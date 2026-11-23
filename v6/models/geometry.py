"""Deterministic, candidate-conditioned geometry for V6 frame tokenization.

All calculations use the *causal prior*, measured LOS and known station state.
No true target range, measurement Jacobian, covariance or future observation is used.
Geometry is calculated in float64 where source precision permits; normalized
features are converted to the projection dtype only inside the neural encoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .feature_builder import FeatureBatch
from .normalization import V6Normalization


@dataclass(frozen=True)
class FrameGeometry:
    """Per-frame physical relations; scalars [..., C], vectors [..., C, 3]."""

    state_scalars: Tensor
    state_vectors: Tensor
    observation_scalars: Tensor
    observation_vectors: Tensor
    los_direct_scalars: Tensor
    los_direct_vectors: Tensor
    predicted_range_m: Tensor
    distance_sensitivity: Tensor  # R / predicted_range_m
    relative_position_m: Tensor
    relative_velocity_mps: Tensor
    geometric_predicted_los: Tensor
    tangent_innovation: Tensor
    predicted_los_rate_per_s: Tensor
    angular_error_rad: Tensor
    geometric_valid_mask: Tensor


def _norm(v: Tensor) -> Tensor:
    return torch.linalg.vector_norm(v, dim=-1)


def _dot(a: Tensor, b: Tensor) -> Tensor:
    return (a * b).sum(dim=-1)


def build_frame_geometry(features: FeatureBatch, normalization: V6Normalization) -> FrameGeometry:
    """Build invariant scalar channels and covariant 3-D vector channels.

    Absolute ECI axes are NOT interpreted as scalar features. For an overall
    ECI coordinate rotation, scalar inputs remain fixed and vector inputs
    rotate consistently (assuming caller rotates all states/observations).
    """
    r_scale = float(normalization.R)
    v_scale = float(normalization.V)
    t_scale = float(normalization.T)
    stats = normalization.stats_scale.detach().to(device=features.prior_state_si.device, dtype=torch.float64)
    eps = 1.0  # one metre: only protects geometrically invalid coincidences
    valid = features.valid_mask.to(torch.bool)

    prior = features.prior_state_si.to(torch.float64)
    station = features.station_state_si.to(torch.float64)
    observed = features.observed_los.to(torch.float64)
    predicted = features.predicted_los.to(torch.float64)
    innovation = features.raw_innovation.to(torch.float64)
    dlos = features.delta_los.to(torch.float64)

    r = prior[..., :3]
    v = prior[..., 3:]
    station_r = station[..., :3]
    station_v = station[..., 3:]
    rel_r = r - station_r
    rel_v = v - station_v
    rho = _norm(rel_r)
    geometry_valid = valid & torch.isfinite(rho) & (rho > eps)
    safe_rho = rho.clamp_min(eps)
    u_geo = rel_r / safe_rho.unsqueeze(-1)
    range_ratio = rho / r_scale
    sensitivity = r_scale / safe_rho
    rel_v_r = _dot(u_geo, rel_v)
    rel_v_t = rel_v - rel_v_r.unsqueeze(-1) * u_geo
    predicted_los_rate = rel_v_t / safe_rho.unsqueeze(-1)

    # Use input predicted LOS to respect the actual innovation definition.
    # The geometric LOS from the prior is also exposed independently so any
    # upstream inconsistency can be diagnosed rather than hidden.
    unit_obs = observed / _norm(observed).clamp_min(1e-15).unsqueeze(-1)
    unit_pred = predicted / _norm(predicted).clamp_min(1e-15).unsqueeze(-1)
    cos_angle = _dot(unit_obs, unit_pred).clamp(-1.0, 1.0)
    p = unit_obs - cos_angle.unsqueeze(-1) * unit_pred
    sin_angle = _norm(p)
    angle = torch.atan2(sin_angle, cos_angle)
    cos_consistency = _dot(unit_pred, u_geo)
    dt_norm = features.delta_t_sec.to(torch.float64) / t_scale
    elapsed_norm = features.relative_time_sec.to(torch.float64) / t_scale

    prev_post_valid = features.metadata.get("previous_posterior_used", torch.zeros_like(valid))
    update_valid = features.metadata.get("update_valid", torch.zeros_like(valid))
    evolution_r = torch.zeros_like(r)
    evolution_v = torch.zeros_like(v)
    upd_r = torch.zeros_like(r)
    upd_v = torch.zeros_like(v)
    prev_post = features.previous_posterior_state_si.to(torch.float64)
    prev_prior = features.previous_prior_state_si.to(torch.float64)
    evolution_r = torch.where(prev_post_valid.unsqueeze(-1), (r-prev_post[..., :3]) / r_scale, evolution_r)
    evolution_v = torch.where(prev_post_valid.unsqueeze(-1), (v-prev_post[..., 3:]) / v_scale, evolution_v)
    upd_r = torch.where(update_valid.unsqueeze(-1), (prev_post[..., :3]-prev_prior[..., :3]) / r_scale, upd_r)
    upd_v = torch.where(update_valid.unsqueeze(-1), (prev_post[..., 3:]-prev_prior[..., 3:]) / v_scale, upd_v)
    r_norm = r / r_scale
    v_norm = v / v_scale
    rel_r_norm = rel_r / r_scale
    rel_v_norm = rel_v / v_scale

    state_scalars = torch.stack((
        _norm(r_norm), _norm(v_norm), _dot(r_norm, v_norm),
        _norm(evolution_r), _norm(evolution_v),
        _norm(upd_r), _norm(upd_v),
        _dot(evolution_r, u_geo), _dot(evolution_v, u_geo),
        _dot(upd_r, u_geo), _dot(upd_v, u_geo),
        dt_norm, elapsed_norm, valid.float(), prev_post_valid.float(), update_valid.float(),
    ), dim=-1)
    state_vectors = torch.stack((
        r_norm, v_norm, evolution_r, evolution_v, upd_r, upd_v,
        rel_r_norm, rel_v_norm,
    ), dim=-2)

    scaled_dlos = dlos / stats[0]
    scaled_innovation = innovation / stats[1]
    station_delta_norm = features.station_delta_position_si.to(torch.float64) / r_scale
    obs_scalars = torch.stack((
        range_ratio, sensitivity, _norm(rel_v_norm),
        rel_v_r / v_scale, _norm(rel_v_t) / v_scale,
        t_scale * _norm(predicted_los_rate),
        cos_angle, sin_angle, angle,
        _norm(scaled_innovation), _norm(scaled_dlos), _dot(p, rel_v_t) / v_scale,
        cos_consistency, dt_norm, elapsed_norm,
        features.station_changed.float(),
        features.station_change_reliable.float(),
        features.difference_valid.float(), valid.float(), geometry_valid.float(),
    ), dim=-1)
    obs_vectors = torch.stack((
        observed, predicted, scaled_innovation, scaled_dlos,
        rel_r_norm, rel_v_norm, u_geo, p,
        t_scale * predicted_los_rate, station_delta_norm,
    ), dim=-2)

    # A genuinely LOS-only representation: no station/range/time/state fields.
    direct_scalars = torch.stack((
        cos_angle, sin_angle, angle, _norm(scaled_innovation),
        _norm(scaled_dlos), valid.float(), features.difference_valid.float(),
    ), dim=-1)
    direct_vectors = torch.stack((observed, predicted, scaled_innovation, scaled_dlos), dim=-2)

    # A geometric coincidence cannot supply a meaningful relative LOS or
    # sensitivity. It is NOT treated as a synthetic range measurement.
    # Preserve measured LOS for the LOS-only path; mask geometric fields only.
    gmask = geometry_valid
    obs_scalars = torch.where(gmask.unsqueeze(-1), obs_scalars, torch.zeros_like(obs_scalars))
    obs_vectors = torch.where(gmask.unsqueeze(-1).unsqueeze(-1), obs_vectors, torch.zeros_like(obs_vectors))
    state_scalars = torch.where(valid.unsqueeze(-1), state_scalars, torch.zeros_like(state_scalars))
    state_vectors = torch.where(valid.unsqueeze(-1).unsqueeze(-1), state_vectors, torch.zeros_like(state_vectors))
    direct_scalars = torch.where(valid.unsqueeze(-1), direct_scalars, torch.zeros_like(direct_scalars))
    direct_vectors = torch.where(valid.unsqueeze(-1).unsqueeze(-1), direct_vectors, torch.zeros_like(direct_vectors))

    return FrameGeometry(
        state_scalars=state_scalars,
        state_vectors=state_vectors,
        observation_scalars=obs_scalars,
        observation_vectors=obs_vectors,
        los_direct_scalars=direct_scalars,
        los_direct_vectors=direct_vectors,
        predicted_range_m=torch.where(gmask, rho, torch.zeros_like(rho)),
        distance_sensitivity=torch.where(gmask, sensitivity, torch.zeros_like(sensitivity)),
        relative_position_m=torch.where(gmask.unsqueeze(-1), rel_r, torch.zeros_like(rel_r)),
        relative_velocity_mps=torch.where(gmask.unsqueeze(-1), rel_v, torch.zeros_like(rel_v)),
        geometric_predicted_los=torch.where(gmask.unsqueeze(-1), u_geo, torch.zeros_like(u_geo)),
        tangent_innovation=torch.where(valid.unsqueeze(-1), p, torch.zeros_like(p)),
        predicted_los_rate_per_s=torch.where(gmask.unsqueeze(-1), predicted_los_rate, torch.zeros_like(predicted_los_rate)),
        angular_error_rad=torch.where(valid, angle, torch.zeros_like(angle)),
        geometric_valid_mask=gmask,
    )

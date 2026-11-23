"""Causal recursive filtering with cached scalar/vector attention.

Physical state and observation prediction use the existing KalmanNet orbit
adapter. This module supplies no alternate dynamics or state-update equation.
"""
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .geometry_relation import build_current_pairwise_geometry
from .scalar_vector_attention import ScalarVectorKVCache


class V6GainHead(nn.Module):
    """Shared V6 gain readout; output is a 6-by-3 gain matrix per sample."""

    def __init__(self, input_dim: int = 240, hidden_dim: int = 128,
                 dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 18, dtype=dtype),
        )

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[-1] != self.input_dim:
            raise ValueError(f"gain input must have shape [B,{self.input_dim}]")
        return self.net(features).reshape(-1, 6, 3)


class V6CorrectionHead(nn.Module):
    """Direct scaled-state correction readout using the shared 240-D adapter."""

    def __init__(self, input_dim: int = 240, hidden_dim: int = 128,
                 dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 6, dtype=dtype),
        )

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[-1] != self.input_dim:
            raise ValueError(f"correction input must have shape [B,{self.input_dim}]")
        return self.net(features).reshape(-1, 6)


@dataclass
class CausalHistoryCache:
    """The full causal prefix and its projected key/value tensors."""
    kv: ScalarVectorKVCache | None
    scalar_tokens: list[Tensor]
    vector_tokens: list[Tensor]
    times_sec: list[Tensor]
    station_states_si: list[Tensor]
    observed_los: list[Tensor]
    station_ids: list[Tensor]
    station_id_reliable: list[Tensor]
    valid: list[bool]
    input_frame_indices: list[int]


class FilterDivergenceError(RuntimeError):
    """A non-finite or undefined physical quantity encountered during a frame."""

    def __init__(self, details: Mapping[str, Any]):
        self.details = dict(details)
        super().__init__(
            f"recursive filter failed at batch={details['batch_index']} "
            f"frame={details['frame_index']} stage={details['stage']}"
        )


def select_last_valid(values: Tensor, valid: Tensor) -> Tensor:
    """Gather the greatest valid time index; all-invalid samples return zero."""
    unbatched = valid.ndim == 1
    if unbatched:
        values = values.unsqueeze(0)
        valid = valid.unsqueeze(0)
    if valid.ndim != 2 or tuple(values.shape[:2]) != tuple(valid.shape):
        raise ValueError("values and valid mask must share [B,T] dimensions")
    batch, steps = valid.shape
    indices = torch.arange(steps, device=valid.device).expand(batch, steps)
    last = torch.where(valid, indices, torch.zeros_like(indices)).amax(dim=1)
    gather_index = last.reshape(batch, 1, *([1] * (values.ndim - 2)))
    gather_index = gather_index.expand(batch, 1, *values.shape[2:])
    result = values.gather(1, gather_index).squeeze(1)
    has_valid = valid.any(dim=1).reshape(batch, *([1] * (values.ndim - 2)))
    result = torch.where(has_valid, result, torch.zeros_like(result))
    return result.squeeze(0) if unbatched else result


@dataclass(frozen=True)
class RecursiveFilterOutput:
    prior_states_si: Tensor
    posterior_states_si: Tensor
    gains: Tensor | None
    correction_scaled: Tensor | None
    correction_si: Tensor | None
    innovations: Tensor
    predicted_los: Tensor
    valid_mask: Tensor
    scalar_tokens: Tensor
    vector_tokens: Tensor
    scalar_attention: Tensor
    vector_attention: Tensor
    los_direct_scalar_tokens: Tensor
    los_direct_vector_tokens: Tensor
    history_lengths: Tensor
    history_cache: Any
    frame_features: Any
    frame_gains: Any
    frame_posteriors: Any
    attention_weights: Any
    attention_logits: Any
    relation_rows: Any
    metadata: Mapping[str, Any]

    @property
    def gain(self) -> Tensor | None:
        return None if self.gains is None else select_last_valid(self.gains, self.valid_mask)

    @property
    def prior(self) -> Tensor:
        return self.prior_states_si

    @property
    def innovation(self) -> Tensor:
        return self.innovations

    @property
    def posterior(self) -> Tensor:
        return self.posterior_states_si

    @property
    def current_correction_scaled(self) -> Tensor | None:
        return (None if self.correction_scaled is None else
                select_last_valid(self.correction_scaled, self.valid_mask))

    @property
    def current_correction_si(self) -> Tensor | None:
        return (None if self.correction_si is None else
                select_last_valid(self.correction_si, self.valid_mask))

    @property
    def posterior_state_si(self) -> Tensor:
        return select_last_valid(self.posterior_states_si, self.valid_mask)

    @property
    def prior_state_si(self) -> Tensor:
        return select_last_valid(self.prior_states_si, self.valid_mask)

    @property
    def raw_innovation(self) -> Tensor:
        return self.innovations

    @property
    def current_innovation_raw(self) -> Tensor:
        return select_last_valid(self.innovations, self.valid_mask)

    @property
    def current_direct_los_scalar(self) -> Tensor:
        return select_last_valid(self.los_direct_scalar_tokens, self.valid_mask)

    @property
    def current_direct_los_vector(self) -> Tensor:
        return select_last_valid(self.los_direct_vector_tokens, self.valid_mask)

    @property
    def token_valid_mask(self) -> Tensor:
        return self.valid_mask.repeat_interleave(2, dim=-1)

    @property
    def token_timestep_mapping(self) -> Tensor:
        frame = torch.arange(self.valid_mask.shape[-1], device=self.valid_mask.device)
        frame = frame.repeat_interleave(2)
        if self.valid_mask.ndim == 2:
            frame = frame.unsqueeze(0).expand(self.valid_mask.shape[0], -1)
        return frame


def _physical_functions():
    """Import the existing differentiable RK4 and LOS functions only."""
    from . import physics
    return physics.state_scale, physics.propagate_scaled, physics.predict_los_from_scaled


def _as_bt(value: Any, batch: int, steps: int, tail: tuple[int, ...],
           device: torch.device, dtype: torch.dtype, name: str) -> Tensor:
    result = torch.as_tensor(value, device=device, dtype=dtype)
    if not tail and result.ndim == 0:
        return result.expand(batch, steps)
    if tail and tuple(result.shape) == tail:
        return result.reshape(1, 1, *tail).expand(batch, steps, *tail)
    if tuple(result.shape) == (steps, *tail):
        result = result.unsqueeze(0)
    if result.ndim != 2 + len(tail) or tuple(result.shape[1:]) != (steps, *tail):
        raise ValueError(f"{name} must be [T,...] or [B,T,...], got {tuple(result.shape)}")
    if result.shape[0] == 1 and batch > 1:
        result = result.expand(batch, *result.shape[1:])
    if result.shape[0] != batch:
        raise ValueError(f"{name} batch dimension must equal {batch}")
    return result


def _batch_epochs(value: Any, batch: int, device: torch.device) -> Tensor:
    epochs = torch.as_tensor(value, device=device, dtype=torch.float64)
    if epochs.ndim == 0:
        return epochs.expand(batch)
    epochs = epochs.reshape(-1)
    if epochs.numel() == 1:
        return epochs.expand(batch)
    if epochs.numel() != batch:
        raise ValueError("init_time_sec must be scalar or have shape [B]")
    return epochs


def _causal_step_features(feature: Any, *, observed: Tensor, station: Tensor,
                          station_id: Tensor, reliable: bool,
                          previous_observed: Tensor | None,
                          previous_station: Tensor | None,
                          previous_station_id: Tensor | None,
                          previous_reliable: bool,
                          normalization: Any) -> Any:
    """Supply the previous-observation context that a standalone step lacks."""
    has_previous = previous_observed is not None
    delta_los = observed - previous_observed if has_previous else torch.zeros_like(observed)
    delta_station = station[:3] - previous_station[:3] if has_previous else torch.zeros_like(station[:3])
    known_change = bool(has_previous and reliable and previous_reliable)
    changed = bool(known_change and station_id.item() != previous_station_id.item())
    device = observed.device
    diff_valid = torch.tensor(has_previous, device=device, dtype=torch.bool)
    first_missing = ~diff_valid
    change_reliable = torch.tensor(known_change, device=device, dtype=torch.bool)
    changed_t = torch.tensor(changed, device=device, dtype=torch.bool)
    update_valid = torch.tensor(has_previous, device=device, dtype=torch.bool)
    obs_features = feature.observation_features.clone()
    state_features = feature.state_features.clone()
    stats = normalization.stats_scale.to(device=device, dtype=obs_features.dtype)
    obs_features[6:9] = normalization.normalize_los(delta_los.to(obs_features.dtype)) / stats[0]
    obs_features[18:21] = normalization.normalize_position(delta_station.to(obs_features.dtype))
    obs_features[21] = float(changed)
    obs_features[25] = float(not has_previous)
    obs_features[26] = float(has_previous)
    obs_features[29] = float(known_change)
    state_features[23] = float(not has_previous)
    state_features[24] = float(has_previous)
    state_features[25] = float(changed)
    state_features[28] = float(has_previous)
    state_features[29] = float(known_change)
    meta = dict(feature.metadata)
    meta.update(
        difference_valid=diff_valid,
        delta_los=delta_los,
        station_delta_position_si=delta_station,
        station_change_reliable=change_reliable,
        update_valid=update_valid,
    )
    return replace(
        feature,
        state_features=state_features,
        observation_features=obs_features,
        delta_los=delta_los,
        station_delta_position_si=delta_station,
        first_frame_missing_mask=first_missing,
        station_changed=changed_t,
        station_change_reliable=change_reliable,
        metadata=meta,
    )


def _gain_override_at(override: Any, batch_index: int, frame_index: int,
                      representation: Tensor, device: torch.device) -> Tensor:
    if callable(override):
        value = override(batch_index=batch_index, frame_index=frame_index,
                         representation=representation)
    else:
        value = torch.as_tensor(override)
        if value.ndim == 3:
            value = value[frame_index]
        elif value.ndim == 4:
            value = value[batch_index, frame_index]
        elif value.ndim != 2:
            raise ValueError("gain_override must be [6,3], [T,6,3], or [B,T,6,3]")
    value = torch.as_tensor(value, device=device)
    if not value.is_floating_point():
        value = value.to(dtype=representation.dtype)
    if tuple(value.shape) != (6, 3):
        raise ValueError("gain_override for one frame must have shape [6,3]")
    return value.unsqueeze(0)


def _fail(stage: str, batch_index: int, frame_index: int, current_time: Tensor,
          **context: Tensor | None) -> None:
    details: dict[str, Any] = {
        "stage": stage,
        "batch_index": batch_index,
        "frame_index": frame_index,
        "time_sec": float(current_time.detach().cpu()),
    }
    for name, value in context.items():
        if isinstance(value, Tensor):
            details[name] = value.detach().cpu().tolist()
    raise FilterDivergenceError(details)


class RecursiveFilterMixin:
    """One-layer online attention with full projected KV and physical history."""

    def forward_filter_sequence(
        self, *,
        initial_state_si: Any,
        init_time_sec: Any,
        times_sec: Any,
        observed_los: Any,
        station_states_si: Any = None,
        station_state_si: Any = None,
        station_ids: Any = None,
        station_id_reliable: Any = None,
        valid_mask: Any = None,
        gain_override: Any = None,
        return_attention: bool = False,
        return_logits: bool = False,
        return_relations: bool = False,
    ) -> RecursiveFilterOutput:
        """Run a predict, observe, attend, gain and update loop for each frame."""
        from .geometry_filter import apply_scaled_state_update, apply_scaled_state_correction

        if station_states_si is None:
            station_states_si = station_state_si
        if station_states_si is None:
            raise ValueError("station_states_si must supply position and velocity")
        if not hasattr(self.attention, "forward_current"):
            raise TypeError("recursive filter requires attention.forward_current")
        device = next(self.parameters()).device
        update_mode = str(getattr(self, "update_mode", "gain"))
        if update_mode not in ("gain", "direct_correction"):
            raise ValueError("model.update_mode must be 'gain' or 'direct_correction'")
        direct_correction = update_mode == "direct_correction"
        if direct_correction and gain_override is not None:
            raise ValueError("gain_override is incompatible with update_mode='direct_correction'")
        times = torch.as_tensor(times_sec, device=device, dtype=torch.float64)
        unbatched = times.ndim == 1
        if unbatched:
            times = times.unsqueeze(0)
        if times.ndim != 2 or times.shape[1] < 1:
            raise ValueError("times_sec must have shape [T] or [B,T], with T > 0")
        batch, steps = map(int, times.shape)
        initial = torch.as_tensor(initial_state_si, device=device, dtype=torch.float64)
        if initial.ndim == 1:
            initial = initial.unsqueeze(0)
        if tuple(initial.shape) != (batch, 6):
            raise ValueError("initial_state_si must have shape [6] or [B,6]")
        init_times = _batch_epochs(init_time_sec, batch, device)
        obs = _as_bt(observed_los, batch, steps, (3,), device, torch.float64, "observed_los")
        stations = _as_bt(station_states_si, batch, steps, (6,), device, torch.float64, "station_states_si")
        valid = (torch.ones((batch, steps), device=device, dtype=torch.bool)
                 if valid_mask is None else _as_bt(valid_mask, batch, steps, (), device, torch.bool, "valid_mask"))
        if station_ids is None:
            ids = torch.zeros((batch, steps), device=device, dtype=torch.int64)
            if station_id_reliable is not None and bool(torch.as_tensor(station_id_reliable).any()):
                raise ValueError("reliable station IDs require station_ids")
        else:
            ids = _as_bt(station_ids, batch, steps, (), device, torch.int64, "station_ids")
        reliable = (torch.zeros((batch, steps), device=device, dtype=torch.bool)
                    if station_id_reliable is None else _as_bt(station_id_reliable, batch, steps, (), device, torch.bool, "station_id_reliable"))
        if not bool(torch.isfinite(initial).all() and torch.isfinite(init_times).all()):
            raise ValueError("initial states and initialization epochs must be finite")
        state_scale, propagate_scaled, predict_los = _physical_functions()
        scale = state_scale(device, torch.float64)
        model_dtype = self.attention.q_scalar.weight.dtype
        samples: list[dict[str, Any]] = []

        for b in range(batch):
            indices = torch.nonzero(valid[b], as_tuple=False).flatten().tolist()
            if indices:
                t_valid = times[b, indices]
                if not bool(torch.isfinite(t_valid).all() and torch.isfinite(obs[b, indices]).all()
                            and torch.isfinite(stations[b, indices]).all()):
                    raise ValueError("valid observation inputs must be finite")
                if bool((t_valid[1:] < t_valid[:-1]).any()) or bool(t_valid[0] < init_times[b]):
                    raise ValueError("valid observations must be ordered and no earlier than init_time_sec")
                if bool((torch.linalg.vector_norm(obs[b, indices], dim=-1) <= 1e-15).any()):
                    raise ValueError("valid LOS vectors must be nonzero")
            slots: dict[str, list[Any]] = {
                name: [None] * steps for name in (
                    "prior", "posterior", "gain", "correction_scaled", "correction_si",
                    "innovation", "predicted",
                    "scalar", "vector", "attn_s", "attn_v", "direct_s", "direct_v",
                    "history_length", "feature", "weights", "logits", "relation",
                )
            }
            history_times: list[Tensor] = []
            history_stations: list[Tensor] = []
            history_obs: list[Tensor] = []
            history_ids: list[Tensor] = []
            history_reliable: list[Tensor] = []
            history = CausalHistoryCache(
                kv=None, scalar_tokens=[], vector_tokens=[], times_sec=history_times,
                station_states_si=history_stations, observed_los=history_obs,
                station_ids=history_ids, station_id_reliable=history_reliable,
                valid=[], input_frame_indices=[],
            )
            previous_prior = initial[b]
            previous_posterior = initial[b]
            previous_time = init_times[b]
            previous_obs = previous_station = previous_id = None
            previous_reliable = False
            last_gain = last_innovation = None

            for k in indices:
                t = times[b, k]
                dt = t - previous_time
                prior_scaled = propagate_scaled(
                    (previous_posterior / scale).reshape(1, 6), float(dt.item())
                )
                prior = prior_scaled.reshape(6) * scale
                if not bool(torch.isfinite(prior).all()):
                    _fail("propagation", b, k, t, prior=prior, previous_posterior=previous_posterior,
                          last_gain=last_gain, last_innovation=last_innovation,
                          station=stations[b, k], observed_los=obs[b, k])
                current_station = stations[b, k]
                rho = torch.linalg.vector_norm(prior[:3] - current_station[:3])
                if not bool(torch.isfinite(rho)) or float(rho.item()) <= 0.0:
                    _fail("predicted_range", b, k, t, prior=prior, station=current_station,
                          predicted_range=rho, previous_posterior=previous_posterior,
                          last_gain=last_gain, last_innovation=last_innovation)
                predicted = predict_los(prior_scaled, current_station[:3].reshape(1, 3)).reshape(3)
                innovation = obs[b, k] - predicted
                if not bool(torch.isfinite(predicted).all() and torch.isfinite(innovation).all()):
                    _fail("los_prediction", b, k, t, prior=prior, predicted_los=predicted,
                          innovation=innovation, station=current_station, predicted_range=rho)
                current_reliable = bool(reliable[b, k].item())
                feature = self.tokenizer.feature_builder.forward_step(
                    time_sec=t, init_time_sec=init_times[b], delta_time_seconds=dt,
                    observed_los=obs[b, k], predicted_los=predicted,
                    raw_innovation=innovation, prior_state_si=prior,
                    previous_prior_state_si=previous_prior,
                    previous_posterior_state_si=previous_posterior,
                    station_state_si=current_station, station_id=ids[b, k],
                    station_id_reliable=current_reliable, valid=True,
                )
                feature = _causal_step_features(
                    feature, observed=obs[b, k], station=current_station,
                    station_id=ids[b, k], reliable=current_reliable,
                    previous_observed=previous_obs, previous_station=previous_station,
                    previous_station_id=previous_id, previous_reliable=previous_reliable,
                    normalization=self.normalization,
                )
                tok = self.tokenizer.forward_sequence(features=feature)
                if tok.scalar_tokens is None or tok.vector_tokens is None:
                    raise RuntimeError("geometry_sv filter requires structured tokenizer outputs")
                s = tok.scalar_tokens.reshape(1, 2, self.scalar_channels)
                v = tok.vector_tokens.reshape(1, 2, self.vector_channels, 3)
                direct_s = tok.los_direct_scalar_token.reshape(1, 1, self.scalar_channels)
                direct_v = tok.los_direct_vector_token.reshape(1, 1, self.vector_channels, 3)
                if not bool(torch.isfinite(s).all() and torch.isfinite(v).all()
                            and torch.isfinite(direct_s).all() and torch.isfinite(direct_v).all()):
                    _fail("tokenizer", b, k, t, prior=prior, previous_posterior=previous_posterior,
                          last_gain=last_gain, last_innovation=last_innovation, innovation=innovation,
                          predicted_los=predicted, observed_los=obs[b, k], station=current_station,
                          predicted_range=rho)

                history_times.append(t)
                history_stations.append(current_station)
                history_obs.append(obs[b, k])
                history_ids.append(ids[b, k])
                history_reliable.append(reliable[b, k])
                history.scalar_tokens.append(s)
                history.vector_tokens.append(v)
                history.valid.append(True)
                history.input_frame_indices.append(k)
                relation, pair_valid = build_current_pairwise_geometry(
                    current_time_sec=t.reshape(1), history_times_sec=torch.stack(history_times).unsqueeze(0),
                    current_station_state_si=current_station.unsqueeze(0),
                    history_station_state_si=torch.stack(history_stations).unsqueeze(0),
                    current_observed_los=obs[b, k].unsqueeze(0),
                    history_observed_los=torch.stack(history_obs).unsqueeze(0),
                    normalization=self.normalization,
                    current_station_id=ids[b, k].reshape(1),
                    history_station_ids=torch.stack(history_ids).unsqueeze(0),
                    current_station_id_reliable=reliable[b, k].reshape(1),
                    history_station_id_reliable=torch.stack(history_reliable).unsqueeze(0),
                )
                allowed = pair_valid.repeat_interleave(2, dim=1).unsqueeze(1).expand(1, 2, -1)
                if bool(getattr(self, "no_historical_state_keys", False)):
                    from .geometry_filter import build_online_no_historical_state_mask
                    allowed = allowed & build_online_no_historical_state_mask(
                        len(history_times) - 1, batch_size=1, device=device
                    )
                direct_values = (direct_s, direct_v) if self.use_direct_attention_values else None
                try:
                    attention = self.attention.forward_current(
                        s, v, history.kv, relation, direct_values,
                        attention_mask=allowed,
                        current_valid_mask=torch.ones((1, 2), device=device, dtype=torch.bool),
                        return_attention=return_attention, return_logits=return_logits,
                    )
                except FloatingPointError:
                    _fail("attention", b, k, t, prior=prior,
                          previous_posterior=previous_posterior, last_gain=last_gain,
                          last_innovation=last_innovation, innovation=innovation,
                          predicted_los=predicted, observed_los=obs[b, k],
                          station=current_station, predicted_range=rho)
                history.kv = attention.cache
                sa, va = attention.output
                bypass_s = direct_s[:, 0] if self.use_direct_gain_bypass else torch.zeros_like(direct_s[:, 0])
                bypass_v = direct_v[:, 0] if self.use_direct_gain_bypass else torch.zeros_like(direct_v[:, 0])
                readout = torch.cat((
                    sa[:, 0], sa[:, 1], va[:, 0].flatten(start_dim=-2),
                    va[:, 1].flatten(start_dim=-2), bypass_s, bypass_v.flatten(start_dim=-2),
                ), dim=-1)
                gain = None
                correction_scaled = None
                correction_si = None
                if direct_correction:
                    if self.correction_head is None:
                        raise RuntimeError("direct_correction requires model.correction_head")
                    correction_parameter = next(self.correction_head.parameters(), None)
                    head_input = readout if correction_parameter is None else readout.to(
                        device=correction_parameter.device, dtype=correction_parameter.dtype
                    )
                    correction_scaled = self.correction_head(head_input)
                    if tuple(correction_scaled.shape) != (1, 6):
                        raise ValueError("correction_head must return [B,6]")
                    correction_si = correction_scaled.to(device=device, dtype=prior.dtype) * scale
                    posterior = apply_scaled_state_correction(
                        prior.unsqueeze(0), correction_scaled
                    ).squeeze(0)
                else:
                    if gain_override is not None:
                        gain = _gain_override_at(gain_override, b, k, readout, device)
                    elif self.enable_gain:
                        gain_parameter = next(self.gain_head.parameters(), None)
                        head_input = readout if gain_parameter is None else readout.to(
                            device=gain_parameter.device, dtype=gain_parameter.dtype
                        )
                        gain = self.gain_head(head_input)
                        if tuple(gain.shape) != (1, 6, 3):
                            raise ValueError("gain_head must return [B,6,3]")
                    else:
                        gain = readout.new_zeros((1, 6, 3))
                    if not bool(torch.isfinite(gain).all()):
                        _fail("gain", b, k, t, prior=prior, gain=gain, innovation=innovation,
                              previous_posterior=previous_posterior, last_gain=last_gain,
                              last_innovation=last_innovation, predicted_los=predicted,
                              observed_los=obs[b, k], station=current_station, predicted_range=rho)
                    gain_frame = gain[0]
                    posterior = apply_scaled_state_update(
                        prior.unsqueeze(0), gain_frame.unsqueeze(0).to(device=device, dtype=prior.dtype),
                        innovation.unsqueeze(0),
                    ).squeeze(0)
                if direct_correction and not bool(torch.isfinite(correction_scaled).all()):
                    _fail("correction", b, k, t, prior=prior, correction_scaled=correction_scaled,
                          innovation=innovation, previous_posterior=previous_posterior,
                          last_gain=last_gain, last_innovation=last_innovation,
                          predicted_los=predicted, observed_los=obs[b, k],
                          station=current_station, predicted_range=rho)
                if not bool(torch.isfinite(posterior).all()):
                    _fail("posterior", b, k, t, prior=prior, posterior=posterior,
                          gain=gain, correction_scaled=correction_scaled, innovation=innovation,
                          predicted_los=predicted, station=current_station, predicted_range=rho)
                for name, value in (
                    ("prior", prior), ("posterior", posterior),
                    ("gain", None if direct_correction else gain_frame),
                    ("correction_scaled", None if correction_scaled is None else correction_scaled[0]),
                    ("correction_si", None if correction_si is None else correction_si[0]),
                    ("innovation", innovation), ("predicted", predicted),
                    ("scalar", s[0]), ("vector", v[0]), ("attn_s", sa[0]), ("attn_v", va[0]),
                    ("direct_s", direct_s[0, 0]), ("direct_v", direct_v[0, 0]),
                    ("history_length", len(history_times)), ("feature", feature),
                    ("weights", attention.weights), ("logits", attention.logits),
                    ("relation", relation if return_relations else None),
                ):
                    slots[name][k] = value
                previous_prior, previous_posterior, previous_time = prior, posterior, t
                previous_obs, previous_station, previous_id = obs[b, k], current_station, ids[b, k]
                previous_reliable = current_reliable
                last_gain, last_innovation = gain, innovation
            slots["cache"] = history
            samples.append(slots)

        templates = {
            "prior": torch.zeros(6, device=device, dtype=torch.float64),
            "posterior": torch.zeros(6, device=device, dtype=torch.float64),
            "gain": torch.zeros((6, 3), device=device, dtype=model_dtype),
            "correction_scaled": torch.zeros(6, device=device, dtype=model_dtype),
            "correction_si": torch.zeros(6, device=device, dtype=torch.float64),
            "innovation": torch.zeros(3, device=device, dtype=torch.float64),
            "predicted": torch.zeros(3, device=device, dtype=torch.float64),
            "scalar": torch.zeros((2, self.scalar_channels), device=device, dtype=model_dtype),
            "vector": torch.zeros((2, self.vector_channels, 3), device=device, dtype=model_dtype),
            "attn_s": torch.zeros((2, self.scalar_channels), device=device, dtype=model_dtype),
            "attn_v": torch.zeros((2, self.vector_channels, 3), device=device, dtype=model_dtype),
            "direct_s": torch.zeros(self.scalar_channels, device=device, dtype=model_dtype),
            "direct_v": torch.zeros((self.vector_channels, 3), device=device, dtype=model_dtype),
        }

        def pack(name: str) -> Tensor:
            zero = templates[name]
            return torch.stack([
                torch.stack([zero if x is None else x for x in sample[name]], dim=0)
                for sample in samples
            ], dim=0)

        values = {name: pack(name) for name in templates}
        values["scalar"] = values["scalar"].reshape(batch, 2 * steps, self.scalar_channels)
        values["vector"] = values["vector"].reshape(batch, 2 * steps, self.vector_channels, 3)
        values["attn_s"] = values["attn_s"].reshape(batch, 2 * steps, self.scalar_channels)
        values["attn_v"] = values["attn_v"].reshape(batch, 2 * steps, self.vector_channels, 3)
        history_lengths = torch.tensor(
            [[v or 0 for v in sample["history_length"]] for sample in samples],
            device=device, dtype=torch.long,
        )

        def nested(name: str) -> Any:
            result = tuple(tuple(sample[name]) for sample in samples)
            return result[0] if unbatched else result

        if unbatched:
            values = {name: value.squeeze(0) for name, value in values.items()}
            valid_out = valid.squeeze(0)
            history_lengths = history_lengths.squeeze(0)
        else:
            valid_out = valid
        return RecursiveFilterOutput(
            prior_states_si=values["prior"], posterior_states_si=values["posterior"],
            gains=None if direct_correction else values["gain"],
            correction_scaled=values["correction_scaled"] if direct_correction else None,
            correction_si=values["correction_si"] if direct_correction else None,
            innovations=values["innovation"],
            predicted_los=values["predicted"], valid_mask=valid_out,
            scalar_tokens=values["scalar"], vector_tokens=values["vector"],
            scalar_attention=values["attn_s"], vector_attention=values["attn_v"],
            los_direct_scalar_tokens=values["direct_s"], los_direct_vector_tokens=values["direct_v"],
            history_lengths=history_lengths,
            history_cache=(samples[0]["cache"] if unbatched else tuple(sample["cache"] for sample in samples)),
            frame_features=nested("feature"),
            frame_gains=None if direct_correction else nested("gain"),
            frame_posteriors=nested("posterior"),
            attention_weights=nested("weights") if return_attention else None,
            attention_logits=nested("logits") if return_logits else None,
            relation_rows=nested("relation") if return_relations else None,
            metadata={
                "model_version": "v6_geometry_sv_recursive_v2",
                "recursive_filter": True,
                "online_kv_cache": True,
                "cache_detached": False,
                "history_truncated": False,
                "inference_complexity": "O(T^2) cached-query attention; O(T) persistent KV/history",
                "bptt_memory": "full graph: O(T^2) attention intermediates plus propagation graphs; no detach",
                "update_mode": update_mode,
                "state_update": ("z_post = z_prior + K @ raw_innovation" if not direct_correction
                                  else "z_post = z_prior + delta_scaled"),
                "physics": "orbit_los_kalman_net.propagate_scaled / EKF propagate_rk4",
                "token_order": "state_0, observation_0, state_1, observation_1, ...",
                "model_config": self.model_config,
            },
        )

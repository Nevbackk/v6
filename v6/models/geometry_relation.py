"""Causal-safe pairwise frame geometry for V6 structured attention.

The relation builder uses only quantities already present in ``FeatureBatch``:
timestamps, station positions and observed LOS vectors.  It never reads the
target state or any future estimate.  For query frame ``i`` and key frame
``j`` the baseline convention is ``station_j - station_i``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from .feature_builder import FeatureBatch
from .normalization import V6Normalization


RELATION_DIM = 8


def _as_batch(value: Tensor, *, tail: int | None = None) -> tuple[Tensor, bool]:
    """Promote an unbatched sequence to ``[1,T,...]`` without detaching."""

    if tail is None:
        if value.ndim == 1:
            return value.unsqueeze(0), True
        if value.ndim == 2:
            return value, False
    else:
        if value.ndim == 2 and value.shape[-1] == tail:
            return value.unsqueeze(0), True
        if value.ndim == 3 and value.shape[-1] == tail:
            return value, False
    raise ValueError(f"expected [T,...] or [B,T,...], got {tuple(value.shape)}")


def _unit_los(value: Tensor, eps: float = 1.0e-15) -> tuple[Tensor, Tensor]:
    norm = torch.linalg.vector_norm(value, dim=-1)
    finite = torch.isfinite(value).all(dim=-1) & torch.isfinite(norm) & (norm > eps)
    safe = norm.clamp_min(eps)
    return value / safe.unsqueeze(-1), finite


def _validate_time_order(times: Tensor, valid: Tensor) -> None:
    """Fail explicitly if valid timestamps decrease.

    Equal timestamps are valid: simultaneous observations may occur at
    different stations, and their signed pairwise ``dt`` is exactly zero.
    """

    pair = valid[:, :, None] & valid[:, None, :]
    ordered = times[:, None, :] >= times[:, :, None]
    # Only pairs with query index i < key index j are checked here.  The
    # relation itself retains signed dt for both directions.
    steps = int(times.shape[1])
    upper = torch.triu(torch.ones((steps, steps), dtype=torch.bool, device=times.device), diagonal=1)
    if bool((pair & upper.unsqueeze(0) & ~ordered).any()):
        raise ValueError("valid observation timestamps must be non-decreasing")


def build_pairwise_geometry(
    features: FeatureBatch,
    normalization: V6Normalization,
    *,
    strict_time_order: bool = True,
) -> tuple[Tensor, Tensor]:
    """Return ``(g_frame, pair_valid)`` for all frame pairs.

    ``g_frame`` has shape ``[T,T,8]`` or ``[B,T,T,8]`` and is ordered as
    ``[dt/T, ||b||/R, b·u_i/R, b·u_j/R, u_i·u_j,
    ||b-(b·u_i)u_i||/R, same_station, station_identity_known]``.
    Query index is ``i`` and key/value index is ``j``.  Invalid pairs are
    explicitly zeroed and represented by ``pair_valid=False``.  Causality is
    deliberately left to the token-level attention mask.
    """

    times, squeezed = _as_batch(features.times_sec.to(dtype=torch.float64))
    valid, valid_squeezed = _as_batch(features.valid_mask.to(dtype=torch.bool))
    station, _ = _as_batch(features.station_state_si.to(dtype=torch.float64), tail=6)
    observed, _ = _as_batch(features.observed_los.to(dtype=torch.float64), tail=3)
    station_ids, _ = _as_batch(features.station_ids.to(dtype=torch.float64))
    station_reliable, _ = _as_batch(features.station_id_reliable.to(dtype=torch.bool))

    if not (squeezed == valid_squeezed):
        raise ValueError("FeatureBatch leading dimensions are inconsistent")
    if tuple(times.shape[:2]) != tuple(valid.shape[:2]):
        raise ValueError("FeatureBatch times and valid_mask shapes disagree")
    if strict_time_order:
        _validate_time_order(times, valid)

    finite_time = torch.isfinite(times)
    finite_station = torch.isfinite(station).all(dim=-1)
    unit_observed, finite_los = _unit_los(observed)
    frame_valid = valid & finite_time & finite_station & finite_los

    # Query i, key j: b_ij = station_j - station_i.
    station_r = station[..., :3]
    baseline = station_r[:, None, :, :] - station_r[:, :, None, :]
    ui = unit_observed[:, :, None, :]
    uj = unit_observed[:, None, :, :]
    dt = times[:, None, :] - times[:, :, None]
    baseline_norm = torch.linalg.vector_norm(baseline, dim=-1)
    radial_i = (baseline * ui).sum(dim=-1)
    radial_j = (baseline * uj).sum(dim=-1)
    transverse_i = baseline - radial_i.unsqueeze(-1) * ui
    transverse_norm = torch.linalg.vector_norm(transverse_i, dim=-1)
    los_cosine = (ui * uj).sum(dim=-1).clamp(-1.0, 1.0)

    known_i = station_reliable[:, :, None]
    known_j = station_reliable[:, None, :]
    identity_known = known_i & known_j
    same_station = identity_known & (station_ids[:, :, None] == station_ids[:, None, :])
    pair_valid = frame_valid[:, :, None] & frame_valid[:, None, :]

    r_scale = float(normalization.R)
    t_scale = float(normalization.T)
    relation = torch.stack(
        (
            dt / t_scale,
            baseline_norm / r_scale,
            radial_i / r_scale,
            radial_j / r_scale,
            los_cosine,
            transverse_norm / r_scale,
            same_station.to(dtype=torch.float64),
            identity_known.to(dtype=torch.float64),
        ),
        dim=-1,
    )
    relation = torch.where(pair_valid.unsqueeze(-1), relation, torch.zeros_like(relation))
    pair_valid = pair_valid.to(dtype=torch.bool)
    if squeezed:
        return relation[0], pair_valid[0]
    return relation, pair_valid


def build_current_pairwise_geometry(
    current_time_sec: Tensor,
    history_times_sec: Tensor,
    current_station_state_si: Tensor,
    history_station_state_si: Tensor,
    current_observed_los: Tensor,
    history_observed_los: Tensor,
    normalization: V6Normalization,
    *,
    current_station_id: Tensor | None = None,
    history_station_ids: Tensor | None = None,
    current_station_id_reliable: Tensor | None = None,
    history_station_id_reliable: Tensor | None = None,
    current_valid: Tensor | None = None,
    history_valid: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Build only newest-query-to-prefix relations, preserving the project sign convention.

    Returns relation [B,K,8] for query=current and key/value=history, where
    baseline is station_key minus station_query. Inputs can be unbatched or
    batched; calculations are performed in float64.
    """
    device = current_station_state_si.device

    def _expand_batch(value: Tensor, batch: int, name: str) -> Tensor:
        if int(value.shape[0]) == 1 and batch > 1:
            return value.expand(batch, *value.shape[1:])
        if int(value.shape[0]) != batch:
            raise ValueError(f"{name} batch dimension does not match {batch}")
        return value

    def _current_vector(value: Tensor, tail: int, name: str, dtype: torch.dtype) -> Tensor:
        result = torch.as_tensor(value, device=device, dtype=dtype)
        if tail == 0:
            if result.ndim == 0:
                result = result.reshape(1)
            elif result.ndim != 1:
                raise ValueError(f"{name} must be scalar or [B]")
        else:
            if result.ndim == 1 and int(result.shape[0]) == tail:
                result = result.unsqueeze(0)
            if result.ndim != 2 or int(result.shape[-1]) != tail:
                raise ValueError(f"{name} must be [{tail}] or [B,{tail}]")
        return result

    def _history_vector(value: Tensor, tail: int, name: str, dtype: torch.dtype, batch: int) -> Tensor:
        result = torch.as_tensor(value, device=device, dtype=dtype)
        if tail == 0:
            if result.ndim == 1:
                result = result.unsqueeze(0)
            if result.ndim != 2:
                raise ValueError(f"{name} must be [K] or [B,K]")
        else:
            if result.ndim == 2 and int(result.shape[-1]) == tail:
                result = result.unsqueeze(0)
            if result.ndim != 3 or int(result.shape[-1]) != tail:
                raise ValueError(f"{name} must be [K,{tail}] or [B,K,{tail}]")
        return _expand_batch(result, batch, name)

    def _current_flag(value: Tensor | None, name: str, batch: int, default: bool) -> Tensor:
        if value is None:
            return torch.full((batch,), default, device=device, dtype=torch.bool)
        result = torch.as_tensor(value, device=device, dtype=torch.bool)
        if result.ndim == 0:
            result = result.reshape(1)
        else:
            result = result.reshape(-1)
        return _expand_batch(result, batch, name)

    def _history_flag(value: Tensor | None, name: str, batch: int, k: int, default: bool) -> Tensor:
        if value is None:
            return torch.full((batch, k), default, device=device, dtype=torch.bool)
        result = torch.as_tensor(value, device=device, dtype=torch.bool)
        if result.ndim == 1:
            result = result.unsqueeze(0)
        if result.ndim != 2 or int(result.shape[1]) != k:
            raise ValueError(f"{name} must be [K] or [B,K]")
        return _expand_batch(result, batch, name)

    def _current_id(value: Tensor | None, name: str, batch: int) -> Tensor:
        if value is None:
            return torch.zeros(batch, device=device, dtype=torch.int64)
        result = torch.as_tensor(value, device=device).reshape(-1)
        return _expand_batch(result, batch, name)

    def _history_id(value: Tensor | None, name: str, batch: int, k: int) -> Tensor:
        if value is None:
            return torch.zeros((batch, k), device=device, dtype=torch.int64)
        result = torch.as_tensor(value, device=device)
        if result.ndim == 1:
            result = result.unsqueeze(0)
        if result.ndim != 2 or int(result.shape[1]) != k:
            raise ValueError(f"{name} must be [K] or [B,K]")
        return _expand_batch(result, batch, name)

    cs = _current_vector(current_station_state_si, 6, "current_station_state_si", torch.float64)
    batch = int(cs.shape[0])
    cu = _expand_batch(
        _current_vector(current_observed_los, 3, "current_observed_los", torch.float64),
        batch,
        "current_observed_los",
    )
    ct = _expand_batch(
        _current_vector(current_time_sec, 0, "current_time_sec", torch.float64),
        batch,
        "current_time_sec",
    )
    ht = _history_vector(history_times_sec, 0, "history_times_sec", torch.float64, batch)
    hs = _history_vector(history_station_state_si, 6, "history_station_state_si", torch.float64, batch)
    hu = _history_vector(history_observed_los, 3, "history_observed_los", torch.float64, batch)
    if tuple(cs.shape) != (batch, 6) or tuple(cu.shape) != (batch, 3):
        raise ValueError("current station and LOS must be [B,6] and [B,3]")
    if tuple(hs.shape[:2]) != tuple(ht.shape) or tuple(hu.shape[:2]) != tuple(ht.shape):
        raise ValueError("history time, station and LOS batch/length dimensions disagree")
    k = int(ht.shape[1])
    if k < 1:
        raise ValueError("history must include the current frame")
    cr = _current_flag(current_station_id_reliable, "current_station_id_reliable", batch, False)
    hr = _history_flag(history_station_id_reliable, "history_station_id_reliable", batch, k, False)
    ci = _current_id(current_station_id, "current_station_id", batch)
    hi = _history_id(history_station_ids, "history_station_ids", batch, k)
    cv = _current_flag(current_valid, "current_valid", batch, True)
    hv = _history_flag(history_valid, "history_valid", batch, k, True)

    unorm = torch.linalg.vector_norm(cu, dim=-1)
    hnorm = torch.linalg.vector_norm(hu, dim=-1)
    current_los_ok = torch.isfinite(cu).all(dim=-1) & torch.isfinite(unorm) & (unorm > 1.0e-15)
    history_los_ok = torch.isfinite(hu).all(dim=-1) & torch.isfinite(hnorm) & (hnorm > 1.0e-15)
    ui = cu / unorm.clamp_min(1.0e-15).unsqueeze(-1)
    uj = hu / hnorm.clamp_min(1.0e-15).unsqueeze(-1)
    station_i = cs[..., :3]
    station_j = hs[..., :3]
    baseline = station_j - station_i[:, None, :]
    radial_i = (baseline * ui[:, None, :]).sum(dim=-1)
    radial_j = (baseline * uj).sum(dim=-1)
    transverse = baseline - radial_i.unsqueeze(-1) * ui[:, None, :]
    same = cr[:, None] & hr & (ci[:, None] == hi)
    identity_known = cr[:, None] & hr
    pair_valid = (
        cv[:, None] & hv & torch.isfinite(ct[:, None]) & torch.isfinite(ht)
        & torch.isfinite(cs).all(dim=-1)[:, None]
        & torch.isfinite(hs).all(dim=-1)
        & current_los_ok[:, None] & history_los_ok
    )
    relation = torch.stack(
        (
            (ht - ct[:, None]) / float(normalization.T),
            torch.linalg.vector_norm(baseline, dim=-1) / float(normalization.R),
            radial_i / float(normalization.R),
            radial_j / float(normalization.R),
            (ui[:, None, :] * uj).sum(dim=-1).clamp(-1.0, 1.0),
            torch.linalg.vector_norm(transverse, dim=-1) / float(normalization.R),
            same.to(torch.float64),
            identity_known.to(torch.float64),
        ),
        dim=-1,
    )
    relation = torch.where(pair_valid.unsqueeze(-1), relation, torch.zeros_like(relation))
    return relation, pair_valid


def build_causal_geometry_chunk(
    times_sec: Tensor,
    station_state_si: Tensor,
    observed_los: Tensor,
    station_ids: Tensor,
    station_id_reliable: Tensor,
    valid_mask: Tensor,
    normalization: V6Normalization,
    *,
    query_start: int,
    query_stop: int,
) -> tuple[Tensor, Tensor]:
    """Build a small batch of causal query rows in one vectorized call.

    The returned tensors have shape ``[B,Q,P,8]`` and ``[B,Q,P]`` where
    ``Q=query_stop-query_start`` and ``P=query_stop``.  Row ``q`` is therefore
    consumed as ``chunk[:, q-query_start, :q+1]``.  The arithmetic mirrors
    :func:`build_current_pairwise_geometry` element-for-element; the only
    additional operation is the explicit causal index mask.  Inputs are
    converted with ``torch.as_tensor`` but are never detached, so callers that
    use this helper directly retain the normal autograd behavior.

    This helper is intended for sequence metadata whose values are fixed for a
    recursive pass.  The caller remains responsible for guarding any path that
    needs gradients with respect to those metadata tensors.
    """
    if times_sec.ndim != 2 or station_state_si.ndim != 3 or observed_los.ndim != 3:
        raise ValueError("static geometry inputs must be batched [B,T,...] tensors")
    if station_ids.ndim != 2 or station_id_reliable.ndim != 2 or valid_mask.ndim != 2:
        raise ValueError("static geometry metadata must be [B,T]")
    batch, steps = times_sec.shape
    if tuple(station_state_si.shape[:2]) != (batch, steps) or tuple(station_state_si.shape[-1:]) != (6,):
        raise ValueError("station_state_si must be [B,T,6]")
    if tuple(observed_los.shape[:2]) != (batch, steps) or tuple(observed_los.shape[-1:]) != (3,):
        raise ValueError("observed_los must be [B,T,3]")
    if tuple(station_ids.shape) != (batch, steps):
        raise ValueError("station_ids must be [B,T]")
    if tuple(station_id_reliable.shape) != (batch, steps):
        raise ValueError("station_id_reliable must be [B,T]")
    if tuple(valid_mask.shape) != (batch, steps):
        raise ValueError("valid_mask must be [B,T]")
    query_start = int(query_start)
    query_stop = int(query_stop)
    if not (0 <= query_start < query_stop <= steps):
        raise ValueError("query chunk must satisfy 0 <= start < stop <= T")

    device = times_sec.device
    times = torch.as_tensor(times_sec, device=device, dtype=torch.float64)
    station = torch.as_tensor(station_state_si, device=device, dtype=torch.float64)
    observed = torch.as_tensor(observed_los, device=device, dtype=torch.float64)
    station_ids = torch.as_tensor(station_ids, device=device)
    station_reliable = torch.as_tensor(
        station_id_reliable, device=device, dtype=torch.bool
    )
    valid = torch.as_tensor(valid_mask, device=device, dtype=torch.bool)

    query_times = times[:, query_start:query_stop]
    history_times = times[:, :query_stop]
    query_station = station[:, query_start:query_stop]
    history_station = station[:, :query_stop]
    query_observed = observed[:, query_start:query_stop]
    history_observed = observed[:, :query_stop]
    query_ids = station_ids[:, query_start:query_stop]
    history_ids = station_ids[:, :query_stop]
    query_reliable = station_reliable[:, query_start:query_stop]
    history_reliable = station_reliable[:, :query_stop]
    query_valid = valid[:, query_start:query_stop]
    history_valid = valid[:, :query_stop]

    query_norm = torch.linalg.vector_norm(query_observed, dim=-1)
    history_norm = torch.linalg.vector_norm(history_observed, dim=-1)
    query_los_ok = (
        torch.isfinite(query_observed).all(dim=-1)
        & torch.isfinite(query_norm)
        & (query_norm > 1.0e-15)
    )
    history_los_ok = (
        torch.isfinite(history_observed).all(dim=-1)
        & torch.isfinite(history_norm)
        & (history_norm > 1.0e-15)
    )
    query_unit = query_observed / query_norm.clamp_min(1.0e-15).unsqueeze(-1)
    history_unit = history_observed / history_norm.clamp_min(1.0e-15).unsqueeze(-1)

    baseline = (
        history_station[:, None, :, :3] - query_station[:, :, None, :3]
    )
    radial_query = (baseline * query_unit[:, :, None, :]).sum(dim=-1)
    radial_history = (baseline * history_unit[:, None, :, :]).sum(dim=-1)
    transverse = baseline - radial_query.unsqueeze(-1) * query_unit[:, :, None, :]
    same_station = (
        query_reliable[:, :, None]
        & history_reliable[:, None, :]
        & (query_ids[:, :, None] == history_ids[:, None, :])
    )
    identity_known = query_reliable[:, :, None] & history_reliable[:, None, :]
    query_indices = torch.arange(query_start, query_stop, device=device)
    history_indices = torch.arange(query_stop, device=device)
    causal = history_indices[None, :] <= query_indices[:, None]
    pair_valid = (
        query_valid[:, :, None]
        & history_valid[:, None, :]
        & causal[None, :, :]
        & torch.isfinite(query_times)[:, :, None]
        & torch.isfinite(history_times)[:, None, :]
        & torch.isfinite(query_station).all(dim=-1)[:, :, None]
        & torch.isfinite(history_station).all(dim=-1)[:, None, :]
        & query_los_ok[:, :, None]
        & history_los_ok[:, None, :]
    )
    relation = torch.stack(
        (
            (history_times[:, None, :] - query_times[:, :, None])
            / float(normalization.T),
            torch.linalg.vector_norm(baseline, dim=-1) / float(normalization.R),
            radial_query / float(normalization.R),
            radial_history / float(normalization.R),
            (query_unit[:, :, None, :] * history_unit[:, None, :, :])
            .sum(dim=-1)
            .clamp(-1.0, 1.0),
            torch.linalg.vector_norm(transverse, dim=-1) / float(normalization.R),
            same_station.to(dtype=torch.float64),
            identity_known.to(dtype=torch.float64),
        ),
        dim=-1,
    )
    relation = torch.where(pair_valid.unsqueeze(-1), relation, torch.zeros_like(relation))
    return relation, pair_valid


def expand_frame_relation_to_tokens(
    frame_relation: Tensor,
    frame_pair_valid: Tensor,
) -> tuple[Tensor, Tensor]:
    """Repeat each frame pair for its interleaved state/observation tokens."""

    if frame_relation.ndim == 3:
        frame_relation = frame_relation.unsqueeze(0)
        frame_pair_valid = frame_pair_valid.unsqueeze(0)
        squeezed = True
    elif frame_relation.ndim == 4:
        squeezed = False
    else:
        raise ValueError("frame_relation must have shape [T,T,8] or [B,T,T,8]")
    relation = frame_relation.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
    pair_valid = frame_pair_valid.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
    if squeezed:
        return relation[0], pair_valid[0]
    return relation, pair_valid


def build_token_causal_mask(valid_mask: Tensor) -> Tensor:
    """Build ``[2T,2T]`` or ``[B,2T,2T]`` frame-causal token masks."""

    valid = valid_mask.to(dtype=torch.bool)
    if valid.ndim == 1:
        valid = valid.unsqueeze(0)
        squeezed = True
    elif valid.ndim == 2:
        squeezed = False
    else:
        raise ValueError("valid_mask must have shape [T] or [B,T]")
    frame_index = torch.arange(valid.shape[1], device=valid.device).repeat_interleave(2)
    token_valid = valid.repeat_interleave(2, dim=1)
    allowed = frame_index[None, :, None] >= frame_index[None, None, :]
    allowed = allowed & token_valid[:, :, None] & token_valid[:, None, :]
    return allowed[0] if squeezed else allowed


__all__ = [
    "RELATION_DIM",
    "build_pairwise_geometry",
    "build_current_pairwise_geometry",
    "build_causal_geometry_chunk",
    "expand_frame_relation_to_tokens",
    "build_token_causal_mask",
]

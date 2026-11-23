"""No-sync batched causal feature construction for streaming frames.

The online recursive filter historically called ``FeatureBuilder.forward_step``
and then ``recursive_filter._causal_step_features`` for every sample.  That
path read previous-observation and station-ID context as Python scalars.  This
module implements the same FeatureBatch layout and normalization directly for
a ``[B, 1, ...]`` frame.  All value-dependent branches are tensor operations;
the hot path has no scalar device reads or reduction used as a Python condition.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .feature_builder import FeatureBatch, FeatureBuilder, _coerce_station_ids


def _as_frame(
    value: Any,
    *,
    batch: int,
    tail: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    name: str,
    optional: bool = False,
) -> Tensor:
    """Coerce a lane to ``[B, 1, *tail]`` without detaching tensors."""

    if value is None:
        if not optional:
            raise ValueError(f"{name} is required")
        return torch.zeros((batch, 1, *tail), device=device, dtype=dtype)
    if isinstance(value, Tensor):
        result = value
        if result.is_complex():
            raise TypeError(f"{name} does not support complex values")
        if result.device != device:
            result = result.to(device=device)
        if result.dtype != dtype:
            result = result.to(dtype=dtype)
    else:
        result = torch.as_tensor(value, device=device, dtype=dtype)

    expected = (batch, 1, *tail)
    if tuple(result.shape) == expected:
        return result
    if tuple(result.shape) == (batch, *tail):
        return result.unsqueeze(1)
    if tuple(result.shape) == (1, *tail):
        return result.reshape(1, 1, *tail).expand(batch, 1, *tail)
    if tuple(result.shape) == tail:
        return result.reshape(1, 1, *tail).expand(batch, 1, *tail)
    if not tail:
        if result.ndim == 0:
            return result.reshape(1, 1).expand(batch, 1)
        if tuple(result.shape) == (batch,):
            return result.unsqueeze(1)
        if tuple(result.shape) == (1,):
            return result.reshape(1, 1).expand(batch, 1)
    raise ValueError(
        f"{name} must have shape [B,1{',...' if tail else ''}] "
        f"for batch={batch}; got {tuple(result.shape)}"
    )


def _finite_or_zero(value: Tensor) -> Tensor:
    if value.is_floating_point():
        return torch.where(torch.isfinite(value), value, torch.zeros_like(value))
    return value


def _replace_slice(value: Tensor, start: int, stop: int, replacement: Tensor) -> Tensor:
    return torch.cat((value[..., :start], replacement, value[..., stop:]), dim=-1)


def _flag(mask: Tensor, dtype: torch.dtype) -> Tensor:
    return mask.to(dtype=dtype).unsqueeze(-1)


def _bool_frame(
    value: Any,
    *,
    batch: int,
    device: torch.device,
    default: bool,
    name: str,
) -> Tensor:
    if value is None:
        return torch.full((batch, 1), default, device=device, dtype=torch.bool)
    raw = _as_frame(
        value,
        batch=batch,
        tail=(),
        device=device,
        dtype=torch.float32,
        name=name,
    )
    return raw > 0.5


def _infer_batch(*values: Any) -> tuple[torch.device, int]:
    """Infer device and batch from static shapes only."""

    device = next(
        (value.device for value in values if isinstance(value, Tensor)),
        torch.device("cpu"),
    )
    batch: int | None = None
    for value in values:
        if isinstance(value, Tensor):
            shape = tuple(value.shape)
        else:
            try:
                shape = tuple(torch.as_tensor(value).shape)
            except (TypeError, ValueError, RuntimeError):
                continue
        if len(shape) == 0:
            continue
        candidate = int(shape[0])
        if batch is None:
            batch = candidate
        elif candidate not in (1, batch):
            raise ValueError(f"inconsistent batch dimensions: expected {batch}, got {candidate}")
    return device, (1 if batch is None else batch)


def build_batch_features(
    feature_builder: FeatureBuilder,
    *,
    time_sec: Any,
    observed_los: Any,
    prior_state_si: Any,
    previous_prior_state_si: Any = None,
    previous_posterior_state_si: Any = None,
    previous_observed_los: Any = None,
    station_state_si: Any,
    previous_station_state_si: Any = None,
    station_id: Any = None,
    previous_station_id: Any = None,
    station_id_reliable: Any = None,
    previous_station_id_reliable: Any = None,
    predicted_los: Any = None,
    raw_innovation: Any = None,
    init_time_sec: Any = None,
    delta_time_seconds: Any = None,
    active: Any = None,
    has_previous: Any = None,
    metadata: Mapping[str, Any] | None = None,
) -> FeatureBatch:
    """Build one causal frame for a batch without per-frame host sync.

    All frame lanes use leading shape ``[B, 1]``; ``[B, ...]`` and one shared
    frame are accepted and broadcast.  The returned FeatureBatch always keeps
    the explicit ``[B, 1, ...]`` dimensions.  ``active`` masks padding and
    ``has_previous`` describes the previous observation/station context.  The
    two masks are combined with finite-input masks using tensor operations.
    """

    device, batch = _infer_batch(
        time_sec,
        observed_los,
        prior_state_si,
        previous_prior_state_si,
        previous_posterior_state_si,
        station_state_si,
    )

    # Match FeatureBuilder's existing precision policy for SI feature lanes;
    # timestamps and explicit dt remain float64.
    feature_dtype = torch.float32
    for value in (prior_state_si, observed_los, station_state_si):
        if isinstance(value, Tensor) and value.is_floating_point():
            feature_dtype = value.dtype
            break

    time_frame = _as_frame(
        time_sec,
        batch=batch,
        tail=(),
        device=device,
        dtype=torch.float64,
        name="time_sec",
    )
    observed = _as_frame(
        observed_los,
        batch=batch,
        tail=(3,),
        device=device,
        dtype=feature_dtype,
        name="observed_los",
    )
    prior = _as_frame(
        prior_state_si,
        batch=batch,
        tail=(6,),
        device=device,
        dtype=feature_dtype,
        name="prior_state_si",
    )
    station = _as_frame(
        station_state_si,
        batch=batch,
        tail=(6,),
        device=device,
        dtype=feature_dtype,
        name="station_state_si",
    )
    previous_prior = _as_frame(
        previous_prior_state_si,
        batch=batch,
        tail=(6,),
        device=device,
        dtype=feature_dtype,
        name="previous_prior_state_si",
        optional=True,
    )
    previous_posterior = _as_frame(
        previous_posterior_state_si,
        batch=batch,
        tail=(6,),
        device=device,
        dtype=feature_dtype,
        name="previous_posterior_state_si",
        optional=True,
    )
    previous_observed = _as_frame(
        previous_observed_los,
        batch=batch,
        tail=(3,),
        device=device,
        dtype=feature_dtype,
        name="previous_observed_los",
        optional=True,
    )
    previous_station = _as_frame(
        previous_station_state_si,
        batch=batch,
        tail=(6,),
        device=device,
        dtype=feature_dtype,
        name="previous_station_state_si",
        optional=True,
    )
    predicted = _as_frame(
        predicted_los,
        batch=batch,
        tail=(3,),
        device=device,
        dtype=feature_dtype,
        name="predicted_los",
        optional=True,
    )
    active_input = _bool_frame(
        active, batch=batch, device=device, default=True, name="active"
    )
    has_previous_input = _bool_frame(
        has_previous, batch=batch, device=device, default=False, name="has_previous"
    )
    current_reliable_input = _bool_frame(
        station_id_reliable,
        batch=batch,
        device=device,
        default=False,
        name="station_id_reliable",
    )
    previous_reliable = _bool_frame(
        previous_station_id_reliable,
        batch=batch,
        device=device,
        default=False,
        name="previous_station_id_reliable",
    )

    # IDs use FeatureBuilder's deterministic numeric/string encoding.  This is
    # metadata conversion only; no value is read on the host.
    station_values, station_ids_present = _coerce_station_ids(
        station_id, batch=batch, steps=1, device=device
    )
    previous_station_values, _ = _coerce_station_ids(
        previous_station_id, batch=batch, steps=1, device=device
    )
    station_id_finite = torch.isfinite(station_values)
    station_id_reliable_effective = current_reliable_input & station_id_finite

    times_finite = torch.isfinite(time_frame)
    observed_finite = torch.isfinite(observed).all(dim=-1)
    prior_finite = torch.isfinite(prior).all(dim=-1)
    station_finite = torch.isfinite(station).all(dim=-1)
    predicted_finite = torch.isfinite(predicted).all(dim=-1)
    # ``predicted_los`` is required on active frames by the old FeatureBuilder;
    # an omitted lane therefore contributes no valid frame while preserving a
    # fully tensorized fast path.
    predicted_present = torch.ones((batch, 1), device=device, dtype=torch.bool)
    if predicted_los is None:
        predicted_present = torch.zeros_like(predicted_present)
    valid = (
        active_input
        & times_finite
        & observed_finite
        & prior_finite
        & station_finite
        & predicted_finite
        & predicted_present
    )

    times = _finite_or_zero(time_frame)
    observed = _finite_or_zero(observed)
    prior = _finite_or_zero(prior)
    previous_prior = _finite_or_zero(previous_prior)
    previous_posterior = _finite_or_zero(previous_posterior)
    station = _finite_or_zero(station)
    predicted = _finite_or_zero(predicted)
    raw_innovation = observed - predicted

    # Single-frame FeatureBuilder semantics: explicit previous prior/posterior
    # values are available to the state lanes even when has_previous is false;
    # the old _causal_step_features helper only changes the context flags and
    # observation/station deltas.  Therefore state validity is based on the
    # supplied previous lanes, while context validity is has_previous below.
    previous_prior_present = previous_prior_state_si is not None
    previous_posterior_present = previous_posterior_state_si is not None
    previous_prior_finite = torch.isfinite(previous_prior).all(dim=-1)
    previous_posterior_finite = torch.isfinite(previous_posterior).all(dim=-1)
    previous_prior_used = valid & previous_prior_present & previous_prior_finite
    previous_posterior_used = valid & previous_posterior_present & previous_posterior_finite
    update_valid_base = previous_prior_used & previous_posterior_used

    if delta_time_seconds is None:
        delta_t = torch.zeros_like(times, dtype=torch.float64)
    else:
        supplied_delta = _as_frame(
            delta_time_seconds,
            batch=batch,
            tail=(),
            device=device,
            dtype=torch.float64,
            name="delta_time_seconds",
        )
        delta_t = torch.where(valid, _finite_or_zero(supplied_delta), torch.zeros_like(supplied_delta))

    if init_time_sec is None:
        initial_time = times
    else:
        initial_raw = torch.as_tensor(init_time_sec, device=device, dtype=torch.float64)
        if initial_raw.ndim == 0:
            initial_time = initial_raw.reshape(1, 1).expand(batch, 1)
        elif initial_raw.ndim == 1:
            if int(initial_raw.shape[0]) == 1:
                initial_time = initial_raw.reshape(1, 1).expand(batch, 1)
            elif int(initial_raw.shape[0]) == batch:
                initial_time = initial_raw.reshape(batch, 1)
            else:
                raise ValueError(f"init_time_sec must be scalar, [B] or [B,1], got {tuple(initial_raw.shape)}")
        elif initial_raw.ndim == 2 and tuple(initial_raw.shape) == (batch, 1):
            initial_time = initial_raw
        else:
            raise ValueError(f"init_time_sec must be scalar, [B] or [B,1], got {tuple(initial_raw.shape)}")
        initial_time = _finite_or_zero(initial_time)
        if delta_time_seconds is None:
            first_delta = times - initial_time
            delta_t = torch.where(valid, first_delta, torch.zeros_like(first_delta))

    station_id_used = station_id_reliable_effective & valid
    station_values = torch.where(
        station_id_used, station_values, torch.zeros_like(station_values)
    )

    # The new batch frame may have an explicit previous observation even though
    # FeatureBuilder sees only one timestep.  This is the vectorized equivalent
    # of _causal_step_features.
    has_previous_frame = has_previous_input & valid
    delta_los = torch.where(
        has_previous_frame.unsqueeze(-1),
        observed - _finite_or_zero(previous_observed),
        torch.zeros_like(observed),
    )
    station_delta_position = torch.where(
        has_previous_frame.unsqueeze(-1),
        station[..., :3] - _finite_or_zero(previous_station[..., :3]),
        torch.zeros_like(station[..., :3]),
    )
    station_change_reliable = has_previous_frame & station_id_used & previous_reliable
    station_changed = station_change_reliable & (station_values != previous_station_values)
    difference_valid = has_previous_frame

    previous_posterior = _finite_or_zero(previous_posterior)
    previous_prior = _finite_or_zero(previous_prior)
    stats_scale = feature_builder.normalization.stats_scale.to(device=device, dtype=feature_dtype)

    # Normalization is exactly the existing FeatureBuilder normalization: SI
    # state/position/time lanes are normalized once, while LOS lanes retain
    # their direct units and use the fixed statistical scales for delta/error.
    prior_norm = feature_builder.normalization.normalize_state(prior)
    previous_prior_norm = feature_builder.normalization.normalize_state(previous_prior)
    previous_posterior_norm = feature_builder.normalization.normalize_state(previous_posterior)
    state_evolution_norm = prior_norm - previous_posterior_norm
    state_evolution_norm = torch.where(
        previous_posterior_used.unsqueeze(-1),
        state_evolution_norm,
        torch.zeros_like(state_evolution_norm),
    )
    state_update_norm = previous_posterior_norm - previous_prior_norm
    state_update_norm = torch.where(
        update_valid_base.unsqueeze(-1),
        state_update_norm,
        torch.zeros_like(state_update_norm),
    )
    state_update_norm = torch.cat(
        (
            state_update_norm[..., :3] / stats_scale[2],
            state_update_norm[..., 3:] / stats_scale[3],
        ),
        dim=-1,
    )
    station_position_norm = feature_builder.normalization.normalize_position(station[..., :3])
    station_state_norm = feature_builder.normalization.normalize_state(station)
    station_delta_norm = feature_builder.normalization.normalize_position(station_delta_position)
    dt_norm = feature_builder.normalization.normalize_time(delta_t).to(dtype=feature_dtype)
    elapsed_sec = torch.where(valid, times - initial_time, torch.zeros_like(times))
    elapsed_norm = feature_builder.normalization.normalize_time(elapsed_sec).to(dtype=feature_dtype)
    observed_direct = feature_builder.normalization.normalize_los(observed)
    predicted_direct = feature_builder.normalization.normalize_los(predicted)
    delta_los_direct = feature_builder.normalization.normalize_los(delta_los) / stats_scale[0]
    innovation_direct = feature_builder.normalization.normalize_los(raw_innovation) / stats_scale[1]
    one_if = lambda mask: _flag(mask, feature_dtype)
    zero = torch.zeros((batch, 1, 1), device=device, dtype=feature_dtype)
    station_code = station_values.to(dtype=feature_dtype).unsqueeze(-1)
    station_code = torch.where(
        station_id_used.unsqueeze(-1), station_code, torch.zeros_like(station_code)
    )

    first_missing = valid & ~has_previous_frame
    station_position_used = valid
    state_parts = [
        prior_norm,
        state_evolution_norm,
        state_update_norm,
        station_position_norm,
        dt_norm.unsqueeze(-1),
        one_if(valid),
        one_if(first_missing),
        one_if(difference_valid),
        one_if(station_changed),
        one_if(station_id_used & valid),
        station_code,
        one_if(update_valid_base),
        one_if(station_change_reliable),
        one_if(station_position_used),
    ]
    state_features = torch.cat(state_parts, dim=-1)
    state_features = torch.cat((state_features, zero), dim=-1)

    time_features = torch.cat((dt_norm.unsqueeze(-1), elapsed_norm.unsqueeze(-1)), dim=-1)
    observation_parts = [
        torch.cat((observed_direct, predicted_direct), dim=-1),
        delta_los_direct,
        innovation_direct,
        torch.cat((station_state_norm, station_delta_norm, one_if(station_changed)), dim=-1),
        time_features,
        one_if(valid),
        one_if(first_missing),
        one_if(difference_valid),
        one_if(station_id_used & valid),
        station_code,
        one_if(station_change_reliable),
        one_if(station_position_used),
        one_if(times_finite),
    ]
    observation_features = torch.cat(observation_parts, dim=-1)

    # _causal_step_features overrides only these fixed layout lanes.  The
    # state numerical evolution/update lanes retain FeatureBuilder semantics.
    state_features = _replace_slice(state_features, 23, 24, one_if(~has_previous_frame))
    state_features = _replace_slice(state_features, 24, 25, one_if(has_previous_frame))
    state_features = _replace_slice(state_features, 25, 26, one_if(station_changed))
    state_features = _replace_slice(state_features, 28, 29, one_if(has_previous_frame))
    state_features = _replace_slice(state_features, 29, 30, one_if(station_change_reliable))
    observation_features = _replace_slice(observation_features, 6, 9, delta_los_direct)
    observation_features = _replace_slice(observation_features, 18, 21, station_delta_norm)
    observation_features = _replace_slice(observation_features, 21, 22, one_if(station_changed))
    observation_features = _replace_slice(observation_features, 25, 26, one_if(~has_previous_frame))
    observation_features = _replace_slice(observation_features, 26, 27, one_if(has_previous_frame))
    observation_features = _replace_slice(observation_features, 29, 30, one_if(station_change_reliable))

    canonical_metadata: dict[str, Any] = dict(metadata or {})
    canonical_metadata.update(
        {
            "causal_prior_used": valid,
            "previous_prior_used": previous_prior_used,
            "previous_posterior_used": previous_posterior_used,
            # Match _causal_step_features: context validity is independent of
            # whether the optional state lanes were supplied.
            "update_valid": has_previous_frame,
            "current_posterior_used": torch.zeros_like(valid),
            "direct_los": observed_direct,
            "delta_los": delta_los,
            "raw_innovation": raw_innovation,
            "station_delta_position_si": station_delta_position,
            "time_features": time_features,
            "delta_t_sec": delta_t,
            "relative_time_sec": elapsed_sec,
            "station_id_used": station_id_used,
            "station_change_reliable": station_change_reliable,
            "difference_valid": difference_valid,
        }
    )

    return FeatureBatch(
        state_features=state_features,
        observation_features=observation_features,
        times_sec=times,
        relative_time_sec=elapsed_sec,
        delta_t_sec=delta_t,
        valid_mask=valid,
        first_frame_missing_mask=first_missing,
        prior_state_si=prior,
        previous_prior_state_si=previous_prior,
        previous_posterior_state_si=previous_posterior,
        observed_los=observed_direct,
        delta_los=delta_los,
        predicted_los=predicted_direct,
        raw_innovation=raw_innovation,
        station_state_si=station,
        station_position_si=station[..., :3],
        station_delta_position_si=station_delta_position,
        station_ids=station_values,
        station_id_reliable=station_id_reliable_effective,
        station_id_used=station_id_used,
        station_changed=station_changed,
        station_change_reliable=station_change_reliable,
        lengths=valid.to(dtype=torch.long).sum(dim=-1),
        stats_scale=feature_builder.normalization.stats_scale,
        metadata=canonical_metadata,
    )


class BatchFeatureBuilder(nn.Module):
    """Callable module wrapper around :func:`build_batch_features`."""

    def __init__(self, feature_builder: FeatureBuilder | None = None, **kwargs: Any) -> None:
        super().__init__()
        self.feature_builder = (
            feature_builder if feature_builder is not None else FeatureBuilder(**kwargs)
        )

    def forward(self, **kwargs: Any) -> FeatureBatch:
        return build_batch_features(self.feature_builder, **kwargs)


BatchedFeatureBuilder = BatchFeatureBuilder
build_batch_feature_batch = build_batch_features
build_batch_causal_features = build_batch_features


__all__ = [
    "BatchFeatureBuilder",
    "BatchedFeatureBuilder",
    "build_batch_features",
    "build_batch_feature_batch",
    "build_batch_causal_features",
]

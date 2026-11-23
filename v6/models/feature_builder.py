"""Causal, differentiable feature construction for V6 observation tokens.

This file owns representation only.  It does not contain attention, recurrent
state, or a training loop.  A sequence is represented by two independent
branches at every valid frame:

``state_features``
    Current causal prior, previous prior, previous posterior, and physical
    station geometry.

``observation_features``
    Direct current LOS, predicted LOS, raw LOS innovation, and observation
    metadata.

The current posterior is deliberately not an accepted input.  Inputs carrying
SI state/position/time values are normalized exactly once here.  The raw LOS,
raw innovation and float64 time differences are returned beside the features
for diagnostics and for downstream heads that need physical units.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import torch
from torch import Tensor, nn

from .normalization import V6Normalization


STATE_DIM = 6
LOS_DIM = 3
STATION_POSITION_DIM = 3
FEATURE_DIM = 32
STATE_FEATURE_DIM = FEATURE_DIM
OBSERVATION_FEATURE_DIM = FEATURE_DIM


def _as_real(value: Any, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> Tensor:
    if isinstance(value, Tensor):
        result = value
        if result.is_complex():
            raise TypeError("complex values are not supported")
        if device is not None and result.device != device:
            result = result.to(device=device)
        if dtype is not None and result.dtype != dtype:
            result = result.to(dtype=dtype)
        if not result.is_floating_point():
            result = result.to(dtype=dtype or torch.get_default_dtype())
        return result
    return torch.as_tensor(value, device=device, dtype=dtype or torch.get_default_dtype())


def _finite_or_zero(value: Tensor) -> Tensor:
    if value.is_floating_point():
        return torch.where(torch.isfinite(value), value, torch.zeros_like(value))
    return value


def _stable_string_code(value: Any) -> float:
    """Map a station label to a deterministic finite code.

    The code is used only when the caller marks the ID reliable.  A cryptographic
    digest avoids Python's process-randomized ``hash`` and keeps runs stable.
    """

    digest = hashlib.sha1(str(value).encode("utf-8", errors="replace")).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return float(integer) / float(2**64 - 1)


def _flatten_nested(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        result: list[Any] = []
        for item in value:
            result.extend(_flatten_nested(item))
        return result
    return [value]


def _mapping_get(mapping: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _set_if_none(current: Any, mapping: Mapping[str, Any], *names: str) -> Any:
    if current is not None:
        return current
    return _mapping_get(mapping, *names)


def _coerce_bt(
    value: Any,
    *,
    batch: int,
    steps: int,
    tail: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    name: str,
    fill: float = 0.0,
) -> tuple[Tensor, bool]:
    """Broadcast an optional ``[B,T,*tail]`` value without detaching it."""

    provided = value is not None
    if value is None:
        return torch.full(
            (batch, steps, *tail),
            float(fill),
            device=device,
            dtype=dtype,
        ), False
    tensor = _as_real(value, device=device, dtype=dtype)
    expected = tuple(tail)
    if tensor.ndim == len(expected) and tuple(tensor.shape) == expected:
        tensor = tensor.reshape((1, 1, *expected)).expand(batch, steps, *expected)
    elif tensor.ndim == len(expected) + 1 and tuple(tensor.shape[1:]) == expected:
        if int(tensor.shape[0]) == steps:
            tensor = tensor.unsqueeze(0).expand(batch, steps, *expected)
        elif int(tensor.shape[0]) == batch and steps == 1:
            tensor = tensor.unsqueeze(1).expand(batch, steps, *expected)
        else:
            raise ValueError(
                f"{name} has shape {tuple(tensor.shape)}; expected [T,*{expected}] "
                f"or [B,T,*{expected}]"
            )
    elif tensor.ndim == len(expected) + 2 and tuple(tensor.shape[-len(expected):]) == expected:
        if tuple(tensor.shape[:2]) != (batch, steps):
            raise ValueError(
                f"{name} has shape {tuple(tensor.shape)}; expected "
                f"[{batch},{steps},*{expected}]"
            )
    elif not expected and tensor.ndim == 0:
        tensor = tensor.reshape(1, 1).expand(batch, steps)
    elif not expected and tensor.ndim == 1:
        if int(tensor.shape[0]) == steps:
            tensor = tensor.unsqueeze(0).expand(batch, steps)
        elif int(tensor.shape[0]) == batch and steps == 1:
            tensor = tensor.unsqueeze(1).expand(batch, steps)
        else:
            raise ValueError(
                f"{name} has shape {tuple(tensor.shape)}; expected [{steps}] or "
                f"[{batch},{steps}]"
            )
    elif not expected and tensor.ndim == 2:
        if tuple(tensor.shape) != (batch, steps):
            raise ValueError(
                f"{name} has shape {tuple(tensor.shape)}; expected [{batch},{steps}]"
            )
    else:
        raise ValueError(
            f"{name} has shape {tuple(tensor.shape)}; expected tail {expected}"
        )
    return tensor, provided


def _coerce_position(
    value: Any,
    *,
    batch: int,
    steps: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> tuple[Tensor, bool]:
    if value is not None:
        raw = _as_real(value, device=device, dtype=dtype)
        if raw.shape[-1:] == (6,):
            value = raw[..., :3]
    return _coerce_bt(
        value,
        batch=batch,
        steps=steps,
        tail=(3,),
        device=device,
        dtype=dtype,
        name=name,
    )


def _coerce_station_state(
    value: Any,
    *,
    batch: int,
    steps: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> tuple[Tensor, bool]:
    """Accept only a station ``[r,v]`` six-dimensional SI state."""

    if value is None:
        return torch.zeros((batch, steps, 6), device=device, dtype=dtype), False
    raw = _as_real(value, device=device, dtype=dtype)
    if raw.shape[-1:] != (6,):
        raise ValueError(
            f"{name} must provide a six-dimensional [position, velocity] SI state; "
            f"got shape {tuple(raw.shape)}"
        )
    value = raw
    return _coerce_bt(
        value,
        batch=batch,
        steps=steps,
        tail=(6,),
        device=device,
        dtype=dtype,
        name=name,
    )


def _coerce_station_ids(
    value: Any,
    *,
    batch: int,
    steps: int,
    device: torch.device,
) -> tuple[Tensor, bool]:
    if value is None:
        return torch.zeros((batch, steps), device=device, dtype=torch.float64), False
    try:
        numeric = torch.as_tensor(value, device=device)
        if numeric.is_complex():
            raise TypeError("complex station IDs are not supported")
        if numeric.is_floating_point() or numeric.is_signed() or numeric.is_unsigned():
            numeric = numeric.to(dtype=torch.float64)
            result, _ = _coerce_bt(
                numeric,
                batch=batch,
                steps=steps,
                tail=(),
                device=device,
                dtype=torch.float64,
                name="station_ids",
            )
            return result, True
    except (TypeError, ValueError, RuntimeError):
        pass
    flat = _flatten_nested(value)
    if len(flat) == 1:
        flat = flat * (batch * steps)
    if len(flat) == steps and batch > 1:
        flat = flat * batch
    if len(flat) != batch * steps:
        raise ValueError(
            f"station_ids has {len(flat)} entries; expected {batch * steps}"
        )
    codes = torch.tensor(
        [_stable_string_code(item) for item in flat],
        device=device,
        dtype=torch.float64,
    ).reshape(batch, steps)
    return codes, True


def _sequence_times(value: Any, *, device: torch.device) -> tuple[Tensor, bool]:
    if value is None:
        raise ValueError("times_sec is required")
    raw = _as_real(value, device=device, dtype=torch.float64)
    if raw.ndim == 0:
        return raw.reshape(1, 1), True
    if raw.ndim == 1:
        return raw.unsqueeze(0), False
    if raw.ndim == 2:
        return raw, False
    raise ValueError(f"times_sec must have shape [], [T] or [B,T], got {tuple(raw.shape)}")


def _infer_device(*values: Any) -> torch.device:
    for value in values:
        if isinstance(value, Tensor):
            return value.device
    return torch.device("cpu")


@dataclass(frozen=True)
class FeatureBatch:
    """Result of causal feature construction.

    The class is attribute-accessible and also supports dictionary-style
    lookup so existing data pipelines can consume it without a conversion.
    Leading dimensions are ``[T]`` or ``[B,T]`` depending on input rank.
    """

    state_features: Tensor
    observation_features: Tensor
    times_sec: Tensor
    relative_time_sec: Tensor
    delta_t_sec: Tensor
    valid_mask: Tensor
    first_frame_missing_mask: Tensor
    prior_state_si: Tensor
    previous_prior_state_si: Tensor
    previous_posterior_state_si: Tensor
    observed_los: Tensor
    delta_los: Tensor
    predicted_los: Tensor
    raw_innovation: Tensor
    station_state_si: Tensor
    station_position_si: Tensor
    station_delta_position_si: Tensor
    station_ids: Tensor
    station_id_reliable: Tensor
    station_id_used: Tensor
    station_changed: Tensor
    station_change_reliable: Tensor
    lengths: Tensor
    stats_scale: Tensor
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def tokens(self) -> Tensor:
        return torch.stack((self.state_features, self.observation_features), dim=-2)

    @property
    def branch_tokens(self) -> Tensor:
        return self.tokens

    @property
    def direct_los(self) -> Tensor:
        return self.observed_los

    @property
    def state_abs_features(self) -> Tensor:
        return self.state_features[..., :6]

    @property
    def state_evolution_features(self) -> Tensor:
        return self.state_features[..., 6:12]

    @property
    def state_update_features(self) -> Tensor:
        return self.state_features[..., 12:18]

    @property
    def los_features(self) -> Tensor:
        return self.observation_features[..., :6]

    @property
    def delta_los_features(self) -> Tensor:
        return self.observation_features[..., 6:9]

    @property
    def innovation_features(self) -> Tensor:
        return self.observation_features[..., 9:12]

    @property
    def station_features(self) -> Tensor:
        return self.observation_features[..., 12:22]

    @property
    def time_features(self) -> Tensor:
        return self.observation_features[..., 22:24]

    @property
    def innovation(self) -> Tensor:
        return self.raw_innovation

    @property
    def mask(self) -> Tensor:
        return self.valid_mask

    @property
    def difference_valid(self) -> Tensor:
        return self.valid_mask & ~self.first_frame_missing_mask

    @property
    def update_valid(self) -> Tensor:
        return self.metadata.get("update_valid", torch.zeros_like(self.valid_mask))

    @property
    def first_missing_mask(self) -> Tensor:
        return self.first_frame_missing_mask

    def __getitem__(self, key: str) -> Any:
        aliases = {
            "state": "state_features",
            "observation": "observation_features",
            "obs": "observation_features",
            "los": "observed_los",
            "direct_los": "observed_los",
            "innovation": "raw_innovation",
            "raw_innovation": "raw_innovation",
            "dt_sec": "delta_t_sec",
            "time_delta_sec": "delta_t_sec",
            "mask": "valid_mask",
            "first_missing_mask": "first_frame_missing_mask",
            "first_frame_missing": "first_frame_missing_mask",
            "station_id_mask": "station_id_used",
            "tokens": "tokens",
        }
        name = aliases.get(str(key), str(key))
        if name == "tokens":
            return self.tokens
        if hasattr(self, name):
            return getattr(self, name)
        if name in self.metadata:
            return self.metadata[name]
        raise KeyError(key)

    def keys(self) -> tuple[str, ...]:
        return (
            "tokens",
            "state_features",
            "observation_features",
            "times_sec",
            "relative_time_sec",
            "delta_t_sec",
            "valid_mask",
            "first_frame_missing_mask",
            "prior_state_si",
            "previous_prior_state_si",
            "previous_posterior_state_si",
            "observed_los",
            "delta_los",
            "predicted_los",
            "raw_innovation",
            "station_position_si",
            "station_state_si",
            "station_delta_position_si",
            "station_ids",
            "station_id_reliable",
            "station_id_used",
            "station_changed",
            "station_change_reliable",
            "difference_valid",
            "update_valid",
            "lengths",
            "stats_scale",
        )

    def as_dict(self) -> dict[str, Any]:
        return {name: self[name] for name in self.keys()}


class FeatureBuilder(nn.Module):
    """Build causal state and observation branches from raw SI sequences."""

    feature_dim = FEATURE_DIM
    state_feature_dim = STATE_FEATURE_DIM
    observation_feature_dim = OBSERVATION_FEATURE_DIM

    def __init__(
        self,
        normalization: V6Normalization | None = None,
        *,
        stats_scales: Sequence[float] | Mapping[str, Any] | None = None,
        stats_json: str | Path | Mapping[str, Any] | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.normalization = normalization or V6Normalization(
            stats_scales=stats_scales,
            stats_json=stats_json,
            dtype=torch.float64,
        )
        self.feature_dtype = dtype

    def _unpack_mapping(
        self,
        times_sec: Any,
        observed_los: Any,
        prior_state_si: Any,
        previous_prior_state_si: Any,
        previous_posterior_state_si: Any,
        station_position_si: Any,
        station_state_si: Any,
        station_ids: Any,
        station_id_reliable: Any,
        predicted_los: Any,
        raw_innovation: Any,
        valid_mask: Any,
        lengths: Any,
        delta_time_seconds: Any,
        init_time_sec: Any,
        metadata: Mapping[str, Any] | None,
        kwargs: MutableMapping[str, Any],
    ) -> tuple[Any, ...]:
        payload: Mapping[str, Any] | None = None
        if isinstance(times_sec, Mapping):
            payload = times_sec
            times_sec = None
        if payload is None and isinstance(observed_los, Mapping):
            payload = observed_los
            observed_los = None
        if payload is not None:
            forbidden_payload = {
                "current_posterior_state_si",
                "current_posterior_state",
                "posterior_state_si",
                "posterior_state",
                "current_posterior",
            }
            leaked_payload = sorted(name for name in forbidden_payload if name in payload)
            if leaked_payload:
                raise ValueError(
                    "current posterior is not a V6 tokenizer input; received "
                    + ", ".join(leaked_payload)
                )
            times_sec = _set_if_none(times_sec, payload, "times_sec", "times", "time_sec", "time")
            observed_los = _set_if_none(
                observed_los, payload, "observed_los", "los", "direct_los", "observation_los"
            )
            prior_state_si = _set_if_none(
                prior_state_si, payload, "prior_state_si", "prior_state", "causal_prior_si", "state_si"
            )
            previous_prior_state_si = _set_if_none(
                previous_prior_state_si,
                payload,
                "previous_prior_state_si",
                "previous_prior_state",
                "prior_prev_state_si",
            )
            previous_posterior_state_si = _set_if_none(
                previous_posterior_state_si,
                payload,
                "previous_posterior_state_si",
                "previous_posterior_state",
                "posterior_prev_state_si",
            )
            station_position_si = _set_if_none(
                station_position_si,
                payload,
                "station_position_si",
                "station_positions_si",
                "observer_position_si",
                "observers_si",
            )
            station_state_si = _set_if_none(
                station_state_si,
                payload,
                "station_state_si",
                "station_state",
                "sat_state",
            )
            if station_state_si is not None:
                # A full station [r,v] lane is richer than a position-only
                # alias and must win when both are present.
                station_position_si = station_state_si
            station_ids = _set_if_none(station_ids, payload, "station_ids", "station_id", "site_ids")
            station_id_reliable = _set_if_none(
                station_id_reliable,
                payload,
                "station_id_reliable",
                "station_ids_reliable",
                "station_id_valid",
            )
            predicted_los = _set_if_none(
                predicted_los, payload, "predicted_los", "los_pred", "predicted_observed_los"
            )
            raw_innovation = _set_if_none(
                raw_innovation, payload, "raw_innovation", "innovation", "los_innovation"
            )
            valid_mask = _set_if_none(valid_mask, payload, "valid_mask", "mask", "observation_mask")
            lengths = _set_if_none(lengths, payload, "lengths", "sequence_lengths", "length")
            delta_time_seconds = _set_if_none(
                delta_time_seconds,
                payload,
                "delta_time_seconds",
                "delta_t_sec",
                "time_delta_sec",
            )
            init_time_sec = _set_if_none(
                init_time_sec,
                payload,
                "init_time_sec",
                "initial_time_sec",
                "t_init_seconds",
            )
            if metadata is None:
                candidate = payload.get("metadata")
                metadata = candidate if isinstance(candidate, Mapping) else None
        aliases = {
            "times": "times_sec",
            "time_sec": "times_sec",
            "observations": "observed_los",
            "los": "observed_los",
            "prior": "prior_state_si",
            "causal_prior": "prior_state_si",
            "previous_prior": "previous_prior_state_si",
            "previous_posterior": "previous_posterior_state_si",
            "station_positions": "station_position_si",
            "observer_positions_si": "station_position_si",
            "station_id_mask": "station_id_reliable",
            "predicted": "predicted_los",
            "innovation": "raw_innovation",
        }
        locals_map = {
            "times_sec": times_sec,
            "observed_los": observed_los,
            "prior_state_si": prior_state_si,
            "previous_prior_state_si": previous_prior_state_si,
            "previous_posterior_state_si": previous_posterior_state_si,
            "station_position_si": station_position_si,
            "station_state_si": station_state_si,
            "station_ids": station_ids,
            "station_id_reliable": station_id_reliable,
            "predicted_los": predicted_los,
            "raw_innovation": raw_innovation,
            "valid_mask": valid_mask,
            "lengths": lengths,
            "delta_time_seconds": delta_time_seconds,
            "init_time_sec": init_time_sec,
        }
        if locals_map["station_state_si"] is not None:
            # Prefer the full [r,v] station state over a position-only alias.
            locals_map["station_position_si"] = locals_map["station_state_si"]
        for old, new in aliases.items():
            if locals_map[new] is None and old in kwargs:
                locals_map[new] = kwargs.pop(old)
        forbidden = (
            "current_posterior_state_si",
            "current_posterior_state",
            "posterior_state_si",
            "posterior_state",
            "current_posterior",
        )
        leaked = [name for name in forbidden if name in kwargs]
        if leaked:
            raise ValueError(
                "current posterior is not a V6 tokenizer input; received " + ", ".join(leaked)
            )
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected feature-builder inputs: {unknown}")
        return (
            locals_map["times_sec"],
            locals_map["observed_los"],
            locals_map["prior_state_si"],
            locals_map["previous_prior_state_si"],
            locals_map["previous_posterior_state_si"],
            locals_map["station_position_si"],
            locals_map["station_ids"],
            locals_map["station_id_reliable"],
            locals_map["predicted_los"],
            locals_map["raw_innovation"],
            locals_map["valid_mask"],
            locals_map["lengths"],
            metadata,
            locals_map["delta_time_seconds"],
            locals_map["init_time_sec"],
        )

    def _build_sequence(
        self,
        *,
        times_sec: Any,
        observed_los: Any,
        prior_state_si: Any,
        previous_prior_state_si: Any,
        previous_posterior_state_si: Any,
        station_position_si: Any,
        station_ids: Any,
        station_id_reliable: Any,
        predicted_los: Any,
        raw_innovation: Any,
        valid_mask: Any,
        lengths: Any,
        delta_time_seconds: Any,
        init_time_sec: Any,
        metadata: Mapping[str, Any] | None,
        squeeze_batch: bool,
        squeeze_time: bool = False,
    ) -> FeatureBatch:
        device = _infer_device(
            times_sec,
            observed_los,
            prior_state_si,
            previous_prior_state_si,
            previous_posterior_state_si,
            station_position_si,
        )
        times, scalar_time = _sequence_times(times_sec, device=device)
        batch, steps = int(times.shape[0]), int(times.shape[1])
        if squeeze_batch and batch != 1:
            raise ValueError("single-step input cannot have multiple batches")
        # Feature arithmetic follows caller state/LOS dtype; float64 is used
        # only for the timestamp lane as required by the time contract.
        feature_dtype = self.feature_dtype
        for candidate in (prior_state_si, observed_los, station_position_si):
            if isinstance(candidate, Tensor) and candidate.is_floating_point():
                feature_dtype = candidate.dtype
                break
        observed, observed_present = _coerce_bt(
            observed_los,
            batch=batch,
            steps=steps,
            tail=(3,),
            device=device,
            dtype=feature_dtype,
            name="observed_los",
        )
        if not observed_present:
            raise ValueError("observed_los is required")
        prior, prior_present = _coerce_bt(
            prior_state_si,
            batch=batch,
            steps=steps,
            tail=(6,),
            device=device,
            dtype=feature_dtype,
            name="prior_state_si",
        )
        if not prior_present:
            raise ValueError("prior_state_si (causal current prior) is required")
        previous_prior, previous_prior_present = _coerce_bt(
            previous_prior_state_si,
            batch=batch,
            steps=steps,
            tail=(6,),
            device=device,
            dtype=feature_dtype,
            name="previous_prior_state_si",
        )
        previous_posterior, previous_posterior_present = _coerce_bt(
            previous_posterior_state_si,
            batch=batch,
            steps=steps,
            tail=(6,),
            device=device,
            dtype=feature_dtype,
            name="previous_posterior_state_si",
        )
        previous_prior_finite = torch.isfinite(previous_prior).all(dim=-1)
        previous_posterior_finite = torch.isfinite(previous_posterior).all(dim=-1)
        # If no explicit previous prior is supplied, derive it causally from
        # the preceding current prior.  This never reads a future/current
        # posterior and keeps the operation differentiable.
        if not previous_prior_present:
            previous_prior = torch.cat((torch.zeros_like(prior[:, :1]), prior[:, :-1]), dim=1)
            previous_prior_finite = torch.isfinite(previous_prior).all(dim=-1)
        station_state, station_position_present = _coerce_station_state(
            station_position_si,
            batch=batch,
            steps=steps,
            device=device,
            dtype=feature_dtype,
            name="station_position_si",
        )
        station_position = station_state[..., :3]
        station_values, station_values_present = _coerce_station_ids(
            station_ids, batch=batch, steps=steps, device=device
        )
        station_reliable, station_reliable_present = _coerce_bt(
            station_id_reliable,
            batch=batch,
            steps=steps,
            tail=(),
            device=device,
            dtype=torch.float32,
            name="station_id_reliable",
            # A station ID is categorical metadata, not a trusted signal by
            # itself.  It is consumed only when the caller explicitly marks it
            # reliable.
            fill=0.0,
        )
        station_reliable = station_reliable > 0.5
        finite_station_id = torch.isfinite(station_values)
        station_id_reliable_effective = station_reliable & finite_station_id
        station_id_used = station_id_reliable_effective
        station_values = torch.where(station_id_used, station_values, torch.zeros_like(station_values))
        pred, predicted_present = _coerce_bt(
            predicted_los,
            batch=batch,
            steps=steps,
            tail=(3,),
            device=device,
            dtype=feature_dtype,
            name="predicted_los",
        )
        pred_finite = torch.isfinite(pred).all(dim=-1)
        pred = _finite_or_zero(pred)
        raw, raw_present = _coerce_bt(
            raw_innovation,
            batch=batch,
            steps=steps,
            tail=(3,),
            device=device,
            dtype=feature_dtype,
            name="raw_innovation",
        )
        if not raw_present:
            raw = observed - pred
        raw_finite = torch.isfinite(raw).all(dim=-1)
        raw = _finite_or_zero(raw)

        # Compute masks before sanitizing invalid padded values.  A finite raw
        # value can still be marked invalid by the caller's variable-length
        # mask, and a masked NaN cannot contaminate features.
        times_finite = torch.isfinite(times).to(dtype=torch.bool)
        times = _finite_or_zero(times)
        observed_finite = torch.isfinite(observed).all(dim=-1)
        prior_finite = torch.isfinite(prior).all(dim=-1)
        if valid_mask is None:
            valid = torch.ones((batch, steps), device=device, dtype=torch.bool)
        else:
            valid, _ = _coerce_bt(
                valid_mask,
                batch=batch,
                steps=steps,
                tail=(),
                device=device,
                dtype=torch.float32,
                name="valid_mask",
            )
            valid = valid > 0.5
        if lengths is not None:
            length_tensor = torch.as_tensor(lengths, device=device, dtype=torch.long).reshape(-1)
            if length_tensor.numel() == 1 and batch > 1:
                length_tensor = length_tensor.expand(batch)
            if int(length_tensor.numel()) != batch:
                raise ValueError(f"lengths must have {batch} values")
            if bool(((length_tensor < 0) | (length_tensor > steps)).any()):
                raise ValueError(f"lengths must lie in [0,{steps}]")
            positions = torch.arange(steps, device=device).reshape(1, steps)
            valid = valid & (positions < length_tensor.reshape(batch, 1))
        valid = valid & times_finite & observed_finite & prior_finite
        station_state_finite = torch.isfinite(station_state).all(dim=-1)
        station_available = bool(station_position_present)
        if not station_available and bool(valid.any()):
            raise ValueError(
                "station_position_si must provide a six-dimensional [position, velocity] SI state"
            )
        if station_available and bool((valid & ~station_state_finite).any()):
            raise ValueError(
                "station_position_si contains non-finite position/velocity on a valid frame"
            )
        valid = valid & station_state_finite
        if not station_available:
            valid = valid & torch.zeros_like(valid)
        if not predicted_present and bool(valid.any()):
            raise ValueError("predicted_los is required for every valid frame")
        if predicted_present and bool((valid & ~pred_finite).any()):
            raise ValueError("predicted_los contains non-finite values on a valid frame")
        # Innovation has one physical definition in V6: e = y - y_hat.  A
        # caller may repeat it as a diagnostic input, but it must agree on
        # every valid frame.  Padding/invalid frames are deliberately exempt
        # so NaN sentinels there cannot poison an otherwise valid sequence.
        expected_raw = observed - pred
        if raw_present:
            if bool((valid & ~raw_finite).any()):
                raise ValueError("raw_innovation contains non-finite values on a valid frame")
            raw_close = torch.isclose(
                raw,
                expected_raw,
                rtol=1.0e-5,
                atol=1.0e-8,
                equal_nan=False,
            ).all(dim=-1)
            if bool((valid & ~raw_close).any()):
                raise ValueError(
                    "raw_innovation must equal observed_los - predicted_los on valid frames"
                )
        # Always carry the canonical differentiable expression downstream,
        # including when the optional diagnostic input was supplied.
        raw = _finite_or_zero(expected_raw)
        raw_finite = torch.isfinite(raw).all(dim=-1)
        station_id_used = station_id_reliable_effective & valid
        station_values = torch.where(station_id_used, station_values, torch.zeros_like(station_values))
        lengths_out = valid.to(dtype=torch.long).sum(dim=-1)

        delta_t = torch.zeros_like(times, dtype=torch.float64)
        if steps > 1:
            differences = times[:, 1:] - times[:, :-1]
            adjacent = valid[:, 1:] & valid[:, :-1]
            delta_t[:, 1:] = torch.where(adjacent, differences, torch.zeros_like(differences))
        if delta_time_seconds is not None:
            supplied_delta, _ = _coerce_bt(
                delta_time_seconds,
                batch=batch,
                steps=steps,
                tail=(),
                device=device,
                dtype=torch.float64,
                name="delta_time_seconds",
            )
            supplied_delta = _finite_or_zero(supplied_delta)
            delta_t = torch.where(valid, supplied_delta, torch.zeros_like(supplied_delta))
        previous_valid = torch.zeros_like(valid)
        if steps > 1:
            previous_valid[:, 1:] = valid[:, :-1]
        first_missing = valid & ~previous_valid
        prior_valid = valid & prior_finite
        previous_prior_used = valid & previous_prior_finite
        if not previous_prior_present:
            previous_prior_used = previous_prior_used & previous_valid
        previous_posterior_used = valid & previous_posterior_present & previous_posterior_finite
        update_valid = previous_prior_used & previous_posterior_used
        previous_prior = _finite_or_zero(previous_prior)
        previous_posterior = _finite_or_zero(previous_posterior)
        prior = _finite_or_zero(prior)
        delta_los = torch.zeros_like(observed)
        if steps > 1:
            los_differences = observed[:, 1:] - observed[:, :-1]
            adjacent = valid[:, 1:] & valid[:, :-1]
            delta_los[:, 1:] = torch.where(
                adjacent.unsqueeze(-1),
                los_differences,
                torch.zeros_like(los_differences),
            )
        station_position_finite = torch.isfinite(station_position).all(dim=-1)
        station_state = _finite_or_zero(station_state)
        station_position = _finite_or_zero(station_position)
        station_position_used = valid & station_position_present & station_state_finite

        prev_station_ids = torch.zeros_like(station_values)
        prev_station_used = torch.zeros_like(station_id_used)
        if steps > 1:
            prev_station_ids[:, 1:] = station_values[:, :-1]
            prev_station_used[:, 1:] = station_id_used[:, :-1] & previous_valid[:, 1:]
        station_change_reliable = station_id_used & prev_station_used & previous_valid
        station_changed = station_change_reliable & (station_values != prev_station_ids)
        difference_valid = valid & previous_valid
        predicted_available = valid & pred_finite & bool(predicted_present)
        raw_available = valid & raw_finite
        station_delta_position = torch.zeros_like(station_position)
        if steps > 1:
            station_differences = station_position[:, 1:] - station_position[:, :-1]
            adjacent = valid[:, 1:] & valid[:, :-1]
            station_delta_position[:, 1:] = torch.where(
                adjacent.unsqueeze(-1),
                station_differences,
                torch.zeros_like(station_differences),
            )

        # ``init_time_sec`` is an optional causal origin.  When it is
        # supplied, the first frame has a real elapsed interval from that
        # origin; it is not silently assigned ``dt=0``.  An explicitly
        # supplied ``delta_time_seconds`` remains authoritative.
        if init_time_sec is None:
            initial_time = times[:, :1]
        else:
            init_raw = _as_real(init_time_sec, device=device, dtype=torch.float64)
            if init_raw.ndim == 0:
                initial_time = init_raw.reshape(1, 1).expand(batch, 1)
            elif init_raw.ndim == 1 and int(init_raw.numel()) in (1, batch):
                initial_time = init_raw.reshape(-1, 1)
                if int(initial_time.shape[0]) == 1 and batch > 1:
                    initial_time = initial_time.expand(batch, 1)
            elif init_raw.ndim == 2 and tuple(init_raw.shape) == (batch, 1):
                initial_time = init_raw
            else:
                raise ValueError(
                    f"init_time_sec must be scalar, [B] or [B,1], got {tuple(init_raw.shape)}"
                )
        initial_time = _finite_or_zero(initial_time)
        if init_time_sec is not None and delta_time_seconds is None:
            first_delta = times[:, :1] - initial_time
            delta_t[:, :1] = torch.where(
                valid[:, :1], first_delta, torch.zeros_like(first_delta)
            )

        # Normalization happens exactly once on SI state/position/time values.
        prior_norm = self.normalization.normalize_state(prior)
        previous_prior_norm = self.normalization.normalize_state(previous_prior)
        previous_posterior_norm = self.normalization.normalize_state(previous_posterior)
        stats_scale = self.normalization.stats_scale.to(device=device, dtype=feature_dtype)
        state_evolution_norm = prior_norm - previous_posterior_norm
        state_evolution_norm = torch.where(
            previous_posterior_used.unsqueeze(-1),
            state_evolution_norm,
            torch.zeros_like(state_evolution_norm),
        )
        state_update_norm = previous_posterior_norm - previous_prior_norm
        state_update_norm = torch.where(
            update_valid.unsqueeze(-1),
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
        station_position_norm = self.normalization.normalize_position(station_position)
        station_state_norm = self.normalization.normalize_state(station_state)
        station_delta_position_norm = self.normalization.normalize_position(station_delta_position)
        dt_norm = self.normalization.normalize_time(delta_t).to(dtype=feature_dtype)
        observed_direct = self.normalization.normalize_los(_finite_or_zero(observed))
        predicted_direct = self.normalization.normalize_los(pred)
        delta_los_direct = self.normalization.normalize_los(delta_los) / stats_scale[0]
        innovation_direct = self.normalization.normalize_los(raw) / stats_scale[1]
        zero = torch.zeros((batch, steps, 1), device=device, dtype=feature_dtype)
        one_if = lambda mask: mask.to(dtype=feature_dtype).unsqueeze(-1)
        station_code = station_values.to(dtype=feature_dtype).unsqueeze(-1)
        station_code = torch.where(one_if(station_id_used).bool(), station_code, torch.zeros_like(station_code))

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
            one_if(station_id_used),
            station_code,
            one_if(update_valid),
            one_if(station_change_reliable),
            one_if(station_position_used),
        ]
        state_features = torch.cat(state_parts, dim=-1)
        elapsed_sec = times - initial_time
        elapsed_sec = torch.where(valid, elapsed_sec, torch.zeros_like(elapsed_sec))
        elapsed_norm = self.normalization.normalize_time(elapsed_sec).to(dtype=feature_dtype)
        los_abs = torch.cat((observed_direct, predicted_direct), dim=-1)
        station_direct = torch.cat(
            (
                station_state_norm,
                station_delta_position_norm,
                one_if(station_changed),
            ),
            dim=-1,
        )
        time_direct = torch.cat((dt_norm.unsqueeze(-1), elapsed_norm.unsqueeze(-1)), dim=-1)
        observation_parts = [
            los_abs,
            delta_los_direct,
            innovation_direct,
            station_direct,
            time_direct,
            one_if(valid),
            one_if(first_missing),
            one_if(difference_valid),
            one_if(station_id_used),
            station_code,
            one_if(station_change_reliable),
            one_if(station_position_used),
            one_if(times_finite),
        ]
        observation_features = torch.cat(observation_parts, dim=-1)
        if state_features.shape[-1] > FEATURE_DIM or observation_features.shape[-1] > FEATURE_DIM:
            raise RuntimeError("V6 feature layout exceeded the declared feature dimension")
        if state_features.shape[-1] < FEATURE_DIM:
            state_features = torch.cat(
                (state_features, torch.zeros((*state_features.shape[:-1], FEATURE_DIM - state_features.shape[-1]), device=device, dtype=feature_dtype)),
                dim=-1,
            )
        if observation_features.shape[-1] < FEATURE_DIM:
            observation_features = torch.cat(
                (observation_features, torch.zeros((*observation_features.shape[:-1], FEATURE_DIM - observation_features.shape[-1]), device=device, dtype=feature_dtype)),
                dim=-1,
            )

        canonical_metadata: dict[str, Any] = dict(metadata or {})
        canonical_metadata.update(
            {
                "causal_prior_used": prior_valid,
                "previous_prior_used": previous_prior_used,
                "previous_posterior_used": previous_posterior_used,
                "update_valid": update_valid,
                "current_posterior_used": torch.zeros_like(valid),
                "direct_los": observed_direct,
                "delta_los": delta_los,
                "raw_innovation": raw,
                "station_delta_position_si": station_delta_position,
                "time_features": time_direct,
            "delta_t_sec": delta_t,
            "relative_time_sec": elapsed_sec,
                "station_id_used": station_id_used,
                "station_change_reliable": station_change_reliable,
                "difference_valid": difference_valid,
            }
        )

        def unbatch(value: Tensor) -> Tensor:
            if squeeze_batch:
                value = value[0]
            if squeeze_time:
                value = value[0]
            return value

        return FeatureBatch(
            state_features=unbatch(state_features),
            observation_features=unbatch(observation_features),
            times_sec=unbatch(times),
            relative_time_sec=unbatch(elapsed_sec),
            delta_t_sec=unbatch(delta_t),
            valid_mask=unbatch(valid),
            first_frame_missing_mask=unbatch(first_missing),
            prior_state_si=unbatch(prior),
            previous_prior_state_si=unbatch(previous_prior),
            previous_posterior_state_si=unbatch(previous_posterior),
            observed_los=unbatch(observed_direct),
            delta_los=unbatch(delta_los),
            predicted_los=unbatch(predicted_direct),
            raw_innovation=unbatch(raw),
            station_state_si=unbatch(station_state),
            station_position_si=unbatch(station_position),
            station_delta_position_si=unbatch(station_delta_position),
            station_ids=unbatch(station_values),
            station_id_reliable=unbatch(station_id_reliable_effective),
            station_id_used=unbatch(station_id_used),
            station_changed=unbatch(station_changed),
            station_change_reliable=unbatch(station_change_reliable),
            lengths=lengths_out[0] if squeeze_batch else lengths_out,
            stats_scale=self.normalization.stats_scale,
            metadata={
                key: (
                    unbatch(value)
                    if isinstance(value, Tensor)
                    and value.ndim > 0
                    and (
                        (squeeze_batch and value.shape[0] == batch)
                        or (squeeze_time and value.shape[0] == steps)
                    )
                    else value
                )
                for key, value in canonical_metadata.items()
            },
        )

    def forward_sequence(
        self,
        times_sec: Any = None,
        observed_los: Any = None,
        prior_state_si: Any = None,
        previous_prior_state_si: Any = None,
        previous_posterior_state_si: Any = None,
        station_position_si: Any = None,
        station_state_si: Any = None,
        station_ids: Any = None,
        station_id_reliable: Any = None,
        predicted_los: Any = None,
        raw_innovation: Any = None,
        valid_mask: Any = None,
        lengths: Any = None,
        delta_time_seconds: Any = None,
        init_time_sec: Any = None,
        t_init_seconds: Any = None,
        t_init_sec: Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> FeatureBatch:
        if init_time_sec is None:
            init_time_sec = t_init_seconds if t_init_seconds is not None else t_init_sec
        values = self._unpack_mapping(
            times_sec,
            observed_los,
            prior_state_si,
            previous_prior_state_si,
            previous_posterior_state_si,
            station_position_si,
            station_state_si,
            station_ids,
            station_id_reliable,
            predicted_los,
            raw_innovation,
            valid_mask,
            lengths,
            delta_time_seconds,
            init_time_sec,
            metadata,
            {},
        )
        raw_time_rank = torch.as_tensor(values[0]).ndim
        return self._build_sequence(
            times_sec=values[0],
            observed_los=values[1],
            prior_state_si=values[2],
            previous_prior_state_si=values[3],
            previous_posterior_state_si=values[4],
            station_position_si=values[5],
            station_ids=values[6],
            station_id_reliable=values[7],
            predicted_los=values[8],
            raw_innovation=values[9],
            valid_mask=values[10],
            lengths=values[11],
            metadata=values[12],
            delta_time_seconds=values[13],
            init_time_sec=values[14],
            squeeze_batch=raw_time_rank == 1,
        )

    def forward_step(
        self,
        time_sec: Any = None,
        observed_los: Any = None,
        prior_state_si: Any = None,
        previous_prior_state_si: Any = None,
        previous_posterior_state_si: Any = None,
        station_position_si: Any = None,
        station_state_si: Any = None,
        station_id: Any = None,
        station_id_reliable: Any = None,
        predicted_los: Any = None,
        raw_innovation: Any = None,
        valid: Any = True,
        delta_time_seconds: Any = None,
        init_time_sec: Any = None,
        t_init_seconds: Any = None,
        t_init_sec: Any = None,
        metadata: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> FeatureBatch:
        if "times_sec" in kwargs:
            time_sec = kwargs.pop("times_sec")
        if "station_ids" in kwargs:
            station_id = kwargs.pop("station_ids")
        if "valid_mask" in kwargs:
            valid = kwargs.pop("valid_mask")
        if init_time_sec is None:
            init_time_sec = t_init_seconds if t_init_seconds is not None else t_init_sec
        return self._build_sequence(
            times_sec=time_sec,
            observed_los=observed_los,
            prior_state_si=prior_state_si,
            previous_prior_state_si=previous_prior_state_si,
            previous_posterior_state_si=previous_posterior_state_si,
            station_position_si=station_state_si if station_state_si is not None else station_position_si,
            station_ids=station_id,
            station_id_reliable=station_id_reliable,
            predicted_los=predicted_los,
            raw_innovation=raw_innovation,
            valid_mask=valid,
            lengths=None,
            delta_time_seconds=delta_time_seconds,
            init_time_sec=init_time_sec,
            metadata=metadata,
            squeeze_batch=True,
            squeeze_time=True,
        )

    def forward(self, times_sec: Any = None, observed_los: Any = None, **kwargs: Any) -> FeatureBatch:
        # Mapping input is always a sequence unless its time value is scalar.
        candidate_times = times_sec
        candidate_los = observed_los
        if isinstance(candidate_times, Mapping):
            payload = candidate_times
            candidate_times = _mapping_get(payload, "times_sec", "times", "time_sec", "time")
            candidate_los = _mapping_get(payload, "observed_los", "los", "direct_los")
            values = self._unpack_mapping(
                times_sec=payload,
                observed_los=None,
                prior_state_si=None,
                previous_prior_state_si=None,
                previous_posterior_state_si=None,
                station_position_si=None,
                station_state_si=None,
                station_ids=None,
                station_id_reliable=None,
                predicted_los=None,
                raw_innovation=None,
                valid_mask=None,
                lengths=None,
                delta_time_seconds=None,
                init_time_sec=None,
                metadata=None,
                kwargs=dict(kwargs),
            )
            raw_time_rank = torch.as_tensor(values[0]).ndim
            los_rank = torch.as_tensor(values[1]).ndim if values[1] is not None else 0
            is_step = raw_time_rank == 0 or los_rank == 1
            return self._build_sequence(
                times_sec=values[0],
                observed_los=values[1],
                prior_state_si=values[2],
                previous_prior_state_si=values[3],
                previous_posterior_state_si=values[4],
                station_position_si=values[5],
                station_ids=values[6],
                station_id_reliable=values[7],
                predicted_los=values[8],
                raw_innovation=values[9],
                valid_mask=values[10],
                lengths=values[11],
                metadata=values[12],
                delta_time_seconds=values[13],
                init_time_sec=values[14],
                squeeze_batch=True if is_step else raw_time_rank == 1,
                squeeze_time=is_step,
            )
        try:
            time_tensor = torch.as_tensor(candidate_times)
            los_tensor = torch.as_tensor(candidate_los) if candidate_los is not None else None
            is_step = time_tensor.ndim == 0 or (los_tensor is not None and los_tensor.ndim == 1)
        except (TypeError, ValueError, RuntimeError):
            is_step = False
        if is_step:
            return self.forward_step(time_sec=times_sec, observed_los=observed_los, **kwargs)
        return self.forward_sequence(times_sec=times_sec, observed_los=observed_los, **kwargs)

    build = forward
    build_sequence = forward_sequence
    build_step = forward_step


V6FeatureBuilder = FeatureBuilder
ObservationFeatureBuilder = FeatureBuilder


__all__ = [
    "STATE_DIM",
    "LOS_DIM",
    "STATION_POSITION_DIM",
    "FEATURE_DIM",
    "STATE_FEATURE_DIM",
    "OBSERVATION_FEATURE_DIM",
    "FeatureBatch",
    "FeatureBuilder",
    "V6FeatureBuilder",
    "ObservationFeatureBuilder",
]

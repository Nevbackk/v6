"""Unit-safe, fixed normalization for the V6 observation tokenizer.

The tokenizer receives physical SI values.  This module is intentionally
small and deterministic: the orbital scales are derived from the constants in
the V6 contract and the four statistical scales are non-trainable buffers.
No input is detached or converted through NumPy, so all arithmetic remains
differentiable with respect to the caller's tensors.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn


MU_EARTH_M3_S2: float = 3.986004418e14
EARTH_RADIUS_M: float = 6378.14e3
R_EARTH_M: float = EARTH_RADIUS_M
V_SCALE_MPS: float = math.sqrt(MU_EARTH_M3_S2 / EARTH_RADIUS_M)
T_SCALE_S: float = EARTH_RADIUS_M / V_SCALE_MPS

# Short names match the notation used in the design document.
R: float = EARTH_RADIUS_M
V: float = V_SCALE_MPS
T: float = T_SCALE_S

# Statistics are not guessed by the tokenizer.  They are four fixed calibration
# values supplied by the caller (or loaded from a JSON calibration artifact),
# in this exact order.
STAT_SCALE_NAMES: tuple[str, str, str, str] = (
    "delta_los",
    "innovation",
    "update_pos",
    "update_vel",
)


def _floating_tensor(
    value: Any,
    *,
    like: Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Convert a value to a real tensor while retaining autograd history."""

    if isinstance(value, Tensor):
        result = value
        if result.is_complex():
            raise TypeError("complex values are not supported")
        if like is not None and result.device != like.device:
            result = result.to(device=like.device)
        target_dtype = dtype
        if target_dtype is None and like is not None and like.is_floating_point():
            target_dtype = like.dtype
        if target_dtype is not None and result.dtype != target_dtype:
            result = result.to(dtype=target_dtype)
        if not result.is_floating_point():
            result = result.to(dtype=target_dtype or torch.get_default_dtype())
        return result
    target_dtype = dtype
    if target_dtype is None and like is not None and like.is_floating_point():
        target_dtype = like.dtype
    if target_dtype is None:
        target_dtype = torch.get_default_dtype()
    return torch.as_tensor(
        value,
        dtype=target_dtype,
        device=None if like is None else like.device,
    )


def _positive_tuple(
    values: Iterable[float], *, name: str, length: int
) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != int(length):
        raise ValueError(f"{name} must contain exactly {length} values")
    if any(not math.isfinite(value) or value <= 0.0 for value in result):
        raise ValueError(f"{name} must contain finite positive values")
    return result


def _stats_from_source(source: Sequence[float] | Mapping[str, Any] | str | Path) -> tuple[float, ...]:
    """Read the four named calibration scales from a sequence or JSON.

    JSON may be either ``{"delta_los": ..., ...}``,
    ``{"stats_scales": { ... }}``, or ``{"stats_scales": [...]}``.  Nested
    records with a ``scale`` field are accepted to keep provenance-rich
    calibration manifests readable.
    """

    value: Any = source
    if isinstance(source, (str, Path)):
        with Path(source).expanduser().open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    if isinstance(value, Mapping):
        nested = value.get("stats_scales", value.get("statistics", value))
        if isinstance(nested, Mapping):
            values: list[float] = []
            for name in STAT_SCALE_NAMES:
                if name not in nested:
                    raise ValueError(f"stats calibration is missing {name!r}")
                item = nested[name]
                if isinstance(item, Mapping):
                    item = item.get("scale", item.get("value"))
                values.append(float(item))
            return tuple(values)
        value = nested
    if isinstance(value, (str, bytes)):
        raise TypeError("stats calibration must be a four-value sequence or named mapping")
    return tuple(float(item) for item in value)


@dataclass(frozen=True)
class PhysicalScales:
    """Canonical orbital scales used by all V6 branches."""

    radius_m: float = EARTH_RADIUS_M
    mu_m3_s2: float = MU_EARTH_M3_S2

    def __post_init__(self) -> None:
        radius = float(self.radius_m)
        mu = float(self.mu_m3_s2)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius_m must be finite and positive")
        if not math.isfinite(mu) or mu <= 0.0:
            raise ValueError("mu_m3_s2 must be finite and positive")

    @property
    def R(self) -> float:
        return float(self.radius_m)

    @property
    def V(self) -> float:
        return math.sqrt(float(self.mu_m3_s2) / float(self.radius_m))

    @property
    def T(self) -> float:
        return float(self.radius_m) / self.V

    @property
    def state_scale(self) -> tuple[float, float, float, float, float, float]:
        return (self.R, self.R, self.R, self.V, self.V, self.V)

    def tensor(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> Tensor:
        return torch.tensor(self.state_scale, device=device, dtype=dtype)

    def normalize_state(self, state_si: Any) -> Tensor:
        state = _floating_tensor(state_si)
        if state.shape[-1:] != (6,):
            raise ValueError(f"state_si must end in six values, got {tuple(state.shape)}")
        return state / self.tensor(device=state.device, dtype=state.dtype)

    def denormalize_state(self, state_normalized: Any) -> Tensor:
        state = _floating_tensor(state_normalized)
        if state.shape[-1:] != (6,):
            raise ValueError(
                "state_normalized must end in six values, "
                f"got {tuple(state.shape)}"
            )
        return state * self.tensor(device=state.device, dtype=state.dtype)

    def normalize_position(self, position_si: Any) -> Tensor:
        position = _floating_tensor(position_si)
        if position.shape[-1:] != (3,):
            raise ValueError(
                f"position_si must end in three values, got {tuple(position.shape)}"
            )
        return position / float(self.R)

    def denormalize_position(self, position_normalized: Any) -> Tensor:
        position = _floating_tensor(position_normalized)
        if position.shape[-1:] != (3,):
            raise ValueError(
                "position_normalized must end in three values, "
                f"got {tuple(position.shape)}"
            )
        return position * float(self.R)

    def normalize_velocity(self, velocity_si: Any) -> Tensor:
        velocity = _floating_tensor(velocity_si)
        if velocity.shape[-1:] != (3,):
            raise ValueError(
                f"velocity_si must end in three values, got {tuple(velocity.shape)}"
            )
        return velocity / float(self.V)

    def denormalize_velocity(self, velocity_normalized: Any) -> Tensor:
        velocity = _floating_tensor(velocity_normalized)
        if velocity.shape[-1:] != (3,):
            raise ValueError(
                "velocity_normalized must end in three values, "
                f"got {tuple(velocity.shape)}"
            )
        return velocity * float(self.V)

    def normalize_time(self, time_seconds: Any) -> Tensor:
        value = _floating_tensor(time_seconds)
        return value / float(self.T)


class V6Normalization(nn.Module):
    """Torch module exposing the frozen V6 normalization contract.

    Inputs are always interpreted as raw SI values.  The module does not try
    to infer whether a caller has already normalized an input; this avoids a
    silent second normalization.  ``stats_scale`` is a four-element buffer,
    not a parameter, and therefore cannot be changed by an optimizer.
    """

    def __init__(
        self,
        *,
        physical_scales: PhysicalScales | None = None,
        stats_scales: Sequence[float] | Mapping[str, Any] | None = None,
        stats_json: str | Path | Mapping[str, Any] | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        physical = physical_scales or PhysicalScales()
        if stats_scales is not None and stats_json is not None:
            raise ValueError("provide either stats_scales or stats_json, not both")
        source = stats_scales if stats_scales is not None else stats_json
        if source is None:
            raise ValueError(
                "V6Normalization requires four fixed positive statistics scales "
                f"named {STAT_SCALE_NAMES}; pass stats_scales or stats_json"
            )
        stats = _positive_tuple(
            _stats_from_source(source), name="stats_scales", length=4
        )
        self.physical = physical
        self.register_buffer(
            "state_scale",
            torch.tensor(physical.state_scale, dtype=dtype),
            persistent=True,
        )

        self.stats_scale_names = STAT_SCALE_NAMES
        self.register_buffer(
            "position_scale",
            torch.tensor(float(physical.R), dtype=dtype),
            persistent=True,
        )
        self.register_buffer(
            "velocity_scale",
            torch.tensor(float(physical.V), dtype=dtype),
            persistent=True,
        )
        self.register_buffer(
            "time_scale",
            torch.tensor(float(physical.T), dtype=dtype),
            persistent=True,
        )
        self.register_buffer(
            "stats_scale",
            torch.tensor(stats, dtype=dtype),
            persistent=True,
        )

    @property
    def R(self) -> float:
        return float(self.physical.R)

    @property
    def V(self) -> float:
        return float(self.physical.V)

    @property
    def T(self) -> float:
        return float(self.physical.T)

    @property
    def stats_scales(self) -> Tensor:
        return self.stats_scale

    @classmethod
    def from_stats_json(
        cls,
        source: str | Path | Mapping[str, Any],
        *,
        physical_scales: PhysicalScales | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> "V6Normalization":
        return cls(
            physical_scales=physical_scales,
            stats_json=source,
            dtype=dtype,
        )

    def _scale_like(self, scale: Tensor, value: Tensor) -> Tensor:
        return scale.to(device=value.device, dtype=value.dtype)

    def normalize_state(self, state_si: Any) -> Tensor:
        state = _floating_tensor(state_si)
        if state.shape[-1:] != (6,):
            raise ValueError(f"state_si must end in six values, got {tuple(state.shape)}")
        return state / self._scale_like(self.state_scale, state)

    normalize_state_si = normalize_state

    def denormalize_state(self, state_normalized: Any) -> Tensor:
        state = _floating_tensor(state_normalized)
        if state.shape[-1:] != (6,):
            raise ValueError(
                "state_normalized must end in six values, "
                f"got {tuple(state.shape)}"
            )
        return state * self._scale_like(self.state_scale, state)

    def denormalize_state_si(self, state_normalized: Any) -> Tensor:
        return self.denormalize_state(state_normalized)

    def normalize_position(self, position_si: Any) -> Tensor:
        position = _floating_tensor(position_si)
        if position.shape[-1:] != (3,):
            raise ValueError(
                f"position_si must end in three values, got {tuple(position.shape)}"
            )
        return position / self._scale_like(self.position_scale, position)

    normalize_position_si = normalize_position

    def normalize_velocity(self, velocity_si: Any) -> Tensor:
        velocity = _floating_tensor(velocity_si)
        if velocity.shape[-1:] != (3,):
            raise ValueError(
                f"velocity_si must end in three values, got {tuple(velocity.shape)}"
            )
        return velocity / self._scale_like(self.velocity_scale, velocity)

    normalize_velocity_si = normalize_velocity

    def normalize_time(self, time_seconds: Any) -> Tensor:
        value = _floating_tensor(time_seconds)
        return value / self._scale_like(self.time_scale, value)

    normalize_time_seconds = normalize_time

    def normalize_los(self, los: Any) -> Tensor:
        """Return direct LOS values without geometric re-normalization."""

        value = _floating_tensor(los)
        if value.shape[-1:] != (3,):
            raise ValueError(f"los must end in three values, got {tuple(value.shape)}")
        return value

    def normalize_stats(self, values: Any) -> Tensor:
        stats = _floating_tensor(values)
        if stats.shape[-1:] != (4,):
            raise ValueError(
                f"stats must end in four values, got {tuple(stats.shape)}"
            )
        return stats / self._scale_like(self.stats_scale, stats)

    def normalize_observation(self, value: Any, *, kind: str) -> Tensor:
        kind_key = str(kind).strip().lower()
        if kind_key in {"state", "state_si"}:
            return self.normalize_state(value)
        if kind_key in {"position", "position_si", "station", "station_si"}:
            return self.normalize_position(value)
        if kind_key in {"velocity", "velocity_si"}:
            return self.normalize_velocity(value)
        if kind_key in {"time", "time_sec", "time_seconds", "dt", "dt_sec"}:
            return self.normalize_time(value)
        if kind_key in {"los", "innovation", "direct_los"}:
            return self.normalize_los(value)
        if kind_key in {"stats", "metadata"}:
            return self.normalize_stats(value)
        raise ValueError(f"unknown normalization kind {kind!r}")


Normalization = V6Normalization
PhysicsNormalization = V6Normalization
V6PhysicalScales = PhysicalScales


__all__ = [
    "MU_EARTH_M3_S2",
    "EARTH_RADIUS_M",
    "R_EARTH_M",
    "V_SCALE_MPS",
    "T_SCALE_S",
    "R",
    "V",
    "T",
    "STAT_SCALE_NAMES",
    "PhysicalScales",
    "V6PhysicalScales",
    "V6Normalization",
    "Normalization",
    "PhysicsNormalization",
]

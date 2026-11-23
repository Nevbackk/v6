"""Batch-safe, differentiable two-body RK4 propagation.

This mirrors the physical rules in
orbit_los_kalman_net._propagate_scaled_batch_eager.  Substep counts are
prepared on the CPU; the differentiable RK4 kernel is tensorized over lanes.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Iterable

import torch


STATE_SCALE_VALUES = (1.0e6, 1.0e6, 1.0e6, 1.0e3, 1.0e3, 1.0e3)
STATE_ORDER = ("rx_m", "ry_m", "rz_m", "vx_mps", "vy_mps", "vz_mps")
DTYPE = torch.float64
MU_EARTH_M3_S2 = 3.986004418e14
MAX_SUBSTEP_SEC = 5.0


def state_scale(
    device: torch.device | str | None = None,
    dtype: torch.dtype = DTYPE,
) -> torch.Tensor:
    """Return the existing scaled-state convention."""

    return torch.tensor(
        STATE_SCALE_VALUES,
        device="cpu" if device is None else device,
        dtype=dtype,
    )


@dataclass(frozen=True)
class SubstepPlan:
    """CPU-owned one-epoch plan.

    Counts are zero for exact zero intervals and invalid lanes.  The integer
    vector and max count are safe to pass to a device-side propagation call.
    """

    dt_sec: torch.Tensor
    n_substeps: torch.Tensor
    max_n_substeps: int
    valid_mask: torch.Tensor
    max_substep_sec: float

    def __iter__(self) -> Iterable[object]:
        """Support unpacking as n_substeps, max_n_substeps."""

        yield self.n_substeps
        yield self.max_n_substeps


def _cpu_vector(
    value: torch.Tensor | Iterable[float] | float,
    *,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.is_complex():
            raise TypeError(f"{name} must be real")
        tensor = value.detach().to(device="cpu", dtype=dtype)
    else:
        tensor = torch.as_tensor(value, device="cpu", dtype=dtype)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1)
    elif tensor.ndim != 1:
        raise ValueError(f"{name} must have shape [batch], got {tuple(tensor.shape)}")
    return tensor.contiguous()


def prepare_substep_plan(
    dt_sec: torch.Tensor | Iterable[float] | float,
    max_substep_sec: float | torch.Tensor | Iterable[bool] = MAX_SUBSTEP_SEC,
    valid_mask: torch.Tensor | Iterable[bool] | None = None,
) -> SubstepPlan:
    """Prepare ceil(abs(dt)/max_substep_sec) on the CPU.

    A boolean vector may be passed as the second positional argument as a
    shorthand for valid_mask.  Invalid lanes receive zero substeps.
    """

    if valid_mask is None and not isinstance(max_substep_sec, numbers.Real):
        candidate = torch.as_tensor(max_substep_sec)
        if candidate.dtype == torch.bool or candidate.ndim > 0:
            valid_mask = max_substep_sec  # type: ignore[assignment]
            max_substep_sec = MAX_SUBSTEP_SEC

    try:
        max_step = float(max_substep_sec)
    except (TypeError, ValueError) as exc:
        raise TypeError("max_substep_sec must be a positive finite scalar") from exc
    if not math.isfinite(max_step) or max_step <= 0.0:
        raise ValueError("max_substep_sec must be finite and positive")

    dt_cpu = _cpu_vector(dt_sec, dtype=DTYPE, name="dt_sec")
    if not bool(torch.isfinite(dt_cpu).all()):
        raise ValueError("dt_sec must contain only finite values")

    if valid_mask is None:
        valid_cpu = torch.ones(dt_cpu.shape, dtype=torch.bool)
    else:
        valid_cpu = _cpu_vector(valid_mask, dtype=torch.bool, name="valid_mask")
        if valid_cpu.shape != dt_cpu.shape:
            raise ValueError(
                "valid_mask must have shape [batch] matching dt_sec, "
                f"got {tuple(valid_cpu.shape)} and {tuple(dt_cpu.shape)}"
            )

    n_substeps = torch.ceil(dt_cpu.abs() / max_step).to(torch.int64)
    n_substeps = torch.where(valid_cpu, n_substeps, torch.zeros_like(n_substeps))
    max_n_substeps = int(n_substeps.max()) if n_substeps.numel() else 0
    return SubstepPlan(
        dt_sec=dt_cpu,
        n_substeps=n_substeps,
        max_n_substeps=max_n_substeps,
        valid_mask=valid_cpu,
        max_substep_sec=max_step,
    )


prepare_substep_plan_cpu = prepare_substep_plan
prepare_batch_substep_plan = prepare_substep_plan


def _validate_state(z: torch.Tensor) -> int:
    if not isinstance(z, torch.Tensor):
        raise TypeError("scaled states must be a torch.Tensor")
    if z.ndim != 2 or z.shape[1] != 6:
        raise ValueError("scaled states must have shape [batch, 6]")
    if not z.is_floating_point() or z.is_complex():
        raise TypeError("scaled states must use a real floating dtype")
    return int(z.shape[0])


def _as_batch_tensor(
    value: torch.Tensor | Iterable[float] | float,
    *,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.is_complex():
            raise TypeError(f"{name} must be real")
        tensor = value.to(device=device, dtype=dtype)
    else:
        tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.ndim == 0:
        tensor = tensor.expand(batch)
    if tensor.shape != (batch,):
        raise ValueError(f"{name} must have shape [batch], got {tuple(tensor.shape)}")
    return tensor


def _as_batch_steps(
    value: torch.Tensor | Iterable[int] | int,
    *,
    batch: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.is_complex():
            raise TypeError(f"{name} must be real")
        tensor = value.to(device=device, dtype=torch.int64)
    else:
        tensor = torch.as_tensor(value, device=device, dtype=torch.int64)
    if tensor.ndim == 0:
        tensor = tensor.expand(batch)
    if tensor.shape != (batch,):
        raise ValueError(f"{name} must have shape [batch], got {tuple(tensor.shape)}")
    return tensor


def _as_valid_mask(
    value: torch.Tensor | Iterable[bool] | bool | None,
    *,
    batch: int,
    device: torch.device,
) -> torch.Tensor:
    if value is None:
        return torch.ones((batch,), device=device, dtype=torch.bool)
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device, dtype=torch.bool)
    else:
        tensor = torch.as_tensor(value, device=device, dtype=torch.bool)
    if tensor.ndim == 0:
        tensor = tensor.expand(batch)
    if tensor.shape != (batch,):
        raise ValueError(f"valid_mask must have shape [batch], got {tuple(tensor.shape)}")
    return tensor


def _propagate_scaled_batch_eager(
    z: torch.Tensor,
    dt_sec: torch.Tensor | Iterable[float] | float,
    n_substeps: torch.Tensor | Iterable[int] | int,
    max_n_substeps: int,
    max_substep_sec: float = MAX_SUBSTEP_SEC,
    valid_mask: torch.Tensor | Iterable[bool] | bool | None = None,
) -> torch.Tensor:
    """Differentiable two-body RK4 for one epoch across a batch.

    The only Python loop is the common upper bound max_n_substeps.  Each lane
    uses its own h = dt_sec / n_substeps; lane activity is tensor-masked.
    """

    batch = _validate_state(z)
    try:
        max_steps = int(max_n_substeps)
    except (TypeError, ValueError) as exc:
        raise TypeError("max_n_substeps must be a non-negative Python integer") from exc
    if max_steps < 0:
        raise ValueError("max_n_substeps must be non-negative")
    if valid_mask is None and not isinstance(max_substep_sec, numbers.Real):
        candidate = torch.as_tensor(max_substep_sec)
        if candidate.dtype == torch.bool or candidate.ndim > 0:
            valid_mask = max_substep_sec  # type: ignore[assignment]
            max_substep_sec = MAX_SUBSTEP_SEC
    try:
        max_step = float(max_substep_sec)
    except (TypeError, ValueError) as exc:
        raise TypeError("max_substep_sec must be a positive finite scalar") from exc
    if not math.isfinite(max_step) or max_step <= 0.0:
        raise ValueError("max_substep_sec must be finite and positive")

    dt = _as_batch_tensor(
        dt_sec, batch=batch, device=z.device, dtype=z.dtype, name="dt_sec"
    )
    steps = _as_batch_steps(
        n_substeps, batch=batch, device=z.device, name="n_substeps"
    )
    valid = _as_valid_mask(valid_mask, batch=batch, device=z.device)

    scale = state_scale(z.device, z.dtype)
    x = z * scale
    steps_safe = steps.clamp_min(1)
    h = dt / steps_safe.to(dtype=z.dtype)

    def f(state: torch.Tensor) -> torch.Tensor:
        r = state[:, :3]
        v = state[:, 3:]
        r_norm = torch.linalg.vector_norm(r, dim=1, keepdim=True).clamp_min(1.0)
        acceleration = -(MU_EARTH_M3_S2 / r_norm.pow(3)) * r
        return torch.cat((v, acceleration), dim=1)

    for k in range(max_steps):
        active = valid & (steps > k)
        active6 = active[:, None]
        safe_x = torch.where(active6, x, torch.zeros_like(x))
        k1 = f(safe_x)
        k2 = f(safe_x + 0.5 * h[:, None] * k1)
        k3 = f(safe_x + 0.5 * h[:, None] * k2)
        k4 = f(safe_x + h[:, None] * k3)
        x_new = safe_x + (h[:, None] / 6.0) * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        )
        x = torch.where(active6, x_new, x)

    return x / scale


propagate_scaled_batch_eager = _propagate_scaled_batch_eager


def propagate_scaled_batch(
    z: torch.Tensor,
    dt_sec: torch.Tensor | Iterable[float] | float,
    n_substeps: torch.Tensor | Iterable[int] | int,
    max_n_substeps: int,
    max_substep_sec: float = MAX_SUBSTEP_SEC,
    valid_mask: torch.Tensor | Iterable[bool] | bool | None = None,
) -> torch.Tensor:
    """Propagate a batch of scaled states with differentiable RK4."""

    return _propagate_scaled_batch_eager(
        z=z,
        dt_sec=dt_sec,
        n_substeps=n_substeps,
        max_n_substeps=max_n_substeps,
        max_substep_sec=max_substep_sec,
        valid_mask=valid_mask,
    )


__all__ = [
    "DTYPE",
    "MAX_SUBSTEP_SEC",
    "MU_EARTH_M3_S2",
    "STATE_ORDER",
    "STATE_SCALE_VALUES",
    "SubstepPlan",
    "prepare_batch_substep_plan",
    "prepare_substep_plan",
    "prepare_substep_plan_cpu",
    "propagate_scaled_batch",
    "propagate_scaled_batch_eager",
    "state_scale",
]

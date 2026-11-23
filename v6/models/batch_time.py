"""CPU preparation for irregular, variable-length recursive batch epochs."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import math
import torch
from torch import Tensor

@dataclass(frozen=True)
class BatchTimePlan:
    times_sec: Tensor
    init_times_sec: Tensor
    valid_mask: Tensor
    delta_sec: Tensor
    substeps: Tensor
    max_substeps: tuple[int, ...]


def _cpu_tensor(value: Any, dtype: torch.dtype, name: str) -> Tensor:
    if isinstance(value, Tensor) and value.device.type != "cpu":
        raise ValueError(f"{name} must be CPU metadata when preparing a batch plan")
    tensor = torch.as_tensor(value, dtype=dtype, device="cpu")
    if tensor.requires_grad:
        raise ValueError(f"{name} must be non-differentiable time metadata")
    return tensor


def prepare_batch_time_plan(times_sec: Any, init_time_sec: Any,
                            valid_mask: Any = None,
                            max_substep_sec: float = 5.0) -> BatchTimePlan:
    """Validate epochs once on CPU and precompute each trajectory's RK4 plan.

    Padding may appear between valid observations; the next valid interval is
    measured from the preceding valid epoch. Equal epochs are accepted.
    """
    times = _cpu_tensor(times_sec, torch.float64, "times_sec")
    if times.ndim == 1:
        times = times.unsqueeze(0)
    if times.ndim != 2 or times.shape[1] == 0:
        raise ValueError("times_sec must be [T] or [B,T] with T > 0")
    batch, length = times.shape
    init = _cpu_tensor(init_time_sec, torch.float64, "init_time_sec").reshape(-1)
    if init.numel() == 1:
        init = init.expand(batch)
    if init.numel() != batch or not bool(torch.isfinite(init).all()):
        raise ValueError("init_time_sec must be finite scalar or [B]")
    if valid_mask is None:
        valid = torch.ones((batch, length), dtype=torch.bool)
    else:
        valid = _cpu_tensor(valid_mask, torch.bool, "valid_mask")
        if valid.ndim == 0:
            valid = valid.expand(batch, length)
        elif valid.ndim == 1:
            valid = valid.unsqueeze(0).expand(batch, length)
        if tuple(valid.shape) != (batch, length):
            raise ValueError("valid_mask must be [T] or [B,T]")
    if not math.isfinite(max_substep_sec) or max_substep_sec <= 0:
        raise ValueError("max_substep_sec must be finite and positive")
    previous = init
    cleaned, intervals, counts, max_counts = [], [], [], []
    for k in range(length):
        active = valid[:, k]
        current = torch.where(active, times[:, k], previous)
        if not bool(torch.isfinite(current).all()):
            raise ValueError("valid observation times must be finite")
        dt = torch.where(active, current - previous, torch.zeros_like(current))
        if bool((dt < 0).any()):
            raise ValueError("valid times must be non-decreasing and at least init_time_sec")
        count = torch.ceil(dt.abs() / max_substep_sec).to(torch.int64).clamp_min(1)
        count = torch.where(active & (dt != 0), count, torch.zeros_like(count))
        cleaned.append(current)
        intervals.append(dt)
        counts.append(count)
        max_counts.append(int(count.max()))
        previous = torch.where(active, current, previous)
    return BatchTimePlan(
        times_sec=torch.stack(cleaned, dim=1), init_times_sec=init,
        valid_mask=valid, delta_sec=torch.stack(intervals, dim=1),
        substeps=torch.stack(counts, dim=1), max_substeps=tuple(max_counts),
    )

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
    kv: Any
    scalar_tokens: list[Tensor]
    vector_tokens: list[Tensor]
    times_sec: list[Tensor]
    station_states_si: list[Tensor]
    observed_los: list[Tensor]
    station_ids: list[Tensor]
    station_id_reliable: list[Tensor]
    valid: list[Tensor | bool]
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


@dataclass(frozen=True)
class TrainingFilterOutput:
    """Minimal differentiable sequence outputs needed by training losses."""

    prior_states_si: Tensor
    posterior_states_si: Tensor
    gains: Tensor | None
    correction_scaled: Tensor | None
    correction_si: Tensor | None
    innovations: Tensor
    predicted_los: Tensor
    valid_mask: Tensor
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


class RecursiveFilterMixin:
    """Batch-parallel causal recursion; the serial implementation is a reference."""

    def forward_filter_sequence(
        self, *args: Any, **kwargs: Any
    ) -> RecursiveFilterOutput | TrainingFilterOutput:
        from .batch_recursive_filter import forward_filter_sequence
        return forward_filter_sequence(self, *args, **kwargs)

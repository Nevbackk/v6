"""Relation-conditioned causal attention for scalar/vector token streams.

Scalars have shape ``[B, N, C_s]`` and vectors have shape ``[B, N, C_v, 3]``.
Vector channels are mixed with shared XYZ coefficients; attention scores use
only scalar products of vectors, making the layer SO(3)-equivariant.  A direct
LOS lane is projected into values at observation tokens only.
"""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .block_kv_cache import BlockwiseKVCache, KVBlock


class ScalarVectorAttentionOutput(NamedTuple):
    """Two-field result, unpackable as ``(scalar, vector)``."""

    scalar: Tensor
    vector: Tensor


def safe_masked_softmax(
    logits: Tensor,
    allowed: Tensor,
    dim: int = -1,
    *,
    check_finite: bool = True,
) -> Tensor:
    """Masked softmax whose all-masked rows are exactly zero, never NaN.

    ``check_finite`` is kept separate from the numerical masking operation so
    the online training path can avoid synchronizing a GPU reduction back to
    Python on every frame.  Debug callers retain the original strict check.
    """

    if logits.shape != allowed.shape:
        try:
            allowed = torch.broadcast_to(allowed, logits.shape)
        except RuntimeError as exc:
            raise ValueError(
                f"allowed mask shape {tuple(allowed.shape)} does not broadcast to "
                f"logits shape {tuple(logits.shape)}"
            ) from exc
    allowed = allowed.to(device=logits.device, dtype=torch.bool)
    if check_finite:
        finite_allowed = torch.isfinite(logits) | ~allowed
        if not bool(finite_allowed.all()):
            raise FloatingPointError("non-finite attention logit on an allowed edge")
    clean = logits
    masked = clean.masked_fill(~allowed, -torch.inf)
    maximum = masked.max(dim=dim, keepdim=True).values
    has_value = allowed.any(dim=dim, keepdim=True)
    maximum = torch.where(has_value, maximum, torch.zeros_like(maximum))
    exponential = torch.exp(masked - maximum).masked_fill(~allowed, 0.0)
    denominator = exponential.sum(dim=dim, keepdim=True)
    return exponential / denominator.clamp_min(torch.finfo(exponential.dtype).tiny)


class ScalarVectorKVCache(NamedTuple):
    """Projected causal history for the single-layer online attention path."""
    key_scalar: Tensor
    key_vector: Tensor
    value_scalar: Tensor
    value_vector: Tensor
    valid_mask: Tensor


class OnlineScalarVectorAttentionOutput(NamedTuple):
    """Current output and graph-preserving extended KV cache."""
    output: ScalarVectorAttentionOutput
    cache: ScalarVectorKVCache | BlockwiseKVCache
    weights: Tensor | None = None
    logits: Tensor | None = None


class RelationConditionedScalarVectorAttention(nn.Module):
    """Relation-conditioned multi-head causal attention for scalar/vector tokens.

    The canonical constructor is ``(scalar_dim, vector_channels,
    relation_dim=8, num_heads=4, head_dim=None)``.  ``scalar_channels``,
    ``heads`` and ``relation_channels`` are accepted as compatibility aliases.
    ``query_chunk_size`` bounds temporary ``[B,H,Q,N]`` score tensors without
    truncating the key/value history.
    """

    def __init__(
        self,
        scalar_dim: int | None = None,
        vector_channels: int | None = None,
        relation_dim: int = 8,
        num_heads: int = 4,
        head_dim: int | None = None,
        output_scalar_dim: int | None = None,
        output_vector_channels: int | None = None,
        *,
        scalar_channels: int | None = None,
        relation_channels: int | None = None,
        heads: int | None = None,
        out_scalar_dim: int | None = None,
        out_vector_channels: int | None = None,
        los_direct_scalar_dim: int | None = None,
        los_direct_vector_channels: int | None = None,
        query_chunk_size: int | None = None,
        allow_same_frame: bool = True,
        causal: bool = True,
        dropout: float = 0.0,
        aggregation_mode: str = "unified",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if scalar_dim is None:
            scalar_dim = scalar_channels
        elif scalar_channels is not None and int(scalar_dim) != int(scalar_channels):
            raise ValueError("scalar_dim and scalar_channels disagree")
        if scalar_dim is None or vector_channels is None:
            raise TypeError("scalar_dim and vector_channels are required")
        if relation_channels is not None:
            if int(relation_dim) != 8 and int(relation_dim) != int(relation_channels):
                raise ValueError("relation_dim and relation_channels disagree")
            relation_dim = int(relation_channels)
        if heads is not None:
            if int(num_heads) != 4 and int(num_heads) != int(heads):
                raise ValueError("num_heads and heads disagree")
            num_heads = int(heads)
        if output_scalar_dim is None:
            output_scalar_dim = out_scalar_dim
        elif out_scalar_dim is not None and int(output_scalar_dim) != int(out_scalar_dim):
            raise ValueError("output_scalar_dim and out_scalar_dim disagree")
        if output_vector_channels is None:
            output_vector_channels = out_vector_channels
        elif out_vector_channels is not None and int(output_vector_channels) != int(out_vector_channels):
            raise ValueError("output_vector_channels and out_vector_channels disagree")
        self.scalar_dim = int(scalar_dim)
        self.vector_channels = int(vector_channels)
        self.relation_dim = int(relation_dim)
        self.num_heads = int(num_heads)
        if min(self.scalar_dim, self.vector_channels, self.relation_dim, self.num_heads) <= 0:
            raise ValueError("attention dimensions must be positive")
        self.head_dim = int(head_dim or max(1, self.scalar_dim // self.num_heads))
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        self.output_scalar_dim = int(output_scalar_dim or self.scalar_dim)
        self.output_vector_channels = int(output_vector_channels or self.vector_channels)
        if min(self.output_scalar_dim, self.output_vector_channels) <= 0:
            raise ValueError("output dimensions must be positive")
        if query_chunk_size is not None and int(query_chunk_size) <= 0:
            raise ValueError("query_chunk_size must be positive or None")
        if not 0 <= float(dropout) < 1:
            raise ValueError("dropout must lie in [0,1)")
        if aggregation_mode not in ("unified", "dual_softmax_gated"):
            raise ValueError(
                "aggregation_mode must be 'unified' or 'dual_softmax_gated'"
            )
        self.query_chunk_size = None if query_chunk_size is None else int(query_chunk_size)
        self.allow_same_frame = bool(allow_same_frame)
        self.causal = bool(causal)
        self.aggregation_mode = str(aggregation_mode)
        width = self.num_heads * self.head_dim
        # Scalar Q/K are built from rotation-invariant channels: the original
        # scalar channels plus one norm for each vector channel.  No XYZ
        # component is flattened into this projection.
        invariant_dim = self.scalar_dim + self.vector_channels
        self.invariant_dim = invariant_dim
        self.q_scalar = nn.Linear(invariant_dim, width, dtype=dtype)
        self.k_scalar = nn.Linear(invariant_dim, width, dtype=dtype)
        self.q_vector_weight = nn.Parameter(torch.empty(self.num_heads, self.head_dim, self.vector_channels, dtype=dtype))
        self.k_vector_weight = nn.Parameter(torch.empty(self.num_heads, self.head_dim, self.vector_channels, dtype=dtype))
        nn.init.xavier_uniform_(self.q_vector_weight)
        nn.init.xavier_uniform_(self.k_vector_weight)
        # A relation feature is a per-head key vector, so it participates in
        # q_i dot (k_j + r_ij), rather than being an unstructured scalar bias.
        relation_hidden = max(width, self.relation_dim)
        self.relation_mlp = nn.Sequential(
            nn.Linear(self.relation_dim, relation_hidden, dtype=dtype),
            nn.GELU(),
            nn.Linear(relation_hidden, width, dtype=dtype),
        )
        self.value_scalar = nn.Linear(self.scalar_dim, width, dtype=dtype)
        self.value_vector_weight = nn.Parameter(torch.empty(self.num_heads, self.head_dim, self.vector_channels, dtype=dtype))
        nn.init.xavier_uniform_(self.value_vector_weight)
        if self.aggregation_mode == "unified":
            # Keep the V6 parameter names, shapes, initialization, and order
            # unchanged so existing V6 state_dicts load strictly.
            self.output_scalar = nn.Linear(width, self.output_scalar_dim, dtype=dtype)
            self.output_vector_weight = nn.Parameter(torch.empty(self.output_vector_channels, self.num_heads, self.head_dim, dtype=dtype))
            nn.init.xavier_uniform_(self.output_vector_weight)
        else:
            # Each key stream gets its own output projection.  Scalar biases
            # are omitted so an empty or fully gated stream contributes zero.
            self.output_scalar_state = nn.Linear(
                width, self.output_scalar_dim, bias=False, dtype=dtype
            )
            self.output_scalar_obs = nn.Linear(
                width, self.output_scalar_dim, bias=False, dtype=dtype
            )
            self.output_vector_state_weight = nn.Parameter(
                torch.empty(self.output_vector_channels, self.num_heads, self.head_dim, dtype=dtype)
            )
            self.output_vector_obs_weight = nn.Parameter(
                torch.empty(self.output_vector_channels, self.num_heads, self.head_dim, dtype=dtype)
            )
            nn.init.xavier_uniform_(self.output_vector_state_weight)
            nn.init.xavier_uniform_(self.output_vector_obs_weight)
            self.context_gate = nn.Sequential(
                nn.Linear(6 * self.head_dim, 2 * self.head_dim, dtype=dtype),
                nn.GELU(),
                nn.Linear(2 * self.head_dim, 2, dtype=dtype),
            )
            nn.init.normal_(self.context_gate[-1].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.context_gate[-1].bias)
        self.scalar_residual = (
            nn.Identity()
            if self.output_scalar_dim == self.scalar_dim
            else nn.Linear(self.scalar_dim, self.output_scalar_dim, dtype=dtype)
        )
        if self.output_vector_channels == self.vector_channels:
            self.register_parameter("vector_residual_weight", None)
        else:
            self.vector_residual_weight = nn.Parameter(
                torch.empty(self.output_vector_channels, self.vector_channels, dtype=dtype)
            )
            nn.init.xavier_uniform_(self.vector_residual_weight)
        self.dropout = nn.Dropout(float(dropout))
        if los_direct_scalar_dim is None:
            self.los_scalar_value = nn.LazyLinear(width, bias=True, dtype=dtype)
        else:
            self.los_scalar_value = nn.Linear(int(los_direct_scalar_dim), width, dtype=dtype)
        if los_direct_vector_channels is None:
            self.los_vector_value = nn.LazyLinear(width, bias=False, dtype=dtype)
        else:
            self.los_vector_value = nn.Linear(int(los_direct_vector_channels), width, bias=False, dtype=dtype)

    @property
    def attention_weight_semantics(self) -> str:
        """Describe whether returned weights use one or two denominators."""
        if self.aggregation_mode == "dual_softmax_gated":
            return "independent_state_observation_softmax"
        return "global_key_softmax"

    @staticmethod
    def _dual_stream_softmax(
        logits: Tensor,
        allowed: Tensor,
        global_key_indices: Tensor,
        *,
        check_finite: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Normalize State and Observation keys independently over all keys."""
        state_key = (global_key_indices.to(device=logits.device) % 2) == 0
        state_allowed = allowed & state_key.view(1, 1, -1)
        obs_allowed = allowed & ~state_key.view(1, 1, -1)
        alpha_state = safe_masked_softmax(
            logits, state_allowed[:, None], dim=-1, check_finite=check_finite
        )
        alpha_obs = safe_masked_softmax(
            logits, obs_allowed[:, None], dim=-1, check_finite=check_finite
        )
        heads = int(logits.shape[1])
        has_state = state_allowed.any(dim=-1)[:, None, :, None].expand(
            -1, heads, -1, -1
        )
        has_obs = obs_allowed.any(dim=-1)[:, None, :, None].expand(
            -1, heads, -1, -1
        )
        return alpha_state, alpha_obs, has_state, has_obs

    def _dual_context_updates(
        self,
        q_scalar: Tensor,
        q_vector: Tensor,
        context_state_scalar: Tensor,
        context_state_vector: Tensor,
        context_obs_scalar: Tensor,
        context_obs_vector: Tensor,
        has_state: Tensor,
        has_obs: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Gate invariant stream contexts, then project each stream separately."""
        if self.aggregation_mode != "dual_softmax_gated":
            raise RuntimeError("dual context updates require dual_softmax_gated mode")
        # The State gate scales only the State Value context. The current
        # State Query residual is fused afterward, so a near-zero State gate
        # is not strictly equivalent to B512's history_obs_only mask while
        # retaining both current-frame tokens.
        gate_input = torch.cat(
            (
                q_scalar,
                torch.linalg.vector_norm(q_vector, dim=-1),
                context_state_scalar,
                torch.linalg.vector_norm(context_state_vector, dim=-1),
                context_obs_scalar,
                torch.linalg.vector_norm(context_obs_vector, dim=-1),
            ),
            dim=-1,
        )
        gate_logits = self.context_gate(gate_input)
        gamma_state = torch.sigmoid(gate_logits[..., 0:1])
        gamma_obs = torch.sigmoid(gate_logits[..., 1:2])
        gamma_state = torch.where(has_state, gamma_state, torch.zeros_like(gamma_state))
        gamma_obs = torch.where(has_obs, gamma_obs, torch.zeros_like(gamma_obs))

        state_scalar_gated = gamma_state * context_state_scalar
        obs_scalar_gated = gamma_obs * context_obs_scalar
        state_vector_gated = gamma_state.unsqueeze(-1) * context_state_vector
        obs_vector_gated = gamma_obs.unsqueeze(-1) * context_obs_vector
        state_scalar_update = self.output_scalar_state(
            state_scalar_gated.permute(0, 2, 1, 3).reshape(
                state_scalar_gated.shape[0], state_scalar_gated.shape[2], -1
            )
        )
        obs_scalar_update = self.output_scalar_obs(
            obs_scalar_gated.permute(0, 2, 1, 3).reshape(
                obs_scalar_gated.shape[0], obs_scalar_gated.shape[2], -1
            )
        )
        state_vector_update = torch.einsum(
            "bqhdc,ohd->bqoc",
            state_vector_gated.permute(0, 2, 1, 3, 4),
            self.output_vector_state_weight,
        )
        obs_vector_update = torch.einsum(
            "bqhdc,ohd->bqoc",
            obs_vector_gated.permute(0, 2, 1, 3, 4),
            self.output_vector_obs_weight,
        )
        return (
            state_scalar_update + obs_scalar_update,
            state_vector_update + obs_vector_update,
            gamma_state,
            gamma_obs,
        )

    def _capture_gate_diagnostics(
        self,
        diagnostics: dict[str, Any] | None,
        gamma_state: Tensor,
        gamma_obs: Tensor,
    ) -> None:
        """Copy final post-mask Stream gates into a caller-owned debug dict."""
        if diagnostics is None:
            return
        if self.aggregation_mode != "dual_softmax_gated":
            raise ValueError("Gate diagnostics require dual_softmax_gated")
        state = gamma_state.squeeze(-1)
        obs = gamma_obs.squeeze(-1)
        assert state.shape == obs.shape
        assert state.ndim == 3
        diagnostics["gamma_state"] = state.detach().clone()
        diagnostics["gamma_obs"] = obs.detach().clone()
        diagnostics["query_order"] = ("state", "observation")
        diagnostics["weight_semantics"] = "independent_sigmoid_gates"

    @staticmethod
    def _merge_block_stream_stats(
        running: tuple[Tensor, Tensor, Tensor, Tensor, Tensor] | None,
        block: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Merge one stream's block LSE and value numerators without -inf math."""
        block_max, block_z, block_scalar_num, block_vector_num, block_has = block
        if running is None:
            running_max = torch.zeros_like(block_max)
            running_z = torch.zeros_like(block_z)
            running_scalar_num = torch.zeros_like(block_scalar_num)
            running_vector_num = torch.zeros_like(block_vector_num)
            running_has = torch.zeros_like(block_has, dtype=torch.bool)
        else:
            running_max, running_z, running_scalar_num, running_vector_num, running_has = running
        new_has = running_has | block_has
        both = running_has & block_has
        new_max = torch.where(
            both,
            torch.maximum(running_max, block_max),
            torch.where(
                running_has,
                running_max,
                torch.where(block_has, block_max, torch.zeros_like(block_max)),
            ),
        )
        old_delta = torch.where(running_has, running_max - new_max, torch.zeros_like(new_max))
        block_delta = torch.where(block_has, block_max - new_max, torch.zeros_like(new_max))
        old_factor = torch.where(running_has, torch.exp(old_delta), torch.zeros_like(old_delta))
        block_factor = torch.where(block_has, torch.exp(block_delta), torch.zeros_like(block_delta))
        merged_z = old_factor * running_z + block_factor * block_z
        merged_scalar_num = old_factor * running_scalar_num + block_factor * block_scalar_num
        merged_vector_num = (
            old_factor.unsqueeze(-1) * running_vector_num
            + block_factor.unsqueeze(-1) * block_vector_num
        )
        return new_max, merged_z, merged_scalar_num, merged_vector_num, new_has

    @staticmethod
    def build_causal_mask(valid_mask: Tensor, *, token_frame_index: Tensor | None = None, allow_same_frame: bool = True) -> Tensor:
        """Build a frame-causal token mask while retaining same-frame edges."""

        valid = valid_mask.to(dtype=torch.bool)
        if valid.ndim == 1:
            valid, squeeze = valid.unsqueeze(0), True
        elif valid.ndim == 2:
            squeeze = False
        else:
            raise ValueError("valid_mask must have shape [N] or [B,N]")
        batch, tokens = map(int, valid.shape)
        if token_frame_index is None:
            frame = torch.arange(tokens, device=valid.device) // 2
            if tokens % 2:
                frame = torch.arange(tokens, device=valid.device)
        else:
            frame = torch.as_tensor(token_frame_index, device=valid.device)
            if frame.ndim == 1 and int(frame.shape[0]) != tokens:
                raise ValueError("token_frame_index must have length N")
            if frame.ndim == 2 and tuple(frame.shape) != (batch, tokens):
                raise ValueError("token_frame_index must match [B,N]")
            if frame.ndim not in (1, 2):
                raise ValueError("token_frame_index must have shape [N] or [B,N]")
        if frame.ndim == 1:
            allowed = frame[:, None] >= frame[None, :]
            if allow_same_frame:
                allowed = allowed | (frame[:, None] == frame[None, :])
            allowed = allowed.unsqueeze(0).expand(batch, tokens, tokens)
        else:
            allowed = frame[:, :, None] >= frame[:, None, :]
            if allow_same_frame:
                allowed = allowed | (frame[:, :, None] == frame[:, None, :])
        allowed = allowed & valid[:, :, None] & valid[:, None, :]
        return allowed[0] if squeeze else allowed

    def _resolve_mask(self, attention_mask: Tensor | None, *, batch: int, tokens: int, device: torch.device, token_frame_index: Tensor | None) -> tuple[Tensor, Tensor]:
        if token_frame_index is None:
            frame = torch.arange(tokens, device=device) // 2
            if tokens % 2:
                frame = torch.arange(tokens, device=device)
        else:
            frame = torch.as_tensor(token_frame_index, device=device)
        if frame.ndim == 1:
            if int(frame.shape[0]) != tokens:
                raise ValueError("token_frame_index must have length N")
            same_frame = (frame[:, None] == frame[None, :]).unsqueeze(0).expand(batch, tokens, tokens)
            frame_causal = (frame[:, None] >= frame[None, :]).unsqueeze(0).expand(batch, tokens, tokens)
        elif frame.ndim == 2 and tuple(frame.shape) == (batch, tokens):
            same_frame = frame[:, :, None] == frame[:, None, :]
            frame_causal = frame[:, :, None] >= frame[:, None, :]
        else:
            raise ValueError("token_frame_index must have shape [N] or [B,N]")
        if attention_mask is None:
            allowed = frame_causal if self.causal else torch.ones_like(frame_causal)
            additive = torch.zeros_like(allowed, dtype=self.q_scalar.weight.dtype)
        else:
            raw = torch.as_tensor(attention_mask, device=device)
            if raw.ndim in (2, 3) and int(raw.shape[-1]) != tokens:
                frame_count = int(frame.max().item()) + 1 if frame.numel() else 0
                if tuple(raw.shape[-2:]) != (frame_count, frame_count):
                    raise ValueError(f"attention mask shape {tuple(raw.shape)} does not match [B,N,N]")
                if raw.ndim == 2:
                    raw = raw[frame[:, None], frame[None, :]]
                else:
                    raw = raw[:, frame[:, None], frame[None, :]]
            if raw.ndim == 2:
                if tuple(raw.shape) != (tokens, tokens):
                    raise ValueError("attention mask must have shape [N,N] or [B,N,N]")
                raw = raw.unsqueeze(0).expand(batch, tokens, tokens)
            elif raw.ndim == 3:
                if tuple(raw.shape) != (batch, tokens, tokens):
                    raise ValueError("attention mask must have shape [N,N] or [B,N,N]")
            else:
                raise ValueError("attention mask must have shape [N,N] or [B,N,N]")
            if raw.dtype == torch.bool:
                allowed = raw
                additive = torch.zeros_like(raw, dtype=self.q_scalar.weight.dtype)
            else:
                additive = raw.to(dtype=self.q_scalar.weight.dtype)
                allowed = torch.isfinite(additive) & (additive > -1e8)
                additive = torch.where(allowed, additive, torch.zeros_like(additive))
            # An explicitly supplied mask is authoritative.  In particular,
            # do not reopen same-frame or padding edges here.  Callers that
            # want same-frame State/Observation interaction must construct it
            # in the token causal mask before passing it to this layer.
        return allowed, additive

    @staticmethod
    def _as_batch(value: Tensor, batch: int, name: str) -> Tensor:
        if value.shape[0] == batch:
            return value
        if value.shape[0] == 1:
            return value.expand(batch, *value.shape[1:])
        raise ValueError(f"{name} batch dimension {value.shape[0]} does not match {batch}")

    @staticmethod
    def _parse_los_direct(los_direct: Any, *, batch: int, dtype: torch.dtype, device: torch.device) -> tuple[Tensor | None, Tensor | None]:
        if los_direct is None:
            return None, None
        scalar: Any = None
        vector: Any = None
        if isinstance(los_direct, dict):
            scalar = los_direct.get("scalar", los_direct.get("s"))
            vector = los_direct.get("vector", los_direct.get("v"))
        elif isinstance(los_direct, (tuple, list)):
            if len(los_direct) != 2:
                raise ValueError("los_direct tuple must be (scalar, vector)")
            scalar, vector = los_direct
        else:
            raw = RelationConditionedScalarVectorAttention._as_batch(torch.as_tensor(los_direct, device=device, dtype=dtype), batch, "los_direct")
            if raw.ndim == 2:
                raw = raw.unsqueeze(-1)
            if raw.ndim == 3 and int(raw.shape[-1]) == 3:
                vector = raw.unsqueeze(-2)
            elif raw.ndim == 3:
                scalar = raw
            elif raw.ndim == 4 and int(raw.shape[-1]) == 3:
                vector = raw
            else:
                raise ValueError("los_direct must be [B,T,D], [B,T,3], [B,T,C,3], or (scalar, vector)")
        if scalar is not None:
            scalar = RelationConditionedScalarVectorAttention._as_batch(torch.as_tensor(scalar, device=device, dtype=dtype), batch, "los_direct scalar")
            if scalar.ndim == 2:
                scalar = scalar.unsqueeze(-1)
            if scalar.ndim != 3:
                raise ValueError("los_direct scalar lane must have shape [B,T,D]")
        if vector is not None:
            vector = RelationConditionedScalarVectorAttention._as_batch(torch.as_tensor(vector, device=device, dtype=dtype), batch, "los_direct vector")
            if vector.ndim == 3 and int(vector.shape[-1]) == 3:
                vector = vector.unsqueeze(-2)
            if vector.ndim != 4 or int(vector.shape[-1]) != 3:
                raise ValueError("los_direct vector lane must have shape [B,T,C,3]")
        return scalar, vector

    @staticmethod
    def _observation_indices(*, batch: int, tokens: int, frames: int, device: torch.device, token_frame_index: Tensor | None, observation_token_mask: Tensor | None) -> tuple[Tensor, Tensor]:
        if observation_token_mask is not None:
            obs = torch.as_tensor(observation_token_mask, device=device, dtype=torch.bool)
            if obs.ndim == 1:
                if int(obs.shape[0]) != tokens:
                    raise ValueError("observation_token_mask must have length N")
                obs = obs.unsqueeze(0).expand(batch, tokens)
            elif tuple(obs.shape) != (batch, tokens):
                raise ValueError("observation_token_mask must have shape [N] or [B,N]")
        elif tokens == 2 * frames:
            obs = (torch.arange(tokens, device=device) % 2 == 1).unsqueeze(0).expand(batch, tokens)
        elif tokens == frames:
            obs = torch.ones((batch, tokens), device=device, dtype=torch.bool)
        else:
            raise ValueError("cannot map LOS frames to tokens; pass observation_token_mask")
        if token_frame_index is None:
            frame = (torch.arange(tokens, device=device) // 2 if tokens == 2 * frames else torch.arange(tokens, device=device)).unsqueeze(0).expand(batch, tokens)
        else:
            frame = torch.as_tensor(token_frame_index, device=device)
            if frame.ndim == 1:
                frame = frame.unsqueeze(0).expand(batch, tokens)
            if tuple(frame.shape) != (batch, tokens):
                raise ValueError("token_frame_index must have shape [N] or [B,N]")
        return obs, frame.to(dtype=torch.long)

    def _inject_direct_values(
        self,
        scalar_values: Tensor,
        vector_values: Tensor,
        los_direct: Any,
        *,
        batch: int,
        tokens: int,
        token_frame_index: Tensor | None,
        observation_token_mask: Tensor | None,
        mode: str = "debug",
    ) -> tuple[Tensor, Tensor]:
        scalar, vector = self._parse_los_direct(los_direct, batch=batch, dtype=scalar_values.dtype, device=scalar_values.device)
        if scalar is None and vector is None:
            return scalar_values, vector_values
        frames = int(scalar.shape[1]) if scalar is not None else int(vector.shape[1])
        if vector is not None and int(vector.shape[1]) != frames:
            raise ValueError("LOS scalar and vector lanes must have the same T")
        obs, frame = self._observation_indices(batch=batch, tokens=tokens, frames=frames, device=scalar_values.device, token_frame_index=token_frame_index, observation_token_mask=observation_token_mask)
        # ``frame`` is validated by the CPU input pipeline in train_fast
        # mode.  Keeping this check out of the hot path avoids a GPU -> Python
        # synchronization on every online frame.
        if mode == "debug" and bool((frame < 0).any() or (frame >= frames).any()):
            raise ValueError("token_frame_index contains a frame outside LOS direct sequence")
        batch_index = torch.arange(batch, device=scalar_values.device)[:, None]
        if scalar is not None:
            projected = self.los_scalar_value(scalar[batch_index, frame]).reshape(batch, tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            scalar_values = scalar_values + projected * obs[:, None, :, None].to(scalar_values.dtype)
        if vector is not None:
            projected = self.los_vector_value(vector[batch_index, frame].transpose(-1, -2)).transpose(-1, -2)
            projected = projected.reshape(batch, tokens, self.num_heads, self.head_dim, 3).permute(0, 2, 1, 3, 4)
            vector_values = vector_values + projected * obs[:, None, :, None, None].to(vector_values.dtype)
        return scalar_values, vector_values

    def _project_current_kv(
        self, s: Tensor, v: Tensor, los_direct: Any = None, *, mode: str = "debug"
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        parameter = self.q_scalar.weight
        s = s.to(device=parameter.device, dtype=parameter.dtype)
        v = v.to(device=parameter.device, dtype=parameter.dtype)
        batch, tokens, _ = s.shape
        if tokens != 2:
            raise ValueError("online attention requires exactly two current tokens")
        scalar_values = self.value_scalar(s).reshape(
            batch, tokens, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        vector_values = torch.einsum(
            "bnci,hdc->bnhdi", v, self.value_vector_weight
        ).permute(0, 2, 1, 3, 4)
        if los_direct is not None:
            frame_index = torch.zeros((batch, tokens), device=s.device, dtype=torch.long)
            observation_mask = torch.tensor(
                [[False, True]], device=s.device, dtype=torch.bool
            ).expand(batch, tokens)
            scalar_values, vector_values = self._inject_direct_values(
                scalar_values, vector_values, los_direct,
                batch=batch, tokens=tokens, token_frame_index=frame_index,
                observation_token_mask=observation_mask,
                mode=mode,
            )
        invariant = torch.cat((s, torch.linalg.vector_norm(v, dim=-1)), dim=-1)
        key_scalar = self.k_scalar(invariant).reshape(
            batch, tokens, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        key_vector = torch.einsum(
            "bnci,hdc->bnhdi", v, self.k_vector_weight
        ).permute(0, 2, 1, 3, 4)
        return key_scalar, key_vector, scalar_values, vector_values

    def forward_current_legacy(
        self,
        current_scalar: Tensor,
        current_vector: Tensor,
        historical_kv_cache: ScalarVectorKVCache | None,
        current_pairwise_geometry: Tensor,
        current_direct_los: Any = None,
        *,
        attention_mask: Tensor | None = None,
        current_valid_mask: Tensor | None = None,
        return_attention: bool = False,
        return_logits: bool = False,
        gate_diagnostics: dict[str, Any] | None = None,
        mode: str = "debug",
    ) -> OnlineScalarVectorAttentionOutput:
        """Compute two current queries and extend graph-preserving projected KV.

        current_valid_mask explicitly marks the two current tokens. Without it,
        validity is inferred from the supplied mask's current query and key
        edges, or assumed true if no mask was supplied.
        """
        if mode not in ("debug", "train_fast"):
            raise ValueError("mode must be 'debug' or 'train_fast'")
        if self.output_scalar_dim != self.scalar_dim or self.output_vector_channels != self.vector_channels:
            raise ValueError("online residual attention requires matching input/output channels")
        parameter = self.q_scalar.weight
        s = current_scalar.to(device=parameter.device, dtype=parameter.dtype)
        v = current_vector.to(device=parameter.device, dtype=parameter.dtype)
        if s.ndim != 3 or tuple(s.shape[1:]) != (2, self.scalar_dim):
            raise ValueError(f"current_scalar must be [B,2,{self.scalar_dim}]")
        if v.ndim != 4 or tuple(v.shape[1:]) != (2, self.vector_channels, 3):
            raise ValueError(f"current_vector must be [B,2,{self.vector_channels},3]")
        batch = int(s.shape[0])
        if current_valid_mask is None:
            if attention_mask is None:
                current_valid = torch.ones((batch, 2), device=s.device, dtype=torch.bool)
            else:
                raw_mask = torch.as_tensor(attention_mask, device=s.device)
                if raw_mask.ndim == 2:
                    raw_mask = raw_mask.unsqueeze(0)
                if raw_mask.ndim != 3 or int(raw_mask.shape[1]) != 2:
                    raise ValueError("online attention_mask must be [B,2,2K]")
                if raw_mask.shape[-1] < 2:
                    raise ValueError("online mask must include the current two key tokens")
                if int(raw_mask.shape[0]) == 1 and batch > 1:
                    raw_mask = raw_mask.expand(batch, *raw_mask.shape[1:])
                if int(raw_mask.shape[0]) != batch:
                    raise ValueError("online attention_mask batch dimension does not match current tokens")
                if raw_mask.dtype == torch.bool:
                    raw_allowed = raw_mask
                else:
                    raw_allowed = torch.isfinite(raw_mask) & (raw_mask > -1e8)
                # A token can be a valid key even when its own query row is
                # intentionally masked. Explicit current_valid_mask resolves
                # cases where neither incoming nor outgoing edges are present.
                current_valid = raw_allowed.any(dim=-1) | raw_allowed[..., -2:].any(dim=-2)
        else:
            current_valid = torch.as_tensor(current_valid_mask, device=s.device, dtype=torch.bool)
            if current_valid.ndim == 1:
                current_valid = current_valid.unsqueeze(0)
            if current_valid.ndim == 2 and int(current_valid.shape[0]) == 1 and batch > 1:
                current_valid = current_valid.expand(batch, 2)
            if tuple(current_valid.shape) != (batch, 2):
                raise ValueError("current_valid_mask must be [B,2]")
        s = torch.where(current_valid.unsqueeze(-1), s, torch.zeros_like(s))
        v = torch.where(current_valid.unsqueeze(-1).unsqueeze(-1), v, torch.zeros_like(v))
        if current_direct_los is not None:
            direct_s, direct_v = self._parse_los_direct(
                current_direct_los, batch=batch, dtype=s.dtype, device=s.device
            )
            observation_valid = current_valid[:, 1:2]
            if direct_s is not None:
                direct_s = torch.where(
                    observation_valid.unsqueeze(-1), direct_s, torch.zeros_like(direct_s)
                )
            if direct_v is not None:
                direct_v = torch.where(
                    observation_valid.unsqueeze(-1).unsqueeze(-1), direct_v, torch.zeros_like(direct_v)
                )
            current_direct_los = (direct_s, direct_v)
        k_s, k_v, value_s, value_v = self._project_current_kv(
            s, v, current_direct_los, mode=mode
        )
        key_mask = current_valid[:, None, :, None]
        vector_key_mask = key_mask.unsqueeze(-1)
        k_s = torch.where(key_mask, k_s, torch.zeros_like(k_s))
        value_s = torch.where(key_mask, value_s, torch.zeros_like(value_s))
        k_v = torch.where(vector_key_mask, k_v, torch.zeros_like(k_v))
        value_v = torch.where(vector_key_mask, value_v, torch.zeros_like(value_v))
        key_valid = current_valid
        if historical_kv_cache is not None:
            cache = historical_kv_cache
            cache_valid = torch.as_tensor(
                cache.valid_mask, device=s.device, dtype=torch.bool
            )
            if cache_valid.ndim == 1:
                cache_valid = cache_valid.unsqueeze(0)
            if cache_valid.ndim != 2:
                raise ValueError("historical KV valid_mask must be [B,N]")
            if int(cache_valid.shape[0]) == 1 and batch > 1:
                cache_valid = cache_valid.expand(batch, cache_valid.shape[1])
            if int(cache_valid.shape[0]) != batch:
                raise ValueError("historical KV cache batch dimension mismatch")
            history_tokens = int(cache_valid.shape[1])

            history_key_scalar = torch.as_tensor(
                cache.key_scalar, device=s.device, dtype=parameter.dtype
            )
            history_key_vector = torch.as_tensor(
                cache.key_vector, device=s.device, dtype=parameter.dtype
            )
            history_value_scalar = torch.as_tensor(
                cache.value_scalar, device=s.device, dtype=parameter.dtype
            )
            history_value_vector = torch.as_tensor(
                cache.value_vector, device=s.device, dtype=parameter.dtype
            )
            if history_key_scalar.ndim != 4 or history_key_vector.ndim != 5:
                raise ValueError("historical KV tensors have invalid rank")
            if history_value_scalar.ndim != 4 or history_value_vector.ndim != 5:
                raise ValueError("historical KV tensors have invalid rank")
            cache_tensors = (
                history_key_scalar,
                history_key_vector,
                history_value_scalar,
                history_value_vector,
            )
            if any(int(tensor.shape[0]) not in (1, batch) for tensor in cache_tensors):
                raise ValueError("historical KV cache batch dimension mismatch")
            if any(int(tensor.shape[0]) == 1 for tensor in cache_tensors) and batch > 1:
                history_key_scalar = history_key_scalar.expand(batch, *history_key_scalar.shape[1:])
                history_key_vector = history_key_vector.expand(batch, *history_key_vector.shape[1:])
                history_value_scalar = history_value_scalar.expand(batch, *history_value_scalar.shape[1:])
                history_value_vector = history_value_vector.expand(batch, *history_value_vector.shape[1:])
            expected_scalar = (batch, self.num_heads, history_tokens, self.head_dim)
            expected_vector = (batch, self.num_heads, history_tokens, self.head_dim, 3)
            if tuple(history_key_scalar.shape) != expected_scalar or tuple(history_value_scalar.shape) != expected_scalar:
                raise ValueError("historical scalar KV shape mismatch")
            if tuple(history_key_vector.shape) != expected_vector or tuple(history_value_vector.shape) != expected_vector:
                raise ValueError("historical vector KV shape mismatch")
            # Padding may contain arbitrary or non-finite storage.  Zero it
            # before the einsums: a zero attention weight multiplied by NaN
            # values would otherwise still produce NaN contexts.
            scalar_history_mask = cache_valid[:, None, :, None]
            vector_history_mask = scalar_history_mask.unsqueeze(-1)
            history_key_scalar = torch.where(
                scalar_history_mask, history_key_scalar, torch.zeros_like(history_key_scalar)
            )
            history_value_scalar = torch.where(
                scalar_history_mask, history_value_scalar, torch.zeros_like(history_value_scalar)
            )
            history_key_vector = torch.where(
                vector_history_mask, history_key_vector, torch.zeros_like(history_key_vector)
            )
            history_value_vector = torch.where(
                vector_history_mask, history_value_vector, torch.zeros_like(history_value_vector)
            )
            k_s = torch.cat((history_key_scalar, k_s), dim=2)
            k_v = torch.cat((history_key_vector, k_v), dim=2)
            value_s = torch.cat((history_value_scalar, value_s), dim=2)
            value_v = torch.cat((history_value_vector, value_v), dim=2)
            key_valid = torch.cat((cache_valid, current_valid), dim=1)
        key_count = int(k_s.shape[2])
        relation = torch.as_tensor(
            current_pairwise_geometry, device=s.device, dtype=parameter.dtype
        )
        if relation.ndim == 2:
            relation = relation.unsqueeze(0)
        if relation.ndim == 3:
            if int(relation.shape[0]) not in (1, batch):
                raise ValueError("frame relation batch dimension does not match current tokens")
            if tuple(relation.shape[1:]) != (key_count // 2, self.relation_dim):
                raise ValueError("frame relation must be [B,K,R] including current frame")
            relation = relation[:, None].expand(batch, 2, -1, -1).repeat_interleave(2, dim=2)
        elif relation.ndim == 4:
            if int(relation.shape[0]) not in (1, batch):
                raise ValueError("token relation batch dimension does not match current tokens")
            if tuple(relation.shape[1:]) != (2, key_count, self.relation_dim):
                raise ValueError("token relation must be [B,2,2K,R]")
        else:
            raise ValueError("current_pairwise_geometry must have rank 2, 3 or 4")
        if int(relation.shape[0]) == 1 and batch > 1:
            relation = relation.expand(batch, *relation.shape[1:])
        if attention_mask is None:
            allowed = torch.ones((batch, 2, key_count), device=s.device, dtype=torch.bool)
        else:
            raw_allowed = torch.as_tensor(attention_mask, device=s.device)
            if raw_allowed.dtype == torch.bool:
                allowed = raw_allowed
            else:
                allowed = torch.isfinite(raw_allowed) & (raw_allowed > -1e8)
            if allowed.ndim == 2:
                allowed = allowed.unsqueeze(0)
            if allowed.ndim == 3 and int(allowed.shape[0]) == 1 and batch > 1:
                allowed = allowed.expand(batch, *allowed.shape[1:])
            if tuple(allowed.shape) != (batch, 2, key_count):
                raise ValueError("online attention_mask must be [B,2,2K]")
        allowed = allowed & current_valid[:, :, None] & key_valid[:, None, :]
        relation = torch.where(allowed.unsqueeze(-1), relation, torch.zeros_like(relation))
        invariant = torch.cat((s, torch.linalg.vector_norm(v, dim=-1)), dim=-1)
        q_scalar = self.q_scalar(invariant).reshape(
            batch, 2, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        q_vector = torch.einsum(
            "bnci,hdc->bnhdi", v, self.q_vector_weight
        ).permute(0, 2, 1, 3, 4)
        rel_hidden = self.relation_mlp(relation).reshape(
            batch, 2, key_count, self.num_heads, self.head_dim
        ).permute(0, 3, 1, 2, 4)
        logits = torch.einsum(
            "bhqd,bhqkd->bhqk", q_scalar, k_s.unsqueeze(2) + rel_hidden
        ) * (float(self.head_dim) ** -0.5)
        logits = logits + torch.einsum(
            "bhqdc,bhkdc->bhqk", q_vector, k_v
        ) * (float(3 * self.head_dim) ** -0.5)
        if self.aggregation_mode == "dual_softmax_gated":
            alpha_state, alpha_obs, has_state, has_obs = self._dual_stream_softmax(
                logits,
                allowed,
                torch.arange(key_count, device=s.device),
                check_finite=(mode == "debug"),
            )
            # Dropout remains post-softmax and is applied independently to
            # both normalized streams. The two streams never share a divisor.
            alpha_state = self.dropout(alpha_state)
            alpha_obs = self.dropout(alpha_obs)
            context_state_scalar = torch.einsum(
                "bhqk,bhkd->bhqd", alpha_state, value_s
            )
            context_state_vector = torch.einsum(
                "bhqk,bhkdc->bhqdc", alpha_state, value_v
            )
            context_obs_scalar = torch.einsum(
                "bhqk,bhkd->bhqd", alpha_obs, value_s
            )
            context_obs_vector = torch.einsum(
                "bhqk,bhkdc->bhqdc", alpha_obs, value_v
            )
            update_s, update_v, gamma_state, gamma_obs = self._dual_context_updates(
                q_scalar,
                q_vector,
                context_state_scalar,
                context_state_vector,
                context_obs_scalar,
                context_obs_vector,
                has_state,
                has_obs,
            )
            weights = alpha_state + alpha_obs
            self._capture_gate_diagnostics(
                gate_diagnostics, gamma_state, gamma_obs
            )
        else:
            weights = self.dropout(
                safe_masked_softmax(
                    logits,
                    allowed[:, None],
                    dim=-1,
                    check_finite=(mode == "debug"),
                )
            )
            scalar_context = torch.einsum(
                "bhqk,bhkd->bhqd", weights, value_s
            ).permute(0, 2, 1, 3).reshape(batch, 2, -1)
            vector_context = torch.einsum(
                "bhqk,bhkdc->bhqdc", weights, value_v
            ).permute(0, 2, 1, 3, 4)
            update_s = self.output_scalar(scalar_context)
            update_v = torch.einsum(
                "bqhdc,ohd->bqoc", vector_context, self.output_vector_weight
            )
        residual_s = self.scalar_residual(s)
        if self.vector_residual_weight is None:
            residual_v = v
        else:
            residual_v = torch.einsum("bnci,oc->bnoi", v, self.vector_residual_weight)
        row_valid = allowed.any(dim=-1)
        out_s = torch.where(
            row_valid.unsqueeze(-1), residual_s + update_s, torch.zeros_like(update_s)
        )
        out_v = torch.where(
            row_valid.unsqueeze(-1).unsqueeze(-1),
            residual_v + update_v, torch.zeros_like(update_v),
        )
        new_cache = ScalarVectorKVCache(k_s, k_v, value_s, value_v, key_valid)
        return OnlineScalarVectorAttentionOutput(
            ScalarVectorAttentionOutput(out_s, out_v), new_cache,
            weights if return_attention else None,
            logits if return_logits else None,
        )

    def forward_current(
        self,
        current_scalar: Tensor,
        current_vector: Tensor,
        historical_kv_cache: ScalarVectorKVCache | BlockwiseKVCache | None,
        current_pairwise_geometry: Tensor | None,
        current_direct_los: Any = None,
        *,
        attention_mask: Tensor | None = None,
        current_valid_mask: Tensor | None = None,
        return_attention: bool = False,
        return_logits: bool = False,
        gate_diagnostics: dict[str, Any] | None = None,
        mode: str = "debug",
        attention_impl: str = "legacy_concat",
        block_frames: int = 32,
        current_geometry_inputs: Mapping[str, Tensor] | None = None,
        _reuse_sanitized_kv: bool = False,
    ) -> OnlineScalarVectorAttentionOutput:
        """Dispatch to the preserved reference or exact blockwise implementation."""
        if gate_diagnostics is not None:
            if mode != "debug":
                raise ValueError("Gate diagnostics require mode='debug'")
            if self.aggregation_mode != "dual_softmax_gated":
                raise ValueError("Gate diagnostics require V7-A")
            if len(gate_diagnostics) != 0:
                raise ValueError("Use an empty diagnostics dict per call")
        if attention_impl == "legacy_concat":
            if current_pairwise_geometry is None:
                raise ValueError("legacy_concat requires current_pairwise_geometry")
            if isinstance(historical_kv_cache, BlockwiseKVCache):
                raise TypeError("legacy_concat cannot consume a BlockwiseKVCache")
            return self.forward_current_legacy(
                current_scalar, current_vector, historical_kv_cache,
                current_pairwise_geometry, current_direct_los,
                attention_mask=attention_mask,
                current_valid_mask=current_valid_mask,
                return_attention=return_attention,
                return_logits=return_logits,
                gate_diagnostics=gate_diagnostics,
                mode=mode,
            )
        if attention_impl not in ("blockwise", "blockwise_checkpoint", "fused_tiles", "fused_tiles_save"):
            raise ValueError(f"unknown attention_impl: {attention_impl}")
        if attention_impl.startswith("fused_tiles") and (
            self.aggregation_mode != "unified" or self.dropout.p or block_frames != 32
        ):
            raise ValueError("fused_tiles requires unified attention, zero dropout and 32-frame blocks")
        # Debug and diagnostic requests use the inspectable exact reference.
        if attention_impl.startswith("fused_tiles") and (
            mode == "debug" or return_attention or return_logits
        ):
            attention_impl = "blockwise_checkpoint"
        return self._forward_current_blockwise(
            current_scalar, current_vector, historical_kv_cache,
            current_pairwise_geometry, current_direct_los,
            attention_mask=attention_mask,
            current_valid_mask=current_valid_mask,
            return_attention=return_attention,
            return_logits=return_logits,
            gate_diagnostics=gate_diagnostics,
            mode=mode,
            attention_impl=attention_impl,
            block_frames=block_frames,
            current_geometry_inputs=current_geometry_inputs,
            _reuse_sanitized_kv=_reuse_sanitized_kv,
        )

    def _forward_current_blockwise(
        self,
        current_scalar: Tensor,
        current_vector: Tensor,
        historical_kv_cache: ScalarVectorKVCache | BlockwiseKVCache | None,
        current_pairwise_geometry: Tensor | None,
        current_direct_los: Any = None,
        *,
        attention_mask: Tensor | None = None,
        current_valid_mask: Tensor | None = None,
        return_attention: bool = False,
        return_logits: bool = False,
        gate_diagnostics: dict[str, Any] | None = None,
        mode: str = "debug",
        attention_impl: str = "blockwise",
        block_frames: int = 32,
        current_geometry_inputs: Mapping[str, Tensor] | None = None,
        _reuse_sanitized_kv: bool = False,
    ) -> OnlineScalarVectorAttentionOutput:
        """Exact all-history attention over immutable KV blocks.

        Each local relation/logit/value reduction is bounded by one block. The
        online log-sum-exp merge uses one denominator over every valid key.
        """
        from .geometry_relation import build_current_pairwise_geometry

        if mode not in ("debug", "train_fast"):
            raise ValueError("mode must be 'debug' or 'train_fast'")
        if self.output_scalar_dim != self.scalar_dim or self.output_vector_channels != self.vector_channels:
            raise ValueError("online residual attention requires matching input/output channels")
        parameter = self.q_scalar.weight
        s = current_scalar.to(device=parameter.device, dtype=parameter.dtype)
        v = current_vector.to(device=parameter.device, dtype=parameter.dtype)
        if s.ndim != 3 or tuple(s.shape[1:]) != (2, self.scalar_dim):
            raise ValueError(f"current_scalar must be [B,2,{self.scalar_dim}]")
        if v.ndim != 4 or tuple(v.shape[1:]) != (2, self.vector_channels, 3):
            raise ValueError(f"current_vector must be [B,2,{self.vector_channels},3]")
        batch = int(s.shape[0])
        if current_valid_mask is None:
            if attention_mask is None:
                current_valid = torch.ones((batch, 2), device=s.device, dtype=torch.bool)
            else:
                raw_mask = torch.as_tensor(attention_mask, device=s.device)
                if raw_mask.ndim == 2:
                    raw_mask = raw_mask.unsqueeze(0)
                if raw_mask.ndim != 3 or int(raw_mask.shape[1]) != 2:
                    raise ValueError("online attention_mask must be [B,2,2K]")
                if raw_mask.shape[-1] < 2:
                    raise ValueError("online mask must include the current two key tokens")
                if int(raw_mask.shape[0]) == 1 and batch > 1:
                    raw_mask = raw_mask.expand(batch, *raw_mask.shape[1:])
                if int(raw_mask.shape[0]) != batch:
                    raise ValueError("online attention_mask batch dimension does not match current tokens")
                raw_allowed = raw_mask if raw_mask.dtype == torch.bool else (torch.isfinite(raw_mask) & (raw_mask > -1e8))
                current_valid = raw_allowed.any(dim=-1) | raw_allowed[..., -2:].any(dim=-2)
        else:
            current_valid = torch.as_tensor(current_valid_mask, device=s.device, dtype=torch.bool)
            if current_valid.ndim == 1:
                current_valid = current_valid.unsqueeze(0)
            if current_valid.ndim == 2 and int(current_valid.shape[0]) == 1 and batch > 1:
                current_valid = current_valid.expand(batch, 2)
            if tuple(current_valid.shape) != (batch, 2):
                raise ValueError("current_valid_mask must be [B,2]")

        s = torch.where(current_valid.unsqueeze(-1), s, torch.zeros_like(s))
        v = torch.where(current_valid.unsqueeze(-1).unsqueeze(-1), v, torch.zeros_like(v))
        if current_direct_los is not None:
            direct_s, direct_v = self._parse_los_direct(
                current_direct_los, batch=batch, dtype=s.dtype, device=s.device
            )
            observation_valid = current_valid[:, 1:2]
            if direct_s is not None:
                direct_s = torch.where(observation_valid.unsqueeze(-1), direct_s, torch.zeros_like(direct_s))
            if direct_v is not None:
                direct_v = torch.where(observation_valid.unsqueeze(-1).unsqueeze(-1), direct_v, torch.zeros_like(direct_v))
            current_direct_los = (direct_s, direct_v)

        # The current frame is projected once, then appended without mutation.
        k_s, k_v, value_s, value_v = self._project_current_kv(
            s, v, current_direct_los, mode=mode
        )
        key_mask = current_valid[:, None, :, None]
        vector_key_mask = key_mask.unsqueeze(-1)
        k_s = torch.where(key_mask, k_s, torch.zeros_like(k_s))
        value_s = torch.where(key_mask, value_s, torch.zeros_like(value_s))
        k_v = torch.where(vector_key_mask, k_v, torch.zeros_like(k_v))
        value_v = torch.where(vector_key_mask, value_v, torch.zeros_like(value_v))

        if historical_kv_cache is None:
            cache = BlockwiseKVCache(block_frames=int(block_frames))
        elif isinstance(historical_kv_cache, BlockwiseKVCache):
            cache = historical_kv_cache
            if cache.block_frames != int(block_frames):
                raise ValueError("block_frames cannot change within one recursive sequence")
        else:
            cache = BlockwiseKVCache.from_legacy(
                historical_kv_cache, block_frames=int(block_frames)
            )
        # V7-A:
        # Historical KV remains append-only.
        # Future smoother-conditioned history refresh
        # will require explicit replacement/re-encoding.
        # No such refresh is implemented in this stage.
        fused = attention_impl.startswith("fused_tiles")
        cache = cache.append_frame(
            k_s, k_v, value_s, value_v, current_valid, pack_groups=fused
        )
        if cache.total_tokens != 2 * cache.total_frames:
            raise RuntimeError("blockwise KV token/frame counts disagree")

        invariant = torch.cat((s, torch.linalg.vector_norm(v, dim=-1)), dim=-1)
        q_scalar = self.q_scalar(invariant).reshape(
            batch, 2, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        q_vector = torch.einsum(
            "bnci,hdc->bnhdi", v, self.q_vector_weight
        ).permute(0, 2, 1, 3, 4)

        if current_geometry_inputs is not None:
            if current_pairwise_geometry is not None:
                raise ValueError("pass geometry inputs or precomputed relation, not both")
            if cache.total_frames != int(current_geometry_inputs["history_times_sec"].shape[1]):
                raise ValueError("geometry history must end at the current frame")
            if "normalization" not in current_geometry_inputs:
                raise ValueError("current_geometry_inputs requires normalization")
            normalization = current_geometry_inputs["normalization"]
        else:
            if current_pairwise_geometry is None:
                raise ValueError("blockwise attention requires relation or geometry inputs")
            relation_source = torch.as_tensor(
                current_pairwise_geometry, device=s.device, dtype=parameter.dtype
            )
            if relation_source.ndim == 2:
                relation_source = relation_source.unsqueeze(0)
            if relation_source.ndim == 3:
                if relation_source.shape[-1] != self.relation_dim or relation_source.shape[1] != cache.total_frames:
                    raise ValueError("frame relation must include exactly the full frame history")
                if relation_source.shape[0] not in (1, batch):
                    raise ValueError("frame relation batch dimension mismatch")
                if relation_source.shape[0] == 1 and batch > 1:
                    relation_source = relation_source.expand(batch, *relation_source.shape[1:])
            elif relation_source.ndim == 4:
                if tuple(relation_source.shape[1:]) != (2, cache.total_tokens, self.relation_dim):
                    raise ValueError("token relation must be [B,2,2T,R]")
                if relation_source.shape[0] not in (1, batch):
                    raise ValueError("token relation batch dimension mismatch")
                if relation_source.shape[0] == 1 and batch > 1:
                    relation_source = relation_source.expand(batch, *relation_source.shape[1:])
            else:
                raise ValueError("current_pairwise_geometry must have rank 2, 3 or 4")
            normalization = None

        if attention_mask is None:
            full_allowed = None
        else:
            full_allowed = torch.as_tensor(attention_mask, device=s.device)
            if full_allowed.dtype != torch.bool:
                full_allowed = torch.isfinite(full_allowed) & (full_allowed > -1e8)
            if full_allowed.ndim == 2:
                full_allowed = full_allowed.unsqueeze(0)
            if full_allowed.ndim == 3 and full_allowed.shape[0] == 1 and batch > 1:
                full_allowed = full_allowed.expand(batch, *full_allowed.shape[1:])
            if tuple(full_allowed.shape) != (batch, 2, cache.total_tokens):
                raise ValueError("online attention_mask must be [B,2,2T]")

        if fused:
            scalar_context, vector_context, row_valid = self._fused_context(
                q_scalar, q_vector, cache, current_valid, full_allowed,
                None if current_geometry_inputs is not None else relation_source,
                current_geometry_inputs,
                recompute=attention_impl == "fused_tiles",
            )
            update_s = self.output_scalar(scalar_context.permute(0, 2, 1, 3).reshape(batch, 2, -1))
            update_v = torch.einsum(
                "bqhdc,ohd->bqoc", vector_context.permute(0, 2, 1, 3, 4),
                self.output_vector_weight,
            )
            return self._online_residual(s, v, update_s, update_v, row_valid, cache)

        running_max: Tensor | None = None
        running_denominator: Tensor | None = None
        running_scalar_numerator: Tensor | None = None
        running_vector_numerator: Tensor | None = None
        running_has_valid: Tensor | None = None
        running_state_stats: tuple[Tensor, Tensor, Tensor, Tensor, Tensor] | None = None
        running_obs_stats: tuple[Tensor, Tensor, Tensor, Tensor, Tensor] | None = None
        diagnostic_logits: list[Tensor] = []
        diagnostic_exp: list[Tensor] = []
        diagnostic_max: list[Tensor] = []
        diagnostic_has: list[Tensor] = []
        diagnostic_state_exp: list[Tensor] = []
        diagnostic_obs_exp: list[Tensor] = []
        diagnostic_state_max: list[Tensor] = []
        diagnostic_obs_max: list[Tensor] = []
        diagnostic_state_has: list[Tensor] = []
        diagnostic_obs_has: list[Tensor] = []
        use_checkpoint = (
            attention_impl == "blockwise_checkpoint" and torch.is_grad_enabled()
        )

        for first_frame, block in cache.iter_blocks():
            block_frames_here = int(block.valid_mask.shape[1]) // 2
            token_start, token_end = 2 * first_frame, 2 * (first_frame + block_frames_here)
            block_valid = block.valid_mask.to(device=s.device, dtype=torch.bool)
            if _reuse_sanitized_kv:
                # The recursive training loop owns its cache throughout this call.
                # Its frame K/V were zeroed before append, and block joins only copy.
                (
                    block_key_scalar, block_key_vector,
                    block_value_scalar, block_value_vector,
                ) = block[:4]
            else:
                scalar_key_mask = block_valid[:, None, :, None]
                vector_key_mask = scalar_key_mask.unsqueeze(-1)
                # Invalid slots retain their logical positions in the cache, but
                # dirty storage must not enter either logits or 0 * NaN value
                # products. This temporary masked view preserves gradients for
                # every valid historical KV and leaves the append-only cache intact.
                block_key_scalar = torch.where(
                    scalar_key_mask, block.key_scalar, torch.zeros_like(block.key_scalar)
                )
                block_value_scalar = torch.where(
                    scalar_key_mask, block.value_scalar, torch.zeros_like(block.value_scalar)
                )
                block_key_vector = torch.where(
                    vector_key_mask, block.key_vector, torch.zeros_like(block.key_vector)
                )
                block_value_vector = torch.where(
                    vector_key_mask, block.value_vector, torch.zeros_like(block.value_vector)
                )
            if full_allowed is None:
                extra_allowed = torch.ones(
                    (batch, 2, token_end - token_start), device=s.device, dtype=torch.bool
                )
            else:
                extra_allowed = full_allowed[:, :, token_start:token_end]

            if current_geometry_inputs is not None:
                physical = current_geometry_inputs
                geo_args = (
                    physical["current_time_sec"],
                    physical["history_times_sec"][:, first_frame:first_frame + block_frames_here],
                    physical["current_station_state_si"],
                    physical["history_station_state_si"][:, first_frame:first_frame + block_frames_here],
                    physical["current_observed_los"],
                    physical["history_observed_los"][:, first_frame:first_frame + block_frames_here],
                    physical["current_station_id"],
                    physical["history_station_ids"][:, first_frame:first_frame + block_frames_here],
                    physical["current_station_id_reliable"],
                    physical["history_station_id_reliable"][:, first_frame:first_frame + block_frames_here],
                    physical["current_valid"],
                    physical["history_valid"][:, first_frame:first_frame + block_frames_here],
                )

                def block_stats_fn(
                    qs: Tensor, qv: Tensor, ks: Tensor, kv: Tensor,
                    vs: Tensor, vv: Tensor, extra: Tensor, valid_keys: Tensor,
                    current_time: Tensor, history_times: Tensor,
                    current_station: Tensor, history_station: Tensor,
                    current_los: Tensor, history_los: Tensor,
                    current_id: Tensor, history_ids: Tensor,
                    current_reliable: Tensor, history_reliable: Tensor,
                    query_valid: Tensor, history_valid: Tensor,
                    global_key_indices: Tensor,
                ) -> tuple[Tensor, ...]:
                    rel, pair_valid = build_current_pairwise_geometry(
                        current_time, history_times, current_station, history_station,
                        current_los, history_los, normalization,
                        current_station_id=current_id, history_station_ids=history_ids,
                        current_station_id_reliable=current_reliable,
                        history_station_id_reliable=history_reliable,
                        current_valid=query_valid, history_valid=history_valid,
                    )
                    frame_count = int(rel.shape[1])
                    relation = rel[:, None].expand(batch, 2, frame_count, self.relation_dim).repeat_interleave(2, dim=2)
                    allowed_local = pair_valid[:, None].expand(batch, 2, frame_count).repeat_interleave(2, dim=-1)
                    allowed_local = allowed_local & extra & query_valid[:, None, None] & valid_keys[:, None, :]
                    return self._block_stats(
                        qs, qv, ks, kv, vs, vv, relation, allowed_local,
                        global_key_indices,
                        check_finite=(mode == "debug"), diagnostics=(return_attention or return_logits),
                    )
            else:
                if relation_source.ndim == 3:
                    rel = relation_source[:, first_frame:first_frame + block_frames_here]
                    relation = rel[:, None].expand(batch, 2, block_frames_here, self.relation_dim).repeat_interleave(2, dim=2)
                else:
                    relation = relation_source[:, :, token_start:token_end]

                def block_stats_fn(
                    qs: Tensor, qv: Tensor, ks: Tensor, kv: Tensor,
                    vs: Tensor, vv: Tensor, extra: Tensor, valid_keys: Tensor,
                    rel_block: Tensor,
                    global_key_indices: Tensor,
                ) -> tuple[Tensor, ...]:
                    allowed_local = extra & current_valid[:, :, None] & valid_keys[:, None, :]
                    return self._block_stats(
                        qs, qv, ks, kv, vs, vv, rel_block, allowed_local,
                        global_key_indices,
                        check_finite=(mode == "debug"), diagnostics=(return_attention or return_logits),
                    )
                geo_args = (relation,)

            stats_args = (
                q_scalar, q_vector, block_key_scalar, block_key_vector,
                block_value_scalar, block_value_vector, extra_allowed, block_valid,
                *geo_args,
                torch.arange(token_start, token_end, device=s.device),
            )
            if use_checkpoint:
                stats = checkpoint(
                    block_stats_fn, *stats_args,
                    use_reentrant=False, preserve_rng_state=True,
                )
            else:
                stats = block_stats_fn(*stats_args)
            if self.aggregation_mode == "dual_softmax_gated":
                state_block = stats[:5]
                obs_block = stats[5:10]
                running_state_stats = self._merge_block_stream_stats(
                    running_state_stats, state_block
                )
                running_obs_stats = self._merge_block_stream_stats(
                    running_obs_stats, obs_block
                )
                if return_attention or return_logits:
                    diagnostic_logits.append(stats[10])
                    diagnostic_state_exp.append(stats[11])
                    diagnostic_obs_exp.append(stats[12])
                    diagnostic_state_max.append(state_block[0])
                    diagnostic_obs_max.append(obs_block[0])
                    diagnostic_state_has.append(state_block[4])
                    diagnostic_obs_has.append(obs_block[4])
            else:
                block_max, block_denominator, block_scalar_numerator, block_vector_numerator, block_has_valid = stats[:5]
                if running_max is None:
                    running_max = torch.zeros_like(block_max)
                    running_denominator = torch.zeros_like(block_denominator)
                    running_scalar_numerator = torch.zeros_like(block_scalar_numerator)
                    running_vector_numerator = torch.zeros_like(block_vector_numerator)
                    running_has_valid = torch.zeros_like(block_has_valid, dtype=torch.bool)

                new_has = running_has_valid | block_has_valid
                both = running_has_valid & block_has_valid
                new_max = torch.where(
                    both,
                    torch.maximum(running_max, block_max),
                    torch.where(running_has_valid, running_max,
                                torch.where(block_has_valid, block_max, torch.zeros_like(block_max))),
                )
                old_delta = torch.where(running_has_valid, running_max - new_max, torch.zeros_like(new_max))
                new_delta = torch.where(block_has_valid, block_max - new_max, torch.zeros_like(new_max))
                old_factor = torch.where(running_has_valid, torch.exp(old_delta), torch.zeros_like(old_delta))
                new_factor = torch.where(block_has_valid, torch.exp(new_delta), torch.zeros_like(new_delta))
                running_denominator = old_factor * running_denominator + new_factor * block_denominator
                running_scalar_numerator = old_factor * running_scalar_numerator + new_factor * block_scalar_numerator
                running_vector_numerator = old_factor.unsqueeze(-1) * running_vector_numerator + new_factor.unsqueeze(-1) * block_vector_numerator
                running_max = new_max
                running_has_valid = new_has
                if return_attention or return_logits:
                    diagnostic_logits.append(stats[5])
                    diagnostic_exp.append(stats[6])
                    diagnostic_max.append(block_max)
                    diagnostic_has.append(block_has_valid)

        if self.aggregation_mode == "dual_softmax_gated":
            assert running_state_stats is not None and running_obs_stats is not None
            state_max, state_z, state_scalar_num, state_vector_num, state_has = running_state_stats
            obs_max, obs_z, obs_scalar_num, obs_vector_num, obs_has = running_obs_stats
            state_denominator = state_z.clamp_min(torch.finfo(state_z.dtype).tiny)
            obs_denominator = obs_z.clamp_min(torch.finfo(obs_z.dtype).tiny)
            context_state_scalar = state_scalar_num / state_denominator
            context_state_vector = state_vector_num / state_denominator.unsqueeze(-1)
            context_obs_scalar = obs_scalar_num / obs_denominator
            context_obs_vector = obs_vector_num / obs_denominator.unsqueeze(-1)
            update_s, update_v, gamma_state, gamma_obs = self._dual_context_updates(
                q_scalar,
                q_vector,
                context_state_scalar,
                context_state_vector,
                context_obs_scalar,
                context_obs_vector,
                state_has,
                obs_has,
            )
            self._capture_gate_diagnostics(
                gate_diagnostics, gamma_state, gamma_obs
            )
            row_valid = (state_has | obs_has)[:, 0, :, 0]
        else:
            assert running_max is not None and running_denominator is not None
            assert running_scalar_numerator is not None and running_vector_numerator is not None
            assert running_has_valid is not None
            denominator = running_denominator.clamp_min(torch.finfo(running_denominator.dtype).tiny)
            scalar_context = (running_scalar_numerator / denominator).permute(0, 2, 1, 3).reshape(batch, 2, -1)
            vector_context = (running_vector_numerator / denominator.unsqueeze(-1)).permute(0, 2, 1, 3, 4)
            update_s = self.output_scalar(scalar_context)
            update_v = torch.einsum("bqhdc,ohd->bqoc", vector_context, self.output_vector_weight)
            row_valid = running_has_valid[:, 0, :, 0]
        weights = logits_out = None
        if return_attention or return_logits:
            if return_logits:
                logits_out = torch.cat(diagnostic_logits, dim=-1)
            if return_attention:
                if self.aggregation_mode == "dual_softmax_gated":
                    state_probabilities = []
                    obs_probabilities = []
                    for exp_weight, block_max, block_has in zip(
                        diagnostic_state_exp, diagnostic_state_max, diagnostic_state_has
                    ):
                        delta = torch.where(
                            block_has, block_max - state_max, torch.zeros_like(block_max)
                        )
                        factor = torch.where(block_has, torch.exp(delta), torch.zeros_like(delta))
                        state_probabilities.append(
                            exp_weight * factor / state_denominator
                        )
                    for exp_weight, block_max, block_has in zip(
                        diagnostic_obs_exp, diagnostic_obs_max, diagnostic_obs_has
                    ):
                        delta = torch.where(
                            block_has, block_max - obs_max, torch.zeros_like(block_max)
                        )
                        factor = torch.where(block_has, torch.exp(delta), torch.zeros_like(delta))
                        obs_probabilities.append(exp_weight * factor / obs_denominator)
                    weights = torch.cat(state_probabilities, dim=-1) + torch.cat(
                        obs_probabilities, dim=-1
                    )
                else:
                    probabilities = []
                    for exp_weight, block_max, block_has in zip(
                        diagnostic_exp, diagnostic_max, diagnostic_has
                    ):
                        delta = torch.where(
                            block_has, block_max - running_max,
                            torch.zeros_like(block_max),
                        )
                        factor = torch.where(
                            block_has, torch.exp(delta), torch.zeros_like(delta)
                        )
                        probabilities.append(exp_weight * factor / denominator)
                    weights = torch.cat(probabilities, dim=-1)
        return self._online_residual(s, v, update_s, update_v, row_valid, cache, weights, logits_out)

    def _online_residual(
        self, s, v, update_s, update_v, row_valid, cache, weights=None, logits=None,
    ) -> OnlineScalarVectorAttentionOutput:
        residual_s = self.scalar_residual(s)
        residual_v = v if self.vector_residual_weight is None else torch.einsum(
            "bnci,oc->bnoi", v, self.vector_residual_weight
        )
        out_s = torch.where(row_valid[..., None], residual_s + update_s, 0.0)
        out_v = torch.where(row_valid[..., None, None], residual_v + update_v, 0.0)
        return OnlineScalarVectorAttentionOutput(
            ScalarVectorAttentionOutput(out_s, out_v), cache, weights, logits
        )

    def _fused_context(
        self, qs, qv, cache, current_valid, full_allowed, relation, physical, *, recompute,
    ):
        from .fused_attention_tiles import tile_context
        from .geometry_relation import build_current_pairwise_geometry

        batch = qs.shape[0]
        query_valid = current_valid
        if physical is not None:
            query_valid = physical["current_valid"].reshape(batch, 1).expand(-1, 2)
            relation, pair_valid = build_current_pairwise_geometry(
                physical["current_time_sec"], physical["history_times_sec"],
                physical["current_station_state_si"], physical["history_station_state_si"],
                physical["current_observed_los"], physical["history_observed_los"],
                physical["normalization"],
                current_station_id=physical["current_station_id"],
                history_station_ids=physical["history_station_ids"],
                current_station_id_reliable=physical["current_station_id_reliable"],
                history_station_id_reliable=physical["history_station_id_reliable"],
                current_valid=physical["current_valid"], history_valid=physical["history_valid"],
            )
            pair_valid = pair_valid.repeat_interleave(2, dim=-1)
            full_allowed = pair_valid[:, None] if full_allowed is None else full_allowed & pair_valid[:, None]

        def group_stats(qs, qv, ks, kv, vs, vv, rel, allowed):
            groups, keys = ks.shape[1], ks.shape[3]
            if rel.ndim == 3:
                # One shared relation per frame. Masked edges never read its
                # embedding, so they contribute zero to the shared gradient.
                rel = torch.where(
                    allowed.reshape(batch, groups, 2, keys // 2, 2).any(dim=(2, 4))
                    .reshape(batch, -1, 1), rel, 0.0,
                )
                hidden = self.relation_mlp(rel.to(qs.dtype)).reshape(
                    batch, groups, keys // 2, self.num_heads, self.head_dim
                )
                hidden = hidden.repeat_interleave(2, dim=2).permute(0, 1, 3, 2, 4)
                hidden = hidden.unsqueeze(3).expand(-1, -1, -1, 2, -1, -1)
            else:
                rel = rel.reshape(batch, 2, groups, keys, self.relation_dim).transpose(1, 2)
                rel = torch.where(allowed[..., None], rel.to(qs.dtype), 0.0)
                hidden = self.relation_mlp(rel).reshape(
                    batch, groups, 2, keys, self.num_heads, self.head_dim
                ).permute(0, 1, 4, 2, 3, 5)
            return tile_context(qs, qv, ks, kv, vs, vv, hidden, allowed)

        statistics = []
        for first, block in cache.iter_packed_groups():
            groups, keys = block.valid_mask.shape[1:]
            stop = first + groups * keys // 2
            token_slice = slice(2 * first, 2 * stop)
            allowed = query_valid[:, None, :, None] & block.valid_mask[:, :, None, :]
            if full_allowed is not None:
                extra = full_allowed[..., token_slice].reshape(batch, -1, groups, keys)
                allowed = allowed & extra.transpose(1, 2)
            rel = relation[:, first:stop] if relation.ndim == 3 else relation[:, :, token_slice]
            args = (qs, qv, *block[:4], rel, allowed)
            if recompute and torch.is_grad_enabled():
                stats = checkpoint(group_stats, *args, use_reentrant=False, preserve_rng_state=False)
            else:
                stats = group_stats(*args)
            statistics.append(stats)

        # Reduce only the compact (max, sum, weighted value) tile statistics.
        # Fixing the max as a scale is valid for the normalized quotient.
        maximum, denominator, scalar, vector = (
            torch.cat(tiles, dim=1) for tiles in zip(*statistics)
        )
        if qs.is_cuda and qs.dtype == torch.float64:
            from .fused_attention_reduce import reduce_tiles
            return reduce_tiles(maximum, denominator, scalar, vector)
        valid = denominator > 0
        maximum = maximum.masked_fill(~valid, -torch.inf)
        row_valid = valid.any(dim=1)
        global_max = torch.where(row_valid, maximum.amax(dim=1), 0.0)
        scale = torch.exp(maximum - global_max[:, None])
        total = (scale * denominator).sum(dim=1).clamp_min(torch.finfo(qs.dtype).tiny)
        scalar = (scale * scalar).sum(dim=1) / total
        vector = (scale.unsqueeze(-1) * vector).sum(dim=1) / total.unsqueeze(-1)
        return scalar, vector, row_valid[:, 0, :, 0]

    def _block_stats(
        self,
        q_scalar: Tensor,
        q_vector: Tensor,
        key_scalar: Tensor,
        key_vector: Tensor,
        value_scalar: Tensor,
        value_vector: Tensor,
        relation: Tensor,
        allowed: Tensor,
        global_key_indices: Tensor,
        *,
        check_finite: bool,
        diagnostics: bool,
    ) -> tuple[Tensor, ...]:
        """Compute one block's local max, denominator, and weighted numerators."""
        relation = relation.to(device=q_scalar.device, dtype=q_scalar.dtype)
        # The frozen implementation zeroes disallowed relations before the
        # RelationMLP; preserve that diagnostic-logit and padding behavior.
        relation = torch.where(allowed.unsqueeze(-1), relation, torch.zeros_like(relation))
        batch, key_count = int(key_scalar.shape[0]), int(key_scalar.shape[2])
        rel_hidden = self.relation_mlp(relation).reshape(
            batch, 2, key_count, self.num_heads, self.head_dim
        ).permute(0, 3, 1, 2, 4)
        logits = torch.einsum(
            "bhqd,bhqkd->bhqk", q_scalar, key_scalar.unsqueeze(2) + rel_hidden
        ) * (float(self.head_dim) ** -0.5)
        logits = logits + torch.einsum(
            "bhqdc,bhkdc->bhqk", q_vector, key_vector
        ) * (float(3 * self.head_dim) ** -0.5)
        allowed_h = allowed[:, None]
        if check_finite and not bool((torch.isfinite(logits) | ~allowed_h).all()):
            raise FloatingPointError("non-finite attention logit on an allowed edge")
        if self.aggregation_mode == "dual_softmax_gated":
            state_key = (global_key_indices.to(device=logits.device) % 2) == 0

            def stream_stats(stream_allowed: Tensor) -> tuple[Tensor, ...]:
                # Mask before exp so invalid keys never enter it with a finite
                # surrogate logit.
                stream_logits = logits.masked_fill(~stream_allowed, -torch.inf)
                has_valid = stream_allowed.any(dim=-1, keepdim=True)
                raw_max = stream_logits.max(dim=-1, keepdim=True).values
                # Empty streams use max=0; masked logits remain -inf, so exp=0.
                block_max = torch.where(
                    has_valid, raw_max, torch.zeros_like(raw_max)
                )
                shifted = stream_logits - block_max
                exponentials = torch.exp(shifted)
                denominator = exponentials.sum(dim=-1, keepdim=True)
                dropped_exponentials = self.dropout(exponentials)
                scalar_numerator = torch.einsum(
                    "bhqk,bhkd->bhqd", dropped_exponentials, value_scalar
                )
                vector_numerator = torch.einsum(
                    "bhqk,bhkdc->bhqdc", dropped_exponentials, value_vector
                )
                return (
                    block_max, denominator, scalar_numerator,
                    vector_numerator, has_valid, dropped_exponentials,
                )

            state_allowed = allowed_h & state_key.view(1, 1, 1, key_count)
            obs_allowed = allowed_h & ~state_key.view(1, 1, 1, key_count)
            state_stats = stream_stats(state_allowed)
            obs_stats = stream_stats(obs_allowed)
            result = state_stats[:5] + obs_stats[:5]
            if diagnostics:
                result = result + (logits, state_stats[5], obs_stats[5])
            return result

        safe_logits = torch.where(allowed_h, logits, torch.zeros_like(logits))
        block_has_valid = allowed_h.any(dim=-1, keepdim=True)
        raw_max = safe_logits.masked_fill(~allowed_h, -torch.inf).max(dim=-1, keepdim=True).values
        block_max = torch.where(block_has_valid, raw_max, torch.zeros_like(raw_max))
        shifted = safe_logits - block_max
        exponentials = torch.where(allowed_h, torch.exp(shifted), torch.zeros_like(shifted))
        denominator = exponentials.sum(dim=-1, keepdim=True)
        dropped_exponentials = self.dropout(exponentials)
        scalar_numerator = torch.einsum(
            "bhqk,bhkd->bhqd", dropped_exponentials, value_scalar
        )
        vector_numerator = torch.einsum(
            "bhqk,bhkdc->bhqdc", dropped_exponentials, value_vector
        )
        result: tuple[Tensor, ...] = (
            block_max, denominator, scalar_numerator, vector_numerator, block_has_valid,
        )
        if diagnostics:
            result = result + (logits, dropped_exponentials)
        return result

    def reference_prefix_attention(
        self, scalar_history: Tensor, vector_history: Tensor,
        frame_relation: Tensor, direct_los_history: Any = None,
    ) -> ScalarVectorAttentionOutput:
        """Run full prefix attention and return its last interleaved pair."""
        if scalar_history.ndim != 3 or vector_history.ndim != 4:
            raise ValueError("reference histories must be batched token sequences")
        tokens = int(scalar_history.shape[1])
        if tokens % 2:
            raise ValueError("interleaved history must have two tokens per frame")
        frames = tokens // 2
        causal = torch.tril(
            torch.ones((frames, frames), device=scalar_history.device, dtype=torch.bool)
        )
        frame_index = torch.arange(frames, device=scalar_history.device).repeat_interleave(2)
        result = self.forward(
            scalar_history, vector_history, relation=frame_relation,
            attention_mask=causal, los_direct=direct_los_history,
            token_frame_index=frame_index,
            observation_token_mask=(torch.arange(tokens, device=scalar_history.device) % 2 == 1),
        )
        return ScalarVectorAttentionOutput(result.scalar[:, -2:], result.vector[:, -2:])

    def forward(self, s: Tensor, v: Tensor, g: Tensor | None = None, attention_mask: Tensor | None = None, los_direct: Any = None, direct_vector: Tensor | None = None, *, direct_scalar: Tensor | None = None, relation: Tensor | None = None, attn_mask: Tensor | None = None, mask: Tensor | None = None, token_frame_index: Tensor | None = None, observation_token_mask: Tensor | None = None, return_attention: bool = False, return_logits: bool = False, return_diagnostics: bool = False, return_weights: bool | None = None, query_chunk_size: int | None = None) -> tuple[Tensor, Tensor] | tuple[Tensor, ...]:
        """Apply attention; masks use ``True`` for allowed edges."""

        if relation is not None:
            if g is not None:
                raise ValueError("provide only one of g and relation")
            g = relation
        if direct_scalar is not None:
            if los_direct is not None:
                raise ValueError("provide direct_scalar either as los_direct or as a keyword")
            los_direct = direct_scalar
        if direct_vector is not None:
            if los_direct is None:
                los_direct = (None, direct_vector)
            elif isinstance(los_direct, (tuple, list)):
                if len(los_direct) != 2 or los_direct[1] is not None:
                    raise ValueError("los_direct already contains a vector lane")
                los_direct = (los_direct[0], direct_vector)
            else:
                los_direct = (los_direct, direct_vector)
        if attn_mask is not None:
            if attention_mask is not None:
                raise ValueError("provide only one of attention_mask and attn_mask")
            attention_mask = attn_mask
        if mask is not None:
            if attention_mask is not None:
                raise ValueError("provide only one of attention_mask, attn_mask and mask")
            attention_mask = mask
        if return_weights is not None:
            return_attention = bool(return_weights)
        if s.ndim != 3 or v.ndim != 4 or int(v.shape[-1]) != 3:
            raise ValueError("s must be [B,N,Cs], v must be [B,N,Cv,3]")
        batch, tokens, channels = map(int, s.shape)
        if channels != self.scalar_dim or tuple(v.shape[:2]) != (batch, tokens) or int(v.shape[2]) != self.vector_channels:
            raise ValueError(f"expected s [B,N,{self.scalar_dim}] and v [B,N,{self.vector_channels},3]")
        parameter = self.q_scalar.weight
        s = s.to(device=parameter.device, dtype=parameter.dtype)
        v = v.to(device=parameter.device, dtype=parameter.dtype)
        relation_is_frame_level = False
        if g is None:
            g = torch.zeros((batch, tokens, tokens, self.relation_dim), device=parameter.device, dtype=parameter.dtype)
        else:
            g = torch.as_tensor(g, device=parameter.device, dtype=parameter.dtype)
            frames = tokens // 2 if tokens % 2 == 0 else -1
            if g.ndim == 3:
                if tuple(g.shape) == (tokens, tokens, self.relation_dim):
                    g = g.unsqueeze(0)
                elif frames > 0 and tuple(g.shape) == (frames, frames, self.relation_dim):
                    g = g.unsqueeze(0)
                    relation_is_frame_level = True
                else:
                    raise ValueError(
                        f"g must be [N,N,{self.relation_dim}] or [T,T,{self.relation_dim}]"
                    )
            elif g.ndim == 4:
                if tuple(g.shape[1:]) == (tokens, tokens, self.relation_dim):
                    relation_is_frame_level = False
                elif frames > 0 and tuple(g.shape[1:]) == (frames, frames, self.relation_dim):
                    relation_is_frame_level = True
                else:
                    raise ValueError(
                        f"g must be [B,N,N,{self.relation_dim}] or [B,T,T,{self.relation_dim}]"
                    )
            else:
                raise ValueError("g must have rank 3 or 4")
            if int(g.shape[0]) == 1 and batch > 1:
                g = g.expand(batch, *g.shape[1:])
            if int(g.shape[0]) != batch:
                raise ValueError("relation batch dimension does not match token batch")
        allowed, additive = self._resolve_mask(attention_mask, batch=batch, tokens=tokens, device=parameter.device, token_frame_index=token_frame_index)
        query_s, query_v, key_s, key_v = s, v, s, v
        key_has_edge = None
        if self.aggregation_mode == "dual_softmax_gated":
            query_has_edge = allowed.any(dim=-1)
            key_has_edge = allowed.any(dim=1)
            query_s = torch.where(query_has_edge.unsqueeze(-1), s, torch.zeros_like(s))
            query_v = torch.where(
                query_has_edge.unsqueeze(-1).unsqueeze(-1), v, torch.zeros_like(v)
            )
            key_s = torch.where(key_has_edge.unsqueeze(-1), s, torch.zeros_like(s))
            key_v = torch.where(
                key_has_edge.unsqueeze(-1).unsqueeze(-1), v, torch.zeros_like(v)
            )
        q_invariant = torch.cat((query_s, torch.linalg.vector_norm(query_v, dim=-1)), dim=-1)
        k_invariant = torch.cat((key_s, torch.linalg.vector_norm(key_v, dim=-1)), dim=-1)
        q_scalar = self.q_scalar(q_invariant).reshape(batch, tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_scalar = self.k_scalar(k_invariant).reshape(batch, tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q_vector = torch.einsum("bnci,hdc->bnhdi", query_v, self.q_vector_weight).permute(0, 2, 1, 3, 4)
        k_vector = torch.einsum("bnci,hdc->bnhdi", key_v, self.k_vector_weight).permute(0, 2, 1, 3, 4)
        scalar_values = self.value_scalar(key_s).reshape(batch, tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        vector_values = torch.einsum("bnci,hdc->bnhdi", key_v, self.value_vector_weight).permute(0, 2, 1, 3, 4)
        scalar_values, vector_values = self._inject_direct_values(scalar_values, vector_values, los_direct, batch=batch, tokens=tokens, token_frame_index=token_frame_index, observation_token_mask=observation_token_mask)
        if key_has_edge is not None:
            key_scalar_mask = key_has_edge[:, None, :, None]
            key_vector_mask = key_scalar_mask.unsqueeze(-1)
            scalar_values = torch.where(key_scalar_mask, scalar_values, torch.zeros_like(scalar_values))
            vector_values = torch.where(key_vector_mask, vector_values, torch.zeros_like(vector_values))
            k_scalar = torch.where(key_scalar_mask, k_scalar, torch.zeros_like(k_scalar))
            k_vector = torch.where(key_vector_mask, k_vector, torch.zeros_like(k_vector))
        chunk = int(query_chunk_size or self.query_chunk_size or tokens)
        if chunk <= 0:
            raise ValueError("query_chunk_size must be positive")
        scalar_outputs: list[Tensor] = []
        vector_outputs: list[Tensor] = []
        attention_outputs: list[Tensor] = []
        logit_outputs: list[Tensor] = []
        finite_logit_count = 0
        total_logit_count = 0
        finite_allowed_logit_count = 0
        total_allowed_logit_count = 0
        scalar_scale = float(self.head_dim) ** -0.5
        vector_scale = float(3 * self.head_dim) ** -0.5
        for start in range(0, tokens, chunk):
            stop = min(tokens, start + chunk)
            q_chunk = q_scalar[:, :, start:stop]
            # Project only the current query block's pairwise relations. This
            # bounds the [B,H,Q,N,D] relation-key workspace while retaining
            # every historical key/value frame.
            if relation_is_frame_level:
                if token_frame_index is None:
                    frame_map = (
                        torch.arange(tokens, device=parameter.device) // 2
                    ).unsqueeze(0).expand(batch, tokens)
                else:
                    frame_map = torch.as_tensor(token_frame_index, device=parameter.device)
                    if frame_map.ndim == 1:
                        frame_map = frame_map.unsqueeze(0).expand(batch, tokens)
                    if tuple(frame_map.shape) != (batch, tokens):
                        raise ValueError("token_frame_index must match [B,N] for frame relations")
                batch_index = torch.arange(batch, device=parameter.device)[:, None, None]
                query_frames = frame_map[:, start:stop, None]
                key_frames = frame_map[:, None, :]
                relation_input = g[batch_index, query_frames, key_frames]
            else:
                relation_input = g[:, start:stop]
            if self.aggregation_mode == "dual_softmax_gated":
                relation_input = torch.where(
                    allowed[:, start:stop, :, None],
                    relation_input,
                    torch.zeros_like(relation_input),
                )
            relation_chunk = self.relation_mlp(relation_input).reshape(
                batch, stop - start, tokens, self.num_heads, self.head_dim
            ).permute(0, 3, 1, 2, 4)
            scores = torch.einsum(
                "bhqd,bhqkd->bhqk", q_chunk,
                k_scalar.unsqueeze(2) + relation_chunk,
            ) * scalar_scale
            scores = scores + torch.einsum(
                "bhqdc,bhkdc->bhqk", q_vector[:, :, start:stop], k_vector
            ) * vector_scale
            scores = scores + additive[:, None, start:stop, :]
            if return_logits:
                logit_outputs.append(scores)
            if return_diagnostics:
                finite_logit_count += int(torch.isfinite(scores).sum().item())
                total_logit_count += int(scores.numel())
                allowed_chunk = allowed[:, None, start:stop, :].expand_as(scores)
                finite_allowed_logit_count += int(
                    (torch.isfinite(scores) & allowed_chunk).sum().item()
                )
                total_allowed_logit_count += int(allowed_chunk.sum().item())
            if self.aggregation_mode == "dual_softmax_gated":
                alpha_state, alpha_obs, has_state, has_obs = self._dual_stream_softmax(
                    scores,
                    allowed[:, start:stop, :],
                    torch.arange(tokens, device=parameter.device),
                    check_finite=True,
                )
                alpha_state = self.dropout(alpha_state)
                alpha_obs = self.dropout(alpha_obs)
                context_state_scalar = torch.einsum(
                    "bhqk,bhkd->bhqd", alpha_state, scalar_values
                )
                context_state_vector = torch.einsum(
                    "bhqk,bhkdc->bhqdc", alpha_state, vector_values
                )
                context_obs_scalar = torch.einsum(
                    "bhqk,bhkd->bhqd", alpha_obs, scalar_values
                )
                context_obs_vector = torch.einsum(
                    "bhqk,bhkdc->bhqdc", alpha_obs, vector_values
                )
                scalar_chunk, vector_chunk, gamma_state, gamma_obs = self._dual_context_updates(
                    q_chunk,
                    q_vector[:, :, start:stop],
                    context_state_scalar,
                    context_state_vector,
                    context_obs_scalar,
                    context_obs_vector,
                    has_state,
                    has_obs,
                )
                weights = alpha_state + alpha_obs
            else:
                weights = safe_masked_softmax(
                    scores, allowed[:, None, start:stop, :], dim=-1
                )
                weights = self.dropout(weights)
                scalar_context = torch.einsum("bhqk,bhkd->bhqd", weights, scalar_values)
                vector_context = torch.einsum("bhqk,bhkdc->bhqdc", weights, vector_values)
                scalar_context = scalar_context.permute(0, 2, 1, 3).reshape(batch, stop - start, -1)
                scalar_chunk = self.output_scalar(scalar_context)
                vector_chunk = torch.einsum("bqhdc,ohd->bqoc", vector_context.permute(0, 2, 1, 3, 4), self.output_vector_weight)
            row_has_edge = allowed[:, start:stop].any(dim=-1)
            scalar_chunk = torch.where(row_has_edge.unsqueeze(-1), scalar_chunk, torch.zeros_like(scalar_chunk))
            vector_chunk = torch.where(row_has_edge.unsqueeze(-1).unsqueeze(-1), vector_chunk, torch.zeros_like(vector_chunk))
            scalar_outputs.append(scalar_chunk)
            vector_outputs.append(vector_chunk)
            if return_attention:
                attention_outputs.append(weights)
        scalar_output = torch.cat(scalar_outputs, dim=1)
        vector_output = torch.cat(vector_outputs, dim=1)
        residual_scalar = self.scalar_residual(query_s)
        if self.vector_residual_weight is None:
            residual_vector = query_v
        else:
            residual_vector = torch.einsum(
                "bnci,oc->bnoi", query_v, self.vector_residual_weight
            )
        row_valid = allowed.any(dim=-1)
        scalar_output = torch.where(
            row_valid.unsqueeze(-1),
            residual_scalar + scalar_output,
            torch.zeros_like(scalar_output),
        )
        vector_output = torch.where(
            row_valid.unsqueeze(-1).unsqueeze(-1),
            residual_vector + vector_output,
            torch.zeros_like(vector_output),
        )
        if return_diagnostics:
            diagnostics = {
                "logits_finite_fraction": finite_logit_count / max(total_logit_count, 1),
                "allowed_logits_finite_fraction": (
                    finite_allowed_logit_count / max(total_allowed_logit_count, 1)
                ),
                "logit_element_count": total_logit_count,
                "allowed_logit_element_count": total_allowed_logit_count,
            }
        else:
            diagnostics = None
        extras: list[Any] = []
        if return_attention:
            extras.append(torch.cat(attention_outputs, dim=2))
        if return_logits:
            extras.append(torch.cat(logit_outputs, dim=2))
        if return_diagnostics:
            extras.append(diagnostics)
        if extras:
            return (scalar_output, vector_output, *extras)
        return ScalarVectorAttentionOutput(scalar_output, vector_output)


ScalarVectorAttention = RelationConditionedScalarVectorAttention
ScalarVectorCausalAttention = RelationConditionedScalarVectorAttention
CausalScalarVectorAttention = RelationConditionedScalarVectorAttention
RelationConditionedAttention = RelationConditionedScalarVectorAttention

__all__ = [
    "ScalarVectorAttentionOutput", "ScalarVectorKVCache",
    "OnlineScalarVectorAttentionOutput", "safe_masked_softmax",
    "RelationConditionedScalarVectorAttention", "ScalarVectorAttention",
    "ScalarVectorCausalAttention", "CausalScalarVectorAttention",
    "RelationConditionedAttention",
]

"""Geometry-conditioned V6 recursive filter and explicit fixed-history encoder.

The default forward methods run causal predict, gain and update per frame.
The representation-only fixed-history path has its own named method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .normalization import V6Normalization
from .tokenizer import TokenizerOutput, V6Tokenizer
from .scalar_vector_attention import RelationConditionedScalarVectorAttention
from .recursive_filter import (
    RecursiveFilterMixin, RecursiveFilterOutput, TrainingFilterOutput,
    V6GainHead, V6CorrectionHead, FilterDivergenceError,
)


KALMAN_STATE_SCALE_VALUES = (1.0e6, 1.0e6, 1.0e6, 1.0e3, 1.0e3, 1.0e3)
V6_MODEL_VERSION = "v6_geometry_sv_recursive_batch_v3"
V7A_MODEL_VERSION = "v7a_geometry_sv_dual_softmax_v1"


@dataclass(frozen=True)
class GeometryFilterOutput:
    """Outputs of the structured history path.

    Sequence fields retain the tokenizer rank: unbatched input is returned as
    ``[2T,...]`` and batched input as ``[B,2T,...]``.  ``gain`` and state fields
    refer to the last valid frame in each sequence.
    """

    tokenizer_output: TokenizerOutput
    scalar_tokens: Tensor
    vector_tokens: Tensor
    scalar_attention: Tensor
    vector_attention: Tensor
    relation: Tensor
    pair_valid: Tensor
    attention_mask: Tensor
    current_scalar: Tensor
    current_vector: Tensor
    raw_innovation: Tensor
    gain: Tensor | None
    posterior_state_si: Tensor | None
    metadata: Mapping[str, Any]
    attention_weights: Tensor | None = None
    attention_logits: Tensor | None = None
    attention_diagnostics: Mapping[str, Any] | None = None
    token_frame_index: Tensor | None = None


def build_token_causal_mask(
    valid_mask: Tensor, *, no_historical_state_keys: bool = False
) -> Tensor:
    """Construct a causal mask for interleaved ``[s_i,o_i]`` tokens.

    When ``no_historical_state_keys`` is enabled, state keys remain visible to
    both same-frame queries, while state keys from earlier frames are hidden.
    Observation keys keep the original causal and validity rules.
    """

    valid = torch.as_tensor(valid_mask, dtype=torch.bool)
    squeeze = valid.ndim == 1
    if squeeze:
        valid = valid.unsqueeze(0)
    if valid.ndim != 2:
        raise ValueError("valid_mask must have shape [T] or [B,T]")
    batch, frames = map(int, valid.shape)
    frame = torch.arange(frames, device=valid.device).repeat_interleave(2)
    allowed = frame[None, :, None] >= frame[None, None, :]
    token_valid = valid.repeat_interleave(2, dim=-1)
    allowed = allowed & token_valid[:, :, None] & token_valid[:, None, :]
    if no_historical_state_keys:
        token_index = torch.arange(2 * frames, device=valid.device)
        token_frame = token_index // 2
        state_key = token_index.remainder(2) == 0
        historical_state_edge = (
            (token_frame[:, None] > token_frame[None, :])
            & state_key[None, :]
        )
        allowed = allowed & ~historical_state_edge.unsqueeze(0)
    return allowed[0] if squeeze else allowed


def build_online_no_historical_state_mask(
    current_frame_index: int, *, batch_size: int, device: torch.device
) -> Tensor:
    """Return ``[B,2,2T]`` key permissions for one causal frame.

    The prefix contains frames ``0..current_frame_index``.  Both query rows
    retain all observation keys and the current state key, while historical
    state keys are disallowed.
    """
    frame = int(current_frame_index)
    batch = int(batch_size)
    if frame < 0:
        raise ValueError("current_frame_index must be nonnegative")
    if batch < 1:
        raise ValueError("batch_size must be positive")
    token_index = torch.arange(2 * (frame + 1), device=device)
    token_frame = token_index // 2
    state_key = token_index.remainder(2) == 0
    key_allowed = (~state_key) | (token_frame == frame)
    return key_allowed.reshape(1, 1, -1).expand(batch, 2, -1)


def _as_batch(value: Tensor, *, name: str) -> tuple[Tensor, bool]:
    if value.ndim == 2:
        return value.unsqueeze(0), True
    if value.ndim >= 3:
        return value, False
    raise ValueError(f"{name} must include a time dimension")


def _last_frame_indices(valid: Tensor) -> Tensor:
    if valid.ndim != 2:
        raise ValueError("valid mask must be [B,T]")
    frame = torch.arange(valid.shape[1], device=valid.device, dtype=torch.long)
    return torch.where(valid, frame[None, :], torch.zeros_like(frame)[None, :]).amax(dim=-1)


def _select_frame(value: Tensor, valid: Tensor) -> Tensor:
    """Gather the last valid frame, returning zeros for all-invalid rows."""

    index = _last_frame_indices(valid)
    gather_shape = (value.shape[0], 1, *value.shape[2:])
    gather_index = index[:, None].reshape(value.shape[0], 1, *([1] * (value.ndim - 2))).expand(gather_shape)
    selected = value.gather(1, gather_index).squeeze(1)
    return torch.where(valid.any(dim=-1).reshape(value.shape[0], *([1] * (value.ndim - 2))), selected, torch.zeros_like(selected))


def _expand_frame_relation(relation: Tensor) -> Tensor:
    if relation.ndim == 3:
        relation = relation.unsqueeze(0)
    if relation.ndim != 4:
        raise ValueError("frame relation must have shape [T,T,G] or [B,T,T,G]")
    return relation.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)


def _expand_frame_pair_mask(pair_valid: Tensor) -> Tensor:
    if pair_valid.ndim == 2:
        pair_valid = pair_valid.unsqueeze(0)
    if pair_valid.ndim != 3:
        raise ValueError("pair_valid must have shape [T,T] or [B,T,T]")
    return pair_valid.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)


def apply_scaled_state_update(prior_state_si: Tensor, gain: Tensor, innovation: Tensor) -> Tensor:
    """Apply the existing KalmanNet scaled-state update exactly once."""

    scale = torch.as_tensor(KALMAN_STATE_SCALE_VALUES, device=prior_state_si.device, dtype=prior_state_si.dtype)
    if prior_state_si.ndim != 2 or prior_state_si.shape[-1] != 6:
        raise ValueError("prior_state_si must have shape [B,6]")
    if gain.shape != (prior_state_si.shape[0], 6, 3) or innovation.shape != (prior_state_si.shape[0], 3):
        raise ValueError("gain must be [B,6,3] and innovation must be [B,3]")
    z_prior = prior_state_si / scale
    z_post = z_prior + torch.bmm(gain, innovation.unsqueeze(-1)).squeeze(-1)
    return z_post * scale


def apply_scaled_state_correction(prior_state_si: Tensor, correction_scaled: Tensor) -> Tensor:
    """Apply a direct scaled-state correction without multiplying by innovation."""

    scale = torch.as_tensor(
        KALMAN_STATE_SCALE_VALUES,
        device=prior_state_si.device,
        dtype=prior_state_si.dtype,
    )
    if prior_state_si.ndim != 2 or prior_state_si.shape[-1] != 6:
        raise ValueError("prior_state_si must have shape [B,6]")
    if correction_scaled.shape != (prior_state_si.shape[0], 6):
        raise ValueError("correction_scaled must be [B,6]")
    return (prior_state_si / scale + correction_scaled.to(
        device=prior_state_si.device, dtype=prior_state_si.dtype
    )) * scale


class GeometryConditionedFilter(RecursiveFilterMixin, nn.Module):
    """Recursive scalar/vector filter with an independent V6 gain head."""

    def __init__(
        self,
        tokenizer: V6Tokenizer | None = None,
        *,
        normalization: V6Normalization | None = None,
        stats_scales: Any = None,
        scalar_channels: int = 32,
        vector_channels: int = 16,
        num_heads: int = 4,
        attention: nn.Module | None = None,
        gain_head: nn.Module | None = None,
        gain_model: nn.Module | None = None,
        enable_gain: bool = True,
        update_mode: str = "gain",
        correction_head: nn.Module | None = None,
        use_direct_attention_values: bool = True,
        use_direct_gain_bypass: bool = True,
        attention_mode: str = "geometry_sv",
        attention_fusion: str = "unified",
        query_chunk_size: int | None = 256,
        attention_impl: str = "legacy_concat",
        block_frames: int = 32,
        checkpoint_rk4: bool = False,
        no_historical_state_keys: bool = False,
        projection_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if str(attention_mode) != "geometry_sv":
            raise ValueError(
                "legacy attention mode is unavailable: this V6 integration "
                "has no connected legacy Attention/gain main model; use "
                "attention_mode='geometry_sv'"
            )
        self.attention_mode = "geometry_sv"
        if attention_fusion not in ("unified", "dual_softmax_gated"):
            raise ValueError(
                "attention_fusion must be 'unified' or 'dual_softmax_gated'"
            )
        self.attention_fusion = str(attention_fusion)
        self.tokenizer = tokenizer or V6Tokenizer(
            normalization=normalization,
            stats_scales=stats_scales,
            scalar_channels=scalar_channels,
            vector_channels=vector_channels,
            projection_dtype=projection_dtype,
            enable_geometry=True,
        )
        if not self.tokenizer.enable_geometry:
            raise ValueError("geometry_sv requires enable_geometry=True on the tokenizer")
        if (self.tokenizer.scalar_channels != int(scalar_channels)
                or self.tokenizer.vector_channels != int(vector_channels)):
            raise ValueError("tokenizer scalar/vector channels must match the filter configuration")
        if attention is None:
            self.attention = RelationConditionedScalarVectorAttention(
                scalar_dim=scalar_channels,
                vector_channels=vector_channels,
                relation_dim=8,
                num_heads=num_heads,
                query_chunk_size=query_chunk_size,
                los_direct_scalar_dim=scalar_channels,
                los_direct_vector_channels=vector_channels,
                aggregation_mode=self.attention_fusion,
                dtype=projection_dtype,
            )
        else:
            custom_aggregation = getattr(attention, "aggregation_mode", None)
            if self.attention_fusion == "dual_softmax_gated" and custom_aggregation != "dual_softmax_gated":
                raise ValueError(
                    "attention_fusion='dual_softmax_gated' requires a custom attention "
                    "module with aggregation_mode='dual_softmax_gated'"
                )
            if custom_aggregation is not None and custom_aggregation != self.attention_fusion:
                raise ValueError(
                    f"custom attention aggregation_mode={custom_aggregation!r} does not "
                    f"match attention_fusion={self.attention_fusion!r}"
                )
            self.attention = attention
        # The integration layer supplies the complete 2T x 2T mask, including
        # legal same-frame State/Observation edges.  Do not let an attention
        # implementation reopen same-frame edges after this mask has removed
        # padding or invalid frames.
        if hasattr(self.attention, "allow_same_frame"):
            self.attention.allow_same_frame = False
        self.scalar_channels = int(scalar_channels)
        self.vector_channels = int(vector_channels)
        self.projection_dtype = projection_dtype
        if attention_impl not in ("legacy_concat", "blockwise", "blockwise_checkpoint", "fused_tiles_save"):
            raise ValueError("invalid attention_impl")
        if int(block_frames) <= 0:
            raise ValueError("block_frames must be positive")
        self.attention_impl = str(attention_impl)
        self.block_frames = int(block_frames)
        self.checkpoint_rk4 = bool(checkpoint_rk4)
        self.no_historical_state_keys = bool(no_historical_state_keys)
        if update_mode not in ("gain", "direct_correction"):
            raise ValueError("update_mode must be 'gain' or 'direct_correction'")
        self.update_mode = str(update_mode)
        if gain_model is not None:
            raise ValueError(
                "gain_model/KalmanNet FC2 is no longer the V6 gain interface; "
                "pass a V6-compatible gain_head explicitly"
            )
        self.use_direct_attention_values = bool(use_direct_attention_values)
        self.use_direct_gain_bypass = bool(use_direct_gain_bypass)
        self.gain_readout_dim = 3 * self.scalar_channels + 9 * self.vector_channels
        if self.update_mode == "gain":
            self.enable_gain = bool(enable_gain)
            self.gain_head = gain_head if gain_head is not None else V6GainHead(
                input_dim=self.gain_readout_dim,
                hidden_dim=128,
                dtype=projection_dtype,
            )
            if correction_head is not None:
                raise ValueError("correction_head is only valid for update_mode='direct_correction'")
            self.correction_head = None
        else:
            if gain_head is not None:
                raise ValueError(
                    "direct_correction requires gain_head=None"
                )
            self.enable_gain = False
            self.gain_head = None
            self.correction_head = correction_head if correction_head is not None else V6CorrectionHead(
                input_dim=self.gain_readout_dim,
                hidden_dim=128,
                dtype=projection_dtype,
            )
        self._refresh_stats_config()
        self.register_load_state_dict_post_hook(self._after_load_stats_config)

    def _refresh_stats_config(self) -> None:
        # This one-time CPU snapshot is refreshed after a state-dict load;
        # model_config can then be emitted without a per-forward GPU sync.
        self._stats_scales_config = self.normalization.stats_scale.detach().cpu().tolist()

    def _after_load_stats_config(self, module: nn.Module, incompatible_keys: Any) -> None:
        self._refresh_stats_config()

    @property
    def model_version(self) -> str:
        return V7A_MODEL_VERSION if self.attention_fusion == "dual_softmax_gated" else V6_MODEL_VERSION

    @property
    def attention_weight_semantics(self) -> str:
        if self.attention_fusion == "dual_softmax_gated":
            return "independent_state_observation_softmax"
        return getattr(self.attention, "attention_weight_semantics", "global_key_softmax")

    @property
    def normalization(self) -> V6Normalization:
        return self.tokenizer.feature_builder.normalization

    @property
    def model_config(self) -> dict[str, Any]:
        """Serializable architecture and initialization metadata for checkpoints."""
        normalization = self.normalization
        config = {
            "model_version": self.model_version,
            "attention_mode": self.attention_mode,
            "tokenizer": {
                "type": type(self.tokenizer).__name__,
                "branch_dim": self.tokenizer.branch_dim,
                "d_model": self.tokenizer.d_model,
                "scalar_channels": self.scalar_channels,
                "vector_channels": self.vector_channels,
                "geometry_enabled": self.tokenizer.enable_geometry,
            },
            "pair_relation_dim": int(getattr(self.attention, "relation_dim", 8)),
            "num_heads": int(getattr(self.attention, "num_heads", 0)),
            "head_dim": int(getattr(self.attention, "head_dim", 0)),
            "query_chunk_size": getattr(self.attention, "query_chunk_size", None),
            "attention_impl": self.attention_impl,
            "block_frames": self.block_frames,
            "checkpoint_rk4": self.checkpoint_rk4,
            "projection_dtype": str(self.projection_dtype),
            "stats_scale_names": list(normalization.stats_scale_names),
            "stats_scales": list(self._stats_scales_config),
            "update_mode": self.update_mode,
            "gain_enabled": self.enable_gain,
            "gain_head": None if self.gain_head is None else type(self.gain_head).__name__,
            "gain_head_input_dim": self.gain_readout_dim if self.gain_head is not None else None,
            "gain_head_output_dim": 18 if self.gain_head is not None else None,
            "gain_head_parameter_count": (
                None if self.gain_head is None else
                sum(p.numel() for p in self.gain_head.parameters())
            ),
            "correction_head": (
                None if self.correction_head is None else type(self.correction_head).__name__
            ),
            "correction_head_input_dim": (
                None if self.correction_head is None else self.gain_readout_dim
            ),
            "correction_head_hidden_dim": (
                None if self.correction_head is None else
                int(getattr(self.correction_head, "hidden_dim", 128))
            ),
            "correction_head_output_dim": 6 if self.correction_head is not None else None,
            "correction_head_parameter_count": (
                None if self.correction_head is None else
                sum(p.numel() for p in self.correction_head.parameters())
            ),
            "use_direct_attention_values": self.use_direct_attention_values,
            "use_direct_gain_bypass": self.use_direct_gain_bypass,
            "online_kv_cache": True,
            "no_historical_state_keys": self.no_historical_state_keys,
        }
        if self.attention_fusion == "dual_softmax_gated":
            config.update({
                "attention_fusion": "dual_softmax_gated",
                "attention_streams": ["state", "observation"],
                "attention_stream_normalization": "independent_softmax",
                "attention_context_fusion": "independent_sigmoid_gates",
                "attention_weight_semantics": "independent_state_observation_softmax",
                "historical_kv_refresh": False,
            })
        return config

    def _build_relation(self, features: Any) -> tuple[Tensor, Tensor]:
        try:
            from .geometry_relation import build_pairwise_geometry
        except ImportError as exc:
            raise ImportError("geometry_filter requires models.geometry_relation.build_pairwise_geometry") from exc
        result = build_pairwise_geometry(
            features, self.normalization, strict_time_order=True
        )
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise TypeError("build_pairwise_geometry must return (relation, pair_valid)")
        return result[0], result[1]

    def forward(self, *args: Any, **kwargs: Any) -> RecursiveFilterOutput | TrainingFilterOutput:
        """Run the actual causal recursive filter."""
        return self.forward_filter_sequence(*args, **kwargs)

    def forward_sequence(self, *args: Any, **kwargs: Any) -> RecursiveFilterOutput | TrainingFilterOutput:
        """Alias for the full recursive filter; initial_state_si is required."""
        return self.forward_filter_sequence(*args, **kwargs)

    def encode_fixed_history(
        self,
        *,
        features: Any = None,
        apply_update: bool | None = None,
        return_attention: bool = False,
        return_logits: bool = False,
        return_diagnostics: bool = False,
        **kwargs: Any,
    ) -> GeometryFilterOutput:
        """Encode fixed supplied states; no dynamics or state update is performed."""
        output = self.tokenizer.forward_sequence(features=features, **kwargs) if features is not None else self.tokenizer.forward_sequence(**kwargs)
        if output.scalar_tokens is None or output.vector_tokens is None:
            raise RuntimeError("geometry_sv mode requires scalar_tokens and vector_tokens")
        scalar, unbatched = _as_batch(output.scalar_tokens, name="scalar_tokens")
        # The vector lane has rank three when unbatched ([2T,Cv,3]), so use
        # the scalar lane's mode decision rather than inferring batching from
        # rank alone.
        vector = output.vector_tokens.unsqueeze(0) if unbatched else output.vector_tokens
        valid = output.valid_mask
        if valid.ndim == 1:
            valid = valid.unsqueeze(0)
        if valid.ndim != 2:
            raise ValueError("tokenizer valid_mask must be [T] or [B,T]")
        relation_frame, pair_frame = self._build_relation(output.features)
        # The physical relation is formed in float64; cast once to the model
        # dtype before expanding each frame pair to its four token pairs.
        relation = torch.as_tensor(
            relation_frame, device=scalar.device, dtype=self.projection_dtype
        )
        pair_valid = _expand_frame_pair_mask(torch.as_tensor(pair_frame, device=scalar.device, dtype=torch.bool))
        causal = build_token_causal_mask(
            valid, no_historical_state_keys=self.no_historical_state_keys
        )
        if causal.ndim == 2:
            causal = causal.unsqueeze(0)
        attention_mask = causal & pair_valid
        frame_index = torch.arange(valid.shape[1], device=scalar.device).repeat_interleave(2).unsqueeze(0).expand(valid.shape[0], -1)
        direct_scalar = output.los_direct_scalar_token
        direct_vector = output.los_direct_vector_token
        if direct_scalar is not None and direct_scalar.ndim == 2:
            direct_scalar = direct_scalar.unsqueeze(0)
        if direct_vector is not None and direct_vector.ndim == 3:
            direct_vector = direct_vector.unsqueeze(0)
        attn_result = self.attention(
            scalar,
            vector,
            relation=relation,
            attention_mask=attention_mask,
            los_direct=(direct_scalar, direct_vector),
            token_frame_index=frame_index,
            observation_token_mask=(torch.arange(scalar.shape[1], device=scalar.device) % 2 == 1).unsqueeze(0).expand(scalar.shape[0], -1),
            return_attention=return_attention,
            return_logits=return_logits,
            return_diagnostics=return_diagnostics,
        )
        attention_weights = None
        attention_logits = None
        attention_diagnostics = None
        if hasattr(attn_result, "scalar"):
            scalar_attention, vector_attention = attn_result.scalar, attn_result.vector
        else:
            scalar_attention, vector_attention = attn_result[:2]
            extra_index = 2
            if return_attention:
                attention_weights = attn_result[extra_index]
                extra_index += 1
            if return_logits:
                attention_logits = attn_result[extra_index]
                extra_index += 1
            if return_diagnostics:
                attention_diagnostics = attn_result[extra_index]
        # Some attention implementations reopen same-frame edges internally
        # for compatibility with a frame mask.  Enforce the integration
        # contract at the final fusion boundary as well: a token row with no
        # valid query/key edge must be exactly zero, including projection
        # biases.  This is essential for both padding and all-invalid rows.
        output_row_valid = attention_mask.any(dim=-1)
        scalar_attention = torch.where(
            output_row_valid.unsqueeze(-1),
            scalar_attention,
            torch.zeros_like(scalar_attention),
        )
        vector_attention = torch.where(
            output_row_valid.unsqueeze(-1).unsqueeze(-1),
            vector_attention,
            torch.zeros_like(vector_attention),
        )
        # Keep the two same-frame token streams distinct until the final
        # readout boundary.  Concatenation is deliberate: summing first would
        # discard which part came from State versus Observation and would not
        # match the 160-dimensional adapter input.
        state_scalar = _select_frame(scalar_attention[:, 0::2], valid)
        observation_scalar = _select_frame(scalar_attention[:, 1::2], valid)
        state_vector = _select_frame(vector_attention[:, 0::2], valid)
        observation_vector = _select_frame(vector_attention[:, 1::2], valid)
        current_scalar = torch.cat((state_scalar, observation_scalar), dim=-1)
        current_vector = torch.cat((state_vector, observation_vector), dim=-2)
        raw_innovation = _select_frame(output.raw_innovation.unsqueeze(0) if output.raw_innovation.ndim == 2 else output.raw_innovation, valid)
        if apply_update:
            raise ValueError(
                "encode_fixed_history does not perform a state update; "
                "use forward_filter_sequence for recursive filtering"
            )
        gain = None
        posterior = None
        metadata = {
            "attention_mode": self.attention_mode,
            "attention_fusion": self.attention_fusion,
            "attention_weight_semantics": self.attention_weight_semantics,
            "pair_relation_dim": relation.shape[-1],
            "relation_storage": "frame-level; mapped to token pairs within query chunks",
            "token_relation_shape": (int(scalar.shape[1]), int(scalar.shape[1]), int(relation.shape[-1])),
            "scalar_channels": self.scalar_channels,
            "vector_channels": self.vector_channels,
            "num_heads": int(getattr(self.attention, "num_heads", 0)),
            "model_version": self.model_version,
            "model_config": self.model_config,
            "kalman_state_scale_values": KALMAN_STATE_SCALE_VALUES,
            "raw_innovation_unchanged": True,
            "update_mode": self.update_mode,
            "gain_head": None if self.gain_head is None else type(self.gain_head).__name__,
            "gain_head_input_dim": self.gain_readout_dim if self.gain_head is not None else None,
            "gain_head_output_dim": 18 if self.gain_head is not None else None,
            "gain_head_parameter_count": (
                None if self.gain_head is None else
                sum(p.numel() for p in self.gain_head.parameters())
            ),
            "correction_head": (
                None if self.correction_head is None else type(self.correction_head).__name__
            ),
            "correction_head_input_dim": (
                None if self.correction_head is None else self.gain_readout_dim
            ),
            "correction_head_hidden_dim": (
                None if self.correction_head is None else
                int(getattr(self.correction_head, "hidden_dim", 128))
            ),
            "correction_head_output_dim": 6 if self.correction_head is not None else None,
            "correction_head_parameter_count": (
                None if self.correction_head is None else
                sum(p.numel() for p in self.correction_head.parameters())
            ),
            "use_direct_attention_values": self.use_direct_attention_values,
            "use_direct_gain_bypass": self.use_direct_gain_bypass,
            "online_kv_cache": False,
            "recursive_filter": False,
            "state_update_performed": False,
            "readout_vector_flatten_boundary": "after_attention_only",
        }
        if unbatched:
            scalar, vector, pair_valid, attention_mask = (x.squeeze(0) for x in (scalar, vector, pair_valid, attention_mask))
            if relation.ndim == 4 and relation.shape[0] == 1:
                relation = relation.squeeze(0)
            scalar_attention, vector_attention = scalar_attention.squeeze(0), vector_attention.squeeze(0)
            current_scalar, current_vector, raw_innovation = current_scalar.squeeze(0), current_vector.squeeze(0), raw_innovation.squeeze(0)
            if attention_weights is not None:
                attention_weights = attention_weights.squeeze(0)
            if attention_logits is not None:
                attention_logits = attention_logits.squeeze(0)
            if gain is not None:
                gain, posterior = gain.squeeze(0), posterior.squeeze(0)
        return GeometryFilterOutput(
            output,
            scalar,
            vector,
            scalar_attention,
            vector_attention,
            relation,
            pair_valid,
            attention_mask,
            current_scalar,
            current_vector,
            raw_innovation,
            gain,
            posterior,
            metadata,
            attention_weights,
            attention_logits,
            attention_diagnostics,
            frame_index.squeeze(0) if unbatched else frame_index,
        )


__all__ = ["GeometryFilterOutput", "RecursiveFilterOutput", "TrainingFilterOutput", "GeometryConditionedFilter",
           "V6GainHead", "V6CorrectionHead", "FilterDivergenceError", "build_token_causal_mask",
           "build_online_no_historical_state_mask",
           "apply_scaled_state_update", "apply_scaled_state_correction", "KALMAN_STATE_SCALE_VALUES"]

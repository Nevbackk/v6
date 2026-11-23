"""V6 dual-branch observation tokenizer.

The tokenizer is a representation boundary only.  It creates one state token
and one observation/LOS token per frame, returns causal masks and physical
metadata, and leaves attention and training to downstream modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from .feature_builder import FEATURE_DIM, FeatureBatch, FeatureBuilder
from .normalization import V6Normalization
from .geometry import FrameGeometry, build_frame_geometry
from .scalar_vector import ScalarVectorEncoder


BRANCH_NAMES: tuple[str, str] = ("state", "observation")


class ResidualFeatureMLP(nn.Module):
    """Project one semantic feature lane with a residual two-layer MLP.

    The skip projection deliberately has no bias.  Keeping the residual path
    linear and bias-free makes the zero-feature point explicit while the
    learned branch remains ``fc1 -> GELU -> fc2``.
    """

    def __init__(
        self,
        input_dim: int,
        branch_dim: int = 32,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        input_dim = int(input_dim)
        branch_dim = int(branch_dim)
        if input_dim <= 0 or branch_dim <= 0:
            raise ValueError("ResidualFeatureMLP dimensions must be positive")
        self.fc1 = nn.Linear(input_dim, branch_dim, dtype=dtype)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(branch_dim, branch_dim, dtype=dtype)
        self.skip = nn.Linear(input_dim, branch_dim, bias=False, dtype=dtype)

    def forward(self, value: Tensor) -> Tensor:
        value = value.to(dtype=self.fc1.weight.dtype)
        return self.skip(value) + self.fc2(self.activation(self.fc1(value)))


def select_last_valid(value: Tensor, valid_mask: Tensor) -> Tensor:
    """Select the vector at the largest valid timestep without Python loops.

    ``value`` may be an unbatched ``[T,*]`` sequence, a batched ``[B,T,*]``
    sequence, or a single-step ``[*]`` vector paired with a scalar mask.  An
    all-invalid row returns zeros.  Index selection is differentiable with
    respect to ``value``; the index itself is intentionally derived only from
    the boolean mask.
    """

    valid = valid_mask.to(dtype=torch.bool, device=value.device)
    if valid.ndim == 0:
        return torch.where(valid, value, torch.zeros_like(value))
    if valid.ndim == 1:
        if value.ndim == 0 or int(value.shape[0]) != int(valid.shape[0]):
            raise ValueError(
                "unbatched value must have the valid-mask length on dimension zero"
            )
        steps = int(valid.shape[0])
        indices = torch.arange(steps, device=value.device, dtype=torch.long)
        index = torch.where(valid, indices, torch.zeros_like(indices)).amax()
        selected = value.index_select(0, index.reshape(1)).squeeze(0)
        return torch.where(valid.any(), selected, torch.zeros_like(selected))
    if valid.ndim == 2:
        if value.ndim < 2 or tuple(value.shape[:2]) != tuple(valid.shape):
            raise ValueError(
                "batched value must have [B,T] leading dimensions matching valid_mask"
            )
        batch, steps = int(valid.shape[0]), int(valid.shape[1])
        indices = torch.arange(steps, device=value.device, dtype=torch.long)
        indices = indices.reshape(1, steps).expand(batch, steps)
        last = torch.where(valid, indices, torch.zeros_like(indices)).amax(dim=-1)
        gather_shape = (batch, 1, *value.shape[2:])
        gather_index = last.reshape(batch, 1, *([1] * (value.ndim - 2))).expand(gather_shape)
        selected = value.gather(dim=1, index=gather_index).squeeze(1)
        any_valid = valid.any(dim=-1).reshape(batch, *([1] * (value.ndim - 2)))
        return torch.where(any_valid, selected, torch.zeros_like(selected))
    raise ValueError(f"valid_mask must have rank zero, one or two, got {tuple(valid.shape)}")


def _causal_mask(valid_mask: Tensor) -> Tensor:
    if valid_mask.ndim == 0:
        valid = valid_mask.reshape(1)
    else:
        valid = valid_mask
    if valid.ndim == 1:
        steps = int(valid.shape[0])
        base = torch.tril(torch.ones((steps, steps), device=valid.device, dtype=torch.bool))
        return base & valid[:, None] & valid[None, :]
    if valid.ndim == 2:
        batch, steps = int(valid.shape[0]), int(valid.shape[1])
        base = torch.tril(torch.ones((steps, steps), device=valid.device, dtype=torch.bool))
        return base.unsqueeze(0) & valid[:, :, None] & valid[:, None, :]
    raise ValueError(f"valid_mask must have rank one or two, got {tuple(valid.shape)}")


@dataclass(frozen=True)
class TokenizerOutput:
    """Dual-token output with raw diagnostic lanes retained."""

    tokens: Tensor
    state_token: Tensor
    observation_token: Tensor
    features: FeatureBatch
    valid_mask: Tensor
    branch_valid_mask: Tensor
    first_frame_missing_mask: Tensor
    raw_innovation: Tensor
    direct_los: Tensor
    delta_los: Tensor
    delta_t_sec: Tensor
    causal_mask: Tensor
    los_direct_sequence_embedding: Tensor
    los_direct_sequence_features: Tensor
    # New structured representation for a geometry-aware Attention layer.
    # Legacy flat tokens remain available for compatibility only.
    scalar_tokens: Tensor | None = None
    vector_tokens: Tensor | None = None
    state_scalar_token: Tensor | None = None
    state_vector_token: Tensor | None = None
    observation_scalar_token: Tensor | None = None
    observation_vector_token: Tensor | None = None
    los_direct_scalar_token: Tensor | None = None
    los_direct_vector_token: Tensor | None = None
    frame_geometry: FrameGeometry | None = None
    branch_names: tuple[str, str] = BRANCH_NAMES
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def observation_los_token(self) -> Tensor:
        return self.observation_token

    @property
    def state_tokens(self) -> Tensor:
        return self.state_token

    @property
    def observation_tokens(self) -> Tensor:
        return self.observation_token

    @property
    def token_valid_mask(self) -> Tensor:
        branch = self.branch_valid_mask
        if branch.ndim >= 2 and branch.shape[-1] == 2:
            return branch.reshape(*branch.shape[:-2], branch.shape[-2] * 2)
        return branch

    @property
    def token_type_ids(self) -> Tensor:
        shape = self.valid_mask.shape + (2,)
        base = torch.tensor((0, 1), device=self.valid_mask.device, dtype=torch.long)
        branch = base.expand(shape)
        if branch.ndim >= 2:
            return branch.reshape(*branch.shape[:-2], branch.shape[-2] * 2)
        return branch

    @property
    def token_timestep_mapping(self) -> Tensor:
        valid = self.valid_mask
        if valid.ndim == 0:
            return torch.zeros((2,), device=valid.device, dtype=torch.long)
        if valid.ndim == 1:
            base = torch.arange(valid.shape[0], device=valid.device, dtype=torch.long)
            return base.repeat_interleave(2)
        base = torch.arange(valid.shape[-1], device=valid.device, dtype=torch.long)
        return base.repeat_interleave(2).expand(valid.shape[0], -1)

    @property
    def token_time_index(self) -> Tensor:
        return self.token_timestep_mapping

    @property
    def time_index(self) -> Tensor:
        return self.token_timestep_mapping

    @property
    def current_innovation_raw(self) -> Tensor:
        return select_last_valid(self.raw_innovation, self.valid_mask)

    @property
    def innovation_raw(self) -> Tensor:
        return self.raw_innovation

    @property
    def current_los_embedding(self) -> Tensor:
        return select_last_valid(self.los_direct_sequence_embedding, self.valid_mask)

    @property
    def current_los_features(self) -> Tensor:
        """Actual LOS, predicted LOS, normalized dLOS and normalized innovation."""

        return select_last_valid(self.los_direct_sequence_features, self.valid_mask)

    @property
    def current_los_direct_sequence_embedding(self) -> Tensor:
        return self.current_los_embedding

    @property
    def current_los_direct_sequence_features(self) -> Tensor:
        return self.current_los_features

    @property
    def los_direct_embedding(self) -> Tensor:
        return self.los_direct_sequence_embedding

    @property
    def los_direct_features(self) -> Tensor:
        return self.los_direct_sequence_features

    @property
    def relative_time(self) -> Tensor:
        return self.features.relative_time_sec

    @property
    def relative_time_sec(self) -> Tensor:
        return self.features.relative_time_sec

    @property
    def innovation(self) -> Tensor:
        return self.raw_innovation

    @property
    def mask(self) -> Tensor:
        return self.valid_mask

    @property
    def lengths(self) -> Tensor:
        return self.features.lengths

    def __getitem__(self, key: str) -> Any:
        aliases = {
            "state": "state_token",
            "observation": "observation_token",
            "obs": "observation_token",
            "los": "direct_los",
            "innovation": "raw_innovation",
            "delta_los": "delta_los",
            "dt_sec": "delta_t_sec",
            "mask": "valid_mask",
            "first_missing_mask": "first_frame_missing_mask",
            "features": "features",
            "los_direct_embedding": "los_direct_sequence_embedding",
            "los_direct_features": "los_direct_sequence_features",
        }
        name = aliases.get(str(key), str(key))
        if hasattr(self, name):
            return getattr(self, name)
        if name in self.metadata:
            return self.metadata[name]
        raise KeyError(key)

    def keys(self) -> tuple[str, ...]:
        return (
            "tokens",
            "scalar_tokens",
            "vector_tokens",
            "state_scalar_token",
            "state_vector_token",
            "observation_scalar_token",
            "observation_vector_token",
            "los_direct_scalar_token",
            "los_direct_vector_token",
            "frame_geometry",
            "state_token",
            "state_tokens",
            "observation_token",
            "observation_tokens",
            "features",
            "valid_mask",
            "branch_valid_mask",
            "token_valid_mask",
            "token_type_ids",
            "token_timestep_mapping",
            "token_time_index",
            "time_index",
            "first_frame_missing_mask",
            "raw_innovation",
            "current_innovation_raw",
            "innovation_raw",
            "direct_los",
            "current_los_embedding",
            "current_los_features",
            "los_direct_sequence_embedding",
            "los_direct_sequence_features",
            "current_los_direct_sequence_embedding",
            "current_los_direct_sequence_features",
            "delta_los",
            "delta_t_sec",
            "relative_time",
            "relative_time_sec",
            "causal_mask",
            "branch_names",
        )

    def as_dict(self) -> dict[str, Any]:
        return {name: self[name] for name in self.keys()}


class V6Tokenizer(nn.Module):
    """Build independent state and direct-LOS tokens from raw SI inputs.

    Each semantic lane first maps to a fixed ``branch_dim`` and the independent
    lanes are then fused to ``d_model``.  The module contains no attention or
    recurrent state; ``token_dim`` is retained as an alias for ``d_model``.
    """

    branch_names = BRANCH_NAMES
    feature_dim = FEATURE_DIM

    def __init__(
        self,
        normalization: V6Normalization | None = None,
        *,
        feature_builder: FeatureBuilder | None = None,
        token_dim: int | None = None,
        branch_dim: int = 32,
        d_model: int = 64,
        stats_scales: Sequence[float] | Mapping[str, Any] | None = None,
        stats_json: str | Path | Mapping[str, Any] | None = None,
        projection_dtype: torch.dtype = torch.float32,
        scalar_channels: int = 32,
        vector_channels: int = 16,
        enable_geometry: bool = True,
    ) -> None:
        super().__init__()
        if feature_builder is not None and normalization is not None:
            raise ValueError("provide either feature_builder or normalization, not both")
        self.feature_builder = feature_builder or FeatureBuilder(
            normalization=normalization,
            stats_scales=stats_scales,
            stats_json=stats_json,
        )
        if token_dim is not None:
            if int(d_model) != 64 and int(d_model) != int(token_dim):
                raise ValueError("token_dim and d_model disagree")
            d_model = int(token_dim)
        branch_dim = int(branch_dim)
        d_model = int(d_model)
        if branch_dim <= 0 or d_model <= 0:
            raise ValueError("branch_dim and d_model must be positive")
        self.branch_dim = branch_dim
        self.d_model = d_model
        self.token_dim = d_model
        self.enable_geometry = bool(enable_geometry)
        self.scalar_channels = int(scalar_channels)
        self.vector_channels = int(vector_channels)
        if self.enable_geometry:
            # Invariant scalar channels + SO(3)-compatible vector channels.
            # The geometry path never mixes XYZ with an arbitrary flat MLP.
            self.geometry_state_encoder = ScalarVectorEncoder(
                16, 8, scalar_channels=self.scalar_channels,
                vector_channels=self.vector_channels, dtype=projection_dtype)
            self.geometry_observation_encoder = ScalarVectorEncoder(
                20, 10, scalar_channels=self.scalar_channels,
                vector_channels=self.vector_channels, dtype=projection_dtype)
            # Independent weights: this path consumes LOS quantities only.
            self.geometry_los_direct_encoder = ScalarVectorEncoder(
                7, 4, scalar_channels=self.scalar_channels,
                vector_channels=self.vector_channels, dtype=projection_dtype)
        # Each semantic lane has its own residual MLP.  Keeping these modules
        # separate is important: a direct LOS perturbation must not share
        # weights with dLOS, innovation, station metadata, or time; likewise
        # state absolute/evolution/update lanes remain independent.
        self.state_abs_embedding = ResidualFeatureMLP(
            6, branch_dim, dtype=projection_dtype
        )
        self.state_evol_embedding = ResidualFeatureMLP(
            6, branch_dim, dtype=projection_dtype
        )
        self.state_update_embedding = ResidualFeatureMLP(
            6, branch_dim, dtype=projection_dtype
        )
        self.los_embedding = ResidualFeatureMLP(6, branch_dim, dtype=projection_dtype)
        self.dlos_embedding = ResidualFeatureMLP(
            3, branch_dim, dtype=projection_dtype
        )
        self.innovation_embedding = ResidualFeatureMLP(
            3, branch_dim, dtype=projection_dtype
        )
        self.station_embedding = ResidualFeatureMLP(
            10, branch_dim, dtype=projection_dtype
        )
        self.time_embedding = ResidualFeatureMLP(2, branch_dim, dtype=projection_dtype)
        self.state_mask_linear = nn.Linear(4, branch_dim, dtype=projection_dtype)
        self.observation_mask_linear = nn.Linear(5, branch_dim, dtype=projection_dtype)
        self.state_fusion = nn.Sequential(
            nn.Linear(4 * branch_dim, 2 * d_model, dtype=projection_dtype),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model, dtype=projection_dtype),
            nn.LayerNorm(d_model, dtype=projection_dtype),
        )
        self.observation_fusion = nn.Sequential(
            nn.Linear(6 * branch_dim, 2 * d_model, dtype=projection_dtype),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model, dtype=projection_dtype),
            nn.LayerNorm(d_model, dtype=projection_dtype),
        )
        self.los_direct_fusion = nn.Sequential(
            nn.Linear(3 * branch_dim, 2 * d_model, dtype=projection_dtype),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model, dtype=projection_dtype),
            nn.LayerNorm(d_model, dtype=projection_dtype),
        )

    # Compatibility names are properties rather than assigned module aliases;
    # this keeps one registration and one state-dict entry per module.
    @property
    def state_abs_linear(self) -> nn.Module:
        return self.state_abs_embedding

    @property
    def state_evol_linear(self) -> nn.Module:
        return self.state_evol_embedding

    @property
    def state_evolution_embedding(self) -> nn.Module:
        return self.state_evol_embedding

    @property
    def state_update_linear(self) -> nn.Module:
        return self.state_update_embedding

    @property
    def los_linear(self) -> nn.Module:
        return self.los_embedding

    @property
    def dlos_linear(self) -> nn.Module:
        return self.dlos_embedding

    @property
    def innovation_linear(self) -> nn.Module:
        return self.innovation_embedding

    @property
    def station_linear(self) -> nn.Module:
        return self.station_embedding

    @property
    def time_linear(self) -> nn.Module:
        return self.time_embedding

    @property
    def los_diff_embedding(self) -> nn.Module:
        return self.dlos_embedding

    @property
    def state_abs_projection(self) -> nn.Module:
        return self.state_abs_embedding

    @property
    def state_evolution_projection(self) -> nn.Module:
        return self.state_evol_embedding

    @property
    def state_update_projection(self) -> nn.Module:
        return self.state_update_embedding

    @property
    def los_projection(self) -> nn.Module:
        return self.los_embedding

    @property
    def dlos_projection(self) -> nn.Module:
        return self.dlos_embedding

    @property
    def innovation_projection(self) -> nn.Module:
        return self.innovation_embedding

    @property
    def station_projection(self) -> nn.Module:
        return self.station_embedding

    @property
    def time_projection(self) -> nn.Module:
        return self.time_embedding

    @property
    def state_projection(self) -> nn.Module:
        return self.state_fusion

    @property
    def observation_projection(self) -> nn.Module:
        return self.observation_fusion

    @property
    def los_direct_projection(self) -> nn.Module:
        return self.los_direct_fusion

    @property
    def los_direct_sequence_fusion(self) -> nn.Module:
        return self.los_direct_fusion

    def _project(self, features: FeatureBatch) -> tuple[Tensor, Tensor, Tensor]:
        # FeatureBuilder's layout reserves fixed slices for each semantic
        # group.  Each lane has its own branch_dim projection, followed by an
        # explicit fusion to d_model; there is no cross-lane weight sharing.
        state_abs = features.state_features[..., :6]
        state_evol = features.state_features[..., 6:12]
        state_update = features.state_features[..., 12:18]
        observation_los = features.observation_features[..., :6]
        observation_dlos = features.observation_features[..., 6:9]
        observation_innovation = features.observation_features[..., 9:12]
        observation_station = features.observation_features[..., 12:22]
        observation_time = features.observation_features[..., 22:24]
        previous_valid = features.valid_mask & ~features.first_frame_missing_mask
        previous_posterior_used = features.metadata.get(
            "previous_posterior_used", torch.zeros_like(features.valid_mask)
        )
        update_valid = features.metadata.get("update_valid", previous_posterior_used)
        evolution_valid = previous_posterior_used
        state_mask = torch.stack(
            (
                features.valid_mask,
                evolution_valid,
                update_valid,
                features.station_change_reliable,
            ),
            dim=-1,
        ).to(dtype=features.state_features.dtype)
        observation_mask = torch.stack(
            (
                features.valid_mask,
                features.difference_valid,
                features.station_id_used,
                features.station_change_reliable,
                features.metadata.get(
                    "causal_prior_used", torch.zeros_like(features.valid_mask)
                ),
            ),
            dim=-1,
        ).to(dtype=features.observation_features.dtype)

        def apply(module: nn.Module, value: Tensor) -> Tensor:
            # Linear parameters have an explicit dtype.  ``to`` preserves the
            # graph from raw SI inputs while making mixed precision explicit.
            parameter = next(module.parameters())
            return module(value.to(dtype=parameter.dtype))

        state_abs_embedding = apply(self.state_abs_embedding, state_abs)
        state_evol_embedding = apply(self.state_evol_embedding, state_evol)
        state_update_embedding = apply(self.state_update_embedding, state_update)
        state_mask_embedding = apply(self.state_mask_linear, state_mask)
        state = self.state_fusion(
            torch.cat(
                (
                    state_abs_embedding,
                    state_evol_embedding,
                    state_update_embedding,
                    state_mask_embedding,
                ),
                dim=-1,
            )
        )

        # Compute each LOS lane once and reuse the same embeddings in the
        # observation branch and the independent direct-LOS branch.
        los_embedding = apply(self.los_embedding, observation_los)
        dlos_embedding = apply(self.dlos_embedding, observation_dlos)
        innovation_embedding = apply(self.innovation_embedding, observation_innovation)
        station_embedding = apply(self.station_embedding, observation_station)
        time_embedding = apply(self.time_embedding, observation_time)
        observation_mask_embedding = apply(self.observation_mask_linear, observation_mask)
        observation = self.observation_fusion(
            torch.cat(
                (
                    los_embedding,
                    dlos_embedding,
                    innovation_embedding,
                    station_embedding,
                    time_embedding,
                    observation_mask_embedding,
                ),
                dim=-1,
            )
        )
        los_direct = self.los_direct_fusion(
            torch.cat((los_embedding, dlos_embedding, innovation_embedding), dim=-1)
        )
        valid = features.valid_mask.to(dtype=torch.bool).unsqueeze(-1)
        state = torch.where(valid, state, torch.zeros_like(state))
        observation = torch.where(valid, observation, torch.zeros_like(observation))
        los_direct = torch.where(valid, los_direct, torch.zeros_like(los_direct))
        return state, observation, los_direct

    def encode_geometry_features(self, features: FeatureBatch):
        """Encode structured lanes using the existing geometry encoders only.

        The recursive geometry filter does not consume legacy flat tokens, so
        it can avoid evaluating their separate MLPs at every time step.
        """
        if not self.enable_geometry:
            raise RuntimeError("structured geometry encoding is disabled")
        geometry = build_frame_geometry(features, self.feature_builder.normalization)
        state_s, state_v = self.geometry_state_encoder(
            geometry.state_scalars, geometry.state_vectors, features.valid_mask
        )
        obs_s, obs_v = self.geometry_observation_encoder(
            geometry.observation_scalars, geometry.observation_vectors,
            geometry.geometric_valid_mask,
        )
        direct_s, direct_v = self.geometry_los_direct_encoder(
            geometry.los_direct_scalars, geometry.los_direct_vectors, features.valid_mask
        )
        scalar = torch.stack((state_s, obs_s), dim=-2)
        vector = torch.stack((state_v, obs_v), dim=-3)
        if features.valid_mask.ndim >= 1:
            scalar = scalar.flatten(start_dim=-3, end_dim=-2)
            vector = vector.flatten(start_dim=-4, end_dim=-3)
        return scalar, vector, direct_s, direct_v, geometry

    def _wrap(self, features: FeatureBatch) -> TokenizerOutput:
        state_token, observation_token, los_direct_sequence_embedding = self._project(features)
        los_direct_sequence_features = features.observation_features[..., :12]
        valid = features.valid_mask.to(dtype=torch.bool).unsqueeze(-1)
        los_direct_sequence_features = torch.where(
            valid,
            los_direct_sequence_features,
            torch.zeros_like(los_direct_sequence_features),
        )
        interleaved = torch.stack((state_token, observation_token), dim=-2)
        if interleaved.ndim >= 3:
            tokens = interleaved.reshape(
                *interleaved.shape[:-3],
                interleaved.shape[-3] * interleaved.shape[-2],
                interleaved.shape[-1],
            )
        else:
            tokens = interleaved
        valid_mask = features.valid_mask
        branch_valid_mask = valid_mask.unsqueeze(-1).expand(*valid_mask.shape, 2)
        structured = {}
        if self.enable_geometry:
            frame_geometry = build_frame_geometry(features, self.feature_builder.normalization)
            s_state, v_state = self.geometry_state_encoder(
                frame_geometry.state_scalars, frame_geometry.state_vectors, features.valid_mask)
            s_obs, v_obs = self.geometry_observation_encoder(
                frame_geometry.observation_scalars, frame_geometry.observation_vectors,
                frame_geometry.geometric_valid_mask)
            s_direct, v_direct = self.geometry_los_direct_encoder(
                frame_geometry.los_direct_scalars, frame_geometry.los_direct_vectors,
                features.valid_mask)
            structured = dict(
                frame_geometry=frame_geometry,
                state_scalar_token=s_state, state_vector_token=v_state,
                observation_scalar_token=s_obs, observation_vector_token=v_obs,
                los_direct_scalar_token=s_direct, los_direct_vector_token=v_direct,
            )
            scalar_pair = torch.stack((s_state, s_obs), dim=-2)
            vector_pair = torch.stack((v_state, v_obs), dim=-3)
            if features.valid_mask.ndim >= 1:
                scalar_pair = scalar_pair.flatten(start_dim=-3, end_dim=-2)
                vector_pair = vector_pair.flatten(start_dim=-4, end_dim=-3)
            structured["scalar_tokens"] = scalar_pair
            structured["vector_tokens"] = vector_pair
        metadata = dict(features.metadata)
        metadata.update(
            {
                "causal_prior_used": features.metadata.get("causal_prior_used"),
                "previous_prior_used": features.metadata.get("previous_prior_used"),
                "previous_posterior_used": features.metadata.get("previous_posterior_used"),
                "current_posterior_used": torch.zeros_like(valid_mask),
                "raw_innovation": features.raw_innovation,
                "direct_los": features.observed_los,
                "station_id_reliable": features.station_id_reliable,
                "station_changed": features.station_changed,
                "los_direct_sequence_embedding": los_direct_sequence_embedding,
                "los_direct_sequence_features": los_direct_sequence_features,
            }
        )
        return TokenizerOutput(
            tokens=tokens,
            state_token=state_token,
            observation_token=observation_token,
            features=features,
            valid_mask=valid_mask,
            branch_valid_mask=branch_valid_mask,
            first_frame_missing_mask=features.first_frame_missing_mask,
            raw_innovation=features.raw_innovation,
            direct_los=features.observed_los,
            delta_los=features.delta_los,
            delta_t_sec=features.delta_t_sec,
            causal_mask=_causal_mask(valid_mask),
            los_direct_sequence_embedding=los_direct_sequence_embedding,
            los_direct_sequence_features=los_direct_sequence_features,
            metadata=metadata,
            **structured,
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
        features: FeatureBatch | None = None,
    ) -> TokenizerOutput:
        if features is not None:
            return self._wrap(features)
        features = self.feature_builder.forward_sequence(
            times_sec=times_sec,
            observed_los=observed_los,
            prior_state_si=prior_state_si,
            previous_prior_state_si=previous_prior_state_si,
            previous_posterior_state_si=previous_posterior_state_si,
            station_position_si=station_position_si,
            station_state_si=station_state_si,
            station_ids=station_ids,
            station_id_reliable=station_id_reliable,
            predicted_los=predicted_los,
            raw_innovation=raw_innovation,
            valid_mask=valid_mask,
            lengths=lengths,
            delta_time_seconds=delta_time_seconds,
            init_time_sec=init_time_sec,
            t_init_seconds=t_init_seconds,
            t_init_sec=t_init_sec,
            metadata=metadata,
        )
        return self._wrap(features)

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
    ) -> TokenizerOutput:
        features = self.feature_builder.forward_step(
            time_sec=time_sec,
            observed_los=observed_los,
            prior_state_si=prior_state_si,
            previous_prior_state_si=previous_prior_state_si,
            previous_posterior_state_si=previous_posterior_state_si,
            station_position_si=station_position_si,
            station_state_si=station_state_si,
            station_id=station_id,
            station_id_reliable=station_id_reliable,
            predicted_los=predicted_los,
            raw_innovation=raw_innovation,
            valid=valid,
            delta_time_seconds=delta_time_seconds,
            init_time_sec=init_time_sec,
            t_init_seconds=t_init_seconds,
            t_init_sec=t_init_sec,
            metadata=metadata,
            **kwargs,
        )
        return self._wrap(features)

    def forward(self, times_sec: Any = None, observed_los: Any = None, **kwargs: Any) -> TokenizerOutput:
        supplied_features = kwargs.pop("features", None)
        if supplied_features is None:
            supplied_features = kwargs.pop("feature_batch", None)
        if isinstance(times_sec, FeatureBatch) and supplied_features is None:
            supplied_features = times_sec
        if supplied_features is not None:
            if not isinstance(supplied_features, FeatureBatch):
                raise TypeError("features must be a FeatureBatch")
            return self._wrap(supplied_features)
        # Let FeatureBuilder handle mapping aliases and leakage checks, but
        # preserve scalar/single-frame convenience for streaming callers.
        candidate_times = times_sec
        candidate_los = observed_los
        if isinstance(candidate_times, Mapping):
            payload = candidate_times
            candidate_times = payload.get("times_sec", payload.get("times", payload.get("time_sec", payload.get("time"))))
            candidate_los = payload.get("observed_los", payload.get("los", payload.get("direct_los")))
        try:
            time_tensor = torch.as_tensor(candidate_times)
            los_tensor = torch.as_tensor(candidate_los) if candidate_los is not None else None
            is_step = time_tensor.ndim == 0 or (los_tensor is not None and los_tensor.ndim == 1)
        except (TypeError, ValueError, RuntimeError):
            is_step = False
        if is_step:
            if isinstance(times_sec, Mapping):
                return self._wrap(self.feature_builder.forward(times_sec, **kwargs))
            return self.forward_step(time_sec=times_sec, observed_los=observed_los, **kwargs)
        if isinstance(times_sec, Mapping):
            return self._wrap(self.feature_builder.forward(times_sec, **kwargs))
        return self.forward_sequence(times_sec=times_sec, observed_los=observed_los, **kwargs)

    tokenize = forward
    tokenize_step = forward_step
    tokenize_sequence = forward_sequence


Tokenizer = V6Tokenizer
V6ObservationTokenizer = V6Tokenizer


__all__ = [
    "BRANCH_NAMES",
    "ResidualFeatureMLP",
    "select_last_valid",
    "TokenizerOutput",
    "V6Tokenizer",
    "V6ObservationTokenizer",
    "Tokenizer",
]

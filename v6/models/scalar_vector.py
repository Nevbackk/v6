"""Rotation-compatible scalar/vector token projection without vector flattening.

Scalar inputs are physical rotation invariants. Vector inputs are 3-vectors.
Vector channels are mixed with shared coefficients across XYZ and gated by
rotation-invariant scalar features. No 3-D vector bias is added.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ScalarVectorEncoder(nn.Module):
    def __init__(
        self,
        scalar_input_dim: int,
        vector_input_channels: int,
        *,
        scalar_channels: int = 32,
        vector_channels: int = 16,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if min(scalar_input_dim, vector_input_channels, scalar_channels, vector_channels) <= 0:
            raise ValueError("scalar/vector dimensions must be positive")
        self.scalar_channels = scalar_channels
        self.vector_channels = vector_channels
        self.scalar_net = nn.Sequential(
            nn.Linear(scalar_input_dim + vector_input_channels, 2 * scalar_channels, dtype=dtype),
            nn.GELU(),
            nn.Linear(2 * scalar_channels, scalar_channels, dtype=dtype),
            nn.LayerNorm(scalar_channels, dtype=dtype),
        )
        self.vector_weights = nn.Parameter(torch.empty(vector_channels, vector_input_channels, dtype=dtype))
        nn.init.xavier_uniform_(self.vector_weights)
        self.vector_gate = nn.Sequential(
            nn.Linear(scalar_channels, vector_channels, dtype=dtype),
            nn.Sigmoid(),
        )

    def forward(self, scalars: Tensor, vectors: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        dtype = self.vector_weights.dtype
        scalars = scalars.to(dtype=dtype)
        vectors = vectors.to(dtype=dtype)
        invariant_norms = torch.linalg.vector_norm(vectors, dim=-1)
        scalar_output = self.scalar_net(torch.cat((scalars, invariant_norms), dim=-1))
        # Channel mixing only, the same weights for x, y, z.
        vector_output = torch.einsum('...nc,on->...oc', vectors, self.vector_weights)
        vector_output = vector_output * self.vector_gate(scalar_output).unsqueeze(-1)
        valid = valid.to(torch.bool)
        scalar_output = torch.where(valid.unsqueeze(-1), scalar_output, torch.zeros_like(scalar_output))
        vector_output = torch.where(valid.unsqueeze(-1).unsqueeze(-1), vector_output, torch.zeros_like(vector_output))
        return scalar_output, vector_output

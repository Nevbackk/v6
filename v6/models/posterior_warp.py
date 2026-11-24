"""A normalized six-dimensional posterior with conditional rational-quadratic spline couplings."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .batched_physics import DTYPE, state_scale


def standard_normal_log_prob(z):
    return -0.5 * (z.square() + math.log(2 * math.pi)).sum(-1)


def _knots(logits, bound):
    # Linear tails preserve support over the entire real line. A machine-scale
    # minimum width prevents coincident knots in the working precision.
    eps = torch.finfo(logits.dtype).eps * 8
    fractions = logits.softmax(-1) * (1 - logits.shape[-1] * eps) + eps
    inside = fractions.cumsum(-1)[..., :-1] * (2 * bound) - bound
    edge = torch.full_like(logits[..., :1], bound)
    return torch.cat((-edge, inside, edge), dim=-1)


def monotone_spline(value, raw, bins, *, inverse=False, bound=4.0):
    """C1 scalar bijection, analytic inverse and log derivative (Durkan et al., 2019)."""
    shift, log_scale = raw[..., 0], torch.asinh(raw[..., 1])
    xk = _knots(raw[..., 2 : 2 + bins], bound)
    yk = _knots(raw[..., 2 + bins : 2 + 2 * bins], bound)
    inner = torch.asinh(raw[..., 2 + 2 * bins :]).exp()
    edge = torch.ones_like(xk[..., :1])
    slopes = torch.cat((edge, inner, edge), dim=-1)
    v = (value - shift) * (-log_scale).exp() if inverse else value
    # Selecting a bin is piecewise constant; values and first derivatives agree at its edges.
    grid = yk if inverse else xk
    index = (v.unsqueeze(-1) >= grid[..., 1:-1]).sum(-1, keepdim=True)

    def take(table, offset=0):
        return table.gather(-1, index + offset).squeeze(-1)

    x0, y0 = take(xk), take(yk)
    width, height = take(xk, 1) - x0, take(yk, 1) - y0
    d0, d1 = take(slopes), take(slopes, 1)
    secant = height / width
    bend = d0 + d1 - 2 * secant
    inside = (v >= -bound) & (v <= bound)
    if inverse:
        y = ((v - y0) / height).clamp(0, 1)
        a = y * bend + secant - d0
        b = d0 - y * bend
        c = -secant * y
        disc = (b.square() - 4 * a * c).clamp_min(0).sqrt()
        # The two equivalent roots avoid cancellation when b is negative.
        small = torch.finfo(v.dtype).tiny
        t = torch.where(
            b >= 0,
            -2 * c / (b + disc).clamp_min(small),
            (-b + disc) / (2 * torch.where(a > 0, a, torch.ones_like(a))),
        )
        t = t.clamp(0, 1)
    else:
        t = ((v - x0) / width).clamp(0, 1)
    cross = t * (1 - t)
    denom = secant + bend * cross
    derivative = (
        secant.square() * (d1 * t.square() + 2 * secant * cross + d0 * (1 - t).square()) / denom.square()
    )
    ld = torch.where(inside, derivative.log(), 0.0) + log_scale
    if inverse:
        return torch.where(inside, x0 + t * width, v), -ld
    y = y0 + height * (secant * t.square() + d0 * cross) / denom
    return torch.where(inside, y, v) * log_scale.exp() + shift, ld


class ConditionalOrbitWarp(nn.Module):
    """Complementary three-coordinate couplings; zero track shape is the identity."""

    def __init__(self, bins=8, hidden=32, dtype=DTYPE):
        super().__init__()
        self.bins = bins
        self.coefficients = 3 * bins + 1
        self.shape_size = 6 * self.coefficients
        self.conditioners = nn.ModuleList(
            nn.Sequential(
                nn.Linear(3, hidden, dtype=dtype),
                nn.SiLU(),
                nn.Linear(hidden, 3 * self.coefficients, dtype=dtype),
            )
            for _ in range(2)
        )
        for net in self.conditioners:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
        self.register_buffer("order", torch.tensor([0, 2, 4, 1, 3, 5]))
        self.register_buffer("undo_order", torch.tensor([0, 3, 1, 4, 2, 5]))

    def map(self, value, shape, *, inverse=False):
        if value.shape[-1] != 6 or shape.shape[-1] != self.shape_size:
            raise ValueError("Expected six coordinates and the track's spline coefficients")
        left, right = value.index_select(-1, self.order).chunk(2, dim=-1)
        raw = shape.unflatten(-1, (2, 3, self.coefficients))
        ld = value.new_zeros(value.shape[:-1])
        for layer in (1, 0) if inverse else (0, 1):
            active, fixed = (left, right) if layer == 0 else (right, left)
            modulation = self.conditioners[layer](torch.asinh(fixed)).unflatten(-1, (3, self.coefficients))
            params = raw[..., layer, :, :] * (1 + modulation.tanh())
            active, log_jac = monotone_spline(active, params, self.bins, inverse=inverse)
            if layer == 0:
                left = active
            else:
                right = active
            ld = ld + log_jac.sum(-1)
        return torch.cat((left, right), dim=-1).index_select(-1, self.undo_order), ld

    def log_prob(self, value, shape):
        z, ld = self.map(value, shape, inverse=True)
        return standard_normal_log_prob(z) + ld

    def velocity(self, at, tau, old, new):
        """Eulerian velocity of the residual path T(z; old + tau*(new-old))."""
        direction = new - old
        shape = old + tau * direction
        base, _ = self.map(at, shape, inverse=True)
        _, velocity = torch.func.jvp(
            lambda a: self.map(base, a)[0], (shape,), (direction.expand_as(shape),)
        )
        return velocity


@dataclass(frozen=True)
class WarpedOrbitPosterior:
    """x_SI = center + D root T(z; shape). Center and root are reference parameters."""

    center: Tensor
    root: Tensor
    shape: Tensor
    warp: ConditionalOrbitWarp

    @property
    def scale(self):
        return state_scale(device=self.center.device, dtype=self.center.dtype)

    def standardize(self, x):
        y = (x - self.center) / self.scale
        return torch.linalg.solve(self.root, y.reshape(-1, 6).T).T.reshape_as(y)

    def from_chart(self, coordinates):
        return self.center + self.scale * (coordinates @ self.root.T)

    def transform(self, z):
        y, ld = self.warp.map(z, self.shape)
        return self.from_chart(y), ld + self.log_affine_volume()

    def inverse(self, x):
        z, ld = self.warp.map(self.standardize(x), self.shape, inverse=True)
        return z, ld - self.log_affine_volume()

    def log_affine_volume(self):
        return torch.linalg.slogdet(self.root).logabsdet + self.scale.log().sum()

    def log_prob(self, x):
        z, ld = self.inverse(x)
        return standard_normal_log_prob(z) + ld

    def sample(self, count, *, generator=None):
        z = torch.randn((count, 6), dtype=self.center.dtype, device=self.center.device, generator=generator)
        return self.transform(z)[0]

    def moments(self, nodes):
        # Center in chart space before converting large ECI components back to SI.
        y, _ = self.warp.map(nodes, self.shape)
        mean = y.mean(0)
        offsets = ((y - mean) @ self.root.T) * self.scale
        return self.from_chart(mean), offsets.T @ offsets / len(y)

    def detached(self):
        return WarpedOrbitPosterior(
            self.center.detach(), self.root.detach(), self.shape.detach(), self.warp
        )

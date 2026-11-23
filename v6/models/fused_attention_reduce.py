"""Stable all-history reduction of compact, ordered attention tile statistics."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _reduce(Max, Den, Scalar, Vector, Scale, OutS, OutV, Valid,
            tiles, H: tl.constexpr, D: tl.constexpr, T: tl.constexpr,
            C: tl.constexpr):
    row = tl.program_id(0)
    batch = row // (H * 2)
    head_query = row % (H * 2)
    tile = tl.arange(0, T)
    offset = (batch * tiles + tile) * (H * 2) + head_query
    den = tl.load(Den + offset, tile < tiles, 0)
    valid = (tile < tiles) & (den > 0)
    max_tile = tl.load(Max + offset, valid, -float("inf"))
    max_all = tl.max(max_tile, axis=0)
    has_keys = tl.sum(valid.to(tl.int32), axis=0) > 0
    max_all = tl.where(has_keys, max_all, 0.0)
    scale = libdevice.exp(max_tile - max_all)
    total = tl.sum(scale * den, axis=0)
    scale /= tl.where(has_keys, total, 1.0)
    tl.store(Scale + offset, scale, tile < tiles)
    c = tl.arange(0, C)
    s = tl.load(Scalar + offset[:, None] * D + c[None, :],
                valid[:, None] & (c[None, :] < D), 0)
    v = tl.load(Vector + offset[:, None] * (D * 3) + c[None, :],
                valid[:, None] & (c[None, :] < D * 3), 0)
    tl.store(OutS + row * D + c, tl.sum(scale[:, None] * s, axis=0), c < D)
    tl.store(OutV + row * (D * 3) + c, tl.sum(scale[:, None] * v, axis=0), c < D * 3)
    tl.store(Valid + row, has_keys)


@triton.jit
def _reduce_ordered(Max, Den, Scalar, Vector, Scale, OutS, OutV, Valid,
                    tiles, H: tl.constexpr, D: tl.constexpr, T: tl.constexpr,
                    C: tl.constexpr):
    """Fold 32-frame tile statistics in the exact reference tile order."""
    row = tl.program_id(0)
    batch = row // (H * 2)
    head_query = row % (H * 2)
    first = batch * tiles * H * 2 + head_query
    c = tl.arange(0, C)
    running_max = tl.full((), 0.0, tl.float64)
    running_den = tl.full((), 0.0, tl.float64)
    running_s = tl.full((C,), 0.0, tl.float64)
    running_v = tl.full((C,), 0.0, tl.float64)
    for tile in range(tiles):
        offset = first + tile * H * 2
        den = tl.load(Den + offset)
        maximum = tl.load(Max + offset)
        has_old = running_den > 0
        has_new = den > 0
        next_max = tl.where(has_old & has_new, tl.maximum(running_max, maximum),
                            tl.where(has_old, running_max, tl.where(has_new, maximum, 0.0)))
        old = tl.where(has_old, libdevice.exp(running_max - next_max), 0.0)
        new = tl.where(has_new, libdevice.exp(maximum - next_max), 0.0)
        s = tl.load(Scalar + offset * D + c, c < D, 0)
        v = tl.load(Vector + offset * (D * 3) + c, c < D * 3, 0)
        running_den = old * running_den + new * den
        running_s = old * running_s + new * s
        running_v = old * running_v + new * v
        running_max = next_max

    has_keys = running_den > 0
    denom = tl.where(has_keys, running_den, 1.0)
    tl.store(OutS + row * D + c, running_s / denom, c < D)
    tl.store(OutV + row * (D * 3) + c, running_v / denom, c < D * 3)
    tl.store(Valid + row, has_keys)
    tile = tl.arange(0, T)
    offset = first + tile * H * 2
    den = tl.load(Den + offset, tile < tiles, 0.0)
    maximum = tl.load(Max + offset, (tile < tiles) & (den > 0), -float("inf"))
    scale = libdevice.exp(maximum - running_max) / denom
    tl.store(Scale + offset, scale, tile < tiles)


@triton.jit
def _reduce_backward(Scale, OutS, OutV, GradS, GradV, GradDen, GradScalar, GradVector,
                     tiles, H: tl.constexpr, D: tl.constexpr, T: tl.constexpr,
                     C: tl.constexpr):
    row = tl.program_id(0)
    batch = row // (H * 2)
    head_query = row % (H * 2)
    tile = tl.arange(0, T)
    c = tl.arange(0, C)
    offset = (batch * tiles + tile) * (H * 2) + head_query
    scale = tl.load(Scale + offset, tile < tiles, 0)
    gs = tl.load(GradS + row * D + c, c < D, 0)
    gv = tl.load(GradV + row * (D * 3) + c, c < D * 3, 0)
    s = tl.load(OutS + row * D + c, c < D, 0)
    v = tl.load(OutV + row * (D * 3) + c, c < D * 3, 0)
    dot = tl.sum(gs * s + gv * v, axis=0)
    tl.store(GradDen + offset, -scale * dot, tile < tiles)
    tl.store(GradScalar + offset[:, None] * D + c[None, :], scale[:, None] * gs[None, :],
             (tile[:, None] < tiles) & (c[None, :] < D))
    tl.store(GradVector + offset[:, None] * (D * 3) + c[None, :], scale[:, None] * gv[None, :],
             (tile[:, None] < tiles) & (c[None, :] < D * 3))


class _ReduceTiles(torch.autograd.Function):
    @staticmethod
    def forward(ctx, maximum, denominator, scalar, vector, ordered):
        b, tiles, h, q, d = scalar.shape
        assert q == 2
        out_s = scalar.new_empty((b, h, q, d))
        out_v = vector.new_empty((b, h, q, d, 3))
        valid = torch.empty((b, h, q), device=scalar.device, dtype=torch.bool)
        scale = torch.empty_like(denominator)
        kernel = _reduce_ordered if ordered else _reduce
        kernel[(b * h * q,)](
            maximum.contiguous(), denominator.contiguous(), scalar.contiguous(),
            vector.contiguous(), scale, out_s, out_v, valid, tiles, h, d,
            triton.next_power_of_2(tiles), triton.next_power_of_2(d * 3),
            enable_fp_fusion=False,
        )
        row_valid = valid[:, 0, :]
        ctx.mark_non_differentiable(row_valid)
        ctx.save_for_backward(scale, out_s, out_v)
        return out_s, out_v, row_valid

    @staticmethod
    def backward(ctx, grad_s, grad_v, _):
        scale, out_s, out_v = ctx.saved_tensors
        b, tiles, h, q, _ = scale.shape
        d = out_s.shape[-1]
        grad_den = torch.empty_like(scale)
        grad_scalar = out_s.new_empty((b, tiles, h, q, d))
        grad_vector = out_v.new_empty((b, tiles, h, q, d, 3))
        _reduce_backward[(b * h * q,)](
            scale, out_s, out_v, grad_s.contiguous(), grad_v.contiguous(),
            grad_den, grad_scalar, grad_vector, tiles, h, d,
            triton.next_power_of_2(tiles), triton.next_power_of_2(d * 3),
            enable_fp_fusion=False,
        )
        return None, grad_den, grad_scalar, grad_vector, None


def reduce_tiles(maximum, denominator, scalar, vector, *, ordered=False):
    return _ReduceTiles.apply(maximum, denominator, scalar, vector, ordered)

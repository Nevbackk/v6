"""Fused statistics for one (or a small group of) attention key tiles.

The online attention implementation consumes the four quantities returned by
``tile_context`` and merges them with an online log-sum-exp reduction.  The
kernel intentionally does *not* normalize the context: keeping the local
maximum and denominator separate is what makes the operation composable over
history blocks.

The public layout is

``qs``
    ``[B, H, 2, D]`` scalar queries.
``qv``
    ``[B, H, 2, D, 3]`` vector queries.
``ks, vs``
    ``[B, G, H, K, D]`` scalar keys/values.
``kv, vv``
    ``[B, G, H, K, D, 3]`` vector keys/values.
``rel``
    ``[B, G, H, 2, K, D]`` relation vectors already projected to head
    dimension.  The scalar logit uses ``qs @ (ks + rel)``.
``allowed``
    ``[B, G, 2, K]`` boolean edge mask.

``G`` is a group of independent key tiles.  Every output row is independent,
so the Triton launch has one program for each ``(B, G, H, query)`` row.  The
backward launch writes query/key/value gradients to query- or key-indexed
scratch buffers first; reductions over ``G`` and the two query rows happen in
PyTorch, avoiding atomics and races.

The module has no mandatory Triton dependency.  The same custom-autograd
contract is implemented with PyTorch on CPU and for non-FP64 callers; a
supported CUDA/FP64 invocation uses Triton strictly so compiler failures are
visible rather than silently changing the execution path.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


try:  # Triton is optional in the reference/CPU test environment.
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - depends on the host installation.
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    libdevice = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False


_QUERY_ROWS = 2
_VECTOR_DIM = 3


def _validate_and_prepare(
    qs: Tensor,
    qv: Tensor,
    ks: Tensor,
    kv: Tensor,
    vs: Tensor,
    vv: Tensor,
    rel: Tensor,
    allowed: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Validate layouts and promote floating inputs to one computation dtype."""

    values = (qs, qv, ks, kv, vs, vv, rel)
    if not all(isinstance(x, Tensor) for x in values) or not isinstance(allowed, Tensor):
        raise TypeError("tile_context inputs must all be torch.Tensor objects")
    device = qs.device
    if any(x.device != device for x in values) or allowed.device != device:
        raise ValueError("all tile_context tensors must be on one device")
    if any(not torch.is_floating_point(x) for x in values):
        raise TypeError("tile_context floating inputs must have a floating dtype")
    if qs.ndim != 4 or qs.shape[2] != _QUERY_ROWS:
        raise ValueError("qs must have shape [B,H,2,D]")
    batch, heads, _, dim = (int(x) for x in qs.shape)
    if batch <= 0 or heads <= 0 or dim <= 0:
        raise ValueError("qs dimensions B, H and D must be positive")
    if qv.ndim != 5 or tuple(qv.shape) != (batch, heads, _QUERY_ROWS, dim, _VECTOR_DIM):
        raise ValueError("qv must have shape [B,H,2,D,3] matching qs")
    if ks.ndim != 5:
        raise ValueError("ks must have shape [B,G,H,K,D]")
    group, key_heads, key_count, key_dim = (int(ks.shape[1]), int(ks.shape[2]), int(ks.shape[3]), int(ks.shape[4]))
    if group <= 0 or key_count <= 0:
        raise ValueError("ks dimensions G and K must be positive")
    if key_heads != heads or key_dim != dim:
        raise ValueError("ks must match qs head and feature dimensions")
    expected_scalar = (batch, group, heads, key_count, dim)
    expected_vector = (batch, group, heads, key_count, dim, _VECTOR_DIM)
    if tuple(vs.shape) != expected_scalar:
        raise ValueError("vs must have shape [B,G,H,K,D] matching ks")
    if tuple(kv.shape) != expected_vector or tuple(vv.shape) != expected_vector:
        raise ValueError("kv and vv must have shape [B,G,H,K,D,3] matching ks")
    expected_relation = (batch, group, heads, _QUERY_ROWS, key_count, dim)
    if tuple(rel.shape) != expected_relation:
        raise ValueError("rel must have shape [B,G,H,2,K,D] matching ks")
    if tuple(allowed.shape) != (batch, group, _QUERY_ROWS, key_count):
        raise ValueError("allowed must have shape [B,G,2,K] matching ks")

    work_dtype = qs.dtype
    for x in values[1:]:
        work_dtype = torch.promote_types(work_dtype, x.dtype)
    if not work_dtype.is_floating_point:
        raise TypeError("tile_context requires floating-point inputs")

    # contiguous() is important for the Triton path.  These copies remain
    # differentiable Copy nodes, so gradients still reach a caller's original
    # non-contiguous/cast tensors.
    prepared = tuple(x.to(dtype=work_dtype).contiguous() for x in values)
    mask = allowed.to(dtype=torch.bool).contiguous()
    return (*prepared, mask)  # type: ignore[return-value]


def _torch_tile_forward(
    qs: Tensor,
    qv: Tensor,
    ks: Tensor,
    kv: Tensor,
    vs: Tensor,
    vv: Tensor,
    rel: Tensor,
    allowed: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Reference forward, with invalid storage masked before any arithmetic."""

    # [B,G,H,2,K] and [B,G,H,2,K,D] views.  Masking each operand before the
    # sums matters when padded KV storage is uninitialized or contains NaN.
    edge = allowed[:, :, None, :, :]
    edge_d = edge.unsqueeze(-1)
    ks_clean = torch.where(edge_d, ks[:, :, :, None, :, :], torch.zeros_like(ks[:, :, :, None, :, :]))
    rel_clean = torch.where(edge_d, rel, torch.zeros_like(rel))
    kv_clean = torch.where(
        edge_d.unsqueeze(-1),
        kv[:, :, :, None, :, :, :],
        torch.zeros_like(kv[:, :, :, None, :, :, :]),
    )
    vs_clean = torch.where(edge_d, vs[:, :, :, None, :, :], torch.zeros_like(vs[:, :, :, None, :, :]))
    vv_clean = torch.where(
        edge_d.unsqueeze(-1),
        vv[:, :, :, None, :, :, :],
        torch.zeros_like(vv[:, :, :, None, :, :, :]),
    )

    q_scalar = qs[:, None, :, :, :]
    q_vector = qv[:, None, :, :, :, :]
    scalar_logits = (q_scalar.unsqueeze(-2) * (ks_clean + rel_clean)).sum(dim=-1)
    vector_logits = (q_vector.unsqueeze(-3) * kv_clean).sum(dim=(-1, -2))
    logits = scalar_logits * (float(qs.shape[-1]) ** -0.5)
    logits = logits + vector_logits * (float(_VECTOR_DIM * qs.shape[-1]) ** -0.5)

    # ``safe_logits`` is finite even on invalid edges.  This avoids NaN
    # propagation in the all-masked-row case while preserving exact zeros.
    safe_logits = torch.where(edge, logits, torch.zeros_like(logits))
    has_valid = edge.any(dim=-1, keepdim=True)
    raw_max = safe_logits.masked_fill(~edge, -torch.inf).amax(dim=-1, keepdim=True)
    block_max = torch.where(has_valid, raw_max, torch.zeros_like(raw_max))
    exponentials = torch.where(
        edge,
        torch.exp(safe_logits - block_max),
        torch.zeros_like(safe_logits),
    )
    denominator = exponentials.sum(dim=-1, keepdim=True)
    scalar_numerator = (exponentials.unsqueeze(-1) * vs_clean).sum(dim=-2)
    vector_numerator = (exponentials.unsqueeze(-1).unsqueeze(-1) * vv_clean).sum(dim=-3)
    return block_max, denominator, scalar_numerator, vector_numerator


def _torch_tile_backward(
    qs: Tensor,
    qv: Tensor,
    ks: Tensor,
    kv: Tensor,
    vs: Tensor,
    vv: Tensor,
    rel: Tensor,
    allowed: Tensor,
    block_max: Tensor,
    grad_denominator: Optional[Tensor],
    grad_scalar_numerator: Optional[Tensor],
    grad_vector_numerator: Optional[Tensor],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Analytic backward for the detached-max forward contract.

    If ``e_k = exp(logit_k - max)`` and the local maximum is treated as a
    constant, ``d e_k = e_k d logit_k``.  This is the only subtle distinction
    from asking PyTorch autograd to differentiate ``amax`` directly.
    """

    edge = allowed[:, :, None, :, :]
    edge_d = edge.unsqueeze(-1)
    ks_clean = torch.where(edge_d, ks[:, :, :, None, :, :], torch.zeros_like(ks[:, :, :, None, :, :]))
    rel_clean = torch.where(edge_d, rel, torch.zeros_like(rel))
    kv_clean = torch.where(
        edge_d.unsqueeze(-1),
        kv[:, :, :, None, :, :, :],
        torch.zeros_like(kv[:, :, :, None, :, :, :]),
    )
    vs_clean = torch.where(edge_d, vs[:, :, :, None, :, :], torch.zeros_like(vs[:, :, :, None, :, :]))
    vv_clean = torch.where(
        edge_d.unsqueeze(-1),
        vv[:, :, :, None, :, :, :],
        torch.zeros_like(vv[:, :, :, None, :, :, :]),
    )
    q_scalar = qs[:, None, :, :, :]
    q_vector = qv[:, None, :, :, :, :]
    scalar_logits = (q_scalar.unsqueeze(-2) * (ks_clean + rel_clean)).sum(dim=-1)
    vector_logits = (q_vector.unsqueeze(-3) * kv_clean).sum(dim=(-1, -2))
    logits = scalar_logits * (float(qs.shape[-1]) ** -0.5)
    logits = logits + vector_logits * (float(_VECTOR_DIM * qs.shape[-1]) ** -0.5)
    exponentials = torch.where(
        edge,
        torch.exp(torch.where(edge, logits, torch.zeros_like(logits)) - block_max),
        torch.zeros_like(logits),
    )

    batch, group, heads, _, key_count, dim = rel.shape
    del batch, heads, key_count, dim  # dimensions remain available through tensors
    gz = torch.zeros_like(block_max) if grad_denominator is None else grad_denominator
    gns = torch.zeros_like(block_max.expand(*block_max.shape[:-1], qs.shape[-1])) if grad_scalar_numerator is None else grad_scalar_numerator
    gnv = torch.zeros_like(block_max.expand(*block_max.shape[:-1], qs.shape[-1], _VECTOR_DIM)) if grad_vector_numerator is None else grad_vector_numerator

    # d loss / d e_k combines all three returned statistics.
    de = gz + (gns.unsqueeze(-2) * vs_clean).sum(dim=-1)
    de = de + (gnv.unsqueeze(-3) * vv_clean).sum(dim=(-1, -2))
    grad_logit = exponentials * de
    scalar_scale = float(qs.shape[-1]) ** -0.5
    vector_scale = float(_VECTOR_DIM * qs.shape[-1]) ** -0.5
    combined_key = ks_clean + rel_clean

    # q gradients are summed over independent key tiles.  Key/value gradients
    # are summed over the two query rows; relation is query-row specific.
    grad_qs_per_group = scalar_scale * (
        grad_logit.unsqueeze(-1) * combined_key
    ).sum(dim=-2)
    grad_qv_per_group = vector_scale * (
        grad_logit.unsqueeze(-1).unsqueeze(-1) * kv_clean
    ).sum(dim=-3)
    grad_qs = grad_qs_per_group.sum(dim=1)
    grad_qv = grad_qv_per_group.sum(dim=1)

    grad_ks_per_query = scalar_scale * grad_logit.unsqueeze(-1) * q_scalar.unsqueeze(-2)
    grad_rel = scalar_scale * grad_logit.unsqueeze(-1) * q_scalar.unsqueeze(-2)
    grad_kv_per_query = vector_scale * grad_logit.unsqueeze(-1).unsqueeze(-1) * q_vector.unsqueeze(-3)
    grad_vs_per_query = exponentials.unsqueeze(-1) * gns.unsqueeze(-2)
    grad_vv_per_query = exponentials.unsqueeze(-1).unsqueeze(-1) * gnv.unsqueeze(-3)

    grad_ks = grad_ks_per_query.sum(dim=3)
    grad_kv = grad_kv_per_query.sum(dim=3)
    grad_vs = grad_vs_per_query.sum(dim=3)
    grad_vv = grad_vv_per_query.sum(dim=3)
    # Every input-gradient tensor must use the original input's dtype.  The
    # custom Function receives one promoted dtype, so this cast is mostly a
    # no-op but also documents the intended FP64 accumulation contract.
    return grad_qs, grad_qv, grad_ks, grad_kv, grad_vs, grad_vv, grad_rel


if _TRITON_AVAILABLE:

    @triton.jit
    def _tile_forward_kernel(
        qs, qv, ks, kv, vs, vv, rel, allowed,
        out_max, out_den, out_ns, out_nv,
        batch, groups, heads, key_count, dim,
        sq_b, sq_h, sq_q, sq_d,
        qv_b, qv_h, qv_q, qv_d, qv_c,
        sk_b, sk_g, sk_h, sk_k, sk_d,
        kv_b, kv_g, kv_h, kv_k, kv_d, kv_c,
        sv_b, sv_g, sv_h, sv_k, sv_d,
        vv_b, vv_g, vv_h, vv_k, vv_d, vv_c,
        r_b, r_g, r_h, r_q, r_k, r_d,
        a_b, a_g, a_q, a_k,
        SCALE_SCALAR: tl.constexpr,
        SCALE_VECTOR: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """One independent (batch, group, head, query) tile row."""

        pid = tl.program_id(0)
        query = pid % 2
        rem = pid // 2
        head = rem % heads
        rem = rem // heads
        group = rem % groups
        batch_index = rem // groups

        d = tl.arange(0, BLOCK_D)
        k = tl.arange(0, BLOCK_K)
        d_mask = d < dim
        k_mask = k < key_count
        c = tl.arange(0, BLOCK_C)
        c_mask = c < 3
        row = (((batch_index * groups + group) * heads + head) * 2 + query)

        q_scalar = tl.load(
            qs + batch_index * sq_b + head * sq_h + query * sq_q + d * sq_d,
            mask=d_mask,
            other=0.0,
        ).to(tl.float64)
        q_vector = tl.load(
            qv + batch_index * qv_b + head * qv_h + query * qv_q
            + d[:, None] * qv_d + c[None, :] * qv_c,
            mask=d_mask[:, None] & c_mask[None, :],
            other=0.0,
        ).to(tl.float64)
        edge = tl.load(
            allowed + batch_index * a_b + group * a_g + query * a_q + k * a_k,
            mask=k_mask,
            other=0,
        ).to(tl.int1)
        edge_d = edge[:, None] & d_mask[None, :]
        edge_v = edge[:, None, None] & d_mask[None, :, None] & c_mask[None, None, :]

        key_scalar = tl.load(
            ks + batch_index * sk_b + group * sk_g + head * sk_h
            + k[:, None] * sk_k + d[None, :] * sk_d,
            mask=edge_d,
            other=0.0,
        ).to(tl.float64)
        relation = tl.load(
            rel + batch_index * r_b + group * r_g + head * r_h + query * r_q
            + k[:, None] * r_k + d[None, :] * r_d,
            mask=edge_d,
            other=0.0,
        ).to(tl.float64)
        key_vector = tl.load(
            kv + batch_index * kv_b + group * kv_g + head * kv_h
            + k[:, None, None] * kv_k + d[None, :, None] * kv_d + c[None, None, :] * kv_c,
            mask=edge_v,
            other=0.0,
        ).to(tl.float64)
        scale_scalar = tl.full((), SCALE_SCALAR, tl.float64)
        scale_vector = tl.full((), SCALE_VECTOR, tl.float64)
        logits_scalar = tl.sum((key_scalar + relation) * q_scalar[None, :], axis=1) * scale_scalar
        logits_vector = tl.sum(tl.sum(key_vector * q_vector[None, :, :], axis=2), axis=1) * scale_vector
        logits = logits_scalar + logits_vector
        neg_inf = tl.full((BLOCK_K,), -float("inf"), tl.float64)
        masked_logits = tl.where(edge, logits, neg_inf)
        has_valid = tl.sum(edge.to(tl.int32), axis=0) > 0
        raw_max = tl.max(masked_logits, axis=0)
        local_max = tl.where(has_valid, raw_max, 0.0)
        exponential = tl.where(edge, libdevice.exp(logits - local_max), 0.0)

        value_scalar = tl.load(
            vs + batch_index * sv_b + group * sv_g + head * sv_h
            + k[:, None] * sv_k + d[None, :] * sv_d,
            mask=edge_d,
            other=0.0,
        ).to(tl.float64)
        value_vector = tl.load(
            vv + batch_index * vv_b + group * vv_g + head * vv_h
            + k[:, None, None] * vv_k + d[None, :, None] * vv_d + c[None, None, :] * vv_c,
            mask=edge_v,
            other=0.0,
        ).to(tl.float64)
        numerator_scalar = tl.sum(exponential[:, None] * value_scalar, axis=0)
        numerator_vector = tl.sum(exponential[:, None, None] * value_vector, axis=0)

        tl.store(out_max + row, local_max)
        tl.store(out_den + row, tl.sum(exponential, axis=0))
        tl.store(out_ns + row * dim + d, numerator_scalar, mask=d_mask)
        tl.store(
            out_nv + row * dim * 3 + d[:, None] * 3 + c[None, :],
            numerator_vector,
            mask=d_mask[:, None] & c_mask[None, :],
        )


    @triton.jit
    def _tile_backward_kernel(
        qs, qv, ks, kv, vs, vv, rel, allowed, block_max,
        grad_den, grad_ns, grad_nv,
        scratch_qs, scratch_qv,
        scratch_ks, scratch_kv, scratch_vs, scratch_vv, grad_rel,
        batch, groups, heads, key_count, dim,
        SCALE_SCALAR: tl.constexpr,
        SCALE_VECTOR: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """One row of analytic backward; reductions happen outside the kernel."""

        pid = tl.program_id(0)
        query = pid % 2
        rem = pid // 2
        head = rem % heads
        rem = rem // heads
        group = rem % groups
        batch_index = rem // groups
        d = tl.arange(0, BLOCK_D)
        k = tl.arange(0, BLOCK_K)
        c = tl.arange(0, BLOCK_C)
        d_mask = d < dim
        k_mask = k < key_count
        c_mask = c < 3
        edge = tl.load(
            allowed + ((batch_index * groups + group) * 2 + query) * key_count + k,
            mask=k_mask,
            other=0,
        ).to(tl.int1)
        edge_d = edge[:, None] & d_mask[None, :]
        edge_v = edge[:, None, None] & d_mask[None, :, None] & c_mask[None, None, :]

        q_scalar = tl.load(
            qs + (batch_index * heads + head) * 2 * dim + query * dim + d,
            mask=d_mask,
            other=0.0,
        ).to(tl.float64)
        q_vector = tl.load(
            qv + ((batch_index * heads + head) * 2 + query) * dim * 3
            + d[:, None] * 3 + c[None, :],
            mask=d_mask[:, None] & c_mask[None, :],
            other=0.0,
        ).to(tl.float64)
        key_scalar = tl.load(
            ks + (((batch_index * groups + group) * heads + head) * key_count) * dim
            + k[:, None] * dim + d[None, :],
            mask=edge_d,
            other=0.0,
        ).to(tl.float64)
        relation = tl.load(
            rel + ((((batch_index * groups + group) * heads + head) * 2 + query) * key_count) * dim
            + k[:, None] * dim + d[None, :],
            mask=edge_d,
            other=0.0,
        ).to(tl.float64)
        key_vector = tl.load(
            kv + ((((batch_index * groups + group) * heads + head) * key_count) * dim * 3)
            + k[:, None, None] * dim * 3 + d[None, :, None] * 3 + c[None, None, :],
            mask=edge_v,
            other=0.0,
        ).to(tl.float64)
        scale_scalar = tl.full((), SCALE_SCALAR, tl.float64)
        scale_vector = tl.full((), SCALE_VECTOR, tl.float64)
        logits = tl.sum((key_scalar + relation) * q_scalar[None, :], axis=1) * scale_scalar
        logits = logits + tl.sum(tl.sum(key_vector * q_vector[None, :, :], axis=2), axis=1) * scale_vector
        max_value = tl.load(
            block_max + (((batch_index * groups + group) * heads + head) * 2 + query),
        ).to(tl.float64)
        exponential = tl.where(edge, libdevice.exp(logits - max_value), 0.0)

        row = (((batch_index * groups + group) * heads + head) * 2 + query)
        grad_z = tl.load(grad_den + row).to(tl.float64)
        grad_scalar = tl.load(
            grad_ns + row * dim + d,
            mask=d_mask,
            other=0.0,
        ).to(tl.float64)
        grad_vector = tl.load(
            grad_nv + row * dim * 3 + d[:, None] * 3 + c[None, :],
            mask=d_mask[:, None] & c_mask[None, :],
            other=0.0,
        ).to(tl.float64)
        value_scalar = tl.load(
            vs + (((batch_index * groups + group) * heads + head) * key_count) * dim
            + k[:, None] * dim + d[None, :],
            mask=edge_d,
            other=0.0,
        ).to(tl.float64)
        value_vector = tl.load(
            vv + ((((batch_index * groups + group) * heads + head) * key_count) * dim * 3)
            + k[:, None, None] * dim * 3 + d[None, :, None] * 3 + c[None, None, :],
            mask=edge_v,
            other=0.0,
        ).to(tl.float64)
        de = grad_z + tl.sum(value_scalar * grad_scalar[None, :], axis=1)
        de = de + tl.sum(tl.sum(value_vector * grad_vector[None, :, :], axis=2), axis=1)
        grad_logit = exponential * de

        grad_qs = tl.sum(grad_logit[:, None] * (key_scalar + relation), axis=0) * scale_scalar
        grad_qv = tl.sum(grad_logit[:, None, None] * key_vector, axis=0) * scale_vector
        qs_key_grad = grad_logit[:, None] * q_scalar[None, :] * scale_scalar
        qv_key_grad = grad_logit[:, None, None] * q_vector[None, :, :] * scale_vector
        rel_grad = qs_key_grad
        scalar_value_grad = exponential[:, None] * grad_scalar[None, :]
        vector_value_grad = exponential[:, None, None] * grad_vector[None, :, :]

        base_q = ((((batch_index * groups + group) * heads + head) * 2 + query) * dim)
        tl.store(scratch_qs + base_q + d, grad_qs, mask=d_mask)
        tl.store(scratch_qv + base_q * 3 + d[:, None] * 3 + c[None, :], grad_qv, mask=d_mask[:, None] & c_mask[None, :])
        base_k = (((batch_index * groups + group) * heads + head) * 2 + query) * key_count * dim
        all_d = k_mask[:, None] & d_mask[None, :]
        all_v = k_mask[:, None, None] & d_mask[None, :, None] & c_mask[None, None, :]
        tl.store(scratch_ks + base_k + k[:, None] * dim + d[None, :], qs_key_grad, mask=all_d)
        tl.store(scratch_kv + base_k * 3 + k[:, None, None] * dim * 3 + d[None, :, None] * 3 + c[None, None, :], qv_key_grad, mask=all_v)
        tl.store(scratch_vs + base_k + k[:, None] * dim + d[None, :], scalar_value_grad, mask=all_d)
        tl.store(scratch_vv + base_k * 3 + k[:, None, None] * dim * 3 + d[None, :, None] * 3 + c[None, None, :], vector_value_grad, mask=all_v)
        rel_base = ((((batch_index * groups + group) * heads + head) * 2 + query) * key_count * dim)
        tl.store(grad_rel + rel_base + k[:, None] * dim + d[None, :], rel_grad, mask=all_d)


def _can_use_triton(inputs: tuple[Tensor, ...]) -> bool:
    """Triton is intentionally limited to the tested contiguous FP64 path."""

    if not _TRITON_AVAILABLE:
        return False
    if not inputs or not all(x.is_cuda for x in inputs):
        return False
    # FP64 is the accuracy contract of this kernel.  The PyTorch path remains
    # available for fp32/bfloat16 callers and unusual devices.
    if inputs[0].dtype != torch.float64 or any(x.dtype != torch.float64 for x in inputs[:-1]):
        return False
    if int(inputs[2].shape[-2]) > 64:
        return False
    return True


def _triton_forward(
    qs: Tensor,
    qv: Tensor,
    ks: Tensor,
    kv: Tensor,
    vs: Tensor,
    vv: Tensor,
    rel: Tensor,
    allowed: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    batch, heads, _, dim = (int(x) for x in qs.shape)
    groups, key_count = int(ks.shape[1]), int(ks.shape[3])
    out_shape = (batch, groups, heads, 2)
    out_max = torch.empty((*out_shape, 1), device=qs.device, dtype=qs.dtype)
    out_den = torch.empty_like(out_max)
    out_ns = torch.empty((*out_shape, dim), device=qs.device, dtype=qs.dtype)
    out_nv = torch.empty((*out_shape, dim, 3), device=qs.device, dtype=qs.dtype)
    block_d = triton.next_power_of_2(dim)
    block_k = triton.next_power_of_2(key_count)

    def grid(meta):
        del meta
        return (batch * groups * heads * 2,)

    # Inputs are contiguous by construction.  Passing strides explicitly keeps
    # this helper correct if a future caller removes that preparation step.
    _tile_forward_kernel[grid](
        qs, qv, ks, kv, vs, vv, rel, allowed,
        out_max, out_den, out_ns, out_nv,
        batch, groups, heads, key_count, dim,
        *qs.stride(), *qv.stride(), *ks.stride(), *kv.stride(), *vs.stride(), *vv.stride(), *rel.stride(), *allowed.stride(),
        SCALE_SCALAR=float(dim) ** -0.5,
        SCALE_VECTOR=float(3 * dim) ** -0.5,
        BLOCK_D=block_d, BLOCK_K=block_k, BLOCK_C=4,
    )
    return out_max, out_den, out_ns, out_nv


def _triton_backward(
    qs: Tensor,
    qv: Tensor,
    ks: Tensor,
    kv: Tensor,
    vs: Tensor,
    vv: Tensor,
    rel: Tensor,
    allowed: Tensor,
    block_max: Tensor,
    grad_denominator: Tensor,
    grad_scalar_numerator: Tensor,
    grad_vector_numerator: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    batch, heads, _, dim = (int(x) for x in qs.shape)
    groups, key_count = int(ks.shape[1]), int(ks.shape[3])
    scratch_qs = torch.empty((batch, groups, heads, 2, dim), device=qs.device, dtype=torch.float64)
    scratch_qv = torch.empty((batch, groups, heads, 2, dim, 3), device=qs.device, dtype=torch.float64)
    scratch_ks = torch.empty((batch, groups, heads, 2, key_count, dim), device=qs.device, dtype=torch.float64)
    scratch_kv = torch.empty((batch, groups, heads, 2, key_count, dim, 3), device=qs.device, dtype=torch.float64)
    scratch_vs = torch.empty_like(scratch_ks)
    scratch_vv = torch.empty_like(scratch_kv)
    grad_rel = torch.empty_like(rel)
    block_d = triton.next_power_of_2(dim)
    block_k = triton.next_power_of_2(key_count)

    def grid(meta):
        del meta
        return (batch * groups * heads * 2,)

    _tile_backward_kernel[grid](
        qs, qv, ks, kv, vs, vv, rel, allowed, block_max,
        grad_denominator, grad_scalar_numerator, grad_vector_numerator,
        scratch_qs, scratch_qv, scratch_ks, scratch_kv, scratch_vs, scratch_vv, grad_rel,
        batch, groups, heads, key_count, dim,
        SCALE_SCALAR=float(dim) ** -0.5,
        SCALE_VECTOR=float(3 * dim) ** -0.5,
        BLOCK_D=block_d, BLOCK_K=block_k, BLOCK_C=4,
    )
    grad_qs = scratch_qs.sum(dim=1)
    grad_qv = scratch_qv.sum(dim=1)
    grad_ks = scratch_ks.sum(dim=3)
    grad_kv = scratch_kv.sum(dim=3)
    grad_vs = scratch_vs.sum(dim=3)
    grad_vv = scratch_vv.sum(dim=3)
    return grad_qs, grad_qv, grad_ks, grad_kv, grad_vs, grad_vv, grad_rel


class _TileContextFunction(torch.autograd.Function):
    """Custom autograd wrapper with a deliberately detached local maximum."""

    @staticmethod
    def forward(
        ctx,
        qs: Tensor,
        qv: Tensor,
        ks: Tensor,
        kv: Tensor,
        vs: Tensor,
        vv: Tensor,
        rel: Tensor,
        allowed: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        use_triton = _can_use_triton((qs, qv, ks, kv, vs, vv, rel, allowed))
        if use_triton:
            # A supported CUDA/F64 invocation is deliberately strict: a
            # compiler/runtime error should be visible to the caller instead
            # of silently turning a production benchmark into a Torch run.
            outputs = _triton_forward(qs, qv, ks, kv, vs, vv, rel, allowed)
        else:
            outputs = _torch_tile_forward(qs, qv, ks, kv, vs, vv, rel, allowed)
        block_max, denominator, scalar_num, vector_num = outputs
        ctx.save_for_backward(qs, qv, ks, kv, vs, vv, rel, allowed, block_max.detach())
        ctx.use_triton = use_triton
        # The derivative of max cancels in a normalized softmax, and the
        # online merge intentionally treats each tile's max as a statistic.
        ctx.mark_non_differentiable(block_max)
        return block_max, denominator, scalar_num, vector_num

    @staticmethod
    def backward(ctx, grad_max, grad_denominator, grad_scalar_num, grad_vector_num):
        del grad_max  # max was marked non-differentiable by forward().
        qs, qv, ks, kv, vs, vv, rel, allowed, block_max = ctx.saved_tensors
        gd = torch.zeros_like(block_max) if grad_denominator is None else grad_denominator.contiguous()
        gns = torch.zeros_like(block_max.expand(*block_max.shape[:-1], qs.shape[-1])) if grad_scalar_num is None else grad_scalar_num.contiguous()
        gnv = torch.zeros_like(block_max.expand(*block_max.shape[:-1], qs.shape[-1], 3)) if grad_vector_num is None else grad_vector_num.contiguous()
        if ctx.use_triton:
            gradients = _triton_backward(
                qs, qv, ks, kv, vs, vv, rel, allowed,
                block_max, gd, gns, gnv,
            )
        else:
            gradients = _torch_tile_backward(
                qs, qv, ks, kv, vs, vv, rel, allowed,
                block_max, gd, gns, gnv,
            )
        gradients = tuple(g.to(dtype=x.dtype) for g, x in zip(gradients, (qs, qv, ks, kv, vs, vv, rel)))
        return (*gradients, None)


def tile_context(
    qs: Tensor,
    qv: Tensor,
    ks: Tensor,
    kv: Tensor,
    vs: Tensor,
    vv: Tensor,
    rel: Tensor,
    allowed: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return local LSE statistics for grouped scalar/vector attention tiles.

    The returned shapes are ``max, denom: [B,G,H,2,1]``, scalar numerator
    ``[B,G,H,2,D]``, and vector numerator ``[B,G,H,2,D,3]``.  Invalid edges
    contribute exactly zero, including when an entire query row is masked.
    ``max`` is intentionally non-differentiable; all other outputs participate
    in the analytic custom backward.
    """

    prepared = _validate_and_prepare(qs, qv, ks, kv, vs, vv, rel, allowed)
    return _TileContextFunction.apply(*prepared)


__all__ = ["tile_context"]

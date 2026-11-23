"""Append-only block storage for differentiable scalar/vector attention KV.

The cache owns only Python tuples of Tensor references.  A frame is copied into
a completed block once, when that block reaches capacity.  Reading an unfinished
block concatenates only its bounded pending entries; completed blocks are never
joined into a full-history tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, NamedTuple

from torch import Tensor
import torch


class KVFrame(NamedTuple):
    key_scalar: Tensor       # [B,H,2,D]
    key_vector: Tensor       # [B,H,2,D,3]
    value_scalar: Tensor     # [B,H,2,D]
    value_vector: Tensor     # [B,H,2,D,3]
    valid_mask: Tensor       # [B,2]


class KVBlock(NamedTuple):
    key_scalar: Tensor       # [B,H,2F,D]
    key_vector: Tensor       # [B,H,2F,D,3]
    value_scalar: Tensor     # [B,H,2F,D]
    value_vector: Tensor     # [B,H,2F,D,3]
    valid_mask: Tensor       # [B,2F]


FUSED_GROUP_BLOCKS = 8


def pack_blocks(blocks: tuple[KVBlock, ...]) -> KVBlock:
    """Stack equally sized tiles along their independent tile dimension."""
    return KVBlock(*(torch.stack(items, dim=1) for items in zip(*blocks)))


def _join_frames(frames: tuple[KVFrame, ...]) -> KVBlock:
    if not frames:
        raise ValueError("cannot materialize an empty KV block")
    return KVBlock(
        torch.cat(tuple(frame.key_scalar for frame in frames), dim=2),
        torch.cat(tuple(frame.key_vector for frame in frames), dim=2),
        torch.cat(tuple(frame.value_scalar for frame in frames), dim=2),
        torch.cat(tuple(frame.value_vector for frame in frames), dim=2),
        torch.cat(tuple(frame.valid_mask for frame in frames), dim=1),
    )


@dataclass(frozen=True)
class BlockwiseKVCache:
    """Immutable cache structure preserving every stored Tensor's autograd graph."""

    block_frames: int = 32
    completed_blocks: tuple[KVBlock, ...] = ()
    pending_entries: tuple[KVFrame, ...] = ()
    total_frames: int = 0
    packed_groups: tuple[KVBlock, ...] = ()
    partial_group: KVBlock | None = None

    def __post_init__(self) -> None:
        if self.block_frames <= 0:
            raise ValueError("block_frames must be positive")
        if len(self.pending_entries) >= self.block_frames:
            raise ValueError("pending_entries must be smaller than one block")
        if self.total_frames != len(self.completed_blocks) * self.block_frames + len(self.pending_entries):
            raise ValueError("KV cache frame count does not match its blocks")

    @property
    def total_tokens(self) -> int:
        return 2 * self.total_frames

    def append_frame(
        self,
        key_scalar: Tensor,
        key_vector: Tensor,
        value_scalar: Tensor,
        value_vector: Tensor,
        valid_mask: Tensor,
        *,
        pack_groups: bool = False,
    ) -> "BlockwiseKVCache":
        """Return an extended cache without modifying any existing Tensor."""
        if key_scalar.ndim != 4 or key_scalar.shape[2] != 2:
            raise ValueError("one frame key_scalar must have shape [B,H,2,D]")
        if value_scalar.shape != key_scalar.shape:
            raise ValueError("one frame value_scalar must match key_scalar")
        if key_vector.ndim != 5 or key_vector.shape[2] != 2:
            raise ValueError("one frame key_vector must have shape [B,H,2,D,3]")
        if value_vector.shape != key_vector.shape:
            raise ValueError("one frame value_vector must match key_vector")
        if tuple(valid_mask.shape) != (key_scalar.shape[0], 2):
            raise ValueError("one frame valid_mask must have shape [B,2]")
        frame = KVFrame(key_scalar, key_vector, value_scalar, value_vector, valid_mask)
        pending = self.pending_entries + (frame,)
        completed = self.completed_blocks
        if len(pending) == self.block_frames:
            # The only join of a completed block: this happens once at capacity.
            completed = completed + (_join_frames(pending),)
            pending = ()
        packed = self.packed_groups
        partial = self.partial_group
        if pack_groups and (completed is not self.completed_blocks or partial is None):
            first = len(packed) * FUSED_GROUP_BLOCKS
            while len(completed) - first >= FUSED_GROUP_BLOCKS:
                packed += (pack_blocks(completed[first:first + FUSED_GROUP_BLOCKS]),)
                first += FUSED_GROUP_BLOCKS
            partial = pack_blocks(completed[first:]) if first < len(completed) else None
        elif completed is not self.completed_blocks:
            partial = None
        return BlockwiseKVCache(
            block_frames=self.block_frames,
            completed_blocks=completed,
            pending_entries=pending,
            total_frames=self.total_frames + 1,
            packed_groups=packed,
            partial_group=partial,
        )

    def iter_blocks(self) -> Iterator[tuple[int, KVBlock]]:
        """Yield ``(first_frame_index, block)`` in strict history order."""
        offset = 0
        for block in self.completed_blocks:
            yield offset, block
            offset += self.block_frames
        if self.pending_entries:
            # At most block_frames-1 entries are joined for the current query.
            yield offset, _join_frames(self.pending_entries)

    def iter_packed_groups(self) -> Iterator[tuple[int, KVBlock]]:
        """Read complete immutable groups and materialize only bounded tails."""
        offset = 0
        for group in self.packed_groups:
            yield offset, group
            offset += FUSED_GROUP_BLOCKS * self.block_frames
        consumed = len(self.packed_groups) * FUSED_GROUP_BLOCKS
        if self.partial_group is not None:
            yield offset, self.partial_group
            offset += self.partial_group.valid_mask.shape[1] * self.block_frames
        else:
            for first in range(consumed, len(self.completed_blocks), FUSED_GROUP_BLOCKS):
                blocks = self.completed_blocks[first:first + FUSED_GROUP_BLOCKS]
                yield offset, pack_blocks(blocks)
                offset += len(blocks) * self.block_frames
        if self.pending_entries:
            yield offset, pack_blocks((_join_frames(self.pending_entries),))

    @classmethod
    def from_legacy(cls, cache: object, *, block_frames: int = 32) -> "BlockwiseKVCache":
        """Convert a legacy cache for focused parity tests and API compatibility."""
        required = ("key_scalar", "key_vector", "value_scalar", "value_vector", "valid_mask")
        if any(not hasattr(cache, name) for name in required):
            raise TypeError("cache is not a legacy projected KV cache")
        valid = cache.valid_mask
        if valid.ndim == 1:
            valid = valid.unsqueeze(0)
        if valid.shape[1] % 2:
            raise ValueError("legacy KV token count must contain whole two-token frames")
        result = cls(block_frames=block_frames)
        for frame in range(valid.shape[1] // 2):
            sl = slice(2 * frame, 2 * frame + 2)
            result = result.append_frame(
                cache.key_scalar[:, :, sl], cache.key_vector[:, :, sl],
                cache.value_scalar[:, :, sl], cache.value_vector[:, :, sl],
                valid[:, sl],
            )
        return result

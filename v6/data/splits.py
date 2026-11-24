"""Source-isolated dataset splits, block/gap detection, and causal checkpoint scheduling.

Implements the strict data hygiene contract shared by Model Plan V1 and Model Plan V2:
- Deterministic source_id partitioning of TRAIN into TRAIN-fit (80%), TRAIN-calib (10%),
  and DEV-select (10%), keeping historical VAL untouched for paired replication only.
- Automatic causal detection of K <= 6 observation blocks separated by 300 s silent gaps.
- Sub-arc A / B / C partitioning inside newly arrived post-gap blocks for V2 re-anchoring.
- Macro-checkpoint scheduling (+30 s, +150 s, +300 s per block) plus the pre-registered
  V2 common physical endpoint (Block 2 start + 30 s, evaluated even if held).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor


B0_ERROR_STRATA_KM: tuple[tuple[float, float, str], ...] = (
    (0.0, 1.0, "[0, 1) km"),
    (1.0, 10.0, "[1, 10) km"),
    (10.0, 100.0, "[10, 100) km"),
    (100.0, 1000.0, "[100, 1000) km"),
    (1000.0, 10000.0, "[1000, 10000) km"),
)

CADENCE_NAMES: tuple[str, ...] = ("dense", "stride5", "stride20", "pois10")


def source_hash_unit(source_id: str, seed: int = 20261028) -> float:
    """Map a source_id deterministically into [0, 1) independent of Python hash randomization."""
    payload = f"{int(seed)}::{source_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return float(value) / float(2**64)


def assign_source_split(
    source_id: str,
    *,
    seed: int = 20261028,
    fit_ratio: float = 0.80,
    calib_ratio: float = 0.10,
) -> str:
    """Assign a TRAIN source_id to 'train_fit', 'train_calib', or 'dev_select'."""
    if not (0.0 < fit_ratio < 1.0 and 0.0 <= calib_ratio < 1.0 and fit_ratio + calib_ratio < 1.0):
        raise ValueError("fit_ratio and calib_ratio must be positive and sum to < 1")
    u = source_hash_unit(str(source_id), seed=seed)
    if u < fit_ratio:
        return "train_fit"
    if u < fit_ratio + calib_ratio:
        return "train_calib"
    return "dev_select"


def partition_prepared_keys_by_source(
    rows: Mapping[str, Mapping[str, Any]],
    *,
    seed: int = 20261028,
    fit_ratio: float = 0.80,
    calib_ratio: float = 0.10,
) -> dict[str, list[str]]:
    """Partition Prepared record keys strictly by source_id so all 4 cadences stay together."""
    splits: dict[str, list[str]] = {
        "train_fit": [],
        "train_calib": [],
        "dev_select": [],
    }
    for key in sorted(rows):
        row = rows[key]
        record_key = row.get("record_key", {})
        source_id = str(record_key.get("source_id", row.get("source_id", key)))
        bucket = assign_source_split(
            source_id, seed=seed, fit_ratio=fit_ratio, calib_ratio=calib_ratio
        )
        splits[bucket].append(key)
    return splits


def b0_stratum_label(error_km: float) -> str:
    """Return the five-tier initial B0 position error stratum label."""
    if not np.isfinite(error_km) or error_km < 0.0:
        return "invalid"
    for low, high, label in B0_ERROR_STRATA_KM:
        if low <= error_km < high:
            return label
    return ">= 10000 km"


@dataclass(frozen=True)
class ObservationBlock:
    """Metadata for one contiguous observation block (typically ~300 s)."""

    block_index: int
    start_idx: int
    end_idx: int  # inclusive index in the valid observation array
    start_time_sec: float
    end_time_sec: float
    preceding_gap_sec: float
    subarc_a_end_idx: int  # inclusive end of fitting sub-arc A
    subarc_b_end_idx: int  # inclusive end of causal selection sub-arc B
    # Sub-arc C is (subarc_b_end_idx + 1 .. end_idx) for unbiased out-of-fit scoring.


@dataclass(frozen=True)
class MacroCheckpoint:
    """One causal publication/refinement checkpoint along a multi-block track."""

    checkpoint_index: int
    block_index: int
    target_offset_sec: float
    eval_time_sec: float
    last_obs_idx: int  # -1 if no post-init observation has arrived by eval_time_sec
    prev_last_obs_idx: int
    crossed_gap: bool
    is_block2_primary_endpoint: bool = False

    @property
    def has_new_observations(self) -> bool:
        return self.last_obs_idx > self.prev_last_obs_idx


def parse_observation_blocks(
    times_sec: Tensor | np.ndarray | Sequence[float],
    valid_mask: Tensor | np.ndarray | Sequence[bool] | None = None,
    *,
    init_time_sec: float = 0.0,
    gap_threshold_sec: float = 60.0,
) -> list[ObservationBlock]:
    """Segment a 1D observation timestamp sequence into contiguous blocks across silent gaps."""
    t_all = np.asarray(times_sec, dtype=np.float64).reshape(-1)
    if valid_mask is None:
        mask = np.ones_like(t_all, dtype=bool)
    else:
        mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
    valid_indices = np.flatnonzero(mask)
    if valid_indices.size == 0:
        return []

    blocks: list[ObservationBlock] = []
    seg_start_pos = 0
    prev_end_time = float(init_time_sec)
    block_idx = 0

    for pos in range(1, valid_indices.size + 1):
        is_split = pos == valid_indices.size
        if not is_split:
            dt = float(t_all[valid_indices[pos]] - t_all[valid_indices[pos - 1]])
            is_split = dt > float(gap_threshold_sec)
        if is_split:
            s_idx = int(valid_indices[seg_start_pos])
            e_idx = int(valid_indices[pos - 1])
            s_time = float(t_all[s_idx])
            e_time = float(t_all[e_idx])
            gap_sec = max(0.0, s_time - prev_end_time) if block_idx > 0 else max(0.0, s_time - float(init_time_sec))

            span_indices = valid_indices[seg_start_pos:pos]
            duration = max(0.0, e_time - s_time)
            if duration >= 30.0 and span_indices.size >= 3:
                # Partition block into Sub-arc A (first 40%), Sub-arc B (middle 30%), Sub-arc C (last 30%)
                t_a = s_time + 0.40 * duration
                t_b = s_time + 0.70 * duration
                a_pos = int(np.searchsorted(t_all[span_indices], t_a, side="right")) - 1
                b_pos = int(np.searchsorted(t_all[span_indices], t_b, side="right")) - 1
                a_pos = max(0, min(a_pos, span_indices.size - 3))
                b_pos = max(a_pos + 1, min(b_pos, span_indices.size - 2))
                sub_a_end = int(span_indices[a_pos])
                sub_b_end = int(span_indices[b_pos])
            else:
                mid = max(0, span_indices.size // 2 - 1)
                sub_a_end = int(span_indices[mid])
                sub_b_end = int(span_indices[min(span_indices.size - 1, mid + 1)])

            blocks.append(
                ObservationBlock(
                    block_index=block_idx,
                    start_idx=s_idx,
                    end_idx=e_idx,
                    start_time_sec=s_time,
                    end_time_sec=e_time,
                    preceding_gap_sec=gap_sec,
                    subarc_a_end_idx=sub_a_end,
                    subarc_b_end_idx=sub_b_end,
                )
            )
            prev_end_time = e_time
            seg_start_pos = pos
            block_idx += 1
    return blocks


def build_macro_checkpoint_schedule(
    times_sec: Tensor | np.ndarray | Sequence[float],
    valid_mask: Tensor | np.ndarray | Sequence[bool] | None = None,
    *,
    init_time_sec: float = 0.0,
    iod_last_time_sec: float | None = None,
    block_offsets_sec: Sequence[float] = (30.0, 150.0, 300.0),
    gap_threshold_sec: float = 60.0,
    nominal_block_sec: float = 300.0,
    nominal_gap_sec: float = 300.0,
    include_block2_primary_endpoint: bool = True,
    block_start_time_sec: float | None = None,
) -> tuple[list[MacroCheckpoint], list[ObservationBlock]]:
    """Build causal macro-checkpoints (+30s, +150s, +300s per block, <=18 total).

    Strictly enforces:
    1. No checkpoint earlier than max(init_time_sec, iod_last_time_sec).
    2. Common physical endpoint at Block 2 start + 30 s is always scheduled even if
       no new observation has arrived in [block2_start, block2_start + 30 s], so the
       held pure two-body propagation is scored without trigger-time selection bias.
    """
    t_all = np.asarray(times_sec, dtype=np.float64).reshape(-1)
    if valid_mask is None:
        mask = np.ones_like(t_all, dtype=bool)
    else:
        mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
    valid_indices = np.flatnonzero(mask)
    valid_times = t_all[valid_indices]

    earliest_allowed = float(init_time_sec)
    if iod_last_time_sec is not None:
        earliest_allowed = max(earliest_allowed, float(iod_last_time_sec))

    blocks = parse_observation_blocks(
        t_all, mask, init_time_sec=init_time_sec, gap_threshold_sec=gap_threshold_sec
    )
    if not blocks:
        return [], []

    # Events are published on their physical clock. Appending an observation
    # never snaps a previously scheduled correction back to the last exposure.
    until = float(valid_times[-1])
    raw = {}
    if block_start_time_sec is not None:
        start = float(block_start_time_sec)
        period = nominal_block_sec + nominal_gap_sec
        starts = [(i, start+i*period, until) for i in range(max(0, int((until-start)//period)+1))]
    else:
        starts = [(b.block_index, b.start_time_sec,
                   blocks[i+1].start_time_sec if i+1 < len(blocks) else until)
                  for i,b in enumerate(blocks)]
    for bi, start, stop in starts:
        for offset in block_offsets_sec:
            at = start + float(offset)
            if earliest_allowed <= at <= min(until, stop):
                raw[at] = (bi, float(offset), False)
    endpoint = float(init_time_sec if block_start_time_sec is None else block_start_time_sec) + nominal_block_sec + nominal_gap_sec + 30.
    if include_block2_primary_endpoint and earliest_allowed <= endpoint <= until:
        raw[endpoint] = (1, 30., True)
    checkpoints = []
    previous, previous_block = -1, -1
    for at,(bi,offset,endpoint_flag) in sorted(raw.items()):
        pos = int(np.searchsorted(valid_times, at, side="right")) - 1
        last = int(valid_indices[pos]) if pos >= 0 else -1
        if last <= previous and not endpoint_flag:
            continue
        checkpoints.append(MacroCheckpoint(len(checkpoints), bi, offset, at, last, previous,
                                           previous_block >= 0 and bi > previous_block, endpoint_flag))
        previous, previous_block = last, bi
    return checkpoints, blocks


__all__ = [
    "B0_ERROR_STRATA_KM",
    "CADENCE_NAMES",
    "MacroCheckpoint",
    "ObservationBlock",
    "assign_source_split",
    "b0_stratum_label",
    "build_macro_checkpoint_schedule",
    "parse_observation_blocks",
    "partition_prepared_keys_by_source",
    "source_hash_unit",
]

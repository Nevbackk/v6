"""Source-isolated partitions and initial error strata."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np


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


__all__ = [
    "B0_ERROR_STRATA_KM",
    "CADENCE_NAMES",
    "assign_source_split",
    "b0_stratum_label",
    "partition_prepared_keys_by_source",
    "source_hash_unit",
]

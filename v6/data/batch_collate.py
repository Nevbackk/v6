"""Collate complete variable-length filter records without truncation."""
from __future__ import annotations
from typing import Any, Mapping, Sequence
import torch
from torch.nn.utils.rnn import pad_sequence
from ..models.batch_time import prepare_batch_time_plan


def collate_filter_sequences(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pad complete records and precompute CPU epochs/substeps for the filter."""
    if not records:
        raise ValueError("records must be nonempty")
    initial, init_times, times, observations, stations = [], [], [], [], []
    valid, ids, reliability = [], [], []
    for record in records:
        t = torch.as_tensor(record["times_sec"], dtype=torch.float64, device="cpu")
        if t.ndim != 1 or t.numel() == 0:
            raise ValueError("each times_sec must be a nonempty one-dimensional sequence")
        length = t.numel()
        los = torch.as_tensor(record["observed_los"], dtype=torch.float64, device="cpu")
        station = torch.as_tensor(record["station_states_si"], dtype=torch.float64, device="cpu")
        b0 = torch.as_tensor(record["initial_state_si"], dtype=torch.float64, device="cpu")
        if tuple(los.shape) != (length, 3) or tuple(station.shape) != (length, 6) or tuple(b0.shape) != (6,):
            raise ValueError("record dimensions must be [T,3], [T,6], [6]")
        initial.append(b0)
        init_times.append(torch.as_tensor(record["init_time_sec"], dtype=torch.float64, device="cpu").reshape(()))
        times.append(t)
        observations.append(los)
        stations.append(station)
        mask = torch.as_tensor(record.get("valid_mask", torch.ones(length, dtype=torch.bool)), dtype=torch.bool, device="cpu")
        station_id = torch.as_tensor(record.get("station_ids", torch.zeros(length, dtype=torch.int64)), dtype=torch.int64, device="cpu")
        known = torch.as_tensor(record.get("station_id_reliable", torch.zeros(length, dtype=torch.bool)), dtype=torch.bool, device="cpu")
        if tuple(mask.shape) != (length,) or tuple(station_id.shape) != (length,) or tuple(known.shape) != (length,):
            raise ValueError("validity and station identifiers must have length T")
        if "station_ids" not in record and bool(known.any()):
            raise ValueError("reliable station identities require station_ids")
        if not bool(torch.isfinite(b0).all() and torch.isfinite(los[mask]).all()
                    and torch.isfinite(station[mask]).all()):
            raise ValueError("initial states and active observation inputs must be finite")
        if bool((torch.linalg.vector_norm(los[mask], dim=-1) <= 1e-15).any()):
            raise ValueError("active LOS vectors must be nonzero")
        valid.append(mask)
        ids.append(station_id)
        reliability.append(known)
    result = {
        "initial_state_si": torch.stack(initial),
        "init_time_sec": torch.stack(init_times),
        "times_sec": pad_sequence(times, batch_first=True),
        "observed_los": pad_sequence(observations, batch_first=True),
        "station_states_si": pad_sequence(stations, batch_first=True),
        "valid_mask": pad_sequence(valid, batch_first=True),
        "station_ids": pad_sequence(ids, batch_first=True),
        "station_id_reliable": pad_sequence(reliability, batch_first=True),
    }
    result["time_plan"] = prepare_batch_time_plan(
        result["times_sec"], result["init_time_sec"], result["valid_mask"]
    )
    return result

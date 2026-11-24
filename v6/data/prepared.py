"""Read the external frozen G4 prepared snapshot."""
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from functools import lru_cache

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from .batch_collate import collate_filter_sequences

IDENTITY = ("source_id", "rhythm", "record_id")


def identity(row):
    return tuple(str(row[name]) for name in IDENTITY)


class Prepared:
    def __init__(self, root, protocol_dir, split):
        self.root = root
        self.split = split
        config = json.loads((protocol_dir / "config.json").read_text())
        spec = config["source_filter"][split]
        allowlist = protocol_dir / str(spec["allowlist_path"]).split("/")[-1]
        raw = allowlist.read_bytes()
        if hashlib.sha256(raw).hexdigest() != spec["allowlist_sha256"]:
            raise ValueError(f"Allowlist checksum mismatch: {allowlist}")
        allowed = {}
        for row in csv.DictReader(raw.decode().splitlines()):
            source = (int(row["shard_id"]), str(row["source_id"]))
            error = float(row["b0_position_error_km"])
            if row["split"] != split or not 0 <= error < 10000 or source in allowed:
                raise ValueError("Invalid source allowlist")
            allowed[source] = error
        keys = set(config[f"fixed_{split}_record_keys"])
        self.rows = {}
        self.groups = defaultdict(list)
        initial = {}
        for path in sorted((root / "initialization_cache" / split).glob("shard_*.jsonl")):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                key = row["cache_key"]
                if key not in keys:
                    continue
                sid = int(row["shard_id"])
                source = sid, identity(row["record_key"])[0]
                b0 = float(row["initial_epoch_sec"]), tuple(row["initial_state"])
                if (source not in allowed or not row["shared_initialization"]
                        or b0 != (float(row["dense_t_init"]), tuple(row["B0"]))
                        or initial.setdefault(source, b0) != b0 or key in self.rows):
                    raise ValueError(f"Invalid B0 or record identity: {key}")
                self.rows[key] = row
                self.groups[sid].append(key)
        if (set(self.rows) != keys or set(initial) != set(allowed)
                or len(keys) != config[f"{split}_records"]):
            raise ValueError(f"Incomplete frozen {split} snapshot")
        self._initial_truth = {}
        self._initial_epoch = {source: b0[0] for source, b0 in initial.items()}
        self._initial_station = {}
        self.shard = lru_cache(maxsize=16)(self._load_shard)

    def _load_shard(self, sid):
        path = self.root / "prepared_cache" / self.split / "shards" / f"shard_{sid:06d}.pt"
        records = torch.load(path, map_location="cpu", weights_only=False)["records"]
        indexed = {identity(row): row for row in records}
        for row in records:
            source, rhythm, _ = identity(row)
            key = (sid, source)
            if key not in self._initial_epoch or not len(row["t"]):
                continue
            epoch = self._initial_epoch[key]
            raw_times = np.asarray(row["t"], dtype=np.float64)
            i = int(np.searchsorted(raw_times, epoch, side="right")) - 1
            if i < 0:
                continue
            sample_time = float(raw_times[i])
            # All cadence variants share the same initialization. Use the latest
            # ground truth already available at that epoch for the clean control.
            if key not in self._initial_truth or sample_time > self._initial_truth[key][0]:
                self._initial_truth[key] = (
                    sample_time,
                    np.asarray(row["tgt_state"][i], dtype=np.float64).copy(),
                )
            if rhythm == "dense":
                station = np.asarray(row["sat_state"][i], dtype=np.float64).copy()
                station[:3] += station[3:] * (epoch - sample_time)
                self._initial_station[key] = station
        if len(indexed) != len(records):
            raise ValueError(f"Duplicate records in {path}")
        return indexed

    def length(self, key):
        return int(self.rows[key]["filter_observation_count"])

    def record(self, key, *, exact_initial=False):
        init = self.rows[key]
        row = self.shard(int(init["shard_id"]))[identity(init["record_key"])]
        t = np.asarray(row["t"], dtype=np.float64)
        epoch = float(init["initial_epoch_sec"])
        start = int(init["initial_observation_index"]) + 1
        if (start != np.searchsorted(t, epoch, side="right")
                or len(t) - start != self.length(key)
                or not np.isfinite(t).all() or (np.diff(t) <= 0).any()):
            raise ValueError(f"Invalid observation times for {key}")
        source = (int(init["shard_id"]), identity(init["record_key"])[0])
        initial_state = init["initial_state"]
        if exact_initial:
            if "initial_truth_state_si" in init:
                initial_state = init["initial_truth_state_si"]
            elif source in self._initial_truth:
                from ..models.reprojection import propagate_si_with_stm

                sample_time, state = self._initial_truth[source]
                initial_state = propagate_si_with_stm(
                    torch.as_tensor(state), epoch - sample_time, compute_stm=False
                )[0].numpy()
            else:
                raise ValueError(
                    f"V1 needs a known six-dimensional truth state at or before initialization: {key}"
                )
            initial_state = np.asarray(initial_state, dtype=np.float64)
            if initial_state.shape != (6,) or not np.isfinite(initial_state).all():
                raise ValueError(f"Invalid known initial state: {key}")
        return dict(
            initial_state_si=initial_state,
            init_time_sec=epoch,
            initialization="known_state" if exact_initial else "gauss_ls",
            initial_station_si=self._initial_station.get(source),
            times_sec=t[start:],
            observed_los=np.asarray(row["los"])[start:],
            station_states_si=np.asarray(row["sat_state"])[start:],
            station_ids=np.asarray(row["sat_id"])[start:],
            station_id_reliable=np.ones(len(t) - start, dtype=bool),
            truth_states_si=np.asarray(row["tgt_state"])[start:],
        )

    def collate(self, keys):
        records = [self.record(key) for key in keys]
        batch = collate_filter_sequences(records)
        batch["truth_states_si"] = pad_sequence(
            [torch.as_tensor(row["truth_states_si"], dtype=torch.float64) for row in records],
            batch_first=True)
        return batch

    def batches(self, epoch, seed, size):
        rng = random.Random(seed + 1009 * epoch)
        shard_ids = list(self.groups)
        rng.shuffle(shard_ids)
        carry = []
        for start in range(0, len(shard_ids), 16):
            buckets = defaultdict(list)
            for sid in shard_ids[start:start + 16]:
                for key in self.groups[sid]:
                    buckets[int(math.log2(max(1, self.length(key))))].append(key)
            local, tails = [], []
            for bucket in sorted(buckets):
                keys = buckets[bucket]
                rng.shuffle(keys)
                full = len(keys) // size * size
                local.extend(keys[i:i + size] for i in range(0, full, size))
                tails.extend(keys[full:])
            tails.sort(key=lambda key: (self.length(key), key))
            local.extend(tails[i:i + size] for i in range(0, len(tails), size))
            rng.shuffle(local)
            for keys in local:
                if len(keys) == size:
                    yield keys
                else:
                    carry.extend(keys)
                    if len(carry) >= size:
                        yield carry[:size]
                        carry = carry[size:]
        if carry:
            yield carry

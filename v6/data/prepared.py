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
        self._dense_start = {}
        self._initial_epoch = {source: b0[0] for source, b0 in initial.items()}
        self._initial_station = {}
        self.shard = lru_cache(maxsize=16)(self._load_shard)

    def _load_shard(self, sid):
        path = self.root / "prepared_cache" / self.split / "shards" / f"shard_{sid:06d}.pt"
        records = torch.load(path, map_location="cpu", weights_only=False)["records"]
        indexed = {identity(row): row for row in records}
        for row in records:
            source, rhythm, _ = identity(row)
            if rhythm == "dense" and len(row["t"]):
                self._dense_start[(sid, source)] = float(row["t"][0])
                key = (sid, source)
                if key in self._initial_epoch:
                    epoch = self._initial_epoch[key]
                    i = np.searchsorted(row["t"], epoch, side="right") - 1
                    if i >= 0:
                        station = np.array(row["sat_state"][i], dtype=np.float64, copy=True)
                        station[:3] += station[3:] * (epoch - float(row["t"][i]))
                        self._initial_station[key] = station
        if len(indexed) != len(records):
            raise ValueError(f"Duplicate records in {path}")
        return indexed

    def length(self, key):
        return int(self.rows[key]["filter_observation_count"])

    def record(self, key):
        init = self.rows[key]
        row = self.shard(int(init["shard_id"]))[identity(init["record_key"])]
        t = np.asarray(row["t"], dtype=np.float64)
        epoch = float(init["initial_epoch_sec"])
        start = int(init["initial_observation_index"]) + 1
        if (start != np.searchsorted(t, epoch, side="right")
                or len(t) - start != self.length(key)
                or not np.isfinite(t).all() or (np.diff(t) <= 0).any()):
            raise ValueError(f"Invalid observation times for {key}")
        # Raw dense acquisition and all cadence variants of a source share one clock.
        # A declared physical block opening takes precedence; older records fall back
        # to the dense opening, or the shared initialization epoch if dense is absent.
        start_time = row.get("block_start_time_sec", self._dense_start.get(
            (int(init["shard_id"]), identity(init["record_key"])[0]), epoch))
        return dict(initial_state_si=init["initial_state"], init_time_sec=epoch,
                    initial_station_si=self._initial_station.get((int(init["shard_id"]),
                                                                  identity(init["record_key"])[0])),
                    block_start_time_sec=float(start_time),
                    times_sec=t[start:], observed_los=np.asarray(row["los"])[start:],
                    station_states_si=np.asarray(row["sat_state"])[start:],
                    station_ids=np.asarray(row["sat_id"])[start:],
                    station_id_reliable=np.ones(len(t) - start, dtype=bool),
                    truth_states_si=np.asarray(row["tgt_state"])[start:])

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

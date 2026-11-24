"""Checkpoint state shared by the orbit training and validation entry points."""
from __future__ import annotations

import json
import random
import sys

import numpy as np
import torch


def read_checkpoint(args, parser, *, evaluation, config_keys, eval_overrides=(), require_weights=True):
    filename = args.resume or (args.run_dir / ("best.pt" if evaluation else "last.pt"))
    if not filename.is_file():
        if args.resume or (evaluation and require_weights):
            parser.error(f"Checkpoint does not exist: {filename}")
        return {}
    ckpt = torch.load(filename, map_location="cpu", weights_only=True)
    if ckpt.get("recursion") != "valid_frame":
        parser.error(
            "This checkpoint predates the per-frame model and selection metric; use a new run directory"
        )
    if hasattr(args, "version") and ckpt.get("version") != args.version:
        parser.error(f"Expected a {args.version} checkpoint for this initialization protocol")
    saved = ckpt.get("config", ckpt)
    explicit = {token.partition("=")[0] for token in sys.argv[1:]}
    for action in parser._actions:
        key = action.dest
        if key not in config_keys or key not in saved:
            continue
        if explicit.isdisjoint(action.option_strings):
            setattr(args, key, saved[key])
        elif getattr(args, key) != saved[key] and not (evaluation and key in eval_overrides):
            parser.error(f"Checkpoint {key}={saved[key]!r}, requested {getattr(args, key)!r}")
    if "split_seed" not in saved and args.split_seed != parser.get_default("split_seed"):
        parser.error("Legacy checkpoint does not identify a nondefault source split")
    return ckpt


def restore_training_state(ckpt, optimizer, run_dir, score_fn):
    if not ckpt:
        return float("inf")
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if "rng" in ckpt:
        rng = ckpt["rng"]
        random.setstate(rng["python"])
        alg, keys, pos, cached, last = rng["numpy"]
        np.random.set_state((alg, np.asarray(keys, dtype=np.uint32), pos, cached, last))
        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_initialized() and rng["cuda"]:
            torch.cuda.set_rng_state_all(rng["cuda"])
    if "best_score" in ckpt:
        return ckpt["best_score"]
    # Earlier versions stored the selected epoch and its metric in separate files.
    previous = run_dir / "best.pt"
    if previous.exists():
        epoch = torch.load(previous, map_location="cpu", weights_only=True)["epoch"]
        report = run_dir / f"dev_select_{epoch:02d}.json"
        if not report.is_file():
            raise ValueError(f"Cannot recover the selected validation metric: {report}")
        return score_fn(json.loads(report.read_text(encoding="utf-8")))
    return float("inf")


def save_epoch(path, last_path, model, optimizer, args, config_keys, epoch, score, best_score, **extra):
    alg, keys, pos, cached, last = np.random.get_state()
    rng = {"python": random.getstate(), "numpy": (alg, keys.tolist(), pos, cached, last),
           "torch": torch.get_rng_state(),
           "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}
    config = {key: getattr(args, key) for key in config_keys}
    payload = {
        "recursion": "valid_frame",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_score": min(score, best_score),
        "config": config,
        "rng": rng,
        **config,
        **extra,
    }
    destinations = [path]
    if score < best_score or not (path.parent / "best.pt").is_file():
        destinations.append(path.parent / "best.pt")
    # last.pt is the resume pointer. Publish it only after all epoch outputs exist.
    destinations.append(last_path)
    for destination in destinations:
        tmp = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(destination)
    return payload["best_score"]

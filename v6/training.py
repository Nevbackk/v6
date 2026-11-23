"""Shared full-history, frame-weighted training and recursive validation."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from .data.prepared import Prepared
from .models import GeometryConditionedFilter

FIELDS = ("initial_state_si", "init_time_sec", "times_sec", "observed_los",
          "station_states_si", "station_ids", "station_id_reliable", "valid_mask", "time_plan")


def model_for(mode, backend, device):
    model = GeometryConditionedFilter(
        stats_scales=[.01, .01, .01, .1], scalar_channels=32, vector_channels=16,
        num_heads=4, attention_fusion="unified", attention_impl=backend, block_frames=32,
        no_historical_state_keys=True, projection_dtype=torch.float64, update_mode=mode)
    head = model.gain_head if mode == "gain" else model.correction_head
    torch.nn.init.zeros_(head.net[-1].weight)
    torch.nn.init.zeros_(head.net[-1].bias)
    return model.to(device)


def rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def predict(model, batch, device):
    inputs = {name: batch[name].to(device) if isinstance(batch[name], torch.Tensor)
              else batch[name] for name in FIELDS}
    return model.forward_filter_sequence(**inputs, mode="train_fast", output_mode="training_minimal")


def errors(out, batch):
    predicted = out.posterior_states_si
    truth = batch["truth_states_si"].to(predicted.device)
    scale = predicted.new_tensor([1e6] * 3 + [1e3] * 3)
    return (predicted[out.valid_mask] / scale - truth[out.valid_mask] / scale).square()


def save(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.no_grad()
def validate(model, data, device, size):
    model.eval()
    sums = torch.zeros(6, dtype=torch.float64, device=device)
    frames = 0
    failures = 0
    for sid in sorted(data.groups):
        ordered = sorted(data.groups[sid], key=lambda key: (data.length(key), key))
        for start in range(0, len(ordered), size):
            batch = data.collate(ordered[start:start + size])
            out = predict(model, batch, device)
            square = errors(out, batch)
            finite = torch.isfinite(square).all(dim=-1)
            failures += int((~finite).sum())
            sums += square[finite].sum(dim=0)
            frames += int(finite.sum())
    if frames == 0:
        raise FloatingPointError("No finite validation frames")
    position, velocity = sums[:3].sum().item() / frames, sums[3:].sum().item() / frames
    return dict(records=len(data.rows), finite_frames=frames, nonfinite_frames=failures,
                scaled_mse=(position + velocity) / 6, position_km=position**.5 * 1000,
                velocity_mps=velocity**.5 * 1000)


def train_batch(model, data, keys, optimizer, device, microbatch, fused):
    ordered = sorted(keys, key=data.length, reverse=True)
    frames = sum(map(data.length, ordered))
    start = 0
    loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    while start < len(ordered):
        top = data.length(ordered[start])
        size = min(microbatch, max(1, (32 * 256**2) // top**2)) if fused else microbatch
        batch = data.collate(ordered[start:start + size])
        value = errors(predict(model, batch, device), batch).sum() / (6 * frames)
        if not torch.isfinite(value):
            raise FloatingPointError(f"Nonfinite loss at {ordered[start]}")
        value.backward()
        loss += value.detach().item()
        start += size
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return loss, norm.item()


def initialize_direct(direct, gain_state):
    state = direct.state_dict()
    for name, value in gain_state.items():
        if name in ("gain_head.net.2.weight", "gain_head.net.2.bias"):
            if torch.count_nonzero(value):
                raise ValueError("Paired Gain initialization must have zero final layer")
            continue
        state[name.replace("gain_head.", "correction_head.", 1)] = value
    direct.load_state_dict(state, strict=True)


def main(mode, evaluation=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--microbatch", type=int)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--gain-step-zero", type=Path)
    parser.add_argument("--backend", choices=("blockwise_checkpoint", "fused_tiles_save"))
    args = parser.parse_args()
    device = torch.device(args.device)
    backend = args.backend or ("fused_tiles_save" if mode == "direct_correction"
                               and device.type == "cuda" and not evaluation else "blockwise_checkpoint")
    microbatch = args.microbatch or (32 if backend == "fused_tiles_save" else 8)
    if min(args.batch_size, microbatch, args.val_batch_size, args.epochs) < 1 or args.lr <= 0:
        parser.error("Batch sizes, epoch count and learning rate must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = model_for(mode, backend, device)
    val_data = Prepared(args.prepared_root, args.protocol_dir, "val")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    last = args.run_dir / "last.pt"
    source = args.resume or (last if last.exists() else None)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False) if source else None
    if checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
    if evaluation:
        if checkpoint is None:
            parser.error("Validation requires --resume or an existing run checkpoint")
        print(json.dumps(validate(model, val_data, device, args.val_batch_size)))
        return
    train_data = Prepared(args.prepared_root, args.protocol_dir, "train")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=.0001)
    protocol = dict(mode=mode, seed=args.seed, batch_size=args.batch_size, lr=args.lr,
                    snapshot=hashlib.sha256((args.protocol_dir / "config.json").read_bytes()).hexdigest())
    step, first_epoch, cursor = 0, 1, 0
    if checkpoint:
        if checkpoint.get("protocol") != protocol:
            raise ValueError("Resume does not match this release's training protocol")
        optimizer.load_state_dict(checkpoint["optimizer"])
        restore(checkpoint["rng"])
        step, first_epoch, cursor = (checkpoint[key] for key in ("global_step", "epoch", "batch_cursor"))
        if checkpoint["epoch_complete"]:
            first_epoch += 1
            cursor = 0
    elif mode == "direct_correction":
        if args.gain_step_zero:
            paired = torch.load(args.gain_step_zero, map_location="cpu", weights_only=False)
            if paired.get("global_step") != 0:
                raise ValueError("--gain-step-zero requires a step-0 checkpoint")
            state = paired["model"]
        else:
            torch.manual_seed(args.seed)
            state = model_for("gain", "blockwise_checkpoint", "cpu").state_dict()
        initialize_direct(model, state)

    def payload(epoch, cursor, complete):
        return dict(model=model.state_dict(), optimizer=optimizer.state_dict(), rng=rng(),
                    global_step=step, epoch=epoch, batch_cursor=cursor,
                    epoch_complete=complete, protocol=protocol)

    if checkpoint is None:
        save(args.run_dir / "step_0.pt", payload(1, 0, False))
    for epoch in range(first_epoch, args.epochs + 1):
        model.train()
        batches = list(train_data.batches(epoch, args.seed, args.batch_size))
        for index, keys in enumerate(batches, 1):
            if index <= cursor:
                continue
            loss, grad = train_batch(model, train_data, keys, optimizer, device,
                                     microbatch, backend == "fused_tiles_save")
            step += 1
            save(last, payload(epoch, index, False))
            print(json.dumps(dict(epoch=epoch, batch=index, global_step=step,
                                  train_loss=loss, gradient_norm=grad)), flush=True)
        scores = validate(model, val_data, device, args.val_batch_size)
        (args.run_dir / f"val_{epoch:02d}.json").write_text(json.dumps(scores, indent=2) + "\n")
        save(args.run_dir / f"epoch_{epoch:02d}.pt", payload(epoch, len(batches), True))
        save(last, payload(epoch, len(batches), True))
        cursor = 0

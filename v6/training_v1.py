"""Training and stratified validation pipeline for Model Plan V1 (LTR-LM, JF-BR, CL-*, FD-ASF).

Supports:
1. Classical zero-parameter controls: cl_gn1_pure, cl_gn1_matched, cl_window_lm, cl_full_prefix_wls.
2. 3x2 robust anchor loss suite ('scaled_mse', 'huber', 'log_charbonnier') on LTR-LM & JF-BR.
3. Zero-BPTT offline proposal training for Flagship FD-ASF (fitting forecast-disagreement q_k*
   or standard posterior pi_k for Posterior-CNF ablation, plus optional geometry DSM).
4. Source-isolated TRAIN-fit / TRAIN-calib / DEV-select splitting and paired VAL replication.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .data.prepared import Prepared
from .checkpoints import read_checkpoint, restore_training_state, save_epoch
from .data.splits import partition_prepared_keys_by_source
from .eval_metrics import (
    RecordEvalRow,
    compute_los_local_errors,
    evaluate_v2_primary_endpoint_forecast300,
    summarize_record_rows,
)
from .models.batched_physics import state_scale
from .models.reprojection import propagate_si_with_stm
from .models.v1_fd_asf import (
    FDASFSequentialFilter,
    proposal_weighted_kl_and_dsm_loss,
)
from .models.v1_ltr_lm import (
    ClassicalLMRefiner,
    JFBRRefiner,
    LTRLMRefiner,
    V1OnlineFilter,
)


V1_LOSS_TYPES = ("scaled_mse", "huber", "log_charbonnier")


def state_regression_loss(
    pred_state_si: Tensor,
    truth_state_si: Tensor,
    *,
    loss_type: str = "log_charbonnier",
    huber_delta: float = 0.05,
    charbonnier_eps: float = 1.0e-3,
) -> Tensor:
    """Compute 6D scaled state loss under 'scaled_mse', 'huber', or 'log_charbonnier'."""
    if loss_type not in V1_LOSS_TYPES:
        raise ValueError(f"loss_type must be one of {V1_LOSS_TYPES}, got {loss_type!r}")
    scale = state_scale(device=pred_state_si.device, dtype=pred_state_si.dtype)
    diff = (pred_state_si - truth_state_si) / scale
    if loss_type == "scaled_mse":
        return diff.square().mean()
    if loss_type == "huber":
        return torch.nn.functional.huber_loss(diff, torch.zeros_like(diff), delta=float(huber_delta))
    eps2 = float(charbonnier_eps) ** 2
    return 0.5 * torch.log1p(diff.square() / eps2).mean()


def build_v1_model(
    arch: str,
    *,
    writeback_mode: str = "joint_6d",
    jfbr_mode: str = "homog_6d",
    proposal_mode: str = "learned_mix",
    num_particles: int = 12,
    device: torch.device | str = "cpu",
) -> V1OnlineFilter | FDASFSequentialFilter:
    """Factory for all Model Plan V1 baselines, learned refiners, and Flagship FD-ASF."""
    dev = torch.device(device)
    arch_key = arch.strip().lower()
    if arch_key in ("cl_gn1_pure", "cl_gn1_matched", "cl_window_lm", "cl_full_prefix_wls"):
        ref = ClassicalLMRefiner(variant=arch_key, writeback_mode=writeback_mode)
        return V1OnlineFilter(ref).to(dev)
    if arch_key == "jf_br":
        ref = JFBRRefiner(mode=jfbr_mode)
        return V1OnlineFilter(ref).to(dev)
    if arch_key == "ltr_lm":
        ref = LTRLMRefiner(writeback_mode=writeback_mode)
        return V1OnlineFilter(ref).to(dev)
    if arch_key in ("fd_asf", "posterior_cnf", "sobol_lambert_lm"):
        pmode = "sobol_nonlearned" if arch_key == "sobol_lambert_lm" else proposal_mode
        return FDASFSequentialFilter(num_particles=num_particles, proposal_mode=pmode).to(dev)
    raise ValueError(f"Unknown V1 architecture: {arch!r}")


def train_v1_refiner_step_on_record(
    model: V1OnlineFilter,
    record: Mapping[str, Any],
    *,
    loss_type: str = "log_charbonnier",
    device: torch.device | str = "cpu",
) -> Tensor:
    """Trainable causal checkpoint anchor step on one record for LTR-LM or JF-BR."""
    dev = torch.device(device)
    dtype = torch.float64
    x0 = torch.as_tensor(record["initial_state_si"], device=dev, dtype=dtype)
    t_init = float(record["init_time_sec"])
    times = torch.as_tensor(record["times_sec"], dtype=dtype)
    obs = torch.as_tensor(record["observed_los"], device=dev, dtype=dtype)
    stn = torch.as_tensor(record["station_states_si"], device=dev, dtype=dtype)
    truth = torch.as_tensor(record["truth_states_si"], device=dev, dtype=dtype)
    valid = torch.as_tensor(
        record.get("valid_mask", torch.ones(times.shape[0], dtype=torch.bool)), dtype=torch.bool
    )

    rollout = model.rollout_track(
        initial_state_si=x0, init_time_sec=t_init, times_sec=times, observed_los=obs,
        station_states_si=stn, valid_mask=valid, block_start_time_sec=record.get("block_start_time_sec"),
        iod_last_time_sec=record.get("iod_last_time_sec"), station_ids=record.get("station_ids"),
        station_id_reliable=record.get("station_id_reliable"), detach_feedback=True)
    losses = []
    for cp, published in rollout["checkpoint_states"]:
        k = cp.last_obs_idx
        target, _ = propagate_si_with_stm(truth[k], cp.eval_time_sec - float(times[k]), compute_stm=False)
        losses.append(state_regression_loss(published, target, loss_type=loss_type))
    if not losses:
        return x0.new_zeros((), requires_grad=True)
    return torch.stack(losses).mean()


def train_fd_asf_proposal_step_on_record(
    model: FDASFSequentialFilter,
    record: Mapping[str, Any],
    *,
    use_forecast_disagreement: bool = True,
    dsm_weight: float = 0.05,
    num_ref_candidates: int = 24,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Zero-BPTT offline amortized proposal training step for FD-ASF (or Posterior-CNF when False)."""
    from .models.orbit_proposals import EmptyTeacherSupport, draw_chart_teacher
    try:
        z, branch, _, w_star, w_pi, context = draw_chart_teacher(
            model.tracklet_encoder, None, record, num_ref_candidates=num_ref_candidates,
            device=device, los_budget=16)
    except EmptyTeacherSupport:
        return (next(model.parameters()) * 0.0).sum()
    return proposal_weighted_kl_and_dsm_loss(
        model.flow, z, branch, w_star if use_forecast_disagreement else w_pi, context, dsm_weight=dsm_weight)


@torch.no_grad()
def evaluate_filter_records(
    model: Any,
    records_with_meta: Sequence[tuple[str, str, str, Mapping[str, Any]]],
    *,
    device: torch.device | str = "cpu",
    good_orbit_harm_threshold_km: float = 20.0,
) -> dict[str, Any]:
    """Evaluate a V1 or V2 sequence filter on (record_id, source_id, cadence, record) items."""
    dev = torch.device(device)
    model.eval()
    rows: list[RecordEvalRow] = []

    for rec_id, src_id, cadence, rec in records_with_meta:
        x0 = torch.as_tensor(rec["initial_state_si"], device=dev, dtype=torch.float64)
        t0 = float(rec["init_time_sec"])
        times = torch.as_tensor(rec["times_sec"], dtype=torch.float64)
        obs = torch.as_tensor(rec["observed_los"], device=dev, dtype=torch.float64)
        stn = torch.as_tensor(rec["station_states_si"], device=dev, dtype=torch.float64)
        truth = torch.as_tensor(rec["truth_states_si"], device=dev, dtype=torch.float64)
        valid = torch.as_tensor(
            rec.get("valid_mask", torch.ones(times.shape[0], dtype=torch.bool)), dtype=torch.bool
        )

        val_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        first_k = int(val_idx[0].item())
        last_k = int(val_idx[-1].item())
        truth_at_t0, _ = propagate_si_with_stm(
            truth[first_k], t0 - float(times[first_k].item()), compute_stm=False
        )
        b0_err_km = float(torch.linalg.vector_norm(x0[:3] - truth_at_t0[:3]).item() / 1000.0)

        pure_priors: list[Tensor] = []
        curr_p = x0
        curr_tp = t0
        for k_i in val_idx.tolist():
            tk_i = float(times[k_i].item())
            curr_p, _ = propagate_si_with_stm(curr_p, tk_i - curr_tp, compute_stm=False)
            curr_tp = tk_i
            pure_priors.append(curr_p)
        pure_prior_stack = torch.stack(pure_priors, dim=0)
        truth_valid = truth[val_idx.to(dev)]
        prior_pos_rmse_km = float(
            torch.sqrt(((pure_prior_stack[:, :3] - truth_valid[:, :3]).square().sum(dim=-1)).mean()).item()
            / 1000.0
        )

        b2_ep_km: float | None = None
        failed_batches = attempted_batches = 0
        if hasattr(model, "rollout_track"):
            from inspect import Parameter, signature
            parameters = signature(model.rollout_track).parameters
            accepts_kwargs = any(p.kind is Parameter.VAR_KEYWORD for p in parameters.values())
            metadata = {key: rec[key] for key in ("station_ids", "station_id_reliable", "block_start_time_sec",
                                                   "readout_until_sec", "iod_last_time_sec", "initial_station_si")
                        if key in rec and (accepts_kwargs or key in parameters)}
            roll = model.rollout_track(
                initial_state_si=x0,
                init_time_sec=t0,
                times_sec=times,
                observed_los=obs,
                station_states_si=stn,
                valid_mask=valid,
                **metadata,
            )
            post_all = roll["posterior_states_si"]
            prior_all = roll["prior_states_si"]
            failed_batches = roll.get("failed_proposals", 0)
            attempted_batches = roll.get("attempted_batches", 0)
            if roll["block2_t30_state_si"] is not None and roll["block2_t30_time_sec"] is not None:
                t_ep = float(roll["block2_t30_time_sec"])
                near_pos = int(torch.argmin((times[val_idx] - t_ep).abs()).item())
                k_near = int(val_idx[near_pos].item())
                truth_ep, _ = propagate_si_with_stm(
                    truth[k_near], t_ep - float(times[k_near].item()), compute_stm=False
                )
                b2_ep_km = float(
                    evaluate_v2_primary_endpoint_forecast300(roll["block2_t30_state_si"], truth_ep).item()
                )
        else:
            out = model.forward_filter_sequence(
                initial_state_si=x0,
                init_time_sec=t0,
                times_sec=times,
                observed_los=obs,
                station_states_si=stn,
                valid_mask=valid,
            )
            post_all = out.posterior_states_si
            prior_all = out.prior_states_si

        post_valid = post_all[val_idx.to(dev)]
        prior_errors = (prior_all[val_idx.to(dev), :3]-truth_valid[:, :3]).norm(dim=-1) / 1000.
        post_errors = (post_valid[:, :3]-truth_valid[:, :3]).norm(dim=-1) / 1000.
        eligible = prior_errors <= good_orbit_harm_threshold_km
        harmful = eligible & ((post_errors > good_orbit_harm_threshold_km) | ~torch.isfinite(post_errors))
        is_fin = bool(torch.isfinite(post_valid).all().item())
        if is_fin:
            scale = state_scale(device=dev, dtype=torch.float64)
            smse = float(((post_valid - truth_valid) / scale).square().mean().item())
            pos_rmse = float(
                torch.sqrt(((post_valid[:, :3] - truth_valid[:, :3]).square().sum(dim=-1)).mean()).item()
                / 1000.0
            )
            vel_rmse = float(
                torch.sqrt(((post_valid[:, 3:] - truth_valid[:, 3:]).square().sum(dim=-1)).mean()).item()
            )
            b6_vel = float(torch.linalg.vector_norm(post_valid[-1, 3:] - truth_valid[-1, 3:]).item())
            local_last = compute_los_local_errors(post_valid[-1], truth_valid[-1], stn[last_k])
            b6_rad = float(local_last["r_R_km"].item())
        else:
            smse = float("inf")
            pos_rmse = float("inf")
            vel_rmse = float("inf")
            b6_vel = float("inf")
            b6_rad = float("inf")

        rows.append(
            RecordEvalRow(
                record_id=rec_id,
                source_id=src_id,
                cadence=cadence,
                b0_pos_error_km=b0_err_km,
                pos_rmse_km=pos_rmse,
                vel_rmse_mps=vel_rmse,
                scaled_mse=smse,
                block6_end_vel_rmse_mps=b6_vel if times[last_k] >= rec.get("block_start_time_sec", t0)+3000. else None,
                block6_end_radial_pos_km=b6_rad if times[last_k] >= rec.get("block_start_time_sec", t0)+3000. else None,
                is_finite=is_fin,
                diverged=(not is_fin) or (pos_rmse > 1000.0),
                prior_pos_rmse_km=prior_pos_rmse_km,
                block2_t30_forecast300_pos_km=b2_ep_km,
                valid_frames=val_idx.numel(),
                eligible_prior_epochs=int(eligible.sum()),
                newly_harmed_epochs=int(harmful.sum()),
                failed_proposals=failed_batches,
                attempted_batches=attempted_batches,
            )
        )

    return {"summary": summarize_record_rows(rows), "rows": rows}


def main(evaluation: bool = False) -> None:
    parser = argparse.ArgumentParser(description="Model Plan V1 Trainer & Validator")
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--arch",
        default="ltr_lm",
        choices=(
            "ltr_lm",
            "jf_br",
            "cl_gn1_pure",
            "cl_gn1_matched",
            "cl_window_lm",
            "cl_full_prefix_wls",
            "fd_asf",
            "posterior_cnf",
            "sobol_lambert_lm",
        ),
    )
    parser.add_argument("--writeback-mode", default="joint_6d", choices=("joint_6d", "schur_vel", "lock_pos_vel"))
    parser.add_argument("--jfbr-mode", default="homog_6d", choices=("vel_only", "homog_6d"))
    parser.add_argument("--loss-type", default="log_charbonnier", choices=V1_LOSS_TYPES)
    parser.add_argument("--num-particles", type=int, default=12)
    parser.add_argument("--dsm-weight", type=float, default=0.0)
    parser.add_argument("--eval-split", default="dev_select", choices=("dev_select", "paired_val"))
    parser.add_argument("--max-records-per-epoch", type=int, default=0, help="0 = full train_fit split")
    parser.add_argument("--max-eval-records", type=int, default=0, help="0 = full eval split")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=20261028, help="Model/training RNG seed")
    parser.add_argument("--split-seed", type=int, default=20261028, help="Frozen source_id split seed")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    if not evaluation and args.eval_split == "paired_val":
        parser.error("Training/model selection MUST use --eval-split dev_select; reserve paired_val for locked validation only.")

    config_keys = ("arch", "writeback_mode", "jfbr_mode", "proposal_mode", "num_particles",
                   "dsm_weight", "loss_type", "split_seed", "seed")
    config_keys = tuple(key for key in config_keys if hasattr(args, key))
    learned = not (args.arch.startswith("cl_") or args.arch == "sobol_lambert_lm")
    ckpt = read_checkpoint(args, parser, evaluation=evaluation, config_keys=config_keys,
                           eval_overrides=("num_particles",), require_weights=learned)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    model = build_v1_model(
        args.arch,
        writeback_mode=args.writeback_mode,
        jfbr_mode=args.jfbr_mode,
        num_particles=args.num_particles,
        device=args.device,
    )
    last_ckpt = args.run_dir / "last.pt"
    start_epoch = 1
    if ckpt:
        model.load_state_dict(ckpt["model"], strict=True)
        start_epoch = int(ckpt.get("epoch", 0)) + 1

    if args.eval_split == "paired_val":
        eval_prep = Prepared(args.prepared_root, args.protocol_dir, "val")
        eval_keys = sorted(eval_prep.rows)
    else:
        eval_prep = Prepared(args.prepared_root, args.protocol_dir, "train")
        eval_keys = partition_prepared_keys_by_source(eval_prep.rows, seed=args.split_seed)["dev_select"]
    if args.max_eval_records > 0:
        eval_keys = eval_keys[: args.max_eval_records]

    def _pack_eval_items(prep: Prepared, keys: Sequence[str]):
        return (
            (k, str(prep.rows[k]["record_key"]["source_id"]), str(prep.rows[k]["record_key"]["rhythm"]), prep.record(k))
            for k in keys
        )

    if evaluation or not learned:
        res = evaluate_filter_records(model, _pack_eval_items(eval_prep, eval_keys), device=args.device)
        out_path = args.run_dir / f"eval_{args.eval_split}.json"
        out_path.write_text(json.dumps(res["summary"], indent=2) + "\n")
        print(json.dumps(res["summary"]), flush=True)
        return

    train_prep = Prepared(args.prepared_root, args.protocol_dir, "train")
    fit_keys = partition_prepared_keys_by_source(train_prep.rows, seed=args.split_seed)["train_fit"]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    def selection_score(summary):
        return float(summary["overall"]["position_km"]["p95"])
    best_p95 = restore_training_state(ckpt, optimizer, args.run_dir, selection_score)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        rng = random.Random(args.seed + epoch)
        shuffled = list(fit_keys)
        rng.shuffle(shuffled)
        if args.max_records_per_epoch > 0:
            shuffled = shuffled[: args.max_records_per_epoch]

        for start in range(0, len(shuffled), args.batch_size):
            batch_keys = shuffled[start : start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            batch_losses = []
            for key in batch_keys:
                rec = train_prep.record(key)
                if isinstance(model, FDASFSequentialFilter):
                    loss_k = train_fd_asf_proposal_step_on_record(
                        model,
                        rec,
                        use_forecast_disagreement=(args.arch == "fd_asf"),
                        dsm_weight=args.dsm_weight,
                        device=args.device,
                    )
                else:
                    loss_k = train_v1_refiner_step_on_record(
                        model, rec, loss_type=args.loss_type, device=args.device
                    )
                batch_losses.append(loss_k)
            loss = torch.stack(batch_losses).mean()
            if not loss.requires_grad:
                continue
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "cursor": start + len(batch_keys),
                        "train_loss": float(loss.detach().item()),
                        "grad_norm": float(grad_norm.item()),
                    }
                ),
                flush=True,
            )

        val_res = evaluate_filter_records(model, _pack_eval_items(eval_prep, eval_keys), device=args.device)
        (args.run_dir / f"dev_select_{epoch:02d}.json").write_text(json.dumps(val_res["summary"], indent=2) + "\n")
        best_p95 = save_epoch(args.run_dir / f"epoch_{epoch:02d}.pt", last_ckpt, model, optimizer,
                              args, config_keys, epoch, selection_score(val_res["summary"]), best_p95)


__all__ = [
    "V1_LOSS_TYPES",
    "build_v1_model",
    "evaluate_filter_records",
    "main",
    "state_regression_loss",
    "train_fd_asf_proposal_step_on_record",
    "train_v1_refiner_step_on_record",
]

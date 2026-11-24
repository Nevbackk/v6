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
from .eval_metrics import metrics_json
from .checkpoints import read_checkpoint, restore_training_state, save_epoch
from .data.splits import partition_prepared_keys_by_source
from .eval_metrics import (
    RecordEvalRow,
    compute_los_local_errors,
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
        initial_state_si=x0,
        init_time_sec=t_init,
        times_sec=times,
        observed_los=obs,
        station_states_si=stn,
        valid_mask=valid,
        station_ids=record.get("station_ids"),
        station_id_reliable=record.get("station_id_reliable"),
        detach_feedback=True,
    )
    if not bool(valid.any()):
        return x0.new_zeros((), requires_grad=True)
    return state_regression_loss(rollout["posterior_states_si"][valid], truth[valid], loss_type=loss_type)


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
    initializations = set()

    for rec_id, src_id, cadence, rec in records_with_meta:
        initializations.add(rec.get("initialization", "supplied"))
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
        if not len(val_idx):
            raise ValueError(f"Cannot score a record without valid observations: {rec_id}")
        first_k = int(val_idx[0].item())
        last_k = int(val_idx[-1].item())
        truth_at_t0, _ = propagate_si_with_stm(
            truth[first_k], t0 - float(times[first_k].item()), compute_stm=False
        )
        b0_err_km = (
            0.0
            if rec.get("initialization") == "known_state"
            else float(torch.linalg.vector_norm(x0[:3] - truth_at_t0[:3]).item() / 1000.0)
        )

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

        observation_nll = None
        posterior_nll = affine_nll = None
        hpd90 = cone90 = los_error = None
        failure = None
        roll = None
        failed_batches = attempted_batches = 0
        try:
            if hasattr(model, "rollout_track"):
                from inspect import Parameter, signature

                parameters = signature(model.rollout_track).parameters
                accepts_kwargs = any(p.kind is Parameter.VAR_KEYWORD for p in parameters.values())
                metadata = {
                    key: rec[key]
                    for key in ("station_ids", "station_id_reliable", "initial_station_si")
                    if key in rec and (accepts_kwargs or key in parameters)
                }
                if "score_states_si" in parameters:
                    metadata["score_states_si"] = truth
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
                if "observation_nll" in roll:
                    observation_nll = float(roll["observation_nll"][val_idx.to(dev)].mean())
                if roll.get("posterior_nll_si") is not None:
                    posterior_nll = float(roll["posterior_nll_si"][val_idx.to(dev)].mean())
                    affine_nll = float(roll["reference_nll_si"][val_idx.to(dev)].mean())
                    hpd90 = float(roll["posterior_hpd90"][val_idx.to(dev)].mean())
                    measured_error = roll["los_error_rad"][val_idx.to(dev)]
                    cone90 = float(
                        (measured_error <= roll["los_cone90_rad"][val_idx.to(dev)]).double().mean()
                    )
                    los_error = float(measured_error.mean())
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
        except (FloatingPointError, torch.linalg.LinAlgError, OverflowError) as error:
            failure = dict(
                type=getattr(error, "cause_type", type(error).__name__),
                message=str(error),
                frame=getattr(error, "frame_index", None),
                time_sec=getattr(error, "time_sec", None),
            )
            prior_all = post_all = torch.full_like(truth, float("nan"))
        post_valid = post_all[val_idx.to(dev)]
        prior_errors = (prior_all[val_idx.to(dev), :3]-truth_valid[:, :3]).norm(dim=-1) / 1000.
        post_errors = (post_valid[:, :3]-truth_valid[:, :3]).norm(dim=-1) / 1000.
        eligible = prior_errors <= good_orbit_harm_threshold_km
        harmful = eligible & ((post_errors > good_orbit_harm_threshold_km) | ~torch.isfinite(post_errors))
        finite_frames = torch.isfinite(post_valid).all(-1) & torch.isfinite(prior_all[val_idx.to(dev)]).all(
            -1
        )
        if roll is not None and failure is None:
            for field in ("observation_nll", "posterior_nll_si"):
                if roll.get(field) is not None:
                    finite_frames &= torch.isfinite(roll[field][val_idx.to(dev)])
        is_fin = bool(finite_frames.all())
        if not is_fin and failure is None:
            bad = int(val_idx[(~finite_frames).nonzero()[0, 0].cpu()])
            failure = dict(
                type="NonFiniteOutput",
                message="Non-finite state or predictive score",
                frame=bad,
                time_sec=float(times[bad]),
            )
        if failure is not None and hasattr(model, "estimator"):
            # Keep failed density estimators in accuracy, coverage and NLL denominators.
            posterior_nll = affine_nll = observation_nll = los_error = float("inf")
            hpd90 = cone90 = 0.0
        elif failure is not None and hasattr(model, "sensor"):
            observation_nll = float("inf")
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
                last_frame_vel_mps=b6_vel,
                last_frame_radial_pos_km=b6_rad,
                is_finite=is_fin,
                diverged=(not is_fin) or (pos_rmse > 1000.0),
                prior_pos_rmse_km=prior_pos_rmse_km,
                observation_nll=observation_nll,
                posterior_nll_si=posterior_nll,
                reference_nll_si=affine_nll,
                initialization=rec.get("initialization", "supplied"),
                failure=failure,
                posterior_hpd90_rate=hpd90,
                los_cone90_rate=cone90,
                los_error_rad=los_error,
                valid_frames=val_idx.numel(),
                eligible_prior_epochs=int(eligible.sum()),
                newly_harmed_epochs=int(harmful.sum()),
                failed_proposals=failed_batches,
                attempted_batches=attempted_batches,
            )
        )

    summary = summarize_record_rows(rows)
    summary["initializations"] = sorted(initializations)
    return {"summary": summary, "rows": rows}


def diffusion_entry(version, evaluation):
    """Select the per-frame estimator from the command line or its saved checkpoint."""
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument("--arch")
    selector.add_argument("--route")
    selector.add_argument("--ablation-mode")
    selector.add_argument("--resume", type=Path)
    selector.add_argument("--run-dir", type=Path)
    chosen, _ = selector.parse_known_args()
    arch = chosen.arch
    if arch is None and chosen.ablation_mode is not None:
        arch = "asd_sc"
    if arch is None and chosen.run_dir is not None:
        path = chosen.resume or chosen.run_dir / ("best.pt" if evaluation else "last.pt")
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=True)
            arch = saved.get("arch", "asd_sc" if "ablation_mode" in saved else "diffusion")
    if arch is None:
        arch = "ppc" if version == "v2" and chosen.route is None else "diffusion"
    if arch in ("ppc", "qr", "gaussian", "diffusion"):
        from .training_diffusion import main as run_diffusion

        run_diffusion(evaluation, version=version, default_arch=arch)
        return True
    return False


def main(evaluation: bool = False) -> None:
    if diffusion_entry("v1", evaluation):
        return
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
    parser.add_argument(
        "--initialization",
        choices=("known_state", "gauss_ls"),
        default="known_state",
        help="Use gauss_ls to compare classical controls with V2 on identical B0",
    )
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

    config_keys = (
        "arch",
        "initialization",
        "writeback_mode",
        "jfbr_mode",
        "proposal_mode",
        "num_particles",
        "dsm_weight",
        "loss_type",
        "split_seed",
        "seed",
    )
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
            (
                k,
                str(prep.rows[k]["record_key"]["source_id"]),
                str(prep.rows[k]["record_key"]["rhythm"]),
                prep.record(k, exact_initial=(args.initialization == "known_state")),
            )
            for k in keys
        )

    if evaluation or not learned:
        res = evaluate_filter_records(model, _pack_eval_items(eval_prep, eval_keys), device=args.device)
        out_path = args.run_dir / f"eval_{args.eval_split}.json"
        out_path.write_text(metrics_json(res["summary"], indent=2) + "\n")
        print(metrics_json(res["summary"]), flush=True)
        return

    train_prep = Prepared(args.prepared_root, args.protocol_dir, "train")
    fit_keys = partition_prepared_keys_by_source(train_prep.rows, seed=args.split_seed)["train_fit"]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    def selection_score(summary):
        return summary["source_equal_summary"]["scaled_mse"]
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
                rec = train_prep.record(key, exact_initial=(args.initialization == "known_state"))
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
        (args.run_dir / f"dev_select_{epoch:02d}.json").write_text(
            metrics_json(val_res["summary"], indent=2) + "\n"
        )
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

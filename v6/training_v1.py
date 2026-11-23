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
from .data.splits import (
    build_macro_checkpoint_schedule,
    partition_prepared_keys_by_source,
)
from .eval_metrics import (
    RecordEvalRow,
    compute_los_local_errors,
    evaluate_v2_primary_endpoint_forecast300,
    summarize_record_rows,
)
from .models.batched_physics import state_scale
from .models.reprojection import (
    estimate_online_los_noise_floor,
    propagate_si_with_stm,
    select_causal_keyframes,
)
from .models.v1_fd_asf import (
    AttributableChart2Plus4,
    FDASFSequentialFilter,
    check_bound_perigee_feasible,
    compute_forecast_disagreement_weights,
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

    checkpoints, _ = build_macro_checkpoint_schedule(
        times, valid, init_time_sec=t_init, include_block2_primary_endpoint=False
    )
    active_cps = [cp for cp in checkpoints if cp.last_obs_idx >= 1]
    if not active_cps:
        return torch.zeros((), device=dev, dtype=dtype, requires_grad=True)

    losses: list[Tensor] = []
    curr_x = x0
    curr_t = t_init

    for cp in active_cps:
        k = cp.last_obs_idx
        tk = float(times[k].item())
        prior_k, _ = propagate_si_with_stm(curr_x.detach(), tk - curr_t, compute_stm=False)
        tau_k, _, _ = estimate_online_los_noise_floor(
            times, obs.detach().cpu(), valid_mask=valid, up_to_idx=k
        )
        kf_idx, _ = select_causal_keyframes(
            times, valid, up_to_idx=k, prev_checkpoint_idx=cp.prev_last_obs_idx, max_keyframes=model.max_keyframes
        )
        if kf_idx.size < 2:
            curr_x, curr_t = prior_k.detach(), tk
            continue
        out = model.refiner.refine_anchor(
            prior_k,
            tk,
            stn[k, :3],
            times[kf_idx].to(device=dev, dtype=dtype),
            stn[kf_idx],
            obs[kf_idx],
            tau_rad=tau_k,
        )
        losses.append(state_regression_loss(out.posterior_state_si, truth[k], loss_type=loss_type))
        curr_x = out.posterior_state_si.detach()
        curr_t = tk

    if not losses:
        return torch.zeros((), device=dev, dtype=dtype, requires_grad=True)
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
    dev = torch.device(device)
    dtype = torch.float64
    t_init = float(record["init_time_sec"])
    times = torch.as_tensor(record["times_sec"], dtype=dtype)
    obs = torch.as_tensor(record["observed_los"], device=dev, dtype=dtype)
    stn = torch.as_tensor(record["station_states_si"], device=dev, dtype=dtype)
    truth = torch.as_tensor(record["truth_states_si"], device=dev, dtype=dtype)
    valid = torch.as_tensor(
        record.get("valid_mask", torch.ones(times.shape[0], dtype=torch.bool)), dtype=torch.bool
    )

    _, blocks = build_macro_checkpoint_schedule(
        times, valid, init_time_sec=t_init, include_block2_primary_endpoint=False
    )
    if not blocks:
        return torch.zeros((), device=dev, dtype=dtype, requires_grad=True)

    blk0 = blocks[0]
    k_end = blk0.end_idx
    _, sigma_k, _ = estimate_online_los_noise_floor(
        times, obs.detach().cpu(), valid_mask=valid, up_to_idx=k_end
    )
    ctx, attr = model.tracklet_encoder.encode_prefix(
        times, obs, stn, valid, up_to_idx=k_end, sigma_u_rad=sigma_k
    )
    if not attr:
        return torch.zeros((), device=dev, dtype=dtype, requires_grad=True)

    t_c = attr["t_c_sec"]
    k_mid = (blk0.start_idx + blk0.end_idx) // 2
    truth_c, _ = propagate_si_with_stm(truth[k_mid], t_c - float(times[k_mid].item()), compute_stm=False)
    rel_r_star = truth_c[:3] - attr["station_c_si"][:3]
    rel_v_star = truth_c[3:] - attr["station_c_si"][3:]
    rho_star = torch.linalg.vector_norm(rel_r_star).clamp_min(100.0)
    rhodot_star = (rel_v_star * (rel_r_star / rho_star)).sum()

    chart = AttributableChart2Plus4(
        t_c_sec=t_c,
        t_ref_sec=t_init,
        station_c_si=attr["station_c_si"],
        u_bar_c=attr["u_bar_c"],
        omega_bar_c=attr["omega_bar_c"],
    )
    z_star = torch.stack(
        (torch.log(rho_star / chart.rho_ref_m), rhodot_star / chart.v_ref_mps), dim=0
    )

    M = int(num_ref_candidates)
    n_local = M // 2
    z_local = z_star.unsqueeze(0) + torch.randn((n_local, 2), device=dev, dtype=dtype) * 0.15
    z_wide = z_star.unsqueeze(0) + torch.randn((M - n_local, 2), device=dev, dtype=dtype) * 0.85
    z_cand = torch.cat((z_local, z_wide), dim=0)
    b_cand = torch.zeros((M,), device=dev, dtype=torch.long)

    thetas_cand, log_det = chart.decode(z_cand)
    kf_idx, _ = select_causal_keyframes(times, valid, up_to_idx=k_end, max_keyframes=12)
    kappa = 1.0 / max(sigma_k * sigma_k, 1.0e-8)
    nll = model._segment_nll(
        thetas_cand,
        t_init,
        times[kf_idx].to(device=dev),
        stn[kf_idx],
        obs[kf_idx],
        kappa,
    )
    feas = check_bound_perigee_feasible(thetas_cand)
    log_pi = torch.where(feas, -nll + log_det, torch.full_like(nll, -1.0e6))

    if len(blocks) >= 2:
        next_idx = blocks[1].start_idx
        t_next = float(times[next_idx].item())
        s_next = stn[next_idx, :3]
    else:
        t_next = float(times[k_end].item()) + 300.0
        s_next_full, _ = propagate_si_with_stm(stn[k_end], 300.0, compute_stm=False)
        s_next = s_next_full[:3]

    w_star, w_pi, _ = compute_forecast_disagreement_weights(
        thetas_cand,
        log_pi,
        t_ref_sec=t_init,
        next_view_time_sec=t_next,
        next_station_pos_si=s_next,
    )
    target_w = w_star if use_forecast_disagreement else w_pi
    return proposal_weighted_kl_and_dsm_loss(
        model.flow, z_cand, b_cand, target_w, ctx, dsm_weight=dsm_weight
    )


@torch.no_grad()
def evaluate_filter_records(
    model: Any,
    records_with_meta: Sequence[tuple[str, str, str, Mapping[str, Any]]],
    *,
    device: torch.device | str = "cpu",
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
        if hasattr(model, "rollout_track"):
            roll = model.rollout_track(
                initial_state_si=x0,
                init_time_sec=t0,
                times_sec=times,
                observed_los=obs,
                station_states_si=stn,
                valid_mask=valid,
            )
            post_all = roll["posterior_states_si"]
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

        post_valid = post_all[val_idx.to(dev)]
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
                block6_end_vel_rmse_mps=b6_vel,
                block6_end_radial_pos_km=b6_rad,
                is_finite=is_fin,
                diverged=(not is_fin) or (pos_rmse > 1000.0),
                prior_pos_rmse_km=prior_pos_rmse_km,
                block2_t30_forecast300_pos_km=b2_ep_km,
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
    parser.add_argument("--dsm-weight", type=float, default=0.05)
    parser.add_argument("--eval-split", default="dev_select", choices=("dev_select", "paired_val"))
    parser.add_argument("--max-records-per-epoch", type=int, default=0, help="0 = full train_fit split")
    parser.add_argument("--max-eval-records", type=int, default=0, help="0 = full eval split")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

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
    src_ckpt = args.resume or (last_ckpt if last_ckpt.exists() else None)
    start_epoch = 1
    if src_ckpt and src_ckpt.exists():
        ckpt = torch.load(src_ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"], strict=True)
        start_epoch = int(ckpt.get("epoch", 0)) + 1

    if args.eval_split == "paired_val":
        eval_prep = Prepared(args.prepared_root, args.protocol_dir, "val")
        eval_keys = sorted(eval_prep.rows)
    else:
        eval_prep = Prepared(args.prepared_root, args.protocol_dir, "train")
        eval_keys = partition_prepared_keys_by_source(eval_prep.rows, seed=args.seed)["dev_select"]
    if args.max_eval_records > 0:
        eval_keys = eval_keys[: args.max_eval_records]

    def _pack_eval_items(prep: Prepared, keys: Sequence[str]):
        return [
            (k, str(prep.rows[k]["record_key"]["source_id"]), str(prep.rows[k]["record_key"]["rhythm"]), prep.record(k))
            for k in keys
        ]

    if evaluation or sum(p.numel() for p in model.parameters()) == 0:
        res = evaluate_filter_records(model, _pack_eval_items(eval_prep, eval_keys), device=args.device)
        out_path = args.run_dir / f"eval_{args.eval_split}.json"
        out_path.write_text(json.dumps(res["summary"], indent=2) + "\n")
        print(json.dumps(res["summary"]), flush=True)
        return

    train_prep = Prepared(args.prepared_root, args.protocol_dir, "train")
    fit_keys = partition_prepared_keys_by_source(train_prep.rows, seed=args.seed)["train_fit"]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    best_p95 = float("inf")

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

        payload = {"model": model.state_dict(), "epoch": epoch, "arch": args.arch, "seed": args.seed}
        torch.save(payload, args.run_dir / f"epoch_{epoch:02d}.pt")
        torch.save(payload, last_ckpt)
        val_res = evaluate_filter_records(model, _pack_eval_items(eval_prep, eval_keys), device=args.device)
        (args.run_dir / f"dev_select_{epoch:02d}.json").write_text(json.dumps(val_res["summary"], indent=2) + "\n")
        curr_p95 = float(val_res["summary"]["overall"]["position_km"]["p95"])
        if curr_p95 <= best_p95:
            best_p95 = curr_p95
            torch.save(payload, args.run_dir / "best.pt")


__all__ = [
    "V1_LOSS_TYPES",
    "build_v1_model",
    "evaluate_filter_records",
    "main",
    "state_regression_loss",
    "train_fd_asf_proposal_step_on_record",
    "train_v1_refiner_step_on_record",
]

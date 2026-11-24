"""Training and stratified evaluation for Round 5 Genuine Diffusion Models (Route A & Route B).

Strict experimental hygiene:
- Frozen --split-seed decoupled from training --seed.
- Refuses --eval-split paired_val during training (DEV-select only for model selection).
- Route A (VP-OrbitDiff): single multi-scale cosine-VP velocity + second-moment NLL objective,
  supporting causal rollout prior aggregation and guidance/K/P/delta ablations.
- Route B1 (Guided Cosine Flow 2D): single importance-corrected 3D-chord cosine-CFM loss
  covering both Attributable and Lambert charts + both branches + 4D angular relaxation.
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
from .data.observation_clock import causal_diffusion_events
from .checkpoints import read_checkpoint, restore_training_state, save_epoch
from .reference_prior import fit_reference_prior
from .data.splits import parse_observation_blocks, partition_prepared_keys_by_source
from .models.batched_physics import state_scale
from .models.diffusion_orbit import (
    HALF_PI,
    RouteBGuidedCosineFlow2D,
    VPOrbitDiffusionFilter,
    propagate_reference_covariance_stm,
    safe_cholesky_spd,
    vp_velocity_moment_loss,
)
from .models.reprojection import (
    estimate_online_los_noise_floor,
    propagate_si_with_stm,
)
from .models.v1_fd_asf import ContinuousTrackletEncoder
from .models.orbit_proposals import EmptyTeacherSupport, RouteBOrbitFilter, draw_chart_teacher
from .training_v1 import evaluate_filter_records


def train_route_a_vp_step_on_record(
    model: VPOrbitDiffusionFilter,
    record: Mapping[str, Any],
    *,
    num_noise_samples: int = 4,
    use_rollout_priors: bool = True,
    rollout_prior_prob: float = 0.5,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Single multi-scale cosine-VP velocity/moment training step on one multi-block record.

    Trains f_theta(z_s, s, c^-) on causal (c^-, z_0) pairs collected from BOTH pure-propagation
    B0 anchors and (when use_rollout_priors=True) detached closed-loop rollout-visited priors.
    Context c^- strictly excludes the fresh unassimilated batch Y so inference guidance never
    double-counts observations!
    """
    dev = torch.device(device)
    dtype = torch.float64
    d_scale = state_scale(device=dev, dtype=dtype)

    x0 = torch.as_tensor(record["initial_state_si"], device=dev, dtype=dtype)
    t_init = float(record["init_time_sec"])
    times = torch.as_tensor(record["times_sec"], dtype=dtype)
    obs = torch.as_tensor(record["observed_los"], device=dev, dtype=dtype)
    stn = torch.as_tensor(record["station_states_si"], device=dev, dtype=dtype)
    truth = torch.as_tensor(record["truth_states_si"], device=dev, dtype=dtype)
    valid = torch.as_tensor(
        record.get("valid_mask", torch.ones(times.shape[0], dtype=torch.bool)), dtype=torch.bool
    )

    blocks = parse_observation_blocks(times, valid, init_time_sec=t_init)
    if not blocks:
        return torch.zeros((), device=dev, dtype=dtype, requires_grad=True)

    # Collect causal anchor tuples: (prior_x_si, P_ref_in, eval_time_sec, truth_at_eval_si, prev_idx, sigma_past)
    anchors: list[tuple[Tensor, Tensor, float, Tensor, int, float]] = []
    if use_rollout_priors and random.random() < float(rollout_prior_prob):
        was_training = model.training
        model.eval()
        roll = model.rollout_track(
            initial_state_si=x0,
            init_time_sec=t_init,
            times_sec=times,
            observed_los=obs,
            station_states_si=stn,
            valid_mask=valid,
            seed=random.randint(1, 2**30 - 1),
            **{key: record[key] for key in ("block_start_time_sec", "station_ids", "station_id_reliable",
                                            "initial_station_si", "readout_until_sec")
               if key in record},
        )
        if was_training:
            model.train()
        for bt in roll["batch_traces"]:
            ev_t = float(bt["eval_time_sec"])
            last_idx = int(bt["last_obs_idx"])
            dt_tr = ev_t - float(times[last_idx].item())
            if abs(dt_tr) > 1.0e-9:
                truth_ev, _ = propagate_si_with_stm(truth[last_idx], dt_tr, compute_stm=False)
            else:
                truth_ev = truth[last_idx]
            anchors.append(
                (
                    bt["prior_state_si"].detach(),
                    bt["P_carry_in"].detach(),
                    ev_t,
                    truth_ev.detach(),
                    int(bt["prev_assimilated_idx"]),
                    float(bt["sigma_past_rad"]),
                )
            )
    else:
        station = record.get("initial_station_si")
        station_at_init = x0.new_zeros(6) if station is None else torch.as_tensor(station).to(x0)
        curr_x, P_ref = model.initial_moments(x0, station_at_init)
        curr_t = t_init
        prev_idx = -1
        events = causal_diffusion_events(times, valid, t_init, record.get("block_start_time_sec"),
                                         readout_until_sec=record.get("readout_until_sec"))
        for ev_t, last_idx, _ in events:
            if last_idx <= prev_idx:
                continue
            dt = ev_t - curr_t
            if abs(dt) > 0.0:
                curr_x, phi_step = propagate_si_with_stm(curr_x.detach(), dt, compute_stm=True)
                assert phi_step is not None
                P_ref = propagate_reference_covariance_stm(P_ref, phi_step)
                curr_t = ev_t
            if prev_idx >= 0:
                _, sigma_past, _ = estimate_online_los_noise_floor(
                    times, obs.detach().cpu(), valid_mask=valid, up_to_idx=prev_idx,
                    default_sigma_rad=model.default_sigma_rad,
                    station_states_si=stn, station_ids=record.get("station_ids"),
                    station_id_reliable=record.get("station_id_reliable"),
                )
            else:
                sigma_past = model.default_sigma_rad
            truth_ev, _ = propagate_si_with_stm(truth[last_idx], curr_t-times[last_idx].item(), compute_stm=False)
            anchors.append((curr_x.detach(), P_ref.detach(), curr_t, truth_ev.detach(), prev_idx, sigma_past))
            prev_idx = last_idx

    if not anchors:
        return torch.zeros((), device=dev, dtype=dtype, requires_grad=True)

    losses: list[Tensor] = []
    M = max(1, int(num_noise_samples))
    for prior_x, P_in, ev_t, truth_ev, prev_idx, sigma_past in anchors:
        L_b = safe_cholesky_spd(P_in)
        ctx_past = model.build_causal_past_context(
            prior_state_si=prior_x,
            L_b=L_b,
            eval_time_sec=ev_t,
            init_time_sec=t_init,
            times_sec=times,
            observed_los=obs,
            station_states_si=stn,
            valid_mask=valid,
            prev_assimilated_idx=prev_idx,
            sigma_u_rad=sigma_past,
        )
        # Whitened clean target z_0 = L_b^{-1} D^{-1} (x* - x^-)
        err_scaled = (truth_ev - prior_x) / d_scale
        z_0 = torch.linalg.solve_triangular(L_b, err_scaled.unsqueeze(-1), upper=False).squeeze(-1).detach()

        # The loss is defined at both endpoints; train over the full cosine path.
        s = torch.rand((M,), device=dev, dtype=dtype) * HALF_PI
        eps = torch.randn((M, 6), device=dev, dtype=dtype)
        alpha = torch.cos(s).unsqueeze(-1)
        sigma = torch.sin(s).unsqueeze(-1)
        z_s = alpha * z_0.unsqueeze(0) + sigma * eps
        v_star = alpha * eps - sigma * z_0.unsqueeze(0)

        v_theta, V_theta, _, _ = model.denoiser(z_s, s, ctx_past)
        losses.append(
            vp_velocity_moment_loss(
                v_star, v_theta, V_theta, fixed_isotropic_v=model.fixed_isotropic_v
            )
        )

    return torch.stack(losses).mean()


def train_route_b_cosine_flow_step_on_record(
    flow_b: RouteBGuidedCosineFlow2D,
    encoder: ContinuousTrackletEncoder,
    record: Mapping[str, Any],
    *,
    use_chord_disagreement: bool = True,
    num_ref_candidates: int = 24,
    los_budget: int | None = None,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Single importance-corrected cosine-CFM step for Route B1 across both charts, branches & 4D eta."""
    z, branch, eta, w_star, w_pi, context = draw_chart_teacher(
        encoder, flow_b, record, num_ref_candidates=num_ref_candidates, device=device, los_budget=los_budget)
    return flow_b.cosine_cfm_loss(z, branch, eta, w_star if use_chord_disagreement else w_pi, context)


def main(evaluation: bool = False) -> None:
    parser = argparse.ArgumentParser(description="Round 5 Genuine Diffusion Trainer & Validator (Route A & B)")
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--route", choices=("a", "b1", "b2"), default="a")
    parser.add_argument("--K", type=int, default=4, help="Core knob 1: artificial reverse diffusion intervals")
    parser.add_argument("--P", type=int, default=8, help="Core knob 2: independent seeds / particles")
    parser.add_argument("--delta", type=float, default=0.10, help="Core knob 3: min reference covariance fraction")
    parser.add_argument("--B-los", type=int, default=0, help="Optional knob 4: raw fresh LOS budget (0 = full fresh batch)")
    parser.add_argument("--guidance-mode", default="vjp", choices=("vjp", "identity_jac", "none"))
    parser.add_argument("--fixed-isotropic-v", action="store_true", help="Ablation: fix V_theta = I")
    parser.add_argument("--no-rollout-priors", action="store_true", help="Ablation: pure B0 propagation priors only")
    parser.add_argument("--eval-split", default="dev_select", choices=("dev_select", "paired_val"))
    parser.add_argument("--max-records-per-epoch", type=int, default=0)
    parser.add_argument("--max-eval-records", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=20261028, help="Model/training RNG seed")
    parser.add_argument("--split-seed", type=int, default=20261028, help="Frozen source_id split seed")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    if not evaluation and args.eval_split == "paired_val":
        parser.error("Training/model selection MUST use --eval-split dev_select; reserve paired_val for locked validation.")
    config_keys = ("route", "K", "P", "delta", "B_los", "guidance_mode", "fixed_isotropic_v",
                   "no_rollout_priors", "split_seed", "seed")
    ckpt = read_checkpoint(args, parser, evaluation=evaluation, config_keys=config_keys,
                           eval_overrides=("K", "P", "B_los", "guidance_mode", "delta"))
    if args.K < 1 or args.P < 1 or args.B_los < 0 or not 0 < args.delta <= 1:
        parser.error("K and P must be positive, B-los nonnegative and delta in (0,1]")
    if args.route != "a" and (args.guidance_mode == "identity_jac" or args.fixed_isotropic_v
                              or args.no_rollout_priors or args.delta != parser.get_default("delta")):
        parser.error("identity_jac, fixed-isotropic-v, no-rollout-priors and delta tuning apply to Route A")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    if args.route == "a":
        model = VPOrbitDiffusionFilter(num_reverse_steps=args.K, num_particles=args.P,
                                       delta=args.delta, los_budget=args.B_los or None,
                                       guidance_mode=args.guidance_mode,
                                       fixed_isotropic_v=args.fixed_isotropic_v)
    else:
        model = RouteBOrbitFilter(args.route, num_steps=args.K, num_particles=args.P,
                                  los_budget=args.B_los or None,
                                  use_guidance=args.guidance_mode != "none")
    model.to(args.device)

    last_ckpt = args.run_dir / "last.pt"
    start_epoch = 1
    if ckpt:
        model.load_state_dict(ckpt["model"], strict=True)
        if args.route == "a":
            model.reference_prior = ckpt.get("reference_prior")
        start_epoch = int(ckpt.get("epoch", 0)) + 1

    if args.eval_split == "paired_val":
        eval_prep = Prepared(args.prepared_root, args.protocol_dir, "val")
        eval_keys = sorted(eval_prep.rows)
    else:
        eval_prep = Prepared(args.prepared_root, args.protocol_dir, "train")
        eval_keys = partition_prepared_keys_by_source(eval_prep.rows, seed=args.split_seed)["dev_select"]
    if args.max_eval_records > 0:
        eval_keys = eval_keys[: args.max_eval_records]

    def _pack_eval(prep: Prepared, keys: Sequence[str]):
        return (
            (k, str(prep.rows[k]["record_key"]["source_id"]), str(prep.rows[k]["record_key"]["rhythm"]), prep.record(k))
            for k in keys
        )

    if evaluation:
        res = evaluate_filter_records(model, _pack_eval(eval_prep, eval_keys), device=args.device)
        (args.run_dir / f"eval_diff_{args.eval_split}.json").write_text(json.dumps(res["summary"], indent=2) + "\n")
        print(json.dumps(res["summary"]), flush=True)
        return

    train_prep = Prepared(args.prepared_root, args.protocol_dir, "train")
    fit_keys = partition_prepared_keys_by_source(train_prep.rows, seed=args.split_seed)["train_fit"]
    if args.route == "a" and model.reference_prior is None:
        model.reference_prior = fit_reference_prior(train_prep, fit_keys)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    def selection_score(summary):
        overall = summary["overall"]
        return float(overall.get("v2_primary_block2_t30_forecast300_pos_km",
                                 overall["position_km"])["capped_mean"])
    best_score = restore_training_state(ckpt, optimizer, args.run_dir, selection_score)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        rng = random.Random(args.seed + epoch)
        shuffled = list(fit_keys)
        rng.shuffle(shuffled)
        if args.max_records_per_epoch > 0:
            shuffled = shuffled[: args.max_records_per_epoch]

        trained_records = 0
        skipped = 0
        for start in range(0, len(shuffled), args.batch_size):
            batch_keys = shuffled[start : start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            batch_losses = []
            for k in batch_keys:
                record = train_prep.record(k)
                if args.route == "a":
                    batch_losses.append(train_route_a_vp_step_on_record(
                        model, record, use_rollout_priors=not args.no_rollout_priors, device=args.device))
                else:
                    try:
                        batch_losses.append(train_route_b_cosine_flow_step_on_record(
                            model.flow, model.encoder, record, device=args.device, los_budget=model.los_budget))
                    except EmptyTeacherSupport:
                        skipped += 1
            if not batch_losses:
                continue
            trained_records += len(batch_losses)
            loss = torch.stack(batch_losses).mean()
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "cursor": start + len(batch_keys),
                        "train_loss": float(loss.detach().item()),
                        "grad_norm": float(gnorm.item()),
                        "teacher_skipped": skipped,
                    }
                ),
                flush=True,
            )

        if trained_records == 0:
            raise ValueError("No usable training record in this epoch")
        val_res = evaluate_filter_records(model, _pack_eval(eval_prep, eval_keys), device=args.device)
        (args.run_dir / f"dev_select_{epoch:02d}.json").write_text(json.dumps(val_res["summary"], indent=2) + "\n")
        best_score = save_epoch(args.run_dir / f"diff_epoch_{epoch:02d}.pt", last_ckpt,
                                model, optimizer, args, config_keys, epoch,
                                selection_score(val_res["summary"]), best_score,
                                reference_prior=getattr(model, "reference_prior", None))


__all__ = [
    "main",
    "train_route_a_vp_step_on_record",
    "train_route_b_cosine_flow_step_on_record",
]

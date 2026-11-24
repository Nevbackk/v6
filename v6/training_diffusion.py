"""Training and held-out evaluation for the two initialization protocols and chart proposals."""

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
from .reference_prior import fit_reference_prior
from .data.splits import partition_prepared_keys_by_source
from .models.batched_physics import state_scale
from .models.diffusion_orbit import (
    HALF_PI,
    RouteBGuidedCosineFlow2D,
    VPOrbitDiffusionFilter,
    safe_cholesky_spd,
    vp_velocity_moment_loss,
)
from .models.v1_fd_asf import ContinuousTrackletEncoder
from .models.ppc_flow import PPCOrbitFilter
from .models.orbit_proposals import EmptyTeacherSupport, RouteBOrbitFilter, draw_chart_teacher
from .training_v1 import evaluate_filter_records


def train_route_a_vp_step_on_record(
    model: VPOrbitDiffusionFilter,
    record: Mapping[str, Any],
    *,
    num_noise_samples: int = 4,
    use_rollout_priors: bool = True,
    rollout_prior_prob: float = 0.5,
    unroll_frames: int = 1,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Supervise the sensor at every arrival and V2 on the six-dimensional prior error."""
    x = torch.as_tensor(record["initial_state_si"], device=device, dtype=torch.float64)
    obs = torch.as_tensor(record["observed_los"]).to(x)
    stations = torch.as_tensor(record["station_states_si"]).to(x)
    truth = torch.as_tensor(record["truth_states_si"]).to(x)
    times = torch.as_tensor(record["times_sec"]).cpu().double()
    t0 = float(record["init_time_sec"])
    mask = record.get("valid_mask")
    pure = model.known_initial or not use_rollout_priors or random.random() >= rollout_prior_prob
    roll = model.rollout_track(
        initial_state_si=x,
        init_time_sec=t0,
        times_sec=times,
        observed_los=obs,
        station_states_si=stations,
        valid_mask=mask,
        training_pairs=True,
        pure_prior=pure,
        seed=random.randrange(2**30),
        **{
            k: record[k]
            for k in ("initial_station_si", "station_ids", "station_id_reliable")
            if k in record
        },
    )
    pairs = roll["training_pairs"]
    if not pairs:
        raise ValueError("No valid frame for diffusion training")
    angular, sensor_inputs, state_errors, state_inputs = [], [], [], []
    hidden = None if model.known_initial else x.new_zeros(model.tracklet_encoder.context_dim)
    scale = state_scale(device=x.device, dtype=x.dtype)
    for count, (frame, prior, carry, tangent, sensor_ctx, sigma) in enumerate(pairs):
        k = frame.index
        true_ray = torch.nn.functional.normalize(truth[k, :3] - stations[k, :3], dim=0)
        angular.append(tangent.T @ (torch.nn.functional.normalize(obs[k], dim=0) - true_ray) / sigma)
        sensor_inputs.append(sensor_ctx)
        if hidden is not None:
            L = safe_cholesky_spd(carry)
            state_errors.append(
                torch.linalg.solve_triangular(L, ((truth[k] - prior) / scale)[:, None], upper=False)[:, 0]
            )
            state_inputs.append(model.state_context(prior, L, frame.time_sec - t0, sigma, count, hidden))
            hidden = model.tracklet_encoder.advance(
                hidden.detach() if count % unroll_frames == 0 else hidden, obs[k], stations[k], frame.dt_sec
            )
    loss = model.sensor.loss(torch.stack(angular), torch.stack(sensor_inputs))
    if model.known_initial:
        return loss
    clean = torch.stack(state_errors).repeat_interleave(num_noise_samples, dim=0)
    ctx = torch.stack(state_inputs).repeat_interleave(num_noise_samples, dim=0)
    s = torch.rand(len(clean), device=x.device, dtype=x.dtype) * HALF_PI
    a, b = s.cos()[:, None], s.sin()[:, None]
    eps = torch.randn_like(clean)
    v, V, _, _ = model.denoiser(a * clean + b * eps, s, ctx)
    return loss + vp_velocity_moment_loss(
        a * eps - b * clean, v, V, fixed_isotropic_v=model.fixed_isotropic_v
    )


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


def main(evaluation: bool = False, *, version="v2", default_arch="diffusion") -> None:
    parser = argparse.ArgumentParser(
        description=f"Per-frame orbit estimation ({version.upper()})",
        epilog="Use --arch ltr_lm or fd_asf for V1 controls; use --arch asd_sc for the V2 control.",
    )
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--arch", choices=("ppc", "qr", "gaussian", "diffusion"), default=default_arch)
    parser.add_argument(
        "--observation-model",
        choices=("diffusion", "direct", "calibrated"),
        default="diffusion",
        help="Angular reverse chain, direct heteroscedastic MLP, or causal measured noise floor",
    )
    parser.add_argument(
        "--initial-bias",
        choices=("none", "train_mean"),
        default="none",
        help="PPC and Gaussian controls can ablate the TRAIN-fit mean correction to B0",
    )
    parser.add_argument(
        "--unroll-frames",
        type=int,
        help="Training gradient span across valid arrivals (PPC default 2, VP control 1)",
    )
    parser.add_argument(
        "--warp-nodes",
        type=int,
        default=128,
        help="Deterministic Gaussian integration points per PPC reference",
    )
    parser.add_argument(
        "--sensor-checkpoint", type=Path, help="Initialize the V2 sensor model from a V1 checkpoint"
    )
    parser.add_argument("--route", choices=("a", "b1", "b2"), default="a")
    parser.add_argument(
        "--K",
        type=int,
        default=4,
        help="PPC likelihood increments and angular reverse steps, or control integration steps",
    )
    parser.add_argument(
        "--P",
        type=int,
        default=8,
        help="Angular sampling and training CFM budget for PPC, or particles for the VP/chart controls",
    )
    parser.add_argument(
        "--delta", type=float, default=0.10, help="Reference covariance fraction for the VP state control"
    )
    parser.add_argument(
        "--B-los",
        type=int,
        default=0,
        help="B1/B2 proposal guidance LOS budget (0 uses the complete prefix)",
    )
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
    args.version = version
    config_keys = (
        "version",
        "arch",
        "warp_nodes",
        "unroll_frames",
        "observation_model",
        "initial_bias",
        "route",
        "K",
        "P",
        "delta",
        "B_los",
        "guidance_mode",
        "fixed_isotropic_v",
        "no_rollout_priors",
        "split_seed",
        "seed",
    )
    ckpt = read_checkpoint(
        args,
        parser,
        evaluation=evaluation,
        config_keys=config_keys,
        eval_overrides=("K", "P", "B_los", "guidance_mode", "delta"),
        require_weights=not (
            args.observation_model == "calibrated" and (args.arch in ("qr", "gaussian") or version == "v1")
        ),
    )
    if args.unroll_frames is None:
        args.unroll_frames = 2 if args.arch == "ppc" else 1
    if args.unroll_frames < 1:
        parser.error("--unroll-frames must be positive")
    if ckpt and ckpt.get("arch", "diffusion") != args.arch:
        parser.error("The requested model differs from the checkpoint architecture")
    if args.arch in ("ppc", "qr", "gaussian"):
        if version != "v2" or args.route != "a":
            parser.error("PPC is a single noisy-initial V2 estimator")
        if (
            args.B_los
            or args.guidance_mode != "vjp"
            or args.fixed_isotropic_v
            or args.no_rollout_priors
            or args.delta != parser.get_default("delta")
        ):
            parser.error("VP state-sampler and chart-proposal ablations apply to --arch diffusion")
    elif args.warp_nodes != parser.get_default("warp_nodes"):
        parser.error("--warp-nodes applies to PPC and its Gaussian controls")
    if args.arch == "diffusion" and args.initial_bias != "none":
        parser.error("--initial-bias applies to PPC and its matched Gaussian controls")
    if args.route != "a" and args.observation_model != "diffusion":
        parser.error("The chart proposal controls have their own observation model")
    if args.K < 1 or args.P < 1 or args.B_los < 0 or not 0 < args.delta <= 1:
        parser.error("K and P must be positive, B-los nonnegative and delta in (0,1]")
    if args.route != "a" and (args.guidance_mode == "identity_jac" or args.fixed_isotropic_v
                              or args.no_rollout_priors or args.delta != parser.get_default("delta")):
        parser.error("identity_jac, fixed-isotropic-v, no-rollout-priors and delta tuning apply to Route A")
    if args.route == "a" and args.B_los:
        parser.error(
            "Route A assimilates one fresh LOS per frame; --B-los limits chart guidance on Routes B1/B2"
        )
    if version == "v1" and (
        args.no_rollout_priors
        or args.fixed_isotropic_v
        or args.guidance_mode != "vjp"
        or args.delta != parser.get_default("delta")
    ):
        parser.error("Orbit prior and guidance ablations apply to noisy-initial V2")
    if version == "v1" and args.route != "a":
        parser.error("The exact-initial diffusion model uses the angular observation chain")
    if args.sensor_checkpoint and (version != "v2" or args.route != "a" or evaluation or ckpt):
        parser.error("--sensor-checkpoint initializes a new V2 Route A training run")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    name = args.arch if args.arch != "diffusion" else "diff"
    if args.arch in ("ppc", "qr", "gaussian"):
        model = PPCOrbitFilter(
            steps=args.K,
            sensor_samples=args.P,
            nodes=args.warp_nodes,
            estimator=args.arch,
            observation_model=args.observation_model,
            initial_bias=args.initial_bias,
        )
    elif args.route == "a":
        model = VPOrbitDiffusionFilter(
            num_reverse_steps=args.K,
            num_particles=args.P,
            delta=args.delta,
            guidance_mode=args.guidance_mode,
            fixed_isotropic_v=args.fixed_isotropic_v,
            known_initial=(version == "v1"),
            observation_model=args.observation_model,
        )
    else:
        model = RouteBOrbitFilter(args.route, num_steps=args.K, num_particles=args.P,
                                  los_budget=args.B_los or None,
                                  use_guidance=args.guidance_mode != "none")
    model.to(args.device)
    if args.sensor_checkpoint:
        source = torch.load(args.sensor_checkpoint, map_location="cpu", weights_only=True)
        if source.get("observation_model", "diffusion") != args.observation_model:
            parser.error("V1 and V2 must use the same observation model for sensor transfer")
        if source.get("version") != "v1":
            parser.error("Expected a V1 angular sensor checkpoint")
        if source.get("recursion") != "valid_frame" or source.get("split_seed") != args.split_seed:
            parser.error("The sensor checkpoint must use the same per-frame protocol and source split seed")
        model.sensor.load_state_dict(
            {k.removeprefix("sensor."): v for k, v in source["model"].items() if k.startswith("sensor.")},
            strict=True,
        )

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
            (
                k,
                str(prep.rows[k]["record_key"]["source_id"]),
                str(prep.rows[k]["record_key"]["rhythm"]),
                prep.record(k, exact_initial=(version == "v1")),
            )
            for k in keys
        )

    learned = any(p.requires_grad for p in model.parameters())
    train_prep = fit_keys = None
    if not evaluation or (not learned and version == "v2"):
        train_prep = Prepared(args.prepared_root, args.protocol_dir, "train")
        fit_keys = partition_prepared_keys_by_source(train_prep.rows, seed=args.split_seed)["train_fit"]
        if args.route == "a" and version == "v2" and model.reference_prior is None:
            model.reference_prior = fit_reference_prior(train_prep, fit_keys)
    if evaluation or not learned:
        res = evaluate_filter_records(model, _pack_eval(eval_prep, eval_keys), device=args.device)
        (args.run_dir / f"eval_{name}_{args.eval_split}.json").write_text(
            metrics_json(res["summary"], indent=2) + "\n"
        )
        print(metrics_json(res["summary"]), flush=True)
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    def selection_score(summary):
        if version == "v1":
            return summary["overall"]["observation_nll"]
        return summary["source_equal_summary"]["scaled_mse"]
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
            total_loss = 0.0
            count = 0
            for key in batch_keys:
                record = train_prep.record(key, exact_initial=(version == "v1"))
                if args.arch in ("ppc", "qr", "gaussian"):
                    mask = record.get("valid_mask")
                    frame_count = (
                        len(record["times_sec"]) if mask is None else int(torch.as_tensor(mask).sum())
                    )
                    if not frame_count:
                        raise ValueError("No valid PPC training frame")
                    for frame_loss in model.training_loss_chunks(record, args.unroll_frames):
                        part = frame_loss / frame_count
                        part.backward()
                        total_loss += float(part.detach())
                    count += 1
                    continue
                if args.route == "a":
                    term = train_route_a_vp_step_on_record(
                        model,
                        record,
                        use_rollout_priors=not args.no_rollout_priors,
                        unroll_frames=args.unroll_frames,
                        device=args.device,
                    )
                else:
                    try:
                        term = train_route_b_cosine_flow_step_on_record(
                            model.flow,
                            model.encoder,
                            record,
                            device=args.device,
                            los_budget=model.los_budget,
                        )
                    except EmptyTeacherSupport:
                        skipped += 1
                        continue
                term.backward()
                total_loss += float(term.detach())
                count += 1
            if not count:
                continue
            trained_records += count
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.div_(count)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "cursor": start + len(batch_keys),
                        "train_loss": total_loss / count,
                        "grad_norm": float(gnorm.item()),
                        "teacher_skipped": skipped,
                    }
                ),
                flush=True,
            )

        if trained_records == 0:
            raise ValueError("No usable training record in this epoch")
        val_res = evaluate_filter_records(model, _pack_eval(eval_prep, eval_keys), device=args.device)
        (args.run_dir / f"dev_select_{epoch:02d}.json").write_text(
            metrics_json(val_res["summary"], indent=2) + "\n"
        )
        best_score = save_epoch(
            args.run_dir / f"{name}_epoch_{epoch:02d}.pt",
            last_ckpt,
            model,
            optimizer,
            args,
            config_keys,
            epoch,
            selection_score(val_res["summary"]),
            best_score,
            reference_prior=getattr(model, "reference_prior", None),
        )


__all__ = [
    "main",
    "train_route_a_vp_step_on_record",
    "train_route_b_cosine_flow_step_on_record",
]

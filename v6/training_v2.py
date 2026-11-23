"""Training, DAgger Rollout Replay, and 2-Checkpoint Unrolling for Model Plan V2 (ASD-SC).

Implements the exact Round 4 training specification from
docs/MODEL_V2_ADAPTIVE_DENOISING_PLAN.md (§2.2, §2.4) and
outputs/research_codex/ROUND4_NOISY_INIT_SELFCORRECT_REVIEW.md (§4, commits b8e7855, 3ec2ba0, ae8708d):
1. Counterfactual triplet prior interventions on the same LOS prefix:
   (a) near-truth good prior, (b) observable bad prior, (c) short-arc scale-invariant fake-range prior.
2. Multi-horizon physical forecast risk E_m(x) at {0 s, +60 s, +300 s} via pure two-body propagation.
3. Switchable step-increment progressive supervision L_{prog,l} targeting T_l d_l on (y_{l+1} - y_l)
   (never mixing cumulative and incremental labels!).
4. Pinball quantile loss on blind error log-scales (0.5 and 0.9 quantiles).
5. One-sided good-prior anti-erasure penalty with stop-gradient on the hold baseline E_m(x_m^-).
6. Reset candidate risk + stop-gradient BCE trigger loss.
7. On-policy DAgger rollout state buffer (with static-B0 toggle for V2-Abl-3) + 2-checkpoint
   connected unrolling across 300 s gaps.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .data.prepared import Prepared
from .data.splits import (
    MacroCheckpoint,
    build_macro_checkpoint_schedule,
    partition_prepared_keys_by_source,
)
from .models.batched_physics import state_scale
from .models.reprojection import propagate_si_with_stm
from .models.v2_asd_sc import (
    ASDSCFilter,
    CheckpointStepOutput,
    PendingSubspaceMemory,
    V2_ABLATION_MODES,
    transport_pending_subspace_qr,
)
from .training_v1 import evaluate_filter_records


def generate_counterfactual_prior_triplet(
    truth_state_si: Tensor,
    station_state_si: Tensor,
    *,
    good_pos_std_m: float = 150.0,
    good_vel_std_mps: float = 0.5,
    bad_pos_std_m: float = 8.0e4,
    bad_vel_std_mps: float = 80.0,
    fake_range_scale: float = 1.35,
    generator: torch.Generator | None = None,
) -> dict[str, Tensor]:
    """Build paired counterfactual priors for the exact same raw LOS prefix (V2 §2.2 / V2-Abl-1)."""
    device, dtype = truth_state_si.device, truth_state_si.dtype
    n_good = torch.randn((6,), generator=generator, device=device, dtype=dtype)
    prior_good = truth_state_si + torch.cat(
        (n_good[:3] * good_pos_std_m, n_good[3:] * good_vel_std_mps), dim=0
    )

    n_bad = torch.randn((6,), generator=generator, device=device, dtype=dtype)
    prior_bad = truth_state_si + torch.cat(
        (n_bad[:3] * bad_pos_std_m, n_bad[3:] * bad_vel_std_mps), dim=0
    )

    rel_r = truth_state_si[:3] - station_state_si[:3]
    rel_v = truth_state_si[3:] - station_state_si[3:]
    rho = torch.linalg.vector_norm(rel_r).clamp_min(100.0)
    u0 = rel_r / rho
    v_par = (rel_v * u0).sum() * u0
    v_perp = rel_v - v_par
    beta = float(fake_range_scale)
    prior_fake = torch.cat(
        (
            station_state_si[:3] + beta * rel_r,
            station_state_si[3:] + v_par + beta * v_perp,
        ),
        dim=0,
    )
    return {
        "good": prior_good,
        "bad_observable": prior_bad,
        "fake_range": prior_fake,
    }


def _elementwise_loss(diff: Tensor, *, use_huber: bool = True, huber_delta: float = 0.05) -> Tensor:
    if use_huber:
        return torch.nn.functional.huber_loss(
            diff, torch.zeros_like(diff), delta=float(huber_delta), reduction="mean"
        )
    return diff.square().mean()


def multi_horizon_physical_risk(
    state_si: Tensor,
    truth_state_m_si: Tensor,
    *,
    horizons_sec: Sequence[float] = (60.0, 300.0),
    horizon_weights: Sequence[float] = (0.5, 0.5),
    lambda_forecast: float = 1.0,
    use_huber: bool = True,
    huber_delta: float = 0.05,
) -> Tensor:
    """Compute unmodified physical multi-horizon forecast risk E_m(x) (ROUND4 §4, Eq. 89-92)."""
    d_scale = state_scale(device=state_si.device, dtype=state_si.dtype)
    curr_err = (state_si - truth_state_m_si) / d_scale
    risk = _elementwise_loss(curr_err, use_huber=use_huber, huber_delta=huber_delta)

    if lambda_forecast <= 0.0 or not horizons_sec:
        return risk

    f_loss = torch.zeros_like(risk)
    for h_sec, w_h in zip(horizons_sec, horizon_weights):
        if float(w_h) <= 0.0 or float(h_sec) <= 0.0:
            continue
        pred_h, _ = propagate_si_with_stm(state_si, float(h_sec), compute_stm=False)
        with torch.no_grad():
            truth_h, _ = propagate_si_with_stm(truth_state_m_si.detach(), float(h_sec), compute_stm=False)
        pos_err_scaled = (pred_h[:3] - truth_h[:3]) / 1.0e6
        f_loss = f_loss + float(w_h) * _elementwise_loss(
            pos_err_scaled, use_huber=use_huber, huber_delta=huber_delta
        )
    return risk + float(lambda_forecast) * f_loss


def compute_progressive_step_target(
    V_phys: Tensor,
    singular_values: Tensor,
    d_remaining_scaled: Tensor,
    *,
    gamma: float = 0.5,
) -> Tensor:
    """Compute geometry-driven progressive step-increment target sg(T_l d_l) (commit ae8708d)."""
    if V_phys.numel() == 0 or singular_values.numel() == 0:
        return torch.zeros_like(d_remaining_scaled)
    s_det = singular_values.detach()
    v_det = V_phys.detach()
    tau_i = (s_det * s_det) / (s_det * s_det + float(gamma) * float(gamma))
    proj = v_det.transpose(0, 1) @ d_remaining_scaled.detach()
    return (v_det @ (tau_i * proj)).detach()


def pinball_loss(target: Tensor, pred: Tensor, quantile: float) -> Tensor:
    """Pinball quantile loss rho_p(u) = u * (p - I[u < 0]), u = sg(target) - pred."""
    u = target.detach() - pred
    p = float(quantile)
    return torch.where(u >= 0.0, p * u, (p - 1.0) * u).mean()


@dataclass(frozen=True)
class V2LossWeights:
    """Hyperparameters for the complete V2 checkpoint & 2-step rollout loss."""

    lambda_forecast: float = 1.0
    horizons_sec: tuple[float, ...] = (60.0, 300.0)
    horizon_weights: tuple[float, ...] = (0.5, 0.5)
    lambda_prog: float = 0.25  # set to 0.0 to ablate progressive supervision cleanly
    prog_gamma: float = 0.5
    lambda_quantile: float = 0.05
    lambda_good_harm: float = 2.0
    delta_good_margin: float = 1.0e-5
    good_pos_thresh_m: float = 10.0e3
    good_vel_thresh_mps: float = 25.0
    lambda_reset: float = 0.2
    delta_reset_label: float = 1.0e-4
    use_huber: bool = True
    huber_delta: float = 0.05
    eps_0: float = 1.0e-4


def compute_v2_checkpoint_loss(
    step_out: CheckpointStepOutput,
    truth_state_m_si: Tensor,
    weights: V2LossWeights = V2LossWeights(),
) -> tuple[Tensor, dict[str, float]]:
    """Evaluate exact L_I checkpoint loss (ROUND4 §4 Eq. 100-105 + ae8708d progressive term)."""
    device, dtype = truth_state_m_si.device, truth_state_m_si.dtype
    d_scale = state_scale(device=device, dtype=dtype)
    L = len(step_out.microsteps)

    if L == 0:
        zero = (step_out.published_state_si * 0.0).sum()
        return zero, {"total": 0.0, "phys_risk": 0.0, "prog": 0.0, "quantile": 0.0, "good_harm": 0.0, "reset": 0.0}

    total_micro = torch.zeros((), device=device, dtype=dtype)
    phys_log, prog_log, q_log = 0.0, 0.0, 0.0
    w_ell = 1.0 / float(L)

    for tr in step_out.microsteps:
        e_m = multi_horizon_physical_risk(
            tr.state_next_si,
            truth_state_m_si,
            horizons_sec=weights.horizons_sec,
            horizon_weights=weights.horizon_weights,
            lambda_forecast=weights.lambda_forecast,
            use_huber=weights.use_huber,
            huber_delta=weights.huber_delta,
        )
        x_prev_si = step_out.prior_state_si + d_scale * tr.y_prev
        d_rem = ((truth_state_m_si - x_prev_si) / d_scale).detach()

        l_prog = torch.zeros_like(e_m)
        if weights.lambda_prog > 0.0 and tr.V_phys.numel() > 0:
            target_inc = compute_progressive_step_target(
                tr.V_phys, tr.singular_values, d_rem, gamma=weights.prog_gamma
            )
            l_prog = _elementwise_loss(
                tr.delta_y - target_inc, use_huber=weights.use_huber, huber_delta=weights.huber_delta
            )

        l_quant = torch.zeros_like(e_m)
        if weights.lambda_quantile > 0.0 and tr.V_phys.numel() > 0:
            proj_abs = (tr.V_phys.detach().transpose(0, 1) @ d_rem).abs()
            z_star = torch.log(float(weights.eps_0) + proj_abs).detach()
            l_quant = pinball_loss(z_star, tr.q50_pred, 0.5) + pinball_loss(z_star, tr.q90_pred, 0.9)

        total_micro = total_micro + w_ell * (
            e_m + weights.lambda_prog * l_prog + weights.lambda_quantile * l_quant
        )
        phys_log += w_ell * float(e_m.detach().item())
        prog_log += w_ell * float(l_prog.detach().item())
        q_log += w_ell * float(l_quant.detach().item())

    prior_pos_err = float(torch.linalg.vector_norm(step_out.prior_state_si[:3] - truth_state_m_si[:3]).detach().item())
    prior_vel_err = float(torch.linalg.vector_norm(step_out.prior_state_si[3:] - truth_state_m_si[3:]).detach().item())
    is_good_prior = (prior_pos_err <= weights.good_pos_thresh_m) and (prior_vel_err <= weights.good_vel_thresh_mps)

    good_harm_loss = torch.zeros_like(total_micro)
    if is_good_prior and weights.lambda_good_harm > 0.0:
        with torch.no_grad():
            hold_risk_sg = multi_horizon_physical_risk(
                step_out.prior_state_si.detach(),
                truth_state_m_si,
                horizons_sec=weights.horizons_sec,
                horizon_weights=weights.horizon_weights,
                lambda_forecast=weights.lambda_forecast,
                use_huber=weights.use_huber,
                huber_delta=weights.huber_delta,
            ).detach()
        post_risk = multi_horizon_physical_risk(
            step_out.local_posterior_si,
            truth_state_m_si,
            horizons_sec=weights.horizons_sec,
            horizon_weights=weights.horizon_weights,
            lambda_forecast=weights.lambda_forecast,
            use_huber=weights.use_huber,
            huber_delta=weights.huber_delta,
        )
        good_harm_loss = torch.relu(post_risk - hold_risk_sg - float(weights.delta_good_margin))

    reset_loss = torch.zeros_like(total_micro)
    if (
        weights.lambda_reset > 0.0
        and step_out.reanchor_candidate_si is not None
        and step_out.reset_logit is not None
    ):
        risk_R = multi_horizon_physical_risk(
            step_out.reanchor_candidate_si,
            truth_state_m_si,
            horizons_sec=weights.horizons_sec,
            horizon_weights=weights.horizon_weights,
            lambda_forecast=weights.lambda_forecast,
            use_huber=weights.use_huber,
            huber_delta=weights.huber_delta,
        )
        with torch.no_grad():
            risk_L_sg = multi_horizon_physical_risk(
                step_out.local_posterior_si.detach(),
                truth_state_m_si,
                horizons_sec=weights.horizons_sec,
                horizon_weights=weights.horizon_weights,
                lambda_forecast=weights.lambda_forecast,
                use_huber=weights.use_huber,
                huber_delta=weights.huber_delta,
            )
            label_reset = (
                (risk_R.detach() + float(weights.delta_reset_label)) < risk_L_sg
            ).to(dtype=step_out.reset_logit.dtype)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(step_out.reset_logit, label_reset)
        reset_loss = risk_R + bce

    total = (
        total_micro
        + float(weights.lambda_good_harm) * good_harm_loss
        + float(weights.lambda_reset) * reset_loss
    )
    return total, {
        "total": float(total.detach().item()),
        "phys_risk": phys_log,
        "prog": prog_log,
        "quantile": q_log,
        "good_harm": float(good_harm_loss.detach().item()),
        "reset": float(reset_loss.detach().item()),
    }


@dataclass(frozen=True)
class RolloutPairSample:
    """Learner-visited or static-prior contiguous checkpoint pair (cp_m, cp_{m+1})."""

    record_key: str
    prior_m_si: Tensor
    memory_m: PendingSubspaceMemory
    cp_m: MacroCheckpoint
    cp_next: MacroCheckpoint | None


class DAggerRolloutBuffer:
    """Visited-state replay buffer supporting both on-policy DAgger and static-B0 ablations."""

    def __init__(self, max_pairs: int = 16384) -> None:
        self.max_pairs = int(max_pairs)
        self.samples: list[RolloutPairSample] = []

    def __len__(self) -> int:
        return len(self.samples)

    @torch.no_grad()
    def collect_from_records(
        self,
        model: ASDSCFilter,
        records_by_key: Mapping[str, Mapping[str, Any]],
        keys: Sequence[str],
        *,
        two_step_pairs: bool = True,
        use_dagger_rollout: bool = True,
        counterfactual_prob: float = 0.25,
        seed: int = 20261028,
    ) -> int:
        model.eval()
        rng = random.Random(seed)
        gen = torch.Generator().manual_seed(seed)
        added = 0

        for key in keys:
            rec = records_by_key[key]
            t_init = float(rec["init_time_sec"])
            times = torch.as_tensor(rec["times_sec"], dtype=torch.float64)
            obs = torch.as_tensor(rec["observed_los"], dtype=torch.float64)
            stn = torch.as_tensor(rec["station_states_si"], dtype=torch.float64)
            truth = torch.as_tensor(rec["truth_states_si"], dtype=torch.float64)
            valid = torch.as_tensor(
                rec.get("valid_mask", torch.ones(times.shape[0], dtype=torch.bool)), dtype=torch.bool
            )

            init_candidates = [torch.as_tensor(rec["initial_state_si"], dtype=torch.float64)]
            if counterfactual_prob > 0.0 and rng.random() < counterfactual_prob:
                val_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
                if val_idx.numel() > 0:
                    k0 = int(val_idx[0].item())
                    t0_truth, _ = propagate_si_with_stm(
                        truth[k0], t_init - float(times[k0].item()), compute_stm=False
                    )
                    s0_stn, _ = propagate_si_with_stm(
                        stn[k0], t_init - float(times[k0].item()), compute_stm=False
                    )
                    cf = generate_counterfactual_prior_triplet(t0_truth, s0_stn, generator=gen)
                    init_candidates.extend([cf["good"], cf["bad_observable"], cf["fake_range"]])

            for x0_cand in init_candidates:
                if use_dagger_rollout:
                    res = model.rollout_track(
                        initial_state_si=x0_cand,
                        init_time_sec=t_init,
                        times_sec=times,
                        observed_los=obs,
                        station_states_si=stn,
                        valid_mask=valid,
                    )
                    cps = res["checkpoint_records"]
                    if not cps:
                        continue
                    prev_mem = PendingSubspaceMemory.empty(torch.device("cpu"), torch.float64)
                    for i, entry in enumerate(cps):
                        cp_m: MacroCheckpoint = entry["checkpoint"]
                        if not cp_m.has_new_observations:
                            prev_mem = entry["memory"]
                            continue
                        cp_n = cps[i + 1]["checkpoint"] if (two_step_pairs and i + 1 < len(cps)) else None
                        mem_cpu = PendingSubspaceMemory(
                            V_p=prev_mem.V_p.detach().cpu(),
                            r_diag_log=prev_mem.r_diag_log.detach().cpu(),
                            r_mix_ratio=prev_mem.r_mix_ratio,
                        )
                        self.samples.append(
                            RolloutPairSample(
                                record_key=key,
                                prior_m_si=entry["prior_state_si"].detach().cpu(),
                                memory_m=mem_cpu,
                                cp_m=cp_m,
                                cp_next=cp_n,
                            )
                        )
                        prev_mem = entry["memory"]
                        added += 1
                else:
                    # V2-Abl-3 static B0 baseline: pure un-updated propagation without learner feedback
                    checkpoints, _ = build_macro_checkpoint_schedule(times, valid, init_time_sec=t_init)
                    curr_p = x0_cand
                    curr_tp = t_init
                    empty_mem = PendingSubspaceMemory.empty(torch.device("cpu"), torch.float64)
                    for i, cp_m in enumerate(checkpoints):
                        curr_p, _ = propagate_si_with_stm(
                            curr_p, cp_m.eval_time_sec - curr_tp, compute_stm=False
                        )
                        curr_tp = cp_m.eval_time_sec
                        if not cp_m.has_new_observations:
                            continue
                        cp_n = checkpoints[i + 1] if (two_step_pairs and i + 1 < len(checkpoints)) else None
                        self.samples.append(
                            RolloutPairSample(
                                record_key=key,
                                prior_m_si=curr_p.detach().cpu(),
                                memory_m=empty_mem,
                                cp_m=cp_m,
                                cp_next=cp_n,
                            )
                        )
                        added += 1

        if len(self.samples) > self.max_pairs:
            self.samples = self.samples[-self.max_pairs :]
        return added


def _truth_at_time(
    times_sec: Tensor, truth_states_si: Tensor, valid_mask: Tensor, target_time_sec: float
) -> Tensor:
    val_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
    t_val = times_sec[val_idx]
    nearest_pos = int(torch.argmin((t_val - float(target_time_sec)).abs()).item())
    k_near = int(val_idx[nearest_pos].item())
    dt = float(target_time_sec) - float(times_sec[k_near].item())
    if abs(dt) < 1.0e-9:
        return truth_states_si[k_near]
    x_t, _ = propagate_si_with_stm(truth_states_si[k_near], dt, compute_stm=False)
    return x_t


def train_two_checkpoint_pair(
    model: ASDSCFilter,
    sample: RolloutPairSample,
    record: Mapping[str, Any],
    *,
    loss_weights: V2LossWeights = V2LossWeights(),
    device: torch.device | str = "cpu",
) -> tuple[Tensor, dict[str, float]]:
    """Unroll 2 contiguous checkpoints (cp_m -> propagate/QR -> cp_{m+1}) with connected gradients."""
    dev = torch.device(device)
    dtype = torch.float64
    times = torch.as_tensor(record["times_sec"], dtype=dtype)
    obs = torch.as_tensor(record["observed_los"], device=dev, dtype=dtype)
    stn = torch.as_tensor(record["station_states_si"], device=dev, dtype=dtype)
    truth = torch.as_tensor(record["truth_states_si"], device=dev, dtype=dtype)
    valid = torch.as_tensor(
        record.get("valid_mask", torch.ones(times.shape[0], dtype=torch.bool)), dtype=torch.bool
    )

    prior_m = sample.prior_m_si.to(device=dev, dtype=dtype).detach()
    mem_m = PendingSubspaceMemory(
        V_p=sample.memory_m.V_p.to(device=dev, dtype=dtype).detach(),
        r_diag_log=sample.memory_m.r_diag_log.to(device=dev, dtype=dtype).detach(),
        r_mix_ratio=sample.memory_m.r_mix_ratio,
    )

    out_1 = model.run_checkpoint(
        prior_state_si=prior_m,
        eval_time_sec=sample.cp_m.eval_time_sec,
        times_sec=times,
        observed_los=obs,
        station_states_si=stn,
        valid_mask=valid,
        last_obs_idx=sample.cp_m.last_obs_idx,
        prev_last_obs_idx=sample.cp_m.prev_last_obs_idx,
        memory_in=mem_m,
        has_new_observations=sample.cp_m.has_new_observations,
    )
    truth_1 = _truth_at_time(times, truth, valid, sample.cp_m.eval_time_sec)
    loss_1, stats_1 = compute_v2_checkpoint_loss(out_1, truth_1, weights=loss_weights)

    if sample.cp_next is None or (not sample.cp_next.has_new_observations):
        return loss_1, stats_1

    dt_gap = sample.cp_next.eval_time_sec - sample.cp_m.eval_time_sec
    prior_2, phi_gap = propagate_si_with_stm(
        out_1.local_posterior_si,
        dt_gap,
        compute_stm=(out_1.memory_out.dim_p > 0),
        ledger=model.ledger,
    )
    mem_2 = (
        transport_pending_subspace_qr(out_1.memory_out, phi_gap)
        if (out_1.memory_out.dim_p > 0 and phi_gap is not None)
        else out_1.memory_out
    )

    out_2 = model.run_checkpoint(
        prior_state_si=prior_2,
        eval_time_sec=sample.cp_next.eval_time_sec,
        times_sec=times,
        observed_los=obs,
        station_states_si=stn,
        valid_mask=valid,
        last_obs_idx=sample.cp_next.last_obs_idx,
        prev_last_obs_idx=sample.cp_next.prev_last_obs_idx,
        memory_in=mem_2,
        has_new_observations=sample.cp_next.has_new_observations,
    )
    truth_2 = _truth_at_time(times, truth, valid, sample.cp_next.eval_time_sec)
    loss_2, stats_2 = compute_v2_checkpoint_loss(out_2, truth_2, weights=loss_weights)

    pair_loss = 0.5 * (loss_1 + loss_2)
    merged = {k: 0.5 * (stats_1[k] + stats_2[k]) for k in stats_1}
    return pair_loss, merged


def _parse_horizons(spec: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    cleaned = [float(x.strip()) for x in spec.split(",") if x.strip()]
    if not cleaned:
        return (), ()
    w = tuple(1.0 / len(cleaned) for _ in cleaned)
    return tuple(cleaned), w


def main(evaluation: bool = False) -> None:
    parser = argparse.ArgumentParser(description="Model Plan V2 (ASD-SC) Trainer & Validator")
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ablation-mode", default="full", choices=V2_ABLATION_MODES)
    parser.add_argument("--disable-reanchor", action="store_true", help="V2-Abl-5: disable Tier-2 re-anchor")
    parser.add_argument("--single-step-grad", action="store_true", help="V2-Abl-3: 1-checkpoint vs 2-checkpoint unroll")
    parser.add_argument("--no-dagger-replay", action="store_true", help="V2-Abl-3: static B0 only vs DAgger replay")
    parser.add_argument("--counterfactual-prob", type=float, default=0.25, help="Paired good/bad/fake-range prior prob")
    parser.add_argument("--horizons-sec", default="60,300", help="Comma-separated forecast horizons (e.g. '60,300' or '60')")
    parser.add_argument("--lambda-prog", type=float, default=0.25, help="Progressive step-increment weight (0 disables)")
    parser.add_argument("--lambda-forecast", type=float, default=1.0, help="V2-Abl-4: multi-horizon forecast weight")
    parser.add_argument("--lambda-good-harm", type=float, default=2.0, help="One-sided good-prior anti-erasure weight")
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

    model = ASDSCFilter(
        ablation_mode=args.ablation_mode,
        enable_reanchor=(not args.disable_reanchor),
    ).to(args.device)
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

    def _pack_eval(prep: Prepared, keys: Sequence[str]):
        return [
            (k, str(prep.rows[k]["record_key"]["source_id"]), str(prep.rows[k]["record_key"]["rhythm"]), prep.record(k))
            for k in keys
        ]

    if evaluation:
        res = evaluate_filter_records(model, _pack_eval(eval_prep, eval_keys), device=args.device)
        (args.run_dir / f"eval_v2_{args.eval_split}.json").write_text(json.dumps(res["summary"], indent=2) + "\n")
        print(json.dumps(res["summary"]), flush=True)
        return

    train_prep = Prepared(args.prepared_root, args.protocol_dir, "train")
    fit_keys = partition_prepared_keys_by_source(train_prep.rows, seed=args.seed)["train_fit"]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    h_vals, h_weights = _parse_horizons(args.horizons_sec)
    loss_weights = V2LossWeights(
        horizons_sec=h_vals,
        horizon_weights=h_weights,
        lambda_prog=args.lambda_prog,
        lambda_forecast=args.lambda_forecast if h_vals else 0.0,
        lambda_good_harm=args.lambda_good_harm,
    )
    buffer = DAggerRolloutBuffer(max_pairs=32768)
    best_ep = float("inf")

    for epoch in range(start_epoch, args.epochs + 1):
        rng = random.Random(args.seed + 101 * epoch)
        shuffled_fit = list(fit_keys)
        rng.shuffle(shuffled_fit)
        active_keys = (
            shuffled_fit[: args.max_records_per_epoch]
            if args.max_records_per_epoch > 0
            else shuffled_fit
        )

        # Stream rollout collection & optimization in manageable chunks so RAM stays bounded on 35k records
        chunk_size = 512
        for c_start in range(0, len(active_keys), chunk_size):
            c_keys = active_keys[c_start : c_start + chunk_size]
            rec_cache = {k: train_prep.record(k) for k in c_keys}
            buffer.collect_from_records(
                model,
                rec_cache,
                c_keys,
                two_step_pairs=(not args.single_step_grad),
                use_dagger_rollout=(not args.no_dagger_replay),
                counterfactual_prob=args.counterfactual_prob,
                seed=args.seed + epoch + c_start,
            )

            model.train()
            recent_pool = list(buffer.samples[-min(len(buffer), chunk_size * 8) :])
            rng.shuffle(recent_pool)
            for start in range(0, len(recent_pool), args.batch_size):
                batch_samples = recent_pool[start : start + args.batch_size]
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for smp in batch_samples:
                    rec_obj = rec_cache.get(smp.record_key) or train_prep.record(smp.record_key)
                    lp, _ = train_two_checkpoint_pair(
                        model,
                        smp,
                        rec_obj,
                        loss_weights=loss_weights,
                        device=args.device,
                    )
                    losses.append(lp)
                loss = torch.stack(losses).mean()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "chunk_start": c_start,
                            "pair_cursor": start + len(batch_samples),
                            "train_loss": float(loss.detach().item()),
                            "grad_norm": float(grad_norm.item()),
                        }
                    ),
                    flush=True,
                )

        payload = {
            "model": model.state_dict(),
            "epoch": epoch,
            "ablation_mode": args.ablation_mode,
            "seed": args.seed,
        }
        torch.save(payload, args.run_dir / f"v2_epoch_{epoch:02d}.pt")
        torch.save(payload, last_ckpt)
        val_res = evaluate_filter_records(model, _pack_eval(eval_prep, eval_keys), device=args.device)
        (args.run_dir / f"dev_select_{epoch:02d}.json").write_text(json.dumps(val_res["summary"], indent=2) + "\n")
        overall = val_res["summary"]["overall"]
        score = float(
            overall.get("v2_primary_block2_t30_forecast300_pos_km", overall["position_km"])["capped_mean"]
        )
        if score <= best_ep:
            best_ep = score
            torch.save(payload, args.run_dir / "best.pt")


__all__ = [
    "DAggerRolloutBuffer",
    "RolloutPairSample",
    "V2LossWeights",
    "compute_progressive_step_target",
    "compute_v2_checkpoint_loss",
    "generate_counterfactual_prior_triplet",
    "main",
    "multi_horizon_physical_risk",
    "pinball_loss",
    "train_two_checkpoint_pair",
]

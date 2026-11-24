"""Model Plan V1 (Deterministic & Classical Refiners): CL-GN1, CL-WindowLM, JF-BR, and LTR-LM.

Implements all deterministic baselines and learned refiners co-signed in
docs/JOINT_RESEARCH_PLAN.md (§2.1–§2.5):
1. Classical non-learned differential correction arms (0 neural parameters):
   - CL-GN1-Pure: fixed diagonal damping Lambda_0 + backtracking line search
   - CL-GN1-Matched: identical deadzone alpha_k, obtuse safeguard & line search as LTR-LM
   - CL-WindowLM: multi-step iterative LM on W=16 causal keyframes
   - CL-FullPrefixWLS: multi-step iterative WLS/LM over the full causal prefix
2. Candidate A — Jacobian-Free Bounded Refiners (JF-BR, pure forward, no STM Phi):
   - JF-BR-Vel: bounded physical velocity-only refiner
   - JF-BR-6D: homogeneous bias-free 6D refiner (exact zero step at zero residual)
3. Candidate B — Learned Trust-Region Preconditioned 1-Step LM (LTR-LM, ~15k params):
   - Predicts per-frame reliability weights w_theta in (w_min, 1]^W and 6D diagonal
     damping Lambda_theta, solving (A^T W_b A + Lambda_theta) y_k = alpha_k A^T W_b nu_0
     under all three R2-verified writeback variants:
       * joint_6d    (LTR-LM-Joint6D)
       * schur_vel   (LTR-LM-SchurVel: Schur complement eliminating unknown position)
       * lock_pos_vel (LTR-LM-LockPosVel: locked position, direct velocity fit)
4. V1OnlineFilter: causal multi-block closed-loop filter (both BackboneFree pure RK4
   propagation + keyframe refiner, and ProtectedBackbone wrapping V6 filter).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from ..data.observation_clock import observation_frames
from .batched_physics import DTYPE, state_scale
from .normalization import EARTH_RADIUS_M, T_SCALE_S
from .recursive_filter import TrainingFilterOutput
from .reprojection import (
    ComputeLedger,
    WindowReprojectionResult,
    apply_local_msc_step,
    deadzone_gate_alpha,
    los_noise_profile,
    geodesic_angle_rad,
    propagate_si_with_stm,
    reproject_causal_window_single,
    select_causal_keyframes,
)


WRITEBACK_MODES = ("joint_6d", "schur_vel", "lock_pos_vel")


@dataclass(frozen=True)
class RefinerStepOutput:
    """Detailed single-trigger output for anchor diagnostics and closed-loop rollout."""

    prior_state_si: Tensor  # [6]
    posterior_state_si: Tensor  # [6]
    y_step_local: Tensor  # [6] dimensionless local MSC increment
    alpha_gate: Tensor  # scalar deadzone gate in [0, 1)
    median_angle_rad: float
    tau_rad: float
    window_span_sec: float
    accepted: bool
    frame_weights: Tensor | None = None  # [W]
    damping_diag: Tensor | None = None  # [6]


def solve_damped_normal_equations(
    A: Tensor,
    nu_0: Tensor,
    *,
    frame_weights: Tensor | None = None,
    damping_diag: Tensor,
    alpha_gate: Tensor | float = 1.0,
    writeback_mode: str = "joint_6d",
    obtuse_fraction: float = 0.0,
    obtuse_damping_boost: float = 100.0,
    ledger: ComputeLedger | None = None,
) -> Tensor:
    """Solve sign-verified weighted damped normal equations in float64 via Cholesky.

    Equation:
        (A^T W_b A + Lambda) y = alpha_k A^T W_b nu_0_flat
    where nu(y) ~= nu_0_flat - A y (A = -D_y nu(0) in R^{2W x 6}).
    """
    if writeback_mode not in WRITEBACK_MODES:
        raise ValueError(f"writeback_mode must be one of {WRITEBACK_MODES}, got {writeback_mode!r}")

    device, dtype = A.device, A.dtype
    W = int(nu_0.shape[0])
    if W == 0 or A.numel() == 0:
        return torch.zeros((6,), device=device, dtype=dtype)

    nu_flat = nu_0.reshape(2 * W)
    if frame_weights is None:
        w_b = torch.ones((2 * W,), device=device, dtype=dtype)
    else:
        w_f = frame_weights.reshape(W).to(device=device, dtype=dtype)
        w_b = w_f.repeat_interleave(2)

    lam = damping_diag.reshape(6).to(device=device, dtype=dtype).clamp_min(1.0e-9)
    if obtuse_fraction >= 0.5:
        lam = lam * float(obtuse_damping_boost)

    alpha = torch.as_tensor(alpha_gate, device=device, dtype=dtype)
    Aw = A * w_b.unsqueeze(-1)  # [2W, 6]
    H = A.transpose(0, 1) @ Aw + torch.diag(lam)  # [6, 6]
    rhs = alpha * (Aw.transpose(0, 1) @ nu_flat)  # [6]

    if ledger is not None:
        ledger.cholesky_solves += 1

    if writeback_mode == "joint_6d":
        L = torch.linalg.cholesky(H)
        y = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)
        return y

    if writeback_mode == "lock_pos_vel":
        H_vv = H[3:, 3:]
        b_v = rhs[3:]
        L_vv = torch.linalg.cholesky(H_vv)
        y_v = torch.cholesky_solve(b_v.unsqueeze(-1), L_vv).squeeze(-1)
        return torch.cat((torch.zeros((3,), device=device, dtype=dtype), y_v), dim=0)

    # writeback_mode == "schur_vel": eliminate unknown position via Schur complement
    H_rr = H[:3, :3]
    H_rv = H[:3, 3:]
    H_vr = H[3:, :3]
    H_vv = H[3:, 3:]
    b_r = rhs[:3]
    b_v = rhs[3:]

    L_rr = torch.linalg.cholesky(H_rr)
    H_rr_inv_H_rv = torch.cholesky_solve(H_rv, L_rr)  # [3, 3]
    H_rr_inv_b_r = torch.cholesky_solve(b_r.unsqueeze(-1), L_rr).squeeze(-1)  # [3]

    S_vv = H_vv - H_vr @ H_rr_inv_H_rv
    # Symmetrize numerical rounding before Cholesky
    S_vv = 0.5 * (S_vv + S_vv.transpose(0, 1)) + 1.0e-12 * torch.eye(3, device=device, dtype=dtype)
    rhs_schur = b_v - H_vr @ H_rr_inv_b_r
    L_s = torch.linalg.cholesky(S_vv)
    y_v = torch.cholesky_solve(rhs_schur.unsqueeze(-1), L_s).squeeze(-1)
    return torch.cat((torch.zeros((3,), device=device, dtype=dtype), y_v), dim=0)


def _window_geodesic_cost(
    state_si: Tensor,
    t_k_sec: float,
    station_k_pos_si: Tensor,
    keyframe_times_sec: Tensor,
    keyframe_stations_si: Tensor,
    keyframe_los_obs: Tensor,
    *,
    ledger: ComputeLedger | None = None,
) -> float:
    rep = reproject_causal_window_single(
        state_si,
        t_k_sec,
        station_k_pos_si,
        keyframe_times_sec,
        keyframe_stations_si,
        keyframe_los_obs,
        compute_jacobian=False,
        ledger=ledger,
    )
    return float(rep.angles_rad.square().mean().item()) if rep.angles_rad.numel() > 0 else 0.0


def _build_keyframe_tokens(
    rep: WindowReprojectionResult,
    tau_rad: float,
    dtype: torch.dtype = DTYPE,
) -> tuple[Tensor, Tensor]:
    """Build per-keyframe geometric/residual tokens [W, 10] and pooled summary [14]."""
    device = rep.nu_0.device
    W = int(rep.nu_0.shape[0])
    if W == 0:
        return (
            torch.zeros((0, 10), device=device, dtype=dtype),
            torch.zeros((14,), device=device, dtype=dtype),
        )
    safe_tau = max(float(tau_rad), 1.0e-6)
    dt_norm = rep.dt_from_k_sec / float(T_SCALE_S)  # <= 0
    rho_norm = rep.pred_ranges_m / float(EARTH_RADIUS_M)
    inv_rho = float(EARTH_RADIUS_M) / rep.pred_ranges_m.clamp_min(100.0)
    ang_over_tau = (rep.angles_rad / safe_tau).clamp_max(100.0)
    log_ang_ratio = torch.log1p(rep.angles_rad / safe_tau)
    nu_scaled = (rep.nu_0 / safe_tau).clamp(-100.0, 100.0)
    new_flag = rep.is_new_segment.to(dtype=dtype)

    per_frame = torch.stack(
        (
            nu_scaled[:, 0],
            nu_scaled[:, 1],
            rep.one_minus_cos,
            log_ang_ratio,
            ang_over_tau,
            dt_norm,
            rho_norm,
            inv_rho,
            rep.cos_angles,
            new_flag,
        ),
        dim=-1,
    ).to(dtype=dtype)

    med_ang = torch.median(rep.angles_rad)
    max_ang = rep.angles_rad.amax()
    span_norm = float(rep.window_span_sec) / float(T_SCALE_S)
    mean_feat = per_frame.mean(dim=0)  # [10]
    summary = torch.cat(
        (
            mean_feat,
            torch.stack(
                (
                    torch.log1p(med_ang / safe_tau),
                    torch.log1p(max_ang / safe_tau),
                    torch.tensor(span_norm, device=device, dtype=dtype),
                    torch.tensor(float(W) / 16.0, device=device, dtype=dtype),
                )
            ),
        ),
        dim=0,
    )
    return per_frame, summary


class ClassicalLMRefiner(nn.Module):
    """Zero-parameter classical differential correction baselines (JOINT_RESEARCH_PLAN §2.4.1).

    Modes:
    - 'cl_gn1_pure': 1-step fixed damping Lambda_0 + backtracking line search (no alpha_k deadzone)
    - 'cl_gn1_matched': matched deadzone alpha_k, obtuse safeguard & line search as LTR-LM
    - 'cl_window_lm': up to max_iters=10 adaptive LM iterations on W=16 keyframes
    - 'cl_full_prefix_wls': iterative LM using the full causal prefix up to t_k
    """

    def __init__(
        self,
        variant: str = "cl_gn1_matched",
        *,
        base_damping: Sequence[float] = (1.0e-2, 1.0e-4, 1.0e-4, 1.0e-2, 1.0e-3, 1.0e-3),
        writeback_mode: str = "joint_6d",
        max_keyframes: int = 16,
        max_iters: int = 10,
        line_search_trials: int = 3,
        min_arc_span_sec: float = 5.0,
        dtype: torch.dtype = DTYPE,
    ) -> None:
        super().__init__()
        valid_variants = ("cl_gn1_pure", "cl_gn1_matched", "cl_window_lm", "cl_full_prefix_wls")
        if variant not in valid_variants:
            raise ValueError(f"variant must be one of {valid_variants}, got {variant!r}")
        self.variant = variant
        self.writeback_mode = writeback_mode
        self.max_keyframes = 100000 if variant == "cl_full_prefix_wls" else int(max_keyframes)
        self.max_iters = 1 if variant in ("cl_gn1_pure", "cl_gn1_matched") else int(max_iters)
        self.line_search_trials = int(line_search_trials)
        self.min_arc_span_sec = float(min_arc_span_sec)
        self.register_buffer(
            "base_damping", torch.tensor(base_damping, dtype=dtype), persistent=True
        )

    def refine_anchor(
        self,
        candidate_state_si: Tensor,
        t_k_sec: float,
        station_k_pos_si: Tensor,
        keyframe_times_sec: Tensor,
        keyframe_stations_si: Tensor,
        keyframe_los_obs: Tensor,
        *,
        tau_rad: float,
        ledger: ComputeLedger | None = None,
    ) -> RefinerStepOutput:
        device, dtype = candidate_state_si.device, candidate_state_si.dtype
        W = int(keyframe_times_sec.shape[0])
        if W < 2:
            return RefinerStepOutput(
                prior_state_si=candidate_state_si,
                posterior_state_si=candidate_state_si,
                y_step_local=torch.zeros((6,), device=device, dtype=dtype),
                alpha_gate=torch.tensor(0.0, device=device, dtype=dtype),
                median_angle_rad=0.0,
                tau_rad=tau_rad,
                window_span_sec=0.0,
                accepted=False,
            )

        curr_x = candidate_state_si
        lam_scale = 1.0
        total_y = torch.zeros((6,), device=device, dtype=dtype)
        accepted_any = False
        first_med = 0.0
        first_alpha = torch.tensor(1.0, device=device, dtype=dtype)
        span_sec = 0.0

        for it in range(self.max_iters):
            rep = reproject_causal_window_single(
                curr_x,
                t_k_sec,
                station_k_pos_si,
                keyframe_times_sec,
                keyframe_stations_si,
                keyframe_los_obs,
                compute_jacobian=True,
                ledger=ledger,
            )
            span_sec = rep.window_span_sec
            med_ang = float(torch.median(rep.angles_rad).item())
            if it == 0:
                first_med = med_ang
            if span_sec < self.min_arc_span_sec:
                return RefinerStepOutput(
                    prior_state_si=candidate_state_si,
                    posterior_state_si=candidate_state_si,
                    y_step_local=torch.zeros((6,), device=device, dtype=dtype),
                    alpha_gate=torch.tensor(0.0, device=device, dtype=dtype),
                    median_angle_rad=first_med,
                    tau_rad=tau_rad,
                    window_span_sec=span_sec,
                    accepted=False,
                )

            if self.variant == "cl_gn1_pure":
                alpha = torch.tensor(1.0, device=device, dtype=dtype)
            else:
                alpha = deadzone_gate_alpha(med_ang, tau_rad).to(device=device, dtype=dtype)
                if it == 0:
                    first_alpha = alpha
                if float(alpha.item()) <= 0.0:
                    break

            assert rep.A_local_msc is not None
            obtuse_frac = float((rep.cos_angles <= 0.0).to(torch.float64).mean().item())
            damping = self.base_damping.to(device=device, dtype=dtype) * lam_scale
            y_raw = solve_damped_normal_equations(
                rep.A_local_msc,
                rep.nu_0,
                damping_diag=damping,
                alpha_gate=alpha,
                writeback_mode=self.writeback_mode,
                obtuse_fraction=obtuse_frac,
                ledger=ledger,
            )
            base_cost = float(rep.angles_rad.square().mean().item())

            step_scale = 1.0
            step_accepted = False
            for _ in range(max(1, self.line_search_trials)):
                if ledger is not None:
                    ledger.line_search_trials += 1
                cand_x = apply_local_msc_step(
                    curr_x, station_k_pos_si, step_scale * y_raw, span_sec
                )
                trial_cost = _window_geodesic_cost(
                    cand_x,
                    t_k_sec,
                    station_k_pos_si,
                    keyframe_times_sec,
                    keyframe_stations_si,
                    keyframe_los_obs,
                    ledger=ledger,
                )
                if trial_cost <= base_cost + 1.0e-15:
                    curr_x = cand_x
                    total_y = total_y + step_scale * y_raw
                    accepted_any = True
                    step_accepted = True
                    lam_scale = max(0.1, lam_scale * 0.5)
                    break
                step_scale *= 0.5

            if not step_accepted:
                lam_scale *= 4.0
                if self.max_iters == 1:
                    break
            elif float(torch.linalg.vector_norm(y_raw).item()) < 1.0e-6:
                break

        return RefinerStepOutput(
            prior_state_si=candidate_state_si,
            posterior_state_si=curr_x,
            y_step_local=total_y,
            alpha_gate=first_alpha,
            median_angle_rad=first_med,
            tau_rad=tau_rad,
            window_span_sec=span_sec,
            accepted=accepted_any,
            damping_diag=self.base_damping.to(device=device, dtype=dtype),
        )


class JFBRRefiner(nn.Module):
    """Candidate A: Jacobian-Free Bounded Physical-Scale Refiner (JF-BR-Vel & JF-BR-6D).

    Pure forward lightweight network (~12k params) needing neither STM Phi nor A:
    - mode='vel_only': delta_r = 0, ||delta_v|| <= min(v_max, rho_bar [a_k - tau_k]_+ / max(T_w, T_min))
    - mode='homog_6d': y_k = alpha_k sum_j W_theta(g_{j|k}, g_bar_k) nu_{j|k}(0), strictly zero at nu_0=0!
    """

    def __init__(
        self,
        mode: str = "homog_6d",
        *,
        hidden_dim: int = 64,
        v_max_mps: float = 800.0,
        t_min_sec: float = 30.0,
        min_arc_span_sec: float = 5.0,
        dtype: torch.dtype = DTYPE,
    ) -> None:
        super().__init__()
        if mode not in ("vel_only", "homog_6d"):
            raise ValueError("JFBRRefiner mode must be 'vel_only' or 'homog_6d'")
        self.mode = mode
        self.hidden_dim = int(hidden_dim)
        self.v_max_mps = float(v_max_mps)
        self.t_min_sec = float(t_min_sec)
        self.min_arc_span_sec = float(min_arc_span_sec)

        self.frame_enc = nn.Sequential(
            nn.Linear(10, hidden_dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, dtype=dtype),
        )
        if mode == "vel_only":
            self.vel_head = nn.Sequential(
                nn.Linear(hidden_dim + 14, hidden_dim, dtype=dtype),
                nn.GELU(),
                nn.Linear(hidden_dim, 3, dtype=dtype),
            )
            nn.init.zeros_(self.vel_head[-1].weight)
            nn.init.zeros_(self.vel_head[-1].bias)
        else:
            # Predicts per-frame 6x2 operator W_theta(g_{j|k}, g_bar_k) applied homogeneously to nu_{j|k}(0)
            self.op_head = nn.Sequential(
                nn.Linear(hidden_dim + 14, hidden_dim, dtype=dtype),
                nn.GELU(),
                nn.Linear(hidden_dim, 12, dtype=dtype),
            )
            nn.init.zeros_(self.op_head[-1].weight)
            nn.init.zeros_(self.op_head[-1].bias)

    def refine_anchor(
        self,
        candidate_state_si: Tensor,
        t_k_sec: float,
        station_k_pos_si: Tensor,
        keyframe_times_sec: Tensor,
        keyframe_stations_si: Tensor,
        keyframe_los_obs: Tensor,
        *,
        tau_rad: float,
        ledger: ComputeLedger | None = None,
    ) -> RefinerStepOutput:
        device, dtype = candidate_state_si.device, candidate_state_si.dtype
        p_dtype = self.frame_enc[0].weight.dtype
        W = int(keyframe_times_sec.shape[0])
        if W < 2:
            return RefinerStepOutput(
                prior_state_si=candidate_state_si,
                posterior_state_si=candidate_state_si,
                y_step_local=torch.zeros((6,), device=device, dtype=dtype),
                alpha_gate=torch.tensor(0.0, device=device, dtype=dtype),
                median_angle_rad=0.0,
                tau_rad=tau_rad,
                window_span_sec=0.0,
                accepted=False,
            )

        rep = reproject_causal_window_single(
            candidate_state_si,
            t_k_sec,
            station_k_pos_si,
            keyframe_times_sec,
            keyframe_stations_si,
            keyframe_los_obs,
            compute_jacobian=False,
            ledger=ledger,
        )
        med_ang_t = torch.median(rep.angles_rad)
        med_ang = float(med_ang_t.detach().item())
        if rep.window_span_sec < self.min_arc_span_sec:
            return RefinerStepOutput(
                prior_state_si=candidate_state_si,
                posterior_state_si=candidate_state_si,
                y_step_local=torch.zeros((6,), device=device, dtype=dtype),
                alpha_gate=torch.tensor(0.0, device=device, dtype=dtype),
                median_angle_rad=med_ang,
                tau_rad=tau_rad,
                window_span_sec=rep.window_span_sec,
                accepted=False,
            )

        alpha = deadzone_gate_alpha(med_ang_t, tau_rad).to(device=device, dtype=dtype)
        per_frame, summary = _build_keyframe_tokens(rep, tau_rad, dtype=p_dtype)
        h_f = self.frame_enc(per_frame)  # [W, H]

        if self.mode == "vel_only":
            pooled = h_f.mean(dim=0)
            raw_v = self.vel_head(torch.cat((pooled, summary), dim=-1)).to(dtype=dtype)
            # Smooth algebraic squashing: ||b_theta||_2 < 1 everywhere AND d(b_theta)/d(raw_v)|_0 = I_3
            b_theta = raw_v / torch.sqrt(1.0 + (raw_v * raw_v).sum())

            excess_ang = torch.relu(med_ang_t - float(tau_rad))
            rho_bar = torch.median(rep.pred_ranges_m)
            v_scale_k = (
                torch.linalg.vector_norm(candidate_state_si[:3] - station_k_pos_si[:3]).clamp_min(100.0)
                / max(rep.window_span_sec, self.t_min_sec)
            )
            phys_cap_mps = torch.clamp(
                rho_bar * excess_ang / max(rep.window_span_sec, self.t_min_sec),
                max=self.v_max_mps,
            )
            y_vel = (phys_cap_mps / v_scale_k.clamp_min(1.0e-6)) * b_theta
            y_step = torch.cat((torch.zeros((3,), device=device, dtype=dtype), y_vel), dim=0)
        else:
            ctx = torch.cat((h_f, summary.unsqueeze(0).expand(W, -1)), dim=-1)
            W_op = torch.tanh(self.op_head(ctx)).reshape(W, 6, 2).to(dtype=dtype)
            # Strictly homogeneous in nu_0 with 1/W averaging so W=0 or nu_0=0 -> exact 0
            y_raw = torch.einsum("wij,wj->i", W_op, rep.nu_0) / float(W)
            y_step = alpha * y_raw

        post = apply_local_msc_step(
            candidate_state_si, station_k_pos_si, y_step, rep.window_span_sec, t_min_sec=self.t_min_sec
        )
        accepted = bool((y_step.detach().abs().max().item()) > 0.0)
        return RefinerStepOutput(
            prior_state_si=candidate_state_si,
            posterior_state_si=post,
            y_step_local=y_step,
            alpha_gate=alpha,
            median_angle_rad=med_ang,
            tau_rad=tau_rad,
            window_span_sec=rep.window_span_sec,
            accepted=accepted,
        )


class LTRLMRefiner(nn.Module):
    """Candidate B: Learned Trust-Region Preconditioned 1-Step LM (LTR-LM, ~15k params).

    Predicts per-frame scalar weights w_theta in (w_min, 1]^W and 6D diagonal damping
    Lambda_theta = diag(lambda_min + softplus(f_theta)), solving the sign-verified
    6x6 normal equations under writeback_mode in {'joint_6d', 'schur_vel', 'lock_pos_vel'}.
    """

    def __init__(
        self,
        *,
        writeback_mode: str = "joint_6d",
        hidden_dim: int = 64,
        w_min: float = 0.05,
        lambda_min: Sequence[float] = (1.0e-3, 1.0e-5, 1.0e-5, 1.0e-3, 1.0e-4, 1.0e-4),
        init_damping: Sequence[float] = (5.0e-2, 1.0e-3, 1.0e-3, 5.0e-2, 5.0e-3, 5.0e-3),
        t_min_sec: float = 30.0,
        min_arc_span_sec: float = 5.0,
        eval_line_search_trials: int = 2,
        dtype: torch.dtype = DTYPE,
    ) -> None:
        super().__init__()
        if writeback_mode not in WRITEBACK_MODES:
            raise ValueError(f"writeback_mode must be one of {WRITEBACK_MODES}")
        self.writeback_mode = writeback_mode
        self.hidden_dim = int(hidden_dim)
        self.w_min = float(w_min)
        self.t_min_sec = float(t_min_sec)
        self.min_arc_span_sec = float(min_arc_span_sec)
        self.eval_line_search_trials = int(eval_line_search_trials)

        lam_min_t = torch.tensor(lambda_min, dtype=dtype)
        init_lam_t = torch.tensor(init_damping, dtype=dtype)
        self.register_buffer("lambda_min", lam_min_t, persistent=True)

        # Frame encoder + reliability weight head + 6D damping head (~15k parameters total)
        self.frame_mlp = nn.Sequential(
            nn.Linear(10, hidden_dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, dtype=dtype),
            nn.LayerNorm(hidden_dim, dtype=dtype),
        )
        self.weight_head = nn.Sequential(
            nn.Linear(hidden_dim + 14, hidden_dim // 2, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1, dtype=dtype),
        )
        # Summary + log diagonal of A^T A (6 scalars) -> 6D damping
        self.damping_head = nn.Sequential(
            nn.Linear(hidden_dim + 14 + 6, hidden_dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_dim, 6, dtype=dtype),
        )
        # Initialize heads near matched classical LM baseline
        nn.init.zeros_(self.weight_head[-1].weight)
        nn.init.constant_(self.weight_head[-1].bias, 2.0)  # sigmoid(2) ~= 0.88 near uniform 1
        nn.init.zeros_(self.damping_head[-1].weight)
        diff = (init_lam_t - lam_min_t).clamp_min(1.0e-6)
        inv_softplus = torch.log(torch.expm1(diff))
        with torch.no_grad():
            self.damping_head[-1].bias.copy_(inv_softplus)

    def refine_anchor(
        self,
        candidate_state_si: Tensor,
        t_k_sec: float,
        station_k_pos_si: Tensor,
        keyframe_times_sec: Tensor,
        keyframe_stations_si: Tensor,
        keyframe_los_obs: Tensor,
        *,
        tau_rad: float,
        use_line_search: bool | None = None,
        ledger: ComputeLedger | None = None,
    ) -> RefinerStepOutput:
        device, dtype = candidate_state_si.device, candidate_state_si.dtype
        p_dtype = self.frame_mlp[0].weight.dtype
        W = int(keyframe_times_sec.shape[0])
        if W < 2:
            return RefinerStepOutput(
                prior_state_si=candidate_state_si,
                posterior_state_si=candidate_state_si,
                y_step_local=torch.zeros((6,), device=device, dtype=dtype),
                alpha_gate=torch.tensor(0.0, device=device, dtype=dtype),
                median_angle_rad=0.0,
                tau_rad=tau_rad,
                window_span_sec=0.0,
                accepted=False,
            )

        rep = reproject_causal_window_single(
            candidate_state_si,
            t_k_sec,
            station_k_pos_si,
            keyframe_times_sec,
            keyframe_stations_si,
            keyframe_los_obs,
            compute_jacobian=True,
            t_min_sec=self.t_min_sec,
            ledger=ledger,
        )
        med_ang_t = torch.median(rep.angles_rad)
        med_ang = float(med_ang_t.detach().item())
        if rep.window_span_sec < self.min_arc_span_sec:
            return RefinerStepOutput(
                prior_state_si=candidate_state_si,
                posterior_state_si=candidate_state_si,
                y_step_local=torch.zeros((6,), device=device, dtype=dtype),
                alpha_gate=torch.tensor(0.0, device=device, dtype=dtype),
                median_angle_rad=med_ang,
                tau_rad=tau_rad,
                window_span_sec=rep.window_span_sec,
                accepted=False,
            )

        alpha = deadzone_gate_alpha(med_ang_t, tau_rad).to(device=device, dtype=dtype)
        assert rep.A_local_msc is not None
        A = rep.A_local_msc

        per_frame, summary = _build_keyframe_tokens(rep, tau_rad, dtype=p_dtype)
        h_f = self.frame_mlp(per_frame)  # [W, H]
        w_logits = self.weight_head(
            torch.cat((h_f, summary.unsqueeze(0).expand(W, -1)), dim=-1)
        ).squeeze(-1)
        w_theta = (self.w_min + (1.0 - self.w_min) * torch.sigmoid(w_logits)).to(dtype=dtype)

        ata_diag = (A * A).sum(dim=0).clamp_min(1.0e-18)
        log_ata = torch.log10(ata_diag).to(dtype=p_dtype) / 10.0
        damp_in = torch.cat((h_f.mean(dim=0), summary, log_ata), dim=-1)
        lam_raw = self.damping_head(damp_in).to(dtype=dtype)
        lam_theta = self.lambda_min.to(device=device, dtype=dtype) + nn.functional.softplus(lam_raw)

        obtuse_frac = float((rep.cos_angles.detach() <= 0.0).to(torch.float64).mean().item())
        y_step = solve_damped_normal_equations(
            A,
            rep.nu_0,
            frame_weights=w_theta,
            damping_diag=lam_theta,
            alpha_gate=alpha,
            writeback_mode=self.writeback_mode,
            obtuse_fraction=obtuse_frac,
            ledger=ledger,
        )

        do_ls = (not self.training and self.eval_line_search_trials > 1) if use_line_search is None else bool(use_line_search)
        if not do_ls or float(alpha.detach().item()) <= 0.0:
            post = apply_local_msc_step(
                candidate_state_si, station_k_pos_si, y_step, rep.window_span_sec, t_min_sec=self.t_min_sec
            )
            accepted = float(alpha.detach().item()) > 0.0
        else:
            base_cost = float(rep.angles_rad.detach().square().mean().item())
            scale_try = 1.0
            post = candidate_state_si
            accepted = False
            final_y = torch.zeros_like(y_step)
            for _ in range(self.eval_line_search_trials):
                if ledger is not None:
                    ledger.line_search_trials += 1
                cand_x = apply_local_msc_step(
                    candidate_state_si,
                    station_k_pos_si,
                    scale_try * y_step,
                    rep.window_span_sec,
                    t_min_sec=self.t_min_sec,
                )
                trial_cost = _window_geodesic_cost(
                    cand_x,
                    t_k_sec,
                    station_k_pos_si,
                    keyframe_times_sec,
                    keyframe_stations_si,
                    keyframe_los_obs,
                    ledger=ledger,
                )
                if trial_cost <= base_cost + 1.0e-12:
                    post = cand_x
                    final_y = scale_try * y_step
                    accepted = True
                    break
                scale_try *= 0.5
            y_step = final_y

        return RefinerStepOutput(
            prior_state_si=candidate_state_si,
            posterior_state_si=post,
            y_step_local=y_step,
            alpha_gate=alpha,
            median_angle_rad=med_ang,
            tau_rad=tau_rad,
            window_span_sec=rep.window_span_sec,
            accepted=accepted,
            frame_weights=w_theta,
            damping_diag=lam_theta,
        )


class PriorProtectionGate(nn.Module):
    """Pre-update baseline protection gate preventing k=1 / short-arc good-orbit corruption.

    When causal arc span T_{w,k} < min_arc_sec or single-frame angular residual <= c_protect * tau_k,
    blocks destructive unconstrained base updates (e.g. preserving 0.092 km B0 instead of jumping to 28.37 km).
    """

    def __init__(self, min_arc_sec: float = 10.0, c_protect: float = 1.5) -> None:
        super().__init__()
        self.min_arc_sec = float(min_arc_sec)
        self.c_protect = float(c_protect)

    def should_hold_prior(
        self, elapsed_arc_sec: float, current_angle_rad: float, tau_rad: float
    ) -> bool:
        if elapsed_arc_sec < self.min_arc_sec and current_angle_rad <= max(self.c_protect * tau_rad, 5.0e-3):
            return True
        return current_angle_rad <= tau_rad


class V1OnlineFilter(nn.Module):
    """Causal multi-block online filter wrapping any V1 refiner (BackboneFree or ProtectedBackbone).

    Compatible with v6's collate_filter_sequences batch contract via `forward_filter_sequence`.
    """

    def __init__(
        self,
        refiner: ClassicalLMRefiner | JFBRRefiner | LTRLMRefiner,
        *,
        base_filter: nn.Module | None = None,
        max_keyframes: int | None = None,
        prior_gate: PriorProtectionGate | None = None,
        default_sigma_rad: float = 5.0e-4,
        c_tau: float = 3.0,
    ) -> None:
        super().__init__()
        self.refiner = refiner
        self.base_filter = base_filter
        self.max_keyframes = (
            int(getattr(refiner, "max_keyframes", 16))
            if max_keyframes is None
            else int(max_keyframes)
        )
        self.prior_gate = prior_gate or PriorProtectionGate()
        self.default_sigma_rad = float(default_sigma_rad)
        self.c_tau = float(c_tau)
        self.ledger = ComputeLedger()

    def rollout_track(
        self, *, initial_state_si: Tensor, init_time_sec: float, times_sec: Tensor,
        observed_los: Tensor, station_states_si: Tensor, station_ids: Tensor | None = None,
        station_id_reliable: Tensor | None = None, valid_mask: Tensor | None = None,
        detach_feedback: bool = False,
    ) -> dict[str, Any]:
        """Propagate the last posterior and refine the causal prefix at each valid arrival."""
        x0 = initial_state_si.to(torch.float64)
        times = times_sec.detach().cpu().double()
        valid = torch.ones_like(times, dtype=torch.bool) if valid_mask is None else valid_mask.detach().cpu().bool()
        obs, stations = observed_los.to(x0), station_states_si.to(x0)
        sigmas, _ = los_noise_profile(
            times,
            obs,
            station_ids,
            station_id_reliable,
            valid,
            default_sigma_rad=self.default_sigma_rad,
            station_states_si=stations,
        )
        anchor = x0
        priors = [torch.zeros_like(x0)] * len(times)
        posts = priors.copy()
        accepted = 0
        base_step = getattr(self.base_filter, "step", None)
        base_propagate = getattr(self.base_filter, "propagate", None)
        if self.base_filter is not None and not callable(base_step) and not callable(base_propagate):
            raise TypeError("base_filter must provide step(previous_state, dt, los, station) or propagate(state, dt)")
        first_time = None
        for frame in observation_frames(times, valid, init_time_sec):
            k, tm, dt = frame.index, frame.time_sec, frame.dt_sec
            if first_time is None:
                first_time = tm
            if callable(base_propagate):
                prior = base_propagate(anchor, dt)
            else:
                prior, _ = propagate_si_with_stm(anchor, dt, compute_stm=False, ledger=self.ledger)
            post = prior
            tau = self.c_tau * sigmas[k]
            if callable(base_step):
                angle = float(geodesic_angle_rad(prior[:3] - stations[k, :3], obs[k]).detach())
                if not self.prior_gate.should_hold_prior(tm - first_time, angle, tau):
                    post = base_step(anchor, dt, obs[k], stations[k])
            idx, _ = select_causal_keyframes(
                times,
                valid,
                up_to_idx=k,
                prev_checkpoint_idx=frame.previous_index,
                max_keyframes=self.max_keyframes,
            )
            step = self.refiner.refine_anchor(
                post,
                tm,
                stations[k, :3],
                times[idx].to(x0),
                stations[idx],
                obs[idx],
                tau_rad=tau,
                ledger=self.ledger,
            )
            post = step.posterior_state_si
            accepted += int(step.accepted)
            priors[k], posts[k] = prior, post
            anchor = post.detach() if detach_feedback else post
        return dict(
            prior_states_si=torch.stack(priors),
            posterior_states_si=torch.stack(posts),
            accepted_triggers=accepted,
            ledger=self.ledger.as_dict(),
        )

    def forward_single_track(self, **kwargs) -> tuple[Tensor, Tensor, dict[str, Any]]:
        result = self.rollout_track(**kwargs)
        return result["prior_states_si"], result["posterior_states_si"], result

    def forward_filter_sequence(
        self,
        *,
        initial_state_si: Tensor,
        init_time_sec: Tensor | float,
        times_sec: Tensor,
        observed_los: Tensor,
        station_states_si: Tensor,
        station_ids: Tensor | None = None,
        station_id_reliable: Tensor | None = None,
        valid_mask: Tensor | None = None,
        time_plan: Any = None,
        mode: str = "train_fast",
        output_mode: str = "training_minimal",
    ) -> TrainingFilterOutput:
        """Batched wrapper matching GeometryConditionedFilter.forward_filter_sequence."""
        unbatched = initial_state_si.ndim == 1
        init_x = initial_state_si.unsqueeze(0) if unbatched else initial_state_si
        batch = int(init_x.shape[0])
        device = next(self.parameters(), init_x).device
        init_x = init_x.to(device=device, dtype=torch.float64)

        init_t = torch.as_tensor(init_time_sec, dtype=torch.float64).reshape(-1)
        if init_t.numel() == 1:
            init_t = init_t.expand(batch)
        t_seq = times_sec.unsqueeze(0) if times_sec.ndim == 1 else times_sec
        obs_seq = observed_los.unsqueeze(0) if observed_los.ndim == 2 else observed_los
        stn_seq = station_states_si.unsqueeze(0) if station_states_si.ndim == 2 else station_states_si
        val_seq = (
            torch.ones(t_seq.shape, dtype=torch.bool)
            if valid_mask is None
            else (valid_mask.unsqueeze(0) if valid_mask.ndim == 1 else valid_mask)
        )
        sid_seq = None if station_ids is None else (station_ids.unsqueeze(0) if station_ids.ndim == 1 else station_ids)
        srel_seq = (
            None
            if station_id_reliable is None
            else (station_id_reliable.unsqueeze(0) if station_id_reliable.ndim == 1 else station_id_reliable)
        )

        priors_b, posts_b, pred_los_b, innov_b = [], [], [], []
        scale = state_scale(device=device, dtype=torch.float64)

        for b in range(batch):
            pr, po, _ = self.forward_single_track(
                initial_state_si=init_x[b],
                init_time_sec=float(init_t[b].item()),
                times_sec=t_seq[b],
                observed_los=obs_seq[b].to(device=device),
                station_states_si=stn_seq[b].to(device=device),
                station_ids=None if sid_seq is None else sid_seq[b],
                station_id_reliable=None if srel_seq is None else srel_seq[b],
                valid_mask=val_seq[b],
            )
            vb = val_seq[b].to(device=device)
            obs_b = obs_seq[b].to(device=device, dtype=torch.float64)
            stn_b = stn_seq[b].to(device=device, dtype=torch.float64)
            rel = pr[:, :3] - stn_b[:, :3]
            u_hat = rel / torch.linalg.vector_norm(rel, dim=-1, keepdim=True).clamp_min(1.0)
            u_hat = torch.where(vb.unsqueeze(-1), u_hat, torch.zeros_like(u_hat))
            inn = torch.where(vb.unsqueeze(-1), obs_b - u_hat, torch.zeros_like(obs_b))
            priors_b.append(pr)
            posts_b.append(po)
            pred_los_b.append(u_hat)
            innov_b.append(inn)

        priors = torch.stack(priors_b, dim=0)
        posts = torch.stack(posts_b, dim=0)
        preds = torch.stack(pred_los_b, dim=0)
        innovs = torch.stack(innov_b, dim=0)
        corr_si = posts - priors
        corr_scaled = corr_si / scale
        val_dev = val_seq.to(device=device)

        if unbatched:
            priors, posts, corr_si, corr_scaled, preds, innovs, val_dev = (
                x.squeeze(0) for x in (priors, posts, corr_si, corr_scaled, preds, innovs, val_dev)
            )
        return TrainingFilterOutput(
            prior_states_si=priors,
            posterior_states_si=posts,
            gains=None,
            correction_scaled=corr_scaled,
            correction_si=corr_si,
            innovations=innovs,
            predicted_los=preds,
            valid_mask=val_dev,
            metadata={"model_family": "v1_online_refiner", "refiner": type(self.refiner).__name__},
        )


__all__ = [
    "ClassicalLMRefiner",
    "JFBRRefiner",
    "LTRLMRefiner",
    "PriorProtectionGate",
    "RefinerStepOutput",
    "V1OnlineFilter",
    "WRITEBACK_MODES",
    "solve_damped_normal_equations",
]

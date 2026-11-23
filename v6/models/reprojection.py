"""Differentiable two-body STM propagation, spherical LOS geometry, and causal noise calibration.

Shared mathematical kernel for Model Plan V1 (CL-GN1, JF-BR, LTR-LM, FD-ASF) and
Model Plan V2 (ASD-SC):
1. Exact discrete RK4 tangent-linear State Transition Matrix Phi(t_j, t_k) in float64.
2. Singularity-free right-handed orthonormal LOS tangent frames U = [u_hat, q_1, q_2] in SO(3).
3. Spherical exponential retraction Exp_{S^2} and log-range MSC local coordinate Jacobian B_k.
4. Sign-verified forward sensitivity A = -D_y nu(0) such that nu(y) ~= nu_0 - A y.
5. Causal unequal-time interpolation LOS noise floor estimator tau_k (compatible with pois10).
6. Causal time-stratified keyframe selection (up to W=16) with old/new segment tagging.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from .batched_physics import (
    DTYPE,
    MAX_SUBSTEP_SEC,
    MU_EARTH_M3_S2,
    STATE_SCALE_VALUES,
    state_scale,
)


@dataclass
class ComputeLedger:
    """Transparent accounting of RK4 substeps, STM propagations, and trial evaluations."""

    rk4_substeps: int = 0
    stm_substeps: int = 0
    reprojection_passes: int = 0
    line_search_trials: int = 0
    cholesky_solves: int = 0

    def reset(self) -> None:
        self.rk4_substeps = 0
        self.stm_substeps = 0
        self.reprojection_passes = 0
        self.line_search_trials = 0
        self.cholesky_solves = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "rk4_substeps": int(self.rk4_substeps),
            "stm_substeps": int(self.stm_substeps),
            "reprojection_passes": int(self.reprojection_passes),
            "line_search_trials": int(self.line_search_trials),
            "cholesky_solves": int(self.cholesky_solves),
        }


def build_los_frame(u_unit: Tensor) -> tuple[Tensor, Tensor]:
    """Construct a branchless right-handed SO(3) frame U = [u, q1, q2] and E_perp = [q1, q2].

    Uses the singularity-free Duff et al. (2017) construction so no point on S^2 divides by zero,
    followed by a single Gram-Schmidt polish to machine precision (~1e-16 in float64).

    Args:
        u_unit: Tensor [..., 3] of nonzero LOS vectors.

    Returns:
        U: Tensor [..., 3, 3] with columns [u_hat, q1, q2] and det(U) = +1.
        E_perp: Tensor [..., 3, 2] with columns [q1, q2] satisfying E_perp^T u_hat = 0.
    """
    u = u_unit / torch.linalg.vector_norm(u_unit, dim=-1, keepdim=True).clamp_min(1.0e-15)
    ux, uy, uz = u[..., 0], u[..., 1], u[..., 2]
    sign = torch.where(uz >= 0.0, torch.ones_like(uz), -torch.ones_like(uz))
    a = -1.0 / (sign + uz)
    b = ux * uy * a
    q1 = torch.stack((1.0 + sign * ux * ux * a, sign * b, -sign * ux), dim=-1)
    # Polish orthogonality & unit norm to machine precision
    q1 = q1 - (q1 * u).sum(dim=-1, keepdim=True) * u
    q1 = q1 / torch.linalg.vector_norm(q1, dim=-1, keepdim=True).clamp_min(1.0e-15)
    q2 = torch.linalg.cross(u, q1, dim=-1)
    q2 = q2 / torch.linalg.vector_norm(q2, dim=-1, keepdim=True).clamp_min(1.0e-15)
    E_perp = torch.stack((q1, q2), dim=-1)  # [..., 3, 2]
    U = torch.stack((u, q1, q2), dim=-1)  # [..., 3, 3]
    return U, E_perp


def safe_sinc(theta: Tensor) -> Tensor:
    """Compute sin(theta)/theta smoothly across theta = 0."""
    small = theta.abs() < 1.0e-4
    t2 = theta * theta
    series = 1.0 - t2 / 6.0 + (t2 * t2) / 120.0
    safe_theta = torch.where(small, torch.ones_like(theta), theta)
    return torch.where(small, series, torch.sin(safe_theta) / safe_theta)


def spherical_exp_map_delta(u_base: Tensor, delta_tangent_3d: Tensor) -> Tensor:
    """Exact incremental retraction Exp_{S^2}(u, d) - u = -2 sin^2(||d||/2) u + sinc(||d||) d.

    Bitwise zero at d = 0 while preserving exact Frechet derivative I_tangent at d = 0
    and avoiding catastrophic cancellation for small angular steps!
    """
    u = u_base / torch.linalg.vector_norm(u_base, dim=-1, keepdim=True).clamp_min(1.0e-15)
    d = delta_tangent_3d - (delta_tangent_3d * u).sum(dim=-1, keepdim=True) * u
    theta = torch.linalg.vector_norm(d, dim=-1, keepdim=True)
    half_sin = torch.sin(0.5 * theta)
    return (-2.0 * half_sin * half_sin) * u + safe_sinc(theta) * d


def spherical_exp_map(u_base: Tensor, delta_tangent_3d: Tensor) -> Tensor:
    """Exponential retraction on S^2: u_base + spherical_exp_map_delta(u_base, d)."""
    u = u_base / torch.linalg.vector_norm(u_base, dim=-1, keepdim=True).clamp_min(1.0e-15)
    res = u + spherical_exp_map_delta(u, delta_tangent_3d)
    return res / torch.linalg.vector_norm(res, dim=-1, keepdim=True).clamp_min(1.0e-15)


def geodesic_angle_rad(u_pred: Tensor, u_obs: Tensor) -> Tensor:
    """Antipodal-safe monotonic geodesic angle atan2(||u_pred x u_obs||, u_pred . u_obs) in [0, pi]."""
    up = u_pred / torch.linalg.vector_norm(u_pred, dim=-1, keepdim=True).clamp_min(1.0e-15)
    uo = u_obs / torch.linalg.vector_norm(u_obs, dim=-1, keepdim=True).clamp_min(1.0e-15)
    cross_norm = torch.linalg.vector_norm(torch.linalg.cross(up, uo, dim=-1), dim=-1)
    dot = (up * uo).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.atan2(cross_norm, dot)


def _two_body_rhs_and_jacobian(
    x: Tensor, dphi: Tensor | None, mu: float = MU_EARTH_M3_S2
) -> tuple[Tensor, Tensor | None]:
    """Evaluate two-body ODE f(x) and exact tangent linear action Df(x) @ dphi."""
    r = x[:, :3]
    v = x[:, 3:]
    r_norm = torch.linalg.vector_norm(r, dim=1, keepdim=True).clamp_min(1.0)
    inv_r3 = mu / r_norm.pow(3)
    acc = -inv_r3 * r
    fx = torch.cat((v, acc), dim=1)
    if dphi is None:
        return fx, None
    r_hat = r / r_norm
    dr = dphi[:, :3, :]  # [B, 3, 6]
    dv = dphi[:, 3:, :]  # [B, 3, 6]
    # G(r) dr = -(mu/r^3) [dr - 3 (r_hat^T dr) r_hat]
    r_hat_dot_dr = (r_hat.unsqueeze(-1) * dr).sum(dim=1, keepdim=True)  # [B, 1, 6]
    dacc = -inv_r3.unsqueeze(-1) * (dr - 3.0 * r_hat_dot_dr * r_hat.unsqueeze(-1))
    df_dphi = torch.cat((dv, dacc), dim=1)
    return fx, df_dphi


def propagate_si_with_stm(
    state_si: Tensor,
    dt_sec: Tensor | float,
    *,
    max_substep_sec: float = MAX_SUBSTEP_SEC,
    compute_stm: bool = True,
    valid_mask: Tensor | None = None,
    ledger: ComputeLedger | None = None,
) -> tuple[Tensor, Tensor | None]:
    """Propagate batched SI states [B, 6] (and exact RK4 tangent STM [B, 6, 6]) over dt_sec.

    Phi(t + dt, t) is the exact analytical Frechet derivative of the discrete RK4 step
    with respect to state_si, matching finite differences to ~1e-11 in float64.
    """
    squeeze = state_si.ndim == 1
    x = state_si.unsqueeze(0) if squeeze else state_si
    if x.ndim != 2 or x.shape[1] != 6:
        raise ValueError(f"state_si must have shape [6] or [B,6], got {tuple(state_si.shape)}")
    batch = x.shape[0]
    device, dtype = x.device, x.dtype

    dt = torch.as_tensor(dt_sec, device=device, dtype=dtype)
    if dt.ndim == 0:
        dt = dt.expand(batch)
    elif dt.shape != (batch,):
        raise ValueError(f"dt_sec must be scalar or [B], got {tuple(dt.shape)}")

    if valid_mask is None:
        valid = torch.ones((batch,), device=device, dtype=torch.bool)
    else:
        valid = torch.as_tensor(valid_mask, device=device, dtype=torch.bool).reshape(batch)

    dt_cpu = dt.detach().to("cpu", dtype=torch.float64)
    valid_cpu = valid.detach().to("cpu")
    steps_cpu = torch.ceil(dt_cpu.abs() / float(max_substep_sec)).to(torch.int64)
    steps_cpu = torch.where(valid_cpu & (dt_cpu != 0.0), steps_cpu.clamp_min(1), torch.zeros_like(steps_cpu))
    max_steps = int(steps_cpu.max().item()) if steps_cpu.numel() > 0 else 0
    total_active_steps = int(steps_cpu.sum().item())

    if ledger is not None:
        ledger.rk4_substeps += total_active_steps
        if compute_stm:
            ledger.stm_substeps += total_active_steps

    steps = steps_cpu.to(device=device)
    h = dt / steps.clamp_min(1).to(dtype=dtype)

    phi: Tensor | None = None
    if compute_stm:
        phi = torch.eye(6, device=device, dtype=dtype).unsqueeze(0).expand(batch, 6, 6).clone()

    for k in range(max_steps):
        active = valid & (steps > k)
        active6 = active[:, None]
        safe_x = torch.where(active6, x, torch.zeros_like(x))
        safe_phi = None
        if phi is not None:
            active66 = active[:, None, None]
            safe_phi = torch.where(active66, phi, torch.zeros_like(phi))

        h1 = h[:, None]
        k1, dk1 = _two_body_rhs_and_jacobian(safe_x, safe_phi)
        k2, dk2 = _two_body_rhs_and_jacobian(
            safe_x + 0.5 * h1 * k1,
            None if safe_phi is None else safe_phi + 0.5 * h1.unsqueeze(-1) * dk1,
        )
        k3, dk3 = _two_body_rhs_and_jacobian(
            safe_x + 0.5 * h1 * k2,
            None if safe_phi is None else safe_phi + 0.5 * h1.unsqueeze(-1) * dk2,
        )
        k4, dk4 = _two_body_rhs_and_jacobian(
            safe_x + h1 * k3,
            None if safe_phi is None else safe_phi + h1.unsqueeze(-1) * dk3,
        )
        x_next = safe_x + (h1 / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        x = torch.where(active6, x_next, x)
        if phi is not None and dk1 is not None and dk2 is not None and dk3 is not None and dk4 is not None:
            phi_next = safe_phi + (h1.unsqueeze(-1) / 6.0) * (dk1 + 2.0 * dk2 + 2.0 * dk3 + dk4)
            phi = torch.where(active[:, None, None], phi_next, phi)

    if squeeze:
        return x.squeeze(0), (None if phi is None else phi.squeeze(0))
    return x, phi


def local_msc_chart_basis(
    state_si: Tensor,
    station_pos_si: Tensor,
    window_span_sec: Tensor | float,
    *,
    t_min_sec: float = 30.0,
    rho_min_m: float = 100.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build V1 local LOS Log-MSC chart basis B_k = blkdiag(rho_k U_k, v_scale_k U_k).

    Args:
        state_si: [B, 6] candidate posterior/prior in SI (m, m/s).
        station_pos_si: [B, 3] observer position at t_k in SI (m).
        window_span_sec: scalar or [B] causal keyframe span T_{w,k} in seconds.

    Returns:
        B_k: [B, 6, 6] Jacobian d x_k / d y_k at y_k = 0.
        U_k: [B, 3, 3] right-handed LOS frame [u_{k|k}, q_{1,k}, q_{2,k}].
        rho_k: [B] predicted slant range ||r_k - s_k|| in metres.
        v_scale_k: [B] velocity scaling rho_k / max(T_{w,k}, T_min) in m/s.
    """
    squeeze = state_si.ndim == 1
    x = state_si.unsqueeze(0) if squeeze else state_si
    s = station_pos_si.unsqueeze(0) if station_pos_si.ndim == 1 else station_pos_si
    batch = x.shape[0]
    rel_r = x[:, :3] - s[:, :3]
    rho_k = torch.linalg.vector_norm(rel_r, dim=-1).clamp_min(rho_min_m)
    u_k = rel_r / rho_k.unsqueeze(-1)
    U_k, _ = build_los_frame(u_k)

    tw = torch.as_tensor(window_span_sec, device=x.device, dtype=x.dtype)
    if tw.ndim == 0:
        tw = tw.expand(batch)
    v_scale_k = rho_k / tw.clamp_min(float(t_min_sec))

    B_k = torch.zeros((batch, 6, 6), device=x.device, dtype=x.dtype)
    B_k[:, :3, :3] = rho_k[:, None, None] * U_k
    B_k[:, 3:, 3:] = v_scale_k[:, None, None] * U_k
    if squeeze:
        return B_k.squeeze(0), U_k.squeeze(0), rho_k.squeeze(0), v_scale_k.squeeze(0)
    return B_k, U_k, rho_k, v_scale_k


def apply_local_msc_step(
    state_si: Tensor,
    station_pos_si: Tensor,
    y_step: Tensor,
    window_span_sec: Tensor | float,
    *,
    t_min_sec: float = 30.0,
    rho_min_m: float = 100.0,
    max_log_rho_step: float = 2.5,
) -> Tensor:
    """Retract dimensionless local step y_k = [d_eta_rho, d_eta_1, d_eta_2, d_v_R, d_v_1, d_v_2] to ECI SI.

    Guaranteed exact identity (x^+ == x) when y_step == 0, positive slant range via exp(d_eta_rho),
    and unit-sphere retraction via Exp_{S^2}.
    """
    squeeze = state_si.ndim == 1
    x = state_si.unsqueeze(0) if squeeze else state_si
    s = station_pos_si.unsqueeze(0) if station_pos_si.ndim == 1 else station_pos_si
    y = y_step.unsqueeze(0) if y_step.ndim == 1 else y_step

    _, U_k, rho_k, v_scale_k = local_msc_chart_basis(
        x, s, window_span_sec, t_min_sec=t_min_sec, rho_min_m=rho_min_m
    )
    u_k = U_k[:, :, 0]
    q1_k = U_k[:, :, 1]
    q2_k = U_k[:, :, 2]

    d_eta_rho = y[:, 0].clamp(-max_log_rho_step, max_log_rho_step)
    d_tangent = y[:, 1:2] * q1_k + y[:, 2:3] * q2_k
    du = spherical_exp_map_delta(u_k, d_tangent)
    u_new = u_k + du
    # Exact cancellation-free increment: s + rho exp(d_eta) u_new = x[:3] + rho * (expm1(d_eta) * u_new + du)
    dr = rho_k.unsqueeze(-1) * (torch.expm1(d_eta_rho).unsqueeze(-1) * u_new + du)
    r_new = x[:, :3] + dr
    v_new = x[:, 3:] + v_scale_k.unsqueeze(-1) * torch.bmm(U_k, y[:, 3:].unsqueeze(-1)).squeeze(-1)

    out = torch.cat((r_new, v_new), dim=-1)
    return out.squeeze(0) if squeeze else out


@dataclass(frozen=True)
class WindowReprojectionResult:
    """Outputs of backward two-body reprojection over W causal keyframes."""

    nu_0: Tensor  # [W, 2] observed-minus-predicted tangent residuals
    angles_rad: Tensor  # [W] antipodal-safe geodesic angles in [0, pi]
    cos_angles: Tensor  # [W] dot(u_pred, u_obs) in [-1, 1]
    one_minus_cos: Tensor  # [W] 1 - cos_angles in [0, 2]
    pred_los: Tensor  # [W, 3] predicted unit LOS vectors
    pred_ranges_m: Tensor  # [W] predicted slant ranges
    E_perp: Tensor  # [W, 3, 2] frozen tangent bases at y = 0
    dt_from_k_sec: Tensor  # [W] t_j - t_k <= 0
    is_new_segment: Tensor  # [W] bool tag for newly arrived observations
    A_eci_si: Tensor | None  # [2W, 6] sensitivity wrt ECI SI delta_x_k: nu(dx) ~= nu_0 - A_eci_si dx
    A_scaled_d: Tensor | None  # [2W, 6] sensitivity wrt D-scaled y: nu(D y) ~= nu_0 - A_scaled_d y
    A_local_msc: Tensor | None  # [2W, 6] sensitivity wrt V1 local MSC y_k: nu(y_k) ~= nu_0 - A_local_msc y_k
    window_span_sec: float


def select_causal_keyframes(
    times_sec: Tensor | np.ndarray,
    valid_mask: Tensor | np.ndarray | None = None,
    *,
    up_to_idx: int,
    prev_checkpoint_idx: int = -1,
    max_keyframes: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Select up to W causal valid observation indices <= up_to_idx, preserving early, middle & recent frames.

    Returns:
        selected_indices: int64 numpy array [W] in strictly increasing time order.
        is_new_segment: bool numpy array [W] marking indices > prev_checkpoint_idx.
    """
    t_all = np.asarray(times_sec, dtype=np.float64).reshape(-1)
    if valid_mask is None:
        mask = np.ones_like(t_all, dtype=bool)
    else:
        mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if up_to_idx < 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=bool)
    prefix_mask = mask[: up_to_idx + 1]
    candidates = np.flatnonzero(prefix_mask)
    if candidates.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=bool)
    if candidates.size <= max_keyframes:
        chosen = candidates.astype(np.int64)
        return chosen, (chosen > int(prev_checkpoint_idx))

    # Reserve up to 4 recent frames (including newly arrived frames) + 12 time-stratified span anchors
    n_recent = min(4, max_keyframes // 4, candidates.size)
    recent = candidates[-n_recent:]
    pool = candidates[:-n_recent]
    n_span = max_keyframes - n_recent
    t_pool = t_all[pool]
    if t_pool[-1] > t_pool[0] + 1.0e-9 and n_span > 1:
        targets = np.linspace(t_pool[0], t_pool[-1], n_span)
        picked: list[int] = []
        used = set(int(i) for i in recent)
        for target in targets:
            pos = int(np.argmin(np.abs(t_pool - target)))
            # Nudge if already picked
            if int(pool[pos]) in used:
                order = np.argsort(np.abs(t_pool - target))
                for alt in order:
                    if int(pool[alt]) not in used:
                        pos = int(alt)
                        break
            idx_val = int(pool[pos])
            if idx_val not in used:
                picked.append(idx_val)
                used.add(idx_val)
        chosen = np.array(sorted(picked + [int(i) for i in recent]), dtype=np.int64)
    else:
        lin = np.linspace(0, candidates.size - 1, max_keyframes).round().astype(np.int64)
        chosen = np.unique(candidates[lin])
    return chosen, (chosen > int(prev_checkpoint_idx))


def reproject_causal_window_single(
    state_k_si: Tensor,
    t_k_sec: float,
    station_k_pos_si: Tensor,
    keyframe_times_sec: Tensor,
    keyframe_stations_si: Tensor,
    keyframe_los_obs: Tensor,
    *,
    is_new_segment: Tensor | None = None,
    compute_jacobian: bool = True,
    frozen_E_perp: Tensor | None = None,
    max_substep_sec: float = MAX_SUBSTEP_SEC,
    t_min_sec: float = 30.0,
    ledger: ComputeLedger | None = None,
) -> WindowReprojectionResult:
    """Single-pass reverse two-body RK4 reprojection from t_k through W causal keyframes.

    Strictly implements the R2/R4 verified sign convention:
        nu_{j|k}(0) = E_{perp,j}^T (u_j - u_hat_{j|k})
        A = -D_y nu(0) = + (1 / rho_{j|k}) E_{perp,j}^T P_j^perp [I_3, 0] Phi(t_j, t_k) B_chart
    so first-order Taylor expansion is nu(y) ~= nu_0 - A y.
    """
    device, dtype = state_k_si.device, state_k_si.dtype
    W = int(keyframe_times_sec.shape[0])
    if W == 0:
        empty2 = torch.zeros((0, 2), device=device, dtype=dtype)
        empty1 = torch.zeros((0,), device=device, dtype=dtype)
        empty3 = torch.zeros((0, 3), device=device, dtype=dtype)
        empty32 = torch.zeros((0, 3, 2), device=device, dtype=dtype)
        emptyA = torch.zeros((0, 6), device=device, dtype=dtype) if compute_jacobian else None
        return WindowReprojectionResult(
            nu_0=empty2,
            angles_rad=empty1,
            cos_angles=empty1,
            one_minus_cos=empty1,
            pred_los=empty3,
            pred_ranges_m=empty1,
            E_perp=empty32,
            dt_from_k_sec=empty1,
            is_new_segment=torch.zeros((0,), device=device, dtype=torch.bool),
            A_eci_si=emptyA,
            A_scaled_d=emptyA,
            A_local_msc=emptyA,
            window_span_sec=0.0,
        )

    if ledger is not None:
        ledger.reprojection_passes += 1

    # Single reverse sweep from t_k down to keyframe_times_sec[0] (since keyframes are sorted ascending)
    curr_x = state_k_si
    curr_t = float(t_k_sec)
    curr_phi = torch.eye(6, device=device, dtype=dtype) if compute_jacobian else None

    states_rev: list[Tensor] = []
    phis_rev: list[Tensor] = []
    t_list = keyframe_times_sec.detach().cpu().tolist()

    for j in range(W - 1, -1, -1):
        target_t = float(t_list[j])
        dt_step = target_t - curr_t
        if abs(dt_step) > 0.0:
            curr_x, step_phi = propagate_si_with_stm(
                curr_x,
                dt_step,
                max_substep_sec=max_substep_sec,
                compute_stm=compute_jacobian,
                ledger=ledger,
            )
            if compute_jacobian and step_phi is not None and curr_phi is not None:
                # Chain rule: Phi(t_j, t_k) = Phi(t_j, t_{j+1}) @ Phi(t_{j+1}, t_k)
                curr_phi = step_phi @ curr_phi
        curr_t = target_t
        states_rev.append(curr_x)
        if compute_jacobian and curr_phi is not None:
            phis_rev.append(curr_phi)

    states_w = torch.stack(states_rev[::-1], dim=0)  # [W, 6]
    rel_r = states_w[:, :3] - keyframe_stations_si[:, :3]  # [W, 3]
    rho_w = torch.linalg.vector_norm(rel_r, dim=-1).clamp_min(10.0)  # [W]
    u_pred = rel_r / rho_w.unsqueeze(-1)  # [W, 3]
    u_obs = keyframe_los_obs / torch.linalg.vector_norm(keyframe_los_obs, dim=-1, keepdim=True).clamp_min(1.0e-15)

    if frozen_E_perp is not None:
        E_perp = frozen_E_perp
    else:
        _, E_perp = build_los_frame(u_pred.detach())

    # Observed minus predicted chord residual in frozen tangent basis: [W, 2]
    diff_u = u_obs - u_pred
    nu_0 = torch.einsum("wci,wc->wi", E_perp, diff_u)
    cos_ang = (u_pred * u_obs).sum(dim=-1).clamp(-1.0, 1.0)
    angles = geodesic_angle_rad(u_pred, u_obs)
    dt_from_k = keyframe_times_sec.to(device=device, dtype=dtype) - float(t_k_sec)
    span_sec = float(max(0.0, float(t_k_sec) - float(t_list[0])))

    if is_new_segment is None:
        new_mask = torch.zeros((W,), device=device, dtype=torch.bool)
    else:
        new_mask = torch.as_tensor(is_new_segment, device=device, dtype=torch.bool).reshape(W)

    A_eci: Tensor | None = None
    A_scaled: Tensor | None = None
    A_msc: Tensor | None = None
    if compute_jacobian:
        phis_w = torch.stack(phis_rev[::-1], dim=0)  # [W, 6, 6]
        # P_j^perp = I_3 - u_pred u_pred^T; project position block of Phi(t_j, t_k)
        phi_r = phis_w[:, :3, :]  # [W, 3, 6]
        u_dot_phir = (u_pred.unsqueeze(-1) * phi_r).sum(dim=1, keepdim=True)  # [W, 1, 6]
        p_perp_phir = phi_r - u_pred.unsqueeze(-1) * u_dot_phir  # [W, 3, 6]
        # A_j_eci = (1 / rho_j) E_{perp,j}^T P_j^perp [I_3, 0] Phi(t_j, t_k): [W, 2, 6]
        A_w_eci = torch.einsum("wca,wcb->wab", E_perp, p_perp_phir) / rho_w[:, None, None]
        A_eci = A_w_eci.reshape(2 * W, 6)

        d_vec = state_scale(device=device, dtype=dtype)  # [6]
        A_scaled = A_eci * d_vec.unsqueeze(0)

        B_k, _, _, _ = local_msc_chart_basis(
            state_k_si, station_k_pos_si, span_sec, t_min_sec=t_min_sec
        )
        A_msc = A_eci @ B_k

    return WindowReprojectionResult(
        nu_0=nu_0,
        angles_rad=angles,
        cos_angles=cos_ang,
        one_minus_cos=1.0 - cos_ang,
        pred_los=u_pred,
        pred_ranges_m=rho_w,
        E_perp=E_perp,
        dt_from_k_sec=dt_from_k,
        is_new_segment=new_mask,
        A_eci_si=A_eci,
        A_scaled_d=A_scaled,
        A_local_msc=A_msc,
        window_span_sec=span_sec,
    )


def estimate_online_los_noise_floor(
    times_sec: Tensor | np.ndarray,
    observed_los: Tensor | np.ndarray,
    station_ids: Tensor | np.ndarray | None = None,
    station_id_reliable: Tensor | np.ndarray | None = None,
    valid_mask: Tensor | np.ndarray | None = None,
    *,
    up_to_idx: int,
    dt_cal_max_sec: float = 25.0,
    kappa_dist: float = 0.8493218002880191,  # 1 / sqrt(2 ln 2)
    c_tau: float = 3.0,
    n_min_triplets: int = 3,
    default_sigma_rad: float = 5.0e-4,
) -> tuple[float, float, int]:
    """Causal unequal-time triplet interpolation angular noise-floor estimator (pois10-safe).

    Uses only triplets (m-1, m, m+1) with m+1 <= up_to_idx, same station, and total span
    t_{m+1} - t_{m-1} <= dt_cal_max_sec:
        a_m = (t_{m+1} - t_m) / (t_{m+1} - t_{m-1}),  b_m = 1 - a_m
        xi_m = E_{perp,m}^T (u_m - a_m u_{m-1} - b_m u_{m+1})
        sigma_hat = kappa_dist * median( ||xi_m||_2 / sqrt(1 + a_m^2 + b_m^2) )

    Returns:
        tau_k: deadzone threshold c_tau * sigma_hat (radians).
        sigma_hat: estimated per-axis angular noise std (radians).
        num_triplets: number of valid causal triplets |C_k|.
    """
    if up_to_idx < 2:
        sigma = float(default_sigma_rad)
        return float(c_tau * sigma), sigma, 0

    t_all = np.asarray(times_sec, dtype=np.float64).reshape(-1)[: up_to_idx + 1]
    u_all = np.asarray(observed_los, dtype=np.float64).reshape(-1, 3)[: up_to_idx + 1]
    if valid_mask is None:
        v_all = np.ones(t_all.shape, dtype=bool)
    else:
        v_all = np.asarray(valid_mask, dtype=bool).reshape(-1)[: up_to_idx + 1]

    idx = np.flatnonzero(v_all)
    if idx.size < 3:
        sigma = float(default_sigma_rad)
        return float(c_tau * sigma), sigma, 0

    t = t_all[idx]
    u = u_all[idx]
    u = u / np.clip(np.linalg.norm(u, axis=-1, keepdims=True), 1.0e-15, None)

    t_prev, t_mid, t_next = t[:-2], t[1:-1], t[2:]
    span = t_next - t_prev
    good = (span > 1.0e-6) & (span <= float(dt_cal_max_sec)) & (t_mid > t_prev) & (t_next > t_mid)

    if station_ids is not None and station_id_reliable is not None:
        sids = np.asarray(station_ids).reshape(-1)[: up_to_idx + 1][idx]
        srel = np.asarray(station_id_reliable, dtype=bool).reshape(-1)[: up_to_idx + 1][idx]
        same_station = (
            (~srel[:-2])
            | (~srel[1:-1])
            | (~srel[2:])
            | ((sids[:-2] == sids[1:-1]) & (sids[1:-1] == sids[2:]))
        )
        good = good & same_station

    triplet_pos = np.flatnonzero(good)
    n_triplets = int(triplet_pos.size)
    if n_triplets < int(n_min_triplets):
        sigma = float(default_sigma_rad)
        return float(c_tau * sigma), sigma, n_triplets

    tp = t_prev[triplet_pos]
    tm = t_mid[triplet_pos]
    tn = t_next[triplet_pos]
    sp = tn - tp
    a_m = (tn - tm) / sp
    b_m = (tm - tp) / sp

    u_m = torch.as_tensor(u[1:-1][triplet_pos], dtype=torch.float64)
    u_p = torch.as_tensor(u[:-2][triplet_pos], dtype=torch.float64)
    u_n = torch.as_tensor(u[2:][triplet_pos], dtype=torch.float64)
    a_t = torch.as_tensor(a_m, dtype=torch.float64).unsqueeze(-1)
    b_t = torch.as_tensor(b_m, dtype=torch.float64).unsqueeze(-1)

    _, E_perp_m = build_los_frame(u_m)
    detrended_3d = u_m - a_t * u_p - b_t * u_n
    xi_m = torch.einsum("mci,mc->mi", E_perp_m, detrended_3d)
    norm_factor = torch.sqrt(1.0 + a_t.squeeze(-1).square() + b_t.squeeze(-1).square())
    scaled_norms = (torch.linalg.vector_norm(xi_m, dim=-1) / norm_factor).numpy()

    sigma_hat = float(kappa_dist * float(np.median(scaled_norms)))
    sigma_hat = max(sigma_hat, 1.0e-7)
    return float(c_tau * sigma_hat), sigma_hat, n_triplets


def deadzone_gate_alpha(median_angle_rad: Tensor | float, tau_rad: Tensor | float) -> Tensor:
    """Smooth noise-floor gate alpha_k = [a_k - tau_k]_+^2 / (tau_k^2 + [a_k - tau_k]_+^2) in [0, 1)."""
    a = torch.as_tensor(median_angle_rad, dtype=torch.float64)
    tau = torch.as_tensor(tau_rad, dtype=torch.float64, device=a.device).clamp_min(1.0e-12)
    excess = torch.relu(a - tau)
    ex2 = excess * excess
    return ex2 / (tau * tau + ex2)


__all__ = [
    "ComputeLedger",
    "WindowReprojectionResult",
    "apply_local_msc_step",
    "build_los_frame",
    "deadzone_gate_alpha",
    "estimate_online_los_noise_floor",
    "geodesic_angle_rad",
    "local_msc_chart_basis",
    "propagate_si_with_stm",
    "reproject_causal_window_single",
    "safe_sinc",
    "select_causal_keyframes",
    "spherical_exp_map",
]

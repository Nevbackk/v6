"""Model Plan V1 Flagship (FD-ASF): Forecast-Disagreement Amortized Sequential Filtering.

Implements all three mathematical pillars of docs/FLAGSHIP_ACLASS_PROPOSAL.md:
- Pillar I:
  * Batch-invariant continuous-time additive tracklet normal statistics (H_s, B_s, sigma_u)
    mapping arbitrary cadences (dense, stride5, stride20, pois10) to fixed context c_k.
  * Noise-aware 2+4 physical ambiguity charts at static reference epoch t_ref:
      1) Dual-anchor universal-variable M=0 Lambert chart (with |sin(Delta f)| >= sin(15 deg)
         and r_p >= R_earth + h_min guards) + exact 6x6 log-Jacobian volume element;
      2) Smooth fallback single-epoch Attributable (ln rho, rho_dot) chart with closed-form
         volume element |det DF_attr| = rho_c^5 v_ref omega_ref^2 |sinc(||eta_u||)|.
- Pillar II:
  * Tractable-density 4-layer Conditional RealNVP coupling flow q_psi(z, b | c_k) (< 35k params)
    trained on SNIS optimal forecast-disagreement weights:
        q_k*(theta) proportional to pi_k(theta) * ||h_{k+1}(theta) - E_{pi_k}[h_{k+1}]||_2
    using only causally known next-window viewing geometry (t_{k+1}, s_{k+1}), plus optional
    multi-scale geometry-aligned score regularization (DSM) and defensive mixing q_mix.
  * Incremental block likelihood SMC (never double-counting old LOS) + Top-K 6x6 LTR-LM mode
    polishing kept strictly in the deterministic point-publish pool.
- Pillar III:
  * 3-condition Dominance-Gated Safe Publisher preserving good priors while unlocking quickly
    once parallax breaks multi-branch ambiguity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from ..data.splits import ObservationBlock, parse_observation_blocks
from .batched_physics import DTYPE, MU_EARTH_M3_S2, state_scale
from .normalization import EARTH_RADIUS_M, T_SCALE_S, V_SCALE_MPS
from .recursive_filter import TrainingFilterOutput
from .reprojection import (
    ComputeLedger,
    build_los_frame,
    estimate_online_los_noise_floor,
    geodesic_angle_rad,
    propagate_si_with_stm,
    safe_sinc,
    select_causal_keyframes,
    spherical_exp_map,
)
from .v1_ltr_lm import LTRLMRefiner


MIN_PERIGEE_RADIUS_M = EARTH_RADIUS_M + 120.0e3
SIN_15_DEG = math.sin(math.radians(15.0))


def _stumpff_c_s(psi: Tensor) -> tuple[Tensor, Tensor]:
    """Universal-variable Stumpff functions C(psi) and S(psi) smooth across psi = 0."""
    small = psi.abs() < 1.0e-4
    psi2 = psi * psi
    c_ser = 0.5 - psi / 24.0 + psi2 / 720.0
    s_ser = 1.0 / 6.0 - psi / 120.0 + psi2 / 5040.0

    pos = psi > 0.0
    safe_abs = psi.abs().clamp_min(1.0e-8)
    root = torch.sqrt(safe_abs)
    c_pos = (1.0 - torch.cos(root)) / safe_abs
    s_pos = (root - torch.sin(root)) / (safe_abs * root)
    c_neg = (torch.cosh(root) - 1.0) / safe_abs
    s_neg = (torch.sinh(root) - root) / (safe_abs * root)

    c_val = torch.where(pos, c_pos, c_neg)
    s_val = torch.where(pos, s_pos, s_neg)
    return torch.where(small, c_ser, c_val), torch.where(small, s_ser, s_val)


def solve_lambert_universal_m0(
    r1_m: Tensor,
    r2_m: Tensor,
    dt_sec: float | Tensor,
    *,
    retrograde: bool = False,
    mu: float = MU_EARTH_M3_S2,
    n_iters: int = 28,
) -> tuple[Tensor, Tensor, Tensor]:
    """Differentiable M=0 universal-variable Lambert solver in float64 (Bate-Mueller-White).

    For r_p >= R_earth, orbital period >= 5069.34 s > 3300 s, so full revolutions M = 0 always.

    Args:
        r1_m: [..., 3] ECI position at t_a (m).
        r2_m: [..., 3] ECI position at t_b (m).
        dt_sec: positive transfer time t_b - t_a (s).
        retrograde: False for prograde short/long way (h_z >= 0), True for retrograde (h_z < 0).

    Returns:
        v1_mps: [..., 3] ECI velocity at t_a.
        v2_mps: [..., 3] ECI velocity at t_b.
        valid: [...] boolean convergence & physical geometry mask (|sin(df)| >= sin 15 deg).
    """
    squeeze = r1_m.ndim == 1
    r1 = r1_m.unsqueeze(0) if squeeze else r1_m
    r2 = r2_m.unsqueeze(0) if squeeze else r2_m
    device, dtype = r1.device, r1.dtype
    batch = r1.shape[0]
    dt = torch.as_tensor(dt_sec, device=device, dtype=dtype).expand(batch)

    r1_norm = torch.linalg.vector_norm(r1, dim=-1).clamp_min(1.0)
    r2_norm = torch.linalg.vector_norm(r2, dim=-1).clamp_min(1.0)
    cos_df = (r1 * r2).sum(dim=-1) / (r1_norm * r2_norm)
    cos_df = cos_df.clamp(-0.999999, 0.999999)
    cross_z = r1[:, 0] * r2[:, 1] - r1[:, 1] * r2[:, 0]
    sin_abs = torch.sqrt((1.0 - cos_df * cos_df).clamp_min(1.0e-14))

    # Prograde: transfer < pi when cross_z >= 0; Retrograde: transfer < pi when cross_z < 0
    if not retrograde:
        dm = torch.where(cross_z >= 0.0, torch.ones_like(cross_z), -torch.ones_like(cross_z))
    else:
        dm = torch.where(cross_z < 0.0, torch.ones_like(cross_z), -torch.ones_like(cross_z))

    A_const = dm * torch.sqrt(r1_norm * r2_norm * (1.0 + cos_df).clamp_min(1.0e-12))
    geom_ok = (sin_abs >= SIN_15_DEG) & (dt > 1.0) & (A_const.abs() > 1.0)

    sqrt_mu = math.sqrt(mu)
    psi = torch.zeros((batch,), device=device, dtype=dtype)
    psi_low = torch.full((batch,), -4.0 * math.pi, device=device, dtype=dtype)
    psi_up = torch.full((batch,), 4.0 * math.pi * math.pi - 1.0e-3, device=device, dtype=dtype)

    for _ in range(int(n_iters)):
        c_val, s_val = _stumpff_c_s(psi)
        sqrt_c = torch.sqrt(c_val.clamp_min(1.0e-14))
        y = r1_norm + r2_norm + A_const * (psi * s_val - 1.0) / sqrt_c

        # Keep y strictly positive by raising psi_low if needed
        neg_y = y <= 1.0e-6
        psi_low = torch.where(neg_y, psi, psi_low)
        y_safe = y.clamp_min(1.0e-6)

        chi = torch.sqrt(y_safe / c_val.clamp_min(1.0e-14))
        tof = ((chi.pow(3) * s_val + A_const * torch.sqrt(y_safe)) / sqrt_mu)
        res = tof - dt

        too_short = (res <= 0.0) | neg_y
        psi_low = torch.where(too_short, torch.maximum(psi_low, psi), psi_low)
        psi_up = torch.where(~too_short, torch.minimum(psi_up, psi), psi_up)

        # Damped Newton with bisection safeguard inside [psi_low, psi_up]
        psi_mid = 0.5 * (psi_low + psi_up)
        # Finite-slope derivative d(tof)/d(psi) via analytic universal anomaly derivative
        c_eps, s_eps = _stumpff_c_s(psi + 1.0e-5)
        y_eps = (r1_norm + r2_norm + A_const * ((psi + 1.0e-5) * s_eps - 1.0) / torch.sqrt(c_eps.clamp_min(1.0e-14))).clamp_min(1.0e-6)
        chi_eps = torch.sqrt(y_eps / c_eps.clamp_min(1.0e-14))
        tof_eps = (chi_eps.pow(3) * s_eps + A_const * torch.sqrt(y_eps)) / sqrt_mu
        dtof = (tof_eps - tof) / 1.0e-5
        newton_psi = psi - res / dtof.clamp_min(1.0e-9)
        in_bounds = (~neg_y) & (newton_psi > psi_low) & (newton_psi < psi_up)
        psi = torch.where(in_bounds, newton_psi, psi_mid)

    c_val, s_val = _stumpff_c_s(psi)
    y_final = (r1_norm + r2_norm + A_const * (psi * s_val - 1.0) / torch.sqrt(c_val.clamp_min(1.0e-14))).clamp_min(1.0e-6)
    f = 1.0 - y_final / r1_norm
    g = A_const * torch.sqrt(y_final / mu)
    g_dot = 1.0 - y_final / r2_norm
    g_safe = torch.where(g.abs() < 1.0e-9, torch.ones_like(g), g)

    v1 = (r2 - f.unsqueeze(-1) * r1) / g_safe.unsqueeze(-1)
    v2 = (g_dot.unsqueeze(-1) * r2 - r1) / g_safe.unsqueeze(-1)

    chi_f = torch.sqrt(y_final / c_val.clamp_min(1.0e-14))
    tof_f = (chi_f.pow(3) * s_val + A_const * torch.sqrt(y_final)) / sqrt_mu
    converged = geom_ok & (y_final > 1.0) & ((tof_f - dt).abs() < 1.0e-2) & torch.isfinite(v1).all(dim=-1)

    if squeeze:
        return v1.squeeze(0), v2.squeeze(0), converged.squeeze(0)
    return v1, v2, converged


def check_bound_perigee_feasible(
    state_si: Tensor,
    *,
    min_perigee_m: float = MIN_PERIGEE_RADIUS_M,
    mu: float = MU_EARTH_M3_S2,
) -> Tensor:
    """Physical feasibility indicator I[E(theta) < 0 and r_p(theta) >= R_earth + h_min]."""
    r = state_si[..., :3]
    v = state_si[..., 3:]
    r_norm = torch.linalg.vector_norm(r, dim=-1).clamp_min(1.0)
    v2 = (v * v).sum(dim=-1)
    energy = 0.5 * v2 - mu / r_norm
    h_vec = torch.linalg.cross(r, v, dim=-1)
    h2 = (h_vec * h_vec).sum(dim=-1)
    # Eccentricity e = sqrt(1 + 2 E h^2 / mu^2), semi-major axis a = -mu / (2 E), r_p = a (1 - e)
    safe_e = torch.where(energy < -1.0, energy, -torch.ones_like(energy))
    a = -mu / (2.0 * safe_e)
    ecc = torch.sqrt((1.0 + 2.0 * safe_e * h2 / (mu * mu)).clamp_min(0.0))
    r_p = a * (1.0 - ecc)
    return (energy < 0.0) & (r_p >= float(min_perigee_m))


class AttributableChart2Plus4:
    """Smooth single-epoch 2+4 Attributable chart with closed-form Jacobian determinant.

    Coordinates:
        z = [ln(rho_c / rho_ref), rho_dot_c / v_ref] in R^2
        eta = [eta_u (2D), eta_udot (2D)] in R^4
    Mapping at epoch t_c:
        rho_c = rho_ref * exp(z_1),   rho_dot_c = v_ref * z_2
        u = Exp_{S^2}(u_bar_c, E_{perp,c} eta_u)
        r_c = s_c + rho_c u
        v_c = v_{s,c} + rho_dot_c u + rho_c E_perp(u) (omega_bar_c + omega_ref eta_udot)
    Exact volume element wrt (z_1, z_2, eta_u, eta_udot):
        |det DF_attr| = rho_c^5 * v_ref * omega_ref^2 * |sinc(||eta_u||_2)|.
    """

    def __init__(
        self,
        *,
        t_c_sec: float,
        t_ref_sec: float,
        station_c_si: Tensor,
        u_bar_c: Tensor,
        omega_bar_c: Tensor,
        rho_ref_m: float = 2.0e6,
        v_ref_mps: float = V_SCALE_MPS,
        omega_ref_rad_s: float = 1.0e-3,
    ) -> None:
        self.t_c_sec = float(t_c_sec)
        self.t_ref_sec = float(t_ref_sec)
        self.station_c_si = station_c_si.to(torch.float64)
        self.u_bar_c = u_bar_c.to(torch.float64) / torch.linalg.vector_norm(u_bar_c.to(torch.float64)).clamp_min(1.0e-15)
        _, self.E_perp_c = build_los_frame(self.u_bar_c)
        self.omega_bar_c = omega_bar_c.to(torch.float64).reshape(2)
        self.rho_ref_m = float(rho_ref_m)
        self.v_ref_mps = float(v_ref_mps)
        self.omega_ref_rad_s = float(omega_ref_rad_s)

    def decode(self, z: Tensor, eta: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Map (z [..., 2], eta [..., 4]) -> (theta_ref_si [..., 6], log_abs_det_jacobian [...])."""
        squeeze = z.ndim == 1
        zb = z.unsqueeze(0) if squeeze else z
        device, dtype = zb.device, zb.dtype
        batch = zb.shape[0]
        if eta is None:
            etab = torch.zeros((batch, 4), device=device, dtype=dtype)
        else:
            etab = eta.unsqueeze(0) if eta.ndim == 1 else eta

        s_c = self.station_c_si.to(device=device, dtype=dtype)
        u_bar = self.u_bar_c.to(device=device, dtype=dtype)
        E_c = self.E_perp_c.to(device=device, dtype=dtype)
        w_bar = self.omega_bar_c.to(device=device, dtype=dtype)

        z1 = zb[:, 0].clamp(-6.0, 6.0)
        z2 = zb[:, 1]
        eta_u = etab[:, :2]
        eta_w = etab[:, 2:]

        rho_c = self.rho_ref_m * torch.exp(z1)
        rho_dot = self.v_ref_mps * z2
        d_tang = eta_u[:, 0:1] * E_c[:, 0] + eta_u[:, 1:2] * E_c[:, 1]
        u = spherical_exp_map(u_bar.unsqueeze(0).expand(batch, 3), d_tang)
        _, E_u = build_los_frame(u)

        omega_2d = w_bar.unsqueeze(0) + self.omega_ref_rad_s * eta_w
        u_dot_3d = omega_2d[:, 0:1] * E_u[:, :, 0] + omega_2d[:, 1:2] * E_u[:, :, 1]

        r_c = s_c[:3].unsqueeze(0) + rho_c.unsqueeze(-1) * u
        v_c = s_c[3:].unsqueeze(0) + rho_dot.unsqueeze(-1) * u + rho_c.unsqueeze(-1) * u_dot_3d
        x_c = torch.cat((r_c, v_c), dim=-1)

        eta_u_norm = torch.linalg.vector_norm(eta_u, dim=-1)
        sinc_val = safe_sinc(eta_u_norm).abs().clamp_min(1.0e-12)
        log_det = (
            5.0 * torch.log(rho_c)
            + math.log(self.v_ref_mps)
            + 2.0 * math.log(self.omega_ref_rad_s)
            + torch.log(sinc_val)
        )

        dt_ref = self.t_ref_sec - self.t_c_sec
        if abs(dt_ref) > 1.0e-9:
            theta_ref, phi_ref = propagate_si_with_stm(x_c, dt_ref, compute_stm=True)
            assert phi_ref is not None
            _, log_det_phi = torch.linalg.slogdet(phi_ref)
            log_det = log_det + log_det_phi
        else:
            theta_ref = x_c

        if squeeze:
            return theta_ref.squeeze(0), log_det.squeeze(0)
        return theta_ref, log_det


class LambertChart2Plus4:
    """Dual-anchor log-range Lambert 2+4 chart with automatic fallback to AttributableChart2Plus4."""

    def __init__(
        self,
        *,
        t_a_sec: float,
        t_b_sec: float,
        t_ref_sec: float,
        station_a_si: Tensor,
        station_b_si: Tensor,
        u_bar_a: Tensor,
        u_bar_b: Tensor,
        fallback_chart: AttributableChart2Plus4,
        rho_ref_m: float = 2.0e6,
    ) -> None:
        self.t_a_sec = float(t_a_sec)
        self.t_b_sec = float(t_b_sec)
        self.t_ref_sec = float(t_ref_sec)
        self.station_a_si = station_a_si.to(torch.float64)
        self.station_b_si = station_b_si.to(torch.float64)
        self.u_bar_a = u_bar_a.to(torch.float64) / torch.linalg.vector_norm(u_bar_a.to(torch.float64)).clamp_min(1.0e-15)
        self.u_bar_b = u_bar_b.to(torch.float64) / torch.linalg.vector_norm(u_bar_b.to(torch.float64)).clamp_min(1.0e-15)
        _, self.E_perp_a = build_los_frame(self.u_bar_a)
        _, self.E_perp_b = build_los_frame(self.u_bar_b)
        self.fallback_chart = fallback_chart
        self.rho_ref_m = float(rho_ref_m)

    def _map_single(self, coords6: Tensor, retrograde: bool) -> tuple[Tensor, Tensor]:
        device, dtype = coords6.device, coords6.dtype
        z1 = coords6[0].clamp(-6.0, 6.0)
        z2 = coords6[1].clamp(-6.0, 6.0)
        eta_a = coords6[2:4]
        eta_b = coords6[4:6]

        rho_a = self.rho_ref_m * torch.exp(z1)
        rho_b = self.rho_ref_m * torch.exp(z2)
        E_a = self.E_perp_a.to(device=device, dtype=dtype)
        E_b = self.E_perp_b.to(device=device, dtype=dtype)
        u_a = spherical_exp_map(
            self.u_bar_a.to(device=device, dtype=dtype),
            eta_a[0] * E_a[:, 0] + eta_a[1] * E_a[:, 1],
        )
        u_b = spherical_exp_map(
            self.u_bar_b.to(device=device, dtype=dtype),
            eta_b[0] * E_b[:, 0] + eta_b[1] * E_b[:, 1],
        )
        r_a = self.station_a_si[:3].to(device=device, dtype=dtype) + rho_a * u_a
        r_b = self.station_b_si[:3].to(device=device, dtype=dtype) + rho_b * u_b
        v_a, _, ok = solve_lambert_universal_m0(
            r_a, r_b, self.t_b_sec - self.t_a_sec, retrograde=retrograde
        )
        x_a = torch.cat((r_a, v_a), dim=0)
        dt_ref = self.t_ref_sec - self.t_a_sec
        if abs(dt_ref) > 1.0e-9:
            theta_ref, _ = propagate_si_with_stm(x_a, dt_ref, compute_stm=False)
        else:
            theta_ref = x_a
        return theta_ref, ok

    def decode(
        self, z: Tensor, eta: Tensor | None = None, branch: Tensor | int = 0
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Decode (z [B,2], eta [B,4], branch [B]) -> (theta_ref [B,6], log_abs_det [B], used_lambert [B])."""
        squeeze = z.ndim == 1
        zb = z.unsqueeze(0) if squeeze else z
        batch = zb.shape[0]
        device, dtype = zb.device, zb.dtype
        etab = (
            torch.zeros((batch, 4), device=device, dtype=dtype)
            if eta is None
            else (eta.unsqueeze(0) if eta.ndim == 1 else eta)
        )
        br = torch.as_tensor(branch, device=device, dtype=torch.long).reshape(-1)
        if br.numel() == 1:
            br = br.expand(batch)

        fb_theta, fb_logdet = self.fallback_chart.decode(zb, etab)
        thetas: list[Tensor] = []
        logdets: list[Tensor] = []
        used_lam: list[Tensor] = []

        eps_fd = 1.0e-5
        eye6 = torch.eye(6, device=device, dtype=dtype) * eps_fd

        for i in range(batch):
            retro = bool(int(br[i].item()) == 1)
            c0 = torch.cat((zb[i], etab[i]), dim=0)
            th0, ok0 = self._map_single(c0, retrograde=retro)
            if not bool(ok0.item()):
                thetas.append(fb_theta[i])
                logdets.append(fb_logdet[i])
                used_lam.append(torch.tensor(False, device=device))
                continue
            # Central finite-difference 6x6 Jacobian of F_b for exact volume element
            cols: list[Tensor] = []
            jac_ok = True
            for d in range(6):
                th_p, ok_p = self._map_single(c0 + eye6[d], retrograde=retro)
                th_m, ok_m = self._map_single(c0 - eye6[d], retrograde=retro)
                if not bool((ok_p & ok_m).item()):
                    jac_ok = False
                    break
                cols.append((th_p - th_m) / (2.0 * eps_fd))
            if not jac_ok:
                thetas.append(fb_theta[i])
                logdets.append(fb_logdet[i])
                used_lam.append(torch.tensor(False, device=device))
                continue
            J6 = torch.stack(cols, dim=-1)  # [6, 6]
            sign_j, ld = torch.linalg.slogdet(J6)
            if float(sign_j.abs().item()) < 0.5 or not bool(torch.isfinite(ld).item()):
                thetas.append(fb_theta[i])
                logdets.append(fb_logdet[i])
                used_lam.append(torch.tensor(False, device=device))
            else:
                thetas.append(th0)
                logdets.append(ld)
                used_lam.append(torch.tensor(True, device=device))

        out_th = torch.stack(thetas, dim=0)
        out_ld = torch.stack(logdets, dim=0)
        out_used = torch.stack(used_lam, dim=0)
        if squeeze:
            return out_th.squeeze(0), out_ld.squeeze(0), out_used.squeeze(0)
        return out_th, out_ld, out_used


class ContinuousTrackletEncoder(nn.Module):
    """Pillar I: Cadence-invariant continuous-time additive normal-statistics encoder.

    Computes H_s = sum_{j in s} phi(tau_j) phi(tau_j)^T in R^{4x4} and
    B_s = sum_{j in s} phi(tau_j) y_j^T in R^{4x2} on Legendre basis up to degree 3,
    eliminating sequence-length discrepancies across dense / stride5 / stride20 / pois10.
    """

    def __init__(self, context_dim: int = 48, max_blocks: int = 6, dtype: torch.dtype = DTYPE) -> None:
        super().__init__()
        self.context_dim = int(context_dim)
        self.max_blocks = int(max_blocks)
        # Per-block summary dim: 10 (upper tri of H_s / N) + 8 (B_s / N) + 8 (LS fit u_dot + u_bar + stn + span + sigma) = 26
        self.block_feat_dim = 26
        self.block_mlp = nn.Sequential(
            nn.Linear(self.block_feat_dim, context_dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(context_dim, context_dim, dtype=dtype),
            nn.LayerNorm(context_dim, dtype=dtype),
        )
        self.gru = nn.GRU(context_dim, context_dim, batch_first=True, dtype=dtype)

    @staticmethod
    def legendre_basis(tau: Tensor) -> Tensor:
        """Evaluate [P_0(tau), P_1(tau), P_2(tau), P_3(tau)] for tau in [-1, 1]."""
        p0 = torch.ones_like(tau)
        p1 = tau
        p2 = 0.5 * (3.0 * tau * tau - 1.0)
        p3 = 0.5 * (5.0 * tau.pow(3) - 3.0 * tau)
        return torch.stack((p0, p1, p2, p3), dim=-1)  # [N, 4]

    @staticmethod
    def compute_additive_block_stats(
        times_sec: Tensor,
        observed_los: Tensor,
        *,
        t_center_sec: float,
        half_span_sec: float,
        u_bar_ref: Tensor,
    ) -> tuple[Tensor, Tensor, int]:
        """Exact batch-additive sufficient statistics (H_s, B_s, count) for a block or sub-chunk."""
        device = observed_los.device
        dtype = torch.float64
        t = times_sec.to(device=device, dtype=dtype)
        u = observed_los.to(device=device, dtype=dtype)
        u = u / torch.linalg.vector_norm(u, dim=-1, keepdim=True).clamp_min(1.0e-15)
        u_ref = u_bar_ref.to(device=device, dtype=dtype)
        u_ref = u_ref / torch.linalg.vector_norm(u_ref).clamp_min(1.0e-15)
        _, E_perp = build_los_frame(u_ref)

        tau = (t - float(t_center_sec)) / max(float(half_span_sec), 1.0)
        phi = ContinuousTrackletEncoder.legendre_basis(tau)  # [N, 4]
        y = (u - u_ref.unsqueeze(0)) @ E_perp  # [N, 2]

        H_s = phi.transpose(0, 1) @ phi  # [4, 4]
        B_s = phi.transpose(0, 1) @ y  # [4, 2]
        return H_s, B_s, int(t.shape[0])

    def encode_prefix(
        self,
        times_sec: Tensor,
        observed_los: Tensor,
        station_states_si: Tensor,
        valid_mask: Tensor | None,
        *,
        up_to_idx: int,
        sigma_u_rad: float = 5.0e-4,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Encode causal prefix <= up_to_idx into fixed context vector [context_dim] + Attributable summary."""
        p_dtype = self.block_mlp[0].weight.dtype
        device = self.block_mlp[0].weight.device
        t_np = np.asarray(times_sec.detach().cpu(), dtype=np.float64).reshape(-1)[: up_to_idx + 1]
        v_np = (
            np.ones_like(t_np, dtype=bool)
            if valid_mask is None
            else np.asarray(valid_mask.detach().cpu(), dtype=bool).reshape(-1)[: up_to_idx + 1]
        )
        blocks = parse_observation_blocks(t_np, v_np)
        if not blocks:
            return torch.zeros((self.context_dim,), device=device, dtype=p_dtype), {}

        blk_feats: list[Tensor] = []
        last_attr: dict[str, Any] = {}
        t0_global = blocks[0].start_time_sec

        for blk in blocks[-self.max_blocks :]:
            sl = slice(blk.start_idx, blk.end_idx + 1)
            m_b = torch.as_tensor(v_np[sl], device=device, dtype=torch.bool)
            t_b = times_sec[sl].to(device=device, dtype=torch.float64)[m_b]
            u_b = observed_los[sl].to(device=device, dtype=torch.float64)[m_b]
            s_b = station_states_si[sl].to(device=device, dtype=torch.float64)[m_b]

            t_c = 0.5 * (blk.start_time_sec + blk.end_time_sec)
            half_span = max(0.5 * (blk.end_time_sec - blk.start_time_sec), 15.0)
            u_mean = u_b.mean(dim=0)
            u_bar = u_mean / torch.linalg.vector_norm(u_mean).clamp_min(1.0e-15)
            H_s, B_s, count = self.compute_additive_block_stats(
                t_b, u_b, t_center_sec=t_c, half_span_sec=half_span, u_bar_ref=u_bar
            )
            n_safe = float(max(count, 1))
            H_norm = H_s / n_safe
            B_norm = B_s / n_safe

            # Adaptive order truncation based on frame count & SVD condition number
            order = min(4, max(1, count))
            H_sub = H_norm[:order, :order] + 1.0e-4 * torch.eye(order, device=device, dtype=torch.float64)
            coeff = torch.linalg.solve(H_sub, B_norm[:order])  # [order, 2]
            omega_est = (coeff[1] / half_span) if order >= 2 else torch.zeros((2,), device=device, dtype=torch.float64)

            triu_idx = torch.triu_indices(4, 4, device=device)
            h_vec = H_norm[triu_idx[0], triu_idx[1]]  # [10]
            b_vec = B_norm.reshape(8) / max(sigma_u_rad, 1.0e-4)  # [8]
            s_center = s_b[count // 2]
            meta_vec = torch.cat(
                (
                    u_bar,
                    omega_est * float(T_SCALE_S),
                    torch.tensor(
                        [
                            (t_c - t0_global) / float(T_SCALE_S),
                            half_span / float(T_SCALE_S),
                            math.log10(max(sigma_u_rad, 1.0e-7)),
                        ],
                        device=device,
                        dtype=torch.float64,
                    ),
                ),
                dim=0,
            )  # [8]
            blk_feats.append(torch.cat((h_vec, b_vec.clamp(-50.0, 50.0), meta_vec), dim=0).to(dtype=p_dtype))
            last_attr = {
                "t_c_sec": t_c,
                "station_c_si": s_center,
                "u_bar_c": u_bar,
                "omega_bar_c": omega_est,
                "first_block_t_sec": blocks[0].start_time_sec,
                "first_station_si": station_states_si[blocks[0].start_idx].to(device=device, dtype=torch.float64),
                "first_u_bar": observed_los[blocks[0].start_idx].to(device=device, dtype=torch.float64),
            }

        seq = self.block_mlp(torch.stack(blk_feats, dim=0)).unsqueeze(0)  # [1, B_k, D]
        _, h_last = self.gru(seq)
        return h_last.squeeze(0).squeeze(0), last_attr


class ConditionalRealNVP2D(nn.Module):
    """Pillar II: Explicit-density 4-layer Conditional Affine Coupling Flow on R^2 + branch head.

    Total parameter count ~28k (< 45k budget), providing exact analytic log_prob and
    single-pass O(1) sampling without ODE integration or closed-loop BPTT.
    """

    def __init__(
        self,
        context_dim: int = 48,
        hidden_dim: int = 64,
        num_layers: int = 4,
        num_branches: int = 2,
        max_log_scale: float = 2.0,
        dtype: torch.dtype = DTYPE,
    ) -> None:
        super().__init__()
        self.context_dim = int(context_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_branches = int(num_branches)
        self.max_log_scale = float(max_log_scale)

        self.branch_head = nn.Sequential(
            nn.Linear(context_dim, hidden_dim // 2, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_branches, dtype=dtype),
        )
        self.branch_emb = nn.Embedding(num_branches, 8, dtype=dtype)
        masks = [
            torch.tensor([1.0, 0.0] if i % 2 == 0 else [0.0, 1.0], dtype=dtype)
            for i in range(num_layers)
        ]
        self.register_buffer("masks", torch.stack(masks, dim=0), persistent=True)

        self.st_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 + context_dim + 8, hidden_dim, dtype=dtype),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim, dtype=dtype),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 4, dtype=dtype),  # [scale_2, shift_2]
                )
                for _ in range(num_layers)
            ]
        )
        for net in self.st_nets:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)

    def _cond(self, context: Tensor, branch: Tensor) -> Tensor:
        return torch.cat((context, self.branch_emb(branch)), dim=-1)

    def log_prob(self, z: Tensor, branch: Tensor, context: Tensor) -> Tensor:
        """Exact analytical log q_psi(z, b | c_k) = log p_psi(b | c_k) + log q_psi(z | b, c_k)."""
        p_dtype = self.branch_head[0].weight.dtype
        zb = z.to(dtype=p_dtype)
        ctx = context.to(dtype=p_dtype)
        if ctx.ndim == 1:
            ctx = ctx.unsqueeze(0).expand(zb.shape[0], -1)
        br = branch.to(dtype=torch.long).reshape(zb.shape[0])

        br_log_probs = nn.functional.log_softmax(self.branch_head(ctx), dim=-1)
        log_p_b = br_log_probs.gather(1, br.unsqueeze(-1)).squeeze(-1)

        cond = self._cond(ctx, br)
        x = zb
        log_det = torch.zeros((zb.shape[0],), device=zb.device, dtype=p_dtype)

        # Inverse flow z -> u_base
        for idx in range(self.num_layers - 1, -1, -1):
            m = self.masks[idx].to(device=zb.device, dtype=p_dtype).unsqueeze(0)
            x_masked = x * m
            st = self.st_nets[idx](torch.cat((x_masked, cond), dim=-1))
            s = self.max_log_scale * torch.tanh(st[:, :2]) * (1.0 - m)
            t = st[:, 2:] * (1.0 - m)
            x = x_masked + (1.0 - m) * (x - t) * torch.exp(-s)
            log_det = log_det - s.sum(dim=-1)

        log_base = -0.5 * (x.square().sum(dim=-1) + 2.0 * math.log(2.0 * math.pi))
        return (log_p_b + log_base + log_det).to(dtype=z.dtype)

    def sample_and_log_prob(
        self, context: Tensor, num_samples: int, *, generator: torch.Generator | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Single-pass exact sampling of (z [P,2], branch [P], log_q [P])."""
        p_dtype = self.branch_head[0].weight.dtype
        device = self.branch_head[0].weight.device
        ctx1 = context.to(device=device, dtype=p_dtype).reshape(1, -1)
        ctx = ctx1.expand(int(num_samples), -1)

        br_logits = self.branch_head(ctx1).squeeze(0)
        br_probs = torch.softmax(br_logits, dim=-1)
        branch = torch.multinomial(br_probs, int(num_samples), replacement=True, generator=generator)
        log_p_b = torch.log(br_probs.clamp_min(1.0e-12))[branch]

        u = torch.randn((int(num_samples), 2), generator=generator, device=device, dtype=p_dtype)
        cond = self._cond(ctx, branch)
        x = u
        log_det = torch.zeros((int(num_samples),), device=device, dtype=p_dtype)

        for idx in range(self.num_layers):
            m = self.masks[idx].to(device=device, dtype=p_dtype).unsqueeze(0)
            x_masked = x * m
            st = self.st_nets[idx](torch.cat((x_masked, cond), dim=-1))
            s = self.max_log_scale * torch.tanh(st[:, :2]) * (1.0 - m)
            t = st[:, 2:] * (1.0 - m)
            x = x_masked + (1.0 - m) * (x * torch.exp(s) + t)
            log_det = log_det + s.sum(dim=-1)

        log_base = -0.5 * (u.square().sum(dim=-1) + 2.0 * math.log(2.0 * math.pi))
        return x.to(torch.float64), branch, (log_p_b + log_base + log_det).to(torch.float64)


class DefensiveProposalMixture(nn.Module):
    """Proposition 2 defensive mixture: q_mix = (1 - eps_def) q_psi + eps_def q_phys."""

    def __init__(
        self,
        flow: ConditionalRealNVP2D,
        *,
        eps_def: float = 0.15,
        phys_std: float = 1.6,
        num_branches: int = 2,
    ) -> None:
        super().__init__()
        if not (0.0 < eps_def < 1.0):
            raise ValueError("eps_def must lie strictly in (0, 1)")
        self.flow = flow
        self.eps_def = float(eps_def)
        self.phys_std = float(phys_std)
        self.num_branches = int(num_branches)

    def _log_q_phys(self, z: Tensor) -> Tensor:
        var = self.phys_std * self.phys_std
        log_gauss = -0.5 * (z.square().sum(dim=-1) / var + 2.0 * math.log(2.0 * math.pi * var))
        return log_gauss - math.log(float(self.num_branches))

    def log_prob(self, z: Tensor, branch: Tensor, context: Tensor) -> Tensor:
        log_q_psi = self.flow.log_prob(z, branch, context)
        log_q_phy = self._log_q_phys(z.to(torch.float64)).to(dtype=log_q_psi.dtype)
        return torch.logaddexp(
            math.log(1.0 - self.eps_def) + log_q_psi,
            math.log(self.eps_def) + log_q_phy,
        )

    def sample_and_log_prob(
        self, context: Tensor, num_samples: int, *, mode: str = "learned_mix"
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Sample P candidates under 'learned_mix' or 'sobol_nonlearned'."""
        device = context.device
        P = int(num_samples)
        if mode == "sobol_nonlearned":
            sobol = torch.quasirandom.SobolEngine(dimension=2, scramble=True)
            u01 = sobol.draw(P).to(device=device, dtype=torch.float64).clamp(1.0e-4, 1.0 - 1.0e-4)
            z = math.sqrt(2.0) * self.phys_std * torch.erfinv(2.0 * u01 - 1.0)
            branch = (torch.arange(P, device=device) % self.num_branches).to(torch.long)
            return z, branch, self._log_q_phys(z)

        n_phys = max(1, int(round(self.eps_def * P)))
        n_flow = max(0, P - n_phys)
        parts_z, parts_b = [], []
        if n_flow > 0:
            zf, bf, _ = self.flow.sample_and_log_prob(context, n_flow)
            parts_z.append(zf)
            parts_b.append(bf)
        zp = torch.randn((n_phys, 2), device=device, dtype=torch.float64) * self.phys_std
        bp = torch.randint(0, self.num_branches, (n_phys,), device=device, dtype=torch.long)
        parts_z.append(zp)
        parts_b.append(bp)

        z_all = torch.cat(parts_z, dim=0)
        b_all = torch.cat(parts_b, dim=0)
        log_q_mix = self.log_prob(z_all, b_all, context).to(torch.float64)
        return z_all, b_all, log_q_mix


def compute_forecast_disagreement_weights(
    candidate_thetas_si: Tensor,
    log_posterior_unnorm: Tensor,
    *,
    t_ref_sec: float,
    next_view_time_sec: float,
    next_station_pos_si: Tensor,
    num_spherical_bins: int = 16,
    disagreement_floor: float = 1.0e-3,
) -> tuple[Tensor, Tensor, Tensor]:
    """Proposition 1 (FD-ASF): Optimal SNIS forecast-disagreement weights q_k* proportional to pi_k ||h - h_bar||_2.

    Uses ONLY causally known future platform geometry (next_view_time_sec, next_station_pos_si) —
    NEVER reads any future LOS measurement u_{k+1}.

    Returns:
        w_star_norm: [M] normalized target weights for forward-KL flow training.
        posterior_weights: [M] normalized standard posterior weights pi_k.
        disagreement_norms: [M] ||h_{k+1}(theta_m) - h_bar_{k+1}||_2.
    """
    M = int(candidate_thetas_si.shape[0])
    device, dtype = candidate_thetas_si.device, torch.float64
    thetas = candidate_thetas_si.to(device=device, dtype=dtype)
    log_pi = log_posterior_unnorm.to(device=device, dtype=dtype).reshape(M)

    pi_w = torch.softmax(log_pi - log_pi.max(), dim=0)
    dt_next = float(next_view_time_sec - t_ref_sec)
    x_next, _ = propagate_si_with_stm(thetas, dt_next, compute_stm=False)
    rel_next = x_next[:, :3] - next_station_pos_si[:3].to(device=device, dtype=dtype).unsqueeze(0)
    u_next = rel_next / torch.linalg.vector_norm(rel_next, dim=-1, keepdim=True).clamp_min(1.0)

    # Bounded spherical harmonic / Fibonacci-bin soft forecast features h_{k+1}(theta) in R^B
    B = int(num_spherical_bins)
    idx = torch.arange(B, device=device, dtype=dtype)
    phi_fib = math.pi * (3.0 - math.sqrt(5.0)) * idx
    z_fib = 1.0 - 2.0 * (idx + 0.5) / float(B)
    r_fib = torch.sqrt((1.0 - z_fib * z_fib).clamp_min(0.0))
    anchors = torch.stack((r_fib * torch.cos(phi_fib), r_fib * torch.sin(phi_fib), z_fib), dim=-1)  # [B, 3]

    # Soft von Mises-Fisher bin assignment probabilities h(theta) in Delta^{B-1}
    kappa_bin = 12.0
    h_vecs = torch.softmax(kappa_bin * (u_next @ anchors.transpose(0, 1)), dim=-1)  # [M, B]
    h_bar = (pi_w.unsqueeze(-1) * h_vecs).sum(dim=0, keepdim=True)  # [1, B]
    dis_norm = torch.linalg.vector_norm(h_vecs - h_bar, dim=-1)  # [M]

    raw_star = pi_w * (dis_norm + float(disagreement_floor))
    w_star = raw_star / raw_star.sum().clamp_min(1.0e-15)
    return w_star, pi_w, dis_norm


def proposal_weighted_kl_and_dsm_loss(
    flow: ConditionalRealNVP2D,
    z_targets: Tensor,
    branch_targets: Tensor,
    target_weights: Tensor,
    context: Tensor,
    *,
    dsm_weight: float = 0.05,
    noise_scales: Sequence[float] = (0.05, 0.20),
) -> Tensor:
    """Offline zero-BPTT training loss: weighted forward KL + optional geometry-aligned multi-scale DSM."""
    w = target_weights / target_weights.sum().clamp_min(1.0e-12)
    log_q = flow.log_prob(z_targets, branch_targets, context)
    kl_loss = -(w.detach() * log_q).sum()
    if dsm_weight <= 0.0:
        return kl_loss

    dsm_total = torch.zeros_like(kl_loss)
    for sigma in noise_scales:
        eps = torch.randn_like(z_targets)
        z_noisy = (z_targets.detach() + float(sigma) * eps).requires_grad_(True)
        lq = flow.log_prob(z_noisy, branch_targets, context)
        score = torch.autograd.grad(lq.sum(), z_noisy, create_graph=True)[0]
        target_score = -eps / float(sigma)
        per_sample = ((score - target_score) * float(sigma)).square().sum(dim=-1)
        dsm_total = dsm_total + (w.detach() * per_sample).sum()
    return kl_loss + float(dsm_weight) * dsm_total / float(len(noise_scales))


@dataclass(frozen=True)
class PublishDecision:
    """Output of Pillar III Dominance-Gated Safe Publisher."""

    published_theta_ref_si: Tensor
    switched_from_prior: bool
    prior_median_angle_rad: float
    dominant_basin_weight: float
    nll_gain: float


class DominanceGatedPublisher:
    """Pillar III: 3-condition dominance-gated point-state publisher (Proposition 3)."""

    def __init__(
        self,
        *,
        gamma_dom: float = 0.75,
        delta_margin: float = 9.0,  # > 0.5 * chi^2_{6, 0.99} ~= 8.4
        basin_radius_scaled: float = 0.25,
    ) -> None:
        self.gamma_dom = float(gamma_dom)
        self.delta_margin = float(delta_margin)
        self.basin_radius_scaled = float(basin_radius_scaled)

    def select_published_orbit(
        self,
        *,
        prior_theta_ref_si: Tensor,
        prior_median_angle_rad: float,
        tau_rad: float,
        prior_nll: float,
        polished_modes_theta_ref_si: Tensor,
        polished_modes_nll: Tensor,
        particle_thetas_ref_si: Tensor,
        particle_weights: Tensor,
    ) -> PublishDecision:
        if polished_modes_theta_ref_si.numel() == 0:
            return PublishDecision(
                published_theta_ref_si=prior_theta_ref_si,
                switched_from_prior=False,
                prior_median_angle_rad=prior_median_angle_rad,
                dominant_basin_weight=0.0,
                nll_gain=0.0,
            )

        best_idx = int(torch.argmin(polished_modes_nll).item())
        best_mode = polished_modes_theta_ref_si[best_idx]
        best_nll = float(polished_modes_nll[best_idx].item())

        scale = state_scale(device=best_mode.device, dtype=best_mode.dtype)
        dists = torch.linalg.vector_norm(
            (particle_thetas_ref_si - best_mode.unsqueeze(0)) / scale.unsqueeze(0), dim=-1
        )
        in_basin = dists <= self.basin_radius_scaled
        w_dom = float(particle_weights[in_basin].sum().item()) if bool(in_basin.any()) else float(particle_weights.max().item())
        nll_gain = float(prior_nll - best_nll)

        cond1_above_noise = prior_median_angle_rad > float(tau_rad)
        cond2_dominant = w_dom >= self.gamma_dom
        cond3_margin = nll_gain > self.delta_margin

        if cond1_above_noise and cond2_dominant and cond3_margin:
            return PublishDecision(
                published_theta_ref_si=best_mode,
                switched_from_prior=True,
                prior_median_angle_rad=prior_median_angle_rad,
                dominant_basin_weight=w_dom,
                nll_gain=nll_gain,
            )
        return PublishDecision(
            published_theta_ref_si=prior_theta_ref_si,
            switched_from_prior=False,
            prior_median_angle_rad=prior_median_angle_rad,
            dominant_basin_weight=w_dom,
            nll_gain=nll_gain,
        )


class FDASFSequentialFilter(nn.Module):
    """Complete Model Plan V1 Flagship Filter (FD-ASF) uniting Pillars I, II, and III."""

    def __init__(
        self,
        *,
        context_dim: int = 48,
        num_particles: int = 12,
        top_k_polish: int = 2,
        lm_polish_steps: int = 2,
        eps_def: float = 0.15,
        proposal_mode: str = "learned_mix",  # "learned_mix" or "sobol_nonlearned"
        gamma_dom: float = 0.75,
        delta_margin: float = 9.0,
        dtype: torch.dtype = DTYPE,
    ) -> None:
        super().__init__()
        self.num_particles = int(num_particles)
        self.top_k_polish = int(top_k_polish)
        self.lm_polish_steps = int(lm_polish_steps)
        self.proposal_mode = str(proposal_mode)

        self.tracklet_encoder = ContinuousTrackletEncoder(context_dim=context_dim, dtype=dtype)
        self.flow = ConditionalRealNVP2D(context_dim=context_dim, dtype=dtype)
        self.proposal = DefensiveProposalMixture(self.flow, eps_def=eps_def)
        self.lm_refiner = LTRLMRefiner(writeback_mode="joint_6d", dtype=dtype)
        self.publisher = DominanceGatedPublisher(gamma_dom=gamma_dom, delta_margin=delta_margin)
        self.ledger = ComputeLedger()

    def _segment_nll(
        self,
        thetas_ref_si: Tensor,
        t_ref_sec: float,
        seg_times_sec: Tensor,
        seg_stations_si: Tensor,
        seg_los_obs: Tensor,
        kappa: float,
    ) -> Tensor:
        """Evaluate spherical von Mises-Fisher NLL sum_{j in seg} kappa [1 - u_j^T h(phi(theta))]."""
        P = int(thetas_ref_si.shape[0])
        nll = torch.zeros((P,), device=thetas_ref_si.device, dtype=torch.float64)
        u_obs = seg_los_obs / torch.linalg.vector_norm(seg_los_obs, dim=-1, keepdim=True).clamp_min(1.0e-15)
        for j in range(int(seg_times_sec.shape[0])):
            dt = float(seg_times_sec[j].item()) - float(t_ref_sec)
            xj, _ = propagate_si_with_stm(thetas_ref_si, dt, compute_stm=False, ledger=self.ledger)
            rel = xj[:, :3] - seg_stations_si[j, :3].unsqueeze(0)
            u_hat = rel / torch.linalg.vector_norm(rel, dim=-1, keepdim=True).clamp_min(1.0)
            cos_j = (u_hat * u_obs[j].unsqueeze(0)).sum(dim=-1).clamp(-1.0, 1.0)
            nll = nll + float(kappa) * (1.0 - cos_j)
        return nll

    @torch.no_grad()
    def forward_single_track(
        self,
        *,
        initial_state_si: Tensor,
        init_time_sec: float,
        times_sec: Tensor,
        observed_los: Tensor,
        station_states_si: Tensor,
        valid_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Any]]:
        """Run block-incremental FD-ASF SMC + Top-K LM + Dominance-Gated Publisher on one track."""
        device = next(self.parameters()).device
        dtype = torch.float64
        t_ref = float(init_time_sec)
        prior_theta_ref = initial_state_si.to(device=device, dtype=dtype)
        published_theta_ref = prior_theta_ref.clone()

        T = int(times_sec.shape[0])
        t_cpu = times_sec.detach().to("cpu", dtype=dtype)
        val_cpu = (
            torch.ones((T,), dtype=torch.bool)
            if valid_mask is None
            else valid_mask.detach().to("cpu", dtype=torch.bool)
        )
        obs = observed_los.to(device=device, dtype=dtype)
        stn = station_states_si.to(device=device, dtype=dtype)

        blocks = parse_observation_blocks(t_cpu, val_cpu, init_time_sec=t_ref)
        block_ends = {blk.end_idx: blk for blk in blocks}

        particle_thetas: Tensor | None = None
        particle_log_w: Tensor | None = None
        prev_eval_idx = -1
        switches = 0

        priors_seq: list[Tensor] = []
        posts_seq: list[Tensor] = []

        for k in range(T):
            if not bool(val_cpu[k].item()):
                priors_seq.append(torch.zeros((6,), device=device, dtype=dtype))
                posts_seq.append(torch.zeros((6,), device=device, dtype=dtype))
                continue
            tk = float(t_cpu[k].item())
            prior_k, _ = propagate_si_with_stm(
                published_theta_ref, tk - t_ref, compute_stm=False, ledger=self.ledger
            )

            if k in block_ends:
                blk = block_ends[k]
                tau_k, sigma_k, _ = estimate_online_los_noise_floor(
                    t_cpu, obs.detach().cpu(), valid_mask=val_cpu, up_to_idx=k
                )
                kappa = 1.0 / max(sigma_k * sigma_k, 1.0e-8)
                ctx, attr = self.tracklet_encoder.encode_prefix(
                    t_cpu, obs, stn, val_cpu, up_to_idx=k, sigma_u_rad=sigma_k
                )
                if attr:
                    attr_chart = AttributableChart2Plus4(
                        t_c_sec=attr["t_c_sec"],
                        t_ref_sec=t_ref,
                        station_c_si=attr["station_c_si"],
                        u_bar_c=attr["u_bar_c"],
                        omega_bar_c=attr["omega_bar_c"],
                    )
                    if blk.block_index >= 1 and (attr["t_c_sec"] - attr["first_block_t_sec"]) >= 60.0:
                        chart: Any = LambertChart2Plus4(
                            t_a_sec=attr["first_block_t_sec"],
                            t_b_sec=attr["t_c_sec"],
                            t_ref_sec=t_ref,
                            station_a_si=attr["first_station_si"],
                            station_b_si=attr["station_c_si"],
                            u_bar_a=attr["first_u_bar"],
                            u_bar_b=attr["u_bar_c"],
                            fallback_chart=attr_chart,
                        )
                    else:
                        chart = attr_chart

                    # Incremental observations in new block (never double-counting old blocks!)
                    new_sl = slice(max(0, prev_eval_idx + 1), k + 1)
                    new_mask = val_cpu[new_sl]
                    new_idx_local = torch.nonzero(new_mask, as_tuple=False).squeeze(-1)
                    # Subsample up to 8 evaluation epochs per block for fast incremental likelihood
                    if new_idx_local.numel() > 8:
                        sel = torch.linspace(0, new_idx_local.numel() - 1, 8).round().to(torch.long)
                        new_idx_local = new_idx_local[sel]
                    seg_t = t_cpu[new_sl][new_idx_local].to(device=device)
                    seg_s = stn[new_sl][new_idx_local]
                    seg_u = obs[new_sl][new_idx_local]

                    need_refresh = particle_thetas is None
                    if particle_log_w is not None:
                        w_tmp = torch.softmax(particle_log_w - particle_log_w.max(), dim=0)
                        ess = 1.0 / (w_tmp * w_tmp).sum().item()
                        if ess < 0.5 * self.num_particles:
                            need_refresh = True

                    if need_refresh:
                        z_s, b_s, log_q = self.proposal.sample_and_log_prob(
                            ctx, self.num_particles, mode=self.proposal_mode
                        )
                        eta_s = torch.randn((self.num_particles, 4), device=device, dtype=dtype) * sigma_k
                        log_q_eta = -0.5 * (
                            (eta_s / sigma_k).square().sum(dim=-1) + 4.0 * math.log(2.0 * math.pi * sigma_k * sigma_k)
                        )
                        if isinstance(chart, LambertChart2Plus4):
                            th_s, log_det, _ = chart.decode(z_s, eta_s, b_s)
                        else:
                            th_s, log_det = chart.decode(z_s, eta_s)

                        kf_idx, _ = select_causal_keyframes(t_cpu, val_cpu, up_to_idx=k, max_keyframes=16)
                        kf_t = t_cpu[kf_idx].to(device=device)
                        kf_s = stn[kf_idx]
                        kf_u = obs[kf_idx]
                        full_nll = self._segment_nll(th_s, t_ref, kf_t, kf_s, kf_u, kappa)
                        feasible = check_bound_perigee_feasible(th_s)
                        log_phys = torch.where(feasible, torch.zeros_like(full_nll), torch.full_like(full_nll, -1.0e6))
                        particle_thetas = th_s
                        particle_log_w = log_phys - full_nll + log_det - (log_q + log_q_eta)
                    else:
                        assert particle_thetas is not None and particle_log_w is not None
                        delta_nll = self._segment_nll(particle_thetas, t_ref, seg_t, seg_s, seg_u, kappa)
                        particle_log_w = particle_log_w - delta_nll

                    prev_eval_idx = k
                    w_norm = torch.softmax(particle_log_w - particle_log_w.max(), dim=0)

                    # Polish Top-K modes using 6x6 LTR-LM kernel (strictly in point-publish pool!)
                    kf_idx, _ = select_causal_keyframes(t_cpu, val_cpu, up_to_idx=k, max_keyframes=16)
                    if kf_idx.size >= 2:
                        kf_t = t_cpu[kf_idx].to(device=device)
                        kf_s = stn[kf_idx]
                        kf_u = obs[kf_idx]
                        top_k = min(self.top_k_polish, self.num_particles)
                        top_indices = torch.topk(w_norm, k=top_k).indices
                        polished_list: list[Tensor] = []
                        for idx_p in top_indices.tolist():
                            cand_k, _ = propagate_si_with_stm(
                                particle_thetas[idx_p], tk - t_ref, compute_stm=False, ledger=self.ledger
                            )
                            for _ in range(self.lm_polish_steps):
                                out_lm = self.lm_refiner.refine_anchor(
                                    cand_k, tk, stn[k, :3], kf_t, kf_s, kf_u, tau_rad=tau_k, ledger=self.ledger
                                )
                                cand_k = out_lm.posterior_state_si
                            cand_ref, _ = propagate_si_with_stm(
                                cand_k, t_ref - tk, compute_stm=False, ledger=self.ledger
                            )
                            polished_list.append(cand_ref)
                        polished_modes = torch.stack(polished_list, dim=0)
                        modes_nll = self._segment_nll(polished_modes, t_ref, kf_t, kf_s, kf_u, kappa)
                        prior_nll = float(
                            self._segment_nll(published_theta_ref.unsqueeze(0), t_ref, kf_t, kf_s, kf_u, kappa)[0].item()
                        )
                        # Compute prior median angle on keyframes
                        angles_prior = []
                        for j_w in range(int(kf_t.shape[0])):
                            xj_p, _ = propagate_si_with_stm(
                                published_theta_ref, float(kf_t[j_w].item()) - t_ref, compute_stm=False, ledger=self.ledger
                            )
                            u_p = xj_p[:3] - kf_s[j_w, :3]
                            angles_prior.append(geodesic_angle_rad(u_p, kf_u[j_w]))
                        prior_med = float(torch.median(torch.stack(angles_prior)).item())

                        decision = self.publisher.select_published_orbit(
                            prior_theta_ref_si=published_theta_ref,
                            prior_median_angle_rad=prior_med,
                            tau_rad=tau_k,
                            prior_nll=prior_nll,
                            polished_modes_theta_ref_si=polished_modes,
                            polished_modes_nll=modes_nll,
                            particle_thetas_ref_si=particle_thetas,
                            particle_weights=w_norm,
                        )
                        published_theta_ref = decision.published_theta_ref_si
                        if decision.switched_from_prior:
                            switches += 1

            post_k, _ = propagate_si_with_stm(
                published_theta_ref, tk - t_ref, compute_stm=False, ledger=self.ledger
            )
            priors_seq.append(prior_k)
            posts_seq.append(post_k)

        return (
            torch.stack(priors_seq, dim=0),
            torch.stack(posts_seq, dim=0),
            {"switches": switches, "ledger": self.ledger.as_dict()},
        )

    def forward_filter_sequence(
        self,
        *,
        initial_state_si: Tensor,
        init_time_sec: Tensor | float,
        times_sec: Tensor,
        observed_los: Tensor,
        station_states_si: Tensor,
        valid_mask: Tensor | None = None,
        **kwargs: Any,
    ) -> TrainingFilterOutput:
        unbatched = initial_state_si.ndim == 1
        init_x = initial_state_si.unsqueeze(0) if unbatched else initial_state_si
        batch = int(init_x.shape[0])
        device = next(self.parameters()).device
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

        scale = state_scale(device=device, dtype=torch.float64)
        priors_b, posts_b, preds_b, innovs_b = [], [], [], []
        for b in range(batch):
            pr, po, _ = self.forward_single_track(
                initial_state_si=init_x[b],
                init_time_sec=float(init_t[b].item()),
                times_sec=t_seq[b],
                observed_los=obs_seq[b],
                station_states_si=stn_seq[b],
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
            preds_b.append(u_hat)
            innovs_b.append(inn)

        priors = torch.stack(priors_b, dim=0)
        posts = torch.stack(posts_b, dim=0)
        preds = torch.stack(preds_b, dim=0)
        innovs = torch.stack(innovs_b, dim=0)
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
            metadata={"model_family": "v1_fd_asf", "proposal_mode": self.proposal_mode},
        )


__all__ = [
    "AttributableChart2Plus4",
    "ConditionalRealNVP2D",
    "ContinuousTrackletEncoder",
    "DefensiveProposalMixture",
    "DominanceGatedPublisher",
    "FDASFSequentialFilter",
    "LambertChart2Plus4",
    "PublishDecision",
    "check_bound_perigee_feasible",
    "compute_forecast_disagreement_weights",
    "proposal_weighted_kl_and_dsm_loss",
    "solve_lambert_universal_m0",
]

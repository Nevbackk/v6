"""Round 5 Converged Genuine Diffusion Architecture (3+1 Core Knobs: K, P, delta, optional B_los).

Implements both mathematically verified diffusion routes from
docs/ROUND5_MINIMAL_DIFFUSION_DESIGN.md and
outputs/research_codex/ROUND5_DIFFUSION_AND_HYPERPARAM_REVIEW.md (acabb03):

1. Route A (Primary): VPOrbitDiffusionFilter (Cosine-VP Conditional Orbital Diffusion)
   - Forward whitened cosine VP path (s in [0, pi/2], alpha = cos s, sigma = sin s):
         x* = x^- + D (e_bar_b + L_b z_0),   z_s = alpha z_0 + sigma eps
         v* = alpha eps - sigma z_0,         (v_theta, V_theta) = f_theta(z_s, s, c^-)
         m_s = alpha z_s - sigma v_theta,    C_s = sigma^2 V_theta,  V_theta = delta I + B B^T
     trained with a SINGLE multi-scale velocity + second-moment Gaussian NLL objective.
   - Fresh-batch-only noisy-time likelihood guidance with exact denoiser Jacobian VJP:
         g_s^loc(z) = (D_z m_s)^T H^T (R + H C_s H^T)^{-1} (Y - h(m_s))
     solved in O(6^3) via Cholesky factor of C_s without inverting 2N x 2N matrices.
   - Explicit K-interval interior-midpoint reverse Probability Flow ODE:
         dz / ds = v_theta(z, s, c^-) - tan(s) g_s^loc(z),   s: pi/2 -> 0
     returning full diffusion trajectories for direct interpretability.
   - Strict separation of artificial noise sigma^2 I, conditional moment C_s, physical
     uncertainty P_m^+/-, and measurement noise R; cross-gap transport of all P independent
     two-body trajectories regularized by the SAME single delta:
         P_carry^+ = (1 - delta) P_samples^+ + delta P_b^+.

2. Route B (Separate Multi-Modal Proposal / Weighted Chain):
   - RouteBGuidedCosineFlow2D (B1 Deterministic Guided Cosine Flow on R^2):
     shared cosine path + guided field v_g = v_psi + omega cot(omega t) g_t, with EXACT
     discrete K-step map log-Jacobian sum_j log|det D T_j| (zero continuous/discrete mismatch).
   - RouteBTwistedDiffusionSMC (B2 Stochastic Twisted Diffusion SMC, Wu et al. NeurIPS 2023):
     explicit stochastic reverse transition p_j, guided proposal q_j, and telescoping potentials
     psi_0 = ell(Y | F_b), psi_K = 1 yielding exact incremental weights G_{j-1}.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor, nn

from ..data.observation_clock import causal_diffusion_events
from .batched_physics import DTYPE, state_scale
from .normalization import EARTH_RADIUS_M, T_SCALE_S, V_SCALE_MPS
from .recursive_filter import TrainingFilterOutput
from .reprojection import (
    ComputeLedger,
    build_los_frame,
    estimate_online_los_noise_floor,
    propagate_si_with_stm,
    reproject_causal_window_single,
)
from .v1_fd_asf import ContinuousTrackletEncoder


HALF_PI = 0.5 * math.pi


def safe_cholesky_spd(mat: Tensor, *, jitter: float = 1.0e-12) -> Tensor:
    """Symmetrize and compute Cholesky factor L such that mat ~= L L^T."""
    sym = 0.5 * (mat + mat.transpose(-1, -2))
    eye = torch.eye(sym.shape[-1], device=sym.device, dtype=sym.dtype)
    eps = float(jitter)
    for _ in range(6):
        try:
            return torch.linalg.cholesky(sym + eps * eye)
        except RuntimeError:
            eps *= 10.0
    eigvals, eigvecs = torch.linalg.eigh(sym)
    eigvals = eigvals.clamp_min(1.0e-10)
    rebuilt = (eigvecs * eigvals.unsqueeze(-2)) @ eigvecs.transpose(-1, -2)
    return torch.linalg.cholesky(0.5 * (rebuilt + rebuilt.transpose(-1, -2)) + 1.0e-10 * eye)


def build_diagonal_los_prior_covariance(
    prior_state_si: Tensor,
    station_pos_si: Tensor,
    *,
    pos_radial_std_scaled: float = 1.0,
    pos_transverse_std_scaled: float = 0.15,
    vel_radial_std_scaled: float = 0.8,
    vel_transverse_std_scaled: float = 0.25,
) -> Tensor:
    """Build initial diagonal-in-local-LOS-frame reference covariance P_0^- in D-scaled coordinates.

    Diagonal initialization in the LOS orthonormal frame U = [u_hat, q1, q2] guarantees
    A_0 P_0^- v_rho == 0 on short arcs before the first gap (zero spurious off-diagonal
    range leakage!). All subsequent cross-correlations arise solely from physical two-body
    propagation and real measurement updates.
    """
    device, dtype = prior_state_si.device, torch.float64
    rel = prior_state_si[:3].to(dtype=dtype) - station_pos_si[:3].to(dtype=dtype)
    U, _ = build_los_frame(rel)  # [3, 3] columns [u, q1, q2]

    U_blk = torch.zeros((6, 6), device=device, dtype=dtype)
    U_blk[:3, :3] = U
    U_blk[3:, 3:] = U

    diag_vars = torch.tensor(
        [
            pos_radial_std_scaled**2,
            pos_transverse_std_scaled**2,
            pos_transverse_std_scaled**2,
            vel_radial_std_scaled**2,
            vel_transverse_std_scaled**2,
            vel_transverse_std_scaled**2,
        ],
        device=device,
        dtype=dtype,
    )
    P0 = U_blk @ torch.diag(diag_vars) @ U_blk.transpose(0, 1)
    return 0.5 * (P0 + P0.transpose(0, 1))


def whitened_reference_covariance_update(
    P_minus: Tensor,
    A_scaled_d: Tensor,
    sigma_u_rad: float,
) -> Tensor:
    """Exact single-batch linearized reference covariance update P^+ = L (I + B^T B)^{-1} L^T.

    With P^- = L L^T and B = R^{-1/2} A L (R = sigma_u^2 I), preserves exact variance
    factor 1.0 in the prior-whitened nullspace ker(B) and avoids inverting 2N x 2N matrices.
    """
    if A_scaled_d.numel() == 0 or A_scaled_d.shape[0] == 0:
        return P_minus
    device, dtype = P_minus.device, P_minus.dtype
    L = safe_cholesky_spd(P_minus)
    inv_sigma = 1.0 / max(float(sigma_u_rad), 1.0e-7)
    B = inv_sigma * (A_scaled_d.to(device=device, dtype=dtype) @ L)  # [2N, 6]
    # R.T R = I + B.T B. QR retains the unit contribution along weak
    # directions even when a well-measured singular value is very large.
    _, upper = torch.linalg.qr(torch.cat((B, torch.eye(6, device=device, dtype=dtype))))
    root_t = torch.linalg.solve_triangular(upper.T, L.T, upper=False)
    P_plus = root_t.T @ root_t
    return 0.5 * (P_plus + P_plus.transpose(0, 1))


def propagate_reference_covariance_stm(P_scaled: Tensor, phi_si: Tensor) -> Tensor:
    """First-order two-body scaled covariance transport Phi_tilde P Phi_tilde^T (Q = 0)."""
    device, dtype = P_scaled.device, P_scaled.dtype
    d_vec = state_scale(device=device, dtype=dtype)
    phi_scaled = (1.0 / d_vec).unsqueeze(-1) * phi_si.to(device=device, dtype=dtype) * d_vec.unsqueeze(0)
    P_next = phi_scaled @ P_scaled @ phi_scaled.transpose(0, 1)
    return 0.5 * (P_next + P_next.transpose(0, 1))


def regularized_carryover_covariance(
    P_samples: Tensor | None,
    P_ref_plus: Tensor,
    delta: float,
) -> Tensor:
    """Single-delta carryover: P_carry^+ = (1 - delta) P_samples^+ + delta P_b^+ >= delta P_b^+."""
    d_val = float(delta)
    if P_samples is None or not bool(torch.isfinite(P_samples).all()):
        return P_ref_plus
    P_mix = (1.0 - d_val) * P_samples + d_val * P_ref_plus
    return 0.5 * (P_mix + P_mix.transpose(0, 1))


def vp_velocity_moment_loss(
    v_star: Tensor,
    v_theta: Tensor,
    V_theta: Tensor,
    *,
    fixed_isotropic_v: bool = False,
) -> Tensor:
    """Single multi-scale Gaussian NLL velocity + second-moment denoising objective (ROUND5 §3).

    L = 0.5 * E[ (v* - v_theta)^T V_theta^{-1} (v* - v_theta) + log det V_theta ].
    When fixed_isotropic_v=True (V_theta = I), reduces exactly to 0.5 * E[||v* - v_theta||^2].
    """
    err = (v_star - v_theta)
    if fixed_isotropic_v:
        return 0.5 * err.square().sum(dim=-1).mean()
    squeeze = err.ndim == 1
    e_b = err.unsqueeze(0) if squeeze else err
    V_b = V_theta.unsqueeze(0) if V_theta.ndim == 2 else V_theta
    L_v = safe_cholesky_spd(V_b)
    sol = torch.cholesky_solve(e_b.unsqueeze(-1), L_v).squeeze(-1)
    quad = (e_b * sol).sum(dim=-1)
    log_det = 2.0 * torch.diagonal(L_v, dim1=-2, dim2=-1).clamp_min(1.0e-12).log().sum(dim=-1)
    return 0.5 * (quad + log_det).mean()


class SinusoidalTimeEmbedding(nn.Module):
    """Fourier embedding for continuous diffusion time s in [0, pi/2]."""

    def __init__(self, dim: int = 16, dtype: torch.dtype = DTYPE) -> None:
        super().__init__()
        half = dim // 2
        freqs = torch.exp(torch.linspace(0.0, math.log(32.0), half, dtype=dtype))
        self.register_buffer("freqs", freqs, persistent=True)

    def forward(self, s: Tensor) -> Tensor:
        s_t = s.reshape(-1, 1).to(device=self.freqs.device, dtype=self.freqs.dtype)
        ang = s_t * self.freqs.unsqueeze(0)
        return torch.cat((torch.sin(ang), torch.cos(ang)), dim=-1)


class CosineVPDenoiserNet(nn.Module):
    """Route A time-conditioned velocity & SPD second-moment denoiser (v_theta, V_theta) = f_theta(z_s, s, c^-).

    Strict causality & zero double-counting: context c^- contains ONLY already-assimilated history,
    current propagated prior x^-, and reference scale L_b — NEVER the fresh unassimilated batch Y!
    """

    def __init__(
        self,
        *,
        context_dim: int = 64,
        hidden_dim: int = 96,
        time_dim: int = 16,
        delta: float = 0.10,
        fixed_isotropic_v: bool = False,
        dtype: torch.dtype = DTYPE,
    ) -> None:
        super().__init__()
        if not (0.0 < delta <= 1.0):
            raise ValueError("delta must lie in (0, 1]")
        self.context_dim = int(context_dim)
        self.hidden_dim = int(hidden_dim)
        self.delta = float(delta)
        self.fixed_isotropic_v = bool(fixed_isotropic_v)

        self.time_emb = SinusoidalTimeEmbedding(time_dim, dtype=dtype)
        in_dim = 6 + time_dim + context_dim
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, dtype=dtype),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, dtype=dtype),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, dtype=dtype),
            nn.SiLU(),
        )
        self.v_head = nn.Linear(hidden_dim, 6, dtype=dtype)
        # Predict 6x6 factor B_theta with V_theta = delta I + B_theta B_theta^T
        # Initialize B_theta = sqrt(1 - delta) I_6 so initial V_theta == I_6!
        self.b_head = nn.Linear(hidden_dim, 36, dtype=dtype)
        nn.init.normal_(self.v_head.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.v_head.bias)
        nn.init.normal_(self.b_head.weight, mean=0.0, std=1.0e-3)
        init_b = math.sqrt(max(1.0 - self.delta, 0.0)) * torch.eye(6, dtype=dtype).reshape(36)
        with torch.no_grad():
            self.b_head.bias.copy_(init_b)

    def forward(
        self, z_s: Tensor, s: Tensor | float, context: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return (v_theta [B,6], V_theta [B,6,6], m_s [B,6], C_s [B,6,6])."""
        squeeze = z_s.ndim == 1
        zb = z_s.unsqueeze(0) if squeeze else z_s
        batch = zb.shape[0]
        device = self.v_head.weight.device
        p_dtype = self.v_head.weight.dtype

        zb_p = zb.to(device=device, dtype=p_dtype)
        s_t = torch.as_tensor(s, device=device, dtype=p_dtype).reshape(-1)
        if s_t.numel() == 1:
            s_t = s_t.expand(batch)
        ctx = context.to(device=device, dtype=p_dtype)
        if ctx.ndim == 1:
            ctx = ctx.unsqueeze(0).expand(batch, -1)

        t_feat = self.time_emb(s_t)
        h = self.backbone(torch.cat((zb_p, t_feat, ctx), dim=-1))
        v_theta = self.v_head(h)

        eye6 = torch.eye(6, device=device, dtype=p_dtype).unsqueeze(0).expand(batch, 6, 6)
        if self.fixed_isotropic_v:
            V_theta = eye6
        else:
            B_mat = self.b_head(h).reshape(batch, 6, 6)
            V_theta = self.delta * eye6 + torch.bmm(B_mat, B_mat.transpose(-1, -2))

        alpha = torch.cos(s_t).unsqueeze(-1)
        sigma = torch.sin(s_t).unsqueeze(-1)
        m_s = alpha * zb_p - sigma * v_theta
        C_s = (sigma * sigma).unsqueeze(-1) * V_theta

        if squeeze:
            return (
                v_theta.squeeze(0).to(z_s.dtype),
                V_theta.squeeze(0).to(z_s.dtype),
                m_s.squeeze(0).to(z_s.dtype),
                C_s.squeeze(0).to(z_s.dtype),
            )
        return (
            v_theta.to(z_s.dtype),
            V_theta.to(z_s.dtype),
            m_s.to(z_s.dtype),
            C_s.to(z_s.dtype),
        )


def compute_noisy_los_guidance_o6(
    z_s: Tensor,
    m_s: Tensor,
    C_s: Tensor,
    *,
    H_z0: Tensor,
    nu_res: Tensor,
    sigma_u_rad: float,
    guidance_mode: str = "vjp",
) -> Tensor:
    """Compute noisy-time likelihood score g_s^loc = (D_z m_s)^T H^T (R + H C_s H^T)^{-1} nu in O(6^3).

    Solve the square-root least-squares problem with design [H L_C / sigma; I]
    and right hand side [nu / sigma; 0], then solve L_C.T g0 = solution.
    Cost is O(N d^2 + d^3) with d=6 and N angular residual components.

    guidance_mode:
      - 'vjp': exact denoiser Jacobian VJP (D_z m_s)^T g0 (default per ROUND5 §3)
      - 'identity_jac': ablation setting D_z m_s = I_6
      - 'none': unconditional diffusion prior ablation (returns 0)
    """
    if guidance_mode == "none" or H_z0.numel() == 0 or H_z0.shape[0] == 0:
        return torch.zeros_like(z_s)

    device, dtype = z_s.device, torch.float64
    H = H_z0.to(device=device, dtype=dtype)
    nu = nu_res.reshape(-1).to(device=device, dtype=dtype)
    C = C_s.detach().to(device=device, dtype=dtype)

    if not math.isfinite(sigma_u_rad) or sigma_u_rad <= 0:
        raise ValueError("sigma_u_rad must be positive and finite")
    L_c = safe_cholesky_spd(C)
    design = torch.cat((H @ L_c / sigma_u_rad, torch.eye(6, device=device, dtype=dtype)))
    q, upper = torch.linalg.qr(design, mode="reduced")
    rhs6 = q[:len(nu)].T @ (nu / sigma_u_rad)
    inner = torch.linalg.solve_triangular(upper, rhs6.unsqueeze(-1), upper=True).squeeze(-1)
    g0 = torch.linalg.solve_triangular(L_c.transpose(-2, -1), inner.unsqueeze(-1), upper=True).squeeze(-1)
    if not bool(torch.isfinite(g0).all()):
        raise FloatingPointError("Non-finite angular likelihood gradient")

    if guidance_mode == "identity_jac":
        return g0.to(dtype=z_s.dtype)
    if guidance_mode == "vjp":
        if not z_s.requires_grad:
            raise RuntimeError("guidance_mode='vjp' requires z_s.requires_grad=True")
        g_vjp = torch.autograd.grad(
            (m_s.to(torch.float64) * g0.detach()).sum(),
            z_s,
            retain_graph=False,
            create_graph=False,
        )[0]
        if not bool(torch.isfinite(g_vjp).all()):
            raise FloatingPointError("Non-finite denoiser VJP")
        return g_vjp.to(dtype=z_s.dtype)
    raise ValueError(f"Unknown guidance_mode: {guidance_mode!r}")


@dataclass(frozen=True)
class ReverseODETrace:
    """Full interior-midpoint reverse diffusion trace exposing step-by-step denoising & guidance."""

    z_final: Tensor  # [P, 6] terminal whitened samples at s = 0
    states_final_si: Tensor  # [P, 6] decoded ECI SI orbit samples
    mean_state_si: Tensor  # [6] pre-registered sample mean point estimate
    sample_cov_scaled: Tensor | None  # Unbiased sample covariance is undefined for P=1.
    carry_cov_scaled: Tensor  # Reference-regularized covariance used at the next epoch.
    s_midpoints: tuple[float, ...]
    z_trajectory: Tensor  # [K+1, P, 6]
    m_trajectory: Tensor  # [K, P, 6]
    guidance_norms: Tensor  # [K, P]


def integrate_vp_reverse_pf_ode(
    z_init: Tensor,
    rhs_fn: Callable[[Tensor, float], tuple[Tensor, Tensor, Tensor]],
    *,
    num_steps: int,
) -> tuple[Tensor, tuple[float, ...], Tensor, Tensor, Tensor]:
    """Integrate dz/ds = f_s(z) from s = pi/2 -> 0 over K uniform interior-midpoint intervals.

    Interval j: s_j -> s_{j-1}, Delta = s_{j-1} - s_j < 0, s_bar = 0.5 * (s_j + s_{j-1}).
    Update:
        k1 = f_{s_bar}(z_j)
        k2 = f_{s_bar}(z_j + 0.5 * Delta * k1)
        z_{j-1} = z_j + Delta * k2
    All evaluations occur strictly in the interior (0, pi/2), avoiding endpoint division by zero.
    """
    K = max(1, int(num_steps))
    s_grid = torch.linspace(HALF_PI, 0.0, K + 1, dtype=torch.float64)
    z = z_init.clone()
    z_hist = [z.detach().clone()]
    m_hist: list[Tensor] = []
    g_norms: list[Tensor] = []
    mids: list[float] = []

    for j in range(K):
        s_hi = float(s_grid[j].item())
        s_lo = float(s_grid[j + 1].item())
        delta_s = s_lo - s_hi  # < 0
        s_bar = 0.5 * (s_hi + s_lo)
        mids.append(s_bar)

        f1, m1, gn1 = rhs_fn(z, s_bar)
        z_mid = z + 0.5 * delta_s * f1
        f2, m2, gn2 = rhs_fn(z_mid, s_bar)
        z = (z + delta_s * f2).detach()

        z_hist.append(z.clone())
        m_hist.append(m2.detach().clone())
        g_norms.append(0.5 * (gn1.detach() + gn2.detach()))

    return (
        z,
        tuple(mids),
        torch.stack(z_hist, dim=0),
        torch.stack(m_hist, dim=0),
        torch.stack(g_norms, dim=0),
    )


class VPOrbitDiffusionFilter(nn.Module):
    """Route A Primary Model: Minimal 3+1 Knob Cosine-VP Conditional Orbital Diffusion Filter.

    Core knobs (3 default + 1 optional compute budget):
      - K (num_reverse_steps): artificial reverse diffusion intervals (default 4)
      - P (num_particles): independent initial Gaussian seeds / trajectories (default 8)
      - delta (min_ref_var_frac): shared floor for V_theta >= delta I and P_carry >= delta P_b^+ (default 0.10)
      - B_los (los_budget): optional cap on raw fresh-batch LOS per likelihood call (None = use all fresh LOS)
    """

    def __init__(
        self,
        *,
        num_reverse_steps: int = 4,
        num_particles: int = 8,
        delta: float = 0.10,
        los_budget: int | None = None,
        guidance_mode: str = "vjp",
        fixed_isotropic_v: bool = False,
        context_dim: int = 64,
        hidden_dim: int = 96,
        default_sigma_rad: float = 5.0e-4,
        dtype: torch.dtype = DTYPE,
    ) -> None:
        super().__init__()
        if int(num_reverse_steps) < 1:
            raise ValueError(f"num_reverse_steps must be >= 1, got {num_reverse_steps}")
        if int(num_particles) < 1:
            raise ValueError(f"num_particles must be >= 1, got {num_particles}")
        if los_budget is not None and int(los_budget) < 1:
            raise ValueError(f"los_budget must be >= 1, got {los_budget}")
        if not (0.0 < float(delta) <= 1.0):
            raise ValueError(f"delta must lie in (0, 1], got {delta}")
        self.num_reverse_steps = int(num_reverse_steps)
        self.num_particles = int(num_particles)
        self.delta = float(delta)
        self.los_budget = None if los_budget is None else int(los_budget)
        if guidance_mode not in ("vjp", "identity_jac", "none"):
            raise ValueError("Unknown likelihood guidance mode")
        self.guidance_mode = str(guidance_mode)
        self.fixed_isotropic_v = bool(fixed_isotropic_v)
        self.default_sigma_rad = float(default_sigma_rad)

        # Past-only causal context encoder (never fed current fresh batch Y!)
        self.tracklet_encoder = ContinuousTrackletEncoder(context_dim=48, dtype=dtype)
        # Context combines tracklet summary [48] + normalized prior state [6] + log diag(L_b) [6] + time/meta [4] = 64
        self.context_dim = int(context_dim)
        self.prior_ctx_proj = nn.Linear(48 + 6 + 6 + 4, context_dim, dtype=dtype)
        self.reference_prior = None
        self.denoiser = CosineVPDenoiserNet(
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            delta=delta,
            fixed_isotropic_v=fixed_isotropic_v,
            dtype=dtype,
        )
        self.ledger = ComputeLedger()

    def initial_moments(self, x0, station_at_init):
        from ..reference_prior import los_state_basis
        if self.reference_prior is None:
            return x0, build_diagonal_los_prior_covariance(x0, station_at_init[:3])
        basis = los_state_basis(x0, station_at_init[:3])
        mean = basis @ self.reference_prior["mean_los"].to(x0)
        cov = basis @ self.reference_prior["cov_los"].to(x0) @ basis.T
        return x0 + state_scale(device=x0.device, dtype=x0.dtype)*mean, cov

    def build_causal_past_context(
        self,
        *,
        prior_state_si: Tensor,
        L_b: Tensor,
        eval_time_sec: float,
        init_time_sec: float,
        times_sec: Tensor,
        observed_los: Tensor,
        station_states_si: Tensor,
        valid_mask: Tensor | None,
        prev_assimilated_idx: int,
        sigma_u_rad: float,
    ) -> Tensor:
        """Build strictly past-only context c^- excluding fresh unassimilated batch Y."""
        device = self.prior_ctx_proj.weight.device
        p_dtype = self.prior_ctx_proj.weight.dtype
        if prev_assimilated_idx >= 0:
            trk_ctx, _ = self.tracklet_encoder.encode_prefix(
                times_sec, observed_los, station_states_si, valid_mask,
                up_to_idx=prev_assimilated_idx, sigma_u_rad=sigma_u_rad,
            )
        else:
            trk_ctx = torch.zeros((48,), device=device, dtype=p_dtype)

        scale_phys = torch.tensor(
            [EARTH_RADIUS_M] * 3 + [V_SCALE_MPS] * 3, device=device, dtype=torch.float64
        )
        x_norm = (prior_state_si.to(device=device, dtype=torch.float64) / scale_phys).to(p_dtype)
        log_diag_L = torch.log(torch.diagonal(L_b).abs().clamp_min(1.0e-8)).to(device=device, dtype=p_dtype)
        meta = torch.tensor(
            [
                (float(eval_time_sec) - float(init_time_sec)) / float(T_SCALE_S),
                1.0 if prev_assimilated_idx >= 0 else 0.0,
                math.log10(max(float(sigma_u_rad), 1.0e-7)),
                math.log1p(max(0, prev_assimilated_idx + 1)),
            ],
            device=device,
            dtype=p_dtype,
        )
        return torch.tanh(self.prior_ctx_proj(torch.cat((trk_ctx.to(p_dtype), x_norm, log_diag_L, meta), dim=-1)))

    def _select_fresh_batch_indices(self, fresh_indices: np.ndarray) -> np.ndarray:
        if self.los_budget is None or fresh_indices.size <= self.los_budget:
            return fresh_indices
        sel = np.linspace(0, fresh_indices.size - 1, self.los_budget).round().astype(np.int64)
        return np.unique(fresh_indices[sel])

    def assimilate_fresh_batch(
        self,
        *,
        prior_state_si: Tensor,
        e_bar_b: Tensor,
        P_ref_minus: Tensor,
        eval_time_sec: float,
        station_at_eval_pos_si: Tensor,
        fresh_times_sec: Tensor,
        fresh_stations_si: Tensor,
        fresh_los_obs: Tensor,
        context_past: Tensor,
        sigma_u_rad: float,
        P_carry_minus: Tensor | None = None,
        num_steps: int | None = None,
        num_particles: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[ReverseODETrace, Tensor]:
        """Run K-interval reverse PF-ODE for P seeds guided ONLY by fresh LOS batch Y.

        Returns:
            trace: ReverseODETrace with P samples, sample mean, empirical sample covariance, and trajectory.
            trace.carry_cov_scaled: regularized carryover covariance.
            P_ref_plus: independent linearized reference covariance updated ONCE by Y.
        """
        device = prior_state_si.device
        dtype = torch.float64
        d_scale = state_scale(device=device, dtype=dtype)
        K = self.num_reverse_steps if num_steps is None else int(num_steps)
        P = self.num_particles if num_particles is None else int(num_particles)
        if K < 1:
            raise ValueError(f"num_steps must be >= 1, got {K}")
        if P < 1:
            raise ValueError(f"num_particles must be >= 1, got {P}")

        P_carry_in = P_ref_minus if P_carry_minus is None else P_carry_minus
        L_b = safe_cholesky_spd(P_carry_in.to(device=device, dtype=dtype))
        eb = e_bar_b.to(device=device, dtype=dtype)
        x_prior = prior_state_si.to(device=device, dtype=dtype)

        # Single linearized reference update P_ref_plus using independent P_ref_minus and fresh batch Y
        rep_ref = reproject_causal_window_single(
            x_prior + d_scale * eb,
            eval_time_sec,
            station_at_eval_pos_si,
            fresh_times_sec,
            fresh_stations_si,
            fresh_los_obs,
            compute_jacobian=True,
            ledger=self.ledger,
        )
        assert rep_ref.A_scaled_d is not None
        P_ref_plus = whitened_reference_covariance_update(P_ref_minus, rep_ref.A_scaled_d, sigma_u_rad)
        if not bool(torch.isfinite(P_ref_plus).all()):
            P_ref_plus = P_ref_minus

        z_init = torch.randn((P, 6), generator=generator, device=device, dtype=dtype)

        def _rhs(z_batch: Tensor, s_bar: float) -> tuple[Tensor, Tensor, Tensor]:
            tan_s = math.tan(s_bar)
            f_list, m_list, gn_list = [], [], []
            for p_idx in range(P):
                with torch.inference_mode(False), torch.enable_grad():
                    z_p = z_batch[p_idx].detach().clone().requires_grad_(self.guidance_mode == "vjp")
                    v_th, _, m_s, C_s = self.denoiser(z_p, s_bar, context_past)
                    if self.guidance_mode == "none" or fresh_times_sec.numel() == 0:
                        g_loc = torch.zeros_like(z_p)
                    else:
                        x_m = (x_prior + d_scale * (eb + L_b @ m_s.detach())).detach()
                        rep_m = reproject_causal_window_single(
                            x_m,
                            eval_time_sec,
                            station_at_eval_pos_si,
                            fresh_times_sec,
                            fresh_stations_si,
                            fresh_los_obs,
                            compute_jacobian=True,
                            ledger=self.ledger,
                        )
                        assert rep_m.A_scaled_d is not None
                        H_z0 = rep_m.A_scaled_d @ L_b  # [2N_Y, 6]
                        g_loc = compute_noisy_los_guidance_o6(
                            z_p,
                            m_s,
                            C_s,
                            H_z0=H_z0,
                            nu_res=rep_m.nu_0,
                            sigma_u_rad=sigma_u_rad,
                            guidance_mode=self.guidance_mode,
                        )
                f_p = v_th.detach() - tan_s * g_loc.detach()
                f_list.append(f_p)
                m_list.append(m_s.detach())
                gn_list.append(torch.linalg.vector_norm(g_loc.detach()))
            return torch.stack(f_list, dim=0), torch.stack(m_list, dim=0), torch.stack(gn_list, dim=0)

        z_fin, mids, z_traj, m_traj, g_norms = integrate_vp_reverse_pf_ode(
            z_init, _rhs, num_steps=K
        )
        # Decode P terminal samples to ECI SI: x = x^- + D (e_bar_b + L_b z_0)
        deltas_scaled = eb.unsqueeze(0) + (z_fin @ L_b.transpose(0, 1))
        states_si = x_prior.unsqueeze(0) + deltas_scaled * d_scale.unsqueeze(0)
        mean_state_si = states_si.mean(dim=0)
        if P > 1:
            centered = (states_si - mean_state_si.unsqueeze(0)) / d_scale.unsqueeze(0)
            P_samples = (centered.transpose(0, 1) @ centered) / float(P - 1)
            if not bool(torch.isfinite(P_samples).all()):
                P_samples = None
        else:
            P_samples = None

        P_carry_plus = regularized_carryover_covariance(P_samples, P_ref_plus, self.delta)
        if not bool(torch.isfinite(P_carry_plus).all()):
            P_carry_plus = P_ref_plus

        trace = ReverseODETrace(
            z_final=z_fin,
            states_final_si=states_si,
            mean_state_si=mean_state_si,
            sample_cov_scaled=P_samples,
            carry_cov_scaled=P_carry_plus,
            s_midpoints=mids,
            z_trajectory=z_traj,
            m_trajectory=m_traj,
            guidance_norms=g_norms,
        )
        return trace, P_ref_plus

    @torch.inference_mode(False)
    @torch.no_grad()
    def rollout_track(
        self,
        *,
        initial_state_si: Tensor,
        init_time_sec: float,
        times_sec: Tensor,
        observed_los: Tensor,
        station_states_si: Tensor,
        valid_mask: Tensor | None = None,
        initial_ref_cov_scaled: Tensor | None = None,
        initial_station_si: Tensor | None = None,
        seed: int = 20261028,
        explicit_flush: bool = False,
        block_start_time_sec: float | None = None,
        readout_until_sec: float | None = None,
        station_ids: Tensor | None = None,
        station_id_reliable: Tensor | None = None,
    ) -> dict[str, Any]:
        """Full multi-block online rollout propagating all P particles independently across gaps."""
        device = self.prior_ctx_proj.weight.device
        dtype = torch.float64
        d_scale = state_scale(device=device, dtype=dtype)
        gen = torch.Generator(device=device).manual_seed(int(seed))

        x0 = initial_state_si.to(device=device, dtype=dtype)
        t_cpu = times_sec.detach().to("cpu", dtype=dtype)
        T = int(t_cpu.shape[0])
        val_cpu = (
            torch.ones((T,), dtype=torch.bool)
            if valid_mask is None
            else valid_mask.detach().to("cpu", dtype=torch.bool)
        )
        obs = observed_los.to(device=device, dtype=dtype)
        stn = station_states_si.to(device=device, dtype=dtype)

        val_indices_all = np.flatnonzero(val_cpu.numpy())
        station_at_init = (x0.new_zeros(6) if initial_station_si is None else
                           torch.as_tensor(initial_station_si, device=device, dtype=dtype))
        x0, P_ref = self.initial_moments(x0, station_at_init)
        if initial_ref_cov_scaled is not None:
            P_ref = initial_ref_cov_scaled.to(device=device, dtype=dtype)
        P_carry = P_ref.clone()
        e_bar = torch.zeros((6,), device=device, dtype=dtype)

        particles_si: Tensor | None = None
        curr_anchor_x = x0
        curr_anchor_t = float(init_time_sec)
        prev_assimilated_idx = -1

        block2_t30_state: Tensor | None = None
        block2_t30_time: float | None = None
        batch_traces: list[dict[str, Any]] = []

        assim_events = causal_diffusion_events(
            t_cpu, val_cpu, init_time_sec, block_start_time_sec,
            explicit_flush=explicit_flush, readout_until_sec=readout_until_sec,
        )

        event_anchors: list[tuple[float, Tensor, Tensor]] = []
        for ev_t, last_idx, is_b2_ep in assim_events:
            dt = float(ev_t) - curr_anchor_t
            if abs(dt) > 0.0:
                prev_anchor_x = curr_anchor_x
                x_ref_prop, phi_gap = propagate_si_with_stm(
                    prev_anchor_x, dt, compute_stm=True, ledger=self.ledger
                )
                assert phi_gap is not None
                if particles_si is not None:
                    # Propagate all P particles independently across gap via two-body RK4!
                    particles_si, _ = propagate_si_with_stm(
                        particles_si, dt, compute_stm=False, ledger=self.ledger
                    )
                    curr_anchor_x = particles_si.mean(dim=0)
                    centered = (particles_si - curr_anchor_x.unsqueeze(0)) / d_scale.unsqueeze(0)
                    P_samp_prop = (
                        (centered.transpose(0, 1) @ centered) / float(particles_si.shape[0] - 1)
                        if particles_si.shape[0] > 1
                        else None
                    )
                else:
                    curr_anchor_x = x_ref_prop
                    P_samp_prop = None
                P_ref = propagate_reference_covariance_stm(P_ref, phi_gap)
                P_carry = regularized_carryover_covariance(P_samp_prop, P_ref, self.delta)
                curr_anchor_t = float(ev_t)

            event_prior = curr_anchor_x
            # Collect strictly fresh observations arriving after prev_assimilated_idx
            if last_idx > prev_assimilated_idx:
                prior_before_assim = curr_anchor_x.detach().clone()
                P_carry_in = P_carry.detach().clone()
                P_ref_in = P_ref.detach().clone()
                prev_before_assim = prev_assimilated_idx

                fresh_all = val_indices_all[
                    (val_indices_all > prev_assimilated_idx) & (val_indices_all <= last_idx)
                ]
                fresh_idx = self._select_fresh_batch_indices(fresh_all)

                # Strict separation: sigma_past uses ONLY <= prev_assimilated_idx (zero leak into c^-!),
                # while sigma_meas calibrates measurement noise R for fresh batch Y.
                if prev_assimilated_idx >= 0:
                    _, sigma_past, _ = estimate_online_los_noise_floor(
                        t_cpu, obs.detach().cpu(), valid_mask=val_cpu, up_to_idx=prev_assimilated_idx,
                        default_sigma_rad=self.default_sigma_rad,
                        station_states_si=stn.detach().cpu(),
                        station_ids=station_ids, station_id_reliable=station_id_reliable,
                    )
                else:
                    sigma_past = self.default_sigma_rad
                fresh_mask = val_cpu.clone()
                fresh_mask[:prev_assimilated_idx+1] = False
                _, sigma_meas, _ = estimate_online_los_noise_floor(
                    t_cpu, obs.detach().cpu(), valid_mask=fresh_mask, up_to_idx=last_idx,
                    default_sigma_rad=sigma_past, station_states_si=stn.detach().cpu(),
                    station_ids=station_ids, station_id_reliable=station_id_reliable,
                )
                L_carry = safe_cholesky_spd(P_carry)
                ctx_past = self.build_causal_past_context(
                    prior_state_si=curr_anchor_x,
                    L_b=L_carry,
                    eval_time_sec=curr_anchor_t,
                    init_time_sec=float(init_time_sec),
                    times_sec=t_cpu,
                    observed_los=obs,
                    station_states_si=stn,
                    valid_mask=val_cpu,
                    prev_assimilated_idx=prev_assimilated_idx,
                    sigma_u_rad=sigma_past,
                )
                trace, P_ref = self.assimilate_fresh_batch(
                    prior_state_si=curr_anchor_x,
                    e_bar_b=e_bar,
                    P_carry_minus=P_carry,
                    P_ref_minus=P_ref,
                    eval_time_sec=curr_anchor_t,
                    station_at_eval_pos_si=stn[int(fresh_idx[-1]), :3],
                    fresh_times_sec=t_cpu[fresh_idx].to(device=device),
                    fresh_stations_si=stn[fresh_idx],
                    fresh_los_obs=obs[fresh_idx],
                    context_past=ctx_past,
                    sigma_u_rad=sigma_meas,
                    generator=gen,
                )
                particles_si = trace.states_final_si
                curr_anchor_x = trace.mean_state_si
                P_carry = trace.carry_cov_scaled
                prev_assimilated_idx = last_idx
                batch_traces.append(
                    {
                        "eval_time_sec": curr_anchor_t,
                        "last_obs_idx": last_idx,
                        "prev_assimilated_idx": prev_before_assim,
                        "sigma_past_rad": sigma_past,
                        "sigma_meas_rad": sigma_meas,
                        "prior_state_si": prior_before_assim,
                        "posterior_state_si": curr_anchor_x.detach().clone(),
                        "P_carry_in": P_carry_in,
                        "P_carry": P_carry.detach().clone(),
                        "P_ref_in": P_ref_in,
                        "P_ref": P_ref.detach().clone(),
                        "trace": trace,
                    }
                )

            if is_b2_ep:
                block2_t30_state = curr_anchor_x.detach().clone()
                block2_t30_time = curr_anchor_t
            published = curr_anchor_x.unsqueeze(0) if particles_si is None else particles_si
            event_anchors.append((curr_anchor_t, published, event_prior))

        # Render per-frame causal states along times_sec
        priors_seq, posts_seq = [], []
        anc_ptr = 0
        act_x, act_t = x0.unsqueeze(0), float(init_time_sec)
        for k in range(T):
            if not bool(val_cpu[k].item()):
                priors_seq.append(torch.zeros_like(x0))
                posts_seq.append(torch.zeros_like(x0))
                continue
            tk = float(t_cpu[k].item())
            # A past event between two frames affects both prior and posterior of
            # the later frame. An event exactly at tk affects only its posterior.
            while anc_ptr < len(event_anchors) and event_anchors[anc_ptr][0] < tk:
                act_t, act_x, _ = event_anchors[anc_ptr]
                anc_ptr += 1
            if anc_ptr < len(event_anchors) and event_anchors[anc_ptr][0] == tk:
                act_t, act_x, pr_k = event_anchors[anc_ptr]
                anc_ptr += 1
                po_k = act_x.mean(dim=0)
            else:
                pushed, _ = propagate_si_with_stm(act_x, tk-act_t, compute_stm=False, ledger=self.ledger)
                pr_k = po_k = pushed.mean(dim=0)
            priors_seq.append(pr_k)
            posts_seq.append(po_k)

        return {
            "prior_states_si": torch.stack(priors_seq, dim=0),
            "posterior_states_si": torch.stack(posts_seq, dim=0),
            "batch_traces": batch_traces,
            "block2_t30_state_si": block2_t30_state,
            "block2_t30_time_sec": block2_t30_time,
            "ledger": self.ledger.as_dict(),
        }

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
        device = self.prior_ctx_proj.weight.device
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
            res = self.rollout_track(
                initial_state_si=init_x[b],
                init_time_sec=float(init_t[b].item()),
                times_sec=t_seq[b],
                observed_los=obs_seq[b],
                station_states_si=stn_seq[b],
                valid_mask=val_seq[b],
            )
            pr, po = res["prior_states_si"], res["posterior_states_si"]
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
            metadata={
                "model_family": "route_a_vp_orbit_diffusion",
                "K": self.num_reverse_steps,
                "P": self.num_particles,
                "delta": self.delta,
                "B_los": self.los_budget,
            },
        )


# ==============================================================================
# Route B (Separate Multi-Modal Proposals: B1 Guided Cosine Flow & B2 Twisted SMC)
# ==============================================================================


class RouteBGuidedCosineFlow2D(nn.Module):
    """Route B1: Deterministic 2D Cosine-Path Flow Matching with EXACT discrete K-step map density.

    Shared cosine probability path (omega = pi/2):
        z_t = cos(omega t) e + sin(omega t) z_1,   t in [0, 1]
        u_target = omega [-sin(omega t) e + cos(omega t) z_1]
    Time-dependent conditional affine subflows use the velocity and the scaled
    guide omega cot(omega t) at two coordinate values. Summing their log scales
    gives the log-Jacobian of the actual finite-step proposal.
    """

    def __init__(
        self,
        context_dim: int = 48,
        hidden_dim: int = 64,
        time_dim: int = 16,
        num_branches: int = 2,
        dtype: torch.dtype = DTYPE,
    ) -> None:
        super().__init__()
        self.context_dim = int(context_dim)
        self.num_branches = int(num_branches)
        self.time_emb = SinusoidalTimeEmbedding(time_dim, dtype=dtype)
        self.branch_emb = nn.Embedding(num_branches, 8, dtype=dtype)
        self.chart_embedding = nn.Embedding(2, context_dim, dtype=dtype)
        nn.init.zeros_(self.chart_embedding.weight)
        # Input: z [2] + t_emb [16] + branch_emb [8] + eta [4] + context [48]
        self.vel_net = nn.Sequential(
            nn.Linear(2 + time_dim + 8 + 4 + context_dim, hidden_dim, dtype=dtype),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, dtype=dtype),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2, dtype=dtype),
        )
        nn.init.normal_(self.vel_net[-1].weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.vel_net[-1].bias)

    def _raw_velocity(self, z: Tensor, t: Tensor | float, branch: Tensor, eta: Tensor, context: Tensor) -> Tensor:
        p_dtype = self.vel_net[0].weight.dtype
        device = self.vel_net[0].weight.device
        squeeze = z.ndim == 1
        zb = z.unsqueeze(0) if squeeze else z
        batch = zb.shape[0]
        t_v = torch.as_tensor(t, device=device, dtype=p_dtype).reshape(-1)
        if t_v.numel() == 1:
            t_v = t_v.expand(batch)
        br = branch.to(device=device, dtype=torch.long).reshape(-1)
        if br.numel() == 1:
            br = br.expand(batch)
        et = eta.to(device=device, dtype=p_dtype)
        if et.ndim == 1:
            et = et.unsqueeze(0).expand(batch, -1)
        ctx = context.to(device=device, dtype=p_dtype)
        if ctx.ndim == 1:
            ctx = ctx.unsqueeze(0).expand(batch, -1)

        inp = torch.cat((zb.to(p_dtype), self.time_emb(t_v), self.branch_emb(br), et, ctx), dim=-1)
        out = self.vel_net(inp).to(dtype=z.dtype)
        return out.squeeze(0) if squeeze else out

    def _affine_field(self, z, t, branch, eta, context, axis, guide=None):
        """Conditional offset and slope, independent of the coordinate being updated."""
        zero = z.clone()
        zero[..., axis] = 0.0
        one = zero.clone()
        one[..., axis] = 1.0
        offset = self._raw_velocity(zero, t, branch, eta, context)[..., axis]
        at_one = self._raw_velocity(one, t, branch, eta, context)[..., axis]
        if guide is not None:
            multiplier = HALF_PI / math.tan(HALF_PI * t)
            offset = offset + multiplier * guide(zero, t)[..., axis]
            at_one = at_one + multiplier * guide(one, t)[..., axis]
        return offset, at_one - offset

    def velocity(self, z: Tensor, t: Tensor | float, branch: Tensor, eta: Tensor, context: Tensor) -> Tensor:
        fields = []
        for axis in range(2):
            offset, slope = self._affine_field(z, t, branch, eta, context, axis)
            fields.append(offset + slope * z[..., axis])
        return torch.stack(fields, dim=-1)

    def transport(self, z, branch, eta, context, *, num_steps, guidance_fn=None, inverse=False):
        """Globally invertible, time-conditioned affine subflows and their actual log-volume."""
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        dt = (-1.0 if inverse else 1.0) / num_steps
        grid = reversed(range(num_steps)) if inverse else range(num_steps)
        log_volume = z.new_zeros(z.shape[:-1])
        for j in grid:
            t = (j + 0.5) / num_steps
            for axis, step in ((0, dt / 2), (1, dt), (0, dt / 2)):
                offset, slope = self._affine_field(z, t, branch, eta, context, axis, guidance_fn)
                # asinh is odd. Using the same conditioner and -step gives the exact inverse.
                log_scale = torch.asinh(step * slope)
                value = log_scale.exp() * z[..., axis] + step * (0.5 * log_scale).exp() * offset
                z = torch.stack((value, z[..., 1]) if axis == 0 else (z[..., 0], value), dim=-1)
                log_volume = log_volume + log_scale
        return z, log_volume

    def cosine_cfm_loss(
        self,
        z_1: Tensor,
        branch: Tensor,
        eta: Tensor,
        weights: Tensor,
        context: Tensor,
    ) -> Tensor:
        """Single weighted cosine-path flow matching velocity regression loss."""
        device, dtype = z_1.device, torch.float64
        M = int(z_1.shape[0])
        w = weights.to(device=device, dtype=dtype)
        w = w / w.sum().clamp_min(1.0e-15)

        t = torch.rand((M,), device=device, dtype=dtype)
        e = torch.randn((M, 2), device=device, dtype=dtype)
        ang = HALF_PI * t.unsqueeze(-1)
        z_t = torch.cos(ang) * e + torch.sin(ang) * z_1.to(dtype=dtype)
        v_target = HALF_PI * (-torch.sin(ang) * e + torch.cos(ang) * z_1.to(dtype=dtype))
        v_pred = self.velocity(z_t, t, branch, eta, context)
        return (w.detach() * (v_pred - v_target).square().sum(dim=-1)).sum()

    def sample_with_exact_discrete_log_prob(
        self,
        *,
        branch: Tensor,
        eta: Tensor,
        context: Tensor,
        num_steps: int = 4,
        guidance_fn: Callable[[Tensor, float], Tensor] | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Sample the trained conditional field with a bijective discrete transport.

        A deterministic guide enters the affine subflow conditioner. The returned mask marks
        finite particles; invertibility follows from the strictly positive coordinate scales.
        """
        device = self.vel_net[0].weight.device
        dtype = torch.float64
        branch = branch.reshape(-1)
        P = branch.numel()
        if P < 1 or num_steps < 1:
            raise ValueError("At least one particle and one flow step are required")
        if eta.ndim == 1:
            eta = eta.expand(P, -1)
        if eta.shape != (P, 4):
            raise ValueError("eta must have shape [P, 4] or [4]")
        e = torch.randn((P, 2), generator=generator, device=device, dtype=dtype)
        log_q0 = -0.5 * (e.square().sum(dim=-1) + 2.0 * math.log(2.0 * math.pi))
        if guidance_fn is None:
            samples, volumes = self.transport(e, branch, eta, context, num_steps=num_steps)
        else:
            pairs = [self.transport(e[i], branch[i:i+1], eta[i],
                                    context[i] if context.ndim == 2 else context,
                                    num_steps=num_steps, guidance_fn=guidance_fn) for i in range(P)]
            samples, volumes = (torch.stack(items) for items in zip(*pairs))
        log_q = log_q0 - volumes
        return samples, log_q, torch.isfinite(samples).all(dim=-1) & torch.isfinite(log_q)


class RouteBTwistedDiffusionSMC:
    """Route B2: Exact Discrete Stochastic Twisted Diffusion SMC (Wu et al., NeurIPS 2023 Alg. 1).

    Maintains exact incremental Feynman-Kac path weights along K discrete reverse steps:
        log G_{j-1} = log p_j(x_{j-1} | x_j) - log q_j(x_{j-1} | x_j, Y)
                      + log psi_{j-1}(x_{j-1}) - log psi_j(x_j)
    with terminal potential psi_0(x_0) = ell(Y | x_0) and initial psi_K = 1, so intermediate
    twisting potentials telescope out. Normalized expectations and log evidence can be biased
    at finite particle count, even when all path weights are exact.
    """

    @staticmethod
    def run_chain(
        *,
        x_init: Tensor,  # [P, D] samples from base noise p_K
        num_steps: int,
        base_transition_params: Callable[[Tensor, int], tuple[Tensor, Tensor]],
        guided_proposal_params: Callable[[Tensor, int], tuple[Tensor, Tensor]],
        log_twist_potential: Callable[[Tensor, int], Tensor],
        resample_ess_frac: float = 1.0,
        on_resample: Callable[[Tensor], None] | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run K-step Twisted Diffusion SMC and return (x_0 [P, D], norm_weights [P], log_Z)."""
        device, dtype = x_init.device, torch.float64
        x = x_init.to(device=device, dtype=dtype)
        P, dim = x.shape
        K = int(num_steps)
        if P < 1 or K < 1 or not 0.0 <= resample_ess_frac <= 1.0:
            raise ValueError("P and K must be positive and the ESS fraction must be in [0, 1]")

        log_w = torch.zeros((P,), device=device, dtype=dtype)
        log_z_est = torch.tensor(0.0, device=device, dtype=dtype)
        # psi_K == 1 => log_psi_curr == 0
        log_psi_curr = torch.zeros((P,), device=device, dtype=dtype)

        def _diag_gauss_log_prob(val: Tensor, mean: Tensor, std: Tensor) -> Tensor:
            if not bool(torch.isfinite(std).all() & (std > 0).all()):
                raise ValueError("Gaussian transition standard deviations must be finite and positive")
            s = std
            return -0.5 * (
                ((val - mean) / s).square() + 2.0 * torch.log(s) + math.log(2.0 * math.pi)
            ).sum(dim=-1)

        for j in range(K, 0, -1):
            mu_p, std_p = base_transition_params(x, j)
            mu_q, std_q = guided_proposal_params(x, j)
            eps = torch.randn((P, dim), generator=generator, device=device, dtype=dtype)
            x_prev = mu_q + std_q * eps

            log_p = _diag_gauss_log_prob(x_prev, mu_p, std_p)
            log_q = _diag_gauss_log_prob(x_prev, mu_q, std_q)
            log_psi_prev = log_twist_potential(x_prev, j - 1)  # at j-1 == 0, equals log ell(Y | x_0)

            if log_psi_prev.shape != (P,):
                raise ValueError("Twist potential must return one scalar per particle")
            alive = torch.isfinite(log_w)
            log_inc = (log_p - log_q) + (log_psi_prev - log_psi_curr)
            log_w = torch.where(alive, log_w + log_inc, -torch.inf)
            if bool(torch.isnan(log_w).any() | torch.isposinf(log_w).any()):
                raise ValueError("Non-finite SMC density ratio")
            if not bool(torch.isfinite(log_w).any()):
                return x_prev, torch.zeros_like(log_w), log_w.new_full((), -torch.inf)
            x = x_prev
            log_psi_curr = log_psi_prev

            # Systematic resampling when ESS drops (except after final step j == 1)
            if j > 1 and resample_ess_frac > 0.0:
                w_max = log_w.max()
                w_unnorm = torch.exp(log_w - w_max)
                w_norm = w_unnorm / w_unnorm.sum().clamp_min(1.0e-15)
                ess = 1.0 / (w_norm * w_norm).sum().item()
                if ess < resample_ess_frac * P:
                    log_z_est = log_z_est + w_max + torch.log(w_unnorm.mean().clamp_min(1.0e-15))
                    cdf = torch.cumsum(w_norm, dim=0)
                    cdf[-1] = 1.0
                    u0 = torch.rand((), generator=generator, device=device, dtype=dtype) / float(P)
                    positions = u0 + torch.arange(P, device=device, dtype=dtype) / float(P)
                    idx = torch.searchsorted(cdf, positions, right=True).clamp(0, P - 1)
                    x = x[idx]
                    log_psi_curr = log_psi_curr[idx]
                    if on_resample is not None:
                        on_resample(idx)
                    log_w = torch.zeros((P,), device=device, dtype=dtype)

        w_max = log_w.max()
        w_unnorm = torch.exp(log_w - w_max)
        log_z_est = log_z_est + w_max + torch.log(w_unnorm.mean().clamp_min(1.0e-15))
        w_norm = w_unnorm / w_unnorm.sum().clamp_min(1.0e-15)
        return x, w_norm, log_z_est


__all__ = [
    "CosineVPDenoiserNet",
    "HALF_PI",
    "ReverseODETrace",
    "RouteBGuidedCosineFlow2D",
    "RouteBTwistedDiffusionSMC",
    "SinusoidalTimeEmbedding",
    "VPOrbitDiffusionFilter",
    "build_diagonal_los_prior_covariance",
    "compute_noisy_los_guidance_o6",
    "integrate_vp_reverse_pf_ode",
    "propagate_reference_covariance_stm",
    "regularized_carryover_covariance",
    "safe_cholesky_spd",
    "vp_velocity_moment_loss",
    "whitened_reference_covariance_update",
]

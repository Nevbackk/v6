"""Per-arrival orbit filtering with angular and state-space cosine VP diffusion."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import Tensor, nn

from ..data.observation_clock import observation_frames
from .batched_physics import DTYPE, state_scale
from .normalization import EARTH_RADIUS_M, T_SCALE_S, V_SCALE_MPS
from .recursive_filter import TrainingFilterOutput
from .reprojection import (
    ComputeLedger,
    build_los_frame,
    RunningLOSNoise,
    propagate_si_with_stm,
)
from .v1_fd_asf import ContinuousTrackletEncoder


HALF_PI = 0.5 * math.pi


def safe_cholesky_spd(mat: Tensor, *, jitter: float = 1.0e-12) -> Tensor:
    """Symmetrize and compute Cholesky factor L such that mat ~= L L^T."""
    sym = 0.5 * (mat + mat.transpose(-1, -2))
    eye = torch.eye(sym.shape[-1], device=sym.device, dtype=sym.dtype)
    factor, info = torch.linalg.cholesky_ex(sym + jitter * eye)
    if bool((info == 0).all()):
        return factor
    # A rank-deficient particle cloud can have a scale much larger than an
    # absolute jitter. Repair only failed members at their own numerical scale.
    flat = sym.reshape(-1, sym.shape[-1], sym.shape[-1])
    failed = info.reshape(-1) != 0
    value, basis = torch.linalg.eigh(flat[failed])
    floor = (32 * torch.finfo(sym.dtype).eps * value.abs().amax(-1, keepdim=True)).clamp_min(jitter)
    clean = (basis * value.clamp_min(floor).unsqueeze(-2)) @ basis.transpose(-1, -2)
    fixed = torch.linalg.cholesky((clean + clean.transpose(-1, -2)) / 2 + floor.unsqueeze(-1) * eye)
    out = factor.reshape_as(flat).clone()
    out[failed] = fixed
    return out.reshape_as(sym)


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
        state_dim: int = 6,
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
        self.state_dim = int(state_dim)
        in_dim = self.state_dim + time_dim + context_dim
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, dtype=dtype),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, dtype=dtype),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, dtype=dtype),
            nn.SiLU(),
        )
        self.v_head = nn.Linear(hidden_dim, self.state_dim, dtype=dtype)
        self.b_head = nn.Linear(hidden_dim, self.state_dim**2, dtype=dtype)
        nn.init.normal_(self.v_head.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.v_head.bias)
        nn.init.normal_(self.b_head.weight, mean=0.0, std=1.0e-3)
        init_b = math.sqrt(max(1.0 - self.delta, 0.0)) * torch.eye(self.state_dim, dtype=dtype).flatten()
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

        eye = torch.eye(self.state_dim, device=device, dtype=p_dtype).expand(batch, -1, -1)
        if self.fixed_isotropic_v:
            V_theta = eye
        else:
            B_mat = self.b_head(h).reshape(batch, self.state_dim, self.state_dim)
            V_theta = self.delta * eye + torch.bmm(B_mat, B_mat.transpose(-1, -2))

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


class AngularNoiseDiffusion(nn.Module):
    """Conditional VP diffusion of standardized two-dimensional sensor residuals."""

    def __init__(self, dtype=DTYPE):
        super().__init__()
        self.net = CosineVPDenoiserNet(state_dim=2, context_dim=14, hidden_dim=48, dtype=dtype)

    @staticmethod
    def context(ray, station, previous_innovation, dt, sigma):
        return torch.cat(
            (
                ray,
                station[:3] / EARTH_RADIUS_M,
                station[3:] / V_SCALE_MPS,
                torch.asinh(previous_innovation / sigma),
                ray.new_tensor([math.log1p(dt / T_SCALE_S), math.log(sigma)]),
            )
        )

    def moments(self, context, steps, particles, generator):
        """Run the cosine VP ancestral chain and average the final conditional moments."""
        z = torch.randn((particles, 2), dtype=context.dtype, device=context.device, generator=generator)
        eye = torch.eye(2, dtype=z.dtype, device=z.device)
        for i in range(steps, 0, -1):
            s = HALF_PI * i / steps
            a = 0.0 if i == steps else math.cos(s)
            prev = math.cos(HALF_PI * (i - 1) / steps)
            ratio = a / prev
            beta = 1.0 - ratio**2
            _, _, clean, clean_cov = self.net(z, s, context)
            coeff = prev * beta / (1.0 - a * a)
            mean = coeff * clean + (ratio * (1.0 - prev * prev) / (1.0 - a * a)) * z
            covariance = coeff**2 * clean_cov + (beta * (1.0 - prev * prev) / (1.0 - a * a)) * eye
            if i == 1:
                average = mean.mean(0)
                centered = mean - average
                return average, covariance.mean(0) + centered.T @ centered / particles
            eps = torch.randn(z.shape, dtype=z.dtype, device=z.device, generator=generator)
            z = mean + (safe_cholesky_spd(covariance) @ eps.unsqueeze(-1)).squeeze(-1)
        raise ValueError("The reverse chain needs at least one step")

    def loss(self, clean, context):
        s = torch.rand(clean.shape[0], device=clean.device, dtype=clean.dtype) * HALF_PI
        noise = torch.randn_like(clean)
        a, b = s.cos()[:, None], s.sin()[:, None]
        v, cov, _, _ = self.net(a * clean + b * noise, s, context)
        return vp_velocity_moment_loss(a * noise - b * clean, v, cov)


def _whitened_likelihood_score(H, residual, covariance):
    """Square-root solve for one or a batch of independent local likelihoods."""
    L = safe_cholesky_spd(covariance.detach())
    eye = torch.eye(L.shape[-1], dtype=L.dtype, device=L.device).expand_as(L)
    q, upper = torch.linalg.qr(torch.cat((H @ L, eye), dim=-2), mode="reduced")
    rhs = q[..., : H.shape[-2], :].transpose(-1, -2) @ residual
    inner = torch.linalg.solve_triangular(upper, rhs, upper=True)
    return torch.linalg.solve_triangular(L.transpose(-1, -2), inner, upper=True).squeeze(-1)


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
    g0 = _whitened_likelihood_score(H / sigma_u_rad, nu[:, None] / sigma_u_rad, C)
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
    sample_cov_scaled: Tensor | None  # Empirical ensemble spread; undefined for P=1.
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


@dataclass
class OrbitStreamState:
    epoch: float
    initial_epoch: float
    posterior: Tensor
    reference_covariance: Tensor | None
    particles: Tensor | None
    history: Tensor | None
    previous_innovation: Tensor
    noise: RunningLOSNoise
    generator: torch.Generator
    ledger: ComputeLedger
    count: int = 0


class VPOrbitDiffusionFilter(nn.Module):
    """Shared angular reverse Markov chain with optional six-dimensional state diffusion."""

    def __init__(
        self,
        *,
        num_reverse_steps: int = 4,
        num_particles: int = 8,
        delta: float = 0.10,
        guidance_mode: str = "vjp",
        fixed_isotropic_v: bool = False,
        context_dim: int = 64,
        hidden_dim: int = 96,
        default_sigma_rad: float = 5.0e-4,
        known_initial: bool = False,
        dtype: torch.dtype = DTYPE,
    ) -> None:
        super().__init__()
        if int(num_reverse_steps) < 1:
            raise ValueError(f"num_reverse_steps must be >= 1, got {num_reverse_steps}")
        if int(num_particles) < 1:
            raise ValueError(f"num_particles must be >= 1, got {num_particles}")
        if not (0.0 < float(delta) <= 1.0):
            raise ValueError(f"delta must lie in (0, 1], got {delta}")
        self.num_reverse_steps = int(num_reverse_steps)
        self.num_particles = int(num_particles)
        self.delta = float(delta)
        if guidance_mode not in ("vjp", "identity_jac", "none"):
            raise ValueError("Unknown likelihood guidance mode")
        self.guidance_mode = str(guidance_mode)
        self.fixed_isotropic_v = bool(fixed_isotropic_v)
        self.default_sigma_rad = float(default_sigma_rad)

        # Past-only causal context encoder (never fed current fresh batch Y!)
        self.tracklet_encoder = (
            ContinuousTrackletEncoder(context_dim=48, dtype=dtype) if not known_initial else None
        )
        # Context combines tracklet summary [48] + normalized prior state [6] + log diag(L_b) [6] + time/meta [4] = 64
        self.context_dim = int(context_dim)
        self.prior_ctx_proj = (
            nn.Linear(48 + 6 + 6 + 4, context_dim, dtype=dtype) if not known_initial else None
        )
        self.known_initial = known_initial
        self.sensor = AngularNoiseDiffusion(dtype)
        self.reference_prior = None
        self.denoiser = (
            CosineVPDenoiserNet(
                context_dim=context_dim,
                hidden_dim=hidden_dim,
                delta=delta,
                fixed_isotropic_v=fixed_isotropic_v,
                dtype=dtype,
            )
            if not known_initial
            else None
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
        past_summary: Tensor | None = None,
    ) -> Tensor:
        """Build strictly past-only context c^- excluding fresh unassimilated batch Y."""
        device = self.prior_ctx_proj.weight.device
        p_dtype = self.prior_ctx_proj.weight.dtype
        if past_summary is not None:
            trk_ctx = past_summary
        elif prev_assimilated_idx >= 0:
            trk_ctx, _ = self.tracklet_encoder.encode_prefix(
                times_sec,
                observed_los,
                station_states_si,
                valid_mask,
                up_to_idx=prev_assimilated_idx,
                init_time_sec=init_time_sec,
            )
        else:
            trk_ctx = torch.zeros((48,), device=device, dtype=p_dtype)

        count = (
            prev_assimilated_idx + 1
            if valid_mask is None
            else int(torch.as_tensor(valid_mask)[: prev_assimilated_idx + 1].sum())
        )
        return self.state_context(
            prior_state_si, L_b, eval_time_sec - init_time_sec, sigma_u_rad, count, trk_ctx
        )

    def state_context(self, prior_state_si, L_b, elapsed, sigma, count, summary):
        """Condition on strictly past observations and the propagated prior."""
        scale = prior_state_si.new_tensor([EARTH_RADIUS_M] * 3 + [V_SCALE_MPS] * 3)
        meta = prior_state_si.new_tensor(
            [elapsed / T_SCALE_S, float(count > 0), math.log10(sigma), math.log1p(count)]
        )
        log_diag = torch.log(torch.diagonal(L_b).clamp_min(1.0e-8))
        return torch.tanh(self.prior_ctx_proj(torch.cat((summary, prior_state_si / scale, log_diag, meta))))

    def assimilate_frame(
        self, prior, P_ref, P_carry, station, observed, noise_scale, noise_shape_eci, context, generator
    ):
        """Use exactly the current LOS in the reference update and the noisy-state VJP."""
        scale = state_scale(device=prior.device, dtype=prior.dtype)
        L = safe_cholesky_spd(P_carry)

        def measurement(states):
            rel = states[..., :3] - station[:3]
            rho = rel.norm(dim=-1, keepdim=True).clamp_min(1.0)
            ray = rel / rho
            _, tangent = build_los_frame(ray)
            Et = tangent.transpose(-1, -2)
            Hpos = Et * scale[:3] / rho.unsqueeze(-1)
            H = torch.cat((Hpos, torch.zeros_like(Hpos)), dim=-1)
            residual = Et @ (observed - ray).unsqueeze(-1)
            root = noise_scale * safe_cholesky_spd(Et @ noise_shape_eci @ tangent)
            return (
                torch.linalg.solve_triangular(root, H, upper=False),
                torch.linalg.solve_triangular(root, residual, upper=False),
            )

        H, _ = measurement(prior)
        ref_plus = whitened_reference_covariance_update(P_ref, H, 1.0)

        def rhs(z, s):
            with torch.inference_mode(False), torch.enable_grad():
                z = z.detach().clone().requires_grad_(self.guidance_mode == "vjp")
                velocity, _, clean, cov = self.denoiser(z, s, context)
                if self.guidance_mode == "none":
                    guide = torch.zeros_like(z)
                else:
                    states = prior + scale * (clean.detach() @ L.T)
                    H, residual = measurement(states)
                    J = H @ L
                    # Two angular residuals per particle; batch all independent solves and VJPs.
                    g0 = _whitened_likelihood_score(J, residual, cov)
                    guide = (
                        torch.autograd.grad(clean, z, grad_outputs=g0.detach())[0]
                        if self.guidance_mode == "vjp"
                        else g0
                    )
                return (
                    (velocity - math.tan(s) * guide).detach(),
                    clean.detach(),
                    guide.norm(dim=-1).detach(),
                )

        # Marginally standard Gaussian seeds with zero ensemble mean prevent spurious
        # random walks between arrivals. The covariance below is the ensemble's second moment.
        count = self.num_particles
        z = torch.randn((count, 6), device=prior.device, dtype=prior.dtype, generator=generator)
        if count > 1:
            z = (z - z.mean(0)) * math.sqrt(count / (count - 1))
        z, mids, trajectory, means, norms = integrate_vp_reverse_pf_ode(
            z, rhs, num_steps=self.num_reverse_steps
        )
        states = prior + scale * (z @ L.T)
        mean = states.mean(0)
        residuals = (states - mean) / scale
        sample_cov = residuals.T @ residuals / len(states) if len(states) > 1 else None
        carry = regularized_carryover_covariance(sample_cov, ref_plus, self.delta)
        return ReverseODETrace(z, states, mean, sample_cov, carry, mids, trajectory, means, norms), ref_plus

    @torch.no_grad()
    def start_track(
        self,
        initial_state_si,
        init_time_sec,
        *,
        initial_ref_cov_scaled=None,
        initial_station_si=None,
        seed=20261028,
    ):
        x = torch.as_tensor(initial_state_si).to(self.sensor.net.v_head.weight)
        t0 = float(init_time_sec)
        if x.shape != (6,) or not math.isfinite(t0):
            raise ValueError("Expected a six-dimensional initial state and finite epoch")
        ref = hidden = None
        if not self.known_initial:
            station = (
                x.new_zeros(6) if initial_station_si is None else torch.as_tensor(initial_station_si).to(x)
            )
            x, ref = self.initial_moments(x, station)
            if initial_ref_cov_scaled is not None:
                ref = torch.as_tensor(initial_ref_cov_scaled).to(x)
            hidden = x.new_zeros(self.tracklet_encoder.context_dim)
        return OrbitStreamState(
            t0,
            t0,
            x,
            ref,
            None,
            hidden,
            x.new_zeros(3),
            RunningLOSNoise(default_sigma_rad=self.default_sigma_rad),
            torch.Generator(device=x.device).manual_seed(int(seed)),
            ComputeLedger(),
        )

    @torch.inference_mode(False)
    @torch.no_grad()
    def step(
        self,
        state,
        time_sec,
        observed_los,
        station_state_si,
        *,
        station_id=None,
        station_id_reliable=None,
        training_prior_only=False,
    ):
        """Advance one track in place and return this arrival's prediction and posterior."""
        now = float(time_sec)
        dt = now - state.epoch
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("Observation times must be finite and nondecreasing")
        station = torch.as_tensor(station_state_si).to(state.posterior)
        obs = torch.as_tensor(observed_los).to(state.posterior)
        prior, phi = propagate_si_with_stm(
            state.posterior, dt, compute_stm=not self.known_initial, ledger=state.ledger
        )
        ref = carry = None
        if phi is not None:
            ref = propagate_reference_covariance_stm(state.reference_covariance, phi)
            spread = None
            if state.particles is not None:
                pushed, _ = propagate_si_with_stm(
                    state.particles, dt, compute_stm=False, ledger=state.ledger
                )
                prior = pushed.mean(0)
                if len(pushed) > 1:
                    centered = (pushed - prior) / state_scale(device=prior.device, dtype=prior.dtype)
                    spread = centered.T @ centered / len(pushed)
            carry = regularized_carryover_covariance(spread, ref, self.delta)
        raw = nn.functional.normalize(obs, dim=0)
        ray = nn.functional.normalize(prior[:3] - station[:3], dim=0)
        _, tangent = build_los_frame(ray)
        sigma = state.noise.sigma
        sensor_ctx = self.sensor.context(ray, station, state.previous_innovation, dt, sigma)
        result = dict(
            prior_state_si=prior,
            P_ref_in=ref,
            P_carry_in=carry,
            tangent=tangent,
            sensor_context=sensor_ctx,
            sigma_past_rad=sigma,
        )
        posterior = prior
        if not training_prior_only:
            bias, R = self.sensor.moments(
                sensor_ctx, self.num_reverse_steps, self.num_particles, state.generator
            )
            innovation = tangent.T @ (raw - ray)
            noise_bias = tangent @ (sigma * bias)
            root = sigma * safe_cholesky_spd(R)
            white = torch.linalg.solve_triangular(root, (innovation - sigma * bias)[:, None], upper=False)
            result.update(
                predicted_los=nn.functional.normalize(ray + noise_bias, dim=0),
                angular_covariance=sigma**2 * R,
                observation_nll=0.5 * white.square().sum()
                + root.diag().log().sum()
                + math.log(2 * math.pi),
            )
            if not self.known_initial:
                ctx = self.state_context(
                    prior,
                    safe_cholesky_spd(carry),
                    now - state.initial_epoch,
                    sigma,
                    state.count,
                    state.history,
                )
                shape = tangent @ R @ tangent.T + (R.trace() / 2) * torch.outer(ray, ray)
                trace, ref = self.assimilate_frame(
                    prior,
                    ref,
                    carry,
                    station,
                    nn.functional.normalize(raw - noise_bias, dim=0),
                    sigma,
                    shape,
                    ctx,
                    state.generator,
                )
                state.particles = trace.states_final_si
                posterior = trace.mean_state_si
                result.update(trace=trace, P_ref=ref, P_carry=trace.carry_cov_scaled)
        if state.history is not None:
            state.history = self.tracklet_encoder.advance(state.history, raw, station, dt)
        state.posterior, state.reference_covariance, state.epoch = posterior, ref, now
        state.previous_innovation = raw - nn.functional.normalize(posterior[:3] - station[:3], dim=0)
        state.noise.observe(
            now,
            obs.detach().cpu().numpy(),
            station.detach().cpu().numpy(),
            None if station_id is None else int(station_id),
            None if station_id_reliable is None else bool(station_id_reliable),
        )
        state.count += 1
        result["posterior_state_si"] = posterior
        return result

    @torch.no_grad()
    def rollout_track(
        self,
        *,
        initial_state_si,
        init_time_sec,
        times_sec,
        observed_los,
        station_states_si,
        valid_mask=None,
        initial_ref_cov_scaled=None,
        initial_station_si=None,
        station_ids=None,
        station_id_reliable=None,
        seed=20261028,
        return_traces=False,
        training_pairs=False,
        pure_prior=False,
    ):
        state = self.start_track(
            initial_state_si,
            init_time_sec,
            initial_station_si=initial_station_si,
            initial_ref_cov_scaled=initial_ref_cov_scaled,
            seed=seed,
        )
        x = state.posterior
        n = len(times_sec)
        priors, posts = x.new_zeros((n, 6)), x.new_zeros((n, 6))
        nll, predicted, cov = x.new_zeros(n), x.new_zeros((n, 3)), x.new_zeros((n, 2, 2))
        traces, pairs = [], []
        for frame in observation_frames(times_sec, valid_mask, init_time_sec):
            k = frame.index
            out = self.step(
                state,
                frame.time_sec,
                observed_los[k],
                station_states_si[k],
                station_id=None if station_ids is None else station_ids[k],
                station_id_reliable=None if station_id_reliable is None else station_id_reliable[k],
                training_prior_only=pure_prior,
            )
            priors[k], posts[k] = out["prior_state_si"], out["posterior_state_si"]
            if not pure_prior:
                nll[k], predicted[k], cov[k] = (
                    out["observation_nll"],
                    out["predicted_los"],
                    out["angular_covariance"],
                )
            if training_pairs:
                pairs.append(
                    (
                        frame,
                        priors[k],
                        out["P_carry_in"],
                        out["tangent"],
                        out["sensor_context"],
                        out["sigma_past_rad"],
                    )
                )
            if return_traces and "trace" in out:
                traces.append(
                    dict(
                        out,
                        eval_time_sec=frame.time_sec,
                        last_obs_idx=k,
                        prev_assimilated_idx=frame.previous_index,
                    )
                )
        return dict(
            prior_states_si=priors,
            posterior_states_si=posts,
            observation_nll=nll,
            predicted_los=predicted,
            angular_covariance=cov,
            processed_frames=state.count,
            batch_traces=traces,
            training_pairs=pairs,
            ledger=state.ledger.as_dict(),
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
        device = self.sensor.net.v_head.weight.device
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
                **{
                    key: (
                        kwargs[key]
                        if unbatched and torch.as_tensor(kwargs[key]).ndim == 1
                        else kwargs[key][b]
                    )
                    for key in ("station_ids", "station_id_reliable", "initial_station_si")
                    if kwargs.get(key) is not None
                },
            )
            pr, po = res["prior_states_si"], res["posterior_states_si"]
            vb = val_seq[b].to(device=device)
            obs_b = obs_seq[b].to(device=device, dtype=torch.float64)
            u_hat = res["predicted_los"]
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
                "model_family": "known_initial_angular_diffusion"
                if self.known_initial
                else "route_a_vp_orbit_diffusion",
                "K": self.num_reverse_steps,
                "P": self.num_particles,
                "delta": self.delta,
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
    "AngularNoiseDiffusion",
    "OrbitStreamState",
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

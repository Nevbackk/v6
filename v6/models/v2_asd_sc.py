"""Per-frame subspace refinement and causal one-arrival re-anchoring."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from ..data.observation_clock import observation_frames
from .batched_physics import DTYPE, state_scale
from .recursive_filter import TrainingFilterOutput
from .reprojection import (
    ComputeLedger,
    estimate_online_los_noise_floor,
    los_noise_profile,
    geodesic_angle_rad,
    propagate_si_with_stm,
    reproject_causal_window_single,
    select_causal_keyframes,
)


V2_ABLATION_MODES = (
    "full",
    "fixed_g",
    "unconstrained_mlp6d",
    "no_quantile_aux",
    "obs_only",
    "state_only",
    "no_pending_memory",
)


@dataclass
class PendingSubspaceMemory:
    """Latent unresolved subspace carried across observation gaps via thin QR."""

    V_p: Tensor  # [6, p] orthonormal columns in D-scaled coordinates (0 <= p <= 6)
    r_diag_log: Tensor  # [p] log|diag(R)| from last QR transport
    r_mix_ratio: float = 0.0  # Frobenius off-diagonal ratio of R
    prev_q50: Tensor | None = None  # [p]
    prev_q90: Tensor | None = None  # [p]

    @property
    def dim_p(self) -> int:
        return int(self.V_p.shape[1])

    @staticmethod
    def empty(device: torch.device, dtype: torch.dtype = DTYPE) -> "PendingSubspaceMemory":
        return PendingSubspaceMemory(
            V_p=torch.zeros((6, 0), device=device, dtype=dtype),
            r_diag_log=torch.zeros((0,), device=device, dtype=dtype),
            r_mix_ratio=0.0,
        )


def orthonormal_complement(V: Tensor) -> Tensor:
    """Return orthonormal basis [6, 6-p] spanning V^perp in R^6."""
    device, dtype = V.device, V.dtype
    p = int(V.shape[1])
    if p == 0:
        return torch.eye(6, device=device, dtype=dtype)
    if p >= 6:
        return torch.zeros((6, 0), device=device, dtype=dtype)
    # Full SVD of V^T [p, 6] -> Vh [6, 6], last 6-p rows span ker(V^T) = V^perp
    _, _, Vh = torch.linalg.svd(V.transpose(0, 1), full_matrices=True)
    return Vh[p:, :].transpose(0, 1).contiguous()


def transport_pending_subspace_qr(
    memory: PendingSubspaceMemory,
    phi_si: Tensor,
) -> PendingSubspaceMemory:
    """Thin-QR transport across interval: D^{-1} Phi D V_p = Q R, V_p^+ = Q."""
    if memory.dim_p == 0:
        return memory
    device, dtype = memory.V_p.device, memory.V_p.dtype
    d_vec = state_scale(device=device, dtype=dtype)
    phi_scaled = (1.0 / d_vec).unsqueeze(-1) * phi_si.to(device=device, dtype=dtype) * d_vec.unsqueeze(0)
    mapped = phi_scaled @ memory.V_p  # [6, p]
    Q, R = torch.linalg.qr(mapped, mode="reduced")
    # Canonicalize column signs so diag(R) > 0 for smooth orientation continuity
    diag_r = torch.diagonal(R)
    signs = torch.where(diag_r >= 0.0, torch.ones_like(diag_r), -torch.ones_like(diag_r))
    Q = Q * signs.unsqueeze(0)
    R = signs.unsqueeze(-1) * R

    abs_diag = torch.diagonal(R).abs().clamp_min(1.0e-12)
    off_diag = R - torch.diag(torch.diagonal(R))
    mix_ratio = float(
        (torch.linalg.norm(off_diag) / torch.linalg.norm(R).clamp_min(1.0e-12)).detach().item()
    )
    return PendingSubspaceMemory(
        V_p=Q.detach(),
        r_diag_log=torch.log(abs_diag).detach(),
        r_mix_ratio=mix_ratio,
        prev_q50=memory.prev_q50,
        prev_q90=memory.prev_q90,
    )


def unlock_pending_directions_and_basis(
    A_plus: Tensor,
    b_plus: Tensor,
    memory: PendingSubspaceMemory,
    *,
    s_unlock: float = 0.15,
    disable_memory: bool = False,
) -> tuple[Tensor, Tensor, PendingSubspaceMemory, Tensor]:
    """Eliminate ordinary directions V_n and test conditional sensitivity K_p = (I - P_n) A_+ V_p.

    Constructs authorized writeback basis E_a = [V_n, V_p R_rel] and held basis E_h = V_p R_hold
    strictly from observed geometry (not elapsed block number!), guaranteeing E_h^T E_a = 0.

    Returns:
        E_a: [6, d_a] orthonormal basis of directions authorized to update.
        E_h: [6, 6 - d_a] orthonormal basis of directions held frozen (E_h^T Delta_y == 0).
        updated_memory: PendingSubspaceMemory retaining only still-unresolved axes E_h.
        cond_singular_values: [p] singular values of K_p for diagnostic logging.
    """
    device, dtype = A_plus.device, A_plus.dtype
    if disable_memory or memory.dim_p == 0:
        return (
            torch.eye(6, device=device, dtype=dtype),
            torch.zeros((6, 0), device=device, dtype=dtype),
            PendingSubspaceMemory.empty(device, dtype),
            torch.zeros((0,), device=device, dtype=dtype),
        )

    V_p = memory.V_p.detach()
    V_n = orthonormal_complement(V_p)  # [6, 6 - p]
    A_det = A_plus.detach()

    if V_n.shape[1] > 0 and A_det.shape[0] > 0:
        AV_n = A_det @ V_n  # [2W, 6 - p]
        # Orthogonal projector onto col(A_+ V_n) via thin QR for numerical stability
        Q_n, R_n = torch.linalg.qr(AV_n, mode="reduced")
        rank_mask = torch.diagonal(R_n).abs() > 1.0e-10
        Q_n_act = Q_n[:, rank_mask]
        AV_p = A_det @ V_p
        K_p = AV_p - Q_n_act @ (Q_n_act.transpose(0, 1) @ AV_p)
    else:
        K_p = A_det @ V_p

    if K_p.shape[0] == 0:
        E_a = V_n
        E_h = V_p
        return E_a, E_h, memory, torch.zeros((memory.dim_p,), device=device, dtype=dtype)

    U_k, S_k, Vh_k = torch.linalg.svd(K_p, full_matrices=False)
    # Pad singular values if 2W < p
    p = memory.dim_p
    if S_k.numel() < p:
        S_full = torch.zeros((p,), device=device, dtype=dtype)
        S_full[: S_k.numel()] = S_k
        _, _, Vh_full = torch.linalg.svd(K_p, full_matrices=True)
    else:
        S_full = S_k
        Vh_full = Vh_k

    rel_mask = S_full >= float(s_unlock)
    R_rel = Vh_full[rel_mask, :].transpose(0, 1)  # [p, p_rel]
    R_hold = Vh_full[~rel_mask, :].transpose(0, 1)  # [p, p_hold]

    V_rel = V_p @ R_rel if R_rel.shape[1] > 0 else torch.zeros((6, 0), device=device, dtype=dtype)
    E_h = V_p @ R_hold if R_hold.shape[1] > 0 else torch.zeros((6, 0), device=device, dtype=dtype)
    E_a_raw = torch.cat((V_n, V_rel), dim=1)
    if E_a_raw.shape[1] > 0:
        E_a, _ = torch.linalg.qr(E_a_raw, mode="reduced")
        # Project out any 1e-16 overlap with E_h so E_h^T E_a == 0 to machine precision
        if E_h.shape[1] > 0:
            E_a = E_a - E_h @ (E_h.transpose(0, 1) @ E_a)
            E_a, _ = torch.linalg.qr(E_a, mode="reduced")
    else:
        E_a = E_a_raw

    new_mem = PendingSubspaceMemory(
        V_p=E_h.detach(),
        r_diag_log=torch.zeros((E_h.shape[1],), device=device, dtype=dtype),
        r_mix_ratio=memory.r_mix_ratio,
    )
    return E_a, E_h, new_mem, S_full


@dataclass(frozen=True)
class MicrostepTrace:
    """Per-microstep outputs needed by multi-horizon, progressive, and quantile losses."""

    y_prev: Tensor  # [6]
    y_next: Tensor  # [6]
    delta_y: Tensor  # [6] = y_next - y_prev
    state_next_si: Tensor  # [6] = x_prior + D y_next
    V_phys: Tensor  # [6, d_a] = E_a V_a active orthonormal axes
    singular_values: Tensor  # [d_a] singular values of A E_a
    q50_pred: Tensor  # [d_a] predicted 0.5 quantile of log(eps_0 + |v_i^T d_l|)
    q90_pred: Tensor  # [d_a] predicted 0.9 quantile of log(eps_0 + |v_i^T d_l|)
    g_shrink: Tensor  # [d_a] shrinkage factors in [g_min, 1]
    c_ridge: Tensor  # [d_a] gated clipped ridge steps
    chi_active: Tensor  # [d_a] boolean activation mask
    groups: tuple[tuple[int, ...], ...] = ()


class SubspaceBlindDenoiserHead(nn.Module):
    """V2 §2.2 shared-weight directional quantile & bounded positive shrinkage head."""

    def __init__(
        self,
        *,
        ablation_mode: str = "full",
        hidden_dim: int = 64,
        g_min: float = 0.08,
        fixed_g_value: float = 0.35,
        lambda_0: float = 0.05,
        s_min: float = 0.05,
        tau_a_threshold: float = 2.6,  # familywise-calibrated normalized residual threshold
        clip_r: float = 3.0,  # max step per micro-iteration in D-scaled units
        degenerate_tol: float = 0.05,
        eps_0: float = 1.0e-4,
        dtype: torch.dtype = DTYPE,
    ) -> None:
        super().__init__()
        if ablation_mode not in V2_ABLATION_MODES:
            raise ValueError(f"ablation_mode must be one of {V2_ABLATION_MODES}")
        self.ablation_mode = ablation_mode
        self.hidden_dim = int(hidden_dim)
        self.g_min = float(g_min)
        self.fixed_g_value = float(fixed_g_value)
        self.lambda_0 = float(lambda_0)
        self.s_min = float(s_min)
        self.tau_a_threshold = float(tau_a_threshold)
        self.clip_r = float(clip_r)
        self.degenerate_tol = float(degenerate_tol)
        self.eps_0 = float(eps_0)

        # Sign-flip-invariant directional feature dim = 12
        self.feat_dim = 12
        self.quantile_net = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2, dtype=dtype),  # [q50, delta_q90 >= 0]
        )
        self.shrink_net = nn.Sequential(
            nn.Linear(self.feat_dim + 2, hidden_dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1, dtype=dtype),
        )
        # Unconstrained 6D MLP ablation arm (V2-Abl-2 / V2-Abl-1)
        self.free_mlp6d = nn.Sequential(
            nn.Linear(18, hidden_dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_dim, 6, dtype=dtype),
        )
        nn.init.zeros_(self.free_mlp6d[-1].weight)
        nn.init.zeros_(self.free_mlp6d[-1].bias)
        nn.init.zeros_(self.shrink_net[-1].weight)
        nn.init.constant_(self.shrink_net[-1].bias, -0.5)

    def _spectral_groups(self, values):
        """Partition ordered singular values into disjoint near-degenerate groups."""
        groups = []
        start = 0
        for i in range(1, len(values)+1):
            if i == len(values) or values[start]-values[i] > self.degenerate_tol*values[start].clamp_min(1.e-12):
                groups.append(tuple(range(start,i)))
                start = i
        return groups

    def forward_microstep(
        self,
        *,
        x_prior_si: Tensor,
        y_curr: Tensor,
        A_weighted: Tensor,  # [2W, 6] = W^{1/2} A_scaled_d
        b_weighted: Tensor,  # [2W] = W^{1/2} nu(y_curr), normalized by sigma_u
        is_new_frame_2w: Tensor,  # [2W] bool
        E_a: Tensor,  # [6, d_a] authorized writeback basis
        u_curr_los: Tensor,  # [3] current unit LOS
        r_mix_ratio: float = 0.0,
        window_span_sec: float = 0.0,
    ) -> MicrostepTrace:
        device, dtype = x_prior_si.device, x_prior_si.dtype
        p_dtype = self.quantile_net[0].weight.dtype
        d_scale = state_scale(device=device, dtype=dtype)
        d_a = int(E_a.shape[1])

        if self.ablation_mode == "unconstrained_mlp6d":
            E_a = torch.eye(6, device=device, dtype=dtype)
            d_a = 6
        if d_a == 0 or A_weighted.shape[0] == 0:
            y_next = y_curr
            empty = torch.zeros((0,), device=device, dtype=dtype)
            return MicrostepTrace(
                y_prev=y_curr,
                y_next=y_next,
                delta_y=torch.zeros_like(y_curr),
                state_next_si=x_prior_si + d_scale * y_next,
                V_phys=torch.zeros((6, 0), device=device, dtype=dtype),
                singular_values=empty,
                q50_pred=empty,
                q90_pred=empty,
                g_shrink=empty,
                c_ridge=empty,
                chi_active=torch.zeros((0,), device=device, dtype=torch.bool),
            )

        # Frame-stabilized Hellmann-Feynman SVD on restricted matrix M = A E_a:
        # orthonormal frames (U, V_a) are computed on M.detach() to guarantee zero NaN
        # backward when singular values are degenerate/zero, while s_i and a_i retain full autograd!
        M = A_weighted @ E_a  # [2W, d_a]
        U_det, S_det, Vh_det = torch.linalg.svd(M.detach(), full_matrices=False)
        k_rank = int(S_det.shape[0])
        U = U_det[:, :k_rank]  # [2W, k_rank]
        V_a = Vh_det[:k_rank, :].transpose(0, 1)  # [d_a, k_rank]
        V_phys = E_a @ V_a  # [6, k_rank]

        # Differentiable Hellmann-Feynman singular values & residual projections
        s_vals = torch.einsum("wi,wk,ki->i", U, M, V_a).clamp_min(0.0)  # [k_rank]
        a_proj = U.transpose(0, 1) @ b_weighted  # [k_rank]

        # Old-window vs Newly-arrived split projections (prevents dilution across observation gaps!)
        old_mask = (~is_new_frame_2w).to(dtype=dtype).unsqueeze(-1)
        new_mask = is_new_frame_2w.to(dtype=dtype).unsqueeze(-1)
        has_old = float(old_mask.sum().item()) > 0.5
        has_new = float(new_mask.sum().item()) > 0.5

        a_old = (U * old_mask).transpose(0, 1) @ b_weighted if has_old else torch.zeros_like(a_proj)
        a_new = (U * new_mask).transpose(0, 1) @ b_weighted if has_new else a_proj
        sign_agree = (a_old * a_new) / (a_old.abs() * a_new.abs() + 1.0)

        pos_energy = V_phys[:3, :].square().sum(dim=0)  # [k_rank] in [0, 1]
        vel_energy = V_phys[3:, :].square().sum(dim=0)
        u_hat = u_curr_los / torch.linalg.vector_norm(u_curr_los).clamp_min(1.0e-15)
        radial_pos_align = (u_hat.unsqueeze(-1) * V_phys[:3, :]).sum(dim=0).abs()
        radial_vel_align = (u_hat.unsqueeze(-1) * V_phys[3:, :]).sum(dim=0).abs()

        b_rms = torch.sqrt(b_weighted.square().mean().clamp_min(1.0e-12))
        b_new_rms = (
            torch.sqrt((b_weighted[is_new_frame_2w].square().mean()).clamp_min(1.0e-12))
            if has_new
            else b_rms
        )
        x_curr_scaled_norm = torch.linalg.vector_norm((x_prior_si / d_scale) + y_curr)

        # Strictly sign-flip-invariant directional features F_i: [k_rank, 12]
        F_raw = torch.stack(
            (
                torch.log1p(s_vals),
                torch.log(s_vals + 1.0e-4) / 10.0,
                a_proj.abs().clamp_max(100.0) / 10.0,
                torch.log1p(a_proj.abs()),
                a_new.abs().clamp_max(100.0) / 10.0,
                sign_agree,
                pos_energy,
                vel_energy,
                radial_pos_align,
                radial_vel_align,
                torch.full_like(s_vals, float(r_mix_ratio)),
                torch.full_like(s_vals, math.log1p(window_span_sec / 60.0)),
            ),
            dim=-1,
        )
        groups = self._spectral_groups(s_vals.detach())
        F_sym = F_raw.clone()
        for group in groups:
            if len(group) == 1:
                continue
            ix = list(group)
            rms_a = a_proj[ix].norm() / math.sqrt(len(ix))
            rms_new = a_new[ix].norm() / math.sqrt(len(ix))
            values = F_raw[ix].mean(0)
            values[2] = rms_a.clamp_max(100.) / 10.
            values[3] = torch.log1p(rms_a)
            values[4] = rms_new.clamp_max(100.) / 10.
            values[5] = (a_old[ix] @ a_new[ix]) / (1. + a_old[ix].norm()*a_new[ix].norm())
            values[8] = radial_pos_align[ix].norm() / math.sqrt(len(ix))
            values[9] = radial_vel_align[ix].norm() / math.sqrt(len(ix))
            F_sym[ix] = values
        F_sym = F_sym.to(dtype=p_dtype)
        q_out = self.quantile_net(F_sym)
        q50 = q_out[:, 0].to(dtype=dtype)
        q90 = (q_out[:, 0] + nn.functional.softplus(q_out[:, 1])).to(dtype=dtype)

        z_in = (
            torch.zeros_like(q_out)
            if self.ablation_mode == "no_quantile_aux"
            else torch.stack((q50, q90), dim=-1).to(dtype=p_dtype)
        )

        if self.ablation_mode == "fixed_g":
            g_i = torch.full((k_rank,), self.fixed_g_value, device=device, dtype=dtype)
        else:
            logits_g = self.shrink_net(torch.cat((F_sym, z_in), dim=-1)).squeeze(-1).to(dtype=dtype)
            g_i = self.g_min + (1.0 - self.g_min) * torch.sigmoid(logits_g)

        if self.ablation_mode == "unconstrained_mlp6d":
            global_in = torch.cat(
                (
                    F_sym.mean(dim=0),
                    (x_prior_si / d_scale + y_curr).to(dtype=p_dtype),
                ),
                dim=-1,
            )
            delta_y = self.free_mlp6d(global_in).to(dtype=dtype).clamp(-self.clip_r, self.clip_r)
            chi = torch.ones((k_rank,), device=device, dtype=torch.bool)
            c_i = V_phys.transpose(0, 1) @ delta_y
        else:
            clusters = groups

            c_i = torch.zeros_like(a_proj)
            chi = torch.zeros((k_rank,), device=device, dtype=torch.bool)
            for cl in clusters:
                cl_idx = torch.tensor(cl, device=device, dtype=torch.long)
                k_cl = len(cl)
                s_cl = s_vals[cl_idx]
                a_cl = a_proj[cl_idx]
                norm_a_cl = torch.linalg.vector_norm(a_cl.detach())
                s_mean_cl = s_cl.detach().mean()

                thresh_cl = self.tau_a_threshold * math.sqrt(k_cl)
                act_cl = (s_mean_cl >= self.s_min) & (norm_a_cl > thresh_cl)
                chi[cl_idx] = act_cl
                if act_cl:
                    raw_c_cl = (s_cl * a_cl) / (s_cl.square() + self.lambda_0)
                    norm_c_cl = torch.linalg.vector_norm(raw_c_cl)
                    max_norm_cl = self.clip_r * math.sqrt(k_cl)
                    if norm_c_cl > max_norm_cl:
                        scale_cl = max_norm_cl / norm_c_cl
                        c_i[cl_idx] = raw_c_cl * scale_cl
                    else:
                        c_i[cl_idx] = raw_c_cl

            delta_y_red = V_a @ (g_i * c_i)  # [d_a]
            # Exact restricted writeback: Delta_y = E_a Delta_y_red => E_h^T Delta_y == 0!
            delta_y = E_a @ delta_y_red

        y_next = y_curr + delta_y
        x_next = x_prior_si + d_scale * y_next
        return MicrostepTrace(
            y_prev=y_curr,
            y_next=y_next,
            delta_y=delta_y,
            state_next_si=x_next,
            V_phys=V_phys,
            singular_values=s_vals,
            q50_pred=q50[:0] if self.ablation_mode == "no_quantile_aux" else q50,
            q90_pred=q90[:0] if self.ablation_mode == "no_quantile_aux" else q90,
            g_shrink=g_i,
            c_ridge=c_i,
            chi_active=chi,
            groups=tuple(groups),
        )


class CausalReanchorHead(nn.Module):
    """V2 §2.3 Tier-2 learned (Delta ln rho, Delta rho_dot, p_reset) re-anchoring head.

    Fitted at the end of Sub-arc A when persistent residual mismatch remains after L=2 local
    microsteps; compared causally against frozen local track x_L on Sub-arc B; evaluated
    unbiasedly on Sub-arc C.
    """

    def __init__(self, hidden_dim: int = 48, max_dlog_rho: float = 1.5, max_drhodot_mps: float = 3000.0, dtype: torch.dtype = DTYPE) -> None:
        super().__init__()
        self.max_dlog_rho = float(max_dlog_rho)
        self.max_drhodot_mps = float(max_drhodot_mps)
        # Input: [b_old_rms, b_new_rms, med_ang/tau, log_rho, rho_dot/1000, span_norm, p_dim/6, cond_s_max] = 8
        self.net = nn.Sequential(
            nn.Linear(8, hidden_dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_dim, 3, dtype=dtype),  # [d_ln_rho, d_rhodot_scaled, logit_reset]
        )
        nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():
            self.net[-1].bias.zero_()
            # Conservative trigger prior ONLY on reset logit (index 2); zero range/rate jump at init!
            self.net[-1].bias[2] = -2.0

    def propose_reanchor(
        self,
        local_state_si: Tensor,
        station_si: Tensor,
        summary_8d: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Propose re-anchored state x_R preserving current LOS direction & angular rate u_dot."""
        p_dtype = self.net[0].weight.dtype
        out = self.net(summary_8d.to(dtype=p_dtype)).to(dtype=local_state_si.dtype)
        d_log_rho = self.max_dlog_rho * torch.tanh(out[0])
        d_rhodot = self.max_drhodot_mps * torch.tanh(out[1])
        logit_reset = out[2]

        rel_r = local_state_si[:3] - station_si[:3]
        rel_v = local_state_si[3:] - station_si[3:]
        rho = torch.linalg.vector_norm(rel_r).clamp_min(100.0)
        u = rel_r / rho
        rho_dot = (rel_v * u).sum()
        u_dot = (rel_v - rho_dot * u) / rho  # transverse angular rate preserved!

        rho_new = rho * torch.exp(d_log_rho)
        rho_dot_new = rho_dot + d_rhodot
        r_new = station_si[:3] + rho_new * u
        v_new = station_si[3:] + rho_dot_new * u + rho_new * u_dot
        return torch.cat((r_new, v_new), dim=0), logit_reset


@dataclass(frozen=True)
class CheckpointStepOutput:
    """Complete differentiable trace of one V2 observation frame for training and rollout."""

    prior_state_si: Tensor  # [6] x_m^-
    local_posterior_si: Tensor  # [6] x_L after L microsteps
    published_state_si: Tensor  # [6] x_m^+ (equals x_L unless B-arc reset accepted)
    microsteps: tuple[MicrostepTrace, ...]
    memory_out: PendingSubspaceMemory
    E_a: Tensor  # [6, d_a]
    E_h: Tensor  # [6, 6 - d_a]
    reanchor_candidate_si: Tensor | None = None
    reset_logit: Tensor | None = None
    switched_to_reanchor: bool = False


class ASDSCFilter(nn.Module):
    """Complete Model Plan V2 Filter (ASD-SC): progressive subspace blind denoiser + self-correction."""

    def __init__(
        self,
        *,
        ablation_mode: str = "full",
        num_microsteps: int = 2,
        max_keyframes: int = 16,
        hidden_dim: int = 64,
        g_min: float = 0.08,
        s_min: float = 0.05,
        s_unlock: float = 0.15,
        enable_reanchor: bool = True,
        delta_reanchor_rad2: float = 1.0e-6,
        default_sigma_rad: float = 5.0e-4,
        dtype: torch.dtype = DTYPE,
    ) -> None:
        super().__init__()
        self.ablation_mode = ablation_mode
        self.num_microsteps = int(num_microsteps)
        self.max_keyframes = int(max_keyframes)
        self.s_min = float(s_min)
        self.s_unlock = float(s_unlock)
        self.enable_reanchor = bool(enable_reanchor)
        self.delta_reanchor_rad2 = float(delta_reanchor_rad2)
        self.default_sigma_rad = float(default_sigma_rad)

        self.denoiser = SubspaceBlindDenoiserHead(
            ablation_mode=ablation_mode,
            hidden_dim=hidden_dim,
            g_min=g_min,
            s_min=s_min,
            dtype=dtype,
        )
        self.reanchor_head = CausalReanchorHead(dtype=dtype) if enable_reanchor else None
        self.ledger = ComputeLedger()

    def run_checkpoint(
        self,
        *,
        prior_state_si: Tensor,
        eval_time_sec: float,
        times_sec: Tensor,
        observed_los: Tensor,
        station_states_si: Tensor,
        valid_mask: Tensor | None,
        last_obs_idx: int,
        prev_last_obs_idx: int,
        memory_in: PendingSubspaceMemory,
        has_new_observations: bool = True,
        sigma_rad: float | None = None,
        station_ids: Tensor | None = None,
        station_id_reliable: Tensor | None = None,
    ) -> CheckpointStepOutput:
        """Execute one observation frame at eval_time_sec (L=2 microsteps + pending memory update)."""
        device, dtype = prior_state_si.device, prior_state_si.dtype
        d_scale = state_scale(device=device, dtype=dtype)

        # A caller without a fresh observation can only propagate.
        if (not has_new_observations) or last_obs_idx < 0:
            return CheckpointStepOutput(
                prior_state_si=prior_state_si,
                local_posterior_si=prior_state_si,
                published_state_si=prior_state_si,
                microsteps=(),
                memory_out=memory_in,
                E_a=torch.eye(6, device=device, dtype=dtype),
                E_h=torch.zeros((6, 0), device=device, dtype=dtype),
            )

        t_cpu = times_sec.detach().to("cpu", dtype=torch.float64)
        val_cpu = (
            torch.ones(t_cpu.shape, dtype=torch.bool)
            if valid_mask is None
            else valid_mask.detach().to("cpu", dtype=torch.bool)
        )
        if sigma_rad is None:
            tau_k, sigma_k, _ = estimate_online_los_noise_floor(
                t_cpu,
                observed_los.detach().cpu(),
                station_ids=station_ids,
                station_id_reliable=station_id_reliable,
                valid_mask=val_cpu,
                up_to_idx=last_obs_idx,
                default_sigma_rad=self.default_sigma_rad,
                station_states_si=station_states_si.detach().cpu(),
            )
        else:
            sigma_k, tau_k = sigma_rad, 3.0 * sigma_rad
        kf_idx, is_new_np = select_causal_keyframes(
            t_cpu,
            val_cpu,
            up_to_idx=last_obs_idx,
            prev_checkpoint_idx=prev_last_obs_idx,
            max_keyframes=self.max_keyframes,
        )
        if kf_idx.size == 0:
            return CheckpointStepOutput(
                prior_state_si=prior_state_si,
                local_posterior_si=prior_state_si,
                published_state_si=prior_state_si,
                microsteps=(),
                memory_out=memory_in,
                E_a=torch.eye(6, device=device, dtype=dtype),
                E_h=torch.zeros((6, 0), device=device, dtype=dtype),
            )

        kf_t = t_cpu[kf_idx].to(device=device, dtype=dtype)
        kf_s = station_states_si[kf_idx].to(device=device, dtype=dtype)
        kf_u = observed_los[kf_idx].to(device=device, dtype=dtype)
        is_new_t = torch.as_tensor(is_new_np, device=device, dtype=torch.bool)
        is_new_2w = is_new_t.repeat_interleave(2)
        stn_curr = station_states_si[last_obs_idx].to(device=device, dtype=dtype)
        u_curr = observed_los[last_obs_idx].to(device=device, dtype=dtype)

        if self.ablation_mode in ("state_only", "obs_only"):
            if self.ablation_mode == "state_only":
                features = torch.cat((prior_state_si / d_scale, prior_state_si.new_zeros(12)))
                anchor = prior_state_si
            else:
                span = kf_t[-1] - kf_t[0]
                timing = torch.stack((torch.log1p(span), torch.log1p(eval_time_sec-kf_t[-1]),
                                      torch.log1p(is_new_t.sum().to(dtype))))
                features = torch.cat((kf_u[0], kf_u[-1], kf_u.mean(0), stn_curr / d_scale, timing))
                anchor = stn_curr
            head = self.denoiser.free_mlp6d
            state = anchor + d_scale * head(features.to(head[0].weight.dtype)).to(dtype)
            dy = (state - prior_state_si) / d_scale
            empty = dy[:0]
            axes = dy.new_empty((6, 0))
            trace = MicrostepTrace(torch.zeros_like(dy), dy, dy, state, axes, empty,
                                   empty, empty, empty, empty, empty.bool())
            return CheckpointStepOutput(prior_state_si, state, state, (trace,),
                                        PendingSubspaceMemory.empty(device, dtype),
                                        torch.eye(6, device=device, dtype=dtype), axes)

        inv_sigma = 1.0 / max(float(sigma_k), 1.0e-5)

        # Initial probe at y_0 = 0 to test conditional unlock K_p and determine E_a, E_h
        rep0 = reproject_causal_window_single(
            prior_state_si,
            eval_time_sec,
            stn_curr[:3],
            kf_t,
            kf_s,
            kf_u,
            is_new_segment=is_new_t,
            compute_jacobian=True,
            ledger=self.ledger,
        )
        assert rep0.A_scaled_d is not None
        A0_w = rep0.A_scaled_d * inv_sigma
        b0_w = rep0.nu_0.reshape(-1) * inv_sigma

        disable_mem = self.ablation_mode == "no_pending_memory"
        if memory_in.dim_p > 0 and not disable_mem:
            E_a, E_h, mem_after_unlock, cond_s = unlock_pending_directions_and_basis(
                A0_w, b0_w, memory_in, s_unlock=self.s_unlock, disable_memory=False
            )
        else:
            # First short-arc checkpoint: enroll currently unobservable directions (s_i < s_min) into V_p
            _, S_init, Vh_init = torch.linalg.svd(A0_w.detach(), full_matrices=True)
            S_full6 = torch.zeros((6,), device=device, dtype=dtype)
            S_full6[: S_init.numel()] = S_init
            if not disable_mem and bool((S_full6 < self.s_min).any()):
                act_m = S_full6 >= self.s_min
                E_a = Vh_init[act_m, :].transpose(0, 1).contiguous()
                E_h = Vh_init[~act_m, :].transpose(0, 1).contiguous()
                mem_after_unlock = PendingSubspaceMemory(
                    V_p=E_h.detach(),
                    r_diag_log=torch.zeros((E_h.shape[1],), device=device, dtype=dtype),
                )
            else:
                E_a = torch.eye(6, device=device, dtype=dtype)
                E_h = torch.zeros((6, 0), device=device, dtype=dtype)
                mem_after_unlock = PendingSubspaceMemory.empty(device, dtype)
            cond_s = torch.zeros((0,), device=device, dtype=dtype)

        y_curr = torch.zeros((6,), device=device, dtype=dtype)
        traces: list[MicrostepTrace] = []

        for ell in range(self.num_microsteps):
            if ell == 0:
                A_w, b_w, span_sec = A0_w, b0_w, rep0.window_span_sec
            else:
                x_ell = prior_state_si + d_scale * y_curr
                rep_ell = reproject_causal_window_single(
                    x_ell,
                    eval_time_sec,
                    stn_curr[:3],
                    kf_t,
                    kf_s,
                    kf_u,
                    is_new_segment=is_new_t,
                    compute_jacobian=True,
                    ledger=self.ledger,
                )
                assert rep_ell.A_scaled_d is not None
                A_w = rep_ell.A_scaled_d * inv_sigma
                b_w = rep_ell.nu_0.reshape(-1) * inv_sigma
                span_sec = rep_ell.window_span_sec

            tr = self.denoiser.forward_microstep(
                x_prior_si=prior_state_si,
                y_curr=y_curr,
                A_weighted=A_w,
                b_weighted=b_w,
                is_new_frame_2w=is_new_2w,
                E_a=E_a,
                u_curr_los=u_curr,
                r_mix_ratio=memory_in.r_mix_ratio,
                window_span_sec=span_sec,
            )
            traces.append(tr)
            y_curr = tr.y_next

        x_local = prior_state_si + d_scale * y_curr

        # Optional Tier-2 Re-anchor proposal (for training loss & sub-arc A->B selection)
        x_reanchor: Tensor | None = None
        logit_reset: Tensor | None = None
        if self.reanchor_head is not None:
            rel_r = x_local[:3] - stn_curr[:3]
            rel_v = x_local[3:] - stn_curr[3:]
            rho_l = torch.linalg.vector_norm(rel_r).clamp_min(100.0)
            rhodot_l = (rel_v * (rel_r / rho_l)).sum()
            b_old_rms = (
                torch.sqrt(b0_w[~is_new_2w].square().mean().clamp_min(1.0e-12))
                if bool((~is_new_2w).any())
                else torch.tensor(0.0, device=device, dtype=dtype)
            )
            b_new_rms = (
                torch.sqrt(b0_w[is_new_2w].square().mean().clamp_min(1.0e-12))
                if bool(is_new_2w.any())
                else torch.sqrt(b0_w.square().mean().clamp_min(1.0e-12))
            )
            med_ang0 = torch.median(rep0.angles_rad)
            cond_max = cond_s.max() if cond_s.numel() > 0 else torch.tensor(0.0, device=device, dtype=dtype)
            summary_8d = torch.stack(
                (
                    b_old_rms.clamp_max(50.0) / 10.0,
                    b_new_rms.clamp_max(50.0) / 10.0,
                    (med_ang0 / max(tau_k, 1.0e-6)).clamp_max(50.0) / 10.0,
                    torch.log(rho_l / 1.0e6),
                    rhodot_l / 1000.0,
                    torch.tensor(math.log1p(rep0.window_span_sec), device=device, dtype=dtype),
                    torch.tensor(float(mem_after_unlock.dim_p) / 6.0, device=device, dtype=dtype),
                    torch.log1p(cond_max),
                )
            )
            x_reanchor, logit_reset = self.reanchor_head.propose_reanchor(
                x_local, stn_curr, summary_8d
            )

        return CheckpointStepOutput(
            prior_state_si=prior_state_si,
            local_posterior_si=x_local,
            published_state_si=x_local,
            microsteps=tuple(traces),
            memory_out=mem_after_unlock,
            E_a=E_a,
            E_h=E_h,
            reanchor_candidate_si=x_reanchor,
            reset_logit=logit_reset,
        )

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
        station_ids=None,
        station_id_reliable=None,
        return_records=False,
    ):
        """Freeze the current range candidate and test it on the next valid arrival."""
        x = torch.as_tensor(initial_state_si).to(next(self.parameters()))
        times = torch.as_tensor(times_sec).cpu().double()
        valid = (
            torch.ones_like(times, dtype=torch.bool)
            if valid_mask is None
            else torch.as_tensor(valid_mask).cpu().bool()
        )
        obs, stations = torch.as_tensor(observed_los).to(x), torch.as_tensor(station_states_si).to(x)
        sigmas, _ = los_noise_profile(
            times,
            obs,
            station_ids,
            station_id_reliable,
            valid,
            station_states_si=stations,
            default_sigma_rad=self.default_sigma_rad,
        )
        memory = PendingSubspaceMemory.empty(x.device, x.dtype)
        priors, posts = x.new_zeros((len(times), 6)), x.new_zeros((len(times), 6))
        pending, records = None, []
        trials = switches = 0
        for frame in observation_frames(times, valid, init_time_sec):
            k, tm, dt = frame.index, frame.time_sec, frame.dt_sec
            prior, phi = propagate_si_with_stm(x, dt, compute_stm=memory.dim_p > 0, ledger=self.ledger)
            if phi is not None:
                memory = transport_pending_subspace_qr(memory, phi)
            memory_in = memory
            out = self.run_checkpoint(
                prior_state_si=prior,
                eval_time_sec=tm,
                times_sec=times,
                observed_los=obs,
                station_states_si=stations,
                valid_mask=valid,
                last_obs_idx=k,
                prev_last_obs_idx=frame.previous_index,
                memory_in=memory,
                sigma_rad=float(sigmas[k]),
            )
            x, memory = out.published_state_si, out.memory_out
            switched = False
            if pending is not None:
                trials += 1
                t_a, local, candidate, logit = pending
                if float(logit) > 0:
                    args = (t_a, times[k : k + 1], stations[k : k + 1], obs[k : k + 1])
                    if (
                        self._blind_arc_mse(local, *args) - self._blind_arc_mse(candidate, *args)
                        > self.delta_reanchor_rad2
                    ):
                        x, _ = propagate_si_with_stm(
                            candidate, tm - t_a, compute_stm=False, ledger=self.ledger
                        )
                        memory = PendingSubspaceMemory.empty(x.device, x.dtype)
                        switched = True
                        switches += 1
            pending = None
            if not switched and self.enable_reanchor and out.reanchor_candidate_si is not None:
                pending = (tm, x, out.reanchor_candidate_si, out.reset_logit)
            priors[k], posts[k] = prior, x
            if return_records:
                records.append(
                    dict(
                        frame=frame,
                        prior_state_si=prior,
                        published_state_si=x,
                        memory_in=memory_in,
                        memory=memory,
                        reanchor_applied=switched,
                    )
                )
        return dict(
            prior_states_si=priors,
            posterior_states_si=posts,
            frame_records=records,
            reanchor_trials=trials,
            reanchor_switches=switches,
            ledger=self.ledger.as_dict(),
        )

    def _blind_arc_mse(
        self, state_si: Tensor, t_state_sec: float, arc_t: Tensor, arc_s: Tensor, arc_u: Tensor
    ) -> float:
        errs: list[Tensor] = []
        for j in range(int(arc_t.shape[0])):
            xj, _ = propagate_si_with_stm(
                state_si, float(arc_t[j].item()) - t_state_sec, compute_stm=False, ledger=self.ledger
            )
            rel = xj[:3] - arc_s[j, :3]
            errs.append(geodesic_angle_rad(rel, arc_u[j]).square())
        return float(torch.stack(errs).mean().item()) if errs else 0.0

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
        """Batched rollout interface compatible with v6/training.py and validate scripts."""
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
            res = self.rollout_track(
                initial_state_si=init_x[b],
                init_time_sec=float(init_t[b].item()),
                times_sec=t_seq[b],
                observed_los=obs_seq[b],
                station_states_si=stn_seq[b],
                valid_mask=val_seq[b],
            )
            pr = res["prior_states_si"]
            po = res["posterior_states_si"]
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
            metadata={"model_family": "v2_asd_sc", "ablation_mode": self.ablation_mode},
        )


__all__ = [
    "ASDSCFilter",
    "CausalReanchorHead",
    "CheckpointStepOutput",
    "MicrostepTrace",
    "PendingSubspaceMemory",
    "SubspaceBlindDenoiserHead",
    "V2_ABLATION_MODES",
    "orthonormal_complement",
    "transport_pending_subspace_qr",
    "unlock_pending_directions_and_basis",
]

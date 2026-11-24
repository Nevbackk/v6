"""Comprehensive Round 5 mathematical, causal, end-to-end, and bug-fix tests.

Covers:
1. All 7 Round 5 CPU mathematical benchmarks from ROUND5_DIFFUSION_AND_HYPERPARAM_REVIEW.md (acabb03):
   - Prior-whitened nullspace preservation (factor 1.0) vs correlated physical marginal leakage (0.64 / 1 -> 0.488).
   - Single-delta covariance carryover floor (var 4 -> 0.4, gain 0.615385).
   - Scalar VP Gaussian noisy-time likelihood score with exact denoiser Jacobian VJP (0.93327441)
     vs identity-Jacobian (0.851613) and midpoint reverse PF-ODE convergence (1.317647 / 0.235294).
   - Zero-hyperparameter 3D unnormalized chord disagreement (90/10 orthogonal branches -> 0.5/0.5 mass).
   - Route B1 exact discrete K-step map log-Jacobian matching finite-difference 2x2 Jacobian to < 1e-8.
   - Route B2 twisted SMC against the analytic two-mode weights.
2. Strict zero-leakage causality invariance of VPOrbitDiffusionFilter past-only context c^- against fresh batch Y.
3. End-to-end Route A (pure B0 + closed-loop rollout priors) and Route B1 training & rollout.
4. Regression verification for all 6 Round 5 audited baseline bug fixes in V1 & V2.
"""

from __future__ import annotations

import math
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

from v6.data.batch_collate import collate_filter_sequences
from v6.models.diffusion_orbit import (
    HALF_PI,
    CosineVPDenoiserNet,
    RouteBGuidedCosineFlow2D,
    RouteBTwistedDiffusionSMC,
    VPOrbitDiffusionFilter,
    build_diagonal_los_prior_covariance,
    compute_noisy_los_guidance_o6,
    integrate_vp_reverse_pf_ode,
    regularized_carryover_covariance,
    safe_cholesky_spd,
    vp_velocity_moment_loss,
    whitened_reference_covariance_update,
)
from v6.models.reprojection import estimate_online_los_noise_floor, propagate_si_with_stm
from v6.models.v1_fd_asf import (
    ConditionalRealNVP2D,
    ContinuousTrackletEncoder,
    DefensiveProposalMixture,
    DominanceGatedPublisher,
    FDASFSequentialFilter,
    compute_forecast_disagreement_weights,
)
from v6.models.v1_ltr_lm import ClassicalLMRefiner, JFBRRefiner, PriorProtectionGate, V1OnlineFilter
from v6.models.v2_asd_sc import ASDSCFilter, CausalReanchorHead, SubspaceBlindDenoiserHead
from v6.training_diffusion import (
    train_route_a_vp_step_on_record,
    train_route_b_cosine_flow_step_on_record,
)
from v6.training_v1 import train_fd_asf_proposal_step_on_record, train_v1_refiner_step_on_record
from v6.training_v2 import DAggerRolloutBuffer, V2LossWeights, train_two_checkpoint_pair

from tests.test_v1_v2_models import _make_synthetic_two_block_record


class TestRound5MathBenchmarks(unittest.TestCase):
    def test_whitened_nullspace_vs_correlated_physical_leakage(self):
        # Correlated prior P = [[1, 0.8], [0.8, 1]], A = [1, 0], R = 0.25 (sigma = 0.5)
        # Embed in 6x6 block-diagonal so we test the exact production function!
        P_minus = torch.eye(6, dtype=torch.float64)
        P_minus[0, 1] = 0.8
        P_minus[1, 0] = 0.8
        A = torch.zeros((1, 6), dtype=torch.float64)
        A[0, 0] = 1.0
        sigma = 0.5

        P_plus = whitened_reference_covariance_update(P_minus, A, sigma)
        # Physical dim 1 has A v = 0, yet A P^- v = 0.8 != 0 -> variance drops 1.0 -> 0.488!
        self.assertAlmostEqual(P_plus[1, 1].item(), 0.488, places=10)
        # Kalman gain for dim 1 on unit innovation: (P^- A^T)/(A P^- A^T + R) = 0.8 / 1.25 = 0.64
        K_gain = (P_minus @ A.T) / (A @ P_minus @ A.T + sigma**2)
        self.assertAlmostEqual(K_gain[1, 0].item(), 0.64, places=10)

        # In contrast, in the prior-whitened coordinate L^{-1}, nullspace ker(A L) has exact factor 1.0!
        L = safe_cholesky_spd(P_minus)
        B = (A @ L) / sigma
        _, _, Vt = torch.linalg.svd(B, full_matrices=True)
        v_null_whitened = Vt[1]  # orthogonal to row space of A L
        u_phys = torch.linalg.solve_triangular(L.T, v_null_whitened.unsqueeze(-1), upper=True).squeeze(-1)
        var_before = (u_phys @ P_minus @ u_phys).item()
        var_after = (u_phys @ P_plus @ u_phys).item()
        self.assertAlmostEqual(var_after / var_before, 1.0, places=10)

    def test_single_delta_carryover_floor(self):
        P_ref_plus = torch.diag(torch.tensor([0.2, 4.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64))
        P_collapsed = torch.zeros((6, 6), dtype=torch.float64)
        P_carry = regularized_carryover_covariance(P_collapsed, P_ref_plus, delta=0.10)
        self.assertAlmostEqual(P_carry[1, 1].item(), 0.4, places=12)
        gain_next = P_carry[1, 1].item() / (P_carry[1, 1].item() + 0.25)
        self.assertAlmostEqual(gain_next, 0.4 / 0.65, places=6)

    def test_scalar_vp_exact_vjp_guidance_and_midpoint_ode_convergence(self):
        # Exact scalar VP Gaussian benchmark from ROUND5_DIFFUSION_AND_HYPERPARAM_REVIEW.md (acabb03):
        # x_0 ~ N(0, var0=4), alpha=0.8, sigma=0.6, z=0.7, Y=1.4, R=0.25 (sigma_u=0.5)
        # In whitened z_0 = x_0 / 2 ~ N(0, 1), observation model is Y = H z_0 + eps with H = 2.0, R = 0.25.
        # Exact posterior in x_0: mean = 4/(4+0.25)*1.4 = 1.3176470588, var = 4*0.25/4.25 = 0.2352941176.
        alpha = 0.8
        sigma = 0.6
        # Prior z_0 ~ N(0, I_6): exact velocity v*(z, s) = 0 -> m_s(z) = alpha * z, C_s = sigma^2 I_6!
        # Wait: let's check x_0 = 2 z_0 with z_0 ~ N(0,1), z_x = alpha x_0 + sigma eps_x (unwhitened var0=4)
        # In unwhitened x-space (var0=4): m_s(z) = (alpha*var0 / (alpha^2*var0 + sigma^2)) * z,
        # C_s = (sigma^2 * var0 / (alpha^2*var0 + sigma^2)).
        var0 = 4.0
        denom = alpha**2 * var0 + sigma**2  # 0.64*4 + 0.36 = 2.92
        coeff = (alpha * var0) / denom
        c_val = (sigma**2 * var0) / denom

        z_vec = torch.zeros(6, dtype=torch.float64, requires_grad=True)
        with torch.no_grad():
            z_vec[0] = 0.7
        m_vec = coeff * z_vec
        C_mat = torch.eye(6, dtype=torch.float64) * c_val
        H = torch.zeros((1, 6), dtype=torch.float64)
        H[0, 0] = 1.0
        nu = torch.tensor([1.4 - float(m_vec[0].detach().item())], dtype=torch.float64)

        g_vjp = compute_noisy_los_guidance_o6(
            z_vec, m_vec, C_mat, H_z0=H, nu_res=nu, sigma_u_rad=0.5, guidance_mode="vjp"
        )
        g_id = compute_noisy_los_guidance_o6(
            z_vec, m_vec, C_mat, H_z0=H, nu_res=nu, sigma_u_rad=0.5, guidance_mode="identity_jac"
        )
        # Exact values verified in acabb03: 0.93327441449 (vjp) vs 0.85161290323 (identity_jac)
        self.assertAlmostEqual(g_vjp[0].item(), 0.93327441449, places=6)
        self.assertAlmostEqual(g_id[0].item(), 0.85161290323, places=6)

        # Now integrate the midpoint reverse PF-ODE in whitened z_0 ~ N(0, I_6) with H = [2, 0..0], Y = 1.4, sigma_u = 0.5
        # For standard normal prior z_0 ~ N(0, I), exact score is -z => v*(z, s) = 0, m_s(z) = cos(s) z, C_s = sin^2(s) I!
        H_w = torch.zeros((1, 6), dtype=torch.float64)
        H_w[0, 0] = 2.0
        y_obs = 1.4
        gen = torch.Generator().manual_seed(1234)
        # Use 200 antithetic Gaussian seeds for tight variance estimate
        z_half = torch.randn((100, 6), generator=gen, dtype=torch.float64)
        z_seeds = torch.cat((z_half, -z_half), dim=0)
        z_seeds[:, 0] = (z_seeds[:, 0] - z_seeds[:, 0].mean()) / z_seeds[:, 0].std(unbiased=False)

        errs_mean, errs_var = [], []
        for K in (1, 4, 8, 16, 64):
            def _rhs(z_b: torch.Tensor, s_bar: float):
                al, sg = math.cos(s_bar), math.sin(s_bar)
                tan_s = sg / al
                # Analytic VJP for m_s = al * z, C_s = sg^2 I:
                # g_s = al * H^T (R + sg^2 H H^T)^{-1} (Y - H m_s)
                # Here H H^T = 4, R = 0.25:
                denom_k = 0.25 + (sg * sg) * 4.0
                nu_k = y_obs - 2.0 * al * z_b[:, 0]
                g0 = (2.0 * nu_k) / denom_k
                g_v = torch.zeros_like(z_b)
                g_v[:, 0] = al * g0
                f_b = -tan_s * g_v  # v_theta == 0 for standard normal prior
                return f_b, al * z_b, g_v.norm(dim=-1)

            z_fin, _, _, _, _ = integrate_vp_reverse_pf_ode(z_seeds, _rhs, num_steps=K)
            x_fin = 2.0 * z_fin[:, 0]
            errs_mean.append(abs(x_fin.mean().item() - 1.3176470588235294))
            errs_var.append(abs(x_fin.var(unbiased=False).item() - 0.23529411764705882))

        # Monotone 2nd-order O(1/K^2) convergence with K, reaching < 1e-4 at K=64!
        self.assertLess(errs_mean[-1], errs_mean[0])
        self.assertLess(errs_var[-1], errs_var[0])
        self.assertLess(errs_var[-1], errs_var[-2])
        self.assertLess(errs_mean[-1], 1.0e-4)
        self.assertLess(errs_var[-1], 1.0e-4)

    def test_zero_hyperparam_3d_chord_disagreement(self):
        # 90/10 two-branch orthogonal LOS case from acabb03:
        # u1 = (1, 0, 0) w=0.9, u2 = (0, 1, 0) w=0.1 -> u_bar = (0.9, 0.1, 0)
        # ||u1 - u_bar|| = sqrt(0.01 + 0.01) = 0.141421356, ||u2 - u_bar|| = sqrt(0.81 + 0.81) = 1.272792206
        # Optimal branch masses: 0.9*0.141421 : 0.1*1.272792 == 0.5 : 0.5!
        stn = torch.zeros(3, dtype=torch.float64)
        # Construct 10 states at t_ref == next_view_time_sec (dt=0 -> exact u1, u2!)
        th = torch.zeros((10, 6), dtype=torch.float64)
        th[:9, 0] = 7.5e6
        th[9, 1] = 7.5e6
        th[:, 5] = 7200.0
        log_pi = torch.zeros(10, dtype=torch.float64)
        w_star, w_pi, dis = compute_forecast_disagreement_weights(
            th, log_pi, t_ref_sec=0.0, next_view_time_sec=0.0, next_station_pos_si=stn
        )
        self.assertAlmostEqual(dis[0].item(), math.sqrt(0.02), places=12)
        self.assertAlmostEqual(dis[9].item(), math.sqrt(1.62), places=12)
        self.assertAlmostEqual(w_pi[9].item(), 0.10, places=12)
        self.assertAlmostEqual(w_star[:9].sum().item(), 0.50, places=12)
        self.assertAlmostEqual(w_star[9].item(), 0.50, places=12)

    def test_route_b1_exact_discrete_jacobian_matches_full_map(self):
        torch.manual_seed(11)
        flow = RouteBGuidedCosineFlow2D(context_dim=16, hidden_dim=32)
        # Give non-zero output weights so the discrete flow is genuinely nonlinear!
        with torch.no_grad():
            flow.vel_net[-1].weight.normal_(0.0, 0.25)
            flow.vel_net[-1].bias.normal_(0.0, 0.1)

        ctx = torch.randn(16, dtype=torch.float64)
        branch = torch.tensor([0, 1, 0], dtype=torch.long)
        eta = torch.randn((3, 4), dtype=torch.float64) * 1.0e-3

        def _guidance(z: torch.Tensor, t_mid: float) -> torch.Tensor:
            return -0.05 * z / (1.0 + z.square().sum())

        gen1 = torch.Generator().manual_seed(77)
        with torch.no_grad():
            z_out, log_q_exact, inv_mask = flow.sample_with_exact_discrete_log_prob(
                branch=branch, eta=eta, context=ctx, num_steps=4, guidance_fn=_guidance, generator=gen1
            )
        self.assertTrue(bool(inv_mask.all()))

        # Check the density independently by finite differences and by the inverse map.
        gen2 = torch.Generator().manual_seed(77)
        e0 = torch.randn((3, 2), generator=gen2, dtype=torch.float64)
        for idx in range(3):
            def transport(z, inverse=False):
                return flow.transport(z, branch[idx:idx+1], eta[idx], ctx,
                                      num_steps=4, guidance_fn=_guidance, inverse=inverse)
            z_check, ld = transport(e0[idx])
            recovered, inverse_ld = transport(z_check, inverse=True)
            torch.testing.assert_close(recovered, e0[idx], atol=1e-12, rtol=1e-12)
            self.assertAlmostEqual(float(ld + inverse_ld), 0.0, places=12)
            torch.testing.assert_close(z_check, z_out[idx], atol=1e-12, rtol=1e-12)
            h = 1e-5
            cols = [(transport(e0[idx]+h*d)[0]-transport(e0[idx]-h*d)[0])/(2*h)
                    for d in torch.eye(2, dtype=torch.float64)]
            sign, fd_logdet = torch.linalg.slogdet(torch.stack(cols, dim=1))
            self.assertGreater(float(sign), 0.)
            log_q0 = -0.5 * (e0[idx].square().sum() + 2*math.log(2*math.pi))
            self.assertAlmostEqual(float(log_q0-fd_logdet), float(log_q_exact[idx]), places=8)

    def test_smc_resampling_ancestry_and_zero_support(self):
        x = torch.linspace(-2, 2, 20, dtype=torch.float64).reshape(-1, 1)
        base = lambda state, j: (state, torch.ones_like(state))
        genealogy = []
        def potential(state, j):
            return -40*state[:,0].square() if j else torch.full_like(state[:,0], -torch.inf)
        _, weights, log_z = RouteBTwistedDiffusionSMC.run_chain(
            x_init=x, num_steps=2, base_transition_params=base, guided_proposal_params=base,
            log_twist_potential=potential, on_resample=lambda idx: genealogy.append(idx.clone()),
            generator=torch.Generator().manual_seed(85))
        self.assertEqual(len(genealogy), 1)
        self.assertLess(genealogy[0].unique().numel(), x.shape[0])
        self.assertEqual(float(weights.sum()), 0.)
        self.assertEqual(float(log_z), -math.inf)

    def test_route_b2_twisted_diffusion_smc_matches_two_mode_posterior(self):
        # Exact 2-step linear Gaussian target where guided proposal q != true reverse p,
        # verifying Twisted SMC eliminates proposal bias and matches analytic posterior mean & log Z!
        # Prior: x_2 ~ N(0, 1), x_1 | x_2 ~ N(0.5 x_2, 0.5^2), x_0 | x_1 ~ N(0.5 x_1, 0.5^2)
        # => marginal x_0 ~ N(0, 0.5^4 * 1 + 0.5^2 * 0.25 + 0.25) = N(0, 0.375).
        # Likelihood: Y = 1.5 | x_0 ~ N(x_0, 0.25) => exact posterior:
        # var_post = 1/(1/0.375 + 4) = 3/20 = 0.15, mean_post = 0.15 * 4 * 1.5 = 0.90,
        # Z = N(1.5; 0, 0.375 + 0.25 = 0.625).
        P = 4096
        gen = torch.Generator().manual_seed(2026)
        x_init = torch.randn((P, 1), generator=gen, dtype=torch.float64)

        def _base_p(x: torch.Tensor, j: int):
            return 0.5 * x, torch.full_like(x, 0.5)

        def _guided_q(x: torch.Tensor, j: int):
            # Intentionally biased/shifted guided proposal toward Y = 1.5
            shift = 0.45 if j == 2 else 0.75
            return 0.5 * x + shift, torch.full_like(x, 0.6)

        def _log_twist(x: torch.Tensor, j_next: int):
            if j_next == 0:
                # Exact terminal log likelihood log N(1.5; x_0, 0.25)
                return -0.5 * (((1.5 - x[:, 0]) ** 2) / 0.25 + math.log(2.0 * math.pi * 0.25))
            # Intermediate approximate twist potential psi_1(x_1)
            return -0.5 * (((1.5 - 0.5 * x[:, 0]) ** 2) / 0.50)

        x0, w_norm, log_z = RouteBTwistedDiffusionSMC.run_chain(
            x_init=x_init,
            num_steps=2,
            base_transition_params=_base_p,
            guided_proposal_params=_guided_q,
            log_twist_potential=_log_twist,
            resample_ess_frac=0.5,
            generator=gen,
        )
        mean_est = (w_norm * x0[:, 0]).sum().item()
        var_est = (w_norm * (x0[:, 0] - mean_est) ** 2).sum().item()
        exact_log_z = -0.5 * ((1.5**2) / 0.625 + math.log(2.0 * math.pi * 0.625))

        self.assertAlmostEqual(mean_est, 0.90, delta=0.03)
        self.assertAlmostEqual(var_est, 0.15, delta=0.025)
        self.assertAlmostEqual(log_z.item(), exact_log_z, delta=0.04)


class TestRouteACausalityAndEndToEnd(unittest.TestCase):
    def test_past_context_bitwise_invariant_to_fresh_batch(self):
        rec = _make_synthetic_two_block_record()
        model = VPOrbitDiffusionFilter(num_reverse_steps=2, num_particles=4, delta=0.10)
        L_b = safe_cholesky_spd(build_diagonal_los_prior_covariance(rec["initial_state_si"], rec["station_states_si"][0, :3]))

        # Corrupt ALL frames strictly after index 11 (Block 2 fresh batch!)
        obs_corrupt = rec["observed_los"].clone()
        obs_corrupt[12:] = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)

        _, sig1, _ = estimate_online_los_noise_floor(rec["times_sec"], rec["observed_los"], up_to_idx=11)
        _, sig2, _ = estimate_online_los_noise_floor(rec["times_sec"], obs_corrupt, up_to_idx=11)
        self.assertEqual(sig1, sig2)

        ctx1 = model.build_causal_past_context(
            prior_state_si=rec["initial_state_si"],
            L_b=L_b,
            eval_time_sec=660.0,
            init_time_sec=0.0,
            times_sec=rec["times_sec"],
            observed_los=rec["observed_los"],
            station_states_si=rec["station_states_si"],
            valid_mask=rec["valid_mask"],
            prev_assimilated_idx=11,
            sigma_u_rad=sig1,
        )
        ctx2 = model.build_causal_past_context(
            prior_state_si=rec["initial_state_si"],
            L_b=L_b,
            eval_time_sec=660.0,
            init_time_sec=0.0,
            times_sec=rec["times_sec"],
            observed_los=obs_corrupt,
            station_states_si=rec["station_states_si"],
            valid_mask=rec["valid_mask"],
            prev_assimilated_idx=11,
            sigma_u_rad=sig2,
        )
        torch.testing.assert_close(ctx1, ctx2, atol=0.0, rtol=0.0)

    def test_route_a_training_and_frame_rollout(self):
        rec = _make_synthetic_two_block_record(
            pos_offset_m=(8000.0, -12000.0, 6000.0), vel_offset_mps=(-12.0, 18.0, -10.0)
        )
        model = VPOrbitDiffusionFilter(
            num_reverse_steps=2, num_particles=4, delta=0.10, guidance_mode="vjp"
        )
        opt = torch.optim.Adam(model.parameters(), lr=1.0e-3)

        # 1) Pure B0 prior training step
        loss_b0 = train_route_a_vp_step_on_record(
            model, rec, num_noise_samples=3, use_rollout_priors=False
        )
        self.assertTrue(torch.isfinite(loss_b0))
        loss_b0.backward()
        self.assertTrue(any(p.grad is not None and p.grad.norm().item() > 0.0 for p in model.denoiser.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.norm().item() > 0.0 for p in model.prior_ctx_proj.parameters()))
        opt.step()
        opt.zero_grad(set_to_none=True)

        # 2) Closed-loop rollout-prior training step
        loss_roll = train_route_a_vp_step_on_record(
            model, rec, num_noise_samples=3, use_rollout_priors=True, rollout_prior_prob=1.0
        )
        self.assertTrue(torch.isfinite(loss_roll))
        loss_roll.backward()
        self.assertTrue(any(p.grad is not None and p.grad.norm().item() > 0.0 for p in model.tracklet_encoder.parameters()))
        opt.step()

        # 3) Full multi-block rollout & forward_filter_sequence
        roll = model.rollout_track(
            initial_state_si=rec["initial_state_si"],
            init_time_sec=0.0,
            times_sec=rec["times_sec"],
            observed_los=rec["observed_los"],
            station_states_si=rec["station_states_si"],
            valid_mask=rec["valid_mask"],
            return_traces=True,
        )
        self.assertEqual(len(roll["batch_traces"]), len(rec["times_sec"]))
        first_tr = roll["batch_traces"][0]["trace"]
        self.assertEqual(tuple(first_tr.z_trajectory.shape), (3, 4, 6))  # [K+1, P, 6]
        self.assertEqual(tuple(first_tr.guidance_norms.shape), (2, 4))  # [K, P]
        self.assertTrue(torch.isfinite(roll["posterior_states_si"]).all())

        out_seq = model.forward_filter_sequence(**collate_filter_sequences([rec]))
        self.assertEqual(tuple(out_seq.posterior_states_si.shape), (1, 24, 6))
        self.assertTrue(torch.isfinite(out_seq.posterior_states_si).all())

    def test_route_b1_cosine_flow_training_step(self):
        rec = _make_synthetic_two_block_record()
        enc = ContinuousTrackletEncoder(context_dim=32)
        flow_b = RouteBGuidedCosineFlow2D(context_dim=32, hidden_dim=32)
        opt = torch.optim.Adam(list(enc.parameters()) + list(flow_b.parameters()), lr=1.0e-3)
        loss = train_route_b_cosine_flow_step_on_record(flow_b, enc, rec, num_ref_candidates=12)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.norm().item() > 0.0 for p in flow_b.parameters()))
        opt.step()


class TestRound5BaselineBugFixRegressions(unittest.TestCase):
    def test_fix1_stride20_noise_floor_triplet_coverage(self):
        # Stride-20s cadence: consecutive triplets span 40s (> old 25s cap, <= new 50s cap!)
        times = torch.arange(0.0, 200.0, 20.0, dtype=torch.float64)
        x_s0 = torch.tensor([6.878e6, 0.0, 0.0, 0.0, 7612.0, 0.0], dtype=torch.float64)
        x_t0 = torch.tensor([7.9e6, 1.0e5, 0.0, 0.0, 7100.0, 800.0], dtype=torch.float64)
        g = torch.Generator().manual_seed(55)
        los_list = []
        for tk in times.tolist():
            xs, _ = propagate_si_with_stm(x_s0, tk, compute_stm=False)
            xt, _ = propagate_si_with_stm(x_t0, tk, compute_stm=False)
            u = (xt[:3] - xs[:3]) / torch.linalg.vector_norm(xt[:3] - xs[:3])
            u = u + torch.randn(3, generator=g, dtype=torch.float64) * 3.0e-4
            los_list.append(u / torch.linalg.vector_norm(u))
        _, sigma_hat, n_trip = estimate_online_los_noise_floor(times, torch.stack(los_list), up_to_idx=9)
        self.assertGreaterEqual(n_trip, 6)
        self.assertTrue(math.isfinite(sigma_hat) and sigma_hat > 0.0)

    def test_fix2_jfbr_vel_only_nonzero_grad_and_full_prefix_wls_cap(self):
        rec = _make_synthetic_two_block_record(
            pos_offset_m=(12000.0, -15000.0, 8000.0), vel_offset_mps=(-20.0, 25.0, -15.0)
        )
        jf_vel = V1OnlineFilter(JFBRRefiner(mode="vel_only"))
        loss = train_v1_refiner_step_on_record(jf_vel, rec, loss_type="log_charbonnier")
        loss.backward()
        grad_norms = [p.grad.norm().item() for p in jf_vel.parameters() if p.grad is not None]
        self.assertTrue(any(g > 0.0 for g in grad_norms))

        full_wls = V1OnlineFilter(ClassicalLMRefiner("cl_full_prefix_wls"))
        self.assertEqual(full_wls.max_keyframes, 100000)

    def test_fix3_empty_basin_dominance_and_defensive_mixture_eps(self):
        pub = DominanceGatedPublisher()
        x_prev = torch.tensor([7.5e6, 0.0, 0.0, 0.0, 7200.0, 0.0], dtype=torch.float64)
        # Place all particles far away from polished mode so basin is empty (zero particles inside radius)
        far_thetas = (x_prev + torch.tensor([5.0e6, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)).unsqueeze(0).expand(4, 6).clone()
        w = torch.tensor([0.5, 0.3, 0.1, 0.1], dtype=torch.float64)
        dec = pub.select_published_orbit(
            prior_theta_ref_si=x_prev,
            prior_median_angle_rad=5.0e-3,
            tau_rad=1.0e-3,
            prior_nll=100.0,
            polished_modes_theta_ref_si=x_prev.unsqueeze(0),
            polished_modes_nll=torch.tensor([10.0], dtype=torch.float64),
            particle_thetas_ref_si=far_thetas,
            particle_weights=w,
        )
        self.assertEqual(dec.dominant_basin_weight, 0.0)
        self.assertFalse(dec.switched_from_prior)
        torch.testing.assert_close(dec.published_theta_ref_si, x_prev, atol=0.0, rtol=0.0)

        # Stratified mixture density uses actual eps_eff = n_phys / P (for P=6, eps_def=0.2 -> n_phys=1 -> 1/6)
        flow = ConditionalRealNVP2D(context_dim=16, hidden_dim=32)
        mix = DefensiveProposalMixture(flow, eps_def=0.2)
        ctx = torch.randn(16, dtype=torch.float64)
        z_s, b_s, log_q = mix.sample_and_log_prob(ctx, num_samples=6)
        self.assertEqual(tuple(z_s.shape), (6, 2))
        eps_eff = 1.0 / 6.0
        expected_log_q = torch.logaddexp(
            math.log(1.0 - eps_eff) + flow.log_prob(z_s, b_s, ctx),
            math.log(eps_eff) + mix._log_q_phys(z_s),
        )
        torch.testing.assert_close(log_q, expected_log_q, atol=1.0e-12, rtol=1.0e-12)

    def test_fix4_v2_reanchor_eligibility_init_and_unconstrained_ablation(self):
        # 1) CausalReanchorHead zero-jump init: bias[:2] == 0, bias[2] == -2
        head = CausalReanchorHead()
        self.assertAlmostEqual(head.net[-1].bias[0].item(), 0.0, places=12)
        self.assertAlmostEqual(head.net[-1].bias[1].item(), 0.0, places=12)
        self.assertAlmostEqual(head.net[-1].bias[2].item(), -2.0, places=12)

        # 2) B-arc re-anchoring actually becomes eligible on a multi-block record (fixes 0/5 bug!)
        rec = _make_synthetic_two_block_record(
            pos_offset_m=(30000.0, -40000.0, 20000.0), vel_offset_mps=(-30.0, 40.0, -25.0)
        )
        v2 = ASDSCFilter(ablation_mode="full", num_microsteps=1, enable_reanchor=True)
        roll = v2.rollout_track(
            initial_state_si=rec["initial_state_si"],
            init_time_sec=0.0,
            times_sec=rec["times_sec"],
            observed_los=rec["observed_los"],
            station_states_si=rec["station_states_si"],
        )
        self.assertGreaterEqual(roll["reanchor_trials"], 1)

        # 3) unconstrained_mlp6d genuinely bypasses E_a restriction
        v2_unc = ASDSCFilter(ablation_mode="unconstrained_mlp6d", num_microsteps=1)
        with torch.no_grad():
            v2_unc.denoiser.free_mlp6d[-1].bias.fill_(0.05)
        E_a = torch.eye(6, dtype=torch.float64)[:, :4]
        E_h = torch.eye(6, dtype=torch.float64)[:, 4:]
        tr = v2_unc.denoiser.forward_microstep(
            x_prior_si=rec["initial_state_si"],
            y_curr=torch.zeros(6, dtype=torch.float64),
            A_weighted=torch.randn((8, 6), dtype=torch.float64),
            b_weighted=torch.randn(8, dtype=torch.float64),
            is_new_frame_2w=torch.ones(8, dtype=torch.bool),
            E_a=E_a,
            u_curr_los=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        )
        self.assertGreater(torch.linalg.norm(E_h.T @ tr.delta_y).item(), 1.0e-3)

    def test_fix5_training_scripts_reject_paired_val(self):
        v6_root = Path(__file__).resolve().parents[1]
        for script_name in ("train_v1.py", "train_v2.py", "train_diffusion.py"):
            proc = subprocess.run(
                [
                    sys.executable,
                    "-s",
                    str(v6_root / "scripts" / script_name),
                    "--prepared-root",
                    "dummy_prep",
                    "--protocol-dir",
                    "dummy_proto",
                    "--run-dir",
                    "dummy_run",
                    "--eval-split",
                    "paired_val",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0, f"{script_name} MUST refuse --eval-split paired_val")
            self.assertIn("dev_select", proc.stderr)


class TestRound6SubmissionBlockerFixes(unittest.TestCase):
    """Regression verification for all Round 6 audited submission blockers."""

    def test_rotation_invariance_degenerate_subspace(self):
        # Verifies 55006ca rotation counterexample fix:
        # Under near-degenerate singular values, orthogonal basis rotation in the restricted
        # subspace leaves the physical microstep delta_y exactly invariant (< 1e-12).
        head = SubspaceBlindDenoiserHead()
        generator = torch.Generator().manual_seed(19)
        with torch.no_grad():
            head.shrink_net[-1].weight.normal_(generator=generator)
        E_a = torch.zeros(6, 2, dtype=torch.float64)
        E_a[0, 0] = 1.0
        E_a[1, 1] = 1.0

        A_w = torch.zeros(4, 6, dtype=torch.float64)
        A_w[0, 0] = 1.0
        A_w[1, 1] = 1.0
        b_w = torch.tensor([3.0, 3.0, 0.0, 0.0], dtype=torch.float64)
        is_new = torch.tensor([True, True, True, True], dtype=torch.bool)
        u = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        x_prior = torch.zeros(6, dtype=torch.float64)
        y_curr = torch.zeros(6, dtype=torch.float64)

        tr1 = head.forward_microstep(
            x_prior_si=x_prior,
            y_curr=y_curr,
            A_weighted=A_w,
            b_weighted=b_w,
            is_new_frame_2w=is_new,
            E_a=E_a,
            u_curr_los=u,
        )

        for th_deg in (30.0, 45.0, 60.0):
            th = math.radians(th_deg)
            R = torch.tensor([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]], dtype=torch.float64)
            E_a_rot = E_a @ R
            tr_rot = head.forward_microstep(
                x_prior_si=x_prior,
                y_curr=y_curr,
                A_weighted=A_w,
                b_weighted=b_w,
                is_new_frame_2w=is_new,
                E_a=E_a_rot,
                u_curr_los=u,
            )
            max_diff = torch.max(torch.abs(tr1.delta_y - tr_rot.delta_y)).item()
            self.assertLess(max_diff, 1.0e-12, f"Subspace rotation by {th_deg} deg must preserve physical delta_y")

    def test_p_ref_vs_p_carry_no_recursive_collapse(self):
        # Verifies separation of independent P_ref and carried P_carry:
        # P_ref evolves via Phi P_ref Phi^T and whitened update on P_ref_minus (never overwriting with update(P_carry)),
        # preventing recursive floor collapse across multiple observation blocks.
        rec = _make_synthetic_two_block_record()
        model = VPOrbitDiffusionFilter(
            num_reverse_steps=2, num_particles=4, delta=0.10
        )
        res = model.rollout_track(
            initial_state_si=rec["initial_state_si"],
            init_time_sec=0.0,
            times_sec=rec["times_sec"],
            observed_los=rec["observed_los"],
            station_states_si=rec["station_states_si"],
            valid_mask=rec["valid_mask"],
            return_traces=True,
        )
        traces = res["batch_traces"]
        self.assertGreaterEqual(len(traces), 2)
        for tr in traces:
            P_ref = tr["P_ref"]
            P_carry = tr["P_carry"]
            sample_cov = tr["trace"].sample_cov_scaled
            self.assertTrue(torch.all(P_ref.diag() > 0.0))
            self.assertTrue(torch.all(P_carry.diag() > 0.0))
            # P_carry has lower bound delta * P_ref
            self.assertTrue(torch.all(P_carry.diag() >= 0.10 * P_ref.diag() - 1.0e-12))
            self.assertTrue(torch.isfinite(sample_cov).all())

    def test_every_arrival_is_assimilated_including_last_frame(self):
        rec = _make_synthetic_two_block_record()
        model = VPOrbitDiffusionFilter(num_reverse_steps=1, num_particles=2)
        inputs = {
            k: rec[k]
            for k in (
                "initial_state_si",
                "init_time_sec",
                "times_sec",
                "observed_los",
                "station_states_si",
                "valid_mask",
            )
        }
        inputs["valid_mask"] = inputs["valid_mask"].clone()
        inputs["valid_mask"][4] = False
        out = model.rollout_track(**inputs, return_traces=True)
        actual = [r["last_obs_idx"] for r in out["batch_traces"]]
        self.assertEqual(actual, inputs["valid_mask"].nonzero().flatten().tolist())
        self.assertEqual(out["batch_traces"][-1]["eval_time_sec"], float(inputs["times_sec"][-1]))
        changed = inputs["observed_los"].clone()
        changed[-1] = torch.nn.functional.normalize(
            changed[-1] + changed.new_tensor([0.0, 0.01, 0.0]), dim=0
        )
        perturbed = model.rollout_track(**{**inputs, "observed_los": changed})
        torch.testing.assert_close(
            perturbed["posterior_states_si"][:-1], out["posterior_states_si"][:-1], rtol=0, atol=0
        )
        self.assertGreater(
            float((perturbed["posterior_states_si"][-1] - out["posterior_states_si"][-1]).norm()), 0.0
        )

    def test_frame_updates_are_invariant_to_time_origin(self):
        rec = _make_synthetic_two_block_record()
        model = VPOrbitDiffusionFilter(num_reverse_steps=1, num_particles=3)
        inputs = {
            k: rec[k]
            for k in ("initial_state_si", "init_time_sec", "times_sec", "observed_los", "station_states_si")
        }
        base = model.rollout_track(**inputs)
        shifted = model.rollout_track(
            **{**inputs, "init_time_sec": 100.0, "times_sec": inputs["times_sec"] + 100.0}
        )
        torch.testing.assert_close(
            base["posterior_states_si"], shifted["posterior_states_si"], rtol=1.0e-8, atol=1.0e-5
        )

    def test_v1_prior_protection_gate_holds_good_orbit(self):
        # Verifies V1OnlineFilter truthfully respects PriorProtectionGate on good orbit priors.
        rec = _make_synthetic_two_block_record(
            pos_offset_m=(0.0, 0.0, 0.0), vel_offset_mps=(0.0, 0.0, 0.0)
        )
        refiner = ClassicalLMRefiner("cl_gn1_matched")
        gate = PriorProtectionGate(min_arc_sec=60.0, c_protect=1.5)
        filter_v1 = V1OnlineFilter(refiner, prior_gate=gate)

        priors, posts, info = filter_v1.forward_single_track(
            initial_state_si=rec["truth_states_si"][0],
            init_time_sec=float(rec["times_sec"][0].item()),
            times_sec=rec["times_sec"][:4],
            observed_los=rec["observed_los"][:4],
            station_states_si=rec["station_states_si"][:4],
        )
        self.assertEqual(info["accepted_triggers"], 0)
        self.assertLess(torch.max(torch.abs(priors - posts)).item(), 1.0e-12)

    def test_fd_asf_all_infeasible_zero_loss(self):
        # Verifies FD-ASF proposal training gracefully returns zero-loss and zero gradient
        # when all decoded candidates are physically infeasible (subterranean or hyperbolic).
        rec = _make_synthetic_two_block_record()
        model = FDASFSequentialFilter()
        rec_bad = {key: value[:12] if torch.is_tensor(value) and value.ndim and len(value) == 24
                   else value for key, value in rec.items()}
        rec_bad["truth_states_si"] = torch.tensor([1.0e5, 0., 0., 0., 100., 0.],
                                                  dtype=torch.float64).expand(12, 6)
        rec_bad["station_states_si"] = torch.zeros_like(rec_bad["station_states_si"])
        rec_bad["observed_los"] = torch.tensor([1., 0., 0.], dtype=torch.float64).expand(12, 3)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(13)
            loss = train_fd_asf_proposal_step_on_record(model, rec_bad)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        self.assertTrue(all(g == 0.0 for g in grad_norms))

    def test_station_aware_noise_skips_handover(self):
        # Verifies estimate_online_los_noise_floor rejects handover triplets spanning different stations.
        times = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        obs = np.tile(np.array([1.0, 0.0, 0.0]), (5, 1))
        stn = np.zeros((5, 6))
        stn[:3, 0] = 7.0e6
        stn[3:, 1] = 7.0e6

        _, _, n_naive = estimate_online_los_noise_floor(times, obs, up_to_idx=4)
        _, _, n_aware = estimate_online_los_noise_floor(times, obs, up_to_idx=4, station_states_si=stn)
        self.assertEqual(n_naive, 3)
        self.assertEqual(n_aware, 1)


if __name__ == "__main__":
    unittest.main()

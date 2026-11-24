"""Comprehensive mathematical & end-to-end tests for Model Plan V1 and Model Plan V2."""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from v6.data.batch_collate import collate_filter_sequences
from v6.data.splits import partition_prepared_keys_by_source
from v6.eval_metrics import (
    cluster_paired_bootstrap_ci,
    directional_benefit_metrics,
    measure_secant_amplification,
)
from v6.models.batched_physics import state_scale
from v6.models.reprojection import (
    apply_local_msc_step,
    estimate_online_los_noise_floor,
    propagate_si_with_stm,
    reproject_causal_window_single,
)
from v6.models.v1_fd_asf import (
    AttributableChart2Plus4,
    ConditionalRealNVP2D,
    ContinuousTrackletEncoder,
    FDASFSequentialFilter,
    compute_forecast_disagreement_weights,
    solve_lambert_universal_m0,
)
from v6.models.v1_ltr_lm import (
    ClassicalLMRefiner,
    JFBRRefiner,
    LTRLMRefiner,
    V1OnlineFilter,
    solve_damped_normal_equations,
)
from v6.models.v2_asd_sc import (
    ASDSCFilter,
    PendingSubspaceMemory,
    transport_pending_subspace_qr,
    unlock_pending_directions_and_basis,
)
from v6.training_v1 import (
    evaluate_filter_records,
    train_fd_asf_proposal_step_on_record,
    train_v1_refiner_step_on_record,
)
from v6.training_v2 import (
    DAggerRolloutBuffer,
    V2LossWeights,
    compute_progressive_step_target,
    generate_counterfactual_prior_triplet,
    train_two_checkpoint_pair,
)


def _make_synthetic_two_block_record(
    *,
    pos_offset_m: tuple[float, float, float] = (0.0, 25000.0, -15000.0),
    vel_offset_mps: tuple[float, float, float] = (15.0, -35.0, 20.0),
    noise_rad: float = 0.0,
    seed: int = 42,
) -> dict[str, torch.Tensor | float]:
    """Generate a realistic 2-block (0..180s, gap 300s, 480..660s) two-body space-based LOS record."""
    g = torch.Generator().manual_seed(seed)
    dtype = torch.float64

    # Observer in LEO (6878 km), Target in higher inclined orbit (7800 km)
    x_stn_0 = torch.tensor([6.878e6, 0.0, 0.0, 0.0, 7612.0, 1000.0], dtype=dtype)
    x_tgt_0 = torch.tensor([7.800e6, 2.5e5, 1.2e5, -200.0, 7100.0, 1400.0], dtype=dtype)

    t_b1 = torch.linspace(15.0, 180.0, 12, dtype=dtype)
    t_b2 = torch.linspace(480.0, 660.0, 12, dtype=dtype)
    times = torch.cat((t_b1, t_b2), dim=0)

    stn_states, tgt_states, obs_los = [], [], []
    for tk in times.tolist():
        xs, _ = propagate_si_with_stm(x_stn_0, tk, compute_stm=False)
        xt, _ = propagate_si_with_stm(x_tgt_0, tk, compute_stm=False)
        rel = xt[:3] - xs[:3]
        u = rel / torch.linalg.vector_norm(rel)
        if noise_rad > 0.0:
            pert = torch.randn((3,), generator=g, dtype=dtype) * noise_rad
            u = (u + pert) / torch.linalg.vector_norm(u + pert)
        stn_states.append(xs)
        tgt_states.append(xt)
        obs_los.append(u)

    b0 = x_tgt_0 + torch.tensor([*pos_offset_m, *vel_offset_mps], dtype=dtype)
    return {
        "initial_state_si": b0,
        "init_time_sec": 0.0,
        "times_sec": times,
        "observed_los": torch.stack(obs_los, dim=0),
        "station_states_si": torch.stack(stn_states, dim=0),
        "station_ids": torch.zeros((times.shape[0],), dtype=torch.long),
        "station_id_reliable": torch.ones((times.shape[0],), dtype=torch.bool),
        "valid_mask": torch.ones((times.shape[0],), dtype=torch.bool),
        "truth_states_si": torch.stack(tgt_states, dim=0),
    }


class TestPhysicsAndV1Refiners(unittest.TestCase):
    def test_rk4_stm_exact_frechet_derivative(self):
        x0 = torch.tensor([7.2e6, -1.1e6, 0.8e6, 1200.0, 6800.0, -2100.0], dtype=torch.float64)
        for dt in (125.0, -300.0):
            _, phi = propagate_si_with_stm(x0, dt, compute_stm=True)
            assert phi is not None
            phi_auto = torch.autograd.functional.jacobian(
                lambda z: propagate_si_with_stm(z, dt, compute_stm=False)[0], x0
            )
            rel_err = torch.linalg.norm(phi - phi_auto) / torch.linalg.norm(phi_auto)
            self.assertLess(rel_err.item(), 1.0e-13)

    def test_reprojection_sign_and_msc_jacobian(self):
        # 1) Scalar R2 check: nu(y) = 1 - y => A = +1, lambda = 1 => y = +0.5 (residual^2: 1 -> 0.25, NOT 2.25)
        A_scalar = torch.zeros((2, 6), dtype=torch.float64)
        A_scalar[0, 0] = 1.0
        nu_scalar = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        y_step = solve_damped_normal_equations(
            A_scalar, nu_scalar, damping_diag=torch.ones(6, dtype=torch.float64)
        )
        self.assertAlmostEqual(y_step[0].item(), 0.5, places=12)
        new_res_sq = (1.0 - y_step[0].item()) ** 2
        self.assertAlmostEqual(new_res_sq, 0.25, places=12)

        # 2) Full 3D two-body finite-difference verification of A_local_msc = -D_y nu(0)
        rec = _make_synthetic_two_block_record(pos_offset_m=(5000.0, -8000.0, 3000.0))
        times = rec["times_sec"][:6]
        stns = rec["station_states_si"][:6]
        obs = rec["observed_los"][:6]
        tk = float(times[-1].item())
        xk, _ = propagate_si_with_stm(rec["initial_state_si"], tk, compute_stm=False)

        rep0 = reproject_causal_window_single(xk, tk, stns[-1, :3], times, stns, obs, compute_jacobian=True)
        assert rep0.A_local_msc is not None
        cols_fd = []
        eps = 1.0e-6
        for j in range(6):
            dy = torch.zeros(6, dtype=torch.float64)
            dy[j] = eps
            xp = apply_local_msc_step(xk, stns[-1, :3], dy, rep0.window_span_sec)
            xm = apply_local_msc_step(xk, stns[-1, :3], -dy, rep0.window_span_sec)
            rp = reproject_causal_window_single(
                xp, tk, stns[-1, :3], times, stns, obs, compute_jacobian=False, frozen_E_perp=rep0.E_perp
            )
            rm = reproject_causal_window_single(
                xm, tk, stns[-1, :3], times, stns, obs, compute_jacobian=False, frozen_E_perp=rep0.E_perp
            )
            # Negative derivative -D_y nu(0)
            cols_fd.append(-(rp.nu_0.reshape(-1) - rm.nu_0.reshape(-1)) / (2.0 * eps))
        A_fd = torch.stack(cols_fd, dim=-1)
        rel_diff = torch.linalg.norm(rep0.A_local_msc - A_fd) / torch.linalg.norm(A_fd)
        self.assertLess(rel_diff.item(), 1.0e-6)

    def test_schur_vel_equivalence_and_lock_pos_difference(self):
        torch.manual_seed(7)
        A = torch.randn((16, 6), dtype=torch.float64)
        nu = torch.randn((8, 2), dtype=torch.float64)
        w = torch.rand(8, dtype=torch.float64) * 0.8 + 0.2
        lam = torch.tensor([0.05, 0.01, 0.01, 0.05, 0.02, 0.02], dtype=torch.float64)

        y_joint = solve_damped_normal_equations(A, nu, frame_weights=w, damping_diag=lam, writeback_mode="joint_6d")
        y_schur = solve_damped_normal_equations(A, nu, frame_weights=w, damping_diag=lam, writeback_mode="schur_vel")
        y_lock = solve_damped_normal_equations(A, nu, frame_weights=w, damping_diag=lam, writeback_mode="lock_pos_vel")

        torch.testing.assert_close(y_schur[:3], torch.zeros(3, dtype=torch.float64), atol=0.0, rtol=0.0)
        torch.testing.assert_close(y_schur[3:], y_joint[3:], atol=1.0e-12, rtol=1.0e-12)
        self.assertGreater(torch.linalg.norm(y_schur[3:] - y_lock[3:]).item(), 1.0e-3)

    def test_pois10_unequal_time_noise_calibrator(self):
        rng = np.random.default_rng(123)
        gaps = rng.exponential(scale=4.0, size=60).clip(0.5, 12.0)
        times = torch.tensor(np.cumsum(gaps), dtype=torch.float64)
        x_s0 = torch.tensor([6.878e6, 0.0, 0.0, 0.0, 7612.0, 0.0], dtype=torch.float64)
        x_t0 = torch.tensor([7.9e6, 1.0e5, 0.0, 0.0, 7100.0, 800.0], dtype=torch.float64)
        true_sigma = 5.0e-4
        g = torch.Generator().manual_seed(99)
        los_list = []
        for tk in times.tolist():
            xs, _ = propagate_si_with_stm(x_s0, tk, compute_stm=False)
            xt, _ = propagate_si_with_stm(x_t0, tk, compute_stm=False)
            u = (xt[:3] - xs[:3]) / torch.linalg.vector_norm(xt[:3] - xs[:3])
            # Add isotropic tangent noise of std = true_sigma per tangent axis
            pert = torch.randn(3, generator=g, dtype=torch.float64)
            pert = pert - (pert * u).sum() * u
            pert = pert / torch.linalg.vector_norm(pert) * (
                true_sigma * math.sqrt(float(torch.randn(2, generator=g, dtype=torch.float64).square().sum().item()))
            )
            u_noisy = (u + pert) / torch.linalg.vector_norm(u + pert)
            los_list.append(u_noisy)
        obs = torch.stack(los_list, dim=0)
        tau_k, sigma_hat, n_trip = estimate_online_los_noise_floor(times, obs, up_to_idx=59)
        self.assertGreaterEqual(n_trip, 15)
        self.assertGreater(sigma_hat, 2.5e-4)
        self.assertLess(sigma_hat, 9.0e-4)
        self.assertAlmostEqual(tau_k, 3.0 * sigma_hat, places=12)

    def test_v1_zero_residual_identity_and_training_step(self):
        # Exact zero-step preservation on noiseless truth orbit
        rec_clean = _make_synthetic_two_block_record(pos_offset_m=(0.0, 0.0, 0.0), vel_offset_mps=(0.0, 0.0, 0.0))
        tk = float(rec_clean["times_sec"][11].item())
        xk = rec_clean["truth_states_si"][11]
        kf_t = rec_clean["times_sec"][:12]
        kf_s = rec_clean["station_states_si"][:12]
        kf_u = rec_clean["observed_los"][:12]

        for refiner in (
            ClassicalLMRefiner("cl_gn1_matched"),
            JFBRRefiner("vel_only"),
            JFBRRefiner("homog_6d"),
            LTRLMRefiner(writeback_mode="joint_6d"),
        ):
            out = refiner.refine_anchor(xk, tk, kf_s[-1, :3], kf_t, kf_s, kf_u, tau_rad=1.0e-3)
            torch.testing.assert_close(out.posterior_state_si, xk, atol=0.0, rtol=0.0)

        # Classical WindowLM error reduction & LTR-LM / JF-BR backward training step on noisy B0
        rec_noisy = _make_synthetic_two_block_record(pos_offset_m=(12000.0, -18000.0, 9000.0), vel_offset_mps=(-20.0, 25.0, -15.0))
        cl_filter = V1OnlineFilter(ClassicalLMRefiner("cl_window_lm", max_iters=6))
        pr_cl, po_cl, _ = cl_filter.forward_single_track(
            initial_state_si=rec_noisy["initial_state_si"],
            init_time_sec=0.0,
            times_sec=rec_noisy["times_sec"],
            observed_los=rec_noisy["observed_los"],
            station_states_si=rec_noisy["station_states_si"],
        )
        err_prior_end = torch.linalg.norm(pr_cl[0, :3] - rec_noisy["truth_states_si"][0, :3]).item()
        err_post_end = torch.linalg.norm(po_cl[-1, :3] - rec_noisy["truth_states_si"][-1, :3]).item()
        self.assertLess(err_post_end, err_prior_end)

        for learned in (
            V1OnlineFilter(LTRLMRefiner(writeback_mode="joint_6d")),
            V1OnlineFilter(JFBRRefiner(mode="homog_6d")),
        ):
            opt = torch.optim.Adam(learned.parameters(), lr=1.0e-3)
            loss = train_v1_refiner_step_on_record(learned, rec_noisy, loss_type="log_charbonnier")
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            grad_norms = [p.grad.norm().item() for p in learned.parameters() if p.grad is not None]
            self.assertTrue(any(g > 0.0 for g in grad_norms))
            opt.step()

            batch = collate_filter_sequences([rec_noisy])
            out_seq = learned.forward_filter_sequence(**batch)
            self.assertEqual(tuple(out_seq.posterior_states_si.shape), (1, 24, 6))


class TestV1FlagshipFDASF(unittest.TestCase):
    def test_tracklet_batch_additivity(self):
        rec = _make_synthetic_two_block_record()
        t = rec["times_sec"][:12]
        u = rec["observed_los"][:12]
        t_c = float(0.5 * (t[0] + t[-1]))
        h_span = float(0.5 * (t[-1] - t[0]))
        u_ref = u.mean(dim=0)

        H_all, B_all, n_all = ContinuousTrackletEncoder.compute_additive_block_stats(
            t, u, t_center_sec=t_c, half_span_sec=h_span, u_bar_ref=u_ref
        )
        H_1, B_1, n_1 = ContinuousTrackletEncoder.compute_additive_block_stats(
            t[:5], u[:5], t_center_sec=t_c, half_span_sec=h_span, u_bar_ref=u_ref
        )
        H_2, B_2, n_2 = ContinuousTrackletEncoder.compute_additive_block_stats(
            t[5:], u[5:], t_center_sec=t_c, half_span_sec=h_span, u_bar_ref=u_ref
        )
        self.assertEqual(n_all, n_1 + n_2)
        torch.testing.assert_close(H_all, H_1 + H_2, atol=1.0e-13, rtol=1.0e-13)
        torch.testing.assert_close(B_all, B_1 + B_2, atol=1.0e-13, rtol=1.0e-13)

    def test_attributable_chart_exact_jacobian_and_lambert_solver(self):
        # 1) Verify closed-form AttributableChart2Plus4 log_abs_det against autograd 6x6 Jacobian
        chart = AttributableChart2Plus4(
            t_c_sec=50.0,
            t_ref_sec=0.0,
            station_c_si=torch.tensor([6.878e6, 1.2e5, 0.0, -100.0, 7600.0, 500.0], dtype=torch.float64),
            u_bar_c=torch.tensor([0.6, 0.5, 0.6245], dtype=torch.float64),
            omega_bar_c=torch.tensor([4.0e-4, -3.0e-4], dtype=torch.float64),
        )
        z = torch.tensor([0.25, -0.15], dtype=torch.float64)
        eta = torch.tensor([0.02, -0.015, 0.1, -0.2], dtype=torch.float64)
        _, log_det_analytic = chart.decode(z, eta)

        def _fn(vec6: torch.Tensor) -> torch.Tensor:
            th, _ = chart.decode(vec6[:2], vec6[2:])
            return th

        J_auto = torch.autograd.functional.jacobian(_fn, torch.cat((z, eta), dim=0))
        _, log_det_auto = torch.linalg.slogdet(J_auto)
        self.assertAlmostEqual(log_det_analytic.item(), log_det_auto.item(), places=7)

        # 2) Verify universal-variable M=0 Lambert solver hits r2 when propagated
        x_true = torch.tensor([7.5e6, 0.0, 0.0, 0.0, 7200.0, 1200.0], dtype=torch.float64)
        dt_lam = 900.0  # >15 deg orbital arc
        x2_true, _ = propagate_si_with_stm(x_true, dt_lam, compute_stm=False)
        v1_sol, _, ok = solve_lambert_universal_m0(x_true[:3], x2_true[:3], dt_lam, retrograde=False)
        self.assertTrue(bool(ok.item()))
        x2_repro, _ = propagate_si_with_stm(torch.cat((x_true[:3], v1_sol), dim=0), dt_lam, compute_stm=False)
        self.assertLess(torch.linalg.norm(x2_repro[:3] - x2_true[:3]).item(), 5.0)

    def test_flow_density_disagreement_weights_and_fd_asf_rollout(self):
        flow = ConditionalRealNVP2D(context_dim=16, hidden_dim=32)
        ctx = torch.randn(16, dtype=torch.float64)
        z_s, b_s, log_q_sample = flow.sample_and_log_prob(ctx, 10)
        log_q_eval = flow.log_prob(z_s, b_s, ctx)
        torch.testing.assert_close(log_q_sample, log_q_eval, atol=1.0e-10, rtol=1.0e-10)

        # Verify 90/10 two-cluster forecast disagreement boosts minority cluster weight
        th_major = torch.tensor([7.8e6, 0.0, 0.0, 0.0, 7100.0, 0.0], dtype=torch.float64).expand(9, 6)
        th_minor = torch.tensor([9.2e6, 0.0, 0.0, 0.0, 6500.0, 1800.0], dtype=torch.float64).unsqueeze(0)
        thetas = torch.cat((th_major, th_minor), dim=0)
        log_pi = torch.zeros(10, dtype=torch.float64)  # uniform per particle => 90% major, 10% minor
        w_star, w_pi, _ = compute_forecast_disagreement_weights(
            thetas,
            log_pi,
            t_ref_sec=0.0,
            next_view_time_sec=600.0,
            next_station_pos_si=torch.tensor([6.878e6, 5.0e5, 0.0], dtype=torch.float64),
        )
        self.assertAlmostEqual(w_pi[-1].item(), 0.10, places=6)
        self.assertGreater(w_star[-1].item(), 0.25)  # minority branch actively boosted!

        rec = _make_synthetic_two_block_record()
        fd_model = FDASFSequentialFilter(context_dim=32, num_particles=6, top_k_polish=1, lm_polish_steps=1)
        opt = torch.optim.Adam(fd_model.parameters(), lr=1.0e-3)
        loss = train_fd_asf_proposal_step_on_record(fd_model, rec, num_ref_candidates=10)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        opt.step()

        out_seq = fd_model.forward_filter_sequence(**collate_filter_sequences([rec]))
        self.assertEqual(tuple(out_seq.posterior_states_si.shape), (1, 24, 6))
        self.assertTrue(torch.isfinite(out_seq.posterior_states_si).all())


class TestModelPlanV2ASDSC(unittest.TestCase):
    def test_qr_transport_and_restricted_writeback_orthogonality(self):
        # Verify E_h^T Delta_y == 0 to machine precision even when full-A singular axes rotate (commit 3ec2ba0)
        V_p_init = torch.zeros((6, 2), dtype=torch.float64)
        V_p_init[0, 0] = 1.0
        V_p_init[3, 1] = 1.0
        mem = PendingSubspaceMemory(V_p=V_p_init, r_diag_log=torch.zeros(2, dtype=torch.float64))

        x0 = torch.tensor([7.5e6, 2.0e5, 0.0, 0.0, 7200.0, 1000.0], dtype=torch.float64)
        _, phi = propagate_si_with_stm(x0, 300.0, compute_stm=True)
        assert phi is not None
        mem_trans = transport_pending_subspace_qr(mem, phi)
        torch.testing.assert_close(
            mem_trans.V_p.transpose(0, 1) @ mem_trans.V_p,
            torch.eye(2, dtype=torch.float64),
            atol=1.0e-13,
            rtol=1.0e-13,
        )

        # Construct A_+ whose singular axes mix V_n and V_p, with 1 pending axis unlocked and 1 still weak
        V_n = torch.linalg.qr(torch.randn((6, 4), dtype=torch.float64), mode="reduced")[0]
        # Orthogonalize V_p against V_n for test setup
        V_p_clean = mem_trans.V_p - V_n @ (V_n.transpose(0, 1) @ mem_trans.V_p)
        V_p_clean, _ = torch.linalg.qr(V_p_clean, mode="reduced")
        mem_test = PendingSubspaceMemory(V_p=V_p_clean, r_diag_log=torch.zeros(2, dtype=torch.float64))

        # First pending column has strong independent sensitivity (0.8), second has near-zero (0.01)
        A_plus = (
            torch.randn((12, 4), dtype=torch.float64) @ V_n.transpose(0, 1)
            + 0.8 * torch.randn((12, 1), dtype=torch.float64) @ V_p_clean[:, 0:1].transpose(0, 1)
            + 0.005 * torch.randn((12, 1), dtype=torch.float64) @ V_p_clean[:, 1:2].transpose(0, 1)
        )
        b_plus = torch.randn(12, dtype=torch.float64) * 8.0
        E_a, E_h, mem_after, _ = unlock_pending_directions_and_basis(
            A_plus, b_plus, mem_test, s_unlock=0.20
        )
        self.assertEqual(E_a.shape[1], 5)
        self.assertEqual(E_h.shape[1], 1)
        self.assertEqual(mem_after.dim_p, 1)

        model_v2 = ASDSCFilter()
        trace = model_v2.denoiser.forward_microstep(
            x_prior_si=x0,
            y_curr=torch.zeros(6, dtype=torch.float64),
            A_weighted=A_plus,
            b_weighted=b_plus,
            is_new_frame_2w=torch.ones(12, dtype=torch.bool),
            E_a=E_a,
            u_curr_los=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        )
        leakage = torch.linalg.norm(E_h.transpose(0, 1) @ trace.delta_y).item()
        self.assertLess(leakage, 1.0e-14)

    def test_progressive_step_increment_targets(self):
        # Commit ae8708d check: tau = s^2/(s^2 + gamma^2) = 0.5 => step 1 targets 0.5, step 2 targets 0.25, sum = 0.75
        V_phys = torch.eye(6, dtype=torch.float64)[:, :1]
        s_val = torch.tensor([0.5], dtype=torch.float64)  # with gamma=0.5, s^2/(s^2+gamma^2) = 0.25/0.5 = 0.5
        d0 = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        inc1 = compute_progressive_step_target(V_phys, s_val, d0, gamma=0.5)
        self.assertAlmostEqual(inc1[0].item(), 0.5, places=12)
        d1 = d0 - inc1
        inc2 = compute_progressive_step_target(V_phys, s_val, d1, gamma=0.5)
        self.assertAlmostEqual(inc2[0].item(), 0.25, places=12)
        self.assertAlmostEqual((inc1[0] + inc2[0]).item(), 0.75, places=12)

    def test_counterfactual_fake_range_preserves_initial_los_and_rate(self):
        x_true = torch.tensor([7.8e6, 2.5e5, 1.2e5, -200.0, 7100.0, 1400.0], dtype=torch.float64)
        x_stn = torch.tensor([6.878e6, 0.0, 0.0, 0.0, 7612.0, 1000.0], dtype=torch.float64)
        triplets = generate_counterfactual_prior_triplet(x_true, x_stn, fake_range_scale=1.4)
        x_fake = triplets["fake_range"]

        def _u_and_udot(x: torch.Tensor):
            dr = x[:3] - x_stn[:3]
            dv = x[3:] - x_stn[3:]
            rho = torch.linalg.norm(dr)
            u = dr / rho
            udot = (dv - (dv * u).sum() * u) / rho
            return u, udot

        u_t, udot_t = _u_and_udot(x_true)
        u_f, udot_f = _u_and_udot(x_fake)
        torch.testing.assert_close(u_t, u_f, atol=1.0e-14, rtol=1.0e-14)
        torch.testing.assert_close(udot_t, udot_f, atol=1.0e-14, rtol=1.0e-14)

    def test_v2_deadzone_hold_dagger_two_frame_unroll(self):
        # 1) Exact deadzone hold on noiseless good prior
        rec_clean = _make_synthetic_two_block_record(pos_offset_m=(0.0, 0.0, 0.0), vel_offset_mps=(0.0, 0.0, 0.0))
        v2 = ASDSCFilter(ablation_mode="full", num_microsteps=2)
        roll_clean = v2.rollout_track(
            initial_state_si=rec_clean["initial_state_si"],
            init_time_sec=0.0,
            times_sec=rec_clean["times_sec"],
            observed_los=rec_clean["observed_los"],
            station_states_si=rec_clean["station_states_si"],
        )
        torch.testing.assert_close(
            roll_clean["posterior_states_si"], rec_clean["truth_states_si"], atol=1.0e-7, rtol=1.0e-7
        )
        self.assertEqual(len(roll_clean["posterior_states_si"]), len(rec_clean["times_sec"]))

        # 2) DAgger rollout buffer + 2-checkpoint cross-gap connected unrolling + Adam step
        rec_noisy = _make_synthetic_two_block_record(
            pos_offset_m=(15000.0, -25000.0, 10000.0), vel_offset_mps=(20.0, -30.0, 15.0)
        )
        buf = DAggerRolloutBuffer()
        added = buf.collect_from_records(v2, {"rec_0": rec_noisy}, ["rec_0"], two_step_pairs=True)
        self.assertGreaterEqual(added, 4)
        cross_gap_samples = [s for s in buf.samples if s.cp_next is not None and s.cp_next.dt_sec > 100.0]
        self.assertGreaterEqual(len(cross_gap_samples), 1)

        v2.train()
        opt = torch.optim.Adam(v2.parameters(), lr=1.0e-3)
        loss, stats = train_two_checkpoint_pair(
            v2, cross_gap_samples[0], rec_noisy, loss_weights=V2LossWeights(lambda_prog=0.25)
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        grads = [p.grad.norm().item() for p in v2.parameters() if p.grad is not None]
        self.assertTrue(any(g > 0.0 for g in grads))
        opt.step()

        # 3) Evaluation suite, directional triplet, paired bootstrap CI & secant amplification
        eval_res = evaluate_filter_records(
            v2, [("rec_0", "src_A", "dense", rec_noisy), ("rec_1", "src_B", "pois10", rec_clean)]
        )
        self.assertEqual(eval_res["summary"]["overall"]["count"], 2)
        ci = cluster_paired_bootstrap_ci(eval_res["rows"], eval_res["rows"], n_boot=50)
        self.assertAlmostEqual(ci["point_estimate"], 0.0, places=12)

        dir_m = directional_benefit_metrics(
            torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
            torch.tensor([[0.5, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
        )
        self.assertAlmostEqual(dir_m["all_anchor_positive_benefit_rate"], 1.0, places=12)

        amp = measure_secant_amplification(
            lambda x_in: v2.rollout_track(
                initial_state_si=x_in,
                init_time_sec=0.0,
                times_sec=rec_noisy["times_sec"],
                observed_los=rec_noisy["observed_los"],
                station_states_si=rec_noisy["station_states_si"],
            )["posterior_states_si"][-1],
            rec_noisy["initial_state_si"],
            torch.tensor([1.0e-3, 0.0, 0.0, 0.0, 1.0e-3, 0.0], dtype=torch.float64),
        )
        self.assertTrue(math.isfinite(amp) and amp >= 0.0)


if __name__ == "__main__":
    unittest.main()

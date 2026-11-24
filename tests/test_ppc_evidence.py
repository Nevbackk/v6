"""Evidence and attribution checks for the per-frame posterior filter."""
import math
import unittest
from unittest.mock import patch

import torch

from v6.eval_metrics import RecordEvalRow, cluster_paired_bootstrap_ci
from v6.models.diffusion_orbit import integrate_vp_reverse_pf_ode, make_angular_model
from v6.models.posterior_warp import WarpedOrbitPosterior
from v6.models.ppc_flow import PPCOrbitFilter, SphericalLOS
from v6.models.reprojection import build_los_frame, propagate_reference_jet, propagate_si_with_stm
from v6.training_v1 import evaluate_filter_records
from tests.test_v1_v2_models import _make_synthetic_two_block_record


class PPCEvidenceTests(unittest.TestCase):
    def test_default_budget_strong_gaussian_conditioning(self):
        model = PPCOrbitFilter(nodes=32)
        with torch.no_grad():
            model.update_shape.network[-1].weight.fill_(0.01)
        x = model.nodes.new_zeros(6)
        prior = WarpedOrbitPosterior(x, torch.eye(6).to(x), x.new_zeros(model.warp.shape_size), model.warp)
        for h in (1.0, 10.0, 100.0, 400.0):
            H = x.new_tensor([[h, 0.0, 0.0, 0.0, 0.0, 0.0]])
            obs = x.new_tensor([0.7])

            def ell(points):
                return -0.5 * (points @ ((H / prior.scale).T) - obs).squeeze(-1).square()

            for steps in (1, 4):
                law = prior
                for _ in range(steps):
                    law = model.condition(
                        law, H, obs - H @ (law.center / prior.scale), ell, power=1 / steps
                    )
                mean, cov = law.moments(model.nodes)
                self.assertAlmostEqual(float(mean[0] / prior.scale[0]), 0.7 * h / (1 + h * h), places=10)
                self.assertAlmostEqual(
                    float(cov[0, 0].sqrt() / prior.scale[0]), 1 / math.sqrt(1 + h * h), places=10
                )
                self.assertLess(float(law.shape.norm()), 1.0e-8)

        def rhs(z, s):
            a = 100.0**2 * math.sin(s) * math.cos(s) / (1 + 100.0**2 * math.sin(s) ** 2)
            return a * z, z, z[:, 0].abs()

        old = integrate_vp_reverse_pf_ode(x.new_ones((1, 1)), rhs, num_steps=4)[0]
        self.assertAlmostEqual(float(old[0, 0]), 0.40507699288398785, places=12)

    def test_spherical_likelihood_integrates_and_detects_antipodes(self):
        n = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        _, axes = build_los_frame(n)
        scale = 0.4
        emission = SphericalLOS(
            n, axes, n.new_tensor([0.2, -0.1]), n.new_tensor([[1.0, 0.2], [0.2, 0.8]]), scale
        )
        station = n.new_zeros(6)
        # Equal-area spherical quadrature, unrelated to the update's integration nodes.
        i = torch.arange(14000, dtype=n.dtype) + 0.5
        z = 1 - 2 * i / len(i)
        phi = i * (math.pi * (3 - math.sqrt(5)))
        obs = torch.stack(((1 - z * z).sqrt() * phi.cos(), (1 - z * z).sqrt() * phi.sin(), z), dim=1)
        transformed, normalizer = emission.observation(obs)
        for ray in (n, -n):
            state = torch.cat((ray, n))
            target = emission.rays(state, station)
            lp = normalizer - 0.5 * ((transformed - target) / scale).square().sum(-1)
            self.assertAlmostEqual(float(4 * math.pi * lp.exp().mean()), 1.0, places=4)
        isotropic = SphericalLOS(n, axes, n.new_zeros(2), torch.eye(2).to(n), 0.001)
        backward, _ = isotropic.observation(-n)
        aligned = torch.cat((n, n))
        flipped = torch.cat((-n, n))
        self.assertAlmostEqual(
            float(
                isotropic.log_kernel(flipped, station, backward)
                - isotropic.log_kernel(aligned, station, backward)
            ),
            2.0e6,
            places=6,
        )
        model = PPCOrbitFilter(nodes=32)
        ref = model.start_track(aligned, 0.0).posterior
        H, b = isotropic.reference_rows(ref, station, backward)
        self.assertEqual(float(H.norm() + b.norm()), 0.0)

    def test_one_reference_second_variation_matches_discrete_rk4(self):
        x = torch.tensor([7.0e6, 2.0e5, 1.0e5, -250.0, 7250.0, 240.0], dtype=torch.float64)
        L = torch.diag(x.new_tensor([15000.0, 22000.0, 8000.0, 25.0, 15.0, 30.0]))
        direction = x.new_tensor([0.7, -0.8, 0.5, 0.2, -0.3, 0.4])
        dt = 137.0
        center, J, S = propagate_reference_jet(x, L, dt)
        correct, stm = propagate_si_with_stm(x, dt)
        torch.testing.assert_close(center, correct)
        torch.testing.assert_close(J, stm @ L, rtol=1.0e-12, atol=1.0e-10)
        exact = propagate_si_with_stm(x + L @ direction, dt, compute_stm=False)[0]
        linear = center + J @ direction
        second = 0.5 * torch.einsum("ijk,j,k->i", S, direction, direction)
        self.assertLess(float((exact - linear - second).norm()), 0.02 * float((exact - linear).norm()))
        model = PPCOrbitFilter(steps=1, nodes=32)
        state = model.start_track(x, 0.0)
        with patch("v6.models.ppc_flow.propagate_reference_jet", wraps=propagate_reference_jet) as solver:
            model.predict(state.posterior, dt)
        self.assertEqual(solver.call_count, 1)
        self.assertEqual(solver.call_args.args[0].shape, (6,))

    def test_calibration_centers_and_direct_moment_control(self):
        rec = _make_synthetic_two_block_record()
        x = rec["initial_state_si"]
        calibration = {"mean_los": x.new_tensor([0.2, 0.03, 0, 0, 0, 0]), "cov_los": torch.eye(6).to(x)}
        for mode in ("none", "train_mean"):
            model = PPCOrbitFilter(steps=1, nodes=16, initial_bias=mode)
            model.reference_prior = calibration
            initial = model.start_track(x, 0.0).posterior
            self.assertEqual(torch.equal(initial.center, x), mode == "none")
        net = make_angular_model("direct")
        context = torch.randn(7, 14, dtype=x.dtype)
        net.loss(torch.randn(7, 2, dtype=x.dtype), context).backward()
        self.assertGreater(float(net.head[-1].weight.grad.norm()), 0.0)
        self.assertGreater(float(torch.linalg.eigvalsh(net.moments(context)[1]).min()), 0.0)

    def test_numerical_failure_stays_in_denominator_and_quantiles_are_paired(self):
        rec = _make_synthetic_two_block_record()
        model = PPCOrbitFilter(steps=1, nodes=16)
        bad = dict(rec)
        bad["observed_los"] = rec["observed_los"].clone()
        bad["observed_los"][2] = torch.nan
        summary = evaluate_filter_records(model, [("ok", "a", "dense", rec), ("bad", "b", "dense", bad)])[
            "summary"
        ]
        self.assertEqual(summary["overall"]["count"], 2)
        self.assertEqual(summary["overall"]["failed_records"], 1)
        self.assertEqual(summary["overall"]["nonfinite_rate"], 0.5)
        self.assertEqual(summary["failures"][0]["frame"], 2)
        self.assertEqual(summary["overall"]["position_km"]["finite_count"], 1)

        def rows(v):
            return [
                RecordEvalRow(str(i), str(i // 2), "dense", 1.0, p, 1.0, 1.0, 1.0, 1.0, True, False)
                for i, p in enumerate(v)
            ]

        left, right = rows([1, 2, 100, 101]), rows([1, 95, 96, 99])
        for name, q in (("p50", 0.50), ("p95", 0.95)):
            ci = cluster_paired_bootstrap_ci(left, right, statistic=name, mode="ratio", n_boot=64)
            data = lambda rows: torch.tensor([r.pos_rmse_km for r in rows], dtype=torch.float64)
            expected = float(torch.quantile(data(left), q) / torch.quantile(data(right), q))
            self.assertAlmostEqual(ci["point_estimate"], expected)

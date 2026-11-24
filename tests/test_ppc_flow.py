"""Gaussian references, normalized nonlinear transport and causal frame integration."""
import math
import unittest

import torch

from tests.test_v1_v2_models import _make_synthetic_two_block_record
from v6.models.ppc_flow import PPCOrbitFilter, qr_reference
from v6.models.posterior_warp import ConditionalOrbitWarp, WarpedOrbitPosterior, standard_normal_log_prob


class PPCFlowTests(unittest.TestCase):
    def setUp(self):
        self.rng = torch.Generator().manual_seed(27)
        self.dtype = torch.float64

    def test_six_dimensional_inverse_volume_and_normalization(self):
        warp = ConditionalOrbitWarp()
        with torch.no_grad():
            for net in warp.conditioners:
                net[-1].weight.normal_(std=0.1, generator=self.rng)
        raw = torch.randn(warp.shape_size, dtype=self.dtype, generator=self.rng) * 0.6
        root = torch.eye(6, dtype=self.dtype)
        root[1, 0], root[3, 1] = 0.4, -0.2
        law = WarpedOrbitPosterior(root.new_tensor([7.0e6, 0.0, 0.0, 0.0, 7200.0, 0.0]), root, raw, warp)
        z = torch.randn(30, 6, dtype=self.dtype, generator=self.rng) * 2
        z[0, 0] = 8.0
        z[1, 1] = -8.0
        samples, ld = law.transform(z)
        inverse, back = law.inverse(samples)
        torch.testing.assert_close(inverse, z, atol=2.0e-10, rtol=2.0e-10)
        torch.testing.assert_close(back, -ld, atol=2.0e-10, rtol=2.0e-10)
        eps = 1.0e-5
        shift = eps * torch.eye(6, dtype=z.dtype)
        jac = (law.transform(z[3] + shift)[0] - law.transform(z[3] - shift)[0]).T / (2 * eps)
        self.assertLess(abs(float(ld[3] - torch.linalg.slogdet(jac).logabsdet)), 2.0e-7)
        torch.testing.assert_close(
            law.log_prob(samples), standard_normal_log_prob(z) - ld, atol=2.0e-10, rtol=2.0e-10
        )
        # Directly integrate a nontrivial scalar deformation embedded in all six coordinates.
        scalar = ConditionalOrbitWarp()
        shape = torch.zeros_like(raw).reshape(2, 3, -1)
        shape[0, 0, 2 : 2 + scalar.bins] = torch.linspace(-0.8, 0.8, scalar.bins)
        line = torch.linspace(-10.0, 10.0, 16001, dtype=z.dtype)
        grid = z.new_zeros((len(line), 6))
        grid[:, 0] = line
        marginal = (scalar.log_prob(grid, shape.flatten()) + 2.5 * math.log(2 * math.pi)).exp()
        self.assertAlmostEqual(float(torch.trapezoid(marginal, line)), 1.0, places=5)

    def test_tempered_linear_gaussian_consumes_one_likelihood(self):
        model = PPCOrbitFilter(nodes=64)
        with torch.no_grad():
            model.update_shape.network[-1].weight.normal_(std=0.03, generator=self.rng)
        P = torch.diag(torch.tensor([2.0, 3.0, 1.1, 0.75, 1.4, 0.9], dtype=self.dtype))
        P[0, 3] = P[3, 0] = 0.45
        H = P.new_tensor([[1.0, 0.7, 0, 0, 0, 0], [0, 1.0, 0.3, 0, 0, 0]]) / 0.5
        y = P.new_tensor([0.4, -0.3]) / 0.5
        initial = WarpedOrbitPosterior(
            P.new_zeros(6), torch.linalg.cholesky(P), P.new_zeros(model.warp.shape_size), model.warp
        )
        law = initial

        def likelihood(x):
            return -0.5 * (y - (x / initial.scale) @ H.T).square().sum(-1)

        for n in range(1, 5):
            b = y - H @ (law.center / initial.scale)
            law = model.condition(law, H, b, likelihood, power=0.25)
            exact = torch.linalg.inv(torch.linalg.inv(P) + (n / 4) * H.T @ H)
            mean, cov = law.moments(model.nodes)
            torch.testing.assert_close(
                mean / initial.scale, (n / 4) * exact @ H.T @ y, atol=2e-13, rtol=2e-13
            )
            torch.testing.assert_close(
                cov / initial.scale[:, None] / initial.scale[None, :], exact, atol=2e-13, rtol=2e-13
            )
            self.assertLess(float(law.shape.norm()), 1.0e-12)
        once = qr_reference(initial, H, y, 1.0)
        torch.testing.assert_close(law.root @ law.root.T, once.root @ once.root.T, atol=1e-13, rtol=1e-13)

    def test_read_only_prediction_mask_and_shape_feedback(self):
        torch.manual_seed(20)
        model = PPCOrbitFilter(steps=2, sensor_samples=2, nodes=32)
        with torch.no_grad():
            model.update_shape.network[-1].weight.normal_(std=0.01, generator=self.rng)
        rec = _make_synthetic_two_block_record()
        ix = torch.tensor([0, 1, 2, 12, 13, 14])
        t, u, st = (rec[key][ix] for key in ("times_sec", "observed_los", "station_states_si"))
        mask = torch.tensor([True, False, True, True, True, True])
        live = model.start_track(rec["initial_state_si"], 0.0, seed=4)
        kwargs = dict(
            initial_state_si=rec["initial_state_si"],
            init_time_sec=0.0,
            times_sec=t,
            observed_los=u,
            station_states_si=st,
            valid_mask=mask,
            seed=4,
        )
        base = model.rollout_track(**kwargs)
        with torch.no_grad():
            original = live.posterior
            for k in mask.nonzero().flatten().tolist():
                random_state = live.generator.get_state().clone()
                model.readout(model.predict(live.posterior, float(t[k]) - live.epoch))
                torch.testing.assert_close(random_state, live.generator.get_state())
                result = model.step(live, float(t[k]), u[k], st[k])
                torch.testing.assert_close(
                    result["posterior_state_si"], base["posterior_states_si"][k], atol=0, rtol=0
                )
            self.assertEqual(live.count, 5)
            self.assertFalse(torch.equal(original.shape, live.posterior.shape))
        altered = u.clone()
        altered[-1] = torch.nn.functional.normalize(altered[-1] + u.new_tensor([0, 0.001, 0]), dim=0)
        changed = model.rollout_track(**{**kwargs, "observed_los": altered})
        torch.testing.assert_close(
            base["posterior_states_si"][:-1], changed["posterior_states_si"][:-1], atol=0, rtol=0
        )
        self.assertGreater(
            float((changed["posterior_states_si"][-1] - base["posterior_states_si"][-1]).norm()), 0.0
        )
        score1 = model.rollout_track(**kwargs, score_states_si=rec["truth_states_si"][ix])
        score2 = model.rollout_track(**kwargs, score_states_si=rec["truth_states_si"][ix] + 1000.0)
        torch.testing.assert_close(
            score1["posterior_states_si"], score2["posterior_states_si"], atol=0, rtol=0
        )
        self.assertFalse(torch.equal(score1["posterior_nll_si"], score2["posterior_nll_si"]))
        batch = model.forward_filter_sequence(**kwargs)
        torch.testing.assert_close(batch.posterior_states_si, base["posterior_states_si"], atol=0, rtol=0)

    def test_shape_learns_non_gaussian_density_and_cfm_has_gradients(self):
        torch.manual_seed(31)
        model = PPCOrbitFilter(nodes=128)
        zero = model.nodes.new_zeros(6)
        initial = WarpedOrbitPosterior(
            zero, torch.eye(6, dtype=zero.dtype), zero.new_zeros(model.warp.shape_size), model.warp
        )
        H = zero.new_zeros((1, 6))
        b = zero.new_zeros(1)

        def log_likelihood(x):
            r = x[..., 0] / initial.scale[0]
            components = torch.stack(((r - 1.25) / 0.2, (r + 1.25) / 0.2), dim=-1)
            return torch.logsumexp(-0.5 * components.square(), dim=-1) + 0.5 * r.square()

        law, teacher = model._condition(initial, H, b, log_likelihood, want_teacher=True)
        teacher.flow_matching_loss(model.warp, 64).backward()
        self.assertGreater(float(model.update_shape.network[-1].weight.grad.norm()), 0.0)
        optimizer = torch.optim.Adam([*model.warp.parameters(), *model.update_shape.parameters()], lr=0.002)

        def targets(n):
            t = torch.randn((n, 6), dtype=zero.dtype, generator=self.rng)
            t[:, 0] = 0.2 * t[:, 0] + 1.25 * torch.where(torch.arange(n) % 2 == 0, 1.0, -1.0)
            return t * initial.scale

        for _ in range(100):
            optimizer.zero_grad(set_to_none=True)
            law, teacher = model._condition(initial, H, b, log_likelihood, want_teacher=True)
            loss = -law.log_prob(targets(96)).mean() / 6 + teacher.flow_matching_loss(model.warp, 16)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            law = model.condition(initial, H, b, log_likelihood)
            unseen = targets(1024)
            self.assertGreater(float((law.log_prob(unseen) - initial.log_prob(unseen)).mean()), 0.3)
            samples = law.sample(8192, generator=self.rng)[:, 0] / initial.scale[0]
            self.assertLess(float((samples.abs() < 0.5).double().mean()), 0.16)
            self.assertTrue(0.25 < float((samples < 0).double().mean()) < 0.75)

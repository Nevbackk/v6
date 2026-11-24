"""Compact, per-observation posterior transport with a Gaussian QR reference."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..data.observation_clock import FrameNumericalError, observation_frames
from .batched_physics import DTYPE, state_scale
from .diffusion_orbit import (
    make_angular_model,
    FrameOrbitFilter,
    build_diagonal_los_prior_covariance,
    safe_cholesky_spd,
)
from .posterior_warp import ConditionalOrbitWarp, WarpedOrbitPosterior, standard_normal_log_prob
from .reprojection import (
    ComputeLedger,
    RunningLOSNoise,
    build_los_frame,
    propagate_si_with_stm,
    propagate_reference_jet,
)


def qr_reference(prior, H, innovation, power):
    """Update a scaled Gaussian reference by one fraction of a whitened likelihood."""
    if not 0 < power <= 1:
        raise ValueError("likelihood power must be in (0, 1]")
    B = H @ prior.root
    eye = torch.eye(6, device=B.device, dtype=B.dtype)
    q, upper = torch.linalg.qr(torch.cat((eye, math.sqrt(power) * B)), mode="reduced")
    signs = torch.where(upper.diag() < 0, -torch.ones_like(upper.diag()), torch.ones_like(upper.diag()))
    upper = signs[:, None] * upper
    rhs = signs * (q[6:].T @ (math.sqrt(power) * innovation))
    shift = torch.linalg.solve_triangular(upper, rhs[:, None], upper=True)[:, 0]
    root = torch.linalg.solve_triangular(upper.T, prior.root.T, upper=False).T
    return WarpedOrbitPosterior(prior.from_chart(shift), root, prior.shape, prior.warp)


class IdentityOrbitWarp(nn.Module):
    """Gaussian reference for exactly the same frame and observation adapter."""

    shape_size = 0

    def map(self, value, shape, *, inverse=False):
        return value, value.new_zeros(value.shape[:-1])

    def log_prob(self, value, shape):
        return standard_normal_log_prob(value)


class SphericalLOS:
    """A normalized spherical emission with the sensor's bias and local covariance.

    A fixed rotation and a projective linear map send the measured sphere to a
    von Mises--Fisher model. Their Jacobian is independent of the candidate state.
    """

    def __init__(self, nominal, axes, bias, covariance, sigma):
        self.sigma = sigma
        root = safe_cholesky_spd(covariance)
        self.A = axes @ torch.linalg.solve_triangular(root, axes.T, upper=False) + torch.outer(
            nominal, nominal
        )
        shifted = F.normalize(nominal + axes @ (sigma * bias), dim=0)
        a, b, c = torch.linalg.cross(nominal, shifted).unbind()
        zero = a.new_zeros(())
        skew = torch.stack((zero, -c, b, c, zero, -a, -b, a, zero)).reshape(3, 3)
        identity = torch.eye(3).to(nominal)
        self.rotation = identity + skew + skew @ skew / (1 + nominal @ shifted)

    def observation(self, obs):
        mapped = F.normalize(obs, dim=-1) @ self.rotation @ self.A.T
        norm = mapped.norm(dim=-1)
        kappa = 1.0 / self.sigma**2
        normalizer = math.log(kappa / (2 * math.pi)) - math.log(-math.expm1(-2 * kappa))
        return (
            mapped / norm[..., None],
            normalizer + torch.linalg.slogdet(self.A).logabsdet - 3 * norm.log(),
        )

    def rays(self, states, station):
        raw = F.normalize(states[..., :3] - station[:3], dim=-1)
        return F.normalize(raw @ self.A.T, dim=-1)

    def log_kernel(self, states, station, measured):
        return -0.5 * ((self.rays(states, station) - measured) / self.sigma).square().sum(-1)

    def reference_rows(self, law, station, measured):
        rel = law.center[:3] - station[:3]
        rho = rel.norm().clamp_min(1.0)
        ray = rel / rho
        mapped = self.A @ ray
        unit = F.normalize(mapped, dim=0)
        _, axes = build_los_frame(unit)
        Hpos = (axes.T @ self.A) @ (torch.eye(3).to(ray) - torch.outer(ray, ray))
        Hpos = Hpos * law.scale[:3] / (rho * mapped.norm() * self.sigma)
        H = torch.cat((Hpos, torch.zeros_like(Hpos)), dim=-1)
        residual = axes.T @ (measured - unit) / self.sigma
        # Local chord curvature is non-positive on the opposite hemisphere.
        trust = (unit @ measured).clamp_min(0.0).sqrt()
        return trust * H, trust * residual

    def predictive(self, states, station, uniform, measured, obs, normalizer):
        directions = self.rays(states, station)
        log_kernel = -0.5 * ((directions - measured) / self.sigma).square().sum(-1)
        nll = math.log(len(directions)) - torch.logsumexp(log_kernel, 0) - normalizer
        # Invert the 3-D vMF polar CDF and the spherical whitening transform.
        kappa = 1.0 / self.sigma**2
        d = -torch.log1p(uniform[:, 0] * math.expm1(-2 * kappa)) / kappa
        angle = 2 * math.pi * uniform[:, 1]
        _, axes = build_los_frame(directions)
        azimuth = axes[..., 0] * angle.cos()[:, None] + axes[..., 1] * angle.sin()[:, None]
        sample = directions * (1 - d[:, None]) + (d * (2 - d)).clamp_min(0).sqrt()[:, None] * azimuth
        physical = F.normalize(torch.linalg.solve(self.A, sample.T).T, dim=-1) @ self.rotation.T
        mean = F.normalize(physical.mean(0), dim=0)
        theta = torch.atan2(
            torch.linalg.cross(mean.expand_as(physical), physical).norm(dim=-1), physical @ mean
        )
        radius = torch.quantile(theta, 0.9)
        error = torch.atan2(torch.linalg.cross(mean, obs).norm(), mean @ obs)
        return mean, nll, radius, error


class ResidualShapeUpdate(nn.Module):
    """Amortized change of spline coefficients from a difference of distribution statistics."""

    def __init__(self, shape_size, bins=8, hidden=96, dtype=DTYPE):
        super().__init__()
        points = (torch.arange(bins, dtype=dtype) + 0.5) / bins
        self.register_buffer("thresholds", torch.special.ndtri(points))
        self.register_buffer("feature_pairs", torch.tril_indices(6, 6))
        self.embedding = nn.Sequential(
            nn.Linear(6, 32, dtype=dtype), nn.SiLU(), nn.Linear(32, 32, dtype=dtype)
        )
        features = 6 + 21 + 12 * bins + 32
        self.network = nn.Sequential(
            nn.Linear(features + shape_size + 14, hidden, dtype=dtype),
            nn.SiLU(),
            nn.Linear(hidden, hidden, dtype=dtype),
            nn.SiLU(),
            nn.Linear(hidden, shape_size, dtype=dtype),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def statistics(self, y):
        x = torch.asinh(y)
        i, j = self.feature_pairs
        # Smooth marginal CDF probes distinguish two peaks from a Gaussian with the same moments.
        cdf = torch.sigmoid((self.thresholds - y.unsqueeze(-1)) * len(self.thresholds)).flatten(-2)
        return torch.cat((x, x[..., i] * x[..., j], cdf, self.embedding(x)), dim=-1)

    def marginal_quantiles(self, y, weights):
        values, order = y.T.contiguous().sort(dim=-1, stable=True)
        mass = weights[order]
        positions = (mass.cumsum(-1) - 0.5 * mass).contiguous()
        p = torch.special.ndtr(self.thresholds).expand(6, -1).contiguous()
        hi = torch.searchsorted(positions, p).clamp(1, len(weights) - 1)
        lo = hi - 1
        p0, p1 = positions.gather(1, lo), positions.gather(1, hi)
        frac = ((p - p0) / (p1 - p0).clamp_min(torch.finfo(y.dtype).eps)).clamp(0, 1)
        return torch.lerp(values.gather(1, lo), values.gather(1, hi), frac)

    def forward(self, old, source, target, source_weights, target_weights, geometry):
        left = self.statistics(source)
        right = left if target is source else self.statistics(target)
        change = (target_weights - source_weights) @ right + source_weights @ (right - left)
        quantile_change = self.marginal_quantiles(target, target_weights) - self.marginal_quantiles(
            source, source_weights
        )
        change = torch.cat((change, torch.asinh(quantile_change).flatten()))
        fixed = torch.cat((torch.asinh(old), torch.asinh(geometry)))
        # Matched measures produce an exact identity residual for any trained network.
        inputs = torch.stack((torch.cat((change, fixed)), torch.cat((torch.zeros_like(change), fixed))))
        step, origin = self.network(inputs).unbind(0)
        return old + (step - origin)


@dataclass
class TransportTeacher:
    source: Tensor
    target: Tensor
    source_weights: Tensor
    target_weights: Tensor
    old_shape: Tensor
    new_shape: Tensor
    ordering: Tensor

    def flow_matching_loss(self, warp, samples):
        # A shared uniform quantile couples the two finite quadrature measures.
        # Sorting selects a training coupling; it never splits the deployed density.
        with torch.no_grad():
            p = (
                torch.arange(samples, device=self.source.device, dtype=self.source.dtype)
                + torch.rand(samples, device=self.source.device, dtype=self.source.dtype)
            ) / samples

            def quantiles(x, weights):
                order = (x @ self.ordering).argsort(stable=True)
                cdf = weights[order].cumsum(0)
                cdf[-1] = 1.0
                ix = torch.searchsorted(cdf, p, right=True).clamp_max(len(cdf) - 1)
                return x[order[ix]]

            start = quantiles(self.source, self.source_weights)
            end = quantiles(self.target, self.target_weights)
            tau = torch.rand((samples, 1), device=p.device, dtype=p.dtype)
            at = torch.lerp(start, end, tau)
        velocity = warp.velocity(at, tau, self.old_shape, self.new_shape)
        return (velocity - (end - start)).square().mean()


@dataclass
class PPCTrack:
    epoch: float
    posterior: WarpedOrbitPosterior
    previous_innovation: Tensor
    noise: RunningLOSNoise
    generator: torch.Generator
    ledger: ComputeLedger
    count: int = 0


class PPCOrbitFilter(FrameOrbitFilter):
    """One invertible orbit density. Temporary integration nodes are never carried as tracks."""

    def __init__(
        self,
        *,
        steps=4,
        sensor_samples=8,
        nodes=128,
        estimator="ppc",
        observation_model="diffusion",
        initial_bias="none",
        dtype=DTYPE,
    ):
        super().__init__()
        if steps < 1 or sensor_samples < 1 or nodes < 8:
            raise ValueError(
                "Positive steps/sensor samples and at least 8 six-dimensional nodes are required"
            )
        self.steps, self.sensor_samples = steps, sensor_samples
        if estimator not in ("ppc", "qr", "gaussian") or initial_bias not in ("none", "train_mean"):
            raise ValueError("Unknown density or initialization control")
        self.estimator, self.initial_bias = estimator, initial_bias
        self.sensor = make_angular_model(observation_model, dtype)
        self.warp = ConditionalOrbitWarp(dtype=dtype) if estimator == "ppc" else IdentityOrbitWarp()
        self.update_shape = (
            ResidualShapeUpdate(self.warp.shape_size, self.warp.bins, dtype=dtype)
            if estimator == "ppc"
            else None
        )
        # Fixed normal cubature, standardized once so Gaussian readouts reproduce I.
        uniform = torch.quasirandom.SobolEngine(6, scramble=True, seed=17).draw(nodes, dtype=torch.float64)
        z = torch.special.ndtri(
            uniform.clamp(torch.finfo(torch.float64).eps, 1 - torch.finfo(torch.float64).eps)
        )
        z = z - z.mean(0)
        z = torch.linalg.solve_triangular(torch.linalg.cholesky(z.T @ z / nodes), z.T, upper=False).T
        self.register_buffer("nodes", z.contiguous().to(dtype))
        self.register_buffer(
            "sensor_uniform",
            torch.rand(nodes, 2, dtype=dtype, generator=torch.Generator().manual_seed(3941)),
        )
        self.reference_prior = None

    def sequence_metadata(self):
        return {
            "model_family": self.estimator,
            "likelihood_steps": self.steps,
            "integration_nodes": len(self.nodes),
        }

    def start_track(
        self,
        initial_state_si,
        init_time_sec,
        *,
        initial_ref_cov_scaled=None,
        initial_station_si=None,
        seed=20261028,
    ):
        from ..reference_prior import los_state_basis

        x = torch.as_tensor(initial_state_si).to(self.nodes)
        t0 = float(init_time_sec)
        if x.shape != (6,) or not math.isfinite(t0):
            raise ValueError("Expected a six-dimensional initial state and finite epoch")
        station = (
            x.new_zeros(6) if initial_station_si is None else torch.as_tensor(initial_station_si).to(x)
        )
        basis = los_state_basis(x, station[:3])
        scale = state_scale(device=x.device, dtype=x.dtype)
        if self.reference_prior is not None and self.initial_bias == "train_mean":
            x = x + scale * (basis @ self.reference_prior["mean_los"].to(x))
        if initial_ref_cov_scaled is not None:
            covariance = basis.T @ torch.as_tensor(initial_ref_cov_scaled).to(x) @ basis
        elif self.reference_prior is not None:
            covariance = self.reference_prior["cov_los"].to(x)
            if self.initial_bias == "none":
                mean = self.reference_prior["mean_los"].to(x)
                covariance = covariance + torch.outer(mean, mean)
        else:
            covariance = basis.T @ build_diagonal_los_prior_covariance(x, station[:3]) @ basis
        # The first base coordinate points along the initial LOS, and the full factor
        # may rotate under the STM. Its orientation is part of the learned chart.
        root = basis @ safe_cholesky_spd(covariance)
        posterior = WarpedOrbitPosterior(x, root, x.new_zeros(self.warp.shape_size), self.warp)
        return PPCTrack(
            t0,
            posterior,
            x.new_zeros(3),
            RunningLOSNoise(),
            torch.Generator(device=x.device).manual_seed(seed),
            ComputeLedger(),
        )

    def _project(self, reference, source, target, base_weights, weights, geometry, ordering, want_teacher):
        if self.estimator == "gaussian":
            # Same integration budget and likelihood, projected back onto one Gaussian.
            ms, mt = base_weights @ source, weights @ target
            ys, yt = source - ms, target - mt
            Cs = ys.T @ (base_weights[:, None] * ys)
            Ct = yt.T @ (weights[:, None] * yt)
            ls, lt = safe_cholesky_spd(Cs), safe_cholesky_spd(Ct)
            affine = torch.linalg.solve_triangular(ls.T, lt.T, upper=True).T
            return (
                WarpedOrbitPosterior(
                    reference.from_chart(mt - affine @ ms),
                    reference.root @ affine,
                    reference.shape,
                    self.warp,
                ),
                None,
            )
        # The control-variate displacement is zero for matched measures.
        shift = (weights - base_weights) @ target + base_weights @ (target - source)
        reference = WarpedOrbitPosterior(
            reference.from_chart(shift), reference.root, reference.shape, self.warp
        )
        target = target - shift
        shape = self.update_shape(reference.shape, source, target, base_weights, weights, geometry)
        law = WarpedOrbitPosterior(reference.center, reference.root, shape, self.warp)
        teacher = None
        if want_teacher:
            teacher = TransportTeacher(
                source.detach(),
                target.detach(),
                base_weights.detach(),
                weights.detach(),
                reference.shape.detach(),
                shape,
                ordering.detach(),
            )
        return law, teacher

    def predict(self, posterior, dt):
        return self._predict(posterior, dt)[0]

    def _predict(self, posterior, dt, *, ledger=None, want_teacher=False):
        """Push the warped measure through RK4 and project its nonlinear departure from the STM chart."""
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("Observation times must be finite and nondecreasing")
        if dt == 0:
            return posterior, None
        with torch.no_grad():
            scale = posterior.scale
            if self.estimator == "qr":
                center, phi = propagate_si_with_stm(posterior.center, dt, compute_stm=True, ledger=ledger)
                root = (phi * scale[None, :] / scale[:, None]) @ posterior.root
                return WarpedOrbitPosterior(center, root, posterior.shape, self.warp), None
            center, J, S = propagate_reference_jet(
                posterior.center, scale[:, None] * posterior.root, dt, ledger=ledger
            )
            root = J / scale[:, None]
            reference = WarpedOrbitPosterior(center, root, posterior.shape, self.warp)
            source, _ = self.warp.map(self.nodes, posterior.shape)
            # Nodes integrate a local chart polynomial. Only one reference orbit was propagated.
            displacement = 0.5 * torch.einsum("ijk,nj,nk->ni", S, source, source)
            target = source + torch.linalg.solve(J, displacement.T).T
            uniform = source.new_full((len(source),), 1.0 / len(source))
            geometry = source.new_zeros(14)
            ordering = F.normalize(root[:3].T @ F.normalize(center[:3], dim=0), dim=0)
        return self._project(reference, source, target, uniform, uniform, geometry, ordering, want_teacher)

    def condition(self, prior, H, innovation, log_likelihood, *, power=1.0):
        return self._condition(prior, H, innovation, log_likelihood, power=power)[0]

    def _condition(
        self,
        prior,
        H,
        innovation,
        log_likelihood,
        *,
        power=1.0,
        ordering=None,
        want_teacher=False,
        position_only=False,
    ):
        """One likelihood power. H and innovation are whitened and use D-scaled state increments."""
        with torch.no_grad():
            reference = qr_reference(prior, H, innovation, power)
            if self.estimator == "qr":
                return reference, None
            z = self.nodes
            base, log_jac = self.warp.map(z, prior.shape)
            # One fixed latent rule evaluated in two affine charts for balanced integration.
            # Transform between charts algebraically. No support node is propagated as an orbit.
            transfer = torch.linalg.solve(prior.root, reference.root)
            offset = prior.standardize(reference.center)
            prior_in_ref = torch.linalg.solve(transfer, (base - offset).T).T
            ref_in_prior = offset + base @ transfer.T
            previous_y = torch.cat((base, ref_in_prior))
            chart_y = torch.cat((prior_in_ref, base))
            forward_log = standard_normal_log_prob(z) - log_jac
            cross_prior, cross_ref = self.warp.log_prob(
                torch.cat((ref_in_prior, prior_in_ref)), prior.shape
            ).chunk(2)
            prior_chart_log = torch.cat((forward_log, cross_prior))
            ref_chart_log = torch.cat((cross_ref, forward_log))
            difference = prior_chart_log - ref_chart_log + torch.linalg.slogdet(transfer).logabsdet
            base_log_weights = -torch.logaddexp(torch.zeros_like(difference), difference)
            base_weights = base_log_weights.softmax(0)
            linear_residual = innovation - previous_y @ (H @ prior.root).T
            points = (
                reference.center[:3] + reference.scale[:3] * (chart_y @ reference.root[:3].T)
                if position_only
                else reference.from_chart(chart_y)
            )
            remaining = (
                prior_chart_log
                - standard_normal_log_prob(previous_y)
                - (ref_chart_log - standard_normal_log_prob(chart_y))
                + power * (log_likelihood(points) + 0.5 * linear_residual.square().sum(-1))
            )
            weights = (base_log_weights + remaining).softmax(0)
            if not torch.isfinite(weights).all():
                raise FloatingPointError("No finite posterior mass on this frame's integration nodes")
            design, residual = math.sqrt(power) * (H @ prior.root), math.sqrt(power) * innovation
            geometry = torch.cat(
                (
                    design.T @ residual,
                    design.square().sum(0),
                    residual.norm().reshape(1),
                    design.norm().reshape(1),
                )
            )
            if ordering is None:
                ordering = chart_y.new_tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        return self._project(
            reference, chart_y, chart_y, base_weights, weights, geometry, ordering, want_teacher
        )

    def readout(self, posterior, *, mass=0.9):
        """Return deterministic moment and marginal equal-tail interval estimates; preserve track state."""
        if not 0 < mass < 1:
            raise ValueError("Interval mass must be between zero and one")
        values, _ = posterior.transform(self.nodes)
        mean = values.mean(0)
        offsets = values - mean
        quantiles = torch.quantile(values, values.new_tensor([(1 - mass) / 2, (1 + mass) / 2]), dim=0)
        return dict(mean=mean, covariance=offsets.T @ offsets / len(values), interval=quantiles)

    def _advance(self, state, now, raw, station, station_id, reliable, *, truth=None):
        dt = now - state.epoch
        learning = truth is not None
        # Sample one residual stage per frame for CFM; all stages still execute.
        selected = int(torch.randint(self.steps + 1, ()).item()) if learning else -1
        prior, teacher = self._predict(
            state.posterior, dt, ledger=state.ledger, want_teacher=(selected == 0)
        )
        with torch.no_grad():
            states, _ = prior.transform(self.nodes)
            prior_mean = states.mean(0)
            nominal = F.normalize(prior_mean[:3] - station[:3], dim=0)
            _, sensor_axes = build_los_frame(nominal)
            sigma = state.noise.sigma
            ctx = self.sensor.context(nominal, station, state.previous_innovation, dt, sigma)
            bias, R = self.sensor.moments(ctx, self.steps, self.sensor_samples, state.generator)
            emission = SphericalLOS(nominal, sensor_axes, bias, R, sigma)
            obs = F.normalize(raw, dim=0)
            measured, normalizer = emission.observation(obs)
            predicted_los, obs_nll, cone, error = emission.predictive(
                states, station, self.sensor_uniform, measured, obs, normalizer
            )
        law = prior
        for stage in range(1, self.steps + 1):
            with torch.no_grad():
                H, b = emission.reference_rows(law, station, measured)
                ray = F.normalize(law.center[:3] - station[:3], dim=0)
                direction = F.normalize(law.root[:3].T @ ray, dim=0)
            law, auxiliary = self._condition(
                law,
                H,
                b,
                lambda x: emission.log_kernel(x, station, measured),
                power=1 / self.steps,
                ordering=direction,
                want_teacher=(selected == stage),
                position_only=True,
            )
            if auxiliary is not None:
                teacher = auxiliary
        with torch.no_grad():
            mean, covariance = law.moments(self.nodes)
            if not bool(torch.isfinite(mean).all() & torch.isfinite(covariance).all()):
                raise FloatingPointError("Non-finite warped orbit moments")
            state.previous_innovation = obs - F.normalize(mean[:3] - station[:3], dim=0)
            state.noise.observe(
                now, raw.detach().cpu().numpy(), station.detach().cpu().numpy(), station_id, reliable
            )
            state.epoch, state.posterior = now, law if learning else law.detached()
            state.count += 1
        result = dict(
            prior_state_si=prior_mean,
            posterior_state_si=mean,
            covariance_si=covariance,
            predicted_los=predicted_los,
            angular_covariance=sigma**2 * R,
            observation_nll=obs_nll,
            distribution=law,
            los_cone90_rad=cone,
            los_error_rad=error,
        )
        if learning:
            true_ray = F.normalize(truth[:3] - station[:3], dim=0)
            angular = sensor_axes.T @ (obs - true_ray) / sigma
            # SI units add a constant to the density loss; normalize by state dimension.
            density_loss = -law.log_prob(truth) / 6
            cfm = (
                teacher.flow_matching_loss(self.warp, self.sensor_samples)
                if teacher is not None
                else density_loss.new_zeros(())
            )
            result["loss"] = density_loss + cfm + self.sensor.loss(angular[None], ctx[None]) / 2
        return result

    @torch.no_grad()
    def step(
        self, state, time_sec, observed_los, station_state_si, *, station_id=None, station_id_reliable=None
    ):
        raw = torch.as_tensor(observed_los).to(self.nodes)
        station = torch.as_tensor(station_state_si).to(self.nodes)
        return self._advance(state, float(time_sec), raw, station, station_id, station_id_reliable)

    def _frames(self, record, *, training=False, unroll=1):
        state = self.start_track(
            record["initial_state_si"],
            record["init_time_sec"],
            initial_station_si=record.get("initial_station_si"),
            initial_ref_cov_scaled=record.get("initial_ref_cov_scaled"),
            seed=record.get("seed", 20261028),
        )
        times = record["times_sec"]
        obs = torch.as_tensor(record["observed_los"]).to(self.nodes)
        stations = torch.as_tensor(record["station_states_si"]).to(self.nodes)
        ids, reliable = record.get("station_ids"), record.get("station_id_reliable")
        truth = torch.as_tensor(record["truth_states_si"]).to(self.nodes) if training else None
        for frame in observation_frames(times, record.get("valid_mask"), state.epoch):
            k = frame.index
            try:
                out = self._advance(
                    state,
                    frame.time_sec,
                    obs[k],
                    stations[k],
                    None if ids is None else int(ids[k]),
                    None if reliable is None else bool(reliable[k]),
                    truth=None if truth is None else truth[k],
                )
            except (FloatingPointError, torch.linalg.LinAlgError, OverflowError) as error:
                raise FrameNumericalError(frame, error) from error
            if training and state.count % unroll == 0:
                state.posterior = state.posterior.detached()
            yield k, out

    def training_loss_chunks(self, record, unroll=2):
        """Sum each truncated chunk; shape credit passes between consecutive valid arrivals."""
        if unroll < 1:
            raise ValueError("Unroll length must be positive")
        terms = []
        for _, out in self._frames(record, training=True, unroll=unroll):
            terms.append(out["loss"])
            if len(terms) == unroll:
                yield torch.stack(terms).sum()
                terms = []
        if terms:
            yield torch.stack(terms).sum()

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
        initial_station_si=None,
        initial_ref_cov_scaled=None,
        station_ids=None,
        station_id_reliable=None,
        seed=20261028,
        score_states_si=None,
    ):
        record = dict(
            initial_state_si=initial_state_si,
            init_time_sec=init_time_sec,
            times_sec=times_sec,
            observed_los=observed_los,
            station_states_si=station_states_si,
            valid_mask=valid_mask,
            initial_station_si=initial_station_si,
            initial_ref_cov_scaled=initial_ref_cov_scaled,
            station_ids=station_ids,
            station_id_reliable=station_id_reliable,
            seed=seed,
        )
        n = len(times_sec)
        priors, posts = self.nodes.new_zeros((n, 6)), self.nodes.new_zeros((n, 6))
        rays, nll = self.nodes.new_zeros((n, 3)), self.nodes.new_zeros(n)
        density_nll = self.nodes.new_zeros(n) if score_states_si is not None else None
        affine_nll = self.nodes.new_zeros(n) if score_states_si is not None else None
        hpd = self.nodes.new_zeros(n) if score_states_si is not None else None
        cone, angle = self.nodes.new_zeros(n), self.nodes.new_zeros(n)
        count = 0
        for k, out in self._frames(record):
            priors[k], posts[k] = out["prior_state_si"], out["posterior_state_si"]
            nll[k], rays[k] = out["observation_nll"], out["predicted_los"]
            cone[k], angle[k] = out["los_cone90_rad"], out["los_error_rad"]
            if density_nll is not None:
                # Evaluation only: predictions and the next track state are already committed.
                value = torch.as_tensor(score_states_si[k]).to(self.nodes)
                posterior = out["distribution"]
                density_nll[k] = -posterior.log_prob(value)
                affine_nll[k] = posterior.log_affine_volume() - standard_normal_log_prob(
                    posterior.standardize(value)
                )
                _, ld = posterior.transform(self.nodes)
                hpd[k] = -density_nll[k] >= torch.quantile(standard_normal_log_prob(self.nodes) - ld, 0.10)
            count += 1
        return dict(
            prior_states_si=priors,
            posterior_states_si=posts,
            predicted_los=rays,
            observation_nll=nll,
            posterior_nll_si=density_nll,
            reference_nll_si=affine_nll,
            posterior_hpd90=hpd,
            los_cone90_rad=cone,
            los_error_rad=angle,
            processed_frames=count,
        )

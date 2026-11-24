"""Chart-conditioned proposals for a causal prefix of optical observations."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..data.observation_clock import observation_frames

from .batched_physics import MU_EARTH_M3_S2
from .diffusion_orbit import HALF_PI, RouteBGuidedCosineFlow2D, RouteBTwistedDiffusionSMC
from .reprojection import (
    build_los_frame,
    estimate_online_los_noise_floor,
    los_noise_profile,
    propagate_si_with_stm,
)
from .v1_fd_asf import (
    MIN_PERIGEE_RADIUS_M,
    SIN_15_DEG,
    AttributableChart2Plus4,
    ContinuousTrackletEncoder,
    LambertChart2Plus4,
    check_bound_perigee_feasible,
    compute_forecast_disagreement_weights,
)


@dataclass
class ChartPrefix:
    chart: AttributableChart2Plus4 | LambertChart2Plus4
    context: Tensor
    indices: Tensor
    eta_std: Tensor
    sigma: float
    center_index: int
    first_index: int

    @property
    def lambert(self):
        return isinstance(self.chart, LambertChart2Plus4)

    def decode(self, z, eta, branch, *, volume):
        # The chart is selected for this prefix. A failed Lambert branch has no mass;
        # silently decoding the same second coordinate as a range rate changes its law.
        if self.lambert and volume:
            states, logdet, valid = self.chart.decode(z, eta, branch)
        elif self.lambert:
            pairs = [self.chart._map_single(torch.cat((zi, ei)), bool(bi.item()))
                     for zi, ei, bi in zip(z, eta, branch)]
            states, valid = (torch.stack(items) for items in zip(*pairs))
            logdet = z.new_zeros(len(z))
        else:
            states, logdet = self.chart.decode(z, eta)
            valid = branch == 0
        range_coords = z if self.lambert else z[:, :1]
        valid = valid & (range_coords.abs() < 6).all(-1) & (eta[:, :2].norm(dim=-1) < math.pi)
        if self.lambert:
            valid = valid & (eta[:, 2:].norm(dim=-1) < math.pi)
        return states, logdet, valid & torch.isfinite(states).all(-1) & torch.isfinite(logdet)


def build_chart_prefix(
    encoder, flow, record, last_index, *, device, los_budget=None, context=None, sigma=None
):
    times = torch.as_tensor(record["times_sec"], dtype=torch.float64, device="cpu")
    n = last_index + 1
    mask = torch.as_tensor(record.get("valid_mask", torch.ones_like(times, dtype=torch.bool)),
                           dtype=torch.bool, device="cpu")
    idx = torch.nonzero(mask[:n], as_tuple=True)[0]
    if not len(idx):
        raise ValueError("No valid observation in the causal prefix")
    k, first = idx[-1].item(), idx[0].item()
    obs = torch.as_tensor(record["observed_los"], dtype=torch.float64, device=device)
    stn = torch.as_tensor(record["station_states_si"], dtype=torch.float64, device=device)
    station_ids = record.get("station_ids")
    reliable = record.get("station_id_reliable")
    sig = sigma
    if sig is None:
        _, sig, _ = estimate_online_los_noise_floor(
            times, obs.detach().cpu(), station_ids, reliable, mask, up_to_idx=k, station_states_si=stn
        )
    t_ref = float(record["init_time_sec"])
    ctx = context
    if ctx is None:
        ctx, _ = encoder.encode_prefix(times, obs, stn, mask, up_to_idx=k, init_time_sec=t_ref)
    tk = times[k].item()
    x0 = torch.as_tensor(record["initial_state_si"], dtype=torch.float64, device=device)
    xk, _ = propagate_si_with_stm(x0, tk - t_ref, compute_stm=False)
    rho = float((xk[:3] - stn[k, :3]).norm().clamp_min(1))
    u = torch.nn.functional.normalize(obs[k], dim=0)
    _, axes = build_los_frame(u)
    if len(idx) > 1:
        prev = idx[-2].item()
        dt = tk - times[prev].item()
    else:
        prev, dt = k, 0.0
    if station_ids is not None and (reliable is None or (bool(reliable[prev]) and bool(reliable[k]))):
        same_station = int(station_ids[prev]) == int(station_ids[k])
    else:
        expected = .5 * (stn[prev, 3:] + stn[k, 3:]) * dt
        same_station = bool((stn[k, :3] - stn[prev, :3] - expected).norm() < 100.)
    chart_args = dict(t_c_sec=tk, t_ref_sec=t_ref, station_c_si=stn[k], u_bar_c=u, rho_ref_m=rho)
    if 0 < dt <= 60 and same_station:
        rate = axes.T @ (u - torch.nn.functional.normalize(obs[prev], dim=0)) / dt
    else:
        rate = axes.T @ (xk[3:] - stn[k, 3:]) / rho
    attr = AttributableChart2Plus4(omega_bar_c=rate, **chart_args)
    chart = attr
    scale_rate = math.sqrt(2) * sig / (dt * attr.omega_ref_rad_s) if 0 < dt <= 60 and same_station else 1.
    eta_std = obs.new_tensor([sig, sig, scale_rate, scale_rate])
    # The nominal orbit and station history select the chart, independently of proposed samples.
    min_period = 2 * math.pi * math.sqrt(MIN_PERIGEE_RADIUS_M**3 / MU_EARTH_M3_S2)
    if 1.0 < tk - float(times[first]) < min_period:
        xa, _ = propagate_si_with_stm(x0, times[first].item() - t_ref, compute_stm=False)
        r1, r2 = xa[:3], xk[:3]
        sine = torch.linalg.cross(r1, r2).norm() / (r1.norm() * r2.norm()).clamp_min(1.)
        if sine >= SIN_15_DEG:
            chart = LambertChart2Plus4(t_a_sec=times[first].item(), t_b_sec=tk, t_ref_sec=t_ref,
                                       station_a_si=stn[first], station_b_si=stn[k],
                                       u_bar_a=obs[first], u_bar_b=u, fallback_chart=attr, rho_ref_m=rho)
            eta_std = obs.new_full((4,), sig)
    if los_budget is not None and len(idx) > los_budget:
        if los_budget < 1:
            raise ValueError("los_budget must be positive")
        idx = idx[torch.linspace(0, len(idx) - 1, los_budget).round().long()] if los_budget > 1 else idx[-1:]
    if flow is not None:
        ctx = ctx + flow.chart_embedding.weight[int(isinstance(chart, LambertChart2Plus4))]
    return ChartPrefix(chart, ctx, idx, eta_std, sig, k, first)


def prefix_nll(states, prefix, record, *, all_observations=False, return_states=False):
    """Stable spherical chord likelihood, using one chronological sweep per orbit."""
    device = states.device
    times = torch.as_tensor(record["times_sec"], dtype=states.dtype, device="cpu")
    obs = torch.as_tensor(record["observed_los"], dtype=states.dtype, device=device)
    stn = torch.as_tensor(record["station_states_si"], dtype=states.dtype, device=device)
    t = float(record["init_time_sec"])
    cost = states.new_zeros(len(states))
    indices = prefix.indices.tolist()
    if all_observations:
        mask = torch.as_tensor(record.get("valid_mask", torch.ones(len(times), dtype=torch.bool))).cpu()
        indices = torch.nonzero(mask[: prefix.center_index + 1], as_tuple=True)[0].tolist()
    sigma_history = record.get("los_sigma_by_frame")
    for k in indices:
        tk = times[k].item()
        states, _ = propagate_si_with_stm(states, tk-t, compute_stm=False)
        t = tk
        los = torch.nn.functional.normalize(states[:, :3] - stn[k, :3], dim=-1)
        sigma = prefix.sigma if sigma_history is None else sigma_history[k]
        cost = cost + (los - torch.nn.functional.normalize(obs[k], dim=0)).square().sum(-1) / (
            2 * sigma**2
        )
    return (cost, states) if return_states else cost


def chart_log_target(z, eta, branch, prefix, record, *, at_frame=False):
    states, ld, valid = prefix.decode(z, eta, branch, volume=True)
    valid = valid & check_bound_perigee_feasible(states)
    log_target = z.new_full((len(z),), -torch.inf)
    if bool(valid.any()):
        cost, terminal = prefix_nll(
            states[valid], prefix, record, all_observations=True, return_states=True
        )
        log_target[valid] = ld[valid] - cost
        if at_frame:
            states = states.clone()
            states[valid] = terminal
    return states, log_target


def normal_log_density(x, std):
    return -0.5 * ((x / std).square() + 2 * torch.log(std) + math.log(2*math.pi)).sum(-1)


class EmptyTeacherSupport(ValueError):
    """The reference batch contains no usable observation prefix or feasible orbit."""


def draw_chart_teacher(encoder, flow, record, *, num_ref_candidates, device, los_budget=None):
    """Return a reference batch with importance-corrected posterior and forecast weights."""
    dev = torch.device(device)
    dtype = torch.float64
    t_init = float(record["init_time_sec"])
    times = torch.as_tensor(record["times_sec"], dtype=dtype)
    stn = torch.as_tensor(record["station_states_si"], device=dev, dtype=dtype)
    truth = torch.as_tensor(record["truth_states_si"], device=dev, dtype=dtype)
    valid = torch.as_tensor(
        record.get("valid_mask", torch.ones(times.shape[0], dtype=torch.bool)), dtype=torch.bool
    )
    if num_ref_candidates < 1:
        raise ValueError("num_ref_candidates must be positive")
    frames = list(observation_frames(times, valid, t_init))
    if not frames:
        raise EmptyTeacherSupport("No valid observation frame")
    k = random.choice([f.index for f in frames])
    prefix = build_chart_prefix(encoder, flow, record, k, device=dev, los_budget=los_budget)
    sigma_history, _ = los_noise_profile(
        times[: k + 1],
        record["observed_los"][: k + 1],
        None if record.get("station_ids") is None else record["station_ids"][: k + 1],
        None if record.get("station_id_reliable") is None else record["station_id_reliable"][: k + 1],
        valid[: k + 1],
        station_states_si=stn[: k + 1],
    )
    record = {**record, "los_sigma_by_frame": sigma_history}
    ic, ia = prefix.center_index, prefix.first_index
    chart = prefix.chart
    rel = truth[ic, :3] - stn[ic, :3]
    rho = rel.norm().clamp_min(1.)
    if prefix.lambert:
        rho_first = (truth[ia, :3] - stn[ia, :3]).norm().clamp_min(1.)
        z_star = torch.stack((rho_first.log(), rho.log())) - math.log(chart.rho_ref_m)
    else:
        rhodot = ((truth[ic, 3:] - stn[ic, 3:]) * rel / rho).sum()
        z_star = torch.stack(((rho / chart.rho_ref_m).log(), rhodot / chart.v_ref_mps))

    # Independent scale/branch draws give the product density below for every M,
    # including odd M. Truth only centers the training proposal, never the filter.
    M = int(num_ref_candidates)
    broad = torch.rand(M, device=dev) < .5
    widths = torch.where(broad, z_star.new_tensor(.85), z_star.new_tensor(.15))
    z = z_star + widths[:, None] * torch.randn((M, 2), device=dev, dtype=dtype)
    nbranches = 2 if prefix.lambert else 1
    branch = torch.randint(nbranches, (M,), device=dev)
    eta = torch.randn((M, 4), device=dev, dtype=dtype) * prefix.eta_std
    dz = z - z_star
    log_qz = torch.logaddexp(normal_log_density(dz, z.new_tensor(.15)),
                             normal_log_density(dz, z.new_tensor(.85))) - math.log(2.)
    log_ref = log_qz + normal_log_density(eta, prefix.eta_std) - math.log(nbranches)
    with torch.no_grad():
        states, log_target = chart_log_target(z, eta, branch, prefix, record)
        supported = torch.isfinite(log_target)
        if not bool(supported.any()):
            raise EmptyTeacherSupport("No physically admissible teacher particle in this prefix")
        log_weight = log_target[supported] - log_ref[supported]
        schedule = record.get("scheduled_forecasts", {}).get(k)
        later = torch.nonzero(valid[k + 1 :], as_tuple=True)[0]
        if schedule is not None or len(later):
            if schedule is None:
                # Training label only. The proposal and its context saw only the prefix through k.
                nxt = k + 1 + int(later[0])
                t_next, station = float(times[nxt]), stn[nxt]
            else:
                t_next, station = schedule
            w_star, w_pi, _ = compute_forecast_disagreement_weights(
                states[supported],
                log_weight,
                t_ref_sec=t_init,
                next_view_time_sec=t_next,
                next_station_pos_si=torch.as_tensor(station).to(stn)[:3],
            )
        else:
            w_pi = w_star = torch.softmax(log_weight, dim=0)
    return z[supported], branch[supported], eta[supported], w_star, w_pi, prefix.context


class RouteBOrbitFilter(nn.Module):
    """B1: chart flow with endpoint IS. B2: twisted finite Gaussian reverse chain.

    The physical target is proportional to the prefix likelihood on the feasible chart,
    with its ECI volume element. The learned velocity guides proposals; its density
    is never substituted for the physical target.
    """
    def __init__(self, route="b1", *, num_steps=4, num_particles=8, los_budget=None,
                 use_guidance=True, context_dim=48):
        super().__init__()
        if route not in ("b1", "b2") or min(num_steps, num_particles) < 1:
            raise ValueError("Choose b1 or b2 and positive K and P")
        if los_budget is not None and los_budget < 1:
            raise ValueError("los_budget must be positive")
        self.route, self.num_steps, self.num_particles = route, num_steps, num_particles
        self.los_budget, self.use_guidance = los_budget, use_guidance
        self.encoder = ContinuousTrackletEncoder(context_dim=context_dim)
        self.flow = RouteBGuidedCosineFlow2D(context_dim=context_dim)

    def _guide(self, z, t, branch, eta, pre, record):
        # A bounded surrogate potential guides proposals only. Endpoint weights use the
        # original likelihood, with no surrogate or guidance multiplier in the target.
        with torch.enable_grad():
            zr = z.detach().clone().requires_grad_(True)
            v = self.flow.velocity(zr, t, branch, eta, pre.context)
            clean = math.sin(HALF_PI*t)*zr + math.cos(HALF_PI*t)*v/HALF_PI
            states, _, valid = pre.decode(clean.reshape(-1, 2), eta.reshape(-1, 4), branch.reshape(-1), volume=False)
            if not bool(valid.any()):
                return torch.zeros_like(z)
            potential = -torch.log1p(prefix_nll(states[valid], pre, record)).sum()
            grad = torch.autograd.grad(potential, zr)[0].detach()
        return torch.where(torch.isfinite(grad), grad, torch.zeros_like(grad))

    def sample_prefix(self, pre, record, generator):
        P, K = self.num_particles, self.num_steps
        dev, dtype = pre.context.device, torch.float64
        branches = torch.randint(2 if pre.lambert else 1, (P,), generator=generator, device=dev)
        if self.route == "b1":
            eta = torch.randn((P, 4), device=dev, dtype=dtype, generator=generator) * pre.eta_std
            base = torch.randn((P, 2), device=dev, dtype=dtype, generator=generator)
            def guide(z, t):
                return self._guide(z, t, branches, eta, pre, record)
            z, volume = self.flow.transport(base, branches, eta, pre.context, num_steps=K,
                                            guidance_fn=guide if self.use_guidance else None)
            states, log_target = chart_log_target(z, eta, branches, pre, record, at_frame=True)
            log_q = normal_log_density(base, base.new_ones(2))-volume+normal_log_density(eta, pre.eta_std)
            if pre.lambert:
                log_q -= math.log(2.)
            log_w = log_target - log_q
            valid = torch.isfinite(log_w)
            if not bool(valid.any()):
                return None
            return states[valid], torch.log_softmax(log_w[valid], 0)

        # The Gaussian OU chain preserves N(0,I_6). Terminal twisting divides that
        # known endpoint density out and includes the linear eta-scale Jacobian.
        x = torch.randn((P, 6), generator=generator, dtype=dtype, device=dev)
        rho, noise = math.cos(HALF_PI/K), math.sin(HALF_PI/K)
        def base(x, j):
            return rho*x, torch.full_like(x, noise)
        def guided(x, j):
            mu, std = base(x, j)
            if self.use_guidance:
                t = 1. - (j-.5)/K
                v = self.flow.velocity(x[:,:2], t, branches, x[:,2:]*pre.eta_std, pre.context)
                # An explicit, normalized Gaussian proposal; the base/proposal ratio
                # below corrects its drift even when the learned direction is poor.
                mu = torch.cat((mu[:,:2] + v/K, mu[:,2:]), dim=-1)
            return mu, std
        terminal_states = None
        def twist(x, j):
            nonlocal terminal_states
            eta = x[:,2:]*pre.eta_std
            if j == 0:
                terminal_states, log_target = chart_log_target(
                    x[:, :2], eta, branches, pre, record, at_frame=True
                )
                return (log_target + pre.eta_std.log().sum() - normal_log_density(x, x.new_ones(6))
                        + (math.log(2.) if pre.lambert else 0.))
            states, _, valid = pre.decode(x[:,:2], eta, branches, volume=False)
            # Invalid intermediate points retain potential 1 so a later move can enter support.
            out = x.new_zeros(len(x))
            if bool(valid.any()):
                cost = prefix_nll(states[valid], pre, record)
                out[valid] = -(1-j/K)*torch.log1p(cost)
            return out
        def resample(idx):
            nonlocal branches
            branches = branches[idx]
        x, w, _ = RouteBTwistedDiffusionSMC.run_chain(
            x_init=x, num_steps=K, base_transition_params=base, guided_proposal_params=guided,
            log_twist_potential=twist, on_resample=resample, generator=generator)
        live = w > 0
        if not bool(live.any()):
            return None
        return terminal_states[live], w[live].log()

    @torch.inference_mode(False)
    @torch.no_grad()
    def rollout_track(self, *, initial_state_si, init_time_sec, times_sec, observed_los,
                      station_states_si, valid_mask=None, seed=20261028, **metadata):
        dev = next(self.parameters()).device
        prior = torch.as_tensor(initial_state_si, dtype=torch.float64, device=dev)
        t = torch.as_tensor(times_sec, dtype=torch.float64, device="cpu")
        obs = torch.as_tensor(observed_los).to(prior)
        stations = torch.as_tensor(station_states_si).to(prior)
        mask = torch.ones_like(t, dtype=torch.bool) if valid_mask is None else torch.as_tensor(valid_mask, device="cpu").bool()
        record = dict(
            initial_state_si=prior,
            init_time_sec=init_time_sec,
            times_sec=t,
            valid_mask=mask,
            observed_los=obs,
            station_states_si=stations,
            **metadata,
        )
        sigmas, _ = los_noise_profile(
            t,
            obs,
            metadata.get("station_ids"),
            metadata.get("station_id_reliable"),
            mask,
            station_states_si=stations,
        )
        gen = torch.Generator(device=dev).manual_seed(int(seed))
        record["los_sigma_by_frame"] = sigmas
        hidden = prior.new_zeros(self.encoder.context_dim)
        particles = log_weights = None
        priors = prior.new_zeros((len(t), 6))
        posts = torch.zeros_like(priors)
        attempted = failures = 0
        for frame in observation_frames(t, mask, init_time_sec):
            k = frame.index
            if particles is not None:
                particles, _ = propagate_si_with_stm(particles, frame.dt_sec, compute_stm=False)
                prior = (log_weights.exp()[:, None] * particles).sum(0)
            else:
                prior, _ = propagate_si_with_stm(prior, frame.dt_sec, compute_stm=False)
            priors[k] = prior
            hidden = self.encoder.advance(hidden, obs[k], stations[k], frame.dt_sec)
            # Existing particles condition on this LOS exactly once. A new proposal
            # is importance-corrected against all received rows when support is depleted.
            if particles is not None:
                rays = torch.nn.functional.normalize(particles[:, :3] - stations[k, :3], dim=-1)
                residual = rays - torch.nn.functional.normalize(obs[k], dim=0)
                log_weights = torch.log_softmax(
                    log_weights - residual.square().sum(-1) / (2 * sigmas[k] ** 2), dim=0
                )
            if (
                particles is None
                or len(particles) == 1
                or 1 / log_weights.exp().square().sum() < self.num_particles / 2
            ):
                pre = build_chart_prefix(
                    self.encoder,
                    self.flow,
                    record,
                    k,
                    device=dev,
                    los_budget=self.los_budget,
                    context=hidden,
                    sigma=sigmas[k],
                )
                bank = self.sample_prefix(pre, record, gen)
                attempted += 1
                if bank is None:
                    failures += 1
                else:
                    particles, log_weights = bank
            if particles is not None:
                prior = (log_weights.exp()[:, None] * particles).sum(0)
            posts[k] = prior
        return dict(prior_states_si=priors, posterior_states_si=posts,
                    failed_proposals=failures, attempted_batches=attempted)

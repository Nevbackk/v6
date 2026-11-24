"""Chart-conditioned proposals for a causal prefix of optical observations."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..data.observation_clock import causal_diffusion_events

from .diffusion_orbit import HALF_PI, RouteBGuidedCosineFlow2D, RouteBTwistedDiffusionSMC
from .reprojection import build_los_frame, estimate_online_los_noise_floor, propagate_si_with_stm
from .v1_fd_asf import (
    SIN_15_DEG, AttributableChart2Plus4, ContinuousTrackletEncoder, LambertChart2Plus4,
    check_bound_perigee_feasible, compute_forecast_disagreement_weights,
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


def build_chart_prefix(encoder, flow, record, last_index, *, device, los_budget=None):
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
    _, sig, _ = estimate_online_los_noise_floor(times, obs.detach().cpu(), station_ids,
                                                 reliable, mask, up_to_idx=k, station_states_si=stn)
    ctx, _ = encoder.encode_prefix(times, obs, stn, mask, up_to_idx=k, sigma_u_rad=sig)
    t_ref = float(record["init_time_sec"])
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
    if bool((torch.diff(times[idx]) > 60).any()):
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


def prefix_nll(states, prefix, record):
    """Stable spherical chord likelihood, using one chronological sweep per orbit."""
    device = states.device
    times = torch.as_tensor(record["times_sec"], dtype=states.dtype, device="cpu")
    obs = torch.as_tensor(record["observed_los"], dtype=states.dtype, device=device)
    stn = torch.as_tensor(record["station_states_si"], dtype=states.dtype, device=device)
    t = float(record["init_time_sec"])
    cost = states.new_zeros(len(states))
    for k in prefix.indices.tolist():
        tk = times[k].item()
        states, _ = propagate_si_with_stm(states, tk-t, compute_stm=False)
        t = tk
        los = torch.nn.functional.normalize(states[:, :3] - stn[k, :3], dim=-1)
        cost = cost + (los - torch.nn.functional.normalize(obs[k], dim=0)).square().sum(-1)
    return cost / (2 * prefix.sigma**2)


def chart_log_target(z, eta, branch, prefix, record):
    states, ld, valid = prefix.decode(z, eta, branch, volume=True)
    valid = valid & check_bound_perigee_feasible(states)
    log_target = z.new_full((len(z),), -torch.inf)
    if bool(valid.any()):
        log_target[valid] = ld[valid] - prefix_nll(states[valid], prefix, record)
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
    events = causal_diffusion_events(times, valid, t_init, record.get("block_start_time_sec"),
                                     readout_until_sec=record.get("readout_until_sec"))
    available = list(dict.fromkeys(k for _, k, _ in events if k >= 0))
    if not available:
        raise EmptyTeacherSupport("No prefix ready for publication")
    k = random.choice(available)
    prefix = build_chart_prefix(encoder, flow, record, k, device=dev, los_budget=los_budget)
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
        # Use an announced viewing geometry if supplied by the operator. The fallback
        # forecasts the present platform at a fixed horizon without reading future rows.
        schedule = record.get("scheduled_forecasts", {}).get(k)
        if schedule is None:
            t_next = float(times[k]) + 300.
            next_station, _ = propagate_si_with_stm(stn[k], 300., compute_stm=False)
        else:
            t_next, station = schedule
            next_station = torch.as_tensor(station, device=dev, dtype=dtype)
        w_star, w_pi, _ = compute_forecast_disagreement_weights(
            states[supported], log_weight, t_ref_sec=t_init, next_view_time_sec=t_next,
            next_station_pos_si=next_station[:3])
    return z[supported], branch[supported], eta[supported], w_star, w_pi, prefix.context


def render_reference_orbits(prior, t_ref, times, mask, anchors):
    """Propagate the latest published reference orbit to each frame, preserving pre-update values."""
    ref, ptr = prior, 0
    priors, posts = [], []
    for k, tk in enumerate(times.tolist()):
        if not mask[k]:
            priors.append(torch.zeros_like(prior))
            posts.append(torch.zeros_like(prior))
            continue
        while ptr < len(anchors) and anchors[ptr][0] < tk:
            ref = anchors[ptr][1]
            ptr += 1
        pred, _ = propagate_si_with_stm(ref, tk-t_ref, compute_stm=False)
        priors.append(pred)
        if ptr < len(anchors) and anchors[ptr][0] == tk:
            ref = anchors[ptr][1]
            ptr += 1
            pred, _ = propagate_si_with_stm(ref, tk-t_ref, compute_stm=False)
        posts.append(pred)
    return torch.stack(priors), torch.stack(posts)


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
            states, log_target = chart_log_target(z, eta, branches, pre, record)
            log_q = normal_log_density(base, base.new_ones(2))-volume+normal_log_density(eta, pre.eta_std)
            if pre.lambert:
                log_q -= math.log(2.)
            log_w = log_target - log_q
            valid = torch.isfinite(log_w)
            if not bool(valid.any()):
                return None
            return (torch.softmax(log_w[valid], 0)[:,None] * states[valid]).sum(0)

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
        def twist(x, j):
            eta = x[:,2:]*pre.eta_std
            if j == 0:
                _, log_target = chart_log_target(x[:,:2], eta, branches, pre, record)
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
        states, _, _ = pre.decode(x[live,:2], x[live,2:]*pre.eta_std, branches[live], volume=False)
        return (w[live,None]*states).sum(0)

    @torch.inference_mode(False)
    @torch.no_grad()
    def rollout_track(self, *, initial_state_si, init_time_sec, times_sec, observed_los,
                      station_states_si, valid_mask=None, seed=20261028, **metadata):
        from ..data.observation_clock import causal_diffusion_events
        dev = next(self.parameters()).device
        prior = torch.as_tensor(initial_state_si, dtype=torch.float64, device=dev)
        record = dict(initial_state_si=prior, init_time_sec=init_time_sec, times_sec=times_sec,
                      observed_los=observed_los, station_states_si=station_states_si, **metadata)
        t = torch.as_tensor(times_sec, dtype=torch.float64, device="cpu")
        mask = torch.ones_like(t, dtype=torch.bool) if valid_mask is None else torch.as_tensor(valid_mask, device="cpu").bool()
        record["valid_mask"] = mask
        events = causal_diffusion_events(t, mask, init_time_sec, metadata.get("block_start_time_sec"),
                                        explicit_flush=metadata.get("explicit_flush", False),
                                        readout_until_sec=metadata.get("readout_until_sec"))
        gen = torch.Generator(device=dev).manual_seed(int(seed))
        anchors = []
        last, ref = -1, prior
        ep_state = ep_time = None
        failures = attempted = 0
        for te, idx, endpoint in events:
            if idx > last:
                attempted += 1
                pre = build_chart_prefix(self.encoder, self.flow, record, idx, device=dev, los_budget=self.los_budget)
                candidate = self.sample_prefix(pre, record, gen)
                if candidate is not None:
                    ref = candidate
                else:
                    failures += 1
                last = idx
            anchors.append((te, ref))
            if endpoint:
                ep_state, _ = propagate_si_with_stm(ref, te-init_time_sec, compute_stm=False)
                ep_time = te
        priors, posts = render_reference_orbits(prior, init_time_sec, t, mask, anchors)
        return dict(prior_states_si=priors, posterior_states_si=posts,
                    block2_t30_state_si=ep_state, block2_t30_time_sec=ep_time,
                    failed_proposals=failures, attempted_batches=attempted)

"""Fit and evaluate the production six-dimensional warp on a two-body range slice (CPU)."""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v6.models.batched_physics import MU_EARTH_M3_S2, state_scale
from v6.models.ppc_flow import PPCOrbitFilter, qr_reference
from v6.models.posterior_warp import WarpedOrbitPosterior, monotone_spline
from v6.models.reprojection import propagate_si_with_stm


class RangeSlice:
    """One ambiguous range coordinate and five independent standard normal coordinates."""

    def __init__(self):
        self.q = torch.linspace(0.35, 1.60, 12001, dtype=torch.float64)
        self.mid = 0.8485621433429708
        self.width = (1.60 - 0.35) / math.sqrt(12)
        self.scale = state_scale(device=self.q.device, dtype=self.q.dtype)
        R = 7.0e6
        observer = self.q.new_tensor([R, 0.0, 0.0, 0.0, 0.0, math.sqrt(MU_EARTH_M3_S2 / R)])
        states = observer.expand(len(self.q), 6).clone()
        states[:, 1] += R * self.q
        states[:, 5] -= 1.0e-4 * R * self.q
        self.rays = {}
        for at in (10, 120, 640):
            x, _ = propagate_si_with_stm(states, at, compute_stm=False)
            s, _ = propagate_si_with_stm(observer, at, compute_stm=False)
            self.rays[at] = F.normalize(x[:, :3] - s[:3], dim=-1)

    @staticmethod
    def curvature(q):
        return (1 - (1 + q * q).pow(-1.5)) / q

    def view(self, q, at):
        if at == 0:
            return self.curvature(q.clamp_min(0.01))[..., None]
        index = ((q - self.q[0]) / (self.q[-1] - self.q[0]) * (len(self.q) - 1)).clamp(0, len(self.q) - 1)
        lower = index.long().clamp_max(len(self.q) - 2)
        return torch.lerp(self.rays[at][lower], self.rays[at][lower + 1], (index - lower)[..., None])

    def observed(self, q, at, noisy):
        y = self.view(q, at)
        if noisy:
            y = y + torch.randn_like(y) * (0.005 if at == 0 else 5.0e-4)
        return F.normalize(y, dim=-1) if at else y

    def update(self, model, law, log_grid, obs, at, learn, *, qr_only=False):
        sigma = 0.005 if at == 0 else 5.0e-4
        center = self.mid + self.width * law.center[0] / self.scale[0]
        deriv = (self.view(center + 1.0e-5, at) - self.view(center - 1.0e-5, at)) / (2.0e-5)
        H = self.q.new_zeros((len(obs), 6))
        H[:, 0] = deriv * self.width / sigma
        innovation = (obs - self.view(center, at)) / sigma

        def score(x):
            z = x[..., 0] / self.scale[0]
            q = self.mid + self.width * z
            energy = -0.5 * ((self.view(q, at) - obs) / sigma).square().sum(-1)
            if at == 0:
                # Reference experiment starts with a uniform prior in the range interval.
                energy = energy + 0.5 * z.square()
            return torch.where((q >= self.q[0]) & (q <= self.q[-1]), energy, -torch.inf)

        if qr_only:
            law, teacher = qr_reference(law, H, innovation, 1.0), None
        else:
            law, teacher = model._condition(law, H, innovation, score, want_teacher=learn)
        log_grid = log_grid - 0.5 * ((self.view(self.q, at) - obs) / sigma).square().sum(-1)
        return law, teacher, log_grid

    def reference_draws(self, weights, n, generator):
        z = torch.randn((n, 6), generator=generator, dtype=self.q.dtype)
        p = (
            torch.arange(n, dtype=self.q.dtype) + torch.rand(n, generator=generator, dtype=self.q.dtype)
        ) / n
        ix = torch.searchsorted(weights.cumsum(0), p).clamp_max(len(self.q) - 1)
        z[:, 0] = (self.q[ix] - self.mid) / self.width
        return z * self.scale

    def start(self, model):
        law = WarpedOrbitPosterior(
            self.q.new_zeros(6),
            torch.eye(6, dtype=self.q.dtype),
            self.q.new_zeros(model.warp.shape_size),
            model.warp,
        )
        log_grid = torch.zeros_like(self.q)
        log_grid[[0, -1]] = -math.log(2.0)
        return law, log_grid


def marginal_density(law, sim):
    """Integrate the first triangular coupling over its three passive normals."""
    warp = law.warp
    z = torch.quasirandom.SobolEngine(3, scramble=True, seed=931).draw(128, dtype=sim.q.dtype)
    passive = torch.special.ndtri(z)
    raw = law.shape.reshape(2, 3, warp.coefficients)[0, 0]
    modulation = warp.conditioners[0](torch.asinh(passive)).reshape(-1, 3, warp.coefficients)[:, 0]
    params = raw * (1 + modulation.tanh())
    q = sim.q[::4]
    values = []
    assert float(law.root[0, 1:].abs().max()) < 1.0e-12
    factor = sim.width * law.root[0, 0]
    center = sim.mid + sim.width * law.center[0] / sim.scale[0]
    for section in q.split(64):
        y = ((section - center) / factor)[:, None].expand(-1, len(passive))
        base, ild = monotone_spline(y, params.expand(len(section), -1, -1), warp.bins, inverse=True)
        values.append((-0.5 * (base.square() + math.log(2 * math.pi)) + ild).exp().mean(-1) / factor.abs())
    return q, torch.cat(values)


def marginal_modes(q, density, sim):
    maxima = ((density[1:-1] > density[:-2]) & (density[1:-1] >= density[2:])).nonzero().flatten() + 1
    peaks = []
    for selected in (maxima[q[maxima] < sim.mid], maxima[q[maxima] > sim.mid]):
        if len(selected):
            peaks.append(int(selected[density[selected].argmax()]))
    result = {
        "modes_km": [float(7000 * q[k]) for k in peaks],
        "all_local_maxima_km": (7000 * q[maxima]).tolist(),
        "all_local_maxima_relative_height": (density[maxima] / density.max()).tolist(),
    }
    if len(peaks) == 2:
        valley = peaks[0] + int(density[peaks[0] : peaks[1] + 1].argmin())
        result.update(
            valley_location_km=float(q[valley] * 7000),
            valley_to_smaller_peak_density=float(density[valley] / density[peaks].min()),
        )
    return result


def plot_density(curves, output):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    fig.subplots_adjust(left=0.095, right=0.985, top=0.78, bottom=0.16, wspace=0.32)
    for ax, (at, x, exact, learned, projected) in zip(axes, curves):
        ax.plot(x, exact, color="black", label="Grid reference", linewidth=1.6)
        ax.plot(x, learned, color="#1967a3", label="Learned spline", linewidth=1.8)
        ax.plot(x, projected, color="#a54e0e", label="Projected Gaussian", linestyle="--")
        if at == 0:
            ax.axvspan(5250, 7140, color="#999999", alpha=0.13, label="Valley interval")
        else:
            ax.set_xlim(3850, 4500)
        ax.set_title("Ambiguous curvature" if at == 0 else "After the resolving LOS at 640 s")
        ax.set_xlabel("Initial slant range (km)")
        ax.set_ylabel("Probability density (1/km)")
        ax.spines[["right", "top"]].set_visible(False)
    labels, names = axes[0].get_legend_handles_labels()
    fig.legend(labels, names, loc="upper center", ncol=4, frameon=False, fontsize=9)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--figure", type=Path, help="Write the grid, learned and projected marginal density curves"
    )
    parser.add_argument(
        "--load-weights", type=Path, help="Evaluate a previously trained range-slice experiment"
    )
    parser.add_argument("--save-weights", type=Path, help="Save weights for this synthetic slice only")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    sim = RangeSlice()
    model = PPCOrbitFilter(nodes=256)
    optimizer = torch.optim.Adam(
        list(model.warp.parameters()) + list(model.update_shape.parameters()), lr=0.002
    )
    gen = torch.Generator().manual_seed(args.seed + 1)
    if args.load_weights:
        saved = torch.load(args.load_weights, map_location="cpu", weights_only=True)
        if saved.get("experiment") != "range_slice":
            raise ValueError("Expected weights from this range-slice experiment")
        model.load_state_dict(saved["model"])
        args.seed, args.iterations = saved["seed"], saved["training_cases"]
    for epoch in range(0 if args.load_weights else args.iterations):
        law, grid = sim.start(model)
        qtrue = (
            0.53 + 0.13 * torch.rand((), generator=gen)
            if epoch % 2 == 0
            else 1.10 + 0.20 * torch.rand((), generator=gen)
        ).double()
        optimizer.zero_grad(set_to_none=True)
        # Train short arcs, an intermediate view, and a resolving view on either side.
        for at in (0, 120, 640):
            obs = sim.observed(qtrue, at, True)
            law, teacher, grid = sim.update(model, law, grid, obs, at, True)
            labels = sim.reference_draws(grid.softmax(0), 64, gen)
            loss = -law.log_prob(labels).mean() / 6 + teacher.flow_matching_loss(model.warp, 16)
            (loss / 3).backward()
            law = law.detached()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

    if args.save_weights:
        torch.save(
            {
                "model": model.state_dict(),
                "seed": args.seed,
                "training_cases": args.iterations,
                "experiment": "range_slice",
            },
            args.save_weights,
        )
    model.eval()
    rows = []
    curves = []
    with torch.no_grad():
        law, grid = sim.start(model)
        affine, _ = sim.start(model)
        matched = PPCOrbitFilter(
            nodes=len(model.nodes), estimator="gaussian", observation_model="calibrated"
        )
        projected, _ = sim.start(matched)
        common = torch.randn((32768, 6), generator=torch.Generator().manual_seed(491), dtype=sim.q.dtype)
        for at in (0, 10, 120, 640):
            obs = sim.observed(sim.q.new_tensor(0.6), at, False)
            law, _, grid = sim.update(model, law, grid, obs, at, False)
            affine, _, _ = sim.update(model, affine, torch.zeros_like(grid), obs, at, False, qr_only=True)
            projected, _, _ = sim.update(matched, projected, torch.zeros_like(grid), obs, at, False)
            w = grid.softmax(0)
            targets = sim.reference_draws(w, 4096, torch.Generator().manual_seed(727))
            z = (sim.q - sim.mid) / sim.width
            oracle_mean, oracle_var = w @ z, w @ ((z - w @ z).square())
            gauss_root = torch.eye(6, dtype=z.dtype)
            gauss_root[0, 0] = oracle_var.sqrt()
            gauss_center = z.new_zeros(6)
            gauss_center[0] = oracle_mean * sim.scale[0]
            moment_gauss = WarpedOrbitPosterior(
                gauss_center, gauss_root, torch.zeros_like(law.shape), law.warp
            )
            row = {
                "epoch_sec": at,
                "reference_near_mass": float(w[sim.q <= sim.mid].sum()),
                "reference_valley_mass": float(w[(sim.q > 0.75) & (sim.q < 1.02)].sum()),
            }
            for name, posterior in (
                ("warp", law),
                ("qr", affine),
                ("projected_gaussian", projected),
                ("moment_gaussian", moment_gauss),
            ):
                draw = posterior.transform(common)[0]
                q = sim.mid + sim.width * draw[:, 0] / sim.scale[0]
                row[name] = {
                    "near_mass": float((q <= sim.mid).double().mean()),
                    "valley_mass": float(((q > 0.75) & (q < 1.02)).double().mean()),
                    "outside_mass": float(((q < sim.q[0]) | (q > sim.q[-1])).double().mean()),
                    "mean_range_km": float(q.mean() * 7000),
                    "sd_range_km": float(q.std() * 7000),
                    "nll_si": float(-posterior.log_prob(targets).mean()),
                }
            qplot, pplot = marginal_density(law, sim)
            row["warp"].update(marginal_modes(qplot, pplot, sim))
            if args.figure and at in (0, 640):
                average = sim.mid + sim.width * projected.center[0] / sim.scale[0]
                spread = sim.width * torch.linalg.vector_norm(projected.root[0])
                gaussian = torch.exp(-0.5 * ((qplot - average) / spread).square()) / (
                    math.sqrt(2 * math.pi) * spread
                )
                step = sim.q[1] - sim.q[0]
                curves.append((at, 7000 * qplot, w[::4] / (step * 7000), pplot / 7000, gaussian / 7000))
            row["reference_mean_km"] = float((w @ sim.q) * 7000)
            row["reference_sd_km"] = float((w @ (sim.q - w @ sim.q).square()).sqrt() * 7000)
            rows.append(row)
    report = {
        "seed": args.seed,
        "training_cases": args.iterations,
        "nodes_per_reference": len(model.nodes),
        "validation_draws": len(common),
        "results": rows,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.figure:
        plot_density(curves, args.figure)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

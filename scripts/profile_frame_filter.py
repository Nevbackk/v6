"""Measure the complete CPU step of one estimator on the same deterministic synthetic stream."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v6.models.diffusion_orbit import VPOrbitDiffusionFilter
from v6.models.ppc_flow import PPCOrbitFilter
from v6.models.reprojection import propagate_si_with_stm


def peak_rss_mib():
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "peak",
                    "working",
                    "paged_peak",
                    "paged",
                    "np_peak",
                    "np",
                    "pagefile",
                    "pagefile_peak",
                )
            ]

        counts = Counters()
        counts.cb = ctypes.sizeof(counts)
        kernel = ctypes.WinDLL("kernel32")
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi")
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counts), counts.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return counts.peak / 2**20
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (2**20 if sys.platform == "darwin" else 1024)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("ppc", "qr", "gaussian", "diffusion"), default="ppc")
    parser.add_argument(
        "--observation-model", choices=("diffusion", "direct", "calibrated"), default="diffusion"
    )
    args = parser.parse_args()
    torch.manual_seed(43)
    torch.set_num_threads(1)
    make_model = (
        VPOrbitDiffusionFilter
        if args.arch == "diffusion"
        else lambda **kw: PPCOrbitFilter(estimator=args.arch, **kw)
    )
    model = make_model(observation_model=args.observation_model).eval()
    x = torch.tensor([7.8e6, 2.5e5, 1.2e5, -200.0, 7100.0, 1400.0], dtype=torch.float64)
    station = x.new_tensor([6.878e6, 0.0, 0.0, 0.0, 7612.0, 1000.0])
    frames = []
    t = 0.0
    for dt in [1.0] * 16 + [143.0] + [1.0] * 15:
        t += dt
        x, _ = propagate_si_with_stm(x, dt, compute_stm=False)
        station, _ = propagate_si_with_stm(station, dt, compute_stm=False)
        obs = torch.nn.functional.normalize(x[:3] - station[:3], dim=0)
        frames.append((t, obs, station))
    init = x.new_tensor([7.8e6, 2.75e5, 1.05e5, -185.0, 7065.0, 1420.0])
    warm = model.start_track(init, 0.0, seed=11)
    model.step(warm, *frames[0])
    stream = model.start_track(init, 0.0, seed=11)
    calls = Counter()
    handles = []
    modules = (
        "sensor.net",
        "sensor.head",
        "denoiser",
        "update_shape.network",
        "update_shape.embedding",
        "warp.conditioners.0",
        "warp.conditioners.1",
        "tracklet_encoder.frame_cell",
    )
    for name, module in model.named_modules():
        if name in modules:
            handles.append(
                module.register_forward_hook(lambda mod, inputs, outputs, name=name: calls.update([name]))
            )
    grad = torch.autograd.grad

    def vjp(*a, **kw):
        calls.update(["autograd.grad"])
        return grad(*a, **kw)

    times = []
    with patch("torch.autograd.grad", side_effect=vjp):
        for row in frames:
            start = time.perf_counter_ns()
            out = model.step(stream, *row)
            times.append((time.perf_counter_ns() - start) / 1.0e6)
            if not torch.isfinite(out["posterior_state_si"]).all():
                raise FloatingPointError("Non-finite benchmark output")
    for handle in handles:
        handle.remove()
    print(
        json.dumps(
            dict(
                arch=args.arch,
                observation_model=args.observation_model,
                frames=len(frames),
                threads=torch.get_num_threads(),
                dtype="float64",
                device="cpu",
                latency_unit="ms",
                frame_p50=float(np.median(times)),
                frame_p95=float(np.quantile(times, 0.95)),
                gap143=times[16],
                total_forward_calls=dict(calls),
                rk4_substeps=stream.ledger.rk4_substeps,
                stm_substeps=stream.ledger.stm_substeps,
                peak_process_rss_mib=peak_rss_mib(),
                trainable_parameters=sum(p.numel() for p in model.parameters()),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

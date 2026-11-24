"""Physical-frame recurrence and cross-version angular diffusion."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from tests.test_submission_paths import make_prepared
from tests.test_v1_v2_models import _make_synthetic_two_block_record
from v6.data.prepared import Prepared
from v6.models.diffusion_orbit import AngularNoiseDiffusion, VPOrbitDiffusionFilter
from v6.models.orbit_proposals import RouteBOrbitFilter
from v6.models.reprojection import propagate_si_with_stm


class FrameDiffusionTests(unittest.TestCase):
    def test_reverse_angular_chain_has_correct_gaussian_marginals(self):
        mean = torch.tensor([0.35, -0.15], dtype=torch.float64)
        cov = torch.tensor([[1.2, 0.25], [0.25, 0.7]], dtype=mean.dtype)
        eye = torch.eye(2, dtype=mean.dtype)

        class GaussianOracle(torch.nn.Module):
            def forward(self, z, s, ctx):
                a = z.new_tensor(s).cos()
                v = 1 - a * a
                gain = torch.linalg.solve(a * a * cov + v * eye, a * cov).T
                clean = mean + (z - a * mean) @ gain.T
                conditional = (cov - a * gain @ cov).expand(len(z), -1, -1)
                return None, None, clean, conditional

        model = AngularNoiseDiffusion()
        model.net = GaussianOracle()
        for k in (1, 4):
            m, C = model.moments(mean.new_zeros(14), k, 8192, torch.Generator().manual_seed(4))
            torch.testing.assert_close(m, mean, atol=0.035, rtol=0)
            torch.testing.assert_close(C, cov, atol=0.045, rtol=0)

    def test_known_state_and_noisy_state_share_a_working_sensor(self):
        torch.manual_seed(5)
        rec = _make_synthetic_two_block_record(pos_offset_m=(0.0, 0.0, 0.0), vel_offset_mps=(0.0, 0.0, 0.0))
        args = {
            k: rec[k]
            for k in ("initial_state_si", "init_time_sec", "times_sec", "station_states_si", "observed_los")
        }
        for key in ("times_sec", "station_states_si", "observed_los"):
            args[key] = args[key][:5]
        args["valid_mask"] = torch.tensor([True, True, False, True, True])
        args["times_sec"] = args["times_sec"].clone()
        args["times_sec"][2] = float("nan")
        v1 = VPOrbitDiffusionFilter(known_initial=True, num_reverse_steps=2, num_particles=4)
        v2 = VPOrbitDiffusionFilter(num_reverse_steps=2, num_particles=4)
        v2.sensor.load_state_dict(v1.sensor.state_dict())
        args2 = {
            **args,
            "initial_state_si": args["initial_state_si"] + torch.tensor([0.0, 28000.0, 0.0, 0.0, 0.0, 0.0]),
        }
        before1, before2 = v1.rollout_track(**args), v2.rollout_track(**args2)
        with torch.no_grad():
            v1.sensor.net.v_head.bias.add_(1.0)
            v2.sensor.load_state_dict(v1.sensor.state_dict())
        after1, after2 = v1.rollout_track(**args), v2.rollout_track(**args2)
        torch.testing.assert_close(
            after1["posterior_states_si"], before1["posterior_states_si"], rtol=0, atol=0
        )
        self.assertGreater(float((after1["predicted_los"] - before1["predicted_los"]).norm()), 0.0)
        self.assertGreater(
            float((after2["posterior_states_si"][0] - before2["posterior_states_si"][0]).norm()), 0.0
        )
        expected, last_t = args["initial_state_si"], 0.0
        for k in [0, 1, 3, 4]:
            t = float(args["times_sec"][k])
            expected, _ = propagate_si_with_stm(expected, t - last_t, compute_stm=False)
            torch.testing.assert_close(after1["posterior_states_si"][k], expected, rtol=0, atol=0)
            self.assertGreater(float(torch.linalg.eigvalsh(after1["angular_covariance"][k]).min()), 0.0)
            last_t = t
        self.assertEqual(float(after2["posterior_states_si"][2].norm()), 0.0)

    def test_online_step_matches_padded_and_compressed_tracks_at_last_arrival(self):
        torch.manual_seed(31)
        t = torch.tensor([10.0, 21.0, 45.0, 610.0, 2874.0, 3299.0], dtype=torch.float64)
        valid = torch.tensor([True, True, True, False, True, True])
        x = t.new_tensor([7.8e6, 2.5e5, 1.2e5, -200.0, 7100.0, 1400.0])
        s = t.new_tensor([6.878e6, 0.0, 0.0, 0.0, 7612.0, 1000.0])

        def sequence(start):
            return torch.stack([propagate_si_with_stm(start, float(tk), compute_stm=False)[0] for tk in t])

        stations, targets = sequence(s), sequence(x)
        obs = torch.nn.functional.normalize(targets[:, :3] - stations[:, :3], dim=1)
        for known in (True, False):
            model = VPOrbitDiffusionFilter(known_initial=known, num_reverse_steps=2, num_particles=4)
            initial = x if known else x + t.new_tensor([0.0, 4000.0, 0.0, 0.0, 0.0, 0.0])
            args = dict(initial_state_si=initial, init_time_sec=0.0, seed=23)
            a = model.rollout_track(
                **args, times_sec=t, observed_los=obs, station_states_si=stations, valid_mask=valid
            )
            b = model.rollout_track(
                **args, times_sec=t[valid], observed_los=obs[valid], station_states_si=stations[valid]
            )
            live = model.start_track(initial, 0.0, seed=23)
            indices = valid.nonzero().flatten().tolist()
            for n, k in enumerate(indices):
                step = model.step(live, float(t[k]), obs[k], stations[k])
                for field, single in (
                    ("prior_states_si", "prior_state_si"),
                    ("posterior_states_si", "posterior_state_si"),
                    ("predicted_los", "predicted_los"),
                    ("angular_covariance", "angular_covariance"),
                ):
                    torch.testing.assert_close(step[single], a[field][k], rtol=0, atol=0)
                    torch.testing.assert_close(step[single], b[field][n], rtol=0, atol=0)
            self.assertEqual(live.count, len(indices))
            self.assertEqual(live.epoch, 3299.0)

    def test_chart_particles_consume_new_rays_between_refreshes(self):
        rec = _make_synthetic_two_block_record()
        x = rec["initial_state_si"]
        states = torch.stack((x, x + x.new_tensor([0.0, 0.01, 0.0, 0.0, 0.0, 0.0])))
        args = {k: rec[k][:2] for k in ("times_sec", "observed_los", "station_states_si")}
        for route in ("b1", "b2"):
            model = RouteBOrbitFilter(route, num_steps=1, num_particles=2)
            with patch.object(
                model,
                "sample_prefix",
                return_value=(states, x.new_full((2,), -float(torch.log(torch.tensor(2.0))))),
            ) as proposal:
                result = model.rollout_track(initial_state_si=x, init_time_sec=0.0, **args)
                self.assertEqual(proposal.call_count, 1)
                self.assertGreater(
                    float((result["posterior_states_si"][1] - result["prior_states_si"][1]).norm()), 0.0
                )

    def test_versioned_cli_initialization_transfer_and_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proto = make_prepared(root)
            prep = Prepared(root, proto, "train")
            key = next(iter(prep.rows))
            clean, noisy = prep.record(key, exact_initial=True), prep.record(key)
            self.assertEqual(clean["init_time_sec"], noisy["init_time_sec"])
            torch.testing.assert_close(
                torch.as_tensor(clean["initial_state_si"]),
                torch.tensor([7.8e6, 2.5e5, 1.2e5, -200.0, 7100.0, 1400.0], dtype=torch.float64),
            )
            self.assertGreater(
                float(
                    (
                        torch.as_tensor(clean["initial_state_si"]) - torch.tensor(noisy["initial_state_si"])
                    ).norm()
                ),
                1.0e3,
            )
            common = [
                "--prepared-root",
                str(root),
                "--protocol-dir",
                str(proto),
                "--device",
                "cpu",
                "--K",
                "2",
                "--P",
                "4",
                "--max-records-per-epoch",
                "1",
                "--max-eval-records",
                "1",
                "--seed",
                "7",
            ]

            def call(script, version, more):
                cmd = [
                    sys.executable,
                    "-s",
                    str(Path(__file__).resolve().parents[1] / "scripts" / (script + "_" + version + ".py")),
                    "--run-dir",
                    str(root / version),
                    *common,
                    *more,
                ]
                return subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    env={
                        **os.environ,
                        "CUDA_VISIBLE_DEVICES": "",
                        "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                    },
                )

            for version in ("v1", "v2"):
                transfer = [] if version == "v1" else ["--sensor-checkpoint", str(root / "v1" / "best.pt")]
                trained = call("train", version, ["--epochs", "1", *transfer])
                self.assertEqual(trained.returncode, 0, trained.stderr)
                tested = call("validate", version, ["--eval-split", "paired_val"])
                self.assertEqual(tested.returncode, 0, tested.stderr)
                summary = json.loads((root / version / "eval_diff_paired_val.json").read_text())
                self.assertEqual(
                    summary["initializations"], ["known_state" if version == "v1" else "gauss_ls"]
                )
                self.assertEqual(summary["overall"]["valid_frames"], 24)
                saved = torch.load(root / version / "best.pt", weights_only=True)
                self.assertEqual(saved["version"], version)
                self.assertEqual(saved["recursion"], "valid_frame")
                self.assertTrue(any(k.startswith("sensor.net.") for k in saved["model"]))
                self.assertEqual(any(k.startswith("denoiser.") for k in saved["model"]), version == "v2")
            wrong = call("validate", "v2", ["--resume", str(root / "v1" / "best.pt")])
            self.assertNotEqual(wrong.returncode, 0)
            self.assertIn("initialization protocol", wrong.stderr)

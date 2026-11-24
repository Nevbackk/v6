"""Public entry points and recovery behavior on a small prepared observation set."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from tests.test_v1_v2_models import _make_synthetic_two_block_record
from v6.data.observation_clock import causal_diffusion_events
from v6.data.splits import assign_source_split
from v6.models.diffusion_orbit import RouteBGuidedCosineFlow2D
from v6.models.orbit_proposals import RouteBOrbitFilter, build_chart_prefix
from v6.training_diffusion import train_route_b_cosine_flow_step_on_record
from v6.models.diffusion_orbit import VPOrbitDiffusionFilter, compute_noisy_los_guidance_o6
from v6.models.reprojection import estimate_online_los_noise_floor, propagate_si_with_stm
from v6.models.v1_ltr_lm import ClassicalLMRefiner, V1OnlineFilter
from v6.models.v2_asd_sc import ASDSCFilter
from v6.training_v2 import DAggerRolloutBuffer, train_two_checkpoint_pair


def make_prepared(root):
    """Materialize the documented Prepared format, including source allowlists."""
    proto = root / "protocol"
    proto.mkdir()
    usable = {}
    for i in range(200):
        source = f"cpu_{i}"
        bucket = assign_source_split(source)
        usable.setdefault(bucket, []).append(source)
    config = {"source_filter": {}}
    fixture = _make_synthetic_two_block_record()
    for split, sources in (("train", usable["train_fit"][:2] + usable["dev_select"][:1]),
                           ("val", ["cpu_validation"])):
        init_dir = root / "initialization_cache" / split
        shard_dir = root / "prepared_cache" / split / "shards"
        init_dir.mkdir(parents=True)
        shard_dir.mkdir(parents=True)
        allow = proto / f"{split}.csv"
        keys, init, raw = [], [], []
        with allow.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(("shard_id", "source_id", "b0_position_error_km", "split"))
            for i, source in enumerate(sources):
                writer.writerow((0, source, 30., split))
                meta = {"source_id": source, "rhythm": "dense", "record_id": str(i)}
                key = f"{split}_{source}"
                x0 = fixture["initial_state_si"].tolist()
                # Distinct calibrated source errors.
                x0[0] += i*250.
                keys.append(key)
                init.append(dict(cache_key=key, record_key=meta, shard_id=0,
                                 shared_initialization=True, initial_epoch_sec=0., dense_t_init=0.,
                                 initial_state=x0, B0=x0, initial_observation_index=-1,
                                 filter_observation_count=len(fixture["times_sec"])))
                raw.append(dict(**meta, block_start_time_sec=0., t=fixture["times_sec"],
                                los=fixture["observed_los"], sat_state=fixture["station_states_si"],
                                tgt_state=fixture["truth_states_si"], sat_id=fixture["station_ids"]))
        config["source_filter"][split] = dict(allowlist_path=allow.name,
                                               allowlist_sha256=hashlib.sha256(allow.read_bytes()).hexdigest())
        config[f"fixed_{split}_record_keys"] = keys
        config[f"{split}_records"] = len(keys)
        (init_dir/"shard_000000.jsonl").write_text("\n".join(map(json.dumps, init)), encoding="utf-8")
        torch.save({"records": raw}, shard_dir/"shard_000000.pt")
    (proto/"config.json").write_text(json.dumps(config), encoding="utf-8")
    return proto


class TestSubmissionPaths(unittest.TestCase):
    def test_high_precision_likelihood_and_denoiser_vjp(self):
        z = torch.zeros(6, dtype=torch.float64, requires_grad=True)
        H = torch.zeros((2, 6), dtype=z.dtype)
        H[0, 0], H[1, 1] = 1.e4, 2.
        C = torch.eye(6, dtype=z.dtype)
        J = 2*C + torch.roll(C, 1, 1)
        nu = z.new_tensor((1., .25))
        sigma = 1.e-7
        actual = compute_noisy_los_guidance_o6(z, J@z, C, H_z0=H, nu_res=nu, sigma_u_rad=sigma)
        dense = J.T @ H.T @ torch.linalg.solve(H@C@H.T + sigma**2*torch.eye(2, dtype=z.dtype), nu)
        torch.testing.assert_close(actual, dense, rtol=1.e-9, atol=1.e-12)

    def test_moving_station_noise_and_preupdate_backbone_gate(self):
        rec = _make_synthetic_two_block_record(pos_offset_m=(0., 0., 0.), vel_offset_mps=(0., 0., 0.))
        t, stn, obs = rec["times_sec"][:5], rec["station_states_si"][:5], rec["observed_los"][:5]
        count = estimate_online_los_noise_floor(t, obs, up_to_idx=4, station_states_si=stn)[2]
        self.assertEqual(count, 3)
        class OffsetBackbone(torch.nn.Module):
            def step(self, previous, dt, los, station):
                return previous + 1.e5
        model = V1OnlineFilter(ClassicalLMRefiner("cl_gn1_matched"), base_filter=OffsetBackbone())
        out = model.rollout_track(initial_state_si=rec["initial_state_si"], init_time_sec=0.,
                                  times_sec=t[:1], observed_los=obs[:1], station_states_si=stn[:1])
        torch.testing.assert_close(out["posterior_states_si"], out["prior_states_si"], atol=0, rtol=0)
        out = model.rollout_track(initial_state_si=rec["initial_state_si"], init_time_sec=0.,
                                  times_sec=t[:1], observed_los=-obs[:1], station_states_si=stn[:1])
        self.assertGreater(float((out["posterior_states_si"]-out["prior_states_si"]).norm()), 1.e4)

    def test_published_prefix_is_invariant_at_a_future_gap(self):
        rec = _make_synthetic_two_block_record()
        for model in (VPOrbitDiffusionFilter(num_reverse_steps=1, num_particles=2, los_budget=2),
                      RouteBOrbitFilter("b1", num_steps=1, num_particles=2, los_budget=2),
                      RouteBOrbitFilter("b2", num_steps=1, num_particles=2, los_budget=2)):
            def rollout(n):
                return model.rollout_track(initial_state_si=rec["initial_state_si"], init_time_sec=0.,
                    times_sec=rec["times_sec"][:n], observed_los=rec["observed_los"][:n],
                    station_states_si=rec["station_states_si"][:n], block_start_time_sec=0., seed=11)
            full = rollout(24)
            for n in (12, 20):
                sliced = rollout(n)
                torch.testing.assert_close(sliced["posterior_states_si"], full["posterior_states_si"][:n],
                                           rtol=0, atol=0)

    def test_replay_uses_the_accepted_state_and_clears_local_memory(self):
        from unittest.mock import patch
        rec = _make_synthetic_two_block_record()
        model = ASDSCFilter(num_microsteps=1, block_offsets_sec=(30., 90., 150.))
        buffer = DAggerRolloutBuffer()
        def candidate(state, station, features):
            return state + 1.e3, state.new_tensor(2.)
        with patch.object(model.reanchor_head, "propose_reanchor", side_effect=candidate), \
             patch.object(model, "_blind_arc_mse", side_effect=[1., 0.]):
            buffer.collect_from_records(model, {"track":rec}, ["track"], counterfactual_prob=0.)
        switched = [x for x in buffer.samples if x.switched_state_si is not None]
        self.assertEqual(len(switched), 1)
        sample = switched[0]
        expected, _ = propagate_si_with_stm(sample.switched_state_si,
                      sample.cp_next.eval_time_sec-sample.cp_m.eval_time_sec, compute_stm=False)
        with patch.object(model, "run_checkpoint", wraps=model.run_checkpoint) as traced:
            loss, _ = train_two_checkpoint_pair(model, sample, rec)
            loss.backward()
        second = traced.call_args_list[1].kwargs
        torch.testing.assert_close(second["prior_state_si"], expected, atol=0, rtol=0)
        self.assertEqual(second["memory_in"].dim_p, 0)

    def test_clock_is_independent_of_future_and_missing_exposures(self):
        t = torch.tensor([10.,50.,300.,670.,898.])
        clock = causal_diffusion_events(t, None, 10., 0.)
        self.assertEqual(clock, [(300.,2,False),(630.,2,True)])
        self.assertEqual(causal_diffusion_events(t[:2], None, 10., 0.), [])
        self.assertEqual(causal_diffusion_events(t[:3], None, 10., 0.), clock[:1])
        held = causal_diffusion_events(t[:3], None, 10., 0., readout_until_sec=630.)
        self.assertEqual(held, clock)

    def test_both_teacher_charts_and_nondefault_candidate_count(self):
        from unittest.mock import patch
        rec = _make_synthetic_two_block_record()
        flow = RouteBOrbitFilter("b1", num_steps=1, num_particles=4, context_dim=16)
        self.assertFalse(build_chart_prefix(flow.encoder, flow.flow, rec, 11, device="cpu").lambert)
        self.assertTrue(build_chart_prefix(flow.encoder, flow.flow, rec, 23, device="cpu").lambert)
        for block in (0,1):
            torch.manual_seed(22)
            flow.zero_grad()
            with patch("v6.training_diffusion.random.choice", side_effect=lambda blocks: blocks[block]):
                loss = train_route_b_cosine_flow_step_on_record(flow.flow, flow.encoder, rec, num_ref_candidates=31)
            loss.backward()
            self.assertGreater(float(flow.flow.chart_embedding.weight.grad[block].norm()), 0.)

    def test_real_train_validate_and_missing_checkpoint(self):
        v6_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proto = make_prepared(root)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS":"1", "MKL_NUM_THREADS":"1"}
            for route in ("a","b1","b2"):
                out = root/route
                common = ["--prepared-root",str(root),"--protocol-dir",str(proto),
                          "--run-dir",str(out),"--device","cpu","--route",route,
                          "--K","1","--P","2","--B-los","2","--batch-size","1",
                          "--max-records-per-epoch","1","--max-eval-records","1","--seed","11"]
                def call(script, more):
                    return subprocess.run([sys.executable,"-s",str(v6_root/"scripts"/script),*common,*more],
                                          cwd=v6_root, env=env, capture_output=True, text=True)
                no_ckpt = call("validate_diffusion.py", ["--resume",str(out/"missing.pt")])
                self.assertNotEqual(no_ckpt.returncode, 0)
                trained = call("train_diffusion.py", ["--epochs","1"] + (["--no-rollout-priors"] if route == "a" else []))
                self.assertEqual(trained.returncode, 0, trained.stderr)
                self.assertTrue((out/"best.pt").is_file())
                val = call("validate_diffusion.py", ["--eval-split","paired_val"])
                self.assertEqual(val.returncode, 0, val.stderr)
                summary = json.loads((out/"eval_diff_paired_val.json").read_text())
                self.assertEqual(summary["overall"]["count"], 1)
                self.assertIn("v2_primary_block2_t30_forecast300_pos_km", summary["overall"])
                if route != "a":
                    self.assertGreater(summary["overall"]["attempted_proposal_batches"], 0)
                else:
                    saved = torch.load(out/"last.pt", weights_only=True)
                    self.assertEqual(saved["reference_prior"]["num_sources"], 2)
                    self.assertIn("optimizer", saved)
                    resumed = call("train_diffusion.py", ["--epochs","2","--no-rollout-priors"])
                    self.assertEqual(resumed.returncode, 0, resumed.stderr)
                    newer = torch.load(out/"last.pt", weights_only=True)
                    self.assertLessEqual(newer["best_score"], saved["best_score"])
                    self.assertEqual(newer["epoch"], 2)
                    mismatched = call("train_diffusion.py", ["--epochs","3","--split-seed","7"])
                    self.assertNotEqual(mismatched.returncode, 0)
                    self.assertIn("split_seed", mismatched.stderr)

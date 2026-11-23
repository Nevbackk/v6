import tempfile
import unittest
from pathlib import Path

import torch

from v6.data.batch_collate import collate_filter_sequences
from v6.training import errors, initialize_direct, model_for, predict, save


class FilterSmoke(unittest.TestCase):
    def test_both_heads(self):
        dtype = torch.float64
        state = torch.tensor([7e6, 0, 0, 0, 7500, 0], dtype=dtype)
        times = torch.tensor([5., 10., 15.], dtype=dtype)
        record = dict(initial_state_si=state, init_time_sec=0., times_sec=times,
                      observed_los=torch.tensor([[1., 0, 0]] * 3, dtype=dtype),
                      station_states_si=torch.zeros((3, 6), dtype=dtype),
                      station_ids=torch.zeros(3, dtype=torch.long),
                      station_id_reliable=torch.ones(3, dtype=torch.bool))
        batch = collate_filter_sequences([record])
        batch["truth_states_si"] = state.expand(1, 3, 6)
        gain = model_for("gain", "blockwise_checkpoint", "cpu")
        direct = model_for("direct_correction", "blockwise_checkpoint", "cpu")
        initialize_direct(direct, gain.state_dict())
        baseline = predict(gain, batch, "cpu").posterior_states_si
        torch.testing.assert_close(predict(direct, batch, "cpu").posterior_states_si, baseline)
        for model in (gain, direct):
            optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
            output = predict(model, batch, "cpu")
            loss = errors(output, batch).mean()
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            head = model.gain_head if model.update_mode == "gain" else model.correction_head
            self.assertGreater(head.net[-1].weight.grad.norm().item(), 0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "last.pt"
                save(path, {"model": model.state_dict(), "optimizer": optimizer.state_dict()})
                loaded = torch.load(path, weights_only=True)
                model_for(model.update_mode, "blockwise_checkpoint", "cpu").load_state_dict(loaded["model"], strict=True)
                self.assertTrue(loaded["optimizer"]["state"])

"""Physical publication times shared by the two diffusion estimators."""

import numpy as np
import torch


def causal_diffusion_events(times, valid_mask, init_time_sec, block_start_time_sec=None,
                            *, explicit_flush=False, readout_until_sec=None):
    """Return (time, last available observation index, primary endpoint) in time order.

    The acquisition plan has 300 s observing blocks separated by 300 s gaps.
    Source metadata can supply the first scheduled opening independently of cadence.
    If absent, the shared initialization epoch defines the nominal clock. An explicit
    end-of-stream flush is distinct from merely receiving a shorter array.
    """
    t = torch.as_tensor(times).detach().cpu().double().numpy().reshape(-1)
    mask = np.ones(len(t), dtype=bool) if valid_mask is None else torch.as_tensor(
        valid_mask).detach().cpu().numpy().astype(bool).reshape(-1)
    idx = np.flatnonzero(mask)
    if not idx.size:
        return []
    valid = t[idx]
    origin = float(init_time_sec if block_start_time_sec is None else block_start_time_sec)
    until = float(valid[-1] if readout_until_sec is None else readout_until_sec)
    first = max(0, int(np.ceil((float(init_time_sec) - origin - 300.) / 600.)))
    last = int(np.floor((until - origin - 300.) / 600.))
    times_out = {origin + 600.*i + 300.: False for i in range(first, last+1)}
    endpoint = origin + 630.
    if init_time_sec <= endpoint <= until:
        times_out[endpoint] = True
    if explicit_flush and float(init_time_sec) <= valid[-1] <= until:
        times_out.setdefault(float(valid[-1]), False)
    events = []
    previous = -1
    for tm, primary in sorted(times_out.items()):
        pos = np.searchsorted(valid, tm, side="right")-1
        k = int(idx[pos]) if pos >= 0 else -1
        if k > previous or primary:
            events.append((tm, k, primary))
            previous = k
    return events

"""Observation time and feedback indices for a single causal track."""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ObservationFrame:
    index: int
    previous_index: int
    time_sec: float
    dt_sec: float


class FrameNumericalError(FloatingPointError):
    """A numerical failure at an identified valid observation of a record."""

    def __init__(self, frame, error):
        self.frame_index = frame.index
        self.time_sec = frame.time_sec
        self.cause_type = type(error).__name__
        super().__init__(str(error))


def observation_frames(times_sec, valid_mask, init_time_sec):
    """Yield each valid arrival, including the last one, using the previous valid epoch."""
    times = torch.as_tensor(times_sec).detach().to(device="cpu", dtype=torch.float64).numpy()
    if times.ndim != 1 or not np.isfinite(init_time_sec):
        raise ValueError("Expected one track and a finite initial epoch")
    mask = (
        np.ones(times.shape, dtype=bool)
        if valid_mask is None
        else torch.as_tensor(valid_mask).detach().cpu().numpy().astype(bool)
    )
    if mask.shape != times.shape:
        raise ValueError("valid_mask must match times_sec")
    previous = -1
    last_time = float(init_time_sec)
    for index in np.flatnonzero(mask):
        now = float(times[index])
        if not np.isfinite(now) or now < last_time:
            raise ValueError("Valid observation times must be finite and nondecreasing from initialization")
        yield ObservationFrame(int(index), previous, now, now - last_time)
        previous, last_time = int(index), now

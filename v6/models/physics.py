"""Single-trajectory adapter to the production batched two-body RK4."""
import math
import torch
from .batched_physics import MAX_SUBSTEP_SEC, propagate_scaled_batch, state_scale

def propagate_scaled(z, dt_sec, max_substep_sec=MAX_SUBSTEP_SEC):
    steps = max(1, math.ceil(abs(dt_sec) / max_substep_sec))
    return propagate_scaled_batch(z, dt_sec, steps, steps, max_substep_sec)


def predict_los_from_scaled(z: torch.Tensor, station_pos_m: torch.Tensor) -> torch.Tensor:
    """Predict ECI LOS using this epoch's station position."""
    scale = state_scale(z.device, z.dtype)
    state_si = z * scale
    if station_pos_m.ndim != 2 or station_pos_m.shape != (z.shape[0], 3):
        raise ValueError("station_pos_m must have shape [batch, 3]")
    delta = state_si[:, :3] - station_pos_m
    rho = torch.linalg.vector_norm(delta, dim=1, keepdim=True)
    return delta / rho

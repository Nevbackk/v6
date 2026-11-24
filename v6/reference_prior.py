"""TRAIN-fit calibration of the shared Gauss-LS initialization error."""

import torch

from .models.batched_physics import state_scale
from .models.reprojection import build_los_frame, propagate_si_with_stm


def los_state_basis(initial, station_position):
    frame, _ = build_los_frame(initial[:3] - station_position[:3].to(initial))
    return torch.block_diag(frame, frame)


@torch.no_grad()
def fit_reference_prior(prepared, keys):
    """One sample per independent source, in a station-relative frame at initialization."""
    seen = set()
    errors = []
    for key in sorted(keys, key=lambda k: (prepared.rows[k]["shard_id"], k)):
        meta = prepared.rows[key]
        source = (meta["shard_id"], meta["record_key"]["source_id"])
        if source in seen:
            continue
        rec = prepared.record(key)
        t = torch.as_tensor(rec["times_sec"], dtype=torch.float64)
        valid = torch.as_tensor(rec.get("valid_mask", torch.ones_like(t, dtype=torch.bool)))
        indices = torch.nonzero(valid, as_tuple=True)[0]
        if not indices.numel():
            continue
        i = indices[0].item()
        dt = float(rec["init_time_sec"]) - t[i].item()
        x0 = torch.as_tensor(rec["initial_state_si"], dtype=torch.float64)
        truth, _ = propagate_si_with_stm(torch.as_tensor(rec["truth_states_si"][i], dtype=x0.dtype),
                                         dt, compute_stm=False)
        station = rec.get("initial_station_si")
        station = x0.new_zeros(6) if station is None else torch.as_tensor(station, dtype=x0.dtype)
        d = state_scale(device=x0.device, dtype=x0.dtype)
        errors.append(los_state_basis(x0, station[:3]).T @ ((truth - x0)/d))
        seen.add(source)
    if not errors:
        raise ValueError("No usable TRAIN-fit initialization for prior calibration")
    samples = torch.stack(errors)
    mean = samples.mean(0)
    if len(samples) == 1:
        cov = torch.eye(6, dtype=mean.dtype)
    else:
        residuals = samples - mean
        cov = residuals.T @ residuals / len(samples)
        # Oracle approximating shrinkage; the scalar is estimated from the samples.
        mu = cov.trace()/6
        alpha = cov.square().mean()
        denominator = (len(samples)+1)*(alpha-mu.square()/6)
        shrink = ((alpha+mu.square())/denominator).clamp(0, 1) if denominator > 0 else 1.
        eye = torch.eye(6, dtype=mean.dtype)
        cov = (1-shrink)*cov + shrink*mu*eye + torch.finfo(mean.dtype).eps*eye
    return {"mean_los": mean, "cov_los": cov, "num_sources": len(samples)}

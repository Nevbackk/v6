"""Stratified evaluation protocol, paired bootstrap CIs, and V2 stability/endpoint audits.

Implements the complete co-signed evaluation specification across:
- docs/JOINT_RESEARCH_PLAN.md (§4–§5)
- docs/FLAGSHIP_ACLASS_PROPOSAL.md (§3–§4)
- docs/MODEL_V2_ADAPTIVE_DENOISING_PLAN.md (§3–§4)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .data.splits import B0_ERROR_STRATA_KM, CADENCE_NAMES, b0_stratum_label
from .models.batched_physics import STATE_SCALE_VALUES
from .models.reprojection import build_los_frame, propagate_si_with_stm


@dataclass(frozen=True)
class RecordEvalRow:
    """Per-record summary for full-table, per-cadence, source-equal, and paired bootstrap stats."""

    record_id: str
    source_id: str
    cadence: str
    b0_pos_error_km: float
    pos_rmse_km: float
    vel_rmse_mps: float
    scaled_mse: float
    block6_end_vel_rmse_mps: float
    block6_end_radial_pos_km: float
    is_finite: bool
    diverged: bool
    prior_pos_rmse_km: float | None = None
    block2_t30_forecast300_pos_km: float | None = None


def compute_los_local_errors(
    pred_state_si: Tensor,
    truth_state_si: Tensor,
    station_state_si: Tensor,
) -> dict[str, Tensor]:
    """Decompose 6D state error (pred - truth) into LOS-radial and two transverse axes."""
    rel_true = truth_state_si[..., :3] - station_state_si[..., :3]
    U_true, _ = build_los_frame(rel_true)
    u_r = U_true[..., :, 0]
    q1 = U_true[..., :, 1]
    q2 = U_true[..., :, 2]

    err_r_km = (pred_state_si[..., :3] - truth_state_si[..., :3]) / 1000.0
    err_v_mps = pred_state_si[..., 3:] - truth_state_si[..., 3:]

    r_R = (err_r_km * u_r).sum(dim=-1)
    r_T1 = (err_r_km * q1).sum(dim=-1)
    r_T2 = (err_r_km * q2).sum(dim=-1)
    r_T = torch.hypot(r_T1, r_T2)

    v_R = (err_v_mps * u_r).sum(dim=-1)
    v_T1 = (err_v_mps * q1).sum(dim=-1)
    v_T2 = (err_v_mps * q2).sum(dim=-1)
    v_T = torch.hypot(v_T1, v_T2)

    return {
        "r_R_km": r_R,
        "r_T1_km": r_T1,
        "r_T2_km": r_T2,
        "r_T_km": r_T,
        "v_R_mps": v_R,
        "v_T1_mps": v_T1,
        "v_T2_mps": v_T2,
        "v_T_mps": v_T,
    }


def directional_benefit_metrics(
    needed_delta_scaled: Tensor,
    proposed_delta_scaled: Tensor,
    *,
    zero_tol: float = 1.0e-12,
    truth_tol: float = 1.0e-9,
) -> dict[str, float]:
    """Selection-bias-free 3-metric directional alignment report (JOINT_RESEARCH_PLAN §4.3).

    Reports simultaneously:
    1. zero_step_fraction: fraction of anchors with ||c|| <= zero_tol
    2. nonzero_cosine_p50: median cos(d, c) among nonzero steps
    3. all_anchor_positive_benefit_rate: fraction of all nonzero-error anchors with
       B(d, c) = 2 d^T c - ||c||^2 > 0
    """
    d = needed_delta_scaled.reshape(-1, needed_delta_scaled.shape[-1]).to(torch.float64)
    c = proposed_delta_scaled.reshape(-1, proposed_delta_scaled.shape[-1]).to(torch.float64)
    d_norm = torch.linalg.vector_norm(d, dim=-1)
    c_norm = torch.linalg.vector_norm(c, dim=-1)

    is_zero_step = c_norm <= float(zero_tol)
    zero_frac = float(is_zero_step.to(torch.float64).mean().item()) if c.shape[0] > 0 else 0.0

    nonzero_mask = (~is_zero_step) & (d_norm > float(truth_tol))
    if bool(nonzero_mask.any()):
        cos_vals = (d[nonzero_mask] * c[nonzero_mask]).sum(dim=-1) / (
            d_norm[nonzero_mask] * c_norm[nonzero_mask]
        )
        cos_p50 = float(torch.quantile(cos_vals, 0.5).item())
    else:
        cos_p50 = 0.0

    valid_truth = d_norm > float(truth_tol)
    if bool(valid_truth.any()):
        benefit = 2.0 * (d[valid_truth] * c[valid_truth]).sum(dim=-1) - c_norm[valid_truth].square()
        pos_benefit_rate = float((benefit > 0.0).to(torch.float64).mean().item())
    else:
        pos_benefit_rate = 0.0

    return {
        "zero_step_fraction": zero_frac,
        "nonzero_cosine_p50": cos_p50,
        "all_anchor_positive_benefit_rate": pos_benefit_rate,
    }


def _distribution_summary(values: np.ndarray, cap_for_diverged: float) -> dict[str, float]:
    """Compute mean, P50, P90, P95, P99, and worst-1% CVaR-99% while keeping failures in denominator."""
    if values.size == 0:
        return {
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "cvar99": 0.0,
        }
    capped = np.where(np.isfinite(values), np.minimum(values, cap_for_diverged), cap_for_diverged)
    p99 = float(np.quantile(capped, 0.99))
    tail_count = max(1, int(np.ceil(0.01 * capped.size)))
    sorted_vals = np.sort(capped)
    cvar99 = float(np.mean(sorted_vals[-tail_count:]))
    finite_only = values[np.isfinite(values)]
    raw_mean = float(np.mean(finite_only)) if finite_only.size == values.size else float("inf")
    return {
        "raw_mean": raw_mean,
        "capped_mean": float(np.mean(capped)),
        "p50": float(np.quantile(capped, 0.50)),
        "p90": float(np.quantile(capped, 0.90)),
        "p95": float(np.quantile(capped, 0.95)),
        "p99": p99,
        "cvar99": cvar99,
    }


def summarize_record_rows(
    rows: Sequence[RecordEvalRow],
    *,
    divergence_pos_km: float = 1000.0,
    good_orbit_harm_threshold_km: float = 20.0,
) -> dict[str, Any]:
    """Produce overall, 4-cadence, source-equal, and 5-stratum tables from RecordEvalRow items."""
    if not rows:
        return {"count": 0}

    def _agg(subset: Sequence[RecordEvalRow]) -> dict[str, Any]:
        n = len(subset)
        if n == 0:
            return {"count": 0}
        pos = np.array([r.pos_rmse_km if r.is_finite else np.inf for r in subset], dtype=np.float64)
        vel = np.array([r.vel_rmse_mps if r.is_finite else np.inf for r in subset], dtype=np.float64)
        smse = np.array([r.scaled_mse if r.is_finite else np.inf for r in subset], dtype=np.float64)
        b6_v = np.array([r.block6_end_vel_rmse_mps if r.is_finite else np.inf for r in subset], dtype=np.float64)
        b6_r = np.array([abs(r.block6_end_radial_pos_km) if r.is_finite else np.inf for r in subset], dtype=np.float64)
        nonfinite_rate = float(np.mean([not r.is_finite for r in subset]))
        diverged_rate = float(np.mean([r.diverged or (not r.is_finite) for r in subset]))

        # Good-orbit new harm rate strictly relative to un-updated propagated prior at same epoch
        good_sub = [r for r in subset if 0.0 <= r.b0_pos_error_km < 10.0 and r.prior_pos_rmse_km is not None]
        if good_sub:
            newly_harmed = [
                (not r.is_finite)
                or (
                    r.prior_pos_rmse_km <= good_orbit_harm_threshold_km
                    and r.pos_rmse_km > good_orbit_harm_threshold_km
                )
                for r in good_sub
            ]
            good_harm_rate = float(np.mean(newly_harmed))
        else:
            good_harm_rate = 0.0

        b2_vals = np.array(
            [
                r.block2_t30_forecast300_pos_km
                for r in subset
                if r.block2_t30_forecast300_pos_km is not None
            ],
            dtype=np.float64,
        )
        out: dict[str, Any] = {
            "count": n,
            "nonfinite_rate": nonfinite_rate,
            "diverged_rate_gt_1000km": diverged_rate,
            "scaled_mse_finite_mean": float(np.mean(smse[np.isfinite(smse)])) if np.isfinite(smse).any() else float("inf"),
            "position_km": _distribution_summary(pos, cap_for_diverged=max(divergence_pos_km * 10.0, 10000.0)),
            "velocity_mps": _distribution_summary(vel, cap_for_diverged=10000.0),
            "block6_end_vel_mps": _distribution_summary(b6_v, cap_for_diverged=10000.0),
            "block6_end_radial_pos_km": _distribution_summary(b6_r, cap_for_diverged=10000.0),
            "good_orbit_lt10km_new_harm_rate": good_harm_rate,
        }
        if b2_vals.size > 0:
            out["v2_primary_block2_t30_forecast300_pos_km"] = _distribution_summary(
                b2_vals, cap_for_diverged=10000.0
            )
        return out

    by_cadence = {
        cad: _agg([r for r in rows if r.cadence == cad])
        for cad in CADENCE_NAMES
    }
    by_stratum = {
        label: _agg([r for r in rows if b0_stratum_label(r.b0_pos_error_km) == label])
        for _, _, label in B0_ERROR_STRATA_KM
    }

    # Source-equal aggregation (each unique source_id weighted equally regardless of frame count)
    source_buckets: dict[str, list[RecordEvalRow]] = {}
    for r in rows:
        source_buckets.setdefault(r.source_id, []).append(r)
    source_means_pos = np.array(
        [
            np.mean([x.pos_rmse_km if x.is_finite else 10000.0 for x in grp])
            for grp in source_buckets.values()
        ],
        dtype=np.float64,
    )
    source_means_vel = np.array(
        [
            np.mean([x.vel_rmse_mps if x.is_finite else 10000.0 for x in grp])
            for grp in source_buckets.values()
        ],
        dtype=np.float64,
    )

    return {
        "overall": _agg(rows),
        "by_cadence": by_cadence,
        "by_b0_stratum": by_stratum,
        "source_equal_summary": {
            "unique_sources": len(source_buckets),
            "position_km": _distribution_summary(source_means_pos, cap_for_diverged=10000.0),
            "velocity_mps": _distribution_summary(source_means_vel, cap_for_diverged=10000.0),
        },
    }


def cluster_paired_bootstrap_ci(
    candidate_rows: Sequence[RecordEvalRow],
    baseline_rows: Sequence[RecordEvalRow],
    *,
    metric: str = "pos_rmse_km",
    mode: str = "difference",  # "difference" (cand - base) or "ratio" (cand / base)
    n_boot: int = 2000,
    seed: int = 20261028,
) -> dict[str, float]:
    """Cluster-paired bootstrap over independent source_id units (JOINT_RESEARCH_PLAN §4.5)."""
    cand_map = {r.record_id: r for r in candidate_rows}
    base_map = {r.record_id: r for r in baseline_rows}
    shared_ids = sorted(set(cand_map).intersection(base_map))
    if not shared_ids:
        raise ValueError("No shared record_ids between candidate and baseline")

    clusters: dict[str, list[tuple[float, float]]] = {}
    for rid in shared_ids:
        rc, rb = cand_map[rid], base_map[rid]
        vc = float(getattr(rc, metric)) if rc.is_finite and getattr(rc, metric) is not None else 10000.0
        vb = float(getattr(rb, metric)) if rb.is_finite and getattr(rb, metric) is not None else 10000.0
        clusters.setdefault(rc.source_id, []).append((vc, vb))

    source_keys = sorted(clusters)
    n_sources = len(source_keys)
    cluster_sums = np.array(
        [[sum(p[0] for p in clusters[s]), sum(p[1] for p in clusters[s]), len(clusters[s])] for s in source_keys],
        dtype=np.float64,
    )

    def _stat(arr: np.ndarray) -> float:
        tot_c, tot_b, tot_n = arr[:, 0].sum(), arr[:, 1].sum(), max(1.0, arr[:, 2].sum())
        mc, mb = tot_c / tot_n, tot_b / tot_n
        if mode == "ratio":
            return float(mc / max(mb, 1.0e-12))
        return float(mc - mb)

    point = _stat(cluster_sums)
    rng = np.random.default_rng(seed)
    samples = np.empty(int(n_boot), dtype=np.float64)
    for b in range(int(n_boot)):
        idx = rng.integers(0, n_sources, size=n_sources)
        samples[b] = _stat(cluster_sums[idx])

    return {
        "point_estimate": point,
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "one_sided_95_upper": float(np.quantile(samples, 0.95)),
        "num_shared_records": len(shared_ids),
        "num_source_clusters": n_sources,
    }


@torch.no_grad()
def evaluate_v2_primary_endpoint_forecast300(
    published_state_at_b2_t30_si: Tensor,
    truth_state_at_b2_t30_si: Tensor,
    *,
    horizon_sec: float = 300.0,
) -> Tensor:
    """Propagate both published state and truth at Block 2 + 30 s purely two-body for 300 s (km)."""
    pred_f, _ = propagate_si_with_stm(
        published_state_at_b2_t30_si, horizon_sec, compute_stm=False
    )
    true_f, _ = propagate_si_with_stm(
        truth_state_at_b2_t30_si, horizon_sec, compute_stm=False
    )
    return torch.linalg.vector_norm(pred_f[..., :3] - true_f[..., :3], dim=-1) / 1000.0


@torch.no_grad()
def measure_secant_amplification(
    rollout_fn: Callable[[Tensor], Tensor],
    base_initial_state_si: Tensor,
    perturbation_scaled: Tensor,
) -> float:
    """Measure empirical closed-loop secant amplification ||F(z + delta) - F(z)|| / ||delta|| in D-scale."""
    scale = torch.tensor(STATE_SCALE_VALUES, device=base_initial_state_si.device, dtype=torch.float64)
    delta_norm = float(torch.linalg.vector_norm(perturbation_scaled).item())
    if delta_norm <= 1.0e-15:
        raise ValueError("perturbation_scaled must be nonzero")
    out_base = rollout_fn(base_initial_state_si)
    out_pert = rollout_fn(base_initial_state_si + perturbation_scaled * scale)
    diff_scaled = (out_pert - out_base) / scale
    return float(torch.linalg.vector_norm(diff_scaled).item() / delta_norm)


__all__ = [
    "RecordEvalRow",
    "cluster_paired_bootstrap_ci",
    "compute_los_local_errors",
    "directional_benefit_metrics",
    "evaluate_v2_primary_endpoint_forecast300",
    "measure_secant_amplification",
    "summarize_record_rows",
]

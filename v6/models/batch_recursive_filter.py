"""Batch-parallel causal V6 recursion; only the time axis is iterated."""
from __future__ import annotations

from typing import Any
import importlib
import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .batch_time import BatchTimePlan, prepare_batch_time_plan
from .recursive_filter import (
    CausalHistoryCache, RecursiveFilterOutput, TrainingFilterOutput,
    _as_bt, _physical_functions,
)
from .geometry_relation import (
    build_causal_geometry_chunk,
    build_current_pairwise_geometry,
)
from .scalar_vector_attention import RelationConditionedScalarVectorAttention
from .batched_physics import propagate_scaled_batch


def _checkpointed_rk4_interval(
    state_scaled: Tensor,
    dt_sec: Tensor,
    n_substeps: Tensor,
    valid_mask: Tensor,
    max_n_substeps: int,
) -> Tensor:
    """Pure, fixed-plan checkpoint region for one batched RK4 interval."""
    return propagate_scaled_batch(
        state_scaled, dt_sec, n_substeps, max_n_substeps,
        valid_mask=valid_mask,
    )


def forward_filter_sequence(
    model: Any, *,
    initial_state_si: Any,
    init_time_sec: Any,
    times_sec: Any,
    observed_los: Any,
    station_states_si: Any = None,
    station_state_si: Any = None,
    station_ids: Any = None,
    station_id_reliable: Any = None,
    valid_mask: Any = None,
    gain_override: Any = None,
    return_attention: bool = False,
    return_logits: bool = False,
    return_relations: bool = False,
    return_gate_diagnostics: bool = False,
    attention_mass_callback: Any = None,
    output_mode: str = "full",
    mode: str = "train_fast",
    time_plan: BatchTimePlan | None = None,
) -> RecursiveFilterOutput | TrainingFilterOutput:
    """Run all batch members concurrently at each observation index.

    Epochs and validity are CPU metadata in a precomputed time_plan. No tensor
    value is read by Python inside the fast per-frame loop. Debug mode adds
    per-frame finite/geometry checks with host synchronization.
    """
    from .batch_feature_builder import build_batch_features
    from .geometry_filter import (
        apply_scaled_state_update,
        apply_scaled_state_correction,
        build_online_no_historical_state_mask,
    )

    if mode not in ("train_fast", "debug"):
        raise ValueError("mode must be 'train_fast' or 'debug'")
    if output_mode not in ("full", "training_minimal"):
        raise ValueError("output_mode must be 'full' or 'training_minimal'")
    update_mode = str(getattr(model, "update_mode", "gain"))
    if update_mode not in ("gain", "direct_correction"):
        raise ValueError("model.update_mode must be 'gain' or 'direct_correction'")
    direct_correction = update_mode == "direct_correction"
    if direct_correction and gain_override is not None:
        raise ValueError("gain_override is incompatible with update_mode='direct_correction'")
    if output_mode == "training_minimal" and (
        return_logits or return_relations
        or (return_attention and attention_mass_callback is None)
    ):
        raise ValueError("training_minimal cannot retain attention or relation diagnostics")
    if attention_mass_callback is not None and not callable(attention_mass_callback):
        raise TypeError("attention_mass_callback must be callable")
    if return_gate_diagnostics:
        if mode != "debug" or output_mode != "full":
            raise ValueError(
                "return_gate_diagnostics requires mode='debug' and output_mode='full'"
            )
        attention_is_v7a = (
            getattr(model, "attention_fusion", None) == "dual_softmax_gated"
            or getattr(getattr(model, "attention", None), "aggregation_mode", None)
            == "dual_softmax_gated"
        )
        if not attention_is_v7a:
            raise ValueError(
                "return_gate_diagnostics requires attention_fusion='dual_softmax_gated'"
            )
    if station_states_si is None:
        station_states_si = station_state_si
    if station_states_si is None:
        raise ValueError("station_states_si must supply position and velocity")
    supplied_plan = time_plan is not None
    if time_plan is None:
        time_plan = prepare_batch_time_plan(times_sec, init_time_sec, valid_mask)
    if not isinstance(time_plan, BatchTimePlan):
        raise TypeError("time_plan must be a BatchTimePlan")
    raw_times = torch.as_tensor(times_sec, dtype=torch.float64)
    if raw_times.device.type != "cpu":
        raise ValueError("times_sec must remain on CPU as batch time metadata")
    unbatched = raw_times.ndim == 1
    batch, steps = time_plan.times_sec.shape
    if supplied_plan:
        raw_times = raw_times.to(torch.float64)
        if raw_times.ndim == 1:
            raw_times = raw_times.unsqueeze(0)
        if tuple(raw_times.shape) != (batch, steps):
            raise ValueError("time_plan and times_sec shapes disagree")
        same_times = torch.where(time_plan.valid_mask, raw_times, time_plan.times_sec)
        if not torch.equal(same_times, time_plan.times_sec):
            raise ValueError("time_plan does not match valid times_sec")
        if isinstance(init_time_sec, Tensor) and init_time_sec.device.type != "cpu":
            raise ValueError("init_time_sec must remain on CPU as batch time metadata")
        raw_init = torch.as_tensor(init_time_sec, dtype=torch.float64, device="cpu").reshape(-1)
        if raw_init.numel() == 1:
            raw_init = raw_init.expand(batch)
        if not torch.equal(raw_init, time_plan.init_times_sec):
            raise ValueError("time_plan does not match init_time_sec")
        if valid_mask is not None:
            if isinstance(valid_mask, Tensor) and valid_mask.device.type != "cpu":
                raise ValueError("valid_mask must remain on CPU as batch time metadata")
            raw_valid = torch.as_tensor(valid_mask, dtype=torch.bool, device="cpu")
            if raw_valid.ndim == 0:
                raw_valid = raw_valid.expand(batch, steps)
            elif raw_valid.ndim == 1:
                raw_valid = raw_valid.unsqueeze(0).expand(batch, steps)
            if not torch.equal(raw_valid, time_plan.valid_mask):
                raise ValueError("time_plan does not match valid_mask")
    device = next(model.parameters()).device
    initial = torch.as_tensor(initial_state_si, device=device, dtype=torch.float64)
    if initial.ndim == 1:
        initial = initial.unsqueeze(0)
    if tuple(initial.shape) != (batch, 6):
        raise ValueError("initial_state_si must have shape [6] or [B,6]")
    observed = _as_bt(observed_los, batch, steps, (3,), device, torch.float64, "observed_los")
    stations = _as_bt(station_states_si, batch, steps, (6,), device, torch.float64, "station_states_si")
    if station_ids is None:
        if station_id_reliable is not None:
            raise ValueError("station_id_reliable requires station_ids")
        ids = torch.zeros((batch, steps), device=device, dtype=torch.int64)
    else:
        ids = _as_bt(station_ids, batch, steps, (), device, torch.int64, "station_ids")
    reliable = (torch.zeros((batch, steps), device=device, dtype=torch.bool)
                if station_id_reliable is None else
                _as_bt(station_id_reliable, batch, steps, (), device, torch.bool, "station_id_reliable"))
    times = time_plan.times_sec.to(device=device)
    init_times = time_plan.init_times_sec.to(device=device)
    valid = time_plan.valid_mask.to(device=device)
    intervals = time_plan.delta_sec.to(device=device)
    counts = time_plan.substeps.to(device=device)
    model_dtype = model.attention.q_scalar.weight.dtype
    state_scale, _, predict_los = _physical_functions()
    scale = state_scale(device, torch.float64)
    safe_observed = torch.where(valid.unsqueeze(-1), observed, torch.zeros_like(observed))
    safe_station = torch.where(valid.unsqueeze(-1), stations, torch.zeros_like(stations))
    safe_ids = torch.where(valid, ids, torch.zeros_like(ids))
    safe_reliable = reliable & valid
    previous_prior = initial
    previous_posterior = initial
    previous_obs = torch.zeros((batch, 3), device=device, dtype=torch.float64)
    previous_station = torch.zeros((batch, 6), device=device, dtype=torch.float64)
    previous_ids = torch.zeros(batch, device=device, dtype=torch.int64)
    previous_reliable = torch.zeros(batch, device=device, dtype=torch.bool)
    has_previous = torch.zeros(batch, device=device, dtype=torch.bool)
    kv = None
    hist = None
    if output_mode == "full":
        hist = CausalHistoryCache(
            kv=None, scalar_tokens=[], vector_tokens=[], times_sec=[],
            station_states_si=[], observed_los=[], station_ids=[],
            station_id_reliable=[], valid=[], input_frame_indices=[],
        )
    update_slot_names = (("correction_scaled", "correction_si") if direct_correction
                         else ("gain",))
    if output_mode == "training_minimal":
        slot_names = ("prior", "posterior", *update_slot_names, "innovation", "predicted")
    else:
        slot_names = (
            "prior", "posterior", *update_slot_names, "innovation", "predicted",
            "scalar", "vector", "attn_s", "attn_v", "direct_s", "direct_v",
            "feature", "weights", "logits", "relation",
            *(("frame_correction_scaled", "frame_correction_si") if direct_correction
              else ("frame_gain",)),
            "frame_post",
        )
        if return_gate_diagnostics:
            slot_names += ("gate_state", "gate_obs")
    slots: dict[str, list[Any]] = {name: [] for name in slot_names}
    offset = torch.tensor((1.0, 0.0, 0.0), device=device, dtype=torch.float64).expand(batch, 3)
    override = None
    if gain_override is not None and not callable(gain_override):
        override = torch.as_tensor(gain_override, device=device)
        if not override.is_floating_point():
            override = override.to(model_dtype)
        if override.ndim == 2 and tuple(override.shape) == (6, 3):
            pass
        elif override.ndim == 3 and tuple(override.shape) == (steps, 6, 3):
            pass
        elif override.ndim == 4 and tuple(override.shape) == (batch, steps, 6, 3):
            pass
        else:
            raise ValueError("gain_override must be [6,3], [T,6,3], or [B,T,6,3]")

    reuse_sanitized_kv = (
        type(model.attention) is RelationConditionedScalarVectorAttention
        and bool(getattr(model, "no_historical_state_keys", False))
        and model.attention_impl != "legacy_concat"
        and output_mode == "training_minimal"
        and mode == "train_fast"
        and attention_mass_callback is None
    )
    # Keep the mask variant's physical two-token KV cache and full logical
    # attention mask, but precompute fixed metadata geometry outside the
    # checkpoint closure.  This private candidate is restricted to the same
    # formal training contract used by the mask optimization.  Differentiable
    # or non-finite metadata stays on the original geometry-input path.
    static_geometry = (
        type(model.attention) is RelationConditionedScalarVectorAttention
        and bool(getattr(model, "no_historical_state_keys", False))
        and model.attention_impl in ("blockwise_checkpoint", "fused_tiles", "fused_tiles_save")
        and model.attention.aggregation_mode == "unified"
        and model.attention.dropout.p == 0.0
        and mode == "train_fast"
        and output_mode == "training_minimal"
        and attention_mass_callback is None
    )
    static_geometry_chunk_frames = 16
    if static_geometry:
        static_geometry = not any(
            tensor.requires_grad for tensor in (times, safe_station, safe_observed)
        )
    if static_geometry:
        static_geometry = bool(
            (
                torch.isfinite(safe_station).all()
                & torch.isfinite(safe_observed).all()
            )
        )
    static_relation_chunk: Tensor | None = None
    static_valid_chunk: Tensor | None = None
    static_chunk_start = 0
    static_chunk_stop = 0
    for k in range(steps):
        active = valid[:, k]
        dt = intervals[:, k]
        if static_geometry and k == static_chunk_stop:
            static_chunk_start = k
            static_chunk_stop = min(steps, k + static_geometry_chunk_frames)
            static_relation_chunk, static_valid_chunk = build_causal_geometry_chunk(
                times,
                safe_station,
                safe_observed,
                safe_ids,
                safe_reliable,
                valid,
                model.normalization,
                query_start=static_chunk_start,
                query_stop=static_chunk_stop,
            )
        if bool(getattr(model, "checkpoint_rk4", False)) and torch.is_grad_enabled():
            prior_z = checkpoint(
                _checkpointed_rk4_interval,
                previous_posterior / scale, dt, counts[:, k], active,
                time_plan.max_substeps[k],
                use_reentrant=False, preserve_rng_state=True,
            )
        else:
            prior_z = propagate_scaled_batch(
                previous_posterior / scale, dt, counts[:, k],
                time_plan.max_substeps[k], valid_mask=active,
            )
        prior = prior_z * scale
        # Inactive rows use a finite, nonzero candidate relative position.
        station = torch.where(
            active[:, None], safe_station[:, k],
            torch.cat((prior[:, :3] - offset, torch.zeros_like(prior[:, 3:])), dim=-1),
        )
        obs = torch.where(active[:, None], safe_observed[:, k], offset)
        predicted = predict_los(prior_z, station[:, :3])
        innovation = obs - predicted
        current_ids = safe_ids[:, k]
        current_reliable = safe_reliable[:, k]
        feature = build_batch_features(
            model.tokenizer.feature_builder,
            prior_state_si=prior,
            previous_prior_state_si=previous_prior,
            previous_posterior_state_si=previous_posterior,
            observed_los=obs,
            previous_observed_los=previous_obs,
            predicted_los=predicted,
            station_state_si=station,
            previous_station_state_si=previous_station,
            station_id=current_ids,
            previous_station_id=previous_ids,
            station_id_reliable=current_reliable,
            previous_station_id_reliable=previous_reliable,
            time_sec=times[:, k],
            init_time_sec=init_times,
            delta_time_seconds=dt,
            active=active,
            has_previous=has_previous,
        )
        structured_s, structured_v, direct_scalar, direct_vector, _ = (
            model.tokenizer.encode_geometry_features(feature)
        )
        s = structured_s.reshape(batch, 2, model.scalar_channels)
        v = structured_v.reshape(batch, 2, model.vector_channels, 3)
        direct_s = direct_scalar.reshape(batch, 1, model.scalar_channels)
        direct_v = direct_vector.reshape(batch, 1, model.vector_channels, 3)
        if model.attention_impl == "legacy_concat":
            relation, pair_valid = build_current_pairwise_geometry(
                current_time_sec=times[:, k], history_times_sec=times[:, :k+1],
                current_station_state_si=station, history_station_state_si=safe_station[:, :k+1],
                current_observed_los=obs, history_observed_los=safe_observed[:, :k+1],
                normalization=model.normalization, current_station_id=current_ids,
                history_station_ids=safe_ids[:, :k+1],
                current_station_id_reliable=current_reliable,
                history_station_id_reliable=safe_reliable[:, :k+1],
                current_valid=active, history_valid=valid[:, :k+1],
            )
            allowed = pair_valid.repeat_interleave(2, dim=1).unsqueeze(1).expand(batch, 2, -1)
            geometry_inputs = None
        elif static_geometry:
            assert static_relation_chunk is not None and static_valid_chunk is not None
            local_query = k - static_chunk_start
            relation = static_relation_chunk[:, local_query, : k + 1]
            pair_valid = static_valid_chunk[:, local_query, : k + 1]
            geometry_inputs = None
            geometry_allowed = (
                pair_valid.repeat_interleave(2, dim=1)
                .unsqueeze(1)
                .expand(batch, 2, -1)
            )
            allowed = build_online_no_historical_state_mask(
                k, batch_size=batch, device=device
            ) & geometry_allowed
        else:
            relation = None
            allowed = None
            geometry_inputs = {
                "current_time_sec": times[:, k],
                "history_times_sec": times[:, :k+1],
                "current_station_state_si": safe_station[:, k],
                "history_station_state_si": safe_station[:, :k+1],
                "current_observed_los": safe_observed[:, k],
                "history_observed_los": safe_observed[:, :k+1],
                "current_station_id": current_ids,
                "history_station_ids": safe_ids[:, :k+1],
                "current_station_id_reliable": current_reliable,
                "history_station_id_reliable": safe_reliable[:, :k+1],
                "current_valid": active,
                "history_valid": valid[:, :k+1],
                "normalization": model.normalization,
            }
        if bool(getattr(model, "no_historical_state_keys", False)) and not static_geometry:
            state_key_allowed = build_online_no_historical_state_mask(
                k, batch_size=batch, device=device
            )
            allowed = state_key_allowed if allowed is None else (allowed & state_key_allowed)
        attention_kwargs = {
            "attention_mask": allowed,
            "current_valid_mask": active[:, None].expand(batch, 2),
            "return_attention": return_attention or attention_mass_callback is not None,
            "return_logits": return_logits,
            "mode": mode,
            "attention_impl": model.attention_impl,
            "block_frames": model.block_frames,
            "current_geometry_inputs": geometry_inputs,
        }
        if reuse_sanitized_kv:
            attention_kwargs["_reuse_sanitized_kv"] = True
        frame_diag: dict[str, Any] | None = {} if return_gate_diagnostics else None
        if return_gate_diagnostics:
            attention_kwargs["gate_diagnostics"] = frame_diag
        attention = model.attention.forward_current(
            s, v, kv, relation,
            (direct_s, direct_v) if model.use_direct_attention_values else None,
            **attention_kwargs,
        )
        if attention_mass_callback is not None:
            if attention.weights is None:
                raise RuntimeError("attention module did not return requested weights")
            attention_mass_callback(
                frame_index=k,
                weights=attention.weights,
                current_time_sec=time_plan.times_sec[:, k],
                init_time_sec=time_plan.init_times_sec,
                history_valid=time_plan.valid_mask[:, :k + 1],
            )
        kv = attention.cache
        sa, va = attention.output
        bypass_s = direct_s[:, 0] if model.use_direct_gain_bypass else torch.zeros_like(direct_s[:, 0])
        bypass_v = direct_v[:, 0] if model.use_direct_gain_bypass else torch.zeros_like(direct_v[:, 0])
        readout = torch.cat((
            sa[:, 0], sa[:, 1], va[:, 0].flatten(start_dim=-2),
            va[:, 1].flatten(start_dim=-2), bypass_s, bypass_v.flatten(start_dim=-2),
        ), dim=-1)
        gain = None
        correction_scaled = None
        correction_si = None
        frame_gain = None
        frame_correction_scaled = None
        frame_correction_si = None
        if direct_correction:
            if model.correction_head is None:
                raise RuntimeError("direct_correction requires model.correction_head")
            parameter = next(model.correction_head.parameters(), None)
            head_input = readout if parameter is None else readout.to(
                dtype=parameter.dtype, device=parameter.device
            )
            correction_scaled = model.correction_head(head_input)
            if tuple(correction_scaled.shape) != (batch, 6):
                raise ValueError("correction_head must return [B,6]")
            correction_scaled = torch.where(
                active[:, None], correction_scaled, torch.zeros_like(correction_scaled)
            )
            correction_si = correction_scaled.to(dtype=prior.dtype, device=device) * scale
            posterior = apply_scaled_state_correction(prior, correction_scaled)
            frame_correction_scaled = correction_scaled[0] if unbatched else correction_scaled
            frame_correction_si = correction_si[0] if unbatched else correction_si
        else:
            if callable(gain_override):
                gain = gain_override(frame_index=k, representation=readout)
            elif override is not None:
                if override.ndim == 2:
                    gain = override.unsqueeze(0).expand(batch, 6, 3)
                elif override.ndim == 3:
                    gain = override[k].unsqueeze(0).expand(batch, 6, 3)
                else:
                    gain = override[:, k]
            elif model.enable_gain:
                parameter = next(model.gain_head.parameters(), None)
                head_input = readout if parameter is None else readout.to(dtype=parameter.dtype, device=parameter.device)
                gain = model.gain_head(head_input)
            else:
                gain = readout.new_zeros((batch, 6, 3))
            if tuple(gain.shape) != (batch, 6, 3):
                raise ValueError("gain_head must return [B,6,3]")
            gain = torch.where(active[:, None, None], gain, torch.zeros_like(gain))
            frame_gain = gain[0] if unbatched else gain
            gain_for_update = frame_gain.unsqueeze(0) if unbatched else frame_gain
            posterior = apply_scaled_state_update(
                prior, gain_for_update.to(dtype=prior.dtype, device=device), innovation
            )
        frame_post = posterior[0] if unbatched else posterior
        posterior_for_feedback = frame_post.unsqueeze(0) if unbatched else frame_post
        if mode == "debug":
            update_finite = (torch.isfinite(correction_scaled).all(dim=-1)
                             if direct_correction else
                             torch.isfinite(gain).all(dim=(-1, -2)))
            okay = (torch.isfinite(prior).all(dim=-1) & torch.isfinite(predicted).all(dim=-1)
                    & update_finite & torch.isfinite(posterior).all(dim=-1)
                    & (torch.linalg.vector_norm(prior[:, :3]-station[:, :3],dim=-1) > 0))
            if bool((active & ~okay).any()):
                from .recursive_filter import FilterDivergenceError
                bad = int(torch.nonzero(active & ~okay, as_tuple=False)[0, 0])
                details = {"stage": "batch_step", "batch_index": bad, "frame_index": k,
                           "time_sec": float(times[bad, k]),
                           "prior": prior[bad].detach().cpu().tolist(),
                           "innovation": innovation[bad].detach().cpu().tolist(),
                           "posterior": posterior[bad].detach().cpu().tolist()}
                if direct_correction:
                    details["correction_scaled"] = correction_scaled[bad].detach().cpu().tolist()
                else:
                    details["gain"] = gain[bad].detach().cpu().tolist()
                raise FilterDivergenceError(details)
        mask = active[:, None]
        mask_v = active[:, None, None]
        slots["prior"].append(torch.where(mask, prior, torch.zeros_like(prior)))
        slots["posterior"].append(torch.where(mask, posterior_for_feedback, torch.zeros_like(posterior)))
        if direct_correction:
            slots["correction_scaled"].append(correction_scaled)
            slots["correction_si"].append(correction_si)
        else:
            slots["gain"].append(gain)
        slots["innovation"].append(torch.where(mask, innovation, torch.zeros_like(innovation)))
        slots["predicted"].append(torch.where(mask, predicted, torch.zeros_like(predicted)))
        if output_mode == "full":
            slots["scalar"].append(s)
            slots["vector"].append(v)
            slots["attn_s"].append(sa)
            slots["attn_v"].append(va)
            slots["direct_s"].append(direct_s[:, 0])
            slots["direct_v"].append(direct_v[:, 0])
            slots["feature"].append(feature)
            slots["weights"].append(attention.weights)
            slots["logits"].append(attention.logits)
            relation_diagnostic = relation
            if return_relations and relation_diagnostic is None:
                relation_diagnostic, _ = build_current_pairwise_geometry(
                    current_time_sec=times[:, k], history_times_sec=times[:, :k+1],
                    current_station_state_si=station, history_station_state_si=safe_station[:, :k+1],
                    current_observed_los=obs, history_observed_los=safe_observed[:, :k+1],
                    normalization=model.normalization, current_station_id=current_ids,
                    history_station_ids=safe_ids[:, :k+1],
                    current_station_id_reliable=current_reliable,
                    history_station_id_reliable=safe_reliable[:, :k+1],
                    current_valid=active, history_valid=valid[:, :k+1],
                )
            slots["relation"].append(relation_diagnostic if return_relations else None)
            if direct_correction:
                slots["frame_correction_scaled"].append(frame_correction_scaled)
                slots["frame_correction_si"].append(frame_correction_si)
            else:
                slots["frame_gain"].append(frame_gain)
            slots["frame_post"].append(frame_post)
            if return_gate_diagnostics:
                assert frame_diag is not None
                try:
                    gamma_state = frame_diag["gamma_state"]
                    gamma_obs = frame_diag["gamma_obs"]
                except KeyError as exc:
                    raise RuntimeError(
                        "V7-A attention did not populate gate_diagnostics"
                    ) from exc
                if gamma_state.ndim != 3 or gamma_state.shape[0] != batch or gamma_state.shape[-1] != 2:
                    raise ValueError(
                        "gate_diagnostics gamma_state must have shape [B,H,2]"
                    )
                if gamma_obs.shape != gamma_state.shape:
                    raise ValueError(
                        "gate_diagnostics gamma tensors must have matching [B,H,2] shapes"
                    )
                slots["gate_state"].append(gamma_state)
                slots["gate_obs"].append(gamma_obs)
        previous_prior = torch.where(mask, prior, previous_prior)
        previous_posterior = torch.where(mask, posterior_for_feedback, previous_posterior)
        previous_obs = torch.where(mask, obs, previous_obs)
        previous_station = torch.where(mask, station, previous_station)
        previous_ids = torch.where(active, current_ids, previous_ids)
        previous_reliable = torch.where(active, current_reliable, previous_reliable)
        has_previous = has_previous | active
        if hist is not None:
            hist.scalar_tokens.append(s)
            hist.vector_tokens.append(v)
            hist.times_sec.append(times[:, k])
            hist.station_states_si.append(station)
            hist.observed_los.append(obs)
            hist.station_ids.append(current_ids)
            hist.station_id_reliable.append(current_reliable)
            hist.valid.append(active)
            hist.input_frame_indices.append(k)
    if hist is not None:
        hist.kv = kv

    def pack(name: str) -> Tensor:
        return torch.stack(slots[name], dim=1)
    values = {name: pack(name) for name in slot_names[:2]}
    values.update({name: pack(name) for name in update_slot_names})
    values.update({name: pack(name) for name in ("innovation", "predicted")})
    if output_mode == "full":
        values.update({name: pack(name) for name in (
            "scalar", "vector", "attn_s", "attn_v", "direct_s", "direct_v",
        )})
        for name in ("scalar", "attn_s"):
            values[name] = values[name].reshape(batch, 2*steps, model.scalar_channels)
        for name in ("vector", "attn_v"):
            values[name] = values[name].reshape(batch, 2*steps, model.vector_channels, 3)
    history_lengths = torch.where(valid, valid.to(torch.long).cumsum(dim=1), torch.zeros_like(valid, dtype=torch.long))
    if unbatched:
        values = {name: value.squeeze(0) for name, value in values.items()}
        valid_out = valid.squeeze(0)
        history_lengths = history_lengths.squeeze(0)
    else:
        valid_out = valid
    metadata = {"model_version":model.model_version, "recursive_filter":True,
                "batch_parallel":True, "online_kv_cache":True, "cache_detached":False,
                "history_truncated":False, "mode":mode, "output_mode":output_mode,
                "attention_impl":model.attention_impl, "block_frames":model.block_frames,
                "attention_fusion":model.attention_fusion,
                "attention_weight_semantics":model.attention_weight_semantics,
                "model_config":model.model_config,
                "update_mode": update_mode,
                "state_update": (
                    "z_post = z_prior + K @ raw_innovation" if not direct_correction
                    else "z_post = z_prior + delta_scaled"
                )}
    if return_gate_diagnostics:
        metadata["gate_diagnostics"] = {
            "gamma_state": torch.stack(slots["gate_state"], dim=1),
            "gamma_obs": torch.stack(slots["gate_obs"], dim=1),
            "valid_mask": valid.detach().clone(),
            "query_order": ("state", "observation"),
        }
    if output_mode == "training_minimal":
        return TrainingFilterOutput(
            prior_states_si=values["prior"], posterior_states_si=values["posterior"],
            gains=None if direct_correction else values["gain"],
            correction_scaled=values.get("correction_scaled"),
            correction_si=values.get("correction_si"),
            innovations=values["innovation"],
            predicted_los=values["predicted"], valid_mask=valid_out, metadata=metadata,
        )
    return RecursiveFilterOutput(
        prior_states_si=values["prior"], posterior_states_si=values["posterior"],
        gains=None if direct_correction else values.get("gain"),
        correction_scaled=values.get("correction_scaled"),
        correction_si=values.get("correction_si"),
        innovations=values["innovation"], predicted_los=values["predicted"],
        valid_mask=valid_out, scalar_tokens=values["scalar"], vector_tokens=values["vector"],
        scalar_attention=values["attn_s"], vector_attention=values["attn_v"],
        los_direct_scalar_tokens=values["direct_s"], los_direct_vector_tokens=values["direct_v"],
        history_lengths=history_lengths, history_cache=hist,
        frame_features=tuple(slots["feature"]),
        frame_gains=None if direct_correction else tuple(slots["frame_gain"]),
        frame_posteriors=tuple(slots["frame_post"]),
        attention_weights=tuple(slots["weights"]) if return_attention else None,
        attention_logits=tuple(slots["logits"]) if return_logits else None,
        relation_rows=tuple(slots["relation"]) if return_relations else None,
        metadata=metadata,
    )

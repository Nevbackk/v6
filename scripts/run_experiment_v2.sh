#!/usr/bin/env bash
# ==============================================================================
# Complete Execution Pipeline — Model Plan V2 (ASD-SC Progressive Denoising)
# ==============================================================================
# Implements the full Round 4 experimental matrix & primary endpoint protocol in
# docs/MODEL_V2_ADAPTIVE_DENOISING_PLAN.md and ROUND4_NOISY_INIT_SELFCORRECT_REVIEW.md:
#   - Stage 0: Local CPU mathematical & structural unit checks
#   - Stage 1: Main ASD-SC Full Progressive Model across 3 pre-registered seeds
#   - Stage 2 (V2-Abl-1): Prior + LOS Synergy (obs_only vs state_only vs no_cf vs full)
#   - Stage 3 (V2-Abl-2): Blind Scale Trust Gate (fixed_g vs unconstrained_mlp6d vs no_quantile_aux)
#   - Stage 4 (V2-Abl-3): 2x2 {Static B0, DAgger Replay} x {1-Step Grad, 2-Step Unroll}
#   - Stage 5 (V2-Abl-4): Multi-Horizon {0s, +60s, +300s} & Progressive Step-Increment T_l d_l
#   - Stage 6 (V2-Abl-5): Cross-Gap QR Pending Memory V_p/K_p/E_a & Tier-2 A/B/C Re-anchoring
#   - Stage 7: Locked Paired Replication on historical VAL (Block 2 +30s -> +300s endpoint)
#
# Usage (Linux / AutoDL CUDA Server):
#   PREPARED_ROOT=/path/to/prepared PROTOCOL_DIR=/path/to/protocol \
#     bash scripts/run_experiment_v2.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -d "${REPO_ROOT}/v6/v6" ]]; then
  V6_DIR="${REPO_ROOT}/v6"
else
  V6_DIR="${REPO_ROOT}"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/runs/exp_v2}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-0.001}"
SEEDS_STR="${SEEDS:-20261028 20261029 20261030}"
STAGE="${STAGE:-all}"
MAX_RECORDS_PER_EPOCH="${MAX_RECORDS_PER_EPOCH:-0}"
MAX_EVAL_RECORDS="${MAX_EVAL_RECORDS:-0}"
RUN_PAIRED_VAL="${RUN_PAIRED_VAL:-0}"

read -r -a SEEDS_ARR <<< "${SEEDS_STR}"
case "${STAGE}" in
  all|test|main|abl1_synergy|abl2_gate|abl3_replay_unroll|abl4_horizon_prog|abl5_memory_reanchor|paired_val) ;;
  *) echo "Unknown stage: ${STAGE}" >&2; exit 2 ;;
esac
ABL_SEED="${ABL_SEED:-${SEEDS_ARR[0]}}"

echo "======================================================================"
echo "[Model Plan V2: ASD-SC] Starting Stage='${STAGE}' | Device='${DEVICE}'"
echo "Codebase: ${V6_DIR}"
echo "Output  : ${OUTPUT_ROOT}"
echo "======================================================================"

# ------------------------------------------------------------------------------
# Stage 0: CPU Mathematical & Orthogonality Unit Checks
# ------------------------------------------------------------------------------
if [[ "${STAGE}" == "all" || "${STAGE}" == "test" ]]; then
  echo "[V2][Stage 0] Running V2 QR, E_a restricted writeback, and progressive target unit tests..."
  (cd "${V6_DIR}" && "${PYTHON_BIN}" -s -m unittest discover -s tests -v)
  if [[ "${STAGE}" == "test" ]]; then
    exit 0
  fi
fi

: "${PREPARED_ROOT:?ERROR: Must set PREPARED_ROOT=/path/to/prepared}"
: "${PROTOCOL_DIR:?ERROR: Must set PROTOCOL_DIR=/path/to/protocol}"

COMMON_ARGS=(
  --prepared-root "${PREPARED_ROOT}"
  --protocol-dir "${PROTOCOL_DIR}"
  --device "${DEVICE}"
  --split-seed "${SPLIT_SEED:-20261028}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --lr "${LR}"
  --max-records-per-epoch "${MAX_RECORDS_PER_EPOCH}"
  --max-eval-records "${MAX_EVAL_RECORDS}"
)

# ------------------------------------------------------------------------------
# Stage 1: Full ASD-SC Main Model (3 Pre-Registered Seeds)
# ------------------------------------------------------------------------------
if [[ "${STAGE}" == "all" || "${STAGE}" == "main" ]]; then
  for SEED in "${SEEDS_ARR[@]}"; do
    RUN_DIR="${OUTPUT_ROOT}/stage1_asd_sc_full/seed_${SEED}"
    echo "[V2][Stage 1] Training Full ASD-SC | seed=${SEED}..."
    "${PYTHON_BIN}" "${V6_DIR}/scripts/train_v2.py" \
      "${COMMON_ARGS[@]}" \
      --run-dir "${RUN_DIR}" \
      --ablation-mode full \
      --horizons-sec "60,300" \
      --lambda-forecast 1.0 \
      --lambda-prog 0.25 \
      --lambda-good-harm 2.0 \
      --counterfactual-prob 0.25 \
      --seed "${SEED}" \
      --eval-split dev_select
  done
fi

# ------------------------------------------------------------------------------
# Stage 2 (V2-Abl-1): Prior + LOS Synergy & Counterfactual Prior Intervention
# ------------------------------------------------------------------------------
if [[ "${STAGE}" == "all" || "${STAGE}" == "abl1_synergy" ]]; then
  for MODE in obs_only state_only; do
    echo "[V2][Abl-1] Training ${MODE} | seed=${ABL_SEED}..."
    "${PYTHON_BIN}" "${V6_DIR}/scripts/train_v2.py" \
      "${COMMON_ARGS[@]}" \
      --run-dir "${OUTPUT_ROOT}/abl1_synergy/${MODE}_seed_${ABL_SEED}" \
      --ablation-mode "${MODE}" \
      --seed "${ABL_SEED}" \
      --eval-split dev_select
  done

  echo "[V2][Abl-1] Training full without counterfactual prior interventions | seed=${ABL_SEED}..."
  "${PYTHON_BIN}" "${V6_DIR}/scripts/train_v2.py" \
    "${COMMON_ARGS[@]}" \
    --run-dir "${OUTPUT_ROOT}/abl1_synergy/no_counterfactual_seed_${ABL_SEED}" \
    --ablation-mode full \
    --counterfactual-prob 0.0 \
    --seed "${ABL_SEED}" \
    --eval-split dev_select
fi

# ------------------------------------------------------------------------------
# Stage 3 (V2-Abl-2): Blind Scale Trust Gate & Quantile Auxiliary Supervision
# ------------------------------------------------------------------------------
if [[ "${STAGE}" == "all" || "${STAGE}" == "abl2_gate" ]]; then
  for MODE in fixed_g unconstrained_mlp6d no_quantile_aux; do
    echo "[V2][Abl-2] Training ${MODE} | seed=${ABL_SEED}..."
    "${PYTHON_BIN}" "${V6_DIR}/scripts/train_v2.py" \
      "${COMMON_ARGS[@]}" \
      --run-dir "${OUTPUT_ROOT}/abl2_gate/${MODE}_seed_${ABL_SEED}" \
      --ablation-mode "${MODE}" \
      --seed "${ABL_SEED}" \
      --eval-split dev_select
  done
fi

# ------------------------------------------------------------------------------
# Stage 4 (V2-Abl-3): 2x2 DAgger Visited-State Replay x Unroll Gradient Length
#    Note: (DAgger + 2-Step) is already trained in Stage 1 (seed=${ABL_SEED});
#   the static-B0 + 1-Step cell is the primary comparator for the Block 2 +30s endpoint!
# ------------------------------------------------------------------------------
if [[ "${STAGE}" == "all" || "${STAGE}" == "abl3_replay_unroll" ]]; then
  for SEED in "${SEEDS_ARR[@]}"; do
    echo "[V2][Abl-3] Training Static-B0 + 1-Step comparator (for Block 2 +30s paired endpoint) | seed=${SEED}..."
    "${PYTHON_BIN}" "${V6_DIR}/scripts/train_v2.py" \
      "${COMMON_ARGS[@]}" \
      --run-dir "${OUTPUT_ROOT}/abl3_replay_unroll/static_1step/seed_${SEED}" \
      --ablation-mode full \
      --no-dagger-replay \
      --single-step-grad \
      --seed "${SEED}" \
      --eval-split dev_select
  done

  echo "[V2][Abl-3] Training Static-B0 + 2-Step | seed=${ABL_SEED}..."
  "${PYTHON_BIN}" "${V6_DIR}/scripts/train_v2.py" \
    "${COMMON_ARGS[@]}" \
    --run-dir "${OUTPUT_ROOT}/abl3_replay_unroll/static_2step_seed_${ABL_SEED}" \
    --ablation-mode full \
    --no-dagger-replay \
    --seed "${ABL_SEED}" \
    --eval-split dev_select

  echo "[V2][Abl-3] Training DAgger + 1-Step | seed=${ABL_SEED}..."
  "${PYTHON_BIN}" "${V6_DIR}/scripts/train_v2.py" \
    "${COMMON_ARGS[@]}" \
    --run-dir "${OUTPUT_ROOT}/abl3_replay_unroll/dagger_1step_seed_${ABL_SEED}" \
    --ablation-mode full \
    --single-step-grad \
    --seed "${ABL_SEED}" \
    --eval-split dev_select
fi

# ------------------------------------------------------------------------------
# Stage 5 (V2-Abl-4): Multi-Horizon Phase Forecast & Progressive Step-Increment T_l d_l
# ------------------------------------------------------------------------------
if [[ "${STAGE}" == "all" || "${STAGE}" == "abl4_horizon_prog" ]]; then
  echo "[V2][Abl-4] Training Current-Only (lambda_forecast=0) | seed=${ABL_SEED}..."
  "${PYTHON_BIN}" "${V6_DIR}/scripts/train_v2.py" \
    "${COMMON_ARGS[@]}" \
    --run-dir "${OUTPUT_ROOT}/abl4_horizon_prog/current_only_seed_${ABL_SEED}" \
    --ablation-mode full \
    --horizons-sec "" \
    --lambda-forecast 0.0 \
    --seed "${ABL_SEED}" \
    --eval-split dev_select

  echo "[V2][Abl-4] Training Current + 60s Only | seed=${ABL_SEED}..."
  "${PYTHON_BIN}" "${V6_DIR}/scripts/train_v2.py" \
    "${COMMON_ARGS[@]}" \
    --run-dir "${OUTPUT_ROOT}/abl4_horizon_prog/horizon_60s_only_seed_${ABL_SEED}" \
    --ablation-mode full \
    --horizons-sec "60" \
    --lambda-forecast 1.0 \
    --seed "${ABL_SEED}" \
    --eval-split dev_select

  echo "[V2][Abl-4] Training without Progressive Step-Increment Supervision (lambda_prog=0) | seed=${ABL_SEED}..."
  "${PYTHON_BIN}" "${V6_DIR}/scripts/train_v2.py" \
    "${COMMON_ARGS[@]}" \
    --run-dir "${OUTPUT_ROOT}/abl4_horizon_prog/no_progressive_aux_seed_${ABL_SEED}" \
    --ablation-mode full \
    --lambda-prog 0.0 \
    --seed "${ABL_SEED}" \
    --eval-split dev_select

  echo "[V2][Abl-4] Training without One-Sided Good-Prior Penalty (lambda_good_harm=0) | seed=${ABL_SEED}..."
  "${PYTHON_BIN}" "${V6_DIR}/scripts/train_v2.py" \
    "${COMMON_ARGS[@]}" \
    --run-dir "${OUTPUT_ROOT}/abl4_horizon_prog/no_good_harm_penalty_seed_${ABL_SEED}" \
    --ablation-mode full \
    --lambda-good-harm 0.0 \
    --seed "${ABL_SEED}" \
    --eval-split dev_select
fi

# ------------------------------------------------------------------------------
# Stage 6 (V2-Abl-5): Cross-Gap QR Pending Memory & Tier-2 A/B/C Re-anchoring
# ------------------------------------------------------------------------------
if [[ "${STAGE}" == "all" || "${STAGE}" == "abl5_memory_reanchor" ]]; then
  echo "[V2][Abl-5] Training Local-Only (no V_p/K_p memory, no Tier-2 re-anchor) | seed=${ABL_SEED}..."
  "${PYTHON_BIN}" "${V6_DIR}/scripts/train_v2.py" \
    "${COMMON_ARGS[@]}" \
    --run-dir "${OUTPUT_ROOT}/abl5_memory_reanchor/local_only_seed_${ABL_SEED}" \
    --ablation-mode no_pending_memory \
    --disable-reanchor \
    --seed "${ABL_SEED}" \
    --eval-split dev_select

  echo "[V2][Abl-5] Training Local + V_p/K_p/E_a Memory (no Tier-2 re-anchor) | seed=${ABL_SEED}..."
  "${PYTHON_BIN}" "${V6_DIR}/scripts/train_v2.py" \
    "${COMMON_ARGS[@]}" \
    --run-dir "${OUTPUT_ROOT}/abl5_memory_reanchor/memory_no_reanchor_seed_${ABL_SEED}" \
    --ablation-mode full \
    --disable-reanchor \
    --seed "${ABL_SEED}" \
    --eval-split dev_select
fi

# ------------------------------------------------------------------------------
# Stage 7: Locked Paired Replication on Historical VAL (Block 2 +30s -> +300s Endpoint)
# ------------------------------------------------------------------------------
if [[ "${RUN_PAIRED_VAL}" == "1" || "${STAGE}" == "paired_val" ]]; then
  echo "[V2][Stage 7] Evaluating locked ASD-SC vs Static-1Step on historical VAL..."
  for SEED in "${SEEDS_ARR[@]}"; do
    FULL_DIR="${OUTPUT_ROOT}/stage1_asd_sc_full/seed_${SEED}"
    [[ -f "${FULL_DIR}/best.pt" ]] || { echo "Missing required checkpoint: ${FULL_DIR}/best.pt" >&2; exit 2; }
    "${PYTHON_BIN}" "${V6_DIR}/scripts/validate_v2.py" \
      --prepared-root "${PREPARED_ROOT}" \
      --protocol-dir "${PROTOCOL_DIR}" \
      --run-dir "${FULL_DIR}" \
      --ablation-mode full \
      --resume "${FULL_DIR}/best.pt" \
      --eval-split paired_val \
      --device "${DEVICE}" --split-seed "${SPLIT_SEED:-20261028}" \
      --max-eval-records "${MAX_EVAL_RECORDS}"

    STAT_DIR="${OUTPUT_ROOT}/abl3_replay_unroll/static_1step/seed_${SEED}"
    [[ -f "${STAT_DIR}/best.pt" ]] || { echo "Missing required checkpoint: ${STAT_DIR}/best.pt" >&2; exit 2; }
    "${PYTHON_BIN}" "${V6_DIR}/scripts/validate_v2.py" \
      --prepared-root "${PREPARED_ROOT}" \
      --protocol-dir "${PROTOCOL_DIR}" \
      --run-dir "${STAT_DIR}" \
      --ablation-mode full \
      --resume "${STAT_DIR}/best.pt" \
      --eval-split paired_val \
      --device "${DEVICE}" --split-seed "${SPLIT_SEED:-20261028}" \
      --max-eval-records "${MAX_EVAL_RECORDS}"
  done
fi

echo "[Model Plan V2: ASD-SC] Pipeline execution finished cleanly."

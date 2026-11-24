#!/usr/bin/env bash
# ==============================================================================
# Complete Execution Pipeline — Round 5 Minimal Genuine Diffusion (VP-OrbitDiff)
# ==============================================================================
# Core knobs (3 default + 1 optional compute budget):
#   K     : artificial reverse diffusion intervals (default 4)
#   P     : independent initial Gaussian seeds / trajectories (default 8)
#   delta : shared floor for V_theta >= delta I and P_carry >= delta P_b^+ (default 0.10)
#   B_los : optional B1/B2 proposal-guidance prefix cap (terminal weights use all observations)
#
# Usage (Linux / AutoDL CUDA Server):
#   PREPARED_ROOT=/path/to/prepared PROTOCOL_DIR=/path/to/protocol \
#     bash scripts/run_experiment_diffusion.sh
#
# Optional environment overrides:
#   OUTPUT_ROOT=runs/exp_diffusion
#   PYTHON_BIN=python
#   DEVICE=cuda
#   EPOCHS=10
#   BATCH_SIZE=32
#   LR=0.001
#   SEEDS="20261028 20261029 20261030"
#   SPLIT_SEED=20261028
#   STAGE=all   (all | test | main | k_curve | abl_guidance | abl_budget_delta | b1 | b2 | paired_val)
#   RUN_PAIRED_VAL=0   (set 1 ONLY after locking model choices on dev_select!)
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
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/runs/exp_diffusion}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-0.001}"
SEEDS_STR="${SEEDS:-20261028 20261029 20261030}"
SPLIT_SEED="${SPLIT_SEED:-20261028}"
STAGE="${STAGE:-all}"
MAX_RECORDS_PER_EPOCH="${MAX_RECORDS_PER_EPOCH:-0}"
MAX_EVAL_RECORDS="${MAX_EVAL_RECORDS:-0}"
RUN_PAIRED_VAL="${RUN_PAIRED_VAL:-0}"

read -r -a SEEDS_ARR <<< "${SEEDS_STR}"
case "${STAGE}" in
  all|test|main|k_curve|abl_guidance|abl_budget_delta|b1|b2|paired_val|paired_b1|paired_b2) ;;
  *) echo "Unknown stage: ${STAGE}" >&2; exit 2 ;;
esac
ABL_SEED="${ABL_SEED:-${SEEDS_ARR[0]}}"

echo "======================================================================"
echo "[Round 5 VP-OrbitDiff] Stage='${STAGE}' | Device='${DEVICE}' | SplitSeed='${SPLIT_SEED}'"
echo "Codebase: ${V6_DIR}"
echo "Output  : ${OUTPUT_ROOT}"
echo "======================================================================"

# ------------------------------------------------------------------------------
# Stage 0: Full Mathematical & Bug-Fix CPU Verification
# ------------------------------------------------------------------------------
if [[ "${STAGE}" == "all" || "${STAGE}" == "test" ]]; then
  echo "[Diff][Stage 0] Running full unit test suite (V6 + V1 + V2 + Round 5 Diffusion)..."
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
  --split-seed "${SPLIT_SEED}"
  --max-records-per-epoch "${MAX_RECORDS_PER_EPOCH}"
  --max-eval-records "${MAX_EVAL_RECORDS}"
)

# ------------------------------------------------------------------------------
# Stage 1: Main Route A (VP-OrbitDiff, K=4, P=8, delta=0.10, exact VJP) x 3 Seeds
# ------------------------------------------------------------------------------
if [[ "${STAGE}" == "all" || "${STAGE}" == "main" ]]; then
  for SEED in "${SEEDS_ARR[@]}"; do
    RUN_DIR="${OUTPUT_ROOT}/stage1_main_K4_P8/seed_${SEED}"
    echo "[Diff][Stage 1] Training VP-OrbitDiff (K=4, P=8, delta=0.10, VJP) | seed=${SEED}..."
    "${PYTHON_BIN}" "${V6_DIR}/scripts/train_diffusion.py" \
      "${COMMON_ARGS[@]}" \
      --run-dir "${RUN_DIR}" \
      --K 4 --P 8 --delta 0.10 \
      --guidance-mode vjp \
      --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}" --lr "${LR}" \
      --seed "${SEED}" \
      --eval-split dev_select
  done
fi

# ------------------------------------------------------------------------------
# Stage 2: Inference-Time Reverse Diffusion Curve K in {1, 2, 4, 8} (Same Trained Model!)
#   Verifies whether multi-step reverse PF-ODE beats single-step K=1 on the exact same weights.
# ------------------------------------------------------------------------------
if [[ "${STAGE}" == "all" || "${STAGE}" == "k_curve" ]]; then
  MAIN_CKPT="${OUTPUT_ROOT}/stage1_main_K4_P8/seed_${ABL_SEED}/best.pt"
  [[ -f "${MAIN_CKPT}" ]] || { echo "Missing required checkpoint: ${MAIN_CKPT}" >&2; exit 2; }
  for K_VAL in 1 2 4 8; do
    echo "[Diff][Stage 2] Evaluating trained VP-OrbitDiff at K=${K_VAL} on dev_select..."
    "${PYTHON_BIN}" "${V6_DIR}/scripts/validate_diffusion.py" \
      "${COMMON_ARGS[@]}" \
      --run-dir "${OUTPUT_ROOT}/stage2_k_curve/K${K_VAL}_seed_${ABL_SEED}" \
      --K "${K_VAL}" --P 8 --delta 0.10 \
      --guidance-mode vjp \
      --resume "${MAIN_CKPT}" \
      --eval-split dev_select
  done
fi

# ------------------------------------------------------------------------------
# Stage 3: Guidance, Conditional Second-Moment V_theta & Rollout-Prior Ablations
#   1) Inference-time guidance_mode in {identity_jac, none} on locked main checkpoint
#   2) Training-time fixed_isotropic_v (V_theta = I) and no_rollout_priors (pure B0 only)
# ------------------------------------------------------------------------------
if [[ "${STAGE}" == "all" || "${STAGE}" == "abl_guidance" ]]; then
  MAIN_CKPT="${OUTPUT_ROOT}/stage1_main_K4_P8/seed_${ABL_SEED}/best.pt"
  [[ -f "${MAIN_CKPT}" ]] || { echo "Missing required checkpoint: ${MAIN_CKPT}" >&2; exit 2; }
  for GMODE in identity_jac none; do
    echo "[Diff][Stage 3] Evaluating guidance_mode=${GMODE} on dev_select..."
    "${PYTHON_BIN}" "${V6_DIR}/scripts/validate_diffusion.py" \
      "${COMMON_ARGS[@]}" \
      --run-dir "${OUTPUT_ROOT}/stage3_abl_guidance/${GMODE}_seed_${ABL_SEED}" \
      --K 4 --P 8 --delta 0.10 \
      --guidance-mode "${GMODE}" \
      --resume "${MAIN_CKPT}" \
      --eval-split dev_select
  done

  echo "[Diff][Stage 3] Training fixed_isotropic_v (V_theta = I) | seed=${ABL_SEED}..."
  "${PYTHON_BIN}" "${V6_DIR}/scripts/train_diffusion.py" \
    "${COMMON_ARGS[@]}" \
    --run-dir "${OUTPUT_ROOT}/stage3_abl_guidance/fixed_isotropic_v_seed_${ABL_SEED}" \
    --K 4 --P 8 --delta 0.10 \
    --guidance-mode vjp --fixed-isotropic-v \
    --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}" --lr "${LR}" \
    --seed "${ABL_SEED}" \
    --eval-split dev_select

  echo "[Diff][Stage 3] Training no_rollout_priors (pure B0 anchors only) | seed=${ABL_SEED}..."
  "${PYTHON_BIN}" "${V6_DIR}/scripts/train_diffusion.py" \
    "${COMMON_ARGS[@]}" \
    --run-dir "${OUTPUT_ROOT}/stage3_abl_guidance/no_rollout_priors_seed_${ABL_SEED}" \
    --K 4 --P 8 --delta 0.10 \
    --guidance-mode vjp --no-rollout-priors \
    --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}" --lr "${LR}" \
    --seed "${ABL_SEED}" \
    --eval-split dev_select
fi

# ------------------------------------------------------------------------------
# Stage 4: State covariance floor sensitivity
# ------------------------------------------------------------------------------
if [[ "${STAGE}" == "all" || "${STAGE}" == "abl_budget_delta" ]]; then
  MAIN_CKPT="${OUTPUT_ROOT}/stage1_main_K4_P8/seed_${ABL_SEED}/best.pt"
  [[ -f "${MAIN_CKPT}" ]] || { echo "Missing required checkpoint: ${MAIN_CKPT}" >&2; exit 2; }
  for DVAL in 0.05 0.20; do
    "${PYTHON_BIN}" "${V6_DIR}/scripts/validate_diffusion.py" "${COMMON_ARGS[@]}" \
      --run-dir "${OUTPUT_ROOT}/stage4_budget_delta/delta${DVAL}_seed_${ABL_SEED}" \
      --delta "${DVAL}" --K 4 --P 8 --resume "${MAIN_CKPT}" --eval-split dev_select
  done
fi

for ROUTE in b1 b2; do
  if [[ "${STAGE}" == "all" || "${STAGE}" == "${ROUTE}" ]]; then
    for SEED in "${SEEDS_ARR[@]}"; do
      "${PYTHON_BIN}" "${V6_DIR}/scripts/train_diffusion.py" "${COMMON_ARGS[@]}" \
        --route "${ROUTE}" --run-dir "${OUTPUT_ROOT}/${ROUTE}/seed_${SEED}" \
        --K 4 --P 8 --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}" --lr "${LR}" --seed "${SEED}"
    done
  fi
done

# ------------------------------------------------------------------------------
# Stage 5: Locked Paired Replication on Historical VAL (ONLY after DEV-select lock!)
# ------------------------------------------------------------------------------
case "${STAGE}" in
  paired_val) LOCKED_ROUTES=(a) ;;
  paired_b1) LOCKED_ROUTES=(b1) ;;
  paired_b2) LOCKED_ROUTES=(b2) ;;
  all) [[ "${RUN_PAIRED_VAL}" == "1" ]] && LOCKED_ROUTES=(a b1 b2) || LOCKED_ROUTES=() ;;
  b1|b2) [[ "${RUN_PAIRED_VAL}" == "1" ]] && LOCKED_ROUTES=("${STAGE}") || LOCKED_ROUTES=() ;;
  *) [[ "${RUN_PAIRED_VAL}" == "1" ]] && LOCKED_ROUTES=(a) || LOCKED_ROUTES=() ;;
esac
for ROUTE in "${LOCKED_ROUTES[@]}"; do
  for SEED in "${SEEDS_ARR[@]}"; do
    RUN_DIR="${OUTPUT_ROOT}/${ROUTE}/seed_${SEED}"
    if [[ "${ROUTE}" == "a" ]]; then RUN_DIR="${OUTPUT_ROOT}/stage1_main_K4_P8/seed_${SEED}"; fi
    CKPT="${RUN_DIR}/best.pt"
    [[ -f "${CKPT}" ]] || { echo "Missing required checkpoint: ${CKPT}" >&2; exit 2; }
    "${PYTHON_BIN}" "${V6_DIR}/scripts/validate_diffusion.py" "${COMMON_ARGS[@]}" \
      --route "${ROUTE}" --run-dir "${RUN_DIR}" --resume "${CKPT}" --eval-split paired_val
  done
done

echo "[Round 5 VP-OrbitDiff] Pipeline execution finished cleanly."

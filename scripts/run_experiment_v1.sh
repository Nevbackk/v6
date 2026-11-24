#!/usr/bin/env bash
# ==============================================================================
# Complete Execution Pipeline — Model Plan V1 (JOINT_RESEARCH_PLAN + FD-ASF)
# ==============================================================================
# Usage (Linux / AutoDL CUDA Server):
#   PREPARED_ROOT=/path/to/prepared PROTOCOL_DIR=/path/to/protocol \
#     bash scripts/run_experiment_v1.sh
#
# Optional environment overrides:
#   OUTPUT_ROOT=runs/exp_v1        (default: runs/exp_v1)
#   PYTHON_BIN=python              (default: python)
#   DEVICE=cuda                    (default: cuda)
#   EPOCHS=10                      (default: 10)
#   BATCH_SIZE=32                  (default: 32)
#   SEEDS="20261028 20261029 20261030"
#   STAGE=all                      (all | test | classical | abl_3x2 | ltr_lm | fd_asf | paired_val)
#   MAX_RECORDS_PER_EPOCH=0        (0 = full TRAIN-fit; >0 = fast pilot cap, e.g. 512)
#   MAX_EVAL_RECORDS=0             (0 = full DEV-select / VAL; >0 = fast pilot cap)
#   RUN_PAIRED_VAL=0               (set 1 ONLY after locking model choices on dev_select!)
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
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/runs/exp_v1}"
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
  all|test|classical|abl_3x2|ltr_lm|fd_asf|paired_val) ;;
  *) echo "Unknown stage: ${STAGE}" >&2; exit 2 ;;
esac

echo "======================================================================"
echo "[Model Plan V1] Starting Stage='${STAGE}' | Device='${DEVICE}'"
echo "Codebase: ${V6_DIR}"
echo "Output  : ${OUTPUT_ROOT}"
echo "======================================================================"

# ------------------------------------------------------------------------------
# Stage 0: Local CPU Mathematical & End-to-End Unit Smoke Verification
# ------------------------------------------------------------------------------
if [[ "${STAGE}" == "all" || "${STAGE}" == "test" ]]; then
  echo "[V1][Stage 0] Running V6 + V1 + V2 mathematical & end-to-end unit tests..."
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
  --max-records-per-epoch "${MAX_RECORDS_PER_EPOCH}"
  --max-eval-records "${MAX_EVAL_RECORDS}"
)

# ------------------------------------------------------------------------------
# Stage 1: Zero-Training Classical Differential Correction Controls (DEV-select)
#   CL-GN1-Pure, CL-GN1-Matched (isolated learning gate), CL-WindowLM, CL-FullPrefixWLS
# ------------------------------------------------------------------------------
if [[ "${STAGE}" == "all" || "${STAGE}" == "classical" ]]; then
  for ARCH in cl_gn1_pure cl_gn1_matched cl_window_lm cl_full_prefix_wls; do
    echo "[V1][Stage 1] Evaluating classical baseline '${ARCH}' on dev_select..."
    "${PYTHON_BIN}" "${V6_DIR}/scripts/validate_v1.py" \
      "${COMMON_ARGS[@]}" \
      --run-dir "${OUTPUT_ROOT}/stage1_classical/${ARCH}" \
      --arch "${ARCH}" \
      --writeback-mode joint_6d \
      --eval-split dev_select
  done
fi

# ------------------------------------------------------------------------------
# Stage 2: 3x2 Orthogonal Loss x Bounded Jacobian-Free Refiner Ablation (3 Seeds)
#   {scaled_mse, huber, log_charbonnier} x {JF-BR vel_only, JF-BR homog_6d}
# ------------------------------------------------------------------------------
if [[ "${STAGE}" == "all" || "${STAGE}" == "abl_3x2" ]]; then
  for LOSS in scaled_mse huber log_charbonnier; do
    for JF_MODE in vel_only homog_6d; do
      for SEED in "${SEEDS_ARR[@]}"; do
        RUN_DIR="${OUTPUT_ROOT}/stage2_jfbr_3x2/${JF_MODE}_${LOSS}/seed_${SEED}"
        echo "[V1][Stage 2] Training JF-BR (${JF_MODE}) | loss=${LOSS} | seed=${SEED}..."
        "${PYTHON_BIN}" "${V6_DIR}/scripts/train_v1.py" \
          "${COMMON_ARGS[@]}" \
          --run-dir "${RUN_DIR}" \
          --arch jf_br \
          --jfbr-mode "${JF_MODE}" \
          --loss-type "${LOSS}" \
          --epochs "${EPOCHS}" \
          --batch-size "${BATCH_SIZE}" \
          --lr "${LR}" \
          --seed "${SEED}" \
          --eval-split dev_select
      done
    done
  done
fi

# ------------------------------------------------------------------------------
# Stage 3: Learned Trust-Region Preconditioned 1-Step LM (LTR-LM) Writeback Duel
#   Joint6D vs SchurVel (eliminating unknown position) vs LockPosVel (3 Seeds)
# ------------------------------------------------------------------------------
if [[ "${STAGE}" == "all" || "${STAGE}" == "ltr_lm" ]]; then
  for WB_MODE in joint_6d schur_vel lock_pos_vel; do
    for SEED in "${SEEDS_ARR[@]}"; do
      RUN_DIR="${OUTPUT_ROOT}/stage3_ltr_lm/${WB_MODE}/seed_${SEED}"
      echo "[V1][Stage 3] Training LTR-LM (${WB_MODE}) | seed=${SEED}..."
      "${PYTHON_BIN}" "${V6_DIR}/scripts/train_v1.py" \
        "${COMMON_ARGS[@]}" \
        --run-dir "${RUN_DIR}" \
        --arch ltr_lm \
        --writeback-mode "${WB_MODE}" \
        --loss-type log_charbonnier \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --lr "${LR}" \
        --seed "${SEED}" \
        --eval-split dev_select
    done
  done
fi

# ------------------------------------------------------------------------------
# Stage 4: A-Class Flagship Multi-Hypothesis Suite (FD-ASF vs Posterior-CNF vs Sobol)
#   1) Non-learned 2+4 Sobol-Lambert + 6D LM across particle budgets P in {4, 12, 16}
#   2) Standard Posterior-Fit Flow (Posterior-CNF, fitting pi_k)
#   3) Flagship FD-ASF (fitting optimal forecast disagreement q_k* prop pi_k ||h - h_bar||_2)
# ------------------------------------------------------------------------------
if [[ "${STAGE}" == "all" || "${STAGE}" == "fd_asf" ]]; then
  for P_BUDGET in 4 12 16; do
    echo "[V1][Stage 4] Evaluating non-learned Sobol-Lambert-LM | P=${P_BUDGET} on dev_select..."
    "${PYTHON_BIN}" "${V6_DIR}/scripts/validate_v1.py" \
      "${COMMON_ARGS[@]}" \
      --run-dir "${OUTPUT_ROOT}/stage4_flagship/sobol_lambert_P${P_BUDGET}" \
      --arch sobol_lambert_lm \
      --num-particles "${P_BUDGET}" \
      --eval-split dev_select
  done

  for ARCH in posterior_cnf fd_asf; do
    for SEED in "${SEEDS_ARR[@]}"; do
      RUN_DIR="${OUTPUT_ROOT}/stage4_flagship/${ARCH}_P12/seed_${SEED}"
      echo "[V1][Stage 4] Training ${ARCH} (P=12, DSM=0.0) | seed=${SEED}..."
      "${PYTHON_BIN}" "${V6_DIR}/scripts/train_v1.py" \
        "${COMMON_ARGS[@]}" \
        --run-dir "${RUN_DIR}" \
        --arch "${ARCH}" \
        --num-particles 12 \
        --dsm-weight 0.0 \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --lr "${LR}" \
        --seed "${SEED}" \
        --eval-split dev_select
    done
  done
fi

# ------------------------------------------------------------------------------
# Stage 5 (Pre-Registered Paired Replication ONLY after locking on DEV-select)
# ------------------------------------------------------------------------------
if [[ "${RUN_PAIRED_VAL}" == "1" || "${STAGE}" == "paired_val" ]]; then
  echo "[V1][Stage 5] Running locked paired replication on historical VAL (10,857 records)..."
  "${PYTHON_BIN}" "${V6_DIR}/scripts/validate_v1.py" \
    "${COMMON_ARGS[@]}" \
    --run-dir "${OUTPUT_ROOT}/stage1_classical/cl_gn1_matched" \
    --arch cl_gn1_matched \
    --eval-split paired_val

  for SEED in "${SEEDS_ARR[@]}"; do
    LTR_DIR="${OUTPUT_ROOT}/stage3_ltr_lm/joint_6d/seed_${SEED}"
    [[ -f "${LTR_DIR}/best.pt" ]] || { echo "Missing required checkpoint: ${LTR_DIR}/best.pt" >&2; exit 2; }
    "${PYTHON_BIN}" "${V6_DIR}/scripts/validate_v1.py" \
      "${COMMON_ARGS[@]}" \
      --run-dir "${LTR_DIR}" \
      --arch ltr_lm \
      --writeback-mode joint_6d \
      --resume "${LTR_DIR}/best.pt" \
      --eval-split paired_val

    FD_DIR="${OUTPUT_ROOT}/stage4_flagship/fd_asf_P12/seed_${SEED}"
    [[ -f "${FD_DIR}/best.pt" ]] || { echo "Missing required checkpoint: ${FD_DIR}/best.pt" >&2; exit 2; }
    "${PYTHON_BIN}" "${V6_DIR}/scripts/validate_v1.py" \
      "${COMMON_ARGS[@]}" \
      --run-dir "${FD_DIR}" \
      --arch fd_asf \
      --num-particles 12 \
      --resume "${FD_DIR}/best.pt" \
      --eval-split paired_val
  done
fi

echo "[Model Plan V1] Pipeline execution finished cleanly."

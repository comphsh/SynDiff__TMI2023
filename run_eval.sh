#!/usr/bin/env bash
# ==============================================================================
# SynDiff Inference Script for BraTS2020
# ==============================================================================
# Usage:
#   bash run_eval.sh <task_timestamp> [ckpt_epoch]
#
# Examples:
#   bash run_eval.sh 20260605_143022          # Default epoch 200
#   bash run_eval.sh 20260605_143022 200      # Specific epoch
#
# This only runs prediction (saves synthesized images).
# Metrics are computed separately by your global evaluation script.
# ==============================================================================

set -e

# --- Global Path Variables (modify for your environment) ---
COMPARE_ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA_ROOT="/devdata/hsh/datasets/seg_dataset/BraTS2020/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
DATALIST_DIR="${COMPARE_ROOT}/datalist/BraTS2020"
OUTPUT_PATH="${COMPARE_ROOT}/results"

# --- Parse Args ---
TASK_TS="${1:?Usage: bash run_eval.sh <task_timestamp> [ckpt_epoch]}"
CKPT_EPOCH="${2:-200}"

TASK_DIR="${OUTPUT_PATH}/task_${TASK_TS}"
MODEL_DIR="${TASK_DIR}/models"
PRED_DIR="${TASK_DIR}/prediction"

# --- Verify ---
echo "============================================"
echo " SynDiff Inference - BraTS2020"
echo "============================================"
echo "Project:   ${COMPARE_ROOT}"
echo "Task ts:   ${TASK_TS}"
echo "Checkpoint: epoch=${CKPT_EPOCH}"
echo "Output:    ${PRED_DIR}/"
echo "============================================"

[ -d "${MODEL_DIR}" ] || { echo "ERROR: Model dir not found: ${MODEL_DIR}"; exit 1; }

echo "Available checkpoints:"
ls -lh "${MODEL_DIR}"/*.pth 2>/dev/null || echo "  (none)"
echo ""

cd "${COMPARE_ROOT}"

python eval.py \
    --input_path "${DATA_ROOT}" \
    --datalist_dir "${DATALIST_DIR}" \
    --output_path "${OUTPUT_PATH}" \
    --task_ts "${TASK_TS}" \
    --ckpt_epoch "${CKPT_EPOCH}" \
    --gpu 0 \
    --image_size 256 \
    --num_channels 2 \
    --num_channels_dae 64 \
    --ch_mult 1 1 2 2 4 4 \
    --num_timesteps 4 \
    --num_res_blocks 2 \
    --embedding_type positional \
    --z_emb_dim 256 \
    --t_emb_dim 256

echo ""
echo "============================================"
echo " Inference complete!"
echo ""
echo " Predictions: ${PRED_DIR}/"
echo "   Subdirs:"
for d in 0001 0010 0011 0100 0101 0110 0111 1000 1001 1010 1011 1100 1101 1110; do
    n=$(ls "${PRED_DIR}/${d}/" 2>/dev/null | wc -l)
    echo "     ${d}/  (${n} patients)"
done
echo ""
echo " To compute metrics, run your global evaluation script on:"
echo "   ${PRED_DIR}/"
echo "============================================"

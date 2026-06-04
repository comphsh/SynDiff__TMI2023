#!/bin/bash
# =============================================================================
# SynDiff Evaluation Script for BraTS2020
# =============================================================================
# Usage:
#   bash run_eval.sh <task_timestamp>
#
# Example:
#   bash run_eval.sh 20260604_120000
#
# Description:
#   Evaluates trained SynDiff model on BraTS2020 test set.
#   - Tests all 14 missing modality patterns
#   - mask_id 1-4:   1 missing modality (4 patterns)
#   - mask_id 5-10:  2 missing modalities (6 patterns)
#   - mask_id 11-14: 3 missing modalities (4 patterns)
#   - Generates synthesized images for missing modalities
#   - Computes SSIM/PSNR/MSE/MAE/LPIPS/FID metrics
#
#   mask format: 'flair_t1_t1ce_t2', 1=available, 0=missing
#
# Results directory structure:
#   results/task_<timestamp>/
#     prediction/{mask_id}/      - generated images + input + ground truth
#     prediction_metric_result/{mask_id}/result.txt  - metrics
# =============================================================================

set -e

# Global paths
DATA_ROOT="/devdata/hsh/datasets/seg_dataset/BraTS2020/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
OUTPUT_PATH="./results"
EVAL_SCRIPT="/devdata2/hsh/program/python/methods/MySparseDiffusion/my_sparse_diff-moe-006/scripts/_01_vae/metrics/syn_metrics.py"

# Parse arguments
if [ $# -lt 1 ]; then
    echo "Usage: bash run_eval.sh <task_timestamp> [which_epoch]"
    echo ""
    echo "Available task directories:"
    ls -d ${OUTPUT_PATH}/task_*/ 2>/dev/null || echo "  (none found)"
    echo ""
    echo "Example:"
    echo "  bash run_eval.sh 20260604_120000"
    echo "  bash run_eval.sh 20260604_120000 200"
    exit 1
fi

TASK_TIMESTAMP=$1
WHICH_EPOCH=${2:-200}
TASK_DIR="${OUTPUT_PATH}/task_${TASK_TIMESTAMP}"
CKPT_PATH="${TASK_DIR}"

if [ ! -d "${CKPT_PATH}/models" ]; then
    echo "Error: Checkpoint directory not found: ${CKPT_PATH}/models"
    echo "Available task directories:"
    ls -d ${OUTPUT_PATH}/task_*/ 2>/dev/null || echo "  (none found)"
    exit 1
fi

echo "================================================"
echo "  SynDiff Evaluation on BraTS2020"
echo "================================================"
echo "Task directory: ${TASK_DIR}"
echo "Checkpoint epoch: ${WHICH_EPOCH}"
echo "Data root: ${DATA_ROOT}"
echo "Eval script: ${EVAL_SCRIPT}"
echo ""

# Run evaluation
python3 eval.py \
    --image_size 256 \
    --num_channels 2 \
    --num_channels_dae 64 \
    --ch_mult 1 1 2 2 4 4 \
    --num_timesteps 4 \
    --num_res_blocks 2 \
    --batch_size 1 \
    --embedding_type positional \
    --z_emb_dim 256 \
    --ngf 64 \
    --ckpt_path "${CKPT_PATH}" \
    --which_epoch "${WHICH_EPOCH}" \
    --gpu_chose 0 \
    --input_path "${DATA_ROOT}" \
    --eval_script "${EVAL_SCRIPT}"

echo ""
echo "================================================"
echo "  Evaluation completed!"
echo ""
echo "Results:"
echo "  Predictions: ${TASK_DIR}/prediction/{mask_id}/"
echo "  Metrics:     ${TASK_DIR}/prediction_metric_result/{mask_id}/result.txt"
echo ""
echo "Missing modality patterns tested:"
echo "  mask_id 1  (0111): flair missing"
echo "  mask_id 2  (1011): t1 missing"
echo "  mask_id 3  (1101): t1ce missing"
echo "  mask_id 4  (1110): t2 missing"
echo "  mask_id 5  (0011): flair+t1 missing"
echo "  mask_id 6  (0101): flair+t1ce missing"
echo "  mask_id 7  (0110): flair+t2 missing"
echo "  mask_id 8  (1001): t1+t1ce missing"
echo "  mask_id 9  (1010): t1+t2 missing"
echo "  mask_id 10 (1100): t1ce+t2 missing"
echo "  mask_id 11 (0001): flair+t1+t1ce missing"
echo "  mask_id 12 (0010): flair+t1+t2 missing"
echo "  mask_id 13 (0100): flair+t1ce+t2 missing"
echo "  mask_id 14 (1000): t1+t1ce+t2 missing"
echo "================================================"

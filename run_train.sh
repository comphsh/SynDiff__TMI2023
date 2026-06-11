#!/usr/bin/env bash
# ==============================================================================
# SynDiff Training Script for BraTS2020
# ==============================================================================
# Usage:
#   bash run_train.sh
#
# Path Configuration (modify as needed for local PC vs server):
#   COMPARE_ROOT:  Project root directory
#   DATA_ROOT:     BraTS2020 dataset root
#   DATALIST_DIR:  train/val/test.list directory
# ==============================================================================

set -e

# --- Global Path Variables (modify for your environment) ---
COMPARE_ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA_ROOT="/devdata/hsh/datasets/seg_dataset/BraTS2020/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
DATALIST_DIR="${COMPARE_ROOT}/datalist/BraTS2020"
OUTPUT_PATH="${COMPARE_ROOT}/results"

# --- Verify ---
echo "============================================"
echo " SynDiff Training - BraTS2020"
echo "============================================"
echo "Project:  ${COMPARE_ROOT}"
echo "Data:     ${DATA_ROOT}"
echo "Datalist: ${DATALIST_DIR}"
echo "Output:   ${OUTPUT_PATH}"
echo "============================================"

[ -d "${DATA_ROOT}" ] || { echo "ERROR: DATA_ROOT not found: ${DATA_ROOT}"; exit 1; }
[ -f "${DATALIST_DIR}/train.list" ] || { echo "ERROR: train.list not found"; exit 1; }

echo "Train: $(wc -l < ${DATALIST_DIR}/train.list) patients"
echo "Val:   $(wc -l < ${DATALIST_DIR}/val.list) patients"
echo "Test:  $(wc -l < ${DATALIST_DIR}/test.list) patients"
echo "============================================"
echo ""

cd "${COMPARE_ROOT}"

python train.py \
    --image_size 256 \
    --num_channels 2 \
    --num_channels_dae 64 \
    --ch_mult 1 1 2 2 4 4 \
    --num_timesteps 4 \
    --num_res_blocks 2 \
    --batch_size 1 \
    --num_epoch 200 \
    --ngf 64 \
    --embedding_type positional \
    --z_emb_dim 256 \
    --t_emb_dim 256 \
    --lr_g 1.6e-4 \
    --lr_d 1e-4 \
    --beta1 0.5 \
    --beta2 0.9 \
    --r1_gamma 0.05 \
    --lazy_reg 10 \
    --lambda_l1_loss 0.5 \
    --nz 100 \
    --save_content \
    --save_ckpt_every 50 \
    --save_content_every 50 \
    --exp BraTS20_syndiff \
    --input_path "${DATA_ROOT}" \
    --datalist_dir "${DATALIST_DIR}" \
    --output_path "${OUTPUT_PATH}"

echo ""
echo "============================================"
echo " Training completed!"
echo " Results: ${OUTPUT_PATH}/task_*/"
echo " TensorBoard: tensorboard --logdir ${OUTPUT_PATH}/task_*/tensorboard"
echo "============================================"

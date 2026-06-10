#!/usr/bin/env bash
# ==============================================================================
# SynDiff Training Script for BraTS2020 Unified Experiment
# ==============================================================================
# Usage:
#   bash run_train.sh
#
# Description:
#   Trains SynDiff (Adversarial Diffusion Model) on BraTS2020 dataset.
#   - Data: 369 cases, 260 train / 36 val / 73 test (70/10/20 split)
#   - Modalities: flair, t1, t1ce, t2
#   - Normalization: Percentile-based (0-99.5) -> [0,1] -> [-1,1]
#   - Training: Random source-target pair sampling from 4 modalities
#   - 200 epochs, batch_size=1, Quadro RTX 8000 48G
#   - Output: results/task_{timestamp}/models/
#   - Logs: TensorBoard in results/task_{timestamp}/tensorboard/
#
# Path Configuration (modify as needed):
#   COMPARE_ROOT:  Project root directory
#   DATA_ROOT:     BraTS2020 dataset root directory
#   DATALIST_DIR:  Directory containing train/val/test.list
#   OUTPUT_PATH:   Results output directory
# ==============================================================================

set -e  # Exit on error

# --- Path Configuration ---
COMPARE_ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA_ROOT="/devdata/hsh/datasets/seg_dataset/BraTS2020/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
DATALIST_DIR="${COMPARE_ROOT}/datalist/BraTS2020"
OUTPUT_PATH="${COMPARE_ROOT}/results"

# --- Verify paths ---
echo "============================================"
echo " SynDiff Training - BraTS2020 Unified Experiment"
echo "============================================"
echo "Project root:    ${COMPARE_ROOT}"
echo "Data root:       ${DATA_ROOT}"
echo "Datalist dir:    ${DATALIST_DIR}"
echo "Output path:     ${OUTPUT_PATH}"
echo "============================================"

if [ ! -d "${DATA_ROOT}" ]; then
    echo "ERROR: Data root not found: ${DATA_ROOT}"
    exit 1
fi

if [ ! -f "${DATALIST_DIR}/train.list" ]; then
    echo "ERROR: train.list not found in ${DATALIST_DIR}"
    exit 1
fi

# --- Training Configuration ---
# Hyperparameters from original SynDiff paper (TMI 2023)
# Adapted for single GPU (Quadro RTX 8000 48G)
BATCH_SIZE=1
NUM_EPOCH=200
IMAGE_SIZE=256
NUM_CHANNELS_DAE=64
CH_MULT="1 1 2 2 4 4"
NUM_TIMESTEPS=4
NUM_RES_BLOCKS=2
NGF=64
EMBEDDING_TYPE="positional"
Z_EMB_DIM=256
T_EMB_DIM=256
LR_G="1.6e-4"
LR_D="1e-4"
BETA1=0.5
BETA2=0.9
R1_GAMMA=0.05
LAZY_REG=10
LAMBDA_L1=0.5
NZ=100
SAVE_CKPT_EVERY=50
SAVE_CONTENT_EVERY=50

echo ""
echo "Training Configuration:"
echo "  batch_size:        ${BATCH_SIZE}"
echo "  num_epoch:         ${NUM_EPOCH}"
echo "  image_size:        ${IMAGE_SIZE}"
echo "  num_timesteps:     ${NUM_TIMESTEPS}"
echo "  lr_g:              ${LR_G}"
echo "  lr_d:              ${LR_D}"
echo "  lambda_l1_loss:    ${LAMBDA_L1}"
echo "  r1_gamma:          ${R1_GAMMA}"
echo "============================================"
echo ""

# --- Run Training ---
cd "${COMPARE_ROOT}"

python train.py \
    --image_size ${IMAGE_SIZE} \
    --num_channels 2 \
    --num_channels_dae ${NUM_CHANNELS_DAE} \
    --ch_mult ${CH_MULT} \
    --num_timesteps ${NUM_TIMESTEPS} \
    --num_res_blocks ${NUM_RES_BLOCKS} \
    --batch_size ${BATCH_SIZE} \
    --num_epoch ${NUM_EPOCH} \
    --ngf ${NGF} \
    --embedding_type ${EMBEDDING_TYPE} \
    --z_emb_dim ${Z_EMB_DIM} \
    --t_emb_dim ${T_EMB_DIM} \
    --lr_g ${LR_G} \
    --lr_d ${LR_D} \
    --beta1 ${BETA1} \
    --beta2 ${BETA2} \
    --r1_gamma ${R1_GAMMA} \
    --lazy_reg ${LAZY_REG} \
    --lambda_l1_loss ${LAMBDA_L1} \
    --nz ${NZ} \
    --save_content \
    --save_ckpt_every ${SAVE_CKPT_EVERY} \
    --save_content_every ${SAVE_CONTENT_EVERY} \
    --exp BraTS20_syndiff \
    --input_path "${DATA_ROOT}" \
    --datalist_dir "${DATALIST_DIR}" \
    --output_path "${OUTPUT_PATH}"

echo ""
echo "============================================"
echo " Training completed!"
echo " Check results in: ${OUTPUT_PATH}/task_*/"
echo " TensorBoard: tensorboard --logdir ${OUTPUT_PATH}/task_*/tensorboard"
echo "============================================"

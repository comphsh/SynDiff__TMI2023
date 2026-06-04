#!/bin/bash
# =============================================================================
# SynDiff Training Script for BraTS2020
# =============================================================================
# Usage:
#   bash run_train.sh
#
# Description:
#   Trains SynDiff (Adversarial Diffusion Model) on BraTS2020 dataset.
#   - 4 modalities: flair, t1, t1ce, t2
#   - Random source-target pair sampling during training
#   - 200 epochs, batch_size=1 (adjust if GPU memory allows)
#   - Models saved to: results/task_<timestamp>/models/
#   - TensorBoard logs: results/task_<timestamp>/tensorboard/
#
# Hardware: Quadro RTX 8000 48G
# =============================================================================

set -e

# Global paths
DATA_ROOT="/devdata/hsh/datasets/seg_dataset/BraTS2020/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
OUTPUT_PATH="./results"

echo "================================================"
echo "  SynDiff Training on BraTS2020"
echo "================================================"
echo "Data root: ${DATA_ROOT}"
echo "Output path: ${OUTPUT_PATH}"
echo ""

# Run training
python3 train.py \
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
    --r1_gamma 0.05 \
    --z_emb_dim 256 \
    --lr_d 1e-4 \
    --lr_g 1.6e-4 \
    --lazy_reg 10 \
    --save_content \
    --save_content_every 50 \
    --save_ckpt_every 50 \
    --lambda_l1_loss 0.5 \
    --contrast1 T1 \
    --contrast2 T2 \
    --exp BraTS20_syndiff \
    --input_path "${DATA_ROOT}" \
    --output_path "${OUTPUT_PATH}"

echo ""
echo "================================================"
echo "  Training completed!"
echo "  Check results in: ${OUTPUT_PATH}/task_*/"
echo "  TensorBoard: tensorboard --logdir ${OUTPUT_PATH}/task_*/tensorboard"
echo "================================================"

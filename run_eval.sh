#!/usr/bin/env bash
# ==============================================================================
# SynDiff Evaluation Pipeline for BraTS2020 Unified Experiment
# ==============================================================================
# Usage:
#   bash run_eval.sh <task_timestamp> [ckpt_epoch] [--metrics_only]
#
# Arguments:
#   task_timestamp: The timestamp of the training task (e.g., 20260605_143022)
#   ckpt_epoch:     (Optional) Checkpoint epoch. Default: "200"
#   --metrics_only:  (Optional) Skip inference, only compute metrics on existing predictions
#
# Examples:
#   bash run_eval.sh 20260605_143022                 # Full pipeline (inference + metrics)
#   bash run_eval.sh 20260605_143022 200             # Use epoch 200 checkpoint
#   bash run_eval.sh 20260605_143022 final --metrics_only  # Only compute metrics
#
# Pipeline:
#   Step 1 (eval.py):      Load model → Generate 14 missing-pattern predictions → Save NIfTI
#   Step 2 (syn_metric.py): Load predictions → Compute SSIM/PSNR/MSE/MAE/LPIPS → Save result.txt
# ==============================================================================

set -e

# --- Path Configuration ---
COMPARE_ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA_ROOT="/devdata/hsh/datasets/seg_dataset/BraTS2020/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
DATALIST_DIR="${COMPARE_ROOT}/datalist/BraTS2020"
OUTPUT_PATH="${COMPARE_ROOT}/results"
EVAL_SCRIPT="/devdata2/hsh/program/python/methods/MySparseDiffusion/my_sparse_diff-moe-006/scripts/_01_vae/metrics/syn_metrics.py"

# --- Parse Arguments ---
METRICS_ONLY=false
TASK_TS=""
CKPT_EPOCH="200"

for arg in "$@"; do
    case $arg in
        --metrics_only)
            METRICS_ONLY=true
            ;;
        *)
            if [ -z "$TASK_TS" ]; then
                TASK_TS="$arg"
            elif [ "$CKPT_EPOCH" = "200" ] && [ "$arg" != "--metrics_only" ]; then
                CKPT_EPOCH="$arg"
            fi
            ;;
    esac
done

if [ -z "$TASK_TS" ]; then
    echo "ERROR: task_timestamp is required."
    echo ""
    echo "Usage: bash run_eval.sh <task_timestamp> [ckpt_epoch] [--metrics_only]"
    echo ""
    echo "Examples:"
    echo "  bash run_eval.sh 20260605_143022                  # Full pipeline"
    echo "  bash run_eval.sh 20260605_143022 200              # Specific epoch"
    echo "  bash run_eval.sh 20260605_143022 final --metrics_only  # Metrics only"
    echo ""
    echo "Available task directories:"
    ls -d ${OUTPUT_PATH}/task_*/ 2>/dev/null || echo "  (none found)"
    exit 1
fi

TASK_DIR="${OUTPUT_PATH}/task_${TASK_TS}"
MODEL_DIR="${TASK_DIR}/models"
PRED_DIR="${TASK_DIR}/prediction"
METRIC_DIR="${TASK_DIR}/prediction_metric_result"

# --- Verify paths ---
echo "============================================"
echo " SynDiff Evaluation Pipeline"
echo "============================================"
echo "Project root:    ${COMPARE_ROOT}"
echo "Data root:       ${DATA_ROOT}"
echo "Datalist dir:    ${DATALIST_DIR}"
echo "Task timestamp:  ${TASK_TS}"
echo "Task dir:        ${TASK_DIR}"
echo "============================================"
echo ""

# --- GPU Configuration ---
GPU=0

if [ ! -d "${TASK_DIR}" ]; then
    echo "ERROR: Task directory not found: ${TASK_DIR}"
    echo "Available task directories:"
    ls -d ${OUTPUT_PATH}/task_*/ 2>/dev/null || echo "  (none found)"
    exit 1
fi

cd "${COMPARE_ROOT}"

# ==============================================================================
# Step 1: Inference (eval.py) — prediction only
# ==============================================================================
if [ "$METRICS_ONLY" = false ]; then
    echo "============================================"
    echo " Step 1/2: Inference (eval.py)"
    echo "============================================"
    echo "Checkpoint:      epoch=${CKPT_EPOCH}"
    echo "Model dir:       ${MODEL_DIR}"
    echo "GPU:             ${GPU}"
    echo "Output:          ${PRED_DIR}/"
    echo "============================================"
    echo ""

    if [ ! -d "${MODEL_DIR}" ]; then
        echo "ERROR: Model directory not found: ${MODEL_DIR}"
        exit 1
    fi

    echo "Available checkpoints:"
    ls -lh ${MODEL_DIR}/*.pth 2>/dev/null || echo "  (none found)"
    echo ""

    python eval.py \
        --input_path "${DATA_ROOT}" \
        --datalist_dir "${DATALIST_DIR}" \
        --output_path "${OUTPUT_PATH}" \
        --task_ts "${TASK_TS}" \
        --ckpt_epoch "${CKPT_EPOCH}" \
        --gpu ${GPU} \
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
    echo "[run_eval] Step 1 complete: predictions saved to ${PRED_DIR}/"
    echo ""
else
    echo "============================================"
    echo " Step 1/2: Inference (SKIPPED --metrics_only)"
    echo "============================================"
    echo ""
fi

# ==============================================================================
# Step 2: Metrics (syn_metric.py)
# ==============================================================================
echo "============================================"
echo " Step 2/2: Metrics (syn_metric.py)"
echo "============================================"
echo "Prediction dir: ${PRED_DIR}/"
echo "Output dir:     ${METRIC_DIR}/"
echo "Eval script:    ${EVAL_SCRIPT}"
echo "============================================"
echo ""

if [ ! -d "${PRED_DIR}" ]; then
    echo "ERROR: Prediction directory not found: ${PRED_DIR}"
    echo "Run inference first: bash run_eval.sh ${TASK_TS}"
    exit 1
fi

if [ ! -f "${EVAL_SCRIPT}" ]; then
    echo "[WARNING] syn_metric.py not found at: ${EVAL_SCRIPT}"
    echo "Skipping metric computation."
    echo "Install syn_metric.py and re-run:"
    echo "  bash run_eval.sh ${TASK_TS} --metrics_only"
    exit 0
fi

# Run syn_metric.py for each mask
for mask_str in 0111 1011 1101 1110 0011 0101 0110 1001 1010 1100 0001 0010 0100 1000; do
    mask_pred_dir="${PRED_DIR}/${mask_str}"
    mask_metric_dir="${METRIC_DIR}/${mask_str}"

    if [ ! -d "${mask_pred_dir}" ]; then
        echo "[WARNING] Missing prediction dir: ${mask_pred_dir}, skipping mask=${mask_str}"
        continue
    fi

    echo "Computing metrics for mask=${mask_str}..."

    # Use syn_metric.py's stream_process interface
    python -c "
import sys
sys.path.insert(0, '$(dirname ${EVAL_SCRIPT})')
sys.path.insert(0, '$(dirname $(dirname ${EVAL_SCRIPT}))')
from syn_metrics import ImageQualityEvaluator, stream_process
import os

mask_pred_dir = '${mask_pred_dir}'
mask_metric_dir = '${mask_metric_dir}'
os.makedirs(mask_metric_dir, exist_ok=True)

evaluator = ImageQualityEvaluator(LPIPS_model_type='nomedical')
stream_process(
    src_path='${DATA_ROOT}',
    gen_path=mask_pred_dir,
    evaluator=evaluator,
    gen_shuffix='syn',
    from_ckpt_name='',
)

# Copy results to standard location
import glob, shutil
metric_files = glob.glob(os.path.join(mask_pred_dir, '*_metrics_results', '*.txt'))
if metric_files:
    result_file = os.path.join(mask_metric_dir, 'result.txt')
    shutil.copy(metric_files[0], result_file)
    print(f'Metrics saved to: {result_file}')
else:
    # Fallback: run syn_metric.py CLI
    print('[WARNING] stream_process produced no output, trying CLI...')
"
done

echo ""
echo "============================================"
echo " Pipeline complete!"
echo ""
echo " Predictions:  ${PRED_DIR}/"
echo "   Subdirs: 0111 1011 1101 1110 0011 0101 0110 1001 1010 1100 0001 0010 0100 1000"
echo " Metrics:      ${METRIC_DIR}/"
echo ""
echo " Quick view:"
echo "   cat ${METRIC_DIR}/1110/result.txt"
echo ""
echo " Summary view (all masks):"
for mask_str in 0111 1011 1101 1110 0011 0101 0110 1001 1010 1100 0001 0010 0100 1000; do
    f="${METRIC_DIR}/${mask_str}/result.txt"
    if [ -f "$f" ]; then
        printf "  mask_%-4s: %s\n" "$mask_str" "$(tail -1 "$f")"
    else
        printf "  mask_%-4s: (pending)\n" "$mask_str"
    fi
done
echo "============================================"

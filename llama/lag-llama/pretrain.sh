#!/bin/bash

# Updated to your specific pretraining path
DATASET_PATH="/root/traffic-shifts/trafpy/trafpy_pretrain_data_extended.csv"
EXPERIMENT_NAME="backbone_v1_ctx128_extended"
RESULTS_DIR="experiments/results"
WANDB_PROJECT="lag-llama-pretraining"

# Model Architecture
CONTEXT_LENGTH=256
PREDICTION_LENGTH=1
N_LAYER=1
N_HEAD=8
N_EMBD_PER_HEAD=16

# Training Hyperparameters
MAX_EPOCHS=30
BATCH_SIZE=64
LEARNING_RATE=1e-4
GPU_ID=0

# Check if file exists before running
if [ ! -f "$DATASET_PATH" ]; then
    echo "ERROR: File not found at $DATASET_PATH"
    exit 1
fi

python run.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --dataset_path "$DATASET_PATH" \
    --results_dir "$RESULTS_DIR" \
    --wandb_project "$WANDB_PROJECT" \
    --context_length "$CONTEXT_LENGTH" \
    --prediction_length "$PREDICTION_LENGTH" \
    --n_layer "$N_LAYER" \
    --n_head "$N_HEAD" \
    --n_embd_per_head "$N_EMBD_PER_HEAD" \
    --lags_seq "h" "min" "s" \
    --max_epochs "$MAX_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    --gpu "$GPU_ID" \
    --data_normalization "mean" \
    --distr_output "studentT" \
    --wandb_mode "offline"

echo "Pretraining process for $EXPERIMENT_NAME has completed."

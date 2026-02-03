#!/bin/bash
BACKBONE_NAME="backbone_latest"
EXP_NAME="finetune_trafpy_specialized"
RESULTS_DIR="experiments/results"

# --- NEW: WIPE OLD RESULTS TO FORCE NEW CHECKPOINT ---
echo "Cleaning old results for $EXP_NAME..."
rm -rf $RESULTS_DIR/$EXP_NAME

python run.py \
    -e $EXP_NAME \
    -d "/root/traffic-shifts/trafpy" \
    --seed 42 \
    -r $RESULTS_DIR \
    --batch_size 64 \
    --max_epochs 15 \
    --single_dataset "trafpy_finetune_normal_data" \
    --get_ckpt_path_from_experiment_name $BACKBONE_NAME \
    --wandb_project "lag-llama-test" \
    --context_length 128 \
    --n_layer 1 \
    --n_head 8 \
    --n_embd_per_head 16 \
    --time_feat \
    --lags_seq $(seq 1 47) \
    --lr 0.0001 \
    --use_dataset_prediction_length \
    --num_validation_windows 1 \
    --wandb_mode "offline"

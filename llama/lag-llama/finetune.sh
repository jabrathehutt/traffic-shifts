#!/bin/bash
# We use the fixed name for the backbone
BACKBONE_NAME="backbone_latest" 
EXP_NAME="finetune_trafpy_specialized"
RESULTS_DIR="experiments/results"

python run.py \
    -e $EXP_NAME \
    -d "/root/traffic-shifts/trafpy" \
    --seed 42 \
    -r $RESULTS_DIR \
    --batch_size 64 \
    --max_epochs 10 \
    --single_dataset "trafpy_finetune_normal_data" \
    --get_ckpt_path_from_experiment_name $BACKBONE_NAME \
    --wandb_project "lag-llama-test" \
    --context_length 512 \
    --n_layer 1 \
    --n_head 8 \
    --n_embd_per_head 16 \
    --time_feat \
    --lags_seq $(seq 1 47) \
    --lr 0.00001 \
    --use_dataset_prediction_length \
    --num_validation_windows 1 \
    --wandb_mode "offline"

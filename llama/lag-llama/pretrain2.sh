rm -rf experiments/results/pretraining_lag_llama_trafpy

python run.py \
    -e "pretraining_lag_llama_trafpy" \
    -d "/root/traffic-shifts/trafpy" \
    --single_dataset "trafpy_pretrain_data_extended" \
    --context_length 256 \
    --n_layer 1 \
    --n_head 8 \
    --n_embd_per_head 16 \
    --time_feat \
    --lags_seq $(seq 1 47) \
    --lr 0.0001 \
    --max_epochs 15\
    --wandb_mode "offline"

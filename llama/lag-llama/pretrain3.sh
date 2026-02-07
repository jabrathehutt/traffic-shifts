rm -rf experiments/results/pretraining_lag_llama_trafpy
rm -f models/backbone_latest.ckpt
python run2.py \
    -e "pretraining_lag_llama_trafpy" \
    -d "/root/traffic-shifts/trafpy" \
    --single_dataset "trafpy_pretrain_data_extended" \
    --context_length 512 \
    --n_layer 1 \
    --n_head 8 \
    --n_embd_per_head 16 \
    --time_feat \
    --lags_seq $(seq 1 95) \
    --lr 0.0001 \
    --max_epochs 15\
    --wandb_mode "offline"


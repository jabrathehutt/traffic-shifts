import pandas as pd
import numpy as np
import torch
import random
import os
from gluonts.dataset.pandas import PandasDataset
from lag_llama.gluon.estimator import LagLlamaEstimator

# --- 1. DETERMINISTIC SEEDING ---
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Critical for bit-for-bit reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Optional: ensure DataLoader uses the same seed for workers
    # os.environ["PL_GLOBAL_SEED"] = str(seed)

seed_everything(42)

# --- CONFIGURATION ---
INPUT_FILE = "/root/traffic-shifts/trafpy/trafpy_pretrain_data_extended.csv"
CKPT_PATH = "/root/traffic-shifts/llama/lag-llama/lag-llama-backbone.ckpt"

def run_extensive_pretraining():
    print(f"Loading {INPUT_FILE} for backbone pretraining...")
    df = pd.read_csv(INPUT_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['traffic_volume_Tbits'] = df['traffic_volume_Tbits'].astype('float32')

    dataset = PandasDataset.from_long_dataframe(
        df,
        target="traffic_volume_Tbits",
        timestamp="timestamp",
        item_id="flow_key_id",
        freq="10min"
    )

    print("Initializing Lag-Llama Estimator (Strict Deterministic Mode)...")
    estimator = LagLlamaEstimator(
        prediction_length=24,
        context_length=512,
        batch_size=64,
        num_parallel_samples=100,
        trainer_kwargs={
            "max_epochs": 20,
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "devices": 1,
            # --- 2. FIXING THE MISCONFIGURATION ---
            # We remove enable_checkpointing=False to avoid the conflict.
            # We keep deterministic=True for the PyTorch Lightning trainer.
            "deterministic": True,
            # If you want to avoid generating local checkpoint files:
            "logger": False, 
        }
    )

    print("Starting Stage 1: Extensive Unsupervised Pretraining...")
    # Estimator.train() handles the internal Lightning callbacks
    predictor = estimator.train(dataset)

    # 3. MANUAL WEIGHT EXPORT
    # This ensures your backbone checkpoint is exactly what you intend to use for finetuning.
    torch.save(predictor.network.state_dict(), CKPT_PATH)
    print(f"Successfully saved deterministic pretrained backbone to {CKPT_PATH}")

if __name__ == "__main__":
    run_extensive_pretraining()

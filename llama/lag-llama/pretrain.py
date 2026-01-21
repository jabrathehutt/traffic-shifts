import pandas as pd
import numpy as np
import torch
from gluonts.dataset.pandas import PandasDataset
from lag_llama.gluon.estimator import LagLlamaEstimator

# --- CONFIGURATION ---
INPUT_FILE = "/root/traffic/llama/lag-llama/trafpy_pretrain_data.csv"
CKPT_PATH = "/root/traffic/llama/lag-llama/lag-llama-backbone.ckpt"

def run_extensive_pretraining():
    print(f"Loading {INPUT_FILE} for backbone pretraining...")
    df = pd.read_csv(INPUT_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sync with current TrafPy column name and float32 requirement
    df['traffic_volume_Tbits'] = df['traffic_volume_Tbits'].astype('float32')

    # --- CRITICAL FIX: Use from_long_dataframe ---
    # This correctly handles multiple flows in a single CSV
    dataset = PandasDataset.from_long_dataframe(
        df,
        target="traffic_volume_Tbits",
        timestamp="timestamp",
        item_id="flow_key_id", # Tells GluonTS how to separate the series
        freq="10min"           # Use '10min' instead of '10T' to avoid deprecation warnings
    )

    print("Initializing Lag-Llama Estimator...")
    estimator = LagLlamaEstimator(
        prediction_length=24,   
        context_length=128,     
        batch_size=64,
        num_parallel_samples=100,
        trainer_kwargs={
            "max_epochs": 50,
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "devices": 1 
        }
    )

    print("Starting Stage 1: Extensive Unsupervised Pretraining...")
    predictor = estimator.train(dataset)

    # Save to your local directory
    torch.save(predictor.network.state_dict(), CKPT_PATH)
    print(f"Successfully saved pretrained backbone to {CKPT_PATH}")

if __name__ == "__main__":
    run_extensive_pretraining()

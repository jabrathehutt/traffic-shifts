import pandas as pd
import numpy as np
import torch
import random
import os
from gluonts.dataset.pandas import PandasDataset
from lag_llama.gluon.estimator import LagLlamaEstimator

def seed_everything(seed=42):
    random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed(seed); torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

INPUT_FILE = "/root/traffic-shifts/trafpy/trafpy_pretrain_data_extended.csv"
CKPT_PATH = "/root/traffic-shifts/llama/lag-llama/lag-llama-backbone.ckpt"

def run_extensive_pretraining():
    print(f"Loading {INPUT_FILE} for backbone pretraining...")
    df = pd.read_csv(INPUT_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['traffic_volume_Tbits'] = df['traffic_volume_Tbits'].astype('float32')

    dataset = PandasDataset.from_long_dataframe(
        df, target="traffic_volume_Tbits", timestamp="timestamp",
        item_id="flow_key_id", freq="10min"
    )

    estimator = LagLlamaEstimator(
        prediction_length=1, # Match detection task
        context_length=128,
        batch_size=64,
        num_parallel_samples=100,
        trainer_kwargs={
            "max_epochs": 30,
            "accelerator": "gpu",
            "devices": 1,
            "deterministic": True,
            "logger": False,
        }
    )

    print("Starting Stage 1: Extensive Unsupervised Pretraining...")
    predictor = estimator.train(dataset)

    # SAVE FIX: Save the underlying transformer (prediction_net)
    torch.save(predictor.prediction_net.state_dict(), CKPT_PATH)
    print(f"Successfully saved backbone weights to {CKPT_PATH}")

if __name__ == "__main__":
    run_extensive_pretraining()

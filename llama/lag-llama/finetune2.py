import torch
import pandas as pd
import numpy as np
import random
import os
from gluonts.dataset.pandas import PandasDataset
from lag_llama.gluon.estimator import LagLlamaEstimator
from gluonts.torch.distributions import StudentTOutput
import warnings

warnings.filterwarnings("ignore")

# Force torch to use float32 globally to match backbone weights
torch.set_default_dtype(torch.float32)

def seed_everything(seed=42):
    random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed(seed); torch.backends.cudnn.deterministic = True

seed_everything(42)

# --- CONFIG ---
WEIGHTS_PATH = "lag-llama-backbone.ckpt" 
METRICS_CSV = "/root/traffic-shifts/trafpy/trafpy_finetune_normal_data.csv"
FINAL_PATH = "specialized_v11_supervised.pt"

CONTEXT_LEN = 512
MAX_CONTEXT_LEN = 1024
EPOCHS = 30
N_LAYER = 1
N_HEAD = 8
N_EMBD_PER_HEAD = 16

def run_official_finetuning():
    # 1. Load Data and Force Float32 at the Source
    df = pd.read_csv(METRICS_CSV)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['traffic_volume_Tbits'] = df['traffic_volume_Tbits'].astype('float32')
    
    dataset = PandasDataset.from_long_dataframe(
        df,
        target="traffic_volume_Tbits",
        timestamp="timestamp",
        item_id="flow_key_id",
        freq="10min"
    )

    # 2. Initialize Estimator
    estimator = LagLlamaEstimator(
        prediction_length=1,
        context_length=CONTEXT_LEN,
        max_context_length=MAX_CONTEXT_LEN,
        batch_size=64,
        lr=1e-4,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd_per_head=N_EMBD_PER_HEAD,
        scaling="mean",
        time_feat=False,
        trainer_kwargs={
            "max_epochs": EPOCHS,
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "devices": 1,
            "gradient_clip_val": 1.0,
            "precision": "32-true", # Explicitly tell Lightning to use Float32
        },
    )

    # 3. MANUAL OVERRIDE AND WEIGHT LOADING
    print("Initializing network and fixing Student-T projection...")
    lightning_module = estimator.create_lightning_module()
    
    # Ensure the module is in float32 before loading weights
    lightning_module = lightning_module.float()
    
    d_model = N_HEAD * N_EMBD_PER_HEAD 
    lightning_module.model.distr_output = StudentTOutput()
    lightning_module.model.param_proj = lightning_module.model.distr_output.get_args_proj(d_model)

    # Load backbone weights
    state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k if k.startswith("model.") else f"model.{k}"
        new_state_dict[name] = v
        
    lightning_module.load_state_dict(new_state_dict, strict=True)
    print("Backbone weights loaded successfully.")

    # 4. Official Training Pipeline
    print("Starting Official Lag-Llama Finetuning pipeline...")
    # 
    transformation = estimator.create_transformation()
    
    # Double-check: ensure weights are float32 right before training
    lightning_module.model.float()

    predictor = estimator.train_model(
        training_data=dataset, 
        transformation=transformation,
        training_network=lightning_module
    ).predictor

    # 5. Save Specialized Weights
    torch.save(predictor.prediction_net.state_dict(), FINAL_PATH)
    print(f"Specialized weights saved to: {FINAL_PATH}")

if __name__ == "__main__":
    run_official_finetuning()

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
torch.set_default_dtype(torch.float32)

def seed_everything(seed=42):
    random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed(seed); torch.backends.cudnn.deterministic = True

seed_everything(42)

WEIGHTS_PATH = "lag-llama-backbone.ckpt"
METRICS_CSV = "/root/traffic-shifts/trafpy/trafpy_finetune_normal_data.csv"
FINAL_PATH = "specialized_v11_supervised.pt"

CONTEXT_LEN = 512
MAX_CONTEXT_LEN = 1024
EPOCHS = 10 # Literature suggests 5-10 is plenty for specialization
LR = 1e-6   # 1e-4 is too high for finetuning; it destroys pretrained knowledge

def run_official_finetuning():
    df = pd.read_csv(METRICS_CSV)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['traffic_volume_Tbits'] = df['traffic_volume_Tbits'].astype('float32')

    dataset = PandasDataset.from_long_dataframe(
        df, target="traffic_volume_Tbits", timestamp="timestamp",
        item_id="flow_key_id", freq="10min"
    )

    estimator = LagLlamaEstimator(
        prediction_length=1, context_length=CONTEXT_LEN,
        max_context_length=MAX_CONTEXT_LEN, batch_size=64,
        lr=LR, n_layer=1, n_head=8, n_embd_per_head=16,
        scaling="mean", time_feat=False,
        trainer_kwargs={
            "max_epochs": EPOCHS, "accelerator": "gpu", "devices": 1,
            "gradient_clip_val": 1.0, "precision": "32-true",
        },
    )

    print("Initializing network and fixing Student-T projection...")
    lightning_module = estimator.create_lightning_module()
    
    # Re-link Student-T Output
    d_model = 8 * 16
    lightning_module.model.distr_output = StudentTOutput()
    lightning_module.model.param_proj = lightning_module.model.distr_output.get_args_proj(d_model)

    # LOAD FIX: Align keys with the model. prefix
    state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
    new_state_dict = { (k if k.startswith("model.") else f"model.{k}"): v for k, v in state_dict.items() }
    
    lightning_module.load_state_dict(new_state_dict, strict=True)
    print("30-day Backbone weights loaded successfully.")

    transformation = estimator.create_transformation()
    predictor = estimator.train_model(
        training_data=dataset, 
        transformation=transformation,
        training_network=lightning_module
    ).predictor

    torch.save(predictor.prediction_net.state_dict(), FINAL_PATH)
    print(f"Specialized weights saved to: {FINAL_PATH}")

if __name__ == "__main__":
    run_official_finetuning()

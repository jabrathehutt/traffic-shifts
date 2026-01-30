import os
# Force expandable segments to help with the 11GB limit
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import pandas as pd
import numpy as np
import torch
from gluonts.dataset.common import ListDataset
from sklearn.metrics import f1_score, precision_score, recall_score
from lag_llama.gluon.estimator import LagLlamaEstimator
from lag_llama.gluon.lightning_module import LagLlamaLightningModule
from gluonts.torch.distributions import StudentTOutput
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# --- PARAMETERS ---
DATA_FILE = "/root/traffic-shifts/trafpy/trafpy_master_univariate_data.csv"
CKPT_PATH = "specialized_v11_supervised.pt"
CONTEXT_LENGTH = 32
MAX_LAG = 168
NUM_SAMPLES = 100 # Kept at official 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_official_safe_evaluation():
    # 1. Load Data
    df = pd.read_csv(DATA_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['traffic_volume_Tbits'] = df['traffic_volume_Tbits'].astype(np.float32)

    # 2. Setup Module
    module = LagLlamaLightningModule(
        context_length=CONTEXT_LENGTH,
        prediction_length=1,
        model_kwargs={
            "input_size": 1, "context_length": CONTEXT_LENGTH, "max_context_length": 2048,
            "lags_seq": [1, 2, 3, 4, 5, 6, 7, 12, 24, 48, 72, 168],
            "distr_output": StudentTOutput(), "n_layer": 8, "n_embd_per_head": 32,
            "n_head": 8, "scaling": "mean", "time_feat": False,
        }
    )
    
    sd = torch.load(CKPT_PATH, map_location=DEVICE)
    if 'state_dict' in sd: sd = sd['state_dict']
    module.load_state_dict({k.replace('model.', ''): v for k, v in sd.items()}, strict=False)
    module = module.to(DEVICE).eval()

    # 3. Official Predictor with Minimal Batching
    # Set batch_size to 1 to avoid the 5.84GB spike caused by NUM_SAMPLES=100
    estimator = LagLlamaEstimator(
        prediction_length=1, 
        context_length=CONTEXT_LENGTH, 
        batch_size=1 
    )
    predictor = estimator.create_predictor(
        transformation=estimator.create_transformation(), 
        module=module
    )

    # 4. Prepare Rolling Windows
    unique_ids = df['flow_key_id'].unique()[:] 
    historical_windows = []
    
    for fid in unique_ids:
        flow_df = df[df['flow_key_id'] == fid].sort_values('timestamp')
        target = flow_df['traffic_volume_Tbits'].values
        for i in range(MAX_LAG + CONTEXT_LENGTH, len(target)):
            historical_windows.append({
                "start": flow_df['timestamp'].iloc[0],
                "target": target[:i], 
                "item_id": f"{fid}|{i}"
            })

    dataset = ListDataset(historical_windows, freq='10min')

    # 5. Inference
    print(f"Running Official Predictor (Batch=1, Samples=100) on {DEVICE}...")
    all_results = []
    
    with torch.no_grad():
        forecast_it = predictor.predict(dataset, num_samples=NUM_SAMPLES)
        
        for forecast in tqdm(forecast_it, total=len(historical_windows), desc="Sampling"):
            fid, idx_str = forecast.item_id.split('|')
            idx = int(idx_str)
            
            q95 = np.quantile(forecast.samples, 0.95)
            actual_row = df[df['flow_key_id'] == fid].iloc[idx]
            
            all_results.append({
                "true": int(actual_row['is_anomaly']),
                "prediction": 1 if actual_row['traffic_volume_Tbits'] > q95 else 0
            })
            
            # Periodically clear cache to prevent buildup
            if len(all_results) % 100 == 0:
                torch.cuda.empty_cache()

    # 6. Metrics
    res_df = pd.DataFrame(all_results)
    print("\n" + "="*45)
    print(f"F1-Score: {f1_score(res_df['true'], res_df['prediction'], zero_division=0):.4f}")
    print("="*45)

if __name__ == "__main__":
    run_official_safe_evaluation()

import os
# Force memory optimization and fragmentation prevention
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
import gc

warnings.filterwarnings("ignore")

# --- CONFIGURATION (Safe for 11GB VRAM) ---
DATA_FILE = "/root/traffic-shifts/trafpy/trafpy_master_univariate_data.csv"
CKPT_PATH = "specialized_v11_supervised.pt"
CONTEXT_LENGTH = 32
NUM_SAMPLES = 20  # Sufficient for 95th percentile confidence
CHUNK_SIZE = 250   # Smaller chunks for more frequent cache clearing
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_evaluation():
    # 1. Load Data
    df = pd.read_csv(DATA_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['traffic_volume_Tbits'] = df['traffic_volume_Tbits'].astype(np.float32)

    # 2. Init Architecture
    print(f"Loading weights into {DEVICE}...")
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

    # 3. Predictor with MINIMAL internal batching (The OOM Fix)
    estimator = LagLlamaEstimator(
        prediction_length=1, 
        context_length=CONTEXT_LENGTH, 
        batch_size=1  # Sequential processing on GPU to minimize peak memory
    )
    predictor = estimator.create_predictor(estimator.create_transformation(), module)

    # 4. Generate Rolling Windows
    unique_ids = df['flow_key_id'].unique()[:5] 
    all_windows = []
    
    print("Building historical rolling windows...")
    for fid in unique_ids:
        flow_df = df[df['flow_key_id'] == fid].sort_values('timestamp')
        target = flow_df['traffic_volume_Tbits'].values
        for i in range(CONTEXT_LENGTH, len(target)):
            all_windows.append({
                "start": flow_df['timestamp'].iloc[0],
                "target": target[:i], 
                "item_id": f"{fid}|{i}"
            })

    # 5. Safe Inference Loop
    all_results = []
    print(f"Executing inference on {len(all_windows)} points...")
    
    # We use a progress bar over the total window count
    pbar = tqdm(total=len(all_windows), desc="Analyzing Traffic")
    
    for i in range(0, len(all_windows), CHUNK_SIZE):
        chunk = all_windows[i:i + CHUNK_SIZE]
        dataset = ListDataset(chunk, freq='10min')
        
        with torch.no_grad():
            forecast_it = predictor.predict(dataset, num_samples=NUM_SAMPLES)
            for forecast in forecast_it:
                fid, idx_str = forecast.item_id.split('|')
                idx = int(idx_str)
                
                # Probabilistic 95% threshold from samples
                q95 = np.quantile(forecast.samples, 0.95)
                row = df[df['flow_key_id'] == fid].iloc[idx]
                
                all_results.append({
                    "actual": row['traffic_volume_Tbits'],
                    "limit": q95,
                    "is_anomaly": row['is_anomaly']
                })
                pbar.update(1)
        
        # Aggressive memory cleanup
        torch.cuda.empty_cache()
        gc.collect()

    pbar.close()

    # 6. Metrics Report
    res_df = pd.DataFrame(all_results)
    res_df['y_pred'] = res_df['actual'] > res_df['limit']
    
    print("\n" + "="*45)
    print("THESIS PERFORMANCE METRICS")
    print("-" * 45)
    print(f"F1 Score:  {f1_score(res_df['is_anomaly'], res_df['y_pred'], zero_division=0):.4f}")
    print(f"Precision: {precision_score(res_df['is_anomaly'], res_df['y_pred'], zero_division=0):.4f}")
    print(f"Recall:    {recall_score(res_df['is_anomaly'], res_df['y_pred'], zero_division=0):.4f}")
    print(f"Points:    {len(res_df)}")
    print("="*45)

if __name__ == "__main__":
    run_evaluation()

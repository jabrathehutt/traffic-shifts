import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import pandas as pd
import numpy as np
import torch
from gluonts.dataset.common import ListDataset
from gluonts.dataset.loader import InferenceDataLoader
from gluonts.torch.batchify import batchify
from sklearn.metrics import f1_score, precision_score, recall_score
from lag_llama.gluon.estimator import LagLlamaEstimator
from lag_llama.gluon.lightning_module import LagLlamaLightningModule
from gluonts.torch.distributions import StudentTOutput
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# --- OFFICIAL SPECS ---
DATA_FILE = "/root/traffic-shifts/trafpy/trafpy_master_univariate_data.csv"
CKPT_PATH = "specialized_v11_supervised.pt"
CONTEXT_LENGTH = 32
MAX_LAG = 168
BATCH_SIZE = 64 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_evaluation():
    # 1. Load Data
    df = pd.read_csv(DATA_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['traffic_volume_Tbits'] = df['traffic_volume_Tbits'].astype(np.float32)

    # 2. Initialize Official Module
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

    # 3. Setup Official Transformation
    estimator = LagLlamaEstimator(
        prediction_length=1, context_length=CONTEXT_LENGTH, batch_size=BATCH_SIZE
    )
    transformation = estimator.create_transformation()

    # 4. Prepare Dataset (Full 50 Flows)
    unique_ids = df['flow_key_id'].unique()[:50]
    historical_entries = []
    
    print(f"Building official rolling windows for {len(unique_ids)} flows...")
    for fid in unique_ids:
        flow_df = df[df['flow_key_id'] == fid].sort_values('timestamp')
        target_values = flow_df['traffic_volume_Tbits'].values
        needed_len = MAX_LAG + CONTEXT_LENGTH
        
        for i in range(needed_len, len(target_values)):
            historical_entries.append({
                "start": flow_df['timestamp'].iloc[0],
                "target": target_values[i - needed_len : i],
                "item_id": f"{fid}|{i}"
            })

    dataset = ListDataset(historical_entries, freq='10min')

    # 5. Official Inference via DataLoader
    loader = InferenceDataLoader(
        dataset,
        transform=transformation,
        batch_size=BATCH_SIZE,
        stack_fn=lambda x: batchify(x, DEVICE)
    )

    all_results = []
    print(f"Analyzing {len(historical_entries)} points on {DEVICE}...")

    with torch.no_grad():
        for batch in tqdm(loader, total=len(historical_entries)//BATCH_SIZE):
            # ROBUST KEY MAPPING: Look for whatever name GluonTS gave the target tensor
            p_target = None
            for key in ["past_target", "past_target_cdf", "target"]:
                if key in batch:
                    p_target = batch[key]
                    break
            
            p_obs = None
            for key in ["past_observed_values", "observed_values"]:
                if key in batch:
                    p_obs = batch[key]
                    break

            # If the keys are still missing, we inspect the batch to find the 2D/3D tensors
            if p_target is None:
                # Fallback to the first tensor that looks like target data
                p_target = next(v for k, v in batch.items() if isinstance(v, torch.Tensor) and v.ndim >= 2)
            if p_obs is None:
                p_obs = torch.ones_like(p_target)

            # Official forward pass
            distr_args, loc, scale = module.model(
                past_target=p_target,
                past_observed_values=p_obs,
                future_target=None
            )
            
            # Official analytical quantile calculation
            distr = module.model.distr_output.distribution(distr_args, loc, scale)
            q95 = distr.quantile(torch.tensor([0.95], device=DEVICE))[:, -1]
            q95_np = q95.cpu().numpy()
            
            for k, item_id in enumerate(batch["item_id"]):
                fid, idx_str = item_id.split('|')
                idx = int(idx_str)
                actual_val = df[df['flow_key_id'] == fid].iloc[idx]
                
                all_results.append({
                    'true': int(actual_val['is_anomaly']),
                    'pred': 1 if actual_val['traffic_volume_Tbits'] > q95_np[k] else 0
                })

    # 6. Final Report
    res_df = pd.DataFrame(all_results)
    print("\n" + "="*45)
    print("OFFICIAL HIGH-SPEED SUMMARY")
    print("-" * 45)
    print(f"F1 Score:  {f1_score(res_df['true'], res_df['pred'], zero_division=0):.4f}")
    print(f"Points:    {len(res_df)}")
    print("="*45)
    
    res_df.to_csv("lag_llama_official_results.csv", index=False)

if __name__ == "__main__":
    run_evaluation()

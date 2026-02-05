import torch
import pandas as pd
import numpy as np
import os
import random
import time
import gc
from gluonts.dataset.pandas import PandasDataset
from lag_llama.gluon.estimator import LagLlamaEstimator
from lag_llama.gluon.lightning_module import LagLlamaLightningModule
from gluonts.torch.distributions import StudentTOutput
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def seed_everything(seed=42):
    random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

seed_everything(42)

# --- CONFIG ---
CSV_PATH = "/root/traffic-shifts/trafpy/trafpy_master_univariate_data.csv"
CKPT_PATH = "/root/traffic-shifts/llama/lag-llama/models/finetune_latest.ckpt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONTEXT_LENGTH = 128
NUM_SAMPLES = 100 
QUANTILE_THRESHOLD = 0.975
SAMPLING_LAG_MINS = 2.5
INF_BATCH_SIZE = 8 # Safe for 11GB VRAM

def run_verified_evaluation():
    df = pd.read_csv(CSV_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['traffic_volume_Tbits'] = df['traffic_volume_Tbits'].astype('float32')

    # 1. Load Model with Surgery
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint['state_dict']
    ckpt_wte_weight = state_dict['model.transformer.wte.weight']
    ckpt_in_dim, embedding_dim = ckpt_wte_weight.shape[1], ckpt_wte_weight.shape[0]

    lags_seq = list(range(1, 96))
    module = LagLlamaLightningModule(context_length=CONTEXT_LENGTH, max_context_length=2048, prediction_length=1,
        model_kwargs={"context_length": CONTEXT_LENGTH, "max_context_length": 2048, "n_layer": 1, "n_head": 8, 
        "n_embd_per_head": 16, "scaling": "mean", "time_feat": False, "input_size": 1, "distr_output": StudentTOutput(), "lags_seq": lags_seq})

    module.to(DEVICE)
    module.model.transformer.wte = torch.nn.Linear(ckpt_in_dim, embedding_dim, bias=True).to(DEVICE)
    
    estimator = LagLlamaEstimator(prediction_length=1, context_length=CONTEXT_LENGTH, batch_size=INF_BATCH_SIZE, 
                                 trainer_kwargs={"accelerator": "cuda", "devices": 1})
    predictor = estimator.create_predictor(estimator.create_transformation(), module)

    with torch.no_grad():
        module.model.transformer.wte.weight.copy_(ckpt_wte_weight)
    module.load_state_dict({k: v for k, v in state_dict.items() if "wte" not in k}, strict=False)
    module.eval()

    all_y_true, all_y_pred, detection_delays, all_inf_lats = [], [], [], []
    unique_flows = df['flow_key_id'].unique()

    for flow_id in unique_flows:
        flow_df = df[df['flow_key_id'] == flow_id].reset_index(drop=True)
        start_idx = CONTEXT_LENGTH + 60
        
        # Prepare windows
        windows = [flow_df.iloc[:i].set_index('timestamp')['traffic_volume_Tbits'] for i in range(start_idx, len(flow_df))]
        flow_ds = PandasDataset(windows, freq="5min")
        
        flow_y_true, flow_y_pred = [], []
        print(f"Streaming Inference for {flow_id}...")
        t_start_flow = time.time()
        
        with torch.no_grad():
            forecast_it = predictor.predict(flow_ds, num_samples=NUM_SAMPLES)
            for idx, forecast in enumerate(tqdm(forecast_it, total=len(windows))):
                # FIXED: Map idx back to absolute flow_df index
                actual_df_idx = start_idx + idx 
                
                samples = forecast.samples
                upper = np.quantile(samples, q=QUANTILE_THRESHOLD)
                
                label = flow_df['is_anomaly'].iloc[actual_df_idx]
                actual = flow_df['traffic_volume_Tbits'].iloc[actual_df_idx]
                pred = 1 if actual > upper else 0
                
                flow_y_true.append(label)
                flow_y_pred.append(pred)

                if idx % 100 == 0:
                    gc.collect(); torch.cuda.empty_cache()

        t_end_flow = time.time()
        avg_latency = (t_end_flow - t_start_flow) / len(windows)
        all_inf_lats.append(avg_latency)
        comp_lag_mins = avg_latency / 60.0

        all_y_true.extend(flow_y_true)
        all_y_pred.extend(flow_y_pred)

        # FIXED DELAY LOGIC: Corrected relative index mapping
        curr_event_start = None
        detected = False
        for i in range(len(flow_y_true)):
            if flow_y_true[i] == 1 and (i == 0 or flow_y_true[i-1] == 0):
                curr_event_start = i
                detected = False
            if flow_y_true[i] == 1 and flow_y_pred[i] == 1 and not detected:
                # Algorithm Lag is based on how many 5-min buckets passed since the event start
                algo_lag_mins = (i - curr_event_start) * 5.0
                detection_delays.append(SAMPLING_LAG_MINS + algo_lag_mins + comp_lag_mins)
                detected = True
            if flow_y_true[i] == 0:
                curr_event_start, detected = None, False

        torch.cuda.empty_cache()

    y_t, y_p = np.array(all_y_true), np.array(all_y_pred)
    print("\n" + "="*45)
    print(f"PRECISION: {precision_score(y_t, y_p):.4f} | RECALL: {recall_score(y_t, y_p):.4f}")
    print(f"F1-SCORE:  {f1_score(y_t, y_p):.4f} | AVG DELAY: {np.mean(detection_delays):.4f} mins")
    print(f"INFERENCE LATENCY: {np.mean(all_inf_lats):.6f}")
    print("="*45)

if __name__ == "__main__":
    run_verified_evaluation()

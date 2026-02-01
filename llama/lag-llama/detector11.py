import torch
import pandas as pd
import numpy as np
import os
import random
from gluonts.dataset.pandas import PandasDataset
from lag_llama.gluon.estimator import LagLlamaEstimator, LagLlamaLightningModule
from gluonts.torch.distributions import StudentTOutput
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

def seed_everything(seed=42):
    random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed(seed); torch.backends.cudnn.deterministic = True

seed_everything(42)

# --- CONFIG ---
CSV_PATH = "/root/traffic-shifts/trafpy/trafpy_master_univariate_data.csv"
CKPT_PATH = "specialized_v11_supervised.pt" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONTEXT_LENGTH = 512
MAX_CONTEXT_LENGTH = 1024
PREDICTION_LENGTH = 1
NUM_SAMPLES = 100 

# --- ADAPTIVE PRECISION TUNING ---
# Instead of a fixed percentile, we use Mean + K * StdDev of the samples.
# In a Student-T distribution, this captures the 'Heavy Tails' better.
K_SIGMA = 5.0 # Increase this to 6.0 or 7.0 if False Positives persist

def run_adaptive_detection():
    df = pd.read_csv(CSV_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['traffic_volume_Tbits'] = df['traffic_volume_Tbits'].astype('float32')

    module = LagLlamaLightningModule(
        context_length=CONTEXT_LENGTH,
        max_context_length=MAX_CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        model_kwargs={
            "context_length": CONTEXT_LENGTH, "max_context_length": MAX_CONTEXT_LENGTH,
            "n_layer": 1, "n_head": 8, "n_embd_per_head": 16, 
            "scaling": "mean", "time_feat": False, "input_size": 1,
            "distr_output": StudentTOutput(), "lags_seq": list(range(1, 85))
        }
    )

    state_dict = torch.load(CKPT_PATH, map_location=DEVICE)
    if any(k.startswith('model.') for k in state_dict.keys()):
        module.load_state_dict(state_dict)
    else:
        module.model.load_state_dict(state_dict)
    module.to(DEVICE).float().eval()

    estimator = LagLlamaEstimator(
        prediction_length=PREDICTION_LENGTH,
        context_length=CONTEXT_LENGTH,
        batch_size=64,
        device=DEVICE
    )
    predictor = estimator.create_predictor(estimator.create_transformation(), module)

    all_y_true, all_y_pred = [], []
    full_series = df.set_index('timestamp')['traffic_volume_Tbits']
    
    print(f"Executing Adaptive Spread Detection (K-Sigma: {K_SIGMA})...")

    for i in tqdm(range(len(df))):
        window_series = full_series.iloc[:i]
        if len(window_series) == 0: continue

        actual_val = df['traffic_volume_Tbits'].iloc[i]
        actual_label = df['is_anomaly'].iloc[i]
        
        window_dataset = PandasDataset(window_series, freq="10min")
        
        forecast_it = predictor.predict(window_dataset, num_samples=NUM_SAMPLES)
        forecast = list(forecast_it)[0]
        
        # --- ADAPTIVE SPREAD LOGIC ---
        # We calculate the mean and std of the 100 'potential futures'
        samples = forecast.samples.flatten()
        pred_mean = np.mean(samples)
        pred_std = np.std(samples)
        
        # Upper bound expands if the model is uncertain (high pred_std)
        # This is key for bursty network traffic
        upper_bound = pred_mean + (K_SIGMA * pred_std)
        
        prediction = 1 if actual_val > upper_bound else 0
        
        all_y_true.append(actual_label)
        all_y_pred.append(prediction)

    # --- FINAL METRICS ---
    precision = precision_score(all_y_true, all_y_pred, zero_division=0)
    recall = recall_score(all_y_true, all_y_pred, zero_division=0)
    f1 = f1_score(all_y_true, all_y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(all_y_true, all_y_pred).ravel()

    print("\n" + "="*45)
    print(f"ADAPTIVE SPREAD RESULTS (No Delay)")
    print("-" * 45)
    print(f"PRECISION: {precision:.4f} | RECALL: {recall:.4f} | F1: {f1:.4f}")
    print(f"TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")
    print("="*45)

if __name__ == "__main__":
    run_adaptive_detection()

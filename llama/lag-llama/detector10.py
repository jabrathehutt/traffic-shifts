import torch
import pandas as pd
import numpy as np
import os
import random
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from lag_llama.gluon.estimator import LagLlamaLightningModule
from gluonts.torch.distributions import StudentTOutput
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

# --- CONFIGURATION ---
CSV_PATH = "/root/traffic-shifts/trafpy/trafpy_master_univariate_data.csv"
CKPT_PATH = "specialized_v11_supervised.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONTEXT_LENGTH = 128
NUM_SAMPLES = 100

# --- THE "ELITE" CALIBRATION ---
SIGMA_THRESHOLD = 3.2    # High sensitivity to capture the start of drifts
LP_THRESHOLD = -2.8       # Moderate surprise gate
INTEGRAL_WINDOW = 3       # Number of steps to accumulate "Error Energy"
ENERGY_THRESHOLD = 8.5    # Cumulative Z-score required to confirm a stealthy drift

SMA_WINDOW = 3
LAGS_SEQ = list(range(1, 85))

def run_evaluation():
    full_df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
    full_df = full_df.sort_values(["flow_key_id", "timestamp"])

    module = LagLlamaLightningModule(
        context_length=CONTEXT_LENGTH, prediction_length=1,
        model_kwargs={
            "context_length": CONTEXT_LENGTH, "max_context_length": 1024,
            "input_size": 1, "distr_output": StudentTOutput(),
            "n_layer": 1, "n_head": 8, "n_embd_per_head": 16,
            "lags_seq": LAGS_SEQ, "scaling": "mean", "time_feat": False,
        },
    )

    sd = torch.load(CKPT_PATH, map_location=DEVICE)
    module.model.load_state_dict({k.replace('model.', ''): v for k, v in sd['state_dict'].items()}, strict=False)
    model = module.model.to(DEVICE).eval()

    all_y_true, all_y_pred = [], []
    unique_flows = full_df["flow_key_id"].unique()[:50]
    
    print(f"Executing Bayesian Residual Integration Evaluator...")

    for flow_id in tqdm(unique_flows):
        flow_data = full_df[full_df["flow_key_id"] == flow_id].copy()
        flow_data["smoothed"] = flow_data["traffic_volume_Tbits"].rolling(window=SMA_WINDOW).mean().fillna(method='bfill')

        values = flow_data["smoothed"].astype("float32").values
        labels = flow_data["is_anomaly"].values
        if len(values) <= CONTEXT_LENGTH: continue

        indices = range(CONTEXT_LENGTH, len(values))
        windows = np.array([values[i-CONTEXT_LENGTH:i] for i in indices])
        actuals = values[indices]
        gt_labels = labels[indices]

        # Flow-specific trackers
        z_history = []
        lp_history = []

        batch_size = 64
        for b in range(0, len(windows), batch_size):
            batch_win = torch.tensor(windows[b : b + batch_size], device=DEVICE)

            with torch.no_grad():
                scale = batch_win.mean(dim=1, keepdim=True) + 1e-5
                distr_args, loc, scale_p = model(
                    past_target=batch_win / scale,
                    past_observed_values=torch.ones_like(batch_win),
                )

                df_v, mu_v, sigma_v = distr_args[0][:, -1], loc[:, -1], scale_p[:, -1]
                distr = torch.distributions.StudentT(df_v, mu_v, sigma_v)
                y_norm = torch.tensor(actuals[b : b + batch_size], device=DEVICE) / scale.squeeze()
                log_probs = distr.log_prob(y_norm).cpu().numpy()

                torch.manual_seed(42)
                samples = distr.sample((NUM_SAMPLES,)).cpu().numpy()
                rescaled_samples = samples * scale.squeeze().cpu().numpy()
                pred_mean = np.mean(rescaled_samples, axis=0)
                pred_std = np.std(rescaled_samples, axis=0)

                for idx in range(len(log_probs)):
                    z = (actuals[b + idx] - pred_mean[idx]) / (pred_std[idx] + 1e-6)
                    z_history.append(max(0, z)) # Only care about upward shifts
                    lp_history.append(log_probs[idx])
                    
                    # CUMULATIVE ENERGY CALCULATION
                    if len(z_history) >= INTEGRAL_WINDOW:
                        energy = sum(z_history[-INTEGRAL_WINDOW:])
                        avg_lp = sum(lp_history[-INTEGRAL_WINDOW:]) / INTEGRAL_WINDOW
                    else:
                        energy = 0
                        avg_lp = 0

                    # HYBRID TRIGGER:
                    # 1. High single-point surprise (Spikes) OR
                    # 2. High cumulative energy + moderate surprise (Drifts)
                    is_spike = (z > SIGMA_THRESHOLD + 2.0) and (log_probs[idx] < LP_THRESHOLD - 1.0)
                    is_drift = (energy > ENERGY_THRESHOLD) and (avg_lp < LP_THRESHOLD)
                    
                    is_anomaly = is_spike or is_drift
                    
                    all_y_pred.append(1 if is_anomaly else 0)
                    all_y_true.append(gt_labels[b + idx])

    p = precision_score(all_y_true, all_y_pred, zero_division=0)
    r = recall_score(all_y_true, all_y_pred, zero_division=0)
    f1 = f1_score(all_y_true, all_y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(all_y_true, all_y_pred).ravel()

    print("\n" + "="*45)
    print(f"RESIDUAL INTEGRATION RESULTS")
    print("-" * 45)
    print(f"PRECISION: {p:.4f} | RECALL: {r:.4f} | F1: {f1:.4f}")
    print(f"TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")
    print("="*45)

if __name__ == "__main__":
    run_evaluation()

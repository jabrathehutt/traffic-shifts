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

CONTEXT_LENGTH = 96
# HYBRID PARAMS
LOG_PROB_THRESHOLD = -5.5 
Z_MAGNITUDE_THRESHOLD = 4.5
SMA_WINDOW = 3
LAGS_SEQ = list(range(1, 85))

def run_evaluation():
    full_df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
    full_df = full_df.sort_values(["flow_key_id", "timestamp"])

    # Official Model Rebuild
    module = LagLlamaLightningModule(
        context_length=CONTEXT_LENGTH,
        prediction_length=1,
        model_kwargs={
            "context_length": CONTEXT_LENGTH, "max_context_length": 1024,
            "input_size": 1, "distr_output": StudentTOutput(),
            "n_layer": 1, "n_head": 8, "n_embd_per_head": 16,
            "lags_seq": LAGS_SEQ, "scaling": "mean", "time_feat": False,
        },
    )

    sd = torch.load(CKPT_PATH, map_location=DEVICE)
    if 'state_dict' in sd: sd = sd['state_dict']
    module.model.load_state_dict({k.replace('model.', ''): v for k, v in sd.items()}, strict=False)
    model = module.model.to(DEVICE).eval()

    all_y_true, all_y_pred = [], []
    unique_flows = full_df["flow_key_id"].unique()[:50]
    
    print(f"Executing Deterministic Hybrid Surprisal Evaluator...")

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

        flow_preds = []
        batch_size = 64
        for b in range(0, len(windows), batch_size):
            batch_win = torch.tensor(windows[b : b + batch_size], device=DEVICE)
            with torch.no_grad():
                # Stabilized Scaling
                scale = batch_win.median(dim=1, keepdim=True)[0] + 1e-5
                distr_args, loc, scale_p = model(past_target=batch_win/scale, past_observed_values=torch.ones_like(batch_win))
                
                df, loc_f, scale_f = distr_args[0][:, -1], loc[:, -1], scale_p[:, -1]
                
                b_actuals = torch.tensor(actuals[b : b + batch_size], device=DEVICE)
                scaled_actuals = b_actuals / scale.squeeze()
                
                # Distribution Evaluation
                distr = torch.distributions.StudentT(df, loc_f, scale_f)
                
                # 1. Log-Probability (Analytical Surprise)
                lp = distr.log_prob(scaled_actuals)
                
                # 2. Standardized Z-Score (Magnitude)
                # Since we can't use .std() analytically on StudentT parameters easily, 
                # we use the theoretical variance of StudentT: var = scale^2 * (df / (df - 2))
                theoretical_std = scale_f * torch.sqrt(df / (df - 2.0 + 1e-6))
                z_score = (scaled_actuals - loc_f) / (theoretical_std + 1e-6)
                
                # LOGIC: Surprise AND Magnitude AND Upward Spike
                batch_preds = ((lp < LOG_PROB_THRESHOLD) & 
                               (z_score > Z_MAGNITUDE_THRESHOLD) & 
                               (scaled_actuals > loc_f)).int().cpu().numpy()
                flow_preds.extend(batch_preds)

        all_y_true.extend(gt_labels)
        all_y_pred.extend(flow_preds)

    precision = precision_score(all_y_true, all_y_pred, zero_division=0)
    recall = recall_score(all_y_true, all_y_pred, zero_division=0)
    f1 = f1_score(all_y_true, all_y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(all_y_true, all_y_pred).ravel()

    print("\n" + "="*45)
    print(f"HYBRID SURPRISAL RESULTS")
    print("-" * 45)
    print(f"PRECISION: {precision:.4f} | RECALL: {recall:.4f} | F1: {f1:.4f}")
    print(f"TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")
    print("="*45)

if __name__ == "__main__":
    run_evaluation()

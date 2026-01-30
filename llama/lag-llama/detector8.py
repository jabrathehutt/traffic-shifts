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

# --- 1. GLOBAL DETERMINISM SETUP ---
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- CONFIGURATION ---
CSV_PATH = "/root/traffic-shifts/trafpy/trafpy_master_univariate_data.csv"
CKPT_PATH = "specialized_v11_supervised.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONTEXT_LENGTH = 512 # Matches your new trained context length
NUM_SAMPLES = 100
SIGMA_THRESHOLD = 6.0
SMA_WINDOW = 3
LAGS_SEQ = list(range(1, 85))

def run_evaluation():
    full_df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
    full_df = full_df.sort_values(["flow_key_id", "timestamp"])

    # 1. Official Model Setup
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

    state_dict = torch.load(CKPT_PATH, map_location=DEVICE)
    if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
    module.model.load_state_dict({k.replace('model.', ''): v for k, v in state_dict.items()}, strict=False)
    model = module.model.to(DEVICE).eval()

    all_y_true = []
    all_y_pred = []

    unique_flows = full_df["flow_key_id"].unique()[:50]
    print(f"Executing Deterministic Smoothed Evaluator (SMA-{SMA_WINDOW}, Sigma-{SIGMA_THRESHOLD})...")

    for flow_id in tqdm(unique_flows):
        flow_data = full_df[full_df["flow_key_id"] == flow_id].copy()
        flow_data["smoothed_volume"] = flow_data["traffic_volume_Tbits"].rolling(window=SMA_WINDOW).mean().fillna(method='bfill')

        values = flow_data["smoothed_volume"].astype("float32").values
        labels = flow_data["is_anomaly"].values

        if len(values) <= CONTEXT_LENGTH: continue

        indices = range(CONTEXT_LENGTH, len(values))
        windows = np.array([values[i-CONTEXT_LENGTH:i] for i in indices])
        actuals = values[indices]
        gt_labels = labels[indices]

        batch_size = 64
        for b in range(0, len(windows), batch_size):
            batch_win = torch.tensor(windows[b : b + batch_size], device=DEVICE)

            with torch.no_grad():
                scale = batch_win.mean(dim=1, keepdim=True) + 1e-5
                distr_args, _, _ = model(
                    past_target=batch_win / scale,
                    past_observed_values=torch.ones_like(batch_win),
                )

                df_p, loc_p, scale_p = distr_args[0][:, -1], distr_args[1][:, -1], distr_args[2][:, -1]
                distr = torch.distributions.StudentT(df_p, loc_p, scale_p)
                
                # --- 2. SAMPLING DETERMINISM ---
                # We seed PyTorch right before sampling to ensure the noise 
                # generator starts from the same state for every batch.
                torch.manual_seed(42) 
                samples = distr.sample((NUM_SAMPLES,))

                rescaled_samples = samples.cpu().numpy() * scale.squeeze().cpu().numpy()
                pred_mean = np.mean(rescaled_samples, axis=0)
                pred_std = np.std(rescaled_samples, axis=0)

                batch_actuals = actuals[b : b + batch_size]
                batch_gt = gt_labels[b : b + batch_size]

                for idx in range(len(batch_actuals)):
                    z_score = (batch_actuals[idx] - pred_mean[idx]) / (pred_std[idx] + 1e-6)
                    pred = 1 if z_score > SIGMA_THRESHOLD else 0
                    all_y_true.append(batch_gt[idx])
                    all_y_pred.append(pred)

    precision = precision_score(all_y_true, all_y_pred, zero_division=0)
    recall = recall_score(all_y_true, all_y_pred, zero_division=0)
    f1 = f1_score(all_y_true, all_y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(all_y_true, all_y_pred).ravel()

    print("\n" + "="*45)
    print(f"DETERMINISTIC RESULTS (SMA-{SMA_WINDOW}, Sigma: {SIGMA_THRESHOLD})")
    print("-" * 45)
    print(f"PRECISION: {precision:.4f} | RECALL: {recall:.4f} | F1: {f1:.4f}")
    print(f"TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")
    print("="*45)

if __name__ == "__main__":
    run_evaluation()

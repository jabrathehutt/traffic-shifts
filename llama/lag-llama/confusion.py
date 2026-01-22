import torch, numpy as np, pandas as pd, os
import matplotlib.pyplot as plt
import seaborn as sns
from lag_llama.gluon.estimator import LagLlamaLightningModule
from gluonts.torch.distributions import StudentTOutput
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
METRICS_CSV = "/root/traffic-shifts/trafpy/trafpy_master_univariate_data.csv"
MODEL_PATH = "specialized_v11_supervised.pt"

CONTEXT_LEN = 96
N_LAYER = 1
N_HEAD = 8
LAGS_SEQ = list(range(1, 85)) 
BATCH_SIZE = 128

def apply_m_of_n_filter(scores, threshold, m, n):
    raw_preds = (scores > threshold).astype(int)
    flex_preds = np.zeros_like(raw_preds)
    for i in range(len(raw_preds)):
        if i >= n - 1:
            window = raw_preds[i-(n-1) : i+1]
            if np.sum(window) >= m:
                flex_preds[i] = 1
    return flex_preds

def run_evaluation_and_plot():
    df = pd.read_csv(METRICS_CSV)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 1. Load Model
    module = LagLlamaLightningModule(
        context_length=CONTEXT_LEN, prediction_length=1,
        model_kwargs={
            "context_length": CONTEXT_LEN, "max_context_length": 1024, "input_size": 1,
            "distr_output": StudentTOutput(), "n_layer": N_LAYER, "n_head": N_HEAD,
            "n_embd_per_head": 16, "lags_seq": LAGS_SEQ, "scaling": "mean", "time_feat": False,
        }
    )
    sd = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    module.model.load_state_dict({k.replace("model.", ""): v for k, v in sd.items()}, strict=True)
    model = module.model.to(DEVICE).eval()

    # 2. Prepare Data
    target_flow = df['flow_key_id'].unique()[0]
    flow_df = df[df['flow_key_id'] == target_flow].sort_values('timestamp')
    vol_col = 'traffic_volume_Tbits' if 'traffic_volume_Tbits' in flow_df.columns else 'traffic_volume_Gbits'
    v = flow_df[vol_col].values.astype('float32')
    y_true = (flow_df['is_anomaly'].values[CONTEXT_LEN:] == 1).astype(int)
    timestamps = flow_df['timestamp'].values[CONTEXT_LEN:]

    # 3. Inference
    z_scores = []
    print("Generating Z-Scores...")
    for i in tqdm(range(CONTEXT_LEN, len(v), BATCH_SIZE)):
        batch_end = min(i + BATCH_SIZE, len(v))
        windows = [v[j-CONTEXT_LEN:j] for j in range(i, batch_end)]
        x_batch = torch.tensor(np.array(windows)).to(DEVICE)
        y_batch = v[i:batch_end]

        with torch.no_grad():
            scale_f = x_batch.mean(dim=1, keepdim=True) + 1e-5
            distr_args, _, _ = model(past_target=x_batch/scale_f, past_observed_values=torch.ones_like(x_batch).to(DEVICE))
            p_loc = (distr_args[0][:, -1] * scale_f.squeeze(-1)).cpu().numpy()
            p_scale = (torch.exp(distr_args[1][:, -1]) * scale_f.squeeze(-1)).cpu().numpy()
            z_scores.extend(np.abs(p_loc - y_batch) / (p_scale + 1e-10))

    z_array = np.array(z_scores)

    # 4. Use your best configuration found previously
    # Threshold: 2.05, M: 3, N: 5
    best_t, best_m, best_n = 2.052632, 3, 5
    y_pred = apply_m_of_n_filter(z_array, best_t, best_m, best_n)

    # 5. Plotting
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2)
    
    # Subplot: Confusion Matrix
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title(f"Confusion Matrix ({best_m}-of-{best_n} Filter)")
    ax1.set_xlabel("Predicted Label")
    ax1.set_ylabel("True Label")
    ax1.set_xticklabels(['Normal', 'Anomaly'])
    ax1.set_yticklabels(['Normal', 'Anomaly'])

    # Subplot: Timeline
    ax2 = fig.add_subplot(gs[1, :])
    v_window = v[CONTEXT_LEN:]
    ax2.plot(timestamps, v_window, label='Traffic Volume', color='gray', alpha=0.4)
    ax2.fill_between(timestamps, 0, v_window.max(), where=(y_true==1), 
                     color='red', alpha=0.15, label='True Anomaly Window')
    
    # Correct detections vs Misses
    hits = (y_true == 1) & (y_pred == 1)
    misses = (y_true == 1) & (y_pred == 0)
    
    ax2.scatter(timestamps[hits], v_window[hits], color='green', s=20, label='True Positives (Correct)')
    ax2.scatter(timestamps[misses], v_window[misses], color='orange', s=20, label='False Negatives (Missed)')
    
    ax2.set_title(f"Detection Timeline for {target_flow}")
    ax2.legend()

    plt.tight_layout()
    plt.savefig('confusion_analysis.png')
    print("Saved confusion_analysis.png")

if __name__ == "__main__":
    run_evaluation_and_plot()

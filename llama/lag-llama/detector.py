import torch, numpy as np, pandas as pd, os
from lag_llama.gluon.estimator import LagLlamaLightningModule
from gluonts.torch.distributions import StudentTOutput
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
METRICS_CSV = "/root/traffic/trafpy/trafpy_master_univariate_data.csv"
MODEL_PATH = "specialized_v11_supervised.pt"
PLOT_OUTPUT = "detection_latency_analysis.png"

# Architecture parameters
CONTEXT_LEN = 96
N_LAYER = 1
N_HEAD = 8
LAGS_SEQ = list(range(1, 85)) 
BATCH_SIZE = 128
FREQ_MIN = 10 # 10-minute intervals

def apply_persistence_filter(scores, threshold, k=3):
    """Only alerts if k consecutive points are above threshold."""
    raw_preds = (scores > threshold).astype(int)
    smoothed_preds = np.zeros_like(raw_preds)
    for i in range(len(raw_preds)):
        if i >= k - 1:
            if np.all(raw_preds[i-(k-1) : i+1] == 1):
                smoothed_preds[i] = 1
    return smoothed_preds

def calculate_ttd(y_true, y_pred, freq_min):
    """Calculates the time delay between anomaly start and first detection."""
    anomaly_indices = np.where(y_true == 1)[0]
    if len(anomaly_indices) == 0: return None
    
    first_true = anomaly_indices[0]
    # Find first detection at or after the anomaly start
    detections = np.where(y_pred[first_true:] == 1)[0]
    
    if len(detections) > 0:
        delay_intervals = detections[0]
        return delay_intervals * freq_min
    return None

def run_evaluation_with_latency():
    if not os.path.exists(METRICS_CSV):
        print(f"Error: {METRICS_CSV} not found."); return

    df = pd.read_csv(METRICS_CSV)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 1. Initialize Model
    module = LagLlamaLightningModule(
        context_length=CONTEXT_LEN, prediction_length=1,
        model_kwargs={
            "context_length": CONTEXT_LEN, "max_context_length": 1024, "input_size": 1,
            "distr_output": StudentTOutput(), "n_layer": N_LAYER, "n_head": N_HEAD,
            "n_embd_per_head": 16, "lags_seq": LAGS_SEQ, "scaling": "mean", "time_feat": False,
        }
    )

    # 2. Load Weights
    print(f"Loading weights: {MODEL_PATH}...")
    sd = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    clean_sd = {k.replace("model.", ""): v for k, v in sd.items()}
    module.model.load_state_dict(clean_sd, strict=True)
    model = module.model.to(DEVICE).eval()

    # 3. Process Data
    target_flow = df['flow_key_id'].unique()[0]
    flow_df = df[df['flow_key_id'] == target_flow].sort_values('timestamp')
    vol_col = 'traffic_volume_Tbits' if 'traffic_volume_Tbits' in flow_df.columns else 'traffic_volume_Gbits'
    
    v = flow_df[vol_col].values.astype('float32')
    labels = flow_df['is_anomaly'].values
    
    windows_x, targets_y, y_true = [], [], []
    for i in range(CONTEXT_LEN, len(v)):
        windows_x.append(v[i-CONTEXT_LEN:i])
        targets_y.append(v[i])
        y_true.append(1 if labels[i] else 0)

    # 4. Inference
    all_z_scores = []
    print("Calculating Z-Scores...")
    for i in tqdm(range(0, len(windows_x), BATCH_SIZE)):
        batch_end = min(i + BATCH_SIZE, len(windows_x))
        x_batch = torch.tensor(np.array(windows_x[i:batch_end])).to(DEVICE)
        y_batch = np.array(targets_y[i:batch_end])

        with torch.no_grad():
            scale_f = x_batch.mean(dim=1, keepdim=True) + 1e-5
            distr_args, _, _ = model(past_target=x_batch/scale_f, past_observed_values=torch.ones_like(x_batch).to(DEVICE))
            p_loc = (distr_args[0][:, -1] * scale_f.squeeze(-1)).cpu().numpy()
            p_scale = (torch.exp(distr_args[1][:, -1]) * scale_f.squeeze(-1)).cpu().numpy()
            z = np.abs(p_loc - y_batch) / (p_scale + 1e-10)
            all_z_scores.extend(z)

    z_array = np.array(all_z_scores)
    y_true_array = np.array(y_true)

    # 5. Latency vs. F1 Trade-off Analysis
    print("\n--- Persistence Analysis (Window Size vs. Accuracy) ---")
    best_threshold = 2.0825 # Using your previously found best threshold
    
    results = []
    for k in [1, 2, 3, 4, 5]:
        y_pred = apply_persistence_filter(z_array, best_threshold, k=k)
        f1 = f1_score(y_true_array, y_pred, zero_division=0)
        prec = precision_score(y_true_array, y_pred, zero_division=0)
        rec = recall_score(y_true_array, y_pred, zero_division=0)
        ttd = calculate_ttd(y_true_array, y_pred, FREQ_MIN)
        
        results.append({"K": k, "F1": f1, "Precision": prec, "Recall": rec, "TTD_min": ttd})

    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))

    # 6. Final Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    time_plot = flow_df['timestamp'].iloc[CONTEXT_LEN:]
    
    ax1.plot(time_plot, v[CONTEXT_LEN:], label="Traffic Volume", alpha=0.5)
    ax1.fill_between(time_plot, 0, v.max(), where=y_true_array==1, color='red', alpha=0.1, label="Anomaly Window")
    
    # Show detections for K=1 and K=3
    pred_k1 = apply_persistence_filter(z_array, best_threshold, k=1)
    pred_k3 = apply_persistence_filter(z_array, best_threshold, k=3)
    ax1.scatter(time_plot[pred_k1==1], v[CONTEXT_LEN:][pred_k1==1], color='orange', s=5, label="Raw Detections (K=1)")
    ax1.scatter(time_plot[pred_k3==1], v[CONTEXT_LEN:][pred_k3==1], color='green', s=20, marker='x', label="Filtered (K=3)")
    
    ax1.set_title("Inference Results: Raw vs. Filtered Persistence")
    ax1.legend()

    ax2.plot(res_df['K'], res_df['F1'], marker='o', label='F1-Score')
    ax2.plot(res_df['K'], res_df['Precision'], marker='s', label='Precision')
    ax2.set_xlabel("Persistence Window (K samples)")
    ax2.set_ylabel("Metric Score")
    ax2.set_title("The Precision-Latency Trade-off")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT)
    print(f"\nAnalysis plot saved as {PLOT_OUTPUT}")

if __name__ == "__main__":
	run_evaluation_with_latency()


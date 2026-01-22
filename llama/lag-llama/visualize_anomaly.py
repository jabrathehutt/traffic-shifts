import torch, numpy as np, pandas as pd, os
import matplotlib.pyplot as plt
from lag_llama.gluon.estimator import LagLlamaLightningModule
from gluonts.torch.distributions import StudentTOutput
from tqdm import tqdm

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_CSV = "/root/traffic-shifts/llama/lag-llama/trafpy_finetune_normal_data.csv"
TEST_CSV = "/root/traffic-shifts/trafpy/trafpy_master_univariate_data.csv"
MODEL_PATH = "specialized_v11_supervised.pt"

CONTEXT_LEN = 256  
N_LAYER = 1
N_HEAD = 8
LAGS_SEQ = list(range(1, 85))

def visualize_anomaly():
    # 1. Load Data & Stats
    df_train = pd.read_csv(TRAIN_CSV)
    df_test = pd.read_csv(TEST_CSV)
    
    train_v = df_train['traffic_volume_Tbits'].values
    GLOBAL_MEAN = np.mean(train_v)
    GLOBAL_STD = np.std(train_v)
    THRESHOLD = 4.5 * GLOBAL_STD

    # 2. Load Model with Required model_kwargs
    module = LagLlamaLightningModule(
        context_length=CONTEXT_LEN, 
        prediction_length=1,
        model_kwargs={
            "context_length": CONTEXT_LEN, 
            "max_context_length": 1024, 
            "input_size": 1,
            "distr_output": StudentTOutput(), 
            "n_layer": N_LAYER, 
            "n_head": N_HEAD,
            "n_embd_per_head": 16, 
            "lags_seq": LAGS_SEQ, 
            "scaling": "mean", 
            "time_feat": False,
        }
    )
    
    sd = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    module.model.load_state_dict({k.replace("model.", ""): v for k, v in sd.items()}, strict=True)
    model = module.model.to(DEVICE).eval()

    # 3. Select Target Flow and Visualization Window
    test_flow = df_test['flow_key_id'].unique()[0]
    flow_df = df_test[df_test['flow_key_id'] == test_flow].sort_values('timestamp').reset_index(drop=True)
    
    # Find anomaly to center the plot
    anomaly_indices = flow_df[flow_df['is_anomaly'] == 1].index
    if len(anomaly_indices) == 0:
        print("No anomaly found in the test set to visualize.")
        return
        
    center_idx = anomaly_indices[0]
    plot_start = max(CONTEXT_LEN, center_idx - 50)
    plot_end = min(len(flow_df), center_idx + 150)
    
    v = flow_df['traffic_volume_Tbits'].values.astype('float32')
    
    preds, actuals, timestamps, ground_truth = [], [], [], []
    
    print(f"Generating visualization for {test_flow}...")
    for i in tqdm(range(plot_start, plot_end)):
        window = torch.tensor(v[i-CONTEXT_LEN:i]).unsqueeze(0).to(DEVICE).float()
        
        with torch.no_grad():
            scale_f = torch.tensor([[GLOBAL_MEAN + 1e-5]]).to(DEVICE).float()
            distr_args, _, _ = model(
                past_target=window/scale_f, 
                past_observed_values=torch.ones_like(window).to(DEVICE).float()
            )
            y_pred = (distr_args[0][:, -1] * scale_f.squeeze(-1)).cpu().item()
            
            preds.append(y_pred)
            actuals.append(v[i])
            timestamps.append(flow_df.loc[i, 'timestamp'])
            ground_truth.append(flow_df.loc[i, 'is_anomaly'])

    # 4. Plotting logic
    residuals = np.abs(np.array(actuals) - np.array(preds))
    y_detected = (residuals > THRESHOLD).astype(int)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Panel 1: Prediction vs Actual
    ax1.plot(timestamps, actuals, label='Actual Traffic', color='#1f77b4', linewidth=1.5)
    ax1.plot(timestamps, preds, label='Lag-Llama Prediction', color='#ff7f0e', linestyle='--', alpha=0.8)
    ax1.set_title(f"Backbone Traffic Prediction (F1: 0.9867)", fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Panel 2: Residuals and Decision Threshold
    ax2.fill_between(timestamps, residuals, color='#d62728', alpha=0.2, label='Prediction Residual')
    ax2.axhline(y=THRESHOLD, color='black', linestyle='-.', label=f'Detection Threshold (4.5σ)')
    ax2.set_ylabel("Error Magnitude (Tbits)")
    ax2.set_title("Anomaly Scoring (Residual Analysis)", fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle=':', alpha=0.6)

    # Panel 3: Ground Truth vs. Detection
    ax3.fill_between(timestamps, ground_truth, color='green', alpha=0.3, label='Ground Truth (Shift Active)')
    ax3.step(timestamps, y_detected, where='post', color='red', label='Model Alarm', linewidth=2)
    ax3.set_ylim(-0.1, 1.1)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['Normal', 'ANOMALY'])
    ax3.set_title("Instantaneous Detection Response", fontsize=12)
    ax3.legend(loc='upper left')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("backbone_anomaly_report.png", dpi=300)
    print("Report saved: backbone_anomaly_report.png")
    plt.show()

if __name__ == "__main__":
    visualize_anomaly()

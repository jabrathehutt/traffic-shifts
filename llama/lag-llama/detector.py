import torch, numpy as np, pandas as pd, os
from lag_llama.gluon.estimator import LagLlamaLightningModule
from gluonts.torch.distributions import StudentTOutput
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_CSV = "/root/traffic-shifts/llama/lag-llama/trafpy_finetune_normal_data.csv"
TEST_CSV = "/root/traffic-shifts/trafpy/trafpy_master_univariate_data.csv"
MODEL_PATH = "specialized_v11_supervised.pt"

CONTEXT_LEN = 256  
BATCH_SIZE = 64
FREQ_MIN = 10 

def run_zero_delay_optimized():
    df_train = pd.read_csv(TRAIN_CSV)
    df_test = pd.read_csv(TEST_CSV)

    # 1. Establish the Global Statistics
    train_v = df_train['traffic_volume_Tbits'].values
    GLOBAL_MEAN = np.mean(train_v)
    GLOBAL_STD = np.std(train_v)

    # 2. Load Model
    module = LagLlamaLightningModule(
        context_length=CONTEXT_LEN, prediction_length=1,
        model_kwargs={
            "context_length": CONTEXT_LEN, "max_context_length": 1024, "input_size": 1,
            "distr_output": StudentTOutput(), "n_layer": 1, "n_head": 8,
            "n_embd_per_head": 16, "lags_seq": list(range(1, 85)), "scaling": "mean", "time_feat": False,
        }
    )
    sd = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    module.model.load_state_dict({k.replace("model.", ""): v for k, v in sd.items()}, strict=True)
    model = module.model.to(DEVICE).eval()

    # 3. Data Prep
    test_flow = df_test['flow_key_id'].unique()[0]
    flow_df = df_test[df_test['flow_key_id'] == test_flow].sort_values('timestamp')
    v = flow_df['traffic_volume_Tbits'].values.astype('float32')
    y_true = (flow_df['is_anomaly'].values[CONTEXT_LEN:] == 1).astype(int)

    residuals = []
    print(f"Applying Optimized Zero-Delay Detection: {test_flow}")
    
    for i in tqdm(range(CONTEXT_LEN, len(v), BATCH_SIZE)):
        batch_end = min(i + BATCH_SIZE, len(v))
        windows = [v[j-CONTEXT_LEN:j] for j in range(i, batch_end)]
        x_batch = torch.tensor(np.array(windows)).to(DEVICE).float()
        y_actual = v[i:batch_end]

        with torch.no_grad():
            scale_f = torch.tensor([[GLOBAL_MEAN + 1e-5]]).to(DEVICE).float()
            distr_args, _, _ = model(past_target=x_batch/scale_f, past_observed_values=torch.ones_like(x_batch).to(DEVICE).float())
            y_pred_val = (distr_args[0][:, -1] * scale_f.squeeze(-1)).cpu().numpy()
            residuals.extend(np.abs(y_actual - y_pred_val))

    res_array = np.array(residuals)
    
    # 4. THRESHOLD TUNING
    # 4.5 Sigma is usually the boundary for 1.0 Precision in backbone traffic
    threshold = 4.5 * GLOBAL_STD 
    y_pred = (res_array > threshold).astype(int)

    # 5. Metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Delay Calculation
    delay = 0
    anomaly_indices = np.where(y_true == 1)[0]
    if len(anomaly_indices) > 0:
        first_true = anomaly_indices[0]
        detections = np.where(y_pred[first_true:] == 1)[0]
        if len(detections) > 0:
            delay = detections[0] * FREQ_MIN

    print("\n" + "="*50)
    print(f"OPTIMIZED ZERO-DELAY (Threshold = {threshold:.4f} Tbits)")
    print("-" * 50)
    print(f"Precision:         {precision:.4f}")
    print(f"Recall:            {recall:.4f}")
    print(f"F1 Score:          {f1:.4f}")
    print(f"Avg Detect Delay:  {delay:.2f} minutes")
    print("="*50)

if __name__ == "__main__":
    run_zero_delay_optimized()

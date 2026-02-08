import torch
import pandas as pd
import numpy as np
import os
import random
import time
from gluonts.dataset.pandas import PandasDataset
from lag_llama.gluon.estimator import LagLlamaEstimator
from lag_llama.gluon.lightning_module import LagLlamaLightningModule
from gluonts.torch.distributions import StudentTOutput
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
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

CSV_PATH = "/root/traffic-shifts/trafpy/trafpy_master_univariate_data.csv"
CKPT_PATH = "/root/traffic-shifts/llama/lag-llama/models/finetune_latest.ckpt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONTEXT_LENGTH = 128
MAX_CONTEXT_LENGTH = 2048
PREDICTION_LENGTH = 1
NUM_SAMPLES = 1000# Restored to original
QUANTILE_THRESHOLD = 0.95
SAMPLING_LAG_MINS = 2.5

def run_verified_evaluation():
    df = pd.read_csv(CSV_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['traffic_volume_Tbits'] = df['traffic_volume_Tbits'].astype('float32')

    # 2. Setup Model & Surgery
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint['state_dict']
    ckpt_wte_weight = state_dict['model.transformer.wte.weight']
    ckpt_wte_bias = state_dict.get('model.transformer.wte.bias', None)
    ckpt_in_dim = ckpt_wte_weight.shape[1]
    embedding_dim = ckpt_wte_weight.shape[0]
    has_bias = ckpt_wte_bias is not None

    lags_seq = list(range(1, 96))
    module = LagLlamaLightningModule(
        context_length=CONTEXT_LENGTH,
        max_context_length=2048,
        prediction_length=1,
        model_kwargs={
            "context_length": CONTEXT_LENGTH,
            "max_context_length": 2048,
            "n_layer": 1, "n_head": 8, "n_embd_per_head": 16,
            "scaling": "mean", "time_feat": False, "input_size": 1,
            "distr_output": StudentTOutput(), "lags_seq": lags_seq
        }
    )

    module.to(DEVICE)
    module.model.transformer.wte = torch.nn.Linear(ckpt_in_dim, embedding_dim, bias=has_bias).to(DEVICE)

    estimator = LagLlamaEstimator(
        prediction_length=PREDICTION_LENGTH,
        context_length=CONTEXT_LENGTH,
        batch_size=64,
        trainer_kwargs={"accelerator": "cuda" if torch.cuda.is_available() else "cpu", "devices": 1}
    )
    estimator.lags_seq = lags_seq
    predictor = estimator.create_predictor(estimator.create_transformation(), module)

    data_dim = ckpt_in_dim
    try:
        list(predictor.predict(PandasDataset.from_long_dataframe(df[df['flow_key_id']==df['flow_key_id'].unique()[0]].iloc[:200], target="traffic_volume_Tbits", item_id="flow_key_id", timestamp="timestamp", freq="5min"), num_samples=1))
    except Exception:
        data_dim = 50

    new_wte = torch.nn.Linear(data_dim, embedding_dim, bias=has_bias).to(DEVICE)
    with torch.no_grad():
        new_wte.weight.fill_(0)
        new_wte.weight[:, :ckpt_in_dim].copy_(ckpt_wte_weight)
        if has_bias: new_wte.bias.copy_(ckpt_wte_bias)

    module.model.transformer.wte = new_wte
    module.load_state_dict({k: v for k, v in state_dict.items() if "wte" not in k}, strict=False)
    module.to(DEVICE).eval()

    # 5. Evaluation
    all_y_true, all_y_pred = [], []
    detection_delays, all_inf_latencies = [], []
    unique_flows = df['flow_key_id'].unique()

    print(f"Executing Sequential Evaluation (Accuracy Focus) on {len(unique_flows)} flows...")

    for flow_id in unique_flows:
        flow_df = df[df['flow_key_id'] == flow_id].reset_index(drop=True)
        current_event_start_idx = None
        event_already_detected = False

        start_idx = CONTEXT_LENGTH + 60
        for i in tqdm(range(start_idx, len(flow_df)), desc=f"Inference {flow_id}"):
            input_data = flow_df.iloc[:i].set_index('timestamp')['traffic_volume_Tbits']
            window_dataset = PandasDataset(input_data, freq="5min")

            # --- MEASURE INFERENCE LATENCY & COMP LAG ---
            start_comp = time.time()
            with torch.no_grad():
                forecast_it = predictor.predict(window_dataset, num_samples=NUM_SAMPLES)
                forecast = list(forecast_it)[0]
                upper_bound = np.quantile(forecast.samples, q=QUANTILE_THRESHOLD)
            end_comp = time.time()

            inf_latency_sec = end_comp - start_comp
            all_inf_latencies.append(inf_latency_sec)
            comp_lag_mins = inf_latency_sec / 60.0

            actual_val = flow_df['traffic_volume_Tbits'].iloc[i]
            is_anomaly_label = flow_df['is_anomaly'].iloc[i]
            pred_label = 1 if actual_val > upper_bound else 0

            all_y_pred.append(pred_label)
            all_y_true.append(is_anomaly_label)

            # --- DELAY LOGIC ---
            if is_anomaly_label == 1 and (i == 0 or flow_df['is_anomaly'].iloc[i-1] == 0):
                current_event_start_idx = i
                event_already_detected = False

            if is_anomaly_label == 1 and pred_label == 1 and not event_already_detected:
                algo_lag_mins = (i - current_event_start_idx) * 5.0
                total_delay = SAMPLING_LAG_MINS + algo_lag_mins + comp_lag_mins
                detection_delays.append(total_delay)
                event_already_detected = True

            if is_anomaly_label == 0:
                current_event_start_idx, event_already_detected = None, False

    # 6. Results
    y_true, y_pred = np.array(all_y_true), np.array(all_y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    avg_delay = np.mean(detection_delays) if detection_delays else 0

    print("\n" + "="*45)
    print(f"      LAG-LLAMA ACCURACY-STRICT REPORT")
    print("-" * 45)
    print(f"PRECISION: {precision:.4f} | RECALL: {recall:.4f}")
    print(f"F1-SCORE:  {f1:.4f} | AVG DELAY: {avg_delay:.4f} mins")
    print(f"INFERENCE LATENCY: {np.mean(all_inf_latencies):.6f}")
    print("-" * 45)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")
    print("="*45)

if __name__ == "__main__":
    run_verified_evaluation()


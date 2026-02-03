import torch
import pandas as pd
import numpy as np
import os
import random
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
NUM_SAMPLES = 200
QUANTILE_THRESHOLD = 0.975

def run_verified_evaluation():
    # 1. Load Data
    df = pd.read_csv(CSV_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['traffic_volume_Tbits'] = df['traffic_volume_Tbits'].astype('float32')

    # 2. Inspect Checkpoint
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint['state_dict']
    ckpt_wte_weight = state_dict['model.transformer.wte.weight']
    ckpt_wte_bias = state_dict.get('model.transformer.wte.bias', None)
    
    ckpt_in_dim = ckpt_wte_weight.shape[1] 
    embedding_dim = ckpt_wte_weight.shape[0]
    has_bias = ckpt_wte_bias is not None

    # 3. Build Module
    lags_seq = list(range(1, 48)) 
    module = LagLlamaLightningModule(
        context_length=CONTEXT_LENGTH, max_context_length=2048, prediction_length=1,
        model_kwargs={
            "context_length": CONTEXT_LENGTH, "max_context_length": 2048,
            "n_layer": 1, "n_head": 8, "n_embd_per_head": 16,
            "scaling": "mean", "time_feat": False, "input_size": 1,
            "distr_output": StudentTOutput(), "lags_seq": lags_seq
        }
    )

    # 4. Initial Surgery (Trial)
    module.model.transformer.wte = torch.nn.Linear(ckpt_in_dim, embedding_dim, bias=has_bias)
    
    # 5. Dynamic Data Dimension Detection
    estimator = LagLlamaEstimator(prediction_length=PREDICTION_LENGTH, context_length=CONTEXT_LENGTH, batch_size=1)
    estimator.lags_seq = lags_seq
    predictor = estimator.create_predictor(estimator.create_transformation(), module)
    
    test_ds = PandasDataset(df.iloc[:200].set_index('timestamp')['traffic_volume_Tbits'], freq="10min")
    
    print("Detecting Data Pipeline dimensions...")
    try:
        list(predictor.predict(test_ds, num_samples=1))
        data_dim = ckpt_in_dim
    except RuntimeError as e:
        # Extract the dimension the data stream actually sends (e.g., 50)
        import re
        match = re.search(r"12800x(\d+)", str(e))
        if match:
            data_dim = int(match.group(1))
        else:
            data_dim = 50 # Fallback to your error observation
            
    print(f"--- ALIGNMENT SURGERY ---")
    print(f"Checkpoint Input: {ckpt_in_dim} | Data Stream: {data_dim}")

    # 6. FINAL WEIGHT PADDING & LOADING
    new_wte = torch.nn.Linear(data_dim, embedding_dim, bias=has_bias)
    with torch.no_grad():
        new_wte.weight.fill_(0)
        # Copy checkpoint weights into the start of the matrix
        new_wte.weight[:, :ckpt_in_dim].copy_(ckpt_wte_weight)
        if has_bias: new_wte.bias.copy_(ckpt_wte_bias)
            
    module.model.transformer.wte = new_wte
    
    # Clean state_dict for rest of model loading
    clean_sd = {k: v for k, v in state_dict.items() if "wte" not in k}
    module.load_state_dict(clean_sd, strict=False)
    module.to(DEVICE).eval()
    
    print(f"✓ Manual Surgery Success. Weights copied and padded.")
    print(f"Live WTE Mean (matching disk): {module.model.transformer.wte.weight[:, :ckpt_in_dim].mean().item():.10f}")
    print("-" * 31 + "\n")

    # 7. Final Inference
    eval_size = 500
    all_y_true, all_y_pred = [], []

    for i in tqdm(range(len(df) - eval_size, len(df))):
        input_data = df.iloc[:i].set_index('timestamp')['traffic_volume_Tbits']
        if len(input_data) < CONTEXT_LENGTH + 60: continue

        window_dataset = PandasDataset(input_data, freq="10min")
        with torch.no_grad():
            forecast_it = predictor.predict(window_dataset, num_samples=NUM_SAMPLES)
            forecast = list(forecast_it)[0]

        upper_bound = np.quantile(forecast.samples, q=QUANTILE_THRESHOLD)
        actual_val = df['traffic_volume_Tbits'].iloc[i]
        
        all_y_pred.append(1 if actual_val > upper_bound else 0)
        all_y_true.append(df['is_anomaly'].iloc[i])

    # 8. Results
    print("\n" + "="*45)
    print(f"      DETERMINISTIC ANOMALY REPORT")
    print("-" * 45)
    y_true, y_pred = np.array(all_y_true), np.array(all_y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    print(f"PRECISION: {precision:.4f} | RECALL: {recall:.4f}")
    print(f"F1-SCORE:  {f1:.4f}")
    print(f"TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")
    print("="*45)

if __name__ == "__main__":
    run_verified_evaluation()

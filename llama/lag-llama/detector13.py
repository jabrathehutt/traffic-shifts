import torch
import pandas as pd
import numpy as np
import os
import random
import functools
from gluonts.dataset.pandas import PandasDataset
from lag_llama.gluon.estimator import LagLlamaEstimator, LagLlamaLightningModule
from gluonts.torch.distributions import StudentTOutput
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([functools.partial])

def seed_everything(seed=42):
    random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed(seed); torch.backends.cudnn.deterministic = True

seed_everything(42)

# --- CONFIG ---
CSV_PATH = "/root/traffic-shifts/trafpy/trafpy_master_univariate_data.csv"
CKPT_PATH = "experiments/results/finetune_trafpy_v1/42/checkpoints/epoch-epoch=09.ckpt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONTEXT_LENGTH = 128
MAX_CONTEXT_LENGTH = 2048
PREDICTION_LENGTH = 1
NUM_SAMPLES = 100
QUANTILE_THRESHOLD = 0.9999

def run_total_dataset_evaluation():
    # 1. Load Data
    df = pd.read_csv(CSV_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['traffic_volume_Tbits'] = df['traffic_volume_Tbits'].astype('float32')

    # 2. Setup Architecture (Base)
    module = LagLlamaLightningModule(
        context_length=CONTEXT_LENGTH,
        max_context_length=MAX_CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        model_kwargs={
            "context_length": CONTEXT_LENGTH, "max_context_length": MAX_CONTEXT_LENGTH,
            "n_layer": 1, "n_head": 8, "n_embd_per_head": 16,
            "scaling": "mean", "time_feat": False, "input_size": 1,
            "distr_output": StudentTOutput(), "lags_seq": list(range(1, 47))
        }
    )

    # 3. Load Checkpoint & Surgery
    if not os.path.exists(CKPT_PATH):
        print(f"Error: {CKPT_PATH} not found.")
        return

    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint['state_dict']

    expected_dim = state_dict['model.transformer.wte.weight'].shape[1]
    embedding_dim = state_dict['model.transformer.wte.weight'].shape[0]

    if module.model.transformer.wte.weight.shape[1] != expected_dim:
        print(f"Surgery: Patching wte to {expected_dim}")
        module.model.transformer.wte = torch.nn.Linear(expected_dim, embedding_dim, bias=False)

    module.load_state_dict(state_dict)
    module.to(DEVICE).float().eval()

    # 4. Predictor Setup
    # Removed 'freq' to fix TypeError. 
    # GluonTS will infer frequency from the PandasDataset below.
    estimator = LagLlamaEstimator(
        prediction_length=PREDICTION_LENGTH,
        context_length=CONTEXT_LENGTH,
        batch_size=1
    )

    estimator.lags_seq = list(range(1, expected_dim))
    predictor = estimator.create_predictor(estimator.create_transformation(), module)

    # 5. Evaluation Loop
    all_y_true, all_y_pred = [], []
    full_series = df.set_index('timestamp')['traffic_volume_Tbits']

    print(f"Executing Evaluation on {len(df)} points...")
    for i in tqdm(range(len(df))):
        window_series = full_series.iloc[:i]
        if len(window_series) == 0: continue

        actual_val = df['traffic_volume_Tbits'].iloc[i]
        actual_label = df['is_anomaly'].iloc[i]

        # The frequency is defined here, which the predictor uses for transformation
        window_dataset = PandasDataset(window_series, freq="10min")
        forecast_it = predictor.predict(window_dataset, num_samples=NUM_SAMPLES)
        forecast = list(forecast_it)[0]

        upper_bound = np.quantile(forecast.samples, q=QUANTILE_THRESHOLD)
        prediction = 1 if actual_val > upper_bound else 0

        all_y_true.append(actual_label)
        all_y_pred.append(prediction)

    # 6. Results
    precision = precision_score(all_y_true, all_y_pred, zero_division=0)
    recall = recall_score(all_y_true, all_y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(all_y_true, all_y_pred).ravel()

    print("\n" + "="*45)
    print(f"SPECIALIZED BACKBONE (EPOCH 09) RESULTS")
    print("-" * 45)
    print(f"PRECISION: {precision:.4f} | RECALL: {recall:.4f}")
    print(f"TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")
    print("="*45)

if __name__ == "__main__":
    run_total_dataset_evaluation()

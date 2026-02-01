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

CSV_PATH = "/root/traffic-shifts/trafpy/trafpy_master_univariate_data.csv"
CKPT_PATH = "/root/traffic-shifts/llama/lag-llama/models/backbone_latest.ckpt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_total_dataset_evaluation():
    df = pd.read_csv(CSV_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['traffic_volume_Tbits'] = df['traffic_volume_Tbits'].astype('float32')

    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint['state_dict']
    expected_dim = state_dict['model.transformer.wte.weight'].shape[1]
    embedding_dim = state_dict['model.transformer.wte.weight'].shape[0]

    # Surgery: Match the backbone width (79 lags detected in your logs)
    found_lags = 0
    module = None
    for trial_count in range(expected_dim - 5, expected_dim + 2):
        trial_lags = list(range(1, trial_count + 1))
        test_module = LagLlamaLightningModule(
            context_length=64, max_context_length=2048, prediction_length=1,
            model_kwargs={
                "context_length": 64, "max_context_length": 2048,
                "n_layer": 1, "n_head": 8, "n_embd_per_head": 16,
                "scaling": "mean", "time_feat": False, "input_size": 1,
                "distr_output": StudentTOutput(), "lags_seq": trial_lags
            }
        )
        if test_module.model.transformer.wte.weight.shape[1] == expected_dim:
            module = test_module
            found_lags = trial_count
            break

    module.load_state_dict(state_dict, strict=True)
    module.to(DEVICE).float().eval()

    estimator = LagLlamaEstimator(prediction_length=1, context_length=64, batch_size=1)
    estimator.lags_seq = list(range(1, found_lags + 1))
    predictor = estimator.create_predictor(estimator.create_transformation(), module)

    all_y_true = []
    all_samples = []
    all_actuals = []
    full_series = df.set_index('timestamp')['traffic_volume_Tbits']

    print(f"Generating Forecast Samples...")
    for i in tqdm(range(len(df))):
        window_series = full_series.iloc[:i]
        if len(window_series) < (64 + found_lags): continue

        actual_val = df['traffic_volume_Tbits'].iloc[i]
        actual_label = df['is_anomaly'].iloc[i]
        window_dataset = PandasDataset(window_series, freq="10min")
        
        forecast = list(predictor.predict(window_dataset, num_samples=100))[0]
        
        all_samples.append(forecast.samples)
        all_actuals.append(actual_val)
        all_y_true.append(actual_label)

    # SWEEP THRESHOLDS
    print("\n" + "="*55)
    print(f"{'THRESHOLD':<12} | {'PRECISION':<10} | {'RECALL':<10} | {'FP':<5}")
    print("-" * 55)
    
    for q in [0.9, 0.99, 0.999, 0.9999]:
        y_pred = []
        for idx, samples in enumerate(all_samples):
            upper_bound = np.quantile(samples, q=q)
            y_pred.append(1 if all_actuals[idx] > upper_bound else 0)
        
        prec = precision_score(all_y_true, y_pred, zero_division=0)
        rec = recall_score(all_y_true, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(all_y_true, y_pred).ravel()
        print(f"{q:<12.4f} | {prec:<10.4f} | {rec:<10.4f} | {fp:<5}")
    
    print("="*55)

if __name__ == "__main__":
    run_total_dataset_evaluation()

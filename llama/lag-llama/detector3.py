import pandas as pd
import numpy as np
import torch
from gluonts.dataset.pandas import PandasDataset
from sklearn.metrics import precision_score, recall_score, f1_score
from lag_llama.gluon.estimator import LagLlamaEstimator
from lag_llama.gluon.lightning_module import LagLlamaLightningModule
from gluonts.torch.distributions import StudentTOutput
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
DATA_FILE = "/root/traffic-shifts/trafpy/trafpy_master_univariate_data.csv"
CKPT_PATH = "specialized_v11_supervised.pt"
TARGET_COLUMN = 'traffic_volume_Tbits'
SERIES_ID_COLUMN = 'flow_key_id'
CONTEXT_LENGTH = 32  
UPPER_QUANTILE = 0.95 
NUM_SAMPLES = 100 
FREQ = '10min'
CHUNK_SIZE = 5 # Smaller chunks to ensure immediate terminal feedback

def run_evaluation():
    # 1. Load Data
    print("Reading data...")
    df = pd.read_csv(DATA_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(np.float32)
    df['is_anomaly'] = df['is_anomaly'].astype(bool)
    
    if 'actual_start' in df.columns:
        df['actual_start'] = pd.to_datetime(df['actual_start'])

    # 2. Initialize Module
    print(f"Loading weights from {CKPT_PATH}...")
    lags_seq = [1, 2, 3, 4, 5, 6, 7, 12, 24, 48, 72, 168]
    
    module = LagLlamaLightningModule(
        context_length=CONTEXT_LENGTH,
        prediction_length=1,
        model_kwargs={
            "input_size": 1, "context_length": CONTEXT_LENGTH, "max_context_length": 2048,
            "lags_seq": lags_seq, "distr_output": StudentTOutput(),
            "n_layer": 8, "n_embd_per_head": 32, "n_head": 8,
            "scaling": "mean", "time_feat": False,
        }
    )

    state_dict = torch.load(CKPT_PATH, map_location=torch.device('cpu'))
    if 'state_dict' in state_dict: 
        state_dict = state_dict['state_dict']
    
    new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    module.load_state_dict(new_state_dict, strict=False)
    module = module.to(torch.device('cpu')).float().eval()

    # 3. Predictor
    estimator = LagLlamaEstimator(prediction_length=1, context_length=CONTEXT_LENGTH)
    predictor = estimator.create_predictor(transformation=estimator.create_transformation(), module=module)

    # 4. Inference with ACTIVE progress tracking
    unique_ids = df[SERIES_ID_COLUMN].unique()
    all_preds = []

    print(f"Starting inference for {len(unique_ids)} flows...")
    
    # We use tqdm here to track progress across flows
    for i in range(0, len(unique_ids), CHUNK_SIZE):
        chunk_ids = unique_ids[i:i + CHUNK_SIZE]
        chunk_df = df[df[SERIES_ID_COLUMN].isin(chunk_ids)]
        
        dataset = PandasDataset.from_long_dataframe(
            chunk_df, target=TARGET_COLUMN, item_id=SERIES_ID_COLUMN, 
            timestamp="timestamp", freq=FREQ, dtype=np.float32
        )

        with torch.no_grad():
            forecast_it = predictor.predict(dataset, num_samples=NUM_SAMPLES)
            # The "Killed" or "Hanging" usually happens when converting the generator to a list.
            # We process them one by one to keep memory low and the CPU active.
            for forecast in tqdm(forecast_it, total=len(chunk_ids), desc=f"Group {i//CHUNK_SIZE + 1}", leave=False):
                q95 = np.quantile(forecast.samples, UPPER_QUANTILE, axis=0)
                all_preds.append({
                    SERIES_ID_COLUMN: forecast.item_id,
                    "timestamp": forecast.start_date.to_timestamp(),
                    "q95_limit": q95[0],
                    "median_pred": np.median(forecast.samples)
                })

    # 5. Result Merging
    print("\nProcessing metrics...")
    pred_df = pd.DataFrame(all_preds)
    results = df.merge(pred_df, on=['timestamp', SERIES_ID_COLUMN], how='inner')
    results['y_pred'] = results[TARGET_COLUMN] > results['q95_limit']

    # 6. Evaluation
    f1 = f1_score(results['is_anomaly'], results['y_pred'], zero_division=0)
    prec = precision_score(results['is_anomaly'], results['y_pred'], zero_division=0)
    rec = recall_score(results['is_anomaly'], results['y_pred'], zero_division=0)

    # Delay Calculation
    
    delays = []
    results['event_id'] = (results['is_anomaly'] != results['is_anomaly'].shift()).cumsum()
    for _, event in results[results['is_anomaly']].groupby('event_id'):
        detections = event[event['y_pred']]
        if not detections.empty:
            delay = (detections['timestamp'].min() - event['actual_start'].iloc[0]).total_seconds() / 60.0
            delays.append(max(0, delay))

    print("\n" + "="*50)
    print("THESIS PERFORMANCE SUMMARY")
    print("-" * 50)
    print(f"F1 Score:  {f1:.4f} (P: {prec:.4f}, R: {rec:.4f})")
    if delays:
        print(f"Avg Delay: {np.mean(delays):.2f} minutes")
    print(f"Detections: {results['y_pred'].sum()} / {results['is_anomaly'].sum()} true anomalies")
    print("="*50)

    results.to_csv("lag_llama_eval_debug.csv", index=False)

if __name__ == "__main__":
    run_evaluation()

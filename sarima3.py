import pandas as pd
import numpy as np
import os
from statsmodels.tsa.seasonal import STL
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
TRAIN_FILE = '/root/traffic-shifts/llama/lag-llama/trafpy_finetune_normal_data.csv'
TEST_FILE = '/root/traffic-shifts/trafpy/trafpy_master_univariate_data.csv'

def run_stl_baseline():
    if not os.path.exists(TRAIN_FILE) or not os.path.exists(TEST_FILE):
        print("Error: Files not found.")
        return

    df_train_raw = pd.read_csv(TRAIN_FILE)
    df_test_raw = pd.read_csv(TEST_FILE)

    # Aligning IDs: Grabbing the first available flow from each
    train_flow_id = df_train_raw['flow_key_id'].unique()[0]
    test_flow_id = df_test_raw['flow_key_id'].unique()[0]
    
    train_df = df_train_raw[df_train_raw['flow_key_id'] == train_flow_id].sort_values('timestamp')
    test_df = df_test_raw[df_test_raw['flow_key_id'] == test_flow_id].sort_values('timestamp')

    train_series = train_df['traffic_volume_Tbits'].values
    test_series = test_df['traffic_volume_Tbits'].values
    y_true = test_df['is_anomaly'].values

    if len(train_series) == 0 or len(test_series) == 0:
        print("Error: Empty series. Check flow IDs.")
        return

    # 1. Log Transform
    train_log = np.log1p(train_series)
    test_log = np.log1p(test_series)

    # 2. Extract Diurnal Pattern (s=144)
    print(f"Decomposing baseline seasonality for {train_flow_id}...")
    stl = STL(train_log, period=144, robust=True)
    res = stl.fit()
    
    # We take exactly one 24-hour cycle (144 points) as the 'Golden Template'
    seasonal_template = res.seasonal[:144]
    
    # Tile it to match test_series length
    reps = int(np.ceil(len(test_log) / 144))
    tiled_seasonality = np.tile(seasonal_template, reps)[:len(test_log)]

    # 3. Calculate Remainder (Observed - Seasonal Template)
    test_remainder = test_log - tiled_seasonality

    # 4. Robust Detection
    median_rem = np.median(test_remainder)
    mad_rem = np.median(np.abs(test_remainder - median_rem)) * 1.4826
    mad_rem = max(mad_rem, 1e-6) # Floor
    
    z_scores = np.abs(test_remainder - median_rem) / mad_rem
    
    # 5. Integrated Energy Gate (Improving Precision)
    # We use a 3-point rolling average of Z-scores to ignore micro-jitter
    z_smoothed = pd.Series(z_scores).rolling(window=3, center=False).mean().fillna(0).values
    
    # Threshold 5.5 is usually the "sweet spot" for de-seasonalized network data
    y_pred = (z_smoothed > 5.5).astype(int)

    # 6. Final Metrics
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print("\n" + "=" * 45)
    print("ROBUST STL-RESIDUAL PERFORMANCE")
    print("-" * 45)
    print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")
    print(f"TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")
    print("=" * 45)

if __name__ == "__main__":
    run_stl_baseline()

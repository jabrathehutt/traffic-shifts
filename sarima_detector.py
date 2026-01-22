import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import warnings
import os

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
# Path to clean baseline data
TRAIN_FILE = '/root/traffic-shifts/llama/lag-llama/trafpy_finetune_normal_data.csv'
# Path to experimental/anomalous data
TEST_FILE = '/root/traffic-shifts/trafpy/trafpy_master_univariate_data.csv'

# Threshold for Robust Z-Score (tuned for Tbit-scale stochasticity)
ROBUST_Z_THRESHOLD = 5.0 
FREQ_MIN = 10 

def run_seasonal_baseline_analysis():
    # 1. Load Datasets
    if not os.path.exists(TRAIN_FILE) or not os.path.exists(TEST_FILE):
        print("Error: Training or Test file missing.")
        return

    df_train_raw = pd.read_csv(TRAIN_FILE)
    df_test_raw = pd.read_csv(TEST_FILE)

    # 2. Extract corresponding flows
    flow_id_train = df_train_raw['flow_key_id'].unique()[0]
    flow_id_test = df_test_raw['flow_key_id'].unique()[0]

    df_train = df_train_raw[df_train_raw['flow_key_id'] == flow_id_train].sort_values('timestamp')
    df_test = df_test_raw[df_test_raw['flow_key_id'] == flow_id_test].sort_values('timestamp')

    # 3. Establishing the Baseline (The "Training" Step)
    # We use the clean data to find the normal Median and MAD
    baseline_series = df_train['traffic_volume_Tbits'].values
    median_baseline = np.median(baseline_series)
    # 1.4826 is the scaling factor to make MAD consistent with Standard Deviation
    mad_baseline = np.median(np.abs(baseline_series - median_baseline)) * 1.4826

    print(f"Baseline established from {flow_id_train}")
    print(f"Median: {median_baseline:.6f}, MAD-Std: {mad_baseline:.6f}")

    # 4. Detection on the Anomalous Dataset
    test_series = df_test['traffic_volume_Tbits'].values
    y_true = (df_test['is_anomaly'].values == 1).astype(int)
    
    # Calculate Robust Z-Scores for the test set relative to the clean baseline
    # Robust Z = (Value - Median_Baseline) / MAD_Baseline
    robust_z_scores = np.abs(test_series - median_baseline) / (mad_baseline + 1e-12)
    
    y_pred = (robust_z_scores > ROBUST_Z_THRESHOLD).astype(int)

    # 5. Metrics Calculation
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # 6. Detection Delay (TTD)
    delay = 0
    anomaly_indices = np.where(y_true == 1)[0]
    if len(anomaly_indices) > 0:
        first_true = anomaly_indices[0]
        # Find first detection at or after the anomaly start
        detections = np.where(y_pred[first_true:] == 1)[0]
        if len(detections) > 0:
            delay = detections[0] * FREQ_MIN

    print("\n" + "=" * 60)
    print("SARIMA-LIKE ROBUST Z-SCORE PERFORMANCE (TRAINED ON CLEAN)")
    print("=" * 60)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Average Detection Delay: {delay:.2f} minutes")
    print("=" * 60)

if __name__ == "__main__":
    run_seasonal_baseline_analysis()

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import precision_score, recall_score, f1_score, mean_absolute_error
from tqdm import tqdm
import warnings
import os

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
# Train on the clean baseline data
TRAIN_FILE = '/root/traffic-shifts/llama/lag-llama/trafpy_finetune_normal_data.csv'
# Test on the data containing anomalies
TEST_FILE = '/root/traffic-shifts/trafpy/trafpy_master_univariate_data.csv'

CONFIDENCE_INTERVAL = 0.95
THRESHOLD_FACTOR = 1.5  # Increased slightly to handle stochastic noise


def run_prophet_cross_dataset_detection(flow_id_train, flow_id_test):
    # 1. Load Datasets
    df_train_raw = pd.read_csv(TRAIN_FILE)
    df_test_raw = pd.read_csv(TEST_FILE)

    # 2. Prepare Training Data (Clean Baseline)
    df_train = df_train_raw[df_train_raw['flow_key_id'] == flow_id_train].copy()
    df_train = df_train.rename(columns={'timestamp': 'ds', 'traffic_volume_Tbits': 'y'})
    df_train['ds'] = pd.to_datetime(df_train['ds'])

    # 3. Prepare Test Data (Experimental/Anomalous)
    df_test = df_test_raw[df_test_raw['flow_key_id'] == flow_id_test].copy()
    df_test = df_test.rename(columns={'timestamp': 'ds', 'traffic_volume_Tbits': 'y'})
    df_test['ds'] = pd.to_datetime(df_test['ds'])

    # 4. Train Prophet on Clean Baseline
    # Since TrafPy is stochastic, we focus on daily seasonality
    m = Prophet(
        interval_width=CONFIDENCE_INTERVAL,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=True,
    )
    m.fit(df_train)

    # 5. Forecast on the Experimental Timeline
    future = pd.DataFrame({'ds': df_test['ds']})
    forecast = m.predict(future)

    # 6. Detection Logic
    df_pred = df_test.merge(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='left')
    df_pred['uncertainty'] = df_pred['yhat_upper'] - df_pred['yhat_lower']

    # Point-by-point detection based on confidence interval
    df_pred['is_anomaly_detected'] = np.where(
        (np.abs(df_pred['y'] - df_pred['yhat']) > THRESHOLD_FACTOR * df_pred['uncertainty']),
        True, False
    )

    # 7. Calculate Delay (TTD)
    delay = None
    actual_anomalies = df_pred[df_pred['is_anomaly'] == 1]
    if not actual_anomalies.empty:
        first_anomaly_time = actual_anomalies['ds'].min()
        detections = df_pred[(df_pred['is_anomaly_detected'] == True) & (df_pred['ds'] >= first_anomaly_time)]
        if not detections.empty:
            first_detection_time = detections['ds'].min()
            delay = (first_detection_time - first_anomaly_time).total_seconds() / 60.0

    return {
        'y_true': df_pred['is_anomaly'].values,
        'y_pred': df_pred['is_anomaly_detected'].values,
        'mae': mean_absolute_error(df_pred['y'].values, df_pred['yhat'].values),
        'delay': delay
    }


def run_comparative_analysis():
    # Get flow lists
    train_flows = pd.read_csv(TRAIN_FILE)['flow_key_id'].unique()
    test_flows = pd.read_csv(TEST_FILE)['flow_key_id'].unique()

    # We match the first training flow with its corresponding test flow
    # (Assuming both follow the 'Backbone' behavior)
    all_y_true, all_y_pred, delays = [], [], []

    print(f"Prophet: Training on {TRAIN_FILE}")
    print(f"Prophet: Testing on {TEST_FILE}")

    # For a fair comparison, we compare corresponding flows
    # (e.g., Backbone_Normal vs Backbone_Anomalous)
    for i in range(min(len(train_flows), len(test_flows))):
        try:
            res = run_prophet_cross_dataset_detection(train_flows[i], test_flows[i])
            all_y_true.extend(res['y_true'])
            all_y_pred.extend(res['y_pred'])
            if res['delay'] is not None:
                delays.append(res['delay'])
        except Exception as e:
            print(f"Error processing flow {i}: {e}")

    # Metrics
    precision = precision_score(all_y_true, all_y_pred, zero_division=0)
    recall = recall_score(all_y_true, all_y_pred, zero_division=0)
    f1 = f1_score(all_y_true, all_y_pred, zero_division=0)
    avg_delay = np.mean(delays) if delays else 0

    print("\n" + "=" * 60)
    print("PROPHET BASELINE PERFORMANCE (TRAINED ON CLEAN DATA)")
    print("=" * 60)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Avg Detect Delay: {avg_delay:.2f} minutes")
    print("=" * 60)


if __name__ == "__main__":
    run_comparative_analysis()

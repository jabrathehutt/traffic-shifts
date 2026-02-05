import pandas as pd
import numpy as np
import time
from prophet import Prophet
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

TRAIN_FILE = '/root/traffic-shifts/trafpy/trafpy_finetune_normal_data.csv'
TEST_FILE = '/root/traffic-shifts/trafpy/trafpy_master_univariate_data.csv'

CONFIDENCE_INTERVAL = 0.95
THRESHOLD_FACTOR = 1.5
SAMPLING_LAG_MINS = 2.5
FREQ_MIN = 5.0

def run_prophet_detection(flow_id_train, flow_id_test):
    df_train_raw = pd.read_csv(TRAIN_FILE)
    df_test_raw = pd.read_csv(TEST_FILE)

    df_train = df_train_raw[df_train_raw['flow_key_id'] == flow_id_train].rename(columns={'timestamp':'ds','traffic_volume_Tbits':'y'})
    df_train['ds'] = pd.to_datetime(df_train['ds'])
    df_test = df_test_raw[df_test_raw['flow_key_id'] == flow_id_test].rename(columns={'timestamp':'ds','traffic_volume_Tbits':'y'})
    df_test['ds'] = pd.to_datetime(df_test['ds'])

    m = Prophet(interval_width=CONFIDENCE_INTERVAL, daily_seasonality=True).fit(df_train)

    start_comp = time.time()
    forecast = m.predict(pd.DataFrame({'ds': df_test['ds']}))
    end_comp = time.time()

    latency = (end_comp - start_comp) / len(df_test)
    comp_lag_mins = latency / 60.0

    df_pred = df_test.merge(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')
    y_pred = np.where((np.abs(df_pred['y'] - df_pred['yhat']) > THRESHOLD_FACTOR * (df_pred['yhat_upper'] - df_pred['yhat_lower'])), 1, 0)
    y_true = df_pred['is_anomaly'].values

    flow_delays = []
    current_event_start = None
    detected = False
    for i in range(len(y_true)):
        if y_true[i] == 1 and (i == 0 or y_true[i-1] == 0):
            current_event_start, detected = i, False
        if y_true[i] == 1 and y_pred[i] == 1 and not detected:
            flow_delays.append(SAMPLING_LAG_MINS + (i - current_event_start) * FREQ_MIN + comp_lag_mins)
            detected = True
        if y_true[i] == 0:
            current_event_start, detected = None, False

    return {'y_true': y_true, 'y_pred': y_pred, 'delays': flow_delays, 'latency': latency}

def main():
    train_df = pd.read_csv(TRAIN_FILE)
    flows = train_df['flow_key_id'].unique()
    all_y_true, all_y_pred, all_delays, all_lats = [], [], [], []

    for f in flows:
        res = run_prophet_detection(f, f)
        all_y_true.extend(res['y_true']); all_y_pred.extend(res['y_pred'])
        all_delays.extend(res['delays']); all_lats.append(res['latency'])

    y_t, y_p = np.array(all_y_true), np.array(all_y_pred)
    print("\n" + "=" * 60)
    print("PROPHET AGGREGATED REPORT")
    print("-" * 60)
    print(f"PRECISION: {precision_score(y_t, y_p, zero_division=0):.4f} | RECALL: {recall_score(y_t, y_p, zero_division=0):.4f}")
    print(f"F1 SCORE:  {f1_score(y_t, y_p, zero_division=0):.4f} | AVG DELAY: {np.mean(all_delays):.4f} mins")
    print(f"INFERENCE LATENCY: {np.mean(all_lats):.6f}")
    print("-" * 60)
    print("=" * 60)

if __name__ == "__main__":
    main()

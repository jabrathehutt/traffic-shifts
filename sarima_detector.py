import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import time
import warnings

warnings.filterwarnings('ignore')

TRAIN_FILE = '/root/traffic-shifts/trafpy/trafpy_finetune_normal_data.csv'
TEST_FILE = '/root/traffic-shifts/trafpy/trafpy_master_univariate_data.csv'

SEASONAL_PERIOD = 288
ARIMA_ORDER = (1, 1, 1)
THRESHOLD_ALPHA = 3.0
SAMPLING_LAG_MINS = 2.5
FREQ_MIN = 5.0

def run_fast_sarima_analysis():
    df_train_raw = pd.read_csv(TRAIN_FILE)
    df_test_raw = pd.read_csv(TEST_FILE)
    flows = df_test_raw['flow_key_id'].unique()

    all_y_true, all_y_pred, all_delays, all_lats = [], [], [], []

    for flow_id in flows:
        train_df = df_train_raw[df_train_raw['flow_key_id'] == flow_id].sort_values('timestamp')
        test_df = df_test_raw[df_test_raw['flow_key_id'] == flow_id].sort_values('timestamp')

        train_s = train_df['traffic_volume_Tbits'].values
        test_s = test_df['traffic_volume_Tbits'].values

        train_diff = train_s[SEASONAL_PERIOD:] - train_s[:-SEASONAL_PERIOD]
        model_fit = ARIMA(train_diff, order=ARIMA_ORDER).fit()

        test_diff = test_s[SEASONAL_PERIOD:] - test_s[:-SEASONAL_PERIOD]

        start_comp = time.time()
        res = model_fit.apply(test_diff)
        forecast_diff = res.fittedvalues
        end_comp = time.time()

        latency = (end_comp - start_comp) / len(test_diff)
        all_lats.append(latency)
        comp_lag_mins = latency / 60.0

        residuals = np.abs(test_diff - forecast_diff)
        thresh = np.mean(np.abs(model_fit.resid)) + (THRESHOLD_ALPHA * np.std(model_fit.resid))

        y_pred = (residuals > thresh).astype(int)
        y_true = test_df['is_anomaly'].values[SEASONAL_PERIOD:].astype(int)

        curr_start, detected = None, False
        for i in range(len(y_true)):
            if y_true[i] == 1 and (i == 0 or y_true[i-1] == 0):
                curr_start, detected = i, False
            if y_true[i] == 1 and y_pred[i] == 1 and not detected:
                all_delays.append(SAMPLING_LAG_MINS + (i - curr_start) * FREQ_MIN + comp_lag_mins)
                detected = True
            if y_true[i] == 0:
                curr_start, detected = None, False

        all_y_true.extend(y_true); all_y_pred.extend(y_pred)

    y_t, y_p = np.array(all_y_true), np.array(all_y_pred)
    print("\n" + "=" * 60)
    print("FAST SEASONAL ARIMA AGGREGATED REPORT")
    print("-" * 60)
    print(f"PRECISION: {precision_score(y_t, y_p, zero_division=0):.4f} | RECALL: {recall_score(y_t, y_p, zero_division=0):.4f}")
    print(f"F1-SCORE:  {f1_score(y_t, y_p, zero_division=0):.4f} | AVG DELAY: {np.mean(all_delays):.4f} mins")
    print(f"INFERENCE LATENCY: {np.mean(all_lats):.6f}")
    print("-" * 60)
    print("=" * 60)

if __name__ == "__main__":
    run_fast_sarima_analysis()

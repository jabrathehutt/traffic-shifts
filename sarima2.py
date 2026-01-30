import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings('ignore')

def run_precision_optimized_sarima():
    df_train_raw = pd.read_csv('/root/traffic-shifts/llama/lag-llama/trafpy_finetune_normal_data.csv')
    df_test_raw = pd.read_csv('/root/traffic-shifts/trafpy/trafpy_master_univariate_data.csv')

    train_series = df_train_raw[df_train_raw['flow_key_id'] == df_train_raw['flow_key_id'].unique()[0]]['traffic_volume_Tbits'].values
    test_df = df_test_raw[df_test_raw['flow_key_id'] == df_test_raw['flow_key_id'].unique()[0]].sort_values('timestamp')
    test_series = test_df['traffic_volume_Tbits'].values
    y_true = test_df['is_anomaly'].values

    # 1. LOG TRANSFORMATION (Critical for Precision)
    # We add 1e-6 to avoid log(0)
    train_log = np.log1p(train_series)
    test_log = np.log1p(test_series)

    print("Fitting SARIMA on Log-Transformed Space...")
    
    # 2. D=0 (No hard seasonal subtraction)
    model = ARIMA(train_log, order=(1, 0, 1), seasonal_order=(1, 0, 0, 144))
    model_fit = model.fit(method='innovations_mle', low_memory=True)

    # 3. Apply to test
    res = model_fit.apply(test_log)
    residuals = res.resid
    
    # 4. Z-Score on Log-Residuals
    resid_std = np.std(model_fit.resid)
    # Using a much higher threshold (7.0) because we are in Log-space
    z_scores = np.abs(residuals) / (resid_std + 1e-6)
    
    # 5. Moving Average on Z-Scores (Filter micro-jitter)
    z_smoothed = pd.Series(z_scores).rolling(window=3).mean().fillna(0).values
    
    y_pred = (z_smoothed > 1500.0).astype(int)

    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n" + "=" * 45)
    print("LOG-TRANSFORMED SARIMA RESULTS")
    print("-" * 45)
    print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")
    print("=" * 45)

if __name__ == "__main__":
    run_precision_optimized_sarima()

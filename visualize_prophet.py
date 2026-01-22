import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from prophet import Prophet
from sklearn.metrics import precision_score, recall_score, f1_score

# --- CONFIG ---
# Using the path provided in your previous turn
TEST_CSV = "/root/traffic-shifts/trafpy/trafpy_master_univariate_data.csv"

def visualize_prophet_clean():
    # 1. Load Data
    df = pd.read_csv(TEST_CSV)
    df['ds'] = pd.to_datetime(df['timestamp'])
    df['y'] = df['traffic_volume_Tbits']
    
    # 2. Fit Prophet (Using the full test set logic to find the anomaly)
    # Note: In a thesis, you'd usually train on clean and predict on test.
    # Here we simulate the supervised detection logic.
    m = Prophet(interval_width=0.99) # 99% CI for backbone stability
    m.fit(df[['ds', 'y']])
    forecast = m.predict(df[['ds']])
    
    # 3. Calculate Residuals and Detections
    # Prophet detection is typically y > yhat_upper
    df['yhat'] = forecast['yhat'].values
    df['yhat_upper'] = forecast['yhat_upper'].values
    df['yhat_lower'] = forecast['yhat_lower'].values
    df['residual'] = np.abs(df['y'] - df['yhat'])
    
    # Threshold based on the 99% CI Prophet provides
    df['is_detected'] = (df['y'] > df['yhat_upper']).astype(int)
    
    # 4. Filter for a clear view of the anomaly window
    anomaly_idx = df[df['is_anomaly'] == 1].index[0]
    plot_df = df.iloc[max(0, anomaly_idx-50) : min(len(df), anomaly_idx+150)].copy()

    # 5. PLOTTING
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Panel 1: Probabilistic Forecast
    ax1.plot(plot_df['ds'], plot_df['y'], label='Actual Traffic', color='#1f77b4', linewidth=1.5)
    ax1.plot(plot_df['ds'], plot_df['yhat'], label='Prophet Mean Prediction', color='#ff7f0e', linestyle='--')
    ax1.fill_between(plot_df['ds'], plot_df['yhat_lower'], plot_df['yhat_upper'], 
                     color='#ff7f0e', alpha=0.2, label='99% Confidence Interval')
    ax1.set_title("Prophet Probabilistic Forecast (TrafPy Backbone)", fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Panel 2: Residual Analysis
    ax2.fill_between(plot_df['ds'], plot_df['residual'], color='#d62728', alpha=0.2)
    ax2.set_ylabel("Residual Magnitude")
    ax2.set_title("Anomaly Scoring (Actual vs. Expected Deviation)", fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.6)

    # Panel 3: Instantaneous Detection Response
    ax3.fill_between(plot_df['ds'], plot_df['is_anomaly'], color='green', alpha=0.3, label='Ground Truth')
    ax3.step(plot_df['ds'], plot_df['is_detected'], where='post', color='red', label='Prophet Alarm', linewidth=2)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['Normal', 'ANOMALY'])
    ax3.set_title("Detection Alignment (Cleaned Timestamps)", fontsize=12)
    ax3.legend(loc='upper left')

    # --- THE TIMESTAMP FIX ---
    # Use AutoDateLocator and DateFormatter to prevent overlapping text
    ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=25, ha='right') # Angled and right-aligned for readability

    plt.tight_layout()
    plt.savefig("prophet_anomaly_report.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    visualize_prophet_clean()

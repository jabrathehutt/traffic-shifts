import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_stochastic_traffic(csv_file):
    # Load data
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Pick the first flow with anomalies
    unique_anomalous_flows = df[df['is_anomaly'] == True]['flow_key_id'].unique()
    if len(unique_anomalous_flows) == 0:
        print("No anomalies found in data. Checking first available flow instead.")
        flow_id = df['flow_key_id'].iloc[0]
    else:
        flow_id = unique_anomalous_flows[0]
        
    flow_data = df[df['flow_key_id'] == flow_id]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [1, 1]})
    
    # --- PANEL 1: Time Series (Matched to Mbits) ---
    ax1.plot(flow_data['timestamp'], flow_data['traffic_volume_Mbits'], 
             label='Stochastic Traffic (Mbits)', color='#1f77b4', alpha=0.8)
    
    # Shade anomaly window
    anomalies = flow_data[flow_data['is_anomaly'] == True]
    if not anomalies.empty:
        ax1.axvspan(anomalies['timestamp'].min(), anomalies['timestamp'].max(), 
                    color='red', alpha=0.2, label='Anomaly Window')

    ax1.set_title(f"Univariate Traffic Flow: {flow_id}", fontsize=14)
    ax1.set_ylabel("Traffic Volume (Mbits)")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # --- PANEL 2: Distribution Comparison (Matched to Mbits) ---
    normal_vol = flow_data[flow_data['is_anomaly'] == False]['traffic_volume_Mbits']
    anomaly_vol = flow_data[flow_data['is_anomaly'] == True]['traffic_volume_Mbits']
    
    sns.kdeplot(normal_vol, ax=ax2, fill=True, color='blue', label='Normal (Lognormal)')
    
    # If the anomaly surge is flat, use a histogram fallback; otherwise, use KDE
    if not anomaly_vol.empty:
        if anomaly_vol.var() > 1e-6:
            sns.kdeplot(anomaly_vol, ax=ax2, fill=True, color='red', label='Anomaly (High-Variance Surge)')
        else:
            ax2.hist(anomaly_vol, bins=10, alpha=0.5, color='red', label='Anomaly (Fixed Surge)', density=True)
    
    ax2.set_title("Statistical Signature: Normal vs. Anomaly", fontsize=14)
    ax2.set_xlabel("Traffic Volume (Mbits)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('trafpy_visualization.png')
    print("New visualization saved as trafpy_visualization.png")

if __name__ == "__main__":
    visualize_stochastic_traffic('trafpy_master_univariate_data.csv')

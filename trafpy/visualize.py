import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def visualize_stochastic_traffic(csv_file):
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Select the first flow that actually contains anomalies
    anomalous_flows = df[df['is_anomaly'] == True]['flow_key_id'].unique()
    if len(anomalous_flows) == 0:
        print("Error: No anomalies found in the dataset. Check your generation script.")
        return

    flow_id = anomalous_flows[0]
    flow_data = df[df['flow_key_id'] == flow_id]

    # Change to 1 row, 1 column and reduce height for a single panel
    fig, ax1 = plt.subplots(1, 1, figsize=(15, 7))

    # --- PANEL 1: Time Series ---
    ax1.plot(flow_data['timestamp'], flow_data['traffic_volume_Tbits'],
             label='Stochastic Traffic (TrafPy)', color='#1f77b4', alpha=0.7)

    anomalies = flow_data[flow_data['is_anomaly'] == True]
    if not anomalies.empty:
        ax1.axvspan(anomalies['timestamp'].min(), anomalies['timestamp'].max(),
                    color='red', alpha=0.15, label='Anomaly Window')

    ax1.set_title(f"Traffic Flow: {flow_id}", fontsize=14)
    ax1.set_ylabel("Traffic Volume (Tbits)")
    ax1.set_xlabel("Timestamp")
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig('trafpy_visualization_single.png')
    print("Plot saved as trafpy_visualization.png")
    plt.show()

if __name__ == "__main__":
    # Ensure this path matches your current file location
    visualize_stochastic_traffic('trafpy_master_univariate_data.csv')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
INPUT_FILE = 'trafpy_master_univariate_data.csv'

def visualize_generated_data():
    # 1. Load the data
    try:
        df = pd.read_csv(INPUT_FILE, parse_dates=['timestamp'])
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found. Run your generator first.")
        return

    # 2. Extract the first flow for visualization
    first_flow_id = df['flow_key_id'].unique()[0]
    flow_df = df[df['flow_key_id'] == first_flow_id].sort_values('timestamp')

    # 3. Create the Plot
    plt.figure(figsize=(15, 7))
    
    # Plot normal traffic
    plt.plot(flow_df['timestamp'], flow_df['traffic_volume_Tbits'], 
             label='Normal Traffic (Tbits)', color='blue', alpha=0.6, linewidth=1)
    
    # Highlight anomalies in red
    anomalies = flow_df[flow_df['is_anomaly'] == True]
    plt.scatter(anomalies['timestamp'], anomalies['traffic_volume_Tbits'], 
                color='red', label='Injected Anomalies', s=15, zorder=5)

    # 4. Annotate for Thesis Clarity
    plt.title(f"Subtle Anomaly Injection Visualization: {first_flow_id}", fontsize=14)
    plt.xlabel("Timestamp", fontsize=12)
    plt.ylabel("Traffic Volume (Tbits)", fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    
    # Add a zoom-in callout for the first drift if it exists
    if not anomalies.empty:
        plt.annotate('Structural Drift/Spike', 
                     xy=(anomalies['timestamp'].iloc[0], anomalies['traffic_volume_Tbits'].iloc[0]), 
                     xytext=(anomalies['timestamp'].iloc[0], anomalies['traffic_volume_Tbits'].iloc[0] + 2),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    plt.tight_layout()
    plt.savefig('anomaly_visualization.png')
    print("Visualization saved as 'anomaly_visualization.png'.")
    plt.show()

if __name__ == "__main__":
	visualize_generated_data()

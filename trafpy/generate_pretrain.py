import trafpy.generator as tpg
from trafpy.generator.src.dists import val_dists
import pandas as pd
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
# Pretraining usually needs longer sequences or more flows
START_DATE = '2025-01-01 00:00'
END_DATE = '2025-01-20 00:00' 
FREQUENCY = '10min' 
OUTPUT_FILE = '/root/traffic-shifts/llama/lag-llama/trafpy_pretrain_data.csv'

def generate_clean_stochastic_volume(time_index, mu_val, sigma_val):
    volumes = []
    events_per_interval = 200 # Higher density for better baseline stability
    for _ in range(len(time_index)):
        # Generate pure lognormal traffic (No anomalies)
        flow_sizes = val_dists.gen_lognormal_dist(mu_val, sigma_val, 1, 1e9, events_per_interval)
        volumes.append(sum(flow_sizes) / 1e12)
    return np.array(volumes)

def generate_pretrain_dataset():
    time_index = pd.date_range(start=START_DATE, end=END_DATE, freq=FREQUENCY, inclusive='left')
    all_flows = []
    
    # Create 10 different "Normal" behaviors for the model to learn
    variations = [
        {'mu': 14, 'sigma': 2.0}, # Low volume
        {'mu': 15, 'sigma': 2.2}, # Medium
        {'mu': 16, 'sigma': 1.8}, # High volume, low burstiness
        {'mu': 14, 'sigma': 3.0}  # Low volume, high burstiness (Heavy tail)
    ]

    print("Generating Pretraining Dataset (Normal Traffic Only)...")
    for idx, var in enumerate(variations):
        for i in range(5): # 5 flows per variation
            flow_id = f"pretrain_var{idx}_flow{i}"
            volume = generate_clean_stochastic_volume(time_index, var['mu'], var['sigma'])
            
            df = pd.DataFrame({
                'timestamp': time_index,
                'traffic_volume_Tbits': volume,
                'is_anomaly': False, # Always False for pretraining
                'flow_key_id': flow_id
            })
            all_flows.append(df)

    pd.concat(all_flows).to_csv(OUTPUT_FILE, index=False)
    print(f"Pretraining file saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_pretrain_dataset()

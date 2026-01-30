import pandas as pd
import numpy as np
from trafpy.generator.src.dists import val_dists
from tqdm import tqdm

# --- CONFIGURATION ---
START_DATE = '2025-01-01 00:00'
END_DATE = '2025-08-30 00:00' 
FREQUENCY = '10min'
OUTPUT_FILE = 'trafpy_pretrain_data_extended.csv'

def generate_pretrain_dataset():
    time_index = pd.date_range(start=START_DATE, end=END_DATE, freq=FREQUENCY, inclusive='left')
    all_flows = []

    # Diversified variations to help the Transformer generalize across load levels
    variations = [
        {'base_mu': 13.5, 'amplitude': 0.8, 'sigma': 1.1},
        {'base_mu': 14.0, 'amplitude': 0.6, 'sigma': 1.2},
        {'base_mu': 14.5, 'amplitude': 0.9, 'sigma': 1.0},
        {'base_mu': 15.0, 'amplitude': 0.7, 'sigma': 1.3}
    ]

    print(f"Generating 4-Month Diurnal Pretraining Dataset ({len(variations)*15} flows)...")
    for v_idx, var in enumerate(variations):
        for i in tqdm(range(15), desc=f"Variation {v_idx}"):
            flow_id = f"pretrain_v{v_idx}_f{i}"
            volumes = []
            
            # Using the exact phase shift from the Master script (ts.hour - 8)
            # This ensures peaks occur at 14:00 (2 PM)
            diurnal_factors = var['amplitude'] * np.sin(2 * np.pi * (time_index.hour - 8) / 24)
            mus = var['base_mu'] + diurnal_factors
            
            for mu in mus:
                flow_sizes = val_dists.gen_lognormal_dist(mu, var['sigma'], 1, 1e8, 100)
                volumes.append(sum(flow_sizes) / 1e12)

            all_flows.append(pd.DataFrame({
                'timestamp': time_index,
                'traffic_volume_Tbits': volumes,
                'is_anomaly': False,
                'flow_key_id': flow_id
            }))

    pd.concat(all_flows).to_csv(OUTPUT_FILE, index=False)
    print(f"Extended pretraining file saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_pretrain_dataset()

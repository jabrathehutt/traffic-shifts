import trafpy.generator as tpg
from trafpy.generator.src.dists import val_dists
import pandas as pd
import numpy as np
import random
from tqdm import tqdm

# --- CONFIGURATION ---
START_DATE = '2025-01-01 00:00'
END_DATE = '2025-01-08 00:00'
FREQUENCY = '10min'
OUTPUT_FILE = 'trafpy_master_univariate_data.csv'
NUM_FLOWS = 5

def generate_diurnal_thesis_dataset():
    time_index = pd.date_range(START_DATE, END_DATE, freq=FREQUENCY, inclusive='left')
    all_flows = []
    
    print(f"Generating {NUM_FLOWS} Diurnal Flows...")

    for f_idx in range(NUM_FLOWS):
        flow_id = f"Flow_{f_idx}"
        volumes = np.zeros(len(time_index))
        is_anomaly = np.zeros(len(time_index), dtype=bool)

        # 1. Baseline: Diurnal Cycle + TrafPy Jitter
        base_mu = 14.0
        sigma = 1.2 # Heavy-tailed but modelable
        
        for i, ts in enumerate(time_index):
            # Diurnal factor: peaks at 14:00 (2 PM), lowest at 02:00
            # Higher amplitude (0.8) makes the cycle very clear for the Transformer
            diurnal_scale = 0.8 * np.sin(2 * np.pi * (ts.hour - 8) / 24)
            current_mu = base_mu + diurnal_scale
            
            # Generate TrafPy heavy-tailed sample
            flow_sizes = val_dists.gen_lognormal_dist(current_mu, sigma, 1, 1e7, 50)
            volumes[i] = sum(flow_sizes) / 1e12

        # 2. Inject Anomalies (Structural)
        for _ in range(4):
            # --- SUDDEN SPIKE ---
            s_idx = random.randint(150, len(time_index) - 20)
            dur = random.randint(1, 3)
            volumes[s_idx : s_idx+dur] += random.uniform(4.0, 7.0)
            is_anomaly[s_idx : s_idx+dur] = True

            # --- GRADUAL DRIFT ---
            d_idx = random.randint(150, len(time_index) - 50)
            while any(is_anomaly[d_idx : d_idx+12]): d_idx = random.randint(150, len(time_index)-50)
            d_dur = random.randint(8, 15)
            # Linear drift ramp
            volumes[d_idx : d_idx+d_dur] += np.linspace(0, 4.5, d_dur)
            is_anomaly[d_idx : d_idx+d_dur] = True

        df = pd.DataFrame({
            'timestamp': time_index,
            'traffic_volume_Tbits': volumes,
            'is_anomaly': is_anomaly,
            'flow_key_id': flow_id
        })
        all_flows.append(df)

    pd.concat(all_flows, ignore_index=True).to_csv(OUTPUT_FILE, index=False)
    print(f"Diurnal dataset saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_diurnal_thesis_dataset()

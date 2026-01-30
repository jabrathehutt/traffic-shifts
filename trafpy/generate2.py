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

def generate_thesis_dataset():
    time_index = pd.date_range(START_DATE, END_DATE, freq=FREQUENCY, inclusive='left')
    all_flows = []
    
    print(f"Generating {NUM_FLOWS} flows: TrafPy Baseline + Structural Anomalies...")

    for f_idx in range(NUM_FLOWS):
        flow_id = f"Flow_{f_idx}"
        volumes = np.zeros(len(time_index))
        is_anomaly = np.zeros(len(time_index), dtype=bool)

        # 1. Baseline: Standard TrafPy Log-Normal Jitter
        # We use mu=14.0 and sigma=1.2 (reduced from 3.0 to make the baseline stable enough to learn)
        for i in range(len(time_index)):
            flow_sizes = val_dists.gen_lognormal_dist(14.0, 1.2, 1, 1e7, 50)
            volumes[i] = sum(flow_sizes) / 1e12

        # 2. Inject 5 Spikes and 5 Drifts per flow
        for _ in range(5):
            # --- SUDDEN SPIKE ---
            s_start = random.randint(100, len(time_index) - 20)
            s_dur = random.randint(2, 4)
            s_indices = np.arange(s_start, s_start + s_dur)
            # Magnitude: Add 3-5x the average volume
            volumes[s_indices] += random.uniform(2.0, 5.0) 
            is_anomaly[s_indices] = True

            # --- GRADUAL DRIFT ---
            d_start = random.randint(100, len(time_index) - 40)
            while any(is_anomaly[d_start:d_start+10]): # Avoid overlap
                d_start = random.randint(100, len(time_index) - 40)
            d_dur = random.randint(10, 20)
            d_indices = np.arange(d_start, d_start + d_dur)
            
            # Linear ramp magnitude
            ramp = np.linspace(0, 3.0, d_dur)
            volumes[d_indices] += ramp
            is_anomaly[d_indices] = True

        df = pd.DataFrame({
            'timestamp': time_index,
            'traffic_volume_Tbits': volumes,
            'is_anomaly': is_anomaly,
            'flow_key_id': flow_id
        })
        all_flows.append(df)

    pd.concat(all_flows, ignore_index=True).to_csv(OUTPUT_FILE, index=False)
    print(f"Dataset generated: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_thesis_dataset()

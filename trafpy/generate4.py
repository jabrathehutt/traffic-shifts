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
NUM_FLOWS = 5 # Increased for better statistical significance in thesis

def generate_subtle_thesis_dataset():
    time_index = pd.date_range(START_DATE, END_DATE, freq=FREQUENCY, inclusive='left')
    all_flows = []

    print(f"Generating {NUM_FLOWS} Subtle Diurnal Flows...")

    for f_idx in range(NUM_FLOWS):
        flow_id = f"Flow_{f_idx}"
        volumes = np.zeros(len(time_index))
        is_anomaly = np.zeros(len(time_index), dtype=bool)

        # 1. Baseline: Diurnal Cycle + TrafPy Jitter
        base_mu = 14.0
        sigma = 1.2 

        for i, ts in enumerate(time_index):
            diurnal_scale = 0.8 * np.sin(2 * np.pi * (ts.hour - 8) / 24)
            current_mu = base_mu + diurnal_scale

            flow_sizes = val_dists.gen_lognormal_dist(current_mu, sigma, 1, 1e7, 50)
            volumes[i] = sum(flow_sizes) / 1e12

        # 2. Inject SUBTLE Anomalies
        for _ in range(3):
            # --- SUBTLE SPIKE ---
            # Magnitude is now 1.5 - 2.5 (similar to a natural TrafPy burst)
            s_idx = random.randint(150, len(time_index) - 20)
            dur = random.randint(1, 2)
            volumes[s_idx : s_idx+dur] += random.uniform(1.5, 2.5) 
            is_anomaly[s_idx : s_idx+dur] = True

            # --- SUBTLE GRADUAL DRIFT ---
            d_idx = random.randint(150, len(time_index) - 50)
            while any(is_anomaly[d_idx : d_idx+20]): 
                d_idx = random.randint(150, len(time_index)-50)
            
            d_dur = random.randint(10, 20)
            # Ramp max is now 1.8. Added random noise to the ramp itself.
            ramp = np.linspace(0, 1.8, d_dur) + np.random.normal(0, 0.2, d_dur)
            volumes[d_idx : d_idx+d_dur] += np.maximum(0, ramp) # Ensure no negative volume
            is_anomaly[d_idx : d_idx+d_dur] = True

        df = pd.DataFrame({
            'timestamp': time_index,
            'traffic_volume_Tbits': volumes,
            'is_anomaly': is_anomaly,
            'flow_key_id': flow_id
        })
        all_flows.append(df)

    pd.concat(all_flows, ignore_index=True).to_csv(OUTPUT_FILE, index=False)
    print(f"Subtle dataset saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_subtle_thesis_dataset()

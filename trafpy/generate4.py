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
NUM_FLOWS = 1 # Verification run

def generate_diurnal_thesis_dataset():
    time_index = pd.date_range(START_DATE, END_DATE, freq=FREQUENCY, inclusive='left')
    all_flows = []

    print("Generating Traffic with raw scale (no more zero flatline)...")

    for f_idx in range(NUM_FLOWS):
        flow_id = f"Flow_{f_idx}"
        volumes = np.zeros(len(time_index))
        is_anomaly = np.zeros(len(time_index), dtype=bool)

        base_mu = 14.0
        sigma = 1.2 

        for i, ts in enumerate(tqdm(time_index, desc=f"Flow {f_idx}")):
            diurnal_scale = 0.8 * np.sin(2 * np.pi * (ts.hour - 8) / 24)
            current_mu = base_mu + diurnal_scale

            # Use int(1000) samples to create a robust baseline
            # NO DIVISION: We take the raw sum so it shows up on the plot
            flow_sizes = val_dists.gen_lognormal_dist(current_mu, sigma, 1, 1e7, int(1000))
            volumes[i] = sum(flow_sizes) 

        # 2. Inject SUBTLE Anomalies
        # Multiplying ensures they scale naturally with the baseline
        for _ in range(5):
            # --- SUBTLE SPIKE ---
            s_idx = random.randint(150, len(time_index) - 20)
            dur = random.randint(1, 2)
            # Only 20-40% increase
            volumes[s_idx : s_idx+dur] *= random.uniform(1.2, 1.4)
            is_anomaly[s_idx : s_idx+dur] = True

            # --- SUBTLE DRIFT ---
            d_idx = random.randint(150, len(time_index) - 50)
            while any(is_anomaly[d_idx : d_idx+15]): 
                d_idx = random.randint(150, len(time_index)-50)
            d_dur = random.randint(12, 18)
            
            # Subtle 20% ramp
            drift_ramp = np.linspace(1.0, 1.2, d_dur)
            volumes[d_idx : d_idx+d_dur] *= drift_ramp
            is_anomaly[d_idx : d_idx+d_dur] = True

        df = pd.DataFrame({
            'timestamp': time_index,
            'traffic_volume_Tbits': volumes,
            'is_anomaly': is_anomaly,
            'flow_key_id': flow_id
        })
        all_flows.append(df)

    pd.concat(all_flows, ignore_index=True).to_csv(OUTPUT_FILE, index=False)
    print(f"Success! Raw scale dataset saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_diurnal_thesis_dataset()

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

    print(f"Generating {NUM_FLOWS} Subtle Diurnal Flows...")

    for f_idx in range(NUM_FLOWS):
        flow_id = f"Flow_{f_idx}"
        volumes = np.zeros(len(time_index))
        is_anomaly = np.zeros(len(time_index), dtype=bool)

        # 1. Baseline: Diurnal Cycle + TrafPy Jitter
        base_mu = 14.0
        sigma = 1.2 

        for i, ts in enumerate(tqdm(time_index, desc=f"Flow {f_idx}")):
            # Diurnal factor
            diurnal_scale = 0.8 * np.sin(2 * np.pi * (ts.hour - 8) / 24)
            current_mu = base_mu + diurnal_scale

            # Generate more flows (500) to ensure the baseline is thick/visible
            # We also set a higher upper bound for the flow sizes
            flow_sizes = val_dists.gen_lognormal_dist(current_mu, sigma, 1, 1e10, 500)
            
            # Use a divisor that keeps the result in the 10-25 Tbit range
            volumes[i] = sum(flow_sizes) / 1e10

        # 2. Inject Subtle Anomalies (Structural)
        for _ in range(4):
            # --- SUBTLE SPIKE ---
            s_idx = random.randint(150, len(time_index) - 20)
            dur = random.randint(1, 2)
            # Magnitude is now 20-40% of the current volume instead of a huge flat +7.0
            spike_factor = random.uniform(1.2, 1.4)
            volumes[s_idx : s_idx+dur] *= spike_factor
            is_anomaly[s_idx : s_idx+dur] = True

            # --- SUBTLE GRADUAL DRIFT ---
            d_idx = random.randint(150, len(time_index) - 50)
            while any(is_anomaly[d_idx : d_idx+12]): 
                d_idx = random.randint(150, len(time_index)-50)
            d_dur = random.randint(10, 18)
            
            # Gradual ramp up to 25% increase
            drift_ramp = np.linspace(1.0, 1.25, d_dur)
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
    print(f"Subtle Diurnal dataset saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_diurnal_thesis_dataset()

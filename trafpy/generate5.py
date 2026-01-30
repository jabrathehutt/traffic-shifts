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
NUM_FLOWS = 1 

def generate_diurnal_thesis_dataset():
    time_index = pd.date_range(START_DATE, END_DATE, freq=FREQUENCY, inclusive='left')
    all_flows = []

    print("Generating Traffic with Persistent Structural Anomalies...")

    for f_idx in range(NUM_FLOWS):
        flow_id = f"Flow_{f_idx}"
        volumes = np.zeros(len(time_index))
        is_anomaly = np.zeros(len(time_index), dtype=bool)

        base_mu = 14.0
        sigma = 1.2 

        for i, ts in enumerate(tqdm(time_index, desc=f"Flow {f_idx}")):
            diurnal_scale = 0.8 * np.sin(2 * np.pi * (ts.hour - 8) / 24)
            current_mu = base_mu + diurnal_scale
            
            # Generate baseline flows
            flow_sizes = val_dists.gen_lognormal_dist(current_mu, sigma, 1, 1e7, int(1000))
            volumes[i] = sum(flow_sizes) 

        # Calculate a "detection floor" based on the average of non-zero bursts
        # This helps us set an anomaly magnitude that isn't invisible
        typical_burst = np.mean(volumes[volumes > 0])
        anomaly_floor = typical_burst * 0.5 

        # 2. Inject Persistent Anomalies
        for _ in range(5):
            # --- PERSISTENT SPIKE ---
            s_idx = random.randint(150, len(time_index) - 20)
            dur = random.randint(1, 3)
            
            # Add a constant floor + a multiplier
            # This ensures the red dot never sits on the 0 line
            volumes[s_idx : s_idx+dur] = (volumes[s_idx : s_idx+dur] + anomaly_floor) * 1.5
            is_anomaly[s_idx : s_idx+dur] = True

            # --- PERSISTENT GRADUAL DRIFT ---
            d_idx = random.randint(150, len(time_index) - 50)
            while any(is_anomaly[d_idx : d_idx+20]): 
                d_idx = random.randint(150, len(time_index)-50)
            d_dur = random.randint(12, 24) # 2 to 4 hour drifts
            
            # Create a ramp that starts from the anomaly_floor
            # This makes the drift look like a new signal emerging from the idle state
            drift_ramp = np.linspace(anomaly_floor * 0.5, anomaly_floor * 1.5, d_dur)
            volumes[d_idx : d_idx+d_dur] += drift_ramp
            is_anomaly[d_idx : d_idx+d_dur] = True

        df = pd.DataFrame({
            'timestamp': time_index,
            'traffic_volume_Tbits': volumes,
            'is_anomaly': is_anomaly,
            'flow_key_id': flow_id
        })
        all_flows.append(df)

    pd.concat(all_flows, ignore_index=True).to_csv(OUTPUT_FILE, index=False)
    print(f"Dataset with visible structural anomalies saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_diurnal_thesis_dataset()

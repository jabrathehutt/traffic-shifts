import trafpy.generator as tpg
from trafpy.generator.src.dists import val_dists
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import os

# --- DETERMINISM SETUP ---
SEED = 42
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(SEED)

# --- CONFIGURATION ---
START_DATE = '2025-01-01 00:00'
END_DATE = '2025-01-08 00:00'
FREQUENCY = '10min'
OUTPUT_FILE = 'trafpy_master_univariate_data.csv'
NUM_FLOWS = 1

def generate_diurnal_thesis_dataset():
    time_index = pd.date_range(START_DATE, END_DATE, freq=FREQUENCY, inclusive='left')
    all_flows = []

    print(f"Generating Deterministic Dataset (Seed: {SEED})...")

    for f_idx in range(NUM_FLOWS):
        flow_id = f"Flow_{f_idx}"
        volumes = np.zeros(len(time_index))
        is_anomaly = np.zeros(len(time_index), dtype=bool)

        base_mu = 10.0
        sigma = 1.0

        # Generate the smooth diurnal wave
        for i, ts in enumerate(tqdm(time_index, desc=f"Flow {f_idx}")):
            diurnal_scale = 1.5 * np.sin(2 * np.pi * (ts.hour - 8) / 24)
            current_mu = base_mu + diurnal_scale

            # TrafPy uses np.random internally
            flow_sizes = val_dists.gen_lognormal_dist(_mu=current_mu,
                                                      _sigma=sigma,
                                                      min_val=0.01,
                                                      max_val=1e9,
                                                      size=int(2000))
            volumes[i] = sum(flow_sizes) / 1e3

        std_dev = np.std(volumes)

        # 2. Inject Persistent Structural Anomalies (Deterministic random calls)
        for _ in range(6):
            # --- SPIKE ---
            s_idx = random.randint(150, len(time_index) - 20)
            dur = random.randint(1, 3)
            volumes[s_idx : s_idx+dur] += (std_dev * 2.5)
            is_anomaly[s_idx : s_idx+dur] = True

            # --- GRADUAL DRIFT ---
            d_idx = random.randint(150, len(time_index) - 50)
            while any(is_anomaly[d_idx : d_idx+50]): # Checked duration safety
                d_idx = random.randint(150, len(time_index)-50)
            d_dur = random.randint(20, 40)

            drift_ramp = np.linspace(0, std_dev * 3.0, d_dur)
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
    print(f"Saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_diurnal_thesis_dataset()

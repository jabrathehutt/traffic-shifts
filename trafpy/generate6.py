import trafpy.generator as tpg
from trafpy.generator.src.dists import val_dists
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import os

# --- DETERMINISM ---
SEED = 42
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(SEED)

# --- CONFIG ---
START_DATE = '2025-01-01 00:00'
END_DATE = '2025-01-08 00:00'
FREQUENCY = '5min'  # Updated to 5 minutes
OUTPUT_FILE = 'trafpy_master_univariate_data.csv'
NUM_FLOWS = 5

def generate_diurnal_thesis_dataset():
    # Frequency updated here
    time_index = pd.date_range(START_DATE, END_DATE, freq=FREQUENCY, inclusive='left')
    all_flows = []

    print(f"Generating {NUM_FLOWS} Deterministic Flows at {FREQUENCY} resolution...")

    for f_idx in range(NUM_FLOWS):
        flow_id = f"Flow_{f_idx}"
        volumes = np.zeros(len(time_index))
        is_anomaly = np.zeros(len(time_index), dtype=bool)

        base_mu = 10.0
        sigma = 1.0

        for i, ts in enumerate(tqdm(time_index, desc=f"Generating {flow_id}")):
            # Diurnal calculation remains accurate for 5-min intervals
            diurnal_scale = 1.5 * np.sin(2 * np.pi * (ts.hour + ts.minute/60 - 8) / 24)
            current_mu = base_mu + diurnal_scale

            flow_sizes = val_dists.gen_lognormal_dist(_mu=current_mu,
                                                      _sigma=sigma,
                                                      min_val=0.01,
                                                      max_val=1e9,
                                                      size=int(2000))
            volumes[i] = sum(flow_sizes) / 1e3

        std_dev = np.std(volumes)

        for _ in range(6):
            # Spike
            s_idx = random.randint(150, len(time_index) - 20)
            dur = random.randint(2, 6) # Increased duration slightly for 5min resolution
            volumes[s_idx : s_idx+dur] += (std_dev * 2.5)
            is_anomaly[s_idx : s_idx+dur] = True

            # Drift
            d_idx = random.randint(150, len(time_index) - 100)
            while any(is_anomaly[d_idx : d_idx+100]):
                d_idx = random.randint(150, len(time_index)-100)
            d_dur = random.randint(40, 80) # Increased duration for 5min resolution
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
    print(f"Master dataset saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_diurnal_thesis_dataset()

import trafpy.generator as tpg
from trafpy.generator.src.dists import val_dists
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import os

# --- DETERMINISM ---
SEED = 99  # Different seed than master test set
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(SEED)

# --- CONFIG ---
# Use a longer window for fine-tuning to provide more 'examples'
START_DATE = '2025-02-01 00:00'
END_DATE = '2025-09-01 00:00' 
FREQUENCY = '5min'
OUTPUT_FILE = 'trafpy_finetune_labeled_data.csv'
NUM_FLOWS = 3 # Fewer flows but longer time is better for fine-tuning

def generate_labeled_finetune_dataset():
    time_index = pd.date_range(START_DATE, END_DATE, freq=FREQUENCY, inclusive='left')
    all_flows = []

    print(f"Generating Labeled Fine-tuning Data ({len(time_index)} points per flow)...")

    for f_idx in range(NUM_FLOWS):
        flow_id = f"FT_Flow_{f_idx}"
        volumes = np.zeros(len(time_index))
        is_anomaly = np.zeros(len(time_index), dtype=bool)

        base_mu = 10.0
        sigma = 1.0

        # 1. Generate Baseline Diurnal
        for i, ts in enumerate(tqdm(time_index, desc=f"Baseline {flow_id}")):
            diurnal_scale = 1.5 * np.sin(2 * np.pi * (ts.hour + ts.minute/60 - 8) / 24)
            current_mu = base_mu + diurnal_scale
            flow_sizes = val_dists.gen_lognormal_dist(_mu=current_mu, _sigma=sigma, 
                                                    min_val=0.01, max_val=1e9, size=2000)
            volumes[i] = sum(flow_sizes) / 1e3

        std_dev = np.std(volumes)

        # 2. Inject anomalies - We increase the count to make the model 'Supervised'
        # Spike Injection
        for _ in range(15): # More spikes for learning
            s_idx = random.randint(150, len(time_index) - 20)
            dur = random.randint(2, 6)
            volumes[s_idx : s_idx+dur] += (std_dev * random.uniform(2.0, 4.0)) # Varying intensity
            is_anomaly[s_idx : s_idx+dur] = True

        # Drift Injection
        for _ in range(10): # More drifts for learning
            d_idx = random.randint(150, len(time_index) - 100)
            if any(is_anomaly[d_idx : d_idx+100]): continue
            d_dur = random.randint(40, 80)
            drift_ramp = np.linspace(0, std_dev * random.uniform(2.5, 3.5), d_dur)
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
    print(f"Fine-tuning labeled dataset saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_labeled_finetune_dataset()

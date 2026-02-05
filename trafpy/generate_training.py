import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import os
from trafpy.generator.src.dists import val_dists

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def generate_multi_flow_clean_data(filename, days, seed, num_flows=5):
    seed_everything(seed)
    start_date = '2025-01-01 00:00'
    end_date = pd.to_datetime(start_date) + pd.Timedelta(days=days)
    # Frequency updated to 5min
    time_index = pd.date_range(start_date, end_date, freq='5min', inclusive='left')

    all_data = []
    base_mu = 10.0
    sigma = 1.0

    print(f"Generating {num_flows} clean flows for {filename} at 5-min intervals...")

    for f_idx in range(num_flows):
        flow_id = f"Flow_{f_idx}"
        volumes = np.zeros(len(time_index))
        for i, ts in enumerate(tqdm(time_index, desc=f"Clean {flow_id}")):
            # ts.minute/60 ensures a smooth sine wave across 5-min steps
            diurnal_scale = 1.5 * np.sin(2 * np.pi * (ts.hour + ts.minute/60 - 8) / 24)
            current_mu = base_mu + diurnal_scale
            flow_sizes = val_dists.gen_lognormal_dist(_mu=current_mu, _sigma=sigma, size=2000)
            volumes[i] = sum(flow_sizes) / 1e3

        df = pd.DataFrame({
            'timestamp': time_index,
            'traffic_volume_Tbits': volumes,
            'is_anomaly': False,
            'flow_key_id': flow_id
        })
        all_data.append(df)

    pd.concat(all_data, ignore_index=True).to_csv(filename, index=False)
    print(f"Saved: {filename}")

if __name__ == "__main__":
    generate_multi_flow_clean_data('trafpy_pretrain_data_extended.csv', days=120, seed=100)
    generate_multi_flow_clean_data('trafpy_finetune_normal_data.csv', days=7, seed=200)

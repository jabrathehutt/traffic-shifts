import trafpy.generator as tpg
from trafpy.generator.src.dists import val_dists
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import os

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def generate_clean_trafpy_dataset(filename, days, seed):
    # Pin seed for this specific file generation
    seed_everything(seed)
    
    start_date = '2025-01-01 00:00'
    end_date = pd.to_datetime(start_date) + pd.Timedelta(days=days)
    time_index = pd.date_range(start_date, end_date, freq='5min', inclusive='left')

    volumes = np.zeros(len(time_index))
    base_mu = 10.0
    sigma = 1.0

    print(f"Generating {days} days of Deterministic Clean Data for {filename} (Seed: {seed})...")

    for i, ts in enumerate(tqdm(time_index)):
        diurnal_scale = 1.5 * np.sin(2 * np.pi * (ts.hour + ts.minute/60 - 8) / 24)
        current_mu = base_mu + diurnal_scale

        flow_sizes = val_dists.gen_lognormal_dist(_mu=current_mu,
                                                  _sigma=sigma,
                                                  min_val=0.01,
                                                  max_val=1e9,
                                                  size=int(2000))
        volumes[i] = sum(flow_sizes) / 1e3

    df = pd.DataFrame({
        'timestamp': time_index,
        'traffic_volume_Tbits': volumes,
        'is_anomaly': False,
        'flow_key_id': 'Flow_0'
    })

    df.to_csv(filename, index=False)
    print(f"Saved: {filename}")

if __name__ == "__main__":
    # Generate Pre-training Baseline (Longer)
    generate_clean_trafpy_dataset('trafpy_pretrain_data_extended.csv', days=120, seed=100)

    # Generate Fine-tuning Baseline (Standard duration)
    generate_clean_trafpy_dataset('trafpy_finetune_normal_data.csv', days=7, seed=200)

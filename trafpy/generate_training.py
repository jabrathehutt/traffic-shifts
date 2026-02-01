import trafpy.generator as tpg
from trafpy.generator.src.dists import val_dists
import pandas as pd
import numpy as np
from tqdm import tqdm

def generate_clean_trafpy_dataset(filename, days):
    start_date = '2025-01-01 00:00'
    end_date = pd.to_datetime(start_date) + pd.Timedelta(days=days)
    time_index = pd.date_range(start_date, end_date, freq='10min', inclusive='left')
    
    volumes = np.zeros(len(time_index))
    # Standard TrafPy Baseline Parameters
    base_mu = 10.0 
    sigma = 1.0 

    print(f"Generating {days} days of Clean TrafPy Data for {filename}...")

    for i, ts in enumerate(tqdm(time_index)):
        # Maintain identical diurnal pattern
        diurnal_scale = 1.5 * np.sin(2 * np.pi * (ts.hour - 8) / 24)
        current_mu = base_mu + diurnal_scale
        
        # High-resolution sampling (2000 samples per slot)
        flow_sizes = val_dists.gen_lognormal_dist(_mu=current_mu, 
                                                  _sigma=sigma, 
                                                  min_val=0.01, 
                                                  max_val=1e9, 
                                                  size=int(2000))
        
        # Normalization to Tbits matching your successful baseline
        volumes[i] = sum(flow_sizes) / 1e3

    df = pd.DataFrame({
        'timestamp': time_index,
        'traffic_volume_Tbits': volumes,
        'is_anomaly': False, # Clean training data
        'flow_key_id': 'Flow_0'
    })

    df.to_csv(filename, index=False)
    print(f"Saved: {filename}")

if __name__ == "__main__":
    # Generate Pre-training Baseline (Longer)
    generate_clean_trafpy_dataset('trafpy_pretrain_data_extended.csv', days=120)
    
    # Generate Fine-tuning Baseline (Standard duration)
    generate_clean_trafpy_dataset('trafpy_finetune_normal_data.csv', days=7)

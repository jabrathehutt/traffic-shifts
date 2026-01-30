import pandas as pd
import numpy as np
from trafpy.generator.src.dists import val_dists
from tqdm import tqdm

OUTPUT_FILE = 'trafpy_finetune_normal_data.csv'

def generate_finetune_dataset():
    # 1 Week of calibration data
    time_index = pd.date_range(start='2025-01-01', end='2025-01-08', freq='10min', inclusive='left')
    all_flows = []

    print("Generating Diurnal Finetuning Dataset (Calibration)...")
    for i in tqdm(range(20), desc="Finetune Flows"):
        flow_id = f"finetune_flow_{i}"
        volumes = []
        
        # Matches the baseline logic in the Master evaluation dataset
        # base_mu=14.0, amplitude=0.8, phase_shift=-8
        diurnal_factors = 0.8 * np.sin(2 * np.pi * (time_index.hour - 8) / 24)
        mus = 14.0 + diurnal_factors
        
        for mu in mus:
            flow_sizes = val_dists.gen_lognormal_dist(mu, 1.2, 1, 1e7, 50)
            volumes.append(sum(flow_sizes) / 1e12)

        all_flows.append(pd.DataFrame({
            'timestamp': time_index,
            'traffic_volume_Tbits': volumes,
            'is_anomaly': False,
            'flow_key_id': flow_id
        }))

    pd.concat(all_flows).to_csv(OUTPUT_FILE, index=False)
    print(f"Finetuning file saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_finetune_dataset()

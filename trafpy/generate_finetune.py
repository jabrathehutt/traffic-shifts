import trafpy.generator as tpg
from trafpy.generator.src.dists import val_dists
import pandas as pd
import numpy as np
from tqdm import tqdm

OUTPUT_FILE = '/root/traffic/llama/lag-llama/trafpy_finetune_normal_data.csv'

def generate_finetune_dataset():
    # Use same time range as your experiment
    time_index = pd.date_range(start='2025-01-01', end='2025-01-08', freq='10min', inclusive='left')
    all_flows = []
    
    # Target groups from your thesis setup
    groups = ['Backbone_Linear_Normal', 'AS_Shift_Normal']

    print("Generating Finetuning Dataset (Targeted Normal Behavior)...")
    for group_name in groups:
        for i in range(10): # Generate 10 examples of "Normal" for each group
            # Use the EXACT 'Normal' params from your main script (mu=14, sigma=2)
            flow_sizes_list = []
            for _ in range(len(time_index)):
                sizes = val_dists.gen_lognormal_dist(14, 2, 1, 1e7, 50)
                flow_sizes_list.append(sum(sizes) / 1e12)
            
            df = pd.DataFrame({
                'timestamp': time_index,
                'traffic_volume_Tbits': np.array(flow_sizes_list),
                'is_anomaly': False,
                'flow_key_id': f"{group_name}_finetune_{i}"
            })
            all_flows.append(df)

    pd.concat(all_flows).to_csv(OUTPUT_FILE, index=False)
    print(f"Finetuning file saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_finetune_dataset()

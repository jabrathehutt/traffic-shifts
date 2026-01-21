import trafpy.generator as tpg
from trafpy.generator.src.dists import val_dists
import pandas as pd
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
START_DATE = '2025-01-01 00:00'
END_DATE = '2025-01-08 00:00'
FREQUENCY = '10min' 
OUTPUT_FILE = 'trafpy_master_univariate_data.csv'
NUM_FLOWS_PER_GROUP = 2 

# AS Constants
AS_IDS = {'AS100': 100, 'AS200': 200, 'AS300': 300, 'AS400': 400}

def generate_stochastic_volume(time_index, is_anomaly_array):
    volumes = []
    events_per_interval = 50
    
    for i in range(len(time_index)):
        if not is_anomaly_array[i]:
            # Normal: mu=14. We divide by 1e3 (Kbits) to keep numbers large and visible
            flow_sizes = val_dists.gen_lognormal_dist(_mu=14, _sigma=2, min_val=1, max_val=1e7, size=events_per_interval)
            val = sum(flow_sizes) / 1e3 
        else:
            # Anomaly: mu=18 + a forced jump of 500,000 units
            flow_sizes = val_dists.gen_lognormal_dist(_mu=18, _sigma=3, min_val=1, max_val=1e9, size=events_per_interval)
            val = (sum(flow_sizes) / 1e3) + 500000 
            
        volumes.append(val)
    return np.array(volumes)

def generate_full_master_dataset():
    time_index = pd.date_range(start=START_DATE, end=END_DATE, freq=FREQUENCY, inclusive='left')
    all_flows = []

    for template in master_templates:
        for i in range(NUM_FLOWS_PER_GROUP):
            src_asn = template['asn']['src'] + i
            dst_asn = template['asn']['dst'] + i
            flow_id = f"{src_asn}-{dst_asn}_{template['group']}"
            is_anomaly = (time_index >= template['anomaly_window'][0]) & (time_index <= template['anomaly_window'][1])
            traffic_volume = generate_stochastic_volume(time_index, is_anomaly)
            
            df = pd.DataFrame({
                'timestamp': time_index,
                'traffic_volume_Mbits': traffic_volume, # Kept name Mbits for visualize.py
                'is_anomaly': is_anomaly,
                'flow_key_id': flow_id,
                'sourceAS': src_asn,
                'destinationAS': dst_asn,
                'handoverAS': template['asn']['hnd'],
                'nexthopAS': template['asn']['nex']
            })
            all_flows.append(df)

    final_df = pd.concat(all_flows, ignore_index=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Success. Tail of data:\n{final_df[['traffic_volume_Mbits', 'is_anomaly']].tail(5)}")

master_templates = [
    {'group': 'TrafPy_Backbone_Linear', 'asn': {'src': 1001, 'dst': 4001, 'hnd': 200, 'nex': 300}, 'anomaly_window': ('2025-01-07 04:00', '2025-01-07 10:00')},
    {'group': 'TrafPy_AS_Shift', 'asn': {'src': 5001, 'dst': 5002, 'hnd': 300, 'nex': 100}, 'anomaly_window': ('2025-01-07 12:00', '2025-01-07 18:00')}
]

if __name__ == "__main__":
    generate_full_master_dataset()

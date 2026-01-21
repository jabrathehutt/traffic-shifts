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
NUM_FLOWS_PER_GROUP = 2 # Set to your desired number of flows per group

# AS Constants
AS_IDS = {'AS100': 100, 'AS200': 200, 'AS300': 300, 'AS400': 400}

# --- TRAFPY VOLUME ENGINE ---

def generate_stochastic_volume(time_index, is_anomaly_array):
    """Generates Mbits using TrafPy distributions based on anomaly state."""
    volumes = []
    events_per_interval = 50
    
    print("Generating volume points...")
    for i in range(len(time_index)):
        if not is_anomaly_array[i]:
            # NORMAL: mu=14. Lower variance, standard baseline.
            flow_sizes = val_dists.gen_lognormal_dist(_mu=14, _sigma=2, min_val=1, max_val=1e7, size=events_per_interval)
            val = sum(flow_sizes) / 1e6 # MBits
        else:
            # ANOMALY: mu=18. Higher volume and higher variance.
            flow_sizes = val_dists.gen_lognormal_dist(_mu=18, _sigma=3, min_val=1, max_val=1e9, size=events_per_interval)
            # Add a forced baseline shift to make the surge undeniable
            val = (sum(flow_sizes) / 1e6) + 500 
            
        volumes.append(val)
        
    return np.array(volumes)

# --- MASTER TEMPLATES ---

master_templates = [
    {
        'group': 'TrafPy_Backbone_Linear',
        'asn': {'src': 1001, 'dst': 4001, 'hnd': AS_IDS['AS200'], 'nex': AS_IDS['AS300']},
        'anomaly_window': ('2025-01-07 04:00', '2025-01-07 10:00')
    },
    {
        'group': 'TrafPy_AS_Shift',
        'asn': {'src': 5001, 'dst': 5002, 'hnd': AS_IDS['AS300'], 'nex': AS_IDS['AS100']},
        'anomaly_window': ('2025-01-07 12:00', '2025-01-07 18:00')
    }
]

# --- EXECUTION ---

def generate_full_master_dataset():
    time_index = pd.date_range(start=START_DATE, end=END_DATE, freq=FREQUENCY, inclusive='left')
    all_flows = []

    for template in master_templates:
        print(f"Processing Group: {template['group']}")
        for i in tqdm(range(NUM_FLOWS_PER_GROUP)):
            src_asn = template['asn']['src'] + i
            dst_asn = template['asn']['dst'] + i
            flow_id = f"{src_asn}-{dst_asn}_{template['group']}"
            
            is_anomaly = (time_index >= template['anomaly_window'][0]) & \
                         (time_index <= template['anomaly_window'][1])
            
            traffic_volume = generate_stochastic_volume(time_index, is_anomaly)
            
            df = pd.DataFrame({
                'timestamp': time_index,
                'traffic_volume_Mbits': traffic_volume,
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
    print(f"\nSaved {len(final_df)} rows to {OUTPUT_FILE}")
    return final_df

if __name__ == "__main__":
    generate_full_master_dataset()

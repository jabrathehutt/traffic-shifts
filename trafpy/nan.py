import pandas as pd
import numpy as np

FILE_PATH = 'trafpy_finetune_labeled_data.csv'

def patch_data():
    df = pd.read_csv(FILE_PATH)
    
    # Store the count for verification
    anomaly_count = df['is_anomaly'].sum()
    
    # CRITICAL: Set volume to NaN where an anomaly exists
    # This forces Lag-Llama's 'AddObservedValuesIndicator' to mask these points
    df.loc[df['is_anomaly'] == True, 'traffic_volume_Tbits'] = np.nan
    
    df.to_csv(FILE_PATH, index=False)
    print(f"Patched {anomaly_count} points to NaN. run.py will now ignore these during loss calculation.")

if __name__ == "__main__":
    patch_data()

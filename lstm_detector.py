import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import time
import warnings

warnings.filterwarnings('ignore')

TRAIN_FILE = '/root/traffic-shifts/trafpy/trafpy_finetune_normal_data.csv'
TEST_FILE = '/root/traffic-shifts/trafpy/trafpy_master_univariate_data.csv'

SEQUENCE_LENGTH = 50
SEASONAL_PERIOD = 288
EPOCHS = 15
BATCH_SIZE = 64
THRESHOLD_ALPHA = 2.5
SAMPLING_LAG_MINS = 2.5
FREQ_MIN = 5.0

class SeasonalDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len
    def __len__(self):
        return len(self.data) - self.seq_len
    def __getitem__(self, index):
        return self.data[index:index + self.seq_len], self.data[index + self.seq_len]

class GitHubLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=100, num_layers=2):
        super(GitHubLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 100).to(x.device)
        c0 = torch.zeros(2, x.size(0), 100).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.linear(out[:, -1, :])

def run_evaluation():
    df_train_raw = pd.read_csv(TRAIN_FILE)
    df_test_raw = pd.read_csv(TEST_FILE)
    unique_flows = df_test_raw['flow_key_id'].unique()

    all_y_true, all_y_pred, all_delays, all_inf_latencies = [], [], [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for flow_id in unique_flows:
        train_df = df_train_raw[df_train_raw['flow_key_id'] == flow_id].sort_values('timestamp')
        test_df = df_test_raw[df_test_raw['flow_key_id'] == flow_id].sort_values('timestamp')

        train_series = train_df['traffic_volume_Tbits'].values
        test_series = test_df['traffic_volume_Tbits'].values

        train_diff = (train_series[SEASONAL_PERIOD:] - train_series[:-SEASONAL_PERIOD]).reshape(-1, 1)
        test_diff = (test_series[SEASONAL_PERIOD:] - test_series[:-SEASONAL_PERIOD]).reshape(-1, 1)

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_diff.astype(np.float32))
        test_scaled = scaler.transform(test_diff.astype(np.float32))

        train_ds = SeasonalDataset(train_scaled, SEQUENCE_LENGTH)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

        model = GitHubLSTM().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        model.train()
        for _ in range(EPOCHS):
            for x, y in train_loader:
                loss = criterion(model(x.to(device)), y.to(device))
                optimizer.zero_grad(); loss.backward(); optimizer.step()

        model.eval()
        train_res = []
        with torch.no_grad():
            for x, y in DataLoader(train_ds, batch_size=1):
                out = model(x.to(device))
                train_res.append(np.abs(out.item() - y.item()))

        thresh = np.mean(train_res) + (THRESHOLD_ALPHA * np.std(train_res))

        test_ds = SeasonalDataset(test_scaled, SEQUENCE_LENGTH)
        y_pred_flow, comp_lags = [], []
        with torch.no_grad():
            for x, y in DataLoader(test_ds, batch_size=1):
                t0 = time.time()
                out = model(x.to(device))
                t1 = time.time()
                
                latency = t1 - t0
                all_inf_latencies.append(latency)
                comp_lags.append(latency / 60.0)
                y_pred_flow.append(1 if np.abs(out.item() - y.item()) > thresh else 0)

        y_true_flow = (test_df['is_anomaly'].values[SEASONAL_PERIOD + SEQUENCE_LENGTH:] == 1).astype(int)
        all_y_true.extend(y_true_flow)
        all_y_pred.extend(y_pred_flow)

        curr_start, detected = None, False
        for i in range(len(y_true_flow)):
            if y_true_flow[i] == 1 and (i == 0 or y_true_flow[i-1] == 0):
                curr_start, detected = i, False
            if y_true_flow[i] == 1 and y_pred_flow[i] == 1 and not detected:
                all_delays.append(SAMPLING_LAG_MINS + ((i - curr_start) * FREQ_MIN) + comp_lags[i])
                detected = True
            if y_true_flow[i] == 0:
                curr_start, detected = None, False

    y_t, y_p = np.array(all_y_true), np.array(all_y_pred)
    print("\n" + "="*50)
    print("      SEASONAL DIFFERENCING LSTM REPORT")
    print("-" * 50)
    print(f"PRECISION: {precision_score(y_t, y_p, zero_division=0):.4f} | RECALL: {recall_score(y_t, y_p, zero_division=0):.4f}")
    print(f"F1-SCORE:  {f1_score(y_t, y_p, zero_division=0):.4f} | AVG DELAY: {np.mean(all_delays):.4f} mins")
    print(f"INFERENCE LATENCY: {np.mean(all_inf_latencies):.6f}")
    print("-" * 50)
    tn, fp, fn, tp = confusion_matrix(y_t, y_p).ravel()
    print(f"TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")
    print("="*50)

if __name__ == "__main__":
    run_evaluation()

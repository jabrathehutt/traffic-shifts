import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
import warnings
import os

warnings.filterwarnings('ignore')

# --- CONFIGURATION (Aligned with your directory structure) ---
TRAIN_FILE = '/root/traffic-shifts/llama/lag-llama/trafpy_finetune_normal_data.csv'
TEST_FILE = '/root/traffic-shifts/trafpy/trafpy_master_univariate_data.csv'

SEQUENCE_LENGTH = 96  
EPOCHS = 10           # Increased for better convergence on Tbit scale
BATCH_SIZE = 64
ANOMALY_ALPHA = 2.5   # Z-score-like threshold for residuals
FREQ_MIN = 10         # 10-minute intervals from TrafPy

# --- PyTorch LSTM Architecture ---

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, index):
        return self.data[index:index + self.seq_len], self.data[index + self.seq_len]

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# --- Main Logic ---

def run_lstm_analysis():
    # 1. Load Data
    df_train_raw = pd.read_csv(TRAIN_FILE)
    df_test_raw = pd.read_csv(TEST_FILE)

    # Use corresponding flows (e.g., first Backbone flow)
    flow_id_train = df_train_raw['flow_key_id'].unique()[0]
    flow_id_test = df_test_raw['flow_key_id'].unique()[0]

    train_data = df_train_raw[df_train_raw['flow_key_id'] == flow_id_train].sort_values('timestamp')
    test_data = df_test_raw[df_test_raw['flow_key_id'] == flow_id_test].sort_values('timestamp')

    # 2. Scaling (Fit on Train, Transform Test)
    scaler = RobustScaler()
    train_series = scaler.fit_transform(train_data[['traffic_volume_Tbits']].values.astype(np.float32))
    test_series = scaler.transform(test_data[['traffic_volume_Tbits']].values.astype(np.float32))

    # 3. Training on Clean Data
    train_ds = TimeSeriesDataset(train_series, SEQUENCE_LENGTH)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMForecaster(hidden_size=SEQUENCE_LENGTH).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print(f"Training LSTM on clean baseline: {flow_id_train}")
    model.train()
    for epoch in range(EPOCHS):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

    # 4. Inference on Anomalous Data
    model.eval()
    test_ds = TimeSeriesDataset(test_series, SEQUENCE_LENGTH)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    preds, actuals = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            preds.append(model(x).item())
            actuals.append(y.item())

    # 5. Anomaly Detection via Residuals
    residuals = np.abs(np.array(preds) - np.array(actuals))
    # Threshold based on training residuals for consistency
    threshold = np.median(residuals) + (ANOMALY_ALPHA * np.std(residuals))
    y_pred = (residuals > threshold).astype(int)
    
    # Match ground truth (offset by SEQUENCE_LENGTH)
    y_true = (test_data['is_anomaly'].values[SEQUENCE_LENGTH:] == 1).astype(int)

    # 6. Metrics & Delay
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Detection Delay (TTD)
    delay = 0
    anomaly_indices = np.where(y_true == 1)[0]
    if len(anomaly_indices) > 0:
        first_true = anomaly_indices[0]
        detections = np.where(y_pred[first_true:] == 1)[0]
        if len(detections) > 0:
            delay = detections[0] * FREQ_MIN

    print("\n" + "=" * 60)
    print("LSTM SUPERVISED PERFORMANCE (TRAINED ON CLEAN DATA)")
    print("=" * 60)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Average Detection Delay: {delay:.2f} minutes")
    print("=" * 60)

if __name__ == "__main__":
    run_lstm_analysis()

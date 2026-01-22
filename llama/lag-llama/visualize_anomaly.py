import torch, numpy as np, pandas as pd, os
import matplotlib.pyplot as plt
from lag_llama.gluon.estimator import LagLlamaLightningModule
from gluonts.torch.distributions import StudentTOutput
from scipy.stats import t
from tqdm import tqdm

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_CSV = "/root/traffic-shifts/llama/lag-llama/trafpy_finetune_normal_data.csv"
TEST_CSV = "/root/traffic-shifts/trafpy/trafpy_master_univariate_data.csv"
MODEL_PATH = "specialized_v11_supervised.pt"
CONTEXT_LEN = 256  

def visualize_with_bands():
    # 1. Load Stats
    df_train = pd.read_csv(TRAIN_CSV)
    GLOBAL_MEAN = np.mean(df_train['traffic_volume_Tbits'].values)
    GLOBAL_STD = np.std(df_train['traffic_volume_Tbits'].values)

    # 2. Load Model
    module = LagLlamaLightningModule(
        context_length=CONTEXT_LEN, prediction_length=1,
        model_kwargs={
            "context_length": CONTEXT_LEN, "max_context_length": 1024, "input_size": 1,
            "distr_output": StudentTOutput(), "n_layer": 1, "n_head": 8,
            "n_embd_per_head": 16, "lags_seq": list(range(1, 85)), "scaling": "mean", "time_feat": False,
        }
    )
    sd = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    module.model.load_state_dict({k.replace("model.", ""): v for k, v in sd.items()}, strict=True)
    model = module.model.to(DEVICE).eval()

    # 3. Data Prep
    df_test = pd.read_csv(TEST_CSV)
    test_flow = df_test['flow_key_id'].unique()[0]
    flow_df = df_test[df_test['flow_key_id'] == test_flow].sort_values('timestamp').reset_index(drop=True)
    anomaly_start = flow_df[flow_df['is_anomaly'] == 1].index[0]
    plot_slice = flow_df.iloc[anomaly_start-50 : anomaly_start+150].copy()
    
    v = flow_df['traffic_volume_Tbits'].values.astype('float32')
    times, actuals, preds, upper_95, lower_95 = [], [], [], [], []

    print("Extracting Probabilistic Bands...")
    for i in tqdm(plot_slice.index):
        window = torch.tensor(v[i-CONTEXT_LEN:i]).unsqueeze(0).to(DEVICE).float()
        with torch.no_grad():
            scale_f = torch.tensor([[GLOBAL_MEAN + 1e-5]]).to(DEVICE).float()
            # distr_args: [loc, scale, df]
            distr_args, _, _ = model(past_target=window/scale_f, past_observed_values=torch.ones_like(window).to(DEVICE).float())
            
            mu = (distr_args[0][:, -1] * scale_f.squeeze(-1)).cpu().item()
            sigma = (torch.exp(distr_args[1][:, -1]) * scale_f.squeeze(-1)).cpu().item()
            df_val = (torch.abs(distr_args[2][:, -1]) + 2.0).cpu().item()
            
            # Calculate 95% interval using Student-T distribution
            interval = t.interval(0.95, df_val, loc=mu, scale=sigma)
            
            times.append(flow_df.loc[i, 'timestamp'])
            actuals.append(v[i])
            preds.append(mu)
            lower_95.append(interval[0])
            upper_95.append(interval[1])

    # 4. Plotting
    plt.figure(figsize=(15, 7))
    plt.plot(times, actuals, label='Actual Traffic (Backbone)', color='#1f77b4', zorder=3)
    plt.plot(times, preds, label='Lag-Llama Mean Prediction', color='#ff7f0e', linestyle='--')
    
    # Fill the uncertainty area
    plt.fill_between(times, lower_95, upper_95, color='#ff7f0e', alpha=0.2, label='95% Confidence Interval')
    
    plt.title(f"Probabilistic Anomaly Detection: {test_flow}")
    plt.xlabel("Timestamp")
    plt.ylabel("Traffic Volume (Tbits)")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("probabilistic_forecast_plot.png")
    plt.show()

if __name__ == "__main__":
	visualize_with_bands()

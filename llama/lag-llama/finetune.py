import sys, types, torch, numpy as np, pandas as pd, random, os
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from lag_llama.gluon.estimator import LagLlamaLightningModule
from gluonts.torch.distributions import StudentTOutput

def seed_everything(seed=42):
    random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
seed_everything(42)

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOAD_PATH = "/root/traffic-shifts/llama/lag-llama/lag-llama-backbone.ckpt"
METRICS_CSV = "/root/traffic-shifts/llama/lag-llama/trafpy_finetune_normal_data.csv"
FINAL_PATH = "/root/traffic-shifts/llama/lag-llama/specialized_v11_supervised.pt"

CONTEXT_LEN = 512
LAGS_SEQ = list(range(1, 85))

class OfficialDataset(Dataset):
    def __init__(self, csv_path, context_len):
        df = pd.read_csv(csv_path)
        self.samples = []
        for flow_id in df['flow_key_id'].unique():
            flow_df = df[df['flow_key_id'] == flow_id].sort_values('timestamp')
            v = flow_df['traffic_volume_Tbits'].values.astype('float32')
            if len(v) > context_len:
                for i in range(context_len, len(v)):
                    self.samples.append({'x': v[i-context_len:i], 'y': v[i]})

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        return torch.tensor(self.samples[i]['x']), torch.tensor(self.samples[i]['y'])

def run_official_finetuning():
    full_dataset = OfficialDataset(METRICS_CSV, CONTEXT_LEN)
    loader = DataLoader(full_dataset, batch_size=64, shuffle=True)

    module = LagLlamaLightningModule(
        context_length=CONTEXT_LEN, prediction_length=1,
        model_kwargs={
            "context_length": CONTEXT_LEN, "max_context_length": 1024, "input_size": 1,
            "distr_output": StudentTOutput(), "n_layer": 1, "n_head": 8,
            "n_embd_per_head": 16, "lags_seq": LAGS_SEQ, "scaling": "mean", "time_feat": False,
        }
    )

    if os.path.exists(LOAD_PATH):
        sd = torch.load(LOAD_PATH, map_location=DEVICE)
        new_sd = { (k if k.startswith("model.") else f"model.{k}"): v for k, v in sd.items() }
        module.load_state_dict(new_sd, strict=True)

    model = module.model.to(DEVICE); model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

    for epoch in range(20):
        pbar = tqdm(loader, desc=f"MLE Finetune Epoch {epoch+1}")
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            
            scale_f = x.mean(dim=1, keepdim=True) + 1e-5
            
            # Forward pass
            distr_args, loc, scale_p = model(
                past_target=x / scale_f, 
                past_observed_values=torch.ones_like(x).to(DEVICE)
            )
            
            # --- FIX: Slicing for Batch broadcast ---
            # distr_args[0] is df, loc is loc, scale_p is scale
            # We take [:, -1] to get the parameters for the very last time step
            df = distr_args[0][:, -1]
            mu = loc[:, -1]
            sigma = scale_p[:, -1]
            
            # Create distribution object for the last step only
            distr = torch.distributions.StudentT(df, mu, sigma)
            
            # Calculate NLL Loss
            y_norm = y / scale_f.squeeze()
            loss = -distr.log_prob(y_norm).mean()

            loss.backward()
            optimizer.step()
            pbar.set_postfix(nll=f"{loss.item():.4f}")

    torch.save(model.state_dict(), FINAL_PATH)
    print(f"Weights saved to {FINAL_PATH}")

if __name__ == "__main__":
    run_official_finetuning()

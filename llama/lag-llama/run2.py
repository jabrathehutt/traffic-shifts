import sys
from types import ModuleType

# --- 3.13 DISTUTILS SHIM ---
def strtobool(val):
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'): return 1
    elif val in ('n', 'no', 'f', 'false', 'off', '0'): return 0
    else: raise ValueError(f"invalid truth value {val}")

mock_distutils_util = ModuleType('distutils.util')
mock_distutils_util.strtobool = strtobool
sys.modules['distutils.util'] = mock_distutils_util

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

import argparse
import os
import shutil
import functools
import torch
import wandb
import numpy as np
from gluonts.torch.distributions import StudentTOutput
from gluonts.transform import (
    AddObservedValuesIndicator,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter
)
from gluonts.dataset.loader import TrainDataLoader, ValidationDataLoader
from gluonts.itertools import Cyclic
from gluonts.torch.batchify import batchify
import lightning
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer

if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([functools.partial])

from data.data_utils import create_train_and_val_datasets_with_dates
from lag_llama.gluon.estimator import LagLlamaLightningModule

def train(args):
    lightning.seed_everything(args.seed)
    os.makedirs("models", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")

    # --- ARCHITECTURE SYNC ---
    is_finetune = False
    pretrained_sd = None
    final_lags_count = 95 

    if args.get_ckpt_path_from_experiment_name:
        ckpt_path = f"models/{args.get_ckpt_path_from_experiment_name}.ckpt"
        if os.path.exists(ckpt_path):
            print(f">>> FINETUNING MODE: Inspecting {ckpt_path} <<<")
            pretrained_sd = torch.load(ckpt_path, map_location=device, weights_only=False)['state_dict']
            target_width = pretrained_sd['model.transformer.wte.weight'].shape[1]

            found_match = False
            for trial_count in range(target_width - 5, target_width + 2):
                trial_lags = list(range(1, trial_count + 1))
                test_module = LagLlamaLightningModule(
                    context_length=args.context_length, max_context_length=2048, prediction_length=1,
                    model_kwargs={
                        "context_length": args.context_length, "max_context_length": 2048,
                        "n_layer": args.n_layer, "n_head": args.n_head, "n_embd_per_head": 16,
                        "scaling": args.data_normalization, "time_feat": False, "input_size": 1,
                        "distr_output": StudentTOutput(), "lags_seq": trial_lags
                    }
                )
                if test_module.model.transformer.wte.weight.shape[1] == target_width:
                    final_lags_count = trial_count
                    found_match = True
                    break
            if not found_match: raise RuntimeError("Architecture mismatch.")
            is_finetune = True

    lags_seq = list(range(1, final_lags_count + 1))
    distr_output = StudentTOutput()

    module = LagLlamaLightningModule(
        context_length=args.context_length,
        max_context_length=2048,
        prediction_length=1,
        model_kwargs={
            "context_length": args.context_length, "max_context_length": 2048,
            "n_layer": args.n_layer, "n_head": args.n_head, "n_embd_per_head": 16,
            "scaling": args.data_normalization, "time_feat": False, "input_size": 1,
            "distr_output": distr_output, "lags_seq": lags_seq
        }
    )

    if is_finetune:
        module.load_state_dict(pretrained_sd, strict=True)
        print("SUCCESS: Backbone weights loaded.")

    # 3. Data Pipeline
    data_root = args.dataset_path.rstrip('/')
    dataset_outputs = create_train_and_val_datasets_with_dates(
        args.single_dataset, data_root, 0, args.context_length + max(lags_seq), 1,
        num_val_windows=args.num_validation_windows
    )
    train_data, val_data = dataset_outputs[0], dataset_outputs[1]

    # --- MASKING LOGIC ---
    # AddObservedValuesIndicator is the official way to tell Lag-Llama 
    # what data is 'real' and what is 'missing/masked'.
    transformation = Chain([
        AddObservedValuesIndicator(target_field="target", output_field="observed_values")
    ])

    instance_splitter = InstanceSplitter(
        target_field="target",
        is_pad_field="is_pad",
        start_field="start",
        forecast_start_field="forecast_start",
        instance_sampler=ExpectedNumInstanceSampler(num_instances=1.0, min_future=1),
        past_length=args.context_length + max(lags_seq),
        future_length=1,
        time_series_fields=["observed_values"], 
        dummy_value=distr_output.value_in_support,
    )

    train_dataloader = TrainDataLoader(
        Cyclic(train_data), 
        transform=transformation + instance_splitter,
        batch_size=args.batch_size, 
        stack_fn=batchify, 
        num_batches_per_epoch=100,
    )

    val_dataloader = ValidationDataLoader(
        val_data, 
        transform=transformation + instance_splitter, 
        batch_size=args.batch_size, 
        stack_fn=batchify
    )

    # 4. Trainer
    logger = WandbLogger(name=args.experiment_name, project=args.wandb_project, mode=args.wandb_mode)
    checkpoint_dir = os.path.join(args.results_dir, args.experiment_name, str(args.seed), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto", 
        devices="auto",
        logger=logger,
        callbacks=[
            EarlyStopping(monitor="train_loss", patience=5, mode="min"),
            ModelCheckpoint(dirpath=checkpoint_dir, filename="best", monitor="train_loss", save_top_k=1, mode="min")
        ]
    )

    trainer.fit(model=module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # 5. Export
    target_path = "models/finetune_latest.ckpt" if is_finetune else "models/backbone_latest.ckpt"
    best_ckpt_path = trainer.checkpoint_callback.best_model_path
    if best_ckpt_path and os.path.exists(best_ckpt_path):
        shutil.copy2(best_ckpt_path, target_path)
    else:
        torch.save({'state_dict': module.state_dict()}, target_path)

    print(f"Final WTE Mean: {module.model.transformer.wte.weight.mean().item():.10f}")
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment_name", type=str, required=True)
    parser.add_argument("-d", "--dataset_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context_length", type=int, default=32)
    parser.add_argument("--n_layer", type=int, default=1)
    parser.add_argument("--n_embd_per_head", type=int, default=16)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--data_normalization", default="mean")
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-m", "--max_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("-r", "--results_dir", type=str, default="experiments/results")
    parser.add_argument("--wandb_project", type=str, default="lag-llama-test")
    parser.add_argument("--wandb_mode", type=str, default="offline")
    parser.add_argument("--get_ckpt_path_from_experiment_name", type=str)
    parser.add_argument("--single_dataset", type=str)
    parser.add_argument("--num_validation_windows", type=int, default=1)
    parser.add_argument("--time_feat", action="store_true")
    parser.add_argument("--use_dataset_prediction_length", action="store_true")
    parser.add_argument("--lags_seq", type=str, nargs="+")
    args = parser.parse_args()
    train(args)

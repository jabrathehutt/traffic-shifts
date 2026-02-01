import copy
import random
import warnings
import json
import os
from pathlib import Path

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

import numpy as np
import pandas as pd
from tqdm import tqdm
from gluonts.dataset.common import ListDataset
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.transform import InstanceSampler
from pandas.tseries.frequencies import to_offset

from data.read_new_dataset import (
    get_ett_dataset,
    create_train_dataset_without_last_k_timesteps,
    TrainDatasets,
    MetaData,
)

class CombinedDatasetIterator:
    def __init__(self, datasets, seed, weights):
        self._datasets = [iter(el) for el in datasets]
        self._weights = weights
        self._rng = random.Random(seed)

    def __next__(self):
        (dataset,) = self._rng.choices(self._datasets, weights=self._weights, k=1)
        return next(dataset)

class CombinedDataset:
    def __init__(self, datasets, seed=None, weights=None):
        self._seed = seed
        self._datasets = datasets
        self._weights = weights
        n_datasets = len(datasets)
        if weights is None:
            self._weights = [1 / n_datasets] * n_datasets

    def __iter__(self):
        return CombinedDatasetIterator(self._datasets, self._seed, self._weights)

    def __len__(self):
        return sum([len(ds) for ds in self._datasets])

class SingleInstanceSampler(InstanceSampler):
    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts)
        window_size = b - a + 1
        if window_size <= 0:
            return np.array([], dtype=int)
        indices = np.random.randint(window_size, size=1)
        return indices + a

def _count_timesteps(left: pd.Timestamp, right: pd.Timestamp, delta: pd.DateOffset) -> int:
    if type(left) == pd.Period: left = left.to_timestamp()
    if type(right) == pd.Period: right = right.to_timestamp()
    assert (right >= left), f"Case where left ({left}) is after right ({right}) is not implemented."
    try:
        return (right - left) // delta
    except TypeError:
        for i in range(100000):
            if left + (i + 1) * delta > right: return i
        raise RuntimeError(f"Too large difference between {left} and {right}")

def create_train_and_val_datasets_with_dates(
    name, dataset_path, data_id, history_length, prediction_length=None,
    num_val_windows=None, val_start_date=None, train_start_date=None,
    freq=None, last_k_percentage=None,
):
    # --- CUSTOM HANDLER FOR LOCAL TRAFPY CSVs ---
    if "trafpy" in name:
        csv_path = os.path.join(dataset_path, f"{name}.csv")
        print(f"Loading local TrafPy CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # GluonTS Metadata shim for local data
        class LocalMetadata:
            def __init__(self):
                self.prediction_length = 1
                self.freq = "10min"
        
        raw_train_ds = PandasDataset.from_long_dataframe(
            df, target="traffic_volume_Tbits", timestamp="timestamp", 
            item_id="flow_key_id", freq="10min"
        )
        # For training, we treat test same as train to satisfy the logic
        raw_dataset = TrainDatasets(metadata=LocalMetadata(), train=raw_train_ds, test=raw_train_ds)
    elif name in ("ett_h1", "ett_h2", "ett_m1", "ett_m2"):
        raw_dataset = get_ett_dataset(name, os.path.join(dataset_path, "ett_datasets"))
    else:
        raw_dataset = get_dataset(name, path=Path(dataset_path))

    if prediction_length is None: prediction_length = raw_dataset.metadata.prediction_length
    if freq is None: freq = raw_dataset.metadata.freq
    
    timestep_delta = to_offset(freq)
    raw_train_dataset = raw_dataset.train

    # Training Data Slicing
    train_data = []
    total_train_points = 0
    max_train_end_date = None
    for i, series in enumerate(raw_train_dataset):
        s_train = series.copy()
        train_end_index = len(series["target"]) - num_val_windows if num_val_windows else len(series["target"])
        
        if last_k_percentage:
            num_vals = int(len(s_train["target"]) * last_k_percentage / 100)
            start_idx = max(0, train_end_index - num_vals)
        else:
            start_idx = 0
            
        s_train["target"] = series["target"][start_idx:train_end_index]
        s_train["item_id"] = i
        s_train["data_id"] = data_id
        train_data.append(s_train)
        total_train_points += len(s_train["target"])
        
    train_data = ListDataset(train_data, freq=freq)

    # Validation Data Slicing
    val_data = []
    total_val_points = 0
    total_val_windows = 0
    for i, series in enumerate(raw_train_dataset):
        s_val = series.copy()
        train_end_idx = len(series["target"]) - num_val_windows if num_val_windows else len(series["target"])
        val_start_idx = max(0, train_end_idx - prediction_length - history_length)
        
        s_val["start"] = series["start"] + val_start_idx * timestep_delta
        s_val["target"] = series["target"][val_start_idx:]
        s_val["item_id"] = i
        s_val["data_id"] = data_id
        val_data.append(s_val)
        total_val_points += len(s_val["target"])
        total_val_windows += max(0, len(s_val["target"]) - prediction_length - history_length)
        
    val_data = ListDataset(val_data, freq=freq)
    total_points = total_train_points + total_val_points

    return train_data, val_data, total_train_points, total_val_points, total_val_windows, max_train_end_date, total_points

def create_test_dataset(name, dataset_path, history_length, freq=None, data_id=None):
    if "trafpy" in name:
        csv_path = os.path.join(dataset_path, f"{name}.csv")
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        raw_ds = PandasDataset.from_long_dataframe(
            df, target="traffic_volume_Tbits", timestamp="timestamp", 
            item_id="flow_key_id", freq="10min"
        )
        class LocalMetadata:
            def __init__(self):
                self.prediction_length = 1
                self.freq = "10min"
        dataset = TrainDatasets(metadata=LocalMetadata(), train=raw_ds, test=raw_ds)
    else:
        dataset = get_dataset(name, path=Path(dataset_path))

    if freq is None: freq = dataset.metadata.freq
    prediction_length = dataset.metadata.prediction_length
    data = []
    total_points = 0
    for i, series in enumerate(dataset.test):
        offset = len(series["target"]) - (history_length + prediction_length)
        if offset > 0:
            target = series["target"][-(history_length + prediction_length) :]
            data.append({"target": target, "start": series["start"] + offset, "item_id": i, "data_id": data_id})
        else:
            s_copy = copy.deepcopy(series)
            s_copy["item_id"], s_copy["data_id"] = i, data_id
            data.append(s_copy)
        total_points += len(data[-1]["target"])
    return ListDataset(data, freq=freq), prediction_length, total_points

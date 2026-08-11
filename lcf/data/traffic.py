import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, Tuple, Optional
import os


class TrafficDataset(Dataset):

    def __init__(
        self,
        x: np.ndarray,
        c: np.ndarray,
        stats: Optional[Dict] = None,
    ):
        self.x = torch.FloatTensor(x)
        self.c = torch.FloatTensor(c)
        self.stats = stats

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {
            'x': self.x[idx],
            'c': self.c[idx],
        }


def load_and_preprocess_traffic_data(
    data_path: str,
    seq_len: int = 96,
    interval: int = 1,
) -> Tuple[Dict, Dict]:
    print(f"Loading traffic data from: {data_path}")


    df = pd.read_csv(data_path)
    df['date_time'] = pd.to_datetime(df['date_time'])
    df = df.sort_values('date_time').reset_index(drop=True)

    print(f"Original shape: {df.shape}")
    print(f"Time range: {df['date_time'].min()} to {df['date_time'].max()}")


    df['temp'] = df['temp'] - 273.15
    df['temp'] = df['temp'].clip(-40, 50)


    df['traffic_volume'] = np.log1p(df['traffic_volume'])


    df['rain_1h'] = df['rain_1h'].clip(0, 100)
    df['snow_1h'] = df['snow_1h'].clip(0, 10)


    weather_categories = ['Clear', 'Clouds', 'Rain', 'Drizzle', 'Mist',
                          'Haze', 'Fog', 'Thunderstorm', 'Snow', 'Squall', 'Smoke']
    weather_to_idx = {cat: idx for idx, cat in enumerate(weather_categories)}
    weather_to_idx['unknown'] = len(weather_categories)
    df['weather_encoded'] = df['weather_main'].map(weather_to_idx).fillna(len(weather_categories))


    df['holiday_encoded'] = (df['holiday'] != 'None').astype(int)


    df['hour'] = df['date_time'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    print("Added hour features: hour_sin, hour_cos")


    print(f"Creating sequences with length {seq_len}...")

    sequences = []
    for start_idx in range(0, len(df) - seq_len + 1, interval):
        end_idx = start_idx + seq_len
        seq_df = df.iloc[start_idx:end_idx]


        time_diff = (seq_df['date_time'].diff().dropna().dt.total_seconds() / 3600).max()
        if time_diff > 1.5:
            continue


        avg_temp = seq_df['temp'].mean()


        x_seq = seq_df['traffic_volume'].values


        c_seq = np.column_stack([
            seq_df['rain_1h'].values,
            seq_df['snow_1h'].values,
            seq_df['clouds_all'].values,
            seq_df['weather_encoded'].values,
            seq_df['holiday_encoded'].values,
            seq_df['hour_sin'].values,
            seq_df['hour_cos'].values,
        ])

        sequences.append({
            'x': x_seq,
            'c': c_seq,
            'avg_temp': avg_temp,
        })

    print(f"Created {len(sequences)} valid sequences")


    train_seqs = [s for s in sequences if s['avg_temp'] < 12]
    val_seqs = [s for s in sequences if 12 <= s['avg_temp'] < 22]
    test_seqs = [s for s in sequences if s['avg_temp'] >= 22]

    print(f"Temperature-based split:")
    print(f"  Train (<12°C): {len(train_seqs)} sequences")
    print(f"  Val (12-22°C): {len(val_seqs)} sequences")
    print(f"  Test (>22°C): {len(test_seqs)} sequences")


    def seqs_to_arrays(seqs):
        if len(seqs) == 0:
            return np.array([]), np.array([])
        x = np.array([s['x'] for s in seqs])[:, :, np.newaxis]
        c = np.array([s['c'] for s in seqs])
        return x, c

    x_train, c_train = seqs_to_arrays(train_seqs)
    x_val, c_val = seqs_to_arrays(val_seqs)
    x_test, c_test = seqs_to_arrays(test_seqs)


    x_scaler = StandardScaler()
    x_train_flat = x_train.reshape(-1, 1)
    x_scaler.fit(x_train_flat)

    x_train_norm = x_scaler.transform(x_train.reshape(-1, 1)).reshape(x_train.shape)
    x_val_norm = x_scaler.transform(x_val.reshape(-1, 1)).reshape(x_val.shape)
    x_test_norm = x_scaler.transform(x_test.reshape(-1, 1)).reshape(x_test.shape)


    c_scalers = []
    c_train_norm = c_train.copy()
    c_val_norm = c_val.copy()
    c_test_norm = c_test.copy()

    for i in range(3):
        scaler = StandardScaler()
        c_train_flat = c_train[:, :, i].reshape(-1, 1)
        scaler.fit(c_train_flat)

        c_train_norm[:, :, i] = scaler.transform(c_train[:, :, i].reshape(-1, 1)).reshape(c_train[:, :, i].shape)
        c_val_norm[:, :, i] = scaler.transform(c_val[:, :, i].reshape(-1, 1)).reshape(c_val[:, :, i].shape)
        c_test_norm[:, :, i] = scaler.transform(c_test[:, :, i].reshape(-1, 1)).reshape(c_test[:, :, i].shape)

        c_scalers.append(scaler)

    stats = {
        'x_scaler': x_scaler,
        'c_scalers': c_scalers,
        'weather_categories': weather_categories,
        'x_dim': 1,
        'c_dim': c_train_norm.shape[-1],
    }

    data_dict = {
        'x_train': x_train_norm.astype(np.float32),
        'x_val': x_val_norm.astype(np.float32),
        'x_test': x_test_norm.astype(np.float32),
        'c_train': c_train_norm.astype(np.float32),
        'c_val': c_val_norm.astype(np.float32),
        'c_test': c_test_norm.astype(np.float32),
    }

    return data_dict, stats


def get_traffic_dataloaders(
    data_path: str = "/root/autodl-tmp/lcf/data_raw/Metro_Interstate_Traffic_Volume.csv",
    seq_len: int = 96,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    data_dict, stats = load_and_preprocess_traffic_data(data_path, seq_len)

    train_dataset = TrafficDataset(
        data_dict['x_train'],
        data_dict['c_train'],
        stats=stats
    )
    val_dataset = TrafficDataset(
        data_dict['x_val'],
        data_dict['c_val'],
        stats=stats
    )
    test_dataset = TrafficDataset(
        data_dict['x_test'],
        data_dict['c_test'],
        stats=stats
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"\n✓ DataLoaders created:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")

    return train_loader, val_loader, test_loader, stats


if __name__ == "__main__":

    train_loader, val_loader, test_loader, stats = get_traffic_dataloaders()

    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  x: {batch['x'].shape}")
    print(f"  c: {batch['c'].shape}")

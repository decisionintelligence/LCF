import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Optional, List
import os
import glob


STATION_SPLIT = {
    'train': [
        'Dongsi',
        'Guanyuan',
        'Tiantan',
        'Wanshouxigong',
        'Aotizhongxin',
        'Nongzhanguan',
        'Wanliu',
        'Gucheng',
    ],
    'val': [
        'Changping',
        'Dingling',
    ],
    'test': [
        'Shunyi',
        'Huairou',
    ]
}


WIND_DIRECTIONS = [
    'N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
    'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'
]


class AirQualityDataset(Dataset):

    def __init__(
        self,
        x: np.ndarray,
        c: np.ndarray,
        stations: Optional[np.ndarray] = None,
        stats: Optional[Dict] = None,
    ):
        self.x = torch.FloatTensor(x)
        self.c = torch.FloatTensor(c)
        self.stations = stations
        self.stats = stats

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {
            'x': self.x[idx],
            'c': self.c[idx],
        }


def load_and_preprocess_aq_data(
    data_dir: str,
    seq_len: int = 96,
    interval: int = 24,
) -> Tuple[Dict, Dict]:
    print(f"Loading Air Quality data from: {data_dir}")


    pattern = os.path.join(data_dir, "PRSA_Data_*_20130301-20170228.csv")
    station_files = glob.glob(pattern)

    if not station_files:
        raise FileNotFoundError(f"No AQ data files found in {data_dir}")

    print(f"Found {len(station_files)} station files")


    all_data = []
    for file_path in sorted(station_files):
        station_name = os.path.basename(file_path).split('_')[2]
        print(f"  Loading: {station_name}")

        df = pd.read_csv(file_path)
        df['station'] = station_name
        all_data.append(df)

    df = pd.concat(all_data, ignore_index=True)
    print(f"Total rows: {len(df)}")


    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])


    continuous_cols = ['PM2.5', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
    for col in continuous_cols:
        df[col] = df.groupby('station')[col].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both')
        )


    df['wd'] = df.groupby('station')['wd'].transform(
        lambda x: x.fillna(x.mode().iloc[0] if len(x.mode()) > 0 else 'N')
    )


    wd_to_idx = {wd: idx for idx, wd in enumerate(WIND_DIRECTIONS)}
    wd_to_idx['unknown'] = len(WIND_DIRECTIONS)
    df['wd_encoded'] = df['wd'].map(wd_to_idx).fillna(len(WIND_DIRECTIONS))


    df['TEMP'] = df['TEMP'].clip(-40, 50)
    df['RAIN'] = df['RAIN'].clip(0, 100)
    df['WSPM'] = df['WSPM'].clip(0, 50)
    df['PM2.5'] = df['PM2.5'].clip(0, 1000)


    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    print("Added hour features: hour_sin, hour_cos")


    before_len = len(df)
    df = df.dropna(subset=['PM2.5', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM', 'wd'])
    print(f"Dropped {before_len - len(df)} rows with missing values")


    print(f"\nCreating sequences with length {seq_len}, interval {interval}...")

    all_sequences = {'train': [], 'val': [], 'test': []}

    for station_name in df['station'].unique():
        station_df = df[df['station'] == station_name].sort_values('datetime').reset_index(drop=True)


        split = None
        for split_name, stations in STATION_SPLIT.items():
            if station_name in stations:
                split = split_name
                break

        if split is None:
            print(f"  Warning: Station {station_name} not in split config, skipping")
            continue


        n_seqs = 0
        for start_idx in range(0, len(station_df) - seq_len + 1, interval):
            end_idx = start_idx + seq_len
            seq_df = station_df.iloc[start_idx:end_idx]


            time_diff = seq_df['datetime'].diff().dropna().dt.total_seconds() / 3600
            if time_diff.max() > 1.5:
                continue


            x_seq = seq_df['PM2.5'].values


            c_seq = np.column_stack([
                seq_df['TEMP'].values,
                seq_df['PRES'].values,
                seq_df['DEWP'].values,
                seq_df['WSPM'].values,
                seq_df['RAIN'].values,
                seq_df['wd_encoded'].values,
                seq_df['hour_sin'].values,
                seq_df['hour_cos'].values,
            ])

            all_sequences[split].append({
                'x': x_seq,
                'c': c_seq,
                'station': station_name,
            })
            n_seqs += 1

        print(f"  {station_name} ({split}): {n_seqs} sequences")

    print(f"\nSequence counts:")
    print(f"  Train: {len(all_sequences['train'])} sequences")
    print(f"  Val: {len(all_sequences['val'])} sequences")
    print(f"  Test: {len(all_sequences['test'])} sequences")


    def seqs_to_arrays(seqs):
        if len(seqs) == 0:
            return np.array([]), np.array([]), np.array([])
        x = np.array([s['x'] for s in seqs])[:, :, np.newaxis]
        c = np.array([s['c'] for s in seqs])
        stations = np.array([s['station'] for s in seqs])
        return x, c, stations

    x_train, c_train, stations_train = seqs_to_arrays(all_sequences['train'])
    x_val, c_val, stations_val = seqs_to_arrays(all_sequences['val'])
    x_test, c_test, stations_test = seqs_to_arrays(all_sequences['test'])


    x_train_log = np.log1p(x_train)
    x_val_log = np.log1p(x_val)
    x_test_log = np.log1p(x_test)

    x_scaler = StandardScaler()
    x_train_flat = x_train_log.reshape(-1, 1)
    x_scaler.fit(x_train_flat)

    x_train_norm = x_scaler.transform(x_train_log.reshape(-1, 1)).reshape(x_train_log.shape)
    x_val_norm = x_scaler.transform(x_val_log.reshape(-1, 1)).reshape(x_val_log.shape)
    x_test_norm = x_scaler.transform(x_test_log.reshape(-1, 1)).reshape(x_test_log.shape)


    c_scalers = []
    c_train_norm = c_train.copy()
    c_val_norm = c_val.copy()
    c_test_norm = c_test.copy()

    for i in range(5):
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
        'wind_directions': WIND_DIRECTIONS,
        'station_split': STATION_SPLIT,
    }

    data_dict = {
        'x_train': x_train_norm.astype(np.float32),
        'x_val': x_val_norm.astype(np.float32),
        'x_test': x_test_norm.astype(np.float32),
        'c_train': c_train_norm.astype(np.float32),
        'c_val': c_val_norm.astype(np.float32),
        'c_test': c_test_norm.astype(np.float32),
        'stations_train': stations_train,
        'stations_val': stations_val,
        'stations_test': stations_test,
    }

    return data_dict, stats


def get_aq_dataloaders(
    data_dir: str = "/data/avatar/lcf/data_raw/AQ",
    seq_len: int = 96,
    interval: int = 24,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    data_dict, stats = load_and_preprocess_aq_data(data_dir, seq_len, interval)

    train_dataset = AirQualityDataset(
        data_dict['x_train'],
        data_dict['c_train'],
        data_dict['stations_train'],
        stats=stats
    )
    val_dataset = AirQualityDataset(
        data_dict['x_val'],
        data_dict['c_val'],
        data_dict['stations_val'],
        stats=stats
    )
    test_dataset = AirQualityDataset(
        data_dict['x_test'],
        data_dict['c_test'],
        data_dict['stations_test'],
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
    print(f"  Train: {len(train_dataset)} samples ({len(train_loader)} batches)")
    print(f"  Val: {len(val_dataset)} samples ({len(val_loader)} batches)")
    print(f"  Test: {len(test_dataset)} samples ({len(test_loader)} batches)")
    print(f"  Feature dims: x={train_dataset.x.shape[-1]}, c={train_dataset.c.shape[-1]}")

    return train_loader, val_loader, test_loader, stats


if __name__ == "__main__":

    print("=" * 60)
    print("Testing Air Quality Data Loader")
    print("=" * 60)

    train_loader, val_loader, test_loader, stats = get_aq_dataloaders()

    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  x: {batch['x'].shape}")
    print(f"  c: {batch['c'].shape}")
    print(f"\nCondition variables: TEMP, PRES, DEWP, WSPM, RAIN, wd_encoded")

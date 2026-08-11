import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, Tuple, Any
from pathlib import Path


class LCFDataset(Dataset):

    def __init__(
        self,
        x_path: str,
        c_path: str,
        window: int = 96,
        batch_size: Optional[int] = None,
        normalize: bool = True,
        norm_type: str = 'zscore',
        dataset_name: Optional[str] = None,
        split_method: Optional[str] = None,
        augment: bool = False,
        augment_prob: float = 0.5,
    ):
        super().__init__()


        self.x = np.load(x_path).astype(np.float32)
        self.c = np.load(c_path).astype(np.float32)


        assert len(self.x) == len(self.c), f"x and c must have same length: {len(self.x)} vs {len(self.c)}"

        self.window = window
        self.normalize = normalize
        self.norm_type = norm_type
        self.dataset_name = dataset_name
        self.split_method = split_method
        self.augment = augment
        self.augment_prob = augment_prob


        self.x_mean = None
        self.x_std = None
        self.c_mean = None
        self.c_std = None

        if normalize and norm_type == 'zscore':
            self._compute_stats()
            self._normalize_data()

        self.length = len(self.x)

        print(f"Loaded dataset: x={self.x.shape}, c={self.c.shape}")

    def _compute_stats(self):

        self.x_mean = self.x.mean()
        self.x_std = self.x.std() + 1e-8


        self.c_mean = self.c.mean(axis=(0, 1), keepdims=True)
        self.c_std = self.c.std(axis=(0, 1), keepdims=True) + 1e-8

    def _normalize_data(self):
        self.x = (self.x - self.x_mean) / self.x_std
        self.c = (self.c - self.c_mean) / self.c_std

    def denormalize_x(self, x: np.ndarray) -> np.ndarray:
        if self.x_mean is not None:
            return x * self.x_std + self.x_mean
        return x

    def denormalize_c(self, c: np.ndarray) -> np.ndarray:
        if self.c_mean is not None:
            return c * self.c_std + self.c_mean
        return c

    def _augment(self, x: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.rand() > self.augment_prob:
            return x, c


        if np.random.rand() < 0.5:
            noise_scale = 0.01 * np.random.rand()
            x = x + noise_scale * np.random.randn(*x.shape).astype(np.float32)


        if np.random.rand() < 0.3:
            scale = 0.9 + 0.2 * np.random.rand()
            x = x * scale

        return x, c

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = self.x[idx]
        c = self.c[idx]


        if self.augment:
            x, c = self._augment(x, c)

        return {
            'x': torch.from_numpy(x).float(),
            'c': torch.from_numpy(c).float(),
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            'x_mean': self.x_mean,
            'x_std': self.x_std,
            'c_mean': self.c_mean,
            'c_std': self.c_std,
            'x_shape': self.x.shape,
            'c_shape': self.c.shape,
        }


class LCFCounterfactualDataset(Dataset):

    def __init__(
        self,
        x_path: str,
        c_path: str,
        x_cf_path: str,
        c_cf_path: str,
        env_path: Optional[str] = None,
        normalize: bool = True,
    ):
        super().__init__()

        self.x = np.load(x_path).astype(np.float32)
        self.c = np.load(c_path).astype(np.float32)
        self.x_cf = np.load(x_cf_path).astype(np.float32)
        self.c_cf = np.load(c_cf_path).astype(np.float32)

        self.env = None
        if env_path and Path(env_path).exists():
            self.env = np.load(env_path).astype(np.float32)

        if normalize:

            self.x_mean = self.x.mean()
            self.x_std = self.x.std() + 1e-8


            self.x = (self.x - self.x_mean) / self.x_std
            self.x_cf = (self.x_cf - self.x_mean) / self.x_std

        self.length = len(self.x)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        result = {
            'x': torch.from_numpy(self.x[idx]).float(),
            'c': torch.from_numpy(self.c[idx]).float(),
            'x_cf': torch.from_numpy(self.x_cf[idx]).float(),
            'c_cf': torch.from_numpy(self.c_cf[idx]).float(),
        }

        if self.env is not None:
            result['env'] = torch.from_numpy(self.env[idx]).float()

        return result


def create_synthetic_dataset(
    num_samples: int = 1000,
    seq_len: int = 96,
    num_envs: int = 4,
    noise_level: float = 0.1,
    save_dir: Optional[str] = None,
) -> Dict[str, np.ndarray]:

    env_idx = np.random.randint(0, num_envs, size=num_samples)
    env_onehot = np.eye(num_envs)[env_idx]


    c_base = np.random.randn(num_samples, seq_len, 1)

    t = np.linspace(0, 4 * np.pi, seq_len)
    c_periodic = np.sin(t).reshape(1, -1, 1)
    c = c_base + 0.5 * c_periodic


    x = np.zeros((num_samples, seq_len, 1), dtype=np.float32)

    for k in range(num_envs):
        mask = (env_idx == k)
        n_k = mask.sum()

        if n_k == 0:
            continue


        freq = 1.0 + 0.5 * k
        amplitude = 1.0 + 0.3 * k
        phase = k * np.pi / num_envs


        base_signal = amplitude * np.sin(freq * t + phase).reshape(1, -1, 1)
        treatment_effect = 0.5 * c[mask]
        noise = noise_level * np.random.randn(n_k, seq_len, 1)

        x[mask] = base_signal + treatment_effect + noise


    c_extra = np.random.randn(num_samples, seq_len, 3)
    c = np.concatenate([c, c_extra], axis=-1)

    result = {
        'x': x.astype(np.float32),
        'c': c.astype(np.float32),
        'env': env_onehot.astype(np.float32),
    }

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        for key, data in result.items():
            np.save(save_path / f'{key}.npy', data)

        print(f"Saved synthetic dataset to {save_path}")

    return result

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.integrate import solve_ivp
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class HarmonicVPConfig:

    m0: float = 0.5
    gamma0: float = 0.1
    k0: float = 1.0


    T: float = 30.0
    seq_len: int = 96


    x0_range: Tuple[float, float] = (-2.0, 2.0)
    v0_range: Tuple[float, float] = (-1.5, 1.5)


    train_alpha: Tuple[float, float] = (0.0, 0.2)
    train_beta: Tuple[float, float] = (0.0, 0.01)
    train_eta: Tuple[float, float] = (0.02, 0.08)


    val_alpha: Tuple[float, float] = (0.3, 0.5)
    val_beta: Tuple[float, float] = (0.018, 0.022)
    val_eta: Tuple[float, float] = (0.18, 0.22)


    test_alpha: Tuple[float, float] = (0.6, 1.0)
    test_beta: Tuple[float, float] = (0.035, 0.04)
    test_eta: Tuple[float, float] = (0.42, 0.5)


class HarmonicVPDataset(Dataset):

    def __init__(
        self,
        n_samples: int = 1000,
        split: str = 'train',
        config: Optional[HarmonicVPConfig] = None,
        seed: int = 42,
        normalize: bool = True,
        stats: Optional[Dict] = None,
    ):
        self.n_samples = n_samples
        self.split = split
        self.config = config or HarmonicVPConfig()
        self.seed = seed
        self.normalize = normalize


        np.random.seed(seed)


        self.alpha_range = getattr(self.config, f'{split}_alpha')
        self.beta_range = getattr(self.config, f'{split}_beta')
        self.eta_range = getattr(self.config, f'{split}_eta')


        self.x, self.c, self.e = self._generate_data()


        if normalize:
            if stats is not None:
                self.stats = stats
            else:
                self.stats = {
                    'x_mean': self.x.mean(),
                    'x_std': self.x.std() + 1e-8,
                    'c_mean': self.c.mean(axis=(0, 1)),
                    'c_std': self.c.std(axis=(0, 1)) + 1e-8,
                }

            self.x = (self.x - self.stats['x_mean']) / self.stats['x_std']
            self.c = (self.c - self.stats['c_mean']) / self.stats['c_std']
        else:
            self.stats = None

        print(f"[{split}] Generated {n_samples} samples")
        print(f"  α ∈ [{self.alpha_range[0]:.2f}, {self.alpha_range[1]:.2f}]")
        print(f"  β ∈ [{self.beta_range[0]:.3f}, {self.beta_range[1]:.3f}]")
        print(f"  η ∈ [{self.eta_range[0]:.2f}, {self.eta_range[1]:.2f}]")

    def _generate_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.config
        t_eval = np.linspace(0, cfg.T, cfg.seq_len)

        x_list = []
        c_list = []
        e_list = []

        for _ in range(self.n_samples):

            alpha = np.random.uniform(*self.alpha_range)
            beta = np.random.uniform(*self.beta_range)
            eta = np.random.uniform(*self.eta_range)


            x0 = np.random.uniform(*cfg.x0_range)
            v0 = np.random.uniform(*cfg.v0_range)


            def dynamics(t, y):
                pos, vel = y
                m = cfg.m0 + alpha * t
                gamma_t = cfg.gamma0 * (1 + beta * t)
                k_t = cfg.k0 * (1 + eta * t)
                acc = (-gamma_t * vel - k_t * pos) / m
                return [vel, acc]


            sol = solve_ivp(
                dynamics,
                [0, cfg.T],
                [x0, v0],
                t_eval=t_eval,
                method='RK45'
            )

            position = sol.y[0]
            velocity = sol.y[1]


            acceleration = np.zeros(cfg.seq_len)
            for i, t in enumerate(t_eval):
                m = cfg.m0 + alpha * t
                gamma_t = cfg.gamma0 * (1 + beta * t)
                k_t = cfg.k0 * (1 + eta * t)
                acceleration[i] = (-gamma_t * velocity[i] - k_t * position[i]) / m

            x_list.append(acceleration[:, np.newaxis])
            c_list.append(np.stack([velocity, position], axis=-1))
            e_list.append([alpha, beta, eta])

        return (
            np.array(x_list, dtype=np.float32),
            np.array(c_list, dtype=np.float32),
            np.array(e_list, dtype=np.float32)
        )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            'x': torch.from_numpy(self.x[idx]),
            'c': torch.from_numpy(self.c[idx]),
            'e': torch.from_numpy(self.e[idx]),
            'alpha': torch.tensor(self.e[idx, 0]),
            'beta': torch.tensor(self.e[idx, 1]),
            'eta': torch.tensor(self.e[idx, 2]),
        }


class HarmonicVPDatasetMixed(HarmonicVPDataset):

    def __init__(
        self,
        n_samples: int = 1000,
        split: str = 'train',
        config: Optional[HarmonicVPConfig] = None,
        seed: int = 42,
        normalize: bool = True,
        stats: Optional[Dict] = None,
        main_ratio: float = 0.8,
    ):
        self.main_ratio = main_ratio

        self.config = config or HarmonicVPConfig()
        self.split = split
        self.n_samples = n_samples
        self.seed = seed
        self.normalize = normalize

        np.random.seed(seed)


        self.alpha_range = getattr(self.config, f'{split}_alpha')
        self.beta_range = getattr(self.config, f'{split}_beta')
        self.eta_range = getattr(self.config, f'{split}_eta')


        self.x, self.c, self.e = self._generate_data()


        if normalize:
            if stats is not None:
                self.stats = stats
            else:
                self.stats = {
                    'x_mean': self.x.mean(),
                    'x_std': self.x.std() + 1e-8,
                    'c_mean': self.c.mean(axis=(0, 1)),
                    'c_std': self.c.std(axis=(0, 1)) + 1e-8,
                }

            self.x = (self.x - self.stats['x_mean']) / self.stats['x_std']
            self.c = (self.c - self.stats['c_mean']) / self.stats['c_std']
        else:
            self.stats = None

    def _sample_params(self, split: str) -> Tuple[float, float, float]:
        alpha_range = getattr(self.config, f'{split}_alpha')
        beta_range = getattr(self.config, f'{split}_beta')
        eta_range = getattr(self.config, f'{split}_eta')

        alpha = np.random.uniform(*alpha_range)
        beta = np.random.uniform(*beta_range)
        eta = np.random.uniform(*eta_range)

        return alpha, beta, eta

    def _generate_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.config
        t_eval = np.linspace(0, cfg.T, cfg.seq_len)


        n_main = int(self.n_samples * self.main_ratio)
        n_cross = self.n_samples - n_main


        other_splits = [s for s in ['train', 'val', 'test'] if s != self.split]

        x_list = []
        c_list = []
        e_list = []

        print(f"\n[{self.split}] CaTSG 80/20 mixed sampling:")
        print(f"  Main ({self.main_ratio*100:.0f}%): {n_main} from {self.split}")
        print(f"  Cross ({(1-self.main_ratio)*100:.0f}%): {n_cross} from {other_splits}")

        for i in range(self.n_samples):

            if i < n_main:

                alpha, beta, eta = self._sample_params(self.split)
            else:

                cross_split = np.random.choice(other_splits)
                alpha, beta, eta = self._sample_params(cross_split)


            x0 = np.random.uniform(*cfg.x0_range)
            v0 = np.random.uniform(*cfg.v0_range)


            def dynamics(t, y):
                pos, vel = y
                m = cfg.m0 + alpha * t
                gamma_t = cfg.gamma0 * (1 + beta * t)
                k_t = cfg.k0 * (1 + eta * t)
                acc = (-gamma_t * vel - k_t * pos) / m
                return [vel, acc]


            sol = solve_ivp(
                dynamics,
                [0, cfg.T],
                [x0, v0],
                t_eval=t_eval,
                method='RK45'
            )

            position = sol.y[0]
            velocity = sol.y[1]


            acceleration = np.zeros(cfg.seq_len)
            for j, t in enumerate(t_eval):
                m = cfg.m0 + alpha * t
                gamma_t = cfg.gamma0 * (1 + beta * t)
                k_t = cfg.k0 * (1 + eta * t)
                acceleration[j] = (-gamma_t * velocity[j] - k_t * position[j]) / m

            x_list.append(acceleration[:, np.newaxis])
            c_list.append(np.stack([velocity, position], axis=-1))
            e_list.append([alpha, beta, eta])

        self.stats = {
            'x_mean': np.array(x_list).mean(),
            'x_std': np.array(x_list).std() + 1e-8,
            'c_mean': np.array(c_list).mean(axis=(0, 1)),
            'c_std': np.array(c_list).std(axis=(0, 1)) + 1e-8,
        }

        return (
            np.array(x_list, dtype=np.float32),
            np.array(c_list, dtype=np.float32),
            np.array(e_list, dtype=np.float32)
        )


def get_harmonic_vp_dataloaders(
    n_train: int = 3000,
    n_val: int = 1000,
    n_test: int = 1000,
    batch_size: int = 64,
    seed: int = 42,
    normalize: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    config = HarmonicVPConfig()

    train_set = HarmonicVPDataset(n_train, 'train', config, seed, normalize)
    val_set = HarmonicVPDataset(n_val, 'val', config, seed+1, normalize, train_set.stats)
    test_set = HarmonicVPDataset(n_test, 'test', config, seed+2, normalize, train_set.stats)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    info = {
        'seq_len': config.seq_len,
        'x_dim': 1,
        'c_dim': 2,
        'e_dim': 3,
        'stats': train_set.stats,
    }

    return train_loader, val_loader, test_loader, info


def get_harmonic_vp_dataloaders_catsg_style(
    n_train: int = 3000,
    n_val: int = 1000,
    n_test: int = 1000,
    batch_size: int = 64,
    seed: int = 42,
    normalize: bool = True,
    main_ratio: float = 0.8,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    config = HarmonicVPConfig()

    print("=" * 60)
    print("  Harmonic VP Dataset - CaTSG-style 80/20 mixed sampling")
    print("=" * 60)
    print(f"  Main sampling ratio: {main_ratio*100:.0f}% / cross sampling: {(1-main_ratio)*100:.0f}%")

    train_set = HarmonicVPDatasetMixed(
        n_train, 'train', config, seed, normalize,
        stats=None, main_ratio=main_ratio
    )
    val_set = HarmonicVPDatasetMixed(
        n_val, 'val', config, seed+1, normalize,
        stats=train_set.stats, main_ratio=main_ratio
    )
    test_set = HarmonicVPDatasetMixed(
        n_test, 'test', config, seed+2, normalize,
        stats=train_set.stats, main_ratio=main_ratio
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    print(f"\n  Dataset created")
    print(f"    X shape: (N, {config.seq_len}, 1)")
    print(f"    C shape: (N, {config.seq_len}, 2)")
    print(f"    E shape: (N, 3) [alpha, beta, eta]")

    info = {
        'seq_len': config.seq_len,
        'x_dim': 1,
        'c_dim': 2,
        'e_dim': 3,
        'stats': train_set.stats,
    }

    return train_loader, val_loader, test_loader, info


if __name__ == '__main__':

    print("Testing Harmonic-VP dataset...")

    train_loader, val_loader, test_loader, info = get_harmonic_vp_dataloaders_catsg_style(
        n_train=100,
        n_val=50,
        n_test=50,
        batch_size=16,
    )

    print("\n" + "=" * 40)
    print("Sample batch:")
    for batch in train_loader:
        print(f"  x: {batch['x'].shape}")
        print(f"  c: {batch['c'].shape}")
        print(f"  e: {batch['e'].shape}")
        print(f"  α range: [{batch['alpha'].min():.3f}, {batch['alpha'].max():.3f}]")
        print(f"  β range: [{batch['beta'].min():.4f}, {batch['beta'].max():.4f}]")
        print(f"  η range: [{batch['eta'].min():.3f}, {batch['eta'].max():.3f}]")
        break

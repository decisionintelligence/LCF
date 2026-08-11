import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.integrate import solve_ivp
from pathlib import Path
import pickle


class HarmonicVMDataset(Dataset):

    def __init__(
        self,
        n_samples: int = 1000,
        seq_len: int = 48,
        alpha_range: tuple = (0.0, 1.0),
        m0: float = 0.5,
        gamma: float = 0.1,
        k: float = 1.0,
        T: float = 30.0,
        x0_range: tuple = (-2.0, 2.0),
        v0_range: tuple = (-1.5, 1.5),
        normalize: bool = True,
    ):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.alpha_range = alpha_range
        self.m0 = m0
        self.gamma = gamma
        self.k = k
        self.T = T
        self.x0_range = x0_range
        self.v0_range = v0_range
        self.normalize = normalize


        self._generate_data()

        if normalize:
            self._normalize()

    def _harmonic_ode(self, t, y, alpha):
        x, v = y
        m = self.m0 + alpha * t
        a = (-self.k * x - self.gamma * v) / m
        return [v, a]

    def _generate_single_trajectory(self, alpha, x0, v0):
        t_span = (0, self.T)
        t_eval = np.linspace(0, self.T, self.seq_len)

        sol = solve_ivp(
            lambda t, y: self._harmonic_ode(t, y, alpha),
            t_span,
            [x0, v0],
            t_eval=t_eval,
            method='RK45'
        )


        x = sol.y[0]
        v = sol.y[1]


        m = self.m0 + alpha * t_eval
        a = (-self.k * x - self.gamma * v) / m

        return x, v, a, t_eval

    def _generate_data(self):
        all_x = []
        all_c = []
        all_e = []

        for _ in range(self.n_samples):

            alpha = np.random.uniform(*self.alpha_range)


            x0 = np.random.uniform(*self.x0_range)
            v0 = np.random.uniform(*self.v0_range)


            pos, vel, acc, _ = self._generate_single_trajectory(alpha, x0, v0)


            all_x.append(acc)


            c = np.stack([pos, vel], axis=-1)
            all_c.append(c)


            all_e.append(alpha)

        self.x_data = np.array(all_x, dtype=np.float32)
        self.c_data = np.array(all_c, dtype=np.float32)
        self.e_data = np.array(all_e, dtype=np.float32)


        self.stats = {
            'x_mean': self.x_data.mean(),
            'x_std': self.x_data.std(),
            'c_mean': self.c_data.mean(axis=(0, 1)),
            'c_std': self.c_data.std(axis=(0, 1)),
            'e_mean': self.e_data.mean(),
            'e_std': self.e_data.std(),
        }

    def _normalize(self):
        self.x_data = (self.x_data - self.stats['x_mean']) / (self.stats['x_std'] + 1e-8)
        self.c_data = (self.c_data - self.stats['c_mean']) / (self.stats['c_std'] + 1e-8)
        self.e_data = (self.e_data - self.stats['e_mean']) / (self.stats['e_std'] + 1e-8)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            'x': torch.from_numpy(self.x_data[idx:idx+1].T),
            'c': torch.from_numpy(self.c_data[idx]),
            'e_true': torch.tensor([self.e_data[idx]]),
        }


def get_harmonic_vm_dataloaders(
    batch_size: int = 64,
    seq_len: int = 48,
    train_alpha: tuple = (0.0, 0.3),
    val_alpha: tuple = (0.3, 0.6),
    test_alpha: tuple = (0.6, 1.0),
    n_train: int = 3000,
    n_val: int = 1000,
    n_test: int = 1000,
    num_workers: int = 0,
):
    print("=" * 60)
    print("  Harmonic VM Dataset - Damped Harmonic Oscillator with Variable Mass")
    print("=" * 60)
    print(f"  Physical system: m(t) = m0 + alpha * t")
    print(f"  Target variable X: acceleration")
    print(f"  Condition variables C: [position, velocity]")
    print(f"  Environment variable E: alpha (mass change rate)")
    print()
    print(f"  Train alpha: {train_alpha}, n={n_train}")
    print(f"  Val alpha: {val_alpha}, n={n_val}")
    print(f"  Test alpha: {test_alpha}, n={n_test}")


    train_dataset = HarmonicVMDataset(
        n_samples=n_train,
        seq_len=seq_len,
        alpha_range=train_alpha,
    )


    val_dataset = HarmonicVMDataset(
        n_samples=n_val,
        seq_len=seq_len,
        alpha_range=val_alpha,
        normalize=False,
    )
    val_dataset.stats = train_dataset.stats
    val_dataset._normalize()

    test_dataset = HarmonicVMDataset(
        n_samples=n_test,
        seq_len=seq_len,
        alpha_range=test_alpha,
        normalize=False,
    )
    test_dataset.stats = train_dataset.stats
    test_dataset._normalize()


    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    config = {
        'seq_len': seq_len,
        'x_dim': 1,
        'c_dim': 2,
        'e_dim': 1,
        'stats': train_dataset.stats,
        'train_alpha': train_alpha,
        'val_alpha': val_alpha,
        'test_alpha': test_alpha,
    }

    print()
    print(f"  Dataset created")
    print(f"    X shape: (N, {seq_len}, 1)")
    print(f"    C shape: (N, {seq_len}, 2)")
    print(f"    E shape: (N, 1)")

    return train_loader, val_loader, test_loader, config


class HarmonicVMDatasetMixed(HarmonicVMDataset):

    def __init__(
        self,
        n_samples: int = 1000,
        seq_len: int = 48,
        main_alpha_range: tuple = (0.0, 0.2),
        cross_alpha_ranges: list = None,
        main_ratio: float = 0.8,
        m0: float = 0.5,
        gamma: float = 0.1,
        k: float = 1.0,
        T: float = 30.0,
        x0_range: tuple = (-2.0, 2.0),
        v0_range: tuple = (-1.5, 1.5),
        normalize: bool = True,
    ):
        self.main_alpha_range = main_alpha_range
        self.cross_alpha_ranges = cross_alpha_ranges or []
        self.main_ratio = main_ratio


        super().__init__(
            n_samples=n_samples,
            seq_len=seq_len,
            alpha_range=main_alpha_range,
            m0=m0,
            gamma=gamma,
            k=k,
            T=T,
            x0_range=x0_range,
            v0_range=v0_range,
            normalize=normalize,
        )

    def _generate_data(self):
        self.x_data = []
        self.c_data = []
        self.e_true = []


        n_main = int(self.n_samples * self.main_ratio)
        n_cross_total = self.n_samples - n_main


        for _ in range(n_main):
            alpha = np.random.uniform(self.main_alpha_range[0], self.main_alpha_range[1])
            x0 = np.random.uniform(*self.x0_range)
            v0 = np.random.uniform(*self.v0_range)
            x, v, a, _ = self._generate_single_trajectory(alpha, x0, v0)
            self.x_data.append(a.reshape(-1, 1))
            self.c_data.append(np.stack([x, v], axis=1))
            self.e_true.append(alpha)


        if self.cross_alpha_ranges and n_cross_total > 0:
            n_cross_each = n_cross_total // len(self.cross_alpha_ranges)
            n_remainder = n_cross_total % len(self.cross_alpha_ranges)

            for i, cross_range in enumerate(self.cross_alpha_ranges):
                n_this = n_cross_each + (1 if i < n_remainder else 0)
                for _ in range(n_this):
                    alpha = np.random.uniform(cross_range[0], cross_range[1])
                    x0 = np.random.uniform(*self.x0_range)
                    v0 = np.random.uniform(*self.v0_range)
                    x, v, a, _ = self._generate_single_trajectory(alpha, x0, v0)
                    self.x_data.append(a.reshape(-1, 1))
                    self.c_data.append(np.stack([x, v], axis=1))
                    self.e_true.append(alpha)


        self.x_data = np.array(self.x_data, dtype=np.float32)
        self.c_data = np.array(self.c_data, dtype=np.float32)
        self.e_true = np.array(self.e_true, dtype=np.float32).reshape(-1, 1)
        self.e_data = self.e_true.squeeze()


        indices = np.random.permutation(len(self.x_data))
        self.x_data = self.x_data[indices]
        self.c_data = self.c_data[indices]
        self.e_true = self.e_true[indices]
        self.e_data = self.e_data[indices]


        self.stats = {
            'x_mean': self.x_data.mean(),
            'x_std': self.x_data.std(),
            'c_mean': self.c_data.mean(axis=(0, 1)),
            'c_std': self.c_data.std(axis=(0, 1)),
            'e_mean': self.e_data.mean(),
            'e_std': self.e_data.std(),
        }


def get_harmonic_vm_dataloaders_catsg_style(
    batch_size: int = 64,
    seq_len: int = 48,
    n_train: int = 3000,
    n_val: int = 1000,
    n_test: int = 1000,
    main_ratio: float = 0.8,
    num_workers: int = 0,
):
    print("=" * 60)
    print("  Harmonic VM Dataset - CaTSG-style 80/20 mixed sampling")
    print("=" * 60)
    print(f"  Main sampling ratio: {main_ratio*100:.0f}% / cross sampling: {(1-main_ratio)*100:.0f}%")
    print()


    all_ranges = {
        'train': (0.0, 0.2),
        'val': (0.3, 0.5),
        'test': (0.6, 1.0),
    }


    train_dataset = HarmonicVMDatasetMixed(
        n_samples=n_train,
        seq_len=seq_len,
        main_alpha_range=all_ranges['train'],
        cross_alpha_ranges=[all_ranges['val'], all_ranges['test']],
        main_ratio=main_ratio,
    )
    n_train_main = int(n_train * main_ratio)
    print(f"  Train: {n_train_main} from [0.0,0.2] + {n_train-n_train_main} from [0.3,0.5] U [0.6,1.0]")


    val_dataset = HarmonicVMDatasetMixed(
        n_samples=n_val,
        seq_len=seq_len,
        main_alpha_range=all_ranges['val'],
        cross_alpha_ranges=[all_ranges['train'], all_ranges['test']],
        main_ratio=main_ratio,
        normalize=False,
    )
    val_dataset.stats = train_dataset.stats
    val_dataset._normalize()
    n_val_main = int(n_val * main_ratio)
    print(f"  Val: {n_val_main} from [0.3,0.5] + {n_val-n_val_main} from [0.0,0.2] U [0.6,1.0]")


    test_dataset = HarmonicVMDatasetMixed(
        n_samples=n_test,
        seq_len=seq_len,
        main_alpha_range=all_ranges['test'],
        cross_alpha_ranges=[all_ranges['train'], all_ranges['val']],
        main_ratio=main_ratio,
        normalize=False,
    )
    test_dataset.stats = train_dataset.stats
    test_dataset._normalize()
    n_test_main = int(n_test * main_ratio)
    print(f"  Test: {n_test_main} from [0.6,1.0] + {n_test-n_test_main} from [0.0,0.2] U [0.3,0.5]")


    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    config = {
        'seq_len': seq_len,
        'x_dim': 1,
        'c_dim': 2,
        'e_dim': 1,
        'stats': train_dataset.stats,
        'train_alpha': all_ranges['train'],
        'val_alpha': all_ranges['val'],
        'test_alpha': all_ranges['test'],
        'main_ratio': main_ratio,
        'sampling_style': 'catsg_mixed',
    }

    print()
    print(f"  Dataset created")
    print(f"    X shape: (N, {seq_len}, 1)")
    print(f"    C shape: (N, {seq_len}, 2)")
    print(f"    E shape: (N, 1)")

    return train_loader, val_loader, test_loader, config


if __name__ == "__main__":

    train_loader, val_loader, test_loader, config = get_harmonic_vm_dataloaders(
        batch_size=32,
        n_train=100,
        n_val=50,
        n_test=50,
    )

    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  x: {batch['x'].shape}")
    print(f"  c: {batch['c'].shape}")
    print(f"  e_true: {batch['e_true'].shape}")


    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))


    dataset = HarmonicVMDataset(n_samples=5, alpha_range=(0.0, 0.1), normalize=False)
    for i in range(3):
        axes[0, 0].plot(dataset.x_data[i], label=f'α≈0.05')

    dataset_high = HarmonicVMDataset(n_samples=5, alpha_range=(0.8, 1.0), normalize=False)
    for i in range(3):
        axes[0, 0].plot(dataset_high.x_data[i], '--', label=f'α≈0.9')

    axes[0, 0].set_title('Acceleration (X) vs Time - Different α')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Acceleration')
    axes[0, 0].legend()


    axes[0, 1].scatter(dataset.c_data[:, :, 0].flatten(),
                       dataset.c_data[:, :, 1].flatten(),
                       alpha=0.3, label='Low α', s=1)
    axes[0, 1].scatter(dataset_high.c_data[:, :, 0].flatten(),
                       dataset_high.c_data[:, :, 1].flatten(),
                       alpha=0.3, label='High α', s=1)
    axes[0, 1].set_title('Phase Space (Position vs Velocity)')
    axes[0, 1].set_xlabel('Position')
    axes[0, 1].set_ylabel('Velocity')
    axes[0, 1].legend()


    all_alphas = np.concatenate([
        np.random.uniform(0.0, 0.3, 1000),
        np.random.uniform(0.3, 0.6, 500),
        np.random.uniform(0.6, 1.0, 500),
    ])
    axes[1, 0].hist(all_alphas[:1000], bins=30, alpha=0.7, label='Train')
    axes[1, 0].hist(all_alphas[1000:1500], bins=30, alpha=0.7, label='Val')
    axes[1, 0].hist(all_alphas[1500:], bins=30, alpha=0.7, label='Test')
    axes[1, 0].set_title('Alpha Distribution by Split')
    axes[1, 0].set_xlabel('Alpha')
    axes[1, 0].legend()


    axes[1, 1].text(0.5, 0.7, 'E (α)', fontsize=20, ha='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue'))
    axes[1, 1].text(0.2, 0.3, 'C (x,v)', fontsize=16, ha='center',
                    bbox=dict(boxstyle='round', facecolor='lightgreen'))
    axes[1, 1].text(0.8, 0.3, 'X (a)', fontsize=16, ha='center',
                    bbox=dict(boxstyle='round', facecolor='lightyellow'))
    axes[1, 1].annotate('', xy=(0.35, 0.3), xytext=(0.65, 0.3),
                        arrowprops=dict(arrowstyle='->', lw=2))
    axes[1, 1].annotate('', xy=(0.5, 0.55), xytext=(0.5, 0.4),
                        arrowprops=dict(arrowstyle='->', lw=2))
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title('Causal Structure')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('harmonic_vm_overview.png', dpi=150)
    print("\n✓ Saved: harmonic_vm_overview.png")

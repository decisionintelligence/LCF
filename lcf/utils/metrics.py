import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error


def compute_mmd(x: np.ndarray, y: np.ndarray, kernel: str = 'rbf', gamma: float = 1.0) -> float:
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)

    if kernel == 'rbf':
        xx = np.sum(x ** 2, axis=1, keepdims=True)
        yy = np.sum(y ** 2, axis=1, keepdims=True)

        xx_dist = xx + xx.T - 2 * np.dot(x, x.T)
        yy_dist = yy + yy.T - 2 * np.dot(y, y.T)
        xy_dist = xx + yy.T - 2 * np.dot(x, y.T)

        k_xx = np.exp(-gamma * xx_dist)
        k_yy = np.exp(-gamma * yy_dist)
        k_xy = np.exp(-gamma * xy_dist)
    elif kernel == 'linear':
        k_xx = np.dot(x, x.T)
        k_yy = np.dot(y, y.T)
        k_xy = np.dot(x, y.T)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    n = x.shape[0]
    m = y.shape[0]

    mmd = (k_xx.sum() / (n * n) + k_yy.sum() / (m * m) - 2 * k_xy.sum() / (n * m))

    return float(np.sqrt(max(0, mmd)))


def compute_wasserstein_distance(x: np.ndarray, y: np.ndarray) -> float:
    x = x.flatten()
    y = y.flatten()
    return float(stats.wasserstein_distance(x, y))


def compute_mdd(real: np.ndarray, generated: np.ndarray) -> float:
    if real.shape[1] != generated.shape[1]:
        raise ValueError("Sequence lengths must match")

    T = real.shape[1]
    distances = []

    for t in range(T):
        w_dist = compute_wasserstein_distance(real[:, t].flatten(), generated[:, t].flatten())
        distances.append(w_dist)

    return float(np.mean(distances))


def compute_discriminative_score(
    real: np.ndarray,
    generated: np.ndarray,
    test_size: float = 0.3,
    hidden_dim: int = 64
) -> float:
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier


    real_flat = real.reshape(real.shape[0], -1)
    gen_flat = generated.reshape(generated.shape[0], -1)


    X = np.vstack([real_flat, gen_flat])
    y = np.concatenate([np.ones(len(real_flat)), np.zeros(len(gen_flat))])


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )


    clf = MLPClassifier(
        hidden_layer_sizes=(hidden_dim, hidden_dim),
        max_iter=500,
        random_state=42
    )
    clf.fit(X_train, y_train)


    score = clf.score(X_test, y_test)


    return float(abs(score - 0.5))


def compute_correlation_score(
    real: np.ndarray,
    generated: np.ndarray,
    condition: np.ndarray
) -> Dict[str, float]:

    real_flat = real.mean(axis=1).flatten()
    gen_flat = generated.mean(axis=1).flatten()
    cond_flat = condition.mean(axis=1).flatten()


    corr_real_cond = np.corrcoef(real_flat, cond_flat[:len(real_flat)])[0, 1]
    corr_gen_cond = np.corrcoef(gen_flat, cond_flat[:len(gen_flat)])[0, 1]

    return {
        'corr_real_cond': float(corr_real_cond) if not np.isnan(corr_real_cond) else 0.0,
        'corr_gen_cond': float(corr_gen_cond) if not np.isnan(corr_gen_cond) else 0.0,
        'corr_diff': float(abs(corr_real_cond - corr_gen_cond)) if not (np.isnan(corr_real_cond) or np.isnan(corr_gen_cond)) else 1.0
    }


def compute_temporal_stats(data: np.ndarray) -> Dict[str, float]:

    mean_per_t = data.mean(axis=(0, 2))
    std_per_t = data.std(axis=(0, 2))


    autocorr_values = []
    for b in range(data.shape[0]):
        series = data[b, :, 0]
        if len(series) > 1:
            autocorr = np.corrcoef(series[:-1], series[1:])[0, 1]
            if not np.isnan(autocorr):
                autocorr_values.append(autocorr)

    return {
        'temporal_mean': float(np.mean(mean_per_t)),
        'temporal_std': float(np.mean(std_per_t)),
        'mean_autocorr': float(np.mean(autocorr_values)) if autocorr_values else 0.0
    }


def compute_all_metrics(
    real: np.ndarray,
    generated: np.ndarray,
    condition: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    metrics = {}


    metrics['mmd_rbf'] = compute_mmd(real, generated, kernel='rbf')
    metrics['mmd_linear'] = compute_mmd(real, generated, kernel='linear')
    metrics['mdd'] = compute_mdd(real, generated)


    metrics['discriminative'] = compute_discriminative_score(real, generated)


    min_samples = min(len(real), len(generated))
    metrics['mse'] = float(mean_squared_error(
        real[:min_samples].flatten(),
        generated[:min_samples].flatten()
    ))
    metrics['mae'] = float(mean_absolute_error(
        real[:min_samples].flatten(),
        generated[:min_samples].flatten()
    ))


    real_stats = compute_temporal_stats(real)
    gen_stats = compute_temporal_stats(generated)
    metrics['temporal_mean_diff'] = abs(real_stats['temporal_mean'] - gen_stats['temporal_mean'])
    metrics['temporal_std_diff'] = abs(real_stats['temporal_std'] - gen_stats['temporal_std'])
    metrics['autocorr_diff'] = abs(real_stats['mean_autocorr'] - gen_stats['mean_autocorr'])


    if condition is not None:
        corr_metrics = compute_correlation_score(real, generated, condition)
        metrics.update(corr_metrics)

    return metrics


class JFTSD:

    def __init__(self, hidden_dim: int = 64):
        self.hidden_dim = hidden_dim

    def __call__(
        self,
        real: np.ndarray,
        generated: np.ndarray,
        condition: Optional[np.ndarray] = None
    ) -> float:

        real_flat = real.reshape(real.shape[0], -1)
        gen_flat = generated.reshape(generated.shape[0], -1)

        mu_real = np.mean(real_flat, axis=0)
        mu_gen = np.mean(gen_flat, axis=0)

        sigma_real = np.cov(real_flat, rowvar=False)
        sigma_gen = np.cov(gen_flat, rowvar=False)


        sigma_real = np.atleast_2d(sigma_real)
        sigma_gen = np.atleast_2d(sigma_gen)


        diff = mu_real - mu_gen
        mean_term = np.dot(diff, diff)


        try:
            from scipy.linalg import sqrtm
            covmean = sqrtm(sigma_real @ sigma_gen)
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            cov_term = np.trace(sigma_real + sigma_gen - 2 * covmean)
        except:

            cov_term = abs(np.trace(sigma_real) - np.trace(sigma_gen))

        fid = mean_term + cov_term

        return float(max(0, fid))


def get_jftsd(
    real: np.ndarray,
    condition: np.ndarray,
    generated: np.ndarray,
    device: Optional[torch.device] = None
) -> float:
    jftsd = JFTSD()
    return jftsd(real, generated, condition)

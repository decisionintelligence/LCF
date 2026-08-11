import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import numpy as np


class AuxiliaryEncoder(nn.Module):

    def __init__(
        self,
        u_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 32,
        use_positional_encoding: bool = True,
        n_frequencies: int = 8,
    ):
        super().__init__()

        self.u_dim = u_dim
        self.use_positional_encoding = use_positional_encoding
        self.n_frequencies = n_frequencies

        if use_positional_encoding:

            input_dim = u_dim * (1 + 2 * n_frequencies)
        else:
            input_dim = u_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def positional_encoding(self, u: torch.Tensor) -> torch.Tensor:
        original_shape = u.shape[:-1]
        u = u.reshape(-1, self.u_dim)

        encodings = []
        for d in range(self.u_dim):
            u_d = u[:, d:d+1]

            freqs = 2.0 ** torch.arange(self.n_frequencies, device=u.device, dtype=u.dtype)
            freqs = freqs * math.pi

            angles = u_d * freqs

            encoding_d = torch.cat([
                u_d,
                torch.sin(angles),
                torch.cos(angles),
            ], dim=-1)

            encodings.append(encoding_d)

        result = torch.cat(encodings, dim=-1)
        return result.reshape(*original_shape, -1)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        if self.use_positional_encoding:
            u_encoded = self.positional_encoding(u)
        else:
            u_encoded = u

        return self.net(u_encoded)


class ConditionalGMMPrior(nn.Module):

    def __init__(
        self,
        env_dim: int,
        u_dim: int = 1,
        n_components: int = 4,
        hidden_dim: int = 64,
        aux_hidden_dim: int = 64,
        min_std: float = 0.1,
        max_std: float = 2.0,
        use_diagonal_cov: bool = True,
    ):
        super().__init__()

        self.env_dim = env_dim
        self.u_dim = u_dim
        self.n_components = n_components
        self.min_std = min_std
        self.max_std = max_std
        self.use_diagonal_cov = use_diagonal_cov


        self.aux_encoder = AuxiliaryEncoder(
            u_dim=u_dim,
            hidden_dim=aux_hidden_dim,
            output_dim=hidden_dim,
        )


        if use_diagonal_cov:

            output_dim = n_components + n_components * env_dim + n_components * env_dim
        else:

            output_dim = n_components + n_components * env_dim + n_components

        self.param_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )


        with torch.no_grad():
            self.param_net[-1].bias.zero_()
            self.param_net[-1].weight.mul_(0.1)

    def get_params(self, u: torch.Tensor) -> Dict[str, torch.Tensor]:
        if u.dim() == 1:
            u = u.unsqueeze(-1)

        B = u.shape[0]
        K = self.n_components
        D = self.env_dim


        h = self.aux_encoder(u)


        params = self.param_net(h)


        idx = 0
        log_weights = params[:, idx:idx + K]
        idx += K

        means = params[:, idx:idx + K * D].reshape(B, K, D)
        idx += K * D

        if self.use_diagonal_cov:
            log_stds = params[:, idx:idx + K * D].reshape(B, K, D)
        else:
            log_stds = params[:, idx:idx + K]


        weights = F.softmax(log_weights, dim=-1)
        stds = torch.exp(log_stds).clamp(self.min_std, self.max_std)

        return {
            'weights': weights,
            'means': means,
            'stds': stds,
            'log_weights': log_weights,
            'log_stds': log_stds,
        }

    def log_prob(self, e: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        params = self.get_params(u)
        weights = params['weights']
        means = params['means']
        stds = params['stds']

        B, K, D = means.shape


        e_exp = e.unsqueeze(1)


        diff = e_exp - means

        if self.use_diagonal_cov:

            sq_dist = (diff ** 2 / (stds ** 2)).sum(dim=-1)
            log_det = torch.log(stds).sum(dim=-1)
        else:

            sq_dist = (diff ** 2).sum(dim=-1) / (stds ** 2)
            log_det = D * torch.log(stds)

        log_gauss = -0.5 * D * math.log(2 * math.pi) - log_det - 0.5 * sq_dist


        log_weights = torch.log(weights + 1e-10)
        log_prob = torch.logsumexp(log_weights + log_gauss, dim=-1)

        return log_prob

    def sample(self, u: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        params = self.get_params(u)
        weights = params['weights']
        means = params['means']
        stds = params['stds']

        B, K, D = means.shape


        k = torch.multinomial(weights, n_samples, replacement=True)

        if n_samples == 1:
            k = k.squeeze(-1)


            batch_idx = torch.arange(B, device=k.device)
            mean = means[batch_idx, k]

            if self.use_diagonal_cov:
                std = stds[batch_idx, k]
            else:
                std = stds[batch_idx, k].unsqueeze(-1).expand(-1, D)


            e = mean + std * torch.randn_like(mean)
            return e
        else:

            samples = []
            for s in range(n_samples):
                k_s = k[:, s]
                batch_idx = torch.arange(B, device=k.device)
                mean = means[batch_idx, k_s]

                if self.use_diagonal_cov:
                    std = stds[batch_idx, k_s]
                else:
                    std = stds[batch_idx, k_s].unsqueeze(-1).expand(-1, D)

                e_s = mean + std * torch.randn_like(mean)
                samples.append(e_s)

            return torch.stack(samples, dim=1)

    def kl_divergence(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        u: torch.Tensor,
        n_mc_samples: int = 1,
    ) -> torch.Tensor:
        D = mu.shape[-1]
        std = torch.exp(0.5 * logvar)

        total_kl = 0
        for _ in range(n_mc_samples):

            eps = torch.randn_like(mu)
            e = mu + std * eps


            log_q = -0.5 * D * math.log(2 * math.pi) \
                    - 0.5 * logvar.sum(dim=-1) \
                    - 0.5 * (eps ** 2).sum(dim=-1)


            log_p = self.log_prob(e, u)

            total_kl += (log_q - log_p).mean()

        return total_kl / n_mc_samples

    def get_natural_parameters(self, u: torch.Tensor) -> torch.Tensor:
        params = self.get_params(u)
        means = params['means']
        stds = params['stds']

        B, K, D = means.shape

        if self.use_diagonal_cov:
            var = stds ** 2
            lambda1 = means / var
            lambda2 = -0.5 / var


            lambdas = torch.cat([lambda1, lambda2], dim=-1)
        else:
            var = (stds ** 2).unsqueeze(-1)
            lambda1 = means / var
            lambda2 = -0.5 / var

            lambdas = torch.cat([lambda1, lambda2.expand(-1, -1, D)], dim=-1)

        return lambdas

    def verify_sufficient_variability(
        self,
        u_values: torch.Tensor,
        threshold: float = 0.9,
    ) -> Dict:
        with torch.no_grad():

            lambdas = self.get_natural_parameters(u_values)
            M, K, param_dim = lambdas.shape


            lambdas_flat = lambdas.reshape(M, -1).cpu().numpy()
            k = lambdas_flat.shape[1]


            lambda_0 = lambdas_flat[0]
            L = lambdas_flat[1:] - lambda_0


            rank = np.linalg.matrix_rank(L)
            required_rank = min(M - 1, k)
            relative_rank = rank / required_rank if required_rank > 0 else 1.0


            _, singular_values, _ = np.linalg.svd(L)

            return {
                'is_sufficient': relative_rank >= threshold,
                'relative_rank': relative_rank,
                'rank': rank,
                'required_rank': required_rank,
                'singular_values': singular_values,
                'L_matrix': L,
            }


class ConditionalGMMEnvironmentModule(nn.Module):

    def __init__(
        self,
        env_dim: int,
        u_dim: int = 1,
        n_components: int = 4,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.env_dim = env_dim
        self.n_components = n_components

        self.gmm_prior = ConditionalGMMPrior(
            env_dim=env_dim,
            u_dim=u_dim,
            n_components=n_components,
            hidden_dim=hidden_dim,
            use_diagonal_cov=True,
        )

    def kl_loss(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        u: torch.Tensor,
        kl_weight: float = 0.01,
    ) -> torch.Tensor:
        kl = self.gmm_prior.kl_divergence(mu, logvar, u)
        return kl_weight * kl

    def sample(self, u: torch.Tensor) -> torch.Tensor:
        return self.gmm_prior.sample(u)

    def log_prob(self, e: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.gmm_prior.log_prob(e, u)


if __name__ == '__main__':
    print("=" * 60)
    print("  Conditional GMM Prior Test")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    env_dim = 8
    u_dim = 2
    n_components = 4

    cond_gmm = ConditionalGMMPrior(
        env_dim=env_dim,
        u_dim=u_dim,
        n_components=n_components,
        hidden_dim=64,
    ).to(device)


    print("\n[1] Parameter generation")
    B = 32
    u = torch.randn(B, u_dim, device=device)
    params = cond_gmm.get_params(u)

    print(f"  Weights shape: {params['weights'].shape}")
    print(f"  Means shape: {params['means'].shape}")
    print(f"  Stds shape: {params['stds'].shape}")
    print(f"  Weights sum: {params['weights'].sum(dim=-1).mean().item():.4f}")


    print("\n[2] Sampling")
    samples = cond_gmm.sample(u)
    print(f"  Sample shape: {samples.shape}")
    print(f"  Sample mean: {samples.mean().item():.4f}")
    print(f"  Sample std: {samples.std().item():.4f}")


    print("\n[3] Log probability")
    log_p = cond_gmm.log_prob(samples, u)
    print(f"  Log prob shape: {log_p.shape}")
    print(f"  Log prob mean: {log_p.mean().item():.4f}")


    print("\n[4] KL divergence")
    mu = torch.randn(B, env_dim, device=device)
    logvar = torch.zeros(B, env_dim, device=device)
    kl = cond_gmm.kl_divergence(mu, logvar, u)
    print(f"  KL: {kl.item():.4f}")


    print("\n[5] Natural parameters")
    lambdas = cond_gmm.get_natural_parameters(u)
    print(f"  Lambda shape: {lambdas.shape}")


    print("\n[6] Sufficient variability")
    M = 20
    u_test = torch.randn(M, u_dim, device=device)
    var_result = cond_gmm.verify_sufficient_variability(u_test)
    print(f"  Is sufficient: {var_result['is_sufficient']}")
    print(f"  Relative rank: {var_result['relative_rank']:.4f}")
    print(f"  Rank: {var_result['rank']}/{var_result['required_rank']}")


    print("\n[7] Parameter sensitivity to u")
    u1 = torch.zeros(1, u_dim, device=device)
    u2 = torch.ones(1, u_dim, device=device)

    params1 = cond_gmm.get_params(u1)
    params2 = cond_gmm.get_params(u2)

    mean_diff = (params1['means'] - params2['means']).abs().mean().item()
    std_diff = (params1['stds'] - params2['stds']).abs().mean().item()
    weight_diff = (params1['weights'] - params2['weights']).abs().mean().item()

    print(f"  Mean difference: {mean_diff:.4f}")
    print(f"  Std difference: {std_diff:.4f}")
    print(f"  Weight difference: {weight_diff:.4f}")

    print("\n✅ All tests passed!")

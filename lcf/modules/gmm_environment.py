import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import numpy as np


class CurveNetwork(nn.Module):

    def __init__(
        self,
        output_dim: int,
        hidden_dim: int = 64,
        n_frequencies: int = 8,
    ):
        super().__init__()

        self.n_frequencies = n_frequencies

        input_dim = 1 + 2 * n_frequencies

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )


        with torch.no_grad():
            self.net[-1].weight.mul_(0.1)
            self.net[-1].bias.mul_(0.1)

    def positional_encoding(self, t: torch.Tensor) -> torch.Tensor:
        t = t.unsqueeze(-1)

        freqs = 2.0 ** torch.arange(self.n_frequencies, device=t.device, dtype=t.dtype)
        freqs = freqs * math.pi

        angles = t * freqs

        encoding = torch.cat([
            t,
            torch.sin(angles),
            torch.cos(angles),
        ], dim=-1)

        return encoding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        encoding = self.positional_encoding(t)
        return self.net(encoding)


class GMMPrior(nn.Module):

    def __init__(
        self,
        env_dim: int,
        n_components: int = 4,
        learnable_weights: bool = True,
        learnable_variance: bool = True,
        init_std: float = 0.5,
        min_std: float = 0.1,
        max_std: float = 2.0,
        use_curve_prior: bool = False,
        curve_hidden_dim: int = 64,
    ):
        super().__init__()

        self.env_dim = env_dim
        self.n_components = n_components
        self.min_std = min_std
        self.max_std = max_std
        self.use_curve_prior = use_curve_prior

        if use_curve_prior:

            self.curve_net = CurveNetwork(
                output_dim=env_dim,
                hidden_dim=curve_hidden_dim,
            )

            t_values = torch.linspace(0, 1, n_components)
            self.register_buffer('t_values', t_values)

            self.t_offsets = nn.Parameter(torch.zeros(n_components) * 0.01)
        else:

            init_centers = torch.randn(n_components, env_dim)
            if env_dim >= n_components:
                q, _ = torch.linalg.qr(init_centers.T)
                init_centers = q.T[:n_components] * 2.0
            self.centers = nn.Parameter(init_centers)


        if learnable_weights:
            self.log_weights = nn.Parameter(torch.zeros(n_components))
        else:
            self.register_buffer('log_weights', torch.zeros(n_components))


        if learnable_variance:
            self.log_stds = nn.Parameter(torch.full((n_components,), math.log(init_std)))
        else:
            self.register_buffer('log_stds', torch.full((n_components,), math.log(init_std)))

    @property
    def centers(self) -> torch.Tensor:
        if self.use_curve_prior:

            t = (self.t_values + self.t_offsets.tanh() * 0.1).clamp(0, 1)
            centers = self.curve_net(t)
        else:
            centers = self._centers


        return F.normalize(centers, p=2, dim=-1)

    @centers.setter
    def centers(self, value):
        if not hasattr(self, 'use_curve_prior') or not self.use_curve_prior:
            self._centers = value

    @property
    def weights(self) -> torch.Tensor:
        return F.softmax(self.log_weights, dim=0)

    @property
    def stds(self) -> torch.Tensor:
        return torch.exp(self.log_stds).clamp(self.min_std, self.max_std)

    def log_prob(self, e: torch.Tensor) -> torch.Tensor:
        B, D = e.shape
        K = self.n_components


        e_exp = e.unsqueeze(1)
        centers_exp = self.centers.unsqueeze(0)
        stds_exp = self.stds.view(1, K, 1)


        diff = e_exp - centers_exp
        sq_dist = (diff ** 2).sum(dim=-1)

        log_gauss = -0.5 * D * math.log(2 * math.pi) \
                    - D * torch.log(self.stds).unsqueeze(0) \
                    - 0.5 * sq_dist / (self.stds ** 2).unsqueeze(0)


        log_weights = torch.log(self.weights + 1e-10).unsqueeze(0)
        log_prob = torch.logsumexp(log_weights + log_gauss, dim=-1)

        return log_prob

    def compute_soft_assignment(self, e: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        B, D = e.shape
        K = self.n_components

        e_exp = e.unsqueeze(1)
        centers_exp = self.centers.unsqueeze(0)


        sq_dist = ((e_exp - centers_exp) ** 2).sum(dim=-1)


        logits = torch.log(self.weights + 1e-10).unsqueeze(0) \
                 - 0.5 * sq_dist / ((self.stds ** 2).unsqueeze(0) * temperature)

        return F.softmax(logits, dim=-1)

    def sample(self, n_samples: int, device: torch.device = None) -> torch.Tensor:
        if device is None:
            device = self.centers.device


        k = torch.multinomial(self.weights.expand(n_samples, -1), 1).squeeze(-1)


        means = self.centers[k]
        stds = self.stds[k].unsqueeze(-1).expand(-1, self.env_dim)

        return means + stds * torch.randn_like(means)

    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor,
                      n_mc_samples: int = 1,
                      free_bits: float = 0.0) -> torch.Tensor:
        B, D = mu.shape
        std = torch.exp(0.5 * logvar)

        total_kl = 0
        for _ in range(n_mc_samples):

            eps = torch.randn_like(mu)
            e = mu + std * eps


            log_q = -0.5 * D * math.log(2 * math.pi) \
                    - 0.5 * logvar.sum(dim=-1) \
                    - 0.5 * (eps ** 2).sum(dim=-1)


            log_p = self.log_prob(e)

            total_kl += (log_q - log_p).mean()

        kl = total_kl / n_mc_samples


        if free_bits > 0:
            kl = torch.max(kl, torch.tensor(free_bits * D, device=mu.device))

        return kl

    def orthogonal_loss(self) -> torch.Tensor:
        centers_norm = F.normalize(self.centers, dim=-1)
        sim = centers_norm @ centers_norm.T


        K = self.n_components
        mask = ~torch.eye(K, dtype=torch.bool, device=sim.device)
        return sim[mask].pow(2).mean()

    def balance_loss(self, assignment: torch.Tensor) -> torch.Tensor:
        usage = assignment.mean(dim=0)

        entropy = -(usage * torch.log(usage + 1e-10)).sum()
        max_entropy = math.log(self.n_components)
        return (max_entropy - entropy).clamp(min=0)


class VICRegLoss(nn.Module):

    def __init__(
        self,
        sim_weight: float = 25.0,
        var_weight: float = 25.0,
        cov_weight: float = 1.0,
        var_target: float = 1.0,
        use_cosine_sim: bool = True,
    ):
        super().__init__()
        self.sim_weight = sim_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.var_target = var_target
        self.use_cosine_sim = use_cosine_sim

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, D = z1.shape


        if self.use_cosine_sim:


            cos_sim = (z1 * z2).sum(dim=-1)
            sim_loss = (1 - cos_sim).mean()
        else:
            sim_loss = F.mse_loss(z1, z2)


        std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
        std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
        var_loss = F.relu(self.var_target - std_z1).mean() + \
                   F.relu(self.var_target - std_z2).mean()


        z1_centered = z1 - z1.mean(dim=0)
        z2_centered = z2 - z2.mean(dim=0)

        cov_z1 = (z1_centered.T @ z1_centered) / (B - 1)
        cov_z2 = (z2_centered.T @ z2_centered) / (B - 1)


        off_diag_z1 = cov_z1.pow(2).sum() - cov_z1.diag().pow(2).sum()
        off_diag_z2 = cov_z2.pow(2).sum() - cov_z2.diag().pow(2).sum()
        cov_loss = (off_diag_z1 + off_diag_z2) / D

        total = self.sim_weight * sim_loss + \
                self.var_weight * var_loss + \
                self.cov_weight * cov_loss

        return {
            'total': total,
            'sim': sim_loss,
            'var': var_loss,
            'cov': cov_loss,
        }

    def single_view_loss(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, D = z.shape


        std_z = torch.sqrt(z.var(dim=0) + 1e-4)
        var_loss = F.relu(self.var_target - std_z).mean()


        z_centered = z - z.mean(dim=0)
        cov_z = (z_centered.T @ z_centered) / (B - 1)
        off_diag = cov_z.pow(2).sum() - cov_z.diag().pow(2).sum()
        cov_loss = off_diag / D

        total = self.var_weight * var_loss + self.cov_weight * cov_loss

        return {
            'total': total,
            'var': var_loss,
            'cov': cov_loss,
        }


class SoftSwAVLoss(nn.Module):

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        gmm_prior: GMMPrior,
    ) -> torch.Tensor:

        q1 = gmm_prior.compute_soft_assignment(z1, self.temperature)
        q2 = gmm_prior.compute_soft_assignment(z2, self.temperature)


        loss = -(q2.detach() * torch.log(q1 + 1e-8)).sum(dim=-1).mean()
        loss += -(q1.detach() * torch.log(q2 + 1e-8)).sum(dim=-1).mean()

        return loss / 2


class EnvironmentPredictor(nn.Module):

    def __init__(self, env_dim: int, hidden_dim: int = 64, n_targets: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(env_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_targets),
        )

    def forward(self, e: torch.Tensor, x: torch.Tensor) -> torch.Tensor:

        x_flat = x.reshape(x.shape[0], -1)

        x_stats = torch.stack([
            x_flat.mean(dim=-1),
            x_flat.std(dim=-1),
            x_flat.max(dim=-1).values,
            x_flat.min(dim=-1).values,
        ], dim=-1)


        pred = self.net(e)

        return F.mse_loss(pred, x_stats)


class EnvironmentSensitivityMonitor:

    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold
        self.history = []

    @torch.no_grad()
    def check_sensitivity(
        self,
        velocity_net: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        e: torch.Tensor,
        n_samples: int = 5,
        perturbation_scale: float = 0.5,
    ) -> Dict[str, float]:

        v_original = velocity_net(x_t, t, c, e)

        sensitivities = []
        for _ in range(n_samples):

            e_perturbed = e + perturbation_scale * torch.randn_like(e)
            v_perturbed = velocity_net(x_t, t, c, e_perturbed)


            diff = (v_original - v_perturbed).abs().mean()
            sensitivities.append(diff.item())

        avg_sensitivity = np.mean(sensitivities)
        max_sensitivity = np.max(sensitivities)


        self.history.append(avg_sensitivity)


        is_ignored = avg_sensitivity < self.threshold

        return {
            'avg_sensitivity': avg_sensitivity,
            'max_sensitivity': max_sensitivity,
            'is_ignored': is_ignored,
            'warning': f"⚠️ e may be ignored! sensitivity={avg_sensitivity:.6f}" if is_ignored else None,
        }

    def get_trend(self, window: int = 10) -> str:
        if len(self.history) < window:
            return "insufficient_data"

        recent = self.history[-window:]
        earlier = self.history[-2*window:-window] if len(self.history) >= 2*window else self.history[:window]

        recent_avg = np.mean(recent)
        earlier_avg = np.mean(earlier)

        if recent_avg > earlier_avg * 1.1:
            return "increasing ✅"
        elif recent_avg < earlier_avg * 0.9:
            return "decreasing ⚠️"
        else:
            return "stable"


class TimeSeriesAugmentation:

    @staticmethod
    def jitter(x: torch.Tensor, sigma: float = 0.05) -> torch.Tensor:
        return x + sigma * torch.randn_like(x)

    @staticmethod
    def scale(x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
        factor = torch.empty(x.shape[0], 1, 1, device=x.device).uniform_(1 - sigma, 1 + sigma)
        return x * factor

    @staticmethod
    def shift(x: torch.Tensor, max_shift: float = 0.1) -> torch.Tensor:
        shift = torch.empty(x.shape[0], 1, 1, device=x.device).uniform_(-max_shift, max_shift)
        return x + shift

    @staticmethod
    def augment(x: torch.Tensor, c: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:

        x_aug = x.clone()


        x_aug = TimeSeriesAugmentation.jitter(x_aug, sigma=0.03)


        if torch.rand(1).item() > 0.5:
            x_aug = TimeSeriesAugmentation.scale(x_aug, sigma=0.05)


        if torch.rand(1).item() > 0.5:
            x_aug = TimeSeriesAugmentation.shift(x_aug, max_shift=0.05)

        return x_aug, c


class GMMEnvironmentModule(nn.Module):

    def __init__(
        self,
        env_dim: int,
        n_components: int = 4,
        vicreg_config: Dict = None,
        use_soft_swav: bool = True,
        swav_temperature: float = 0.1,
        use_env_predictor: bool = True,
        env_predictor_hidden_dim: int = 64,
        use_curve_prior: bool = False,
        curve_hidden_dim: int = 64,
    ):
        super().__init__()

        self.env_dim = env_dim
        self.n_components = n_components
        self.use_curve_prior = use_curve_prior


        self.gmm_prior = GMMPrior(
            env_dim=env_dim,
            n_components=n_components,
            learnable_weights=True,
            learnable_variance=True,
            use_curve_prior=use_curve_prior,
            curve_hidden_dim=curve_hidden_dim,
        )


        vicreg_config = vicreg_config or {}
        self.vicreg = VICRegLoss(
            sim_weight=vicreg_config.get('sim_weight', 25.0),
            var_weight=vicreg_config.get('var_weight', 25.0),
            cov_weight=vicreg_config.get('cov_weight', 1.0),
        )


        self.use_soft_swav = use_soft_swav
        if use_soft_swav:
            self.soft_swav = SoftSwAVLoss(temperature=swav_temperature)


        self.use_env_predictor = use_env_predictor
        if use_env_predictor:
            self.env_predictor = EnvironmentPredictor(
                env_dim=env_dim,
                hidden_dim=env_predictor_hidden_dim,
            )


        self.sensitivity_monitor = EnvironmentSensitivityMonitor(threshold=0.01)


        self.augment = TimeSeriesAugmentation.augment

    def warmup_loss(
        self,
        mu: torch.Tensor,
        x: torch.Tensor,
        c: torch.Tensor,
        encoder_fn,
        ortho_weight: float = 0.1,
        balance_weight: float = 0.5,
        swav_weight: float = 1.0,
        env_pred_weight: float = 0.5,
    ) -> Dict[str, torch.Tensor]:

        x_aug, c_aug = self.augment(x, c)


        enc_aug = encoder_fn(x_aug, c_aug)
        mu_aug = enc_aug['mu']


        vicreg_losses = self.vicreg(mu, mu_aug)


        swav_loss = torch.tensor(0.0, device=mu.device)
        if self.use_soft_swav:
            swav_loss = self.soft_swav(mu, mu_aug, self.gmm_prior)


        ortho_loss = self.gmm_prior.orthogonal_loss()


        assignment = self.gmm_prior.compute_soft_assignment(mu)
        balance_loss = self.gmm_prior.balance_loss(assignment)


        env_pred_loss = torch.tensor(0.0, device=mu.device)
        if self.use_env_predictor:
            env_pred_loss = self.env_predictor(mu, x)


        total = vicreg_losses['total'] + \
                swav_weight * swav_loss + \
                ortho_weight * ortho_loss + \
                balance_weight * balance_loss + \
                env_pred_weight * env_pred_loss

        return {
            'total': total,
            'vicreg': vicreg_losses['total'],
            'vicreg_sim': vicreg_losses['sim'],
            'vicreg_var': vicreg_losses['var'],
            'vicreg_cov': vicreg_losses['cov'],
            'swav': swav_loss,
            'ortho': ortho_loss,
            'balance': balance_loss,
            'env_pred': env_pred_loss,
            'mu_std': mu.std().item(),
        }

    def training_loss(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        x: torch.Tensor = None,
        kl_weight: float = 0.01,
        var_weight: float = 1.0,
        env_pred_weight: float = 0.1,
        free_bits: float = 0.1,
    ) -> Dict[str, torch.Tensor]:

        kl_loss = self.gmm_prior.kl_divergence(mu, logvar, n_mc_samples=1, free_bits=free_bits)


        vicreg_losses = self.vicreg.single_view_loss(mu)


        e = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        assignment = self.gmm_prior.compute_soft_assignment(e)
        balance_loss = self.gmm_prior.balance_loss(assignment)


        env_pred_loss = torch.tensor(0.0, device=mu.device)
        if self.use_env_predictor and x is not None:
            env_pred_loss = self.env_predictor(e, x)

        total = kl_weight * kl_loss + \
                var_weight * vicreg_losses['total'] + \
                0.1 * balance_loss + \
                env_pred_weight * env_pred_loss

        return {
            'total': total,
            'kl': kl_loss,
            'vicreg': vicreg_losses['total'],
            'balance': balance_loss,
            'env_pred': env_pred_loss,
            'mu_std': mu.std().item(),
        }

    def sample_from_prior(self, n_samples: int, device: torch.device = None) -> torch.Tensor:
        return self.gmm_prior.sample(n_samples, device)

    def get_soft_assignment(self, e: torch.Tensor) -> torch.Tensor:
        return self.gmm_prior.compute_soft_assignment(e)

    def get_component_usage(self, e: torch.Tensor) -> Dict[str, float]:
        assignment = self.gmm_prior.compute_soft_assignment(e)
        hard_assignment = assignment.argmax(dim=-1)

        usage = {}
        for k in range(self.n_components):
            count = (hard_assignment == k).sum().item()
            usage[f'component_{k}'] = count / len(hard_assignment)

        return usage


if __name__ == '__main__':
    print("=" * 60)
    print("  GMM Environment Module Test")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    env_module = GMMEnvironmentModule(
        env_dim=8,
        n_components=4,
        use_soft_swav=True,
    ).to(device)


    print("\n[GMM Prior]")
    gmm = env_module.gmm_prior
    print(f"  Centers shape: {gmm.centers.shape}")
    print(f"  Weights: {gmm.weights.detach().cpu().numpy()}")
    print(f"  Stds: {gmm.stds.detach().cpu().numpy()}")


    samples = gmm.sample(100, device)
    print(f"  Samples shape: {samples.shape}")
    print(f"  Samples mean: {samples.mean().item():.4f}")
    print(f"  Samples std: {samples.std().item():.4f}")


    log_p = gmm.log_prob(samples)
    print(f"  Log prob shape: {log_p.shape}")
    print(f"  Log prob mean: {log_p.mean().item():.4f}")


    assignment = gmm.compute_soft_assignment(samples)
    print(f"  Assignment shape: {assignment.shape}")
    print(f"  Assignment sum: {assignment.sum(dim=-1).mean().item():.4f}")


    mu = torch.randn(32, 8, device=device)
    logvar = torch.zeros(32, 8, device=device)
    kl = gmm.kl_divergence(mu, logvar)
    print(f"  KL divergence: {kl.item():.4f}")


    print("\n[VICReg]")
    z1 = torch.randn(32, 8, device=device)
    z2 = z1 + 0.1 * torch.randn_like(z1)
    vicreg_losses = env_module.vicreg(z1, z2)
    print(f"  Total: {vicreg_losses['total'].item():.4f}")
    print(f"  Sim: {vicreg_losses['sim'].item():.4f}")
    print(f"  Var: {vicreg_losses['var'].item():.4f}")
    print(f"  Cov: {vicreg_losses['cov'].item():.4f}")


    print("\n[Warmup Loss Simulation]")


    class DummyEncoder(nn.Module):
        def __init__(self, input_dim, env_dim):
            super().__init__()
            self.net = nn.Linear(input_dim, env_dim * 2)

        def forward(self, x, c):
            h = x.mean(dim=1)
            out = self.net(h)
            mu, logvar = out.chunk(2, dim=-1)
            return {'mu': mu, 'logvar': logvar}

    encoder = DummyEncoder(1, 8).to(device)

    x = torch.randn(32, 96, 1, device=device)
    c = torch.randn(32, 96, 7, device=device)

    enc_out = encoder(x, c)
    mu = enc_out['mu']

    warmup_losses = env_module.warmup_loss(
        mu, x, c,
        encoder_fn=lambda x, c: encoder(x, c),
    )

    print(f"  Total: {warmup_losses['total'].item():.4f}")
    print(f"  VICReg: {warmup_losses['vicreg'].item():.4f}")
    print(f"  SwAV: {warmup_losses['swav'].item():.4f}")
    print(f"  Ortho: {warmup_losses['ortho'].item():.4f}")
    print(f"  Balance: {warmup_losses['balance'].item():.4f}")
    print(f"  μ std: {warmup_losses['mu_std']:.4f}")

    print("\n✅ All tests passed!")

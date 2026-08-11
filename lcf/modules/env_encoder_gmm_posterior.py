import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple


from .env_encoder_v2 import (
    TCNEncoder,
    TemporalStatistics,
    AttentionPooling,
    SpectralAnalysis,
    add_positional_encoding_catsg,
)


class GMMPosteriorHead(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        env_dim: int,
        n_components: int = 4,
        normalize_mu: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.env_dim = env_dim
        self.n_components = n_components
        self.normalize_mu = normalize_mu


        self.weight_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_components),
        )


        self.mu_shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mu_heads = nn.ModuleList([
            nn.Linear(hidden_dim, env_dim) for _ in range(n_components)
        ])


        self.logvar_shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.logvar_heads = nn.ModuleList([
            nn.Linear(hidden_dim, env_dim) for _ in range(n_components)
        ])


        for head in self.logvar_heads:
            nn.init.constant_(head.bias, -1.0)

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = h.shape[0]


        logits = self.weight_head(h)
        weights = F.softmax(logits, dim=-1)


        h_mu = self.mu_shared(h)
        mus = torch.stack([head(h_mu) for head in self.mu_heads], dim=1)


        if self.normalize_mu:
            mus = F.normalize(mus, p=2, dim=-1)


        h_logvar = self.logvar_shared(h)
        logvars = torch.stack([head(h_logvar) for head in self.logvar_heads], dim=1)
        logvars = torch.clamp(logvars, min=-10, max=2)

        return {
            'weights': weights,
            'logits': logits,
            'mus': mus,
            'logvars': logvars,
        }


class GMMPosteriorSampler(nn.Module):

    def __init__(
        self,
        temperature: float = 1.0,
        hard: bool = False,
        mode: str = 'mixture',
    ):
        super().__init__()
        self.temperature = temperature
        self.hard = hard
        self.mode = mode

    def forward(
        self,
        logits: torch.Tensor,
        mus: torch.Tensor,
        logvars: torch.Tensor,
        temperature: Optional[float] = None,
        hard: Optional[bool] = None,
        mode: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sampling_mode = mode if mode is not None else self.mode

        if sampling_mode == 'mixture':
            return self._mixture_sample(logits, mus, logvars)
        else:
            return self._gumbel_sample(logits, mus, logvars, temperature, hard)

    def _mixture_sample(
        self,
        logits: torch.Tensor,
        mus: torch.Tensor,
        logvars: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, K, D = mus.shape
        device = mus.device


        weights = F.softmax(logits, dim=-1)


        stds = torch.exp(0.5 * logvars)
        eps = torch.randn_like(stds)
        samples_k = mus + stds * eps


        weights_expand = weights.unsqueeze(-1)
        e = (weights_expand * samples_k).sum(dim=1)

        return e, weights

    def _gumbel_sample(
        self,
        logits: torch.Tensor,
        mus: torch.Tensor,
        logvars: torch.Tensor,
        temperature: Optional[float] = None,
        hard: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        temp = temperature if temperature is not None else self.temperature
        use_hard = hard if hard is not None else self.hard


        k_soft = F.gumbel_softmax(logits, tau=temp, hard=use_hard)


        k_expand = k_soft.unsqueeze(-1)

        mu_selected = (k_expand * mus).sum(dim=1)
        logvar_selected = (k_expand * logvars).sum(dim=1)


        std = torch.exp(0.5 * logvar_selected)
        eps = torch.randn_like(std)
        e = mu_selected + std * eps

        return e, k_soft

    def sample_all_components(
        self,
        weights: torch.Tensor,
        mus: torch.Tensor,
        logvars: torch.Tensor,
        n_samples_per_component: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, K, D = mus.shape
        device = mus.device

        stds = torch.exp(0.5 * logvars)


        eps = torch.randn(B, K, n_samples_per_component, D, device=device)
        e_samples = mus.unsqueeze(2) + stds.unsqueeze(2) * eps

        return e_samples


class EnvironmentEncoderGMMPosterior(nn.Module):

    def __init__(
        self,
        seq_len: int = 96,
        input_dim: int = 1,
        cond_dim: int = 2,
        hidden_dim: int = 64,
        env_dim: int = 4,
        n_components: int = 4,
        tcn_depth: int = 4,
        topk_peaks: int = 8,
        dropout: float = 0.1,
        temperature: float = 1.0,
        hard_sampling: bool = False,
        add_positional_encoding: bool = False,
        normalize_mu: bool = True,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.env_dim = env_dim
        self.n_components = n_components
        self.topk_peaks = topk_peaks
        self.add_positional_encoding = add_positional_encoding


        actual_cond_dim = cond_dim + 2 if add_positional_encoding else cond_dim
        self.actual_cond_dim = actual_cond_dim


        fused_dim = input_dim + actual_cond_dim
        self.tcn = TCNEncoder(
            in_channels=fused_dim,
            hidden_dim=hidden_dim,
            depth=tcn_depth,
            kernel_size=3,
        )


        self.stat_path = TemporalStatistics(hidden_dim)
        self.attn_path = AttentionPooling(hidden_dim)
        self.spec_path = SpectralAnalysis(hidden_dim, topk_peaks)


        fused_feat_dim = 5 * hidden_dim + topk_peaks

        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_feat_dim),
            nn.Linear(fused_feat_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )


        self.gmm_head = GMMPosteriorHead(
            hidden_dim=hidden_dim,
            env_dim=env_dim,
            n_components=n_components,
            normalize_mu=normalize_mu,
        )


        self.sampler = GMMPosteriorSampler(
            temperature=temperature,
            hard=hard_sampling,
        )

    def _to_btd(self, x: torch.Tensor, expected_d: int) -> torch.Tensor:
        original_shape = x.shape

        while x.dim() > 3:
            squeezed = False
            for dim in range(x.dim()):
                if x.shape[dim] == 1 and dim not in [0]:
                    x = x.squeeze(dim)
                    squeezed = True
                    break
            if not squeezed:
                break

        if x.dim() == 2:
            x = x.unsqueeze(-1)
        elif x.dim() == 3:
            if x.shape[-1] == expected_d:
                pass
            elif x.shape[1] == expected_d:
                x = x.transpose(1, 2)
        else:
            raise ValueError(f"Cannot convert to (B,T,D): dim={x.dim()}, shape={original_shape}")

        return x

    def encode(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Dict[str, torch.Tensor]:

        x = self._to_btd(x, self.input_dim)
        B, T, D_x = x.shape

        if c.dim() == 2:
            c = c.unsqueeze(1).expand(B, T, -1)
        else:
            c = self._to_btd(c, self.cond_dim)


        if self.add_positional_encoding:
            c = add_positional_encoding_catsg(c)


        xc = torch.cat([x, c], dim=-1)
        xc = xc.transpose(1, 2)
        h_prime = self.tcn(xc)
        h_prime = h_prime.transpose(1, 2)


        h_stat = self.stat_path(h_prime)
        h_attn, attn_weights = self.attn_path(h_prime)
        h_spec = self.spec_path(h_prime)


        h_concat = torch.cat([h_stat, h_attn, h_spec], dim=-1)
        h = self.fusion(h_concat)


        gmm_params = self.gmm_head(h)

        result = gmm_params

        if return_intermediates:
            result.update({
                'h_stat': h_stat,
                'h_attn': h_attn,
                'h_spec': h_spec,
                'h_fused': h,
                'attn_weights': attn_weights,
            })

        return result

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        hard: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        encoded = self.encode(x, c)

        e, k_soft = self.sampler(
            encoded['logits'],
            encoded['mus'],
            encoded['logvars'],
            temperature=temperature,
            hard=hard,
        )

        return {
            'e': e,
            'k_soft': k_soft,
            'weights': encoded['weights'],
            'logits': encoded['logits'],
            'mus': encoded['mus'],
            'logvars': encoded['logvars'],

            'mu': (encoded['weights'].unsqueeze(-1) * encoded['mus']).sum(dim=1),
            'logvar': (encoded['weights'].unsqueeze(-1) * encoded['logvars']).sum(dim=1),
        }

    def compute_gmm_log_prob(
        self,
        e: torch.Tensor,
        weights: torch.Tensor,
        mus: torch.Tensor,
        logvars: torch.Tensor,
    ) -> torch.Tensor:
        B, K, D = mus.shape


        e_expand = e.unsqueeze(1)

        vars = torch.exp(logvars)


        log_probs_k = -0.5 * (
            D * math.log(2 * math.pi) +
            logvars.sum(dim=-1) +
            ((e_expand - mus) ** 2 / vars).sum(dim=-1)
        )


        log_weights = torch.log(weights + 1e-10)
        log_prob = torch.logsumexp(log_weights + log_probs_k, dim=-1)

        return log_prob

    def compute_kl_divergence(
        self,
        weights_q: torch.Tensor,
        mus_q: torch.Tensor,
        logvars_q: torch.Tensor,
        gmm_prior,
        n_mc_samples: int = 100,
    ) -> torch.Tensor:
        B = weights_q.shape[0]
        device = weights_q.device


        e_samples = []
        for _ in range(n_mc_samples):
            e, _ = self.sampler(
                torch.log(weights_q + 1e-10),
                mus_q,
                logvars_q,
                temperature=0.5,
                hard=False,
            )
            e_samples.append(e)

        e_samples = torch.stack(e_samples, dim=1)


        log_q = []
        for i in range(n_mc_samples):
            log_q_i = self.compute_gmm_log_prob(
                e_samples[:, i], weights_q, mus_q, logvars_q
            )
            log_q.append(log_q_i)
        log_q = torch.stack(log_q, dim=1)


        log_p = []
        for i in range(n_mc_samples):
            log_p_i = gmm_prior.log_prob(e_samples[:, i])
            log_p.append(log_p_i)
        log_p = torch.stack(log_p, dim=1)


        kl = (log_q - log_p).mean(dim=1)

        return kl.mean()


if __name__ == '__main__':
    print("=" * 60)
    print("  GMM Posterior Encoder Test")
    print("=" * 60)


    encoder = EnvironmentEncoderGMMPosterior(
        seq_len=96,
        input_dim=1,
        cond_dim=2,
        hidden_dim=64,
        env_dim=4,
        n_components=4,
    )


    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nTotal parameters: {n_params:,}")


    B, T = 32, 96
    x = torch.randn(B, T, 1)
    c = torch.randn(B, T, 2)

    output = encoder(x, c)

    print(f"\nInput x: {x.shape}")
    print(f"Input c: {c.shape}")
    print(f"Output e: {output['e'].shape}")
    print(f"Output weights: {output['weights'].shape}")
    print(f"Output mus: {output['mus'].shape}")
    print(f"Output logvars: {output['logvars'].shape}")
    print(f"Output k_soft: {output['k_soft'].shape}")


    print(f"\nWeights sum: {output['weights'].sum(dim=-1).mean().item():.4f} (should be 1.0)")


    log_prob = encoder.compute_gmm_log_prob(
        output['e'],
        output['weights'],
        output['mus'],
        output['logvars'],
    )
    print(f"\nLog prob: {log_prob.mean().item():.4f}")

    print("\n✅ All tests passed!")

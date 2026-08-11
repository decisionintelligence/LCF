import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple


class SamePadConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int = 1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, :-self.remove]
        return out


class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int, final: bool = False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation)
        self.projector = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels or final else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class TCNEncoder(nn.Module):

    def __init__(self, in_channels: int, hidden_dim: int,
                 depth: int = 4, kernel_size: int = 3):
        super().__init__()


        layers = []
        for i in range(depth):
            dilation = 2 ** i
            in_ch = in_channels if i == 0 else hidden_dim
            out_ch = hidden_dim
            layers.append(ConvBlock(in_ch, out_ch, kernel_size, dilation, final=(i == depth - 1)))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalStatistics(nn.Module):

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_mean = h.mean(dim=1)
        h_std = h.std(dim=1) + 1e-6
        h_max = h.max(dim=1).values

        return torch.cat([h_mean, h_std, h_max], dim=-1)


class AttentionPooling(nn.Module):

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score_fn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        scores = self.score_fn(h)
        weights = F.softmax(scores, dim=1)


        h_attn = (h * weights).sum(dim=1)

        return h_attn, weights


class SpectralAnalysis(nn.Module):

    def __init__(self, hidden_dim: int, topk_peaks: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.topk_peaks = topk_peaks

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, T, H = h.shape
        device = h.device


        fft = torch.fft.rfft(h, dim=1)
        psd = fft.real ** 2 + fft.imag ** 2

        n_freqs = psd.shape[1]


        freq = torch.linspace(0, 1, n_freqs, device=device).view(1, -1, 1)
        psd_sum = psd.sum(dim=1, keepdim=True) + 1e-8
        centroid = (psd * freq).sum(dim=1) / psd_sum.squeeze(1)


        psd_mean = psd.mean(dim=2)


        k = min(self.topk_peaks, n_freqs)
        topk_vals, _ = torch.topk(psd_mean, k=k, dim=-1)


        topk_vals = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-8)


        if k < self.topk_peaks:
            padding = torch.zeros(B, self.topk_peaks - k, device=device)
            topk_vals = torch.cat([topk_vals, padding], dim=-1)

        return torch.cat([centroid, topk_vals], dim=-1)


def add_positional_encoding_catsg(c: torch.Tensor) -> torch.Tensor:
    B, T, D = c.shape
    device = c.device
    dtype = c.dtype


    t = torch.arange(T, device=device, dtype=dtype)
    phi = (2.0 * math.pi) * t / float(T)
    pos_enc = torch.stack([torch.sin(phi), torch.cos(phi)], dim=-1)
    pos_enc = pos_enc.unsqueeze(0).expand(B, T, 2)

    return torch.cat([c, pos_enc], dim=-1)


class EnvironmentEncoderV2(nn.Module):

    def __init__(
        self,
        seq_len: int = 96,
        input_dim: int = 1,
        cond_dim: int = 2,
        hidden_dim: int = 64,
        env_dim: int = 4,
        tcn_depth: int = 4,
        topk_peaks: int = 8,
        dropout: float = 0.1,
        add_positional_encoding: bool = False,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.env_dim = env_dim
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


        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, env_dim),

        )

        self.logvar_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, env_dim),
        )


        nn.init.constant_(self.logvar_head[-1].bias, -1.0)


        self.normalize_mu = True


        self.flow_prior = None

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

        x_orig_shape = x.shape
        x = self._to_btd(x, self.input_dim)
        B, T, D_x = x.shape

        c_orig_shape = c.shape
        if c.dim() == 2:
            c = c.unsqueeze(1).expand(B, T, -1)
        else:
            c = self._to_btd(c, self.cond_dim)


        if x.shape[1] != c.shape[1]:
            raise ValueError(
                f"T dimension mismatch: x.shape={x.shape} (orig={x_orig_shape}), "
                f"c.shape={c.shape} (orig={c_orig_shape})"
            )


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


        mu = self.mu_head(h)


        if self.normalize_mu:
            mu = F.normalize(mu, p=2, dim=-1)

        logvar = self.logvar_head(h)
        logvar = torch.clamp(logvar, min=-10, max=2)

        result = {'mu': mu, 'logvar': logvar}

        if return_intermediates:
            result.update({
                'h_stat': h_stat,
                'h_attn': h_attn,
                'h_spec': h_spec,
                'h_fused': h,
                'attn_weights': attn_weights,
            })

        return result

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)

        if num_samples == 1:
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            B, D = mu.shape
            eps = torch.randn(B, num_samples, D, device=mu.device, dtype=mu.dtype)
            return mu.unsqueeze(1) + std.unsqueeze(1) * eps

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        num_samples: int = 1,
    ) -> Dict[str, torch.Tensor]:
        encoded = self.encode(x, c)
        mu, logvar = encoded['mu'], encoded['logvar']
        e = self.reparameterize(mu, logvar, num_samples)

        return {
            'e': e,
            'mu': mu,
            'logvar': logvar,
        }

    def compute_kl_divergence(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        e_samples: Optional[torch.Tensor] = None,
        free_bits: float = 0.0,
    ) -> torch.Tensor:
        if self.flow_prior is not None and e_samples is not None:


            log_q = self._gaussian_log_prob(e_samples, mu, logvar)
            log_p = self.flow_prior.log_prob(e_samples)
            kl_per_sample = log_q - log_p
            return kl_per_sample.mean()
        else:

            kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            if free_bits > 0:
                kl_per_dim = F.relu(kl_per_dim - free_bits) + free_bits
            return kl_per_dim.sum(dim=-1).mean()

    def _gaussian_log_prob(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        var = torch.exp(logvar)
        log_prob = -0.5 * (
            torch.log(2 * torch.pi * var) +
            (x - mu).pow(2) / var
        ).sum(dim=-1)
        return log_prob


class AffineCoupling(nn.Module):

    def __init__(self, dim: int, hidden_dim: int = 64, mask_type: str = 'even'):
        super().__init__()
        self.dim = dim
        self.mask_type = mask_type


        if mask_type == 'even':
            self.register_buffer('mask', torch.arange(dim) % 2 == 0)
        else:
            self.register_buffer('mask', torch.arange(dim) % 2 == 1)


        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim * 2),
        )


        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_masked = x * self.mask.float()

        params = self.net(x_masked)
        scale, shift = params.chunk(2, dim=-1)
        scale = torch.tanh(scale) * 2


        e = x_masked + (~self.mask).float() * (x * torch.exp(scale) + shift)


        log_det = (scale * (~self.mask).float()).sum(dim=-1)

        return e, log_det

    def inverse(self, e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        e_masked = e * self.mask.float()

        params = self.net(e_masked)
        scale, shift = params.chunk(2, dim=-1)
        scale = torch.tanh(scale) * 2


        x = e_masked + (~self.mask).float() * ((e - shift) * torch.exp(-scale))


        log_det = -(scale * (~self.mask).float()).sum(dim=-1)

        return x, log_det


class RealNVPPrior(nn.Module):

    def __init__(self, dim: int, n_blocks: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.dim = dim


        self.blocks = nn.ModuleList([
            AffineCoupling(dim, hidden_dim, mask_type='even' if i % 2 == 0 else 'odd')
            for i in range(n_blocks)
        ])

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        e = z
        log_det = 0

        for block in self.blocks:
            e, ld = block.forward(e)
            log_det = log_det + ld

        return e, log_det

    def inverse(self, e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = e
        log_det = 0

        for block in reversed(self.blocks):
            z, ld = block.inverse(z)
            log_det = log_det + ld

        return z, log_det

    def log_prob(self, e: torch.Tensor) -> torch.Tensor:
        z, log_det = self.inverse(e)


        log_pz = -0.5 * (z.pow(2) + math.log(2 * math.pi)).sum(dim=-1)

        return log_pz + log_det

    def sample(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device

        z = torch.randn(batch_size, self.dim, device=device)
        e, _ = self.forward(z)
        return e


def create_encoder_v2(
    seq_len: int = 96,
    input_dim: int = 1,
    cond_dim: int = 2,
    hidden_dim: int = 64,
    env_dim: int = 4,
    use_flow_prior: bool = False,
    flow_blocks: int = 4,
) -> EnvironmentEncoderV2:
    encoder = EnvironmentEncoderV2(
        seq_len=seq_len,
        input_dim=input_dim,
        cond_dim=cond_dim,
        hidden_dim=hidden_dim,
        env_dim=env_dim,
    )

    if use_flow_prior:
        encoder.flow_prior = RealNVPPrior(
            dim=env_dim,
            n_blocks=flow_blocks,
            hidden_dim=hidden_dim,
        )

    return encoder


if __name__ == '__main__':
    print("=" * 60)
    print("  Environment Encoder V2 Test")
    print("=" * 60)


    encoder = create_encoder_v2(
        seq_len=96,
        input_dim=1,
        cond_dim=2,
        hidden_dim=64,
        env_dim=4,
        use_flow_prior=True,
    )


    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nTotal parameters: {n_params:,}")


    B, T = 32, 96
    x = torch.randn(B, T, 1)
    c = torch.randn(B, T, 2)

    output = encoder(x, c, num_samples=1)
    print(f"\nInput x: {x.shape}")
    print(f"Input c: {c.shape}")
    print(f"Output e: {output['e'].shape}")
    print(f"Output μ: {output['mu'].shape}")
    print(f"Output logvar: {output['logvar'].shape}")


    encoded = encoder.encode(x, c, return_intermediates=True)
    print(f"\nIntermediate features:")
    print(f"  h_stat: {encoded['h_stat'].shape}")
    print(f"  h_attn: {encoded['h_attn'].shape}")
    print(f"  h_spec: {encoded['h_spec'].shape}")
    print(f"  h_fused: {encoded['h_fused'].shape}")


    kl_gaussian = encoder.compute_kl_divergence(
        output['mu'], output['logvar']
    )
    print(f"\nKL (Gaussian prior): {kl_gaussian.item():.4f}")

    kl_flow = encoder.compute_kl_divergence(
        output['mu'], output['logvar'], output['e']
    )
    print(f"KL (Flow prior): {kl_flow.item():.4f}")


    if encoder.flow_prior is not None:
        samples = encoder.flow_prior.sample(100)
        print(f"\nFlow prior samples: {samples.shape}")
        print(f"  mean: {samples.mean().item():.4f}")
        print(f"  std: {samples.std().item():.4f}")

    print("\n✅ All tests passed!")

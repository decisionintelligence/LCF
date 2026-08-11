import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from lcf.modules.env_encoder_v2 import EnvironmentEncoderV2
from lcf.modules.velocity_net import VelocityNetwork
from lcf.modules.gmm_environment import GMMEnvironmentModule


class GMMBasedLCF(nn.Module):

    def __init__(
        self,
        encoder: nn.Module,
        velocity_net: nn.Module,
        env_module: GMMEnvironmentModule,
    ):
        super().__init__()
        self.encoder = encoder
        self.velocity_net = velocity_net
        self.env_module = env_module


        self.training_step_count = 0

    def encode(self, x: torch.Tensor, c: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.encoder(x, c)


    def warmup_step(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        ortho_weight: float = 0.1,
        balance_weight: float = 0.5,
        swav_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:

        for param in self.velocity_net.parameters():
            param.requires_grad = False


        enc_out = self.encoder(x, c)
        mu = enc_out['mu']


        losses = self.env_module.warmup_loss(
            mu, x, c,
            encoder_fn=lambda x, c: self.encoder(x, c),
            ortho_weight=ortho_weight,
            balance_weight=balance_weight,
            swav_weight=swav_weight,
        )


        for param in self.velocity_net.parameters():
            param.requires_grad = True

        return losses

    def training_step(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        kl_weight: float = 0.01,
        var_weight: float = 0.5,
        env_pred_weight: float = 0.1,
        free_bits: float = 0.1,
        env_recon_weight: float = 0.5,
        check_sensitivity: bool = False,
        zero_env: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B = x.shape[0]
        device = x.device


        t = torch.rand(B, device=device)
        noise = torch.randn_like(x)


        t_expand = t.view(-1, 1, 1)
        x_t = t_expand * x + (1 - t_expand) * noise


        v_target = x - noise


        enc_out = self.encoder(x_t, c)
        mu, logvar = enc_out['mu'], enc_out['logvar']


        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        e = mu + std * eps


        e_for_vel = torch.zeros_like(e) if zero_env else e


        v_output = self.velocity_net(x_t, t, c, e_for_vel)


        if isinstance(v_output, tuple):
            v_pred, e_recon = v_output
        else:
            v_pred = v_output
            e_recon = None


        fm_loss = F.mse_loss(v_pred, v_target)


        env_recon_loss = torch.tensor(0.0, device=device)
        if e_recon is not None and env_recon_weight > 0:
            env_recon_loss = F.mse_loss(e_recon, e.detach())


        env_losses = self.env_module.training_loss(
            mu, logvar,
            x=x,
            kl_weight=kl_weight,
            var_weight=var_weight,
            env_pred_weight=env_pred_weight,
            free_bits=free_bits,
        )


        sensitivity_info = None
        if check_sensitivity and hasattr(self.env_module, 'sensitivity_monitor'):
            sensitivity_info = self.env_module.sensitivity_monitor.check_sensitivity(
                self.velocity_net, x_t[:8], t[:8], c[:8], e[:8]
            )


        total_loss = fm_loss + env_losses['total'] + env_recon_weight * env_recon_loss


        self.training_step_count += 1

        result = {
            'total': total_loss,
            'fm': fm_loss,
            'env': env_losses['total'],
            'kl': env_losses['kl'],
            'vicreg': env_losses['vicreg'],
            'balance': env_losses['balance'],
            'env_pred': env_losses['env_pred'],
            'env_recon': env_recon_loss,
            'mu_std': env_losses['mu_std'],
        }

        if sensitivity_info is not None:
            result['e_sensitivity'] = sensitivity_info['avg_sensitivity']
            if sensitivity_info['warning']:
                print(f"    {sensitivity_info['warning']}")

        return result


    @torch.no_grad()
    def generate(
        self,
        c: torch.Tensor,
        n_steps: int = 50,
        use_prior: bool = True,
        x_ref: Optional[torch.Tensor] = None,
        dynamic_env: bool = False,
        update_interval: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = c.shape
        D = 1
        device = c.device


        if use_prior:
            e = self.env_module.sample_from_prior(B, device)
        else:
            if x_ref is None:
                raise ValueError("x_ref required when use_prior=False")
            enc_out = self.encoder(x_ref, c)
            e = enc_out['mu'] + torch.exp(0.5 * enc_out['logvar']) * torch.randn_like(enc_out['mu'])


        x = torch.randn(B, T, D, device=device)
        dt = 1.0 / n_steps

        for i in range(n_steps):
            t_val = i * dt
            t = torch.full((B,), t_val, device=device)


            if dynamic_env and i > 0 and i % update_interval == 0:
                enc_out = self.encoder(x, c)
                e = enc_out['mu'] + torch.exp(0.5 * enc_out['logvar']) * torch.randn_like(enc_out['mu'])

            v = self.velocity_net(x, t, c, e)
            x = x + v * dt

        return x, e

    @torch.no_grad()
    def generate_causal_mc(
        self,
        c: torch.Tensor,
        n_steps: int = 50,
        n_mc_samples: int = 10,
        use_posterior: bool = True,
    ) -> torch.Tensor:
        B, T, C = c.shape
        D = 1
        device = c.device


        x = torch.randn(B, T, D, device=device)
        dt = 1.0 / n_steps

        for i in range(n_steps):
            t_val = i * dt
            t = torch.full((B,), t_val, device=device)

            if use_posterior:


                enc_out = self.encoder(x, c)
                mu, logvar = enc_out['mu'], enc_out['logvar']
                std = torch.exp(0.5 * logvar)


                v_samples = []
                for _ in range(n_mc_samples):
                    eps = torch.randn_like(mu)
                    e_k = mu + std * eps
                    v_k = self.velocity_net(x, t, c, e_k)
                    v_samples.append(v_k)


                v_do = torch.stack(v_samples).mean(dim=0)
            else:

                v_samples = []
                for _ in range(n_mc_samples):
                    e_k = self.env_module.sample_from_prior(B, device)
                    v_k = self.velocity_net(x, t, c, e_k)
                    v_samples.append(v_k)

                v_do = torch.stack(v_samples).mean(dim=0)


            x = x + v_do * dt

        return x

    @torch.no_grad()
    def generate_causal_gmm(
        self,
        c: torch.Tensor,
        n_steps: int = 50,
    ) -> torch.Tensor:
        B, T, C = c.shape
        D = 1
        device = c.device
        K = self.env_module.n_components


        gmm_centers = self.env_module.gmm_prior.centers

        x = torch.randn(B, T, D, device=device)
        dt = 1.0 / n_steps

        for i in range(n_steps):
            t_val = i * dt
            t = torch.full((B,), t_val, device=device)


            enc_out = self.encoder(x, c)
            mu = enc_out['mu']


            assignment = self.env_module.gmm_prior.compute_soft_assignment(mu)


            v_components = []
            for k in range(K):
                e_k = gmm_centers[k].unsqueeze(0).expand(B, -1)
                v_k = self.velocity_net(x, t, c, e_k)
                v_components.append(v_k)

            v_stack = torch.stack(v_components, dim=1)


            weights = assignment.view(B, K, 1, 1)
            v_do = (weights * v_stack).sum(dim=1)

            x = x + v_do * dt

        return x

    @torch.no_grad()
    def generate_gmm_enhanced(
        self,
        c: torch.Tensor,
        n_steps: int = 50,
    ) -> torch.Tensor:
        B, T, C = c.shape
        D = 1
        device = c.device

        gmm_centers = self.env_module.gmm_prior.centers

        x = torch.randn(B, T, D, device=device)
        dt = 1.0 / n_steps

        for i in range(n_steps):
            t_val = i * dt
            t = torch.full((B,), t_val, device=device)


            enc_out = self.encoder(x, c)
            mu = enc_out['mu']


            assignment = self.env_module.gmm_prior.compute_soft_assignment(mu)


            e_enhanced = torch.einsum('bk,kd->bd', assignment, gmm_centers)


            v = self.velocity_net(x, t, c, e_enhanced)

            x = x + v * dt

        return x

    @torch.no_grad()
    def generate_hybrid(
        self,
        c: torch.Tensor,
        n_steps: int = 50,
        switch_ratio: float = 0.7,
        warmup_mode: str = 'fast',
    ) -> Tuple[torch.Tensor, Dict]:
        B, T, C = c.shape
        D = 1
        device = c.device
        K = self.env_module.n_components

        gmm_centers = self.env_module.gmm_prior.centers

        x = torch.randn(B, T, D, device=device)
        dt = 1.0 / n_steps


        fast_steps = 0
        causal_steps = 0
        assignment_history = []

        switch_step = int(n_steps * switch_ratio)

        for i in range(n_steps):
            t_val = i * dt
            t = torch.full((B,), t_val, device=device)


            enc_out = self.encoder(x, c)
            mu = enc_out['mu']


            assignment = self.env_module.gmm_prior.compute_soft_assignment(mu)
            assignment_history.append(assignment.detach().cpu())

            if i < switch_step:

                fast_steps += 1

                if warmup_mode == 'fast':

                    e = torch.einsum('bk,kd->bd', assignment, gmm_centers)
                else:

                    e = mu

                v = self.velocity_net(x, t, c, e)
            else:

                causal_steps += 1


                v_components = []
                for k in range(K):
                    e_k = gmm_centers[k].unsqueeze(0).expand(B, -1)
                    v_k = self.velocity_net(x, t, c, e_k)
                    v_components.append(v_k)

                v_stack = torch.stack(v_components, dim=1)
                weights = assignment.view(B, K, 1, 1)
                v = (weights * v_stack).sum(dim=1)

            x = x + v * dt

        info = {
            'fast_steps': fast_steps,
            'causal_steps': causal_steps,
            'switch_ratio': switch_ratio,
            'switch_step': switch_step,
            'assignment_history': torch.stack(assignment_history),
        }

        return x, info

    @torch.no_grad()
    def generate_adaptive_hybrid(
        self,
        c: torch.Tensor,
        n_steps: int = 50,
        entropy_threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, Dict]:
        B, T, C = c.shape
        D = 1
        device = c.device
        K = self.env_module.n_components
        max_entropy = torch.log(torch.tensor(K, dtype=torch.float32))

        gmm_centers = self.env_module.gmm_prior.centers

        x = torch.randn(B, T, D, device=device)
        dt = 1.0 / n_steps


        fast_steps = 0
        causal_steps = 0
        entropy_history = []
        mode_history = []

        for i in range(n_steps):
            t_val = i * dt
            t = torch.full((B,), t_val, device=device)

            enc_out = self.encoder(x, c)
            mu = enc_out['mu']
            assignment = self.env_module.gmm_prior.compute_soft_assignment(mu)


            entropy = -(assignment * torch.log(assignment + 1e-8)).sum(dim=-1)
            normalized_entropy = (entropy / max_entropy).mean().item()
            entropy_history.append(normalized_entropy)

            if normalized_entropy > entropy_threshold:

                causal_steps += 1
                mode_history.append('causal')

                v_components = []
                for k in range(K):
                    e_k = gmm_centers[k].unsqueeze(0).expand(B, -1)
                    v_k = self.velocity_net(x, t, c, e_k)
                    v_components.append(v_k)

                v_stack = torch.stack(v_components, dim=1)
                weights = assignment.view(B, K, 1, 1)
                v = (weights * v_stack).sum(dim=1)
            else:

                fast_steps += 1
                mode_history.append('fast')

                e = torch.einsum('bk,kd->bd', assignment, gmm_centers)
                v = self.velocity_net(x, t, c, e)

            x = x + v * dt

        info = {
            'fast_steps': fast_steps,
            'causal_steps': causal_steps,
            'entropy_threshold': entropy_threshold,
            'entropy_history': entropy_history,
            'mode_history': mode_history,
        }

        return x, info


    def get_gmm_centers(self) -> torch.Tensor:
        return self.env_module.gmm_prior.centers

    def get_gmm_assignment(self, mu: torch.Tensor) -> torch.Tensor:
        return self.env_module.gmm_prior.compute_soft_assignment(mu)

    def sample_from_prior(self, n_samples: int, device: torch.device) -> torch.Tensor:
        return self.env_module.sample_from_prior(n_samples, device)


    @torch.no_grad()
    def ode_forward(
        self,
        x_T: torch.Tensor,
        c: torch.Tensor,
        e: torch.Tensor,
        n_steps: int = 100,
    ) -> torch.Tensor:
        x = x_T.clone()
        dt = 1.0 / n_steps
        B = x.shape[0]
        device = x.device

        for i in range(n_steps):
            t_val = i * dt
            t = torch.full((B,), t_val, device=device)
            v = self.velocity_net(x, t, c, e)
            x = x + v * dt

        return x

    @torch.no_grad()
    def ode_backward(
        self,
        x_0: torch.Tensor,
        c: torch.Tensor,
        e: torch.Tensor,
        n_steps: int = 100,
    ) -> torch.Tensor:
        x = x_0.clone()
        dt = 1.0 / n_steps
        B = x.shape[0]
        device = x.device


        for i in range(n_steps):
            t_val = 1.0 - i * dt
            t = torch.full((B,), t_val, device=device)
            v = self.velocity_net(x, t, c, e)
            x = x - v * dt

        return x

    @torch.no_grad()
    def generate_weak_counterfactual(
        self,
        x_obs: torch.Tensor,
        c_obs: torch.Tensor,
        c_cf: torch.Tensor,
        n_steps: int = 100,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        enc_out = self.encoder(x_obs, c_obs)
        e = enc_out['mu']


        B, T, D = x_obs.shape
        x_T = torch.randn(B, T, D, device=x_obs.device)
        x_cf = self.ode_forward(x_T, c_cf, e, n_steps)

        return x_cf, e

    @torch.no_grad()
    def generate_strong_counterfactual(
        self,
        x_obs: torch.Tensor,
        c_obs: torch.Tensor,
        c_cf: torch.Tensor,
        n_steps: int = 100,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        enc_out = self.encoder(x_obs, c_obs)
        e = enc_out['mu']


        x_T = self.ode_backward(x_obs, c_obs, e, n_steps)


        x_cf = self.ode_forward(x_T, c_cf, e, n_steps)

        return x_cf, e, x_T

    @torch.no_grad()
    def verify_inversion(
        self,
        x_obs: torch.Tensor,
        c: torch.Tensor,
        n_steps: int = 100,
    ) -> Dict[str, torch.Tensor]:

        enc_out = self.encoder(x_obs, c)
        e = enc_out['mu']


        x_T = self.ode_backward(x_obs, c, e, n_steps)
        x_recon = self.ode_forward(x_T, c, e, n_steps)


        mse = ((x_obs - x_recon) ** 2).mean()
        mae = (x_obs - x_recon).abs().mean()
        max_err = (x_obs - x_recon).abs().max()

        return {
            'x_recon': x_recon,
            'x_T': x_T,
            'mse': mse,
            'mae': mae,
            'max_error': max_err,
        }


def create_gmm_lcf(
    x_dim: int = 1,
    c_dim: int = 7,
    env_dim: int = 8,
    hidden_dim: int = 128,
    n_components: int = 4,
    num_layers: int = 4,
    seq_len: int = 96,
    cond_dropout: float = 0.2,
    use_curve_prior: bool = True,
    curve_hidden_dim: int = 64,
    use_soft_swav: bool = True,
    swav_temperature: float = 0.1,
) -> GMMBasedLCF:
    import math


    encoder = EnvironmentEncoderV2(
        seq_len=seq_len,
        input_dim=x_dim,
        cond_dim=c_dim,
        env_dim=env_dim,
        hidden_dim=hidden_dim,
    )


    velocity_net = VelocityNetwork(
        seq_len=seq_len,
        input_dim=x_dim,
        cond_dim=c_dim,
        env_dim=env_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        cond_dropout=cond_dropout,
    )


    var_target = 1.0 / math.sqrt(env_dim)

    env_module = GMMEnvironmentModule(
        env_dim=env_dim,
        n_components=n_components,
        vicreg_config={
            'sim_weight': 25.0,
            'var_weight': 25.0,
            'cov_weight': 1.0,
            'var_target': var_target,
        },
        use_soft_swav=use_soft_swav,
        swav_temperature=swav_temperature,
        use_curve_prior=use_curve_prior,
        curve_hidden_dim=curve_hidden_dim,
    )


    model = GMMBasedLCF(encoder, velocity_net, env_module)

    return model

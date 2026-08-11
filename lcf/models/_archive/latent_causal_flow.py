import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from typing import Dict, Optional, Tuple, List
from torch.utils.data import DataLoader
from contextlib import contextmanager
from tqdm import tqdm


class EMA:

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._initialized = False

    def initialize(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        self._initialized = True

    def update(self):
        if not self._initialized:
            self.initialize()
            return

        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                device = param.device
                if self.shadow[name].device != device:
                    self.shadow[name] = self.shadow[name].to(device)
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self):
        if not self._initialized:
            return
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                device = param.device
                if self.shadow[name].device != device:
                    self.shadow[name] = self.shadow[name].to(device)
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


class LatentCausalFlow(pl.LightningModule):

    def __init__(
        self,

        seq_len: int = 96,
        channels: int = 1,
        cond_channels: int = 4,
        env_dim: int = 16,
        hid_dim: int = 128,


        num_mc_samples_train: int = 1,
        num_mc_samples_eval: int = 10,


        sigma_min: float = 1e-4,


        kl_weight: float = 0.01,
        kl_annealing: bool = True,
        kl_warmup_steps: int = 2000,
        free_bits: float = 0.25,


        consistency_weight: float = 0.1,


        c_dropout_rate: float = 0.3,
        c_dropout_schedule: str = "constant",


        stage1_epochs: int = 30,
        stage1_c_dropout: float = 0.5,
        stage2_c_dropout: float = 0.1,


        orth_weight: float = 0.1,


        cfg_scale: float = 1.5,


        env_encoder_config: Optional[Dict] = None,
        velocity_net_config: Optional[Dict] = None,


        use_ema: bool = True,
        ema_decay: float = 0.9999,


        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,

        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()


        self.seq_len = seq_len
        self.channels = channels
        self.cond_channels = cond_channels
        self.env_dim = env_dim
        self.hid_dim = hid_dim


        self.num_mc_samples_train = num_mc_samples_train
        self.num_mc_samples_eval = num_mc_samples_eval


        self.sigma_min = sigma_min


        self.kl_weight = kl_weight
        self.kl_annealing = kl_annealing
        self.kl_warmup_steps = kl_warmup_steps
        self.free_bits = free_bits


        self.consistency_weight = consistency_weight


        self.c_dropout_rate = c_dropout_rate
        self.c_dropout_schedule = c_dropout_schedule
        self.stage1_epochs = stage1_epochs
        self.stage1_c_dropout = stage1_c_dropout
        self.stage2_c_dropout = stage2_c_dropout
        self._current_c_dropout = c_dropout_rate


        self.orth_weight = orth_weight


        self.cfg_scale = cfg_scale


        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_ema = use_ema
        self.ema_decay = ema_decay


        self._build_networks(env_encoder_config, velocity_net_config)


        self.register_buffer(
            'null_cond',
            torch.zeros(1, seq_len, cond_channels)
        )
        self.register_buffer(
            'null_env',
            torch.zeros(1, env_dim)
        )


        self.ema = EMA(self, decay=ema_decay) if use_ema else None

    def _build_networks(self, env_encoder_config, velocity_net_config):
        from lcf.modules.env_encoder import EnvironmentEncoder
        from lcf.modules.velocity_net import VelocityNetwork


        if env_encoder_config is None:
            self.env_encoder = EnvironmentEncoder(
                seq_len=self.seq_len,
                input_dim=self.channels,
                cond_dim=self.cond_channels,
                hidden_dim=self.hid_dim,
                env_dim=self.env_dim,
            )
        else:
            from lcf.utils.util import instantiate_from_config
            self.env_encoder = instantiate_from_config(env_encoder_config)


        if velocity_net_config is None:
            self.velocity_net = VelocityNetwork(
                seq_len=self.seq_len,
                input_dim=self.channels,
                cond_dim=self.cond_channels,
                env_dim=self.env_dim,
                hidden_dim=self.hid_dim,
            )
        else:
            from lcf.utils.util import instantiate_from_config
            self.velocity_net = instantiate_from_config(velocity_net_config)


        enc_params = sum(p.numel() for p in self.env_encoder.parameters())
        vel_params = sum(p.numel() for p in self.velocity_net.parameters())
        print(f"[LCF] Environment Encoder: {enc_params:,} params")
        print(f"[LCF] Velocity Network: {vel_params:,} params")


    def _to_btd(self, x: torch.Tensor, expected_last_dim: int) -> torch.Tensor:
        if x.dim() == 2:
            return x.unsqueeze(-1)
        if x.dim() == 3:
            if x.shape[-1] == expected_last_dim:
                return x
            elif x.shape[1] == expected_last_dim:
                return x.transpose(1, 2)
        return x

    def _to_bdt(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            return x.transpose(1, 2)
        return x


    def ot_conditional_flow(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        t_exp = t.view(-1, 1, 1)


        x_t = (1 - t_exp) * x_0 + t_exp * x_1


        if self.sigma_min > 0 and self.training:
            x_t = x_t + self.sigma_min * torch.randn_like(x_t)


        u_t = x_1 - x_0

        return x_t, u_t

    def get_kl_weight(self) -> float:
        if not self.kl_annealing:
            return self.kl_weight


        if hasattr(self, 'trainer') and self.trainer is not None:
            step = self.trainer.global_step
        else:
            step = getattr(self, '_global_step', 0)

        progress = min(1.0, step / max(1, self.kl_warmup_steps))
        return self.kl_weight * progress

    def get_c_dropout_rate(self) -> float:
        if not hasattr(self, 'trainer') or self.trainer is None:
            return self.c_dropout_rate

        epoch = self.trainer.current_epoch

        if self.c_dropout_schedule == "constant":
            return self.c_dropout_rate

        elif self.c_dropout_schedule == "two_stage":
            if epoch < self.stage1_epochs:
                return self.stage1_c_dropout
            else:

                transition_epochs = 20
                epochs_in_stage2 = epoch - self.stage1_epochs
                if epochs_in_stage2 < transition_epochs:
                    alpha = epochs_in_stage2 / transition_epochs
                    return self.stage1_c_dropout * (1 - alpha) + self.stage2_c_dropout * alpha
                return self.stage2_c_dropout

        elif self.c_dropout_schedule == "linear_decay":
            max_epochs = self.trainer.max_epochs or 100
            alpha = min(1.0, epoch / (0.7 * max_epochs))
            return self.c_dropout_rate * (1 - 0.7 * alpha)

        return self.c_dropout_rate

    def apply_c_dropout(self, c: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return c

        p = self.get_c_dropout_rate()
        if p <= 0:
            return c


        B, T, D = c.shape
        mask = torch.bernoulli(torch.full((B, 1, D), 1 - p, device=c.device))
        return c * mask / (1 - p + 1e-8)

    def compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:

        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())


        if self.free_bits > 0:

            kl_per_dim = F.relu(kl_per_dim - self.free_bits) + self.free_bits


        return kl_per_dim.sum(dim=-1).mean()

    def compute_orthogonal_loss(self, e_samples: torch.Tensor) -> torch.Tensor:
        if e_samples.dim() == 3:

            e = e_samples.mean(dim=1)
        else:
            e = e_samples

        B = e.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=e.device)


        e_norm = F.normalize(e, dim=-1)


        sim = e_norm @ e_norm.T


        off_diag_mask = ~torch.eye(B, dtype=torch.bool, device=e.device)
        orth_loss = (sim[off_diag_mask] ** 2).mean()

        return orth_loss

    def compute_diversity_loss(self, mu: torch.Tensor) -> torch.Tensor:
        B = mu.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=mu.device)


        mu_norm = F.normalize(mu, dim=-1)
        dist = 1 - (mu_norm @ mu_norm.T)


        off_diag_mask = ~torch.eye(B, dtype=torch.bool, device=mu.device)
        mean_dist = dist[off_diag_mask].mean()


        return -mean_dist * 0.1


    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, loss_dict = self._shared_step(batch, stage="train")

        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/c_dropout", self.get_c_dropout_rate(), prog_bar=False, logger=True)
        self.log("train/kl_weight", self.get_kl_weight(), prog_bar=False, logger=True)

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        with self.ema_scope():
            loss, loss_dict = self._shared_step(batch, stage="val")

        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss

    def _shared_step(
        self,
        batch: Dict[str, torch.Tensor],
        stage: str = "train"
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        x_1 = batch['x']
        c = batch['c']
        c_mean = batch.get('c_mean', None)


        x_1 = self._to_btd(x_1, self.channels)
        c = self._to_btd(c, self.cond_channels)

        B, T, _ = x_1.shape
        device = x_1.device


        x_0 = torch.randn_like(x_1)


        t = torch.rand(B, device=device)


        x_t, u_t = self.ot_conditional_flow(x_0, x_1, t)


        if c_mean is not None:
            c_for_encoder = c_mean
        else:
            c_for_encoder = c.mean(dim=1)


        env_out_t = self.env_encoder(x_t, c_for_encoder, t=t)
        mu_t = env_out_t['mu']
        logvar_t = env_out_t['logvar']
        e = env_out_t['e']


        with torch.no_grad():
            env_out_1 = self.env_encoder(x_1, c_for_encoder, t=None)
            mu_1 = env_out_1['mu']


        c_dropped = self.apply_c_dropout(c) if stage == "train" else c


        v_pred = self.velocity_net(x_t, t, c_dropped, e)


        loss_fm = F.mse_loss(v_pred, u_t)


        loss_kl = self.compute_kl_loss(mu_t, logvar_t)


        loss_consist = F.mse_loss(mu_t, mu_1.detach())


        loss_orth = self.compute_orthogonal_loss(e)


        loss_div = self.compute_diversity_loss(mu_t)


        kl_w = self.get_kl_weight() if stage == "train" else self.kl_weight
        loss_total = (
            loss_fm
            + kl_w * loss_kl
            + self.consistency_weight * loss_consist
            + self.orth_weight * loss_orth
            + loss_div
        )


        loss_dict = {
            f'{stage}/loss': loss_total,
            f'{stage}/loss_fm': loss_fm,
            f'{stage}/loss_kl': loss_kl,
            f'{stage}/loss_consist': loss_consist,
            f'{stage}/loss_orth': loss_orth,
            f'{stage}/loss_div': loss_div,
        }


        with torch.no_grad():
            std = torch.exp(0.5 * logvar_t)
            loss_dict[f'{stage}/env_mu_norm'] = mu_t.norm(dim=-1).mean()
            loss_dict[f'{stage}/env_std_mean'] = std.mean()
            loss_dict[f'{stage}/env_std_min'] = std.min()

        return loss_total, loss_dict

    def on_train_batch_end(self, *args, **kwargs):
        if self.ema is not None:
            self.ema.update()

    @contextmanager
    def ema_scope(self, context=None):
        if self.ema is not None:
            self.ema.apply_shadow()
        try:
            yield
        finally:
            if self.ema is not None:
                self.ema.restore()


    @torch.no_grad()
    def sample(
        self,
        c: torch.Tensor,
        batch_size: Optional[int] = None,
        num_steps: int = 100,
        num_mc_samples: Optional[int] = None,
        method: str = 'euler',
        temperature: float = 1.0,
        use_prior: bool = False,
        e_fixed: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
        use_ema: bool = True,
        cfg_scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:

        c = self._to_btd(c, self.cond_channels)

        if batch_size is None:
            batch_size = c.shape[0]

        if num_mc_samples is None:
            num_mc_samples = self.num_mc_samples_eval

        if cfg_scale is None:
            cfg_scale = self.cfg_scale

        device = c.device


        c_mean = c.mean(dim=1)

        ctx = self.ema_scope() if use_ema and self.ema is not None else contextmanager(lambda: iter([None]))()

        with ctx:

            x_t = torch.randn(batch_size, self.seq_len, self.channels, device=device)

            intermediates = [x_t.clone()] if return_intermediates else None
            dt = 1.0 / num_steps

            for step in tqdm(range(num_steps), desc="Sampling", leave=False):
                t_val = step * dt
                t = torch.full((batch_size,), t_val, device=device)


                v = self._get_velocity_cfg(
                    x_t, t, c, c_mean,
                    num_mc_samples=num_mc_samples,
                    use_prior=use_prior,
                    e_fixed=e_fixed,
                    cfg_scale=cfg_scale,
                )


                if method == 'euler':
                    x_t = x_t + v * temperature * dt

                elif method == 'midpoint':
                    t_mid = torch.full((batch_size,), t_val + 0.5 * dt, device=device)
                    x_mid = x_t + v * temperature * (dt / 2)
                    v_mid = self._get_velocity_cfg(
                        x_mid, t_mid, c, c_mean,
                        num_mc_samples=num_mc_samples,
                        use_prior=use_prior,
                        e_fixed=e_fixed,
                        cfg_scale=cfg_scale,
                    )
                    x_t = x_t + v_mid * temperature * dt

                elif method == 'rk4':
                    k1 = v * temperature

                    t2 = torch.full((batch_size,), t_val + 0.5 * dt, device=device)
                    k2 = self._get_velocity_cfg(
                        x_t + k1 * dt / 2, t2, c, c_mean,
                        num_mc_samples=num_mc_samples,
                        use_prior=use_prior,
                        e_fixed=e_fixed,
                        cfg_scale=cfg_scale,
                    ) * temperature

                    k3 = self._get_velocity_cfg(
                        x_t + k2 * dt / 2, t2, c, c_mean,
                        num_mc_samples=num_mc_samples,
                        use_prior=use_prior,
                        e_fixed=e_fixed,
                        cfg_scale=cfg_scale,
                    ) * temperature

                    t3 = torch.full((batch_size,), t_val + dt, device=device)
                    k4 = self._get_velocity_cfg(
                        x_t + k3 * dt, t3, c, c_mean,
                        num_mc_samples=num_mc_samples,
                        use_prior=use_prior,
                        e_fixed=e_fixed,
                        cfg_scale=cfg_scale,
                    ) * temperature

                    x_t = x_t + (k1 + 2*k2 + 2*k3 + k4) * dt / 6

                if return_intermediates:
                    intermediates.append(x_t.clone())

            return x_t, intermediates

    def _get_velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        c_mean: torch.Tensor,
        num_mc_samples: int,
        use_prior: bool = False,
        e_fixed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = x_t.shape[0]
        device = x_t.device

        if e_fixed is not None:

            return self.velocity_net(x_t, t, c, e_fixed)

        if use_prior:

            if num_mc_samples == 1:
                e = torch.randn(B, self.env_dim, device=device)
                return self.velocity_net(x_t, t, c, e)
            else:
                e_samples = torch.randn(B, num_mc_samples, self.env_dim, device=device)
                return self._mc_velocity(x_t, t, c, e_samples)


        if num_mc_samples == 1:
            env_out = self.env_encoder(x_t, c_mean, t=t)
            e = env_out['e']
            return self.velocity_net(x_t, t, c, e)
        else:

            env_out = self.env_encoder(x_t, c_mean, t=t, num_samples=num_mc_samples)
            e_samples = env_out['e']
            return self._mc_velocity(x_t, t, c, e_samples)

    def _get_velocity_cfg(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        c_mean: torch.Tensor,
        num_mc_samples: int,
        use_prior: bool = False,
        e_fixed: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:

        v_cond = self._get_velocity(
            x_t, t, c, c_mean,
            num_mc_samples=num_mc_samples,
            use_prior=use_prior,
            e_fixed=e_fixed,
        )


        if cfg_scale <= 0:
            return v_cond

        B = x_t.shape[0]
        device = x_t.device


        c_null = self.null_cond.expand(B, -1, -1).to(device)
        e_null = self.null_env.expand(B, -1).to(device)
        v_uncond = self.velocity_net(x_t, t, c_null, e_null)


        omega = cfg_scale - 1.0
        v_cfg = (1 + omega) * v_cond - omega * v_uncond

        return v_cfg

    def _mc_velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        e_samples: torch.Tensor,
    ) -> torch.Tensor:
        B, N, D_e = e_samples.shape


        x_t_exp = x_t.unsqueeze(1).expand(-1, N, -1, -1).reshape(B * N, *x_t.shape[1:])
        t_exp = t.unsqueeze(1).expand(-1, N).reshape(B * N)
        c_exp = c.unsqueeze(1).expand(-1, N, -1, -1).reshape(B * N, *c.shape[1:])
        e_flat = e_samples.reshape(B * N, D_e)


        v_flat = self.velocity_net(x_t_exp, t_exp, c_exp, e_flat)


        v_samples = v_flat.reshape(B, N, *v_flat.shape[1:])
        return v_samples.mean(dim=1)

    @torch.no_grad()
    def encode_environment(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._to_btd(x, self.channels)

        if c.dim() == 3:
            c = self._to_btd(c, self.cond_channels)
            c_mean = c.mean(dim=1)
        else:
            c_mean = c

        with self.ema_scope():
            env_out = self.env_encoder(x, c_mean, t=t)

        mu = env_out['mu']
        std = torch.exp(0.5 * env_out['logvar'])

        return mu, std


    def configure_optimizers(self):
        params = list(self.env_encoder.parameters()) + list(self.velocity_net.parameters())

        optimizer = torch.optim.AdamW(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )


        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs if self.trainer else 100,
            eta_min=self.learning_rate * 0.01,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }


    def set_datasets(self, train_set, val_set, batch_size: int = 32, num_workers: int = 4):
        self._train_set = train_set
        self._val_set = val_set
        self._batch_size = batch_size
        self._num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self._train_set,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_set,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=True,
        )

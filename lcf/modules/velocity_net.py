import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
    emb = t.unsqueeze(-1).float() * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class AdaLN(nn.Module):

    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, hidden_dim * 2),
        )

        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.modulation(cond).chunk(2, dim=-1)

        if x.dim() == 3:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)

        return self.norm(x) * (1 + scale) + shift


class TransformerBlock(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.adaln1 = AdaLN(hidden_dim, cond_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.adaln2 = AdaLN(hidden_dim, cond_dim)
        mlp_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:

        h = self.adaln1(x, cond)
        h, _ = self.attn(h, h, h)
        x = x + h


        h = self.adaln2(x, cond)
        x = x + self.mlp(h)

        return x


def add_positional_encoding_catsg(c: torch.Tensor) -> torch.Tensor:
    B, T, D = c.shape
    device = c.device
    dtype = c.dtype


    t = torch.arange(T, device=device, dtype=dtype)
    phi = (2.0 * math.pi) * t / float(T)
    pos_enc = torch.stack([torch.sin(phi), torch.cos(phi)], dim=-1)
    pos_enc = pos_enc.unsqueeze(0).expand(B, T, 2)

    return torch.cat([c, pos_enc], dim=-1)


class VelocityNetwork(nn.Module):

    def __init__(
        self,
        seq_len: int = 96,
        input_dim: int = 1,
        cond_dim: int = 4,
        env_dim: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        cond_dropout: float = 0.2,
        full_cond_mask_prob: float = 0.15,
        direct_env_inject: bool = True,
        add_positional_encoding: bool = False,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.env_dim = env_dim
        self.hidden_dim = hidden_dim
        self.cond_dropout = cond_dropout
        self.full_cond_mask_prob = full_cond_mask_prob
        self.direct_env_inject = direct_env_inject
        self.add_positional_encoding = add_positional_encoding


        actual_cond_dim = cond_dim + 2 if add_positional_encoding else cond_dim
        self.actual_cond_dim = actual_cond_dim


        time_dim = hidden_dim
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )


        self.env_proj = nn.Sequential(
            nn.Linear(env_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )


        adaln_cond_dim = time_dim + hidden_dim


        input_proj_dim = input_dim + actual_cond_dim
        if direct_env_inject:
            input_proj_dim += env_dim

        self.input_proj = nn.Sequential(
            nn.Linear(input_proj_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )


        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)


        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim,
                cond_dim=adaln_cond_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])


        self.output_norm = AdaLN(hidden_dim, adaln_cond_dim)
        self.output_proj = nn.Linear(hidden_dim, input_dim)


        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _to_btd(self, x: torch.Tensor, expected_d: int) -> torch.Tensor:

        while x.dim() > 3:
            if x.shape[-1] == 1:
                x = x.squeeze(-1)
            else:
                x = x.squeeze(-2)

        if x.dim() == 3:
            if x.shape[-1] == expected_d:
                return x
            elif x.shape[1] == expected_d:
                return x.transpose(1, 2)
        elif x.dim() == 2:

            x = x.unsqueeze(-1)
        return x

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        e: torch.Tensor,
        return_hidden: bool = False,
    ):

        input_transposed = False
        if x_t.dim() == 3 and x_t.shape[1] == self.input_dim and x_t.shape[2] == self.seq_len:
            x_t = x_t.transpose(1, 2)
            input_transposed = True

        x_t = self._to_btd(x_t, self.input_dim)
        c = self._to_btd(c, self.cond_dim)

        B, T, _ = x_t.shape


        if self.add_positional_encoding:
            c = add_positional_encoding_catsg(c)


        if self.training:

            full_mask = (torch.rand(B, 1, 1, device=c.device) < self.full_cond_mask_prob).float()

            partial_mask = (torch.rand(B, 1, 1, device=c.device) < self.cond_dropout).float()

            c = c * (1 - full_mask) * (1 - 0.5 * partial_mask * (1 - full_mask))


        t_emb = sinusoidal_embedding(t, self.hidden_dim)
        t_emb = self.time_embed(t_emb)


        e_emb = self.env_proj(e)


        cond = torch.cat([t_emb, e_emb], dim=-1)


        if self.direct_env_inject:

            e_seq = e.unsqueeze(1).expand(B, T, -1)
            xce = torch.cat([x_t, c, e_seq], dim=-1)
            h = self.input_proj(xce)
        else:

            xc = torch.cat([x_t, c], dim=-1)
            h = self.input_proj(xc)


        h = h + self.pos_emb[:, :T, :]


        for block in self.blocks:
            h = block(h, cond)


        h = self.output_norm(h, cond)
        v = self.output_proj(h)


        if input_transposed:
            v = v.transpose(1, 2)

        if return_hidden:
            return v, h
        return v


class ConditionalVelocityNet(VelocityNetwork):

    def __init__(
        self,
        seq_len: int = 96,
        in_channels: int = 1,
        out_channels: int = 1,
        model_channels: int = 64,
        env_dim: int = 32,
        cond_dim: int = 64,
        num_transformer_blocks: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(
            seq_len=seq_len,
            input_dim=in_channels,
            cond_dim=cond_dim,
            env_dim=env_dim,
            hidden_dim=model_channels * 4,
            num_layers=num_transformer_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward_mc(
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


        v_flat = self.forward(x_t_exp, t_exp, c_exp, e_flat)


        v_samples = v_flat.reshape(B, N, *v_flat.shape[1:])
        return v_samples.mean(dim=1)


class VectorVelocityNet(nn.Module):

    def __init__(
        self,
        x_dim: int,
        c_dim: int,
        env_dim: int = 4,
        hidden_dim: int = 128,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.x_dim = x_dim
        self.c_dim = c_dim
        self.env_dim = env_dim
        self.hidden_dim = hidden_dim


        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )


        self.cond_embed = nn.Sequential(
            nn.Linear(c_dim + env_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )


        self.input_proj = nn.Linear(x_dim, hidden_dim)


        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(VectorAdaLNBlock(hidden_dim, hidden_dim, dropout))


        self.output_proj = nn.Linear(hidden_dim, x_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        e: torch.Tensor,
    ) -> torch.Tensor:

        t_emb = sinusoidal_embedding(t, self.hidden_dim)
        t_emb = self.time_embed(t_emb)


        ce = torch.cat([c, e], dim=-1)
        ce_emb = self.cond_embed(ce)


        cond = t_emb + ce_emb


        h = self.input_proj(x_t)


        for block in self.blocks:
            h = block(h, cond)

        return self.output_proj(h)

    def forward_mc(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        e_samples: torch.Tensor,
    ) -> torch.Tensor:
        B, N, D_e = e_samples.shape

        x_t_exp = x_t.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)
        t_exp = t.unsqueeze(1).expand(-1, N).reshape(B * N)
        c_exp = c.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)
        e_flat = e_samples.reshape(B * N, -1)

        v_flat = self.forward(x_t_exp, t_exp, c_exp, e_flat)
        v_samples = v_flat.reshape(B, N, -1)

        return v_samples.mean(dim=1)


class VectorAdaLNBlock(nn.Module):

    def __init__(self, hidden_dim: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()

        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, hidden_dim * 3),
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        params = self.modulation(cond)
        scale, shift, gate = params.chunk(3, dim=-1)

        h = self.norm(x) * (1 + scale) + shift
        h = self.mlp(h)

        return x + gate * h


def create_velocity_net_with_cpd(
    seq_len: int = 96,
    input_dim: int = 1,
    cond_dim: int = 4,
    env_dim: int = 16,
    hidden_dim: int = 128,
    num_layers: int = 6,
    num_heads: int = 4,
    dropout: float = 0.1,
    cond_dropout: float = 0.2,
    full_cond_mask_prob: float = 0.15,
    direct_env_inject: bool = True,
    add_positional_encoding: bool = False,
):
    from lcf.modules.causal_attention_plugin import wrap_with_cpd


    base_net = VelocityNetwork(
        seq_len=seq_len,
        input_dim=input_dim,
        cond_dim=cond_dim,
        env_dim=env_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        cond_dropout=cond_dropout,
        full_cond_mask_prob=full_cond_mask_prob,
        direct_env_inject=direct_env_inject,
        add_positional_encoding=add_positional_encoding,
    )


    return wrap_with_cpd(
        base_net,
        hidden_dim=hidden_dim,
        env_dim=env_dim,
        cond_dim=cond_dim,
        output_dim=input_dim,
    )

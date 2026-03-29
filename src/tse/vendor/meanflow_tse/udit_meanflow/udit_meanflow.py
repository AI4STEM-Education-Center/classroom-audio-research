"""UDiT: U-Net style Diffusion Transformer for MeanFlow-TSE.

Vendored from: https://github.com/rikishimizu/MeanFlow-TSE/blob/main/models/udit_meanflow/udit_meanflow.py
License: MIT (Meta Platforms, Inc. / rikishimizu)

Stripped: DiT config factory functions (training-only), yaml import, __main__ block.
"""

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .attention import Attention


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PositionalConvEmbedding(nn.Module):
    def __init__(self, dim=768, kernel_size=128, groups=16):
        super().__init__()
        self.conv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=groups, bias=True
        )
        self.conv = nn.utils.parametrizations.weight_norm(self.conv, name="weight", dim=2)

    def forward(self, x):
        x = x.transpose(2, 1)
        x = self.conv(x)
        x = F.gelu(x[:, :, :-1])
        x = x.transpose(2, 1)
        return x


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim, length):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.length = length
        self.dim = dim
        self.register_buffer("pe", self._generate_positional_encoding(length, dim))

    def _generate_positional_encoding(self, length, dim):
        pe = torch.zeros(length, dim)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class PE_wrapper(nn.Module):
    def __init__(self, dim=768, method="none", length=None):
        super().__init__()
        self.method = method
        if method == "abs":
            self.length = length
            self.abs_pe = nn.Parameter(torch.zeros(1, length, dim))
            trunc_normal_(self.abs_pe, std=0.02)
        elif method == "conv":
            self.conv_pe = PositionalConvEmbedding(dim=dim)
        elif method == "sinu":
            self.sinu_pe = SinusoidalPositionalEncoding(dim=dim, length=length)
        elif method == "none":
            self.id = nn.Identity()
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.method == "abs":
            _, L, _ = x.shape
            assert L <= self.length
            x = x + self.abs_pe[:, :L, :]
        elif self.method == "conv":
            x = x + self.conv_pe(x)
        elif self.method == "sinu":
            x = self.sinu_pe(x)
        elif self.method == "none":
            x = self.id(x)
        else:
            raise NotImplementedError
        return x


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiTBlock(nn.Module):
    """DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, skip=False, skip_norm=True, use_checkpoint=False, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.skip_linear = nn.Linear(2 * hidden_size, hidden_size) if skip else None
        self.skip_norm = (
            nn.LayerNorm(2 * hidden_size, elementwise_affine=False, eps=1e-6)
            if skip_norm
            else nn.Identity()
        )
        self.use_checkpoint = use_checkpoint

    def forward(self, x, c, skip=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, c, skip)
        else:
            return self._forward(x, c, skip)

    def _forward(self, x, c, skip=None):
        if self.skip_linear is not None:
            cat = torch.cat([x, skip], dim=-1)
            cat = self.skip_norm(cat)
            x = self.skip_linear(cat)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """The final layer of DiT."""

    def __init__(self, hidden_size, output_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, output_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class UDiT(nn.Module):
    """U-Net Diffusion Transformer for MeanFlow-TSE.

    Default config (from config_MeanFlowTSE_clean.yaml):
        input_dim=512, output_dim=512, hidden_size=1024, depth=16, num_heads=16
    """

    def __init__(
        self,
        input_dim=512,
        output_dim=512,
        pos_method="none",
        pos_length=500,
        hidden_size=1024,
        depth=16,
        num_heads=16,
        mlp_ratio=4.0,
        use_checkpoint=False,
    ):
        super().__init__()
        self.num_heads = num_heads

        self.input_proj = nn.Linear(input_dim, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.delta_embedder = TimestepEmbedder(hidden_size)
        self.pos_embed = PE_wrapper(dim=hidden_size, method=pos_method, length=pos_length)

        self.in_blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, use_checkpoint=use_checkpoint)
                for _ in range(depth // 2)
            ]
        )
        self.mid_block = DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, use_checkpoint=use_checkpoint)
        self.out_blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, skip=True, use_checkpoint=use_checkpoint)
                for _ in range(depth // 2)
            ]
        )

        self.final_layer = FinalLayer(hidden_size, output_dim)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.normal_(self.input_proj.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.delta_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.delta_embedder.mlp[2].weight, std=0.02)

        for block in self.in_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.mid_block.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.mid_block.adaLN_modulation[-1].bias, 0)

        for block in self.out_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, r, enrollment):
        """Forward pass.

        Args:
            x: (N, C, T) input spectrogram (mixture or intermediate state)
            t: (N,) current timestep in [0, 1]
            r: (N,) target timestep in [0, 1]
            enrollment: (N, C, T_enroll) reference speaker spectrogram

        Returns:
            (N, C, T) predicted velocity field
        """
        # Scale timesteps to [0, 1000]
        timesteps_t = t * 1000
        timesteps_r = r * 1000

        # Prepend enrollment to input along time axis
        enrollment_length = enrollment.shape[2]
        x = torch.cat((enrollment, x), dim=-1)

        # Project: (N, C, T) -> (N, T, C) -> (N, T, D)
        x = x.transpose(2, 1)
        x = self.input_proj(x)
        x = self.pos_embed(x)

        # Ensure timesteps are tensors
        if not torch.is_tensor(timesteps_t):
            timesteps_t = torch.tensor([timesteps_t], dtype=torch.long, device=x.device)
        elif torch.is_tensor(timesteps_t) and len(timesteps_t.shape) == 0:
            timesteps_t = timesteps_t[None].to(x.device)

        if not torch.is_tensor(timesteps_r):
            timesteps_r = torch.tensor([timesteps_r], dtype=torch.long, device=x.device)
        elif torch.is_tensor(timesteps_r) and len(timesteps_r.shape) == 0:
            timesteps_r = timesteps_r[None].to(x.device)

        # Embed timesteps
        t_emb = self.t_embedder(timesteps_t)
        delta_emb = self.delta_embedder(timesteps_r - timesteps_t)
        c = t_emb + delta_emb

        # U-Net transformer blocks
        skips = []
        for blk in self.in_blocks:
            x = blk(x, c)
            skips.append(x)

        x = self.mid_block(x, c)

        for blk in self.out_blocks:
            x = blk(x, c, skips.pop())

        x = self.final_layer(x, c)

        # Project back: (N, T, D) -> (N, D, T), remove enrollment portion
        x = x.transpose(2, 1)
        x = x[:, :, enrollment_length:]

        return x

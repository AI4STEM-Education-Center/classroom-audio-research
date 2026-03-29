"""Rotary position embeddings for UDiT attention.

Vendored from: https://github.com/rikishimizu/MeanFlow-TSE/blob/main/models/udit_meanflow/rotary.py
License: MIT
"""

import torch


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    cos = cos[:, :, : x.shape[-2], :]
    sin = sin[:, :, : x.shape[-2], :]
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(torch.nn.Module):
    """Rotary position embeddings (RoPE) from RoFormer (Su et al.)."""

    def __init__(self, dim: int):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_dimension=-2):
        seq_len = x.shape[seq_dimension]

        if (
            seq_len != self._seq_len_cached
            or self._cos_cached.device != x.device
            or self._cos_cached.dtype != x.dtype
        ):
            self._seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dimension], device=x.device, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(x.dtype))
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self._cos_cached = emb.cos()[None, None, :, :].to(x.dtype)
            self._sin_cached = emb.sin()[None, None, :, :].to(x.dtype)

        return self._cos_cached, self._sin_cached

    def forward(self, q, k):
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(
            q.float(), seq_dimension=-2
        )
        if k is not None:
            return (
                apply_rotary_pos_emb(q.float(), self._cos_cached, self._sin_cached).type_as(q),
                apply_rotary_pos_emb(k.float(), self._cos_cached, self._sin_cached).type_as(k),
            )
        else:
            return (
                apply_rotary_pos_emb(q.float(), self._cos_cached, self._sin_cached).type_as(q),
                None,
            )

"""Caché incremental LFM2 (KV para GQA + estado para conv corta).

Cada capa de atención guarda k/v rotados (B, n_kv, L, D).
Cada capa conv guarda las últimas k-1 activaciones `y` (B, d, k-1).
"""

from __future__ import annotations

import torch


class LFM2Cache:
    """Contenedor de cachés por capa, independiente de `transformers`."""

    def __init__(self) -> None:
        self.k: dict[int, torch.Tensor] = {}
        self.v: dict[int, torch.Tensor] = {}
        self.conv: dict[int, torch.Tensor] = {}
        self.seq_len: int = 0

    def __len__(self) -> int:
        return self.seq_len

    def get_seq_length(self) -> int:
        return self.seq_len

    # -- atención ---------------------------------------------------------
    def append_kv(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatena k/v nuevos (ya rotados) y devuelve los completos."""
        if layer_idx in self.k:
            k = torch.cat([self.k[layer_idx], k_new], dim=2)
            v = torch.cat([self.v[layer_idx], v_new], dim=2)
        else:
            k, v = k_new, v_new
        self.k[layer_idx] = k
        self.v[layer_idx] = v
        return k, v

    def set_kv(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        self.k[layer_idx] = k
        self.v[layer_idx] = v

    # -- conv -------------------------------------------------------------
    def set_conv(self, layer_idx: int, state: torch.Tensor) -> None:
        self.conv[layer_idx] = state

    def get_conv(self, layer_idx: int) -> torch.Tensor | None:
        return self.conv.get(layer_idx)

    # -- utilidades -------------------------------------------------------
    def advance(self, n_tokens: int) -> None:
        self.seq_len += n_tokens

    def batch_select(self, indices: torch.Tensor) -> "LFM2Cache":
        """Subselección de batch (para beam search)."""
        out = LFM2Cache()
        for i, t in self.k.items():
            out.k[i] = t[indices]
        for i, t in self.v.items():
            out.v[i] = t[indices]
        for i, t in self.conv.items():
            out.conv[i] = t[indices]
        out.seq_len = self.seq_len
        return out

    def to(self, device: torch.device | str) -> "LFM2Cache":
        for i in list(self.k):
            self.k[i] = self.k[i].to(device)
            self.v[i] = self.v[i].to(device)
        for i in list(self.conv):
            self.conv[i] = self.conv[i].to(device)
        return self

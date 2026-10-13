"""Caché incremental TURING-1.

- KV por **grupo CLA** (no por capa): N atenciones -> ~N/2 cachés.
- Grupos totalmente deslizantes: buffer rodante (trim a la ventana).
- Grupos mixtos/globales: historial completo (lo exige la capa global).
- Estados conv por capa (12 vectores multiescala).
- `snapshot()` barato (vistas) para rollback en decodificación especulativa.
"""

from __future__ import annotations

import torch


class TuringCache:
    def __init__(self) -> None:
        self.k: dict[int, torch.Tensor] = {}
        self.v: dict[int, torch.Tensor] = {}
        self.conv: dict[int, torch.Tensor] = {}
        self.group_window: dict[int, int | None] = {}  # gid -> W o None
        self.seq_len: int = 0

    def get_seq_length(self) -> int:
        return self.seq_len

    # -- grupos KV ------------------------------------------------------
    def set_group_window(self, gid: int, window: int | None) -> None:
        self.group_window[gid] = window

    def append_group_kv(
        self, gid: int, k_new: torch.Tensor, v_new: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if gid in self.k:
            k = torch.cat([self.k[gid], k_new], dim=2)
            v = torch.cat([self.v[gid], v_new], dim=2)
        else:
            k, v = k_new, v_new
        w = self.group_window.get(gid)
        if w is not None and k.shape[2] > w:
            k, v = k[:, :, -w:, :], v[:, :, -w:, :]
        self.k[gid], self.v[gid] = k, v
        return k, v

    def set_group_kv(self, gid: int, k: torch.Tensor, v: torch.Tensor) -> None:
        w = self.group_window.get(gid)
        if w is not None and k.shape[2] > w:
            k, v = k[:, :, -w:, :], v[:, :, -w:, :]
        self.k[gid], self.v[gid] = k, v

    def get_group_kv(self, gid: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return self.k.get(gid), self.v.get(gid)

    # -- conv -----------------------------------------------------------
    def set_conv(self, layer_idx: int, state: torch.Tensor) -> None:
        self.conv[layer_idx] = state

    def get_conv(self, layer_idx: int) -> torch.Tensor | None:
        return self.conv.get(layer_idx)

    # -- utilidades -----------------------------------------------------
    def snapshot(self) -> "TuringCache":
        """Copia barata (los tensores son inmutables en nuestro flujo)."""
        out = TuringCache()
        out.k, out.v, out.conv = dict(self.k), dict(self.v), dict(self.conv)
        out.group_window = dict(self.group_window)
        out.seq_len = self.seq_len
        return out

    def to(self, device: torch.device | str) -> "TuringCache":
        for d in (self.k, self.v, self.conv):
            for i in list(d):
                d[i] = d[i].to(device)
        return self

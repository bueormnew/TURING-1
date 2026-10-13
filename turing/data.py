"""Datos byte-level para LM real sin tokenizador (vocab 256 = bytes UTF-8).

Cualquier .txt sirve: se empaqueta en trozos contiguos (el modelo desplaza
labels internamente), split train/val por OFFSET de bytes (sin fuga) y
lotes deterministas con remezcla por época.
"""

from __future__ import annotations

import torch


def read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def make_chunks(data: bytes, seq_len: int) -> torch.Tensor:
    """Trocea en (n, seq_len) uint8->long. Descarta la cola incompleta."""
    n = len(data) // seq_len
    assert n >= 1, f"muy poco texto para seq_len={seq_len}"
    return torch.frombuffer(bytearray(data[: n * seq_len]), dtype=torch.uint8).view(n, seq_len).long()


def split_chunks(chunks: torch.Tensor, val_frac: float = 0.02) -> tuple[torch.Tensor, torch.Tensor]:
    n_val = max(int(len(chunks) * val_frac), 1) if len(chunks) > 1 else 0
    if n_val == 0:
        return chunks, chunks[:0]
    return chunks[:-n_val], chunks[-n_val:]


class ByteDataLoader:
    """Lotes infinitos deterministas (re-mezcla por época con seed+época)."""

    def __init__(self, chunks: torch.Tensor, batch: int, seed: int = 0, shuffle: bool = True) -> None:
        assert len(chunks) >= batch, "menos chunks que batch"
        self.chunks, self.batch, self.seed, self.shuffle = chunks, batch, seed, shuffle
        self.epoch = 0

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        idx = torch.randperm(len(self.chunks), generator=g) if self.shuffle else torch.arange(len(self.chunks))
        self.epoch += 1
        for s in range(0, len(idx) - self.batch + 1, self.batch):
            yield self.chunks[idx[s : s + self.batch]]


def encode_text(s: str) -> torch.Tensor:
    """Texto -> ids byte (1, L). Sin tokenizador: UTF-8 directo."""
    return torch.tensor(list(s.encode("utf-8")), dtype=torch.long).unsqueeze(0)


def decode_ids(ids) -> str:
    """Ids -> texto (ids fuera de 0..255 se vuelven '?')."""
    if torch.is_tensor(ids):
        ids = ids.reshape(-1).tolist()
    return bytes([b if 0 <= b < 256 else 63 for b in ids]).decode("utf-8", errors="replace")

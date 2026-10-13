"""TURING-1 v1 (PyTorch puro, construido sobre la base LFM2).

Híbrido conv/atención con: KV compartido por pares (CLA), ventana
deslizante 2/3 + global 1/3, MoE fino top-6/8 + shared (estilo Kimi),
cabeza MTP, conv multiescala con gate y diseño 4-bit-friendly.

Uso mínimo:
    import turing
    model = turing.TuringForCausalLM.from_preset("tiny")
"""

__version__ = "0.1.0"

from .cache import TuringCache
from .config import TuringConfig, get_preset
from .data import ByteDataLoader, decode_ids, encode_text, make_chunks, read_bytes, split_chunks
from .model import TuringForCausalLM, TuringModel
from .modules import TuringAttention, TuringMoE, TuringMTPHead, TuringMultiscaleConv
from .train import TrainConfig, forward_backward, load_checkpoint, save_checkpoint, train, train_step
from .utils import QUANT_RECIPE, memory_report

__all__ = [
    "__version__",
    "TuringConfig",
    "get_preset",
    "TuringCache",
    "TuringModel",
    "TuringForCausalLM",
    "TuringAttention",
    "TuringMoE",
    "TuringMTPHead",
    "TuringMultiscaleConv",
    "TrainConfig",
    "forward_backward",
    "train_step",
    "train",
    "save_checkpoint",
    "load_checkpoint",
    "ByteDataLoader",
    "make_chunks",
    "read_bytes",
    "split_chunks",
    "encode_text",
    "decode_ids",
    "memory_report",
    "QUANT_RECIPE",
]

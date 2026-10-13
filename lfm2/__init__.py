"""Paquete LFM2 puro en PyTorch (variante densa, sin MoE).

Implementación desde cero (solo `torch`) de la arquitectura densa
Liquid Foundation Models v2 (Liquid AI, 2025) según el Technical Report
arXiv:2511.23404, sección 2.2 y los `config.json` oficiales de HF.

Arquitectura:
- Híbrido mínimo: N capas, mayoría de convoluciones cortas con doble
  puerta + minoría de bloques GQA con QK-Norm y RoPE, cada bloque seguido
  de MLP SwiGLU. Pre-norm RMSNorm en ambos sub-bloques.
"""

from .cache import LFM2Cache
from .config import LFM2Config, get_preset
from .model import LFM2ForCausalLM, LFM2Model
from .modules import (
    LFM2Attention,
    LFM2RMSNorm,
    LFM2RotaryEmbedding,
    LFM2ShortConv,
    LFM2SwiGLU,
)

__all__ = [
    "LFM2Config",
    "get_preset",
    "LFM2Cache",
    "LFM2Model",
    "LFM2ForCausalLM",
    "LFM2RMSNorm",
    "LFM2RotaryEmbedding",
    "LFM2Attention",
    "LFM2ShortConv",
    "LFM2SwiGLU",
]

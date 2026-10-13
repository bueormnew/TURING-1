"""Configuración LFM2.

Fiel a la Tabla 1 del Technical Report (arXiv:2511.23404) y a los
`config.json` oficiales de HuggingFace (LiquidAI/LFM2-*).

Arquitectura (sección 2.2 del report):
    - Híbrido mínimo: mayoría de bloques de convolución corta con doble
      puerta + minoría de bloques GQA, cada uno seguido de MLP SwiGLU.
    - Pre-norm RMSNorm en ambos sub-bloques, RoPE + QK-Norm en atención.
    - Conv depthwise causal k=3, in_proj: d -> 3d (B, C, x), out_proj: d -> d.
    - SwiGLU con ajuste de dimensión: inter = 2*block_ff_dim/3 redondeado
      a múltiplo de `block_multiple_of` (cuando `auto_adjust=True`).
"""

from __future__ import annotations

from dataclasses import dataclass


def _adjust_ff_dim(block_ff_dim: int, multiple_of: int = 256, multiplier: float = 1.0) -> int:
    """Réplica exacta de `Lfm2MLP.__init__` en HF transformers.

    intermediate = int(2 * block_ff_dim / 3)
    intermediate = int(multiplier * intermediate)
    intermediate = multiple_of * ceil(intermediate / multiple_of)
    """
    inter = int(2 * block_ff_dim / 3)
    if multiplier is not None:
        inter = int(multiplier * inter)
    inter = multiple_of * ((inter + multiple_of - 1) // multiple_of)
    return inter


@dataclass
class LFM2Config:
    """Configuración del backbone denso LFM2."""

    vocab_size: int = 65536
    hidden_size: int = 2048
    num_hidden_layers: int = 16
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    # Dimensión "cruda" antes del ajuste SwiGLU (equivale a `block_ff_dim` en HF).
    # La dimensión efectiva se obtiene con `effective_intermediate_size`.
    block_ff_dim: int = 12288
    block_multiple_of: int = 256
    block_ffn_dim_multiplier: float = 1.0
    block_auto_adjust_ff_dim: bool = True
    # Si es None se deriva de `full_attn_idxs`.
    layer_types: list[str] | None = None
    full_attn_idxs: list[int] | None = None
    conv_kernel_size: int = 3  # `conv_L_cache` en HF
    conv_bias: bool = False
    max_position_embeddings: int = 128000
    rope_theta: float = 1_000_000.0
    norm_eps: float = 1e-5
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 7

    def __post_init__(self) -> None:
        if self.layer_types is None:
            if self.full_attn_idxs is None:
                self.full_attn_idxs = list(range(self.num_hidden_layers))
            self.layer_types = [
                "full_attention" if i in set(self.full_attn_idxs) else "conv"
                for i in range(self.num_hidden_layers)
            ]
        else:
            assert len(self.layer_types) == self.num_hidden_layers, (
                f"layer_types len {len(self.layer_types)} != num_hidden_layers {self.num_hidden_layers}"
            )
            for t in self.layer_types:
                assert t in ("conv", "full_attention"), f"layer_type desconocido: {t}"
            if self.full_attn_idxs is None:
                self.full_attn_idxs = [i for i, t in enumerate(self.layer_types) if t == "full_attention"]

    @property
    def head_dim(self) -> int:
        assert self.hidden_size % self.num_attention_heads == 0
        return self.hidden_size // self.num_attention_heads

    @property
    def num_key_value_groups(self) -> int:
        assert self.num_attention_heads % self.num_key_value_heads == 0
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def effective_intermediate_size(self) -> int:
        if self.block_auto_adjust_ff_dim:
            return _adjust_ff_dim(self.block_ff_dim, self.block_multiple_of, self.block_ffn_dim_multiplier)
        return self.block_ff_dim

    @property
    def num_attention_blocks(self) -> int:
        return sum(1 for t in (self.layer_types or []) if t == "full_attention")

    @property
    def num_conv_blocks(self) -> int:
        return self.num_hidden_layers - self.num_attention_blocks


# ---------------------------------------------------------------------------
# Presets oficiales densos (Tabla 1 del report + config.json de HF)
# ---------------------------------------------------------------------------

# LFM2-2.6B: 30 capas, 8 atenciones. Lista literal del config.json oficial.
_LFM2_26B_LAYER_TYPES: list[str] = [
    "conv", "conv", "full_attention", "conv", "conv",
    "full_attention", "conv", "conv", "conv", "full_attention",
    "conv", "conv", "conv", "full_attention", "conv",
    "conv", "conv", "full_attention", "conv", "conv",
    "conv", "full_attention", "conv", "conv",
    "full_attention", "conv", "conv", "full_attention",
    "conv", "conv",
]

# LFM2-8B-A1B (MoE): no incluido por el momento (variante dispersa, sección 2.3).

# Modelos de 16 capas (350M/700M/1.2B): mismo layout, 10 conv + 6 GQA.
_LFM2_16L_ATTN_IDXS: list[int] = [2, 5, 8, 10, 12, 14]


def lfm2_350m() -> LFM2Config:
    """LFM2-350M: d=1024, 16 capas, FF efectiva 4608 (6656 ajustada), 16/8/64."""
    return LFM2Config(
        hidden_size=1024,
        num_hidden_layers=16,
        num_attention_heads=16,
        num_key_value_heads=8,
        block_ff_dim=6656,
        block_auto_adjust_ff_dim=True,
        full_attn_idxs=list(_LFM2_16L_ATTN_IDXS),
    )


def lfm2_700m() -> LFM2Config:
    """LFM2-700M: d=1536, 16 capas, FF efectiva 6912 (10240 ajustada), 24/8/64."""
    return LFM2Config(
        hidden_size=1536,
        num_hidden_layers=16,
        num_attention_heads=24,
        num_key_value_heads=8,
        block_ff_dim=10240,
        block_auto_adjust_ff_dim=True,
        full_attn_idxs=list(_LFM2_16L_ATTN_IDXS),
    )


def lfm2_12b() -> LFM2Config:
    """LFM2-1.2B: d=2048, 16 capas, FF efectiva 8192 (12288 ajustada), 32/8/64."""
    return LFM2Config(
        hidden_size=2048,
        num_hidden_layers=16,
        num_attention_heads=32,
        num_key_value_heads=8,
        block_ff_dim=12288,
        block_auto_adjust_ff_dim=True,
        full_attn_idxs=list(_LFM2_16L_ATTN_IDXS),
    )


def lfm2_26b() -> LFM2Config:
    """LFM2-2.6B: d=2048, 30 capas, FF 10752 sin ajuste, 32/8/64, 8 atenciones."""
    return LFM2Config(
        hidden_size=2048,
        num_hidden_layers=30,
        num_attention_heads=32,
        num_key_value_heads=8,
        block_ff_dim=10752,
        block_auto_adjust_ff_dim=False,
        layer_types=list(_LFM2_26B_LAYER_TYPES),
    )


_PRESETS = {
    "350m": lfm2_350m,
    "700m": lfm2_700m,
    "1.2b": lfm2_12b,
    "2.6b": lfm2_26b,
}


def get_preset(name: str) -> LFM2Config:
    """Devuelve el preset oficial denso por nombre: 350m, 700m, 1.2b, 2.6b."""
    key = name.strip().lower()
    if key not in _PRESETS:
        raise ValueError(f"Preset desconocido {name!r}. Opciones: {sorted(_PRESETS)}")
    return _PRESETS[key]()

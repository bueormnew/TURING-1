"""Configuración TURING-1 (v1, densa + MoE, sin SSM).

TURING-1 = backbone híbrido LFM2 + 6 cambios:
1. KV compartido por pares de atenciones consecutivas (estilo CLA).
2. Ventana deslizante: patrón cíclico [sliding, sliding, global] sobre las
   atenciones en orden (2 de cada 3 con ventana, la tercera global).
3. 1 cabeza MTP (profundidad configurable 1-2) con la head LM compartida.
4. MoE nativo estilo Kimi/DeepSeek: muchos expertos finos, top-6/top-8,
   router softmax + sesgo aux-free, 1 experto compartido.
5. Conv multiescala sin recurrencia: rama k=3 (idéntica a LFM2) + rama
   dilatada k=7/d=2, mezcla por gate inicializado a identidad LFM2.
6. Diseño 4-bit-friendly (sin QAT en v1): QK-Norm, puertas acotadas,
   sin matrices gigantes nuevas (MTP reusa la head atada).
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _adjust_ff_dim(block_ff_dim: int, multiple_of: int = 256, multiplier: float = 1.0) -> int:
    inter = int(2 * block_ff_dim / 3)
    if multiplier is not None:
        inter = int(multiplier * inter)
    return multiple_of * ((inter + multiple_of - 1) // multiple_of)


@dataclass
class TuringConfig:
    # -- backbone (mismas dimensiones que LFM2 para comparar 1:1) --
    vocab_size: int = 65536
    hidden_size: int = 1024
    num_hidden_layers: int = 16
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    block_ff_dim: int = 6656
    block_multiple_of: int = 256
    block_ffn_dim_multiplier: float = 1.0
    block_auto_adjust_ff_dim: bool = True
    layer_types: list[str] | None = None
    full_attn_idxs: list[int] | None = None

    # -- 2. sliding window (patrón cíclico sobre atenciones en orden) --
    attn_window_pattern: list[str] = field(default_factory=lambda: ["sliding", "sliding", "global"])
    sliding_window: int = 2048

    # -- 1. KV compartido por pares consecutivos (CLA) --
    kv_share_pairs: bool = True

    # -- 4. MoE estilo Kimi (finos, top-6/8, shared) --
    use_moe: bool = True
    num_dense_layers: int = 2  # primeras capas con SwiGLU denso (estabilidad)
    num_experts: int = 32
    num_experts_per_tok: int = 6
    moe_intermediate_size: int | None = None  # None -> dense // 8
    moe_shared_experts: int = 1
    moe_norm_topk_prob: bool = True
    moe_z_loss_weight: float = 0.0  # 0 = off; 1e-3 típico (router z-loss estilo DeepSeek)
    moe_dispatch: str = "auto"  # "loop" | "batched" | "auto" (M<=8 loop, si no bmm)

    # -- 3. MTP --
    use_mtp: bool = True
    mtp_depth: int = 1  # 1 ahora, 2 cuando se quiera predecir un token más
    mtp_loss_weight: float = 0.2

    # -- 5. conv multiescala (sin recurrencia) --
    use_multiscale_conv: bool = True
    conv_kernel_size: int = 3
    conv_dilated_kernel: int = 7
    conv_dilation: int = 2
    conv_bias: bool = False
    multiscale_gate_init: float = -3.0  # sigmoid(-3)≈0.05: arranca como LFM2

    # -- varios --
    max_position_embeddings: int = 128000
    rope_theta: float = 1_000_000.0
    norm_eps: float = 1e-5
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    use_sdpa: bool = True  # backend SDPA (flash/mem-efficient) cuando es exacto
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 7

    # Derivados (se calculan en post_init, no pasar manualmente):
    attn_kinds: list[str] = field(default_factory=list)       # por atención en orden
    attn_group: list[int] = field(default_factory=list)       # grupo CLA por atención
    attn_is_producer: list[bool] = field(default_factory=list)
    group_all_sliding: dict[int, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.layer_types is None:
            if self.full_attn_idxs is None:
                self.full_attn_idxs = list(range(self.num_hidden_layers))
            self.layer_types = [
                "full_attention" if i in set(self.full_attn_idxs) else "conv"
                for i in range(self.num_hidden_layers)
            ]
        assert len(self.layer_types) == self.num_hidden_layers
        n_attn = sum(1 for t in self.layer_types if t == "full_attention")
        pat = self.attn_window_pattern
        assert all(p in ("sliding", "global") for p in pat) and len(pat) > 0
        self.attn_kinds = [pat[i % len(pat)] for i in range(n_attn)]
        # Grupos CLA: pares consecutivos (0,1), (2,3)...; impar suelto = grupo propio.
        self.attn_group, self.attn_is_producer = [], []
        for i in range(n_attn):
            if self.kv_share_pairs:
                g = i // 2
                prod = (i % 2 == 0)
            else:
                g, prod = i, True
            self.attn_group.append(g)
            self.attn_is_producer.append(prod)
        self.group_all_sliding = {}
        for i, g in enumerate(self.attn_group):
            self.group_all_sliding.setdefault(g, True)
            if self.attn_kinds[i] != "sliding":
                self.group_all_sliding[g] = False
        if self.moe_intermediate_size is None:
            self.moe_intermediate_size = max(64, self.effective_intermediate_size // 8)
        assert 1 <= self.mtp_depth <= 2
        assert self.num_experts_per_tok <= self.num_experts

    @property
    def head_dim(self) -> int:
        assert self.hidden_size % self.num_attention_heads == 0
        return self.hidden_size // self.num_attention_heads

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def effective_intermediate_size(self) -> int:
        if self.block_auto_adjust_ff_dim:
            return _adjust_ff_dim(self.block_ff_dim, self.block_multiple_of, self.block_ffn_dim_multiplier)
        return self.block_ff_dim

    @property
    def num_attention_blocks(self) -> int:
        return sum(1 for t in self.layer_types if t == "full_attention")

    @property
    def num_conv_blocks(self) -> int:
        return self.num_hidden_layers - self.num_attention_blocks

    @property
    def num_kv_groups(self) -> int:
        return len(set(self.attn_group))

    @property
    def conv_state_len(self) -> int:
        if self.use_multiscale_conv:
            return (self.conv_dilated_kernel - 1) * self.conv_dilation  # 12 por defecto
        return self.conv_kernel_size - 1

    def attn_info_of_layer(self, layer_idx: int) -> tuple[int, str, int, bool] | None:
        """(attn_order, kind, group, is_producer) o None si la capa es conv."""
        if self.layer_types[layer_idx] != "full_attention":
            return None
        order = sum(1 for t in self.layer_types[:layer_idx] if t == "full_attention")
        return order, self.attn_kinds[order], self.attn_group[order], self.attn_is_producer[order]


_LFM2_16L_ATTN = [2, 5, 8, 10, 12, 14]
# Mismo layout de 30 capas que LFM2-2.6B (22 conv + 8 GQA).
_LFM2_30L_TYPES = [
    "conv", "conv", "full_attention", "conv", "conv",
    "full_attention", "conv", "conv", "conv", "full_attention",
    "conv", "conv", "conv", "full_attention", "conv",
    "conv", "conv", "full_attention", "conv", "conv",
    "conv", "full_attention", "conv", "conv",
    "full_attention", "conv", "conv", "full_attention",
    "conv", "conv",
]


def turing_tiny() -> TuringConfig:
    """Preset mínimo para tests/CPU (6 capas: 3 conv + 3 atenciones S,S,G)."""
    return TuringConfig(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        block_ff_dim=128,
        full_attn_idxs=[1, 2, 4],
        sliding_window=8,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        moe_shared_experts=1,
        max_position_embeddings=64,
    )


def turing_350m() -> TuringConfig:
    return TuringConfig(
        hidden_size=1024, num_hidden_layers=16, num_attention_heads=16,
        num_key_value_heads=8, block_ff_dim=6656, full_attn_idxs=list(_LFM2_16L_ATTN),
        sliding_window=2048, num_experts=32, num_experts_per_tok=6,
        moe_intermediate_size=512, moe_shared_experts=1,
    )


def turing_700m() -> TuringConfig:
    return TuringConfig(
        hidden_size=1536, num_hidden_layers=16, num_attention_heads=24,
        num_key_value_heads=8, block_ff_dim=10240, full_attn_idxs=list(_LFM2_16L_ATTN),
        sliding_window=2048, num_experts=32, num_experts_per_tok=6,
        moe_intermediate_size=768, moe_shared_experts=1,
    )


def turing_12b() -> TuringConfig:
    return TuringConfig(
        hidden_size=2048, num_hidden_layers=16, num_attention_heads=32,
        num_key_value_heads=8, block_ff_dim=12288, full_attn_idxs=list(_LFM2_16L_ATTN),
        sliding_window=2048, num_experts=48, num_experts_per_tok=6,
        moe_intermediate_size=1024, moe_shared_experts=1,
    )


def turing_26b() -> TuringConfig:
    return TuringConfig(
        hidden_size=2048, num_hidden_layers=30, num_attention_heads=32,
        num_key_value_heads=8, block_ff_dim=10752, block_auto_adjust_ff_dim=False,
        layer_types=list(_LFM2_30L_TYPES),
        sliding_window=2048, num_experts=32, num_experts_per_tok=8,
        moe_intermediate_size=1024, moe_shared_experts=1,
    )


_PRESETS = {
    "tiny": turing_tiny,
    "350m": turing_350m,
    "700m": turing_700m,
    "1.2b": turing_12b,
    "2.6b": turing_26b,
}


def get_preset(name: str) -> TuringConfig:
    key = name.strip().lower().removeprefix("turing-").removeprefix("turing")
    if key not in _PRESETS:
        raise ValueError(f"Preset desconocido {name!r}. Opciones: {sorted(_PRESETS)}")
    return _PRESETS[key]()

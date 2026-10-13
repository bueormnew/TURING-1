"""Utilidades TURING-1: conteo de parámetros y reporte de memoria.

El punto 6 del diseño ("4 bits sin perder detalle") se concreta aquí en dos
piezas honestas para v1:
1. La arquitectura evita matrices gigantes nuevas (MTP reusa la head atada,
   puertas acotadas, QK-Norm) para que el PTQ a 4 bits no degrade.
2. `memory_report` cuantifica el ahorro real (CLA + SWA + int4) y
   `QUANT_RECIPE` documenta la receta de cuantización prevista
   (el QAT queda para fase 2, solo si el gap PTQ lo justifica).
"""

from __future__ import annotations

from .config import TuringConfig

QUANT_RECIPE = """Receta de cuantización TURING-1 v1 (post-entreno + recovery):
- Pesos: int4 por grupos de 128 (estilo Q4_K / AWQ). Embeddings y head LM
  atada: int4 o int8 (tabla pequeña, sensible a outliers).
- Activaciones: dinámica int8 en inferencia (W4A8); la QK-Norm y las puertas
  sigmoide acotadas mantienen los outliers bajo control por diseño.
- Caché KV: int8 por cabeza (las capas globales) / fp16 si sobra memoria.
- Estados conv (12 vectores/capa): fp16 (son ~KB, no vale cuantizarlos).
- Recovery: 200-500 pasos de destilación KD a LR bajo sobre el modelo
  cuantizado si el gap en benchmarks supera el 1%.
- NO hacer en v1: QAT completo desde cero (coste alto, beneficio marginal
  sobre PTQ+recovery para pesos a 4 bits).
"""


def count_params(config: TuringConfig) -> dict[str, int]:
    """Parámetros totales vs activos por token (sin instanciar el modelo)."""
    d, ff = config.hidden_size, config.effective_intermediate_size
    nq, nkv, hd = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    mi = config.moe_intermediate_size
    E, k, s = config.num_experts, config.num_experts_per_tok, config.moe_shared_experts

    embed = config.vocab_size * d  # head atada: se cuenta una vez
    attn_q_out = d * nq * hd + nq * hd * d
    attn_kv = 2 * d * nkv * hd  # solo productoras CLA
    conv = 3 * d * d + d * config.conv_kernel_size + d * d + d  # +d: gate mix
    if config.use_multiscale_conv:
        conv += d * config.conv_dilated_kernel
    mlp = 3 * d * ff
    moe_layer_total = E * 3 * d * mi + s * 3 * d * mi + d * E
    moe_layer_active = (k + s) * 3 * d * mi
    norms = 2 * d

    backbone, active = 0, 0
    for i, t in enumerate(config.layer_types):
        if t == "full_attention":
            info = config.attn_info_of_layer(i)
            assert info is not None
            backbone += attn_q_out + (attn_kv if info[3] else 0)
            active += attn_q_out + attn_kv  # cómputo equivalente a denso
        else:
            backbone += conv
            active += conv
        backbone += norms
        active += norms
        if config.use_moe and i >= config.num_dense_layers:
            backbone += moe_layer_total
            active += moe_layer_active
        else:
            backbone += mlp
            active += mlp
    backbone += d  # norma final
    active += d
    mtp = config.mtp_depth * 2 * d * d if config.use_mtp else 0
    return {
        "embeddings_head": embed,
        "dense_backbone": backbone - 0,  # sin embeddings/head
        "moe_included_above": True,
        "mtp_heads": mtp,
        "total": embed + backbone + mtp,
        "active_per_token": active,
    }


def memory_report(config: TuringConfig, seq_len: int = 32768) -> dict:
    """Memoria estimada (bytes) a `seq_len` + comparativa con LFM2 denso.

    KV TURING: un caché por grupo CLA; grupos deslizantes capados a W.
    KV LFM2: un caché completo por capa de atención.
    """
    nkv, hd = config.num_key_value_heads, config.head_dim
    W = config.sliding_window
    per_tok_kv = nkv * hd * 2 * 2  # K+V en bf16

    kv_turing, kv_lfm2 = 0, 0
    for gid, all_sl in config.group_all_sliding.items():
        kv_turing += min(seq_len, W if all_sl else seq_len) * per_tok_kv
    kv_lfm2 = config.num_attention_blocks * seq_len * per_tok_kv

    conv_state = config.num_conv_blocks * config.hidden_size * config.conv_state_len * 2
    counts = count_params(config)
    total = counts["total"]
    return {
        "params_total": total,
        "params_active_per_token": counts["active_per_token"],
        "weights_bf16_MB": total * 2 / 1e6,
        "weights_int4_MB": total * 0.5 / 1e6,
        "kv_turing_MB": kv_turing / 1e6,
        "kv_lfm2_baseline_MB": kv_lfm2 / 1e6,
        "kv_reduction": 1 - kv_turing / max(kv_lfm2, 1),
        "conv_state_KB": conv_state / 1e3,
        "seq_len": seq_len,
        "sliding_window": W,
        "num_kv_groups": config.num_kv_groups,
    }


def print_report(config: TuringConfig, seq_len: int = 32768) -> None:
    r = memory_report(config, seq_len)
    print(f"TURING-1 @ ctx {r['seq_len']}:")
    print(f"  params totales: {r['params_total']/1e9:.2f}B | activos/token: {r['params_active_per_token']/1e9:.2f}B")
    print(f"  pesos bf16: {r['weights_bf16_MB']:.0f} MB | int4: {r['weights_int4_MB']:.0f} MB")
    print(f"  KV caché: {r['kv_turing_MB']:.0f} MB vs LFM2 {r['kv_lfm2_baseline_MB']:.0f} MB "
          f"(-{100*r['kv_reduction']:.0f}%, {r['num_kv_groups']} grupos, W={r['sliding_window']})")
    print(f"  estado conv: {r['conv_state_KB']:.0f} KB")

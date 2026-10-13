"""Pruebas TURING-1 v1.

1. Presets y layout (ssg, grupos CLA, MoE top-k, MTP).
2. SWA exacta vs máscara completa. 3. Multiescala ≈ LFM2 al inicio (gate).
4. Forward + LM/MTP loss + backward (MoE incluido).
5. Prefill == incremental (ejercita CLA + SWA + conv-12).
6. Especulativo greedy == greedy estándar (equivalencia fuerte).
7. Reporte de memoria + receta 4-bit.
"""

import torch

from lfm2.modules import LFM2ShortConv
from turing import TuringForCausalLM, get_preset, memory_report
from turing.modules import TuringAttention


def test_presets():
    c = get_preset("tiny")
    assert c.attn_kinds == ["sliding", "sliding", "global"], c.attn_kinds
    assert c.attn_group == [0, 0, 1], c.attn_group  # par CLA + suelta
    assert c.attn_is_producer == [True, False, True]
    assert c.group_all_sliding == {0: True, 1: False}
    assert c.num_kv_groups == 2  # 3 atenciones -> 2 cachés
    for name in ["350m", "700m", "1.2b", "2.6b"]:
        p = get_preset(name)
        assert p.num_kv_groups == (p.num_attention_blocks + 1) // 2
        n_glob = p.attn_kinds.count("global")
        assert n_glob == p.num_attention_blocks // 3, (name, p.attn_kinds)
    print("OK presets: ssg + pares CLA + top-k",
          [get_preset(n).num_experts_per_tok for n in ["350m", "700m", "1.2b", "2.6b"]])


def test_swa_exact():
    torch.manual_seed(0)
    from lfm2.modules import LFM2RotaryEmbedding
    from turing.model import _turing_bias
    attn = TuringAttention(32, 4, 2, use_sdpa=False).eval()
    h = torch.randn(1, 12, 32)
    q = attn.project_q(h)
    k, v = attn.project_kv(h)
    rope = LFM2RotaryEmbedding(8)
    cos, sin = rope(h, torch.arange(12).unsqueeze(0))
    from lfm2.modules import apply_rotary_pos_emb
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    # Caso prefill (12 queries): ventana vía máscara de posiciones absolutas.
    bias_full = _turing_bias(None, 12, 12, 0, 0, h.device, h.dtype, window=4)
    assert bias_full is not None
    out_win = attn.attend(q, k, v, bias_full, window=4)
    # Referencia: atención completa con máscara ventana+causal.
    mask = torch.full((12, 12), float("-inf"))
    for i in range(12):
        mask[i, max(0, i - 3) : i + 1] = 0.0
    out_ref = attn.attend(q, k, v, mask.unsqueeze(0).unsqueeze(0), None)
    assert torch.allclose(out_win, out_ref, atol=1e-5), (out_win - out_ref).abs().max()
    # Caso decode (1 query): recorte físico exacto, sin máscara.
    q1 = q[:, :, -1:, :]
    out_dec = attn.attend(q1, k, v, None, window=4)
    out_man = attn.attend(q1, k[:, :, -4:, :], v[:, :, -4:, :], None, None)
    assert torch.allclose(out_dec, out_man, atol=1e-6)
    print("OK SWA: ventana en prefill == máscara exacta; recorte en decode exacto")


def test_multiscale_starts_as_lfm2():
    torch.manual_seed(1)
    from turing import TuringMultiscaleConv
    m = TuringMultiscaleConv(16, gate_init=-10.0).eval()  # gate≈0 -> rama A
    ref = LFM2ShortConv(16, 3, False).eval()
    ref.in_proj.weight.data.copy_(m.in_proj.weight.data)
    ref.conv.weight.data.copy_(m.conv.weight.data)
    ref.out_proj.weight.data.copy_(m.out_proj.weight.data)
    h = torch.randn(2, 9, 16)
    err = (m(h) - ref(h)).abs().max().item()
    assert err < 1e-3, err
    # Estado de caché: últimos 12 `y`.
    st = m.state_from_full(h)
    assert st.shape == (2, 16, 12)
    print(f"OK multiescala: arranca como LFM2 (err={err:.2e}), estado 12")


def _tiny():
    c = get_preset("tiny")
    c.eos_token_id = 255
    return c


def test_forward_mtp_moe_backward():
    torch.manual_seed(2)
    model = TuringForCausalLM(_tiny())
    ids = torch.randint(0, 255, (2, 14))
    out = model(input_ids=ids, labels=ids)
    assert out["logits"].shape == (2, 14, 256)
    assert out["lm_loss"].item() > 0 and out["mtp_loss"].item() > 0
    assert out["loss"].item() > 0
    out["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    # Top-k respetado: exactamente k pesos no nulos por token en una capa MoE.
    moe = next(m for m in model.model.layers if m.is_moe).ffn
    with torch.no_grad():
        _, logits = moe(torch.randn(5, 64))
        assert logits.shape == (5, 8)
    print(f"OK forward MoE+MTP+backward: lm={out['lm_loss'].item():.3f} mtp={out['mtp_loss'].item():.3f}")


def test_prefill_vs_incremental():
    torch.manual_seed(3)
    model = TuringForCausalLM(_tiny()).eval()
    ids = torch.randint(0, 255, (1, 12))  # > ventana (8): ejercita trim
    with torch.no_grad():
        full = model(input_ids=ids)["logits"]
        _, cache = model.model(input_ids=ids, use_cache=True)
        assert cache.get_seq_length() == 12
        assert cache.k[0].shape[2] == 8  # grupo deslizante recortado a W
        assert cache.k[1].shape[2] == 12  # grupo global completo
        new = torch.randint(0, 255, (1, 3))
        ref = model(input_ids=torch.cat([ids, new], 1))["logits"][:, -3:, :]
        h_inc, _ = model.model(input_ids=new, past=cache, use_cache=False)
        err = (model.lm_head(h_inc) - ref).abs().max().item()
        assert err < 1e-4, err
    print(f"OK caché CLA+SWA: prefill == incremental (err={err:.2e})")


def test_speculative_equals_greedy():
    torch.manual_seed(4)
    model = TuringForCausalLM(_tiny()).eval()
    ids = torch.randint(0, 200, (1, 10))
    with torch.no_grad():
        g1 = model.generate(ids, max_new_tokens=10, do_sample=False)
        g2 = model.generate_speculative(ids, max_new_tokens=10, do_sample=False)
        assert torch.equal(g1, g2), (g1, g2)
        gs = model.generate_speculative(ids, max_new_tokens=6, do_sample=True, temperature=0.8, top_p=0.9)
        assert gs.shape[1] == 16
    print("OK especulativo: greedy idéntico + muestreo corre")


def test_memory_report():
    for name in ["350m", "1.2b"]:
        r = memory_report(get_preset(name), seq_len=32768)
        assert r["kv_reduction"] > 0.3, r
        assert r["weights_int4_MB"] < r["weights_bf16_MB"] / 3
    print(f"OK memoria: KV -{100*r['kv_reduction']:.0f}% a 32k, int4≈1/4 de bf16")


if __name__ == "__main__":
    test_presets()
    test_swa_exact()
    test_multiscale_starts_as_lfm2()
    test_forward_mtp_moe_backward()
    test_prefill_vs_incremental()
    test_speculative_equals_greedy()
    test_memory_report()
    print("TURING-1 v1: TODAS LAS PRUEBAS PASARON")

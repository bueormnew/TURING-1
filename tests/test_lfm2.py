"""Pruebas de la implementación densa LFM2 (sin MoE).

Cubre:
1. Presets y dimensiones efectivas (4608/6912/8192/10752).
2. Ecuación del bloque conv con doble puerta (vs referencia ingenua).
3. Causalidad de la conv (el futuro no afecta al pasado).
4. Forward completo + pérdida + backward.
5. Equivalencia prefill vs incremental (KV + estado conv).
6. Generación greedy determinista.
"""

import torch

from lfm2 import LFM2ForCausalLM, get_preset
from lfm2.modules import LFM2ShortConv


def test_presets():
    c350 = get_preset("350m")
    assert (c350.hidden_size, c350.num_hidden_layers) == (1024, 16)
    assert c350.effective_intermediate_size == 4608, c350.effective_intermediate_size
    assert c350.layer_types.count("full_attention") == 6
    assert c350.layer_types.count("conv") == 10
    c700 = get_preset("700m")
    assert c700.effective_intermediate_size == 6912, c700.effective_intermediate_size
    c12 = get_preset("1.2b")
    assert c12.effective_intermediate_size == 8192, c12.effective_intermediate_size
    c26 = get_preset("2.6b")
    assert (c26.num_hidden_layers, c26.effective_intermediate_size) == (30, 10752)
    assert c26.layer_types.count("full_attention") == 8
    print("OK presets: 4608/6912/8192/10752 + layouts 10+6 / 22+8")


def test_short_conv_equation():
    torch.manual_seed(0)
    conv = LFM2ShortConv(hidden_size=32, kernel_size=3, bias=False).eval()
    h = torch.randn(2, 8, 32)
    out = conv(h)
    # Referencia ingenua de la ecuación (B,C,x)=in_proj; y=B*x; z=conv causal; o=out(C*z)
    with torch.no_grad():
        bcx = conv.in_proj(h).transpose(-1, -2)
        B, C, x = bcx.chunk(3, dim=-2)
        y = B * x
        import torch.nn.functional as F

        z = F.conv1d(y, conv.conv.weight, conv.conv.bias, padding=2, groups=32)[:, :, :8]
        ref = conv.out_proj((C * z).transpose(-1, -2).contiguous())
    assert torch.allclose(out, ref, atol=1e-5), (out - ref).abs().max()
    print("OK conv: ecuación (B,C,x)/puertas/conv causal depthwise")


def test_conv_causality():
    torch.manual_seed(1)
    conv = LFM2ShortConv(hidden_size=16, kernel_size=3, bias=False).eval()
    h = torch.randn(1, 6, 16)
    o1 = conv(h)
    h2 = h.clone()
    h2[:, 3:, :] += 5.0  # perturbar el futuro
    o2 = conv(h2)
    # Los 3 primeros pasos solo ven pasado con ventana k=3 -> idénticos.
    assert torch.allclose(o1[:, :2, :], o2[:, :2, :], atol=1e-5)
    print("OK conv: causalidad (pasado invariante ante cambios futuros)")


def _tiny_config():
    c = get_preset("350m")
    c.vocab_size = 512
    c.hidden_size = 64
    c.num_attention_heads = 4
    c.num_key_value_heads = 2
    c.num_hidden_layers = 4
    c.full_attn_idxs = [1, 3]
    c.layer_types = None
    c.__post_init__()
    c.block_ff_dim = 128
    c.max_position_embeddings = 64
    return c


def test_forward_loss_backward():
    torch.manual_seed(2)
    model = LFM2ForCausalLM(_tiny_config())
    ids = torch.randint(0, 512, (2, 16))
    out = model(input_ids=ids, labels=ids)
    assert out["logits"].shape == (2, 16, 512)
    assert out["loss"].item() > 0
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)
    print(f"OK forward+loss+backward: loss={out['loss'].item():.3f}")


def test_prefill_vs_incremental():
    torch.manual_seed(3)
    model = LFM2ForCausalLM(_tiny_config()).eval()
    ids = torch.randint(0, 512, (1, 12))
    with torch.no_grad():
        full = model(input_ids=ids)["logits"]
        _, cache = model.model(input_ids=ids, use_cache=True)
        assert cache.get_seq_length() == 12
        # Decode incremental de 4 tokens nuevos y comparar contra full.
        new_ids = torch.randint(0, 512, (1, 4))
        ref = model(input_ids=torch.cat([ids, new_ids], dim=1))["logits"][:, -4:, :]
        h_inc, _ = model.model(input_ids=new_ids, past=cache, use_cache=False)
        inc = model.lm_head(h_inc)
        err = (ref - inc).abs().max().item()
        assert err < 1e-4, err
    print(f"OK caché: prefill == incremental (err={err:.2e})")


def test_generate():
    torch.manual_seed(4)
    cfg = _tiny_config()
    cfg.eos_token_id = 511
    model = LFM2ForCausalLM(cfg).eval()
    ids = torch.randint(0, 511, (1, 8))
    with torch.no_grad():
        g1 = model.generate(ids, max_new_tokens=8, do_sample=False)
        g2 = model.generate(ids, max_new_tokens=8, do_sample=False)
        assert torch.equal(g1, g2) and g1.shape[1] == 16
        gs = model.generate(ids, max_new_tokens=8, do_sample=True, temperature=0.7, top_p=0.9)
        assert gs.shape[1] == 16
    print("OK generate: greedy determinista + muestreo top-p")


if __name__ == "__main__":
    test_presets()
    test_short_conv_equation()
    test_conv_causality()
    test_forward_loss_backward()
    test_prefill_vs_incremental()
    test_generate()
    print("TODAS LAS PRUEBAS DENSAS PASARON")

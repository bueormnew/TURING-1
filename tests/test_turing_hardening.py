"""Endurecimiento TURING-1 para escalar a modelos reales (solo añade tests;
refinamientos del fuente en esta ronda: expert_bias persistente,
TuringMoE.update_bias_ y moe_z_loss_weight opcional).

R1. Salud a 12 capas: std de activaciones y grads por capa (sin explosión /
    desvanecimiento; escala MoE vs denso acotada).
R2. bf16: prefill==incremental + cercanía bf16 vs fp32.
R3. fp16 forward a ancho moderado (d=256, lo típico en GPU).
R4. Invarianza RoPE con offset grande (30000: contexto largo real).
R5. Incremental paso a paso 5->20, error en CADA paso (frontera del trim).
R6. Combinado máscara+MTP+labels(-100) con backward.
R7. Extremos de muestreo (temp 0.1/5.0, top_p 0.01, min_p 0.5, rep 2.0).
R8. expert_bias persiste en checkpoints + update_bias_ mueve el signo bien.
R9. z-loss: off ausente, on finito con backward; flag del backbone.
R10. Config ancha (d=256, E=16 top-4) forward+backward finitos.
"""

import torch

from turing import TuringForCausalLM
from turing.config import TuringConfig


def _base(**kw) -> TuringConfig:
    args = dict(
        vocab_size=256, hidden_size=32, num_hidden_layers=6,
        num_attention_heads=4, num_key_value_heads=2, block_ff_dim=64,
        full_attn_idxs=[1, 2, 4], sliding_window=8,
        num_experts=8, num_experts_per_tok=2, moe_intermediate_size=16,
        moe_shared_experts=1, mtp_depth=1, max_position_embeddings=128,
    )
    args.update(kw)
    return TuringConfig(**args)


def _finite(t: torch.Tensor, name: str) -> None:
    bad = (~torch.isfinite(t)).sum().item()
    assert bad == 0, f"{name}: {bad} no finitos"


# ------------------------------------------------------- R1. salud 12 capas
def test_r1_depth_health():
    torch.manual_seed(501)
    cfg = _base(num_hidden_layers=12, full_attn_idxs=[1, 2, 4, 7, 8, 10])
    assert cfg.attn_kinds == ["sliding", "sliding", "global"] * 2
    model = TuringForCausalLM(cfg)
    stream_in: dict[int, float] = {}
    ffn_out: dict[int, float] = {}
    for i, ly in enumerate(model.model.layers):
        ly.operator_norm.register_forward_hook(
            lambda m, inp, out, _i=i: stream_in.__setitem__(_i, inp[0].detach().float().std().item()))
        ly.ffn.register_forward_hook(
            lambda m, inp, out, _i=i: ffn_out.__setitem__(
                _i, (out[0] if isinstance(out, tuple) else out).detach().float().std().item()))
    final_in: dict[str, float] = {}
    model.model.final_norm.register_forward_hook(
        lambda m, inp, out: final_in.__setitem__("s", inp[0].detach().float().std().item()))
    ids = torch.randint(0, 255, (2, 24))
    loss = model(input_ids=ids, labels=ids)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    print("    capa | stream_in | grad_std")
    gstd: dict[str, list] = {}
    for n, p in model.named_parameters():
        if p.grad is not None:
            _finite(p.grad, f"R1.grad[{n}]")
            key = ".".join(n.split(".")[:3]) if n.startswith("model.layers") else n.split(".")[0]
            gstd.setdefault(key, []).append(p.grad.detach().float().reshape(-1))
    s_vals = [stream_in[i] for i in range(12)] + [final_in["s"]]
    assert all(1e-2 < v < 1e2 for v in s_vals), s_vals
    moe_sc, dense_sc = [], []
    for i, ly in enumerate(model.model.layers):
        gs = torch.cat(gstd.get(f"model.layers.{i}", [torch.tensor(float("nan"))])).std().item()
        print(f"    {i:4d} | {stream_in[i]:9.4f} | {gs:.2e}")
        (moe_sc if ly.is_moe else dense_sc).append(ffn_out[i])
    import statistics
    ratio = statistics.mean(moe_sc) / max(statistics.mean(dense_sc), 1e-12)
    assert ratio < 10, f"salida MoE {ratio:.1f}x la densa"
    vals = [torch.cat(g).std().item() for g in gstd.values()]
    assert max(vals) / max(min(vals), 1e-12) < 1e6, vals
    print(f"OK R1: 12 capas estables (stream {s_vals[0]:.3f}->{s_vals[-1]:.3f}, MoE/denso={ratio:.2f}x)")


# ------------------------------------------------------------- R2. bf16 ==
def test_r2_bf16_consistency():
    torch.manual_seed(502)
    cfg = _base()
    m32 = TuringForCausalLM(cfg).eval()
    m16 = TuringForCausalLM(cfg).to(torch.bfloat16).eval()
    m16.load_state_dict({k: v.to(torch.bfloat16) if v.is_floating_point() else v
                         for k, v in m32.state_dict().items()})
    ids = torch.randint(0, 200, (1, 20))
    new = torch.randint(0, 200, (1, 5))
    with torch.no_grad():
        ref32, _ = m32.model(input_ids=torch.cat([ids, new], 1), use_cache=False)
        _, c16 = m16.model(input_ids=ids, use_cache=True)
        assert c16 is not None
        h_inc, _ = m16.model(input_ids=new, past=c16, use_cache=False)
        full16, _ = m16.model(input_ids=torch.cat([ids, new], 1), use_cache=False)
    e_inc = (full16[:, -5:, :].float() - h_inc.float()).abs().max().item()
    e_dt = (full16.float() - ref32).abs().max().item()
    assert e_inc < 0.1, e_inc
    assert e_dt < 0.5, e_dt
    print(f"OK R2: bf16 incremental==full ({e_inc:.2e}), bf16 vs fp32 ({e_dt:.2e})")


# ---------------------------------------------------------- R3. fp16 ancho
def test_r3_fp16_wide():
    torch.manual_seed(503)
    cfg = _base(vocab_size=1024, hidden_size=256, num_hidden_layers=4,
                num_attention_heads=8, num_key_value_heads=4, block_ff_dim=512,
                full_attn_idxs=[1, 2], sliding_window=32)
    model = TuringForCausalLM(cfg).to(torch.float16).eval()
    ids = torch.randint(0, 1024, (1, 24))
    with torch.no_grad():
        out = model(input_ids=ids, labels=ids)
    _finite(out["logits"].float(), "R3.logits_fp16")
    assert torch.isfinite(out["loss"]), "R3.loss_fp16"
    print(f"OK R3: fp16 d=256 finito (max={out['logits'].abs().max().item():.1f})")


# ---------------------------------------------------- R4. offset RoPE largo
def test_r4_rope_far_offset():
    torch.manual_seed(504)
    model = TuringForCausalLM(_base()).eval()
    ids = torch.randint(0, 200, (1, 16))
    p0 = torch.arange(16).unsqueeze(0)
    pB = (30000 + torch.arange(16)).unsqueeze(0)
    with torch.no_grad():
        h0, _ = model.model(input_ids=ids, position_ids=p0, use_cache=False)
        hB, _ = model.model(input_ids=ids, position_ids=pB, use_cache=False)
    err = (h0 - hB).abs().max().item()
    assert err < 1e-3, err
    print(f"OK R4: RoPE relativo invariante en pos 30000 (err={err:.2e})")


# ---------------------------------------------------- R5. cada paso incremental
def test_r5_every_step():
    torch.manual_seed(505)
    model = TuringForCausalLM(_base()).eval()
    ids = torch.randint(0, 200, (1, 5))
    tail = torch.randint(0, 200, (1, 15))
    worst = 0.0
    with torch.no_grad():
        _, cache = model.model(input_ids=ids, use_cache=True)
        assert cache is not None
        for t in range(15):
            h_step, cache = model.model.step(tail[:, t : t + 1], cache)
            full, _ = model.model(input_ids=torch.cat([ids, tail[:, : t + 1]], 1), use_cache=False)
            worst = max(worst, (full[:, -1:, :] - h_step).abs().max().item())
    assert worst < 1e-3, worst
    print(f"OK R5: 15 pasos 5->20, peor err={worst:.2e} (cruza el trim W=8)")


# ------------------------------------------------------- R6. todo combinado
def test_r6_combined_mask_mtp():
    torch.manual_seed(506)
    model = TuringForCausalLM(_base())
    ids = torch.tensor([[10, 20, 30, 40, 50, 0, 0, 0, 0, 0], [7, 8, 9, 0, 0, 0, 0, 0, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]])
    out = model(input_ids=ids, attention_mask=mask, labels=torch.where(mask.bool(), ids, -100))
    _finite(out["logits"], "R6.logits")
    assert torch.isfinite(out["loss"]) and out["mtp_loss"] is not None
    assert torch.isfinite(out["mtp_loss"])
    out["loss"].backward()
    for n, p in model.named_parameters():
        if p.grad is not None:
            _finite(p.grad, f"R6.grad[{n}]")
    print(f"OK R6: máscara+MTP+-100 combinados (loss={out['loss'].item():.3f})")


# ----------------------------------------------------- R7. extremos muestreo
def test_r7_sample_extremes():
    torch.manual_seed(507)
    model = TuringForCausalLM(_base()).eval()
    ids = torch.randint(0, 200, (1, 6))
    with torch.no_grad():
        for temp, top_p, min_p, rep in [(0.1, 1.0, 0.0, 1.0), (5.0, 0.5, 0.0, 1.0),
                                        (1.0, 0.01, 0.0, 1.0), (1.0, 1.0, 0.5, 2.0)]:
            for fn in (model.generate, model.generate_speculative):
                # eos imposible (9999): se verifica longitud completa, no parada temprana.
                g = fn(ids, max_new_tokens=4, do_sample=True, temperature=temp,
                       top_p=top_p, min_p=min_p, repetition_penalty=rep, eos_token_id=9999)
                assert g.shape == (1, 10) and bool(((g >= 0) & (g < 256)).all()), (temp, top_p, g.shape)
    print("OK R7: temp 0.1/5.0, top_p 0.01, min_p 0.5, rep 2.0 sin fallos")


# ------------------------------------------------- R8. bias persiste y actúa
def test_r8_bias_persist_update():
    torch.manual_seed(508)
    import tempfile
    model = TuringForCausalLM(_base()).eval()
    moes = [ly.ffn for ly in model.model.layers if ly.is_moe]
    assert len(moes) == 4
    with torch.no_grad():
        for j, m in enumerate(moes):
            m.expert_bias.copy_(0.005 + 0.01 * torch.arange(m.num_experts))
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    assert "model.layers.2.ffn.expert_bias" in model.state_dict(), "bias fuera del checkpoint!"
    torch.save(model.state_dict(), path)
    fresh = TuringForCausalLM(_base())
    fresh.load_state_dict(torch.load(path, weights_only=True), strict=True)
    for a, b in zip(moes, [ly.ffn for ly in fresh.model.layers if ly.is_moe]):
        assert torch.equal(a.expert_bias, b.expert_bias)
    m = moes[0]
    before = m.expert_bias.clone()
    idx = torch.zeros(10, m.top_k, dtype=torch.long)  # todo al experto 0
    m.update_bias_(idx, lr=0.01)
    assert abs(m.expert_bias[0].item() - (before[0].item() - 0.01)) < 1e-7
    assert all(abs(v.item() - (b + 0.01)) < 1e-7 for v, b in zip(m.expert_bias[1:], before[1:].tolist()))
    print("OK R8: bias persiste en ckpt + update aux-free con signo correcto")


# ---------------------------------------------------------------- R9. z-loss
def test_r9_z_loss():
    torch.manual_seed(509)
    m0 = TuringForCausalLM(_base())
    ids = torch.randint(0, 255, (2, 14))
    out0 = m0(input_ids=ids, labels=ids)
    assert out0["z_loss"] is None, "con peso 0 no debe existir z-loss"
    h, c = m0.model(input_ids=ids)
    assert isinstance((h, c), tuple) and len((h, c)) == 2, "compatibilidad 2-tupla"
    torch.manual_seed(509)
    m1 = TuringForCausalLM(_base(moe_z_loss_weight=1e-3))
    out1 = m1(input_ids=ids, labels=ids)
    assert out1["z_loss"] is not None and torch.isfinite(out1["z_loss"]) and out1["z_loss"] > 0
    expect = out1["lm_loss"] + 0.2 * out1["mtp_loss"] + 1e-3 * out1["z_loss"]
    assert torch.allclose(out1["loss"], expect, atol=1e-6)
    out1["loss"].backward()
    for n, p in m1.named_parameters():
        if p.grad is not None:
            _finite(p.grad, f"R9.grad[{n}]")
    h2, _, rlogs = m1.model(input_ids=ids, return_router_logits=True)
    assert len(rlogs) == 4 and all(r.shape == (2, 14, 8) for r in rlogs)
    assert all(torch.isfinite(r).all() for r in rlogs)
    print(f"OK R9: z-loss off=None, on={out1['z_loss'].item():.4f} con backward + flag backbone")


# ------------------------------------------------------------- R10. ancho
def test_r10_wide():
    torch.manual_seed(510)
    cfg = _base(vocab_size=1024, hidden_size=256, num_hidden_layers=4,
                num_attention_heads=8, num_key_value_heads=4, block_ff_dim=512,
                full_attn_idxs=[1, 2], sliding_window=32,
                num_experts=16, num_experts_per_tok=4, moe_intermediate_size=128)
    model = TuringForCausalLM(cfg)
    n = sum(p.numel() for p in model.parameters())
    ids = torch.randint(0, 1024, (2, 32))
    out = model(input_ids=ids, labels=ids)
    _finite(out["logits"], "R10.logits")
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            _finite(p.grad, f"R10.grad[{name}]")
    print(f"OK R10: d=256 E=16 top-4 ({n/1e6:.1f}M params) fwd+bwd finitos")


if __name__ == "__main__":
    suite = [test_r1_depth_health, test_r2_bf16_consistency, test_r3_fp16_wide,
             test_r4_rope_far_offset, test_r5_every_step, test_r6_combined_mask_mtp,
             test_r7_sample_extremes, test_r8_bias_persist_update, test_r9_z_loss,
             test_r10_wide]
    failed = []
    for fn in suite:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"FALLO {fn.__name__}: {type(e).__name__}: {e}")
            failed.append(fn.__name__)
    print(f"--- {len(suite) - len(failed)}/{len(suite)} pasaron ---")
    if failed:
        print("FALLARON:", failed)
        raise SystemExit(1)
    print("TURING-1 ENDURECIMIENTO: TODO SANO")

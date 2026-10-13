"""Tests de estabilidad y estrés TURING-1 v1 (solo lectura del código fuente).

Objetivo: cazar NaN/inf, fallos y rupturas de causalidad bajo condiciones
fuertes, sin modificar la arquitectura:
A. Barrido de finitud (matriz de flags, bf16, padding, gradientes).
B. Causalidad (extremo a extremo, ventana larga, loop incremental largo,
   invarianza por offset, anti-degeneración).
C. Conv multiescala: step() incremental == forward completo (taps dilatados).
D. MTP/Speculative robusto (depth=2, labels con -100, muestreo multiseed,
   bordes: 0 tokens, prompt de 1 token, MTP desactivado).
E. Batch + padding (invarianza ante padding).
F. Estrés (60 tokens greedy, prefill 300 con trim, snapshot/restore).
G. Invariantes MoE/router + determinismo.
"""

import dataclasses

import torch

from turing import TuringForCausalLM, get_preset
from turing.config import TuringConfig
from turing.modules import TuringMultiscaleConv


def _base(**kw) -> TuringConfig:
    args = dict(
        vocab_size=256,
        hidden_size=32,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        block_ff_dim=64,
        full_attn_idxs=[1, 2, 4],
        sliding_window=8,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        moe_shared_experts=1,
        mtp_depth=1,
        max_position_embeddings=128,
    )
    args.update(kw)
    return TuringConfig(**args)


def _assert_finite(t: torch.Tensor, name: str) -> None:
    bad = (~torch.isfinite(t)).sum().item()
    assert bad == 0, f"{name}: {bad} valores no finitos (max={t.abs().max().item()})"


def _assert_all_grads_finite(model: torch.nn.Module, name: str) -> None:
    n = 0
    for pname, p in model.named_parameters():
        if p.grad is not None:
            n += 1
            _assert_finite(p.grad, f"{name}.grad[{pname}]")
    assert n > 0, f"{name}: ningún parámetro recibió gradiente"


# ---------------------------------------------------------------- A. finitud
def test_a1_flag_matrix():
    torch.manual_seed(11)
    flags = ["kv_share_pairs", "use_multiscale_conv", "use_moe", "use_mtp", "use_sdpa"]
    combos = []
    for mask in range(2 ** len(flags)):
        combos.append({f: bool(mask & (1 << i)) for i, f in enumerate(flags)})
    for ci, kw in enumerate(combos):
        cfg = _base(**kw)
        # Sin MoE/MTP también debe correr con depth>1 pedido (se ignora sin MTP).
        model = TuringForCausalLM(cfg)
        ids = torch.randint(0, 255, (2, 12))
        out = model(input_ids=ids, labels=ids)
        _assert_finite(out["logits"], f"A1[{ci}].logits")
        assert out["loss"] is not None and torch.isfinite(out["loss"]), f"A1[{ci}].loss"
        if kw["use_mtp"]:
            assert out["mtp_loss"] is not None and torch.isfinite(out["mtp_loss"]), f"A1[{ci}].mtp"
        model.zero_grad()
        out["loss"].backward()
        _assert_all_grads_finite(model, f"A1[{ci}]")
    print(f"OK A1: matriz 2^5={len(combos)} combinaciones de flags, logits+loss+grads finitos")


def test_a2_bf16_forward():
    torch.manual_seed(12)
    model = TuringForCausalLM(_base()).to(torch.bfloat16).eval()
    ids = torch.randint(0, 255, (1, 16))
    with torch.no_grad():
        out = model(input_ids=ids, labels=ids)
    _assert_finite(out["logits"].float(), "A2.logits_bf16")
    assert torch.isfinite(out["loss"]), "A2.loss_bf16"
    print("OK A2: forward bf16 finito (loss incluido)")


def test_a3_padding_grads():
    torch.manual_seed(13)
    model = TuringForCausalLM(_base())
    ids = torch.tensor([[5, 7, 9, 11, 0, 0], [3, 4, 0, 0, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0]])
    out = model(input_ids=ids, attention_mask=mask, labels=torch.where(mask.bool(), ids, -100))
    _assert_finite(out["logits"], "A3.logits_pad")
    assert torch.isfinite(out["loss"]), "A3.loss_pad"
    out["loss"].backward()
    _assert_all_grads_finite(model, "A3")
    print("OK A3: padding con máscara + grads finitos")


# ------------------------------------------------------------- B. causalidad
def test_b1_end_to_end_causality():
    torch.manual_seed(21)
    model = TuringForCausalLM(_base()).eval()
    ids = torch.randint(0, 255, (1, 16))
    with torch.no_grad():
        ref = model(input_ids=ids)["logits"]
        ids2 = ids.clone()
        ids2[:, 6:] = torch.randint(0, 255, (1, 10))
        alt = model(input_ids=ids2)["logits"]
    err = (ref[:, :6, :] - alt[:, :6, :]).abs().max().item()
    assert err < 1e-5, err
    print(f"OK B1: pasado invariante ante futuro (err={err:.2e})")


def test_b2_long_window_causality():
    torch.manual_seed(22)
    model = TuringForCausalLM(_base()).eval()  # W=8, 3 conv + S,S,G
    ids = torch.randint(0, 255, (1, 40))
    with torch.no_grad():
        ref = model(input_ids=ids)["logits"]
        ids2 = ids.clone()
        ids2[:, 30:] = torch.randint(0, 255, (1, 10))
        alt = model(input_ids=ids2)["logits"]
    err = (ref[:, :20, :] - alt[:, :20, :]).abs().max().item()
    assert err < 1e-5, err
    print(f"OK B2: causalidad con seq 40 > ventana 8 (err={err:.2e})")


def test_b3_long_incremental_loop():
    torch.manual_seed(23)
    model = TuringForCausalLM(_base()).eval()
    ids = torch.randint(0, 255, (1, 8))
    with torch.no_grad():
        _, cache = model.model(input_ids=ids, use_cache=True)
        assert cache is not None
        tails = torch.randint(0, 255, (1, 25))
        h_steps = []
        for t in range(25):
            h, cache = model.model.step(tails[:, t : t + 1], cache)
            h_steps.append(h)
        h_inc = torch.cat(h_steps, dim=1)
        full, _ = model.model(input_ids=torch.cat([ids, tails], dim=1), use_cache=False)
    err = (full[:, 8:, :] - h_inc).abs().max().item()
    assert err < 1e-3, err
    print(f"OK B3: 25 pasos incrementales == full (err={err:.2e}, con trim+CLA+conv12)")


def test_b4_offset_invariance_with_trim():
    torch.manual_seed(24)
    model = TuringForCausalLM(_base()).eval()  # prefill 12 > W=8 recorta grupo 0
    ids = torch.randint(0, 255, (1, 12))
    new = torch.randint(0, 255, (1, 5))
    with torch.no_grad():
        _, cache = model.model(input_ids=ids, use_cache=True)
        assert cache is not None and cache.k[0].shape[2] == 8
        h_inc, _ = model.model(input_ids=new, past=cache, use_cache=False)
        ref, _ = model.model(input_ids=torch.cat([ids, new], 1), use_cache=False)
    err = (ref[:, -5:, :] - h_inc).abs().max().item()
    assert err < 1e-4, err
    print(f"OK B4: offsets RoPE correctos con caché recortada (err={err:.2e})")


def test_b5_sensitivity_antidegeneracy():
    torch.manual_seed(25)
    model = TuringForCausalLM(_base()).eval()
    ids = torch.randint(0, 255, (1, 16))
    with torch.no_grad():
        ref = model(input_ids=ids)["logits"]
        ids2 = ids.clone()
        ids2[:, :4] = (ids2[:, :4] + 37) % 255
        alt = model(input_ids=ids2)["logits"]
    diff = (ref[:, 8:, :] - alt[:, 8:, :]).abs().max().item()
    assert diff > 1e-6, "el pasado NO afecta al futuro: máscara degenerada?"
    print(f"OK B5: el pasado sí fluye (cambio={diff:.2e}, anti-degeneración)")


# ------------------------------------------------------- C. conv step==full
def test_c1_dilated_step_matches_full():
    for gate_init in (-3.0, 0.0):  # identidad LFM2 y mezcla 50/50
        torch.manual_seed(31)
        conv = TuringMultiscaleConv(32, gate_init=gate_init).eval()
        h = torch.randn(2, 13, 32)
        with torch.no_grad():
            full = conv(h)
            state = conv.state_from_full(h[:, :10, :])
            assert state.shape == (2, 32, 12)
            outs = []
            for t in range(10, 13):
                o, state = conv.step(h[:, t : t + 1, :], state)
                outs.append(o)
            inc = torch.cat(outs, dim=1)
        err = (full[:, 10:, :] - inc).abs().max().item()
        assert err < 1e-5, (gate_init, err)
    print(f"OK C1: step() dilatado == full (err={err:.2e}, gates -3 y 0)")


def test_c2_gate_bounds():
    torch.manual_seed(32)
    conv = TuringMultiscaleConv(16).eval()
    g = torch.sigmoid(conv.mix.detach())
    assert bool(((g > 0) & (g < 1)).all()), "gate fuera de (0,1)"
    assert abs(g.mean().item() - 0.0474) < 1e-3, g.mean().item()  # sigmoid(-3)
    print(f"OK C2: gate acotado, media={g.mean().item():.4f} (≈sigmoid(-3))")


# ----------------------------------------------------- D. MTP/especulativo
def test_d1_mtp_depth2():
    torch.manual_seed(41)
    cfg = _base(mtp_depth=2)
    model = TuringForCausalLM(cfg)
    ids = torch.randint(0, 255, (1, 14))
    out = model(input_ids=ids, labels=ids)
    assert out["mtp_loss"] is not None and torch.isfinite(out["mtp_loss"])
    out["loss"].backward()
    _assert_all_grads_finite(model, "D1")
    assert len(model.mtp_heads) == 2
    with torch.no_grad():
        s = model.generate_speculative(torch.randint(0, 200, (1, 6)), max_new_tokens=6, do_sample=False)
        assert s.shape[1] == 12
    print(f"OK D1: MTP depth=2 (mtp={out['mtp_loss'].item():.3f}) + especulativo")


def test_d2_labels_with_ignore():
    torch.manual_seed(42)
    model = TuringForCausalLM(_base())
    ids = torch.randint(0, 255, (2, 14))
    labels = ids.clone()
    labels[:, ::3] = -100  # ignora también zona de inputs/targets MTP
    out = model(input_ids=ids, labels=labels)
    assert torch.isfinite(out["loss"]) and torch.isfinite(out["mtp_loss"]), (out["loss"], out["mtp_loss"])
    out["loss"].backward()
    _assert_all_grads_finite(model, "D2")
    print("OK D2: labels con -100 (incl. zona MTP) finitos")


def test_d3_speculative_sampling_multiseed():
    cfg = _base()
    cfg.eos_token_id = 255
    model = TuringForCausalLM(cfg).eval()
    ids = torch.randint(0, 200, (1, 10))
    with torch.no_grad():
        for seed in (1, 2, 3):
            torch.manual_seed(seed)
            s = model.generate_speculative(ids, max_new_tokens=8, do_sample=True,
                                           temperature=0.8, top_p=0.9, min_p=0.05,
                                           repetition_penalty=1.1)
            # Muestreo: EOS puede parar antes (longitud variable por diseño).
            assert 10 < s.shape[1] <= 18, s.shape
            assert bool(((s >= 0) & (s < 256)).all()), "ids fuera de vocab"
            torch.manual_seed(seed)  # misma seed -> idéntico (determinismo)
            s2 = model.generate_speculative(ids, max_new_tokens=8, do_sample=True,
                                            temperature=0.8, top_p=0.9, min_p=0.05,
                                            repetition_penalty=1.1)
            assert torch.equal(s, s2), "muestreo no determinista con misma seed"
    print("OK D3: especulativo con muestreo, 3 seeds, rangos + determinismo")


def test_d4_sample_param_matrix():
    torch.manual_seed(44)
    model = TuringForCausalLM(_base()).eval()
    ids = torch.randint(0, 200, (1, 6))
    with torch.no_grad():
        for temp, top_p, min_p, rep in [(0.5, 1.0, 0.0, 1.0), (2.0, 0.8, 0.1, 1.2), (1.0, 0.95, 0.0, 1.0)]:
            torch.manual_seed(1000 + int(temp * 10))
            g = model.generate(ids, max_new_tokens=4, do_sample=True, temperature=temp,
                               top_p=top_p, min_p=min_p, repetition_penalty=rep)
            torch.manual_seed(1000 + int(temp * 10))
            g2 = model.generate(ids, max_new_tokens=4, do_sample=True, temperature=temp,
                                top_p=top_p, min_p=min_p, repetition_penalty=rep)
            assert torch.equal(g, g2), "generate con muestreo no determinista"
            torch.manual_seed(2000 + int(temp * 10))
            s = model.generate_speculative(ids, max_new_tokens=4, do_sample=True, temperature=temp,
                                           top_p=top_p, min_p=min_p, repetition_penalty=rep)
            assert 6 < g.shape[1] <= 10 and 6 < s.shape[1] <= 10, (g.shape, s.shape)
            assert bool(((g >= 0) & (g < 256)).all()) and bool(((s >= 0) & (s < 256)).all())
    print("OK D4: matriz temp/top-p/min-p/repetición en ambos generates (rango+determinismo)")


def test_d5_edges():
    torch.manual_seed(45)
    model = TuringForCausalLM(_base()).eval()
    ids = torch.randint(0, 200, (1, 8))
    with torch.no_grad():
        g0 = model.generate(ids, max_new_tokens=0)
        assert torch.equal(g0, ids), "0 tokens debe devolver el prompt intacto"
        g1 = model.generate(torch.randint(0, 200, (1, 1)), max_new_tokens=5, do_sample=False)
        assert g1.shape == (1, 6)
        s1 = model.generate_speculative(torch.randint(0, 200, (1, 1)), max_new_tokens=5, do_sample=False)
        assert s1.shape == (1, 6)
    print("OK D5: bordes (0 tokens, prompt de 1 token) en ambos modos")


def test_d6_no_mtp_raises():
    torch.manual_seed(46)
    model = TuringForCausalLM(_base(use_mtp=False)).eval()
    ids = torch.randint(0, 200, (1, 6))
    with torch.no_grad():
        try:
            model.generate_speculative(ids, max_new_tokens=4)
        except AssertionError:
            print("OK D6: sin MTP el especulativo falla con AssertionError (esperado)")
        else:
            raise AssertionError("D6: debió fallar sin MTP")


# ------------------------------------------------------------- E. batch/pad
def test_e1_batch():
    torch.manual_seed(51)
    model = TuringForCausalLM(_base()).eval()
    ids = torch.randint(0, 200, (3, 10))
    with torch.no_grad():
        out = model(input_ids=ids, labels=ids)
        _assert_finite(out["logits"], "E1.logits")
        assert torch.isfinite(out["loss"])
        g = model.generate(ids[:2], max_new_tokens=5, do_sample=False)
        assert g.shape == (2, 15)
    print("OK E1: batch=3 forward+loss, batch=2 generate")


def test_e2_padding_invariance():
    torch.manual_seed(52)
    model = TuringForCausalLM(_base()).eval()
    core = torch.randint(1, 200, (1, 5))
    pad_ids = torch.cat([core, torch.zeros(1, 3, dtype=torch.long)], dim=1)
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]])
    with torch.no_grad():
        a = model(input_ids=pad_ids, attention_mask=mask)["logits"][:, :5, :]
        b = model(input_ids=core)["logits"]
    err = (a - b).abs().max().item()
    assert err < 1e-4, err
    print(f"OK E2: padding no contamina (err={err:.2e})")


# ----------------------------------------------------------------- F. estrés
def test_f1_long_greedy():
    torch.manual_seed(61)
    cfg = _base()
    cfg.eos_token_id = 255
    model = TuringForCausalLM(cfg).eval()
    ids = torch.randint(0, 200, (1, 4))
    with torch.no_grad():
        g = model.generate(ids, max_new_tokens=60, do_sample=False)
        assert g.shape[1] == 64, g.shape
        assert bool(((g >= 0) & (g < 256)).all())
        s = model.generate_speculative(ids, max_new_tokens=60, do_sample=False)
        assert torch.equal(g, s), "divergencia especulativa en secuencia larga"
    print("OK F1: 60 tokens greedy == especulativo, rangos válidos")


def test_f2_long_prefill_trim():
    torch.manual_seed(62)
    model = TuringForCausalLM(_base()).eval()
    ids = torch.randint(0, 255, (1, 300))
    with torch.no_grad():
        hidden, cache = model.model(input_ids=ids, use_cache=True)
        assert cache is not None
        _assert_finite(hidden, "F2.hidden300")
        assert cache.k[0].shape[2] == 8, cache.k[0].shape      # sliding recortado
        assert cache.k[1].shape[2] == 300, cache.k[1].shape    # global completo
        assert cache.get_conv(0).shape == (1, 32, 12)
    print("OK F2: prefill 300 (37x ventana), trim y estados correctos")


def test_f3_snapshot_restore():
    torch.manual_seed(63)
    model = TuringForCausalLM(_base()).eval()
    ids = torch.randint(0, 200, (1, 6))
    tail = torch.randint(0, 200, (1, 3))
    with torch.no_grad():
        _, cache = model.model(input_ids=ids, use_cache=True)
        assert cache is not None
        snap = cache.snapshot()
        # Tres avances idénticos desde el snapshot: reproducibilidad total.
        hb, cache_b = model.model(input_ids=tail[:, :2], past=snap, use_cache=True)
        hc, cache_c = model.model(input_ids=tail[:, :2], past=snap, use_cache=True)
        assert torch.equal(hb, hc), "snapshot no reproducible"
        for gid in cache_b.k:
            assert torch.equal(cache_b.k[gid], cache_c.k[gid]), f"KV difiere grupo {gid}"
        # El snapshot original sigue intacto (longitud 6).
        assert snap.get_seq_length() == 6
    print("OK F3: snapshot independiente y reproducible")


# ------------------------------------------------------- G. MoE/determinismo
def test_g1_router_invariants():
    torch.manual_seed(71)
    cfg = _base(num_experts=12, num_experts_per_tok=6)
    model = TuringForCausalLM(cfg).eval()
    moe = next(lyr.ffn for lyr in model.model.layers if lyr.is_moe)
    with torch.no_grad():
        h = torch.randn(7, 32)
        out, logits = moe(h)
        _assert_finite(out, "G1.moe_out")
        scores = torch.softmax(logits, -1)
        topw, idx = torch.topk(scores + moe.expert_bias, k=6, dim=-1)
        assert bool(((idx >= 0) & (idx < 12)).all()), "índices fuera de rango"
        w = torch.gather(scores, 1, idx)
        w = w / (w.sum(-1, keepdim=True) + 1e-6)
        assert (w.sum(-1) - 1).abs().max().item() < 1e-5, "pesos no suman 1"
    print("OK G1: top-6 válido, índices en rango, pesos renormalizados")


def test_g2_shared_expert_learns():
    torch.manual_seed(72)
    model = TuringForCausalLM(_base())
    ids = torch.randint(0, 255, (2, 12))
    model(input_ids=ids, labels=ids)["loss"].backward()
    n = 0
    for lyr in model.model.layers:
        if lyr.is_moe:
            for m in lyr.ffn.shared:
                assert m.w1.weight.grad is not None, "shared sin grad"
                _assert_finite(m.w1.weight.grad, "G2.shared_grad")
                n += 1
    assert n > 0
    print(f"OK G2: {n} expertos compartidos con gradiente finito")


def test_g3_determinism():
    torch.manual_seed(73)
    model = TuringForCausalLM(_base()).eval()
    ids = torch.randint(0, 255, (1, 12))
    with torch.no_grad():
        a = model(input_ids=ids)["logits"]
        b = model(input_ids=ids)["logits"]
        assert torch.equal(a, b), "forward no determinista en eval"
    print("OK G3: determinismo bit a bit en eval")


def test_presets_smoke():
    torch.manual_seed(74)
    for name in ["tiny", "350m", "700m", "1.2b", "2.6b"]:
        get_preset(name)  # construye y deriva layouts sin error
    print("OK presets TURING-1 construyen (tiny/350m/700m/1.2b/2.6b)")


if __name__ == "__main__":
    suite = [
        test_a1_flag_matrix,
        test_a2_bf16_forward,
        test_a3_padding_grads,
        test_b1_end_to_end_causality,
        test_b2_long_window_causality,
        test_b3_long_incremental_loop,
        test_b4_offset_invariance_with_trim,
        test_b5_sensitivity_antidegeneracy,
        test_c1_dilated_step_matches_full,
        test_c2_gate_bounds,
        test_d1_mtp_depth2,
        test_d2_labels_with_ignore,
        test_d3_speculative_sampling_multiseed,
        test_d4_sample_param_matrix,
        test_d5_edges,
        test_d6_no_mtp_raises,
        test_e1_batch,
        test_e2_padding_invariance,
        test_f1_long_greedy,
        test_f2_long_prefill_trim,
        test_f3_snapshot_restore,
        test_g1_router_invariants,
        test_g2_shared_expert_learns,
        test_g3_determinism,
        test_presets_smoke,
    ]
    failed = []
    for fn in suite:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - reporte de estabilidad, seguir con el resto
            print(f"FALLO {fn.__name__}: {type(e).__name__}: {e}")
            failed.append(fn.__name__)
    print(f"--- {len(suite) - len(failed)}/{len(suite)} pasaron ---")
    if failed:
        print("FALLARON:", failed)
        raise SystemExit(1)
    print("TURING-1 v1 ESTABILIDAD: TODAS PASARON")

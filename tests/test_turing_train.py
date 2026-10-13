"""Tests de estabilidad REAL de entrenamiento TURING-1 (solo lectura del fuente).

Verifica que la arquitectura sirve para entrenar modelos de verdad:
T1. Dress rehearsal: AdamW + clip + cosine, 25 pasos (pérdida baja, sin NaN).
T2. Memorización: 4 seqs fijas, 150 pasos (converge, el gate aprende, MTP baja).
T3. LR agresivo 1e-2, 5 pasos (no explota: sin NaN/inf).
T4. Acumulación de gradiente == batch grande (para entrenar con poca VRAM).
T5. Checkpoint modelo save/load: bit a bit idéntico.
T6. Checkpoint caché save/load: continúa idéntico.
T7. Train == eval bit a bit (no hay estocasticidad oculta).
T8. Contexto largo: 512 fwd+bwd, 1024 fwd (RoPE/SWA/trim bajo carga).
T9. AMP bf16 en CPU, 3 pasos (grads finitos).
T10. Post-entreno: especulativo == greedy con pesos APRENDIDOS + step==full.
T11. count_params(utils) vs parámetros reales (delta exacto y explicado).
T12. Álgebra de pérdida: total == lm + λ·mtp.
T13. Throughput informativo (sin asserts).
"""

import copy
import tempfile
import time

import torch

from turing import TuringForCausalLM
from turing.config import TuringConfig
from turing.utils import count_params


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


def _finite(t: torch.Tensor, name: str) -> None:
    bad = (~torch.isfinite(t)).sum().item()
    assert bad == 0, f"{name}: {bad} no finitos"


def _all_grads_finite(model: torch.nn.Module, name: str) -> int:
    n = 0
    for pname, p in model.named_parameters():
        if p.grad is not None:
            n += 1
            _finite(p.grad, f"{name}.grad[{pname}]")
    assert n > 0, f"{name}: sin gradientes"
    return n


# --------------------------------------------------------------- T1. loop
def test_t1_training_loop():
    torch.manual_seed(101)
    model = TuringForCausalLM(_base())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=25)
    batch = torch.randint(0, 255, (4, 32))
    losses, norms = [], []
    for _ in range(25):
        opt.zero_grad()
        out = model(input_ids=batch, labels=batch)
        _finite(out["logits"], "T1.logits")
        assert torch.isfinite(out["loss"]), "T1.loss NaN/inf"
        out["loss"].backward()
        nrm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        assert torch.isfinite(nrm), "T1.gradnorm NaN"
        norms.append(float(nrm))
        opt.step()
        sched.step()
        losses.append(out["loss"].item())
    import statistics
    early, late = statistics.mean(losses[:5]), statistics.mean(losses[-5:])
    assert late < early, f"no converge: {early:.3f} -> {late:.3f}"
    assert max(norms) < 1e4, f"gradnorm desbocada: {max(norms)}"
    _all_grads_finite(model, "T1")
    print(f"OK T1: 25 pasos AdamW+clip+cosine {early:.3f}->{late:.3f}, gradnorm max={max(norms):.2f}")


# -------------------------------------------------------- T2. memorización
def test_t2_memorization():
    torch.manual_seed(102)
    model = TuringForCausalLM(_base())
    data = torch.randint(0, 255, (4, 16))
    conv0 = next(lyr.conv for lyr in model.model.layers if not lyr.is_attention)
    mix0 = conv0.mix.detach().clone()
    mtp0 = None
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(150):
        opt.zero_grad()
        out = model(input_ids=data, labels=data)
        if step == 0:
            mtp0, loss0 = out["mtp_loss"].item(), out["loss"].item()
        assert torch.isfinite(out["loss"]), f"T2 NaN en paso {step}"
        out["loss"].backward()
        opt.step()
    lossf, mtpf = out["loss"].item(), out["mtp_loss"].item()
    assert lossf < 0.6 * loss0, f"no memoriza: {loss0:.3f}->{lossf:.3f}"
    assert mtpf < mtp0, f"MTP no baja: {mtp0:.3f}->{mtpf:.3f}"
    assert not torch.equal(conv0.mix.detach(), mix0), "el gate de la conv no aprendió"
    print(f"OK T2: memoriza 4 seqs {loss0:.3f}->{lossf:.3f}, MTP {mtp0:.3f}->{mtpf:.3f}, gate se mueve")


# ---------------------------------------------------------- T3. LR extremo
def test_t3_aggressive_lr():
    torch.manual_seed(103)
    model = TuringForCausalLM(_base())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    batch = torch.randint(0, 255, (4, 32))
    for step in range(5):
        opt.zero_grad()
        loss = model(input_ids=batch, labels=batch)["loss"]
        assert torch.isfinite(loss), f"T3 NaN/inf en paso {step} con lr=1e-2"
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                _finite(p.grad, f"T3.grad paso {step}")
        opt.step()
    print("OK T3: 5 pasos con lr=1e-2 sin NaN/inf (tolera LR agresivo)")


# ------------------------------------------------------- T4. acumulación
def test_t4_grad_accumulation():
    torch.manual_seed(104)
    cfg = _base()
    a = TuringForCausalLM(cfg)
    torch.manual_seed(104)
    b = TuringForCausalLM(cfg)  # init idéntico
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)
    full = torch.randint(0, 255, (4, 16))
    la = a(input_ids=full, labels=full)["loss"]
    la.backward()
    micro_losses = []
    for m in full.chunk(2, dim=0):
        l = b(input_ids=m, labels=m)["loss"] / 2
        l.backward()
        micro_losses.append(l.item() * 2)
    assert abs(la.item() - sum(micro_losses) / 2) < 1e-6, (la.item(), micro_losses)
    worst = 0.0
    for pa, pb in zip(a.parameters(), b.parameters()):
        if pa.grad is not None:
            worst = max(worst, (pa.grad - pb.grad).abs().max().item())
    assert worst < 1e-4, worst
    print(f"OK T4: 2 micro-batches == batch grande (dif grad max={worst:.2e})")


# ------------------------------------------------------ T5/T6. checkpoints
def test_t5_model_checkpoint():
    torch.manual_seed(105)
    cfg = _base()
    model = TuringForCausalLM(cfg).eval()
    ids = torch.randint(0, 200, (1, 12))
    with torch.no_grad():
        ref = model(input_ids=ids)["logits"]
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    torch.save(model.state_dict(), path)
    fresh = TuringForCausalLM(cfg).eval()
    missing, unexpected = fresh.load_state_dict(torch.load(path, weights_only=True), strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    with torch.no_grad():
        got = fresh(input_ids=ids)["logits"]
    assert torch.equal(ref, got), "reload no es bit a bit"
    print("OK T5: checkpoint modelo save/load bit a bit")


def test_t6_cache_checkpoint():
    torch.manual_seed(106)
    model = TuringForCausalLM(_base()).eval()
    ids = torch.randint(0, 200, (1, 8))
    tok = torch.randint(0, 200, (1, 1))
    with torch.no_grad():
        _, cache = model.model(input_ids=ids, use_cache=True)
        assert cache is not None
        h_ref, _ = model.model.step(tok, cache)
        blob = {"k": dict(cache.k), "v": dict(cache.v), "conv": dict(cache.conv),
                "gw": dict(cache.group_window), "seq": cache.get_seq_length()}
        from turing.cache import TuringCache
        c2 = TuringCache()
        c2.k, c2.v, c2.conv = dict(blob["k"]), dict(blob["v"]), dict(blob["conv"])
        c2.group_window = dict(blob["gw"])
        c2.seq_len = blob["seq"]
        torch.save(blob, blob_path := tempfile.mktemp(suffix=".pt"))
        blob2 = torch.load(blob_path, weights_only=True)
        c3 = TuringCache()
        c3.k, c3.v, c3.conv = dict(blob2["k"]), dict(blob2["v"]), dict(blob2["conv"])
        c3.group_window = dict(blob2["gw"])
        c3.seq_len = blob2["seq"]
        h_re, _ = model.model.step(tok, c3)
        assert torch.equal(h_ref, h_re), "caché recargada diverge"
    print("OK T6: checkpoint de caché (disco) continúa idéntico")


# ------------------------------------------------------------- T7. modos
def test_t7_train_equals_eval():
    torch.manual_seed(107)
    model = TuringForCausalLM(_base())
    ids = torch.randint(0, 200, (1, 14))
    with torch.no_grad():
        model.eval()
        a = model(input_ids=ids)["logits"]
        model.train()
        b = model(input_ids=ids)["logits"]
    assert torch.equal(a, b), "train != eval: estocasticidad oculta?"
    print("OK T7: train == eval bit a bit (sin dropout)")


# -------------------------------------------------------- T8. contexto largo
def test_t8_long_context_train():
    torch.manual_seed(108)
    model = TuringForCausalLM(_base())
    ids = torch.randint(0, 255, (2, 512))
    out = model(input_ids=ids, labels=ids)
    _finite(out["logits"], "T8.logits512")
    assert torch.isfinite(out["loss"]) and torch.isfinite(out["mtp_loss"] or torch.tensor(0.0))
    out["loss"].backward()
    _all_grads_finite(model, "T8")
    with torch.no_grad():
        big = model(input_ids=torch.randint(0, 255, (1, 1024)))["logits"]
        _finite(big, "T8.logits1024")
    print("OK T8: 512 fwd+bwd + 1024 fwd finitos (RoPE/SWA/trim bajo carga)")


# ----------------------------------------------------------------- T9. AMP
def test_t9_bf16_amp():
    torch.manual_seed(109)
    model = TuringForCausalLM(_base())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cpu")
    batch = torch.randint(0, 255, (2, 24))
    for step in range(3):
        opt.zero_grad()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = model(input_ids=batch, labels=batch)["loss"]
        assert torch.isfinite(loss.float()), f"T9 loss no finito paso {step}"
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    _all_grads_finite(model, "T9")
    print("OK T9: 3 pasos AMP bf16 en CPU, grads finitos")


# ----------------------------------------------- T10. post-entreno consistente
def test_t10_posttrain_consistency():
    torch.manual_seed(110)
    model = TuringForCausalLM(_base())
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    data = torch.randint(0, 255, (2, 20))
    for _ in range(10):
        opt.zero_grad()
        l = model(input_ids=data, labels=data)["loss"]
        assert torch.isfinite(l)
        l.backward()
        opt.step()
    model.eval()
    prompt = torch.randint(0, 200, (1, 8))
    with torch.no_grad():
        g = model.generate(prompt, max_new_tokens=10, do_sample=False)
        s = model.generate_speculative(prompt, max_new_tokens=10, do_sample=False)
        assert torch.equal(g, s), "post-entreno: especulativo diverge"
        # step==full también con el gate ya movido por el entrenamiento.
        lyr = next(ly for ly in model.model.layers if not ly.is_attention)
        h = torch.randn(1, 13, 32)
        full = lyr.conv(h)
        st = lyr.conv.state_from_full(h[:, :10, :])
        outs = []
        for t in range(10, 13):
            o, st = lyr.conv.step(h[:, t : t + 1, :], st)
            outs.append(o)
        err = (full[:, 10:, :] - torch.cat(outs, 1)).abs().max().item()
        assert err < 1e-5, err
    print(f"OK T10: post-entreno consistente (espec==greedy, conv err={err:.2e})")


# ------------------------------------------------- T11. conteo parámetros
def test_t11_param_count():
    torch.manual_seed(111)
    cfg = _base()
    model = TuringForCausalLM(cfg)
    seen, total = set(), 0
    for p in model.parameters():
        if id(p) not in seen:  # head atada: contar una vez
            seen.add(id(p))
            total += p.numel()
    rep = count_params(cfg)
    # utils no cuenta QK-Norms (hd por Q + hd por K productora) ni norms MTP:
    # 3 attn × 8 + 2 productoras × 8 + 1 MTP × 32 = 72.
    expected_missing = (cfg.num_attention_blocks * cfg.head_dim
                        + sum(1 for i in cfg.attn_is_producer if i) * cfg.head_dim
                        + cfg.mtp_depth * cfg.hidden_size)
    assert total - rep["total"] == expected_missing, (total, rep["total"], expected_missing)
    print(f"OK T11: utils vs real difieren solo en {expected_missing} params "
          f"(QK-norms+MTP-norms, {100*expected_missing/total:.3f}% — documentado)")


# ------------------------------------------------------ T12. álgebra loss
def test_t12_loss_algebra():
    torch.manual_seed(112)
    cfg = _base()
    model = TuringForCausalLM(cfg)
    ids = torch.randint(0, 255, (2, 14))
    out = model(input_ids=ids, labels=ids)
    expect = out["lm_loss"] + cfg.mtp_loss_weight * out["mtp_loss"]
    assert torch.allclose(out["loss"], expect, atol=1e-6)
    print(f"OK T12: total == lm + {cfg.mtp_loss_weight}·mtp "
          f"({out['lm_loss'].item():.3f} + {out['mtp_loss'].item():.3f})")


# ------------------------------------------------------ T13. throughput
def test_t13_throughput_info():
    torch.manual_seed(113)
    model = TuringForCausalLM(_base()).eval()
    ids = torch.randint(0, 200, (1, 128))
    with torch.no_grad():
        t0 = time.perf_counter()
        _, cache = model.model(input_ids=ids, use_cache=True)
        pre = time.perf_counter() - t0
        assert cache is not None
        t0 = time.perf_counter()
        g = model.generate(ids, max_new_tokens=20, do_sample=False)
        dec = (time.perf_counter() - t0) / 20
        assert g.shape == (1, 148)
    print(f"INFO T13 (tiny d=32, CPU): prefill {128/pre:.0f} tok/s, decode {1/dec:.0f} tok/s")


if __name__ == "__main__":
    suite = [test_t1_training_loop, test_t2_memorization, test_t3_aggressive_lr,
             test_t4_grad_accumulation, test_t5_model_checkpoint, test_t6_cache_checkpoint,
             test_t7_train_equals_eval, test_t8_long_context_train, test_t9_bf16_amp,
             test_t10_posttrain_consistency, test_t11_param_count, test_t12_loss_algebra,
             test_t13_throughput_info]
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
    print("TURING-1 ENTRENAMIENTO: TODO ESTABLE")

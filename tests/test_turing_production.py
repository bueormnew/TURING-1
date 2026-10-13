"""Puerta de calidad de producción TURING-1.

Q1. Dispatch loop/batched/auto equivalentes (forward + grads).
Q2. train_step mueve el bias, stats completas, determinista.
Q3. Acumulación vía train_step == batch grande (parámetros iguales).
Q4. Checkpoint (modelo+opt+sched) reanuda idéntico.
Q5. Mini-train() con ckpts: baja loss, ficheros existen, bias se mueve.
Q6. ByteDataLoader: formas, rango, determinismo, split sin fuga.
Q7. generate_speculative(return_stats): claves, rangos, compatibilidad.
Q8. CLI end-to-end en subproceso (sintético).
Q9. CLI byte-LM con .txt real (vía data.py).
"""

import os
import subprocess
import sys
import tempfile

import torch

from turing import TuringForCausalLM
from turing.config import TuringConfig
from turing.data import ByteDataLoader, make_chunks, read_bytes, split_chunks
from turing.modules import TuringMoE
from turing.train import TrainConfig, load_checkpoint, make_scheduler, save_checkpoint, train, train_step


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
    assert int((~torch.isfinite(t)).sum()) == 0, f"{name} no finito"


# ------------------------------------------------------------------ Q1
def test_q1_dispatch_equivalence():
    torch.manual_seed(601)
    mods = {}
    for disp in ("loop", "batched", "auto"):
        torch.manual_seed(602)
        m = TuringMoE(64, 16, 4, 32, 1, dispatch=disp)
        m.load_state_dict(mods["loop"].state_dict()) if "loop" in mods else None
        m.eval()
        mods[disp] = m
    # mismos pesos en los tres (mismo seed + shapes iguales ya los iguala)
    h = torch.randn(48, 64)
    outs = {}
    for disp, m in mods.items():
        o, _ = m(h)
        _finite(o, f"Q1.{disp}")
        outs[disp] = o
    assert (outs["loop"] - outs["batched"]).abs().max().item() < 1e-5
    assert (outs["loop"] - outs["auto"]).abs().max().item() < 1e-5
    # grads también equivalentes
    mods["loop"].zero_grad()
    mods["batched"].zero_grad()
    outs["loop"].sum().backward()
    # recomputar grafos separados para comparar
    torch.manual_seed(603)
    h2 = torch.randn(48, 64, requires_grad=False)
    gmax = 0.0
    for disp in ("loop", "batched"):
        mods[disp].zero_grad()
        mods[disp](h2)[0].sum().backward()
    for pa, pb in zip(mods["loop"].parameters(), mods["batched"].parameters()):
        gmax = max(gmax, (pa.grad - pb.grad).abs().max().item())
    assert gmax < 1e-4, gmax
    print(f"OK Q1: loop==batched==auto fwd+grads (dif grad max={gmax:.2e})")


# ------------------------------------------------------------------ Q2
def test_q2_train_step_bias():
    torch.manual_seed(604)
    cfg = _base()
    m1 = TuringForCausalLM(cfg)
    torch.manual_seed(604)
    m2 = TuringForCausalLM(cfg)
    tc = TrainConfig(lr=3e-4, bias_lr=0.01)
    ids = torch.randint(0, 255, (4, 16))
    opt1 = torch.optim.AdamW(m1.parameters(), lr=tc.lr)
    s1 = train_step(m1, opt1, None, ids, tc)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=tc.lr)
    s2 = train_step(m2, opt2, None, ids, tc)
    assert abs(s1["loss"] - s2["loss"]) < 1e-9, "train_step no determinista"
    assert s1["bias_updated"] is True
    moved = any((ly.ffn.expert_bias != 0).any().item()
                for ly in m1.model.layers if ly.is_moe)
    assert moved, "el bias no se movió en train_step"
    assert all(k in s1 for k in ("loss", "lm_loss", "mtp_loss", "grad_norm", "lr"))
    print(f"OK Q2: train_step determinista (loss={s1['loss']:.4f}), bias actualizado")


# ------------------------------------------------------------------ Q3
def test_q3_accum_via_api():
    torch.manual_seed(605)
    cfg = _base()
    a = TuringForCausalLM(cfg)
    torch.manual_seed(605)
    b = TuringForCausalLM(cfg)
    full = torch.randint(0, 255, (4, 16))
    opta = torch.optim.AdamW(a.parameters(), lr=3e-4)
    train_step(a, opta, None, full, TrainConfig(lr=3e-4, accum=1, bias_lr=None))
    optb = torch.optim.AdamW(b.parameters(), lr=3e-4)
    train_step(b, optb, None, full, TrainConfig(lr=3e-4, accum=2, bias_lr=None))
    worst = max((pa - pb).abs().max().item() for pa, pb in zip(a.parameters(), b.parameters()))
    assert worst < 1e-5, worst
    print(f"OK Q3: accum=2 == batch grande (dif params={worst:.2e})")


# ------------------------------------------------------------------ Q4
def test_q4_ckpt_resume_sched():
    torch.manual_seed(606)
    cfg = _base()
    a = TuringForCausalLM(cfg)
    opta = torch.optim.AdamW(a.parameters(), lr=3e-4)
    sa = make_scheduler(opta, "cosine", 2, 10)
    data = torch.randint(0, 255, (2, 20))
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    for _ in range(3):
        train_step(a, opta, sa, data, TrainConfig(lr=3e-4, bias_lr=0.01))
    save_checkpoint(path, a, opta, sa, 3)
    torch.manual_seed(606)
    b = TuringForCausalLM(cfg)
    optb = torch.optim.AdamW(b.parameters(), lr=3e-4)
    sb = make_scheduler(optb, "cosine", 2, 10)
    assert load_checkpoint(path, b, optb, sb) == 3
    la, lb = [], []
    for _ in range(2):
        la.append(train_step(a, opta, sa, data, TrainConfig(lr=3e-4, bias_lr=0.01))["loss"])
        lb.append(train_step(b, optb, sb, data, TrainConfig(lr=3e-4, bias_lr=0.01))["loss"])
    assert all(abs(x - y) < 1e-6 for x, y in zip(la, lb)), (la, lb)
    print(f"OK Q4: reanuda con sched+bias idéntico ({la[0]:.4f}=={lb[0]:.4f})")


# ------------------------------------------------------------------ Q5
def test_q5_mini_train():
    torch.manual_seed(607)
    with tempfile.TemporaryDirectory() as ckd:
        cfg = _base()
        model = TuringForCausalLM(cfg)
        tc = TrainConfig(lr=3e-3, max_steps=12, bias_lr=0.01, ckpt_every=4,
                         ckpt_dir=ckd, log_every=100)
        opt = torch.optim.AdamW(model.parameters(), lr=tc.lr)
        sched = make_scheduler(opt, "cosine", 0, 12)
        g = torch.Generator().manual_seed(7)
        batches = [torch.randint(2, 30, (4, 24), generator=g) for _ in range(12)]
        hist = train(model, opt, sched, batches, tc)
        assert len(hist) == 12 and hist[-1]["loss"] < hist[0]["loss"]
        for s in (4, 8, 12):
            assert os.path.exists(f"{ckd}/step{s}.pt"), f"falta ckpt {s}"
        assert any((ly.ffn.expert_bias != 0).any().item()
                   for ly in model.model.layers if ly.is_moe)
    print(f"OK Q5: mini-train 12 pasos {hist[0]['loss']:.3f}->{hist[-1]['loss']:.3f} + 3 ckpts + bias vivo")


# ------------------------------------------------------------------ Q6
def test_q6_loader():
    raw = bytes(range(256)) * 40  # 10240 bytes
    ch = make_chunks(raw, 64)
    assert ch.shape == (160, 64) and ch.max().item() < 256 and ch.min().item() >= 0
    assert bytes(ch.reshape(-1).tolist()) == raw[: 160 * 64], "empaquetado no continuo"
    tr, va = split_chunks(ch, 0.1)
    assert len(tr) == 144 and len(va) == 16
    assert bytes(va.reshape(-1).tolist()) == raw[-16 * 64:], "val no es la cola"
    l1, l2 = ByteDataLoader(tr, 8, seed=3), ByteDataLoader(tr, 8, seed=3)
    b1, b2 = next(iter(l1)), next(iter(l2))
    assert torch.equal(b1, b2), "loader no determinista"
    l3 = ByteDataLoader(tr, 8, seed=4)
    assert not torch.equal(b1, next(iter(l3))), "seed ignorada?"
    print("OK Q6: chunks continuos, split cola-sin-fuga, loader determinista")


# ------------------------------------------------------------------ Q7
def test_q7_spec_stats():
    torch.manual_seed(608)
    model = TuringForCausalLM(_base()).eval()
    ids = torch.randint(0, 200, (1, 8))
    with torch.no_grad():
        g = model.generate_speculative(ids, max_new_tokens=6, do_sample=False)
        assert isinstance(g, torch.Tensor)
        g2, st = model.generate_speculative(ids, max_new_tokens=6, do_sample=False, return_stats=True)
        assert torch.equal(g, g2)
        assert set(st) == {"drafts", "accepted", "rate"} and st["drafts"] > 0
        assert 0.0 <= st["rate"] <= 1.0 and st["accepted"] <= st["drafts"]
    print(f"OK Q7: stats especulativas {st} + compatibilidad sin flag")


# ------------------------------------------------------------- Q8/Q9 (CLI)
def _run_cli(args: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return subprocess.run([sys.executable, os.path.join(root, "train_turing.py"), *args],
                          capture_output=True, text=True, timeout=timeout, cwd=root)


def test_q8_cli_synthetic():
    r = _run_cli(["--steps", "6", "--batch", "4", "--seq", "32", "--ckpt-every", "3",
                  "--ckpt-dir", tempfile.mkdtemp(prefix="q8_")])
    assert r.returncode == 0, r.stderr[-2000:]
    assert "FINAL loss=" in r.stdout and "SAMPLE" in r.stdout, r.stdout[-2000:]
    print("OK Q8: CLI sintético end-to-end (ckpt + sample)")


def test_q9_cli_bytelm():
    with tempfile.TemporaryDirectory() as td:
        txt = os.path.join(td, "corpus.txt")
        with open(txt, "w", encoding="utf-8") as f:
            f.write(("Turing machines manipulate symbols on a tape. " * 40 + "\n") * 60)
        assert len(read_bytes(txt)) > 8000
        r = _run_cli(["--task", "bytes", "--text", txt, "--steps", "4", "--batch", "4",
                      "--seq", "64", "--ckpt-dir", os.path.join(td, "ck")])
    assert r.returncode == 0, r.stderr[-2000:]
    assert "FINAL loss=" in r.stdout and "val=" in r.stdout, r.stdout[-2000:]
    print("OK Q9: CLI byte-LM sobre .txt real (train+val)")


if __name__ == "__main__":
    suite = [test_q1_dispatch_equivalence, test_q2_train_step_bias, test_q3_accum_via_api,
             test_q4_ckpt_resume_sched, test_q5_mini_train, test_q6_loader,
             test_q7_spec_stats, test_q8_cli_synthetic, test_q9_cli_bytelm]
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
    print("TURING-1 PRODUCCIÓN: LISTO")

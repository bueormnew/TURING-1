"""Entrenamiento real TURING-1 en esta máquina.

Uso:
  python train_turing.py --task echo --steps 60            # humo sintético
  python train_turing.py --text datos.txt --seq 256 --steps 200 --batch 8
  python train_turing.py --preset 350m --text datos.txt ...  (más capacidad)

Byte-LM sin tokenizador (vocab 256). Guarda checkpoints + muestra final.
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from turing import TuringForCausalLM, get_preset  # noqa: E402
from turing.data import ByteDataLoader, make_chunks, read_bytes, split_chunks  # noqa: E402
from turing.train import TrainConfig, load_checkpoint, make_scheduler, save_checkpoint, set_seed, train  # noqa: E402


def echo_batches(n: int, batch: int, seq: int, seed: int):
    """Mezcla echo2/echo9/echo16 (tokens 2..31, V=32)."""
    V, TASKS = 32, ["echo2", "echo16", "echo9", "echo16"]
    out = []
    for i in range(n):
        g = torch.Generator().manual_seed(seed + i)
        arr = []
        for j in range(batch):
            t, L = TASKS[(i * batch + j) % len(TASKS)], seq
            s = torch.randint(2, 30, (L,), generator=g)
            k = {"echo2": 2, "echo9": 9, "echo16": 16}[t]
            for p in range(k, L):
                s[p] = s[p - k]
            arr.append(s)
        out.append(torch.stack(arr))
    return out, V


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="tiny")
    ap.add_argument("--task", default="echo", choices=["echo", "bytes"])
    ap.add_argument("--text", default="")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--bias-lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-dir", default="checkpoints_turing")
    ap.add_argument("--ckpt-every", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=0, help="override de hidden_size (0=preset)")
    ap.add_argument("--layers", type=int, default=0, help="override de num_hidden_layers")
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = get_preset(args.preset)
    if args.preset == "tiny" or args.hidden or args.layers:
        # Escala de juguete por defecto: rápida en CPU.
        import dataclasses
        cfg = dataclasses.replace(
            cfg, vocab_size=256,
            hidden_size=args.hidden or 64, num_hidden_layers=args.layers or 6,
            num_attention_heads=4, num_key_value_heads=2, block_ff_dim=128,
            full_attn_idxs=[1, 2, 4], sliding_window=32,
            num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
            max_position_embeddings=512)
    print(f"cfg: d={cfg.hidden_size} L={cfg.num_hidden_layers} attn={cfg.num_attention_blocks} "
          f"E={cfg.num_experts}x{cfg.num_experts_per_tok} mtp={cfg.mtp_depth} W={cfg.sliding_window}", flush=True)

    model = TuringForCausalLM(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params / 1e6:.2f}M", flush=True)

    if args.task == "echo" or not args.text:
        batches, _ = echo_batches(args.steps, args.batch, args.seq, args.seed)
        val = echo_batches(1, args.batch, args.seq, 99999)[0][0]  # un batch tensor
    else:
        raw = read_bytes(args.text)
        print(f"texto: {len(raw) / 1e6:.2f} MB", flush=True)
        tr, va = split_chunks(make_chunks(raw, args.seq))
        loader = ByteDataLoader(tr, args.batch, seed=args.seed)
        it = iter(loader)
        batches = [[next(it) for _ in range(1)][0] for _ in range(args.steps)]
        val = va[: min(len(va), 4)]

    tcfg = TrainConfig(lr=args.lr, max_steps=args.steps, bias_lr=args.bias_lr,
                       ckpt_every=args.ckpt_every, ckpt_dir=args.ckpt_dir,
                       log_every=max(args.steps // 10, 1), seed=args.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=0.01)
    sched = make_scheduler(opt, "cosine", 0, args.steps)

    def eval_fn(m):
        m.eval()
        with torch.no_grad():
            out = m(input_ids=val, labels=val)
            v = out["loss"].item()
        m.train()
        return {"val_loss": v}

    t0 = time.perf_counter()
    hist = train(model, opt, sched, batches, tcfg, eval_fn=eval_fn)
    dt = time.perf_counter() - t0
    final_eval = hist[-1].get("eval") or eval_fn(model)
    print(f"FINAL loss={hist[-1]['loss']:.4f} val={final_eval['val_loss']:.4f} "
          f"tiempo={dt:.0f}s ({args.steps * args.batch * args.seq / dt:.0f} tok/s)", flush=True)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    save_checkpoint(f"{args.ckpt_dir}/final.pt", model, opt, sched, args.steps)
    print(f"ckpt: {args.ckpt_dir}/final.pt", flush=True)

    model.eval()
    with torch.no_grad():
        prompt = batches[0][:1, :16]
        gen = model.generate(prompt, max_new_tokens=32, do_sample=False)
        ids = gen[0].tolist()
    try:
        txt = bytes(ids).decode("utf-8", errors="replace")
        print("SAMPLE:", txt[:200].replace("\n", "\\n"), flush=True)
    except Exception as e:  # noqa: BLE001
        print("SAMPLE ids:", ids[:40], e, flush=True)


if __name__ == "__main__":
    main()

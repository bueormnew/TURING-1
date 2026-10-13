"""Bucle de entrenamiento productivo TURING-1 (PyTorch puro, CPU/GPU).

Incluye lo que un entreno real necesita y los tests no daban:
- `train_step` con acumulación, clip, scheduler y actualización del sesgo
  MoE aux-free (el `expert_bias` por fin se mueve en cada optimizer-step).
- Checkpoints completos (modelo + optimizador + scheduler + paso).
- `train()` mínimo con logging, eval periódica y reanudación.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 0
    max_steps: int = 100
    sched: str = "cosine"  # "cosine" | "constant"
    clip: float = 1.0
    accum: int = 1
    bias_lr: float | None = 0.01  # None = no tocar expert_bias
    log_every: int = 10
    ckpt_every: int = 0  # 0 = no guardar
    ckpt_dir: str = "checkpoints"
    seed: int = 0


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_scheduler(opt: torch.optim.Optimizer, kind: str, warmup: int, max_steps: int):
    if kind == "constant" or max_steps <= 0:
        return torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    def fn(step: int) -> float:
        import math
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        p = (step - warmup) / max(max_steps - warmup, 1)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


def forward_backward(model, ids: torch.Tensor, *, labels: torch.Tensor | None = None,
                     mask: torch.Tensor | None = None, scale: float = 1.0,
                     want_router: bool = False) -> dict:
    """Un micro-paso: forward + backward (sin optimizer-step)."""
    out = model(input_ids=ids, labels=ids if labels is None else labels,
                attention_mask=mask, return_router_logits=want_router)
    loss = out["loss"]
    assert loss is not None
    (loss * scale).backward()
    stats = {"loss": loss.item()}
    for k in ("lm_loss", "mtp_loss", "z_loss"):
        v = out.get(k)
        stats[k] = None if v is None else float(v.detach())
    if want_router:
        assert out["router_logits"] is not None, "el modelo no devolvió router_logits"
        stats["router_idx"] = [
            torch.topk(torch.softmax(rl.detach(), dim=-1)
                       + _bias_of(model, i).to(rl.device, rl.dtype),
                       k=_topk_of(model, i), dim=-1).indices.reshape(-1, _topk_of(model, i))
            for i, rl in enumerate(out["router_logits"])
        ]
    return stats


def _moe_modules(model) -> list:
    return [ly.ffn for ly in model.model.layers if ly.is_moe]


def _bias_of(model, i: int) -> torch.Tensor:
    return _moe_modules(model)[i].expert_bias


def _topk_of(model, i: int) -> int:
    return _moe_modules(model)[i].top_k


@torch.no_grad()
def update_moe_biases(model, idx_list: list[torch.Tensor], lr: float) -> None:
    """Aplica update_bias_ por capa con los top-k del último batch."""
    for moe, idx in zip(_moe_modules(model), idx_list):
        moe.update_bias_(idx.to(moe.expert_bias.device), lr=lr)


def finish_step(model, opt: torch.optim.Optimizer, sched=None, *, clip: float = 1.0) -> float:
    """Clip + optimizer-step + scheduler-step. Devuelve grad-norm pre-clip."""
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    opt.step()
    if sched is not None:
        sched.step()
    opt.zero_grad()
    return float(norm)


def save_checkpoint(path: str, model, opt, sched, step: int) -> None:
    torch.save({"model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict() if sched is not None else None,
                "step": step}, path)


def load_checkpoint(path: str, model, opt=None, sched=None, map_location: str = "cpu") -> int:
    blob = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(blob["model"], strict=True)
    if opt is not None:
        opt.load_state_dict(blob["opt"])
    if sched is not None and blob["sched"] is not None:
        sched.load_state_dict(blob["sched"])
    return int(blob["step"])


def train_step(model, opt, sched, ids: torch.Tensor, cfg: TrainConfig, *,
               labels: torch.Tensor | None = None, mask: torch.Tensor | None = None) -> dict:
    """Un optimizer-step con acumulación interna (ids ya es 1 batch).

    Divide el batch en `cfg.accum` micro-lotes para equivalencia exacta.
    """
    B = ids.shape[0]
    assert B % cfg.accum == 0, "batch divisible por accum"
    opt.zero_grad()
    stats: dict = {"loss": 0.0}
    last_router = None
    want_router = cfg.bias_lr is not None and any(ly.is_moe for ly in model.model.layers)
    masks = [None] * cfg.accum if mask is None else list(mask.chunk(cfg.accum, dim=0))
    labs = ids if labels is None else labels
    for micro, mlab, mm in zip(ids.chunk(cfg.accum, dim=0), labs.chunk(cfg.accum, dim=0), masks):
        s = forward_backward(model, micro, labels=mlab, mask=mm,
                             scale=1.0 / cfg.accum, want_router=want_router)
        stats["loss"] += s["loss"] / cfg.accum
        for k in ("lm_loss", "mtp_loss", "z_loss"):
            if s[k] is not None:
                stats[k] = stats.get(k, 0.0) + s[k] / cfg.accum
        last_router = s.get("router_idx", last_router)
    stats["grad_norm"] = finish_step(model, opt, sched, clip=cfg.clip)
    if cfg.bias_lr is not None and last_router is not None:
        update_moe_biases(model, last_router, cfg.bias_lr)
        stats["bias_updated"] = True
    else:
        stats["bias_updated"] = False
    stats["lr"] = opt.param_groups[0]["lr"]
    return stats


def train(model, opt, sched, batches, cfg: TrainConfig, eval_fn=None) -> list[dict]:
    """Loop mínimo productivo sobre un iterable (finito) de batches."""
    history: list[dict] = []
    for step, ids in enumerate(batches):
        if step >= cfg.max_steps:
            break
        st = train_step(model, opt, sched, ids, cfg)
        st["step"] = step
        if eval_fn is not None and (step % cfg.log_every == 0):
            st["eval"] = eval_fn(model)
        history.append(st)
        if cfg.log_every and step % cfg.log_every == 0:
            print(f"step {step}: loss={st['loss']:.4f} lr={st['lr']:.2e} gnorm={st['grad_norm']:.2f}", flush=True)
        if cfg.ckpt_every and (step + 1) % cfg.ckpt_every == 0:
            import os
            os.makedirs(cfg.ckpt_dir, exist_ok=True)
            save_checkpoint(f"{cfg.ckpt_dir}/step{step + 1}.pt", model, opt, sched, step + 1)
    return history

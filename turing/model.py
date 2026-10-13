"""Backbone y CausalLM TURING-1.

Capa:  h = h + Operador(RMSNorm(h));  h = h + FFN(RMSNorm(h))
  - Operador = conv multiescala o GQA (productora/consumidora CLA, SWA/global).
  - FFN = SwiGLU denso (2 primeras capas) o MoE fino top-6/8 + shared.

Pérdida: L = L_lm + λ·L_mtp (MTP con teacher forcing, head LM compartida).
Generación: estándar token-a-token + especulativa (borrador MTP, batch=1).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from lfm2.modules import LFM2RMSNorm, LFM2RotaryEmbedding, LFM2SwiGLU

from .cache import TuringCache
from .config import TuringConfig
from .modules import TuringAttention, TuringMoE, TuringMTPHead, TuringMultiscaleConv


def _rope_single(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """RoPE a un solo tensor (B, H, L, D) con cos/sin (B, L, D)."""
    d = x.shape[-1]
    rot = torch.cat((-x[..., d // 2 :], x[..., : d // 2]), dim=-1)
    return x * cos.unsqueeze(1) + rot * sin.unsqueeze(1)


def _turing_bias(
    attention_mask: torch.Tensor | None,
    lq: int,
    lk: int,
    q_start: int,
    k_start: int,
    device: torch.device,
    dtype: torch.dtype,
    window: int | None = None,
) -> torch.Tensor | None:
    """Sesgo aditivo (B,1,Lq,Lk) causal por posiciones absolutas + padding + ventana.

    Query i (pos abs q_start+i) ve key j (pos abs k_start+j) si k<=q y,
    con ventana W, si k>=q-W+1.
    `attention_mask` (B, new_len) cubre [0, new_len); se recorta a las claves.
    """
    ar_q = (q_start + torch.arange(lq, device=device)).unsqueeze(1)
    ar_k = (k_start + torch.arange(lk, device=device)).unsqueeze(0)
    allowed = ar_k <= ar_q
    if window is not None:
        allowed = allowed & (ar_k >= ar_q - window + 1)
    bias = torch.where(
        allowed,
        torch.zeros((lq, lk), device=device, dtype=torch.float32),
        torch.full((lq, lk), float("-inf"), device=device, dtype=torch.float32),
    ).unsqueeze(0).unsqueeze(0)
    if attention_mask is not None:
        full = attention_mask.to(torch.float32)
        key_part = full[:, k_start : k_start + lk]
        if key_part.shape[1] != lk:  # pragma: no cover
            raise ValueError(f"máscara incompatible: {full.shape} vs Lk={lk} @ {k_start}")
        key_pad = (1.0 - key_part).unsqueeze(1).unsqueeze(2) * (torch.finfo(torch.float32).min / 2)
        bias = bias + key_pad
    if attention_mask is None and lq == 1:
        return None  # decode de 1 token sin padding: todo visible
    return bias.to(dtype)


class TuringDecoderLayer(nn.Module):
    def __init__(self, config: TuringConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        info = config.attn_info_of_layer(layer_idx)
        self.is_attention = info is not None
        if self.is_attention:
            _, kind, _, is_producer = info
            self.attn_kind = kind
            self.is_producer = is_producer
            self.window = config.sliding_window if kind == "sliding" else None
            self.self_attn = TuringAttention(
                config.hidden_size, config.num_attention_heads, config.num_key_value_heads,
                config.norm_eps, with_kv=is_producer, use_sdpa=config.use_sdpa,
            )
        else:
            if config.use_multiscale_conv:
                self.conv = TuringMultiscaleConv(
                    config.hidden_size, config.conv_kernel_size, config.conv_dilated_kernel,
                    config.conv_dilation, config.conv_bias, config.multiscale_gate_init,
                )
            else:  # pragma: no cover
                from lfm2.modules import LFM2ShortConv
                self.conv = LFM2ShortConv(config.hidden_size, config.conv_kernel_size, config.conv_bias)
        if config.use_moe and layer_idx >= config.num_dense_layers:
            self.ffn = TuringMoE(
                config.hidden_size, config.num_experts, config.num_experts_per_tok,
                config.moe_intermediate_size, config.moe_shared_experts, config.moe_norm_topk_prob,
                dispatch=getattr(config, "moe_dispatch", "auto"),
            )
            self.is_moe = True
        else:
            self.ffn = LFM2SwiGLU(config.hidden_size, config.effective_intermediate_size)
            self.is_moe = False
        self.operator_norm = LFM2RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = LFM2RMSNorm(config.hidden_size, eps=config.norm_eps)


class TuringModel(nn.Module):
    def __init__(self, config: TuringConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([TuringDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.rotary = LFM2RotaryEmbedding(config.head_dim, config.max_position_embeddings, config.rope_theta)
        self.final_norm = LFM2RMSNorm(config.hidden_size, eps=config.norm_eps)
        # Grupo CLA por capa de atención (orden de capa -> gid).
        self._layer_group: dict[int, int] = {}
        order = 0
        for i, t in enumerate(config.layer_types):
            if t == "full_attention":
                self._layer_group[i] = config.attn_group[order]
                order += 1
        self._init_weights()

    def _init_weights(self) -> None:
        std = self.config.initializer_range
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, mean=0.0, std=std)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
            elif isinstance(mod, nn.Embedding):
                nn.init.normal_(mod.weight, mean=0.0, std=std)
            elif isinstance(mod, nn.Conv1d):
                nn.init.normal_(mod.weight, mean=0.0, std=std)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
            elif isinstance(mod, LFM2RMSNorm):
                nn.init.ones_(mod.weight)
            elif isinstance(mod, TuringMoE):
                nn.init.normal_(mod.gate_up, mean=0.0, std=std)
                nn.init.normal_(mod.down, mean=0.0, std=std)
                # expert_bias ya es ceros; mix de la conv conserva su init.

    # -- forward --------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past: TuringCache | None = None,
        use_cache: bool = False,
        return_router_logits: bool = False,
    ) -> tuple[torch.Tensor, TuringCache | None] | tuple[torch.Tensor, TuringCache | None, list[torch.Tensor]]:
        cfg = self.config
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Especificar exactamente uno de input_ids / inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        b, lq, _ = inputs_embeds.shape
        past_seq = past.get_seq_length() if past is not None else 0
        new_len = past_seq + lq

        conv_mask = attention_mask if (attention_mask is not None and past is None) else None
        full_mask = attention_mask
        if attention_mask is not None and past_seq > 0 and attention_mask.shape[-1] == lq:
            prefix = torch.ones(b, past_seq, device=attention_mask.device, dtype=attention_mask.dtype)
            full_mask = torch.cat([prefix, attention_mask], dim=1)

        if position_ids is None:
            position_ids = (past_seq + torch.arange(lq, device=inputs_embeds.device)).unsqueeze(0).expand(b, lq)
        cos, sin = self.rotary(inputs_embeds, position_ids)

        new_cache = TuringCache() if use_cache else None
        if new_cache is not None:
            for gid, all_sl in cfg.group_all_sliding.items():
                new_cache.set_group_window(gid, cfg.sliding_window if all_sl else None)
            if past is not None:
                for gid in past.k:
                    new_cache.k[gid] = past.k[gid]
                    new_cache.v[gid] = past.v[gid]
                for gid, w in past.group_window.items():
                    new_cache.group_window.setdefault(gid, w)
                for i, s in past.conv.items():
                    new_cache.conv[i] = s

        group_kv: dict[int, tuple[torch.Tensor | None, torch.Tensor | None]] = {}
        if past is not None:
            for gid in set(self._layer_group.values()):
                group_kv[gid] = past.get_group_kv(gid)

        h = inputs_embeds
        router_logits: list[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            h_norm = layer.operator_norm(h)
            if layer.is_attention:
                gid = self._layer_group[i]
                q = _rope_single(layer.self_attn.project_q(h_norm), cos, sin)
                k_base, v_base = group_kv.get(gid, (None, None))
                if layer.is_producer:
                    k_new, v_new = layer.self_attn.project_kv(h_norm)
                    k_new = _rope_single(k_new, cos, sin)
                    # v no rota
                    k_full = k_new if k_base is None else torch.cat([k_base, k_new], dim=2)
                    v_full = v_new if v_base is None else torch.cat([v_base, v_new], dim=2)
                    group_kv[gid] = (k_full, v_full)
                    if new_cache is not None:
                        new_cache.set_group_kv(gid, k_full, v_full)
                else:
                    k_full, v_full = k_base, v_base
                    assert k_full is not None, f"consumidora {i}: productora no corrió antes"
                # Recorte de CÓMPUTO exacto para grupos deslizantes: la query
                # más antigua (past_seq) necesita claves desde past_seq-W+1,
                # luego basta conservar los últimos W+lq-1. (En prefill lq~L no
                # recorta nada; en decode lq=1 recorta a W: el ahorro real.)
                # El ALMACÉN (caché) sí se recorta a W en set_group_kv.
                if cfg.group_all_sliding.get(gid):
                    keep = cfg.sliding_window + lq - 1
                    if k_full.shape[2] > keep:
                        k_full = k_full[:, :, -keep:, :]
                        v_full = v_full[:, :, -keep:, :]
                k_start = new_len - k_full.shape[2]
                bias = _turing_bias(full_mask, lq, k_full.shape[2], past_seq, k_start,
                                    h.device, h.dtype, window=layer.window)
                out = layer.self_attn.attend(q, k_full, v_full, bias, layer.window)
            else:
                if past_seq > 0:
                    state = past.conv.get(i) if past is not None else None
                    if state is None:  # pragma: no cover
                        state = layer.conv.init_state(b, h.device, h_norm.dtype)
                    outs, cur = [], state
                    for t in range(lq):
                        o_t, cur = layer.conv.step(h_norm[:, t : t + 1, :], cur)
                        outs.append(o_t)
                    out = torch.cat(outs, dim=1)
                    if new_cache is not None:
                        new_cache.set_conv(i, cur)
                else:
                    out = layer.conv(h_norm, conv_mask)
                    if new_cache is not None:
                        new_cache.set_conv(i, layer.conv.state_from_full(h_norm))
            h = h + out
            ffn_in = layer.ffn_norm(h)
            if layer.is_moe:
                ffn_out, rlog = layer.ffn(ffn_in)
                if return_router_logits:
                    router_logits.append(rlog)
            else:
                ffn_out = layer.ffn(ffn_in)
            h = h + ffn_out

        if new_cache is not None:
            new_cache.seq_len = new_len
        h = self.final_norm(h)
        if return_router_logits:
            return h, new_cache, router_logits
        return h, new_cache

    def step(self, token: torch.Tensor, past: TuringCache) -> tuple[torch.Tensor, TuringCache]:
        """Un paso incremental (verifica posición contigua)."""
        assert past.get_seq_length() >= 0
        h, nc = self.forward(input_ids=token, past=past, use_cache=True)
        assert nc is not None
        return h, nc


class TuringForCausalLM(nn.Module):
    def __init__(self, config: TuringConfig) -> None:
        super().__init__()
        self.config = config
        self.model = TuringModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        else:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=config.initializer_range)
        self.mtp_heads = nn.ModuleList(
            [TuringMTPHead(config.hidden_size, config.norm_eps) for _ in range(config.mtp_depth)]
        ) if config.use_mtp else nn.ModuleList()
        if config.use_mtp:
            for m in self.mtp_heads:
                nn.init.normal_(m.proj.weight, mean=0.0, std=config.initializer_range)

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "TuringForCausalLM":
        """Una línea para notebooks: from_preset("tiny" | "350m" | "700m" | "1.2b" | "2.6b").

        `overrides` reescribe campos de TuringConfig (p. ej. hidden_size=128).
        """
        import dataclasses

        from .config import get_preset

        cfg = get_preset(name)
        if overrides:
            cfg = dataclasses.replace(cfg, **overrides)
        return cls(cfg)

    # -- forward con pérdidas LM + MTP ----------------------------------
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past: TuringCache | None = None,
        use_cache: bool = False,
        labels: torch.Tensor | None = None,
        return_router_logits: bool = False,
    ) -> dict:
        need_router = (self.config.moe_z_loss_weight > 0 or return_router_logits) and labels is not None \
            and self.config.use_moe
        backbone_out = self.model(
            input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            position_ids=position_ids, past=past, use_cache=use_cache,
            return_router_logits=need_router,
        )
        hidden, new_cache = backbone_out[0], backbone_out[1]
        logits = self.lm_head(hidden)
        lm_loss, mtp_loss, z_loss = None, None, None
        if labels is not None:
            lm_loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1), ignore_index=-100,
            )
            if self.config.use_mtp and labels.shape[1] >= 3 + len(self.mtp_heads):
                mtp_loss = self._mtp_loss(hidden, labels)
            if need_router:
                assert len(backbone_out) == 3
                z_terms = [(rl.float().logsumexp(dim=-1) ** 2).mean() for rl in backbone_out[2]]
                z_loss = torch.stack(z_terms).mean().to(dtype=logits.dtype) if z_terms else None
        loss = lm_loss
        if mtp_loss is not None:
            loss = loss + self.config.mtp_loss_weight * mtp_loss
        if z_loss is not None:
            loss = loss + self.config.moe_z_loss_weight * z_loss
        out = {"logits": logits, "loss": loss, "past": new_cache, "hidden_states": hidden}
        out["lm_loss"], out["mtp_loss"], out["z_loss"] = lm_loss, mtp_loss, z_loss
        out["router_logits"] = backbone_out[2] if need_router and len(backbone_out) == 3 else None
        return out

    def _mtp_loss(self, hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """MTP con teacher forcing: cabeza d predice t+d+1 desde (h_d-1, emb(t+d))."""
        V = self.config.vocab_size
        L = labels.shape[1]
        losses = []
        h_prev = hidden
        for d, head in enumerate(self.mtp_heads):
            n_pred = L - (d + 2)  # posiciones t=0..n_pred-1 predicen labels[:, d+2+t]
            if n_pred <= 0:
                break
            emb_in = self.model.embed_tokens(labels[:, d + 1 : d + 1 + n_pred].clamp_min(0))
            x = head(h_prev[:, :n_pred, :], emb_in)
            l = self.lm_head(x)
            losses.append(F.cross_entropy(
                l.contiguous().view(-1, V), labels[:, d + 2 : d + 2 + n_pred].contiguous().view(-1),
                ignore_index=-100,
            ))
            h_prev = x  # la cabeza 2 parte del rasgo de la 1 (cadena con gradiente)
        return torch.stack(losses).mean()

    # -- muestreo --------------------------------------------------------
    @staticmethod
    def _penalize(logits: torch.Tensor, generated: torch.Tensor, penalty: float) -> torch.Tensor:
        if penalty == 1.0:
            return logits
        out = logits.clone()
        for bi in range(generated.shape[0]):
            for tok in torch.unique(generated[bi]).tolist():
                out[bi, tok] = out[bi, tok] / penalty if out[bi, tok] > 0 else out[bi, tok] * penalty
        return out

    @staticmethod
    def _min_p(logits: torch.Tensor, min_p: float) -> torch.Tensor:
        if min_p <= 0:
            return logits
        probs = F.softmax(logits, dim=-1)
        return logits.masked_fill(probs < min_p * probs.max(-1, keepdim=True).values, float("-inf"))

    @staticmethod
    def _top_p_idx(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        sp, si = torch.sort(probs, descending=True, dim=-1)
        cut = torch.cumsum(sp, dim=-1) > top_p
        cut[:, 0] = False
        sp = sp.masked_fill(cut, 0.0)
        sp = sp / sp.sum(-1, keepdim=True).clamp_min(1e-12)
        return torch.gather(si, -1, torch.multinomial(sp, 1))

    def _pick(self, logits: torch.Tensor, generated: torch.Tensor, do_sample: bool,
              temperature: float, top_p: float, min_p: float, rep: float,
              return_probs: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        lg = self._min_p(self._penalize(logits, generated, rep), min_p)
        if not do_sample:
            tok = lg.argmax(-1, keepdim=True)
            return tok, (F.softmax(lg, -1) if return_probs else None)
        lg = lg / max(temperature, 1e-6)
        tok = self._top_p_idx(lg, top_p) if top_p < 1.0 else torch.multinomial(F.softmax(lg, -1), 1)
        return tok, (F.softmax(lg, -1) if return_probs else None)

    # -- generación estándar ---------------------------------------------
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 64, do_sample: bool = False,
                 temperature: float = 1.0, top_p: float = 1.0, min_p: float = 0.0,
                 repetition_penalty: float = 1.0, eos_token_id: int | None = None) -> torch.Tensor:
        self.eval()
        eos = self.config.eos_token_id if eos_token_id is None else eos_token_id
        hidden, cache = self.model(input_ids=input_ids, use_cache=True)
        assert cache is not None
        h_last, logits_next = hidden[:, -1:, :], self.lm_head(hidden[:, -1:, :])
        gen = input_ids
        for _ in range(max_new_tokens):
            tok, _ = self._pick(logits_next[:, -1, :], gen, do_sample, temperature, top_p, min_p,
                                repetition_penalty)
            gen = torch.cat([gen, tok], dim=1)
            if eos is not None and bool((tok == eos).all()):
                break
            h_last, cache = self.model.step(tok, cache)
            logits_next = self.lm_head(h_last)
        return gen

    # -- generación especulativa (borrador MTP, 1 token extra) ------------
    @torch.no_grad()
    def generate_speculative(self, input_ids: torch.Tensor, max_new_tokens: int = 64,
                             do_sample: bool = False, temperature: float = 1.0, top_p: float = 1.0,
                             min_p: float = 0.0, repetition_penalty: float = 1.0,
                             eos_token_id: int | None = None,
                             return_stats: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """Decode especulativo con el propio modelo (sin draft externo).

        Por iteración: t1 sale de la LM (siempre aceptado); la cabeza MTP
        propone t2; UN forward verifica [t1, t2]. Greedy = idéntico a
        `generate(do_sample=False)`; muestreo = speculative sampling estándar.
        Solo batch=1 (la aceptación diverge por secuencia).
        Con `return_stats=True` devuelve (gen, {"drafts", "accepted", "rate"}).
        """
        assert self.config.use_mtp and len(self.mtp_heads) >= 1, "MTP requerido"
        assert input_ids.shape[0] == 1, "especulativo solo batch=1"
        self.eval()
        eos = self.config.eos_token_id if eos_token_id is None else eos_token_id
        prompt_len = input_ids.shape[1]
        hidden, cache = self.model(input_ids=input_ids, use_cache=True)
        assert cache is not None
        h_last, logits_next = hidden[:, -1:, :], self.lm_head(hidden[:, -1:, :])
        gen, made = input_ids, 0
        n_draft = n_accept = 0
        while made < max_new_tokens:
            t1, _ = self._pick(logits_next[:, -1, :], gen, do_sample, temperature, top_p, min_p,
                               repetition_penalty)
            x = self.mtp_heads[0](h_last, self.model.embed_tokens(t1))
            dl = self.lm_head(x)
            t2, q = self._pick(dl[:, -1, :], torch.cat([gen, t1], 1), do_sample, temperature,
                               top_p, min_p, repetition_penalty, return_probs=do_sample)
            snap = cache.snapshot()
            h_v, cache_v = self.model(input_ids=torch.cat([t1, t2], 1), past=cache, use_cache=True)
            assert cache_v is not None
            v1, v2 = self.lm_head(h_v[:, 0:1, :]), self.lm_head(h_v[:, 1:2, :])
            n_draft += 1
            if not do_sample:
                if v1.argmax(-1, keepdim=True).item() == t2.item():
                    n_accept += 1
                    gen = torch.cat([gen, t1, t2], 1)
                    cache, h_last, logits_next, made = cache_v, h_v[:, 1:2, :], v2, made + 2
                else:
                    h_t1, cache_t1 = self.model.step(t1, snap)
                    gen = torch.cat([gen, t1], 1)
                    cache, h_last, logits_next, made = cache_t1, h_t1, self.lm_head(h_t1), made + 1
            else:
                p = F.softmax(self._min_p(v1[:, -1, :], min_p) / max(temperature, 1e-6), -1)
                qq = q[0] if q is not None else None  # (V,)
                acc = True
                if qq is not None:
                    ratio = float(p[0, t2.item()] / qq[t2.item()].clamp_min(1e-12))
                    acc = torch.rand(()).item() < min(1.0, ratio)
                if acc:
                    n_accept += 1
                    gen = torch.cat([gen, t1, t2], 1)
                    cache, h_last, logits_next, made = cache_v, h_v[:, 1:2, :], v2, made + 2
                else:
                    resid = (p[0] - qq).clamp_min(0)
                    t2b = torch.multinomial(resid / resid.sum().clamp_min(1e-12), 1).unsqueeze(0)
                    h_b, cache_b = self.model(input_ids=torch.cat([t1, t2b], 1), past=snap, use_cache=True)
                    assert cache_b is not None
                    gen = torch.cat([gen, t1, t2b], 1)
                    cache, h_last, logits_next = cache_b, h_b[:, 1:2, :], self.lm_head(h_b[:, 1:2, :])
                    made += 2
            if eos is not None and bool((gen[:, -2:] == eos).any()):
                break
        gen = gen[:, : prompt_len + max_new_tokens]
        if return_stats:
            return gen, {"drafts": n_draft, "accepted": n_accept,
                         "rate": (n_accept / n_draft) if n_draft else 0.0}
        return gen

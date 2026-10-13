"""Backbone y CausalLM LFM2 densos (PyTorch puro, sin MoE).

Estructura por capa (sección 2.2 del report):
    h = h + Operador(RMSNorm(h))     # Operador = GQA o ShortConv
    h = h + SwiGLU(RMSNorm(h))

Modelo:
    embed -> N capas -> RMSNorm final -> lm_head (atado opcional).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .cache import LFM2Cache
from .config import LFM2Config
from .modules import (
    LFM2Attention,
    LFM2RMSNorm,
    LFM2RotaryEmbedding,
    LFM2ShortConv,
    LFM2SwiGLU,
)


def _build_attention_bias(
    attention_mask: torch.Tensor | None,
    lq: int,
    lk: int,
    past_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Construye sesgo aditivo (B,1,Lq,Lk) causal + padding.

    - `attention_mask`: (B, Lk) con 1=válido, 0=padding (longitud total
      pasado+actual), o None (= solo causal).
    - Causal: query global i (past_len+i) atiende a key j <= past_len+i.
    """
    causal = torch.full((lq, lk), float("-inf"), device=device, dtype=torch.float32)
    ar_q = torch.arange(past_len, past_len + lq, device=device).unsqueeze(1)  # (Lq,1)
    ar_k = torch.arange(lk, device=device).unsqueeze(0)  # (1,Lk)
    causal = torch.where(ar_k > ar_q, causal, torch.zeros_like(causal))
    bias = causal.unsqueeze(0).unsqueeze(0)  # (1,1,Lq,Lk)
    if attention_mask is not None:
        # (B,1,1,Lk): -inf en keys de padding.
        key_pad = (1.0 - attention_mask.to(torch.float32)).unsqueeze(1).unsqueeze(2)
        key_pad = key_pad * torch.finfo(torch.float32).min / 2  # aprox -inf estable
        # Alinear longitudes: si la máscara solo cubre lo nuevo, extender con unos.
        if key_pad.shape[-1] != lk:
            if key_pad.shape[-1] == lq and past_len > 0:
                prefix = torch.zeros(
                    key_pad.shape[0], 1, 1, past_len, device=device, dtype=torch.float32
                )
                key_pad = torch.cat([prefix, key_pad], dim=-1)
            else:  # pragma: no cover
                raise ValueError(f"attention_mask len {key_pad.shape[-1]} != Lk {lk}")
        bias = bias + key_pad
    if attention_mask is None and lq == 1 and past_len > 0:
        return None  # decode de 1 token sin padding: todo visible
    return bias.to(dtype)


class LFM2DecoderLayer(nn.Module):
    """Una capa del backbone: operador (conv o GQA) + SwiGLU, ambos pre-norm."""

    def __init__(self, config: LFM2Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.is_attention = config.layer_types[layer_idx] == "full_attention"
        if self.is_attention:
            self.self_attn = LFM2Attention(
                config.hidden_size, config.num_attention_heads, config.num_key_value_heads, config.norm_eps
            )
        else:
            self.conv = LFM2ShortConv(config.hidden_size, config.conv_kernel_size, config.conv_bias)
        self.feed_forward = LFM2SwiGLU(config.hidden_size, config.effective_intermediate_size)
        self.operator_norm = LFM2RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = LFM2RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward_full(
        self,
        h: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_bias: torch.Tensor | None,
        conv_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        residual = h
        if self.is_attention:
            out, _, _ = self.self_attn(self.operator_norm(h), position_embeddings, attention_bias)
        else:
            out = self.conv(self.operator_norm(h), conv_mask)
        h = residual + out
        return h + self.feed_forward(self.ffn_norm(h))


class LFM2Model(nn.Module):
    """Backbone LFM2: embeddings + capas híbridas + norma final."""

    def __init__(self, config: LFM2Config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([LFM2DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.rotary = LFM2RotaryEmbedding(config.head_dim, config.max_position_embeddings, config.rope_theta)
        self.final_norm = LFM2RMSNorm(config.hidden_size, eps=config.norm_eps)
        self._init_weights()

    # -- init -----------------------------------------------------------
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

    # -- utilidades -----------------------------------------------------
    def num_parameters(self, without_embeddings: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if without_embeddings:
            n -= self.embed_tokens.weight.numel()
        return n

    def _position_ids(
        self, b: int, lq: int, past_len: int, device: torch.device, position_ids: torch.Tensor | None
    ) -> torch.Tensor:
        if position_ids is not None:
            return position_ids
        pos = torch.arange(past_len, past_len + lq, device=device).unsqueeze(0).expand(b, lq)
        return pos

    # -- forward --------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past: LFM2Cache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, LFM2Cache | None]:
        """Forward completo o incremental.

        - Sin `past` (o `past.seq_len==0`): prefill/entrenamiento con toda
          la secuencia (conv full + atención full con sns causal/padding).
        - Con `past` y `use_cache`: decode incremental token a token (o por
          bloques): atención reutiliza KV, conv usa `step()` con estado.

        Devuelve `(hidden, new_cache|None)`.
        """
        cfg = self.config
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Especificar exactamente uno de input_ids / inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        b, lq, _ = inputs_embeds.shape
        past_len = past.get_seq_length() if past is not None else 0

        # Conv mask: HF anula estados de padding antes de in_proj.
        conv_mask = None
        if attention_mask is not None and attention_mask.shape[-1] == lq and past_len == 0:
            conv_mask = attention_mask
        elif attention_mask is not None and past is None:
            conv_mask = attention_mask

        pos_ids = self._position_ids(b, lq, past_len, inputs_embeds.device, position_ids)
        cos, sin = self.rotary(inputs_embeds, pos_ids)
        position_embeddings = (cos, sin)

        lk = past_len + lq
        full_mask = attention_mask
        if attention_mask is not None and past_len > 0 and attention_mask.shape[-1] == lq:
            # Máscara solo de lo nuevo: anteponer unos (el pasado se asume válido).
            prefix = torch.ones(b, past_len, device=attention_mask.device, dtype=attention_mask.dtype)
            full_mask = torch.cat([prefix, attention_mask], dim=1)
        attn_bias = _build_attention_bias(full_mask, lq, lk, past_len, inputs_embeds.device, inputs_embeds.dtype)

        new_cache = LFM2Cache() if use_cache else None

        h = inputs_embeds
        if past is not None and past_len > 0:
            # ---- decode incremental: reutiliza KV + estado conv ----
            if new_cache is not None:
                for i, t in past.k.items():
                    new_cache.k[i] = t
                for i, t in past.v.items():
                    new_cache.v[i] = t
                for i, t in past.conv.items():
                    new_cache.conv[i] = t
            for i, layer in enumerate(self.layers):
                h_norm = layer.operator_norm(h)
                if layer.is_attention:
                    out, k_full, v_full = layer.self_attn(
                        h_norm, position_embeddings, attn_bias, past.k.get(i), past.v.get(i)
                    )
                    if new_cache is not None:
                        new_cache.set_kv(i, k_full, v_full)
                else:
                    state = past.conv.get(i)
                    if state is None:  # pragma: no cover
                        state = layer.conv.init_state(b, h.device, h_norm.dtype)
                    outs = []
                    cur = state
                    for t in range(lq):
                        o_t, cur = layer.conv.step(h_norm[:, t : t + 1, :], cur)
                        outs.append(o_t)
                    out = torch.cat(outs, dim=1)
                    if new_cache is not None:
                        new_cache.set_conv(i, cur)
                h = h + out
                h = h + layer.feed_forward(layer.ffn_norm(h))
            if new_cache is not None:
                new_cache.seq_len = lk
        else:
            # ---- prefill / entrenamiento ----
            for layer in self.layers:
                h = layer.forward_full(h, position_embeddings, attn_bias, conv_mask)
            if new_cache is not None:
                # Caché exacta con segundo pase ligero (solo al pedir caché).
                new_cache = self._fill_cache(inputs_embeds, full_mask, pos_ids)
                new_cache.seq_len = lk
        h = self.final_norm(h)
        return h, new_cache

    @torch.no_grad()
    def _fill_cache(
        self, inputs_embeds: torch.Tensor, full_mask: torch.Tensor | None, pos_ids: torch.Tensor
    ) -> LFM2Cache:
        """Reconstruye KV + estados conv exactos tras un prefill (bajo no_grad)."""
        cache = LFM2Cache()
        cos, sin = self.rotary(inputs_embeds, pos_ids)
        pe = (cos, sin)
        b, lq, _ = inputs_embeds.shape
        lk = lq  # prefill: pasado vacío
        bias = _build_attention_bias(full_mask, lq, lk, 0, inputs_embeds.device, inputs_embeds.dtype)
        conv_mask = full_mask if full_mask is not None and full_mask.shape[-1] == lq else None
        h = inputs_embeds
        for i, layer in enumerate(self.layers):
            h_norm = layer.operator_norm(h)
            if layer.is_attention:
                out, k_full, v_full = layer.self_attn(h_norm, pe, bias)
                cache.set_kv(i, k_full, v_full)
            else:
                out = layer.conv(h_norm, conv_mask)
                cache.set_conv(i, layer.conv.state_from_full(h_norm))
            h = h + out
            h = h + layer.feed_forward(layer.ffn_norm(h))
        return cache

    # -- decode de un paso (helper para generación) ---------------------
    def step(
        self, token: torch.Tensor, past: LFM2Cache, position: int
    ) -> tuple[torch.Tensor, LFM2Cache]:
        """Un paso de generación. token: (B,1). Devuelve (hidden (B,1,d), caché)."""
        assert past is not None
        h_emb = self.embed_tokens(token)  # (B,1,d)
        pos_ids = torch.full((token.shape[0], 1), position, device=token.device, dtype=torch.long)
        cos, sin = self.rotary(h_emb, pos_ids)
        pe = (cos, sin)
        new_cache = LFM2Cache()
        h = h_emb
        for i, layer in enumerate(self.layers):
            h_norm = layer.operator_norm(h)
            if layer.is_attention:
                out, k_full, v_full = layer.self_attn(h_norm, pe, None, past.k.get(i), past.v.get(i))
                new_cache.set_kv(i, k_full, v_full)
            else:
                state = past.conv.get(i)
                if state is None:  # pragma: no cover
                    state = layer.conv.init_state(token.shape[0], h.device, h_norm.dtype)
                out, cur = layer.conv.step(h_norm, state)
                new_cache.set_conv(i, cur)
            h = h + out
            h = h + layer.feed_forward(layer.ffn_norm(h))
        new_cache.seq_len = position + 1
        return self.final_norm(h), new_cache


class LFM2ForCausalLM(nn.Module):
    """Modelo causal completo: backbone + head de lenguaje (atado opcional)."""

    def __init__(self, config: LFM2Config) -> None:
        super().__init__()
        self.config = config
        self.model = LFM2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self._init_head()

    def _init_head(self) -> None:
        if not self.config.tie_word_embeddings:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past: LFM2Cache | None = None,
        use_cache: bool = False,
        labels: torch.Tensor | None = None,
    ) -> dict:
        hidden, new_cache = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past=past,
            use_cache=use_cache,
        )
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            # Next-token loss con desplazamiento, ignora -100.
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100
            )
        return {"logits": logits, "loss": loss, "past": new_cache, "hidden_states": hidden}

    # -- generación -----------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        eos_token_id: int | None = None,
        pad_token_id: int = 0,
    ) -> torch.Tensor:
        """Generación autoregresiva pura (greedy o muestreo).

        - Prefill con caché (`_fill_cache` + `step`), luego loop token a token.
        - Soporta temperature, top-p, min-p y repetition penalty.
        - Devuelve la secuencia completa (prompt + generados).
        """
        self.eval()
        device = input_ids.device
        b, prompt_len = input_ids.shape
        eos = self.config.eos_token_id if eos_token_id is None else eos_token_id

        # Prefill exacto con caché.
        embeds = self.model.embed_tokens(input_ids)
        pos_ids = torch.arange(prompt_len, device=device).unsqueeze(0).expand(b, prompt_len)
        cache = self.model._fill_cache(embeds, None, pos_ids)
        cache.seq_len = prompt_len
        # Hidden del último token vía forward completo (exacto y simple).
        hidden_full, _ = self.model(input_ids=input_ids, use_cache=False)
        next_logits = self.lm_head(hidden_full[:, -1:, :])

        generated = input_ids
        for _ in range(max_new_tokens):
            logits = next_logits[:, -1, :]  # (B, V)
            if repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(logits, generated, repetition_penalty)
            if do_sample:
                if temperature != 1.0:
                    logits = logits / max(temperature, 1e-6)
                if min_p > 0.0:
                    logits = self._apply_min_p(logits, min_p)
                if top_p < 1.0:
                    next_tok = self._sample_top_p(logits, top_p)
                else:
                    next_tok = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            else:
                next_tok = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)
            if eos is not None and bool((next_tok == eos).all()):
                break
            # Un paso incremental exacto.
            h_step, cache = self.model.step(next_tok, cache, cache.seq_len)
            next_logits = self.lm_head(h_step)
        return generated

    @staticmethod
    def _apply_repetition_penalty(logits: torch.Tensor, generated: torch.Tensor, penalty: float) -> torch.Tensor:
        out = logits.clone()
        for bi in range(generated.shape[0]):
            for tok in torch.unique(generated[bi]).tolist():
                if out[bi, tok] > 0:
                    out[bi, tok] = out[bi, tok] / penalty
                else:
                    out[bi, tok] = out[bi, tok] * penalty
        return out

    @staticmethod
    def _apply_min_p(logits: torch.Tensor, min_p: float) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        top = probs.max(dim=-1, keepdim=True).values
        mask = probs < (min_p * top)
        return logits.masked_fill(mask, float("-inf"))

    @staticmethod
    def _sample_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cumsum > top_p
        cutoff[:, 0] = False
        sorted_probs[cutoff] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        sampled = torch.multinomial(sorted_probs, num_samples=1)
        return torch.gather(sorted_idx, -1, sampled)

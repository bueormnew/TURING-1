"""Módulos TURING-1 (PyTorch puro, reutiliza la base LFM2).

- `TuringAttention`: GQA + QK-Norm + RoPE con ventana deslizante opcional
  (recorte exacto a los últimos W) y backend SDPA cuando es exacto.
  `with_kv=False` para capas consumidoras CLA (solo q_proj).
- `TuringMultiscaleConv`: rama A k=3 (idéntica a LFM2) + rama B dilatada
  k=7/d=2 (campo receptivo 13), mezcla por gate por canal con init a
  identidad LFM2. Sin recurrencia: prefill paralelo, caché de 12 estados.
- `TuringMoE`: expertos finos estilo Kimi/DeepSeek (top-6/8, softmax,
  sesgo aux-free, 1+ expertos compartidos siempre activos).
- `TuringMTPHead`: borrador lineal barato para predicción multi-token.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from lfm2.modules import (
    LFM2RMSNorm,
    LFM2RotaryEmbedding,
    LFM2ShortConv,
    LFM2SwiGLU,
    apply_rotary_pos_emb,
)

__all__ = [
    "TuringAttention",
    "TuringMultiscaleConv",
    "TuringMoE",
    "TuringMTPHead",
    "LFM2RMSNorm",
    "LFM2RotaryEmbedding",
    "LFM2SwiGLU",
]


def _repeat_kv(hidden: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden
    b, n_kv, l, d = hidden.shape
    hidden = hidden[:, :, None, :, :].expand(b, n_kv, n_rep, l, d)
    return hidden.reshape(b, n_kv * n_rep, l, d)


# ---------------------------------------------------------------------------
# Atención con ventana deslizante + SDPA
# ---------------------------------------------------------------------------

class TuringAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        norm_eps: float = 1e-5,
        with_kv: bool = True,
        use_sdpa: bool = True,
    ) -> None:
        super().__init__()
        assert hidden_size % num_attention_heads == 0
        assert num_attention_heads % num_key_value_heads == 0
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scaling = self.head_dim ** -0.5
        self.with_kv = with_kv
        self.use_sdpa = use_sdpa

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=False)
        self.q_norm = LFM2RMSNorm(self.head_dim, eps=norm_eps)
        if with_kv:
            self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
            self.k_norm = LFM2RMSNorm(self.head_dim, eps=norm_eps)
        # Proyección de salida (mezcla de cabezas). Existe en productoras y
        # consumidoras CLA: ambas producen salida al residual. Sin ella las
        # cabezas se concatenan sin mezclar (bug silencioso: nq*hd == d
        # oculta el error de formas). Como en LFM2, sin sesgo.
        self.out_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=False)

    def project_q(self, h: torch.Tensor) -> torch.Tensor:
        b, l, _ = h.shape
        q = self.q_proj(h).view(b, l, self.num_attention_heads, self.head_dim)
        return self.q_norm(q).transpose(1, 2)  # (B, Hq, L, D)

    def project_kv(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.with_kv, "capa consumidora CLA: sin proyecciones K/V"
        b, l, _ = h.shape
        k = self.k_proj(h).view(b, l, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(h).view(b, l, self.num_key_value_heads, self.head_dim)
        return self.k_norm(k).transpose(1, 2), v.transpose(1, 2)

    def attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_bias: torch.Tensor | None = None,
        window: int | None = None,
    ) -> torch.Tensor:
        """Núcleo de atención.

        La ventana se aplica de dos formas exactas según el caso:
        - Decode (1 query): recorte físico a los últimos W (la query final
          solo necesita esos; el resto jamás se atiende).
        - Prefill/verificación (varias queries): SIN recorte; la ventana
          viene en `attention_bias` (máscara por posiciones absolutas que
          construye el modelo). Recortar aquí sería incorrecto porque las
          queries iniciales necesitan claves antiguas.
        """
        lq = q.shape[2]
        if lq == 1 and window is not None and k.shape[2] > window:
            k = k[:, :, -window:, :]
            v = v[:, :, -window:, :]
            if attention_bias is not None:
                attention_bias = attention_bias[:, :, :, -window:]
        if attention_bias is None and self.use_sdpa:
            # Sin máscara: SDPA exacto (causal bottom-right, válido con pasado
            # porque aquí nunca hay recorte sin bias). KV expandido antes por
            # compatibilidad de backends con GQA.
            k = _repeat_kv(k, self.num_key_value_groups)
            v = _repeat_kv(v, self.num_key_value_groups)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=(lq > 1))
            out = out.transpose(1, 2).contiguous().view(q.shape[0], lq, -1)
            return self.out_proj(out)
        # Ruta eager (con máscara: padding, ventana en prefill o verificación).
        k = _repeat_kv(k, self.num_key_value_groups)
        v = _repeat_kv(v, self.num_key_value_groups)
        if attention_bias is None and lq > 1:
            # Solo capas globales sin trim llegan aquí (deslizantes en prefill
            # siempre traen bias con ventana). lk>=lq garantizado.
            lk = k.shape[2]
            causal = torch.full((lq, lk), float("-inf"), device=q.device, dtype=q.dtype)
            causal = torch.triu(causal, diagonal=(lk - lq) + 1)
            attention_bias = causal.unsqueeze(0).unsqueeze(0)
        w = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        if attention_bias is not None:
            w = w + attention_bias.to(dtype=w.dtype)
        w = F.softmax(w, dim=-1, dtype=torch.float32).to(dtype=q.dtype)
        out = torch.matmul(w, v)
        out = out.transpose(1, 2).contiguous().view(q.shape[0], lq, -1)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Conv multiescala (k=3 + dilatada k=7/d=2, gate a identidad LFM2)
# ---------------------------------------------------------------------------

class TuringMultiscaleConv(LFM2ShortConv):
    """Mejora de la conv corta sin recurrencia.

    z1 = DWConv k=3 (rama LFM2, pesos compatibles para warm-start).
    z2 = DWConv k=7, dilatación 2 (campo receptivo 13, rango medio).
    z  = (1-g)·z1 + g·z2,  g = sigmoid(mix) por canal, mix₀=-3 (≈LFM2).
    Estado de caché: últimos 12 `y` (sirve a ambas ramas).
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 3,
        dilated_kernel: int = 7,
        dilation: int = 2,
        bias: bool = False,
        gate_init: float = -3.0,
    ) -> None:
        super().__init__(hidden_size, kernel_size, bias)
        self.dilated_kernel = dilated_kernel
        self.dilation = dilation
        self.state_len = (dilated_kernel - 1) * dilation
        self.conv_dil = nn.Conv1d(
            hidden_size, hidden_size, dilated_kernel,
            groups=hidden_size, bias=bias, dilation=dilation,
            padding=(dilated_kernel - 1) * dilation,
        )
        self.mix = nn.Parameter(torch.full((hidden_size,), gate_init))

    def _blend(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.mix).view(1, -1, 1).to(dtype=z1.dtype)
        return (1.0 - g) * z1 + g * z2

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if attention_mask is not None:
            hidden_states = (hidden_states * attention_mask.to(dtype=hidden_states.dtype)[:, :, None]).to(
                hidden_states.dtype
            )
        b_gate, c_gate, x = self._split_gates(hidden_states)
        y = b_gate * x
        z1 = self._causal_depthwise_conv(y)
        z2 = F.conv1d(
            y, self.conv_dil.weight, self.conv_dil.bias,
            padding=(self.dilated_kernel - 1) * self.dilation,
            dilation=self.dilation, groups=self.hidden_size,
        )[:, :, : y.shape[-1]]
        o = c_gate * self._blend(z1, z2)
        return self.out_proj(o.transpose(-1, -2).contiguous())

    @torch.no_grad()
    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, self.state_len, device=device, dtype=dtype)

    def step(self, hidden_t: torch.Tensor, conv_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b_gate, c_gate, x = self._split_gates(hidden_t)
        y_t = b_gate * x  # (B, d, 1)
        buf = torch.cat([conv_state.to(dtype=y_t.dtype), y_t], dim=-1)  # (B, d, 13)
        w1 = self.conv.weight.squeeze(1)  # (d, 3)
        z1 = (buf[:, :, -3:] * w1.unsqueeze(0)).sum(-1, keepdim=True)
        if self.conv.bias is not None:
            z1 = z1 + self.conv.bias.view(1, -1, 1)
        w2 = self.conv_dil.weight.squeeze(1)  # (d, 7): taps t-12, t-10, ..., t
        # Orden ascendente: taps[j] = y[t-12+2j], igual que F.conv1d con
        # dilatación (w[0] pondera el tap más antiguo). El orden descendente
        # invertiría los pesos y rompería la equivalencia full/incremental.
        tap_idx = torch.arange(0, self.state_len + 1, 2, device=buf.device)  # 0,2,...,12
        taps = buf[:, :, tap_idx]
        z2 = (taps * w2.unsqueeze(0)).sum(-1, keepdim=True)
        if self.conv_dil.bias is not None:
            z2 = z2 + self.conv_dil.bias.view(1, -1, 1)
        new_state = buf[:, :, -self.state_len :].contiguous()
        o_t = (c_gate * self._blend(z1, z2)).transpose(-1, -2).contiguous()
        return self.out_proj(o_t), new_state.to(conv_state.dtype)

    def state_from_full(self, hidden_states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            b_gate, _, x = self._split_gates(hidden_states)
            y = b_gate * x  # (B, d, L)
            s = self.state_len
            if y.shape[-1] >= s:
                return y[:, :, -s:].contiguous()
            pad = torch.zeros(y.shape[0], y.shape[1], s - y.shape[-1], device=y.device, dtype=y.dtype)
            return torch.cat([pad, y], dim=-1)


# ---------------------------------------------------------------------------
# MoE fino estilo Kimi/DeepSeek (top-6/8, softmax, sesgo aux-free, shared)
# ---------------------------------------------------------------------------

class TuringMoE(nn.Module):
    """MoE con expertos SwiGLU finos.

    - Router softmax + sesgo `expert_bias` (solo decide el top-k, estilo
      aux-loss-free de DeepSeek/Kimi); el peso usado es la prob. softmax
      renormalizada entre los k elegidos.
    - `shared_experts` SwiGLU siempre activos (conocimiento común).
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int = 32,
        num_experts_per_tok: int = 6,
        moe_intermediate_size: int = 512,
        num_shared_experts: int = 1,
        norm_topk_prob: bool = True,
        dispatch: str = "auto",
    ) -> None:
        super().__init__()
        assert num_experts_per_tok <= num_experts
        assert dispatch in ("loop", "batched", "auto")
        self.num_experts = num_experts
        self.top_k = num_experts_per_tok
        self.intermediate = moe_intermediate_size
        self.norm_topk_prob = norm_topk_prob
        self.dispatch = dispatch
        # Umbral medido en CPU (E=32,d=512,k=6): N=1 loop, N>=2 batched.
        self.auto_threshold = 8
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        # Persistente (va en state_dict/checkpoints) pero NO es parámetro:
        # el optimizador no lo toca; se ajusta con update_bias_ (aux-free).
        self.register_buffer("expert_bias", torch.zeros(num_experts), persistent=True)
        self.gate_up = nn.Parameter(torch.empty(num_experts, 2 * moe_intermediate_size, hidden_size))
        self.down = nn.Parameter(torch.empty(num_experts, hidden_size, moe_intermediate_size))
        self.shared = nn.ModuleList(
            [LFM2SwiGLU(hidden_size, moe_intermediate_size) for _ in range(num_shared_experts)]
        )
        self.reset_parameters()  # nunca basura: torch.empty puede reciclar NaNs

    def reset_parameters(self, std: float = 0.02) -> None:
        """Init Normal (como nn.Linear). TuringModel._init_weights la repite
        con el std de la config; esto solo garantiza no-garbage standalone."""
        nn.init.normal_(self.gate_up, mean=0.0, std=std)
        nn.init.normal_(self.down, mean=0.0, std=std)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        orig_shape = h.shape
        flat = h.reshape(-1, h.shape[-1])  # (N, d)
        logits = self.router(flat)  # (N, E)
        scores = F.softmax(logits, dim=-1)
        _, idx = torch.topk(scores + self.expert_bias.to(dtype=scores.dtype), k=self.top_k, dim=-1)
        w = torch.gather(scores, 1, idx)
        if self.norm_topk_prob:
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-6)

        out = torch.zeros_like(flat)
        for m in self.shared:
            out = out + m(flat)
        n_drafts = flat.shape[0] * self.top_k
        use_batched = (self.dispatch == "batched") or (self.dispatch == "auto" and n_drafts > self.auto_threshold)
        if use_batched:
            out = out + self._batched_experts(flat, idx, w)
        else:
            # Despacho por experto (bucle solo sobre expertos tocados;
            # producción GPU: grouped GEMM; en CPU ver benchmark loop/batched).
            for e in torch.unique(idx).tolist():
                rows, cols = torch.where(idx == e)
                tok = flat[rows]  # (M, d)
                gu = tok @ self.gate_up[e].T
                g, u = gu.chunk(2, dim=-1)
                y = (F.silu(g) * u) @ self.down[e].T
                out.index_add_(0, rows, (y * w[rows, cols].unsqueeze(-1)).to(dtype=out.dtype))
        return out.view(orig_shape), logits.view(*orig_shape[:-1], -1)

    def _batched_experts(self, flat: torch.Tensor, idx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Despacho vectorizado: ordena drafts por experto y usa 2 bmm.

        Agrupa los N*k drafts por experto (argsort), los empaqueta en un lote
        acolchado (E, maxM, d) y computa gate_up/down con 2 bmm + scatter-add.
        Exacto salvo orden de suma fp. El acolchado se anula con máscara
        (y además da cero solo: silu(0)*0=0).
        """
        N, d = flat.shape
        k = idx.shape[1]
        E = self.num_experts
        M = N * k
        order = torch.argsort(idx.reshape(-1), stable=True)  # (M,) por experto
        rows = torch.arange(N, device=flat.device).repeat_interleave(k)[order]
        wrow = w.reshape(-1)[order]
        expert = idx.reshape(-1)[order]
        cnt = torch.bincount(expert, minlength=E)  # (E,)
        starts = torch.cumsum(cnt, dim=0) - cnt
        slot = torch.arange(M, device=flat.device) - starts[expert]
        maxM = int(cnt.max().item())
        tok_padded = flat.new_zeros((E, maxM, d))
        tok_padded[expert, slot] = flat[rows]
        w_padded = flat.new_zeros((E, maxM))
        # Cast explícito: bajo autocast AMP el softmax corre en fp32 y el
        # index_put exige mismo dtype (el path loop ya castea con .to()).
        w_padded[expert, slot] = wrow.to(w_padded.dtype)
        mask = (torch.arange(maxM, device=flat.device).unsqueeze(0) < cnt.unsqueeze(1)).to(tok_padded.dtype)
        gu = torch.bmm(tok_padded, self.gate_up.transpose(-1, -2))
        g, u = gu.chunk(2, dim=-1)
        o = torch.bmm(F.silu(g) * u, self.down.transpose(-1, -2))
        o = o * (w_padded * mask).unsqueeze(-1)
        out = flat.new_zeros((N, d))
        out.index_add_(0, rows, o[expert, slot].to(out.dtype))
        return out

    @torch.no_grad()
    def update_bias_(self, topk_idx: torch.Tensor, lr: float = 0.01) -> None:
        """Balanceo aux-loss-free estilo DeepSeek/Kimi (llamar tras cada paso).

        `topk_idx`: (N, k) expertos elegidos en el batch. Los sobrecargados
        bajan su sesgo (se eligen menos) y viceversa. Solo afecta a la
        SELECCIÓN top-k, nunca a los pesos usados.
        """
        counts = torch.bincount(topk_idx.reshape(-1), minlength=self.num_experts).to(
            dtype=self.expert_bias.dtype
        )
        self.expert_bias.add_(torch.sign(counts.mean() - counts) * lr)


# ---------------------------------------------------------------------------
# Cabeza MTP (borrador barato para decodificación especulativa)
# ---------------------------------------------------------------------------

class TuringMTPHead(nn.Module):
    """Borrador EAGLE-lite: Linear(2d->d) + RMSNorm + residual de `h`.

    Entrada: (h_t, emb(t+1)) -> rasgo para predecir t+2 con la head LM
    compartida (sin matrices gigantes nuevas -> 4-bit-friendly).
    """

    def __init__(self, hidden_size: int, norm_eps: float = 1e-5) -> None:
        super().__init__()
        self.proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.norm = LFM2RMSNorm(hidden_size, eps=norm_eps)

    def forward(self, h: torch.Tensor, emb_next: torch.Tensor) -> torch.Tensor:
        x = self.proj(torch.cat([h, emb_next], dim=-1))
        return self.norm(x) + h

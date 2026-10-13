"""Bloques elementales LFM2 (densos, PyTorch puro).

Referencia: sección 2.2 del Technical Report + `modeling_lfm2.py` de HF.

1. `LFM2RMSNorm`: pre-norm y norma final, QK-Norm por cabeza.
2. `LFM2RotaryEmbedding`: RoPE clásico, theta=1e6, dim de cabeza=64.
3. `LFM2Attention`: GQA con QK-Norm + RoPE + softmax causal.
4. `LFM2ShortConv`: convolución corta causal depthwise con doble puerta:
       (B, C, x) = in_proj(h);  y = B * x
       z = Conv_k(y)            # depthwise 1D causal, k=3
       o = out_proj(C * z)
5. `LFM2SwiGLU`: MLP SwiGLU w2(silu(w1(x)) * w3(x)).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class LFM2RMSNorm(nn.Module):
    """RMSNorm idéntico al de HF (`Lfm2RMSNorm`, equivalente a T5LayerNorm)."""

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.variance_epsilon)
        return (self.weight * x.to(dtype)).to(dtype)

    def extra_repr(self) -> str:  # pragma: no cover
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

class LFM2RotaryEmbedding(nn.Module):
    """RoPE clásico (tipo 'default').

    inv_freq[i] = 1 / (theta ** (2i / head_dim)), i = 0..head_dim/2-1.
    No usa escalado de atención (factor 1.0).
    """

    def __init__(self, head_dim: int, max_position_embeddings: int = 128000, theta: float = 1_000_000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, L, d) solo se usa para dispositivo/dtype; position_ids: (B, L)
        freqs = position_ids.to(torch.float32)[..., None] * self.inv_freq.to(device=x.device, dtype=torch.float32)
        emb = torch.cat((freqs, freqs), dim=-1)  # (B, L, head_dim)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aplica RoPE a q/k.

    q/k: (B, H, L, D); cos/sin: (B, L, D).
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _repeat_kv(hidden: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expande KV de (B, n_kv, L, D) a (B, n_q, L, D) repitiendo grupos."""
    if n_rep == 1:
        return hidden
    b, n_kv, l, d = hidden.shape
    hidden = hidden[:, :, None, :, :].expand(b, n_kv, n_rep, l, d)
    return hidden.reshape(b, n_kv * n_rep, l, d)


# ---------------------------------------------------------------------------
# Atención GQA con QK-Norm
# ---------------------------------------------------------------------------

class LFM2Attention(nn.Module):
    """GQA + QK-Norm (RMS por cabeza) + RoPE + salida lineal. Sin sesgos."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        norm_eps: float = 1e-5,
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

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=False)
        self.q_layernorm = LFM2RMSNorm(self.head_dim, eps=norm_eps)
        self.k_layernorm = LFM2RMSNorm(self.head_dim, eps=norm_eps)

    def _project_qkv(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, l, _ = hidden.shape
        q = self.q_proj(hidden).view(b, l, self.num_attention_heads, self.head_dim)
        k = self.k_proj(hidden).view(b, l, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(hidden).view(b, l, self.num_key_value_heads, self.head_dim)
        q = self.q_layernorm(q).transpose(1, 2)  # (B, Hq, L, D)
        k = self.k_layernorm(k).transpose(1, 2)  # (B, Hkv, L, D)
        v = v.transpose(1, 2)
        return q, k, v

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_bias: torch.Tensor | None = None,
        key_states: torch.Tensor | None = None,
        value_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward con KV ya concatenados (incluyen pasado + actual).

        - `hidden_states`: (B, Lq, d) ya normalizado.
        - `key_states`/`value_states`: (B, Hkv, Lk, D) completos (rotados) o None.
        - `attention_bias`: (B, 1, Lq, Lk) aditivo (-inf donde enmascarar) o None (= causal).

        Devuelve `(salida, k_full, v_full)` donde k_full/v_full son los
        estados rotados completos listos para guardar en caché.
        """
        cos, sin = position_embeddings
        if key_states is None or value_states is None:
            q, k, v = self._project_qkv(hidden_states)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            k_full, v_full = k, v
        else:
            q_new, k_new, v_new = self._project_qkv(hidden_states)
            q_new, k_new = apply_rotary_pos_emb(q_new, k_new, cos, sin)
            q = q_new
            k_full = torch.cat([key_states, k_new], dim=2)
            v_full = torch.cat([value_states, v_new], dim=2)
            k, v = k_full, v_full

        k_exp = _repeat_kv(k, self.num_key_value_groups)
        v_exp = _repeat_kv(v, self.num_key_value_groups)

        lq, lk = q.shape[2], k.shape[2]
        if attention_bias is None and lq > 1:
            # Máscara causal densa (entrenamiento / prefill sin padding).
            causal = torch.full((lq, lk), float("-inf"), device=q.device, dtype=q.dtype)
            # La ventana válida depende del desplazamiento pasado = lk - lq.
            past = lk - lq
            causal = torch.triu(causal, diagonal=past + 1)
            attention_bias = causal.unsqueeze(0).unsqueeze(0)  # (1,1,Lq,Lk)

        attn_weights = torch.matmul(q, k_exp.transpose(2, 3)) * self.scaling
        if attention_bias is not None:
            attn_weights = attn_weights + attention_bias.to(dtype=attn_weights.dtype)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype=q.dtype)
        attn_out = torch.matmul(attn_weights, v_exp)  # (B, Hq, Lq, D)
        attn_out = attn_out.transpose(1, 2).contiguous().view(q.shape[0], lq, -1)
        return self.out_proj(attn_out), k_full, v_full


# ---------------------------------------------------------------------------
# Convolución corta con doble puerta (bloque LIV)
# ---------------------------------------------------------------------------

class LFM2ShortConv(nn.Module):
    """Bloque de convolución corta con doble puerta de LFM2.

    Ecuación del report (sección 2.2):
        (B, C, x) = Linear(h)      # d -> 3d por posición
        y = B * x
        z = Conv_k(y)              # depthwise 1D causal, k=3
        o = Linear_out(C * z)      # d -> d

    El `nn.Conv1d` se guarda como parámetro (grupos=d) y el forward usa
    `F.conv1d` con pad izquierdo k-1 (causal) + truncado, idéntico a
    `causal_conv1d_fn` de HF. Para decode paso a paso se usa el método
    `step()` con estado (B, d, k-1) que guarda las últimas activaciones `y`.
    """

    def __init__(self, hidden_size: int, kernel_size: int = 3, bias: bool = False) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.in_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=bias,
            padding=kernel_size - 1,
        )
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

    def _split_gates(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # h: (B, L, d) -> BCx: (B, L, 3d) -> B,C,x: (B, d, L) cada uno
        bcx = self.in_proj(h).transpose(-1, -2)
        return bcx.chunk(3, dim=-2)

    def _causal_depthwise_conv(self, y: torch.Tensor) -> torch.Tensor:
        # y: (B, d, L) -> z: (B, d, L) causal (pad izq. k-1, truncar).
        z = F.conv1d(y, self.conv.weight, self.conv.bias, padding=self.kernel_size - 1, groups=self.hidden_size)
        return z[:, :, : y.shape[-1]]

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward completo (entrenamiento / prefill). h: (B, L, d)."""
        if attention_mask is not None:
            # Cero en tokens de padding (igual que HF `apply_mask_to_padding_states`).
            hidden_states = (hidden_states * attention_mask.to(dtype=hidden_states.dtype)[:, :, None]).to(
                hidden_states.dtype
            )
        b_gate, c_gate, x = self._split_gates(hidden_states)  # (B, d, L)
        y = b_gate * x
        z = self._causal_depthwise_conv(y)
        o = c_gate * z  # (B, d, L)
        return self.out_proj(o.transpose(-1, -2).contiguous())

    @torch.no_grad()
    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, self.kernel_size - 1, device=device, dtype=dtype)

    def step(
        self, hidden_t: torch.Tensor, conv_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Un paso de decode. hidden_t: (B, 1, d); conv_state: (B, d, k-1).

        Devuelve (salida (B,1,d), nuevo estado). El estado guarda las últimas
        k-1 activaciones `y` (pre-conv), igual que HF `causal_conv1d_update`.
        """
        b_gate, c_gate, x = self._split_gates(hidden_t)  # (B, d, 1)
        y_t = b_gate * x  # (B, d, 1)
        buf = torch.cat([conv_state.to(dtype=y_t.dtype), y_t], dim=-1)  # (B, d, k)
        w = self.conv.weight.squeeze(1)  # (d, k)
        z_t = (buf * w.unsqueeze(0)).sum(dim=-1, keepdim=True)  # (B, d, 1)
        if self.conv.bias is not None:
            z_t = z_t + self.conv.bias.view(1, -1, 1)
        new_state = buf[:, :, -(self.kernel_size - 1):].contiguous() if self.kernel_size > 1 else buf[:, :, :0]
        o_t = (c_gate * z_t).transpose(-1, -2).contiguous()  # (B, 1, d)
        return self.out_proj(o_t), new_state.to(conv_state.dtype)

    def state_from_full(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Calcula el estado (últimas k-1 `y`) tras un prefill, sin grad."""
        with torch.no_grad():
            _, _, _ = hidden_states.shape
            b_gate, _, x = self._split_gates(hidden_states)
            y = b_gate * x  # (B, d, L)
            k = self.kernel_size - 1
            if y.shape[-1] >= k and k > 0:
                return y[:, :, -k:].contiguous()
            pad = torch.zeros(y.shape[0], y.shape[1], k - y.shape[-1], device=y.device, dtype=y.dtype)
            return torch.cat([pad, y], dim=-1)


# ---------------------------------------------------------------------------
# MLP SwiGLU
# ---------------------------------------------------------------------------

class LFM2SwiGLU(nn.Module):
    """SwiGLU: w2(silu(w1(x)) * w3(x)). Sin sesgos."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

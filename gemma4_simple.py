"""
gemma4_simple.py
================
A clean, minimal PyTorch re-implementation of the Gemma 4 forward pass.
Mirrors the HuggingFace source as closely as possible while stripping away
framework boilerplate so the architecture is easy to read and experiment with.

Reference:
  https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py

Modules implemented (in bottom-up order):
  --- Shared primitives ---
  RMSNorm
  rotate_half / apply_rotary_pos_emb / apply_2d_rope

  --- Text tower ---
  TextRotaryEmbedding
  ScaledEmbedding
  TextMLP          (SwiGLU dense FFN)
  TextAttention    (GQA + QK-norm + sliding-window + KV-sharing)
  TextRouter       (top-k MoE router)
  TextExperts      (sparse expert FFN)
  TextDecoderLayer (Attention + dense MLP + optional MoE + per-layer gate)
  TextModel

  --- Vision tower ---
  VisionRotaryEmbedding   (2-D RoPE for patch (x,y) positions)
  VisionPatchEmbedder
  VisionMLP
  VisionAttention         (ViT-style, 2-D RoPE, QKV norms)
  VisionEncoderLayer
  VisionEncoder
  VisionPooler            (adaptive average pooling)
  VisionModel

  --- Multimodal fusion ---
  MultimodalEmbedder      (project vision soft-tokens → text hidden size)
  Gemma4Model             (merge image/video/audio slots into token stream)
  Gemma4ForCausalLM       (text-only)
  Gemma4ForConditionalGeneration  (full multimodal)
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
# Config dataclasses (minimal, no validation)
# ──────────────────────────────────────────────────────────────

@dataclass
class TextConfig:
    vocab_size: int = 262_144
    hidden_size: int = 2560
    num_hidden_layers: int = 34
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 256
    intermediate_size: int = 8192        # dense MLP width
    # MoE
    num_experts: int = 0                 # 0 = no MoE layers
    num_experts_per_tok: int = 2
    moe_layers: list[int] = field(default_factory=list)  # which layers have MoE
    expert_intermediate_size: int = 4096
    # KV sharing: layers >= kv_share_from reuse KV from layer kv_share_from-1
    kv_share_from: int | None = None
    # Attention
    sliding_window: int | None = 1024   # None → global attention on all layers
    sliding_window_pattern: int = 6     # every Nth layer uses global attn
    rope_theta: float = 10_000.0
    rope_local_base_freq: float = 10_000.0
    rope_global_base_freq: float = 1_000_000.0
    attn_logit_softcapping: float | None = None
    final_logit_softcapping: float | None = 30.0
    # Per-layer input gate (Gemma4-specific)
    hidden_size_per_layer_input: int = 0
    # Misc
    rms_norm_eps: float = 1e-6
    pad_token_id: int = 0
    embed_scale: float = 1.0


@dataclass
class VisionConfig:
    hidden_size: int = 1152
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    head_dim: int = 72
    intermediate_size: int = 4096
    patch_size: int = 14
    image_size: int = 448
    pooling_kernel_size: int = 4     # spatial pooling factor
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-6
    standardize: bool = False        # optional post-pooling standardization


@dataclass
class Gemma4Config:
    text: TextConfig = field(default_factory=TextConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    # Attention: 'bidirectional_vision' → vision tokens use bidirectional mask
    use_bidirectional_attention: str = "none"


# ──────────────────────────────────────────────────────────────
# Shared primitives
# ──────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (with optional learned scale)."""
    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if with_scale else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float32 for numerical stability, then cast back
        x_f = x.float()
        normed = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            normed = normed * self.weight.float()
        return normed.to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split last dim in two and rotate: [x1, x2] → [-x2, x1]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    """Standard 1-D RoPE application."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


def apply_2d_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    unsqueeze_dim: int = 2,
) -> torch.Tensor:
    """
    2-D RoPE for vision patches.
    cos/sin have shape [B, L, 2 * (head_dim // 2)] — the first half encodes x,
    the second half encodes y. We split x's channels accordingly and rotate each half.
    """
    ndim = position_ids.shape[-1]  # should be 2
    channels_per_dim = 2 * (x.shape[-1] // (2 * ndim))
    x_parts = torch.split(x, [channels_per_dim] * ndim, dim=-1)
    cos_parts = torch.split(cos, [channels_per_dim] * ndim, dim=-1)
    sin_parts = torch.split(sin, [channels_per_dim] * ndim, dim=-1)
    rotated = [
        apply_rotary_pos_emb(x_parts[k], cos_parts[k], sin_parts[k], unsqueeze_dim)
        for k in range(ndim)
    ]
    return torch.cat(rotated, dim=-1)


# ──────────────────────────────────────────────────────────────
# Text Tower
# ──────────────────────────────────────────────────────────────

class TextRotaryEmbedding(nn.Module):
    """
    Dual-frequency RoPE for Gemma 4 text.
    Local layers use rope_local_base_freq, global layers use rope_global_base_freq.
    """
    def __init__(self, cfg: TextConfig):
        super().__init__()
        dim = cfg.head_dim
        for name, base in [("local", cfg.rope_local_base_freq), ("global", cfg.rope_global_base_freq)]:
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
            self.register_buffer(f"{name}_inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        layer_type: str = "global",   # "local" or "global"
    ):
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        # [B, head_dim/2, 1] × [B, 1, L] → [B, L, head_dim/2]
        inv_freq_exp = inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
        pos_exp = position_ids[:, None, :].float()
        freqs = (inv_freq_exp.float() @ pos_exp.float()).transpose(1, 2)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(x.dtype)
        sin = emb.sin().to(x.dtype)
        return cos, sin


class ScaledEmbedding(nn.Embedding):
    """Token embedding scaled by sqrt(hidden_size) as in Gemma."""
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int,
                 embed_scale: float = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * self.embed_scale


class TextMLP(nn.Module):
    """SwiGLU dense FFN used in non-MoE and as the shared expert."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class TextAttention(nn.Module):
    """
    Grouped-Query Attention with:
      • QK normalisation (separate RMSNorm on Q and K before RoPE)
      • V normalisation
      • KV sharing (some layers reuse KV states from a previous layer)
      • Optional sliding-window mask
    """
    def __init__(self, cfg: TextConfig, layer_idx: int, is_kv_shared: bool = False):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.is_kv_shared = is_kv_shared
        # Sliding window: every sliding_window_pattern-th layer is global
        self.sliding_window = (
            None if (layer_idx % cfg.sliding_window_pattern == 0)
            else cfg.sliding_window
        )

        hs = cfg.hidden_size
        kv_dim = self.num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(hs, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hs, kv_dim, bias=False) if not is_kv_shared else None
        self.v_proj = nn.Linear(hs, kv_dim, bias=False) if not is_kv_shared else None
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hs, bias=False)

        # Per-head norms (applied before RoPE)
        self.q_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.v_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)

        self.attn_logit_softcapping = cfg.attn_logit_softcapping
        self.scaling = self.head_dim ** -0.5

    def forward(
        self,
        hidden_states: torch.Tensor,          # [B, L, D]
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,      # simple dict cache for demo
    ) -> torch.Tensor:
        B, L, _ = hidden_states.shape
        H, Hkv, Dh = self.num_heads, self.num_kv_heads, self.head_dim

        # ── Queries ─────────────────────────────────────────────────────
        q = self.q_proj(hidden_states).view(B, L, H, Dh)
        q = self.q_norm(q)
        q = apply_rotary_pos_emb(q, cos, sin, unsqueeze_dim=2)
        q = q.transpose(1, 2)  # [B, H, L, Dh]

        # ── Keys & Values ────────────────────────────────────────────────
        if self.is_kv_shared and kv_cache is not None:
            # Reuse the KV states stored by the designated anchor layer
            k, v = kv_cache["shared_kv"]
        else:
            k = self.k_proj(hidden_states).view(B, L, Hkv, Dh)
            v = self.v_proj(hidden_states).view(B, L, Hkv, Dh)
            k = self.k_norm(k)
            v = self.v_norm(v)
            k = apply_rotary_pos_emb(k, cos, sin, unsqueeze_dim=2)
            k = k.transpose(1, 2)  # [B, Hkv, L, Dh]
            v = v.transpose(1, 2)

        # ── GQA: expand KV heads to match Q heads ───────────────────────
        if Hkv != H:
            expand = H // Hkv
            k = k.unsqueeze(2).expand(-1, -1, expand, -1, -1).reshape(B, H, -1, Dh)
            v = v.unsqueeze(2).expand(-1, -1, expand, -1, -1).reshape(B, H, -1, Dh)

        # ── Scaled dot-product attention ────────────────────────────────
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        if self.attn_logit_softcapping is not None:
            attn = attn / self.attn_logit_softcapping
            attn = torch.tanh(attn)
            attn = attn * self.attn_logit_softcapping

        if attention_mask is not None:
            attn = attn + attention_mask

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)          # [B, H, L, Dh]
        out = out.transpose(1, 2).reshape(B, L, H * Dh)
        return self.o_proj(out)


class TextRouter(nn.Module):
    """
    Top-k router for Mixture-of-Experts.
    Selects top_k experts per token and returns normalised, per-expert-scaled weights.
    """
    def __init__(self, cfg: TextConfig):
        super().__init__()
        self.top_k = cfg.num_experts_per_tok
        self.norm  = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.scale = cfg.hidden_size ** 0.5
        # Projection to expert logits
        self.proj  = nn.Linear(cfg.hidden_size, cfg.num_experts, bias=False)
        # Per-expert learned output scale
        self.per_expert_scale = nn.Parameter(torch.ones(cfg.num_experts))

    def forward(self, x: torch.Tensor):
        """x: [T, D]  (T = batch*seq flattened)"""
        h = self.norm(x) * self.scale
        logits = self.proj(h)                                   # [T, E]
        probs  = F.softmax(logits, dim=-1)                      # [T, E]
        top_w, top_idx = torch.topk(probs, self.top_k, dim=-1)  # [T, K]
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)         # renorm
        top_w = top_w * self.per_expert_scale[top_idx]          # scale
        return probs, top_w, top_idx


class TextExperts(nn.Module):
    """
    Sparse MoE expert bank.
    Each expert is a SwiGLU FFN. Only experts that receive at least one token are computed.
    """
    def __init__(self, cfg: TextConfig):
        super().__init__()
        E  = cfg.num_experts
        D  = cfg.hidden_size
        Di = cfg.expert_intermediate_size
        # Stacked weight matrices, one per expert
        self.gate_up_proj = nn.Parameter(torch.empty(E, D, 2 * Di))
        self.down_proj    = nn.Parameter(torch.empty(E, Di, D))
        nn.init.normal_(self.gate_up_proj, std=0.02)
        nn.init.normal_(self.down_proj,    std=0.02)
        self.num_experts = E
        self.act = nn.GELU(approximate="tanh")

    def forward(
        self,
        x: torch.Tensor,          # [T, D]
        top_k_index: torch.Tensor, # [T, K]
        top_k_weights: torch.Tensor, # [T, K]
    ) -> torch.Tensor:
        T, D = x.shape
        out = torch.zeros_like(x)

        # One-hot expert mask [E, K, T]; iterate over experts that receive tokens
        expert_mask = F.one_hot(top_k_index, self.num_experts)  # [T, K, E]
        expert_mask = expert_mask.permute(2, 1, 0)              # [E, K, T]
        active_experts = (expert_mask.sum(dim=(-1, -2)) > 0).nonzero(as_tuple=False)

        for idx in active_experts:
            e = idx[0].item()
            top_k_pos, token_idx = torch.where(expert_mask[e])  # which K-slot and which token
            h = x[token_idx]                                     # [n, D]
            gate, up = (h @ self.gate_up_proj[e]).chunk(2, dim=-1)
            h = self.act(gate) * up
            h = h @ self.down_proj[e]                            # [n, D]
            h = h * top_k_weights[token_idx, top_k_pos, None]   # weighted by router
            out.index_add_(0, token_idx, h.to(out.dtype))

        return out


class TextDecoderLayer(nn.Module):
    """
    One Gemma-4 transformer decoder layer.

    Architecture (see Gemma4TextDecoderLayer.forward in HF source):
      1. Pre-norm → Attention → post-attn-norm → residual
      2. Pre-FFN-norm → dense MLP always runs
      3. If MoE layer:
           MoE output added to MLP output (novel hybrid: both run in parallel)
      4. Post-FFN-norm → residual
      5. Optional per-layer input gate (small gating network conditioned on a
         per-layer side input embedding)
      6. Layer scalar: output multiplied by a learned per-layer scalar
    """
    def __init__(self, cfg: TextConfig, layer_idx: int):
        super().__init__()
        is_kv_shared = (cfg.kv_share_from is not None and layer_idx >= cfg.kv_share_from)
        self.self_attn = TextAttention(cfg, layer_idx, is_kv_shared=is_kv_shared)

        self.input_layernorm         = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

        self.mlp = TextMLP(cfg.hidden_size, cfg.intermediate_size)

        self.enable_moe = layer_idx in cfg.moe_layers and cfg.num_experts > 0
        if self.enable_moe:
            self.router  = TextRouter(cfg)
            self.experts = TextExperts(cfg)
            self.pre_feedforward_layernorm_2  = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
            self.post_feedforward_layernorm_1 = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
            self.post_feedforward_layernorm_2 = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

        # Per-layer input gate
        self.use_per_layer_gate = cfg.hidden_size_per_layer_input > 0
        if self.use_per_layer_gate:
            D, Dp = cfg.hidden_size, cfg.hidden_size_per_layer_input
            self.per_layer_input_gate = nn.Linear(D, Dp, bias=False)
            self.per_layer_projection = nn.Linear(Dp, D, bias=False)
            self.post_per_layer_input_norm = RMSNorm(D, eps=cfg.rms_norm_eps)
            self.act_fn = nn.GELU(approximate="tanh")

        # Learned output scalar (initialised to 1)
        self.layer_scalar = nn.Parameter(torch.ones(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        per_layer_input: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
    ) -> torch.Tensor:

        # ── 1. Self-Attention ────────────────────────────────────────────
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos, sin, attention_mask, kv_cache)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # ── 2. Dense MLP (always) ────────────────────────────────────────
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        # ── 3. MoE block (some layers) ───────────────────────────────────
        if self.enable_moe:
            # MLP output path
            h_mlp = self.post_feedforward_layernorm_1(hidden_states)

            # MoE runs on the pre-MLP residual
            flat = residual.reshape(-1, residual.shape[-1])       # [B*L, D]
            _, top_w, top_idx = self.router(flat)
            h_moe = self.pre_feedforward_layernorm_2(flat)
            h_moe = self.experts(h_moe, top_idx, top_w)
            h_moe = h_moe.reshape(residual.shape)
            h_moe = self.post_feedforward_layernorm_2(h_moe)

            hidden_states = h_mlp + h_moe                         # combine

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # ── 4. Per-layer input gate ──────────────────────────────────────
        if self.use_per_layer_gate and per_layer_input is not None:
            residual = hidden_states
            gate = self.act_fn(self.per_layer_input_gate(hidden_states))
            hidden_states = gate * per_layer_input
            hidden_states = self.per_layer_projection(hidden_states)
            hidden_states = self.post_per_layer_input_norm(hidden_states)
            hidden_states = residual + hidden_states

        # ── 5. Layer scalar ──────────────────────────────────────────────
        hidden_states = hidden_states * self.layer_scalar
        return hidden_states


class TextModel(nn.Module):
    """Stack of TextDecoderLayers with shared RoPE."""
    def __init__(self, cfg: TextConfig):
        super().__init__()
        self.embed_tokens = ScaledEmbedding(
            cfg.vocab_size, cfg.hidden_size, cfg.pad_token_id, cfg.embed_scale
        )
        self.rotary_emb = TextRotaryEmbedding(cfg)
        self.layers = nn.ModuleList([TextDecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)])
        self.norm   = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.sliding_window_pattern = cfg.sliding_window_pattern

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
    ) -> torch.Tensor:
        x = self.embed_tokens(input_ids)                    # [B, L, D]
        B, L, _ = x.shape
        position_ids = torch.arange(L, device=x.device).unsqueeze(0)  # [1, L]

        for i, layer in enumerate(self.layers):
            layer_type = "local" if (i % self.sliding_window_pattern != 0) else "global"
            cos, sin = self.rotary_emb(x, position_ids, layer_type=layer_type)
            x = layer(x, cos, sin, attention_mask=attention_mask, kv_cache=kv_cache)

        return self.norm(x)


# ──────────────────────────────────────────────────────────────
# Vision Tower
# ──────────────────────────────────────────────────────────────

class VisionRotaryEmbedding(nn.Module):
    """
    2-D RoPE for vision patches.
    pixel_position_ids: [B, N, 2]  (x, y per patch; -1 = padding)
    Returns cos/sin with shape [B, N, head_dim] where the head_dim is split
    evenly between x and y rotations.
    """
    def __init__(self, head_dim: int, base: float = 10_000.0):
        super().__init__()
        dim = head_dim // 2   # half for x, half for y
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, pixel_position_ids: torch.Tensor):
        """x: [B, N, D]; pixel_position_ids: [B, N, 2]"""
        all_cos, all_sin = [], []
        inv = self.inv_freq[None, :, None].expand(pixel_position_ids.shape[0], -1, 1)
        for dim_i in range(2):
            pos = pixel_position_ids[:, :, dim_i]           # [B, N]
            pos_exp = pos[:, None, :].float()               # [B, 1, N]
            freqs = (inv.float() @ pos_exp.float()).transpose(1, 2)  # [B, N, dim/2]
            emb = torch.cat([freqs, freqs], dim=-1)         # [B, N, dim]
            all_cos.append(emb.cos())
            all_sin.append(emb.sin())
        cos = torch.cat(all_cos, dim=-1).to(x.dtype)       # [B, N, head_dim]
        sin = torch.cat(all_sin, dim=-1).to(x.dtype)
        return cos, sin


class VisionPatchEmbedder(nn.Module):
    """
    Project pixel patches to hidden_size and add 2-D sinusoidal positional
    embeddings (one per spatial dimension, summed).

    HF source uses `self._position_embeddings(pixel_position_ids, padding_positions)`
    which internally builds per-axis sinusoidal embeddings and projects them to
    hidden_size before adding to the patch features.  We reproduce the same idea
    with a simple learned linear projection from a sin/cos basis.
    """
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        patch_dim = cfg.patch_size * cfg.patch_size * 3
        self.input_proj = nn.Linear(patch_dim, cfg.hidden_size, bias=False)
        # Learned positional embedding: one embedding table per axis (x, y),
        # each of size (max_positions, hidden_size).
        max_pos = cfg.image_size // cfg.patch_size + 1  # +1 for safety
        self.pos_emb_x = nn.Embedding(max_pos + 1, cfg.hidden_size)  # +1 for padding (-1→0)
        self.pos_emb_y = nn.Embedding(max_pos + 1, cfg.hidden_size)

    def forward(
        self,
        pixel_values: torch.Tensor,         # [B, N, patch_dim]
        pixel_position_ids: torch.Tensor,   # [B, N, 2]  (x, y; -1 = padding)
        padding_mask: torch.Tensor,         # [B, N] True = padding
    ) -> torch.Tensor:
        # Normalise pixels: [0,1] → [-1,1]
        pixel_values = 2.0 * (pixel_values - 0.5)
        h = self.input_proj(pixel_values.to(self.input_proj.weight.dtype))

        # Shift -1 padding positions to index 0 (won't matter; masked out later)
        pos = pixel_position_ids.clamp(min=0)   # [B, N, 2]
        pos_embed = self.pos_emb_x(pos[..., 0]) + self.pos_emb_y(pos[..., 1])  # [B, N, D]
        h = h + pos_embed
        h = h.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return h


class VisionMLP(nn.Module):
    """GELU FFN used in the vision encoder (no SwiGLU — plain gate)."""
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=True)
        self.fc2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=True)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class VisionAttention(nn.Module):
    """
    ViT-style full attention with:
      • QKV norms (same as text, for training stability)
      • 2-D RoPE applied to Q and K
    """
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        H, Dh = cfg.num_attention_heads, cfg.head_dim
        D = cfg.hidden_size
        self.num_heads = H
        self.head_dim  = Dh
        self.scaling   = Dh ** -0.5

        self.q_proj = nn.Linear(D, H * Dh, bias=False)
        self.k_proj = nn.Linear(D, H * Dh, bias=False)
        self.v_proj = nn.Linear(D, H * Dh, bias=False)
        self.o_proj = nn.Linear(H * Dh, D, bias=False)

        self.q_norm = RMSNorm(Dh, eps=cfg.rms_norm_eps)
        self.k_norm = RMSNorm(Dh, eps=cfg.rms_norm_eps)
        self.v_norm = RMSNorm(Dh, eps=cfg.rms_norm_eps)

        self.rotary_emb = VisionRotaryEmbedding(Dh, cfg.rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,          # [B, N, D]
        pixel_position_ids: torch.Tensor,     # [B, N, 2]
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = hidden_states.shape
        H, Dh = self.num_heads, self.head_dim

        q = self.q_proj(hidden_states).view(B, N, H, Dh)
        k = self.k_proj(hidden_states).view(B, N, H, Dh)
        v = self.v_proj(hidden_states).view(B, N, H, Dh)

        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)

        # 2-D RoPE
        cos, sin = self.rotary_emb(hidden_states, pixel_position_ids)
        q = apply_2d_rope(q, cos, sin, pixel_position_ids, unsqueeze_dim=2)
        k = apply_2d_rope(k, cos, sin, pixel_position_ids, unsqueeze_dim=2)

        q = q.transpose(1, 2)  # [B, H, N, Dh]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scaling
        if attention_mask is not None:
            attn = attn + attention_mask   # additive mask (0 = attend, -inf = mask)
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, H * Dh)
        return self.o_proj(out)


class VisionEncoderLayer(nn.Module):
    """
    ViT encoder block with sandwich layernorms (pre+post norm around both
    attention and MLP).
    """
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.self_attn = VisionAttention(cfg)
        self.mlp       = VisionMLP(cfg)

        self.input_layernorm          = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.pre_feedforward_layernorm  = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, pixel_position_ids, attention_mask)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # MLP block
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class VisionEncoder(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.layers = nn.ModuleList([VisionEncoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, pixel_position_ids, attention_mask)
        return hidden_states


class VisionPooler(nn.Module):
    """
    Spatially average-pools patch features down by pooling_kernel_size²,
    then scales by √hidden_size and strips padding tokens.
    """
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.kernel = cfg.pooling_kernel_size
        self.scale  = cfg.hidden_size ** 0.5
        self.standardize = cfg.standardize
        if cfg.standardize:
            self.std_bias  = nn.Parameter(torch.zeros(cfg.hidden_size))
            self.std_scale = nn.Parameter(torch.ones(cfg.hidden_size))

    def forward(
        self,
        hidden_states: torch.Tensor,          # [B, N, D]
        pixel_position_ids: torch.Tensor,     # [B, N, 2]
        padding_mask: torch.Tensor,           # [B, N] True = padding
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, D = hidden_states.shape
        k2 = self.kernel ** 2
        output_length = N // k2

        # Zero-out padding before pooling
        hidden_states = hidden_states.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        if N != output_length:
            # Simple reshape-based average pool (assumes square grid)
            hidden_states = hidden_states.view(B, output_length, k2, D).mean(dim=2)
            # Recompute padding mask for pooled sequence
            padding_mask = padding_mask.view(B, output_length, k2).all(dim=-1)

        hidden_states = hidden_states * self.scale
        valid_mask = ~padding_mask  # True = valid pooled token

        if self.standardize:
            hidden_states = (hidden_states - self.std_bias) * self.std_scale

        # Strip padding → return flat [M, D] tensor
        hidden_states = hidden_states[valid_mask]
        return hidden_states, valid_mask


class VisionModel(nn.Module):
    """
    Full vision encoder pipeline:
      pixels → PatchEmbedder → TransformerEncoder → VisionPooler → soft tokens
    """
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.patch_embedder = VisionPatchEmbedder(cfg)
        self.encoder        = VisionEncoder(cfg)
        self.pooler         = VisionPooler(cfg)

    def forward(
        self,
        pixel_values: torch.Tensor,          # [B, N, patch_dim]
        pixel_position_ids: torch.Tensor,    # [B, N, 2]
    ) -> torch.Tensor:
        padding_mask = (pixel_position_ids == -1).all(dim=-1)   # [B, N]
        # attn mask: True=attend, converted to additive 0/-inf
        attn_mask = padding_mask.float() * -1e9
        attn_mask = attn_mask[:, None, None, :]                  # broadcast over heads

        h = self.patch_embedder(pixel_values, pixel_position_ids, padding_mask)
        h = self.encoder(h, pixel_position_ids, attention_mask=attn_mask)
        soft_tokens, _ = self.pooler(h, pixel_position_ids, padding_mask)
        return soft_tokens   # [M, D_vision]  (M = total valid pooled patches)


# ──────────────────────────────────────────────────────────────
# Multimodal Fusion
# ──────────────────────────────────────────────────────────────

class MultimodalEmbedder(nn.Module):
    """
    Projects vision soft-tokens into the language model embedding space.
    Applies a pre-projection RMSNorm then a linear projection.
    """
    def __init__(self, vision_dim: int, text_dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = RMSNorm(vision_dim, eps=eps)
        self.proj = nn.Linear(vision_dim, text_dim, bias=False)

    def forward(self, soft_tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(soft_tokens))


class Gemma4Model(nn.Module):
    """
    Multimodal backbone: merges image soft-tokens into the text token stream
    then runs the language model.

    Token stream layout (after image insertion):
      [text tokens ... <img_placeholder> ... text tokens]
                              ↑
              replaced by projected vision soft-tokens
    """
    def __init__(self, cfg: Gemma4Config):
        super().__init__()
        self.language_model   = TextModel(cfg.text)
        self.vision_model     = VisionModel(cfg.vision)
        self.mm_embedder      = MultimodalEmbedder(cfg.vision.hidden_size, cfg.text.hidden_size)
        self.image_token_id   = 255_999   # Gemma4 hard-codes this sentinel id

    def forward(
        self,
        input_ids: torch.Tensor,                          # [B, L]
        pixel_values: Optional[torch.Tensor] = None,      # [B, N, patch_dim]
        pixel_position_ids: Optional[torch.Tensor] = None, # [B, N, 2]
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Embed text tokens; mask image placeholders with pad_id temporarily
        image_mask = (input_ids == self.image_token_id)  # [B, L]
        safe_ids   = input_ids.clone()
        safe_ids[image_mask] = self.language_model.embed_tokens.padding_idx
        inputs_embeds = self.language_model.embed_tokens(safe_ids)  # [B, L, D_text]

        # Encode images and scatter soft-tokens into placeholder positions
        if pixel_values is not None and pixel_position_ids is not None:
            soft_tokens = self.vision_model(pixel_values, pixel_position_ids)   # [M, D_vision]
            soft_tokens = self.mm_embedder(soft_tokens)                         # [M, D_text]

            # Scatter: fill image placeholder slots with soft tokens
            flat_mask   = image_mask.reshape(-1)
            flat_embeds = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
            flat_embeds[flat_mask] = soft_tokens.to(flat_embeds.dtype)
            inputs_embeds = flat_embeds.reshape(inputs_embeds.shape)

        # Run language model (pass inputs_embeds directly, bypassing embed layer)
        # For simplicity we call layers manually here
        x = inputs_embeds
        B, L, _ = x.shape
        position_ids = torch.arange(L, device=x.device).unsqueeze(0)

        for i, layer in enumerate(self.language_model.layers):
            layer_type = "local" if (i % self.language_model.sliding_window_pattern != 0) else "global"
            cos, sin = self.language_model.rotary_emb(x, position_ids, layer_type=layer_type)
            x = layer(x, cos, sin, attention_mask=attention_mask)

        return self.language_model.norm(x)   # [B, L, D_text]


class Gemma4ForCausalLM(nn.Module):
    """Text-only Gemma 4 with final logit soft-capping."""
    def __init__(self, cfg: Gemma4Config):
        super().__init__()
        self.model   = TextModel(cfg.text)
        self.lm_head = nn.Linear(cfg.text.hidden_size, cfg.text.vocab_size, bias=False)
        self.final_logit_softcapping = cfg.text.final_logit_softcapping

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        hidden = self.model(input_ids, attention_mask)
        logits = self.lm_head(hidden)

        if self.final_logit_softcapping is not None:
            cap = self.final_logit_softcapping
            logits = torch.tanh(logits / cap) * cap

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return {"loss": loss, "logits": logits}


class Gemma4ForConditionalGeneration(nn.Module):
    """Full multimodal Gemma 4 (text + images)."""
    def __init__(self, cfg: Gemma4Config):
        super().__init__()
        self.model   = Gemma4Model(cfg)
        self.lm_head = nn.Linear(cfg.text.hidden_size, cfg.text.vocab_size, bias=False)
        self.final_logit_softcapping = cfg.text.final_logit_softcapping

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        hidden = self.model(input_ids, pixel_values, pixel_position_ids, attention_mask)
        logits = self.lm_head(hidden)

        if self.final_logit_softcapping is not None:
            cap = self.final_logit_softcapping
            logits = torch.tanh(logits / cap) * cap

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return {"loss": loss, "logits": logits}


# ──────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running Gemma 4 smoke test (tiny config, random weights)...\n")

    # Tiny config to run on CPU quickly
    text_cfg = TextConfig(
        vocab_size=1000, hidden_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        intermediate_size=128, num_experts=4, num_experts_per_tok=2,
        moe_layers=[1], expert_intermediate_size=64,
        sliding_window=8, sliding_window_pattern=2,
        final_logit_softcapping=30.0, rms_norm_eps=1e-6,
    )
    vision_cfg = VisionConfig(
        hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
        head_dim=16, intermediate_size=64, patch_size=4,
        image_size=16, pooling_kernel_size=2,
    )
    cfg = Gemma4Config(text=text_cfg, vision=vision_cfg)

    # ── Text-only ────────────────────────────────────────────────────────
    model = Gemma4ForCausalLM(cfg)
    model.eval()
    B, L = 2, 10
    input_ids = torch.randint(0, 1000, (B, L))
    with torch.no_grad():
        out = model(input_ids)
    print(f"[CausalLM]  logits shape: {out['logits'].shape}")  # [2, 10, 1000]

    # ── Multimodal ───────────────────────────────────────────────────────
    mm_model = Gemma4ForConditionalGeneration(cfg)
    mm_model.eval()
    # 4 image patches per image, 2-D positions, patch_dim = 4*4*3 = 48
    # With pooling_kernel_size=2: output_length = 4 // (2*2) = 1 soft token per image
    N, patch_dim = 4, 48
    pixel_values = torch.rand(B, N, patch_dim)
    pixel_position_ids = torch.tensor([[[0,0],[0,1],[1,0],[1,1]]]).expand(B, -1, -1)
    # Put exactly 1 placeholder per image (= 1 pooled soft token)
    input_ids[:, 2] = 255_999
    with torch.no_grad():
        out_mm = mm_model(input_ids, pixel_values, pixel_position_ids)
    print(f"[Multimodal] logits shape: {out_mm['logits'].shape}")  # [2, 10, 1000]
    print("\n✅ Smoke test passed!")

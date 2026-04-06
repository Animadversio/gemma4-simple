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
    # Global attention layers use a larger head_dim (e.g. 512 for E4B)
    global_head_dim: int | None = None           # None → same as head_dim for all layers
    global_partial_rotary_factor: float = 1.0    # fraction of global_head_dim to rotate
    # v_norm is only used when attention_k_eq_v=True (larger Gemma 4 variants)
    use_v_norm: bool = False
    # Per-layer input gate (Gemma4-specific)
    hidden_size_per_layer_input: int = 0
    # Misc
    rms_norm_eps: float = 1e-6
    pad_token_id: int = 0
    embed_scale: float = 1.0


@dataclass
class VisionConfig:
    hidden_size: int = 768
    num_hidden_layers: int = 16
    num_attention_heads: int = 12
    head_dim: int = 64
    intermediate_size: int = 3072
    patch_size: int = 16
    position_embedding_size: int = 10240  # lookup table size per axis
    pooling_kernel_size: int = 3     # spatial pooling factor
    rope_theta: float = 100.0
    rms_norm_eps: float = 1e-6
    standardize: bool = False        # optional post-pooling standardization
    use_clipped_linears: bool = True  # clamp activations in ClippableLinear


@dataclass
class Gemma4Config:
    text: TextConfig = field(default_factory=TextConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    # Attention: 'bidirectional_vision' → vision tokens use bidirectional mask
    use_bidirectional_attention: str = "none"
    image_token_id: int = 258_880   # sentinel id for image placeholder tokens


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
    Local layers use rope_local_base_freq / head_dim.
    Global layers use rope_global_base_freq / global_head_dim, with optional partial
    rotation (partial_rotary_factor): only rope_angles = int(factor * global_head_dim // 2)
    dimensions are rotated; the rest carry zero inv_freq (→ cos=1, sin=0 = identity).
    Matches HF's _compute_proportional_rope_parameters exactly.
    """
    def __init__(self, cfg: TextConfig):
        super().__init__()
        # Local (sliding attention) — full rotation over head_dim
        local_dim = cfg.head_dim
        local_inv_freq = 1.0 / (
            cfg.rope_local_base_freq ** (torch.arange(0, local_dim, 2).float() / local_dim)
        )
        self.register_buffer("local_inv_freq", local_inv_freq, persistent=False)
        self._local_head_dim = local_dim

        # Global (full attention) — proportional/partial rotation over global_head_dim
        global_dim = cfg.global_head_dim if cfg.global_head_dim is not None else cfg.head_dim
        self._global_head_dim = global_dim
        partial = getattr(cfg, "global_partial_rotary_factor", 1.0)
        # rope_angles: number of (freq, freq) pairs that are actually rotated
        rope_angles = int(partial * global_dim // 2)
        # inv_freq for the rotated part: denominator is global_head_dim (matches HF)
        rotated_inv = 1.0 / (
            cfg.rope_global_base_freq
            ** (torch.arange(0, 2 * rope_angles, 2).float() / global_dim)
        )
        # Pad the remaining nope dimensions with zero (identity: cos=1, sin=0)
        nope_count = global_dim // 2 - rope_angles
        if nope_count > 0:
            global_inv_freq = torch.cat(
                [rotated_inv, torch.zeros(nope_count)], dim=0
            )
        else:
            global_inv_freq = rotated_inv
        self.register_buffer("global_inv_freq", global_inv_freq, persistent=False)

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
        # emb: [B, L, head_dim]  (first half and second half mirror each other)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


class ScaledEmbedding(nn.Embedding):
    """Token embedding scaled by sqrt(hidden_size) as in Gemma.
    The scale is stored as a float32 buffer and cast to the weight dtype at forward
    time, matching HF's Gemma4TextScaledWordEmbedding exactly (important for bf16).
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int,
                 embed_scale: float = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)


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
        self.is_kv_shared = is_kv_shared
        # Use _layer_types if set (authoritative), otherwise fall back to pattern
        _layer_types = getattr(cfg, "_layer_types", None)
        is_global = (
            _layer_types[layer_idx] == "full_attention"
            if _layer_types is not None
            else (layer_idx % cfg.sliding_window_pattern == 0)
        )
        self.sliding_window = None if is_global else cfg.sliding_window
        # Global attention layers may use a larger head_dim
        self.head_dim = (
            cfg.global_head_dim if (is_global and cfg.global_head_dim is not None)
            else cfg.head_dim
        )

        hs = cfg.hidden_size
        kv_dim = self.num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(hs, self.num_heads * self.head_dim, bias=False)
        # KV-shared layers still have k/v projections; sharing only applies with a KV cache
        self.k_proj = nn.Linear(hs, kv_dim, bias=False)
        self.v_proj = nn.Linear(hs, kv_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hs, bias=False)

        # Per-head norms (applied before RoPE)
        self.q_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        # v_norm is always applied (no learnable scale by default).
        # use_v_norm=True adds a learnable scale (for larger Gemma 4 variants).
        self.v_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps, with_scale=cfg.use_v_norm)

        self.attn_logit_softcapping = cfg.attn_logit_softcapping
        # QK-norm (RMSNorm on q and k) is always applied, so scaling is 1.0, matching HF.
        self.scaling = 1.0

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
        if self.is_kv_shared and kv_cache is not None and "shared_kv" in kv_cache:
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

        # Upcast to float32 for numerical stability (matches HF eager_attention_forward)
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(hidden_states.dtype)
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
        self._layer_types = getattr(cfg, "_layer_types", None)

        # Per-layer input gate embedding and projection chain (optional; E4B and larger).
        # Matches HF Gemma4TextModel exactly:
        #   embed_tokens_per_layer : Embedding(vocab, N*Dp), scaled by sqrt(Dp)
        #   per_layer_model_projection : Linear(D, N*Dp), scaled by D**-0.5
        #   per_layer_projection_norm  : RMSNorm(Dp)
        #   per_layer_input_scale      : 2**-0.5 (constant)
        Dp = cfg.hidden_size_per_layer_input
        N  = cfg.num_hidden_layers
        if Dp > 0:
            self.embed_tokens_per_layer = ScaledEmbedding(
                cfg.vocab_size, N * Dp, cfg.pad_token_id, embed_scale=Dp ** 0.5
            )
            self.per_layer_model_projection = nn.Linear(cfg.hidden_size, N * Dp, bias=False)
            self.per_layer_model_projection_scale = cfg.hidden_size ** -0.5
            self.per_layer_projection_norm = RMSNorm(Dp, eps=cfg.rms_norm_eps)
            self.per_layer_input_scale = 2.0 ** -0.5
        else:
            self.embed_tokens_per_layer = None
        self._num_layers = N
        self._per_layer_dim = Dp

    def _get_per_layer_embed(self, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
        """Token-embedding part only — [B, L, N, Dp]. No projection yet."""
        if self.embed_tokens_per_layer is None:
            return None
        return self.embed_tokens_per_layer(input_ids).reshape(
            *input_ids.shape, self._num_layers, self._per_layer_dim
        )

    def _project_per_layer(self, inputs_embeds: torch.Tensor,
                            embed_part: torch.Tensor) -> torch.Tensor:
        """Add the inputs_embeds projection to embed_part and scale. → [B, L, N, Dp]."""
        proj = self.per_layer_model_projection(inputs_embeds) * self.per_layer_model_projection_scale
        proj = proj.reshape(*inputs_embeds.shape[:-1], self._num_layers, self._per_layer_dim)
        proj = self.per_layer_projection_norm(proj)
        return (proj + embed_part) * self.per_layer_input_scale

    def _compute_per_layer_inputs(self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor) -> Optional[torch.Tensor]:
        """Compute [B, L, N, Dp] per-layer inputs (text-only path)."""
        embed = self._get_per_layer_embed(input_ids)
        if embed is None:
            return None
        return self._project_per_layer(inputs_embeds, embed)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        per_layer_inputs: Optional[torch.Tensor] = None,  # [B, L, num_layers, Dp] if pre-computed
        inputs_embeds: Optional[torch.Tensor] = None,     # pre-merged embeddings (multimodal)
    ) -> torch.Tensor:
        if inputs_embeds is None:
            x = self.embed_tokens(input_ids)                # [B, L, D]
        else:
            x = inputs_embeds                               # [B, L, D], already embedded

        B, L, _ = x.shape
        position_ids = torch.arange(L, device=x.device).unsqueeze(0)  # [1, L]

        # Compute per-layer inputs from input_ids when not provided externally
        if per_layer_inputs is None and input_ids is not None and self.embed_tokens_per_layer is not None:
            per_layer_inputs = self._compute_per_layer_inputs(input_ids, x)

        for i, layer in enumerate(self.layers):
            if self._layer_types is not None:
                layer_type = "local" if self._layer_types[i] == "sliding_attention" else "global"
            else:
                layer_type = "local" if (i % self.sliding_window_pattern != 0) else "global"
            cos, sin = self.rotary_emb(x, position_ids, layer_type=layer_type)
            pli = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
            x = layer(x, cos, sin, attention_mask=attention_mask,
                      per_layer_input=pli, kv_cache=kv_cache)

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
    Project pixel patches to hidden_size and add learned 2-D positional embeddings.

    Matches HF Gemma4VisionPatchEmbedder exactly:
      - input_proj: Linear(patch_dim, hidden_size, bias=False)
      - position_embedding_table: Parameter(2, position_embedding_size, hidden_size)
        where axis 0 = x, axis 1 = y; lookup via one-hot → matmul, then sum axes.
    """
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        patch_dim = cfg.patch_size * cfg.patch_size * 3
        self.input_proj = nn.Linear(patch_dim, cfg.hidden_size, bias=False)
        # Single parameter table covering both spatial axes, matching HF layout
        self.position_embedding_table = nn.Parameter(
            torch.ones(2, cfg.position_embedding_size, cfg.hidden_size)
        )
        self.position_embedding_size = cfg.position_embedding_size

    def _position_embeddings(
        self,
        pixel_position_ids: torch.Tensor,  # [B, N, 2]
        padding_mask: torch.Tensor,        # [B, N] True = padding
    ) -> torch.Tensor:
        pos = pixel_position_ids.clamp(min=0)  # [B, N, 2]
        # one_hot: [B, N, 2, position_embedding_size]
        one_hot = F.one_hot(pos, num_classes=self.position_embedding_size)
        # permute to [B, 2, N, pos_emb_size] for matmul with table [2, pos_emb_size, D]
        one_hot = one_hot.permute(0, 2, 1, 3).to(self.position_embedding_table)
        # [B, 2, N, D] → sum over axes → [B, N, D]
        pos_embed = (one_hot @ self.position_embedding_table).sum(dim=1)
        # Zero out padding
        pos_embed = torch.where(padding_mask.unsqueeze(-1), torch.zeros_like(pos_embed), pos_embed)
        return pos_embed

    def forward(
        self,
        pixel_values: torch.Tensor,         # [B, N, patch_dim]
        pixel_position_ids: torch.Tensor,   # [B, N, 2]  (x, y; -1 = padding)
        padding_mask: torch.Tensor,         # [B, N] True = padding
    ) -> torch.Tensor:
        pixel_values = 2.0 * (pixel_values - 0.5)  # [0,1] → [-1,1]
        h = self.input_proj(pixel_values.to(self.input_proj.weight.dtype))
        h = h + self._position_embeddings(pixel_position_ids, padding_mask)
        h = h.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return h


class ClippableLinear(nn.Module):
    """
    nn.Linear wrapped with optional input/output clamping (matches HF Gemma4ClippableLinear).
    Clip bounds are stored as buffers and loaded from checkpoint.
    When use_clipped_linears=False (default), bounds stay at ±inf (no-op clamp).
    """
    def __init__(self, in_features: int, out_features: int, use_clipped_linears: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        if use_clipped_linears:
            self.register_buffer("input_min",  torch.tensor(-float("inf")))
            self.register_buffer("input_max",  torch.tensor( float("inf")))
            self.register_buffer("output_min", torch.tensor(-float("inf")))
            self.register_buffer("output_max", torch.tensor( float("inf")))
        self.use_clipped_linears = use_clipped_linears

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_clipped_linears:
            x = torch.clamp(x, self.input_min, self.input_max)
        x = self.linear(x)
        if self.use_clipped_linears:
            x = torch.clamp(x, self.output_min, self.output_max)
        return x


class VisionMLP(nn.Module):
    """SwiGLU FFN used in the vision encoder (matches HF Gemma4VisionMLP)."""
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        clip = cfg.use_clipped_linears
        self.gate_proj = ClippableLinear(cfg.hidden_size, cfg.intermediate_size, clip)
        self.up_proj   = ClippableLinear(cfg.hidden_size, cfg.intermediate_size, clip)
        self.down_proj = ClippableLinear(cfg.intermediate_size, cfg.hidden_size, clip)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


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
        # HF Gemma4VisionAttention uses scaling=1.0, not 1/sqrt(head_dim)
        self.scaling   = 1.0

        clip = cfg.use_clipped_linears
        self.q_proj = ClippableLinear(D, H * Dh, clip)
        self.k_proj = ClippableLinear(D, H * Dh, clip)
        self.v_proj = ClippableLinear(D, H * Dh, clip)
        self.o_proj = ClippableLinear(H * Dh, D, clip)

        self.q_norm = RMSNorm(Dh, eps=cfg.rms_norm_eps, with_scale=True)
        self.k_norm = RMSNorm(Dh, eps=cfg.rms_norm_eps, with_scale=True)
        # v_norm has NO learnable scale (with_scale=False) — just bare normalisation
        self.v_norm = RMSNorm(Dh, eps=cfg.rms_norm_eps, with_scale=False)

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
        attn = F.softmax(attn.float(), dim=-1).to(q.dtype)

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
    Spatially average-pools patch features using pixel_position_ids to group
    patches into k×k kernels (matches HF Gemma4VisionPooler exactly).
    Scales by √hidden_size and strips padding tokens.
    """
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.kernel = cfg.pooling_kernel_size
        self.scale  = cfg.hidden_size ** 0.5
        self.standardize = cfg.standardize
        if cfg.standardize:
            self.std_bias  = nn.Parameter(torch.zeros(cfg.hidden_size))
            self.std_scale = nn.Parameter(torch.ones(cfg.hidden_size))

    def _avg_pool_by_positions(
        self,
        hidden_states: torch.Tensor,   # [B, N, D]
        pixel_position_ids: torch.Tensor,  # [B, N, 2]
        output_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Position-aware average pooling — groups patches by their (x//k, y//k) bin."""
        N = hidden_states.shape[1]
        k = int((N // output_length) ** 0.5)
        k2 = k * k
        if k2 * output_length != N:
            raise ValueError(f"Cannot pool {N} patches to {output_length}: {k}^2 × {output_length} ≠ {N}")

        # Clamp padding (-1) positions to 0 — padding is zeroed out already
        pos = pixel_position_ids.clamp(min=0)              # [B, N, 2]
        max_x = pos[..., 0].max(dim=-1, keepdim=True)[0] + 1  # [B, 1]
        # Kernel index: which (col_bin, row_bin) does each patch fall into?
        kx = torch.div(pos[..., 0], k, rounding_mode="floor")  # [B, N]
        ky = torch.div(pos[..., 1], k, rounding_mode="floor")  # [B, N]
        kernel_idxs = kx + (max_x // k) * ky               # [B, N] linear index

        # One-hot weighted average: each kernel bin accumulates 1/k² of each patch
        weights = F.one_hot(kernel_idxs.long(), output_length).float() / k2  # [B, N, L]
        output = weights.transpose(1, 2) @ hidden_states.float()             # [B, L, D]
        valid_mask = (weights != 0).any(dim=1)                               # [B, L]
        return output.to(hidden_states.dtype), valid_mask

    def forward(
        self,
        hidden_states: torch.Tensor,          # [B, N, D]
        pixel_position_ids: torch.Tensor,     # [B, N, 2]
        padding_mask: torch.Tensor,           # [B, N] True = padding
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, D = hidden_states.shape
        output_length = N // (self.kernel ** 2)

        # Zero-out padding before pooling
        hidden_states = hidden_states.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        if N != output_length:
            hidden_states, valid_mask = self._avg_pool_by_positions(
                hidden_states, pixel_position_ids, output_length
            )
        else:
            valid_mask = ~padding_mask

        hidden_states = hidden_states * self.scale

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
        # HF uses with_scale=False (no learnable weight, pure normalization)
        self.norm = RMSNorm(vision_dim, eps=eps, with_scale=False)
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
        self.image_token_id = cfg.image_token_id

    def forward(
        self,
        input_ids: torch.Tensor,                           # [B, L]
        pixel_values: Optional[torch.Tensor] = None,       # [B, N, patch_dim]
        pixel_position_ids: Optional[torch.Tensor] = None, # [B, N, 2]
        attention_mask: Optional[torch.Tensor] = None,
        per_layer_inputs: Optional[torch.Tensor] = None,   # [B, L, N, Dp] pre-computed
    ) -> torch.Tensor:
        # Embed text tokens; replace image placeholders with pad_id for embedding
        image_mask = (input_ids == self.image_token_id)  # [B, L]
        text_ids = input_ids.clone()
        text_ids[image_mask] = self.language_model.embed_tokens.padding_idx
        inputs_embeds = self.language_model.embed_tokens(text_ids)  # [B, L, D_text]

        # Get token-embedding part of per-layer inputs BEFORE merging images
        # (image positions use pad_id → correct token embedding lookup)
        if per_layer_inputs is None:
            pli_embed = self.language_model._get_per_layer_embed(text_ids)
        else:
            pli_embed = None  # caller provided fully-projected per_layer_inputs

        # Encode images and merge soft-tokens into placeholder positions
        if pixel_values is not None and pixel_position_ids is not None:
            soft_tokens = self.vision_model(pixel_values, pixel_position_ids)  # [M, D_vision]
            soft_tokens = self.mm_embedder(soft_tokens)                        # [M, D_text]
            img_mask_exp = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                img_mask_exp, soft_tokens.to(inputs_embeds.dtype)
            )

        # Project per-layer inputs using MERGED embeddings (image features at image positions),
        # matching HF's Gemma4Model which calls project_per_layer_inputs after merging.
        if pli_embed is not None:
            per_layer_inputs = self.language_model._project_per_layer(inputs_embeds, pli_embed)

        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            per_layer_inputs=per_layer_inputs,
        )


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

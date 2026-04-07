"""
gemma4_simple.py
================
A clean, minimal PyTorch re-implementation of the Gemma 4 forward pass.
Mirrors the HuggingFace source as closely as possible while stripping away
framework boilerplate so the architecture is easy to read and experiment with.

Reference:
  https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py

## Modules implemented (bottom-up order)

**Shared primitives**

* `RMSNorm` — root-mean-square layer normalisation
* `rotate_half` / `apply_rotary_pos_emb` — 1-D RoPE helpers
* `apply_2d_rope` — 2-D RoPE for vision patch grids

**Text tower**

* `TextRotaryEmbedding` — dual-frequency RoPE (local / global)
* `ScaledEmbedding` — token embeddings scaled by √hidden_size
* `TextMLP` — SwiGLU dense FFN
* `TextAttention` — GQA + QK-norm + sliding-window + KV-sharing
* `TextRouter` — top-k MoE router with per-expert scale
* `TextExperts` — sparse expert FFN bank
* `TextDecoderLayer` — full layer: Attention + MLP + optional MoE + per-layer gate
* `TextModel` — stack of decoder layers with shared RoPE

**Vision tower**

* `VisionRotaryEmbedding` — 2-D RoPE for patch (x, y) positions
* `VisionPatchEmbedder` — raw pixels → patch vectors
* `VisionMLP` — SwiGLU FFN with optional activation clipping
* `VisionAttention` — ViT-style full attention with 2-D RoPE and QKV norms
* `VisionEncoderLayer` — single ViT layer
* `VisionEncoder` — stack of ViT layers
* `VisionPooler` — k×k spatial average pooling → soft tokens
* `VisionModel` — patch embedder → encoder → pooler

**Multimodal fusion**

* `MultimodalEmbedder` — project vision soft-tokens → text hidden size
* `Gemma4Model` — merge image soft-tokens into token stream, run language model
* `Gemma4ForCausalLM` — text-only causal LM head
* `Gemma4ForConditionalGeneration` — full multimodal generation model
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
    # Global attention layers use a larger head_dim and may have different KV head count
    global_head_dim: int | None = None           # None → same as head_dim for all layers
    global_partial_rotary_factor: float = 1.0    # fraction of global_head_dim to rotate
    num_global_key_value_heads: int | None = None  # None → same as num_key_value_heads
    # attention_k_eq_v=True: full-attention layers share K/V projection (v_proj=None, V=k_proj(x))
    # use_v_norm applies v_norm regardless; when attention_k_eq_v=True, v_norm is always applied
    attention_k_eq_v: bool = False
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
    r"""
    ## Root Mean Square Layer Normalization

    Unlike LayerNorm, RMSNorm omits the mean-centering step and normalises only
    by the root-mean-square of the activations:

    $$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \cdot w$$

    where $w \in \mathbb{R}^d$ is a learned per-channel scale initialised to 1.

    The computation is done in **float32** for numerical stability regardless of
    the input dtype, then cast back — matching the HuggingFace implementation exactly.
    Pass `with_scale=False` to get the scale-free variant used for $v$-normalisation
    inside attention.
    """
    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if with_scale else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upcast to float32 for numerical stability, then cast back
        x_f = x.float()
        normed = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            normed = normed * self.weight.float()
        return normed.to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    r"""
    ## Rotary helper: rotate the second half into the first

    Splits the last dimension into two halves $[x_1, x_2]$ and returns
    $[-x_2, x_1]$. Combined with the cosine term this implements the
    complex-number rotation $x \cdot e^{i\theta}$.
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    r"""
    ## 1-D Rotary Position Embedding (RoPE)

    Encodes absolute position into the query/key vectors by rotating pairs of
    channels by a position-dependent angle $\theta_i \cdot m$ (where $m$ is
    the token position and $\theta_i = \text{base}^{-2i/d}$):

    $$x^\prime = x \cos\theta + \text{rotate\_half}(x) \sin\theta$$

    Because rotation is an isometry, the dot product $q \cdot k$ depends only
    on the *relative* offset $m - n$, giving the model free relative-position
    information without any learned parameters.
    """
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
    r"""
    ## 2-D Rotary Position Embedding for Vision Patches

    Vision patches have a 2-D grid position $(r, c)$. We encode both axes
    independently by splitting the head channels into two equal halves and
    applying 1-D RoPE with the row frequencies to the first half and column
    frequencies to the second:

    $$x^\prime_{\text{row}} = \text{RoPE}(x_{:d/2},\ \theta_r),\quad
      x^\prime_{\text{col}} = \text{RoPE}(x_{d/2:},\ \theta_c)$$

    `cos`/`sin` carry both sets of frequencies concatenated along the last dim.
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
        # Select inv_freq table for this layer type (local=small base, global=large base)
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        # Outer product: inv_freq [D/2] × position [L] → [B, L, D/2]
        # Each entry (b,l,i) = inv_freq[i] * position_ids[b,l]
        inv_freq_exp = inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
        pos_exp = position_ids[:, None, :].float()
        freqs = (inv_freq_exp.float() @ pos_exp.float()).transpose(1, 2)
        # Duplicate to get full head_dim: [cos(θ_0·m), …, cos(θ_{D/2}·m), cos(θ_0·m), …]
        # This layout means the second half mirrors the first, enabling rotate_half
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
    r"""
    ## SwiGLU Feed-Forward Network

    A gated variant of the FFN where one linear branch acts as a *gate*:

    $$\text{MLP}(x) = W_{\text{down}}\bigl(\text{GELU}(W_{\text{gate}}\, x) \odot W_{\text{up}}\, x\bigr)$$

    The GELU gate (tanh approximation) selectively suppresses or amplifies
    each dimension of the intermediate representation before the final
    down-projection, giving the network a multiplicative, content-dependent
    nonlinearity at essentially no parameter cost beyond the extra gate projection.

    Used as the **dense shared FFN** that runs on every layer. In MoE layers it
    runs *in parallel* with the sparse expert bank and both outputs are summed.
    """
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GELU(gate) ⊙ up  →  down
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class TextAttention(nn.Module):
    r"""
    ## Grouped-Query Attention (GQA)

    Standard scaled dot-product attention with several Gemma-4 twists:

    $$\text{Attn}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

    **QK-normalisation** — RMSNorm is applied to each query and key head *before*
    RoPE. Because the norms are learnable scales, the effective temperature is
    absorbed into the weights, so the scaling constant is $1.0$ rather than
    $1/\sqrt{d_k}$.

    **V-normalisation** — A scale-free RMSNorm is applied to values, stabilising
    training in mixed-precision without introducing extra parameters.

    **Grouped-Query Attention** — there are fewer KV heads than Q heads
    ($H_{kv} < H_q$). Each KV head is shared across $H_q / H_{kv}$ query heads
    via `expand + reshape` before the dot product.

    **Local vs global layers** — layers at positions divisible by
    `sliding_window_pattern` use *full* (global) attention; the rest apply a
    causal sliding-window mask of size `sliding_window`, limiting each token to
    attending only the nearest $W$ positions.

    **KV sharing** — the last `num_kv_shared_layers` layers reuse the key/value
    states computed by the layer just before the shared block starts, saving
    memory without a significant quality drop.

    **`attention_k_eq_v`** — in global layers of large models, $V$ is derived from
    the *same* linear projection as $K$ (i.e. `v_proj = None` and $V = k\_proj(x)$
    before `k_norm`). This halves the KV projection cost.
    """
    def __init__(self, cfg: TextConfig, layer_idx: int, is_kv_shared: bool = False):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = cfg.num_attention_heads
        self.is_kv_shared = is_kv_shared
        # Use _layer_types if set (authoritative), otherwise fall back to pattern
        _layer_types = getattr(cfg, "_layer_types", None)
        is_global = (
            _layer_types[layer_idx] == "full_attention"
            if _layer_types is not None
            else (layer_idx % cfg.sliding_window_pattern == 0)
        )
        self.sliding_window = None if is_global else cfg.sliding_window
        # Global attention layers may use a larger head_dim and different KV head count
        self.head_dim = (
            cfg.global_head_dim if (is_global and cfg.global_head_dim is not None)
            else cfg.head_dim
        )
        self.num_kv_heads = (
            cfg.num_global_key_value_heads
            if (is_global and cfg.num_global_key_value_heads is not None)
            else cfg.num_key_value_heads
        )

        hs = cfg.hidden_size
        kv_dim = self.num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(hs, self.num_heads * self.head_dim, bias=False)
        # KV-shared layers still have k/v projections; sharing only applies with a KV cache
        self.k_proj = nn.Linear(hs, kv_dim, bias=False)
        # attention_k_eq_v: full-attention layers share K projection for V (v_proj=None)
        # In that case V = v_norm(k_proj(x)) (same input projection, different norm, no RoPE)
        self.use_alternative_attention = cfg.attention_k_eq_v and is_global
        self.v_proj = (
            None if self.use_alternative_attention
            else nn.Linear(hs, kv_dim, bias=False)
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hs, bias=False)

        # Per-head norms (applied before RoPE)
        self.q_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        # v_norm: no learnable scale unless use_v_norm=True (4B+); always applied when k_eq_v
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
            k_raw = self.k_proj(hidden_states).view(B, L, Hkv, Dh)
            # attention_k_eq_v: V uses the same raw k projection (before k_norm and RoPE)
            v_raw = k_raw if self.use_alternative_attention else self.v_proj(hidden_states).view(B, L, Hkv, Dh)
            k = self.k_norm(k_raw)
            v = self.v_norm(v_raw)
            k = apply_rotary_pos_emb(k, cos, sin, unsqueeze_dim=2)
            k = k.transpose(1, 2)  # [B, Hkv, L, Dh]
            v = v.transpose(1, 2)

        # ── GQA: expand KV heads to match Q heads ───────────────────────
        # Each KV head is broadcast to H/Hkv query heads by inserting a repeat dim
        if Hkv != H:
            expand = H // Hkv
            k = k.unsqueeze(2).expand(-1, -1, expand, -1, -1).reshape(B, H, -1, Dh)
            v = v.unsqueeze(2).expand(-1, -1, expand, -1, -1).reshape(B, H, -1, Dh)

        # ── Scaled dot-product attention ────────────────────────────────
        # scaling=1.0 because QK-norms already control the variance (no 1/√d needed)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        # Optional logit soft-capping: tanh(logit/cap) * cap keeps logits in (-cap, cap)
        # This prevents softmax collapse in very long contexts
        if self.attn_logit_softcapping is not None:
            attn = attn / self.attn_logit_softcapping
            attn = torch.tanh(attn)
            attn = attn * self.attn_logit_softcapping

        # Causal / sliding-window mask: upper-triangle = -inf, recent window = 0
        if attention_mask is not None:
            attn = attn + attention_mask

        # Upcast to float32 for numerically stable softmax (matches HF eager impl)
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(hidden_states.dtype)
        # Weighted sum of values, merge heads, project back to hidden_size
        out = torch.matmul(attn, v)          # [B, H, L, Dh]
        out = out.transpose(1, 2).reshape(B, L, H * Dh)
        return self.o_proj(out)


class TextRouter(nn.Module):
    r"""
    ## MoE Token Router

    Selects the $K$ best experts for each token and computes their routing weights.

    For a token representation $x \in \mathbb{R}^D$:

    1. **Normalise** — scale-free RMSNorm stabilises the router input.
    2. **Scale** — element-wise multiply by a learned per-dimension scale
       $s \in \mathbb{R}^D$, then multiply by $D^{-1/2}$.
    3. **Logits** — linear projection to $E$ expert scores.
    4. **Softmax** → top-$K$ selection → renormalise the top-$K$ weights to sum to 1.
    5. **Per-expert scale** — multiply each weight by a learned scalar
       $\alpha_e$ (one per expert, init 1), giving the model a way to globally
       up- or down-weight specific experts during training.
    """
    def __init__(self, cfg: TextConfig):
        super().__init__()
        self.top_k = cfg.num_experts_per_tok
        # Bug #2 fix: HF uses RMSNorm with_scale=False (pure normalisation, no learnable weight)
        self.norm  = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, with_scale=False)
        # Bug #1 fix: HF has a learned per-dim scale vector (NOT a scalar sqrt(D))
        self.scale = nn.Parameter(torch.ones(cfg.hidden_size))
        self._scale_factor = cfg.hidden_size ** -0.5
        # Projection to expert logits
        self.proj  = nn.Linear(cfg.hidden_size, cfg.num_experts, bias=False)
        # Per-expert learned output scale
        self.per_expert_scale = nn.Parameter(torch.ones(cfg.num_experts))

    def forward(self, x: torch.Tensor):
        """x: [T, D]  (T = batch*seq flattened)"""
        # Step 1: normalise + scale → stable router input (zero-mean, unit-ish variance)
        # norm is scale-free RMSNorm; self.scale is learned per-dim; _scale_factor = D^{-0.5}
        h = self.norm(x) * self.scale * self._scale_factor
        # Step 2: project to E expert logits — one score per expert per token
        logits = self.proj(h)                                   # [T, E]
        # Step 3: softmax over experts → probability distribution for each token
        probs  = F.softmax(logits, dim=-1)                      # [T, E]
        # Step 4: top-K selection — choose the K highest-probability experts
        top_w, top_idx = torch.topk(probs, self.top_k, dim=-1)  # [T, K]
        # Step 5: renormalise the K weights to sum to 1 (sparse softmax)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)
        # Step 6: apply learned per-expert scalar α_e — allows global expert re-weighting
        top_w = top_w * self.per_expert_scale[top_idx]
        # Return full prob distribution (for aux loss), top-K weights, and expert indices
        return probs, top_w, top_idx


class TextExperts(nn.Module):
    r"""
    ## Sparse Expert Bank (MoE)

    Holds $E$ independent SwiGLU FFNs. For each token only the $K$ experts
    selected by the router are evaluated — the rest are skipped entirely,
    keeping compute proportional to $K/E$ of the full dense cost.

    **Weight layout** — `gate_up_proj[e]` $\in \mathbb{R}^{2D_i \times D}$ and
    `down_proj[e]` $\in \mathbb{R}^{D \times D_i}$ are stacked along the first
    dimension so that loading a single expert is a single tensor slice.

    **Dispatch loop** — we build an `[E, K, T]` one-hot mask, iterate only over
    *active* experts (those that received ≥ 1 token), gather the relevant token
    vectors, run the SwiGLU FFN, and scatter-add results back via `index_add_`.

    **Numerics note** — `index_add_` accumulates contributions in a deterministic
    order (one token at a time), matching HuggingFace's eager implementation
    bit-exactly in bfloat16. The alternative `grouped_mm` kernel accumulates in
    a different order, producing ~0.06 diff per layer that grows to ~25 max_diff
    over 30 layers (see `TEST_REPORT_26B_A4B.md`).
    """
    def __init__(self, cfg: TextConfig):
        super().__init__()
        E  = cfg.num_experts
        D  = cfg.hidden_size
        Di = cfg.expert_intermediate_size
        # Bug #3 fix: match HF weight layout [E, out, in] so checkpoint loads without permute
        self.gate_up_proj = nn.Parameter(torch.empty(E, 2 * Di, D))
        self.down_proj    = nn.Parameter(torch.empty(E, D, Di))
        nn.init.normal_(self.gate_up_proj, std=0.02)
        nn.init.normal_(self.down_proj,    std=0.02)
        self.num_experts = E
        self.act = nn.GELU(approximate="tanh")

    def forward(
        self,
        x: torch.Tensor,           # [T, D]
        top_k_index: torch.Tensor,  # [T, K]
        top_k_weights: torch.Tensor, # [T, K]
    ) -> torch.Tensor:
        T, D = x.shape
        # Accumulator: we scatter expert outputs back into this tensor
        out = torch.zeros(T, D, dtype=x.dtype, device=x.device)

        # Build a 3-way membership mask then transpose to expert-major order
        # expert_mask[e, k, t] = 1  iff token t was assigned to expert e in slot k
        expert_mask = F.one_hot(top_k_index, self.num_experts)  # [T, K, E]
        expert_mask = expert_mask.permute(2, 1, 0)              # [E, K, T]
        # Only iterate over experts that actually received tokens (skip dead experts)
        active_experts = (expert_mask.sum(dim=(1, 2)) > 0).nonzero(as_tuple=True)[0]

        for expert_idx in active_experts:
            e = expert_idx.item()
            # Find which (k-slot, token) pairs are assigned to expert e
            k_slot, tok_idx = torch.where(expert_mask[e])   # [n], [n]
            h = x[tok_idx]                                   # [n, D]  — gather tokens
            # SwiGLU FFN for this expert (gate and up projections packed into one matrix)
            gate, up = F.linear(h, self.gate_up_proj[e]).chunk(2, dim=-1)
            h = self.act(gate) * up                          # GELU gate
            h = F.linear(h, self.down_proj[e])               # [n, D]
            # Scale by routing weight (how much this expert contributes for this token)
            h = h * top_k_weights[tok_idx, k_slot, None]
            # index_add_: deterministic accumulation order → bit-exact with HF eager
            out.index_add_(0, tok_idx, h.to(out.dtype))

        return out


class TextDecoderLayer(nn.Module):
    r"""
    ## Gemma 4 Transformer Decoder Layer

    Each layer runs the following sequence of operations:

    **1. Self-Attention block**
    $$h = x + \text{PostAttnNorm}\!\left(\text{Attn}\!\left(\text{PreNorm}(x)\right)\right)$$

    **2. Dense MLP** (always active, even in MoE layers)
    $$h_{\text{mlp}} = \text{MLP}\!\left(\text{PreFFNNorm}(h)\right)$$

    **3. Sparse MoE** (only in MoE-designated layers; 26B model)

    The dense MLP and the sparse expert bank run *in parallel* on the same
    pre-FFN residual and their outputs are added:
    $$h = h + \text{PostFFNNorm}\!\left(h_{\text{mlp}} + h_{\text{moe}}\right)$$

    **4. Per-layer input gate** (E4B and larger models)

    A lightweight gating network injects a *per-layer side-channel* $p_\ell$
    (derived from the full token embeddings) at every layer:
    $$h = h + \text{Norm}\!\left(W_{\text{proj}}\!\left(\text{GELU}(W_{\text{gate}}\,h) \odot p_\ell\right)\right)$$

    This allows every layer to directly reference the original token context,
    acting as a persistent residual information highway.

    **5. Layer scalar**

    The entire layer output is multiplied by a learned scalar $\lambda$ (init 1),
    giving the optimiser a soft mechanism to reduce a layer's contribution early
    in training.
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
            # Separate post-norm path for the dense MLP output (26B model only).
            # The dense MLP and sparse MoE run *in parallel* on the same pre-FFN residual
            # and their contributions are summed before the shared post-FFN norm below.
            h_mlp = self.post_feedforward_layernorm_1(hidden_states)

            # Flatten to [B*L, D] so the router and experts see tokens, not (batch, seq)
            flat = residual.reshape(-1, residual.shape[-1])       # [B*L, D]
            # Router selects top-K experts for each token: returns routing weights + indices
            _, top_w, top_idx = self.router(flat)
            # Pre-norm before expert FFNs (separate norm for MoE path)
            h_moe = self.pre_feedforward_layernorm_2(flat)
            # Sparse expert dispatch: only active experts run; scatter back via index_add_
            h_moe = self.experts(h_moe, top_idx, top_w)
            h_moe = h_moe.reshape(residual.shape)
            # Post-norm on MoE output (separate from the dense MLP norm above)
            h_moe = self.post_feedforward_layernorm_2(h_moe)

            # Dense MLP + sparse MoE contributions are combined additively
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
    r"""
    ## Full Text Tower

    Stacks $N$ `TextDecoderLayer`s with a shared `TextRotaryEmbedding` that
    pre-computes $(cos, sin)$ tensors for the full input sequence length.

    **Per-layer input pipeline** (E4B model, `hidden_size_per_layer_input > 0`):

    Before the transformer runs, a compact *side-channel* tensor
    $P \in \mathbb{R}^{B \times L \times N \times D_p}$ is computed by blending
    two projections of the input:

    $$P = \frac{1}{\sqrt{2}}\;\text{Norm}\!\left(\frac{W_{\text{proj}}\,x}{\sqrt{D}} + E_{\text{tok}}\right)$$

    where $E_{\text{tok}}$ is a second token embedding table (scaled by $\sqrt{D_p}$).
    The $\ell$-th slice $P_{:,:,\ell,:}$ is passed as `per_layer_input` to layer $\ell$.

    When `inputs_embeds` is provided instead of `input_ids` (multimodal path), the
    per-layer projection is computed from the *merged* embeddings (vision + text),
    so image features influence the side-channel at every layer.
    """
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
        # Accept either token ids (text-only) or pre-built embeddings (multimodal path)
        if inputs_embeds is None:
            x = self.embed_tokens(input_ids)                # [B, L, D]
        else:
            x = inputs_embeds                               # [B, L, D], already embedded

        B, L, _ = x.shape
        # Simple 0..L-1 position indices — causal order; no offset needed for prefill
        position_ids = torch.arange(L, device=x.device).unsqueeze(0)  # [1, L]

        # Compute the full per-layer side-channel P ∈ [B, L, N, Dp] once up front
        # (text-only path; multimodal path pre-computes this with merged embeddings)
        if per_layer_inputs is None and input_ids is not None and self.embed_tokens_per_layer is not None:
            per_layer_inputs = self._compute_per_layer_inputs(input_ids, x)

        for i, layer in enumerate(self.layers):
            # Alternate between local (sliding-window) and global RoPE frequencies
            if self._layer_types is not None:
                layer_type = "local" if self._layer_types[i] == "sliding_attention" else "global"
            else:
                layer_type = "local" if (i % self.sliding_window_pattern != 0) else "global"
            cos, sin = self.rotary_emb(x, position_ids, layer_type=layer_type)
            # Slice the i-th layer's per-layer input vector — [B, L, Dp]
            pli = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
            x = layer(x, cos, sin, attention_mask=attention_mask,
                      per_layer_input=pli, kv_cache=kv_cache)

        # Final RMSNorm before the output projection / logit head
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
        # inv_freq shape [F] → expand to [B, F, 1] for matrix multiply with positions
        inv = self.inv_freq[None, :, None].expand(pixel_position_ids.shape[0], -1, 1)
        for dim_i in range(2):
            # Process x-axis (dim_i=0) then y-axis (dim_i=1) separately
            pos = pixel_position_ids[:, :, dim_i]           # [B, N] integer patch position
            pos_exp = pos[:, None, :].float()               # [B, 1, N] for broadcast matmul
            # Outer product: frequencies × positions → [B, F, N] → transpose to [B, N, F]
            freqs = (inv.float() @ pos_exp.float()).transpose(1, 2)  # [B, N, dim/2]
            # Duplicate freqs to fill full half-head_dim (standard RoPE cos/sin embedding)
            emb = torch.cat([freqs, freqs], dim=-1)         # [B, N, dim]
            all_cos.append(emb.cos())
            all_sin.append(emb.sin())
        # Concatenate x-axis and y-axis embeddings → [B, N, head_dim] where head_dim = 2*dim
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
        # Normalise pixel values from [0, 1] to [-1, 1] (standard ViT preprocessing)
        pixel_values = 2.0 * (pixel_values - 0.5)
        # Linear projection: each flattened patch (patch_size² × 3 channels) → hidden_size
        h = self.input_proj(pixel_values.to(self.input_proj.weight.dtype))
        # Add 2-D learned position embeddings (row + column axes combined, see _position_embeddings)
        h = h + self._position_embeddings(pixel_position_ids, padding_mask)
        # Zero out padding patch positions so they don't contaminate downstream computation
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

        # Project to Q, K, V and reshape to per-head tensors [B, N, H, Dh]
        q = self.q_proj(hidden_states).view(B, N, H, Dh)
        k = self.k_proj(hidden_states).view(B, N, H, Dh)
        v = self.v_proj(hidden_states).view(B, N, H, Dh)

        # Per-head RMSNorm: stabilises attention logit scale, scaling=1.0 (no 1/√d)
        q = self.q_norm(q)
        k = self.k_norm(k)
        # v_norm is scale-free (no learnable weight) — just normalises the magnitude
        v = self.v_norm(v)

        # Compute 2-D RoPE frequencies from (row, col) patch positions
        cos, sin = self.rotary_emb(hidden_states, pixel_position_ids)
        # Apply 2-D RoPE: first head_dim/2 channels encode row, second half encode col
        q = apply_2d_rope(q, cos, sin, pixel_position_ids, unsqueeze_dim=2)
        k = apply_2d_rope(k, cos, sin, pixel_position_ids, unsqueeze_dim=2)

        # Transpose to [B, H, N, Dh] for batched matrix multiply
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product: scaling=1.0 because QK norms handle the variance
        attn = (q @ k.transpose(-2, -1)) * self.scaling
        # Optional attention mask (e.g. to block padding patches from attending)
        if attention_mask is not None:
            attn = attn + attention_mask   # additive: 0 = attend, -inf = block
        # Upcast to float32 for numerically stable softmax
        attn = F.softmax(attn.float(), dim=-1).to(q.dtype)

        # Weighted sum of values, merge heads, project back to D
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
        # ── Attention block with sandwich layernorms ────────────────────────
        # Pre-norm: normalise before attention (stabilises gradient flow)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, pixel_position_ids, attention_mask)
        # Post-norm: second normalisation on the attention output before adding residual
        # (double-norm "sandwich" differs from standard pre-norm transformers)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # ── MLP block with sandwich layernorms ──────────────────────────────
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # Post-FFN norm: same sandwich pattern as attention block
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
        # Sequential ViT layers: each applies sandwich-norm attention + MLP with residual connections
        for layer in self.layers:
            hidden_states = layer(hidden_states, pixel_position_ids, attention_mask)
        # Final hidden states: [B, N, D] — all patch positions, including padding
        return hidden_states


class VisionPooler(nn.Module):
    r"""
    ## Vision Spatial Pooler

    Reduces the $N$ patch tokens to $N / k^2$ *soft tokens* by averaging
    each non-overlapping $k \times k$ block of patches (default $k = 3$, so
    9 patches → 1 soft token):

    $$\text{soft\_tok}_{(r',c')} = \frac{1}{k^2} \sum_{dr=0}^{k-1}\sum_{dc=0}^{k-1} h_{(kr'+dr,\, kc'+dc)}$$

    The output is scaled by $\sqrt{D_v}$ before returning, matching the HF
    implementation.  Padding tokens (indicated by `padding_positions`) are
    masked out and stripped from the output so downstream layers see only
    valid content.

    This aggressive pooling is key to keeping the token count manageable:
    a $336 \times 336$ image at patch size 16 gives $21 \times 21 = 441$ raw
    patches, which pool down to $7 \times 7 = 49$ soft tokens.
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
        # Target output length after pooling: N / k² soft tokens per image
        output_length = N // (self.kernel ** 2)

        # Zero-out padding patches before pooling so they don't bias the averages
        hidden_states = hidden_states.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        if N != output_length:
            # Spatial pooling: each k×k neighborhood of patches → one soft token
            hidden_states, valid_mask = self._avg_pool_by_positions(
                hidden_states, pixel_position_ids, output_length
            )
        else:
            # No pooling needed (kernel=1); valid mask is just the non-padding positions
            valid_mask = ~padding_mask

        # Scale pooled vectors by √D_vision (matches HF implementation)
        hidden_states = hidden_states * self.scale

        # Optional channel-wise standardization (bias + scale per feature dim)
        if self.standardize:
            hidden_states = (hidden_states - self.std_bias) * self.std_scale

        # Strip padding soft tokens → return flat [M, D] tensor where M = valid pooled patches
        # valid_mask: [B, L] boolean indicates which soft tokens are real vs padding
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
        # Mark padding patches: position_id == (-1, -1) signals a padding slot
        padding_mask = (pixel_position_ids == -1).all(dim=-1)   # [B, N]
        # Build additive attention mask: 0 for real patches, -1e9 for padding (pre-softmax block)
        attn_mask = padding_mask.float() * -1e9
        # Expand to [B, 1, 1, N] so it broadcasts over all heads and query positions
        attn_mask = attn_mask[:, None, None, :]                  # broadcast over heads

        # Stage 1: project each patch to hidden_size + add 2-D positional embeddings
        h = self.patch_embedder(pixel_values, pixel_position_ids, padding_mask)
        # Stage 2: run all ViT encoder layers with 2-D RoPE and sandwich-norm attention
        h = self.encoder(h, pixel_position_ids, attention_mask=attn_mask)
        # Stage 3: k×k spatial average pooling → M soft tokens, stripping padding positions
        soft_tokens, _ = self.pooler(h, pixel_position_ids, padding_mask)
        return soft_tokens   # [M, D_vision]  (M = total valid pooled patches across batch)


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
    r"""
    ## Multimodal Backbone

    Fuses the vision and text towers into a single forward pass:

    **Step 1 — Text embedding**
    Token ids → `ScaledEmbedding` (with image placeholders replaced by `pad_id`
    so the embedding table doesn't see the sentinel token).

    **Step 2 — Vision encoding**
    Raw patches → `VisionModel` (ViT encoder + spatial pooler) → $M$ soft tokens
    of shape $[M, D_v]$.  Then `MultimodalEmbedder` (LayerNorm + Linear) projects
    them to text dimension $D$, producing $[M, D]$.

    **Step 3 — Token stream merge**
    Image placeholder positions in the embedding sequence are overwritten by the
    projected soft tokens via `masked_scatter`:

    $$e_i = \begin{cases} \text{soft\_tok}_j & \text{if } i \text{ is an image placeholder} \\ \text{text\_emb}_i & \text{otherwise} \end{cases}$$

    **Step 4 — Per-layer side-channel**
    The per-layer projection is computed from the *merged* embeddings (not the
    original text-only ids), so vision features appear in $P_\ell$ for every layer.

    **Step 5 — Language model**
    The merged embedding sequence is passed through `TextModel` as normal.
    Vision information reaches the transformer through *two routes*:
    (a) directly as soft tokens in the input positions, and
    (b) via the per-layer side-channel $P_\ell$ injected at every decoder layer.
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
        # Locate image placeholder positions in the token sequence
        image_mask = (input_ids == self.image_token_id)  # [B, L]
        # Replace placeholders with pad_id so the embedding table sees valid indices
        text_ids = input_ids.clone()
        text_ids[image_mask] = self.language_model.embed_tokens.padding_idx
        inputs_embeds = self.language_model.embed_tokens(text_ids)  # [B, L, D_text]

        # Compute the token-embedding half of per-layer inputs BEFORE we overwrite
        # the placeholder positions — image slots should use pad embedding, not vision
        if per_layer_inputs is None:
            pli_embed = self.language_model._get_per_layer_embed(text_ids)
        else:
            pli_embed = None  # caller provided fully-projected per_layer_inputs

        # ── Vision path: encode pixels → soft tokens → inject into embedding sequence ──
        if pixel_values is not None and pixel_position_ids is not None:
            # ViT encoder + k×k spatial pooler → [M, D_vision] soft tokens
            soft_tokens = self.vision_model(pixel_values, pixel_position_ids)
            # Linear projection to text hidden dimension → [M, D_text]
            soft_tokens = self.mm_embedder(soft_tokens)
            # Overwrite image placeholder embeddings with projected vision features
            img_mask_exp = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                img_mask_exp, soft_tokens.to(inputs_embeds.dtype)
            )

        # Recompute the projection half using MERGED embeddings so that vision features
        # appear in the per-layer side-channel P_ℓ passed to every decoder layer
        if pli_embed is not None:
            per_layer_inputs = self.language_model._project_per_layer(inputs_embeds, pli_embed)

        # Run the full text tower with vision features baked into the embedding sequence
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
        # Run the full text decoder stack; returns final hidden states [B, L, D]
        hidden = self.model(input_ids, attention_mask)
        # Unembedding: project each position to vocabulary logits [B, L, V]
        logits = self.lm_head(hidden)

        # Logit soft-capping: tanh(z/cap)*cap squashes logits to (-cap, +cap),
        # preventing extreme values that destabilise softmax in long-context generation
        if self.final_logit_softcapping is not None:
            cap = self.final_logit_softcapping
            logits = torch.tanh(logits / cap) * cap

        loss = None
        if labels is not None:
            # Shift by 1: predict token i+1 from position i (causal LM objective)
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
        # Gemma4Model handles vision encoding + token stream merge + text tower in one call
        hidden = self.model(input_ids, pixel_values, pixel_position_ids, attention_mask)
        # Project hidden states to vocabulary logits [B, L, V]
        logits = self.lm_head(hidden)

        # Soft-cap logits: same as text-only model — squashes extremes to (-cap, +cap)
        if self.final_logit_softcapping is not None:
            cap = self.final_logit_softcapping
            logits = torch.tanh(logits / cap) * cap

        loss = None
        if labels is not None:
            # Causal LM cross-entropy: predict token at position i+1 from position i
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

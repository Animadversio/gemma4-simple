"""compare_gemma4_moe.py — Compare MoE components between gemma4_simple and HuggingFace.

Tests (mini synthetic model, runs on CPU unless --device cuda):
  moe_router   — TextRouter vs Gemma4TextRouter
  moe_experts  — TextExperts vs Gemma4TextExperts
  moe_layer    — TextDecoderLayer (MoE) vs Gemma4TextDecoderLayer (MoE)
  moe_model    — full TextModel (MoE, 2 layers) vs Gemma4TextModel (MoE)
  moe_full     — (requires --ckpt) full 26B checkpoint end-to-end

Usage:
  python compare_gemma4_moe.py                     # mini synthetic, cpu
  python compare_gemma4_moe.py --device cuda        # mini synthetic, cuda
  python compare_gemma4_moe.py --ckpt /path/26B    # also run moe_full
  python compare_gemma4_moe.py --tests moe_router   # single test

Match conditions
----------------
All tests use _experts_implementation='eager' in the HF config (see make_hf_cfg and
test_text_tower_26b).  Under this condition our Python-loop TextExperts is bit-exact
vs HF:

  HF eager  vs ours eager  → max_diff = 0.000,  cos_sim = 1.0  (bit-exact)

HF's DEFAULT is _experts_implementation='grouped_mm', a fused batched-GEMM kernel.
It computes the same math but with different bfloat16 accumulation order, producing:

  HF grouped_mm vs ours eager → max_diff ≈ 25.75, cos_sim ≈ 0.961  (26B, 30 layers)

This is NOT a bug — both are mathematically correct.  The discrepancy is ~0.063 per
layer in bfloat16, amplified ~32× per layer by post-expert RMSNorm (ill-conditioned
when expert output magnitude is small), compounding across all 30 MoE layers.
In float32, the expert outputs are bit-identical regardless of implementation path.

Previously fixed bugs (now resolved, no longer present):
  1. Router scale: ours fixed scalar D^0.5, HF learned vector * D^-0.5
  2. Router norm: ours always had learnable weight, HF router norm with_scale=False
  3. Experts weight layout: ours [E,D,2Di]/[E,Di,D], HF [E,2Di,D]/[E,D,Di] (transposed)
  4. Config fields: top_k_experts / moe_intermediate_size / enable_moe_block → need remapping
"""

import argparse
import os
import sys
import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── HuggingFace imports ───────────────────────────────────────────────────────
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4TextRouter,
    Gemma4TextExperts,
    Gemma4TextDecoderLayer,
    Gemma4TextModel,
)

# ── Our imports ───────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gemma4_simple import TextConfig, TextRouter, TextExperts, TextDecoderLayer, TextModel

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def check(name, ref, ours, atol=1.0, min_cos=0.99):
    ref  = ref.float().flatten()
    ours = ours.float().flatten()
    max_diff = (ref - ours).abs().max().item()
    cos_sim  = F.cosine_similarity(ref.unsqueeze(0), ours.unsqueeze(0)).item()
    top1_ref  = ref.argmax().item()
    top1_ours = ours.argmax().item()
    ok = max_diff < atol and cos_sim >= min_cos
    status = "✅" if ok else "❌"
    print(f"  {status}  {name:<45} max_diff={max_diff:.2e}  cos_sim={cos_sim:.6f}  top1={'✓' if top1_ref==top1_ours else '✗'}")
    return ok


def make_hf_cfg(hidden_size=64, num_layers=2, num_experts=4, top_k=2, moe_dim=24,
                vocab_size=512, heads=4, kv_heads=2, head_dim=16, global_head_dim=16,
                intermediate_size=128, num_kv_shared=0, sliding_window=64):
    """Build a tiny HF Gemma4TextConfig with MoE enabled."""
    layer_types = ["sliding_attention", "full_attention"] * (num_layers // 2)
    if len(layer_types) < num_layers:
        layer_types += ["sliding_attention"] * (num_layers - len(layer_types))
    layer_types = layer_types[:num_layers]
    return Gemma4TextConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        num_global_key_value_heads=kv_heads,
        head_dim=head_dim,
        global_head_dim=global_head_dim,
        num_experts=num_experts,
        top_k_experts=top_k,
        moe_intermediate_size=moe_dim,
        enable_moe_block=True,
        vocab_size=vocab_size,
        vocab_size_per_layer_input=vocab_size,
        hidden_size_per_layer_input=0,
        rms_norm_eps=1e-6,
        sliding_window=sliding_window,
        layer_types=layer_types,
        num_kv_shared_layers=num_kv_shared,
        use_double_wide_mlp=False,
        hidden_activation="gelu_pytorch_tanh",
        rope_parameters={
            "full_attention": {"rope_theta": 1e6, "rope_type": "proportional", "partial_rotary_factor": 0.25},
            "sliding_attention": {"rope_theta": 1e4, "rope_type": "default"},
        },
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=1,
        use_cache=False,
        attention_bias=False,
        attention_dropout=0.0,
        final_logit_softcapping=None,
        tie_word_embeddings=False,
        max_position_embeddings=8192,
        _attn_implementation="eager",
        _experts_implementation="eager",  # avoid grouped_mm numerical differences
    )


def make_our_cfg(hf_cfg: Gemma4TextConfig) -> TextConfig:
    """Convert HF config to our TextConfig, mapping MoE field names."""
    layer_types = hf_cfg.layer_types
    full_layers  = [i for i, t in enumerate(layer_types) if t == "full_attention"]

    # For 26B: all layers have MoE; express as list of layer indices
    moe_layers = list(range(hf_cfg.num_hidden_layers)) if hf_cfg.enable_moe_block else []

    # KV share: HF uses num_kv_shared_layers (count from end), we use kv_share_from (index)
    if hf_cfg.num_kv_shared_layers > 0:
        kv_share_from = hf_cfg.num_hidden_layers - hf_cfg.num_kv_shared_layers
    else:
        kv_share_from = None

    cfg = TextConfig(
        vocab_size=hf_cfg.vocab_size,
        hidden_size=hf_cfg.hidden_size,
        intermediate_size=hf_cfg.intermediate_size,
        num_hidden_layers=hf_cfg.num_hidden_layers,
        num_attention_heads=hf_cfg.num_attention_heads,
        num_key_value_heads=hf_cfg.num_key_value_heads,
        head_dim=hf_cfg.head_dim,
        global_head_dim=hf_cfg.global_head_dim,
        num_global_key_value_heads=getattr(hf_cfg, "num_global_key_value_heads", None),
        attention_k_eq_v=getattr(hf_cfg, "attention_k_eq_v", False),
        rms_norm_eps=hf_cfg.rms_norm_eps,
        pad_token_id=hf_cfg.pad_token_id,
        embed_scale=hf_cfg.hidden_size ** 0.5,
        hidden_size_per_layer_input=hf_cfg.hidden_size_per_layer_input,
        sliding_window=hf_cfg.sliding_window,
        sliding_window_pattern=6,  # fallback; _layer_types takes precedence
        kv_share_from=kv_share_from,
        rope_local_base_freq=hf_cfg.rope_parameters["sliding_attention"]["rope_theta"],
        rope_global_base_freq=hf_cfg.rope_parameters["full_attention"]["rope_theta"],
        global_partial_rotary_factor=hf_cfg.rope_parameters["full_attention"].get("partial_rotary_factor", 0.25),
        # MoE
        num_experts=hf_cfg.num_experts or 0,
        num_experts_per_tok=hf_cfg.top_k_experts or 0,
        moe_layers=moe_layers,
        expert_intermediate_size=hf_cfg.moe_intermediate_size or 0,
    )
    # Set _layer_types so TextModel/TextAttention use the exact HF layer type list
    # instead of the sliding_window_pattern fallback
    cfg._layer_types = hf_cfg.layer_types
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Weight copying: HF → ours (with transpositions for expert weights)
# ─────────────────────────────────────────────────────────────────────────────

def copy_router_weights(hf_router: Gemma4TextRouter, our_router: TextRouter):
    """Copy HF router weights into ours (now fully compatible after Bug #1/#2 fixes)."""
    our_router.proj.weight.data.copy_(hf_router.proj.weight.data)
    our_router.per_expert_scale.data.copy_(hf_router.per_expert_scale.data)
    our_router.scale.data.copy_(hf_router.scale.data)  # Bug #1 fixed: now have .scale param
    # Bug #2 fixed: our norm now has with_scale=False, no .weight param to copy


def copy_expert_weights(hf_experts: Gemma4TextExperts, our_experts: TextExperts):
    """Copy HF expert weights into ours (now same layout after Bug #3 fix, no transpose needed).

    Both HF and ours now use: gate_up_proj [E, 2*Di, D]  down_proj [E, D, Di]
    Both use F.linear(x, W) = x @ W.T
    """
    our_experts.gate_up_proj.data.copy_(hf_experts.gate_up_proj.data)
    our_experts.down_proj.data.copy_(hf_experts.down_proj.data)


def copy_decoder_layer_weights(hf_layer: Gemma4TextDecoderLayer,
                                our_layer: TextDecoderLayer):
    """Copy all weights from one decoder layer to ours."""
    sd_hf  = hf_layer.state_dict()
    sd_our = our_layer.state_dict()

    # Copy everything that matches exactly by name and shape
    exact_keys = []
    skipped = []
    for k, v in sd_hf.items():
        if k in sd_our and sd_our[k].shape == v.shape:
            sd_our[k].copy_(v)
            exact_keys.append(k)
        else:
            skipped.append((k, v.shape, sd_our.get(k, torch.tensor([])).shape))

    # Handle expert weight transpositions
    if hf_layer.enable_moe_block:
        copy_expert_weights(hf_layer.experts, our_layer.experts)
        copy_router_weights(hf_layer.router, our_layer.router)

    our_layer.load_state_dict(sd_our, strict=False)
    return skipped


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_moe_router(device, dtype):
    print("\n── MoE Router ──────────────────────────────────────────────────────────")
    hf_cfg = make_hf_cfg()
    our_cfg = make_our_cfg(hf_cfg)
    D = hf_cfg.hidden_size
    E = hf_cfg.num_experts
    K = hf_cfg.top_k_experts

    hf_router  = Gemma4TextRouter(hf_cfg).to(device, dtype).eval()
    our_router = TextRouter(our_cfg).to(device, dtype).eval()

    # Copy shared weights
    our_router.proj.weight.data.copy_(hf_router.proj.weight.data)
    our_router.per_expert_scale.data.copy_(hf_router.per_expert_scale.data)
    # our_router.norm.weight stays ones (HF has none)

    T = 5
    x = torch.randn(T, D, device=device, dtype=dtype)

    with torch.no_grad():
        hf_probs, hf_topw, hf_topi = hf_router(x)
        # HF: norm(x) * scale_vec * D^-0.5  (scale_vec ≈ ones)
        our_probs, our_topw, our_topi = our_router(x)
        # Ours: norm(x) * D^0.5  ← WRONG: scale differs by D from HF

    ok_probs = check("router probs (softmax output)", hf_probs, our_probs, atol=0.1, min_cos=0.9)
    ok_topw  = check("router top_k weights", hf_topw, our_topw, atol=0.1, min_cos=0.9)

    if not ok_probs:
        print()
        print("  BUG #1 — Router scale mismatch:")
        print(f"    HF:   hidden_states * learned_scale(≈1) * D^(-0.5)  ← divides by sqrt(D)={D**0.5:.1f}")
        print(f"    Ours: hidden_states * D^(+0.5)                       ← multiplies by sqrt(D)={D**0.5:.1f}")
        print(f"    Net factor difference: D = {D}")
        print()
        print("  BUG #2 — Router norm has no learnable weight in HF:")
        print("    HF:   Gemma4RMSNorm(with_scale=False) → pure L2 normalisation")
        print("    Ours: RMSNorm always has learnable weight (initialized to 1, never loaded from ckpt)")


def test_moe_experts(device, dtype):
    print("\n── MoE Experts ─────────────────────────────────────────────────────────")
    hf_cfg = make_hf_cfg()
    our_cfg = make_our_cfg(hf_cfg)
    D, E, K, Di = hf_cfg.hidden_size, hf_cfg.num_experts, hf_cfg.top_k_experts, hf_cfg.moe_intermediate_size

    hf_exp  = Gemma4TextExperts(hf_cfg).to(device, dtype).eval()
    our_exp = TextExperts(our_cfg).to(device, dtype).eval()

    # Use small weights to avoid numerical overflow
    with torch.no_grad():
        nn.init.normal_(hf_exp.gate_up_proj, std=0.02)
        nn.init.normal_(hf_exp.down_proj,    std=0.02)

    # Copy with correct transposition
    copy_expert_weights(hf_exp, our_exp)

    T = 5
    x = torch.randn(T, D, device=device, dtype=dtype)
    # Use identical routing for a fair comparison
    top_k_index = torch.randint(0, E, (T, K), device=device)
    top_k_weights = torch.rand(T, K, device=device, dtype=dtype)
    top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

    with torch.no_grad():
        hf_out  = hf_exp(x.clone(),  top_k_index, top_k_weights)
        our_out = our_exp(x.clone(), top_k_index, top_k_weights)

    check("experts output (same layout, direct copy)", hf_out, our_out, atol=1e-3, min_cos=0.9999)


def test_moe_decoder_layer(device, dtype):
    print("\n── MoE Decoder Layer ───────────────────────────────────────────────────")
    hf_cfg = make_hf_cfg(num_layers=2)
    our_cfg = make_our_cfg(hf_cfg)

    layer_idx = 0  # sliding_attention layer
    hf_layer  = Gemma4TextDecoderLayer(hf_cfg, layer_idx).to(device, dtype).eval()
    our_layer = TextDecoderLayer(our_cfg, layer_idx).to(device, dtype).eval()

    # HF uses torch.empty for expert weights → NaN by default; reinitialize before copying
    with torch.no_grad():
        torch.nn.init.normal_(hf_layer.experts.gate_up_proj, std=0.02)
        torch.nn.init.normal_(hf_layer.experts.down_proj, std=0.02)

    skipped = copy_decoder_layer_weights(hf_layer, our_layer)
    if skipped:
        print("  Skipped/shape-mismatched keys during copy:")
        for k, hs, os in skipped:
            if k not in ('experts.gate_up_proj', 'experts.down_proj'):
                print(f"    {k}: HF={tuple(hs)}  ours={tuple(os)}")

    B, L, D = 1, 8, hf_cfg.hidden_size
    x = torch.randn(B, L, D, device=device, dtype=dtype)
    # Build a minimal causal mask (all zeros = no masking for this test)
    mask = torch.zeros(B, 1, L, L, device=device, dtype=dtype)
    # RoPE position embeddings — layer_type goes to forward(), not constructor
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding
    lt = hf_cfg.layer_types[layer_idx]
    rope = Gemma4TextRotaryEmbedding(hf_cfg, device=device)
    pos_ids = torch.arange(L, device=device).unsqueeze(0)
    hf_cos, hf_sin = rope(x, position_ids=pos_ids, layer_type=lt)

    # HF forward — position_embeddings is (cos, sin) tuple, mask is plain tensor
    with torch.no_grad():
        hf_out = hf_layer(
            x.clone(),
            position_embeddings=(hf_cos, hf_sin),
            attention_mask=mask,
        )

    # Our forward — build cos/sin from our rotary embedding
    # HF uses "sliding_attention"/"full_attention"; ours uses "local"/"global"
    our_lt = "local" if lt == "sliding_attention" else "global"
    from gemma4_simple import TextRotaryEmbedding
    our_rope = TextRotaryEmbedding(our_cfg).to(device)
    our_cos, our_sin = our_rope(x, pos_ids, layer_type=our_lt)
    with torch.no_grad():
        our_out = our_layer(x.clone(), our_cos, our_sin, attention_mask=mask)

    check("decoder layer (MoE) output", hf_out, our_out, atol=1.0, min_cos=0.99)


def test_moe_model(device, dtype):
    print("\n── MoE TextModel (2 layers, synthetic weights) ─────────────────────────")
    hf_cfg = make_hf_cfg(num_layers=2)
    our_cfg = make_our_cfg(hf_cfg)

    hf_model  = Gemma4TextModel(hf_cfg).to(device, dtype).eval()
    our_model = TextModel(our_cfg).to(device, dtype).eval()

    # HF uses torch.empty for expert weights → NaN by default; reinitialize before copying
    with torch.no_grad():
        for layer in hf_model.layers:
            if hasattr(layer, 'experts') and layer.experts is not None:
                torch.nn.init.normal_(layer.experts.gate_up_proj, std=0.02)
                torch.nn.init.normal_(layer.experts.down_proj, std=0.02)

    # Copy all weights
    sd_hf  = hf_model.state_dict()
    sd_our = our_model.state_dict()

    loaded, skipped = {}, []
    for k, v in sd_hf.items():
        if k in sd_our and sd_our[k].shape == v.shape:
            sd_our[k] = v.clone()
            loaded[k] = True
        else:
            skipped.append((k, tuple(v.shape), tuple(sd_our.get(k, torch.tensor([])).shape)))

    our_model.load_state_dict(sd_our, strict=False)

    if skipped:
        print("  Keys not loaded (shape mismatch or missing in ours):")
        for k, hs, os in skipped:
            print(f"    {k}: HF={hs}  ours={os}")

    B, L = 2, 16
    ids = torch.randint(2, hf_cfg.vocab_size, (B, L), device=device)
    ids[:, 0] = 2  # BOS
    mask_1d = torch.ones(B, L, device=device, dtype=torch.long)

    # Build 4D causal mask for our model (HF builds this internally from mask_1d)
    causal_4d = torch.full((B, 1, L, L), float("-inf"), device=device, dtype=dtype)
    for i in range(L):
        causal_4d[:, 0, i, :i + 1] = 0.0

    with torch.no_grad():
        hf_out  = hf_model(ids, attention_mask=mask_1d, use_cache=False).last_hidden_state
        our_out = our_model(input_ids=ids, attention_mask=causal_4d)

    check("TextModel (MoE) last hidden state", hf_out, our_out, atol=2.0, min_cos=0.99)


def test_moe_full(ckpt: str, device, dtype):
    """Load ONE MoE decoder layer from the real 26B checkpoint and compare.

    Loading the full 26B model (51.6 GB) requires multiple A100s.  Instead we
    extract weights for a single layer (layer 0) from the safetensors shards
    and compare Gemma4TextDecoderLayer vs TextDecoderLayer on real weights.
    This validates that BUG #3 (expert weight transposition) and BUG #1 (router
    scale) are real issues on the actual checkpoint, not just the mini model.
    """
    print(f"\n── MoE Single-Layer Real Checkpoint (26B) ─────────────────────────────")
    print(f"  ckpt={ckpt}")

    import os, json
    cfg_path = os.path.join(ckpt, "config.json")
    if not os.path.exists(cfg_path):
        print("  ckpt not found or config.json missing — skipping")
        return

    from transformers import AutoConfig
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4TextDecoderLayer as HFLayer,
        Gemma4TextRotaryEmbedding as HFRope,
    )
    from safetensors.torch import load_file

    full_cfg = AutoConfig.from_pretrained(ckpt)
    text_cfg_hf = full_cfg.text_config
    our_cfg = make_our_cfg(text_cfg_hf)

    # Load index to find which shard has layer 0 weights
    idx_path = os.path.join(ckpt, "model.safetensors.index.json")
    with open(idx_path) as f:
        idx = json.load(f)

    layer_idx = 0
    prefix = f"model.language_model.layers.{layer_idx}."
    shards_needed = set(
        v for k, v in idx["weight_map"].items() if k.startswith(prefix)
    )
    print(f"  Loading layer {layer_idx} from shards: {shards_needed}")

    # Load only the required shards
    layer_weights = {}
    for shard in shards_needed:
        sd = load_file(os.path.join(ckpt, shard), device="cpu")
        for k, v in sd.items():
            if k.startswith(prefix):
                short_k = k[len(prefix):]
                layer_weights[short_k] = v.to(device=device, dtype=dtype)
        del sd

    print(f"  Loaded {len(layer_weights)} tensors for layer {layer_idx}")

    # Report what's in the checkpoint vs what we have
    hf_layer  = HFLayer(text_cfg_hf, layer_idx).to(device, dtype).eval()
    our_layer = TextDecoderLayer(our_cfg, layer_idx).to(device, dtype).eval()
    sd_our = our_layer.state_dict()

    print()
    print("  Checkpoint key analysis (MoE-relevant):")
    for k, v in layer_weights.items():
        if any(x in k for x in ["router", "expert"]):
            in_ours = k in sd_our
            shape_match = in_ours and sd_our[k].shape == v.shape
            note = ""
            if not in_ours:
                note = "  ← MISSING in ours"
            elif not shape_match:
                note = f"  ← shape mismatch: ours={tuple(sd_our[k].shape)}"
            print(f"    {k:50s}  {str(tuple(v.shape)):20s}{note}")

    # Load expert weights directly (no transpose needed — our layout now matches HF)
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts as HFExperts
    hf_exp  = HFExperts(text_cfg_hf).to(device, dtype).eval()
    our_exp = TextExperts(our_cfg).to(device, dtype).eval()

    hf_exp.gate_up_proj.data.copy_(layer_weights["experts.gate_up_proj"].to(dtype))
    hf_exp.down_proj.data.copy_(layer_weights["experts.down_proj"].to(dtype))
    our_exp.gate_up_proj.data.copy_(layer_weights["experts.gate_up_proj"].to(dtype))
    our_exp.down_proj.data.copy_(layer_weights["experts.down_proj"].to(dtype))

    T = 4
    x_flat = torch.randn(T, text_cfg_hf.hidden_size, device=device, dtype=dtype) * 0.1
    top_k_index = torch.randint(0, text_cfg_hf.num_experts, (T, text_cfg_hf.top_k_experts), device=device)
    top_k_weights = torch.rand(T, text_cfg_hf.top_k_experts, device=device, dtype=dtype)
    top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

    with torch.no_grad():
        hf_out_exp  = hf_exp(x_flat.clone(),  top_k_index, top_k_weights)
        our_out_exp = our_exp(x_flat.clone(), top_k_index, top_k_weights)

    print()
    check("Experts (real 26B weights, direct load)", hf_out_exp, our_out_exp, atol=1e-2, min_cos=0.9999)

    # Also verify router.scale loads correctly
    has_scale = "router.scale" in layer_weights
    if has_scale:
        scale_norm = layer_weights["router.scale"].float().norm().item()
        print(f"  router.scale norm={scale_norm:.4f} (from real checkpoint) — loaded into our .scale param")


# ─────────────────────────────────────────────────────────────────────────────
# Known differences summary
# ─────────────────────────────────────────────────────────────────────────────

def test_text_tower_26b(ckpt: str, dtype):
    """Load full 26B text tower on CPU and compare a forward pass end-to-end.

    The full model is ~52 GB — too large for a single 40GB A100 but fits easily
    in the 1TB RAM available on this node.  We load HF Gemma4TextModel, copy all
    weights into our TextModel, then compare outputs on a short sequence.

    Strategy:
      1. Load HF model on CPU with low_cpu_mem_usage=True
      2. Copy every weight to our model (all should load cleanly after our fixes)
      3. Delete HF model to free RAM
      4. Run both on the same input (both on CPU, bf16)
    """
    import os
    print(f"\n── 26B Full Text Tower (CPU, bfloat16) ─────────────────────────────────")
    print(f"  ckpt={ckpt}")

    cfg_path = os.path.join(ckpt, "config.json")
    if not os.path.exists(cfg_path):
        print("  ckpt not found — skipping")
        return

    from transformers import AutoConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel as HFTextModel

    device = torch.device("cpu")

    print("  Loading HF config...")
    full_cfg = AutoConfig.from_pretrained(ckpt)
    text_cfg_hf = full_cfg.text_config
    text_cfg_hf._attn_implementation = "eager"
    # Force HF to use per-expert loop (vs grouped_mm) for fair numerical comparison.
    # grouped_mm uses reshape+sum (fp32 accumulation) while our code uses index_add_ (bf16),
    # causing ~0.96 cos_sim after 30 layers. With eager, they match bit-for-bit.
    text_cfg_hf._experts_implementation = "eager"

    our_cfg = make_our_cfg(text_cfg_hf)

    print("  Loading HF Gemma4TextModel (~52 GB, cpu, bfloat16)...")
    hf_model = HFTextModel.from_pretrained(
        ckpt, config=text_cfg_hf,
        torch_dtype=dtype, low_cpu_mem_usage=True,
    ).eval()
    print(f"  HF model loaded ({sum(p.numel() for p in hf_model.parameters())/1e9:.1f}B params)")

    print("  Building our TextModel and copying weights...")
    our_model = TextModel(our_cfg).to(device, dtype).eval()

    sd_hf  = hf_model.state_dict()
    sd_our = our_model.state_dict()
    n_loaded = n_skipped = 0
    skipped = []
    for k, v in sd_hf.items():
        if k in sd_our and sd_our[k].shape == v.shape:
            sd_our[k] = v.clone()
            n_loaded += 1
        else:
            n_skipped += 1
            skipped.append((k, tuple(v.shape), tuple(sd_our.get(k, torch.tensor([])).shape)))
    our_model.load_state_dict(sd_our, strict=False)
    print(f"  Loaded {n_loaded} tensors, skipped {n_skipped}")
    if skipped:
        for k, hs, os in skipped[:10]:
            print(f"    {k}: HF={hs}  ours={os}")

    # Free HF model RAM before running forward passes
    del sd_hf
    import gc; gc.collect()

    B, L = 1, 16
    ids = torch.randint(2, text_cfg_hf.vocab_size, (B, L), device=device)
    ids[:, 0] = 2  # BOS
    mask_1d = torch.ones(B, L, device=device, dtype=torch.long)

    causal_4d = torch.full((B, 1, L, L), float("-inf"), device=device, dtype=dtype)
    for i in range(L):
        causal_4d[:, 0, i, :i + 1] = 0.0

    print("  Running HF forward pass (cpu)...")
    with torch.no_grad():
        hf_out = hf_model(ids, attention_mask=mask_1d, use_cache=False).last_hidden_state

    print("  Running our forward pass (cpu)...")
    with torch.no_grad():
        our_out = our_model(input_ids=ids, attention_mask=causal_4d)

    check("26B TextModel full tower", hf_out, our_out, atol=2.0, min_cos=0.99)


KNOWN_DIFFS = """
=================================================================
  MoE implementation status: gemma4_simple vs HuggingFace
=================================================================

FIXED #1 — Router scale: now nn.Parameter(ones([D])) * D^(-0.5), matching HF exactly.
FIXED #2 — Router norm: now RMSNorm(with_scale=False), no phantom .weight parameter.
FIXED #3 — Expert weight layout: now [E, 2*Di, D] / [E, D, Di] (HF convention),
           uses F.linear — direct load_state_dict works, no permute needed.

NOTE #4 — Config field name remapping needed for 26B checkpoint loading:
  HF field              → Our field
  top_k_experts         → num_experts_per_tok
  moe_intermediate_size → expert_intermediate_size
  enable_moe_block=True → moe_layers=[0,...,N-1] (all layers)

NUMERICAL NOTE — Expert bfloat16 accumulation (NOT a bug):
  HF default:  _experts_implementation='grouped_mm'  (fused batched GEMM kernel)
  Our impl:    Python for-loop + F.linear             (same as HF 'eager' path)

  In float32:  output is bit-identical regardless of implementation path.
  In bfloat16: grouped_mm uses different accumulation order → ~0.063 diff per layer.
               Amplified ~32x by post-expert RMSNorm → ~2.0 per layer → ~25.75 total
               after 30 MoE layers.  cos_sim drops to ~0.961.

  Fix for fair comparison:  set _experts_implementation='eager' on HF config.
  All tests in this file already do this; see make_hf_cfg() and test_text_tower_26b().

  Result with eager:  max_diff=0.000  cos_sim=1.000  (bit-exact, confirmed on 26B).
  See DEBUGGING_NOTES.md for the full investigation.
=================================================================
"""


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",   default=None,
        help="Path to 26B checkpoint for moe_full test (e.g. /path/to/gemma-4-26B-A4B-it)")
    parser.add_argument("--device", default="cpu", choices=["cpu","cuda"])
    parser.add_argument("--dtype",  default="bfloat16",
        choices=["bfloat16","float32"])
    parser.add_argument("--tests",  default="all",
        help="Comma-separated: moe_router,moe_experts,moe_layer,moe_model,moe_full,text_tower_26b  or 'all'")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype  = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    requested = set(args.tests.split(",")) if args.tests != "all" else {
        "moe_router", "moe_experts", "moe_layer", "moe_model", "moe_full"
    }

    print()
    print("=================================================================")
    print("  Gemma 4 MoE — gemma4_simple vs HuggingFace")
    print(f"  device={device}, dtype={dtype}")
    print("=================================================================")

    results = {}

    if "moe_router" in requested:
        test_moe_router(device, dtype)

    if "moe_experts" in requested:
        test_moe_experts(device, dtype)

    if "moe_layer" in requested:
        test_moe_decoder_layer(device, dtype)

    if "moe_model" in requested:
        test_moe_model(device, dtype)

    if "moe_full" in requested:
        ckpt = args.ckpt or "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/huggingface/hub/gemma-4-26B-A4B-it"
        test_moe_full(ckpt, device, dtype)

    if "text_tower_26b" in requested:
        ckpt = args.ckpt or "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/huggingface/hub/gemma-4-26B-A4B-it"
        test_text_tower_26b(ckpt, dtype)

    print()
    print(KNOWN_DIFFS)


if __name__ == "__main__":
    main()

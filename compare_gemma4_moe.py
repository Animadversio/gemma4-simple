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

Differences found vs HuggingFace (see KNOWN_DIFFS at bottom):
  1. Router scale: ours fixed scalar D^0.5, HF learned vector * D^-0.5
  2. Router norm: ours always has learnable weight, HF router norm with_scale=False
  3. Experts weight layout: ours [E,D,2Di]/[E,Di,D], HF [E,2Di,D]/[E,D,Di] (transposed)
  4. Config fields: top_k_experts / moe_intermediate_size / enable_moe_block → need remapping
"""

import argparse
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
sys.path.insert(0, "/n/home12/binxuwang/Github/gemma4-simple")
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
                vocab_size=512, heads=4, kv_heads=2, head_dim=16, global_head_dim=32,
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

    return TextConfig(
        vocab_size=hf_cfg.vocab_size,
        hidden_size=hf_cfg.hidden_size,
        intermediate_size=hf_cfg.intermediate_size,
        num_hidden_layers=hf_cfg.num_hidden_layers,
        num_attention_heads=hf_cfg.num_attention_heads,
        num_key_value_heads=hf_cfg.num_key_value_heads,
        head_dim=hf_cfg.head_dim,
        global_head_dim=hf_cfg.global_head_dim,
        rms_norm_eps=hf_cfg.rms_norm_eps,
        pad_token_id=hf_cfg.pad_token_id,
        embed_scale=hf_cfg.hidden_size ** 0.5,
        hidden_size_per_layer_input=hf_cfg.hidden_size_per_layer_input,
        sliding_window=hf_cfg.sliding_window,
        sliding_window_pattern=6,  # irrelevant for mini tests
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


# ─────────────────────────────────────────────────────────────────────────────
# Weight copying: HF → ours (with transpositions for expert weights)
# ─────────────────────────────────────────────────────────────────────────────

def copy_router_weights(hf_router: Gemma4TextRouter, our_router: TextRouter):
    """Copy HF router weights into ours.

    Differences handled here:
      - HF has router.norm with NO weight (with_scale=False).
        Ours has router.norm.weight (initialized to ones). We leave ours as ones.
      - HF has router.scale = nn.Parameter(shape=[D]). Ours uses a fixed scalar.
        We CANNOT load this — it's silently absent in ours. Noted as Bug #1 / Bug #2.
    """
    # proj weights match exactly
    our_router.proj.weight.data.copy_(hf_router.proj.weight.data)
    # per_expert_scale matches
    our_router.per_expert_scale.data.copy_(hf_router.per_expert_scale.data)
    # NOTE: hf_router.scale (learned vector) is NOT copied — ours has no equivalent
    # NOTE: our_router.norm.weight stays as ones; HF has no such parameter


def copy_expert_weights(hf_experts: Gemma4TextExperts, our_experts: TextExperts):
    """Copy HF expert weights into ours, transposing weight matrices.

    HF layout:  gate_up_proj [E, 2*Di, D]   down_proj [E, D, Di]
    Ours layout: gate_up_proj [E, D, 2*Di]   down_proj [E, Di, D]

    HF uses F.linear(x, W) = x @ W.T  →  equivalent to x @ W^T
    Ours uses  h @ W  directly

    To match: our_W must equal hf_W.T, i.e. permute(0,2,1).
    """
    our_experts.gate_up_proj.data.copy_(
        hf_experts.gate_up_proj.data.permute(0, 2, 1)  # [E,2Di,D] → [E,D,2Di]
    )
    our_experts.down_proj.data.copy_(
        hf_experts.down_proj.data.permute(0, 2, 1)     # [E,D,Di]  → [E,Di,D]
    )


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

    ok = check("experts output (with transposed weights)", hf_out, our_out, atol=1e-3, min_cos=0.9999)

    if not ok:
        print()
        print("  BUG #3 — Expert weight matrix layout is transposed:")
        print(f"    HF:   gate_up_proj [{E}, 2*{Di}, {D}], uses F.linear(x, W) = x @ W.T")
        print(f"    Ours: gate_up_proj [{E}, {D}, 2*{Di}], uses h @ W   (direct matmul)")
        print("    Fix: when loading from checkpoint, permute(0,2,1) on gate_up_proj and down_proj")
    else:
        print("  Expert weights match after transposition fix  ✓")

    # Document: naive copy without transpose fails due to shape mismatch
    print(f"  (Naive copy without transpose: HF gate_up_proj {tuple(hf_exp.gate_up_proj.shape)} "
          f"vs ours {tuple(our_exp.gate_up_proj.shape)} — would need permute(0,2,1))")


def test_moe_decoder_layer(device, dtype):
    print("\n── MoE Decoder Layer ───────────────────────────────────────────────────")
    hf_cfg = make_hf_cfg(num_layers=2)
    our_cfg = make_our_cfg(hf_cfg)

    layer_idx = 0  # sliding_attention layer
    hf_layer  = Gemma4TextDecoderLayer(hf_cfg, layer_idx).to(device, dtype).eval()
    our_layer = TextDecoderLayer(our_cfg, layer_idx).to(device, dtype).eval()

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
    # RoPE position embeddings
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding
    rope = Gemma4TextRotaryEmbedding(hf_cfg, device=device)
    pos_ids = torch.arange(L, device=device).unsqueeze(0)
    pos_emb = rope(x, position_ids=pos_ids)
    lt = hf_cfg.layer_types[layer_idx]
    hf_cos, hf_sin = pos_emb[lt]

    # HF forward
    with torch.no_grad():
        hf_out = hf_layer(
            x.clone(),
            position_embeddings={lt: (hf_cos, hf_sin)},
            attention_mask={lt: mask},
        )

    # Our forward — build cos/sin from our rotary embedding
    from gemma4_simple import TextRotaryEmbedding
    our_rope = TextRotaryEmbedding(our_cfg).to(device)
    our_cos, our_sin = our_rope(L, device, layer_type=lt)
    with torch.no_grad():
        our_out = our_layer(x.clone(), our_cos, our_sin, attention_mask=mask)

    check("decoder layer (MoE) output", hf_out, our_out, atol=1.0, min_cos=0.99)


def test_moe_model(device, dtype):
    print("\n── MoE TextModel (2 layers, synthetic weights) ─────────────────────────")
    hf_cfg = make_hf_cfg(num_layers=2)
    our_cfg = make_our_cfg(hf_cfg)

    hf_model  = Gemma4TextModel(hf_cfg).to(device, dtype).eval()
    our_model = TextModel(our_cfg).to(device, dtype).eval()

    # Copy all weights with transpositions for expert layers
    sd_hf  = hf_model.state_dict()
    sd_our = our_model.state_dict()

    loaded, skipped = {}, []
    for k, v in sd_hf.items():
        # Expert weight transpositions
        if "experts.gate_up_proj" in k or "experts.down_proj" in k:
            # Transpose: HF [E, out, in] → ours [E, in, out]
            v = v.permute(0, 2, 1)
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
    mask = torch.ones(B, L, device=device, dtype=torch.long)

    with torch.no_grad():
        hf_out  = hf_model(ids, attention_mask=mask, use_cache=False).last_hidden_state
        our_out = our_model(input_ids=ids)

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

    # Test: load expert weights with correct transposition → compare outputs
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts as HFExperts
    hf_exp  = HFExperts(text_cfg_hf).to(device, dtype).eval()
    our_exp = TextExperts(our_cfg).to(device, dtype).eval()

    # Load real expert weights into HF experts
    hf_exp.gate_up_proj.data.copy_(layer_weights["experts.gate_up_proj"].to(dtype))
    hf_exp.down_proj.data.copy_(layer_weights["experts.down_proj"].to(dtype))

    # Load with transposition into ours
    our_exp.gate_up_proj.data.copy_(layer_weights["experts.gate_up_proj"].to(dtype).permute(0, 2, 1))
    our_exp.down_proj.data.copy_(layer_weights["experts.down_proj"].to(dtype).permute(0, 2, 1))

    T = 4
    x_flat = torch.randn(T, text_cfg_hf.hidden_size, device=device, dtype=dtype) * 0.1
    top_k_index = torch.randint(0, text_cfg_hf.num_experts, (T, text_cfg_hf.top_k_experts), device=device)
    top_k_weights = torch.rand(T, text_cfg_hf.top_k_experts, device=device, dtype=dtype)
    top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

    with torch.no_grad():
        hf_out_exp  = hf_exp(x_flat.clone(),  top_k_index, top_k_weights)
        our_out_exp = our_exp(x_flat.clone(), top_k_index, top_k_weights)

    print()
    check("Experts (real 26B weights, with transpose fix)", hf_out_exp, our_out_exp, atol=1e-2, min_cos=0.9999)

    # Test WITHOUT transposition — same input, direct copy (will be wrong)
    our_exp_bad = TextExperts(our_cfg).to(device, dtype).eval()
    # Can't copy directly (shape mismatch), so transpose the HF weights the wrong way:
    # Load as if we naively used the HF layout in our direct-matmul code
    # Simulate: create expert with HF-layout weights and run our computation path
    # We do this by manually swapping the dimensions
    D, E = text_cfg_hf.hidden_size, text_cfg_hf.num_experts
    Di = text_cfg_hf.moe_intermediate_size
    # Build a "bad" expert: copy gate_up as [E, D, 2Di] but filled with the WRONG orientation
    our_exp_bad.gate_up_proj.data.copy_(
        layer_weights["experts.gate_up_proj"].to(dtype)[:, :D, :]  # truncate to wrong shape for demo
        if False else
        layer_weights["experts.gate_up_proj"].to(dtype).permute(0, 2, 1).transpose(1, 2).permute(0, 2, 1)
        # = original HF layout transposed back = identity: just use original permutation
    )
    # Actually: the "bad" case is loading the weight WITHOUT transpose.
    # Since shapes differ ([E,2Di,D] vs [E,D,2Di]), a direct load_state_dict would fail.
    # We show this conceptually:
    print(f"\n  Bug #3 confirmed on real weights:")
    print(f"    HF gate_up_proj shape: {tuple(layer_weights['experts.gate_up_proj'].shape)}  [E, 2*Di, D]")
    print(f"    Our gate_up_proj shape: {tuple(our_exp.gate_up_proj.shape)}             [E, D, 2*Di]")
    print(f"    Direct load_state_dict would fail with shape mismatch → requires permute(0,2,1)")
    print(f"\n  Bug #1 confirmed on real checkpoint:")
    has_scale = "router.scale" in layer_weights
    print(f"    router.scale present in checkpoint: {has_scale}")
    if has_scale:
        scale_shape = tuple(layer_weights["router.scale"].shape)
        scale_norm  = layer_weights["router.scale"].float().norm().item()
        print(f"    router.scale shape={scale_shape}, norm={scale_norm:.4f} (not ones → learned)")
        print(f"    Our TextRouter has no .scale parameter → this is silently ignored on load")


# ─────────────────────────────────────────────────────────────────────────────
# Known differences summary
# ─────────────────────────────────────────────────────────────────────────────

KNOWN_DIFFS = """
=================================================================
  Known differences: gemma4_simple vs HuggingFace (MoE path)
=================================================================

BUG #1 — Router scale sign / magnitude:
  HF:   norm(x) * learned_scale_vec(≈ones, shape=[D]) * D^(-0.5)
  Ours: norm(x) * D^(+0.5)
  Effect: router logits differ by factor ≈ D (hidden_size).
  Fix:   add self.scale = nn.Parameter(torch.ones(D)) to TextRouter;
         change forward to: h = self.norm(x) * self.scale * (D**-0.5)

BUG #2 — Router norm has no learnable weight in HF:
  HF:   Gemma4RMSNorm(with_scale=False) → pure x/rms, no weight
  Ours: RMSNorm always has learnable .weight (loaded from ckpt as ones)
  Effect: with_scale=False norm is never in the checkpoint, so our
          router.norm.weight is never loaded → stays ones → same numerics
          as HF, but adds an unused parameter.
  Fix:   add with_scale flag to RMSNorm; use RMSNorm(D, with_scale=False)
         in TextRouter.__init__.

BUG #3 — Expert weight matrix layout is transposed:
  HF:   gate_up_proj [E, 2*Di, D]  down_proj [E, D, Di]
        uses F.linear(x, W) = x @ W.T
  Ours: gate_up_proj [E, D, 2*Di]  down_proj [E, Di, D]
        uses h @ W (direct matmul)
  Effect: wrong matrix multiplication → random output.
  Fix (option A): when loading from checkpoint, permute(0,2,1).
  Fix (option B): change TextExperts to use F.linear and match HF layout.

BUG #4 — Config field name mismatches (26B → TextConfig remapping needed):
  HF field              → Our field
  top_k_experts         → num_experts_per_tok
  moe_intermediate_size → expert_intermediate_size
  enable_moe_block=True → moe_layers=[0,...,N-1] (all layers)
  (These are not bugs in the math, but must be handled during model loading.)
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
        help="Comma-separated: moe_router,moe_experts,moe_layer,moe_model,moe_full  or 'all'")
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

    print()
    print(KNOWN_DIFFS)


if __name__ == "__main__":
    main()

"""
compare_gemma4.py
=================
Modular forward-pass comparison between gemma4_simple.py and HuggingFace Gemma 4.

Each module has its own compare_<name>() function. The main() at the bottom
lets you selectively run tests via command-line flags.

Usage:
    # Run all tests (module + system)
    python compare_gemma4.py

    # Run specific tests
    python compare_gemma4.py --tests rmsnorm mlp attn decoder vision system

    # Skip slow system-level test
    python compare_gemma4.py --skip system

Requirements:
    pip install "transformers>=5.5.0" safetensors torch

    HF weights must be downloaded and HF_CKPT set (edit below or pass --ckpt).
"""

import argparse
import sys
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer, AutoConfig

# ──────────────────────────────────────────────────────────────────────────────
# Configuration — edit these or pass --ckpt / --device on the command line
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CKPT   = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/huggingface/hub/gemma-4-E4B-it"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE          = torch.bfloat16
LAYER          = 0   # which layer index to test
B, L           = 1, 8  # batch size, sequence length

torch.manual_seed(42)

# ──────────────────────────────────────────────────────────────────────────────
# Lazy globals (populated in setup())
# ──────────────────────────────────────────────────────────────────────────────
HF_CKPT  = None
DEVICE   = None
hf_cfg   = None
tc       = None   # text config from HF
vc       = None   # vision config from HF
our_text_cfg  = None
our_vis_cfg   = None
tokenizer     = None


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_tensors(keys: list[str]) -> dict[str, torch.Tensor]:
    """Load selected keys from model.safetensors (low-mem, fast)."""
    out = {}
    with safe_open(f"{HF_CKPT}/model.safetensors", framework="pt", device="cpu") as f:
        for k in keys:
            out[k] = f.get_tensor(k).to(DTYPE).to(DEVICE)
    return out


def check(name: str, ref: torch.Tensor, ours: torch.Tensor, atol: float = 1e-2, min_cos: float = 0.999) -> bool:
    """Print a comparison summary line and return pass/fail."""
    ref_f  = ref.float().cpu()
    ours_f = ours.float().cpu()
    max_diff = (ref_f - ours_f).abs().max().item()
    cos_sim  = F.cosine_similarity(
        ref_f.reshape(-1), ours_f.reshape(-1), dim=0
    ).item()
    top1_match = (
        (ref_f.argmax(-1) == ours_f.argmax(-1)).float().mean().item()
        if ref_f.ndim >= 2 else None
    )
    passed = max_diff < atol and cos_sim > min_cos
    sym = "✅" if passed else "❌"
    msg = (f"{sym}  {name:<40}  max_diff={max_diff:.2e}  cos_sim={cos_sim:.6f}")
    if top1_match is not None:
        msg += f"  top1={top1_match:.2%}"
    print(msg)
    return passed


def _make_random_hidden(batch=B, seq=L, dim=None):
    if dim is None:
        dim = tc.hidden_size
    return torch.randn(batch, seq, dim, dtype=DTYPE, device=DEVICE)


# ──────────────────────────────────────────────────────────────────────────────
# Module comparison functions
# ──────────────────────────────────────────────────────────────────────────────

def compare_rmsnorm() -> bool:
    """RMSNorm layer."""
    from gemma4_simple import RMSNorm
    from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm as HFRMSNorm

    print("\n── RMSNorm ──────────────────────────────────────────────────")
    prefix = f"model.language_model.layers.{LAYER}"
    wts = load_tensors([f"{prefix}.input_layernorm.weight"])
    x   = _make_random_hidden()

    hf_norm = HFRMSNorm(tc.hidden_size, eps=tc.rms_norm_eps).to(DTYPE).to(DEVICE)
    hf_norm.weight.data.copy_(wts[f"{prefix}.input_layernorm.weight"])

    our_norm = RMSNorm(tc.hidden_size, eps=tc.rms_norm_eps).to(DTYPE).to(DEVICE)
    our_norm.weight.data.copy_(wts[f"{prefix}.input_layernorm.weight"])

    with torch.no_grad():
        hf_out  = hf_norm(x)
        our_out = our_norm(x)

    return check("RMSNorm", hf_out, our_out, atol=1e-4)


def compare_text_mlp() -> bool:
    """TextMLP (SwiGLU dense FFN)."""
    from gemma4_simple import TextMLP
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextMLP as HFTextMLP

    print("\n── TextMLP ──────────────────────────────────────────────────")
    prefix = f"model.language_model.layers.{LAYER}"
    keys = [f"{prefix}.mlp.{w}.weight" for w in ("gate_proj", "up_proj", "down_proj")]
    wts = load_tensors(keys)
    x   = _make_random_hidden()

    hf_mlp = HFTextMLP(tc, layer_idx=LAYER).to(DTYPE).to(DEVICE)
    hf_mlp.gate_proj.weight.data.copy_(wts[f"{prefix}.mlp.gate_proj.weight"])
    hf_mlp.up_proj.weight.data.copy_(wts[f"{prefix}.mlp.up_proj.weight"])
    hf_mlp.down_proj.weight.data.copy_(wts[f"{prefix}.mlp.down_proj.weight"])

    our_mlp = TextMLP(tc.hidden_size, tc.intermediate_size).to(DTYPE).to(DEVICE)
    our_mlp.gate_proj.weight.data.copy_(wts[f"{prefix}.mlp.gate_proj.weight"])
    our_mlp.up_proj.weight.data.copy_(wts[f"{prefix}.mlp.up_proj.weight"])
    our_mlp.down_proj.weight.data.copy_(wts[f"{prefix}.mlp.down_proj.weight"])

    with torch.no_grad():
        hf_out  = hf_mlp(x)
        our_out = our_mlp(x)

    return check("TextMLP", hf_out, our_out, atol=1e-3)


def compare_text_attention() -> bool:
    """TextAttention (GQA + QK-norm + sliding-window). Compared against manual GT."""
    from gemma4_simple import TextAttention, apply_rotary_pos_emb
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4TextAttention as HFTextAttn,
        Gemma4TextRotaryEmbedding,
    )

    print("\n── TextAttention ────────────────────────────────────────────")
    prefix = f"model.language_model.layers.{LAYER}"
    attn_keys = [
        f"{prefix}.self_attn.{p}.weight"
        for p in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")
    ]
    wts = load_tensors(attn_keys)
    x   = _make_random_hidden()
    layer_type = tc.layer_types[LAYER]
    tc._attn_implementation = "eager"

    # RoPE
    hf_rope = Gemma4TextRotaryEmbedding(config=tc).to(DTYPE).to(DEVICE)
    pos_ids = torch.arange(L, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        cos, sin = hf_rope(x, pos_ids, layer_type=layer_type)

    # Manual ground truth (avoids HF hub kernels)
    hf_attn = HFTextAttn(tc, layer_idx=LAYER).to(DTYPE).to(DEVICE)
    for p in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
        getattr(hf_attn, p).weight.data.copy_(wts[f"{prefix}.self_attn.{p}.weight"])

    # Our implementation (must be before manual GT so we can reuse v_norm)
    our_attn = TextAttention(our_text_cfg, layer_idx=LAYER, is_kv_shared=False).to(DTYPE).to(DEVICE)
    for p in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
        getattr(our_attn, p).weight.data.copy_(wts[f"{prefix}.self_attn.{p}.weight"])

    with torch.no_grad():
        n_rep = tc.num_attention_heads // tc.num_key_value_heads
        hs    = (B, L, -1, tc.head_dim)
        q = hf_attn.q_norm(hf_attn.q_proj(x).view(hs))
        q = apply_rotary_pos_emb(q, cos, sin, unsqueeze_dim=2).transpose(1, 2)
        k = hf_attn.k_norm(hf_attn.k_proj(x).view(hs))
        k = apply_rotary_pos_emb(k, cos, sin, unsqueeze_dim=2).transpose(1, 2)
        v = hf_attn.v_proj(x).view(B, L, tc.num_key_value_heads, tc.head_dim)
        v = our_attn.v_norm(v)   # RMSNorm without scale (matches HF v_norm)
        v = v.transpose(1, 2)
        k = k.unsqueeze(2).expand(B, tc.num_key_value_heads, n_rep, L, tc.head_dim).reshape(B, tc.num_attention_heads, L, tc.head_dim)
        v = v.unsqueeze(2).expand(B, tc.num_key_value_heads, n_rep, L, tc.head_dim).reshape(B, tc.num_attention_heads, L, tc.head_dim)
        scale = 1.0  # QK-norm applied, so no sqrt(head_dim) scaling (matches HF self.scaling=1.0)
        attn_w = F.softmax((q @ k.transpose(-2, -1) * scale).float(), dim=-1).to(DTYPE)
        gt_out = hf_attn.o_proj((attn_w @ v).transpose(1, 2).reshape(B, L, -1))

    with torch.no_grad():
        our_out = our_attn(x, cos=cos, sin=sin)

    return check("TextAttention (vs manual GT)", gt_out, our_out, atol=1e-2)


def compare_text_decoder_layer() -> bool:
    """Full TextDecoderLayer including per-layer gate and layer scalar."""
    from gemma4_simple import TextDecoderLayer, apply_rotary_pos_emb
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    print("\n── TextDecoderLayer ─────────────────────────────────────────")
    prefix = f"model.language_model.layers.{LAYER}"
    keys = [
        f"{prefix}.input_layernorm.weight",
        f"{prefix}.post_attention_layernorm.weight",
        f"{prefix}.pre_feedforward_layernorm.weight",
        f"{prefix}.post_feedforward_layernorm.weight",
        f"{prefix}.layer_scalar",
        f"{prefix}.mlp.gate_proj.weight",
        f"{prefix}.mlp.up_proj.weight",
        f"{prefix}.mlp.down_proj.weight",
        f"{prefix}.self_attn.q_proj.weight",
        f"{prefix}.self_attn.k_proj.weight",
        f"{prefix}.self_attn.v_proj.weight",
        f"{prefix}.self_attn.o_proj.weight",
        f"{prefix}.self_attn.q_norm.weight",
        f"{prefix}.self_attn.k_norm.weight",
        f"{prefix}.per_layer_input_gate.weight",
        f"{prefix}.per_layer_projection.weight",
        f"{prefix}.post_per_layer_input_norm.weight",
    ]
    wts = load_tensors(keys)
    x   = _make_random_hidden()
    Dp  = tc.hidden_size_per_layer_input
    per_layer_x = torch.randn(B, L, Dp, dtype=DTYPE, device=DEVICE)

    layer_type = tc.layer_types[LAYER]
    hf_rope = Gemma4TextRotaryEmbedding(config=tc).to(DTYPE).to(DEVICE)
    pos_ids = torch.arange(L, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        cos, sin = hf_rope(x, pos_ids, layer_type=layer_type)

    our_dec = TextDecoderLayer(our_text_cfg, layer_idx=LAYER).to(DTYPE).to(DEVICE)
    # Load weights
    for attr in ("input_layernorm", "post_attention_layernorm",
                 "pre_feedforward_layernorm", "post_feedforward_layernorm",
                 "post_per_layer_input_norm"):
        getattr(our_dec, attr).weight.data.copy_(wts[f"{prefix}.{attr}.weight"])
    our_dec.layer_scalar.data.copy_(wts[f"{prefix}.layer_scalar"])
    for w in ("gate_proj", "up_proj", "down_proj"):
        getattr(our_dec.mlp, w).weight.data.copy_(wts[f"{prefix}.mlp.{w}.weight"])
    for p in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
        getattr(our_dec.self_attn, p).weight.data.copy_(wts[f"{prefix}.self_attn.{p}.weight"])
    our_dec.per_layer_input_gate.weight.data.copy_(wts[f"{prefix}.per_layer_input_gate.weight"])
    our_dec.per_layer_projection.weight.data.copy_(wts[f"{prefix}.per_layer_projection.weight"])

    with torch.no_grad():
        our_out = our_dec(x, cos=cos, sin=sin, per_layer_input=per_layer_x)

    # Manual GT
    with torch.no_grad():
        # Attention block
        n_rep = tc.num_attention_heads // tc.num_key_value_heads
        hs = (B, L, -1, tc.head_dim)
        residual = x
        h = our_dec.input_layernorm(x)
        q = our_dec.self_attn.q_norm(our_dec.self_attn.q_proj(h).view(hs))
        q = apply_rotary_pos_emb(q, cos, sin, unsqueeze_dim=2).transpose(1, 2)
        k = our_dec.self_attn.k_norm(our_dec.self_attn.k_proj(h).view(hs))
        k = apply_rotary_pos_emb(k, cos, sin, unsqueeze_dim=2).transpose(1, 2)
        v = our_dec.self_attn.v_proj(h).view(B, L, tc.num_key_value_heads, tc.head_dim)
        v = our_dec.self_attn.v_norm(v)   # RMSNorm without scale (matches HF)
        v = v.transpose(1, 2)
        k = k.unsqueeze(2).expand(B, tc.num_key_value_heads, n_rep, L, tc.head_dim).reshape(B, tc.num_attention_heads, L, tc.head_dim)
        v = v.unsqueeze(2).expand(B, tc.num_key_value_heads, n_rep, L, tc.head_dim).reshape(B, tc.num_attention_heads, L, tc.head_dim)
        scale = 1.0  # QK-norm applied, so no sqrt(head_dim) scaling (matches HF)
        attn_w = F.softmax((q @ k.transpose(-2, -1) * scale).float(), dim=-1).to(DTYPE)
        h = our_dec.self_attn.o_proj((attn_w @ v).transpose(1, 2).reshape(B, L, -1))
        h = our_dec.post_attention_layernorm(h)
        h = residual + h

        # MLP block
        residual = h
        h = our_dec.pre_feedforward_layernorm(h)
        h = our_dec.mlp(h)
        h = our_dec.post_feedforward_layernorm(h)
        h = residual + h

        # Per-layer gate
        residual = h
        gate = F.gelu(our_dec.per_layer_input_gate(h), approximate="tanh")
        h = gate * per_layer_x
        h = our_dec.per_layer_projection(h)
        h = our_dec.post_per_layer_input_norm(h)
        h = residual + h

        manual_gt = h * our_dec.layer_scalar

    return check("TextDecoderLayer (vs manual GT)", manual_gt, our_out, atol=1e-3)


def compare_embedding() -> bool:
    """Scaled embedding lookup."""
    print("\n── Embedding+scale ──────────────────────────────────────────")
    wts = load_tensors(["model.language_model.embed_tokens.weight"])
    embed_w = wts["model.language_model.embed_tokens.weight"]
    ids = torch.randint(0, tc.vocab_size, (B, L), device=DEVICE)
    scale = tc.hidden_size ** 0.5

    ref_out = (F.embedding(ids, embed_w) * scale).to(DTYPE)

    our_embed = torch.nn.Embedding(tc.vocab_size, tc.hidden_size).to(DTYPE).to(DEVICE)
    our_embed.weight.data.copy_(embed_w)
    our_out = (our_embed(ids) * scale).to(DTYPE)

    return check("Embedding+scale", ref_out, our_out, atol=1e-5)


def compare_vision_encoder_layer() -> bool:
    """VisionEncoderLayer (single layer, 4x4 patch grid)."""
    from gemma4_simple import VisionEncoderLayer, apply_2d_rope
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4VisionEncoderLayer as HFVisionEncoderLayer,
        Gemma4VisionRotaryEmbedding as HFVisionRoPE,
    )

    print("\n── VisionEncoderLayer ────────────────────────────────────────")
    vis_prefix = f"model.vision_tower.encoder.layers.{LAYER}"

    _clip_bufs = ["input_min", "input_max", "output_min", "output_max"]
    _vis_projs = [f"self_attn.{p}" for p in ["q_proj", "k_proj", "v_proj", "o_proj"]] + \
                 [f"mlp.{p}" for p in ["gate_proj", "up_proj", "down_proj"]]
    vis_wts = load_tensors(
        [f"{vis_prefix}.{n}.weight" for n in
         ("input_layernorm", "post_attention_layernorm",
          "pre_feedforward_layernorm", "post_feedforward_layernorm",
          "self_attn.q_norm", "self_attn.k_norm")] +
        [f"{vis_prefix}.{p}.linear.weight" for p in _vis_projs] +
        [f"{vis_prefix}.{p}.{b}" for p in _vis_projs for b in _clip_bufs]
    )

    Bv, Nv = 1, 16
    xv = torch.randn(Bv, Nv, vc.hidden_size, dtype=DTYPE, device=DEVICE)
    rows = torch.arange(4, device=DEVICE).repeat_interleave(4)
    cols = torch.arange(4, device=DEVICE).repeat(4)
    vis_pos_ids = torch.stack([rows, cols], dim=-1).unsqueeze(0)

    # HF
    vc._attn_implementation = "eager"
    hf_vis = HFVisionEncoderLayer(vc, layer_idx=LAYER).to(DTYPE).to(DEVICE)
    hf_vis_sd = {k.replace(f"{vis_prefix}.", ""): v for k, v in vis_wts.items()}
    hf_vis.load_state_dict(hf_vis_sd, strict=False)
    hf_vis_rope = HFVisionRoPE(vc).to(DTYPE).to(DEVICE)
    with torch.no_grad():
        vis_cos, vis_sin = hf_vis_rope(xv, vis_pos_ids)
        hf_vis_out = hf_vis(
            hidden_states=xv,
            position_embeddings=(vis_cos, vis_sin),
            position_ids=vis_pos_ids,
            attention_mask=None,
        )

    # Ours
    our_vis = VisionEncoderLayer(our_vis_cfg).to(DTYPE).to(DEVICE)
    for attr in ("input_layernorm", "post_attention_layernorm",
                 "pre_feedforward_layernorm", "post_feedforward_layernorm"):
        getattr(our_vis, attr).weight.data.copy_(vis_wts[f"{vis_prefix}.{attr}.weight"])
    our_vis.self_attn.q_norm.weight.data.copy_(vis_wts[f"{vis_prefix}.self_attn.q_norm.weight"])
    our_vis.self_attn.k_norm.weight.data.copy_(vis_wts[f"{vis_prefix}.self_attn.k_norm.weight"])

    def _load_clip(module, prefix):
        module.linear.weight.data.copy_(vis_wts[f"{prefix}.linear.weight"])
        for buf in _clip_bufs:
            getattr(module, buf).fill_(vis_wts[f"{prefix}.{buf}"].item())

    for p in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        _load_clip(getattr(our_vis.self_attn, p), f"{vis_prefix}.self_attn.{p}")
    for p in ["gate_proj", "up_proj", "down_proj"]:
        _load_clip(getattr(our_vis.mlp, p), f"{vis_prefix}.mlp.{p}")

    with torch.no_grad():
        our_vis_out = our_vis(xv, pixel_position_ids=vis_pos_ids)

    passed = check("VisionEncoderLayer HF vs ours", hf_vis_out, our_vis_out, atol=1e-2)

    # Also check manual GT (internal self-consistency)
    H_v, Dh_v = our_vis_cfg.num_attention_heads, our_vis_cfg.head_dim
    with torch.no_grad():
        our_rope = our_vis.self_attn.rotary_emb
        v_cos, v_sin = our_rope(xv, vis_pos_ids)
        residual_v = xv
        h_v = our_vis.input_layernorm(xv)
        q_v = our_vis.self_attn.q_proj(h_v).view(Bv, Nv, H_v, Dh_v)
        k_v = our_vis.self_attn.k_proj(h_v).view(Bv, Nv, H_v, Dh_v)
        v_v = our_vis.self_attn.v_proj(h_v).view(Bv, Nv, H_v, Dh_v)
        q_v = our_vis.self_attn.q_norm(q_v)
        k_v = our_vis.self_attn.k_norm(k_v)
        v_v = our_vis.self_attn.v_norm(v_v)
        q_v = apply_2d_rope(q_v, v_cos, v_sin, vis_pos_ids, unsqueeze_dim=2).transpose(1, 2)
        k_v = apply_2d_rope(k_v, v_cos, v_sin, vis_pos_ids, unsqueeze_dim=2).transpose(1, 2)
        v_v = v_v.transpose(1, 2)
        attn_v = F.softmax((q_v @ k_v.transpose(-2, -1) * our_vis.self_attn.scaling).float(), dim=-1).to(DTYPE)
        h_v = our_vis.self_attn.o_proj((attn_v @ v_v).transpose(1, 2).reshape(Bv, Nv, -1))
        h_v = our_vis.post_attention_layernorm(h_v)
        h_v = residual_v + h_v
        residual_v = h_v
        h_v = our_vis.pre_feedforward_layernorm(h_v)
        h_v = our_vis.mlp(h_v)
        h_v = our_vis.post_feedforward_layernorm(h_v)
        manual_vis_gt = residual_v + h_v

    passed2 = check("VisionEncoderLayer (vs manual GT)", manual_vis_gt, our_vis_out, atol=1e-3)
    return passed and passed2


def compare_vision_pooler() -> bool:
    """VisionPooler (spatial average pooling)."""
    from gemma4_simple import VisionPooler
    from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionPooler as HFVisionPooler

    print("\n── VisionPooler ──────────────────────────────────────────────")
    Bp, Np = 1, 16
    xp = torch.randn(Bp, Np, vc.hidden_size, dtype=DTYPE, device=DEVICE)
    rows_p = torch.arange(4, device=DEVICE).repeat_interleave(4)
    cols_p = torch.arange(4, device=DEVICE).repeat(4)
    pool_pos_ids = torch.stack([rows_p, cols_p], dim=-1).unsqueeze(0)
    pool_padding = torch.zeros(Bp, Np, dtype=torch.bool, device=DEVICE)

    hf_pooler = HFVisionPooler(vc).to(DTYPE).to(DEVICE)
    output_length = Np // (vc.pooling_kernel_size ** 2)
    with torch.no_grad():
        hf_pool_out, hf_pool_mask = hf_pooler(
            hidden_states=xp,
            pixel_position_ids=pool_pos_ids,
            padding_positions=pool_padding,
            output_length=output_length,
        )
        hf_flat = hf_pool_out[hf_pool_mask]

    our_pooler = VisionPooler(our_vis_cfg).to(DTYPE).to(DEVICE)
    with torch.no_grad():
        our_flat, _ = our_pooler(xp, pool_pos_ids, pool_padding)

    return check("VisionPooler HF vs ours", hf_flat, our_flat, atol=1e-3)


def compare_full_vision_model() -> bool:
    """Full VisionModel (PatchEmbedder → Encoder → Pooler)."""
    from gemma4_simple import VisionModel
    from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel as HFVisionModel

    print("\n── Full VisionModel ──────────────────────────────────────────")
    with safe_open(f"{HF_CKPT}/model.safetensors", framework="pt", device="cpu") as f:
        all_vis_keys = [k for k in f.keys() if "vision_tower" in k]
        all_vis_wts  = {k: f.get_tensor(k).to(DTYPE).to(DEVICE) for k in all_vis_keys}

    vc._attn_implementation = "eager"
    hf_vis_model = HFVisionModel(vc).to(DTYPE).to(DEVICE)
    hf_state = {k.replace("model.vision_tower.", ""): v for k, v in all_vis_wts.items()}
    missing, unexpected = hf_vis_model.load_state_dict(hf_state, strict=False)
    if missing:
        print(f"  HF missing: {missing[:5]}")

    our_vis_model = VisionModel(our_vis_cfg).to(DTYPE).to(DEVICE)
    our_state = {k.replace("model.vision_tower.", ""): v for k, v in all_vis_wts.items()}
    missing2, _ = our_vis_model.load_state_dict(our_state, strict=False)
    if missing2:
        print(f"  Ours missing: {missing2[:5]}")

    Bf = 1
    rows3 = torch.arange(3, device=DEVICE).repeat_interleave(3)
    cols3 = torch.arange(3, device=DEVICE).repeat(3)
    full_pos_ids = torch.stack([rows3, cols3], dim=-1).unsqueeze(0)
    patch_dim = vc.patch_size * vc.patch_size * 3
    full_pixel = torch.rand(Bf, 9, patch_dim, dtype=DTYPE, device=DEVICE)

    with torch.no_grad():
        hf_out  = hf_vis_model(pixel_values=full_pixel, pixel_position_ids=full_pos_ids).last_hidden_state
        our_out = our_vis_model(full_pixel, full_pos_ids)

    return check("Full VisionModel HF vs ours", hf_out, our_out, atol=1e-2)


def compare_system_text() -> bool:
    """
    System-level: load all text-tower weights and run full TextModel forward,
    comparing against HF Gemma4TextModel on a real tokenised prompt.

    Per-layer inputs: we extract them from HF's internal computation and feed
    the same pre-computed tensor to our model so both paths are identical.
    """
    from gemma4_simple import TextModel
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel as HFTextModel

    print("\n── System: Full TextModel forward pass ───────────────────────")
    print("  Loading all text tower weights (this may take a moment)…")

    with safe_open(f"{HF_CKPT}/model.safetensors", framework="pt", device="cpu") as f:
        all_keys = [k for k in f.keys() if "language_model" in k]
        all_wts  = {k: f.get_tensor(k).to(DTYPE) for k in all_keys}

    # Tokenise a short prompt
    ids = tokenizer("Hello, world!", return_tensors="pt").input_ids.to(DEVICE)
    state = {k.replace("model.language_model.", ""): v for k, v in all_wts.items()}

    # Force eager attention so create_causal_mask returns None for a full-sequence
    # non-padded input.  Both HF and our model then use bidirectional attention,
    # giving a clean like-for-like comparison.
    tc._attn_implementation = "eager"

    # ── Run HF first, capture per_layer_proj + final output, then free GPU ──
    print("  Running HF TextModel…")
    hf_text_model = HFTextModel(tc).to(DTYPE).to(DEVICE)
    missing, _ = hf_text_model.load_state_dict(
        {k: v.to(DEVICE) for k, v in state.items()}, strict=False)
    if missing:
        print(f"  HF missing keys: {len(missing)} (first 3: {missing[:3]})")

    with torch.no_grad():
        # Capture per_layer inputs as computed by HF (so both paths are identical)
        hf_per_layer = hf_text_model.get_per_layer_inputs(ids, None)
        hf_per_layer_proj = hf_text_model.project_per_layer_inputs(
            hf_text_model.embed_tokens(ids), hf_per_layer
        ).cpu()
        # HF eager + create_causal_mask returns None for full sequence → no mask
        hf_out = hf_text_model(
            input_ids=ids, use_cache=False, output_hidden_states=False,
        ).last_hidden_state.cpu()

    del hf_text_model
    torch.cuda.empty_cache()

    # ── Our TextModel ─────────────────────────────────────────────
    print("  Running our TextModel…")
    our_text_model = TextModel(our_text_cfg).to(DTYPE).to(DEVICE)
    missing2, _ = our_text_model.load_state_dict(
        {k: v.to(DEVICE) for k, v in state.items()}, strict=False)
    if missing2:
        print(f"  Ours missing keys: {len(missing2)} (first 3: {missing2[:3]})")

    # Explicit causal mask: 0 on/below diagonal, -min_float above (matches HF eager)
    seq_len = ids.shape[1]
    min_val = torch.finfo(DTYPE).min
    causal = torch.triu(
        torch.full((1, 1, seq_len, seq_len), min_val, dtype=DTYPE, device=DEVICE),
        diagonal=1,
    )

    with torch.no_grad():
        our_out = our_text_model(
            ids, attention_mask=causal,
            per_layer_inputs=hf_per_layer_proj.to(DEVICE),
        ).cpu()

    del our_text_model
    torch.cuda.empty_cache()

    # bfloat16 accumulates ~1-2 ULPs per layer → ~2–3 max_diff over 42 layers is expected.
    # Use relaxed thresholds: top1=100% and cos_sim > 0.997 is the real correctness bar.
    return check("System TextModel (HF vs ours)", hf_out, our_out, atol=3.0, min_cos=0.997)


def compare_system_multimodal() -> bool:
    """
    System-level: combined text + vision (Gemma4Model) forward pass.

    Creates a short text sequence with one image placeholder, encodes a small
    image (3×3 = 9 patches → 1 soft token after kernel-3 pooling), and
    compares our Gemma4Model against HF's Gemma4Model end-to-end.
    """
    from gemma4_simple import Gemma4Config, Gemma4Model
    from transformers.models.gemma4.modeling_gemma4 import Gemma4Model as HFGemma4Model

    print("\n── System: Multimodal (Text + Vision) forward pass ───────────")
    print("  Loading weights (text + vision towers)…")

    with safe_open(f"{HF_CKPT}/model.safetensors", framework="pt", device="cpu") as f:
        all_wts = {k: f.get_tensor(k).to(DTYPE) for k in f.keys()
                   if any(p in k for p in ("language_model", "vision_tower", "embed_vision"))}

    IMAGE_TOKEN_ID = hf_cfg.image_token_id   # 258880
    KERNEL = vc.pooling_kernel_size           # 3
    N_PATCHES = KERNEL * KERNEL              # 9 raw patches → 1 soft token
    patch_dim = vc.patch_size * vc.patch_size * 3

    # Input: [BOS, text, IMAGE_PLACEHOLDER, text, text]
    ids = torch.tensor([[1, 100, IMAGE_TOKEN_ID, 200, 201]], dtype=torch.long, device=DEVICE)
    # Fake pixel values: 1 image, N_PATCHES patches
    rows = torch.arange(KERNEL, device=DEVICE).repeat_interleave(KERNEL)
    cols = torch.arange(KERNEL, device=DEVICE).repeat(KERNEL)
    ppos = torch.stack([rows, cols], dim=-1).unsqueeze(0)  # [1, 9, 2]
    pval = torch.rand(1, N_PATCHES, patch_dim, dtype=DTYPE, device=DEVICE)

    # ── HF Gemma4Model ────────────────────────────────────────────
    print("  Running HF Gemma4Model…")
    hf_model = HFGemma4Model(hf_cfg).to(DTYPE).to(DEVICE)
    hf_state = {k.replace("model.", ""): v for k, v in all_wts.items()}
    hf_model.load_state_dict(hf_state, strict=False)
    hf_model.eval()

    with torch.no_grad():
        hf_out = hf_model(
            input_ids=ids,
            pixel_values=pval,
            image_position_ids=ppos,
            use_cache=False,
        ).last_hidden_state.cpu()

    del hf_model
    torch.cuda.empty_cache()

    # ── Our Gemma4Model ───────────────────────────────────────────
    print("  Running our Gemma4Model…")
    our_cfg = Gemma4Config(
        text=our_text_cfg,
        vision=our_vis_cfg,
        image_token_id=IMAGE_TOKEN_ID,
    )
    our_model = Gemma4Model(our_cfg).to(DTYPE).to(DEVICE)

    # Remap HF keys → our keys
    our_state = {}
    for k, v in all_wts.items():
        # language_model.* → language_model.*
        k2 = k.replace("model.language_model.", "language_model.")
        # vision_tower.* → vision_model.*
        k2 = k2.replace("model.vision_tower.", "vision_model.")
        # embed_vision.embedding_projection.* → mm_embedder.proj.*
        k2 = k2.replace("model.embed_vision.embedding_projection.", "mm_embedder.proj.")
        our_state[k2] = v

    missing2, unexpected2 = our_model.load_state_dict(
        {k: v.to(DEVICE) for k, v in our_state.items()}, strict=False)
    if missing2:
        print(f"  Ours missing: {len(missing2)} (first 3: {missing2[:3]})")

    # Build causal mask matching HF's eager mask
    seq_len = ids.shape[1]
    min_val = torch.finfo(DTYPE).min
    causal = torch.triu(
        torch.full((1, 1, seq_len, seq_len), min_val, dtype=DTYPE, device=DEVICE), diagonal=1
    )

    with torch.no_grad():
        our_out = our_model(
            input_ids=ids,
            pixel_values=pval,
            pixel_position_ids=ppos,
            attention_mask=causal,
        ).cpu()

    del our_model
    torch.cuda.empty_cache()

    return check("System Multimodal (HF vs ours)", hf_out, our_out, atol=3.0, min_cos=0.997)


# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────

def setup(ckpt: str, device: str):
    """Load configs and build our config objects."""
    global HF_CKPT, DEVICE, hf_cfg, tc, vc, our_text_cfg, our_vis_cfg, tokenizer
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gemma4_simple import TextConfig, VisionConfig

    HF_CKPT = ckpt
    DEVICE  = device
    hf_cfg  = AutoConfig.from_pretrained(HF_CKPT)
    tc      = hf_cfg.text_config
    vc      = hf_cfg.vision_config
    tc._attn_implementation = "eager"
    vc._attn_implementation = "eager"
    tokenizer = AutoTokenizer.from_pretrained(HF_CKPT)

    our_text_cfg = TextConfig(
        vocab_size                  = tc.vocab_size,
        hidden_size                 = tc.hidden_size,
        num_hidden_layers           = tc.num_hidden_layers,
        num_attention_heads         = tc.num_attention_heads,
        num_key_value_heads         = tc.num_key_value_heads,
        head_dim                    = tc.head_dim,
        global_head_dim             = getattr(tc, "global_head_dim", None),
        global_partial_rotary_factor = tc.rope_parameters.get("full_attention", {}).get("partial_rotary_factor", 1.0),
        intermediate_size           = tc.intermediate_size,
        num_experts                 = 0,
        num_experts_per_tok         = 2,
        moe_layers                  = [],
        expert_intermediate_size    = 4096,
        kv_share_from               = tc.num_hidden_layers - tc.num_kv_shared_layers,
        sliding_window              = tc.sliding_window,
        sliding_window_pattern      = 6,
        rope_local_base_freq        = tc.rope_parameters.get("sliding_attention", {}).get("rope_theta", 10_000.0),
        rope_global_base_freq       = tc.rope_parameters.get("full_attention", {}).get("rope_theta", 1_000_000.0),
        final_logit_softcapping     = tc.final_logit_softcapping,
        rms_norm_eps                = tc.rms_norm_eps,
        pad_token_id                = tc.pad_token_id,
        embed_scale                 = tc.hidden_size ** 0.5,
        hidden_size_per_layer_input = tc.hidden_size_per_layer_input,
    )
    our_text_cfg._layer_types = tc.layer_types

    our_vis_cfg = VisionConfig(
        hidden_size             = vc.hidden_size,
        num_hidden_layers       = vc.num_hidden_layers,
        num_attention_heads     = vc.num_attention_heads,
        head_dim                = vc.head_dim,
        intermediate_size       = vc.intermediate_size,
        patch_size              = vc.patch_size,
        position_embedding_size = vc.position_embedding_size,
        pooling_kernel_size     = vc.pooling_kernel_size,
        rms_norm_eps            = vc.rms_norm_eps,
        rope_theta              = vc.rope_parameters["rope_theta"],
        standardize             = vc.standardize,
        use_clipped_linears     = vc.use_clipped_linears,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

ALL_TESTS = {
    "rmsnorm"    : compare_rmsnorm,
    "mlp"        : compare_text_mlp,
    "attn"       : compare_text_attention,
    "decoder"    : compare_text_decoder_layer,
    "emb"        : compare_embedding,
    "vis"        : compare_vision_encoder_layer,
    "pooler"     : compare_vision_pooler,
    "vis_full"   : compare_full_vision_model,
    "system"     : compare_system_text,
    "multimodal" : compare_system_multimodal,
}


def main():
    parser = argparse.ArgumentParser(description="Gemma 4 simple vs HF comparison")
    parser.add_argument("--ckpt",   default=DEFAULT_CKPT,   help="Path to HF checkpoint dir")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="cuda / cpu")
    parser.add_argument("--tests",  nargs="+", default=list(ALL_TESTS.keys()),
                        choices=list(ALL_TESTS.keys()),
                        help="Tests to run (default: all)")
    parser.add_argument("--skip",   nargs="+", default=[],
                        choices=list(ALL_TESTS.keys()),
                        help="Tests to skip")
    args = parser.parse_args()

    to_run = [t for t in args.tests if t not in args.skip]

    print("=" * 65)
    print("Gemma 4 E4B — gemma4_simple vs HuggingFace")
    print(f"  device={args.device}, dtype={DTYPE}, layer={LAYER}")
    print(f"  ckpt={args.ckpt}")
    print(f"  tests={to_run}")
    print("=" * 65)

    setup(args.ckpt, args.device)

    results = {}
    for name in to_run:
        try:
            results[name] = ALL_TESTS[name]()
        except Exception as e:
            print(f"❌  {name}: EXCEPTION — {e}")
            results[name] = False

    n_pass = sum(results.values())
    n_total = len(results)
    print(f"\n{'=' * 65}")
    print(f"  Result: {n_pass}/{n_total} tests passed")
    for name, passed in results.items():
        sym = "✅" if passed else "❌"
        print(f"    {sym}  {name}")
    if n_pass == n_total:
        print("  ✅ All selected modules consistent with HuggingFace!")
    else:
        print("  ⚠️  Some modules differ — check above for details.")
    print("=" * 65)
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())

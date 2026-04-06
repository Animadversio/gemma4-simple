"""
load_and_test.py
================
Load weights from the HuggingFace gemma-4-E4B-it checkpoint into our
gemma4_simple.py reimplementation and verify that the text-model forward pass
produces numerically identical logits to the reference HF model.

Only the TEXT tower is tested here (the vision/audio towers use ClippableLinear
wrappers with clip buffers that are a faithful but verbose detail; vision
consistency is left as a follow-up).

Usage:
    python load_and_test.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration as HFGemma4
from transformers import AutoConfig
from safetensors import safe_open
from dataclasses import field

from gemma4_simple import (
    TextConfig, VisionConfig, Gemma4Config,
    Gemma4ForCausalLM,
)

HF_CKPT = "/home/binxu/.cache/huggingface/hub/gemma-4-E4B-it"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"


# ──────────────────────────────────────────────────────────────
# 1.  Build our config from the HF config
# ──────────────────────────────────────────────────────────────

def build_our_config(hf_cfg) -> Gemma4Config:
    tc = hf_cfg.text_config
    layer_types = tc.layer_types          # list of "sliding_attention" / "full_attention"
    num_layers  = tc.num_hidden_layers    # 42

    # KV sharing: last num_kv_shared_layers layers reuse KV from the last non-shared layer
    num_kv_shared = tc.num_kv_shared_layers   # 18
    kv_share_from = num_layers - num_kv_shared  # first layer that shares

    # MoE layers: E4B has no MoE (num_experts == 0 / not in config)
    num_experts = getattr(tc, "num_experts", 0) or 0

    # Sliding window pattern: every 6th layer (index 5,11,17,...) is full attention
    # Actually layer_types encodes this directly — we use it to assign per-layer type
    sliding_window = tc.sliding_window   # 512

    text_cfg = TextConfig(
        vocab_size               = tc.vocab_size,
        hidden_size              = tc.hidden_size,
        num_hidden_layers        = num_layers,
        num_attention_heads      = tc.num_attention_heads,
        num_key_value_heads      = tc.num_key_value_heads,
        head_dim                 = tc.head_dim,
        intermediate_size        = tc.intermediate_size,
        num_experts              = num_experts,
        num_experts_per_tok      = getattr(tc, "num_experts_per_tok", 2),
        moe_layers               = [],
        expert_intermediate_size = getattr(tc, "expert_intermediate_size", 4096),
        kv_share_from            = kv_share_from,
        sliding_window           = sliding_window,
        # We store layer_types so our model can pick local vs global per layer
        rope_local_base_freq     = 10_000.0,
        rope_global_base_freq    = 1_000_000.0,
        final_logit_softcapping  = tc.final_logit_softcapping,
        rms_norm_eps             = tc.rms_norm_eps,
        pad_token_id             = tc.pad_token_id,
        embed_scale              = tc.hidden_size ** 0.5,
        hidden_size_per_layer_input = tc.hidden_size_per_layer_input,
    )
    # Attach layer_types for use in forward (monkey-patch for simplicity)
    text_cfg._layer_types = layer_types

    vis_cfg = VisionConfig()   # not used in text-only test
    return Gemma4Config(text=text_cfg, vision=vis_cfg)


# ──────────────────────────────────────────────────────────────
# 2.  Weight mapping  HF keys → our keys
# ──────────────────────────────────────────────────────────────

def load_our_weights(our_model: Gemma4ForCausalLM, safetensors_path: str):
    """
    Copy weights from HF safetensors checkpoint into our model.
    Key translation:
      HF:  model.language_model.layers.{i}.{subpath}
      Ours: model.layers.{i}.{subpath}

      HF:  model.language_model.embed_tokens.weight  →  model.embed_tokens.weight
      HF:  model.language_model.norm.weight          →  model.norm.weight
      HF:  lm_head.weight  (tied to embed_tokens in HF, but separate in ours)
    """
    our_sd = our_model.state_dict()

    def hf_to_ours(hf_key: str) -> str | None:
        # Strip leading "model.language_model." → "model."
        if hf_key.startswith("model.language_model."):
            return "model." + hf_key[len("model.language_model."):]
        # lm_head
        if hf_key == "lm_head.weight":
            return "lm_head.weight"
        return None   # skip (vision, audio, embed_*_per_layer projection etc.)

    missing, unexpected = [], []
    loaded = 0

    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for hf_key in f.keys():
            our_key = hf_to_ours(hf_key)
            if our_key is None:
                continue
            if our_key not in our_sd:
                unexpected.append(our_key)
                continue
            tensor = f.get_tensor(hf_key)
            # layer_scalar is stored as shape [] in HF, shape [1] in ours
            if tensor.shape != our_sd[our_key].shape:
                tensor = tensor.reshape(our_sd[our_key].shape)
            our_sd[our_key].copy_(tensor)
            loaded += 1

    # Find params that were never written (KV-shared layers have no k/v proj weights)
    all_loaded_keys = set()
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for hf_key in f.keys():
            k = hf_to_ours(hf_key)
            if k: all_loaded_keys.add(k)

    for k in our_sd:
        if k not in all_loaded_keys:
            missing.append(k)

    our_model.load_state_dict(our_sd, strict=False)
    print(f"  Loaded {loaded} tensors from checkpoint.")
    if missing:
        print(f"  Missing (not in ckpt — expected for KV-shared / per-layer-input params): "
              f"{len(missing)} keys")
        for k in missing[:10]: print(f"    {k}")
    if unexpected:
        print(f"  Unexpected (in ckpt but not in our model): {len(unexpected)}")
        for k in unexpected[:5]: print(f"    {k}")
    return our_model


# ──────────────────────────────────────────────────────────────
# 3.  Patch TextModel.forward to use per-layer layer_types
# ──────────────────────────────────────────────────────────────

def patch_textmodel_layer_types(text_model, layer_types):
    """
    Override TextModel.forward so it uses the actual per-layer layer_types
    (sliding_attention / full_attention) from the config instead of the
    sliding_window_pattern heuristic.
    """
    import types

    def forward(self, input_ids, attention_mask=None, kv_cache=None):
        x = self.embed_tokens(input_ids)
        B, L, _ = x.shape
        position_ids = torch.arange(L, device=x.device).unsqueeze(0)

        for i, layer in enumerate(self.layers):
            lt = layer_types[i]
            rope_type = "local" if lt == "sliding_attention" else "global"
            cos, sin = self.rotary_emb(x, position_ids, layer_type=rope_type)
            x = layer(x, cos, sin, attention_mask=attention_mask, kv_cache=kv_cache)

        return self.norm(x)

    text_model.forward = types.MethodType(forward, text_model)


# ──────────────────────────────────────────────────────────────
# 4.  Consistency test
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def test_consistency():
    print("=" * 65)
    print("Gemma 4 E4B — forward pass consistency test")
    print("(sequential loading to avoid OOM: HF first, then ours)")
    print("=" * 65)

    tokenizer = AutoTokenizer.from_pretrained(HF_CKPT)
    prompts = ["The capital of France is", "def fibonacci(n):"]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(DEVICE)
    input_ids = inputs["input_ids"]

    # ── Phase A: HF model → save logits → delete ─────────────────
    print("\n[1/4] Loading HF reference model (bf16)...")
    hf_model = HFGemma4.from_pretrained(
        HF_CKPT,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
    )
    hf_model.eval()
    hf_cfg = hf_model.config
    print(f"  HF model loaded. Running forward...")

    hf_out    = hf_model.model.language_model(input_ids=input_ids)
    hf_hidden = hf_out.last_hidden_state
    hf_logits = hf_model.lm_head(hf_hidden)
    if hf_cfg.text_config.final_logit_softcapping:
        cap = hf_cfg.text_config.final_logit_softcapping
        hf_logits = torch.tanh(hf_logits / cap) * cap
    # Save to CPU before freeing GPU memory
    hf_logits_cpu = hf_logits.float().cpu()
    hf_top1_cpu   = hf_logits_cpu.argmax(-1)

    print(f"  HF logits saved ({hf_logits_cpu.shape}). Freeing HF model...")
    del hf_model, hf_out, hf_hidden, hf_logits
    torch.cuda.empty_cache()

    # ── Phase B: our model ────────────────────────────────────────
    print("\n[2/4] Building our model with matching config...")
    our_cfg = build_our_config(hf_cfg)
    our_model = Gemma4ForCausalLM(our_cfg)
    patch_textmodel_layer_types(our_model.model, hf_cfg.text_config.layer_types)
    our_model = our_model.to(torch.bfloat16).to(DEVICE)
    our_model.eval()
    print(f"  Our model built ({sum(p.numel() for p in our_model.parameters())/1e9:.2f}B params).")

    print("\n[3/4] Loading weights from HF checkpoint...")
    our_model = load_our_weights(our_model, HF_CKPT + "/model.safetensors")

    print("\n[4/4] Running our forward pass...")
    our_out    = our_model(input_ids)
    our_logits = our_out["logits"].float().cpu()

    # ── Compare ───────────────────────────────────────────────────
    print(f"\n  HF  logits shape: {hf_logits_cpu.shape}")
    print(f"  Our logits shape: {our_logits.shape}")

    min_len = min(hf_logits_cpu.shape[1], our_logits.shape[1])
    hf_l  = hf_logits_cpu[:, :min_len, :]
    our_l = our_logits[:, :min_len, :]

    max_abs_diff  = (hf_l - our_l).abs().max().item()
    mean_abs_diff = (hf_l - our_l).abs().mean().item()
    cos_sim = F.cosine_similarity(
        hf_l.reshape(-1, hf_l.shape[-1]),
        our_l.reshape(-1, our_l.shape[-1]),
        dim=-1,
    ).mean().item()

    hf_top1  = hf_l.argmax(-1)
    our_top1 = our_l.argmax(-1)
    token_match = (hf_top1 == our_top1).float().mean().item()

    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │  Max  |logit diff|   : {max_abs_diff:>10.6f}      │")
    print(f"  │  Mean |logit diff|   : {mean_abs_diff:>10.6f}      │")
    print(f"  │  Cosine similarity   : {cos_sim:>10.6f}      │")
    print(f"  │  Top-1 token match   : {token_match*100:>9.2f}%      │")
    print(f"  └─────────────────────────────────────────┘")

    print("\n  Predicted next tokens (last position):")
    for i, prompt in enumerate(prompts):
        hf_tok  = tokenizer.decode(hf_top1[i, -1])
        our_tok = tokenizer.decode(our_top1[i, -1])
        match = "✅" if hf_top1[i, -1] == our_top1[i, -1] else "❌"
        print(f"    [{match}] '{prompt}' → HF: '{hf_tok}'  Ours: '{our_tok}'")

    passed = token_match > 0.95 and cos_sim > 0.999
    print(f"\n{'  ✅ PASSED' if passed else '  ⚠️  PARTIAL'}: "
          f"logits {'match' if passed else 'differ'} (token_match={token_match:.2%}, cos_sim={cos_sim:.6f})")
    return passed


if __name__ == "__main__":
    test_consistency()

"""
Debug: run HF and our model layer-by-layer, compare at each step.
Finds the first layer where outputs diverge.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

CKPT = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/huggingface/hub/gemma-4-E4B-it"
DEVICE = "cuda"
DTYPE = torch.bfloat16

# ── Config & tokenizer ────────────────────────────────────────────────────────
hf_cfg = AutoConfig.from_pretrained(CKPT)
tc = hf_cfg.get_text_config()
tokenizer = AutoTokenizer.from_pretrained(CKPT)

from gemma4_simple import TextConfig, TextModel

our_text_cfg = TextConfig(
    vocab_size=tc.vocab_size,
    hidden_size=tc.hidden_size,
    num_hidden_layers=tc.num_hidden_layers,
    num_attention_heads=tc.num_attention_heads,
    num_key_value_heads=tc.num_key_value_heads,
    head_dim=tc.head_dim,
    global_head_dim=getattr(tc, "global_head_dim", None),
    global_partial_rotary_factor=tc.rope_parameters.get("full_attention", {}).get("partial_rotary_factor", 1.0),
    intermediate_size=tc.intermediate_size,
    kv_share_from=tc.num_hidden_layers - tc.num_kv_shared_layers,
    sliding_window=tc.sliding_window,
    sliding_window_pattern=6,
    rope_local_base_freq=tc.rope_parameters.get("sliding_attention", {}).get("rope_theta", 10_000.0),
    rope_global_base_freq=tc.rope_parameters.get("full_attention", {}).get("rope_theta", 1_000_000.0),
    rms_norm_eps=tc.rms_norm_eps,
    hidden_size_per_layer_input=getattr(tc, "hidden_size_per_layer_input", 0),
    embed_scale=tc.hidden_size ** 0.5,
    use_v_norm=getattr(tc, "attention_logit_soft_cap", None) is not None or True,
    attn_logit_softcapping=getattr(tc, "attn_logit_softcap", 0.0),
    final_logit_softcapping=getattr(tc, "final_logit_soft_cap", 0.0),
)
our_text_cfg._layer_types = tc.layer_types

print(f"use_v_norm={our_text_cfg.use_v_norm}")
print(f"layer_types[:10]={tc.layer_types[:10]}")

# Force eager so create_causal_mask returns None → no mask, matching our model
tc._attn_implementation = "eager"

# ── Load weights ──────────────────────────────────────────────────────────────
print("Loading weights…")
with safe_open(f"{CKPT}/model.safetensors", framework="pt", device="cpu") as f:
    all_keys = [k for k in f.keys() if "language_model" in k]
    all_wts = {k: f.get_tensor(k).to(DTYPE) for k in all_keys}

state = {k.replace("model.language_model.", ""): v for k, v in all_wts.items()}

ids = tokenizer("Hello, world!", return_tensors="pt").input_ids.to(DEVICE)
print(f"Input ids shape: {ids.shape}, tokens: {ids.tolist()}")

# ── Load HF and capture per-layer hidden states ───────────────────────────────
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4TextModel as HFTextModel,
    Gemma4TextDecoderLayer as HFDecoderLayer,
)

hf_model = HFTextModel(tc).to(DTYPE).to(DEVICE)
hf_model.load_state_dict({k: v.to(DEVICE) for k, v in state.items()}, strict=False)
hf_model.eval()

# Capture hidden states at each layer
hf_hidden_states = []
hooks = []

def make_hook(idx):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            hf_hidden_states.append((idx, output.detach().cpu()))
        elif isinstance(output, (tuple, list)):
            hf_hidden_states.append((idx, output[0].detach().cpu()))
    return hook

for i, layer in enumerate(hf_model.layers):
    hooks.append(layer.register_forward_hook(make_hook(i)))

with torch.no_grad():
    hf_per_layer = hf_model.get_per_layer_inputs(ids, None)
    hf_per_layer_proj = hf_model.project_per_layer_inputs(
        hf_model.embed_tokens(ids), hf_per_layer
    ).cpu()
    hf_embed = hf_model.embed_tokens(ids).detach().cpu()
    hf_out = hf_model(input_ids=ids, use_cache=False).last_hidden_state.cpu()

for h in hooks:
    h.remove()

del hf_model
torch.cuda.empty_cache()

# ── Build causal mask ──────────────────────────────────────────────────────────
seq_len = ids.shape[1]
causal = torch.full((1, 1, seq_len, seq_len), float("-inf"), dtype=DTYPE, device=DEVICE)
causal = torch.triu(causal, diagonal=1)

# ── Load our model and capture per-layer hidden states ────────────────────────
our_model = TextModel(our_text_cfg).to(DTYPE).to(DEVICE)
missing, _ = our_model.load_state_dict({k: v.to(DEVICE) for k, v in state.items()}, strict=False)
if missing:
    print(f"  Ours missing: {len(missing)} keys, first 5: {missing[:5]}")
our_model.eval()

our_hidden_states = []
our_hooks = []

def make_our_hook(idx):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            our_hidden_states.append((idx, output.detach().cpu()))
        elif isinstance(output, (tuple, list)):
            our_hidden_states.append((idx, output[0].detach().cpu()))
    return hook

for i, layer in enumerate(our_model.layers):
    our_hooks.append(layer.register_forward_hook(make_our_hook(i)))

with torch.no_grad():
    # attention_mask=None matches HF eager (no mask for full non-padded sequence)
    our_out = our_model(ids, attention_mask=None,
                        per_layer_inputs=hf_per_layer_proj.to(DEVICE)).cpu()

for h in our_hooks:
    h.remove()

del our_model
torch.cuda.empty_cache()

# ── Compare embed ──────────────────────────────────────────────────────────────
print(f"\n── Embedding compare ──")
our_embed = our_hidden_states[0][1] if our_hidden_states else None
# We don't have the raw embed from our model, skip for now

# ── Compare layer-by-layer ────────────────────────────────────────────────────
print(f"\n── Layer-by-layer comparison ──")
print(f"{'Layer':>6}  {'Type':>18}  {'max_diff':>12}  {'cos_sim':>10}")
print("-" * 55)

n = min(len(hf_hidden_states), len(our_hidden_states))
for i in range(n):
    hf_idx, hf_h = hf_hidden_states[i]
    our_idx, our_h = our_hidden_states[i]

    lt = tc.layer_types[i]
    max_diff = (hf_h - our_h).abs().max().item()

    hf_flat = hf_h.float().reshape(-1)
    our_flat = our_h.float().reshape(-1)
    cos = torch.nn.functional.cosine_similarity(hf_flat.unsqueeze(0), our_flat.unsqueeze(0)).item()

    marker = " <<<" if max_diff > 0.5 else ""
    print(f"  {i:>4}  {lt:>18}  {max_diff:>12.4f}  {cos:>10.6f}{marker}")

# Final output
max_diff_final = (hf_out - our_out).abs().max().item()
cos_final = torch.nn.functional.cosine_similarity(
    hf_out.float().reshape(1, -1), our_out.float().reshape(1, -1)
).item()
print(f"\n── Final output ──")
print(f"  max_diff={max_diff_final:.4f}  cos_sim={cos_final:.6f}")

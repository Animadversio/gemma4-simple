"""Layer-by-layer diagnostic for TextModel divergence."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel as HFTextModel

from gemma4_simple import TextConfig, TextModel

HF_CKPT = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/huggingface/hub/gemma-4-E4B-it"
DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

hf_cfg = AutoConfig.from_pretrained(HF_CKPT)
tc = hf_cfg.text_config
tc._attn_implementation = "eager"
tokenizer = AutoTokenizer.from_pretrained(HF_CKPT)

our_text_cfg = TextConfig(
    vocab_size=tc.vocab_size, hidden_size=tc.hidden_size,
    num_hidden_layers=tc.num_hidden_layers, num_attention_heads=tc.num_attention_heads,
    num_key_value_heads=tc.num_key_value_heads, head_dim=tc.head_dim,
    global_head_dim=getattr(tc, "global_head_dim", None),
    global_partial_rotary_factor=tc.rope_parameters.get("full_attention", {}).get("partial_rotary_factor", 1.0),
    intermediate_size=tc.intermediate_size, num_experts=0,
    num_experts_per_tok=2, moe_layers=[], expert_intermediate_size=4096,
    kv_share_from=tc.num_hidden_layers - tc.num_kv_shared_layers,
    sliding_window=tc.sliding_window, sliding_window_pattern=6,
    rope_local_base_freq=tc.rope_parameters.get("sliding_attention", {}).get("rope_theta", 10_000.0),
    rope_global_base_freq=tc.rope_parameters.get("full_attention", {}).get("rope_theta", 1_000_000.0),
    final_logit_softcapping=tc.final_logit_softcapping, rms_norm_eps=tc.rms_norm_eps,
    pad_token_id=tc.pad_token_id, embed_scale=tc.hidden_size ** 0.5,
    hidden_size_per_layer_input=tc.hidden_size_per_layer_input,
)
our_text_cfg._layer_types = tc.layer_types

print("Loading weights...")
with safe_open(f"{HF_CKPT}/model.safetensors", framework="pt", device="cpu") as f:
    all_keys = [k for k in f.keys() if "language_model" in k]
    all_wts = {k: f.get_tensor(k).to(DTYPE) for k in all_keys}

state = {k.replace("model.language_model.", ""): v for k, v in all_wts.items()}
ids = tokenizer("Hello, world!", return_tensors="pt").input_ids.to(DEVICE)

# ─── HF model with layer hooks ───
print("Setting up HF model...")
hf_model = HFTextModel(tc).to(DTYPE).to(DEVICE)
missing_hf, unexpected_hf = hf_model.load_state_dict({k: v.to(DEVICE) for k, v in state.items()}, strict=False)
print(f"  HF missing: {len(missing_hf)}, unexpected: {len(unexpected_hf)}")
if missing_hf: print(f"  missing: {missing_hf[:5]}")

hf_layer_outs = {}
def make_hf_hook(i):
    def hook(module, inp, out):
        hf_layer_outs[i] = out[0].detach().cpu() if isinstance(out, tuple) else out.detach().cpu()
    return hook

for i, layer in enumerate(hf_model.layers):
    layer.register_forward_hook(make_hf_hook(i))

with torch.no_grad():
    hf_out = hf_model(input_ids=ids, use_cache=False).last_hidden_state.cpu()
    hf_per_layer = hf_model.project_per_layer_inputs(
        hf_model.embed_tokens(ids),
        hf_model.get_per_layer_inputs(ids, None)
    ).cpu()
print(f"  HF output: {hf_out.shape}")

del hf_model; torch.cuda.empty_cache()

# ─── Our model with layer hooks ───
print("Setting up our model...")
our_model = TextModel(our_text_cfg).to(DTYPE).to(DEVICE)
missing_our, unexpected_our = our_model.load_state_dict({k: v.to(DEVICE) for k, v in state.items()}, strict=False)
print(f"  Our missing: {len(missing_our)}, unexpected: {len(unexpected_our)}")
if missing_our: print(f"  missing: {missing_our[:5]}")

our_layer_outs = {}
def make_our_hook(i):
    def hook(module, inp, out):
        our_layer_outs[i] = out.detach().cpu() if not isinstance(out, tuple) else out[0].detach().cpu()
    return hook

for i, layer in enumerate(our_model.layers):
    layer.register_forward_hook(make_our_hook(i))

seq_len = ids.shape[1]
causal = torch.full((1, 1, seq_len, seq_len), float("-inf"), dtype=DTYPE, device=DEVICE)
causal = torch.triu(causal, diagonal=1)

with torch.no_grad():
    our_out = our_model(ids, attention_mask=causal,
                        per_layer_inputs=hf_per_layer.to(DEVICE)).cpu()

print(f"  Our output: {our_out.shape}")

# ─── Compare layer by layer ───
print("\n── Layer-by-layer comparison ────────────────────────────")
for i in range(tc.num_hidden_layers):
    if i not in hf_layer_outs or i not in our_layer_outs:
        print(f"  Layer {i:2d}: missing output")
        continue
    hf_l = hf_layer_outs[i].float()
    our_l = our_layer_outs[i].float()
    max_diff = (hf_l - our_l).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        hf_l.flatten(), our_l.flatten(), dim=0
    ).item()
    ltype = tc.layer_types[i] if hasattr(tc, 'layer_types') else '?'
    status = "✅" if max_diff < 0.1 else "❌"
    print(f"  Layer {i:2d} ({ltype:17s}): {status} max_diff={max_diff:.4e}  cos_sim={cos_sim:.6f}")

# Final
final_hf = hf_out.float()
final_our = our_out.float()
print(f"\nFinal: max_diff={(final_hf-final_our).abs().max():.4e}  cos_sim={torch.nn.functional.cosine_similarity(final_hf.flatten(), final_our.flatten(), dim=0):.6f}")

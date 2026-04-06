"""
Compare step-by-step: embeddings, per_layer_inputs, then layer 0 output.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel as HFTextModel

CKPT = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/huggingface/hub/gemma-4-E4B-it"
DEVICE = "cuda"; DTYPE = torch.bfloat16

hf_cfg = AutoConfig.from_pretrained(CKPT)
tc = hf_cfg.get_text_config()
tc._attn_implementation = "eager"
tok = AutoTokenizer.from_pretrained(CKPT)

from gemma4_simple import TextConfig, TextModel

our_cfg = TextConfig(
    vocab_size=tc.vocab_size, hidden_size=tc.hidden_size,
    num_hidden_layers=tc.num_hidden_layers,
    num_attention_heads=tc.num_attention_heads,
    num_key_value_heads=tc.num_key_value_heads,
    head_dim=tc.head_dim, global_head_dim=getattr(tc, "global_head_dim", None),
    global_partial_rotary_factor=tc.rope_parameters.get("full_attention", {}).get("partial_rotary_factor", 1.0),
    intermediate_size=tc.intermediate_size,
    kv_share_from=tc.num_hidden_layers - tc.num_kv_shared_layers,
    sliding_window=tc.sliding_window, sliding_window_pattern=6,
    rope_local_base_freq=tc.rope_parameters.get("sliding_attention", {}).get("rope_theta", 10_000.0),
    rope_global_base_freq=tc.rope_parameters.get("full_attention", {}).get("rope_theta", 1_000_000.0),
    rms_norm_eps=tc.rms_norm_eps, hidden_size_per_layer_input=getattr(tc, "hidden_size_per_layer_input", 0),
    embed_scale=tc.hidden_size**0.5, use_v_norm=False,
)
our_cfg._layer_types = tc.layer_types

print("Loading weights...")
with safe_open(f"{CKPT}/model.safetensors", framework="pt", device="cpu") as f:
    all_keys = [k for k in f.keys() if "language_model" in k]
    state = {k.replace("model.language_model.", ""): f.get_tensor(k).to(DTYPE) for k in all_keys}

ids = tok("Hello, world!", return_tensors="pt").input_ids.to(DEVICE)
print(f"ids: {ids}")

# ── Load both models ──────────────────────────────────────────────────────────
hf = HFTextModel(tc).to(DTYPE).to(DEVICE)
m, _ = hf.load_state_dict({k: v.to(DEVICE) for k, v in state.items()}, strict=False)
if m: print(f"HF missing: {m[:3]}")
hf.eval()

our = TextModel(our_cfg).to(DTYPE).to(DEVICE)
m2, _ = our.load_state_dict({k: v.to(DEVICE) for k, v in state.items()}, strict=False)
if m2: print(f"Ours missing: {m2[:3]}")
our.eval()

def cmp(name, a, b):
    d = (a.float() - b.float()).abs()
    cos = F.cosine_similarity(a.float().reshape(1,-1), b.float().reshape(1,-1)).item()
    print(f"  {name:40s}  max_diff={d.max().item():.4f}  cos={cos:.6f}")

with torch.no_grad():
    # Embeddings
    hf_emb = hf.embed_tokens(ids)  # [1,4,2560]
    our_emb = our.embed_tokens(ids)
    cmp("embed_tokens", hf_emb, our_emb)

    # Per-layer token embeddings
    hf_pli_raw = hf.embed_tokens_per_layer(ids)  # [1,4,42*256]
    our_pli_raw = our.embed_tokens_per_layer(ids)
    cmp("embed_tokens_per_layer", hf_pli_raw, our_pli_raw)

    # Per-layer projection
    N, Dp = tc.num_hidden_layers, tc.hidden_size_per_layer_input
    hf_proj_raw = hf.per_layer_model_projection(hf_emb) * (tc.hidden_size ** -0.5)
    our_proj_raw = our.per_layer_model_projection(our_emb) * our.per_layer_model_projection_scale
    cmp("per_layer_model_projection", hf_proj_raw, our_proj_raw)

    # Per-layer projection norm
    hf_proj_norm = hf.per_layer_projection_norm(hf_proj_raw.reshape(1,4,N,Dp))
    our_proj_norm = our.per_layer_projection_norm(our_proj_raw.reshape(1,4,N,Dp))
    cmp("per_layer_projection_norm", hf_proj_norm, our_proj_norm)

    # Full per_layer_inputs
    hf_pli_full = hf.get_per_layer_inputs(ids, None)  # [1,4,N,Dp]
    hf_pli_proj = hf.project_per_layer_inputs(hf_emb, hf_pli_full)  # [1,4,N,Dp]
    our_pli_full = our._compute_per_layer_inputs(ids, our_emb)  # [1,4,N,Dp]
    cmp("per_layer_inputs[:,:,0,:]", hf_pli_proj[:,:,0,:], our_pli_full[:,:,0,:])
    cmp("per_layer_inputs[:,:,5,:]", hf_pli_proj[:,:,5,:], our_pli_full[:,:,5,:])

    # Now run both models completely and compare
    hf_out = hf(input_ids=ids, use_cache=False).last_hidden_state
    our_out = our(ids, attention_mask=None)  # no causal mask
    cmp("FINAL OUTPUT", hf_out, our_out)

    # Also test with HF per_layer_inputs fed to our model
    our_out2 = our(ids, attention_mask=None, per_layer_inputs=hf_pli_proj)
    cmp("FINAL OUTPUT (HF pli)", hf_out, our_out2)

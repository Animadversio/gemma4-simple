"""
Persistent Gemma4 26B session.
Imports setuptools FIRST to prevent circular import with triton.
"""
import setuptools
import setuptools.dist
import setuptools.extension

import sys
sys.path.insert(0, '/n/home12/binxuwang/Github/gemma4-simple')

print("Importing torch...", flush=True)
import torch
import torch.nn.functional as F
print("Importing transformers...", flush=True)
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel as HFTextModel
from transformers import AutoConfig
print("Transformers OK", flush=True)
from safetensors.torch import load_file as st_load

from gemma4_simple import TextModel, TextConfig

CKPT = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/huggingface/hub/gemma-4-26B-A4B-it"
DTYPE = torch.bfloat16

print("Loading config...", flush=True)
full_cfg = AutoConfig.from_pretrained(CKPT)
text_cfg_hf = full_cfg.text_config
text_cfg_hf._attn_implementation = "eager"

# Parse rope params from rope_parameters dict
rope_params = getattr(text_cfg_hf, 'rope_parameters', {}) or {}
rope_local  = rope_params.get('sliding_attention', {}).get('rope_theta', 10_000.0)
rope_global = rope_params.get('full_attention',    {}).get('rope_theta', 1_000_000.0)
global_partial = rope_params.get('full_attention', {}).get('partial_rotary_factor', 0.25)

print(f"  rope_local={rope_local}, rope_global={rope_global}, global_partial={global_partial}", flush=True)
print(f"  num_global_kv_heads={text_cfg_hf.num_global_key_value_heads}, attention_k_eq_v={text_cfg_hf.attention_k_eq_v}", flush=True)

# Build HF text model (blank, no weights yet)
print("Building HF TextModel (blank)...", flush=True)
hf_model = HFTextModel(text_cfg_hf).to(DTYPE).eval()
print(f"  {sum(p.numel() for p in hf_model.parameters())/1e9:.1f}B params", flush=True)

# Load safetensors directly, stripping model.language_model. prefix
import os, json
index_path = os.path.join(CKPT, "model.safetensors.index.json")
with open(index_path) as f:
    index = json.load(f)

# Find unique shard files that have language_model keys
lang_prefix = "model.language_model."
shard_files = sorted(set(
    v for k, v in index["weight_map"].items()
    if k.startswith(lang_prefix)
))
print(f"Loading weights from {len(shard_files)} shard(s)...", flush=True)

hf_sd = hf_model.state_dict()
loaded = 0
for shard in shard_files:
    shard_path = os.path.join(CKPT, shard)
    print(f"  Loading {shard}...", flush=True)
    shard_dict = st_load(shard_path, device="cpu")
    for k, v in shard_dict.items():
        if not k.startswith(lang_prefix):
            continue
        short_k = k[len(lang_prefix):]
        if short_k in hf_sd and hf_sd[short_k].shape == v.shape:
            hf_sd[short_k] = v.to(DTYPE)
            loaded += 1
    del shard_dict

hf_model.load_state_dict(hf_sd, strict=False)
hf_model.eval()
print(f"HF model ready — loaded {loaded} tensors", flush=True)

# Build our model
def make_our_cfg():
    moe_layers = list(range(text_cfg_hf.num_hidden_layers)) if text_cfg_hf.enable_moe_block else []
    kv_share_from = None
    if getattr(text_cfg_hf, 'num_kv_shared_layers', 0) > 0:
        kv_share_from = text_cfg_hf.num_hidden_layers - text_cfg_hf.num_kv_shared_layers

    cfg = TextConfig(
        vocab_size=text_cfg_hf.vocab_size,
        hidden_size=text_cfg_hf.hidden_size,
        num_hidden_layers=text_cfg_hf.num_hidden_layers,
        num_attention_heads=text_cfg_hf.num_attention_heads,
        num_key_value_heads=text_cfg_hf.num_key_value_heads,
        head_dim=text_cfg_hf.head_dim,
        global_head_dim=text_cfg_hf.global_head_dim,
        intermediate_size=text_cfg_hf.intermediate_size,
        rms_norm_eps=text_cfg_hf.rms_norm_eps,
        pad_token_id=text_cfg_hf.pad_token_id,
        embed_scale=text_cfg_hf.hidden_size ** 0.5,
        hidden_size_per_layer_input=text_cfg_hf.hidden_size_per_layer_input,
        sliding_window=text_cfg_hf.sliding_window,
        sliding_window_pattern=6,
        kv_share_from=kv_share_from,
        rope_local_base_freq=rope_local,
        rope_global_base_freq=rope_global,
        global_partial_rotary_factor=global_partial,
        num_experts=text_cfg_hf.num_experts or 0,
        num_experts_per_tok=text_cfg_hf.top_k_experts or 0,
        moe_layers=moe_layers,
        expert_intermediate_size=text_cfg_hf.moe_intermediate_size or 0,
        num_global_key_value_heads=getattr(text_cfg_hf, 'num_global_key_value_heads', None),
        attention_k_eq_v=getattr(text_cfg_hf, 'attention_k_eq_v', False),
        use_v_norm=False,  # HF uses RMSNorm(with_scale=False) for v_norm
    )
    cfg._layer_types = text_cfg_hf.layer_types
    return cfg

our_cfg = make_our_cfg()
print("Building our TextModel...", flush=True)
our_model = TextModel(our_cfg).to(DTYPE).eval()
print(f"  {sum(p.numel() for p in our_model.parameters())/1e9:.1f}B params", flush=True)

# Copy weights from hf_model to our_model (direct state dict match)
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
print(f"Copied {n_loaded} tensors, skipped {n_skipped}", flush=True)
if skipped:
    for k, hs, os in skipped[:10]:
        print(f"  SKIPPED {k}: HF={hs} ours={os}", flush=True)

def make_causal_mask(seq_len, dtype=torch.bfloat16):
    """Build a [1, 1, L, L] causal mask matching HF's format."""
    mask = torch.full((1, 1, seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype)
    for i in range(seq_len):
        mask[:, :, i, :i+1] = 0.0
    return mask

def run_comparison(seq_len=16, seed=42):
    """Compare HF and our model outputs with identical causal mask."""
    torch.manual_seed(seed)
    input_ids = torch.randint(2, our_cfg.vocab_size, (1, seq_len))
    causal = make_causal_mask(seq_len)

    with torch.no_grad():
        hf_out = hf_model(input_ids=input_ids)          # HF builds mask internally
        our_out = our_model(input_ids, attention_mask=causal)  # pass explicit mask

    hf_hs = hf_out.last_hidden_state
    our_hs = our_out

    diff = (hf_hs - our_hs).abs().max().item()
    cos = F.cosine_similarity(hf_hs.flatten(), our_hs.flatten(), dim=0).item()
    print(f"  max_diff={diff:.6f}  cos_sim={cos:.8f}")
    return diff, cos

print("\nRunning initial comparison...", flush=True)
run_comparison()

print("\nSession ready. Use run_comparison() to test.", flush=True)
print("  %autoreload 2 is active — edit gemma4_simple.py and call reload_our_model() to pick up changes.", flush=True)

def reload_our_model():
    """Reload gemma4_simple, rebuild our_model, copy weights from hf_model."""
    import importlib
    import gemma4_simple
    importlib.reload(gemma4_simple)
    from gemma4_simple import TextModel, TextConfig
    global our_model, our_cfg
    our_cfg = make_our_cfg()
    our_model = TextModel(our_cfg).to(DTYPE).eval()
    sd_hf  = hf_model.state_dict()
    sd_our = our_model.state_dict()
    n_loaded = n_skipped = 0
    for k, v in sd_hf.items():
        if k in sd_our and sd_our[k].shape == v.shape:
            sd_our[k] = v.clone()
            n_loaded += 1
        else:
            n_skipped += 1
    our_model.load_state_dict(sd_our, strict=False)
    print(f"  Rebuilt our_model: {n_loaded} loaded, {n_skipped} skipped")
    return our_model

try:
    import IPython
    IPython.start_ipython(argv=[], user_ns={**globals(), **locals()},
                          display_banner=False,
                          exec_lines=["%load_ext autoreload", "%autoreload 2"])
except ImportError:
    import code
    code.interact(local={**globals(), **locals()})

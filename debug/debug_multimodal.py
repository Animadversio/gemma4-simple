"""Debug the combined multimodal model."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

CKPT = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/huggingface/hub/gemma-4-E4B-it"
DEVICE = "cuda"
DTYPE = torch.bfloat16

hf_cfg = AutoConfig.from_pretrained(CKPT)
tc = hf_cfg.text_config
vc = hf_cfg.vision_config
tc._attn_implementation = "eager"
vc._attn_implementation = "eager"

from gemma4_simple import Gemma4Config, Gemma4Model, TextConfig, VisionConfig

IMAGE_TOKEN_ID = hf_cfg.image_token_id
KERNEL = vc.pooling_kernel_size
N_PATCHES = KERNEL * KERNEL
patch_dim = vc.patch_size * vc.patch_size * 3

print(f"IMAGE_TOKEN_ID={IMAGE_TOKEN_ID}, KERNEL={KERNEL}, N_PATCHES={N_PATCHES}")
print(f"patch_dim={patch_dim}, vc.hidden_size={vc.hidden_size}")

# Test input
ids = torch.tensor([[1, 100, IMAGE_TOKEN_ID, 200, 201]], dtype=torch.long, device=DEVICE)
rows = torch.arange(KERNEL, device=DEVICE).repeat_interleave(KERNEL)
cols = torch.arange(KERNEL, device=DEVICE).repeat(KERNEL)
ppos = torch.stack([rows, cols], dim=-1).unsqueeze(0)
pval = torch.rand(1, N_PATCHES, patch_dim, dtype=DTYPE, device=DEVICE)

print(f"ids shape: {ids.shape}, pval shape: {pval.shape}, ppos shape: {ppos.shape}")

# Load weights
print("Loading weights...")
with safe_open(f"{CKPT}/model.safetensors", framework="pt", device="cpu") as f:
    all_keys = list(f.keys())
    all_wts = {k: f.get_tensor(k).to(DTYPE) for k in all_keys
               if any(p in k for p in ("language_model", "vision_tower", "embed_vision"))}

print(f"Loaded {len(all_wts)} weight tensors")
# Print some key names
embed_vis_keys = [k for k in all_wts if "embed_vision" in k]
print(f"embed_vision keys: {embed_vis_keys}")

# Check our Gemma4Model state dict keys
our_text_cfg = TextConfig(
    vocab_size=tc.vocab_size, hidden_size=tc.hidden_size,
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
)
our_text_cfg._layer_types = tc.layer_types

our_vis_cfg = VisionConfig(
    hidden_size=vc.hidden_size,
    num_hidden_layers=vc.num_hidden_layers,
    num_attention_heads=vc.num_attention_heads,
    head_dim=vc.head_dim,
    intermediate_size=vc.intermediate_size,
    patch_size=vc.patch_size,
    position_embedding_size=vc.position_embedding_size,
    pooling_kernel_size=vc.pooling_kernel_size,
    rms_norm_eps=vc.rms_norm_eps,
    rope_theta=vc.rope_parameters["rope_theta"],
    standardize=vc.standardize,
    use_clipped_linears=vc.use_clipped_linears,
)

our_cfg = Gemma4Config(text=our_text_cfg, vision=our_vis_cfg, image_token_id=IMAGE_TOKEN_ID)
our_model = Gemma4Model(our_cfg)

# Show our model's expected keys for mm_embedder
print("\nOur mm_embedder keys:")
for name, p in our_model.mm_embedder.named_parameters():
    print(f"  mm_embedder.{name}: {p.shape}")

# Remap
our_state = {}
for k, v in all_wts.items():
    k2 = k.replace("model.language_model.", "language_model.")
    k2 = k2.replace("model.vision_tower.", "vision_model.")
    k2 = k2.replace("model.embed_vision.embedding_projection.", "mm_embedder.proj.")
    our_state[k2] = v

# Check key overlap
our_model_keys = set(n for n, _ in our_model.named_parameters())
our_state_keys = set(our_state.keys())
missing = our_model_keys - our_state_keys
unexpected = our_state_keys - our_model_keys
print(f"\nOur model keys: {len(our_model_keys)}")
print(f"Our state keys after remap: {len(our_state_keys)}")
print(f"Missing from state (not loaded): {len(missing)}, first 5: {list(missing)[:5]}")
print(f"Unexpected in state (ignored): {len(unexpected)}, first 5: {list(unexpected)[:5]}")

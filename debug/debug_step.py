"""Step-by-step comparison: find the first point where HF and ours diverge."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch, torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

CKPT = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/huggingface/hub/gemma-4-E4B-it"
DEVICE = "cuda"; DTYPE = torch.bfloat16

hf_cfg = AutoConfig.from_pretrained(CKPT)
tc = hf_cfg.get_text_config()
tc._attn_implementation = "eager"
tok = AutoTokenizer.from_pretrained(CKPT)

from gemma4_simple import TextConfig, TextModel
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel as HFTextModel

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
    use_v_norm=False,
    pad_token_id=tc.pad_token_id,
)
our_text_cfg._layer_types = tc.layer_types

print("Loading weights…")
with safe_open(f"{CKPT}/model.safetensors", framework="pt", device="cpu") as f:
    all_wts = {k: f.get_tensor(k).to(DTYPE) for k in f.keys() if "language_model" in k}
state = {k.replace("model.language_model.", ""): v for k, v in all_wts.items()}

ids = tok("Hello, world!", return_tensors="pt").input_ids.to(DEVICE)
print(f"ids: {ids.tolist()}")

# ── HF model ──
hf_model = HFTextModel(tc).to(DTYPE).to(DEVICE)
m, u = hf_model.load_state_dict({k: v.to(DEVICE) for k, v in state.items()}, strict=False)
print(f"HF: missing={len(m)} unexpected={len(u)}")
if m: print(f"  HF missing: {m[:3]}")
if u: print(f"  HF unexpected: {u[:3]}")
hf_model.eval()

# ── Our model ──
our_model = TextModel(our_text_cfg).to(DTYPE).to(DEVICE)
m2, u2 = our_model.load_state_dict({k: v.to(DEVICE) for k, v in state.items()}, strict=False)
print(f"Ours: missing={len(m2)} unexpected={len(u2)}")
if m2: print(f"  Ours missing: {m2[:5]}")
if u2: print(f"  Ours unexpected: {u2[:5]}")
our_model.eval()

def cmp(name, a, b):
    diff = (a.float() - b.float()).abs().max().item()
    cs = F.cosine_similarity(a.float().reshape(1,-1), b.float().reshape(1,-1)).item()
    print(f"  {name}: max_diff={diff:.4f}  cos_sim={cs:.6f}")
    return diff

with torch.no_grad():
    # 1. Embedding
    hf_emb = hf_model.embed_tokens(ids)
    our_emb = our_model.embed_tokens(ids)
    cmp("embed_tokens", hf_emb, our_emb)

    # 2. Per-layer inputs
    hf_pli_raw = hf_model.get_per_layer_inputs(ids, None)
    hf_pli = hf_model.project_per_layer_inputs(hf_emb, hf_pli_raw)
    our_pli = our_model._compute_per_layer_inputs(ids, our_emb)
    cmp("per_layer_inputs", hf_pli, our_pli)
    cmp("per_layer_inputs[0]", hf_pli[:,:,0,:], our_pli[:,:,0,:])

    # 3. Run both models fully
    hf_out = hf_model(input_ids=ids, use_cache=False).last_hidden_state
    our_out = our_model(ids, attention_mask=None)
    cmp("FINAL (both compute pli internally)", hf_out.cpu(), our_out.cpu())

    # 4. Run ours with HF's pli
    our_out2 = our_model(ids, attention_mask=None, per_layer_inputs=hf_pli.cpu().to(DEVICE))
    cmp("FINAL (ours uses HF pli)", hf_out.cpu(), our_out2.cpu())

    # Check weight identity for a few critical weights
    print("\nWeight identity checks:")
    for k in ["layers.0.self_attn.q_proj.weight", "layers.5.self_attn.q_proj.weight",
              "embed_tokens.weight", "embed_tokens_per_layer.weight",
              "per_layer_model_projection.weight"]:
        hf_w = dict(hf_model.named_parameters()).get(k) or dict(hf_model.named_buffers()).get(k)
        our_w = dict(our_model.named_parameters()).get(k) or dict(our_model.named_buffers()).get(k)
        if hf_w is not None and our_w is not None:
            diff = (hf_w.float() - our_w.float()).abs().max().item()
            print(f"  {k}: diff={diff:.6f}")
        else:
            print(f"  {k}: HF={hf_w is not None} ours={our_w is not None}")

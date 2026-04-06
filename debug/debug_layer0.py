"""
Debug layer 0 sub-operations in detail.
Run with: exec(open('/n/home12/binxuwang/Github/gemma4-simple/debug_layer0.py').read(), globals())
"""
import torch
import torch.nn.functional as F
from transformers.models.gemma4.modeling_gemma4 import (
    create_causal_mask, create_sliding_window_causal_mask
)
from transformers.cache_utils import DynamicCache

torch.manual_seed(42)
seq_len = 8
ids = torch.randint(2, our_cfg.vocab_size, (1, seq_len))
pos_ids = torch.arange(seq_len).unsqueeze(0)

pv = DynamicCache(config=text_cfg_hf)
inputs_embeds = hf_model.embed_tokens(ids)
mask_kwargs = dict(config=text_cfg_hf, inputs_embeds=inputs_embeds,
    attention_mask=None, past_key_values=pv, position_ids=pos_ids)
mask_sliding = create_sliding_window_causal_mask(**mask_kwargs)

hf_sd = hf_model.state_dict()
our_sd = our_model.state_dict()
our_only = sorted([k for k in our_sd if k not in hf_sd and 'layers.0.' in k])
hf_only  = sorted([k for k in hf_sd  if k not in our_sd and 'layers.0.' in k])
print(f"Layer 0 keys only in OURS: {our_only}")
print(f"Layer 0 keys only in HF:   {hf_only}")
print()

lt0 = our_cfg._layer_types[0]
with torch.no_grad():
    hf_h = hf_model.embed_tokens(ids).clone()
    our_h = our_model.embed_tokens(ids).clone()
    print(f"Embed diff: {(hf_h - our_h).abs().max():.6f}")

    hf_cos, hf_sin = hf_model.rotary_emb(hf_h, pos_ids, layer_type=lt0)
    our_cos, our_sin = our_model.rotary_emb(our_h, pos_ids, layer_type="local")
    print(f"RoPE cos diff: {(hf_cos - our_cos).abs().max():.6f}")

    hf_ln1  = hf_model.layers[0].input_layernorm(hf_h)
    our_ln1 = our_model.layers[0].input_layernorm(our_h)
    print(f"input_layernorm diff: {(hf_ln1 - our_ln1).abs().max():.6f}")

    hf_attn_out, _ = hf_model.layers[0].self_attn(
        hf_ln1, position_embeddings=(hf_cos, hf_sin), attention_mask=mask_sliding
    )
    our_attn_out = our_model.layers[0].self_attn(our_ln1, our_cos, our_sin, mask_sliding)
    print(f"attn output diff:     {(hf_attn_out - our_attn_out).abs().max():.6f}")

    hf_pan  = hf_model.layers[0].post_attention_layernorm(hf_attn_out)
    our_pan = our_model.layers[0].post_attention_layernorm(our_attn_out)
    hf_h1   = hf_h + hf_pan
    our_h1  = our_h + our_pan
    print(f"post_attn_norm diff:  {(hf_pan - our_pan).abs().max():.6f}")
    print(f"after attn residual:  {(hf_h1 - our_h1).abs().max():.6f}")
    print()

    hf_preffn  = hf_model.layers[0].pre_feedforward_layernorm(hf_h1)
    our_preffn = our_model.layers[0].pre_feedforward_layernorm(our_h1)
    print(f"pre_feedforward_layernorm diff: {(hf_preffn - our_preffn).abs().max():.6f}")

    hf_mlp  = hf_model.layers[0].mlp(hf_preffn)
    our_mlp = our_model.layers[0].mlp(our_preffn)
    print(f"mlp output diff:      {(hf_mlp - our_mlp).abs().max():.6f}")

    hf_l0 = hf_model.layers[0]
    our_l0 = our_model.layers[0]
    print(f"HF enable_moe_block: {hf_l0.enable_moe_block},  Ours enable_moe: {our_l0.enable_moe}")

    hf_h_mlp  = hf_l0.post_feedforward_layernorm_1(hf_mlp)
    our_h_mlp = our_l0.post_feedforward_layernorm_1(our_mlp)
    print(f"post_ffn_ln1 diff:    {(hf_h_mlp - our_h_mlp).abs().max():.6f}")

    hf_flat  = hf_h1.reshape(-1, hf_h1.shape[-1])
    our_flat = our_h1.reshape(-1, our_h1.shape[-1])
    hf_probs, hf_tw, hf_ti = hf_l0.router(hf_flat)
    our_probs, our_tw, our_ti = our_l0.router(our_flat)
    print(f"router probs diff:    {(hf_probs - our_probs).abs().max():.6f}")
    print(f"router top_w diff:    {(hf_tw - our_tw).abs().max():.6f}")
    print(f"router top_idx match: {(hf_ti == our_ti).all().item()}")

    hf_h2_pre  = hf_l0.pre_feedforward_layernorm_2(hf_flat)
    our_h2_pre = our_l0.pre_feedforward_layernorm_2(our_flat)
    print(f"pre_ffn_ln2 diff:     {(hf_h2_pre - our_h2_pre).abs().max():.6f}")

    hf_exp  = hf_l0.experts(hf_h2_pre, hf_ti, hf_tw)
    our_exp = our_l0.experts(our_h2_pre, our_ti, our_tw)
    print(f"experts diff (own inputs): {(hf_exp - our_exp).abs().max():.6f}")
    our_exp2 = our_l0.experts(hf_h2_pre, hf_ti, hf_tw)
    print(f"experts diff (HF inputs):  {(hf_exp - our_exp2).abs().max():.6f}")

    hf_h2  = hf_l0.post_feedforward_layernorm_2(hf_exp.reshape(hf_h1.shape))
    our_h2 = our_l0.post_feedforward_layernorm_2(our_exp.reshape(our_h1.shape))
    print(f"post_ffn_ln2 diff:    {(hf_h2 - our_h2).abs().max():.6f}")

    hf_combined  = hf_h_mlp + hf_h2
    our_combined = our_h_mlp + our_h2
    print(f"combined diff:        {(hf_combined - our_combined).abs().max():.6f}")

    hf_postffn  = hf_l0.post_feedforward_layernorm(hf_combined)
    our_postffn = our_l0.post_feedforward_layernorm(our_combined)
    print(f"post_ffn_ln diff:     {(hf_postffn - our_postffn).abs().max():.6f}")

    hf_out  = hf_h1 + hf_postffn
    our_out = our_h1 + our_postffn
    print(f"after FFN residual:   {(hf_out - our_out).abs().max():.6f}")

    hf_final  = hf_out  * hf_l0.layer_scalar
    our_final = our_out * our_l0.layer_scalar
    print(f"after layer_scalar:   {(hf_final - our_final).abs().max():.6f}")
    print(f"layer_scalar HF={hf_l0.layer_scalar.item():.6f}, ours={our_l0.layer_scalar.item():.6f}")

    hf_l0_full  = hf_l0(hf_h, position_embeddings=(hf_cos, hf_sin), attention_mask=mask_sliding)
    our_l0_full = our_l0(our_h, cos=our_cos, sin=our_sin, attention_mask=mask_sliding)
    print(f"\nFull layer diff: {(hf_l0_full - our_l0_full).abs().max():.6f}")

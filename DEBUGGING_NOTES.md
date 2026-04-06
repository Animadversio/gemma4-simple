# Gemma 4 26B Implementation Debugging Notes

**Date:** 2026-04-06
**Session summary:** Systematic layer-by-layer investigation of numerical divergence between
`gemma4_simple.TextModel` and HuggingFace `Gemma4TextModel` on the 26B checkpoint.

---

## Starting point

Initial comparison with the 26B model (A4B-it variant, bfloat16, CUDA) showed:

```
max_diff = 25.750000   cos_sim = 0.960938
```

Goal: determine whether this reflects a real implementation bug or acceptable numerical noise.

---

## Investigation methodology

All debugging was done interactively in an IPython session via `start_session.py`,
with both models loaded side-by-side on a single A100 GPU.

### Phase 1 — Component tests (L=1, isolated)

Tested each sub-component with the same random input vector fed to both HF and our
implementation.  All confirmed bit-exact (diff = 0) at L=1:

| Component | diff |
|---|---|
| `embed_tokens` | 0.0 |
| `input_layernorm` | 0.0 |
| `q_proj / k_proj / v_proj` | 0.0 |
| `q_norm / k_norm` | 0.0 |
| `RoPE (local & global)` | 0.0 |
| `TextMLP` | 0.0 |
| `TextRouter` (probs, top-k weights, indices) | 0.0 |
| `pre_feedforward_layernorm_2` | 0.0 |

### Phase 2 — Layer-by-layer trace (L=8, real token sequence)

Fed the same embeddings to both models and propagated through each layer in sequence.
Errors accumulate:

```
Embed diff: 0.000000
L  0 (slid): max_diff=0.3750  cos=1.000000
L  1 (slid): max_diff=0.4980  cos=0.996094
...
L 11 (full): max_diff=3.9375  cos=0.996094
L 12 (slid): max_diff=8.0000  cos=0.988281  ← large jump
...
After final norm: diff=33.0000
```

The "jump" at L=12 is a compounding effect: once accumulated error exceeds ~4 units, each
subsequent layer amplifies it further.

### Phase 3 — Per-layer isolated test (same input, independent layers)

Fed the **same fresh input** independently to each layer (no accumulation).  All layers
show small per-layer diff (0.0625–0.375), with `rope_diff = 0.000` throughout.
This ruled out RoPE, attention mask, and weight-loading bugs.

### Phase 4 — Sub-component drill-down (L=1, full layer)

Within a single decoder layer (L=1, same input for both models):

```
Attention output diff:                  0.00000000   ✓ attention is clean
pre_feedforward_layernorm_2 diff:       0.00000000   ✓ norm is clean
Expert output diff (bfloat16):          0.06250000   ← HERE
Expert output diff (float32):           0.00000000   ← float32 is exact!
post_feedforward_layernorm_2 diff:      2.00000000   ← amplified ~32× by RMSNorm
Full layer diff (L=1):                  0.18750000
```

**Key observation:** In float32 the expert output is bit-identical to HF.
The bfloat16 discrepancy must come from a different accumulation path.

---

## Root cause

```python
print(getattr(text_cfg_hf, '_experts_implementation', 'NOT SET'))
# → 'grouped_mm'
```

HuggingFace defaults to `config._experts_implementation = 'grouped_mm'`, which uses a
fused batched GEMM kernel (`torch.ops.fbgemm.gmm` or equivalent) to process all expert
tokens in a single kernel call.

Our `TextExperts.forward` uses a Python for-loop with sequential `F.linear` calls — the
same code path as HF's `'eager'` implementation.

The two paths compute the same mathematical operation but with different bfloat16
accumulation order, producing a ~0.0625 per-layer discrepancy.  This small difference
is then **amplified ~32× by `post_feedforward_layernorm_2`** (RMSNorm is sensitive to
small absolute errors when the input magnitude is small), and the amplified error cascades
across 30 MoE layers.

### Confirmation

```python
text_cfg_hf._experts_implementation = 'eager'
run_comparison()
# → max_diff=0.000000  cos_sim=1.00000000
```

**Perfect match.** Our implementation is mathematically correct.

---

## Error amplification mechanism

The `post_feedforward_layernorm_2` RMSNorm amplifies the 0.0625 diff to ~2.0:

- Expert output magnitude (RMS): ~1.2
- Relative error: 0.0625 / 1.2 ≈ 5.2%
- After normalization the weight vector scales this up, and numerical conditioning
  of the norm (dividing by RMS) turns small absolute differences into larger ones
  when the overall magnitude is moderate but uneven across channels.

The amplification accumulates across all 30 layers because each layer's output is the
input to the next layer's norm.

---

## Summary

| Condition | max_diff | cos_sim | Notes |
|---|---|---|---|
| HF `grouped_mm` vs our eager (bfloat16) | 25.750 | 0.9609 | Default HF config |
| HF `eager` vs our eager (bfloat16) | 0.000 | 1.0000 | Bit-exact match |
| Expert output only (float32) | 0.000 | 1.0000 | Same math, same result |
| Expert output only (bfloat16) | 0.063 | ~1.000 | Accumulation order differs |

**Conclusion:** No implementation bugs. The divergence is purely a numerical precision
artifact of `grouped_mm` vs sequential-loop bfloat16 accumulation.

---

## Fixes applied

1. `compare_gemma4_moe.py`: added `_experts_implementation='eager'` to `make_hf_cfg()`
   and `test_text_tower_26b()` for fair comparison.
2. `gemma4_simple.py` (`TextExperts` docstring): documented the numerical behavior.

---

## Debug scripts

| Script | Purpose |
|---|---|
| `start_session.py` | Interactive IPython session with both models loaded side-by-side |
| `debug_layers.py` | Layer-by-layer trace comparing HF vs ours on a real token sequence |
| `debug_layer0.py` | Focused component tests on a single decoder layer |
| `debug_system.py` | Full-model system test (logits, top-k tokens) |
| `debug_step.py` | Step-by-step forward pass with intermediate tensor comparisons |
| `debug_embed.py` | Embedding and per-layer-input debugging |
| `debug_multimodal.py` | Multimodal (vision+text) forward pass debugging |

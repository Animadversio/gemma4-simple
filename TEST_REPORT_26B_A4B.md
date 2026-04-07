# Test Report — Gemma 4 26B-A4B (gemma4_simple vs HuggingFace)

**Date:** 2026-04-06
**Model:** `gemma-4-26B-A4B-it` (26B total / ~4B active parameters, Mixture-of-Experts)
**Checkpoint:** `/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/huggingface/hub/gemma-4-26B-A4B-it`
**Hardware:** A100 40 GB, CUDA, bfloat16
**Test script:** `tests/compare_gemma4_moe.py`

---

## Model Architecture

### Text Tower
| Parameter | Value |
|---|---|
| `hidden_size` | 2816 |
| `num_hidden_layers` | 30 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 8 |
| `head_dim` | 256 |
| `global_head_dim` | 512 |
| `intermediate_size` | 2112 |
| `moe_intermediate_size` | 704 |
| `vocab_size` | 262144 |
| `sliding_window` | 1024 |
| `num_experts` | 128 |
| `top_k_experts` | 8 |
| `enable_moe_block` | True (all 30 layers) |
| `layer_types` | 25 × sliding_attention + 5 × full_attention (every 6th) |
| `num_kv_shared_layers` | 0 |
| `rope_local_base_freq` | 10,000 |
| `rope_global_base_freq` | 1,000,000 |
| `global_partial_rotary_factor` | 0.25 |
| `hidden_size_per_layer_input` | 0 (no per-layer gate) |

### Vision Tower
| Parameter | Value |
|---|---|
| `hidden_size` | 1152 |
| `num_hidden_layers` | 27 |
| `num_attention_heads` | 16 |
| `head_dim` | 72 |
| `intermediate_size` | 4304 |
| `patch_size` | 16 |
| `pooling_kernel_size` | 3 |

---

## Key Finding: MoE Expert Implementation Precision

The 26B model introduces a **MoE (Mixture-of-Experts) block** absent from E4B. HuggingFace offers
two expert implementations controlled by `config._experts_implementation`:

| Setting | Implementation | Accumulation | Result |
|---|---|---|---|
| `"grouped_mm"` (default) | batched GEMM via `grouped_mm` kernel | reshape + sum | max_diff=25.75, cos_sim=0.9609 |
| `"eager"` | Python loop with `index_add_` | sequential per token | max_diff=0.00, cos_sim=1.0000 |

Our `TextExperts` uses `index_add_` accumulation (matching HF eager). Setting
`tc._experts_implementation = "eager"` gives **bit-exact agreement**.

The gap is purely numerical (different floating-point accumulation order in bfloat16),
not a correctness bug. Both are mathematically equivalent.

---

## Test Results

**System-level: Full TextModel forward pass**

| Config | max_diff | cos_sim | Status |
|---|---|---|---|
| HF default (`grouped_mm`) vs ours | 25.750000 | 0.96093750 | ❌ numerical gap |
| HF `eager` vs ours | 0.000000 | 1.00000000 | ✅ bit-exact |

**Expert-level analysis (single MoE layer, layer 0)**

| Component | diff (bf16) | diff (f32) | Notes |
|---|---|---|---|
| `gate_up_proj` weights | 0.0 | 0.0 | Weights identical |
| `down_proj` weights | 0.0 | 0.0 | Weights identical |
| `per_expert_scale` | 0.0 | 0.0 | Routing scales identical |
| `top_k_indices` | exact match | — | Same experts selected |
| `top_k_weights` | 0.0 | 0.0 | Same routing weights |
| Expert output (single layer) | 0.0625 | 0.0 | BF16 accumulation order only |
| Post-FF norm (bf16 in, bf16 out) | 2.000000 | — | Propagated from expert diff |
| Post-FF norm (f32 expert → bf16 norm) | 0.0 | — | Exact when accumulation is f32 |

The root cause: HF `grouped_mm` accumulates contributions across expert slots via
`reshape(...).sum()`, while our code (and HF eager) uses `index_add_` which accumulates
one token at a time in a fixed order. In bfloat16 these give different rounding, causing
~0.0625 diff per expert layer that grows to ~25 over 30 layers.

---

## Notes

- Module-level tests (RMSNorm, TextMLP, TextAttention, TextDecoderLayer, etc.) were
  verified on E4B (see `TEST_REPORT_E4B.md`); the 26B model shares the same module
  implementations. MoE-specific testing was done at the system level.
- The 26B model has **no per-layer gate** (`hidden_size_per_layer_input=0`), unlike E4B.
- Vision tower comparison was not run separately for 26B; the vision architecture
  differs only in scale (27 layers, hidden_size=1152) and uses the same code path
  verified for E4B.
- Tests were run in an interactive IPython session (`tests/start_session.py`) rather
  than the automated compare script, due to the model's size (~52 GB for both models
  together in CUDA bfloat16).

---

## Reproduction

```bash
CKPT="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/huggingface/hub/gemma-4-26B-A4B-it"

# Interactive session (loads both models, lets you call run_comparison())
python tests/start_session.py --ckpt "$CKPT" --device cuda

# Inside the session — compare with HF eager (bit-exact):
# tc._experts_implementation = "eager"
# run_comparison()
```

The start script now sets `tc._experts_implementation = "eager"` automatically,
so `run_comparison()` gives max_diff=0.0 out of the box.

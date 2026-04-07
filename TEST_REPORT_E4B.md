# Test Report — Gemma 4 E4B (gemma4_simple vs HuggingFace)

**Date:** 2026-04-06
**Model:** `gemma-4-E4B-it` (4B active / ~8.5B total parameters)
**Checkpoint:** `/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/huggingface/hub/gemma-4-E4B-it`
**Hardware:** A100 40 GB, CUDA, bfloat16
**Test script:** `tests/compare_gemma4.py`

---

## Model Architecture

### Text Tower
| Parameter | Value |
|---|---|
| `hidden_size` | 2560 |
| `num_hidden_layers` | 42 |
| `num_attention_heads` | 8 |
| `num_key_value_heads` | 2 |
| `head_dim` | 256 |
| `intermediate_size` | 10240 |
| `vocab_size` | 262144 |
| `sliding_window` | 512 |
| `num_experts` | None (dense MLP, no MoE) |

### Vision Tower
| Parameter | Value |
|---|---|
| `hidden_size` | 768 |
| `num_hidden_layers` | 16 |
| `num_attention_heads` | 12 |
| `head_dim` | 64 |
| `intermediate_size` | 3072 |
| `patch_size` | 16 |
| `pooling_kernel_size` | 3 |
| `position_embedding_size` | 10240 |

---

## Test Results

**Overall: 10/10 PASS** ✅

| # | Test | Module | max_diff | cos_sim | top1 | Status |
|---|---|---|---|---|---|---|
| 1 | `rmsnorm` | RMSNorm | 0.00e+00 | 1.000003 | 100% | ✅ |
| 2 | `mlp` | TextMLP | 0.00e+00 | 1.000004 | 100% | ✅ |
| 3 | `attn` | TextAttention (layer 0) | 0.00e+00 | 1.000005 | 100% | ✅ |
| 4 | `decoder` | TextDecoderLayer (layer 0) | 0.00e+00 | 1.000001 | 100% | ✅ |
| 5 | `emb` | Embedding + scale | 0.00e+00 | 1.000003 | 100% | ✅ |
| 6 | `vis` | VisionEncoderLayer (layer 0) | 0.00e+00 | 1.000000 | 100% | ✅ |
| 7 | `pooler` | VisionPooler | 0.00e+00 | 1.000000 | 100% | ✅ |
| 8 | `vis_full` | Full VisionModel (all 16 layers) | 0.00e+00 | 1.000000 | 100% | ✅ |
| 9 | `system` | Full TextModel (all 42 layers) | 0.00e+00 | 1.000002 | 100% | ✅ |
| 10 | `multimodal` | Full Gemma4Model (text + vision) | 0.00e+00 | 1.000002 | 100% | ✅ |

All outputs are **bit-exact** (max_diff = 0 in bfloat16) vs HuggingFace on every module.

---

## Test Descriptions

### Unit tests (1–8)

Each test loads only the weights for that module from the safetensors file (low memory),
constructs both the HF and our module with identical weights, feeds the same random
input, and compares outputs.

- **rmsnorm**: `RMSNorm` vs `Gemma4RMSNorm` — forward with learnable scale.
- **mlp**: `TextMLP` vs `Gemma4TextMLP` — SwiGLU FFN (gate × up → down).
- **attn**: `TextAttention` vs `Gemma4TextAttention` — GQA with dual-freq RoPE (layer 0 = sliding).
- **decoder**: `TextDecoderLayer` vs `Gemma4TextDecoderLayer` — full pre+post norm block.
- **emb**: `ScaledEmbedding` vs `Gemma4TextScaledWordEmbedding` — token embed × √D.
- **vis**: `VisionEncoderLayer` vs `Gemma4VisionEncoderLayer` — ViT layer with 2D RoPE.
- **pooler**: `VisionPooler` vs `Gemma4VisionPooler` — average pooling with stride.
- **vis_full**: Full `VisionModel` (all 16 ViT layers + pooler) end-to-end.

### System tests (9–10)

- **system**: Full `TextModel` (42 layers, causal attention) on a real tokenized sequence.
  Loads all text tower weights, verifies final hidden states match HF exactly.
- **multimodal**: Full `Gemma4Model` (vision encoder + text decoder) on a synthetic image
  (random pixel values) + text token sequence. Verifies last hidden state of the
  combined multimodal forward pass matches HF.

---

## Numerical notes

- All comparisons in **bfloat16** on CUDA — no float32 upcast needed for E4B.
- The E4B model is **dense** (no MoE experts), so the `grouped_mm` vs `eager` numerical
  issue does not apply here. Results are bit-exact with default HF settings.
- For the 26B A4B model (MoE), see `DEBUGGING_NOTES.md` for details on the expert
  implementation precision difference.

---

## Reproduction

```bash
CKPT="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/huggingface/hub/gemma-4-E4B-it"
python3 tests/compare_gemma4.py --ckpt "$CKPT" --device cuda
```

To run individual tests:
```bash
python3 tests/compare_gemma4.py --ckpt "$CKPT" --tests system multimodal
```

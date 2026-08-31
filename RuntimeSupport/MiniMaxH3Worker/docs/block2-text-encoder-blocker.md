# Block 2 — Qwen3-VL text encoder

Status: **implemented and running with the real MiniMax H3 checkpoint**.

The encoder uses the text-only portion of
`text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf`. It reads the tokenizer from
the matching FL2VA processor directory and does not materialize the `visual.*`
tensors.

## Checkpoint and architecture

- GGUF v3, 902 source tensors: 551 `model.*` and 351 `visual.*`.
- Source quantization: Q4_K 390, Q5_K 27, Q6_K 50, F32 433, F16 2.
- 50 retained language layers; the upstream config describes 64, but this
  checkpoint contains layers 0 through 49 only.
- Hidden size 5120, vocabulary size 151936, 64 query heads, 8 KV heads,
  head dimension 128, FFN size 25600.
- RoPE theta is 5,000,000 and RMSNorm epsilon is 1e-6.
- The output is the raw hidden state after layer 50; no final RMSNorm or
  language-model head is applied because the checkpoint contains neither.
- Attention is causal, uses GQA without manually tiling K/V, and uses the
  split-half RoPE convention.

## Loading policy

Production loading decodes the GGUF K-quant tensors once from a single mapped
file and re-quantizes the 350 linear modules to MLX affine INT8 with group
size 64. Dense tensors, including the token embedding and 200 normalization
tensors, remain FP32. The visual tower is not loaded by the text-only path.

The `text-check --dense --layers N` mode is diagnostic-only. It loads layers
0 through `N - 1` as dense FP32 to separate model math from production INT8
quantization error. `--token-ids` allows parity to use exactly the same input
IDs as the independent Python reference.

## Real-weight verification

The matching tokenizer at
`/Volumes/extSSD/AI_Models/Normal/minimax-h3-gguf-fl2va-q4-0/upstream/FL2VA/processor`
produces valid Chinese and English IDs with vocabulary size 151936. A full
production run loaded 551 language-model tensors, including 350 INT8 linear
modules, and returned `[14, 5120]` hidden states with real weights.

The independent reference is
`scripts/minimax-h3-parity/qwen_text_ref.py`. It uses the installed Python
`gguf` package for K-quant decoding and PyTorch for the attention math. With
the same 14 token IDs and real checkpoint, dense Swift parity was:

| Layers | Output | Relative max diff | Threshold | Result |
|---:|---|---:|---:|---|
| 2 | `[14, 5120]` | `1.107613e-05` | `1e-4` | PASS |
| 5 | `[14, 5120]` | `4.767765e-06` | `1e-4` | PASS |
| 10 | `[14, 5120]` | `3.312539e-07` | `1e-4` | PASS |

These are real-weight dense parity checks, not random-weight tests. A dense
50-layer reference is not practical on this 64GB machine because it would
materialize the full 32B checkpoint in FP32; the production 50-layer Swift
path has been run separately with real INT8 weights.

## Commands

Build and test:

```sh
swift build --package-path RuntimeSupport/MiniMaxH3Worker -c release
swift test --package-path RuntimeSupport/MiniMaxH3Worker
```

Full text encoder check:

```sh
RuntimeSupport/MiniMaxH3Worker/.build/out/Products/Release/GenImageMiniMaxH3Worker text-check \
  --text-encoder /Volumes/extSSD/AI_Models/Normal/minimax-h3-gguf-fl2va-q4-0/text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  --tokenizer /Volumes/extSSD/AI_Models/Normal/minimax-h3-gguf-fl2va-q4-0/upstream/FL2VA/processor \
  --prompt "一隻狐狸在雪地裡奔跑，a cinematic blue hour."
```

Dense parity check:

```sh
ids=14777,110048,114647,18493,100167,29490,100501,111484,3837,64,64665,6303,6460,13
RuntimeSupport/MiniMaxH3Worker/.build/out/Products/Release/GenImageMiniMaxH3Worker text-check \
  --text-encoder /path/to/qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  --tokenizer /path/to/FL2VA/processor --token-ids "$ids" \
  --layers 2 --dense --output /tmp/h3-qwen-swift.safetensors
python3 scripts/minimax-h3-parity/qwen_text_ref.py \
  /path/to/qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  --ids "$ids" --layers 2 --out /tmp/h3-qwen-reference.npy
python3 scripts/minimax-h3-parity/compare.py \
  /tmp/h3-qwen-reference.npy /tmp/h3-qwen-swift.safetensors hidden
```

## Known scope

This block implements the pure text-to-video/audio conditioning path. Vision
token expansion, image/video preprocessing, and modality tags for FL2VA image
anchors remain separate conditioning work. The pipeline stages the text
encoder before loading the H3 transformer so the two large models do not need
to remain resident together.

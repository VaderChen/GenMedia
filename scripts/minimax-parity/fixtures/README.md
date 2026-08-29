# MiniMax Music 3 多 chunk parity fixtures

這兩個 `.safetensors` 是由真實 MiniMax Music 3 管線產生的固定 fixture，不是
獨立隨機合成的 latent。產生流程為：

`AR frame hiddens → condition encoder → denoise chunks → vocoder → crop/concat`

模型來源為 `minimax-music3-mlx-4bit`，seed 為 `7`，denoise steps 為 `1`。
每個 fixture 都保存 `latent_chunk_N`、Python 最終 `audio`，以及 metadata 中的
`chunk_diagnostics`（crop、raw end 與保留 samples）。

| Fixture | 真實 frames | chunks | 最終 audio shape | BF16 SNR |
|---|---:|---:|---|---:|
| `music3-2chunks.safetensors` | 210 | 2 | `[1, 2, 370176]` | `36.03255579 dB` |
| `music3-3chunks.safetensors` | 310 | 3 | `[1, 2, 546816]` | `37.07690447 dB` |

兩個 fixture 均固定保存為 BF16 計算路徑；最終 `audio` 張量仍是 FP32。
`CHUNK_FRAMES = 200` 作用於 autoregressive `frame_hiddens`，不是 latent
chunk 長度：完整 chunk 為 689 latent frames，最後的部分 chunk 為 378 latent
frames。

使用 Swift PoC 產生 actual 後比較：

```bash
RuntimeSupport/MiniMaxMusic3Worker/.build/arm64-apple-macosx/release/GenImageMiniMaxMusic3PoC \
  decode-chunks \
  --model-dir <model-dir> \
  --input scripts/minimax-parity/fixtures/music3-2chunks.safetensors \
  --output /tmp/music3-2chunks-swift.safetensors

RuntimeSupport/MiniMaxMusic3Worker/.build/arm64-apple-macosx/release/GenImageMiniMaxMusic3PoC \
  compare \
  --reference scripts/minimax-parity/fixtures/music3-2chunks.safetensors \
  --actual /tmp/music3-2chunks-swift.safetensors \
  --key audio
```

最終音訊以 SNR dB 為主要 parity 指標；`relative max diff` 與 `mean abs diff`
僅作輔助資訊。逐點 deterministic tensor 則維持依 dtype 判定的 relative max
diff 門檻。

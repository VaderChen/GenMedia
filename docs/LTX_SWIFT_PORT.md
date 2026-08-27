# LTX-2 純 Swift 移植紀錄

## 範圍

本階段只完成 Block 0 比對基礎設施與 Block 1 影片 VAE 解碼。現有
`Sources/GenImageRuntime/LTXVideoGenerationService.swift` 與 Python 生成路徑保持不變，
尚未把 Swift PoC 接進 App。

## 參考版本與模型

- Python 參考：`dgrauet/ltx-2-mlx` `v0.14.22`
  (`64b0a1d2ba358232bb73c6730095593e6e9f3ffb`)。
- 模型：`dgrauet/ltx-2.3-mlx-q4`
  (`56a5866d638ecfe37c54d348e88938235185c2d4`)。
- `vae_decoder.safetensors` SHA-256：
  `f404026fe0b59418eaec4a3fcdc474125c798f0b787dc390f6eb4e79934d4160`。
- VAE decoder 載入 86 個 BF16 張量、0 個量化模組；Q4 只套用 Transformer
  Linear，不套用影片 VAE。
- Swift 套件獨立釘定 `mlx-swift` `0.31.6`；Python 參考依 v0.14.22
  lockfile 使用 MLX `0.31.1`。

## 實作

- `RuntimeSupport/LTXVideoWorker/`：獨立 SwiftPM 套件。
- `LTXVideoSwiftRuntime`：3D temporal padding、PixelNorm、ResBlock、
  depth-to-space、spatial unpatchify、VAE decoder、safetensors 權重載入及
  可串流的時間／空間 tiling 拼接。
- 精度策略沿用 MiniMax Music 3：卷積主幹保留 BF16，跨 tile 的 blend mask、
  加權、累加與正規化固定使用 FP32；FP32 只用在對捨入敏感的邊界運算，
  不把整個 VAE 常駐權重升為 FP32。
- `GenImageLTXVideoPoC decode`：讀取 Python fixture 的 latent，單獨執行
  `latent -> frames`。
- `GenImageLTXVideoPoC compare`：沿用 Music3 的 Double 累加與
  `max|reference-actual| / max|reference|`，依 input dtype 自動選擇門檻。
- `scripts/ltx-parity/dump.py vae-decode`：支援 `--input-dtype`、`--seed`、
  單 tile／多 tile，輸出含 metadata 的 safetensors。

本階段新增 Swift 1,432 行：runtime 1,048、PoC 283、測試 101；Python parity
腳本 151 行。

## 真實權重比對

| dtype | tile 數 | latent | frames | relative max diff | 門檻 | 結果 |
|---|---:|---|---|---:|---:|---|
| BF16 | 1 | `[1,128,2,1,1]` | `[1,3,9,32,32]` | `0` | `1e-2` | PASS |
| BF16 | 3 | `[1,128,4,1,1]` | `[1,3,25,32,32]` | `1.090779110e-7` | `1e-2` | PASS |
| FP32 | 1 | `[1,128,2,1,1]` | `[1,3,9,32,32]` | `0` | `1e-4` | PASS |
| FP32 | 3 | `[1,128,4,1,1]` | `[1,3,25,32,32]` | `1.092554558e-7` | `1e-4` | PASS |

初版讓 BF16 decoded tile 先乘 BF16 ramp mask，再寫入 FP32 buffer，3-tile
relative max diff 為 `2.382806965e-3`。改用 Music3 的混合精度策略後，mask、
乘法、累加與除法全程使用 FP32，誤差降低約 21,845 倍至 `1.090779110e-7`，
與全 FP32 路徑的 `1.092554558e-7` 相同量級。這次差異因此判定為過早 BF16
捨入，而不是 fused kernel 數值地板；單 tile 在 BF16、FP32 都逐值一致。

## 驗證指令

```bash
swift build --package-path RuntimeSupport/LTXVideoWorker -c release
swift test --package-path RuntimeSupport/LTXVideoWorker

RuntimeSupport/LTXVideoWorker/.build/arm64-apple-macosx/release/GenImageLTXVideoPoC \
  decode --model-dir <model-dir> --input <reference.safetensors> \
  --output <actual.safetensors>

RuntimeSupport/LTXVideoWorker/.build/arm64-apple-macosx/release/GenImageLTXVideoPoC \
  compare --reference <reference.safetensors> --actual <actual.safetensors> \
  --key frames

swift build -c release
swift test
```

## 後續區塊估計

以下只做估算，本階段不實作：

- Block 2，影片／音訊 Transformer 與位置編碼：約 1,800–2,500 行 Swift。
- Block 3，scheduler、CFG 與去噪迴圈：約 600–900 行。
- Block 4，Gemma／Connector prompt conditioning：約 1,200–1,800 行；若能重用
  現有 MLXLMCommon，實際行數可能較低。
- Block 5，Audio VAE、vocoder 與影音同步：約 1,300–2,000 行。
- Block 6，完整 Worker protocol、取消／進度／輸出串流與 App 切換：約
  700–1,100 行。

排除項目仍為 `ic_lora.py`、`retake.py`、`keyframe_interpolation.py` 與
`ltx-trainer`。

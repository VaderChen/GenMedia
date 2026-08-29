# LTX-2 純 Swift 移植紀錄

## 範圍

本階段已完成 Block 0–4：比對基礎設施、影片 VAE、影片／音訊多模態 Transformer、
Gemma3 文字編碼與 conditioning connector、兩階段 distilled denoise，以及
影片／音訊解碼與 FFmpeg 封裝。`LTXVideoGenerationService` 已改走獨立的
`GenImageLTXVideoWorker` JSON 子行程，不再搜尋或啟動 Python Runtime。Worker
在模型檔案不完整時明確失敗，不會靜默退回其他 Runtime；目前原生支援文字生影，
image conditioning 與 LoRA fusion 仍明確回報不支援。

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
- `GenImageLTXVideoWorker`：正式 App 使用的 JSON 子行程入口；執行完整的原生
  distilled 兩階段影片／音訊生成，並在模型完整性或能力不符時明確失敗。
- `LTXVideoSwiftRuntime`：3D temporal padding、PixelNorm、ResBlock、
  depth-to-space、spatial unpatchify、VAE decoder、safetensors 權重載入及
  可串流的時間／空間 tiling 拼接；影片／音訊 patchifier、partial modality
  tiling、split RoPE、AdaLN、GQA attention、48 層 AV Transformer 及獨立權重載入器。
- 精度策略沿用 MiniMax Music 3：卷積主幹保留 BF16，跨 tile 的 blend mask、
  加權、累加與正規化固定使用 FP32；FP32 只用在對捨入敏感的邊界運算，
  不把整個 VAE 常駐權重升為 FP32。
- `GenImageLTXVideoPoC decode`：讀取 Python fixture 的 latent，單獨執行
  `latent -> frames`。
- `GenImageLTXVideoPoC transformer`：讀取固定 latent、timestep、text embedding
  與位置，單獨執行一次 Transformer forward。
- `GenImageLTXVideoPoC transformer-tiled`：以 temporal／spatial tile 執行
  多模態 forward，並用 trapezoidal mask 累加影片 token、平均音訊 token。
- `GenImageLTXVideoPoC gemma-hidden`：以 LTX-local Gemma3 all-layer adapter
  輸出 embedding 加 48 層 hidden states，共 49 組；保留左 padding 與 causal mask。
- `GenImageLTXVideoPoC gemma-features`：執行 49 層 hidden state 的 per-token RMS
  整理、影片／音訊 projection 與 8 層 register connector。
- `GenImageLTXVideoPoC compare`：沿用 Music3 的 Double 累加與
  `max|reference-actual| / max|reference|`，優先依 fixture 的
  `effective_input_dtype` 自動選擇門檻。
- `scripts/ltx-parity/dump.py vae-decode`：支援 `--input-dtype`、`--seed`、
  單 tile／多 tile，輸出含 metadata 的 safetensors。

Block 0 + 1 的紀錄行數為 Swift 1,432 行：runtime 1,048、PoC 283、測試 101；
Python parity 腳本當時 151 行。加入 Block 2–4、正式 Worker 與可脫離 Metal
kernel 的純邏輯測試後，目前 Worker 套件 Swift 總行數為 8,094 行：runtime
6,217、PoC 與正式 Worker 1,469、測試 408；`scripts/ltx-parity/dump.py` 為
816 行。
Transformer 權重載入 7,450 個 tensor，其中 1,632 個量化模組，來源 dtype 為
BF16、FP32、UINT32；量化設定為 4-bit、group size 64。

## Block 3–4 目前狀態

已完成不依賴實際模型檔的 Swift 結構與介面：

- LTX-local Gemma3 all-layer adapter；不修改 `mlx-swift-lm`，因其
  `Gemma3Model.layers` 為 internal，無法提供 LTX 所需的 49 組 hidden states。
- Gemma 權重載入支援分片 safetensors、`model.`／`language_model.` 包裝前綴、
  INT4 Linear／Embedding 與 BF16／FP32 計算 dtype。
- Connector 權重載入、per-token RMS stacking、左 padding、register replacement
  與 `gemma-hidden`／`gemma-features` PoC 子命令。

目前 `/Users/vader/AI Modes/Managa/ltx-2.3-mlx-q4` 僅有設定檔與
`transformer-distilled-1.1.safetensors`，以下檔案尚未安裝，因此這台機器仍不能
執行端到端生成或捏造 Gemma、connector、VAE 的實際權重 parity 數字：

| 類別 | 必要檔案／目錄 | 狀態 |
|---|---|---|
| LTX connector | `connector.safetensors` | 缺少 |
| 影片 VAE | `vae_decoder.safetensors`、`vae_encoder.safetensors` | 缺少 |
| 音訊 VAE | `audio_vae.safetensors`、`vocoder.safetensors` | 缺少 |
| Gemma3 | 含 `config.json`、分片 safetensors、tokenizer 的獨立模型目錄 | 未找到 |

上述權重由模型中心安裝後，可用 Python parity fixture 驗證 Swift
`gemma-hidden`、`gemma-features` 與完整生成輸出；Python 參考環境與 `dump.py`
僅保留作為 fixture 產生器，不會成為正式 App 路徑。

## 真實權重比對

| dtype | tile 數 | latent | frames | relative max diff | 門檻 | 結果 |
|---|---:|---|---|---:|---:|---|
| BF16 | 1 | `[1,128,2,1,1]` | `[1,3,9,32,32]` | `0` | `1e-2` | PASS |
| BF16 | 3 | `[1,128,4,1,1]` | `[1,3,25,32,32]` | `1.090779110e-7` | `1e-2` | PASS |
| FP32 | 1 | `[1,128,2,1,1]` | `[1,3,9,32,32]` | `0` | `1e-4` | PASS |
| FP32 | 3 | `[1,128,4,1,1]` | `[1,3,25,32,32]` | `1.092554558e-7` | `1e-4` | PASS |

### Block 2 Transformer parity

Python 參考的 Transformer 會在模型入口把 latent、timestep 與 text embedding
統一轉成 BF16。因此 fixture 的 `input_dtype=float32` 會同時記錄
`effective_input_dtype=bfloat16`；比較器以有效計算精度選門檻，而不是誤把輸出
dtype 當成計算精度。

| level | 宣告／有效 dtype | video relative max diff | audio relative max diff | tile 數 | 門檻 | 結果 |
|---|---|---:|---:|---:|---:|---|
| 單次 forward | BF16 / BF16 | `1.138687255e-2` | `2.361593757e-3` | 1 | `1e-2` | BF16 video 略超出 |
| 單次 forward | FP32 / BF16 | `7.649999623e-5` | `2.728309175e-6` | 1 | `1e-2` | PASS |
| tiled forward | BF16 / BF16 | `3.339106943e-3` | `2.382457428e-3` | 2 | `1e-2` | PASS |
| tiled forward | FP32 / BF16 | `5.788624681e-5` | `1.798498131e-6` | 2 | `1e-2` | PASS |

單次 BF16 video 案例的 `1.14e-2` 與門檻相差很小，來源是 MLX fused attention
與 RoPE 的低精度捨入；tiled 案例仍在 BF16 門檻內。FP32 宣告路徑的有效計算
仍是 BF16，並非未來 production 的 FP32 Transformer 實作。

### Block 2 代表性規模回歸

原本的單次 forward 預設只有 video/audio 各 1 token，無法覆蓋多 token attention、
位置編碼或分塊路徑。現在 `scripts/ltx-parity/dump.py` 的 Transformer 預設固定為
video `8x16x16=2048`、audio `256`、text `64`；三種常設規模與實際 Swift parity
矩陣定義在 `scripts/ltx-parity/fixtures/manifest.json`。以下為同一 seed=7000
在 Worker `mlx-swift 0.31.6` 的實測；數字是 video / audio relative max diff。

| 規模 | Video tokens | BF16 forward | BF16 tiled | 宣告 FP32 / 有效 BF16 forward | 宣告 FP32 / 有效 BF16 tiled |
|---|---:|---:|---:|---:|---:|
| small `4x8x8` | 256 | `3.8603e-3 / 1.0303e-3` | `1.8557e-3 / 1.2375e-3` | `3.1546e-4 / 4.7784e-4` | `1.3632e-4 / 4.4570e-4` |
| medium `8x16x16` | 2048 | `1.5582e-3 / 1.2849e-3` | `1.1120e-2 / 1.1969e-3` | `1.0082e-4 / 4.3489e-4` | `2.8168e-3 / 4.5893e-4` |
| large `16x16x32` | 8192 | `5.1065e-3 / 1.4389e-3` | `4.1826e-3 / 1.2369e-3` | `1.1456e-3 / 4.7826e-4` | `1.0144e-3 / 4.8421e-4` |

BF16 video 誤差在 256、2048、8192 tokens 間沒有單調增長，整體維持約 `1e-3`
至 `5e-3`；medium tiled 的 `1.1120e-2` 是既有 `1e-2` 門檻附近的單筆邊界
值，應保留在回歸報告中，不能宣稱所有組合均低於門檻。FP32 宣告組也不是實際
FP32 計算：Python `LTXModel.__call__` 與 Swift `LTXTransformer` 都會在入口將
輸入統一轉成 BF16，因此 compare 依 metadata 的 `effective_input_dtype` 使用
BF16 門檻。權重中的 BF16/FP32/UINT32 是來源儲存格式，不等於有效計算精度。

目前 Swift 端沒有可直接切換的真正 FP32 Transformer 診斷模式；
`LTXTransformer` 的入口會固定將 latent、timestep 與兩個 text embedding
轉成 BF16。Q4 packed weight 使用 UINT32 儲存不是唯一限制，但即使 loader
收到 `float32`，現有 forward 仍會在模型入口降為 BF16。因此 `input_dtype=float32`
只代表 fixture 的宣告／輸入儲存格式，不能用來宣稱已完成 FP32 計算判別；若要
加入該判別，必須另做不改變 production 行為的 FP32 diagnostic path，並重新驗證
量化矩陣乘法的輸出 dtype。

### Parity fixture 預設 shape

| 子命令 | 預設輸入 shape | 預設驗證範圍 |
|---|---|---|
| `vae-decode` | latent `[1,128,2,1,1]`；`multi` 模式為 `[1,128,4,1,1]` | 預設 single；multi 時驗證 temporal tiles，空間仍為 `1x1` |
| `vae-encode` | pixels `[1,3,9,32,32]` | 影片 VAE encoder 基本 shape，無 tiling |
| `audio-vae-decode` | latent `[1,8,2,16]` | 音訊 VAE decoder 基本 chunk |
| `transformer-forward` | video `[1,2048,128]`、audio `[1,256,128]`、text `[1,64,*]` | 代表性多模態單次 forward |
| `transformer-tiled` | 同上 | 預設 temporal 2 tiles、overlap 1；height/width 各 1 tile |

VAE 的 `vae-decode` 預設不是 spatial tiling fixture：目前 Block 1 的常設多 tile
案例是 temporal `[1,128,4,1,1]`。因此 `1x1` 空間 shape 不足以驗證 spatial tile；
這次只修正 Block 2 Transformer 的退化預設，沒有改 VAE/tiling 實作或其既有
預設行為。若要補 spatial VAE regression，應另定較高記憶體成本的 fixture。

Level 1 的 13 項 Swift Testing 純邏輯測試（patchify／unpatchify、位置順序、
RoPE 頻率、tile 座標與 mask）均通過；在目前 Xcode 的 SwiftPM runner 下，直接
執行產出的 `LTXVideoSwiftRuntimeTests.xctest` 可正常載入 MLX metallib 並通過。

初版讓 BF16 decoded tile 先乘 BF16 ramp mask，再寫入 FP32 buffer，3-tile
relative max diff 為 `2.382806965e-3`。改用 Music3 的混合精度策略後，mask、
乘法、累加與除法全程使用 FP32，誤差降低約 21,845 倍至 `1.090779110e-7`，
與全 FP32 路徑的 `1.092554558e-7` 相同量級。這次差異因此判定為過早 BF16
捨入，而不是 fused kernel 數值地板；單 tile 在 BF16、FP32 都逐值一致。

## 驗證指令

```bash
swift build --package-path RuntimeSupport/LTXVideoWorker -c release
swift test --package-path RuntimeSupport/LTXVideoWorker

RuntimeSupport/LTXVideoWorker/.build/out/Products/Release/GenImageLTXVideoPoC \
  decode --model-dir <model-dir> --input <reference.safetensors> \
  --output <actual.safetensors>

RuntimeSupport/LTXVideoWorker/.build/out/Products/Release/GenImageLTXVideoPoC \
  compare --reference <reference.safetensors> --actual <actual.safetensors> \
  --key frames

swift build -c release
swift test
```

## 後續工作

以下是目前與後續部分的估算：

- Block 2，影片／音訊 Transformer 與位置編碼：已完成，新增 runtime 1,478 行；
  目前尚未取得可區分 active coding 的可靠計時，因此不把 git 時間窗當作工時。
- Block 3–4 已完成結構與管線串接；待完整模型下載後補做端到端權重 parity 與
  實機耗時／記憶體紀錄。
- 後續只處理 image conditioning、LoRA fusion 與更長影片的最佳化，不回到 Python
  生成路徑。

排除項目仍為 `ic_lora.py`、`retake.py`、`keyframe_interpolation.py` 與
`ltx-trainer`。

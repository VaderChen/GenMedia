# ACE-Step 純 Swift PoC

`ACEStepSwiftPoC` 用來驗證 ACE-Step 1.5 Turbo 在 macOS 上的完整 Swift／MLX 推論鏈。驗證完成後，核心程式已抽離為 `ACEStepSwiftRuntime`，並由 GenMedia 正式音樂生成服務直接使用。

`ACEStepSwiftRuntime` 內的階段型別原本沿用 PoC 時期的名稱（`ConditioningPoC`、`GeneratedAudioPoC`、`VAEDecodePoC`、`QwenEmbeddingPoC`），但它們是正式生成路徑的一部分，已改名為 `ACEStepConditioningStage`、`ACEStepAudioGenerationStage`、`ACEStepVAEDecodeStage` 與 `ACEStepTextEmbedder`。只有 `ACEStepDiTForwardProbe`（原 `DiTForwardPoC`）僅供本執行檔診斷使用。

## 已驗證範圍

- 解析 ACE-Step、Qwen3 Embedding 與 Oobleck VAE 設定。
- 直接讀取 safetensors header，檢查 Tensor 名稱、形狀、型別與資料範圍。
- 檢查直接條件生成所需的 DiT、Qwen3 Embedding 與 VAE 權重。
- 執行 MLX Swift 的 Conv1D、ConvTransposed1D、Scaled Dot Product Attention 與 RMSNorm。
- 選擇性用 MLX 直接開啟模型權重，驗證 DiT 軸向轉換與 VAE weight normalization 融合。
- 以純 Swift 建立完整 Oobleck VAE decoder、載入原始權重並輸出 16-bit PCM WAV。
- 直接重用原生 Qwen Transformer，以 Swift tokenizer 產生 Prompt／歌詞 Token 與 Hidden States。
- 完成歌詞、音色與 silence latent 條件編碼，輸出 DiT context latents。
- 完成 ACE-Step Turbo DiT 24 層前向、1～20 step timestep schedule 與 Euler sampler。
- 完成 Prompt、選填歌詞、Turbo diffusion、VAE 與 WAV 的端到端生成。
- 長音訊使用重疊分塊 VAE 解碼及 PCM 串流寫入，正式 Runtime 支援 10～300 秒目標長度。

## 執行方式

快速檢查模型結構與 MLX 算子：

```bash
swift run ACEStepSwiftPoC \
  --model-root "/path/to/models/ace-step-1.5-turbo"
```

列出每個元件前五個 Tensor：

```bash
swift run ACEStepSwiftPoC \
  --model-root "/path/to/models/ace-step-1.5-turbo" \
  --list-tensors 5
```

實際以 MLX 載入並轉換 DiT 代表權重：

```bash
swift run ACEStepSwiftPoC \
  --model-root "/path/to/models/ace-step-1.5-turbo" \
  --load-component dit
```

`--load-component` 可使用 `dit`、`embedding`、`vae` 或 `language-model`。這個模式會開啟大型權重，記憶體需求高於預設的 header 檢查。

執行完整 VAE decoder 並產生 WAV：

```bash
swift run ACEStepSwiftPoC \
  --model-root "/path/to/models/ace-step-1.5-turbo" \
  --decode-vae "/tmp/ace-step-swift-vae.wav" \
  --latent-frames 8
```

VAE 測試使用固定數學序列建立 latent，不依賴隨機數，因此後續可與其他實作進行逐值比對。ACE-Step 的 VAE hop length 為設定檔中所有 downsampling ratio 的乘積；程式會檢查輸出 sample 數是否完全吻合。

編碼 Prompt 與選填歌詞，並保存 condition encoder 的輸入：

```bash
swift run ACEStepSwiftPoC \
  --model-root "/path/to/models/ace-step-1.5-turbo" \
  --encode-prompt "Cinematic electronic music with a steady pulse" \
  --lyrics "" \
  --language en \
  --embedding-output "/tmp/ace-step-swift-embedding.safetensors"
```

輸出包含 `text_hidden_states`、`text_attention_mask`、`lyric_hidden_states` 與 `lyric_attention_mask`，格式可直接供下一階段 ACE condition encoder 使用。

端到端生成：

```bash
swift run ACEStepSwiftPoC \
  --model-root "/path/to/models/ace-step-1.5-turbo" \
  --encode-prompt "Cinematic electronic music with a steady pulse" \
  --lyrics "" \
  --language en \
  --encode-condition "/tmp/ace-step-condition.safetensors" \
  --condition-frames 250 \
  --generate-audio "/tmp/ace-step-swift-generated.wav" \
  --inference-steps 8 \
  --seed 42
```

## 正式接入

`ACEStepMusicGenerationService` 使用 `.mlxSwift` Profile 直接呼叫 `ACEStepNativeGenerator`。模型中心只需提供 Qwen3 Embedding、ACE-Step Turbo DiT、silence latent 與 Oobleck VAE 權重；生成完成後再由共用音訊輸出層轉為 MP3、M4A、AAC 或 FLAC。

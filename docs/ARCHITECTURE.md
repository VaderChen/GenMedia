# GenImage 架構

繁體中文 | [English](ARCHITECTURE.en.md) | [日本語](ARCHITECTURE.ja.md) | [한국어](ARCHITECTURE.ko.md)

## 設計目標

1. 文生圖、圖生文、圖生圖、影片生成、音樂生成與 Upscale 是獨立能力，不互相依賴。
2. 圖片、影片與音訊輸出可以形成工作流與分支。
3. UI 不直接依賴 MLX、Core ML 或任何特定模型。
4. 模型更新時，以 Profile 切換模型版本與推論架構。
5. 舊作品保存 Profile 快照，不受日後 Profile 修改影響。

## 分層

```text
HTML / CSS / JavaScript UI
            │
            │ JSON commands + state snapshots
            ▼
HybridBridgeController (WKWebView Bridge)
            │
            ▼
AppStore / Workflow coordination
            │
            ├── Model manager
            ├── Asset repository
            ├── Job queue
            └── Profile registry
                    │
                    ▼
        Independent inference services
        ├── TextToImageGenerating
        ├── ImageDescribing
        ├── ImageToImageGenerating
        ├── VideoGenerating
        ├── MusicGenerating
        └── ImageUpscaling
                    │
                    ▼
       MLX Swift / Core ML / External CLI
```

Web UI 只能透過 Bridge 使用本機能力，不可直接讀取任意檔案、模型資料夾或系統 API。

## Profile

`InferenceProfile` 包含：

- 功能類型。
- 模型 ID。
- 模型 revision。
- 推論架構：MLX Swift、Core ML、本機服務或外部 CLI。
- 功能預設值。
- Profile revision。

執行工作時，`WorkflowOperation.profileSnapshot` 保存完整值，而不是只保存 Profile ID。

內建 Profile 不直接修改。使用者需要變更時先複製，再儲存為新的 revision。

## 資產與流程

`ImageAsset.parentAssetID` 表示圖片來源。沒有 parent 的資產是獨立工作的根節點：

- 獨立文生圖：生成圖片沒有 parent。
- 獨立圖生文：先匯入一張根圖片，描述輸出寫入 Recipe。
- 獨立 Upscale：先匯入一張根圖片，放大結果以原圖為 parent。
- 串接生成：生成結果以選取圖片為 parent。
- 獨立文生影：MP4 資產沒有 parent。
- 圖生影：MP4 資產以來源圖片為 parent。
- 獨立文生音樂：MP3、M4A、AAC 或 FLAC 資產沒有 parent，並記錄實際時長、取樣率與聲道數。

`WorkflowGraph` 提供 lineage 與 children 查詢，UI 不需要推測資產關係。

## 推論 Runtime

文生圖固定使用 `Z-Image.swift` commit `28bfcf3148c041a554629247170eb54d9ac46830`：

- macOS 14+、Swift 6。
- `ZImageGenerationRequest` 支援 Prompt、負向 Prompt、尺寸、步數、Seed、模型與 runtime options。
- `ZImageTextToImageService` 包裝 `ZImagePipeline.generate` 並提供逐階段進度。
- 去噪迴圈會檢查 Swift Task cancellation。
- 支援模型卸載、LoRA 卸載、取消與記憶體快取清理。

圖生文固定使用 `mlx-swift-lm 2.30.6`：

- `QwenVLImageDescriptionService` 透過 `VLMModelFactory` 載入本機 Qwen3-VL。
- 模型容器在服務生命週期內快取，避免同一 Profile 重複載入。
- 支援繁中、英文、日文、韓文輸出提示。

Upscale 由 `CoreMLUpscaleService` 執行 Real-ESRGAN 512 tile 與 4× 拼接。

影片由 `LTXVideoGenerationService` 呼叫 `ltx-2-mlx generate`：

- 文生影與圖生影共用 `VideoGenerating` 與 `VideoGenerationRequest`。
- Swift 驗證 Profile、模型路徑、尺寸、幀數、FPS 與輸出數量。
- LTX-2.3 額外要求幀數符合 `8n+1`。
- 外部 Process 支援 Task cancellation、日誌錯誤回報與百分比進度擷取。
- MP4 輸出以 `generatedVideo` 資產加入工作區，Web UI 使用原生 `<video>` 播放。
- Runtime 可透過 `GENIMAGE_LTX_RUNTIME` 或標準安裝位置替換，不將 Python 實作耦合進 UI 或 AppStore。

音樂由 `MiniMaxMusic3GenerationService` 呼叫 `mlx-minimax-music3 generate`：

- 文生音樂使用 `MusicGenerating`、`MusicGenerationRequest` 與 `MusicGenerationOptions`。
- Swift 驗證 Profile、模型完整性、風格 Prompt、5～300 秒長度（最長 5 分鐘）、步數與輸出格式；歌詞為選填，空值會轉為純音樂標記與無人聲 Prompt。
- Runtime 固定產生暫存 WAV，再由 FFmpeg 轉碼為 MP3 320 kbps、M4A AAC 256 kbps、ADTS AAC 256 kbps 或 FLAC 無損音訊。
- 成功、失敗或取消時都會清理 WAV、Prompt、歌詞與日誌；只有完成的壓縮音訊會以 `generatedAudio` 資產保留。
- Web UI 使用原生 `<audio controls>` 播放，Inspector 顯示實際時長、格式、44.1 kHz 取樣率與聲道數。
- Runtime 可透過 `GENIMAGE_MINIMAX_MUSIC3_RUNTIME` 或標準 Application Support 位置替換；模型以固定 revision 安裝並維持其獨立 Community License。

MLX metallib 已由 `RuntimeSupport/mlx.metallib` 與 `build.command` 放入 Release 執行目錄。發佈前仍需完成模型授權檢查、16/24/32GB 壓力測試、App bundle、簽章與公證。

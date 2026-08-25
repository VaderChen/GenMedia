# GenMedia 架構

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
       MLX Swift / Core ML / Local REST Service / External CLI
```

Web UI 只能透過 Bridge 使用本機能力，不可直接讀取任意檔案、模型資料夾或系統 API。

### Web UI 更新策略

- Swift 推送狀態時，Web UI 會在 Prompt、負向 Prompt 或歌詞欄位聚焦期間保留本機編輯值，並延後非必要的完整渲染，避免游標、選取範圍與輸入法組字被重設。
- 生成類型與 Prompt／歌詞／輸出設定 TAB 使用獨立的創作面板 renderer，不替換預覽、播放器、Inspector 或側欄 DOM。
- 全域狀態確實需要完整渲染時，播放中的 `<audio>`、`<video>` 與音訊視覺化節點會先脫離再接回相同資產位置，保持播放進度與 Web Audio 連線。

## 內部結構整理

本次整理只調整程式分層與責任邊界，不改變既有生成能力、使用流程或 Web Bridge 協定：

- `ApplicationSupport` 統一定義 `Models`、`Runtime`、`Workspace`、`Pasted` 與 `Generated` 的 Application Support 路徑，並在啟動時接回舊的 `GenMedia` 工作區資料。
- `OutputGeometry` 集中輸出尺寸的上下限、16 倍數對齊、比例換算與圖生圖生成畫布策略；Web UI 的 `js/geometry.js` 維持鏡像實作，避免 Native、MCP 與 UI 各自計算出不同結果。
- `AppStore` 只保留型別宣告、儲存屬性與初始化；Persistence、Paths、Selection、Profiles、OutputSettings、Assets、ImageGeneration、MediaGeneration、Jobs 與 ModelInstallation 依職責拆成 `AppStore+*.swift`。
- Web UI 將側邊欄與路由、工作區分頁、尺寸運算及全量渲染保護拆至 `chrome.js`、`workspace-tabs.js`、`geometry.js` 與 `render-preservation.js`；`app.js` 保留橋接與應用程式協調。
- `SubprocessRuntime` 統一外部 Worker、影片 CLI、音樂 CLI 與 FFmpeg 的可執行檔搜尋、環境、日誌、進度、停滯偵測、取消及終止語意。
- 相依套件修正由 `Patches/manifest.txt` 描述，統一交由 `scripts/apply-runtime-patches.command` 套用與驗證；版本不符或修正失敗時建置會停止，不會靜默使用未修正的來源碼。
- ACE-Step 正式 Runtime 的階段型別已移除 PoC 命名；診斷專用的 DiT probe 與正式生成階段分開，避免實驗程式與產品路徑混淆。

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

`MediaAsset.parentAssetID` 表示媒體來源。沒有 parent 的資產是獨立工作的根節點：

- 獨立文生圖：生成圖片沒有 parent。
- 獨立圖生文：先匯入一張根圖片，描述輸出寫入 Recipe。
- 獨立 Upscale：先匯入一張根圖片，放大結果以原圖為 parent。
- 串接生成：生成結果以選取圖片為 parent。
- 獨立文生影：MP4 資產沒有 parent。
- 圖生影：MP4 資產以來源圖片為 parent。
- 獨立文生音樂：MP3、M4A、AAC 或 FLAC 資產沒有 parent，並記錄實際時長、取樣率與聲道數。

`WorkflowGraph` 提供 lineage 與 children 查詢，UI 不需要推測資產關係。

開啟中的工作區分頁是生成專案的生命週期邊界。Swift 將 `Project`、`MediaAsset`、`WorkflowOperation` 與選取狀態以原子 JSON 快照保存至 Application Support；一般 App 結束不會清除。Web UI 的分頁狀態保存在 WebKit localStorage，關閉分頁時透過 Bridge 通知原生層移除該分頁資產與 lineage 索引，但不刪除已輸出的媒體檔。

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

音樂由 `MusicGenerationRouter` 依 `MusicRuntimeAdapter.supports` 分派，不在 Router 內集中硬編碼模型 ID：

- 文生音樂使用 `MusicGenerating`、`MusicGenerationRequest` 與 `MusicGenerationOptions`。
- `ACEStepMusicGenerationService` 使用 `.mlxSwift` Profile，直接呼叫 `ACEStepSwiftRuntime` 完成 Qwen3 Embedding、條件編碼、Turbo DiT、Euler sampler 與 Oobleck VAE，不啟動外部服務或 Process。
- ACE-Step 支援 10～300 秒、1～20 steps、可選歌詞與純音樂；latent 長度由 VAE 取樣率與 hop length 計算，長音訊使用重疊分塊解碼並串流寫入 PCM，以限制峰值記憶體。程式與模型採 MIT License。
- 音樂 Profile 以 `ProfileMusicConfiguration` 提供長度上下限及「目標／最長」語意，Web UI 不需要依模型 ID 判斷欄位行為。
- `MiniMaxMusic3GenerationService` 使用 `.externalCLI` Profile 呼叫 `mlx-minimax-music3 generate`；5～300 秒參數代表最長長度，模型可在輸出音訊結束標記時提前自然結束。模型維持其獨立 Community License。
- 兩個 Adapter 都固定取得暫存 WAV，再由 `AudioOutputEncoder` 轉碼為 MP3 320 kbps、M4A AAC 256 kbps、ADTS AAC 256 kbps 或 FLAC 無損音訊。
- 成功、失敗或取消時清理暫存檔；只有完成的壓縮音訊會以 `generatedAudio` 資產保留，並記錄實際時長、取樣率與聲道數。
- ACE-Step 權重由模型中心管理，原生 Runtime 編譯於 App 內，不使用獨立安裝路徑或服務環境變數。

MLX metallib 已由 `RuntimeSupport/mlx.metallib` 與 `build.command` 放入 Release 執行目錄。發佈前仍需完成模型授權檢查、16/24/32GB 壓力測試、App bundle、簽章與公證。

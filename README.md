# GenMedia

繁體中文 | [English](README.en.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

GenMedia 是一款**原生支援 Apple Silicon** 的本機 AI 媒體生成 App。目前專案已建立可編譯的混合式應用程式：

- Swift 負責模型、Profile、工作佇列、檔案與 MLX／Core ML 推論。
- `WKWebView` 內嵌 HTML、CSS 與 JavaScript UI，不需要網路或 npm runtime。
- 文生圖、圖生文、圖生圖、文生影、圖生影、文生音樂、字幕生成與 Upscale 都可獨立執行，也可以透過資產來源關係串接。
- 每次操作保存 Profile 快照，模型或架構更新後仍可追蹤當時的版本。
- 獨立設定頁支援繁體中文、英文、日文、韓文及六套可持久保存的配色。
- 設定頁可用 Switch 啟動只綁定本機的 MCP HTTP API；另保留可在 App 未啟動時獨立運作的 JSON-RPC 2.0 stdio server。

## 預覽

![GenMedia 媒體智能生成介面](images/cap001.jpg)

## 執行

需求：macOS 14+、Apple Silicon、Xcode 16+。

```bash
./build.command
./run.command
```

`build.command` 會建立 Release 執行檔，並在 `dist/` 輸出標準 `GenMedia.app`。App 內含 WebUI 資源、MLX Metal runtime、MCP server 及模型診斷工具。

```bash
# 建置 Release 執行檔與 App
./build.command

# 只做增量 Release 建置，不建立 App bundle
./build.command --no-app

# 指定版本與 Bundle ID
GENIMAGE_VERSION=1.1.0 GENIMAGE_BUNDLE_ID=com.example.genimage ./build.command
```

`run.command` 會自動使用 `--no-app`，日常啟動不會重複建立 App bundle。對外發佈的 DMG 由獨立本機流程完成 Developer ID Application 簽章、Apple Notarization、Staple 與 Gatekeeper 驗證。

### 影片 Runtime

影片生成使用可替換的 `ltx-2-mlx` 外部 Runtime；Swift App 負責 Profile、參數驗證、工作佇列、取消、進度、資產與影片播放。第一次使用前安裝 CLI 與 FFmpeg：

```bash
brew install uv ffmpeg
./scripts/install-ltx-runtime.command
```

App 會依序尋找 `GENIMAGE_LTX_RUNTIME`、`GENIMAGE_LTX_RUNTIME_ROOT/.venv/bin/ltx-2-mlx`、App Helpers、`~/.local/bin/ltx-2-mlx`、常見 Homebrew 路徑及 `PATH`。若執行檔位於自訂位置，可指定：

```bash
GENIMAGE_LTX_RUNTIME="/absolute/path/to/ltx-2-mlx" ./run.command
```

`ltx-2-mlx` 預設會使用其 Gemma 文字編碼器設定；已有本機 Gemma 模型時可透過 `GENIMAGE_LTX_GEMMA_MODEL` 指定模型目錄或 Hugging Face ID。App 的 DMG 目前不內含 Python Runtime、Gemma 權重或 FFmpeg，正式散佈時應將其視為選用外部元件，並分別確認 Runtime 與模型授權。

### 字幕生成

字幕流程可匯入影片或音訊，抽取音訊軌後由 `SubtitleGenerationRouter` 選擇原生 Core ML ASR Adapter，保留片段時間軸並輸出 SRT 或 WebVTT。支援多語言 Whisper Large v3 Turbo、中文 Paraformer Large 與日文 Parakeet 0.6B；來源語言可自動偵測或由 Profile 指定。

辨識完成後可選用本機 Qwen3.5／Qwen3.8 MLX 文生文模型，在不改變時間軸的前提下翻譯為繁中、簡中、英、日、韓、西、法、德、義、葡、俄、阿拉伯、印地、孟加拉、印尼、越南、泰、土耳其、波蘭、荷蘭、瑞典、捷克、烏克蘭、馬來或菲律賓文，共 25 種目標語言。字幕會以 `generatedSubtitle` 資產保存於目前工作區，並以來源影片或音訊為 parent。

`GenImageASRPoC` 是獨立的 WhisperKit 驗證工具，用來在不修改主 App 工作區的情況下檢查媒體解碼、語言辨識與時間碼輸出；主 App 的正式字幕流程使用相同的純 Swift／Core ML 邊界。詳見 [ASR 字幕 PoC](docs/ASR_POC.md)。

Qwen3-VL、Qwen3.5 與 Qwen3.8 屬於多模態模型，因此模型中心會同時歸類為「圖生文」與「文生文」，並各自提供對應 Profile。受管理的模型下載會一併取得 `processor_config.json`、影像／影片前處理設定、Tokenizer、Chat Template 與完整權重索引，完成必要檔案驗證後才標記為已安裝。

### Civitai LoRA

模型中心也提供數個以 `ZImageTurbo` 為基底的 Civitai LoRA（Asian Beauties、Turbo Lightning、Flat AnimeStyle、Diorama）。Civitai 下載端點依創作者設定可能要求登入；若下載回應 401，請在啟動 App 前設定個人 Civitai API Token：

```bash
CIVITAI_TOKEN="your-civitai-api-token" ./run.command
```

Token 只會透過 HTTPS `Authorization: Bearer` 標頭傳給 Civitai，不會寫入模型 manifest 或專案檔案。

### Profile、工作佇列與記憶體

- Profile 依「使用中、可用、下載中、不可用」排序；模型與 LoRA 相依項目完整時使用淡綠色外框，下載完成後會立即重新排序。
- 工作取消會先進入 `cancelling`，Runtime Task 結束後自動轉成 `cancelled` 並解除所有生成與記憶體按鈕。ETA 在進度 35% 且執行滿 15 秒後顯示數字，樣本不足時會使用整體耗時備援估算。
- Z-Image MLX 量化相容層支援 `quantize_config.json`、affine／mxfp4、packed pad token 與 FP16→BF16 載入修正。andrevp Z-Image Turbo MLX 4-bit 已完成實際生成驗證。
- 相依套件的原始碼修正列於 `Patches/manifest.txt`，由 `scripts/apply-runtime-patches.command` 在 Swift Package resolve 後套用；`build.command` 會自動呼叫。相依套件版本與 manifest 記載不符、修正檔遺失、套用失敗或套用後找不到預期標記，都會中止建置而不會以未修正的原始碼繼續。執行 `scripts/apply-runtime-patches.command --verify` 可只做檢查。
- 文生圖完成後保留模型權重與暖機 buffer；5 分鐘後只清理可重用的 MLX 暫存 buffer，不卸載模型。按下側欄「釋放記憶體」、切換模型，或切換 Profile 時 RAM 超過 90%，才會卸載不再需要的 Runtime。
- 下載保留來源原始檔名；生成輸出使用 `Image-YYYYMMDD-HHmm`、`Video-YYYYMMDD-HHmm` 或 `Music-YYYYMMDD-HHmm`，同分鐘重複時自動加上流水號，並可在設定頁更改輸出目錄。
- 每個開啟的工作區分頁視為一個生成專案；資產與 lineage 會原子寫入 Application Support，App 關閉後仍可恢復。只有明確關閉分頁時才移除該專案的工作區索引，已輸出的媒體檔仍保留於磁碟。
- App 的資料一律位於 `~/Library/Application Support/GenImage/`（`Models`、`Runtime`、`Workspace`、`Pasted`、`Generated`），由 `GenImageCore/ApplicationSupport.swift` 統一定義。工作區索引曾寫在 `GenMedia/`，啟動時會自動接回目前的根目錄；同名項目一律保留現有的，不覆蓋也不合併。目錄名稱維持 `GenImage` 而非改為與 App 一致的 `GenMedia`，因為 `Runtime/` 底下的 Python venv 把絕對路徑寫死在啟動 script 內，改名會讓已安裝的 LTX 與 MiniMax Runtime 失效。
- Prompt 與歌詞編輯期間會保留游標、選取範圍及輸入法組字狀態；生成類型、Prompt、歌詞與輸出設定 TAB 只局部更新創作面板。必要的完整畫面更新會沿用播放中的音訊或影片節點，避免中斷播放。
- 工作區底片列提供圖片匯入按鈕，並支援從 Finder 拖放一張或多張 PNG、JPEG、WebP、GIF、TIFF、HEIC 與 HEIF 圖片；音樂生成模式會停用圖片匯入，避免混用媒體來源。圖片生成時若已選取來源圖片，主按鈕會自動使用圖生圖 Profile，未選取時則使用文生圖 Profile。
- 圖片與影片比例選項改為下拉選單；圖生圖選取來源圖片後才會顯示「原解析度」，並依來源尺寸換算為符合 Runtime 的 16 倍數寬高。
- 圖生圖的寬高設定會實際傳入 Qwen Image Edit Runtime。當指定比例不同於來源圖片時，來源圖會保持自身解析度並以邊緣延展補足為輸出比例的畫布，再進行生成，以保留完整來源內容。
- 條件影像會以生成解析度編碼，使條件網格與輸出網格的 RoPE 位置完全對齊；若沿用固定 1024² 面積的條件網格，模型只會對準來源中央，輸出即成為裁切後的放大結果。
- 生成解析度與輸出解析度分離：指定面積小於 1024² 時，Runtime 會以指定比例、1024² 面積生成（與 diffusers 參考實作相同的約 4096 個 latent token），再以 Lanczos 縮放為指定尺寸輸出。DiT 的 token 數遠低於訓練規模時去噪會劣化並崩解成條紋，因此 `128 × 192` 這類低解析度改由縮放產生，畫質明顯較佳。
- 指定面積達到或超過 1024² 時直接以指定尺寸生成，不再縮放；解析度越高，Runtime 的記憶體用量與生成時間也會同步增加。低於 1024² 的輸出仍以 1024² 面積生成，因此生成時間不會隨輸出尺寸縮小而減少。
- 圖生圖按下生成時，若輸出面積小於 `512 × 512` 會先跳出警告對話框，可選擇「取消」或「仍要生成」；同一次執行期間確認過一次後不再重複提醒。

### 音樂 Runtime

文生音樂由 `MusicGenerationRouter` 依 Profile 分派至 ACE-Step 1.5 或 MiniMax Music 3 Adapter。Swift App 統一管理音樂風格、可選 Prompt、可選歌詞、步數、Seed、工作取消、時間推估及音訊資產資訊；Prompt 留空時使用所選音樂風格，歌詞留空時生成純音樂。各 Runtime 產生的暫存 WAV 都由共用 FFmpeg 輸出層轉為 MP3、M4A、AAC 或 FLAC。

- **ACE-Step 1.5 Turbo MLX**：建議 Profile，透過 App 內建的 Apple Silicon 原生 Swift／MLX Runtime 生成 10～300 秒歌曲或純音樂，不需要安裝額外服務。程式與模型採 MIT License，可商業使用；長音訊採重疊分塊 VAE 解碼以控制記憶體用量。

- **MiniMax Music 3 MLX 8-bit**：透過外部 `mlx-minimax-music3` CLI 執行，設定值是 5～300 秒的最長長度；模型可依歌曲結構提前自然結束，完成後 App 會顯示實際生成長度。模型仍適用其 Community License。

ACE-Step 原生 Runtime 隨 App 提供；模型、MiniMax Music 3 Runtime 與 FFmpeg 維持外部選用元件。

## 驗證

```bash
swift test

for file in Sources/GenImageApp/Resources/WebUI/js/*.js; do
  node --check "$file"
done
```

診斷本機模型與自動建立的 Profiles：

```bash
swift run GenImageDoctor

# 或指定自訂模型目錄
GENIMAGE_MODEL_ROOT="/path/to/models" swift run GenImageDoctor
```

啟動標準 MCP stdio server：

```bash
MCP_BIN_DIR="$(swift build -c release --show-bin-path)"
"$MCP_BIN_DIR/GenImageMCP"
```

MCP 支援 `initialize`、`ping`、`tools/list`、`tools/call`，工具包含本機模型、Profile、原生 Z-Image 文生圖、Qwen 圖生圖／圖生文、Core ML Upscale 與獨立字幕生成。

若從 App 使用，在「設定 → MCP 整合」開啟 Switch 後會顯示 `http://127.0.0.1:12181/mcp`；該端點只接受本機 HTTP POST JSON-RPC。HTTP 與 stdio 共用同一套 MCP 工具核心，但 stdio 執行檔仍可在 GenMedia.app 未開啟時獨立工作。

已完成 MCP 端到端實測：`genimage_generate_image` 可使用本機 Z-Image Turbo Q4 輸出 PNG；`genimage_describe_image` 可用 Qwen3-VL 輸出繁體中文描述；`genimage_upscale_image` 可使用本機 Real-ESRGAN Core ML 模型輸出 4× 圖片。

## 專案結構

本次內部整理只調整程式分層與責任邊界，不改變既有生成能力、使用流程或 Web Bridge 協定。

```text
Sources/
├── GenImageCore/
│   ├── ApplicationSupport.swift  # Application Support 資料位置的唯一定義
│   ├── DomainModels.swift        # 資產、配方、工作、模型與 Profile
│   ├── InferenceServices.swift   # 圖片、文字、影片、音樂與字幕推論介面
│   ├── ModelCatalog.swift        # 內建模型及 Profile
│   ├── OutputFileNaming.swift    # 圖片、影片、音樂與字幕輸出命名
│   ├── OutputGeometry.swift      # 輸出尺寸運算的唯一定義（WebUI 端由 js/geometry.js 鏡像）
│   ├── ProjectWorkspacePersistence.swift # 開啟中生成專案持久化
│   ├── SubtitleDocument.swift    # SRT／WebVTT 文件輸出
│   └── WorkflowGraph.swift       # 資產來源與分支關係
├── GenImageRuntime/
│   ├── ZImageTextToImageService.swift
│   ├── QwenVLImageDescriptionService.swift
│   ├── Qwen2511ImageToImageService.swift
│   ├── LTXVideoGenerationService.swift
│   ├── MusicGenerationRouter.swift
│   ├── ACEStepMusicGenerationService.swift
│   ├── MiniMaxMusic3GenerationService.swift
│   ├── MediaAudioPreparer.swift
│   ├── WhisperSubtitleTranscriber.swift
│   ├── ParaformerChineseSubtitleTranscriber.swift
│   ├── ParakeetJapaneseSubtitleTranscriber.swift
│   ├── SubtitleGenerationRouter.swift
│   ├── QwenTextGenerationService.swift
│   ├── SubprocessRuntime.swift   # 外部 Runtime 子行程的共用執行流程
│   ├── AudioOutputEncoder.swift
│   └── CoreMLUpscaleService.swift
├── GenImageApp/
    ├── AppStore.swift            # 型別宣告、儲存屬性與 init
    ├── AppStore+SubtitleGeneration.swift
    ├── AppStore+Workspaces.swift
    ├── AppStore+*.swift          # 其他依職責拆分的 AppStore 行為
    ├── HybridBridgeController.swift
    ├── LocalMCPServiceController.swift # App 內 MCP HTTP Switch 與狀態
    ├── HybridWebView.swift
    ├── AssetSchemeHandler.swift  # 安全提供本機圖片、影片與音訊給 Web UI
    └── Resources/WebUI/
        ├── js/automatic-flow.js  # 自動流程頁面骨架
        └── …                     # 其他 HTML/CSS/JavaScript 前端
├── GenImageASRPoC/
    └── main.swift                # 獨立 ASR 驗證工具
└── GenImageMCPServer/
    ├── MCPServer.swift           # 獨立 stdio JSON-RPC server 核心
    └── MCPHTTPServer.swift       # App Switch 使用的 localhost HTTP transport
Patches/
├── MLX-Swift-LM-Qwen35-Text-Only.patch # Qwen3.5 純文字輸入相容修正
└── manifest.txt                  # 建置時套用的相依套件修正清單
scripts/
└── apply-runtime-patches.command  # 依 manifest 套用與驗證相依套件修正
```

## 目前狀態

App 已接入真實本機推論：Z-Image Turbo 文生圖、Qwen3-VL／Qwen3.5／Qwen3.8 多模態圖生文與文生文、Qwen 2511 圖生圖、LTX-2.3 MLX 文生影／圖生影、ACE-Step 1.5 Turbo MLX 與 MiniMax Music 3 MLX 8-bit 文生音樂、Whisper／Paraformer／Parakeet 字幕生成，以及 Core ML Real-ESRGAN Upscale。影片以 MP4 加入工作區；音樂可輸出 MP3、M4A、AAC 或 FLAC；字幕可輸出 SRT 或 WebVTT，並保存語言、時間軸、Profile 快照與 lineage。

更多資訊：

- [更新紀錄](UpdateNote.md)
- [架構說明](docs/ARCHITECTURE.md)
- [Web Bridge](docs/WEB_BRIDGE.md)
- [開發路線](docs/ROADMAP.md)
- [MCP 介面](docs/MCP.md)
- [ASR 字幕 PoC](docs/ASR_POC.md)
- [本機模型測試](docs/MODEL_TEST_REPORT.md)

## 授權

本專案採 GPLv3 與商業授權雙軌：

- 開源使用依 [GNU General Public License v3.0](LICENSE) 授權。
- 若要在無法或不願遵守 GPLv3 的情境下使用，例如閉源整合、專有產品發行或需要客製商業條款，請聯絡著作權人另行取得商業授權。

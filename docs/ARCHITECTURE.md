# GenMedia 架構

繁體中文 | [English](ARCHITECTURE.en.md) | [日本語](ARCHITECTURE.ja.md) | [한국어](ARCHITECTURE.ko.md)

## 設計目標

1. 文生圖、圖生文、圖生圖、影片生成、音樂生成、字幕生成與 Upscale 是獨立能力，不互相依賴。
2. 圖片、影片、音訊與字幕輸出可以形成工作流與分支。
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
        ├── MediaTranscribing
        ├── SubtitleGenerating
        ├── TextGenerating
        └── ImageUpscaling
                    │
                    ▼
       MLX Swift / Core ML / Local REST Service / External CLI
```

Web UI 只能透過 Bridge 使用本機能力，不可直接讀取任意檔案、模型資料夾或系統 API。

### Web UI 更新策略

- `WebAppState` 負責完整內容同步；`WebActivityState` 僅同步任務、安裝狀態、系統資源、訊息與記憶體釋放狀態。純進度更新不重算資產或全畫面內容簽章，階段切換才重新檢查相關控制項。
- 系統資源讀取在背景執行；`WebAsset` 與資產 URL 索引依資產值快取。App 再次成為前景時清除索引，以更新外部修改的字幕。
- 圖片格線與底片列透過 `/thumbnail` 取得最大 384 px 的 ImageIO 縮圖，採延遲載入；原圖預覽仍讀完整檔案。PNG 縮圖保留透明度及 EXIF 方向，記憶體快取限制為 16 MiB／256 張。

- Swift 推送狀態時，Web UI 會在 Prompt、負向 Prompt 或歌詞欄位聚焦期間保留本機編輯值，並延後非必要的完整渲染，避免游標、選取範圍與輸入法組字被重設。
- 生成類型與 Prompt／歌詞／輸出設定 TAB 使用獨立的創作面板 renderer，不替換預覽、播放器、Inspector 或側欄 DOM。
- 工作區分頁 schema v3 讓每個 Tab 保存自己的工作類型、Profile 參照、Prompt、輸出參數及自動流程步驟；切換 Tab 時以單一 Bridge 命令套回 Native 創作狀態。
- 全域狀態確實需要完整渲染時，播放中的 `<audio>`、`<video>` 與音訊視覺化節點會先脫離再接回相同資產位置，保持播放進度與 Web Audio 連線。

## 內部結構整理

本次整理只調整程式分層與責任邊界，不改變既有生成能力、使用流程或 Web Bridge 協定：

- `ApplicationSupport` 統一定義 `Models`、`Runtime`、`Workspace`、`Pasted` 與 `Generated` 的 Application Support 路徑，並在啟動時接回舊的 `GenMedia` 工作區資料。
- `OutputGeometry` 集中輸出尺寸的上下限、16 倍數對齊、比例換算與圖生圖生成畫布策略；Web UI 的 `js/geometry.js` 維持鏡像實作，避免 Native、MCP 與 UI 各自計算出不同結果。
- `AppStore` 只保留型別宣告、儲存屬性與初始化；Persistence、Paths、Selection、Profiles、OutputSettings、Assets、ImageGeneration、MediaGeneration、Jobs 與 ModelInstallation 依職責拆成 `AppStore+*.swift`。
- Web UI 將側邊欄與路由、工作區分頁、尺寸運算及全量渲染保護拆至 `chrome.js`、`workspace-tabs.js`、`geometry.js` 與 `render-preservation.js`；`app.js` 保留橋接與應用程式協調。
- `SubprocessRuntime` 統一外部 Worker、影片 CLI、音樂 CLI 與 FFmpeg 的環境、日誌、進度、停滯偵測、取消及終止語意。
- `MediaCompatibilityService` 是 `ffmpeg`／`ffprobe` 的唯一定位與探測入口；正式 App 優先使用 `Contents/Resources/bin/`，開發執行才依環境變數、Homebrew 與 `PATH` 回退。
- `MediaSourceCompatibilityService` 讓所有匯入影片與音訊先經 `ffprobe` 分類；可直接播放時保留原檔，H.264／HEVC 優先只改封裝，其餘才以 VideoToolbox H.264／AAC 或 M4A AAC 建立播放代理。`AppStore+MediaImport` 將匯入與轉檔建立為可取消的 Job 並回報 FFmpeg 進度，轉碼位元率依影像面積調整。
- `AssetSchemeHandler` 對時間性媒體回應 HTTP Range，使用背景 `FileHandle` 以 512 KiB 分塊送出，並以停止任務集合避免 WebKit 已停止的請求再收到資料；圖片仍維持一次送出。啟動時 `ApplicationSupport.orphanMediaCacheFiles` 會找出沒有對應資產的 UUID 快取檔並清除。
- `MediaCompositionService` 建立不依賴模型的圖片循環影片與影音合併工作，沿用共用 FFmpeg 子行程、進度、取消及輸出命名；工作結果仍以 `generatedVideo` 資產和 `WorkflowOperation` 進入既有 lineage。
- `automatic-flow.js` 以宣告式範本進行 Profile 預檢並建立 Workspace/Tabs。步驟透過資產 ID 與來源 Tab 連結；第一個「簡單 MV」流程依序包含主視覺、背景音樂、圖片循環及影音合併。
- 相依套件修正由 `Patches/manifest.txt` 描述，統一交由 `scripts/apply-runtime-patches.command` 套用與驗證；版本不符或修正失敗時建置會停止，不會靜默使用未修正的來源碼。
- `GenImageMCP` 維持自行持有 `InferenceServices` 的獨立 stdio server；App 的 `LocalMCPServiceController` 只管理 localhost HTTP transport 的生命週期，兩者直接共用 `MCPServer` 工具核心，不形成 stdio→HTTP 代理。
- ACE-Step 正式 Runtime 的階段型別已移除 PoC 命名；診斷專用的 DiT probe 與正式生成階段分開，避免實驗程式與產品路徑混淆。

## Profile

App 先以內建目錄及保存的自訂 Profile 建立 UI，再由 `ModelDiscoveryController` 在背景讀取模型磁碟。控制器取消舊工作並驗證請求識別碼，只有最新目錄能發布結果；失敗不替換既有清單。成功時重新對應 Profile 選取與 LoRA，仍套用 64 GB 可見性規則。模型目錄掃描期間暫不接受生成、模型下載／修復或移除操作。

下載後驗證及手動修復透過 `BackgroundTask` 執行，取消會傳遞到背景工作；套用結果前再次檢查下載任務識別碼與根目錄。LoRA 清單使用獨立的掃描控制器，不因安裝單一模型而重新統計所有大型模型目錄。同步檔案系統呼叫仍須等待作業系統返回，但不占用 MainActor，取消後也不會套用舊結果。

模型操作由 `SerialTaskQueue` 按模型 ID 排序；替換工作先取消並等待前一個工作的所有清理，即使 UI 已顯示暫停，也保留其佇列尾端。不同模型可同時下載。切換模型根目錄前取消下載／驗證，並透過 `ModelDiscoveryController.beforeDiscovery` 等待全部模型操作結束。移除在背景執行且不能暫停；開始移除前確認沒有生成工作，移除結束前也不允許啟動新生成。目錄切換會等移除完成並更新舊狀態後才開始掃描。

`FileDownloadDelegate.start` 返回前會等待 URLSession 取消回呼與續傳資訊寫入完成；連線建立前收到取消也會記錄。下載結果先驗證大小，再移入目標磁碟的唯一暫存檔，最後使用同磁碟 rename 替換目的檔；搬移或替換失敗保留既有檔案。模型硬連結重用也透過 `ModelFileReplacement` 先建立暫存連結，失敗時保留目的檔並回退下載。HTTP 錯誤本文只讀前 2 KiB。


`InferenceProfile` 包含：

- 功能類型。
- 模型 ID。
- 模型 revision。
- 推論架構：MLX Swift、Core ML、本機服務或外部 CLI。
- 功能預設值。
- Profile revision。

`ProfileVisibility` 暫時隱藏任何必要模型的建議記憶體超過 64 GB 的 Profile，64 GB 保留；會比對模型 ID 與已登錄本機路徑。建議需求代表 Runtime 規劃值，不將分階段載入的元件相加；未知模型維持可見。UI、Profile 選取與啟動預設選取共用此規則，模型與原始 Profile 資料不刪除。

執行工作時，`WorkflowOperation.profileSnapshot` 保存完整值，而不是只保存 Profile ID。

內建 Profile 不直接修改。使用者需要變更時先複製，再儲存為新的 revision。

`CustomProfilePersistence` 將完整自訂 Profile 陣列保存至 UserDefaults 的 `GenImage.customProfiles.v1`，啟動時先還原定義，再還原啟用狀態。自訂 Profile 的選取簽章使用穩定 UUID，重新命名不會失去選取；複製會建立新 UUID 並保留音樂長度設定。資料無法解碼時保留原值並停用該次執行的自訂 Profile 保存。

## 資產與流程

`MediaAsset.parentAssetID` 表示媒體來源。沒有 parent 的資產是獨立工作的根節點：

- 獨立文生圖：生成圖片沒有 parent。
- 獨立圖生文：先匯入一張根圖片，描述輸出寫入 Recipe。
- 獨立 Upscale：先匯入一張根圖片，放大結果以原圖為 parent。
- 串接生成：生成結果以選取圖片為 parent。
- 獨立文生影：MP4 資產沒有 parent。
- 圖生影：MP4 資產以來源圖片為 parent。
- 獨立文生音樂：MP3、M4A、AAC 或 FLAC 資產沒有 parent，並記錄實際時長、取樣率與聲道數。
- 字幕生成：先以 `importedVideo` 或 `importedAudio` 匯入來源，SRT／WebVTT 結果以 `generatedSubtitle` 保存並以來源媒體為 parent。

`WorkflowGraph` 提供 lineage 與 children 查詢，UI 不需要推測資產關係。

`ProjectWorkspaceWriter` 在串行背景佇列合併 300 ms 內的工作區快照，原子存檔；App 結束通知會同步 flush 最後一份修改。

`ProjectWorkspacePersistence.restore` 區分檔案不存在與讀取／版本錯誤。發生錯誤時 App 保留原始索引、跳過孤兒快取清理並停用工作區自動存檔，顯示錯誤原因；需修復索引並重新啟動才能恢復保存，該次執行的工作區變更不會覆寫原檔。

`OutputFileNaming` 使用類型、分鐘時間戳與 UUID，避免尚未寫入的批次結果或不同服務取得相同輸出路徑。切換輸出目錄時同時更新推論、字幕與影音合成服務。 圖片、Upscale 與字幕 actor 使用具備鎖保護的 `OutputDirectoryStorage`，讓設定更新同步完成而不必排入 actor；每個工作開始時取得路徑快照，跨 await 後仍使用同一路徑。字幕依來源產生 sidecar 的規則及明確指定的輸出路徑優先權維持不變。

`MediaAssetFiles` 統一處理媒體檔案的保留與重新命名。移除檔案前檢查全部工作區的來源／播放 URL，仍有參照時保留檔案；自動清理只接受 MediaCache 內的檔案，並額外保護這次被關閉資產的來源。孤兒快取判斷同時檢查資產 ID 與檔案 URL，避免另一次匯入產生新 ID 後誤刪正在使用的檔案。刪除命名工作區也使用相同清理規則。

重新命名會更新所有指向同一個目錄項目的資產，保持 ID 與 lineage；只解析父目錄的符號連結，移動符號連結本身時不改掉直接使用其目標的資產。原生層在生成／匯入工作執行或取消中拒絕改名、移除資產及關閉結果分頁，Bridge 將錯誤傳給 UI，避免 UI 先移除分頁但原生操作未完成。

`WarmRuntimeWorker` 在行程退出後先排空日誌，再判斷是否缺少完成事件；`IncrementalLogReader` 回報是否仍有未讀資料，仍限制每輪 1 MiB、單行 64 KiB。只有確定寫入端已退出且讀到 EOF 才交付無換行的最後一行。普通子行程在啟動前、返回結果前都檢查取消，避免已取消工作仍執行程式或回報成功。

`AssetSchemeHandler` 每個請求保有不可逆的取消狀態；背景讀取的每個媒體區塊最多 512 KiB，等待主執行緒交付後才繼續讀取。停止後不再交付資料或完成回呼，handler 自身不會將整段影片累積在待交付佇列；WebKit 內部的媒體緩衝由 WebKit 管理。

開啟中的工作區分頁是生成專案的生命週期邊界。Swift 將 `Project`、`MediaAsset`、`WorkflowOperation` 與選取狀態以原子 JSON 快照保存至 Application Support；一般 App 結束不會清除。Web UI 的分頁狀態保存在 WebKit localStorage，關閉分頁時透過 Bridge 通知原生層移除該分頁資產與 lineage 索引，但不刪除已輸出的媒體檔。

命名工作區位於分頁之上，每個工作區維護自己的分頁集合。建立與刪除由 Bridge 進入 `AppStore+Workspaces`，刪除前必須確認；切換工作區只切換對應分頁與選取狀態，不重建 Runtime 或媒體播放器。

## 推論 Runtime

`SafetensorsHeader` 僅讀 8 bytes 長度與最多 16 MiB JSON，拒絕超過檔案或上限的標頭。`ZImageLoRAAdapterNormalizer` 改寫 LoRA A/B 鍵名後，以 1 MiB 區塊複製張量資料，透過 autoreleasepool 釋放區塊，並在每次複製前檢查取消。暫存檔完整寫入且來源大小／修改時間／檔案識別未變後，以原子 rename 發布；舊快取若標頭或檔案長度不符會重建。

`WarmRuntimeWorker` 的 stdin 設為非阻塞，遇到滿管線會短暫讓出執行權並檢查取消；寫入期限為 30 秒。送出請求期間仍維持單一任務限制，取消或失敗會終止該 Worker 並丟棄 session。

文生圖固定使用 `Z-Image.swift` commit `28bfcf3148c041a554629247170eb54d9ac46830`：

- macOS 14+、Swift 6。
- `ZImageGenerationRequest` 支援 Prompt、負向 Prompt、尺寸、步數、Seed、模型與 runtime options。
- `ZImageTextToImageService` 透過 `WarmRuntimeWorker` 與獨立 Worker 的 `--serve` JSON-line 協定通訊，使用 request ID 關聯逐階段進度與結果；Worker 持有並重用 `ZImagePipeline`。
- Worker 閒置 5 分鐘或記憶體壓力時卸載；若仍在生成，等工作結束後釋放。取消、失敗或長時間無回應會終止 Worker，下次請求重建；可重用 MLX buffer 上限為 512 MiB 或實體 RAM 的 1/16，取較小者。
- `IncrementalLogReader` 記錄檔案 offset 與未完成行，每次最多讀取 1 MiB，單行限制 64 KiB；log tail 以 seek 讀取末尾，不重讀整份日誌。
- Upscale 將每個 tile 寫入單一 bitmap，避免累積 Core Image 合成圖；H3 MP4 每次只轉換一格 CPU 圖片。完整的 Upscale 輸出畫布與 H3 解碼像素張量仍需留在記憶體中。
- 去噪迴圈會檢查 Swift Task cancellation。
- 支援模型卸載、LoRA 卸載、取消與記憶體快取清理。

圖生文與文生文的多模態路徑固定使用 `mlx-swift-lm` revision `7da33441c7c08b010ff1aa8da9dc3d82277272f5`：

- `QwenVLImageDescriptionService` 透過 `VLMModelFactory` 載入本機 Qwen3-VL。
- `QwenTextGenerationService` 使用相同多模態容器的純文字模式；Qwen3-VL、Qwen3.5、Qwen3.8 模型描述同時宣告 `.imageToText` 與 `.textToText`，各能力使用獨立 Profile。
- Qwen3.5 純文字輸入的上游相容修正由 `Patches/MLX-Swift-LM-Qwen35-Text-Only.patch` 套用。
- 受管理下載會取得並驗證 `processor_config.json`、影像／影片前處理設定、Tokenizer、Chat Template、權重檔與索引，避免只有文字權重而缺少多模態處理檔。
- 模型容器在服務生命週期內快取，避免同一 Profile 重複載入。
- 支援繁中、英文、日文、韓文輸出提示。

Upscale 由 `CoreMLUpscaleService` 執行 Real-ESRGAN 512 tile 與 4× 拼接。

影片由 `LTXVideoGenerationService` 啟動 App 隨附的 `GenImageLTXVideoWorker` Swift 子行程：

- 文生影與圖生影共用 `VideoGenerating` 與 `VideoGenerationRequest`。
- Swift 驗證 Profile、模型路徑、尺寸、幀數、FPS 與輸出數量。
- LTX-2.3 額外要求幀數符合 `8n+1`。
- JSON request 與逐階段 progress event 經由既有 `RuntimeProcess` 執行，支援 Task cancellation、日誌錯誤回報與百分比進度擷取。
- LTX LoRA 控制影片由共用 FFmpeg 層以 VideoToolbox H.264 建立，不依賴 GPL `libx264`。
- MP4 輸出以 `generatedVideo` 資產加入工作區，Web UI 使用原生 `<video>` 播放。
- Worker 編譯時放入 App Bundle 的 `Contents/Helpers`；開發建置可用 `GENIMAGE_LTX_WORKER` 覆寫，正式流程不依賴外部 Runtime。

音樂由 `MusicGenerationRouter` 依 `MusicRuntimeAdapter.supports` 分派，不在 Router 內集中硬編碼模型 ID：

- 文生音樂使用 `MusicGenerating`、`MusicGenerationRequest` 與 `MusicGenerationOptions`。
- `ACEStepMusicGenerationService` 使用 `.mlxSwift` Profile，直接呼叫 `ACEStepSwiftRuntime` 完成 Qwen3 Embedding、條件編碼、Turbo DiT、Euler sampler 與 Oobleck VAE，不啟動外部服務或 Process。
- ACE-Step 支援 10～300 秒、1～20 steps、可選歌詞與純音樂；latent 長度由 VAE 取樣率與 hop length 計算，長音訊使用重疊分塊解碼並串流寫入 PCM，以限制峰值記憶體。程式與模型採 MIT License。
- 音樂 Profile 以 `ProfileMusicConfiguration` 提供長度上下限及「目標／最長」語意，Web UI 不需要依模型 ID 判斷欄位行為。
- `MiniMaxMusic3GenerationService` 使用 `.externalCLI` Profile 啟動 App 隨附的 `GenImageMiniMaxMusic3Worker`；App 寫入 JSON request，Worker 以固定 bfloat16 production 路徑執行 Swift／MLX pipeline，5～300 秒參數代表最長長度，模型可在輸出音訊結束標記時提前自然結束。
- 8-bit 與 `mlx-community/MiniMax-Music3-4bit` checkpoint 共用同一 Worker。Worker 逐 frame 回報自迴歸進度、依 chunk×step 回報去噪進度、依 chunk 回報 vocoder 進度；App 沿用 `RuntimeProcess` 提供取消與強制終止，完成後由內建 FFmpeg 將 WAV 轉為使用者選擇的格式。MiniMax Music 3 不再需要 Python Runtime。
- `Mothersuperior/minimax-music3-composer-5.7b-distilled` 以 Model Center 的音樂元件列管理，預設只安裝 `lr-6e-5` 權重；它不是獨立 Profile，也不會被誤送入音樂生成 Service，必須等相容 Composer override Runtime 後才能套用。
- 兩個 Adapter 都固定取得暫存 WAV，再由 `AudioOutputEncoder` 經內建 FFmpeg 轉碼為 MP3 320 kbps、M4A AAC 256 kbps、ADTS AAC 256 kbps 或 FLAC 無損音訊。
- 成功、失敗或取消時清理暫存檔；只有完成的壓縮音訊會以 `generatedAudio` 資產保留，並記錄實際時長、取樣率與聲道數。
- ACE-Step 權重由模型中心管理，原生 Runtime 編譯於 App 內，不使用獨立安裝路徑或服務環境變數。

媒體資產同時保留 `fileURL` 與選填的 `playbackURL`：前者永遠指向使用者原檔，供字幕輸出、來源關係與明確刪除使用；後者只指向 `Application Support/GenImage/MediaCache` 內的相容代理，供 WebKit 預覽。移除資產或關閉專案時會清理代理，不會改寫原始媒體。

字幕生成位於多媒體匯入與文字模型之間：

1. `MediaAudioPreparer` 透過內建 `ffprobe` 確認音訊軌，再由內建 `ffmpeg` 統一解碼為 16 kHz 單聲道 PCM WAV 與暫存路徑。
2. `SubtitleGenerationRouter` 以 `MediaTranscribing.supports(profile:)` 依序選擇第一個相符 Adapter。
3. Whisper Large v3 Turbo 負責多語言辨識，Paraformer Large 負責中文，Parakeet 0.6B 負責日文；三者都在本機 Core ML 路徑執行。
4. 可選的 `QwenTextGenerationService` 以 Qwen3.5／Qwen3.8 MLX 對每批字幕翻譯，但不改變開始與結束時間。
5. `SubtitleDocument` 將結果寫為 SRT／WebVTT，再以 `generatedSubtitle` 資產加入目前工作區。

`GenImageASRPoC` 是獨立驗證執行檔，只測試 WhisperKit 的媒體解碼、語言辨識與時間軸輸出；不寫入 App 工作區，也不是主 App 的替代執行路徑。

`scripts/build-ffmpeg-macos.sh` 產生 Apple Silicon、LGPL-only、動態連結且可重新替換的 FFmpeg 發佈目錄。`build.command` 驗證未啟用 GPL／nonfree 編碼器，複製 `ffmpeg`、`ffprobe`、dylib 與授權文件，將 install name 改為 `@rpath`，依序簽署 dylib、工具與 App，再交由既有 DMG 公證流程處理。Developer ID 測試確認 FFmpeg dylib 與工具使用相同 Team ID 時不需額外 library-validation entitlement；本機 ad-hoc 建置則不對 FFmpeg 啟用 hardened runtime。預建二進位位於被 Git 忽略的 `third_party/ffmpeg/`，不進入 GitHub Source archive。

MLX metallib 依 MLX Swift 版本分開管理。主程式、Qwen、MiniMax Music 3 與 LTX Worker 使用 `mlx-swift 0.31.6`，建置時由 `RuntimeSupport/mlx-swift-0.31.6.metallib` 複製到各自的 Release 執行目錄；Z-Image Worker 因相依套件限制使用 `mlx-swift 0.30.6`，則使用 `RuntimeSupport/mlx-swift-0.30.6-zimage.metallib`，並放在 App Bundle 的 `Contents/Helpers/ZImage/`。`build.command` 會從各套件的 `Package.resolved` 讀取版本、檢查相容 Worker 版本一致，再以 MLX kernel 與 lockfile 指紋決定是否重建，避免不同版本共用或誤用未標版本的 `mlx.metallib`。發佈前仍需完成模型授權檢查與 16/24/32GB 壓力測試。

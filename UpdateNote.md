# GenMedia Update Notes

本文件採「最新內容在上」的方式維護。開發期間先更新 `Unreleased`；建立 GitHub Release 時，將標題改為正式版本與日期，再新增下一個空白的 `Unreleased` 區段。

## 1.26.0831 — 2026-08-31

Release 狀態：已完成 Developer ID 簽章、Apple Notarization、Staple 與 Gatekeeper 驗證。

- 新增 MiniMax H3 GGUF 純 Swift／MLX 影片 Runtime 與主系統分派，支援 H3 文生影及部分單圖生影 Profile；H3 目前處於測試階段，不保證生成結果或所有模型變體在不同硬體上的正確性。
- App 內的 Qwen、MiniMax Music 3、LTX 與 MiniMax H3 Worker 共用同版本 `mlx.metallib`；`Contents/Helpers/mlx.metallib` 使用指向 `Contents/MacOS/mlx.metallib` 的相對 symlink，Z-Image 的 `mlx-swift 0.30.6` 版本仍保留獨立檔案。
- 補齊 MiniMax H3 Worker 的 Developer ID 簽章、secure timestamp 與 hardened runtime，避免 Apple Notary Service 將 H3 可執行檔判定為無效。
- 新增 Profile 卡片的 `×` 暫時停用操作，點擊後使用自訂確認對話框，停用狀態可持久化並可重新啟用。
- 系統已移除絕大多數 Python 依賴，最終目標仍是完全移除；若遷移後仍有功能問題，會持續修正原生 Swift Runtime。

## Unreleased

- `build.command` 在第一次建立 App 或內建 FFmpeg 不完整時，會自動呼叫 `scripts/build-ffmpeg-macos.sh` 下載來源、建立 LGPL `ffmpeg`／`ffprobe` 與必要 dylib，不再要求使用者手動先執行建立腳本或安裝 `pkg-config`。
- LTX 正式流程移除外部 Python Runtime：`LTXVideoGenerationService` 現在只啟動 App 隨附的 `GenImageLTXVideoWorker` Swift 子行程，並由 `build.command` 複製與簽署 Worker 及其 Metal library。LTX Q4 模型安裝方案也會一併下載 Gemma 3 12B 文字編碼器，完整模型約 42 GiB，建議 48 GB 以上記憶體；原有 `~/Library/Application Support/GenImage/Runtime/ltx-2-mlx/` 舊資料不會自動刪除，確認不再使用後可手動移除。
- FFmpeg 建置支援含空白的專案路徑，並會清除外接磁碟產生的 AppleDouble dylib sidecar；失敗時保留原有可用版本，相關排除方式已加入四語 README。
- 新增獨立的 `RuntimeSupport/LTXVideoWorker`，完成 LTX-2 純 Swift Block 0–4：包含 Gemma 文字編碼、conditioning connector、兩階段 distilled denoise、影片／音訊 VAE、vocoder、FFmpeg 封裝、safetensors 權重載入、dtype parity 比較及純邏輯測試。正式服務已改用 Swift Worker，不提供 Python fallback；目前 image conditioning 與 LoRA fusion 仍明確回報不支援。
- LTX 影片 VAE 沿用 MiniMax Music 3 的混合精度策略：神經網路主幹維持 BF16，跨 tile 的 mask、加權、累加與除法固定使用 FP32，避免 BF16 在寫入 FP32 buffer 前提早捨入。
- MiniMax Music 3 新增真實多 chunk vocoder parity 診斷：Python `decode-chunks` 會從 AR frame hiddens 與 denoise 實際產生 latent chunks，Swift PoC 以相同 chunks 執行 `decodeChunks`；2 chunks 與 3 chunks 的 crop 參數、各段保留 samples 及最終 audio shape 全部一致。BF16 relative max diff 分別為 `1.571428571e-2`、`1.881720430e-2`，但改用 SNR 主指標後分別為 `36.0326 dB`、`37.0769 dB`，均通過 `>=30 dB` 門檻；同一批 3 chunks 改用 FP32 vocoder 後 SNR 為 `117.6912 dB`，通過 `>=80 dB` 門檻。差異分散於各 chunk 的 vocoder 輸出，不是裁切或 concat；正式 Decoder 實作未修改。
- 最終音訊 parity 改以長度無關的 SNR dB 為主要指標；`relative max diff` 與 `mean abs diff` 保留為輔助資訊。依真實案例分布，BF16 音訊門檻訂為 `>=30 dB`（目前最低 `33.9346 dB`，保留內容與長度變動緩衝），FP32 訂為 `>=80 dB`（目前 `117.6912 dB`，仍要求極低誤差能量）。
- 2 chunks 與 3 chunks 的真實 parity fixture 收錄於 `scripts/minimax-parity/fixtures/`；fixture 由 AR／denoise 實際產生，保存每個 latent chunk、Python 最終音訊與 crop 診斷，不使用合成 latent。
- `MiniMaxMusic3ChunkLayout.chunkFrames = 200` 明確指 `frame_hiddens` 的窗口，不是 vocoder latent 長度；condition encoder 會將完整 200-frame window 擴展為 689 latent frames，最後部分窗口則可能是 378 latent frames。

## 1.26.0827 — 2026-08-27

- 設定頁新增 Civitai API Token 管理：Token 只保存於 macOS Keychain，模型中心會以 HTTPS Bearer Authorization 直接下載 Civitai LoRA；移除 401 時開啟網站手動下載的流程，並保留 `CIVITAI_TOKEN` 作為舊版相容 fallback。
- MiniMax Music 3 Block 5 已落地：App 改由隨附的 `GenImageMiniMaxMusic3Worker` Swift 子行程執行 8-bit／4-bit checkpoint，透過 JSON request 與逐 frame、chunk×step、vocoder chunk 的真實進度事件工作；已移除 App 對 Python／`mlx_audio` 的依賴。舊的 Python Runtime 不會自動刪除，如不再需要可手動移除 `~/Library/Application Support/GenImage/Runtime/minimax-music3/` 與 `~/Library/Application Support/GenImage/Runtime/minimax-music3-mlx-audio/`，刪除前請確認沒有其他工作使用；可回收空間依本機內容而定，請以 `du -sh` 查詢。
- 新增 `mlx-community/MiniMax-Music3-4bit` 完整 MLX affine 4-bit 音樂模型與文生音樂 Profile；新增 `Mothersuperior/minimax-music3-composer-5.7b-distilled` Composer 元件下載方案，預設取用 `lr-6e-5`，並明確避免將 Composer 當成可獨立生成模型。
- AssetSchemeHandler 對影片與音訊加入背景 HTTP Range 分塊串流，支援漸進播放、拖曳與停止任務保護；媒體匯入與 FFmpeg 相容轉檔改走可取消、可回報進度的工作佇列。
- MediaCache 啟動時清理沒有對應資產的 UUID 孤兒檔；轉碼影片位元率依來源解析度調整，VideoToolbox 失敗訊息補充硬體編碼器與原始檔匯出／下載提示。Developer ID 實測確認 FFmpeg 使用同一 Team ID 簽章後不需額外 library-validation entitlement，已移除該檔案並保留 ad-hoc 本機建置相容性。
- 自動流程頁面由建置中骨架改為宣告式範本；首個「簡單 MV」會建立獨立 Workspace，並準備主視覺、背景音樂、圖片循環與影音合併四個相依 Tabs。分頁 schema 升級為 v3，每個 Tab 個別保存工作類型、Profile、Prompt 與各類參數，切換分頁或重新啟動不會互相覆蓋。
- 新增不依賴模型的圖片循環與影音合併工作。`MediaCompositionService` 透過內建 FFmpeg 支援多圖循環、單張秒數、總長度、解析度、FPS、Cover／Contain，以及音軌取代／混合、音量與長度策略；工作沿用既有 Job 取消、進度、`Video-YYYYMMDD-HHmm` 命名與資產 lineage。
- 影片與音訊來源全面接入內建 FFmpeg 相容層：匯入時以 `ffprobe` 取得 Codec、音軌、時間與旋轉後尺寸；可播放格式沿用原檔，H.264／HEVC 優先無損改封裝，其餘影片使用 VideoToolbox H.264／AAC、音訊使用 M4A AAC 建立播放代理。`MediaAsset` 分離原始 `fileURL` 與代理 `playbackURL`，字幕輸出與刪除仍以原檔為準，代理只存於 App 管理快取並隨資產或專案清理。
- 參考 PicViewer 導入 App 內建的 LGPL 動態版 FFmpeg 相容層：`MediaCompatibilityService` 統一 `ffmpeg`／`ffprobe` 定位、媒體探測、字幕音訊正規化、音樂輸出轉碼與 LTX 控制影片；正式 App 優先使用 Bundle 內工具，開發模式才回退環境與系統路徑。建置流程會封裝並依序簽署 dylib、工具與 App，預建二進位不進入 GitHub Source archive。
- 字幕翻譯目標語言由 5 種擴充為 25 種；App 與獨立 MCP server 共用 `SubtitleTranslationLanguage` 清單，避免介面、驗證與工具 Schema 不一致。
- 工作區刪除圖示改為符合主題的柔和紅色；媒體刪除對話框依安全順序排列「刪除檔案／取消／只移除」；MCP 未啟動時仍顯示 API 區塊並改為停用狀態。
- 媒體移除改為顯示三選一對話框：只從工作區移除並保留檔案、從工作區移除並刪除磁碟檔案，或取消；四語系介面同步更新。
- 字幕輸出改為優先寫入來源影片或音訊檔案所在的目錄，並沿用來源檔名、只替換為 `.srt` 或 `.vtt` 副檔名；若來源路徑不存在，才退回既有輸出目錄與時間戳命名。
- MiniMax Music 3 的預設推論步數由 30 調整為 20；新工作區與缺少步數欄位的設定會採用 20，已保存的使用者設定維持不變。
- Qwen3-VL、Qwen3.5 與 Qwen3.8 多模態模型同時歸類為圖生文與文生文，模型中心篩選與 Profile 各自對應兩種能力；下載流程同步取得並驗證 Processor、影像／影片前處理、Tokenizer、Chat Template 與完整權重索引。
- `mlx-swift-lm` 更新為支援 Qwen3.5 的 revision `7da33441c7c08b010ff1aa8da9dc3d82277272f5`，並由 `MLX-Swift-LM-Qwen35-Text-Only.patch` 修正 Qwen3.5 多模態 Runtime 的純文字輸入。
- 設定頁新增 MCP Switch；啟動後揭露 localhost API `http://127.0.0.1:12181/mcp`。HTTP transport 與獨立 stdio server 共用同一工具核心，stdio 仍自行持有推論服務並可在 GenMedia.app 未執行時工作。
- 新增影片／音訊字幕生成：`SubtitleGenerationRouter` 依 Profile 選擇多語言 Whisper Large v3 Turbo、中文 Paraformer Large 或日文 Parakeet 0.6B Core ML Adapter，輸出 SRT／WebVTT；可選用 Qwen3.5／Qwen3.8 MLX 在不改變時間軸下翻譯為 25 種目標語言。
- 新增命名 workspace 的建立、切換與確認刪除；每個 workspace 維護自己的生成專案分頁集合，切換時只替換對應 tabs 與選取狀態。
- 資產種類新增 `importedVideo`、`importedAudio` 與 `generatedSubtitle`，讓多媒體匯入、字幕 parent lineage 與圖片／時間性媒體判斷有明確型別。
- 新增 13 項不需模型權重的測試，涵蓋 SRT 時間碼與邊界、`AssetKind` 全 case 分類、字幕 Adapter 選擇、來源同名字幕輸出、16 kHz 單聲道換算、ASR 暫存輸出路徑與 MCP HTTP transport；原有 43 項加上新增項目共 56 項。
- 獨立 stdio MCP server 新增第七個工具 `genimage_generate_subtitle`，直接在 MCP 行程內執行 Core ML ASR 與可選 MLX 翻譯，不依賴 GenMedia.app 啟動。
- 除上述字幕與 workspace 能力外，本輪其餘項目為內部架構整理；既有生成流程與 Web Bridge 協定維持相容，變更集中在責任邊界、資料路徑一致性、可測試性與建置可靠性。
- `ImageAsset` 更名為 `MediaAsset`：圖片、影片與音訊一直共用這個型別，音樂生成也是回傳它，名稱與實際用途不符。63 處引用一併更新；欄位名稱不動，因此持久化 JSON 與 Web UI 的鍵名不變，既有資料不需遷移。
- `ACEStepSwiftRuntime` 內沿用 PoC 名稱的階段型別更名：`ConditioningPoC` → `ACEStepConditioningStage`、`GeneratedAudioPoC` → `ACEStepAudioGenerationStage`、`VAEDecodePoC` → `ACEStepVAEDecodeStage`、`QwenEmbeddingPoC` → `ACEStepTextEmbedder`。這四個是音樂生成的正式路徑（ACEStepMusicGenerationService → ACEStepNativeGenerator → 這裡），名稱看起來像實驗殘留，容易在清理時被誤刪。
- `DiTForwardPoC` 更名為 `ACEStepDiTForwardProbe`，並在檔頭註明它只供 `ACEStepSwiftPoC` 診斷使用；沒有搬進該執行檔，是因為它依賴的 DiT 型別都是 internal，搬出去得為了一個診斷工具把它們改成 public。
- `ACEStepSwiftPoC` 的模型搜尋路徑移除已作廢的 `GenMedia/Models`，與統一後的資料根目錄一致。
- Application Support 的兩個根目錄合併為一個：工作區索引原本寫在 `GenMedia/`，其餘（模型、Runtime、貼上圖片、預設輸出）在 `GenImage/`。位置改由 `GenImageCore/ApplicationSupport.swift` 統一定義，7 處各自組路徑的程式碼改為呼叫它。
- 啟動時自動把舊根目錄的內容接回現在的根目錄：只搬移目前沒有的項目，同名一律保留現有的（不覆蓋、不合併），搬空後移除舊目錄，可重複執行。升級後既有的工作區專案不會消失。
- 目錄名稱維持 `GenImage` 而非改為與 App 一致的 `GenMedia`：`Runtime/` 底下的 Python venv 把絕對路徑寫死在啟動 script 與 `pyvenv.cfg`，改名會讓已安裝的 LTX 與 MiniMax Runtime 直接失效，`Models/` 也可能是數十 GB。
- 刪除資產前的「是否為 App 管理檔案」保護改用同一個根目錄定義，先前只認得 `GenImage/`，寫在另一個根目錄下的內容不在保護與清理範圍內。
- 新增 `ApplicationSupportTests`（8 項），涵蓋子目錄一致性、管理範圍判斷、搬移、同名不覆蓋、空目錄清理、重複執行，以及本機實際分裂版面的完整情境。
- 外部 Runtime 子行程的共用流程集中到 `GenImageRuntime/SubprocessRuntime.swift`：可執行檔搜尋、子行程環境、log 檔讀寫、停滯偵測與執行迴圈只有一份。Qwen Image Edit Worker、LTX 影片、MiniMax 音樂與 ffmpeg 音訊轉檔共 5 個呼叫點改用同一套。
- 統一取消與錯誤時的終止語意：任何錯誤（取消、逾時、停滯）都會先 terminate、必要時 SIGKILL，再往外拋。圖生圖 Worker 原本只在 `CancellationError` 時終止子行程，其他錯誤會留下孤兒行程；取消時也只送 terminate，不保證結束。
- 可執行檔搜尋排除目錄：`isExecutableFile` 對目錄回答 true，PATH 內若有同名目錄會挑到永遠啟動不了的路徑。
- 移除 `Qwen2511ImageToImageService` 中只寫不讀的 `runningProcess` 屬性。
- 新增 `SubprocessRuntimeTests`（9 項）：以 `/bin/sh` 實際跑過成功、非零結束、onPoll 丟錯終止、取消終止四條路徑，以及 log 停滯偵測與可執行檔搜尋。這是 Runtime 目錄中少數不需模型權重即可執行的測試。
- `AppStore.swift`（2,586 行、87 個函式、0 個 MARK）依職責拆為 11 個檔案：主檔只留型別宣告、儲存屬性與 init，其餘進入 `AppStore+Persistence`、`+Paths`、`+Selection`、`+Profiles`、`+OutputSettings`、`+Assets`、`+ImageGeneration`、`+MediaGeneration`、`+Jobs`、`+ModelInstallation` 擴充。宣告數量拆分前後皆為 174 個，無增減。
- Swift 的 `private` 只到檔案範圍，因此 54 個跨檔案共用的成員改為 internal（仍侷限於 GenImageApp target），其餘 22 個維持 `private`；主檔開頭記錄了這個約定與檔案分工。
- Web UI 新增 `js/render-preservation.js`：全量重繪時的游標、選取範圍、捲動位置、播放中的音訊／影片節點與 Web Audio 視覺化圖，全部收斂到單一進入點 `withPreservedView(root, paint)`，取代原本散在 `render()` 內、順序錯了就會出錯的八次呼叫。
- Web UI 另拆出 `js/workspace-tabs.js`（分頁簿記與待落地輸出的分頁路由）與 `js/chrome.js`（側邊欄、路由、更新橫幅、系統資源列、對話框與 toast）；兩者都不讀取模組層級可變狀態，`state` 與 `ui` 一律由呼叫端傳入。`app.js` 由 2,182 行降至 1,360 行。
- 相依套件原始碼修正改由 `Patches/manifest.txt` 宣告、`scripts/apply-runtime-patches.command` 統一套用，取代 `build.command` 內兩組手寫的 grep／patch 流程；目前共管理 8 個 patch。
- 修正流程改為絕不安靜失敗：checkout 不存在、修正檔遺失、`Package.resolved` 版本與 manifest 不符、套用失敗、或套用後找不到預期標記，一律中止建置。舊流程在這些情況會直接以未修正的原始碼繼續編譯。
- `patch` 改用 `-i` 讀取檔案並加上 `-N -t -F 0`：不再互動詢問（舊流程用 stdin 餵 patch，遇到 `Assume -R?` 會把修正檔內容當成回答而可能反向套用），也禁止模糊比對避免貼到錯誤位置；成功後清除殘留的 `.orig` 備份。
- 新增 `scripts/apply-runtime-patches.command --verify`，可在不修改任何檔案的情況下確認 8 項修正是否都已套用。
- 輸出尺寸運算集中到 `GenImageCore/OutputGeometry.swift` 與 Web UI 的 `js/geometry.js`：對齊倍數、上下限、比例換算、來源尺寸換算與圖生圖生成畫布策略各只有一份定義，兩份實作互為鏡像並在檔頭互相標註。
- 圖生圖的尺寸策略由 Worker 移到 `Qwen2511ImageToImageService`：Worker 改為接收 `generationWidth`、`generationHeight`、`outputWidth`、`outputHeight` 並直接執行，不再自行決定生成畫布。MCP 走同一條服務路徑，因此同樣套用該策略。
- 影片輸出寬高改用就近對齊（原本無條件捨去），與 Web UI 滑桿一致；`GenerationRecipe`、`VideoGenerationRequest` 與 MCP 參數驗證改用 `OutputGeometry` 的上下限與倍數定義。
- 新增 `OutputGeometryTests`（12 項，涵蓋對齊、夾限、diffusers `calculate_dimensions` 對齊值與圖生圖生成計畫），並修正 `WorkflowGraphTests` 中因模型目錄成長而失效的斷言；`swift test` 由紅轉全綠（21 項）。
- 工作區底片列新增圖片匯入 ICON，支援 Finder 拖放一張或多張 PNG、JPEG、WebP、GIF、TIFF、HEIC 與 HEIF 圖片；音樂生成模式會停用圖片匯入按鈕。
- 圖片生成主按鈕會依目前是否選取有效來源圖片，自動路由至圖生圖或文生圖 Profile，避免圖生圖誤要求文生圖 Profile。
- 圖片與影片比例選項改為下拉選單；圖生圖有來源圖片時才顯示「原解析度」，並將來源尺寸換算為 16 倍數。
- 圖生圖指定寬高會真正傳入 Qwen Image Edit Runtime；來源比例與輸出比例不同時，先以邊緣延展補足為輸出比例的畫布，避免生成前裁切來源內容。過低的 `128 × 192` 仍可能降低細節與構圖穩定性。
- 修正圖生圖調整輸出解析度後輸出被裁切放大的問題：條件影像改以輸出解析度編碼，讓條件網格與輸出網格的 RoPE 位置一致；先前條件網格固定為 1024² 面積，網格較大時模型只會對準來源中央。
- 來源圖補畫布時保留自身解析度（上限為生成長邊或 1024），不再先縮到輸出尺寸，避免縮小後再由 Runtime 重新縮放造成的兩次取樣與細節流失。
- 修正低解析度圖生圖輸出崩解成條紋的問題：生成解析度與輸出解析度分離，指定面積小於 1024² 時以指定比例、1024² 面積生成（約 4096 latent token，與參考實作一致），再以 Lanczos 縮放輸出為指定尺寸；達到 1024² 以上則直接以指定尺寸生成。
- 圖生圖輸出面積小於 `512 × 512` 時，按下生成會先跳出警告對話框，提供「取消」與「仍要生成」；同一次執行期間確認過後不再提醒。對話框文案提供繁體中文、英文、日文與韓文。
- 音樂風格新增「動漫」，並同步提供繁體中文、英文、日文與韓文標籤；Prompt 留空時會使用 `Anime soundtrack` 作為生成風格。
- ACE-Step 1.5 已使用 App 內建的純 Swift／MLX Runtime，直接執行文字編碼、Turbo DiT、Euler sampler 與 Oobleck VAE，不使用 Python Adapter 或本機 Python 服務。
- 每個工作區分頁作為持久化生成專案；資產、lineage、操作與選取狀態會在 App 重新啟動後恢復，關閉分頁時才移除工作區索引。
- Prompt 與歌詞輸入在 Native 狀態更新期間保留游標、選取範圍與輸入法組字；生成類型及創作設定 TAB 改為創作面板區域渲染。
- 必要的完整 Web UI 更新會保留播放中的音訊、影片與 Web Audio 視覺化節點，避免切換 TAB 時中斷播放。
- 圖片、影片與音樂輸出統一使用 `Image-YYYYMMDD-HHmm`、`Video-YYYYMMDD-HHmm`、`Music-YYYYMMDD-HHmm`，同分鐘重複時加上流水號。
- 將實際產生的音訊範例放入 `Outputs/Music/`，供 GitHub 上的專案實例與檢查使用。產生的 macOS `._*` 中繼資料不會納入版本控管。
- 音訊預覽加入 Web Audio API 即時頻譜視覺化，播放時顯示頻率能量柱狀動畫，並保留原有音訊控制列。
- 音訊視覺化改用 8192 點 FFT，頻譜以 18 Hz–24 kHz 的對數頻率分配，上限會根據音源取樣率的 Nyquist 頻率自動限制；波形繪製上限為 1,600 點。
- 低頻對數頻譜改用分段中心頻率與 FFT bin 線性插值，避免相鄰柱狀重複讀取同一組 bin 而產生平台。

## 1.26.0815 — 2026-08-15

更新日期：2026-08-15
Release 狀態：已完成 Developer ID 簽章、Apple Notarization、Staple 與 Gatekeeper 驗證。

### 重點更新

- App 中文名稱與圖示更新為「剪影重生」，WebUI、App bundle 與系統顯示共用 App Icon。
- 工作區支援多分頁圖片列表、圖片縮放拖曳、輸出目錄快速開啟，以及 Prompt／負向 Prompt／輸出設定分頁。
- 文生圖、圖生文、圖生圖、文生影、圖生影與 Upscale 可獨立運作，並可透過資產 lineage 串接。
- Profile 依使用中、可用、下載中、不可用排序；可用 Profile 使用淡綠色外框，下載完成後即時重新排序。

### 模型與 Runtime

- 加入 Z-Image Turbo MLX 2-bit、4-bit、8-bit、Giniiki 4-bit，以及 Qwen3-VL 無審核內容描述 Profile。
- 擴充 `quantize_config.json` 轉換，支援 affine／mxfp4、packed pad token 與 FP16→BF16 載入。
- 修正 andrevp Z-Image Turbo MLX 4-bit 的 480／3840 維度錯誤與 FP16 溢位；已完成 256×256、9 steps 實際生成驗證。
- Z-Image Runtime 修正集中於 `Patches/`，`build.command` 在 Swift Package resolve 後自動套用。
- 文生圖完成後保留模型權重與暖機 buffer；閒置 5 分鐘後只修剪 MLX 暫存 buffer，不卸載模型。
- 側欄新增「釋放記憶體」ICON；有任務執行或取消中時停用。切換 Profile 且 RAM 超過 90% 時釋放非焦點 Runtime。
- 文生圖支援 LoRA，模型中心加入相容的 Hugging Face 與 Civitai LoRA；Civitai 需要登入時會提示並開啟下載網址。

### 下載與模型管理

- 改善 Hugging Face 分段並行下載、進度節流、取消／暫停與大型權重優先策略。
- 依遠端檔案清單校正實際總容量，避免估算值與下載進度超過 100%。
- 下載保留來源原始檔名；已安裝模型提供刪除按鈕與確認對話框。
- 下載中心恢復已下載區域，並降低下載期間主執行緒與 WebUI 重繪負擔。

### 工作佇列與操作

- 取消任務先進入 `cancelling` 並顯示 spinner，Runtime Task 結束後可靠地轉成 `cancelled`。
- 修正取消完成後側欄仍顯示執行中、生成與釋放記憶體按鈕持續停用的問題。
- 生成進度最多每秒同步一次；ETA 在 35% 且執行滿 15 秒後顯示數字，並加入整體耗時備援估算。
- 圖生文執行時停用 Prompt 與生成按鈕，完成後直接更新 Prompt。

### 輸出、系統與介面

- 輸出檔名統一為 `Image-MMDD-HHmmss` 或 `Video-MMDD-HHmmss`，並支援設定自訂輸出目錄。
- 剪貼簿圖片支援 Command+V／Control+V；長邊超過 2048 px 時先縮圖。僅在已啟用圖生文 Profile 時詢問是否排入圖生文。
- RAM／GPU 顯示以 60% 與 80% 為分界套用三種顏色，並採局部 DOM 更新避免干擾滑桿操作。
- 支援繁體中文、英文、日文、韓文與六套可持久化配色。

### 建置與驗證

- `build.command` 預設只建立 `.app`；使用 `--dmg` 才建立 DMG。
- 提供 `run.command`、`clean.command`、`backup.command` 與獨立 DMG 打包流程。
- Metal Toolchain 缺少時提供明確安裝提示。
- Swift 正式建置、13 項測試、JavaScript 語法、Shell 語法及 Git diff 格式檢查均已通過。

### 發佈驗證

- [x] 完成 Release DMG 打包與磁碟映像驗證。
- [x] 確認 Developer ID 簽章、Apple Notarization、Staple 與 Gatekeeper 狀態。
- [x] 建立 GitHub Release，附上 DMG、SHA-256 與安裝說明。
- [x] 將本節標題改為正式版本號與發佈日期。

## 1.26.0814

前一個 GitHub 版本。歷史內容請參考 Git tag `v1.26.0814`。

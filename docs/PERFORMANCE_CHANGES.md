# 記憶體與執行效率調整（2026-09-19）

## 行為

- Profile 依必要模型的 `recommendedMemoryGB` 篩選：大於 64 GB 隱藏、64 GB 保留。已登錄本機路徑使用同一規則；未知模型不猜測需求。保留原始 Profile 與模型檔案。
- Z-Image 以 JSON-line 常駐 Worker 重用模型，閒置 300 秒或記憶體壓力時釋放；生成中的壓力通知延至完成後釋放。取消、失敗與失去回應時終止 Worker，下次重建。MLX 可重用 buffer 上限為 512 MiB 與實體 RAM 的 1/16 之中較小者。
- 任務進度與系統資源使用輕量 Web Bridge，資產及字幕描述有快取；系統資源讀取與工作區存檔移到背景，存檔合併 300 ms 內的修改並在退出前 flush。
- 格線／底片列延遲載入最大 384 px PNG 縮圖；快取限制 16 MiB／256 張。完整預覽保留原圖。
- 日誌按 offset 讀新增資料，每輪最多 1 MiB、單行最多 64 KiB；錯誤訊息只讀檔尾。
- Upscale 每次重用一個 Vision 模型包裝，逐 tile 寫入單一畫布；H3 MP4 逐幀轉換並修正 BGRA 格式標記，避免額外保留整段 CPU 影格陣列。

## 驗證

以下為 2026-09-19 當時環境的歷史驗證；目前位置及最新結果見下方 2026-09-21 章節。

- 根專案 Debug 建置成功，108 項測試通過，涵蓋 Profile 邊界、工作區最後快照、縮圖方向、tile 像素一致性、增量日誌、Worker 重用／取消／失敗復原／閒置卸載。
- Z-Image Worker Debug 建置成功，使用 lockfile 的 mlx-swift 0.30.6，並套用既有四項 Z-Image 相依修正。
- H3 Worker Debug 建置成功，影片寫入 5 項測試通過，包括 152 幀含音訊影片、24 幀張量輸出的解碼影格數、尺寸、時長與逐幀色彩。
- Web UI 的兩項 Node 測試通過，涵蓋進度更新保留本機編輯與媒體參照、工作及安裝階段切換刷新控制項；所有 JavaScript 檔案通過語法檢查。

當時使用的驗證指令（SwiftPM 預設建置系統）：

```sh
swift test -j 4
swift build --package-path RuntimeSupport/ZImageWorker -j 4
swift test --package-path RuntimeSupport/MiniMaxH3Worker -j 3 --filter MiniMaxH3VideoWriterTests
node --test Tests/WebUI/*.test.mjs
```

當時測試曾將既有 Debug MLX Metal library 放在 `.xctest/Contents/MacOS/`。目前工具鏈會驗證該目錄內的程式碼簽章，請使用下方最新驗證方式；重新產生 library 需要可用的 `metal` 與 `metallib`。

本輪未進行大型模型的完整生成與 RAM／耗時基準測量；不宣稱特定效能提升百分比。H3 完整解碼張量與 Upscale 最終畫布仍需記憶體，64 GB 的 Profile 篩選不代表所有尺寸與片長均可在 64 GB 內完成。

## 2026-09-20 建置流程清理

- 移除外接磁碟專用的複製限制、AppleDouble 反覆清理與 DMG 中繼資料轉換。App／DMG 暫存位於專案 `dist/`，備份暫存位於 `Backups/`，FFmpeg 原始碼快取位於 `.build/ffmpeg-source`。
- 一次合併並清除 60,893 個 AppleDouble 附屬檔案；Git 索引警告已消失，連通性檢查通過。SwiftPM 二進位相依快取中的舊專案路徑已更新，套件版本保持不變。
- 六支 shell 腳本語法檢查、備份 ZIP 完整性與快取排除、FFmpeg 打包／執行／dylib 簽章，以及 App 替換／失敗還原均通過。
- 一般 `swift build --product GenImage -j 4` 已進入編譯，但 Metal 階段因缺少 Xcode Metal Toolchain 而失敗，未宣稱完整建置通過。可用 `xcodebuild -downloadComponent MetalToolchain` 安裝；`build.command` 也保留自動安裝此必要工具的流程。

## 2026-09-21 背景掃描與 LoRA 記憶體修正

- 啟動、模型目錄切換及 LoRA 清單掃描改在背景執行。連續切換目錄時舊結果不得覆蓋最新目錄；掃描失敗保留既有清單。尚未完成掃描時不寫掉保存的 LoRA 選擇；自訂 Profile、停用狀態、穩定選取與 >64 GB 篩選保持有效。
- 下載後驗證與手動修復驗證離開 MainActor；取消傳遞至背景工作，檔案驗證迴圈會檢查取消。結果發布前重新檢查任務識別碼與模型根目錄，避免過期結果污染狀態。
- LoRA 相容性檢查只讀 safetensors 標頭，JSON 上限 16 MiB。格式轉換改以 1 MiB 分塊複製權重，在完整寫入後原子替換快取；不再組合包含整個模型的 Data。
- 常駐 Worker stdin 改為非阻塞。滿管線等待期間仍能處理取消，寫入期限 30 秒；失敗的 session 不重用。
- 修正 `abs(optionalSize ?? 0 - resolvedSize)` 的運算優先順序，讓下載大小未變時不再更新整份模型清單；延遲的下載進度不得將驗證階段改回下載中。Profile 切換重用每秒背景更新的資源讀值。

### 合成記憶體量測

使用相同 256 MiB 稀疏 safetensors 檔、同一 Debug 工具鏈與獨立程序，分別執行修改前後的 LoRA 格式轉換，以 `/usr/bin/time -l` 量測：

| 指標 | 修改前 | 修改後 |
| --- | ---: | ---: |
| 最大 RSS | 546,717,696 bytes（521.39 MiB） | 12,173,312 bytes（11.61 MiB） |
| Peak memory footprint | 271,778,776 bytes | 5,309,016 bytes |
| 輸出檔案大小 | 268,435,552 bytes | 268,435,552 bytes |

這是單次合成格式轉換測試，不是實際模型推論或整個 App 的記憶體基準；未以此宣稱推論加速。另一個使用非零權重的回歸測試確認轉換前後張量 payload 完全一致，且原始檔不變。

### 驗證

- 完整根套件編譯成功；122 項 Swift 測試通過（Core 53、Runtime 52、MCP 6、GGUF 9、App 2）。
- 新測試涵蓋慢掃描晚於新掃描完成、取消後不發布結果、8 GiB 稀疏權重檔只讀標頭、非法長度、分塊複製內容一致、損壞快取重建，以及 Worker 完全不讀 stdin 時的取消。
- Worker 滿管線案例送入 512 KiB 請求，要求取消至返回少於 2 秒；原先的同步寫入會等替身程序 5 秒後結束才返回。
- 實際專案位於 ExFAT 磁碟；完整建置在內部 APFS 磁碟的暫存副本執行。未恢復正式腳本的外接磁碟處理。
- 驗證先執行 `swift build --build-tests -j 4`，將本次產生的 `.build/out/Products/Debug/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib` 複製至暫存套件根目錄，再在該目錄執行 `swift test --skip-build -j 4`。

仍待實機量測：完整 App 啟動與大型模型磁碟斷線／重連、真實模型生成、Profile 操作時的長時間磁碟延遲。此輪尚未處理模型刪除與 Profile 的小型檔案檢查；模型刪除已於下方後續修正移到背景，並未宣稱所有檔案操作均已離開 MainActor。

日誌：`genimage-followup-tests.log`、`genimage-followup-final-build.log`、`genimage-lora-memory-old.log`、`genimage-lora-memory-new.log`。暫存副本與本次 `.bak` 在最終驗證後移除，其他備份保留。


## 2026-09-21 模型操作與輸出目錄一致性修正

- 以 `SerialTaskQueue` 按模型 ID 排序下載、驗證、修復與移除。新操作會等待已取消的舊任務完成清理，不同模型仍可並行；在佇列中被取消的任務也會結束 UI 的進行中狀態。
- 模型移除透過 `BackgroundTask` 執行。生成與移除互斥，移除期間不能暫停或重複操作同一模型。切換根目錄先停止下載／驗證並等待清理；已開始的移除則完成並更新舊狀態後才掃描，避免新路徑無效時留下錯誤的可用模型狀態。
- `FileDownloadDelegate` 記錄連線建立前的取消；`start` 等 URLSession 完成與取消回呼的續傳資訊寫入都結束才返回，避免暫停／續傳／移除互相覆寫。
- 下載及硬連結重用先準備目標磁碟上的唯一暫存檔，再透過 `ModelFileReplacement` 原子替換。大小不符、來源消失、目的地是目錄或替換失敗時保留舊內容；下載 HTTP 錯誤本文最多讀取 2 KiB。
- 圖片、Upscale 與字幕服務使用具備鎖保護的輸出路徑儲存，設定同步完成；每個生成工作取得一次快照，進行中的工作不因目錄切換改寫目的地。字幕 sidecar 與明確指定輸出路徑的優先權保留。

### 驗證

- 完整根套件 Debug 編譯成功；**130 項 Swift 測試通過**：Core 56、Runtime 57、MCP 6、GGUF 9、App 2，包含參數化案例。
- 新增 8 項測試：同模型取消／後續操作順序、不同模型不互相阻塞、排隊取消清理、掃描等待移除，以及字幕執行途中切換輸出目錄；其餘下載與硬連結案例詳見下方。
- 下載測試使用真正的 URLSession 與本機 HTTP fixture，涵蓋開始前取消、中途取消、成功替換、大小不符與目的地目錄保留；取消返回後再寫入新續傳標記，確認沒有延遲回呼覆蓋。
- 硬連結測試涵蓋成功重用且 inode 相同、來源消失及目的地目錄保留；失敗與成功均不留下暫存檔。
- 實際專案仍位於 ExFAT，完整編譯／測試採內部磁碟暫存副本，沒有修改正式建置腳本。已核對 159 個來源、測試、資源及套件設定檔，與測試副本內容一致；`git diff --check` 通過。
- 日誌：`genimage-lifecycle-final-build.log`、`genimage-lifecycle-tests.log`。驗證成功後移除本輪 `.bak` 及約 3.5 GiB 的完整建置暫存副本，保留其他備份。

未執行真實模型下載／生成、完整 WebKit 操作、長時間外接磁碟斷線／重連或 Release App／DMG 公證。此輪沒有新的實際推論速度或 RAM 量測。


## 2026-09-21 媒體檔案保護與 Worker 結束處理

- 常駐 Worker 結束後先讀完已寫入的日誌，再判斷是否缺少完成事件。保留每輪 1 MiB／單行 64 KiB 上限；只有寫入端確定退出且到達 EOF 時，才交付沒有換行的最後一行。一般子行程在啟動前與返回結果前檢查取消。
- `MediaAssetFiles` 在刪除前檢查全部工作區的來源及播放參照；仍被使用的檔案會保留並顯示原因。重新命名會更新所有指向同一路徑的資產，保留 ID 與 lineage；父目錄符號連結可對應同一檔案，移動符號連結本身則不更動直接引用目標的資產。
- 自動媒體清理範圍縮小為 MediaCache，保護來源檔及共用代理。啟動時同時檢查實際檔案 URL 與資產 ID，避免同一快取再次匯入後因新 ID 被判成孤兒。刪除工作區時也回收不再使用的播放代理。
- 原生層在生成、媒體匯入或取消中拒絕改名、刪除媒體及關閉結果分頁；Bridge 傳回錯誤，UI 不會在拒絕後繼續移除對應分頁或資產。

### 重現與驗證

- 修改前的隔離重現有 3 個失敗案例：預先取消仍嘗試啟動不存在的程式；Worker 寫入 3 MiB 日誌並以 exit 0 結束時，無論最後事件是否換行都被判為失敗。修正後全部通過。
- 完整根套件 Debug 編譯成功；**141 項 Swift 測試通過**：Core 64、Runtime 60、MCP 6、GGUF 9、App 2。本輪新增 11 項測試，另含參數化案例。
- 檔案測試涵蓋共用來源／播放檔、最後參照移除、快取範圍與符號連結、跨工作區改名、目錄及無效名稱拒絕、既有目的檔保留、重複匯入快取的啟動清理。
- 完整測試曾觀察到既有 Worker 重用測試偶發提早換行程，診斷執行未再重現，沒有將其認定為已證實的產品缺陷。原測試同時啟用 0.4 秒閒置回收與系統壓力通知，無法保證重用前提；已分開驗證重用與閒置回收，測試隔離外部壓力通知，並等待行程確實退出而非固定睡眠。正式程式仍預設啟用記憶體壓力卸載。
- 實際來源、測試、資源及套件設定共 161 個檔案，與測試副本逐檔核對一致；`git diff --check` 通過。ExFAT 工作區的完整編譯沿用內部磁碟暫存副本，沒有修改正式建置腳本。
- 日誌：`genimage-media-reproduction.log`、`genimage-media-final-build.log`、`genimage-media-final-tests.log`。成功後移除本輪 `.bak` 與測試副本，保留其他備份。

此輪未測量真實模型推論速度或峰值 RAM，也未執行完整 WebKit 操作、磁碟斷線／重連及 Release App／DMG 發佈。媒體檔案操作仍同步執行；本輪重點是檔案與參照一致性，沒有宣稱所有磁碟操作已離開 MainActor。


## 2026-10-03 檔案讀取與 Qwen 2.1 固定資料重用

本次保持 UI、操作方式、生成參數、模型精度與功能不變，沒有修改 WebUI 資源。

- WAV 資訊改以 `FileHandle` 讀取 RIFF／chunk 標頭及 16 bytes 的格式資料，跳過音訊與未知 chunk，不再載入整份 WAV。保留原有 chunk 順序、奇數長度 padding、最後有效 fmt/data 優先及不完整尾端的處理方式。
- 原始圖片與其他非影音預覽使用與影音共用的分段傳送，每段至多 512 KiB，等待主執行緒接收後才讀下一段；每段釋放 Foundation 暫存物件，取消後停止讀取與回呼。MIME、完整內容、縮圖快取及影音 Range 行為保持不變。這限制的是原生傳送端的暫存，WebKit 仍需要圖片解碼及顯示記憶體。
- Qwen-Image 2.1 在每張圖片開始去噪前，準備文字投影、索引、RoPE cos/sin 與時間頻率，供同張圖片的後續步驟重用。RoPE 的固定分母只計算一次。latents、時間嵌入、調制、注意力與 Euler 更新仍逐步計算；保留 BF16／FP32 運算順序與每步 MLX 快取清理。準備資料隨去噪階段結束釋放，不跨工作保存。

### WAV 記憶體量測

以修改前後的實際解析函式建立獨立 Swift `-O` 程序，讀取相同 256 MiB PCM WAV，用 `/usr/bin/time -l` 記錄：

| 指標 | 修改前 | 修改後 |
| --- | ---: | ---: |
| 最大 RSS | 274,563,072 bytes（261.84 MiB） | 6,127,616 bytes（5.84 MiB） |
| Peak memory footprint | 270,582,408 bytes | 1,950,152 bytes |
| 解析結果 | 48,000 Hz、2 聲道、1398.1013333333333 秒 | 完全相同 |

這是 WAV 資訊解析的單次合成測試，不代表整個 App 或模型推論的記憶體降幅。

### 小尺寸實際生成比對

使用已安裝的 Qwen-Image 2.1 MLX 4-bit 模型、256×256、seed 42：

- 文生圖：20 步，提示詞 `A red apple on a white table, studio photograph.`，修改前後 PNG SHA-256 均為 `2336b785ffa68663249b92d556e45beb8f6ec0ef8c7f34dcbfddd4197ba9af42`。
- 圖像編輯：以上述文生圖作為輸入，3 步，提示詞 `Make the apple green.`，修改前後 PNG SHA-256 均為 `46a370da84fc8b196d29ea0859df163ce45dbf05fd0a42f2085a279ff1e0f7ae`。此案例驗證執行與輸出一致性，不用於評估編輯品質。
- 兩個案例的輸出檔案均逐位元組相同。觀察到的單次總時間差距很小，且修改後執行期間另有編譯工作，未據此宣稱整體推論速度或模型峰值記憶體有明顯改善。

量測請求、日誌、解析函式量測來源與圖片保存在本機 `Outputs/performance-2026-10-03/`（Git 忽略）。


### 編譯與回歸測試

- 根套件 Debug 及測試目標編譯成功（`swift build --build-system native --build-tests -j 4`）；`GenImage` 與 `GenImageQwen21Worker` 的 Release 產品以預設建置後端各自編譯成功。
- 相關 **21 項測試、6 個 suite 全部通過**，指令為 `swift test --build-system native --skip-build --no-parallel -j 4 --filter 'AudioOutputEncoderTests|AssetSchemeHandlerTests|QwenImage21RuntimeTests'`。測試時提供本次編譯的 MLX Metal library，驗證後移除根目錄的暫存副本。
- 新增驗證涵蓋 PCM／float WAV、延伸 fmt、chunk 順序與 padding、重複有效 chunk、512 MiB 稀疏音訊、損壞及空資料；原始圖片完整 bytes、512 KiB 傳送上限、回應後／首段後取消；Qwen 獨立 attention oracle，以及同一準備資料跨不同 sigma／latents 重用的輸出一致性。
- **完整套件測試未全部完成**：既有 `WarmRuntimeWorkerTests.reusesProcessAndUnloadsAfterIdle` 遇到 60 秒逾時，取樣顯示卡在 Foundation `Process.waitUntilExit()`；Runtime 分組重試也在 `Qwen21ServiceTests.workerProtocolPreservesLinksAndCleansFailedBatches` 的程序等待停住。原因尚未確認，本次沒有修改這些程序管理程式，不將相關測試記為通過。已停止本次卡住的測試行程，保留日誌與堆疊取樣供後續追查。
- `git diff --check` 通過；WebUI、操作參數與模型設定檔均未修改。本次 `.bak`、舊編譯快取備份與大型合成測試檔在驗證後移除。

主要日誌：`test-native-final-build.log`、`tests-targeted.log`、`app-release-build.log`、`qwen-final-build.log`、`tests-final.log`、`tests-runtime.log`，以及兩份 `*-sample.txt`，皆位於前述本機量測目錄。

## 2026-10-03 函式層級最佳化

本輪以開始修改時的工作區內容作為基準，保留先前的新模型整合。UI、操作流程、生成參數與模型精度保持不變。

- `Qwen21Transformer.callAsFunction`：每個去噪步驟先在兩列調制資料上完成 `1 + scale` 與 `tanh`，再展開至 token；所有 Transformer 區塊共用已實體化的結果，省去各層重複相加，並縮小 `tanh` 的運算範圍。步驟相關資料仍於每步重新計算。
- `LTXGemma4TextEncoder.allHiddenStates`／`PreparedRotary`：同次文字編碼的 Q、K 與各層共用 local／global 兩組位置表，保留 FP32 旋轉及原 dtype 輸出。位置表只存在於單次呼叫中，不跨提示詞保存。
- `LTXMediaEncoding.wav`／Worker `writeWAV`：直接在最終 WAV 緩衝區寫入 PCM16，合併有限值檢查與取樣轉換。連續的 Float32 輸入借用 MLX 記憶體；非連續輸入仍使用安全的連續副本。保留裁切、朝零截斷、little-endian、聲道交錯及原子寫檔。
- `rgb24Frame`／`writeVideoFrames`：由 MLX 將非連續影格整理為連續排列，兩條影片輸出路徑共用 byte 轉換，每次封裝重用一個 RGB `Data`。保留原本逐像素取整、非有限值拒絕、影格順序及進度回報；借用的 MLX 資料不會逸出張量生命週期。

### 函式量測

Apple M4、16 GB RAM，Swift Release 編譯。相同輸入先暖機，新舊版本交替量測 9 次，以下為中位數；新版本的位置表準備時間也包含於 RoPE 量測。

| 範圍 | 修改前 | 修改後 |
| --- | ---: | ---: |
| 10 秒、48 kHz 雙聲道轉 PCM16 WAV bytes | 11.52 ms | 1.20 ms |
| 同一 256×256 影格轉 RGB24，重複 9 次 | 9.91 ms | 5.77 ms |
| 48 層 Q/K RoPE，64 tokens、Q 2 heads／K 1 head | 18.09 ms | 13.85 ms |

上述為函式合成量測，未包含模型去噪、檔案寫入或 FFmpeg；不能換算成整個 App 或文生圖／文生影的加速倍率。RoPE 使用 local 256／global 512 維與原有 partial rotation，未包含 attention 或線性投影。

以 10 秒雙聲道為例，連續 Float32 WAV 路徑省去 3,840,000 bytes 的 Swift 浮點副本與 1,920,000 bytes 的中間 PCM 副本；這是移除的資料容量，不是程序峰值記憶體量測。影格轉換仍需 MLX 的排列緩衝區及最終 RGB 緩衝區，不宣稱零記憶體配置。

### 數值與成品驗證

- 根套件 Debug 測試執行成功，回報 199 項／50 個 suite。LTX 套件 Release 回歸測試執行成功，回報 50 項／7 個 suite，其中 4 個需指定模型環境的案例維持條件略過；真實模型比對另外執行如下。
- 新增回歸涵蓋 WAV 標頭、PCM16 裁切／截斷／交錯、單聲道、非連續張量、RGB 所有取整邊界、緩衝區重用／保留舊副本、NaN／Inf 拒絕，以及 FP32／BF16／FP16 的完整與部分 RoPE。既有 Qwen 與 Gemma4 獨立數值 oracle 亦通過。
- 使用 LTX-2.5 真實 Gemma4 4-bit 權重，比較修改前後 48 層、49 組 hidden states；5 個 token 含 2 個左側 padding，轉成 Float32 後逐個位元模式完全相同。
- Qwen Turbo 6 步、256×256、Seed 42 的 PNG 逐位元組相同，SHA-256 均為 `4154ab965a87529ca02e3fa304708e90aed5cea2115d5e29a83edb0b23871400`。
- LTX 使用同一份先前實際生成的 latent，分別經新舊 Worker 解碼與封裝。兩份影片都是 256×256、9 幀、24 FPS、AAC 48 kHz 雙聲道、0.375 秒。AVFoundation 逐幀解碼的像素雜湊與時間戳全部相同，AAC 封包也逐位元組相同。MP4 有 2 bytes 的 VideoToolbox SEI 附加資訊差異，因此**不宣稱整份 MP4 檔案逐位元組相同**。
- `GenImageQwen21Worker`、`GenImageLTXVideoWorker` 的正式 Release 產品均建置成功，`git diff --check` 通過。

本輪沒有重新執行 LTX 完整去噪、長片或高解析度生成，也未打包／發布 Release。成品比對期間另有編譯工作，因此不以這些單次執行時間比較整體推論效能。

量測原始來源、當時工作區基準、9 次交替量測、請求、圖片／影片、逐幀雜湊與日誌保存於本機 `Outputs/function-optimization-2026-10-03/`（Git 忽略）。比對用的臨時測試程式已移出測試目標，永久保留媒體轉換及 RoPE 的回歸測試。本輪 `.bak` 於驗證成功後移除；未同步 GitHub。


## 2026-10-03 全專案函式檢查與最佳化

本輪基準為開始時已包含前述模型整合及函式最佳化的工作區，沒有回復到 Git HEAD。檢查涵蓋第一方 Swift、WebUI JavaScript、各 Worker、工具與測試；檔案清單約 328 份，排除 `.build`、第三方套件、模型與產生的成品。宣告掃描僅用於定位函式與呼叫路徑，不代表每個函式都需要改寫。

### 各範圍的處理

| 範圍 | 本輪結果 |
| --- | --- |
| Core／App | `WorkflowGraph` 建立首筆 UUID 索引，查找為平均 O(1)，lineage 改為 O(深度)；保留重複 ID 首筆優先、循環保護及值語意。Profile 清單一次建立超過記憶體門檻的模型 ID／本機路徑集合，避免每個 Profile 重掃模型、重複標準化路徑；搜尋字串只 trim 一次。 |
| WebUI 工作區 | `reconcileWorkspaceTabs` 在一次對帳內共用素材歸屬、分頁與完成工作索引，新增輸出時立即更新歸屬；Shift／多選共用 image ID 集合；lineage 以 append 後 reverse 建立。保持分頁優先順序、選取行為、素材順序及生成輸出的分頁歸屬。HTML、CSS、介面文字與橋接格式未改。 |
| 圖生文 Runtime | `DescriptionTextStatistics` 只掃描新收到的 Unicode scalars，保留跨 chunk 的重複字元狀態；不再每次重建整段文字的 scalar 陣列。原本的最短字數、標點比例、重複字元、重試與停止時機均保留。 |
| GGUF／LTX／Music 3 | 三處量化解碼以 Float 陣列直接建立 MLX 張量，移除中間 `Data(bytes:)` 的完整 Float32 複本；量化公式、型別及張量形狀不變。 |
| MiniMax H3 | LoRA 套用完成後先計算一次文字 refiner，所有去噪步驟共用；在載入 VAE 前結束其作用域。sigma、AdaLN、條件噪音與音／影排程仍逐步計算。 |
| MiniMax Music 3 | 每個音樂區塊準備一次 RoPE 與無條件輸入，conditional／unconditional 推論共用；既有未提供預先計算的位置表的呼叫仍可使用。WAV 直接寫入最終 PCM 緩衝區，有限值檢查與取樣轉換合併。 |
| ACE-Step | 完整 WAV 路徑借用可連續讀取的 MLX 浮點資料，省去 Swift 浮點、sanitized 浮點及中間 PCM 副本。串流 append 使用自有 Data 清理非有限值，finish 以定長緩衝區轉換，保持原本暫存檔與 256 KiB 讀取上限。 |
| Qwen 2.1／Gemma4／LTX 影音 | 保留上一輪已驗證的調制、RoPE 與 RGB/WAV 最佳化；本輪只另改 LTX GGUF 的中間拷貝。 |
| Qwen 2511／Z-Image | 第一方 Worker 主要為參數驗證、模型生命週期及既有套件呼叫；沒有證據支持再次改動推論公式或跨請求保存 GPU 張量，維持既有實作與釘選依賴。 |
| MCP、下載、子程序、字幕、媒體服務、建置工具 | 檢查請求處理、日誌讀取、檔案與影音輸出路徑；保留既有範圍讀取、分塊傳輸、縮圖快取及取消處理，避免引入過期的檔案快取。啟動／維護腳本回歸通過。 |

App 的 `sourceImages(for:)` 曾比較只索引所選素材的版本；在 5,000 筆素材、32 個來源的測量中，原版約 0.229 ms，候選版約 0.277 ms，因此保留原本實作。沒有為了增加修改數量而留下未證實有效的改寫。

### 函式基準量測

Apple M4、16 GB RAM；Swift 使用 `-O`，MLX 0.31.6；JavaScript 使用本機 Node。新舊版本先暖機兩輪，再交替測量 9 次，下表為中位數。原版來源取自本輪開始時的快照，兩版使用相同輸入。

| 函式／輸入 | 修改前 | 修改後 |
| --- | ---: | ---: |
| 工作區對帳：5,000 筆素材、40 分頁、2,500 個連續操作 | 130.65 ms | 1.68 ms |
| Shift 範圍選取：5,000 筆素材 | 42.56 ms | 0.55 ms |
| WorkflowGraph 建立索引與 5,000 層 lineage | 14.47 ms | 1.62 ms |
| Profile 可見性：83 個 Profile、67 個附本機路徑的模型 | 8.48 ms | 0.030 ms |
| 串流文字品質檢查：128 chunks／2,176 scalars | 5.19 ms | 0.078 ms |
| ACE-Step PCM16 WAV：10 秒、48 kHz 雙聲道，含寫檔 | 12.94 ms | 4.48 ms |
| Music 3 PCM16 WAV：10 秒、48 kHz 雙聲道，含寫檔 | 12.47 ms | 2.33 ms |
| ACE-Step 串流 WAV：10 個一秒區塊，含暫存檔與輸出 | 21.70 ms | 4.08 ms |
| 已解碼 GGUF Float 陣列建立 MLX 張量：16 MiB | 0.755 ms | 0.350 ms |

工作區壓力案例不包含瀏覽器繪製與 localStorage 實際 I/O（儲存函式以 stub 代替）；Graph 包含建構索引成本。GGUF 數字不含讀檔、量化解碼或完整模型載入。H3、Music 3 的模型計算只做數值一致性回歸，沒有把小型測試的速度換算成大型模型的生成加速倍率。所有表格數字均為特定函式的測量，不能當作整個 App 或文生影的效能承諾。

### 記憶體與輸出規則

- 16 MiB 的解碼 Float 陣列轉成 MLX 時，省去一份 16 MiB 的中間 Data；原有 Float 陣列和 MLX 張量仍存在。
- 10 秒／48 kHz／雙聲道 ACE-Step 完整 WAV 路徑，移除兩份各 3,840,000 bytes 的 Swift 浮點陣列與一份 1,920,000 bytes 的中間 PCM Data。最終 WAV 與 MLX 輸入仍需記憶體。
- Music 3 移除兩次全長浮點陣列拷貝，每次在上述輸入下為 3,840,000 bytes；兩次拷貝並非同時存在，不能將容量相加當成峰值節省。
- 上述是移除的中間資料容量，未量測整個 App 的峰值 RSS。非 Float32 或非連續張量仍可能需要型別轉換／連續副本。借用資料限定在張量生命週期內；串流清理非有限值使用自有副本，不修改輸入張量。
- 保留各模型原有 PCM 規則：ACE-Step 非有限值補零及 peak-based gain；Music 3 拒絕非有限值、負滿幅為 -32768；LTX 既有朝零截斷不變。沒有將這些不同規則合併。

### 回歸與驗證界線

- 根套件：205 項／52 suites 通過，涵蓋新的 Graph、Profile、Unicode 串流檢查及 ACE-Step 音訊測試，另含既有 Qwen 數值 oracle。
- WebUI：14 項測試通過；另以修改前實作做 500 組狀態、6,000 次選取事件的差異比對，結果一致，包含重複 ID、刪除素材與 pending job。
- Music 3：24 項／2 suites 通過。小型兩層 Transformer 在多個長度、sigma、rotary dimension、conditional／unconditional 輸入下逐位元相同；WAV 裁切、取整、交錯、非連續輸入及錯誤時保留既有檔案均通過。
- H3：新測試在一般／Pruned 架構、含／不含 LoRA、含／不含音影條件、四個 sigma 下，音影 velocity 的 Float32 位元模式相同。全套 70 項／11 suites 仍回報兩項先前已有的 VAE 測試失敗：`temporalTilingPlan` 的 padded token 預期、`temporalOutputFrameCount` 的輸出幀數預期；本輪未更動相關 VAE 實作或測試預期，不能宣稱 H3 全套通過。
- LTX：50 項／7 suites 回歸通過；需要另指定真實模型環境的案例維持條件略過。
- 建置／啟動／維護腳本：12 項回歸通過。
- `./build.command --no-app` 完整建置通過；主程式、MCP、Doctor、Qwen 2.1 及五個獨立 Worker 共 9 個 Release 執行檔均存在且可執行。最終 App 資源中的兩份工作區 JavaScript 與原始碼 SHA-256 一致，不含本輪暫存備份；`git diff --check` 通過。
- 音訊基準中的新舊 WAV 逐位元組相同；ACE-Step 串流回歸額外跨越 256 KiB 邊界，確認增益、聲道順序及原始張量不變。

本輪使用小型合成張量與現有回歸測試，未重新執行高解析度、長片或完整大型模型生成。前一節的真實 256×256／9 幀驗證屬於前一輪，不當作本輪新增的實測。

基準來源、差異比對、原始量測與日誌位於本機 `Outputs/project-function-optimization-2026-10-03/`（Git 忽略）。本輪驗證成功後已移除自己的 `.bak` 備份；原始基準快照保留於該目錄。沒有同步 GitHub、建立 commit 或發布 Release。


## 1.26.1004 發布整理

以上 2026-10-03 的模型整合、函式最佳化與全專案最佳化一併納入 1.26.1004。各階段的「未同步 GitHub／未發布」為當時狀態；正式版本資訊見[更新紀錄](../UpdateNote.md)，安裝包簽章、公證與驗證範圍見[驗證紀錄](VALIDATION.md)。函式量測不代表整體生成速度，已知 H3 VAE 測試失敗維持揭露。

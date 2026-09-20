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

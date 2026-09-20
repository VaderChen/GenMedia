# GenImage 深度檢查報告（2026-09-20）

初次檢查基準為 `31ae29b`；本報告包含後續效能、操作一致性與媒體檔案修正，最新驗證日期為 2026-09-21。初次檢查只做重現；後續已依使用者要求完成下列修正。原始問題說明與行號保留作為歷史證據。

初次確認的 12 項問題（6 項 P1、6 項 P2）均已修正；下方保留原始嚴重度與重現證據。本報告涵蓋現存缺陷與近期修改，不表示問題全部由近期修改引入。

## 2026-09-21 後續修正

前輪 12 項修正保留，另完成背景模型／LoRA 掃描、過期掃描結果隔離、背景安裝與修復驗證、LoRA 有上限標頭讀取與分塊轉換、Worker 滿管線可取消，以及下載大小比較與延遲進度回呼修正。

此輪完整驗證為 **122 項 Swift 測試通過**；前輪操作一致性修正為 **130 項**；最新媒體檔案與 Worker 修正為 **141 項**，詳見下方紀錄。256 MiB 合成 LoRA 轉換的程序最大 RSS 由約 **521 MiB 降至 11.6 MiB**；不代表真實模型或整個 App 的峰值。詳細方法與限制見 [效能修正紀錄](PERFORMANCE_CHANGES.md#2026-09-21-背景掃描與-lora-記憶體修正)。下方保留前輪的驗證數量與原始問題證據。

## 2026-09-21 操作一致性追加檢查

此輪針對上一輪新增背景操作後的完成順序，以及既有下載器的檔案替換行為進行檢查並修正：

| 問題 | 修正結果 | 驗證 |
| --- | --- | --- |
| 暫停後重啟或移除會與舊任務清理重疊 | 同模型佇列保留取消中的尾端，新任務等待舊任務完成；切換模型目錄也等待清理 | 舊檔案寫入完成後才移除，不同模型仍獨立完成；取消排隊任務仍清除狀態 |
| 取消早於連線建立時被漏接；續傳資訊可能在取消返回後才寫入 | 記錄取消狀態，等待下載完成回呼與續傳資訊寫入 | 連線前取消與本機 HTTP 中途取消測試通過，後續寫入的標記未被舊回呼覆蓋 |
| 下載或硬連結重用先刪除目的檔，後續失敗造成資料遺失 | 先在目的磁碟準備暫存檔，再原子替換；錯誤本文只讀 2 KiB | 成功、大小不符、來源消失、目的地目錄等案例均通過，失敗保留原內容 |
| 移除模型占用 MainActor，背景化後又可能與生成衝突 | 背景移除，生成與移除互斥；目錄掃描等待移除完成 | 佇列／掃描等待測試及完整 App 編譯通過 |
| 輸出路徑先顯示更新，服務仍等待 actor 更新；字幕跨 await 混用目錄 | 同步更新具備鎖保護的路徑，每個工作持有開始時快照 | 第一份字幕等待中切換目錄，第一份仍寫舊位置，第二份寫新位置 |

完整根套件編譯成功，**130 項 Swift 測試通過**（Core 56、Runtime 57、MCP 6、GGUF 9、App 2）。本機 HTTP 案例有實際傳輸及取消，但不代表已測試遠端模型下載、完整 App 操作或外接磁碟斷線。未執行大型模型推論或發佈。

編譯與測試日誌：`genimage-lifecycle-final-build.log`、`genimage-lifecycle-tests.log`。159 個來源、測試、資源及套件設定檔與通過驗證的內部磁碟副本逐檔一致；正式建置腳本未恢復外接磁碟處理。

## 2026-09-21 媒體檔案與 Worker 追加檢查

| 問題 | 修正結果 | 驗證 |
| --- | --- | --- |
| Worker 正常退出時仍有未讀日誌，成功結果被誤判失敗 | 退出後分批排空日誌，處理無換行尾行，維持讀取上限 | 修改前兩個 3 MiB 案例皆失敗，修正後成功；每輪仍不超過 1 MiB |
| 預先取消仍嘗試啟動子行程 | 啟動前及返回前檢查取消 | 修改前拋啟動錯誤，修正後拋 CancellationError |
| 刪除或重新命名一筆資產破壞其他工作區的檔案參照 | 刪除保護共用來源／播放檔；改名同步更新全部對應資產 | 跨工作區、共用播放路徑、父目錄符號連結及移動連結本身案例通過 |
| 自動清理範圍過大，只依 ID 判斷孤兒 | 限制 MediaCache，保護來源與仍被引用的 URL | 快取重新匯入取得新 ID 後仍保留；外部檔案與目錄不被刪除 |
| 執行工作期間可以刪除或改名正在使用的媒體 | 原生層拒絕操作，Bridge 正確傳回錯誤 | 完整 App 編譯通過，已檢查 UI 只在 invoke 成功後移除本地參照 |

完整根套件編譯成功，**141 項 Swift 測試通過**（Core 64、Runtime 60、MCP 6、GGUF 9、App 2）；161 個來源、測試、資源及套件設定檔與測試副本一致。

既有 Worker 重用測試曾有一次不穩定結果，診斷未再次重現；已將短閒置回收與重用分開驗證，隔離測試中的系統壓力通知，並等待行程實際退出。正式程式的壓力卸載功能維持啟用。詳見 [效能與穩定性紀錄](PERFORMANCE_CHANGES.md#2026-09-21-媒體檔案保護與-worker-結束處理)。

日誌：`genimage-media-reproduction.log`、`genimage-media-final-build.log`、`genimage-media-final-tests.log`。未執行真實模型推論、完整 UI 操作、磁碟斷線或發佈。

## 修正與驗證

本次已修正 12 項問題；>64 GB Profile 的既有隱藏規則保留。未移動模型或使用者輸出。

| 項目 | 修正結果 | 驗證 |
| --- | --- | --- |
| 1 批次覆寫 | 輸出名稱加入 UUID，不需先建立空檔 | 真實服務＋替身 Worker：4 個資產、4 個檔案，依序保留 frame-0～3；失敗會清除部分輸出 |
| 2、11 媒體取消／排隊 | 請求取消不可逆，512 KiB 區塊逐塊等待交付 | 512 MiB 稀疏檔與慢回呼：停止後 0 次回呼；Range 206 與空 Range 錯誤測試通過 |
| 3 MCP HTTP | 限定 loopback、驗證 Host、拒絕 Origin／預檢、要求 JSON，移除 CORS | 原生 ping 成功；外部／null／本機 Origin、錯誤 Host 與 Content-Type 拒絕；正式 MCP 6 項測試通過 |
| 4 工作區還原 | 區分不存在與讀取失敗，錯誤時禁止覆寫及孤兒快取清理 | 不存在、損壞 JSON、新版 schema、不可讀檔案案例通過，原始資料保留 |
| 5 FFmpeg 回復 | 只有本次建立的 prefix 才可刪除；signal／EXIT 只回復一次 | 下載失敗、configure 失敗、TERM、已存在備份四種隔離測試通過 |
| 6 授權打包 | App／DMG 使用現有五份授權文件，編譯前檢查 | 實際打包複製片段的五份內容與來源一致；缺文件時在建置前失敗 |
| 7、12 Profile 保存／複製 | 保存完整自訂定義，選取使用穩定 UUID，複製保留 music | 完整值保存／重讀一致；副本有新 ID 且音樂範圍、預設值、語意不變 |
| 8 輸出目錄 | 同步重建 MediaCompositionService | App 完整編譯通過，已追查兩種影音合成工作的服務取得路徑 |
| 9、10 備份／清理 | 共用 Worker 套件列舉，保留所有 .bak 與資料目錄 | 五個現有 Worker 加未來 Worker fixture：ZIP 排除快取，clean 移除全部 .build 並保留備份 |

### 最終驗證

- 完整根套件編譯成功；Swift Testing **116 項通過**：Core 49、Runtime 50、MCP 6、GGUF 9、App URL scheme 2（另含參數化案例）。
- `python3 -m unittest discover -s Tests/Scripts -v`：**7 項通過**，只操作隔離的暫存檔案樹。
- `node --test Tests/WebUI/*.test.mjs`：**2 項通過**；9 個 Shell 腳本語法檢查及 `git diff --check` 通過。
- 初次重現 harness 再測：批次輸出 `distinct_files=4`；不受信任 Origin 回覆 `403` 且無 CORS；媒體 `callbacks_after_stop=0`。
- App 授權檔實際複製片段驗證 5 份文件內容一致；未執行完整 release App／DMG 簽署、公證或發佈。

完整驗證使用內部 APFS 磁碟的暫存副本；驗證後已清除該副本及本次修改前的 `.bak`，其他備份保留。專案所在磁碟為 USB／ExFAT，原位置直接編譯會因 `.__CodeSignature` 等 AppleDouble 檔案而簽章失敗。未將外接磁碟清理或暫存轉移行為加回正式建置腳本。已補齊 Xcode Metal Toolchain，並校正主專案套件快取中搬移前的絕對路徑。

GGUF 測試需要 MLX Metal library：在暫存副本將本次編譯的 `mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib` 複製至套件根目錄，再於該目錄執行 `swift test -j 4`；這是此次測試環境準備，沒有修改依賴套件或正式產品來源。

工作區或自訂 Profile 讀取失敗時會顯示原因並保留原始資料；該次執行的相關變更不會保存，需先修復資料並重新啟動。MCP 的本機原生客戶端仍屬信任邊界，Host／Origin 檢查不等同客戶端身分認證。

未執行真實大型模型推論、完整 WebKit 操作、H3 長影片／音訊及 ASR／翻譯整合，也未量測真實推論速度或峰值 RAM，故不宣稱量化效能改善。後續仍可處理 MainActor 上的大型模型掃描與實機效能量測。

## 優先處理

### 1. [P1] Z-Image 批次輸出使用相同檔名，前面的圖片被覆寫

- 位置：`Sources/GenImageRuntime/ZImageTextToImageService.swift:65–67`；`Sources/GenImageCore/OutputFileNaming.swift:83–91`。
- 觸發：一次生成 2～8 張圖片。
- 原因：先用 `map` 配置全部輸出路徑，但命名函式只檢查磁碟上已存在的檔案。此時第一張尚未寫入，因此每次都取得相同路徑。
- 影響：Worker 依序覆寫同一檔案，UI 建立多個資產卻全部顯示最後一張；刪除其中一個結果的實體檔也會影響其他結果。
- **已重現**：使用原始 `ZImageTextToImageService`、`WarmRuntimeWorker`，搭配只寫入序號的假 Worker；要求 4 張，得到 `assets=4 distinct_files=1`，全部內容為 `frame-3`。未使用真實模型。
- 修正方向：批次內預留唯一名稱，並處理不同服務／MCP 同時配置檔名的競爭；新增批次輸出整合測試。

### 2. [P1] 媒體停止後仍送出 WebKit 回呼

- 位置：`Sources/GenImageApp/AssetSchemeHandler.swift:98–102,288–298`。
- 觸發：影片讀取尚未完成時切換預覽、關閉分頁或停止載入。
- 原因：背景讀取把回呼排到主執行緒；停止後，`serve` 的 `defer` 又移除停止標記。先前排隊的回呼因此重新通過 `isStopped` 檢查。
- 影響：已停止的 `WKURLSchemeTask` 仍收到 `didReceive`／結束回呼，有造成 WebKit 例外的風險。
- **已重現**：直接編譯未修改的 handler，使用記錄型 `WKURLSchemeTask` 與暫存稀疏媒體檔；停止後仍收到 **18 次回呼**。此測試確認錯誤回呼，未刻意讓正式 App 崩潰。
- 修正方向：每個請求使用獨立且不可逆的取消狀態；待所有排隊回呼結束後才清理生命週期。

### 3. [P1] MCP HTTP 接受未受信任 Origin，且回傳萬用 CORS

- 位置：`Sources/GenImageMCPServer/MCPHTTPServer.swift:257–272,311–320,332–334`。
- 觸發：使用者開啟本機 MCP HTTP 服務，有來源能連到該本機端點。
- 原因：HTTP 標頭解析後未驗證 Origin／Host，也沒有認證；OPTIONS 被接受，回覆包含 `Access-Control-Allow-Origin: *`。
- 影響：伺服器本身不區分受信任 MCP 客戶端與其他來源。可呼叫的正式工具包含模型路徑探索、圖片描述及生成等。瀏覽器是否能到達本機端點還受其本機網路權限限制，不能以此替代伺服器驗證。
- **已重現**：使用原始 HTTP transport 與 JSON-RPC dispatcher、無副作用的工具替身，送入 `Origin: http://untrusted.invalid`；得到 HTTP 200、`cors=*`，工具確實被 dispatch。未執行真實推論，也未向外部服務送出資料。
- 修正方向：明確限制 Origin／Host，保留 loopback 綁定，移除萬用 CORS，依實際客戶端需求加入認證。

### 4. [P1] 工作區讀取失敗會被空白工作區覆寫

- 位置：`Sources/GenImageApp/AppStore.swift:304,333–336,426–448`。
- 觸發：既有 `open-projects.json` 損壞、schema 版本不支援，或啟動時暫時無法讀取但稍後可寫入。
- 原因：`try?` 把讀取錯誤當成沒有存檔，建立示範專案與空資產，接著啟用保存並立即排程寫回原位置。
- 影響：既有專案／資產關係的原始索引被覆寫；孤兒快取清理還會因空的資產集合而刪掉既有播放代理。輸出原檔不會由這段程式直接刪除。
- **程式碼確認**：讀取、預設值、快取清理及保存路徑已完整追查；未對使用者真正的工作區製造故障。
- 修正方向：區分「不存在」與「讀取失敗」，保留損壞原檔或 `.bak`；成功還原前禁止覆寫及孤兒清理。

### 5. [P1] FFmpeg 下載失敗會刪掉既有安裝

- 位置：`scripts/build-ffmpeg-macos.sh:44–58,70–76`。
- 觸發：已有 FFmpeg 安裝，但下載或解壓縮階段失敗。
- 原因：先註冊 EXIT trap；trap 無條件刪除 `$PREFIX`。既有安裝要到下載完成後才移到 `$PREFIX_BACKUP`，早期失敗時沒有備份可還原。
- 影響：一次失敗的更新即可移除原本可用的 FFmpeg／ffprobe。
- **已重現**：執行完整原腳本，將所有可寫路徑導向隔離目錄，使用固定回傳 22 的 curl 替身；腳本 exit 22，既有安裝的 sentinel 消失。
- 修正方向：只在已開始替換安裝時回復；讓回復可重複執行，避免 EXIT／signal 路徑重複刪除。

### 6. [P1] App／DMG 打包仍依賴已刪除的授權檔

- 位置：`build.command:631`；本機 `package-dmg.sh:256–260`（此腳本被 Git 忽略）。
- 觸發：執行預設 App 打包，或執行 DMG 打包流程。
- 原因：App 腳本仍複製根目錄 `LICENSE` 成 `GPL-3.0.txt`，但目前僅有 `LICENSE.md` 等檔案；DMG 腳本也仍強制檢查舊檔名。
- 影響：即使 Swift 編譯全部完成，預設 App 打包仍會中止。只改第一處也會留下 DMG 檢查失敗。
- **已確認**：根目錄 `LICENSE` 不存在，腳本啟用 `set -e`；未重新執行昂貴的完整 release build。
- 修正方向：依專案現行授權檔同步調整兩個打包流程，並在編譯前檢查必要包裝資源；不要僅把現行文件重新命名成舊授權名稱。

## 其他確認問題

### 7. [P2] 自訂 Profile 在重新啟動後消失

- 位置：`Sources/GenImageApp/AppStore+Profiles.swift:251–266,314–315,357–375`；`AppStore.swift:348,379`；`AppStore+Persistence.swift:73–102,171–184`。
- 新增、複製及修改只更新記憶體的 `profiles`；UserDefaults 保存的是啟用／停用簽章，工作區快照也沒有 Profile 定義。啟動時只合併模型探索與內建 Profile，因此無法還原自訂設定及其啟用狀態。
- **程式碼確認**：已追查 Profile 所有寫入與啟動還原入口；未以正式 App 做重啟測試。
- 修正方向：獨立保存完整自訂 Profile、穩定 ID 與 revision，啟動先載入定義，再還原選取。

### 8. [P2] 輸出目錄切換漏掉影音合成服務

- 位置：`Sources/GenImageApp/AppStore+Paths.swift:28–42`；`Sources/GenImageRuntime/MediaCompositionService.swift:80–84`。
- `setOutputDirectory` 更新圖片、字幕、影片與音樂服務，卻沒有重建 `mediaCompositionService`。該服務保存不可變的初始化路徑。
- 影響：切換 A → B 後，圖片循環影片及影音合併仍寫到 A，直到重新啟動；畫面卻顯示 B。
- **程式碼確認**：已追查兩種合成工作的服務取得與輸出路徑。
- 修正方向：納入共同目錄更新流程，並測試每一種輸出功能。

### 9. [P2] 清理腳本會刪除 Backups 內的 `.bak`

- 位置：`clean.command:26–32`。
- `find` 遞迴刪除整個專案內的 `*.bak`，沒有排除 `Backups`，但結尾宣告 Backups 未變更。
- **已重現**：隔離目錄中的 `Backups/important.bak` 執行後消失。
- 修正方向：只清理明確定義的中間產物，排除備份及使用者資料目錄，讓顯示訊息與實際行為一致。

### 10. [P2] 備份及清理漏列四個 Worker 的編譯目錄

- 位置：`backup.command:31–40`；`clean.command:15–24`。
- 只處理根 `.build` 與 `Qwen2511Worker/.build`，漏掉 ZImage、LTXVideo、MiniMaxMusic3、MiniMaxH3。
- 影響：備份會收錄大量可重建的快取；清理後還殘留 Worker 舊路徑／舊建置。搬移專案後尤其容易誤以為已完成清理。
- **已重現**：五個 Worker 各放一個假快取檔，產生的 ZIP 包含上述四個 `.build/cache.bin`。目前真實工作區也存在 GB 級 Worker 編譯目錄。
- 修正方向：統一列舉 Worker package 根目錄，所有備份／清理流程共用排除規則。

### 11. [P2] 媒體分塊讀取沒有等待交付，無法限制排隊記憶體

- 位置：`Sources/GenImageApp/AssetSchemeHandler.swift:272–280,288–298`。
- 每讀取 512 KiB 就立即將捕捉該 Data 的 closure 排入 `DispatchQueue.main`，未等待交付也未限制排隊數量。
- 影響：主執行緒渲染或忙碌時，大範圍媒體請求仍可把大量甚至整個請求區間保留在排隊 closure 中；「分塊」本身沒有保證低記憶體。
- **程式碼確認**；第 2 項重現也實際觀察到多筆等待交付的資料。未進行大型真實影片的峰值 RAM 量測。
- 修正方向：限制待交付位元組或區塊數，等前一批送出再讀下一批，同時保持取消可即時生效。

### 12. [P2] 複製音樂 Profile 時遺失音樂設定

- 位置：`Sources/GenImageApp/AppStore+Profiles.swift:252–264`；`Sources/GenImageCore/DomainModels.swift:171,186`；`Resources/WebUI/js/workspace.js:737–746`。
- 複製時沒有傳入 `music: profile.music`，新 Profile 的 `music` 變成 nil。
- 影響：MiniMax 等 Profile 的最長長度語意及模型專屬上下限，會在副本中退回通用設定，使用相同模型卻顯示不同控制。
- **程式碼確認**：型別預設值與 Web UI fallback 路徑一致。
- 修正方向：複製完整設定，再明確更換 ID、名稱、revision 與內建旗標。

## 環境與建置阻礙

這些是目前工作區狀態，與上列產品缺陷分開處理。

1. 以 `diskutil info` 查詢專案所在磁碟，回報 **ExFAT、USB**。當時專案位於外接檔案系統，與先前使用的內部磁碟工作區不同。未自行搬移或格式化。
2. 搬移後 `._` 中繼檔再次出現，包括 `.git/objects/pack/._*.idx`。`git status`／`git log` 曾回報 `non-monotonic index`；此訊息來自被 Git 掃到的 AppleDouble 側檔，不能據此斷定真正 pack 已損壞。本次保留現場，未清理或修改 Git 索引。
3. 原專案執行 `swift test -j 4` 失敗：編譯快取仍指向舊工作區內的 `.build/artifacts/fluidaudio/NemoTextProcessing/NemoTextProcessing.xcframework`。這次未再次手動改寫 generated workspace-state。
4. `xcrun -sdk macosx metal -v` 仍回報缺少 Metal Toolchain。修正舊路徑後，完整 MLX 建置仍需處理此環境依賴。本次未下載元件。

## 初次檢查完成的驗證

| 檢查 | 結果及範圍 |
| --- | --- |
| 原專案 `swift test -j 4` | 因上述 XCFramework 舊路徑失敗，沒有宣稱完整測試通過 |
| 隔離 Swift 測試 | **82 項通過**：Core 46、Runtime 36；直接複製現行來源與既有測試，移除大型模型套件依賴 |
| Web UI 既有測試 | **2 項通過**：activity state 合併與結構變更 |
| JavaScript 語法 | **16 個模組通過** `node --check` |
| Shell 語法 | **9 支腳本通過** `zsh -n` |
| 缺陷重現 | 批次路徑覆寫、停止後回呼、MCP Origin、FFmpeg 回復、備份清理範圍 |
| `git diff --check` | 通過 |

隔離 Swift 測試包含 Profile 64 GB 邊界、工作區快照與 flush、尺寸計算、字幕與 sidecar、媒體探測、子行程終止／取消、warm Worker 重用／閒置回收／失敗重建、增量日誌、縮圖方向及 bitmap tile 拼接。

隔離測試**不等於**完整 App 測試。未重跑 GGUF／MLX、真實模型推論、H3 實際影片／音訊編碼、ASR／翻譯模型整合、真實模型網路下載、DMG 公證及整套 WebKit 操作流程。此次也未量測真實模型速度或峰值 RAM，因此不宣稱效能改善幅度。

## 後續效能與測試工作

- 媒體交付的排隊上限已修正；後續量測長影片切換、快速取消及多預覽時的 RAM。
- `LocalModelDiscovery` 與安裝後驗證已移至背景並加入取消／結果版本管理；LoRA 檢查已限制標頭讀取。後續量測真實磁碟延遲，並檢查模型刪除與 Profile 小型檔案操作。
- 已新增服務端「一次多張輸出」、工作區還原政策與真正 handler 的 URL scheme 取消測試；仍需完整 App 啟動／重啟與真實模型端到端測試。
- 優先順序：輸出覆寫及資料保留 → 媒體取消與 MCP 存取 → 打包／回復 → Profile 與目錄一致性 → 備份範圍及效能量測。

## 重現證據

本次隔離環境使用系統暫存目錄。暫存目錄可能被系統日後清除，因此關鍵結果已記錄於本報告。

- `shell-repros.json`：原腳本在隔離檔案樹的執行結果。
- `probe.log`：4 個資產／1 個輸出檔、未受信任 Origin 的 HTTP 200，以及停止後 18 次回呼。
- `isolated-tests.log`：82 項 Swift 測試。
- `checks/Sources/AuditProbe`：無副作用的重現 harness；後續已換入修正後來源再次驗證。
- `genimage-review-swift-test.log`：原工作區完整測試的建置阻礙。

修正後驗證日誌：

- `genimage-final-swift-tests.log`：完整 116 項 Swift 測試。
- `genimage-maintenance-tests.log`：7 項隔離腳本測試。
- `genimage-fixed-probe.log`：修正後三個主要重現結果。
- `genimage-full-fix-tests.log`：原 ExFAT 路徑的簽章限制。


## GitHub 同步前的文件整理

四語 README、架構、Web Bridge 與路線文件已同步輸出 UUID 命名、資料位置與操作順序；README 授權摘要改為連結現有的原始碼公開・禁止商業販售授權 v1.1，授權條文本身未變更。[驗證指南](VALIDATION.md) 區分最新根套件結果、歷史 Worker 結果及未涵蓋的整合測試。

同步前只清除了已辨識為 AppleDouble 的 Git pack-index／refs 附屬檔，實際 Git 索引、refs 與物件保留；`git fsck --connectivity-only --no-dangling` 通過。這是一次性的工作區中繼資料整理，未恢復正式建置腳本的外接磁碟處理。

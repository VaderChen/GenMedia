# GenMedia Update Notes

本文件採「最新內容在上」的方式維護。開發期間先更新 `Unreleased`；建立 GitHub Release 時，將標題改為正式版本與日期，再新增下一個空白的 `Unreleased` 區段。

## Unreleased

- 工作區底片列新增圖片匯入 ICON，支援 Finder 拖放一張或多張 PNG、JPEG、WebP、GIF、TIFF、HEIC 與 HEIF 圖片；音樂生成模式會停用圖片匯入按鈕。
- 圖片生成主按鈕會依目前是否選取有效來源圖片，自動路由至圖生圖或文生圖 Profile，避免圖生圖誤要求文生圖 Profile。
- 圖片與影片比例選項改為下拉選單；圖生圖有來源圖片時才顯示「原解析度」，並將來源尺寸換算為 16 倍數。
- 圖生圖指定寬高會真正傳入 Qwen Image Edit Runtime；來源比例與輸出比例不同時，先等比例縮放並以邊緣延展補足畫布，避免生成前裁切來源內容。過低的 `128 × 192` 仍可能降低細節與構圖穩定性。
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

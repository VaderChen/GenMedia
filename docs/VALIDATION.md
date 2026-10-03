# 驗證方式與結果

最近更新：2026-10-03。本文記錄可重複執行的檢查，以及各次結果的實際範圍。

## 1.26.1004 回歸與安裝包（2026-10-03）

本版包含下方的新模型整合與兩輪函式最佳化。最終回歸在 Apple M4、16 GB RAM、Swift 6.4／macOS 27 SDK 執行；小尺寸真實模型結果與純函式測試分開記錄。

| 範圍 | 結果與限制 |
| --- | --- |
| 根套件 | 205 項／52 suites 通過，包含模型目錄、Profile、程序退出、Qwen 數值回歸、素材關係、串流文字及 ACE-Step 音訊。 |
| WebUI | 14 項通過；另以 500 組狀態、6,000 次選取事件比較修改前後，結果一致。 |
| MiniMax Music 3 | 24 項／2 suites 通過，涵蓋條件／無條件推論、RoPE 重用及 WAV 輸出。 |
| LTX | 50 項／7 suites 通過；4 個需指定真實模型環境的案例條件略過，實際模型另依下方記錄驗證。 |
| MiniMax H3 | 全套 70 項／11 suites 中有 2 項既有 VAE 時間分段測試失敗；本版未更改相關實作或測試預期。新增文字條件重用與 LoRA 數值回歸通過，不能宣稱全套通過。 |
| 建置、啟動與維護腳本 | 12 項通過。 |
| Release 建置與安裝包 | 9 個出貨產品建置成功；App／DMG 均完成 Developer ID 簽章、公證、Staple 與 Gatekeeper 驗證。複製回專案及唯讀掛載後再次核對通過。 |
| 安裝包啟動檢查 | 內附 Qwen Turbo、LTX-2.5、H3 v1.2 Worker 正確拒絕不相容步數；FFmpeg／FFprobe 可執行。僅驗證封裝與參數檢查，未執行模型生成。 |
| 小尺寸輸出一致性 | Qwen Turbo 256×256／6 步在最佳化前後的 PNG 相同；LTX 的 256×256／9 幀 latent 回放解碼，逐幀像素、時間戳及 AAC 資料相同。這不是重新執行完整文生影。 |

效能數字是指定函式與輸入的局部量測，不代表整個 App 或模型生成的加速倍率。H3 的已知失敗為 `temporalTilingPlan` 與 `temporalOutputFrameCount`；詳見下方初始整合紀錄。完整方法、記憶體拷貝範圍與數值一致性結果見[效能紀錄](PERFORMANCE_CHANGES.md)。

原始日誌保存於本機 `Outputs/function-optimization-2026-10-03/`、`Outputs/project-function-optimization-2026-10-03/` 與 `Outputs/release-1.26.1004/`，不納入 Git。

### 1.26.1004 公證安裝包

- App 版本 `1.26.1004`，Build `2102`，Bundle ID `com.vader.genimage`；Apple Silicon。
- App 公證：`91f5bca7-d82a-40bd-b655-f1ae46a5cb59` — **Accepted**。
- DMG 公證：`d2c4b966-1bb0-4e92-a4cd-6c6057ab9ce3` — **Accepted**。
- `GenMedia-1.26.1004-arm64.dmg`，`214,624,691` bytes；App／DMG 均完成 Staple，Gatekeeper 回報 `Notarized Developer ID`。
- SHA-256：`5e46c3d6cd1f85c2c9d38dc91e568f0b3ae17ecec4443e341700eb5eab69eff9`。
- 安裝包於 APFS 製作，複製回專案磁碟後雜湊一致；再次唯讀掛載並核對版本、主程式與 7 個內附執行元件的 Developer ID、hardened runtime、secure timestamp，以及 22 份 WebUI 資源與授權文件。建置輸入雜湊與封裝前一致。
- [下載與中英文版本說明](https://github.com/VaderChen/GenMedia/releases/tag/v1.26.1004)。日誌與雜湊紀錄保存於本機 `Outputs/release-1.26.1004/`。

## 1.26.1004 新模型初始整合驗證（2026-10-03）

維持純 Swift／MLX 與既有 UI；實際圖片驗證使用 256×256。以下為整合階段的驗證快照，最終回歸及安裝包結果依上方 1.26.1004 紀錄為準。

| 範圍 | 結果與限制 |
| --- | --- |
| Qwen Viggle Turbo v0.3 r128 | 227 組 LoRA 配對、量化投影與專用 6 步時間表通過。256×256／Seed 42 文生圖約 31.01 秒，紅茶壺改藍色約 45.82 秒；已檢視輸出。 |
| Qwen 一般／Turbo 切換 | Turbo 文生圖、Turbo 編輯後，再以一般模型 2 步完成生成，約 22.60 秒。2 步只檢查流程，不評畫質。 |
| Qwen PE | T2I／I2I 4-bit 實際下載、安裝器與目錄重新辨識通過。兩個模型均完成原生文字推論；I2I 回覆偶發缺少圖片參照的起始引號，加入限定欄位修補並以實際回覆回放驗證。PE I2I＋Turbo 6 步已實際完成 256×256 改色編輯，包含前處理約 190.56 秒；來源圖片連結與尺寸保持正確。 |
| H3 LightX2V 4 步 v1.2 | 實際下載及雜湊驗證完成，624 個張量／208 組配對通過基底結構核對，代表性殘差非零且有限。7 項 LoRA 測試通過，涵蓋 v1.2、既有 8 步與 Turbo v4。沒有完整 H3 影片生成或加速倍率實測。 |
| LTX-2.5 | 43 項 LTX Runtime 測試通過（實際模型測試需另行啟用）。新增 Gemma4 獨立 FP64 純量 oracle，誤差門檻 `3e-5`，含 padding 不變性、BOS／截斷、FF bias、首幀 mask 與 ancestral Euler。27.21 GB 蒸餾模型包通過下載雜湊、安裝及重新辨識；Gemma4 真實 4-bit 權重的 48 層／49 組 hidden states 另已通過載入與有限值檢查。真實音訊／影片解碼器亦已通過雙聲道、9 幀 256×256 與有限值檢查。 |
| LTX-2.5 影音輸出 | 真實提示詞生成紅色茶壺；封裝修正後重用同一份 latent，確認 H.264 256×256／9 幀／24 FPS、AAC 48 kHz 雙聲道與 0.375 秒長度。首、中、末幀已檢視，茶壺與桌面構圖一致。未驗證長片、高解析度、音訊品質或與官方 BF16 的品質一致性。 |
| 根套件 | 完整 199 項測試通過，包含目錄、安裝、請求、LoRA、回覆解析、Profile 分頁還原與既有流程；另有 9 項 WebUI 測試通過。 |
| H3 全套回歸 | 69 項中有 2 項既有時間分段測試失敗；不能宣稱 H3 全套通過。詳見下方說明。 |
| 本機 Release | `build.command --no-app` 成功，包含 App、MCP、Qwen 2.1 及各獨立 Worker；未製作新 DMG。 |

本次重跑也捕捉到既有 `WarmRuntimeWorker.unload()` 偶發卡在 Foundation `waitUntilExit()`。已移除完成路徑的多餘等待，強制結束則檢查子行程存活狀態，避免進入同步 RunLoop。新增 32 次並行自然退出／強制結束競爭測試，與常駐 Worker 重用、閒置釋放、取消測試一併通過。分頁草稿改為先比對 Profile ID，重開時再以名稱區分一般／Turbo／PE；無法唯一辨識的舊草稿須重新選取，避免默默換錯模型。

LTX 真實模型載入時發現 `upsampler.0` 被扁平權重重建流程誤判成陣列；已改為保留模型原有的字典與陣列結構。新增空間／時間放大器的 safetensors 實際載入及固定輸出測試，兩個案例均通過。 完整去噪後另發現音訊 resampler 的無條件 `squeeze` 移除了單一批次維度，以及左右聲道應堆疊成 `[B,2,T]`；均已修正，補上常數訊號／單一批次回歸，以及真實解碼器的獨立驗證（約 2.98 秒）。 封裝實測另捕捉到較短的 0.33 秒音軌令 FFmpeg `-shortest` 將 9 幀裁成 7 幀；現在補齊／裁切音訊至影片長度，完成事件也回報影片時長。以真實生成 latent 回放解碼，`ffprobe` 確認修正後保留 9 幀；另新增以真實 Worker／FFmpeg 執行的短音軌封裝回歸，約 2.99 秒通過。

H3 失敗位於未修改的 `MiniMaxH3VideoVAETests.swift`：`paddedTokenCount` 期望 12、實際 15；3 個 latent frames 的 `outputFrameCount` 期望 9、實際 6。該 VAE 實作及這兩項測試與任務開始的 `9ba5ea2` 相同，新增加速 LoRA 的測試另已通過。本次保留失敗紀錄，未更改 VAE 演算法或調整期望值來掩蓋問題。

LTX 的初次完整生成至原封裝約 869.92 秒，Worker peak memory footprint 約 13.44 GB；16 GB 測試機出現明顯記憶體壓力，仍建議 32 GB 以上。修正封裝後以相同 latent 回放解碼／封裝約 2.13 秒，**不是 2.13 秒文生影，也不是最終版本單次完整執行的效能量測**。最終樣片為 `ltx25-256/video-4.mp4`，生成紀錄為 `worker-3.log`、修正後回放與媒體核對為 `worker-4.log`／`verification-4.json`。

實跑記錄在本機 `Outputs/new-model-integration-2026-10-03/`，不納入 Git。模型原位於 `Vader Ext3`；驗證途中磁碟離線，LTX 下載中止，改用 `VaderHD` 下的暫存 `Models` 目錄續做驗證。這些暫存路徑不是 App 的預設安裝目錄。

新增可選實測需明確設定環境變數，普通測試不會自動下載數十 GB 權重：

- `GENIMAGE_PE_INSTALL_ROOT`：以正式安裝器驗證 PE T2I／I2I 與 Qwen Turbo。
- `GENIMAGE_PE_SMOKE_ROOT`、`GENIMAGE_PE_SMOKE_IMAGE`、`GENIMAGE_PE_SMOKE_OUTPUT`：執行原生 PE；可用 `GENIMAGE_PE_SMOKE_KIND=I2I` 選單一模型。
- `GENIMAGE_PE_REPLAY_OUTPUT`：回放原生 I2I 模型的完整回覆，檢查嚴格 JSON 解析與限定引號修補。
- `GENIMAGE_QWEN_PE_E2E_ROOT`、`GENIMAGE_QWEN_PE_E2E_MODEL`、`GENIMAGE_QWEN_PE_E2E_WORKER`、`GENIMAGE_QWEN_PE_E2E_IMAGE`、`GENIMAGE_QWEN_PE_E2E_OUTPUT`：測試 PE 釋放後交接原生 Turbo Worker，輸出 256×256 編輯圖。
- `GENIMAGE_LTX25_INSTALL_ROOT`：安裝及重新辨識蒸餾模型包，不下載 Dev 權重。
- LTX Worker 套件的 `GENIMAGE_LTX25_MODEL`：實際載入 Gemma4 全部 48 層並檢查 49 組 hidden states，另可執行雙聲道音訊與 9 幀影片的解碼器測試。
- `GENIMAGE_LTX_DEBUG_LATENTS`：選填的 `.safetensors` 路徑，保留去噪後的影音 latent 供解碼診斷；正常生成不輸出這份檔案。 `GENIMAGE_LTX_DEBUG_LATENTS_INPUT` 可在 LTX-2.5 重用它進行解碼診斷，會檢查 latent 尺寸、幀數與有限值；一般請求仍執行完整文字編碼與去噪。
- `GENIMAGE_LTX25_WORKER`、`GENIMAGE_LTX25_FFMPEG`、`GENIMAGE_LTX25_FFPROBE`：搭配 `GENIMAGE_LTX25_MODEL` 啟用實際 Worker 封裝回歸，檢查短音軌仍保留 9 幀及完整時長。

## 歷史結果（1.26.1003）

| 範圍 | 結果 | 說明 |
| --- | --- | --- |
| 根套件 Debug／測試目標 | 編譯通過 | 目前工具鏈使用 `--build-system native` 編譯測試 |
| LoRA／Profile／模型目錄相關回歸 | 82 項通過 | 含 H3 步數、基底相容性、錯誤請求、下載驗證與 Profile 合併；實際下載測試另行執行 |
| 記憶體最佳化相關回歸 | 21 項通過 | WAV 標頭讀取、原圖分段傳送、Qwen 2.1 固定資料重用 |
| H3 Runtime 回歸 | 29 項通過 | LoRA 數值、Dense／INT8、alpha/rank、排程、基底結構與量化；實際權重另行測試 |
| 小尺寸實際文生圖／編輯 | 通過限定案例 | 256×256 Qwen 2.1 文生圖／編輯在最佳化前後輸出完全相同；Z-Image LoRA 套用／清除與連續生成通過 |
| 影片 LoRA 載入 | 層級驗證通過 | LTX Dolly In 480 層、H3 LightX2V 8 步 208 層、Turbo v4 259 層；不代表完整影片生成品質 |
| 完整根套件測試 | 未全部完成 | 既有兩項程序等待測試停住，詳見效能紀錄；沒有以舊版完整通過數量代表本版 |
| 1.26.1003 安裝包 | 簽章、公證及 Gatekeeper 通過 | App／DMG 均 Accepted、Staple；複製回專案及唯讀掛載後再次驗證 |

各組測試的範圍不同且部分重疊，數量不相加。影片 LoRA 的完整影片品質與整段加速倍率尚未驗證。H3 4 步版本次只核對公開權重標頭與採樣規格；8 步版與 Turbo v4 則實際下載完整 LoRA 並檢查非零、有限的殘差。詳見 [LoRA 指南](LORAS.md) 與 [效能紀錄](PERFORMANCE_CHANGES.md)。

## 2026-09-22～23 歷史結果（1.26.0922）

| 範圍 | 結果 | 說明 |
| --- | --- | --- |
| 根套件 Debug 編譯 | 通過 | 包含 App、Runtime、MCP、GGUF 及全部根套件測試目標 |
| Release 建置／啟動 | 通過 | 在實際專案目錄完成 `build.command --no-app`，並透過 `run.command` 開啟主視窗 |
| Swift Testing | 163 項通過 | Core 66、Runtime 64、MCP 6、GGUF 9、App 5、QwenImage21 13；另含參數化案例 |
| 建置／啟動及維護腳本 | 12 項通過 | 四個出貨產品建置、失敗／缺檔攔截、啟動參數、備份／清理、FFmpeg 失敗回復及授權打包 |
| Web UI | 6 項通過 | 活動狀態合併、結構變更通知、連續生成與獨立文生圖／圖生圖按鈕 |
| 1.26.0922 App／DMG | 簽章、公證及 Gatekeeper 通過 | APFS 製作，Developer ID、hardened runtime、secure timestamp、App／DMG 公證與 Staple、掛載後 App 驗證均通過 |

使用 Apple Silicon、Swift 6.4 與 macOS 27 SDK。專案的最低部署目標與這次實際測試環境不同；此結果不代表舊版作業系統、工具鏈或所有模型組合均已驗證。

根套件測試不會自動執行 `RuntimeSupport/` 的獨立 Worker 測試。2026-09-19 曾完成 Z-Image Worker 建置與 H3 影片寫入 5 項測試，屬於歷史結果，沒有併入上述根套件測試。詳細紀錄見[效能修正紀錄](PERFORMANCE_CHANGES.md)。

## 根套件

在專案根目錄執行：

```sh
swift build -c release --product GenImage
swift build --build-system native --build-tests -j 4
```

MLX 測試需要與目前相依版本匹配的 Metal library。本次測試使用原生建置後端，Metal library 則由 Release 建置產生；若測試無法找到它，可在測試期間放到套件根目錄，完成後移除該暫存副本：

```sh
(
  if [ -e ./default.metallib ] || [ -L ./default.metallib ]; then
    echo "default.metallib 已存在；請先確認其來源，不自動覆蓋。" >&2
    exit 1
  fi
  cp .build/out/Products/Release/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib ./default.metallib || exit 1
  trap 'rm -f ./default.metallib' EXIT
  swift test --build-system native --skip-build -j 4 \
    --filter 'GenImageCoreTests|LoRAIntegrationTests|ZImageLoRAAdapterNormalizerTests|LoRAProfileTests|MiniMaxH3AccelerationTests|Qwen21ProfileCompatibilityTests'
)
```

其他 SwiftPM 版本或建置後端的輸出位置可能不同。請使用本次建置產生的 library，不要覆蓋既有的自訂檔案，也不要將額外檔案放進已簽章的 `.xctest/Contents/MacOS/`；這會影響 bundle 簽章驗證。

1.26.0922 的完整驗證在內部 APFS 暫存副本進行，來源、測試、資源與套件設定共 183 個檔案均與專案逐檔一致。原 ExFAT 工作區的 AppleDouble 中繼資料曾影響簽章；正式建置腳本不再提供外接磁碟專用的中繼資料清理流程。測試副本不包含模型權重；Qwen-Image 2.1 實跑另讀取專案 Models 下的固定 revision 權重。

## 維護腳本與 Web UI

```sh
python3 -m unittest discover -s Tests/Scripts -v
node --test Tests/WebUI/*.test.mjs
git diff --check
```

維護腳本測試會建立暫存專案與替身工具，不修改真實模型、FFmpeg 安裝或使用者備份。

### 連續生成修正（2026-09-22）

生成完成後，App 會選取輸出圖片供預覽；原本介面因此把唯一的生成按鈕切成圖生圖，沒有啟用圖生圖 Profile 時便無法繼續。現在文生圖按鈕固定保留，選取圖片時另外顯示圖生圖按鈕，分別依自己的 Profile 決定是否可用。

新增 4 項回歸測試，涵蓋連續兩次完成後再生成、編輯完成後繼續、執行／取消／失敗狀態、來源圖片與模型安裝限制。測試使用 256×256 的模擬狀態，驗證實際介面渲染與活動狀態合併，未執行模型推論。另以 WKWebView 載入實際 JS／CSS，確認繁中、英、日、韓四種語言在 584 與 946 px 面板寬度下，圖生文、文生圖、圖生圖按鈕均可用且不被裁切；窄面板會自動換行。

## 1.26.0922 涵蓋的回歸案例

- Profile 的 64 GB 可見性邊界、自訂 Profile 保存與工作區讀取失敗保護。
- 模型掃描的過期結果、同模型取消清理順序、目錄切換等待移除完成。
- 本機 HTTP 下載的取消、續傳資訊寫入、大小不符與既有目的檔保留。
- safetensors 有上限標頭讀取、LoRA 分塊轉換與內容一致性。
- Worker 重用、取消、閒置回收、stdin 滿管線、大量日誌後退出及無換行的最後事件。
- 不同工作區的共用媒體、檔名更新、符號連結及仍被引用的快取保留。
- 媒體 URL scheme 停止後不再回呼、Range 請求與 MCP HTTP 存取限制。

## 限制

尚未驗證所有真實模型的完整推論、所有 WebKit 操作與長時間磁碟斷線／重連。安裝包的簽署公證與發佈結果依版本記錄於下方，不以程式編譯或單元測試代替。

256 MiB 合成 LoRA 轉換的最大 RSS 約由 521 MiB 降到 11.6 MiB，僅代表該次格式轉換，不能當成整個 App 或模型推論的記憶體／速度提升。原始方法及數值見[量測紀錄](PERFORMANCE_CHANGES.md#合成記憶體量測)。

完整問題與修正證據見[深度檢查報告](PROJECT_REVIEW_2026-09-20.md)。暫存日誌可能被系統清除，因此結果與限制也保存在這些文件中。


## Qwen-Image 2.1（2026-09-22）

本輪使用內部 APFS 暫存副本建置完整根套件與測試，避免原 ExFAT 工作區 AppleDouble 檔案造成 SwiftPM bundle 簽章失敗。新增的 Qwen3-VL patch 已套用，並以反向 `git apply --check` 確認一致；完整 patch manifest 的 `--verify` 因其他獨立 Worker checkout 尚未解析而中止，未將此項列為通過。`build.command` 會先解析這些套件再套用 patch。

156 項 Swift Testing 全數通過；另有 7 項維護腳本、2 項 Web UI 測試與 `zsh -n build.command`、`git diff --check` 通過。

新增測試涵蓋：前 RMSNorm hidden states 與一般聊天 logits 的分離、獨立 NumPy DiT 參考值、INT4 matmul、區塊因果／三軸位置、FlowMatch schedule、VAE 時間補零與通道重排、PNG alpha／非有限數值拒絕、無效參數、路由、安裝缺檔偵測、批次失敗清理與素材父子連結。根套件與 Worker 已 Debug 編譯，未執行完整 Release App／DMG 打包或公證。

真實固定 revision 權重已完成 256×256、1 步文生圖端到端測試，輸出 RGBA PNG，耗時 18.14 秒。這是單步連通性測試，不能當成圖片品質驗證。後續實跑結果如下。

512×512、40 步文生圖已完成，耗時 253.40 秒；人工檢視產生的紅蘋果與白色背景符合提示詞。`/usr/bin/time -l` 回報該 Worker peak memory footprint 約 4.53 GB，此數值不是整台電腦的總用量，也不能推論到 1024／2048 或批次生成。測試期間同機有少量編譯／單元測試活動，因此耗時僅供參考。

使用真實 `HuggingFaceModelInstaller.install` 驗證已下載檔案，成功寫入 manifest，再由 `verify` 與 `GenImageDoctor`／`LocalModelDiscovery` 確認可辨識兩項能力。測試權重保留在專案 `Models/qwen-image-2.1-mlx-4bit`，不列入 Git；App 使用其他模型根目錄時，可選取此 Models 目錄或透過模型中心安裝。

實際 RGBA VAE 往返檢查（512×512）平均絕對誤差為 0.0025348（像素範圍 0～1），確認編碼／解碼沒有造成明顯失真。檢查圖像編輯時發現直接沿用 Core Image 線性光空間的正規化會偏離模型要求的 sRGB 像素正規化：在該測試圖的 [-1,1] 資料上平均差異為 0.1649012，最大差異 0.5742644。已在 Qwen 2.1 Runtime 改成先 render 為 sRGB，再以 MLX 執行 mean/std 正規化與時間／空間 patch 重排，並加入中灰像素回歸測試；未修改其他 VLM 的前處理行為。

修正後的 512×512、40 步單圖編輯實跑完成，耗時 496.24 秒，Worker peak memory footprint 約 5.21 GB。提示詞要求將紅蘋果改成綠蘋果並保留構圖；輸出有遵循顏色及構圖要求，但仍有明顯顆粒／金屬感。因此只記錄為端到端流程通過，**不列為編輯品質驗證通過**。sRGB 修正有獨立數值回歸測試支持，但沒有消除這次編輯輸出的品質問題，原因尚未完全定位。圖像編輯持續標示為實驗性，未宣稱與官方推論品質一致。

實測圖片、VAE 往返結果與測試日誌保存在 `Outputs/qwen21-validation`（已由 Git 排除）。未測試 1024／2048、多參考圖、長時間連續工作或所有提示詞類型。


### 小尺寸續驗（2026-09-22）

依需求，這輪真實模型驗證統一使用 256×256，參考圖片也縮至同面積，沒有再次執行 512／1024／2048。編輯指令、種子 42 與來源紅蘋果沿用前輪。

| 測試 | 耗時 | 檢視結果 |
| --- | --- | --- |
| 修正前圖像編輯，20 步 | 80.14 秒 | 綠蘋果、白色桌面正常 |
| 修正後圖像編輯，20 步 | 78.77 秒 | 顏色及構圖正常 |
| 修正後圖像編輯，40 步 | 146.39 秒 | 顏色及構圖正常 |
| 修正後文生圖，20 步 | 47.54 秒 | 紅蘋果與白色桌面正常 |

小尺寸修正前後都沒有重現先前明顯的金屬顆粒，因此不能把畫質改善歸因於這輪程式修正，也不能認定 512 尺寸的問題已排除。上述耗時只供此機器與提示詞參考；20 步可作為後續開發的快速驗證設定。

本輪修正有獨立數值／功能證據：

- Qwen3-VL vision MLP 改用模型要求的 tanh GELU，原本 Swift 相依套件使用 sigmoid 近似。拆成獨立 `Qwen3VL-Vision-GELU.patch`，讓已套用 conditioning patch 的 checkout 也會取得修正。
- 零中心 RMSNorm 在 FP32 執行 `weight + 1` 與 normalization，最後才轉回 BF16，避免縮放權重提早四捨五入。
- Core Image 輸出直接要求 straight-alpha sRGB RGBA，移除色彩轉換前的 unpremultiply；vision 疊白底改在 sRGB tensor 執行。半透明像素的舊版回歸誤差約 0.256，vision 疊白底的正規化誤差約 0.472；修正後通過。
- App 圖生圖 Profile 相容性允許原生 MLX，修正安裝後無法啟用 Qwen 2.1 的問題；切換到圖生圖時保留共用影像路由，避免記憶體清理誤卸載同一個 Runtime。

新增 Qwen3-VL 獨立 NumPy oracle，以 40 個固定權重、vision/text 各一層、非正方形 2×4 與 4×2 patch 驗證位置插值、視覺與文字 RoPE、deepstack、causal attention、最終 RMSNorm 前輸出。錯用 sigmoid GELU 的 CPU 參考誤差為 0.0101／0.0236，高於測試容許誤差 0.0001。NumPy 僅產生測試 fixture，正式推論仍全為 Swift／MLX。

兩份 Qwen3-VL patch 已在臨時乾淨原檔上依建置腳本的 `patch` 命令重新套用，與實測依賴檔逐位元一致。完整跨 Worker patch manifest 及 Release App／DMG 打包仍維持前述未驗證範圍。

本輪完整 Debug 建置與 **163 項 Swift Testing 全數通過**（Core 66、Runtime 64、MCP 6、GGUF 9、App 5、QwenImage21 13）；新增的視覺 oracle、精度、RGBA／疊白底及 Profile 相容性測試均包含在內。原始碼、測試、資源與套件設定共 183 個檔案與 APFS 測試副本一致。這輪小圖與請求 JSON、建置／測試日誌另存於 `Outputs/qwen21-validation` 的 `qwen21-*-256-*`／`small-*` 檔案，未納入 Git。


### Release 啟動修正（2026-09-22）

`run.command` 原本在建置結束後找不到 `.build/out/Products/Release/GenImage`。原因是根套件建置把多個 `--product` 放進同一次 `swift build`，實際只建置最後的 Qwen 2.1 Worker。現在逐一建置 GenImage、GenImageMCP、GenImageDoctor、GenImageQwen21Worker，每項完成後檢查執行檔；啟動腳本也在執行前確認主程式存在且可執行。

新增 5 項腳本回歸測試：四個產品均產生、單項編譯失敗立即停止、成功但缺檔時拒絕繼續、路徑含空白時啟動及參數傳遞、缺少主程式時明確報錯。連同既有維護測試共 12 項通過；`zsh -n build.command run.command` 與 `git diff --check` 通過。

實際主程式為 arm64 Mach-O，`codesign --verify --strict` 通過。所有 Worker checkout 已解析，完整 patch manifest 的 9 項修正也已通過 `--verify`，補足前輪尚未驗證的範圍。

在實際專案目錄完成 `build.command --no-app`（exit 0），四個根套件執行檔與五個獨立 Worker 均成功建置。接著實際執行 `run.command`，確認 GenImage 程序存活，Core Graphics 回報該程序具有可見的 1440×900 主視窗。建置、啟動、視窗證據及腳本測試日誌保存於 `Outputs/launch-validation`（不納入 Git）。啟動修正階段直接執行 Release 程式；App Bundle／DMG 的後續結果見下節。


### 1.26.0922 安裝包（2026-09-22～23）

以 `GENIMAGE_VERSION=1.26.0922`、Build `2357` 建立 Developer ID App，包含 Qwen 2.1 Worker、更新後 WebUI 與第三方授權。App 與所有 Worker 的 Team ID、hardened runtime、secure timestamp 均驗證通過；DMG 的完整性、簽章與唯讀掛載後的內含 App 簽章也通過。

專案位於 ExFAT。此次打包修正 WebUI 比對及 dylib 處理對 AppleDouble 中繼資料的誤判，並新增 `GENIMAGE_DIST_DIR`，在內部 APFS 製作 App 與 DMG。WebUI 正常複本可通過內容檢查，刻意修改 HTML 後仍會拒絕；完整資源及深層簽章檢查沒有略過。DMG 複製回專案 `dist/` 後 SHA-256 一致。

2026-09-23 完成 Apple 公證。App 與 DMG 均獲得 Accepted、附加公證票據並通過 Staple 驗證；Gatekeeper 顯示 `accepted`、`source=Notarized Developer ID`。複製回 ExFAT 的最終 DMG，以及從該 DMG 唯讀掛載的 App，也均通過票據與 Gatekeeper 驗證。

- App 公證 ID：`df0a5eac-9278-4952-bf22-8240a9825d6c`。
- DMG 公證 ID：`0b5c7965-a081-4b40-81e0-e82c0e8ec3fe`。

最終 DMG SHA-256：`bf2d839ad9803da300f54387f57b42fda9cfef7e74255c8cd5d069f93fa51b03`。本機建置、測試及安裝包日誌位於 `Outputs/release-1.26.0922`（不納入 Git）。


## 1.26.1003 安裝包（2026-10-03）

版本 `1.26.1003`、Build `1236`。完整執行 Release 打包，根套件四個出貨產品與五個獨立生成 Worker 均建置成功，並驗證 Runtime patch 清單。安裝包包含圖片風格、影片鏡頭、H3 低步數 LoRA 與記憶體最佳化的原始碼版本；模型權重不包含於 App。

App／DMG 在內部 APFS 製作，兩次 Apple 公證均為 **Accepted**，票據附加與驗證成功。DMG 複製至專案 `dist/` 後逐位元組雜湊一致，通過磁碟映像完整性、簽章、票據及 Gatekeeper 檢查；再以唯讀方式掛載，核對 App 版本與 Build，並驗證內含 App 的簽章、票據及 Gatekeeper。主程式與 7 個 Helper 的 Developer ID Team、hardened runtime、secure timestamp 均符合。未將未公證或驗證失敗的檔案列為發布資產。

- App 公證 ID：`e05bf83e-77ce-49ec-a1dc-3aab6b3493af`。
- DMG 公證 ID：`cf591ddf-8c8a-474d-8596-4e64d0dd8409`。
- DMG：`GenMedia-1.26.1003-arm64.dmg`，214,612,336 bytes。
- SHA-256：`f2a9eedf21b34374b0c8565f9873cc3ac733f6b3e8d5705fb86c03634615d5b8`。

打包後另以 256×256、4 幀的無模型請求檢查 H3 Helper，確認能識別新 LoRA 協定並在步數不符時回報錯誤；此項沒有生成影片。新的 LoRA 數值、採樣與小尺寸生成驗證依本文最新結果及 [LoRA 指南](LORAS.md) 的範圍為準。

本機日誌、來源雜湊、公證狀態及掛載驗證紀錄位於 `Outputs/release-1.26.1003/`（不納入 Git）。本版未重新執行完整根套件測試，既有程序等待案例的限制維持公開記錄。

# 驗證方式與結果

最近更新：2026-09-23。本文記錄可重複執行的檢查，以及各次結果的實際範圍。

## 最新結果

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
swift build --build-tests -j 4
swift test --skip-build -j 4
```

MLX 測試需要與目前相依版本匹配的 Metal library。此次 Swift 6.4 使用的 SwiftPM 建置方式將 library 輸出到以下位置；若測試無法找到它，可在測試期間放到套件根目錄，完成後移除該暫存副本：

```sh
(
  if [ -e ./default.metallib ] || [ -L ./default.metallib ]; then
    echo "default.metallib 已存在；請先確認其來源，不自動覆蓋。" >&2
    exit 1
  fi
  cp .build/out/Products/Debug/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib ./default.metallib || exit 1
  trap 'rm -f ./default.metallib' EXIT
  swift test --skip-build -j 4
)
```

其他 SwiftPM 版本或建置後端的輸出位置可能不同。請使用本次建置產生的 library，不要覆蓋既有的自訂檔案，也不要將額外檔案放進已簽章的 `.xctest/Contents/MacOS/`；這會影響 bundle 簽章驗證。

本輪完整驗證在內部 APFS 暫存副本進行，來源、測試、資源與套件設定共 183 個檔案均與專案逐檔一致。原 ExFAT 工作區的 AppleDouble 中繼資料曾影響簽章；正式建置腳本不再提供外接磁碟專用的中繼資料清理流程。測試副本不包含模型權重；Qwen-Image 2.1 實跑另讀取專案 Models 下的固定 revision 權重。

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

## 這次涵蓋的回歸案例

- Profile 的 64 GB 可見性邊界、自訂 Profile 保存與工作區讀取失敗保護。
- 模型掃描的過期結果、同模型取消清理順序、目錄切換等待移除完成。
- 本機 HTTP 下載的取消、續傳資訊寫入、大小不符與既有目的檔保留。
- safetensors 有上限標頭讀取、LoRA 分塊轉換與內容一致性。
- Worker 重用、取消、閒置回收、stdin 滿管線、大量日誌後退出及無換行的最後事件。
- 不同工作區的共用媒體、檔名更新、符號連結及仍被引用的快取保留。
- 媒體 URL scheme 停止後不再回呼、Range 請求與 MCP HTTP 存取限制。

## 限制

此次沒有執行所有真實模型的完整推論、完整 WebKit 操作、長時間磁碟斷線／重連、Release App／DMG 簽署公證或發佈。

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

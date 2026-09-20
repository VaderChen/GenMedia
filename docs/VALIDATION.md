# 驗證方式與結果

最近更新：2026-09-21。本文記錄可重複執行的檢查，以及各次結果的實際範圍。

## 最新結果

| 範圍 | 結果 | 說明 |
| --- | --- | --- |
| 根套件 Debug 編譯 | 通過 | 包含 App、Runtime、MCP、GGUF 及全部根套件測試目標 |
| Swift Testing | 141 項通過 | Core 64、Runtime 60、MCP 6、GGUF 9、App 2；另含參數化案例 |
| 維護腳本 | 7 項通過 | 在隔離檔案樹測試備份／清理、FFmpeg 失敗回復及授權打包 |
| Web UI | 2 項通過 | 輕量活動狀態合併與結構變更通知 |

使用 Apple Silicon、Swift 6.4 與 macOS 27 SDK。專案的最低部署目標與這次實際測試環境不同；此結果不代表舊版作業系統、工具鏈或所有模型組合均已驗證。

根套件測試不會自動執行 `RuntimeSupport/` 的獨立 Worker 測試。2026-09-19 曾完成 Z-Image Worker 建置與 H3 影片寫入 5 項測試，屬於歷史結果，沒有併入上述 141 項。詳細紀錄見[效能修正紀錄](PERFORMANCE_CHANGES.md)。

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

本輪完整驗證在內部 APFS 暫存副本進行，來源、測試、資源與套件設定共 161 個檔案均與專案逐檔一致。原 ExFAT 工作區的 AppleDouble 中繼資料曾影響簽章；正式建置腳本不再提供外接磁碟專用的中繼資料清理流程。測試副本不包含模型權重，驗證後已移除。

## 維護腳本與 Web UI

```sh
python3 -m unittest discover -s Tests/Scripts -v
node --test Tests/WebUI/*.test.mjs
git diff --check
```

維護腳本測試會建立暫存專案與替身工具，不修改真實模型、FFmpeg 安裝或使用者備份。

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

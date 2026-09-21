# README 操作 GIF 重錄

`record.cjs` 透過 Playwright 操作目前專案的 WebUI，再由 `encode.py` 合成 `images/operation-demo.gif`。包含提示詞、圖片比例、影片設定、歌詞／音樂風格、模型搜尋，以及簡單 MV 四個分頁的建立。動畫約 23 秒、1280 × 857，無限循環。

這是**介面操作示範**：`fixture.json` 的模型與 Profiles 摘自 `Sources/GenImageCore/ModelCatalog.swift`，全部標為未安裝；範例圖片使用專案 App 圖示。Native bridge 僅在隔離瀏覽器內回應設定及工作區操作，不會下載模型、執行推論或讀寫使用者的 App 資料。畫面上方持續標示示範資料。

## 執行

需要 Node.js、Playwright 與 Chromium，以及 Python 3 和 Pillow。依賴可安裝在外部工具目錄，不必修改 Swift 專案。

```sh
# Playwright 已可由 Node.js 載入，且已安裝 Chromium
node scripts/readme-demo/record.cjs

# 指定其他輸出位置
node scripts/readme-demo/record.cjs /tmp/operation-demo.gif
```

可使用 `NODE_PATH` 指定既有 Playwright 模組目錄、`CHROMIUM_EXECUTABLE` 指定 Chromium 執行檔、`PYTHON` 指定具有 Pillow 的 Python。腳本只開啟綁定 `127.0.0.1` 的暫時 HTTP server，結束後關閉 browser 與 server。

輸出會列出 GIF 大小、片段時間與暫存 PNG 路徑。錄製時會檢查瀏覽器錯誤、16:9 選項、LTX 搜尋結果及四個流程分頁；編碼後逐幀解碼檢查。更新模型目錄或介面 selector 時，請同步調整 fixture／錄製腳本，並檢查成品的文字、操作節奏及四份 README 的相對路徑。

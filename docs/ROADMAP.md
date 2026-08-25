# 開發路線

繁體中文 | [English](ROADMAP.en.md) | [日本語](ROADMAP.ja.md) | [한국어](ROADMAP.ko.md)

## 已完成：核心能力

- Swift Package、macOS 14+ App、Apple Silicon 原生 MLX／Core ML 推論與混合式 Web UI。
- 文生圖、圖生文、圖生圖、文生影、圖生影、文生音樂及 4× Upscale 獨立服務與資產 lineage。
- Z-Image Turbo、Qwen3-VL、Qwen 2511、LTX-2.3、ACE-Step 1.5、MiniMax Music 3 與 Real-ESRGAN Profile。
- 音樂 Prompt、可選歌詞、常見音樂風格、5～300 秒設定，以及 MP3／M4A／AAC／FLAC 輸出。
- 模型中心下載、暫停、續傳、磁碟預檢、修復、刪除、Profile 相依檢查與安裝後自動排序。
- 工作佇列、取消、進度、預估剩餘時間、生成耗時、模型快取及手動記憶體釋放。
- 每個工作區分頁作為持久化生成專案；App 重啟後恢復資產、操作、選取狀態與 Profile 快照。
- 創作設定區域渲染、輸入游標與 IME 保護，以及音訊／影片連續播放。
- `Image-YYYYMMDD-HHmm`、`Video-YYYYMMDD-HHmm`、`Music-YYYYMMDD-HHmm` 輸出命名與同分鐘防碰撞流水號。
- Release App bundle、MLX metallib、原生 MCP 推論工具及獨立 DMG 簽章／公證流程。

## 目前階段：穩定與驗證

1. 完成 ACE-Step 長音訊的 16GB、24GB、32GB 記憶體、熱壓與取消恢復測試。
2. 擴充模型下載中斷、雜湊不符、磁碟不足及重新啟動恢復測試。
3. 驗證多分頁專案在 App 異常退出、資產遺失與索引修復情境下的一致性。
4. 完成各 Runtime、模型與 LoRA 的授權頁面及散佈清單。

## 後續

1. 加入 Profile 匯入／匯出、版本遷移與相容性檢查。
2. 擴充更多 Apple Silicon 原生 MLX 媒體生成引擎。
3. 擴充 MCP 的影片、音樂與專案工作流工具。
4. 建立可重現的模型與 Runtime 效能基準。

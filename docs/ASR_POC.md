# ASR 字幕 PoC

繁體中文 | [English](ASR_POC.en.md) | [日本語](ASR_POC.ja.md) | [한국어](ASR_POC.ko.md)

`GenImageASRPoC` 是獨立的本機語音辨識驗證程式，不會修改 GenMedia 主 App 的 UI、工作區或資產資料。它使用原生 Swift／Core ML 的 WhisperKit，將影片或音訊檔轉成帶時間軸的原文字幕。

主 App 已整合完整字幕流程，並支援 Whisper Small（預設速度優先）與 Whisper Large v3 Turbo、中文 Paraformer Large、日文 Parakeet 0.6B、SRT／WebVTT 資產與可選的 Qwen MLX 字幕翻譯。PoC 保留為 WhisperKit 媒體解碼、語言辨識與時間碼的獨立驗證入口。

## 支援的輸入

PoC 會透過 macOS `AVFoundation` 讀取媒體並抽取音訊軌，因此可以接受：

- 有音訊軌的影片，例如 MP4、MOV、M4V
- 音訊檔，例如 WAV、M4A、MP3、AAC、AIFF，以及 macOS 可解碼的其他格式

沒有音訊軌的影片會直接回報錯誤。PoC 目前不使用 Python、FFmpeg 或外部 ASR CLI。

## 執行

第一次執行會下載 WhisperKit Core ML 模型，預設放在：

```text
~/Library/Application Support/GenImage/Models/WhisperKit/
```

輸出預設放在：

```text
~/Library/Application Support/GenImage/Generated/ASR/
```

```bash
swift run GenImageASRPoC --input "/path/to/video.mp4"
swift run GenImageASRPoC --input "/path/to/video.mp4" --input "/path/to/audio.m4a"
swift run GenImageASRPoC "/path/to/audio.wav" --language zh
```

可使用 `--language auto` 自動偵測語言，也可以指定 `zh`、`ja`、`ko` 或 `en` 等語言代碼。Whisper 的中文輸出可能使用簡體字；若要輸出繁體字幕，可使用 `--chinese-script traditional`。預設 `asr` 會保留模型原始字形：

```bash
swift run GenImageASRPoC \
  --input "/path/to/video.mp4" \
  --language zh \
  --chinese-script traditional \
  --word-timestamps \
  --output-dir "/path/to/asr-output"
```

若已經準備好本機模型，可以避免下載：

```bash
swift run GenImageASRPoC \
  --input "/path/to/video.mp4" \
  --model-folder "/path/to/whisperkit-model" \
  --no-download
```

## 輸出

每個輸入會產生三個檔案：

- `<檔名>-asr.json`：原文、辨識語言、模型、片段時間軸、信心資訊；加上 `--word-timestamps` 時包含單字時間軸
- `<檔名>-asr.srt`：一般字幕格式
- `<檔名>-asr.vtt`：WebVTT 格式

PoC 只驗證原文 ASR 與時間軸。主 App 的翻譯流程另接本機 Qwen 文生文模型，沿用同一批字幕片段的開始與結束時間，不使用 Whisper 的英文限定翻譯任務取代多語翻譯。

## PoC 邊界

- PoC 僅使用 WhisperKit；Paraformer、Parakeet 與 Qwen 翻譯由主 App Runtime 提供
- PoC 輸出獨立檔案，不建立 `MediaAsset`、workspace 或 lineage
- 說話者辨識
- 將字幕燒錄或封裝回影片

WhisperKit 與其 Core ML 模型的授權、模型來源與散佈條件，正式接入前需獨立列入依賴與模型授權清單。

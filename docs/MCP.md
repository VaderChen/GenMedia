# GenImage MCP Server

繁體中文 | [English](MCP.en.md) | [日本語](MCP.ja.md) | [한국어](MCP.ko.md)

GenImage 以同一套工具核心提供兩種 JSON-RPC 2.0 MCP transport，協定版本 `2025-06-18`：可獨立執行的 stdio server，以及由 GenMedia.app 設定 Switch 控制的 localhost HTTP API。

## 獨立 stdio

```bash
./build.command
MCP_BIN_DIR="$(swift build -c release --show-bin-path)"
"$MCP_BIN_DIR/GenImageMCP"
```

正式整合請使用 `build.command` 輸出的絕對路徑；單獨執行 `swift build` 不會複製 metallib。

執行 `swift build -c release --show-bin-path` 可取得目前工具鏈實際使用的 Release 目錄；不要硬編碼舊版 SwiftPM 的 `.build/arm64-apple-macosx/release` 路徑。

stdio server 自己持有 `InferenceServices` 並在 `GenImageMCP` 行程內完成推論，不需要 GenMedia.app 執行。

## App HTTP API

在 GenMedia.app 的「設定 → MCP 整合」開啟 Switch 後，畫面會顯示：

```text
http://127.0.0.1:12181/mcp
```

此端點只綁定本機並接受 HTTP `POST` JSON-RPC；關閉 Switch 會立即停止監聽。HTTP transport 直接呼叫與 stdio 相同的 `MCPServer` 工具核心，不會啟動 stdio 子行程，也不是轉送到另一個外部服務。HTTP 模式屬於 App 內嵌服務，因此使用時 GenMedia.app 必須保持執行；需要 headless 整合時請使用上方的獨立 stdio 執行檔。

## MCP 方法

- `initialize`
- `notifications/initialized`
- `ping`
- `tools/list`
- `tools/call`

stdio 訊息為一行一個 UTF-8 JSON-RPC 物件。stdout 只輸出協定訊息；診斷資訊應寫入 stderr。

## 工具

- `genimage_models_list`
- `genimage_profiles_list`
- `genimage_upscale_image`
- `genimage_generate_image`
- `genimage_edit_image`
- `genimage_describe_image`
- `genimage_generate_subtitle`

文生圖、圖生圖、圖生文與可選字幕翻譯使用原生 MLX Swift Runtime；字幕辨識使用原生 Core ML Runtime。Release 執行檔旁必須有 `mlx.metallib`，`build.command` 會自動處理。

模型根目錄可透過 `model_root` 或 `GENIMAGE_MODEL_ROOT` 指定。

### `genimage_generate_subtitle`

必要參數為絕對來源路徑 `input_path`、ASR 模型目錄 `model_path` 與模型識別碼 `model_id`。支援的 ASR 模型 ID：

- `argmaxinc/whisperkit-coreml@large-v3-turbo`：多語言 Whisper
- `FluidInference/paraformer-large-zh-coreml`：中文 Paraformer
- `FluidInference/parakeet-0.6b-ja-coreml`：日文 Parakeet

`format` 可為 `srt` 或 `vtt`，預設 `srt`；`language_code` 可指定來源語言或 `auto`；`output_path` 可指定尚不存在且副檔名相符的絕對輸出路徑。若提供 `target_language_code`，必須同時提供 `translation_model_path` 與 `translation_model_id`，由本機 Qwen 文生文模型翻譯並保留原始時間軸。支援的 25 種目標語言代碼為 `zh-Hant`、`zh-Hans`、`en`、`ja`、`ko`、`es`、`fr`、`de`、`it`、`pt`、`ru`、`ar`、`hi`、`bn`、`id`、`vi`、`th`、`tr`、`pl`、`nl`、`sv`、`cs`、`uk`、`ms`、`fil`。

這個工具直接在 `GenImageMCP` 行程內建立 `SubtitleGenerationRouter` 與推論服務；它不連線到 GenMedia.app，也不要求 App 執行。

## 設定範例

使用編譯後的執行檔可避免 MCP Client 的工作目錄差異：

```json
{
  "mcpServers": {
    "genimage": {
      "command": "/absolute/path/to/GenImageMCP",
      "env": {
        "GENIMAGE_MODEL_ROOT": "/absolute/path/to/models"
      }
    }
  }
}
```

MCP Client 應使用 `build.command` 產生的 Release 執行檔，確保 MLX Runtime 與執行檔位於同一層。

## 已驗證行為

- `initialize` 與列出七個工具的 `tools/list` 煙霧測試通過。
- App HTTP transport 可在 localhost 直接回應 `tools/list`，並與 stdio 回傳相同七個工具。
- 未知方法會回傳 JSON-RPC `-32601`。
- `genimage_generate_image` 已使用本機 Z-Image Turbo Q4 完成 256×256、1-step 端到端生成。
- `genimage_describe_image` 已使用本機 Qwen3-VL 4-bit 完成繁體中文圖片描述。
- `genimage_upscale_image` 已使用本機 Real-ESRGAN Core ML 模型完成 4× 放大。
- 模型內部 Logger 不會寫入 stdout，stdio 保持純 JSON-RPC。

# 本機模型測試報告

繁體中文 | [English](MODEL_TEST_REPORT.en.md) | [日本語](MODEL_TEST_REPORT.ja.md) | [한국어](MODEL_TEST_REPORT.ko.md)

測試日期：2026-08-03
測試平台：Apple M4、macOS、Xcode 26.6、Swift 6.3

模型根目錄：本機測試目錄（未納入版本控制）

## 文生圖

模型：`z-image-turbo-q4`

- 大小：約 7.8GB。
- Diffusers `ZImagePipeline` 目錄結構完整。
- Transformer 與 text encoder 為 4-bit affine、group size 32。
- `Z-Image.swift` commit `28bfcf3148c041a554629247170eb54d9ac46830` 成功載入。
- 256×256、1-step、Seed 1 測試成功。
- 產生 256×256 RGBA PNG。
- 模型載入、Prompt encoding、denoising、VAE decoding 與存檔皆完成。
- MCP `genimage_generate_image` 端到端測試成功：256×256、1-step、Seed 7，輸出為 8-bit RGBA PNG。
- App 使用的 `ZImageTextToImageService` 已直接包裝原生 `ZImagePipeline`，不再呼叫外部 CLI。

SwiftPM 執行需要 `mlx.metallib`。相容檔案已存放於 `RuntimeSupport`，`build.command` 會複製到 Release 執行檔旁，不再於執行時依賴其他 App。

## Upscale

模型：

- `upscale/realesrgan512.mlmodel`
- `upscale/realesrganAnime512.mlmodel`

兩者都已通過 `coremlcompiler compile`：

- 輸入：RGB image，512×512。
- 輸出：RGB image，2048×2048。
- 放大倍率：4×。

## 圖生文

模型：`Qwen3-VL-4B-Instruct-4bit`

- 模型架構：`Qwen3VLForConditionalGeneration`。
- `model_type`：`qwen3_vl`。
- 量化：4-bit affine、group size 64。
- 模型、tokenizer、chat template、preprocessor 設定完整。

`mlx-swift-lm` revision `7da33441c7c08b010ff1aa8da9dc3d82277272f5` 的 Qwen 多模態 Runtime 已接入：

- 本機 4-bit 模型載入成功。
- Z-Image 生成的 PNG 可直接作為輸入。
- 96-token 繁體中文圖片描述測試成功。
- App 會將描述放入文生圖 Prompt，供後續修改或直接生成。

## MCP 驗證

標準 JSON-RPC 2.0 stdio MCP server 已完成以下煙霧與端到端測試：

- `initialize` 回傳協定版本 `2025-06-18`。
- `tools/list` 可列出模型、Profile、Upscale、文生圖、圖生圖、圖生文與字幕生成七項工具。
- `genimage_generate_image` 已直接呼叫原生 Z-Image Runtime，並確認輸出 PNG 尺寸與格式。
- `genimage_describe_image` 已直接呼叫原生 Qwen3-VL Runtime，並輸出繁體中文描述。
- `genimage_upscale_image` 使用本機 Real-ESRGAN Core ML 模型完成 4× 輸出。
- stdout 已驗證只包含逐行 JSON-RPC 回應，不包含模型 Logger。

# ローカルモデルテスト報告

[繁體中文](MODEL_TEST_REPORT.md) | [English](MODEL_TEST_REPORT.en.md) | 日本語 | [한국어](MODEL_TEST_REPORT.ko.md)

テスト日：2026-08-03
テスト環境：Apple M4、macOS、Xcode 26.6、Swift 6.3

モデルルート：ローカルテストディレクトリ（バージョン管理には含まれません）

## テキストから画像

モデル：`z-image-turbo-q4`

- サイズ：約 7.8 GB。
- Diffusers `ZImagePipeline` のディレクトリ構成が完全です。
- Transformer と text encoder は 4-bit affine、group size 32 です。
- `Z-Image.swift` commit `28bfcf3148c041a554629247170eb54d9ac46830` の読み込みに成功しました。
- 256×256、1-step、Seed 1 のテストに成功しました。
- 256×256 RGBA PNG を生成しました。
- モデル読み込み、Prompt encoding、denoising、VAE decoding、ファイル保存がすべて完了しました。
- MCP `genimage_generate_image` の 256×256、1-step、Seed 7 エンドツーエンドテストに成功し、8-bit RGBA PNG を出力しました。
- アプリの `ZImageTextToImageService` はネイティブ `ZImagePipeline` を直接ラップし、外部 CLI を呼び出しません。

SwiftPM の実行には `mlx.metallib` が必要です。互換ファイルは `RuntimeSupport` に保存され、`build.command` が Release 実行ファイルの隣へコピーするため、実行時に他のアプリへ依存しません。

## アップスケール

モデル：

- `upscale/realesrgan512.mlmodel`
- `upscale/realesrganAnime512.mlmodel`

両方のモデルが `coremlcompiler compile` に合格しています。

- 入力：RGB image、512×512。
- 出力：RGB image、2048×2048。
- 拡大倍率：4 倍。

## 画像からテキスト

モデル：`Qwen3-VL-4B-Instruct-4bit`

- モデルアーキテクチャ：`Qwen3VLForConditionalGeneration`。
- `model_type`：`qwen3_vl`。
- 量子化：4-bit affine、group size 64。
- モデル、tokenizer、chat template、preprocessor の設定が完全です。

`mlx-swift-lm` revision `7da33441c7c08b010ff1aa8da9dc3d82277272f5` の Qwen マルチモーダル Runtime を統合済みです。

- ローカル 4-bit モデルの読み込みに成功しました。
- Z-Image が生成した PNG をそのまま入力として使用できます。
- 96-token の繁体字中国語画像説明テストに成功しました。
- アプリは説明をテキストから画像の Prompt に入力し、編集または直接生成に利用できます。

## MCP 検証

標準 JSON-RPC 2.0 stdio MCP server で次のスモークテストとエンドツーエンドテストを完了しています。

- `initialize` はプロトコル版 `2025-06-18` を返します。
- `tools/list` はモデル、プロファイル、アップスケール、テキストから画像、画像編集、画像からテキスト、字幕生成の 7 ツールを返します。
- `genimage_generate_image` はネイティブ Z-Image Runtime を直接呼び出し、出力 PNG のサイズと形式を確認しました。
- `genimage_describe_image` はネイティブ Qwen3-VL Runtime を直接呼び出し、繁体字中国語の説明を出力しました。
- `genimage_upscale_image` はローカル Real-ESRGAN Core ML モデルで 4 倍出力を生成しました。
- stdout には行単位の JSON-RPC 応答だけが含まれ、モデル Logger は含まれません。

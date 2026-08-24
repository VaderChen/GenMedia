# GenMedia

[繁體中文](README.md) | [English](README.en.md) | 日本語 | [한국어](README.ko.md)

GenMedia は **Apple Silicon をネイティブサポート**するローカル AI メディア生成アプリです。本プロジェクトは、次の機能を備えたビルド可能なハイブリッドアプリケーションを提供します。

- Swift がモデル、プロファイル、ジョブキュー、ファイル、MLX／Core ML 推論を管理します。
- `WKWebView` に HTML、CSS、JavaScript UI を組み込み、ネットワーク接続や npm ランタイムを必要としません。
- テキストから画像、画像からテキスト、画像から画像、テキストから動画、画像から動画、テキストから音楽、アップスケールを個別に実行でき、アセットの系譜を通じて連携できます。
- 各操作でプロファイルのスナップショットを保存し、モデルやアーキテクチャの更新後も当時のバージョンを追跡できます。
- 専用の設定画面で繁体字中国語、英語、日本語、韓国語、および永続化可能な 6 種類のカラーテーマを利用できます。
- 標準 JSON-RPC 2.0 stdio MCP サーバーを Agent や自動化ツールから利用できます。

## プレビュー

![GenMedia メディア生成インターフェース](images/cap001.jpg)

## 実行

要件：macOS 14 以降、Apple Silicon、Xcode 16 以降。

```bash
./build.command
./run.command
```

`build.command` は Release 実行ファイルと標準の `GenMedia.app` を `dist/` に生成します。App には WebUI リソース、MLX Metal ランタイム、MCP サーバー、モデル診断ツールが含まれます。

```bash
# Release 実行ファイルと App をビルド
./build.command

# App bundle を作成せずに増分 Release ビルドのみ実行
./build.command --no-app

# バージョンと Bundle ID を指定
GENIMAGE_VERSION=1.1.0 GENIMAGE_BUNDLE_ID=com.example.genimage ./build.command
```

`run.command` は自動的に `--no-app` を使用するため、日常の起動で App bundle を繰り返し生成しません。配布用 DMG は、Developer ID Application 署名、Apple の公証、Staple、Gatekeeper 検証を行う別のローカルフローで処理します。

### 動画 Runtime

動画生成には交換可能な外部 `ltx-2-mlx` Runtime を使用します。Swift アプリがプロファイル、パラメータ検証、ジョブキュー、キャンセル、進捗、アセット、動画再生を管理します。初回使用前に CLI と FFmpeg をインストールしてください。

```bash
brew install uv ffmpeg
./scripts/install-ltx-runtime.command
```

アプリは `GENIMAGE_LTX_RUNTIME`、`GENIMAGE_LTX_RUNTIME_ROOT/.venv/bin/ltx-2-mlx`、App Helpers、`~/.local/bin/ltx-2-mlx`、一般的な Homebrew パス、`PATH` の順に検索します。実行ファイルが独自の場所にある場合は、次のように指定します。

```bash
GENIMAGE_LTX_RUNTIME="/absolute/path/to/ltx-2-mlx" ./run.command
```

`ltx-2-mlx` は既定で Gemma テキストエンコーダー設定を使用します。ローカルの Gemma モデルがある場合は、`GENIMAGE_LTX_GEMMA_MODEL` にモデルディレクトリまたは Hugging Face ID を指定できます。現在のアプリ DMG には Python Runtime、Gemma の重み、FFmpeg は含まれていません。正式配布前に、これらを任意の外部コンポーネントとして扱い、Runtime とモデルのライセンスを個別に確認してください。

### プロファイル、ジョブ、メモリ

- プロファイルは「使用中、利用可能、ダウンロード中、利用不可」の順に並びます。モデルと LoRA の依存関係が揃ったプロファイルは淡い緑色の枠で表示され、ダウンロード完了時に直ちに再整列されます。
- キャンセル時はまず `cancelling` になり、Runtime Task の終了後に `cancelled` へ移行して生成・メモリ関連の操作を再び有効にします。ETA は進捗 35% かつ開始 15 秒後から数値表示し、安定したサンプルが不足する場合は全体経過時間を使用します。
- Z-Image MLX 互換レイヤーは `quantize_config.json`、affine／mxfp4、packed pad token、FP16 から BF16 への読み込みを処理します。`build.command` は Swift Package 解決後に `Patches/` の修正を自動適用します。andrevp Z-Image Turbo MLX 4-bit は実際の画像生成で検証済みです。
- テキストから画像の完了後もモデル重みと暖機 buffer を保持します。5 分間アイドルになると再利用可能な MLX 一時 buffer だけを整理し、モデルはアンロードしません。側面のメモリ解放、モデル切り替え、またはプロファイル切り替え時の RAM 90% 超過保護でのみ不要な Runtime を解放します。
- ダウンロードでは配布元のファイル名を保持し、生成出力は `Image-MMDD-HHmmss` または `Video-MMDD-HHmmss` を使用します。出力ディレクトリは設定画面で変更できます。

### 音楽 Runtime

テキストから音楽は、外部 `mlx-minimax-music3` Runtime とモデルセンターの MiniMax Music 3 MLX 8-bit プロファイルを使用します。Swift アプリが音楽スタイル、任意のプロンプト、任意の歌詞、5～300 秒の長さ（最長 5 分）、ステップ、シード、キャンセル、残り時間推定、音声アセット情報を管理します。プロンプトが空欄の場合は選択した音楽スタイルを使用し、歌詞が空欄の場合はインストゥルメンタルを生成します。Runtime の一時 WAV は FFmpeg で選択した MP3、M4A、AAC、FLAC に変換され、完了後に WAV、テキスト、ログの一時ファイルを削除します。

アプリは `GENIMAGE_MINIMAX_MUSIC3_RUNTIME`、App Helpers、`~/Library/Application Support/GenImage/Runtime/minimax-music3/.venv/bin/mlx-minimax-music3`、一般的なインストール先、`PATH` の順に検索します。Runtime、モデル、FFmpeg は任意の外部コンポーネントとして App bundle／DMG には含めず、MiniMax Music 3 モデルには Community License が適用されます。

## 検証

```bash
swift test

for file in Sources/GenImageApp/Resources/WebUI/js/*.js; do
  node --check "$file"
done
```

ローカルモデルと自動生成されたプロファイルを診断します。

```bash
swift run GenImageDoctor

# または独自のモデルディレクトリを指定
GENIMAGE_MODEL_ROOT="/path/to/models" swift run GenImageDoctor
```

標準 MCP stdio サーバーを起動します。

```bash
.build/arm64-apple-macosx/release/GenImageMCP
```

MCP は `initialize`、`ping`、`tools/list`、`tools/call` をサポートします。ツールにはローカルモデル、プロファイル、ネイティブ Z-Image のテキストから画像生成、Qwen3-VL の画像説明、Core ML アップスケールが含まれます。

MCP のエンドツーエンド検証は完了しています。`genimage_generate_image` はローカル Z-Image Turbo Q4 で PNG を出力し、`genimage_describe_image` は Qwen3-VL で繁体字中国語の説明を生成し、`genimage_upscale_image` はローカル Real-ESRGAN Core ML モデルで 4 倍のアップスケールを実行します。

## プロジェクト構成

```text
Sources/
├── GenImageCore/
│   ├── DomainModels.swift        # アセット、レシピ、ジョブ、モデル、プロファイル
│   ├── InferenceServices.swift   # 画像、テキスト、動画、音楽の推論インターフェース
│   ├── ModelCatalog.swift        # 組み込みモデルとプロファイル
│   ├── OutputFileNaming.swift    # 画像・動画の出力名
│   └── WorkflowGraph.swift       # アセットの系譜と分岐関係
├── GenImageRuntime/
│   ├── ZImageTextToImageService.swift
│   ├── QwenVLImageDescriptionService.swift
│   ├── Qwen2511ImageToImageService.swift
│   ├── LTXVideoGenerationService.swift
│   ├── MiniMaxMusic3GenerationService.swift
│   └── CoreMLUpscaleService.swift
└── GenImageApp/
    ├── AppStore.swift            # アプリケーション状態とジョブ調整
    ├── HybridBridgeController.swift
    ├── HybridWebView.swift
    ├── AssetSchemeHandler.swift  # ローカル画像、動画、音声を安全に WebUI へ提供
    └── Resources/WebUI/          # HTML/CSS/JavaScript フロントエンド
Patches/                           # ビルド時に適用する Z-Image MLX 互換修正
```

## 現在の状態

アプリは Z-Image Turbo のテキストから画像、Qwen3-VL の画像からテキスト、Qwen 2511 の画像から画像、LTX-2.3 MLX のテキストから動画／画像から動画、MiniMax Music 3 MLX 8-bit のテキストから音楽、Core ML Real-ESRGAN のアップスケールによるローカル推論に接続済みです。動画は MP4、音楽は MP3／M4A／AAC／FLAC として追加され、実際の長さ、サンプルレート、チャンネル数、プロファイルスナップショット、系譜を保持します。

詳細情報：

- [更新履歴](UpdateNote.md)
- [アーキテクチャ](docs/ARCHITECTURE.ja.md)
- [Web Bridge](docs/WEB_BRIDGE.ja.md)
- [ロードマップ](docs/ROADMAP.ja.md)
- [MCP インターフェース](docs/MCP.ja.md)
- [ローカルモデルテスト報告](docs/MODEL_TEST_REPORT.ja.md)

## ライセンス

本プロジェクトは GPLv3 と商用ライセンスのデュアルライセンス方式を採用しています。

- オープンソースでの利用は [GNU General Public License v3.0](LICENSE) に基づきます。
- クローズドソースへの統合、プロプライエタリ製品の配布、個別の商用条件など、GPLv3 に準拠できない、または準拠を希望しない場合は、著作権者に連絡して別途商用ライセンスを取得してください。

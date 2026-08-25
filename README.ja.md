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
- ダウンロードでは配布元のファイル名を保持し、生成出力は `Image-YYYYMMDD-HHmm`、`Video-YYYYMMDD-HHmm`、`Music-YYYYMMDD-HHmm` を使用します。同じ分に重複する場合は連番を追加します。出力ディレクトリは設定画面で変更できます。
- 開いている各ワークスペースタブを生成プロジェクトとして扱います。アセットと lineage は Application Support にアトミック保存され、アプリ再起動後も復元されます。タブを明示的に閉じた場合のみプロジェクトのワークスペース索引を削除し、出力済みメディアはディスクに保持します。
- プロンプトと歌詞の編集中は、カーソル、選択範囲、IME の変換状態をネイティブ状態更新から保護します。生成タイプ、プロンプト、歌詞、出力設定のタブは作成パネルだけを再描画します。避けられない全体更新でも、再生中の音声・動画ノードを再利用して再生を中断しません。
- ワークスペースのフィルムストリップには画像読み込みボタンがあり、Finder から PNG、JPEG、WebP、GIF、TIFF、HEIC、HEIF を 1 枚以上ドロップできます。音楽生成中はメディアソースの混在を防ぐため画像読み込みを無効にします。画像生成で入力画像を選択するとメインボタンは画像から画像のプロファイルを使用し、未選択時はテキストから画像のプロファイルを使用します。
- 画像と動画の比率項目はドロップダウンになっています。画像から画像では入力画像を選択した場合だけ「元の解像度」を表示し、入力画像のサイズを Runtime が扱える 16 の倍数へ変換します。
- 画像から画像の幅と高さは Qwen Image Edit Runtime へ実際に渡されます。指定比率が入力画像と異なる場合、入力画像は自身の解像度を保ったまま端の延長で出力比率のキャンバスに配置してから生成するため、元画像全体を保持できます。
- 条件画像は生成解像度で符号化し、条件グリッドと出力グリッドの RoPE 位置を完全に一致させます。条件グリッドを 1024² 面積に固定すると、モデルは入力画像の中央だけに整列し、出力は切り取られた拡大結果になります。
- 生成解像度と出力解像度を分離しました。指定面積が 1024² 未満の場合、Runtime は指定比率のまま 1024² 面積（diffusers 参考実装と同じ約 4096 latent トークン）で生成し、Lanczos で指定サイズへ縮小します。学習時のトークン数を大きく下回ると DiT のノイズ除去は劣化し縞状に破綻するため、`128 × 192` のような低解像度は縮小で生成した方が明らかに高品質です。
- 指定面積が 1024² 以上の場合は指定サイズでそのまま生成し、縮小は行いません。解像度を上げるほど Runtime のメモリ使用量と生成時間も増加します。1024² 未満の出力も 1024² 面積で生成するため、出力を小さくしても生成時間は短くなりません。
- 画像から画像の生成開始時、出力面積が `512 × 512` を下回る場合は警告ダイアログを表示し、「キャンセル」か「このまま生成」を選べます。一度確認すると、その実行中は再表示しません。

### 音楽 Runtime

テキストから音楽は、`MusicGenerationRouter` がアクティブなプロファイルに応じて ACE-Step 1.5 または MiniMax Music 3 Adapter へ振り分けます。Swift アプリが音楽スタイル、任意のプロンプト、任意の歌詞、ステップ、シード、キャンセル、残り時間推定、音声情報を一貫して管理します。プロンプトが空欄の場合は選択したスタイルを使用し、歌詞が空欄の場合はインストゥルメンタルを生成します。各 Runtime の一時 WAV は共通 FFmpeg 層で MP3、M4A、AAC、FLAC に変換されます。

- **ACE-Step 1.5 Turbo MLX**：推奨プロファイルです。アプリ内蔵の Apple Silicon ネイティブ Swift／MLX Runtime により、追加サービスをインストールせずに 10～300 秒の歌またはインストゥルメンタルを生成します。コードとモデルは商用利用可能な MIT License です。長時間音声はオーバーラップ付き分割 VAE デコードでメモリ使用量を抑えます。

- **MiniMax Music 3 MLX 8-bit**：外部 `mlx-minimax-music3` CLI で動作します。5～300 秒の設定値は最長時間であり、曲構成に応じてモデルが早めに自然終了する場合があります。完了後、アプリに実際の出力時間を表示します。モデルには Community License が適用されます。

ACE-Step ネイティブ Runtime はアプリに同梱されます。モデル、MiniMax Music 3 Runtime、FFmpeg は任意の外部コンポーネントです。

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
│   ├── OutputFileNaming.swift    # 画像・動画・音楽の出力名
│   ├── ProjectWorkspacePersistence.swift # 開いている生成プロジェクトの永続化
│   └── WorkflowGraph.swift       # アセットの系譜と分岐関係
├── GenImageRuntime/
│   ├── ZImageTextToImageService.swift
│   ├── QwenVLImageDescriptionService.swift
│   ├── Qwen2511ImageToImageService.swift
│   ├── LTXVideoGenerationService.swift
│   ├── MusicGenerationRouter.swift
│   ├── ACEStepMusicGenerationService.swift
│   ├── MiniMaxMusic3GenerationService.swift
│   ├── AudioOutputEncoder.swift
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

アプリは Z-Image Turbo のテキストから画像、Qwen3-VL の画像からテキスト、Qwen 2511 の画像から画像、LTX-2.3 MLX のテキストから動画／画像から動画、ACE-Step 1.5 Turbo MLX と MiniMax Music 3 MLX 8-bit のテキストから音楽、Core ML Real-ESRGAN のアップスケールによるローカル推論に接続済みです。動画は MP4、音楽は MP3／M4A／AAC／FLAC として追加され、実際の長さ、サンプルレート、チャンネル数、プロファイルスナップショット、系譜を保持します。

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

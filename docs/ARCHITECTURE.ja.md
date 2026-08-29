# GenMedia アーキテクチャ

[繁體中文](ARCHITECTURE.md) | [English](ARCHITECTURE.en.md) | 日本語 | [한국어](ARCHITECTURE.ko.md)

## 設計目標

1. テキストから画像、画像からテキスト、画像から画像、動画生成、音楽生成、字幕生成、アップスケールは相互に依存しない独立した機能です。
2. 画像、動画、音声、字幕の出力からワークフローや分岐を構成できます。
3. UI は MLX、Core ML、特定のモデルに直接依存しません。
4. モデル更新時はプロファイルによってモデル版と推論アーキテクチャを切り替えます。
5. 過去の作品はプロファイルのスナップショットを保持し、その後のプロファイル変更による影響を受けません。

## レイヤー

```text
HTML / CSS / JavaScript UI
            │
            │ JSON commands + state snapshots
            ▼
HybridBridgeController (WKWebView Bridge)
            │
            ▼
AppStore / Workflow coordination
            │
            ├── Model manager
            ├── Asset repository
            ├── Job queue
            └── Profile registry
                    │
                    ▼
        Independent inference services
        ├── TextToImageGenerating
        ├── ImageDescribing
        ├── ImageToImageGenerating
        ├── VideoGenerating
        ├── MusicGenerating
        ├── MediaTranscribing
        ├── SubtitleGenerating
        ├── TextGenerating
        └── ImageUpscaling
                    │
                    ▼
       MLX Swift / Core ML / Local REST Service / External CLI
```

Web UI は Bridge を通じてのみローカル機能を利用できます。任意のファイル、モデルディレクトリ、システム API を直接読み取ることはできません。

### Web UI 更新戦略

- プロンプト、ネガティブプロンプト、歌詞フィールドのフォーカス中は、Swift から状態を受信してもローカル編集値を保持し、不要な全体再描画を遅延させることで、カーソル、選択範囲、IME 変換状態のリセットを防ぎます。
- 生成タイプとプロンプト／歌詞／出力設定タブは独立した作成パネル renderer を使用し、プレビュー、プレイヤー、Inspector、サイドバーの DOM を置き換えません。
- ワークスペースタブ schema v3 は、各タブの作業タイプ、Profile 参照、Prompt、出力設定、自動フロー工程を保存します。タブ切り替え時は単一の Bridge コマンドで Native の作成状態へ復元します。
- 全体状態の変更で完全な再描画が必要な場合は、再生中の `<audio>`、`<video>`、音声ビジュアライザーノードを一度切り離して同じアセット位置へ戻し、再生位置と Web Audio 接続を維持します。

## 内部構造の整理

今回の整理はコードの境界と責務だけを変更するもので、既存の生成機能、利用手順、Web Bridge プロトコルは変更しません。

- `ApplicationSupport` が `Models`、`Runtime`、`Workspace`、`Pasted`、`Generated` の Application Support パスを一元定義し、起動時に旧 `GenMedia` のワークスペースデータを引き継ぎます。
- `OutputGeometry` がサイズの上下限、16 の倍数への整列、比率変換、画像から画像の生成計画を管理します。Web UI の `js/geometry.js` はその鏡像であり、Native、MCP、UI の計算結果を統一します。
- `AppStore` は型宣言、保存プロパティ、初期化だけを保持し、Persistence、Paths、Selection、Profiles、OutputSettings、Assets、ImageGeneration、MediaGeneration、Jobs、ModelInstallation を責務ごとの `AppStore+*.swift` へ分離しました。
- Web UI はサイドバーとルーティング、ワークスペースタブ、サイズ計算、全体再描画の保護を `chrome.js`、`workspace-tabs.js`、`geometry.js`、`render-preservation.js` に分離し、`app.js` は Bridge とアプリケーション調整を担当します。
- `SubprocessRuntime` が外部 Worker、動画・音楽 CLI、FFmpeg の環境、ログ、進捗、停滞検出、キャンセル、終了処理を共通化します。
- `MediaCompatibilityService` は `ffmpeg`／`ffprobe` の唯一の検索・解析入口です。正式 App は `Contents/Resources/bin/` を優先し、開発実行時だけ環境変数、Homebrew、`PATH` へフォールバックします。
- `MediaSourceCompatibilityService` は読み込んだすべての動画・音声を `ffprobe` で分類します。直接再生できる原本は変更せず、H.264／HEVC は可能なら再多重化だけを行い、残りの場合にだけ VideoToolbox H.264／AAC または M4A AAC の再生プロキシを作成します。`AppStore+MediaImport` は読み込みと変換をキャンセル可能な Job にまとめて FFmpeg の進捗を返し、変換ビットレートは画像面積に応じて調整します。
- `AssetSchemeHandler` は時間メディアに HTTP Range を返し、バックグラウンドの `FileHandle` で 512 KiB ごとに読み込みます。停止済みタスク集合により WebKit がキャンセルした要求へのコールバックを防ぎ、画像は従来どおり一括応答します。起動時には `ApplicationSupport.orphanMediaCacheFiles` が対応するアセットのない UUID キャッシュを探して削除します。
- `MediaCompositionService` はモデル不要の画像ループ動画と映像・音声結合を、共通 FFmpeg サブプロセス、進捗、キャンセル、出力命名で実行します。結果は `generatedVideo` と `WorkflowOperation` として既存の系譜へ戻ります。
- `automatic-flow.js` は宣言的テンプレートから Profile を事前検査し、ワークスペースとタブを作成します。工程は元タブとアセット ID で接続され、最初の「シンプル MV」はキービジュアル、BGM、画像ループ、メディア結合で構成されます。
- 依存パッケージのソース修正は `Patches/manifest.txt` に記述し、`scripts/apply-runtime-patches.command` が適用と検証を行います。ピンの不一致や修正失敗時は、未修正のソースで続行せずビルドを停止します。
- `GenImageMCP` は自身で `InferenceServices` を保持する独立 stdio server のままです。App の `LocalMCPServiceController` は localhost HTTP transport のライフサイクルだけを管理し、両者は同じ `MCPServer` ツールコアを直接使用するため、stdio→HTTP プロキシにはなりません。
- 本番用 ACE-Step の段階型は PoC 名称を使わなくなりました。診断専用の DiT probe は本番生成段階から分離し、実験コードとアプリの実行経路を明確にしています。

## プロファイル

`InferenceProfile` には次の情報が含まれます。

- 機能種別。
- モデル ID。
- モデル revision。
- 推論アーキテクチャ：MLX Swift、Core ML、ローカルサービス、外部 CLI。
- 機能の既定値。
- プロファイル revision。

ジョブ実行時、`WorkflowOperation.profileSnapshot` はプロファイル ID だけでなく完全な値を保存します。

組み込みプロファイルは直接変更しません。変更する場合は複製し、新しい revision として保存します。

## アセットとワークフロー

`MediaAsset.parentAssetID` はメディアの生成元を示します。parent を持たないアセットは独立ジョブのルートノードです。

- 単独のテキストから画像：生成画像に parent はありません。
- 単独の画像からテキスト：最初にルート画像を読み込み、説明結果を Recipe に書き込みます。
- 単独のアップスケール：最初にルート画像を読み込み、拡大結果は元画像を parent とします。
- 連携生成：生成結果は選択画像を parent とします。
- 単独のテキストから動画：MP4 アセットに parent はありません。
- 画像から動画：MP4 アセットは入力画像を parent とします。
- 単独のテキストから音楽：MP3、M4A、AAC、FLAC アセットに parent はなく、実際の長さ、サンプルレート、チャンネル数を記録します。
- 字幕生成：入力を `importedVideo` または `importedAudio` として読み込み、SRT／WebVTT を `generatedSubtitle` として保存し、入力メディアを parent とします。

`WorkflowGraph` が lineage と children の問い合わせを提供するため、UI がアセット関係を推測する必要はありません。

開いているワークスペースタブが生成プロジェクトのライフサイクル境界です。Swift は `Project`、`MediaAsset`、`WorkflowOperation`、選択状態を Application Support の JSON スナップショットへアトミック保存し、通常のアプリ終了では削除しません。Web UI はタブ情報を WebKit localStorage に保持します。タブを閉じると Bridge 経由でネイティブ層へ通知し、そのタブのアセットと lineage 索引だけを削除して出力済みメディアは残します。

名前付きワークスペースはタブの上位にあり、それぞれが独自のタブ集合を保持します。作成と削除は Bridge から `AppStore+Workspaces` に入り、削除前には確認が必要です。ワークスペース切り替えは対応するタブと選択状態だけを変更し、Runtime やメディアプレイヤーを再構築しません。

## 推論 Runtime

テキストから画像には `Z-Image.swift` commit `28bfcf3148c041a554629247170eb54d9ac46830` を使用します。

- macOS 14 以降、Swift 6。
- `ZImageGenerationRequest` はプロンプト、ネガティブプロンプト、サイズ、ステップ、シード、モデル、Runtime オプションをサポートします。
- `ZImageTextToImageService` は `ZImagePipeline.generate` をラップし、段階ごとの進捗を提供します。
- ノイズ除去ループは Swift Task のキャンセルを確認します。
- モデルと LoRA のアンロード、キャンセル、メモリキャッシュの解放をサポートします。

マルチモーダルの画像からテキスト／テキストからテキストには `mlx-swift-lm` revision `7da33441c7c08b010ff1aa8da9dc3d82277272f5` を使用します。

- `QwenVLImageDescriptionService` は `VLMModelFactory` を通じてローカル Qwen3-VL を読み込みます。
- `QwenTextGenerationService` は同じマルチモーダルコンテナの純テキスト入力を使用します。Qwen3-VL、Qwen3.5、Qwen3.8 の記述子は `.imageToText` と `.textToText` の両方を宣言し、能力ごとに独立 Profile を持ちます。
- Qwen3.5 純テキスト入力の上流互換修正は `Patches/MLX-Swift-LM-Qwen35-Text-Only.patch` で適用します。
- 管理対象ダウンロードは `processor_config.json`、画像／動画前処理設定、Tokenizer、Chat Template、重みとインデックスを取得・検証し、テキスト重みだけの状態をインストール済みとして扱いません。
- 同じプロファイルの再読み込みを避けるため、サービスのライフサイクル中はモデルコンテナをキャッシュします。
- 繁体字中国語、英語、日本語、韓国語の出力プロンプトをサポートします。

アップスケールは `CoreMLUpscaleService` が Real-ESRGAN の 512 タイルと 4 倍結合で実行します。

動画は `LTXVideoGenerationService` がアプリに同梱された `GenImageLTXVideoWorker` Swift サブプロセスを起動して生成します。

- テキストから動画と画像から動画は `VideoGenerating` と `VideoGenerationRequest` を共有します。
- Swift がプロファイル、モデルパス、サイズ、フレーム数、FPS、出力数を検証します。
- LTX-2.3 ではフレーム数が `8n+1` を満たす必要があります。
- JSON request と段階別 progress event は既存の `RuntimeProcess` を通し、Task のキャンセル、ログによるエラー報告、パーセント進捗の抽出をサポートします。
- LTX LoRA 制御動画は共通 FFmpeg レイヤーが VideoToolbox H.264 で生成し、GPL の `libx264` には依存しません。
- MP4 出力は `generatedVideo` アセットとしてワークスペースに追加され、Web UI のネイティブ `<video>` で再生されます。
- Worker は App Bundle の `Contents/Helpers` に配置されます。開発ビルドでは `GENIMAGE_LTX_WORKER` を指定できますが、本番フローは外部 Runtime に依存しません。

音楽は `MusicGenerationRouter` が `MusicRuntimeAdapter.supports` に従って振り分け、Router にモデル ID を集中してハードコードしません。

- テキストから音楽は `MusicGenerating`、`MusicGenerationRequest`、`MusicGenerationOptions` を使用します。
- `ACEStepMusicGenerationService` は `.mlxSwift` プロファイルを使用し、`ACEStepSwiftRuntime` を直接呼び出して Qwen3 Embedding、条件エンコード、Turbo DiT、Euler sampler、Oobleck VAE を実行します。外部サービスや Process は起動しません。
- ACE-Step は 10～300 秒、1～20 steps、任意の歌詞、インストゥルメンタル生成に対応します。latent 長は VAE のサンプルレートと hop length から算出し、長時間音声はオーバーラップ付き分割デコードと PCM ストリーム書き込みでピークメモリを抑えます。コードとモデルは MIT License です。
- 音楽プロファイルは `ProfileMusicConfiguration` を通じて時間の上下限と目標／最長の意味を公開するため、Web UI がモデル ID で分岐する必要はありません。
- `MiniMaxMusic3GenerationService` は `.externalCLI` プロファイルでアプリ同梱の `GenImageMiniMaxMusic3Worker` を起動します。アプリが JSON request を書き出し、Worker が固定 bfloat16 production 経路で Swift／MLX pipeline を実行します。5～300 秒は最長時間であり、音声終了トークンで早期終了できます。
- 8-bit と `mlx-community/MiniMax-Music3-4bit` checkpoint は同じ Worker を共有します。Worker は自動回帰を frame 単位、denoise を chunk×step 単位、vocoder を chunk 単位で進捗報告します。アプリは `RuntimeProcess` のキャンセル／強制終了を維持し、完成 WAV を内蔵 FFmpeg で変換します。MiniMax Music 3 は Python Runtime を必要としません。
- `Mothersuperior/minimax-music3-composer-5.7b-distilled` は音楽コンポーネントとしてモデルセンターで管理し、既定では `lr-6e-5` 重みだけをインストールします。独立した Profile ではなく、単独で音楽 Service に渡しません。適用には互換 Composer override Runtime が必要です。
- 両 Adapter は一時 WAV を取得し、`AudioOutputEncoder` が内蔵 FFmpeg を通じて MP3 320 kbps、M4A AAC 256 kbps、ADTS AAC 256 kbps、または可逆圧縮 FLAC に変換します。
- 成功、失敗、キャンセル時に一時ファイルを削除し、完成した圧縮音声だけを実際の長さ、サンプルレート、チャンネル数付き `generatedAudio` アセットとして保持します。
- ACE-Step の重みはモデルセンターが管理し、ネイティブ Runtime はアプリにコンパイルされるため、独立したインストール先やサービス用環境変数を使用しません。

メディアアセットは `fileURL` と任意の `playbackURL` を保持します。前者は字幕出力、系譜、明示的な削除に使うユーザー原本を常に指し、後者は WebKit プレビュー用として `Application Support/GenImage/MediaCache` 内の互換プロキシだけを指します。アセット削除またはプロジェクト終了時にプロキシを削除し、元メディアは書き換えません。

字幕生成はメディア読み込みとテキスト生成の間に位置します。

1. `MediaAudioPreparer` が内蔵 `ffprobe` で音声トラックを確認し、内蔵 `ffmpeg` で 16 kHz モノラル PCM WAV と一時パスへ統一します。
2. `SubtitleGenerationRouter` は `MediaTranscribing.supports(profile:)` を順番に評価し、最初に一致した Adapter を選択します。
3. Whisper Large v3 Turbo は多言語、Paraformer Large は中国語、Parakeet 0.6B は日本語を担当し、すべてローカル Core ML パスで動作します。
4. 任意の `QwenTextGenerationService` は Qwen3.5／Qwen3.8 MLX で字幕を分割翻訳し、開始・終了時刻は変更しません。
5. `SubtitleDocument` が SRT／WebVTT を生成し、`generatedSubtitle` アセットとして現在のワークスペースに保存します。

`GenImageASRPoC` は WhisperKit のメディアデコード、言語認識、時間軸出力だけを検証する独立実行ファイルです。App ワークスペースには書き込まず、正式フローの代替でもありません。

`scripts/build-ffmpeg-macos.sh` は Apple Silicon 向け、LGPL-only、動的リンク、差し替え可能な FFmpeg 配布物を生成します。`build.command` は GPL／nonfree Encoder を拒否し、`ffmpeg`、`ffprobe`、dylib、ライセンスをコピーして install name を `@rpath` へ変更し、dylib、ツール、App の順に署名して既存 DMG 公証フローへ渡します。Developer ID で同じ Team ID を使って署名した FFmpeg の dylib とツールは、追加の library-validation entitlement なしで動作することを確認済みです。本機 ad-hoc ビルドでは FFmpeg に hardened runtime を付けません。事前ビルド済みバイナリは Git で無視される `third_party/ffmpeg/` に置かれ、GitHub の Source archive には入りません。

MLX metallib は `RuntimeSupport/mlx.metallib` から `build.command` によって Release 実行ディレクトリへコピーされます。モデルライセンスの確認と 16／24／32 GB の負荷試験は引き続きリリース要件です。

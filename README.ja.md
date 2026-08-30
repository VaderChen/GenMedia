# GenMedia

[繁體中文](README.md) | [English](README.en.md) | 日本語 | [한국어](README.ko.md)

GenMedia は **Apple Silicon をネイティブサポート**するローカル AI メディア生成アプリです。本プロジェクトは、次の機能を備えたビルド可能なハイブリッドアプリケーションを提供します。

- Swift がモデル、プロファイル、ジョブキュー、ファイル、MLX／Core ML 推論を管理します。
- `WKWebView` に HTML、CSS、JavaScript UI を組み込み、ネットワーク接続や npm ランタイムを必要としません。
- テキストから画像、画像からテキスト、画像から画像、テキストから動画、画像から動画、テキストから音楽、字幕生成、アップスケールを個別に実行でき、アセットの系譜を通じて連携できます。
- 自動フローは個別設定を持つワークスペースタブを作成します。「シンプル MV」ではキービジュアル、BGM、画像ループ、映像・音声結合の工程を準備します。
- 各操作でプロファイルのスナップショットを保存し、モデルやアーキテクチャの更新後も当時のバージョンを追跡できます。
- 専用の設定画面で繁体字中国語、英語、日本語、韓国語、および永続化可能な 6 種類のカラーテーマを利用できます。
- 設定画面のスイッチで localhost 専用 MCP HTTP API を起動でき、App を起動していない場合も独立した JSON-RPC 2.0 stdio サーバーを利用できます。

## プレビュー

![GenMedia メディア生成インターフェース](images/cap001.jpg)

## 実行

要件：macOS 14 以降、Apple Silicon、Xcode 16 以降。

```bash
./build.command
./run.command
```

`build.command` は Release 実行ファイルと標準の `GenMedia.app` を `dist/` に生成します。App には WebUI リソース、MLX Metal ランタイム、MCP サーバー、モデル診断ツール、および統一メディア互換レイヤーとして LGPL 動的版 `ffmpeg`／`ffprobe` が含まれます。初回の App bundle ビルド時に `build.command` がソースを自動ダウンロードし、内蔵 FFmpeg を構築します。FFmpeg の手動準備や `pkg-config` のインストールは不要です。

```bash
# Release 実行ファイルと App をビルド
./build.command

# App bundle を作成せずに増分 Release ビルドのみ実行
./build.command --no-app

# バージョンと Bundle ID を指定
GENIMAGE_VERSION=1.1.0 GENIMAGE_BUNDLE_ID=com.example.genimage ./build.command
```

`run.command` は自動的に `--no-app` を使用するため、日常の起動で App bundle を繰り返し生成しません。配布用 DMG は、Developer ID Application 署名、Apple の公証、Staple、Gatekeeper 検証を行う別のローカルフローで処理します。

### FFmpeg ビルドのトラブルシューティング

- 初回の `./build.command` は FFmpeg と LAME のソース取得にネットワーク接続が必要です。以後はキャッシュしたソースと `third_party/ffmpeg` を再利用し、不足または不完全な場合だけ自動で再構築します。
- Homebrew の `pkg-config` は不要です。LAME の構成確認専用 fallback を空白のない一時パスから実行するため、プロジェクトパスに空白が含まれていてもビルドできます。
- ExFAT などの外部ファイルシステムが作成する `._*` AppleDouble sidecar は dylib 処理前に削除され、Mach-O と誤認されません。
- 中断または失敗時は、以前の使用可能な FFmpeg を復元します。ネットワークまたは Xcode の問題を解消して `./build.command` を再実行してください。出力先は `GENMEDIA_FFMPEG_ROOT` で変更できます。

### 動画 Runtime

動画生成はアプリに同梱された `GenImageLTXVideoWorker` Swift サブプロセスで実行します。Swift アプリがプロファイル、パラメータ検証、ジョブキュー、キャンセル、進捗、アセット、動画再生を管理し、追加の動画 Runtime は必要ありません。

モデルセンターの `dgrauet/ltx-2.3-mlx-q4` は、ネイティブ MLX INT4 Transformer、Video／Audio VAE、vocoder、空間アップスケーラーに加えて、`google/gemma-3-12b-it-qat-q4_0-unquantized` の Gemma 3 12B テキストエンコーダーもダウンロードします。全体で約 42 GiB の空き容量が必要で、48 GB 以上のメモリを推奨します。

開発ビルドでは `GENIMAGE_LTX_WORKER` で Worker を指定できます。リリース版は Bundle の `Contents/Helpers/GenImageLTXVideoWorker` を使用します。`GENIMAGE_LTX_GEMMA_MODEL` を設定すると Gemma ディレクトリを上書きでき、未設定時は LTX モデル内の `gemma-3-12b` を使用します。

LTX Worker は JSON request と段階別進捗イベントでアプリと通信し、モデルファイルはアプリに同梱しません。旧版の Runtime データは自動削除しないため、不要であることを確認した後に `~/Library/Application Support/GenImage/Runtime/ltx-2-mlx/` を手動で削除できます。

### 互換メディア読み込み

読み込んだ動画と音声は、内蔵 `ffprobe` がコンテナ、Codec、トラック、長さ、回転後の表示サイズを検査し、長時間の変換は進捗付きのキャンセル可能なジョブになります。WebKit で直接再生できる場合は原本を維持し、H.264／HEVC でコンテナまたは音声だけが非互換の場合は MP4 へ無劣化で再多重化します。それ以外の動画には VideoToolbox H.264／AAC、非互換音声には M4A AAC の再生用プロキシを作成します。再生プロキシは `AssetSchemeHandler` がバックグラウンドで HTTP Range の分割ストリームとして提供するため、プログレッシブ再生とシークに対応します。原本パスは字幕ソースとして保持されるため、字幕の出力先と名前は元ファイルに従います。起動時には対応するアセットがない MediaCache の孤立ファイルを削除し、プロジェクトを閉じるかアセットを削除した時も App 管理の互換キャッシュだけを削除します。

### 自動フローとメディア合成

自動フローは宣言的な工程から新しいワークスペースとタブを作成します。各タブは作業タイプ、Profile、Prompt、画像／動画／音楽／メディア処理設定を個別に保存するため、タブ切り替えや App の再起動で別の下書きを上書きしません。最初の「シンプル MV」テンプレートは、テキストから画像、テキストから音楽、画像ループ動画、映像・音声結合を連携します。

画像ループと映像・音声結合は AI モデルを必要とせず、`MediaCompositionService` が内蔵 FFmpeg で実行します。画像ループは複数画像、1 枚ごとの秒数、総時間、解像度、FPS、Cover／Contain に対応します。結合では元音声の置換／ミックス、音量、出力時間方針を選択できます。工程間はファイル名ではなくアセット ID で接続します。

### 字幕生成

字幕フローは動画または音声を読み込み、内蔵 `ffprobe` でトラックを検査し、内蔵 `ffmpeg` で選択した音声を 16 kHz モノラル PCM に統一してから `SubtitleGenerationRouter` がネイティブ Core ML ASR Adapter を選択します。区間タイムラインを保持したまま SRT または WebVTT を出力します。高速でメモリ使用量の少ない多言語 Whisper Small を既定の推奨とし、高い認識品質が必要な場合は Whisper Large v3 Turbo に切り替えられます。中国語 Paraformer Large と日本語 Parakeet 0.6B もサポートし、入力言語は自動検出または Profile で指定できます。

Whisper Small と Large v3 Turbo は Model Center からそれぞれ `argmaxinc/whisperkit-coreml@small` と `argmaxinc/whisperkit-coreml@large-v3-turbo` としてダウンロードできます。Small は約 216 MB、推奨メモリ 8 GB、Large は約 954 MB、推奨メモリ 16 GB です。どちらも同じネイティブ WhisperKit／Core ML Runtime を使用します。

認識後はローカル Qwen3.5／Qwen3.8 MLX テキストモデルを任意で使用し、タイムラインを変えずに繁体字中国語、簡体字中国語、英語、日本語、韓国語、スペイン語、フランス語、ドイツ語、イタリア語、ポルトガル語、ロシア語、アラビア語、ヒンディー語、ベンガル語、インドネシア語、ベトナム語、タイ語、トルコ語、ポーランド語、オランダ語、スウェーデン語、チェコ語、ウクライナ語、マレー語、フィリピン語の 25 言語へ翻訳できます。結果は入力動画または音声を parent とする `generatedSubtitle` アセットとして現在のワークスペースに保存されます。

`GenImageASRPoC` はメイン App のワークスペースを変更せず、メディアデコード、言語認識、タイムコードを確認する独立 WhisperKit 検証ツールです。正式な App フローも同じ Swift／Core ML 境界を使用します。詳細は [ASR 字幕 PoC](docs/ASR_POC.ja.md) を参照してください。

Qwen3-VL、Qwen3.5、Qwen3.8 はマルチモーダルモデルのため、モデルセンターでは「画像からテキスト」と「テキストからテキスト」の両方に分類され、それぞれの Profile が提供されます。管理対象ダウンロードには `processor_config.json`、画像／動画前処理設定、Tokenizer、Chat Template、完全な重みインデックスが含まれ、必須ファイルの検証後にのみインストール済みになります。

### Civitai LoRA

モデルセンターには `ZImageTurbo` ベースの Civitai LoRA（Asian Beauties、Turbo Lightning、Flat AnimeStyle、Diorama）も含まれます。設定の「Civitai LoRA」カードに API Token を貼り付けて保存してください。Token は macOS Keychain にのみ保存し、モデル manifest、ワークスペース、ログには書き込みません。以後、モデルセンターのダウンロードは Civitai サイトを開いて手動保存するのではなく、HTTPS の `Authorization: Bearer` ヘッダーでファイルを直接取得します。

旧来の UI なし起動スクリプト向けには、互換性 fallback として `CIVITAI_TOKEN` も引き続き使用できます。

### プロファイル、ジョブ、メモリ

- プロファイルは「使用中、利用可能、ダウンロード中、利用不可」の順に並びます。モデルと LoRA の依存関係が揃ったプロファイルは淡い緑色の枠で表示され、ダウンロード完了時に直ちに再整列されます。
- キャンセル時はまず `cancelling` になり、Runtime Task の終了後に `cancelled` へ移行して生成・メモリ関連の操作を再び有効にします。ETA は進捗 35% かつ開始 15 秒後から数値表示し、安定したサンプルが不足する場合は全体経過時間を使用します。
- Z-Image MLX 互換レイヤーは `quantize_config.json`、affine／mxfp4、packed pad token、FP16 から BF16 への読み込みを処理します。andrevp Z-Image Turbo MLX 4-bit は実際の画像生成で検証済みです。
- 依存パッケージへのソース修正は `Patches/manifest.txt` に列挙し、Swift Package 解決後に `scripts/apply-runtime-patches.command` が適用します（`build.command` から自動実行）。依存パッケージのバージョンが manifest と異なる、修正ファイルが無い、適用に失敗した、適用後に想定した目印が見つからない場合はビルドを中止し、未修正のソースのまま進むことはありません。`scripts/apply-runtime-patches.command --verify` で確認のみ実行できます。
- テキストから画像の完了後もモデル重みと暖機 buffer を保持します。5 分間アイドルになると再利用可能な MLX 一時 buffer だけを整理し、モデルはアンロードしません。側面のメモリ解放、モデル切り替え、またはプロファイル切り替え時の RAM 90% 超過保護でのみ不要な Runtime を解放します。
- ダウンロードでは配布元のファイル名を保持し、生成出力は `Image-YYYYMMDD-HHmm`、`Video-YYYYMMDD-HHmm`、`Music-YYYYMMDD-HHmm` を使用します。同じ分に重複する場合は連番を追加します。出力ディレクトリは設定画面で変更できます。
- 開いている各ワークスペースタブを生成プロジェクトとして扱います。アセットと lineage は Application Support にアトミック保存され、アプリ再起動後も復元されます。タブを明示的に閉じた場合のみプロジェクトのワークスペース索引を削除し、出力済みメディアはディスクに保持します。
- アプリのデータはすべて `~/Library/Application Support/GenImage/`（`Models`、`Runtime`、`Workspace`、`Pasted`、`Generated`）に置き、`GenImageCore/ApplicationSupport.swift` が唯一の定義元です。ワークスペース索引は以前 `GenMedia/` に書かれており、起動時に現在のルートへ引き取ります。同名の項目は既存のものを残し、上書きも統合もしません。既存モデルと旧版 Runtime データとの互換性を保つため、アプリ名に合わせた `GenMedia` へは改名せず `GenImage` のままにしています。
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

- **MiniMax Music 3 MLX 8-bit**：アプリに同梱された独立 Swift Worker で動作します。5～300 秒の設定値は最長時間であり、曲構成に応じてモデルが早めに自然終了する場合があります。完了後、アプリに実際の出力時間を表示します。モデルには Community License が適用されます。
- **MiniMax Music 3 MLX 4-bit**：アプリに同梱された独立 Swift Worker で動作し、完全な affine 4-bit MLX checkpoint を使用します。5～300 秒の設定値は最長時間であり、曲構成に応じてモデルが早めに自然終了する場合があります。モデルには MiniMax Music 3 Community License が適用されます。
- **MiniMax Music 3 Composer 5.7B Distilled**：モデルセンターで任意の Composer 高速化コンポーネントとして提供します。インストーラーは `lr-6e-5` 重みを選択します。完全な Music 3 checkpoint と互換 Runtime が必要で、単独で音楽を生成することはできません。

ACE-Step ネイティブ Runtime、MiniMax Music 3 Swift Worker、LGPL FFmpeg 互換レイヤーはアプリに同梱されます。MiniMax Music 3 モデルは任意のコンポーネントです。

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
MCP_BIN_DIR="$(swift build -c release --show-bin-path)"
"$MCP_BIN_DIR/GenImageMCP"
```

MCP は `initialize`、`ping`、`tools/list`、`tools/call` をサポートします。ツールにはローカルモデル、プロファイル、ネイティブ Z-Image のテキストから画像生成、Qwen の画像編集／説明、Core ML アップスケール、独立字幕生成が含まれます。

App から使用する場合は「設定 → MCP 連携」のスイッチを有効にすると、localhost 専用 HTTP POST JSON-RPC エンドポイント `http://127.0.0.1:12181/mcp` が表示されます。HTTP と stdio は同じ MCP ツールコアを共有しますが、stdio 実行ファイルは GenMedia.app を閉じた状態でも独立して動作します。

MCP のエンドツーエンド検証は完了しています。`genimage_generate_image` はローカル Z-Image Turbo Q4 で PNG を出力し、`genimage_describe_image` は Qwen3-VL で繁体字中国語の説明を生成し、`genimage_upscale_image` はローカル Real-ESRGAN Core ML モデルで 4 倍のアップスケールを実行します。

## プロジェクト構成

今回の内部整理はコードの分割と責務だけを変更するもので、既存の生成機能、利用手順、Web Bridge プロトコルは変更しません。

```text
Sources/
├── GenImageCore/
│   ├── ApplicationSupport.swift  # アプリデータ配置の唯一の定義
│   ├── CivitaiTokenStore.swift   # Civitai Token の macOS Keychain 保存
│   ├── DomainModels.swift        # アセット、レシピ、ジョブ、モデル、プロファイル
│   ├── InferenceServices.swift   # 画像、テキスト、動画、音楽、字幕の推論インターフェース
│   ├── ModelCatalog.swift        # 組み込みモデルとプロファイル
│   ├── OutputFileNaming.swift    # 画像・動画・音楽・字幕の出力名
│   ├── OutputGeometry.swift      # 出力サイズ演算の唯一の定義（js/geometry.js が鏡像）
│   ├── ProjectWorkspacePersistence.swift # 開いている生成プロジェクトの永続化
│   ├── SubtitleDocument.swift    # SRT／WebVTT の出力
│   └── WorkflowGraph.swift       # アセットの系譜と分岐関係
├── GenImageRuntime/
│   ├── ZImageTextToImageService.swift
│   ├── QwenVLImageDescriptionService.swift
│   ├── Qwen2511ImageToImageService.swift
│   ├── LTXVideoGenerationService.swift
│   ├── MusicGenerationRouter.swift
│   ├── ACEStepMusicGenerationService.swift
│   ├── MiniMaxMusic3GenerationService.swift
│   ├── MediaCompatibilityService.swift # 内蔵 FFmpeg／ffprobe の検索・解析・変換
│   ├── MediaSourceCompatibilityService.swift # 入力メディアの直接再生・再多重化・再生プロキシ
│   ├── MediaCompositionService.swift # 画像ループ動画と映像・音声結合
│   ├── MediaAudioPreparer.swift
│   ├── WhisperSubtitleTranscriber.swift
│   ├── ParaformerChineseSubtitleTranscriber.swift
│   ├── ParakeetJapaneseSubtitleTranscriber.swift
│   ├── SubtitleGenerationRouter.swift
│   ├── QwenTextGenerationService.swift
│   ├── SubprocessRuntime.swift   # 外部 Runtime 子プロセスの共通処理
│   ├── AudioOutputEncoder.swift
│   └── CoreMLUpscaleService.swift
├── GenImageApp/
    ├── AppStore.swift            # 型宣言、格納プロパティ、init
    ├── AppStore+Credentials.swift # 設定画面の資格情報操作
    ├── AppStore+SubtitleGeneration.swift
    ├── AppStore+MediaImport.swift
    ├── AppStore+MediaComposition.swift
    ├── AppStore+Workspaces.swift
    ├── AppStore+*.swift          # その他の責務別 AppStore 実装
    ├── HybridBridgeController.swift
    ├── LocalMCPServiceController.swift # App 内 MCP HTTP スイッチと状態
    ├── HybridWebView.swift
    ├── AssetSchemeHandler.swift  # ローカル画像、動画、音声を安全に WebUI へ提供
    └── Resources/WebUI/
        ├── js/automatic-flow.js  # テンプレート、Profile 事前検査、ワークスペース計画
        └── …                     # その他の HTML/CSS/JavaScript フロントエンド
├── GenImageASRPoC/
    └── main.swift                # 独立 ASR 検証ツール
└── GenImageMCPServer/
    ├── MCPServer.swift           # 独立 stdio JSON-RPC server コア
    └── MCPHTTPServer.swift       # App スイッチ用 localhost HTTP transport
Patches/
├── MLX-Swift-LM-Qwen35-Text-Only.patch # Qwen3.5 純テキスト入力の互換修正
└── manifest.txt                  # ビルド時に適用する依存パッチ一覧
scripts/
├── apply-runtime-patches.command  # manifest に従って依存パッケージ修正を適用・検証
└── build-ffmpeg-macos.sh          # バンドル可能で再リンク可能な LGPL FFmpeg を構築
```

## 現在の状態

アプリは Z-Image Turbo のテキストから画像、Qwen3-VL／Qwen3.5／Qwen3.8 マルチモーダルの画像からテキスト／テキストからテキスト、Qwen 2511 の画像から画像、LTX-2.3 MLX のテキストから動画／画像から動画、ACE-Step 1.5 Turbo MLX と MiniMax Music 3 MLX 8-bit／4-bit のテキストから音楽、Whisper／Paraformer／Parakeet の字幕生成、Core ML Real-ESRGAN のアップスケールによるローカル推論に接続済みです。動画は MP4、音楽は MP3／M4A／AAC／FLAC、字幕は SRT／WebVTT として追加され、言語、時間軸、プロファイルスナップショット、系譜を保持します。

詳細情報：

- [更新履歴](UpdateNote.md)
- [アーキテクチャ](docs/ARCHITECTURE.ja.md)
- [Web Bridge](docs/WEB_BRIDGE.ja.md)
- [ロードマップ](docs/ROADMAP.ja.md)
- [MCP インターフェース](docs/MCP.ja.md)
- [ASR 字幕 PoC](docs/ASR_POC.ja.md)
- [ローカルモデルテスト報告](docs/MODEL_TEST_REPORT.ja.md)

## ライセンス

本プロジェクトは GPLv3 と商用ライセンスのデュアルライセンス方式を採用しています。

- オープンソースでの利用は [GNU General Public License v3.0](LICENSE) に基づきます。
- クローズドソースへの統合、プロプライエタリ製品の配布、個別の商用条件など、GPLv3 に準拠できない、または準拠を希望しない場合は、著作権者に連絡して別途商用ライセンスを取得してください。
- 内蔵 FFmpeg と LAME はそれぞれの LGPL 条項に従います。ライセンス本文、正確なソース版、ビルド情報は App の `Contents/Resources/Licenses/` に収録されます。

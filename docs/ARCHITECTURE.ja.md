# GenImage アーキテクチャ

[繁體中文](ARCHITECTURE.md) | [English](ARCHITECTURE.en.md) | 日本語 | [한국어](ARCHITECTURE.ko.md)

## 設計目標

1. テキストから画像、画像からテキスト、画像から画像、動画生成、音楽生成、アップスケールは相互に依存しない独立した機能です。
2. 画像、動画、音声の出力からワークフローや分岐を構成できます。
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
        └── ImageUpscaling
                    │
                    ▼
       MLX Swift / Core ML / External CLI
```

Web UI は Bridge を通じてのみローカル機能を利用できます。任意のファイル、モデルディレクトリ、システム API を直接読み取ることはできません。

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

`ImageAsset.parentAssetID` はアセットの生成元を示します。parent を持たないアセットは独立ジョブのルートノードです。

- 単独のテキストから画像：生成画像に parent はありません。
- 単独の画像からテキスト：最初にルート画像を読み込み、説明結果を Recipe に書き込みます。
- 単独のアップスケール：最初にルート画像を読み込み、拡大結果は元画像を parent とします。
- 連携生成：生成結果は選択画像を parent とします。
- 単独のテキストから動画：MP4 アセットに parent はありません。
- 画像から動画：MP4 アセットは入力画像を parent とします。
- 単独のテキストから音楽：MP3、M4A、AAC、FLAC アセットに parent はなく、実際の長さ、サンプルレート、チャンネル数を記録します。

`WorkflowGraph` が lineage と children の問い合わせを提供するため、UI がアセット関係を推測する必要はありません。

## 推論 Runtime

テキストから画像には `Z-Image.swift` commit `28bfcf3148c041a554629247170eb54d9ac46830` を使用します。

- macOS 14 以降、Swift 6。
- `ZImageGenerationRequest` はプロンプト、ネガティブプロンプト、サイズ、ステップ、シード、モデル、Runtime オプションをサポートします。
- `ZImageTextToImageService` は `ZImagePipeline.generate` をラップし、段階ごとの進捗を提供します。
- ノイズ除去ループは Swift Task のキャンセルを確認します。
- モデルと LoRA のアンロード、キャンセル、メモリキャッシュの解放をサポートします。

画像からテキストには `mlx-swift-lm 2.30.6` を使用します。

- `QwenVLImageDescriptionService` は `VLMModelFactory` を通じてローカル Qwen3-VL を読み込みます。
- 同じプロファイルの再読み込みを避けるため、サービスのライフサイクル中はモデルコンテナをキャッシュします。
- 繁体字中国語、英語、日本語、韓国語の出力プロンプトをサポートします。

アップスケールは `CoreMLUpscaleService` が Real-ESRGAN の 512 タイルと 4 倍結合で実行します。

動画は `LTXVideoGenerationService` が `ltx-2-mlx generate` を呼び出して生成します。

- テキストから動画と画像から動画は `VideoGenerating` と `VideoGenerationRequest` を共有します。
- Swift がプロファイル、モデルパス、サイズ、フレーム数、FPS、出力数を検証します。
- LTX-2.3 ではフレーム数が `8n+1` を満たす必要があります。
- 外部 Process は Task のキャンセル、ログによるエラー報告、パーセント進捗の抽出をサポートします。
- MP4 出力は `generatedVideo` アセットとしてワークスペースに追加され、Web UI のネイティブ `<video>` で再生されます。
- Python 実装を UI や `AppStore` に結合せず、`GENIMAGE_LTX_RUNTIME` または標準インストール場所によって Runtime を置き換えられます。

音楽は `MiniMaxMusic3GenerationService` が `mlx-minimax-music3 generate` を呼び出して生成します。

- テキストから音楽は `MusicGenerating`、`MusicGenerationRequest`、`MusicGenerationOptions` を使用します。
- Swift がプロファイル、モデルの完全性、スタイルプロンプト、5～300 秒の長さ（最長 5 分）、ステップ、出力形式を検証します。歌詞は任意で、空欄の場合はインストゥルメンタル用のマーカーとボーカルなしのプロンプトへ変換します。
- Runtime は一時 WAV を生成し、FFmpeg が MP3 320 kbps、M4A AAC 256 kbps、ADTS AAC 256 kbps、または可逆圧縮 FLAC に変換します。
- 成功、失敗、キャンセルのいずれでも WAV、プロンプト、歌詞、ログを削除し、完成した圧縮音声だけを `generatedAudio` アセットとして保持します。
- Web UI はネイティブ `<audio controls>` で再生し、Inspector に実際の長さ、形式、44.1 kHz サンプルレート、チャンネル数を表示します。
- Runtime は `GENIMAGE_MINIMAX_MUSIC3_RUNTIME` または標準の Application Support 配置で交換でき、モデルは固定 revision から導入して独立した Community License を維持します。

MLX metallib は `RuntimeSupport/mlx.metallib` から `build.command` によって Release 実行ディレクトリへコピーされます。配布前にはモデルライセンスの確認、16／24／32 GB の負荷試験、App bundle、署名、公証を完了する必要があります。

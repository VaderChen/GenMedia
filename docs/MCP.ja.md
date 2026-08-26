# GenImage MCP Server

[繁體中文](MCP.md) | [English](MCP.en.md) | 日本語 | [한국어](MCP.ko.md)

GenImage は同じ MCP ツールコアを、プロトコル版 `2025-06-18` の 2 種類の JSON-RPC 2.0 transport で提供します。独立 stdio server と、GenMedia.app のスイッチで制御する localhost HTTP API です。

## 独立 stdio

```bash
./build.command
MCP_BIN_DIR="$(swift build -c release --show-bin-path)"
"$MCP_BIN_DIR/GenImageMCP"
```

正式な連携では `build.command` が出力した絶対パスを使用してください。`swift build` だけを実行しても metallib はコピーされません。

`swift build -c release --show-bin-path` を実行して、現在のツールチェーンが使用する Release ディレクトリを取得してください。旧 SwiftPM の `.build/arm64-apple-macosx/release` パスを固定しないでください。

stdio server は自身で `InferenceServices` を保持し、`GenImageMCP` プロセス内で推論を完了します。GenMedia.app の起動は不要です。

## App HTTP API

GenMedia.app の「設定 → MCP 連携」を有効にすると、次の URL が表示されます。

```text
http://127.0.0.1:12181/mcp
```

このエンドポイントは localhost のみにバインドし、HTTP `POST` の JSON-RPC を受け付けます。スイッチを切ると直ちに待受を停止します。HTTP transport は stdio と同じ `MCPServer` ツールコアを直接呼び出し、stdio 子プロセスを起動せず、別サービスへのプロキシでもありません。App 内蔵モードのため利用中は GenMedia.app を起動したままにし、headless 連携では独立 stdio 実行ファイルを使用してください。

## MCP メソッド

- `initialize`
- `notifications/initialized`
- `ping`
- `tools/list`
- `tools/call`

stdio メッセージは 1 行につき 1 つの UTF-8 JSON-RPC オブジェクトです。stdout にはプロトコルメッセージだけを出力し、診断情報は stderr に書き込みます。

## ツール

- `genimage_models_list`
- `genimage_profiles_list`
- `genimage_upscale_image`
- `genimage_generate_image`
- `genimage_edit_image`
- `genimage_describe_image`
- `genimage_generate_subtitle`

テキストから画像、画像編集、画像説明、任意の字幕翻訳はネイティブ MLX Swift Runtime を使用し、字幕認識はネイティブ Core ML Runtime を使用します。Release 実行ファイルの隣に `mlx.metallib` が必要で、`build.command` が自動的に配置します。

モデルのルートディレクトリは `model_root` または `GENIMAGE_MODEL_ROOT` で指定できます。

### `genimage_generate_subtitle`

必須引数は入力の絶対パス `input_path`、ASR `model_path`、`model_id` です。対応する ASR モデル ID：

- `argmaxinc/whisperkit-coreml@large-v3-turbo`：多言語 Whisper
- `FluidInference/paraformer-large-zh-coreml`：中国語 Paraformer
- `FluidInference/parakeet-0.6b-ja-coreml`：日本語 Parakeet

`format` は `srt` または `vtt` で、既定値は `srt` です。`language_code` は入力言語または `auto` を指定します。`output_path` には、まだ存在せず拡張子が形式と一致する絶対パスを指定できます。`target_language_code` を使う場合は `translation_model_path` と `translation_model_id` も必須で、ローカル Qwen テキストモデルが時間軸を保持したまま `zh-Hant`、`zh-Hans`、`en`、`ja`、`ko` へ翻訳します。

このツールは `GenImageMCP` プロセス内で `SubtitleGenerationRouter` と推論サービスを直接生成します。GenMedia.app へ接続せず、App の起動も必要ありません。

## 設定例

コンパイル済み実行ファイルを使用すると、MCP Client ごとの作業ディレクトリの違いを回避できます。

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

MCP Client は `build.command` が生成した Release 実行ファイルを使用し、MLX Runtime と実行ファイルを同じディレクトリに配置してください。

## 検証済みの動作

- `initialize` と 7 ツールを返す `tools/list` のスモークテストに合格しています。
- App HTTP transport は localhost 上で `tools/list` に応答し、stdio と同じ 7 ツールを返します。
- 未知のメソッドは JSON-RPC `-32601` を返します。
- `genimage_generate_image` はローカル Z-Image Turbo Q4 を使用した 256×256、1-step のエンドツーエンド生成を完了しています。
- `genimage_describe_image` はローカル Qwen3-VL 4-bit を使用した繁体字中国語の画像説明を完了しています。
- `genimage_upscale_image` はローカル Real-ESRGAN Core ML モデルを使用した 4 倍拡大を完了しています。
- モデル内部の Logger は stdout に書き込まず、stdio は純粋な JSON-RPC を維持します。

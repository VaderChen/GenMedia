# ASR 字幕 PoC

[繁體中文](ASR_POC.md) | [English](ASR_POC.en.md) | 日本語 | [한국어](ASR_POC.ko.md)

`GenImageASRPoC` は独立したローカル音声認識検証実行ファイルです。GenMedia の UI、ワークスペース、アセットデータを変更しません。ネイティブ Swift／Core ML の WhisperKit を使い、動画または音声を時間軸付き原文字幕へ変換します。

メイン App には中国語 Paraformer Large、日本語 Parakeet 0.6B、SRT／WebVTT アセット、任意の Qwen MLX 字幕翻訳を含む正式な字幕フローが統合済みです。PoC は WhisperKit のメディアデコード、言語認識、時間コードを独立して確認するために残しています。

## 対応入力

PoC は macOS `AVFoundation` でメディアを読み込み音声トラックを抽出するため、次を利用できます。

- 音声トラックを持つ MP4、MOV、M4V などの動画
- WAV、M4A、MP3、AAC、AIFF、および macOS がデコードできる音声

音声トラックのない動画はエラーになります。Python、FFmpeg、外部 ASR CLI は使用しません。

## 実行

初回実行では WhisperKit Core ML モデルを次へダウンロードします。

```text
~/Library/Application Support/GenImage/Models/WhisperKit/
```

既定の出力先：

```text
~/Library/Application Support/GenImage/Generated/ASR/
```

```bash
swift run GenImageASRPoC --input "/path/to/video.mp4"
swift run GenImageASRPoC --input "/path/to/video.mp4" --input "/path/to/audio.m4a"
swift run GenImageASRPoC "/path/to/audio.wav" --language ja
```

`--language auto` で自動検出し、`zh`、`ja`、`ko`、`en` などのコードも指定できます。中国語は `--chinese-script traditional` で繁体字へ変換できます。単語単位の時刻には `--word-timestamps` を追加します。

既存のローカルモデルを使い、ダウンロードを禁止する場合：

```bash
swift run GenImageASRPoC \
  --input "/path/to/video.mp4" \
  --model-folder "/path/to/whisperkit-model" \
  --no-download
```

## 出力

入力ごとに次を生成します。

- `<名前>-asr.json`：本文、認識言語、モデル、区間時間、信頼度、任意の単語時間
- `<名前>-asr.srt`：SubRip 字幕
- `<名前>-asr.vtt`：WebVTT 字幕

PoC は原文 ASR と時間軸だけを検証します。メイン App の翻訳はローカル Qwen テキストモデルを使用し、各区間の開始・終了時刻を保持します。

## PoC の境界

- WhisperKit のみ使用し、Paraformer、Parakeet、Qwen 翻訳はメイン App Runtime が提供
- 独立ファイルだけを書き込み、`MediaAsset`、workspace、lineage は作成しない
- 話者識別は行わない
- 字幕の動画への焼き込み／多重化は行わない

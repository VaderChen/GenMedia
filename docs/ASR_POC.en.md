# ASR Subtitle PoC

[繁體中文](ASR_POC.md) | English | [日本語](ASR_POC.ja.md) | [한국어](ASR_POC.ko.md)

`GenImageASRPoC` is a standalone local speech-recognition validation executable. It does not modify the GenMedia UI, workspaces, or asset data. It uses native Swift/Core ML WhisperKit to convert video or audio files into source-language subtitles with timestamps.

The main app now contains the complete subtitle workflow, including Chinese Paraformer Large, Japanese Parakeet 0.6B, SRT/WebVTT assets, and optional Qwen MLX subtitle translation. The PoC remains an isolated entry point for validating WhisperKit media decoding, language recognition, and timestamps.

## Supported Input

The PoC reads media and extracts its audio track with macOS `AVFoundation`, so it accepts:

- Video with audio tracks, such as MP4, MOV, and M4V
- Audio such as WAV, M4A, MP3, AAC, AIFF, and other formats decodable by macOS

A video without an audio track reports an error. The PoC does not use Python, FFmpeg, or an external ASR CLI.

## Run

The first run downloads the WhisperKit Core ML model to:

```text
~/Library/Application Support/GenImage/Models/WhisperKit/
```

Output defaults to:

```text
~/Library/Application Support/GenImage/Generated/ASR/
```

```bash
swift run GenImageASRPoC --input "/path/to/video.mp4"
swift run GenImageASRPoC --input "/path/to/video.mp4" --input "/path/to/audio.m4a"
swift run GenImageASRPoC "/path/to/audio.wav" --language ja
```

Use `--language auto` for detection, or specify a language code such as `zh`, `ja`, `ko`, or `en`. For Chinese, `--chinese-script traditional` converts the recognized text to Traditional Chinese. Add `--word-timestamps` for word-level timing.

To use an existing local model without downloading:

```bash
swift run GenImageASRPoC \
  --input "/path/to/video.mp4" \
  --model-folder "/path/to/whisperkit-model" \
  --no-download
```

## Output

Each input produces:

- `<name>-asr.json`: text, detected language, model, segment timing, confidence, and optional word timing
- `<name>-asr.srt`: SubRip subtitles
- `<name>-asr.vtt`: WebVTT subtitles

The PoC validates source-language ASR and timing only. Main-app translation uses a local Qwen text model and preserves every segment's start and end time.

## PoC Boundary

- Uses WhisperKit only; Paraformer, Parakeet, and Qwen translation are provided by the main app runtime
- Writes standalone files and creates no `MediaAsset`, workspace, or lineage records
- Does not perform speaker diarization
- Does not burn or mux subtitles into video

# ASR 자막 PoC

[繁體中文](ASR_POC.md) | [English](ASR_POC.en.md) | [日本語](ASR_POC.ja.md) | 한국어

`GenImageASRPoC`는 독립 로컬 음성 인식 검증 실행 파일입니다. GenMedia UI, 작업 공간, 에셋 데이터를 변경하지 않습니다. 네이티브 Swift/Core ML WhisperKit으로 비디오 또는 오디오를 타임라인이 있는 원문 자막으로 변환합니다.

메인 앱에는 중국어 Paraformer Large, 일본어 Parakeet 0.6B, SRT/WebVTT 에셋, 선택적 Qwen MLX 자막 번역을 포함한 정식 자막 흐름이 통합되어 있습니다. PoC는 WhisperKit 미디어 디코딩, 언어 인식, 타임코드를 독립적으로 확인하는 용도로 유지합니다.

## 지원 입력

PoC는 macOS `AVFoundation`으로 미디어를 읽고 오디오 트랙을 추출하므로 다음을 사용할 수 있습니다.

- 오디오 트랙이 있는 MP4, MOV, M4V 등의 비디오
- WAV, M4A, MP3, AAC, AIFF 및 macOS가 디코딩할 수 있는 오디오

오디오 트랙이 없는 비디오는 오류를 반환합니다. Python, FFmpeg, 외부 ASR CLI는 사용하지 않습니다.

## 실행

첫 실행에서 WhisperKit Core ML 모델을 다음 위치에 다운로드합니다.

```text
~/Library/Application Support/GenImage/Models/WhisperKit/
```

기본 출력 위치:

```text
~/Library/Application Support/GenImage/Generated/ASR/
```

```bash
swift run GenImageASRPoC --input "/path/to/video.mp4"
swift run GenImageASRPoC --input "/path/to/video.mp4" --input "/path/to/audio.m4a"
swift run GenImageASRPoC "/path/to/audio.wav" --language ko
```

`--language auto`로 자동 감지하거나 `zh`, `ja`, `ko`, `en` 등의 코드를 지정할 수 있습니다. 중국어는 `--chinese-script traditional`로 번체 변환이 가능하며 단어별 시간은 `--word-timestamps`를 추가합니다.

기존 로컬 모델을 사용하고 다운로드를 막으려면:

```bash
swift run GenImageASRPoC \
  --input "/path/to/video.mp4" \
  --model-folder "/path/to/whisperkit-model" \
  --no-download
```

## 출력

입력마다 다음 파일을 생성합니다.

- `<이름>-asr.json`: 본문, 감지 언어, 모델, 구간 시간, 신뢰도, 선택적 단어 시간
- `<이름>-asr.srt`: SubRip 자막
- `<이름>-asr.vtt`: WebVTT 자막

PoC는 원문 ASR과 타임라인만 검증합니다. 메인 앱 번역은 로컬 Qwen 텍스트 모델을 사용하며 각 구간의 시작과 종료 시간을 유지합니다.

## PoC 경계

- WhisperKit만 사용하며 Paraformer, Parakeet, Qwen 번역은 메인 앱 Runtime에서 제공
- 독립 파일만 기록하고 `MediaAsset`, workspace, lineage를 만들지 않음
- 화자 분리 없음
- 자막을 비디오에 굽거나 mux하지 않음

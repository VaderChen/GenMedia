# GenImage 아키텍처

[繁體中文](ARCHITECTURE.md) | [English](ARCHITECTURE.en.md) | [日本語](ARCHITECTURE.ja.md) | 한국어

## 설계 목표

1. 텍스트→이미지, 이미지→텍스트, 이미지→이미지, 비디오 생성, 음악 생성, 업스케일은 서로 의존하지 않는 독립 기능입니다.
2. 이미지, 비디오, 오디오 출력으로 작업 흐름과 분기를 구성할 수 있습니다.
3. UI는 MLX, Core ML 또는 특정 모델에 직접 의존하지 않습니다.
4. 모델 업데이트 시 프로필을 통해 모델 버전과 추론 아키텍처를 전환합니다.
5. 기존 작업은 프로필 스냅샷을 보존하므로 이후 프로필 변경의 영향을 받지 않습니다.

## 계층

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

Web UI는 Bridge를 통해서만 로컬 기능을 사용할 수 있으며 임의의 파일, 모델 디렉터리 또는 시스템 API를 직접 읽을 수 없습니다.

## 프로필

`InferenceProfile`에는 다음 정보가 포함됩니다.

- 기능 유형.
- 모델 ID.
- 모델 revision.
- 추론 아키텍처: MLX Swift, Core ML, 로컬 서비스 또는 외부 CLI.
- 기능 기본값.
- 프로필 revision.

작업 실행 시 `WorkflowOperation.profileSnapshot`은 프로필 ID만이 아니라 전체 값을 저장합니다.

기본 제공 프로필은 직접 수정하지 않습니다. 변경하려면 먼저 복제한 뒤 새 revision으로 저장합니다.

## 에셋과 작업 흐름

`ImageAsset.parentAssetID`는 에셋의 원본을 나타냅니다. parent가 없는 에셋은 독립 작업의 루트 노드입니다.

- 독립 텍스트→이미지: 생성된 이미지에 parent가 없습니다.
- 독립 이미지→텍스트: 먼저 루트 이미지를 가져오고 설명 결과를 Recipe에 기록합니다.
- 독립 업스케일: 먼저 루트 이미지를 가져오고 확대 결과는 원본 이미지를 parent로 사용합니다.
- 연결 생성: 생성 결과는 선택한 이미지를 parent로 사용합니다.
- 독립 텍스트→비디오: MP4 에셋에 parent가 없습니다.
- 이미지→비디오: MP4 에셋은 원본 이미지를 parent로 사용합니다.
- 독립 텍스트→음악: MP3, M4A, AAC 또는 FLAC 에셋에 parent가 없으며 실제 길이, 샘플링 레이트, 채널 수를 기록합니다.

`WorkflowGraph`가 lineage와 children 조회를 제공하므로 UI에서 에셋 관계를 추측할 필요가 없습니다.

## 추론 Runtime

텍스트→이미지는 `Z-Image.swift` commit `28bfcf3148c041a554629247170eb54d9ac46830`을 사용합니다.

- macOS 14 이상, Swift 6.
- `ZImageGenerationRequest`는 프롬프트, 네거티브 프롬프트, 크기, 스텝, 시드, 모델, Runtime 옵션을 지원합니다.
- `ZImageTextToImageService`는 `ZImagePipeline.generate`를 감싸고 단계별 진행률을 제공합니다.
- 노이즈 제거 루프는 Swift Task 취소를 확인합니다.
- 모델 언로드, LoRA 언로드, 취소, 메모리 캐시 정리를 지원합니다.

이미지→텍스트는 `mlx-swift-lm 2.30.6`을 사용합니다.

- `QwenVLImageDescriptionService`는 `VLMModelFactory`를 통해 로컬 Qwen3-VL을 불러옵니다.
- 같은 프로필의 반복 로드를 피하기 위해 서비스 수명 동안 모델 컨테이너를 캐시합니다.
- 번체 중국어, 영어, 일본어, 한국어 출력 프롬프트를 지원합니다.

업스케일은 `CoreMLUpscaleService`가 Real-ESRGAN 512 타일과 4배 결합으로 처리합니다.

비디오는 `LTXVideoGenerationService`가 `ltx-2-mlx generate`를 호출하여 생성합니다.

- 텍스트→비디오와 이미지→비디오는 `VideoGenerating` 및 `VideoGenerationRequest`를 공유합니다.
- Swift가 프로필, 모델 경로, 크기, 프레임 수, FPS, 출력 수를 검증합니다.
- LTX-2.3은 추가로 프레임 수가 `8n+1` 형식이어야 합니다.
- 외부 Process는 Task 취소, 로그 기반 오류 보고, 백분율 진행률 추출을 지원합니다.
- MP4 출력은 `generatedVideo` 에셋으로 작업 공간에 추가되며 Web UI의 네이티브 `<video>`로 재생됩니다.
- Python 구현을 UI나 `AppStore`에 결합하지 않고 `GENIMAGE_LTX_RUNTIME` 또는 표준 설치 위치로 Runtime을 교체할 수 있습니다.

음악은 `MiniMaxMusic3GenerationService`가 `mlx-minimax-music3 generate`를 호출하여 생성합니다.

- 텍스트→음악은 `MusicGenerating`, `MusicGenerationRequest`, `MusicGenerationOptions`를 사용합니다.
- Swift가 프로필, 모델 완전성, 스타일 프롬프트, 5~300초 길이(최대 5분), 스텝, 출력 형식을 검증합니다. 가사는 선택 사항이며 비어 있으면 연주곡 마커와 보컬 없음 프롬프트로 변환합니다.
- Runtime은 임시 WAV를 생성하고 FFmpeg가 MP3 320 kbps, M4A AAC 256 kbps, ADTS AAC 256 kbps 또는 무손실 FLAC으로 변환합니다.
- 성공, 실패, 취소 시 모두 WAV, 프롬프트, 가사, 로그를 정리하며 완료된 압축 오디오만 `generatedAudio` 에셋으로 보존합니다.
- Web UI는 네이티브 `<audio controls>`로 재생하고 Inspector에 실제 길이, 형식, 44.1 kHz 샘플링 레이트, 채널 수를 표시합니다.
- Runtime은 `GENIMAGE_MINIMAX_MUSIC3_RUNTIME` 또는 표준 Application Support 위치로 교체할 수 있으며, 모델은 고정 revision에서 설치되고 별도의 Community License를 유지합니다.

MLX metallib은 `RuntimeSupport/mlx.metallib`에서 `build.command`를 통해 Release 실행 디렉터리로 복사됩니다. 배포 전에는 모델 라이선스 검토, 16/24/32 GB 부하 테스트, App bundle, 서명, 공증을 완료해야 합니다.

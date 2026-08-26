# GenMedia 아키텍처

[繁體中文](ARCHITECTURE.md) | [English](ARCHITECTURE.en.md) | [日本語](ARCHITECTURE.ja.md) | 한국어

## 설계 목표

1. 텍스트→이미지, 이미지→텍스트, 이미지→이미지, 비디오 생성, 음악 생성, 자막 생성, 업스케일은 서로 의존하지 않는 독립 기능입니다.
2. 이미지, 비디오, 오디오, 자막 출력으로 작업 흐름과 분기를 구성할 수 있습니다.
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
        ├── MediaTranscribing
        ├── SubtitleGenerating
        ├── TextGenerating
        └── ImageUpscaling
                    │
                    ▼
       MLX Swift / Core ML / Local REST Service / External CLI
```

Web UI는 Bridge를 통해서만 로컬 기능을 사용할 수 있으며 임의의 파일, 모델 디렉터리 또는 시스템 API를 직접 읽을 수 없습니다.

### Web UI 업데이트 전략

- 프롬프트, 네거티브 프롬프트 또는 가사 필드에 포커스가 있는 동안에는 Swift 상태를 받아도 로컬 편집 값을 유지하고 불필요한 전체 렌더링을 지연하여 커서, 선택 범위, IME 조합 상태가 초기화되지 않도록 합니다.
- 생성 유형과 프롬프트/가사/출력 설정 탭은 독립 생성 패널 renderer를 사용하며 미리보기, 플레이어, Inspector 또는 사이드바 DOM을 교체하지 않습니다.
- 전역 상태 변경으로 전체 렌더링이 꼭 필요하면 재생 중인 `<audio>`, `<video>`, 오디오 시각화 노드를 분리한 뒤 같은 에셋 위치에 다시 연결하여 재생 위치와 Web Audio 연결을 유지합니다.

## 내부 구조 정리

이번 정리는 코드 경계와 책임만 변경하며 기존 생성 기능, 사용자 흐름, Web Bridge 프로토콜은 변경하지 않습니다.

- `ApplicationSupport`가 `Models`, `Runtime`, `Workspace`, `Pasted`, `Generated`의 Application Support 경로를 한곳에서 정의하고, 실행 시 이전 `GenMedia` 작업 공간 데이터를 현재 위치로 가져옵니다.
- `OutputGeometry`가 크기 제한, 16배수 정렬, 비율 변환, 이미지→이미지 생성 계획을 담당합니다. Web UI의 `js/geometry.js`가 이를 미러링하여 Native, MCP, UI의 계산 결과를 일치시킵니다.
- `AppStore`는 타입 선언, 저장 속성, 초기화만 유지하고 Persistence, Paths, Selection, Profiles, OutputSettings, Assets, ImageGeneration, MediaGeneration, Jobs, ModelInstallation을 책임별 `AppStore+*.swift` 확장으로 분리했습니다.
- Web UI는 사이드바와 라우팅, 작업 공간 탭, 크기 계산, 전체 렌더링 보호를 `chrome.js`, `workspace-tabs.js`, `geometry.js`, `render-preservation.js`로 분리하고 `app.js`는 Bridge와 애플리케이션 조정을 담당합니다.
- `SubprocessRuntime`가 외부 Worker, 비디오·음악 CLI, FFmpeg의 실행 파일 검색, 환경, 로그, 진행률, 정체 감지, 취소, 종료 처리를 공통화합니다.
- 의존 패키지 소스 수정은 `Patches/manifest.txt`에 정의하고 `scripts/apply-runtime-patches.command`가 적용 및 검증합니다. pin 불일치나 수정 실패 시 수정되지 않은 소스로 계속하지 않고 빌드를 중단합니다.
- `GenImageMCP`는 자체 `InferenceServices`를 보유하는 독립 stdio server로 유지됩니다. 앱의 `LocalMCPServiceController`는 localhost HTTP transport 수명 주기만 관리하며 둘 다 동일한 `MCPServer` 도구 코어를 직접 사용하므로 stdio→HTTP 프록시가 아닙니다.
- 운영용 ACE-Step 단계 타입은 더 이상 PoC 이름을 사용하지 않습니다. 진단 전용 DiT probe를 운영 생성 단계와 분리하여 실험 코드와 앱 실행 경로를 명확히 했습니다.

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

`MediaAsset.parentAssetID`는 미디어 원본을 나타냅니다. parent가 없는 에셋은 독립 작업의 루트 노드입니다.

- 독립 텍스트→이미지: 생성된 이미지에 parent가 없습니다.
- 독립 이미지→텍스트: 먼저 루트 이미지를 가져오고 설명 결과를 Recipe에 기록합니다.
- 독립 업스케일: 먼저 루트 이미지를 가져오고 확대 결과는 원본 이미지를 parent로 사용합니다.
- 연결 생성: 생성 결과는 선택한 이미지를 parent로 사용합니다.
- 독립 텍스트→비디오: MP4 에셋에 parent가 없습니다.
- 이미지→비디오: MP4 에셋은 원본 이미지를 parent로 사용합니다.
- 독립 텍스트→음악: MP3, M4A, AAC 또는 FLAC 에셋에 parent가 없으며 실제 길이, 샘플링 레이트, 채널 수를 기록합니다.
- 자막 생성: 원본을 `importedVideo` 또는 `importedAudio`로 가져오고 SRT/WebVTT 결과를 `generatedSubtitle`로 저장하며 원본 미디어를 parent로 사용합니다.

`WorkflowGraph`가 lineage와 children 조회를 제공하므로 UI에서 에셋 관계를 추측할 필요가 없습니다.

열려 있는 작업 공간 탭이 생성 프로젝트의 수명 주기 경계입니다. Swift는 `Project`, `MediaAsset`, `WorkflowOperation`, 선택 상태를 Application Support의 JSON 스냅샷으로 원자 저장하며 일반적인 앱 종료 시 지우지 않습니다. Web UI는 탭 정보를 WebKit localStorage에 보관합니다. 탭을 닫으면 Bridge를 통해 네이티브 계층에 알리고 해당 탭의 에셋 및 lineage 인덱스만 제거하며 출력된 미디어 파일은 유지합니다.

이름이 있는 작업 공간은 탭 상위에 있으며 각 작업 공간은 자체 탭 집합을 유지합니다. 생성과 삭제는 Bridge를 통해 `AppStore+Workspaces`로 전달되고 삭제 전 확인이 필요합니다. 작업 공간 전환은 해당 탭과 선택 상태만 바꾸며 Runtime이나 미디어 플레이어를 다시 만들지 않습니다.

## 추론 Runtime

텍스트→이미지는 `Z-Image.swift` commit `28bfcf3148c041a554629247170eb54d9ac46830`을 사용합니다.

- macOS 14 이상, Swift 6.
- `ZImageGenerationRequest`는 프롬프트, 네거티브 프롬프트, 크기, 스텝, 시드, 모델, Runtime 옵션을 지원합니다.
- `ZImageTextToImageService`는 `ZImagePipeline.generate`를 감싸고 단계별 진행률을 제공합니다.
- 노이즈 제거 루프는 Swift Task 취소를 확인합니다.
- 모델 언로드, LoRA 언로드, 취소, 메모리 캐시 정리를 지원합니다.

멀티모달 이미지→텍스트 및 텍스트→텍스트 경로는 `mlx-swift-lm` revision `7da33441c7c08b010ff1aa8da9dc3d82277272f5`를 사용합니다.

- `QwenVLImageDescriptionService`는 `VLMModelFactory`를 통해 로컬 Qwen3-VL을 불러옵니다.
- `QwenTextGenerationService`는 같은 멀티모달 컨테이너의 텍스트 전용 입력을 사용합니다. Qwen3-VL, Qwen3.5, Qwen3.8 설명자는 `.imageToText`와 `.textToText`를 모두 선언하고 기능별 독립 Profile을 제공합니다.
- Qwen3.5 텍스트 전용 입력의 업스트림 호환 수정은 `Patches/MLX-Swift-LM-Qwen35-Text-Only.patch`로 적용합니다.
- 관리형 다운로드는 `processor_config.json`, 이미지/비디오 전처리 설정, Tokenizer, Chat Template, 가중치와 인덱스를 내려받아 검증하므로 텍스트 가중치만 있는 모델을 설치 완료로 처리하지 않습니다.
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

음악은 `MusicGenerationRouter`가 `MusicRuntimeAdapter.supports`에 따라 분배하며 Router에 모델 ID를 집중 하드코딩하지 않습니다.

- 텍스트→음악은 `MusicGenerating`, `MusicGenerationRequest`, `MusicGenerationOptions`를 사용합니다.
- `ACEStepMusicGenerationService`는 `.mlxSwift` 프로필을 사용하고 `ACEStepSwiftRuntime`을 직접 호출하여 Qwen3 Embedding, 조건 인코딩, Turbo DiT, Euler sampler, Oobleck VAE를 실행합니다. 외부 서비스나 Process를 시작하지 않습니다.
- ACE-Step은 10~300초, 1~20 steps, 선택적 가사, 연주곡 생성을 지원합니다. latent 길이는 VAE 샘플링 레이트와 hop length로 계산하며 긴 오디오는 오버랩 분할 디코딩과 PCM 스트림 쓰기로 최대 메모리 사용량을 제한합니다. 코드와 모델은 MIT License입니다.
- 음악 프로필은 `ProfileMusicConfiguration`으로 길이 범위와 목표/최대 의미를 제공하므로 Web UI가 모델 ID에 따라 분기할 필요가 없습니다.
- `MiniMaxMusic3GenerationService`는 `.externalCLI` 프로필로 `mlx-minimax-music3 generate`를 호출합니다. 5~300초 매개변수는 최대 길이이며 모델이 오디오 종료 토큰을 출력하면 더 일찍 자연 종료될 수 있습니다. 모델은 별도의 Community License를 유지합니다.
- 두 Adapter 모두 임시 WAV를 얻고 `AudioOutputEncoder`가 MP3 320 kbps, M4A AAC 256 kbps, ADTS AAC 256 kbps 또는 무손실 FLAC으로 변환합니다.
- 성공, 실패 또는 취소 시 임시 파일을 정리하며 완료된 압축 오디오만 실제 길이, 샘플링 레이트, 채널 수와 함께 `generatedAudio` 에셋으로 보존합니다.
- ACE-Step 가중치는 모델 센터에서 관리하며 네이티브 Runtime은 앱에 컴파일되므로 별도 설치 경로나 서비스 환경 변수를 사용하지 않습니다.

자막 생성은 미디어 가져오기와 텍스트 생성 사이에 위치합니다.

1. `MediaAudioPreparer`가 비디오 또는 오디오에서 인식용 오디오 트랙과 임시 경로를 준비합니다.
2. `SubtitleGenerationRouter`가 `MediaTranscribing.supports(profile:)`를 순서대로 평가해 첫 번째 일치 Adapter를 선택합니다.
3. Whisper Large v3 Turbo는 다국어, Paraformer Large는 중국어, Parakeet 0.6B는 일본어를 담당하며 모두 로컬 Core ML 경로로 실행됩니다.
4. 선택적 `QwenTextGenerationService`는 Qwen3.5/Qwen3.8 MLX로 자막을 묶음 번역하되 구간 시작과 종료 시간은 바꾸지 않습니다.
5. `SubtitleDocument`가 SRT/WebVTT를 만들고 현재 작업 공간에 `generatedSubtitle` 에셋으로 저장합니다.

`GenImageASRPoC`는 WhisperKit 미디어 디코딩, 언어 인식, 타임라인 출력만 검증하는 독립 실행 파일입니다. 앱 작업 공간에 쓰지 않으며 정식 흐름의 대체 경로도 아닙니다.

MLX metallib은 `RuntimeSupport/mlx.metallib`에서 `build.command`를 통해 Release 실행 디렉터리로 복사됩니다. 배포 전에는 모델 라이선스 검토, 16/24/32 GB 부하 테스트, App bundle, 서명, 공증을 완료해야 합니다.

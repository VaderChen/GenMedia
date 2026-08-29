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
- 작업 공간 탭 schema v3는 각 탭의 작업 유형, Profile 참조, Prompt, 출력 설정, 자동 흐름 단계를 저장합니다. 탭 전환 시 하나의 Bridge 명령으로 Native 생성 상태를 복원합니다.
- 전역 상태 변경으로 전체 렌더링이 꼭 필요하면 재생 중인 `<audio>`, `<video>`, 오디오 시각화 노드를 분리한 뒤 같은 에셋 위치에 다시 연결하여 재생 위치와 Web Audio 연결을 유지합니다.

## 내부 구조 정리

이번 정리는 코드 경계와 책임만 변경하며 기존 생성 기능, 사용자 흐름, Web Bridge 프로토콜은 변경하지 않습니다.

- `ApplicationSupport`가 `Models`, `Runtime`, `Workspace`, `Pasted`, `Generated`의 Application Support 경로를 한곳에서 정의하고, 실행 시 이전 `GenMedia` 작업 공간 데이터를 현재 위치로 가져옵니다.
- `OutputGeometry`가 크기 제한, 16배수 정렬, 비율 변환, 이미지→이미지 생성 계획을 담당합니다. Web UI의 `js/geometry.js`가 이를 미러링하여 Native, MCP, UI의 계산 결과를 일치시킵니다.
- `AppStore`는 타입 선언, 저장 속성, 초기화만 유지하고 Persistence, Paths, Selection, Profiles, OutputSettings, Assets, ImageGeneration, MediaGeneration, Jobs, ModelInstallation을 책임별 `AppStore+*.swift` 확장으로 분리했습니다.
- Web UI는 사이드바와 라우팅, 작업 공간 탭, 크기 계산, 전체 렌더링 보호를 `chrome.js`, `workspace-tabs.js`, `geometry.js`, `render-preservation.js`로 분리하고 `app.js`는 Bridge와 애플리케이션 조정을 담당합니다.
- `SubprocessRuntime`가 외부 Worker, 비디오·음악 CLI, FFmpeg의 환경, 로그, 진행률, 정체 감지, 취소, 종료 처리를 공통화합니다.
- `MediaCompatibilityService`는 `ffmpeg`/`ffprobe`의 유일한 검색 및 탐색 진입점입니다. 정식 App은 `Contents/Resources/bin/`을 우선하며 개발 실행에서만 환경 변수, Homebrew, `PATH`로 대체합니다.
- `MediaSourceCompatibilityService`는 가져온 모든 비디오와 오디오를 `ffprobe`로 분류합니다. 바로 재생할 수 있는 원본은 유지하고 H.264/HEVC는 가능하면 재다중화만 수행하며, 나머지 경우에만 VideoToolbox H.264/AAC 또는 M4A AAC 재생 프록시를 만듭니다. `AppStore+MediaImport`는 가져오기와 변환을 취소 가능한 Job으로 묶어 FFmpeg 진행률을 반환하고 변환 비트레이트를 이미지 면적에 따라 조정합니다.
- `AssetSchemeHandler`는 시간 기반 미디어에 HTTP Range를 응답하고 백그라운드 `FileHandle`로 512 KiB 청크를 전송합니다. 중지된 작업 집합으로 WebKit이 취소한 요청에 대한 콜백을 막으며 이미지는 기존처럼 한 번에 응답합니다. 시작 시 `ApplicationSupport.orphanMediaCacheFiles`가 대응 에셋이 없는 UUID 캐시 파일을 찾아 삭제합니다.
- `MediaCompositionService`는 모델이 필요 없는 이미지 루프 비디오와 미디어 병합 작업을 공용 FFmpeg 서브프로세스, 진행률, 취소, 출력 이름 규칙으로 실행합니다. 결과는 `generatedVideo` 에셋과 `WorkflowOperation`으로 기존 계보에 다시 연결됩니다.
- `automatic-flow.js`는 선언형 템플릿으로 Profile을 사전 검사하고 작업 공간과 탭을 만듭니다. 단계는 소스 탭과 에셋 ID로 연결되며 첫 ‘간단한 MV’ 흐름은 키 비주얼, 배경 음악, 이미지 루프, 미디어 병합으로 구성됩니다.
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

비디오는 `LTXVideoGenerationService`가 앱에 포함된 `GenImageLTXVideoWorker` Swift 하위 프로세스를 실행하여 생성합니다.

- 텍스트→비디오와 이미지→비디오는 `VideoGenerating` 및 `VideoGenerationRequest`를 공유합니다.
- Swift가 프로필, 모델 경로, 크기, 프레임 수, FPS, 출력 수를 검증합니다.
- LTX-2.3은 추가로 프레임 수가 `8n+1` 형식이어야 합니다.
- JSON request와 단계별 progress event는 기존 `RuntimeProcess`를 통해 실행되며 Task 취소, 로그 기반 오류 보고, 백분율 진행률 추출을 지원합니다.
- LTX LoRA 제어 비디오는 공용 FFmpeg 계층이 VideoToolbox H.264로 생성하며 GPL `libx264`에 의존하지 않습니다.
- MP4 출력은 `generatedVideo` 에셋으로 작업 공간에 추가되며 Web UI의 네이티브 `<video>`로 재생됩니다.
- Worker는 App Bundle의 `Contents/Helpers`에 복사됩니다. 개발 빌드에서는 `GENIMAGE_LTX_WORKER`로 덮어쓸 수 있지만 운영 흐름은 외부 Runtime에 의존하지 않습니다.

음악은 `MusicGenerationRouter`가 `MusicRuntimeAdapter.supports`에 따라 분배하며 Router에 모델 ID를 집중 하드코딩하지 않습니다.

- 텍스트→음악은 `MusicGenerating`, `MusicGenerationRequest`, `MusicGenerationOptions`를 사용합니다.
- `ACEStepMusicGenerationService`는 `.mlxSwift` 프로필을 사용하고 `ACEStepSwiftRuntime`을 직접 호출하여 Qwen3 Embedding, 조건 인코딩, Turbo DiT, Euler sampler, Oobleck VAE를 실행합니다. 외부 서비스나 Process를 시작하지 않습니다.
- ACE-Step은 10~300초, 1~20 steps, 선택적 가사, 연주곡 생성을 지원합니다. latent 길이는 VAE 샘플링 레이트와 hop length로 계산하며 긴 오디오는 오버랩 분할 디코딩과 PCM 스트림 쓰기로 최대 메모리 사용량을 제한합니다. 코드와 모델은 MIT License입니다.
- 음악 프로필은 `ProfileMusicConfiguration`으로 길이 범위와 목표/최대 의미를 제공하므로 Web UI가 모델 ID에 따라 분기할 필요가 없습니다.
- `MiniMaxMusic3GenerationService`는 `.externalCLI` 프로필로 앱에 포함된 `GenImageMiniMaxMusic3Worker`를 실행합니다. 앱이 JSON request를 기록하고 Worker가 고정 bfloat16 production 경로로 Swift/MLX pipeline을 실행합니다. 5~300초는 최대 길이이며 오디오 종료 token에서 조기 종료할 수 있습니다.
- 8-bit와 `mlx-community/MiniMax-Music3-4bit` checkpoint는 같은 Worker를 공유합니다. Worker는 autoregressive를 frame 단위, denoise를 chunk×step 단위, vocoder를 chunk 단위로 진행 보고합니다. 앱은 `RuntimeProcess` 취소와 강제 종료를 유지하고 완성된 WAV를 내장 FFmpeg로 변환합니다. MiniMax Music 3는 Python Runtime이 필요하지 않습니다.
- `Mothersuperior/minimax-music3-composer-5.7b-distilled`는 음악 구성 요소로 모델 센터에서 관리하며 기본적으로 `lr-6e-5` 가중치만 설치합니다. 독립 Profile이 아니므로 음악 Service에 단독으로 전달하지 않으며, 적용하려면 호환되는 Composer override Runtime이 필요합니다.
- 두 Adapter 모두 임시 WAV를 얻고 `AudioOutputEncoder`가 내장 FFmpeg를 통해 MP3 320 kbps, M4A AAC 256 kbps, ADTS AAC 256 kbps 또는 무손실 FLAC으로 변환합니다.
- 성공, 실패 또는 취소 시 임시 파일을 정리하며 완료된 압축 오디오만 실제 길이, 샘플링 레이트, 채널 수와 함께 `generatedAudio` 에셋으로 보존합니다.
- ACE-Step 가중치는 모델 센터에서 관리하며 네이티브 Runtime은 앱에 컴파일되므로 별도 설치 경로나 서비스 환경 변수를 사용하지 않습니다.

미디어 에셋은 `fileURL`과 선택적 `playbackURL`을 함께 보존합니다. 전자는 자막 출력, 계보, 명시적 삭제에 쓰는 사용자 원본을 항상 가리키고, 후자는 WebKit 미리보기용 `Application Support/GenImage/MediaCache` 호환 프록시만 가리킵니다. 에셋을 제거하거나 프로젝트를 닫을 때 프록시를 정리하며 원본 미디어는 변경하지 않습니다.

자막 생성은 미디어 가져오기와 텍스트 생성 사이에 위치합니다.

1. `MediaAudioPreparer`가 내장 `ffprobe`로 오디오 트랙을 확인한 뒤 내장 `ffmpeg`로 16 kHz 모노 PCM WAV와 임시 경로를 준비합니다.
2. `SubtitleGenerationRouter`가 `MediaTranscribing.supports(profile:)`를 순서대로 평가해 첫 번째 일치 Adapter를 선택합니다.
3. Whisper Large v3 Turbo는 다국어, Paraformer Large는 중국어, Parakeet 0.6B는 일본어를 담당하며 모두 로컬 Core ML 경로로 실행됩니다.
4. 선택적 `QwenTextGenerationService`는 Qwen3.5/Qwen3.8 MLX로 자막을 묶음 번역하되 구간 시작과 종료 시간은 바꾸지 않습니다.
5. `SubtitleDocument`가 SRT/WebVTT를 만들고 현재 작업 공간에 `generatedSubtitle` 에셋으로 저장합니다.

`GenImageASRPoC`는 WhisperKit 미디어 디코딩, 언어 인식, 타임라인 출력만 검증하는 독립 실행 파일입니다. 앱 작업 공간에 쓰지 않으며 정식 흐름의 대체 경로도 아닙니다.

`scripts/build-ffmpeg-macos.sh`는 Apple Silicon용 LGPL-only 동적 링크 및 교체 가능한 FFmpeg 배포물을 생성합니다. `build.command`는 GPL/nonfree Encoder를 거부하고 `ffmpeg`, `ffprobe`, dylib, 라이선스를 복사하며 install name을 `@rpath`로 바꾼 뒤 dylib, 도구, App 순서로 서명해 기존 DMG 공증 흐름으로 전달합니다. 동일한 Team ID로 서명한 Developer ID FFmpeg dylib와 도구는 별도 library-validation entitlement 없이 동작함을 확인했으며, 로컬 ad-hoc 빌드에서는 FFmpeg에 hardened runtime을 적용하지 않습니다. 사전 빌드 바이너리는 Git에서 제외되는 `third_party/ffmpeg/`에 있으며 GitHub Source archive에 포함되지 않습니다.

MLX metallib은 `RuntimeSupport/mlx.metallib`에서 `build.command`를 통해 Release 실행 디렉터리로 복사됩니다. 모델 라이선스 검토와 16/24/32 GB 부하 테스트는 계속해서 릴리스 요구 사항입니다.

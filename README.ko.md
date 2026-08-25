# GenMedia

[繁體中文](README.md) | [English](README.en.md) | [日本語](README.ja.md) | 한국어

GenMedia는 **Apple Silicon을 네이티브로 지원**하는 로컬 AI 미디어 생성 앱입니다. 이 프로젝트는 다음 기능을 갖춘 빌드 가능한 하이브리드 애플리케이션을 제공합니다.

- Swift가 모델, 프로필, 작업 대기열, 파일, MLX/Core ML 추론을 관리합니다.
- `WKWebView`에 HTML, CSS, JavaScript UI를 내장하여 네트워크 연결이나 npm 런타임이 필요하지 않습니다.
- 텍스트→이미지, 이미지→텍스트, 이미지→이미지, 텍스트→비디오, 이미지→비디오, 텍스트→음악, 업스케일을 독립적으로 실행하거나 에셋 계보를 통해 연결할 수 있습니다.
- 각 작업은 프로필 스냅샷을 보존하므로 모델이나 아키텍처가 업데이트된 뒤에도 당시 버전을 추적할 수 있습니다.
- 별도의 설정 화면에서 번체 중국어, 영어, 일본어, 한국어와 저장 가능한 6가지 색상 테마를 지원합니다.
- 표준 JSON-RPC 2.0 stdio MCP 서버를 Agent 및 자동화 도구에서 사용할 수 있습니다.

## 미리보기

![GenMedia 지능형 미디어 생성 인터페이스](images/cap001.jpg)

## 실행

요구 사항: macOS 14 이상, Apple Silicon, Xcode 16 이상.

```bash
./build.command
./run.command
```

`build.command`는 Release 실행 파일과 표준 `GenMedia.app`를 `dist/`에 생성합니다. App에는 WebUI 리소스, MLX Metal 런타임, MCP 서버, 모델 진단 도구가 포함됩니다.

```bash
# Release 실행 파일과 App 빌드
./build.command

# App bundle 없이 증분 Release 빌드만 실행
./build.command --no-app

# 버전과 Bundle ID 지정
GENIMAGE_VERSION=1.1.0 GENIMAGE_BUNDLE_ID=com.example.genimage ./build.command
```

`run.command`는 자동으로 `--no-app`를 사용하므로 일반적인 개발 실행에서 App bundle을 반복 생성하지 않습니다. 배포용 DMG는 Developer ID Application 서명, Apple 공증, Staple, Gatekeeper 검증을 수행하는 별도의 로컬 흐름에서 처리합니다.

### 비디오 Runtime

비디오 생성은 교체 가능한 외부 `ltx-2-mlx` Runtime을 사용합니다. Swift 앱은 프로필, 매개변수 검증, 작업 대기열, 취소, 진행률, 에셋, 비디오 재생을 관리합니다. 처음 사용하기 전에 CLI와 FFmpeg를 설치하세요.

```bash
brew install uv ffmpeg
./scripts/install-ltx-runtime.command
```

앱은 `GENIMAGE_LTX_RUNTIME`, `GENIMAGE_LTX_RUNTIME_ROOT/.venv/bin/ltx-2-mlx`, App Helpers, `~/.local/bin/ltx-2-mlx`, 일반적인 Homebrew 경로, `PATH` 순서로 검색합니다. 실행 파일이 사용자 지정 위치에 있다면 다음과 같이 지정합니다.

```bash
GENIMAGE_LTX_RUNTIME="/absolute/path/to/ltx-2-mlx" ./run.command
```

`ltx-2-mlx`는 기본적으로 Gemma 텍스트 인코더 설정을 사용합니다. 로컬 Gemma 모델이 있다면 `GENIMAGE_LTX_GEMMA_MODEL`에 모델 디렉터리 또는 Hugging Face ID를 지정할 수 있습니다. 현재 앱 DMG에는 Python Runtime, Gemma 가중치, FFmpeg가 포함되지 않습니다. 정식 배포 전에 이를 선택적 외부 구성 요소로 취급하고 Runtime 및 모델 라이선스를 각각 확인해야 합니다.

### 프로필, 작업 및 메모리

- 프로필은 사용 중, 사용 가능, 다운로드 중, 사용 불가 순으로 정렬됩니다. 모델과 LoRA 종속성이 모두 준비된 프로필에는 연한 녹색 테두리가 표시되며 다운로드가 끝나면 즉시 다시 정렬됩니다.
- 취소 시 먼저 `cancelling` 상태로 전환되고 Runtime Task가 끝나면 `cancelled`로 변경되어 생성 및 메모리 버튼이 다시 활성화됩니다. ETA는 진행률 35% 및 실행 15초 이후 숫자로 표시되며 안정적인 샘플이 부족하면 전체 경과 시간을 사용합니다.
- Z-Image MLX 호환 계층은 `quantize_config.json`, affine/mxfp4, packed pad token, FP16에서 BF16으로의 로딩을 처리합니다. `build.command`는 Swift Package 해석 후 `Patches/`의 Runtime 수정 사항을 자동 적용합니다. andrevp Z-Image Turbo MLX 4-bit는 실제 이미지 생성으로 검증했습니다.
- 텍스트→이미지 작업이 끝난 뒤에도 모델 가중치와 워밍업 buffer를 유지합니다. 5분 동안 유휴 상태가 되면 재사용 가능한 MLX 임시 buffer만 정리하고 모델은 언로드하지 않습니다. 사이드바의 메모리 해제, 모델 전환 또는 프로필 전환 시 RAM 90% 초과 보호가 동작할 때만 불필요한 Runtime을 해제합니다.
- 다운로드는 원본 파일명을 유지합니다. 생성 결과는 `Image-YYYYMMDD-HHmm`, `Video-YYYYMMDD-HHmm` 또는 `Music-YYYYMMDD-HHmm`을 사용하며 같은 분에 중복되면 일련번호를 추가합니다. 설정에서 출력 디렉터리를 변경할 수 있습니다.
- 열려 있는 각 작업 공간 탭을 생성 프로젝트로 취급합니다. 에셋과 lineage는 Application Support에 원자적으로 저장되며 앱을 다시 실행해도 복원됩니다. 탭을 명시적으로 닫을 때만 해당 프로젝트의 작업 공간 인덱스를 제거하고 출력된 미디어 파일은 디스크에 유지합니다.
- 프롬프트와 가사를 편집하는 동안 커서, 선택 범위, IME 조합 상태를 네이티브 상태 업데이트로부터 보존합니다. 생성 유형, 프롬프트, 가사, 출력 설정 탭은 생성 패널만 다시 렌더링합니다. 불가피한 전체 업데이트에서도 재생 중인 오디오와 비디오 노드를 재사용하여 재생이 끊기지 않도록 합니다.
- 작업 공간 필름스트립에 이미지 가져오기 버튼이 있으며 Finder에서 PNG, JPEG, WebP, GIF, TIFF, HEIC, HEIF 파일을 하나 이상 끌어 놓을 수 있습니다. 음악 생성 중에는 미디어 소스가 섞이지 않도록 이미지 가져오기를 비활성화합니다. 이미지 생성에서 원본 이미지를 선택하면 기본 버튼이 이미지→이미지 프로필을 사용하고, 선택하지 않으면 텍스트→이미지 프로필을 사용합니다.
- 이미지와 비디오 비율 항목은 드롭다운으로 제공됩니다. 이미지→이미지에서는 원본 이미지를 선택한 뒤에만 `원본 해상도`가 표시되며, 원본 크기를 Runtime에서 사용할 수 있는 16의 배수로 변환합니다.
- 이미지→이미지의 너비와 높이는 Qwen Image Edit Runtime에 실제로 전달됩니다. 지정한 비율이 원본과 다르면 원본을 비율을 유지한 채 축소하고 가장자리 확장으로 목표 캔버스를 채운 후 생성하므로 원본 전체 내용을 보존합니다. `128 × 192`도 사용할 수 있지만 너무 낮은 해상도에서는 세부 묘사와 구도 안정성이 떨어질 수 있어 `512 × 768` 이상을 권장합니다.

### 음악 Runtime

텍스트→음악은 `MusicGenerationRouter`가 활성 프로필에 따라 ACE-Step 1.5 또는 MiniMax Music 3 Adapter로 분배합니다. Swift 앱은 음악 스타일, 선택적 프롬프트, 선택적 가사, 스텝, 시드, 취소, 남은 시간 추정, 오디오 정보를 일관되게 관리합니다. 프롬프트를 비워 두면 선택한 스타일을 사용하고, 가사를 비워 두면 연주곡을 생성합니다. 각 Runtime의 임시 WAV는 공용 FFmpeg 계층에서 MP3, M4A, AAC 또는 FLAC으로 변환됩니다.

- **ACE-Step 1.5 Turbo MLX**: 권장 프로필입니다. 앱에 내장된 Apple Silicon 네이티브 Swift/MLX Runtime으로 추가 서비스 설치 없이 10~300초 길이의 노래 또는 연주곡을 생성합니다. 코드와 모델은 상업적으로 사용할 수 있는 MIT License입니다. 긴 오디오는 오버랩 분할 VAE 디코딩으로 메모리 사용량을 제어합니다.

- **MiniMax Music 3 MLX 8-bit**: 외부 `mlx-minimax-music3` CLI로 실행됩니다. 5~300초 설정값은 최대 길이이며 곡 구조에 따라 모델이 더 일찍 자연 종료될 수 있습니다. 완료 후 앱에 실제 출력 길이가 표시됩니다. 모델에는 Community License가 적용됩니다.

ACE-Step 네이티브 Runtime은 앱에 포함됩니다. 모델, MiniMax Music 3 Runtime, FFmpeg는 선택적 외부 구성 요소입니다.

## 검증

```bash
swift test

for file in Sources/GenImageApp/Resources/WebUI/js/*.js; do
  node --check "$file"
done
```

로컬 모델과 자동 생성된 프로필을 진단합니다.

```bash
swift run GenImageDoctor

# 또는 사용자 지정 모델 디렉터리 지정
GENIMAGE_MODEL_ROOT="/path/to/models" swift run GenImageDoctor
```

표준 MCP stdio 서버를 시작합니다.

```bash
.build/arm64-apple-macosx/release/GenImageMCP
```

MCP는 `initialize`, `ping`, `tools/list`, `tools/call`을 지원합니다. 제공 도구에는 로컬 모델, 프로필, 네이티브 Z-Image 텍스트→이미지 생성, Qwen3-VL 이미지 설명, Core ML 업스케일이 포함됩니다.

MCP 엔드투엔드 검증을 완료했습니다. `genimage_generate_image`는 로컬 Z-Image Turbo Q4 모델로 PNG를 출력하고, `genimage_describe_image`는 Qwen3-VL로 번체 중국어 설명을 생성하며, `genimage_upscale_image`는 로컬 Real-ESRGAN Core ML 모델로 4배 업스케일을 수행합니다.

## 프로젝트 구조

```text
Sources/
├── GenImageCore/
│   ├── DomainModels.swift        # 에셋, 레시피, 작업, 모델, 프로필
│   ├── InferenceServices.swift   # 이미지, 텍스트, 비디오, 음악 추론 인터페이스
│   ├── ModelCatalog.swift        # 기본 제공 모델과 프로필
│   ├── OutputFileNaming.swift    # 이미지, 비디오 및 음악 출력 이름
│   ├── ProjectWorkspacePersistence.swift # 열린 생성 프로젝트 영속화
│   └── WorkflowGraph.swift       # 에셋 계보와 분기 관계
├── GenImageRuntime/
│   ├── ZImageTextToImageService.swift
│   ├── QwenVLImageDescriptionService.swift
│   ├── Qwen2511ImageToImageService.swift
│   ├── LTXVideoGenerationService.swift
│   ├── MusicGenerationRouter.swift
│   ├── ACEStepMusicGenerationService.swift
│   ├── MiniMaxMusic3GenerationService.swift
│   ├── AudioOutputEncoder.swift
│   └── CoreMLUpscaleService.swift
└── GenImageApp/
    ├── AppStore.swift            # 애플리케이션 상태와 작업 조정
    ├── HybridBridgeController.swift
    ├── HybridWebView.swift
    ├── AssetSchemeHandler.swift  # 로컬 이미지, 비디오, 오디오를 WebUI에 안전하게 제공
    └── Resources/WebUI/          # HTML/CSS/JavaScript 프런트엔드
Patches/                           # 빌드 시 적용되는 Z-Image MLX 호환성 수정
```

## 현재 상태

앱은 Z-Image Turbo 텍스트→이미지, Qwen3-VL 이미지→텍스트, Qwen 2511 이미지→이미지, LTX-2.3 MLX 텍스트→비디오 및 이미지→비디오, ACE-Step 1.5 Turbo MLX와 MiniMax Music 3 MLX 8-bit 텍스트→음악, Core ML Real-ESRGAN 업스케일의 로컬 추론에 연결되어 있습니다. 비디오는 MP4로, 음악은 MP3/M4A/AAC/FLAC으로 추가되며 실제 길이, 샘플링 레이트, 채널 수, 프로필 스냅샷, 계보를 보존합니다.

추가 정보:

- [업데이트 노트](UpdateNote.md)
- [아키텍처](docs/ARCHITECTURE.ko.md)
- [Web Bridge](docs/WEB_BRIDGE.ko.md)
- [로드맵](docs/ROADMAP.ko.md)
- [MCP 인터페이스](docs/MCP.ko.md)
- [로컬 모델 테스트 보고서](docs/MODEL_TEST_REPORT.ko.md)

## 라이선스

이 프로젝트는 GPLv3와 상용 라이선스의 이중 라이선스 방식을 사용합니다.

- 오픈 소스 사용은 [GNU General Public License v3.0](LICENSE)에 따라 허가됩니다.
- 비공개 소스 통합, 독점 제품 배포, 맞춤형 상용 조건 등 GPLv3를 준수할 수 없거나 준수하지 않으려는 경우 저작권자에게 별도의 상용 라이선스를 요청하세요.

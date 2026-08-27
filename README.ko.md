# GenMedia

[繁體中文](README.md) | [English](README.en.md) | [日本語](README.ja.md) | 한국어

GenMedia는 **Apple Silicon을 네이티브로 지원**하는 로컬 AI 미디어 생성 앱입니다. 이 프로젝트는 다음 기능을 갖춘 빌드 가능한 하이브리드 애플리케이션을 제공합니다.

- Swift가 모델, 프로필, 작업 대기열, 파일, MLX/Core ML 추론을 관리합니다.
- `WKWebView`에 HTML, CSS, JavaScript UI를 내장하여 네트워크 연결이나 npm 런타임이 필요하지 않습니다.
- 텍스트→이미지, 이미지→텍스트, 이미지→이미지, 텍스트→비디오, 이미지→비디오, 텍스트→음악, 자막 생성, 업스케일을 독립적으로 실행하거나 에셋 계보를 통해 연결할 수 있습니다.
- 자동 흐름은 독립 설정을 가진 작업 공간 탭을 만듭니다. ‘간단한 MV’ 템플릿은 키 비주얼, 배경 음악, 이미지 루프, 미디어 병합 단계를 준비합니다.
- 각 작업은 프로필 스냅샷을 보존하므로 모델이나 아키텍처가 업데이트된 뒤에도 당시 버전을 추적할 수 있습니다.
- 별도의 설정 화면에서 번체 중국어, 영어, 일본어, 한국어와 저장 가능한 6가지 색상 테마를 지원합니다.
- 설정 화면의 스위치로 localhost 전용 MCP HTTP API를 시작할 수 있으며, 앱이 실행 중이 아니어도 독립 JSON-RPC 2.0 stdio 서버를 사용할 수 있습니다.

## 미리보기

![GenMedia 지능형 미디어 생성 인터페이스](images/cap001.jpg)

## 실행

요구 사항: macOS 14 이상, Apple Silicon, Xcode 16 이상.

```bash
./build.command
./run.command
```

`build.command`는 Release 실행 파일과 표준 `GenMedia.app`를 `dist/`에 생성합니다. App에는 WebUI 리소스, MLX Metal 런타임, MCP 서버, 모델 진단 도구와 통합 미디어 호환 계층인 LGPL 동적 `ffmpeg`/`ffprobe`가 포함됩니다. 첫 App bundle 빌드 시 `build.command`가 소스를 자동으로 다운로드하고 내장 FFmpeg 배포판을 준비합니다. FFmpeg를 수동으로 준비하거나 `pkg-config`를 설치할 필요가 없습니다.

```bash
# Release 실행 파일과 App 빌드
./build.command

# App bundle 없이 증분 Release 빌드만 실행
./build.command --no-app

# 버전과 Bundle ID 지정
GENIMAGE_VERSION=1.1.0 GENIMAGE_BUNDLE_ID=com.example.genimage ./build.command
```

`run.command`는 자동으로 `--no-app`를 사용하므로 일반적인 개발 실행에서 App bundle을 반복 생성하지 않습니다. 배포용 DMG는 Developer ID Application 서명, Apple 공증, Staple, Gatekeeper 검증을 수행하는 별도의 로컬 흐름에서 처리합니다.

### FFmpeg 빌드 문제 해결

- 첫 `./build.command` 실행에는 FFmpeg와 LAME 소스를 내려받기 위한 네트워크 연결이 필요합니다. 이후 빌드는 캐시된 소스와 `third_party/ffmpeg`를 재사용하며, 없거나 불완전할 때만 자동으로 다시 빌드합니다.
- Homebrew `pkg-config`는 필요하지 않습니다. LAME 구성 확인 전용 fallback을 공백이 없는 임시 경로에서 실행하므로 프로젝트 경로에 공백이 있어도 빌드할 수 있습니다.
- ExFAT 같은 외부 파일 시스템이 만드는 `._*` AppleDouble sidecar는 dylib 처리 전에 제거되어 Mach-O 파일로 잘못 인식되지 않습니다.
- 빌드가 중단되거나 실패하면 이전의 사용 가능한 FFmpeg 배포판을 복원합니다. 네트워크 또는 Xcode 문제를 해결한 뒤 `./build.command`를 다시 실행하세요. `GENMEDIA_FFMPEG_ROOT`로 출력 위치를 변경할 수 있습니다.

### 비디오 Runtime

비디오 생성은 교체 가능한 외부 `ltx-2-mlx` Runtime을 사용합니다. Swift 앱은 프로필, 매개변수 검증, 작업 대기열, 취소, 진행률, 에셋, 비디오 재생을 관리합니다. 처음 사용할 때는 비디오 CLI만 설치하면 됩니다.

```bash
brew install uv
./scripts/install-ltx-runtime.command
```

앱은 `GENIMAGE_LTX_RUNTIME`, `GENIMAGE_LTX_RUNTIME_ROOT/.venv/bin/ltx-2-mlx`, App Helpers, `~/.local/bin/ltx-2-mlx`, 일반적인 Homebrew 경로, `PATH` 순서로 검색합니다. 실행 파일이 사용자 지정 위치에 있다면 다음과 같이 지정합니다.

```bash
GENIMAGE_LTX_RUNTIME="/absolute/path/to/ltx-2-mlx" ./run.command
```

`ltx-2-mlx`는 기본적으로 Gemma 텍스트 인코더 설정을 사용합니다. 로컬 Gemma 모델이 있다면 `GENIMAGE_LTX_GEMMA_MODEL`에 모델 디렉터리 또는 Hugging Face ID를 지정할 수 있습니다. App DMG에는 Python Runtime과 Gemma 가중치는 포함되지 않지만 LGPL 동적 FFmpeg는 내장됩니다. 배포 전에 비디오 Runtime과 모델 라이선스를 각각 확인하세요.

### 호환 미디어 가져오기

가져온 비디오와 오디오는 내장 `ffprobe`가 컨테이너, Codec, 트랙, 길이, 회전이 반영된 표시 크기를 검사하며, 긴 변환은 진행률이 표시되는 취소 가능한 작업으로 실행됩니다. WebKit에서 직접 재생할 수 있으면 원본을 유지하고, H.264/HEVC에서 컨테이너나 오디오만 호환되지 않으면 MP4로 무손실 재다중화합니다. 그 밖의 비디오는 VideoToolbox H.264/AAC, 호환되지 않는 오디오는 M4A AAC 재생 프록시를 생성합니다. 재생 프록시는 `AssetSchemeHandler`가 백그라운드 HTTP Range 청크 스트림으로 제공하므로 점진 재생과 탐색을 지원합니다. 원본 경로는 자막 소스로 유지되므로 자막 출력 폴더와 이름은 원본 파일을 따릅니다. 시작 시 대응 에셋이 없는 MediaCache 고아 파일을 삭제하며, 프로젝트를 닫거나 에셋을 제거할 때도 App이 관리하는 호환 캐시만 삭제합니다.

### 자동 흐름과 미디어 합성

자동 흐름은 선언형 단계로 새 작업 공간과 탭을 만듭니다. 각 탭은 작업 유형, Profile, Prompt, 이미지/비디오/음악/미디어 처리 설정을 개별 보존하므로 탭 전환이나 App 재시작 시 다른 초안을 덮어쓰지 않습니다. 첫 번째 ‘간단한 MV’ 템플릿은 텍스트→이미지, 텍스트→음악, 이미지 루프 비디오, 미디어 병합을 연결합니다.

이미지 루프와 미디어 병합은 AI 모델이 필요하지 않으며 `MediaCompositionService`가 내장 FFmpeg로 실행합니다. 이미지 루프는 여러 이미지, 이미지당 시간, 전체 길이, 해상도, FPS, Cover/Contain을 지원합니다. 미디어 병합은 원본 오디오 교체/믹스, 볼륨, 출력 길이 정책을 지원합니다. 단계 간 입력은 파일명이 아니라 에셋 ID로 전달합니다.

### 자막 생성

자막 흐름은 비디오 또는 오디오를 가져와 내장 `ffprobe`로 트랙을 검사하고, 내장 `ffmpeg`로 선택한 오디오를 16 kHz 모노 PCM으로 통일한 뒤 `SubtitleGenerationRouter`가 네이티브 Core ML ASR Adapter를 선택합니다. 구간 타임라인을 보존하며 SRT 또는 WebVTT로 출력합니다. 다국어 Whisper Large v3 Turbo, 중국어 Paraformer Large, 일본어 Parakeet 0.6B를 지원하고 입력 언어는 자동 감지하거나 Profile에서 지정할 수 있습니다.

인식 후에는 로컬 Qwen3.5/Qwen3.8 MLX 텍스트 모델로 타임라인을 유지한 채 번체 중국어, 간체 중국어, 영어, 일본어, 한국어, 스페인어, 프랑스어, 독일어, 이탈리아어, 포르투갈어, 러시아어, 아랍어, 힌디어, 벵골어, 인도네시아어, 베트남어, 태국어, 터키어, 폴란드어, 네덜란드어, 스웨덴어, 체코어, 우크라이나어, 말레이어, 필리핀어의 25개 언어로 번역할 수 있습니다. 결과는 원본 비디오 또는 오디오를 parent로 하는 `generatedSubtitle` 에셋으로 현재 작업 공간에 저장됩니다.

`GenImageASRPoC`는 메인 앱 작업 공간을 수정하지 않고 미디어 디코딩, 언어 인식, 타임코드를 확인하는 독립 WhisperKit 검증 도구입니다. 정식 앱 흐름도 동일한 Swift/Core ML 경계를 사용합니다. 자세한 내용은 [ASR 자막 PoC](docs/ASR_POC.ko.md)를 참고하세요.

Qwen3-VL, Qwen3.5, Qwen3.8은 멀티모달 모델이므로 모델 센터에서 이미지→텍스트와 텍스트→텍스트 두 분류에 모두 표시되고 각 기능별 Profile을 제공합니다. 관리형 다운로드에는 `processor_config.json`, 이미지/비디오 전처리 설정, Tokenizer, Chat Template, 전체 가중치 인덱스가 포함되며 필수 파일 검증이 끝난 뒤에만 설치 완료로 표시됩니다.

### Civitai LoRA

모델 센터에는 `ZImageTurbo` 기반 Civitai LoRA(Asian Beauties, Turbo Lightning, Flat AnimeStyle, Diorama)도 포함되어 있습니다. 설정의 **Civitai LoRA** 카드에 API Token을 붙여 넣고 저장하세요. Token은 macOS Keychain에만 저장하며 모델 manifest, 작업 공간 또는 로그에 기록하지 않습니다. 이후 모델 센터가 Civitai 웹사이트를 열어 수동 다운로드를 요구하는 대신 HTTPS `Authorization: Bearer` 헤더로 파일을 직접 다운로드합니다.

기존 UI 없는 실행 스크립트와의 호환을 위해 `CIVITAI_TOKEN` 환경 변수도 fallback으로 계속 지원합니다.

### 프로필, 작업 및 메모리

- 프로필은 사용 중, 사용 가능, 다운로드 중, 사용 불가 순으로 정렬됩니다. 모델과 LoRA 종속성이 모두 준비된 프로필에는 연한 녹색 테두리가 표시되며 다운로드가 끝나면 즉시 다시 정렬됩니다.
- 취소 시 먼저 `cancelling` 상태로 전환되고 Runtime Task가 끝나면 `cancelled`로 변경되어 생성 및 메모리 버튼이 다시 활성화됩니다. ETA는 진행률 35% 및 실행 15초 이후 숫자로 표시되며 안정적인 샘플이 부족하면 전체 경과 시간을 사용합니다.
- Z-Image MLX 호환 계층은 `quantize_config.json`, affine/mxfp4, packed pad token, FP16에서 BF16으로의 로딩을 처리합니다. andrevp Z-Image Turbo MLX 4-bit는 실제 이미지 생성으로 검증했습니다.
- 의존 패키지에 대한 소스 수정은 `Patches/manifest.txt`에 나열하며 Swift Package 해석 후 `scripts/apply-runtime-patches.command`가 적용합니다(`build.command`가 자동 호출). 의존 패키지 버전이 manifest와 다르거나, 수정 파일이 없거나, 적용에 실패하거나, 적용 후 예상한 표시를 찾지 못하면 빌드를 중단하며 수정되지 않은 소스로 계속 진행하지 않습니다. `scripts/apply-runtime-patches.command --verify`로 검사만 할 수 있습니다.
- 텍스트→이미지 작업이 끝난 뒤에도 모델 가중치와 워밍업 buffer를 유지합니다. 5분 동안 유휴 상태가 되면 재사용 가능한 MLX 임시 buffer만 정리하고 모델은 언로드하지 않습니다. 사이드바의 메모리 해제, 모델 전환 또는 프로필 전환 시 RAM 90% 초과 보호가 동작할 때만 불필요한 Runtime을 해제합니다.
- 다운로드는 원본 파일명을 유지합니다. 생성 결과는 `Image-YYYYMMDD-HHmm`, `Video-YYYYMMDD-HHmm` 또는 `Music-YYYYMMDD-HHmm`을 사용하며 같은 분에 중복되면 일련번호를 추가합니다. 설정에서 출력 디렉터리를 변경할 수 있습니다.
- 열려 있는 각 작업 공간 탭을 생성 프로젝트로 취급합니다. 에셋과 lineage는 Application Support에 원자적으로 저장되며 앱을 다시 실행해도 복원됩니다. 탭을 명시적으로 닫을 때만 해당 프로젝트의 작업 공간 인덱스를 제거하고 출력된 미디어 파일은 디스크에 유지합니다.
- 앱 데이터는 모두 `~/Library/Application Support/GenImage/`(`Models`, `Runtime`, `Workspace`, `Pasted`, `Generated`)에 있으며 `GenImageCore/ApplicationSupport.swift`가 유일한 정의처입니다. 작업 공간 인덱스는 예전에 `GenMedia/`에 기록되었고 실행 시 현재 루트로 가져옵니다. 이름이 같은 항목은 기존 것을 유지하며 덮어쓰거나 병합하지 않습니다. `Runtime/` 아래 Python venv가 실행 스크립트에 절대 경로를 새겨 두기 때문에 앱 이름에 맞춘 `GenMedia`로 바꾸지 않고 `GenImage`를 유지합니다.
- 프롬프트와 가사를 편집하는 동안 커서, 선택 범위, IME 조합 상태를 네이티브 상태 업데이트로부터 보존합니다. 생성 유형, 프롬프트, 가사, 출력 설정 탭은 생성 패널만 다시 렌더링합니다. 불가피한 전체 업데이트에서도 재생 중인 오디오와 비디오 노드를 재사용하여 재생이 끊기지 않도록 합니다.
- 작업 공간 필름스트립에 이미지 가져오기 버튼이 있으며 Finder에서 PNG, JPEG, WebP, GIF, TIFF, HEIC, HEIF 파일을 하나 이상 끌어 놓을 수 있습니다. 음악 생성 중에는 미디어 소스가 섞이지 않도록 이미지 가져오기를 비활성화합니다. 이미지 생성에서 원본 이미지를 선택하면 기본 버튼이 이미지→이미지 프로필을 사용하고, 선택하지 않으면 텍스트→이미지 프로필을 사용합니다.
- 이미지와 비디오 비율 항목은 드롭다운으로 제공됩니다. 이미지→이미지에서는 원본 이미지를 선택한 뒤에만 `원본 해상도`가 표시되며, 원본 크기를 Runtime에서 사용할 수 있는 16의 배수로 변환합니다.
- 이미지→이미지의 너비와 높이는 Qwen Image Edit Runtime에 실제로 전달됩니다. 지정한 비율이 원본과 다르면 원본은 자체 해상도를 유지한 채 가장자리 확장으로 출력 비율 캔버스에 배치된 후 생성되므로 원본 전체 내용을 보존합니다.
- 조건 이미지는 생성 해상도로 인코딩하여 조건 그리드와 출력 그리드의 RoPE 위치를 정확히 맞춥니다. 조건 그리드를 1024² 면적으로 고정하면 모델이 원본 중앙에만 정렬되어 출력이 잘린 확대 결과가 됩니다.
- 생성 해상도와 출력 해상도를 분리했습니다. 지정 면적이 1024² 미만이면 Runtime이 지정 비율 그대로 1024² 면적(diffusers 참조 구현과 동일한 약 4096 latent 토큰)으로 생성한 뒤 Lanczos로 지정 크기까지 축소합니다. 학습 토큰 수를 크게 밑돌면 DiT의 디노이즈가 저하되다가 줄무늬로 붕괴하므로 `128 × 192` 같은 저해상도는 축소로 만드는 편이 확실히 낫습니다.
- 지정 면적이 1024² 이상이면 지정 크기로 그대로 생성하며 축소하지 않습니다. 해상도가 높아질수록 Runtime 메모리 사용량과 생성 시간도 함께 증가합니다. 1024² 미만 출력도 1024² 면적으로 생성하므로 출력이 작아져도 생성 시간은 줄지 않습니다.
- 이미지→이미지 생성을 시작할 때 출력 면적이 `512 × 512` 미만이면 경고 대화 상자를 먼저 표시하며 ‘취소’ 또는 ‘그래도 생성’을 선택할 수 있습니다. 한 번 확인하면 해당 실행 동안 다시 표시하지 않습니다.

### 음악 Runtime

텍스트→음악은 `MusicGenerationRouter`가 활성 프로필에 따라 ACE-Step 1.5 또는 MiniMax Music 3 Adapter로 분배합니다. Swift 앱은 음악 스타일, 선택적 프롬프트, 선택적 가사, 스텝, 시드, 취소, 남은 시간 추정, 오디오 정보를 일관되게 관리합니다. 프롬프트를 비워 두면 선택한 스타일을 사용하고, 가사를 비워 두면 연주곡을 생성합니다. 각 Runtime의 임시 WAV는 공용 FFmpeg 계층에서 MP3, M4A, AAC 또는 FLAC으로 변환됩니다.

- **ACE-Step 1.5 Turbo MLX**: 권장 프로필입니다. 앱에 내장된 Apple Silicon 네이티브 Swift/MLX Runtime으로 추가 서비스 설치 없이 10~300초 길이의 노래 또는 연주곡을 생성합니다. 코드와 모델은 상업적으로 사용할 수 있는 MIT License입니다. 긴 오디오는 오버랩 분할 VAE 디코딩으로 메모리 사용량을 제어합니다.

- **MiniMax Music 3 MLX 8-bit**: 앱에 포함된 독립 Swift Worker로 실행됩니다. 5~300초 설정값은 최대 길이이며 곡 구조에 따라 모델이 더 일찍 자연 종료될 수 있습니다. 완료 후 앱에 실제 출력 길이가 표시됩니다. 모델에는 Community License가 적용됩니다.
- **MiniMax Music 3 MLX 4-bit**: 앱에 포함된 독립 Swift Worker로 실행되며 완전한 affine 4-bit MLX checkpoint를 사용합니다. 5~300초 설정값은 최대 길이이며 곡 구조에 따라 더 일찍 자연 종료될 수 있습니다. 모델에는 MiniMax Music 3 Community License가 적용됩니다.
- **MiniMax Music 3 Composer 5.7B Distilled**: 모델 센터에서 선택적 Composer 가속 구성 요소로 제공합니다. 설치 프로그램은 `lr-6e-5` 가중치를 선택하며, 완전한 Music 3 checkpoint와 호환 Runtime이 필요하고 단독으로 음악을 생성할 수 없습니다.

ACE-Step 네이티브 Runtime, MiniMax Music 3 Swift Worker와 LGPL FFmpeg 호환 계층은 앱에 포함됩니다. MiniMax Music 3 모델은 선택적 구성 요소입니다.

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
MCP_BIN_DIR="$(swift build -c release --show-bin-path)"
"$MCP_BIN_DIR/GenImageMCP"
```

MCP는 `initialize`, `ping`, `tools/list`, `tools/call`을 지원합니다. 제공 도구에는 로컬 모델, 프로필, 네이티브 Z-Image 텍스트→이미지 생성, Qwen 이미지 편집/설명, Core ML 업스케일, 독립 자막 생성이 포함됩니다.

앱에서 사용할 때는 설정 → MCP 연동의 스위치를 켜면 localhost 전용 HTTP POST JSON-RPC 엔드포인트 `http://127.0.0.1:12181/mcp`가 표시됩니다. HTTP와 stdio는 같은 MCP 도구 코어를 공유하지만 stdio 실행 파일은 GenMedia.app이 꺼져 있어도 독립적으로 동작합니다.

MCP 엔드투엔드 검증을 완료했습니다. `genimage_generate_image`는 로컬 Z-Image Turbo Q4 모델로 PNG를 출력하고, `genimage_describe_image`는 Qwen3-VL로 번체 중국어 설명을 생성하며, `genimage_upscale_image`는 로컬 Real-ESRGAN Core ML 모델로 4배 업스케일을 수행합니다.

## 프로젝트 구조

이번 내부 리팩터링은 코드 경계와 책임만 변경하며 기존 생성 기능, 사용자 흐름, Web Bridge 프로토콜은 변경하지 않습니다.

```text
Sources/
├── GenImageCore/
│   ├── ApplicationSupport.swift  # 앱 데이터 위치의 유일한 정의
│   ├── CivitaiTokenStore.swift   # Civitai Token의 macOS Keychain 저장
│   ├── DomainModels.swift        # 에셋, 레시피, 작업, 모델, 프로필
│   ├── InferenceServices.swift   # 이미지, 텍스트, 비디오, 음악, 자막 추론 인터페이스
│   ├── ModelCatalog.swift        # 기본 제공 모델과 프로필
│   ├── OutputFileNaming.swift    # 이미지, 비디오, 음악 및 자막 출력 이름
│   ├── OutputGeometry.swift      # 출력 크기 연산의 유일한 정의(js/geometry.js가 미러)
│   ├── ProjectWorkspacePersistence.swift # 열린 생성 프로젝트 영속화
│   ├── SubtitleDocument.swift    # SRT/WebVTT 출력
│   └── WorkflowGraph.swift       # 에셋 계보와 분기 관계
├── GenImageRuntime/
│   ├── ZImageTextToImageService.swift
│   ├── QwenVLImageDescriptionService.swift
│   ├── Qwen2511ImageToImageService.swift
│   ├── LTXVideoGenerationService.swift
│   ├── MusicGenerationRouter.swift
│   ├── ACEStepMusicGenerationService.swift
│   ├── MiniMaxMusic3GenerationService.swift
│   ├── MediaCompatibilityService.swift # 내장 FFmpeg/ffprobe 검색, 탐색, 변환
│   ├── MediaSourceCompatibilityService.swift # 입력 미디어 직접 재생, 재다중화, 재생 프록시
│   ├── MediaCompositionService.swift # 이미지 루프 비디오와 미디어 병합
│   ├── MediaAudioPreparer.swift
│   ├── WhisperSubtitleTranscriber.swift
│   ├── ParaformerChineseSubtitleTranscriber.swift
│   ├── ParakeetJapaneseSubtitleTranscriber.swift
│   ├── SubtitleGenerationRouter.swift
│   ├── QwenTextGenerationService.swift
│   ├── SubprocessRuntime.swift   # 외부 Runtime 서브프로세스 공통 처리
│   ├── AudioOutputEncoder.swift
│   └── CoreMLUpscaleService.swift
├── GenImageApp/
    ├── AppStore.swift            # 타입 선언, 저장 속성, init
    ├── AppStore+Credentials.swift # 설정 자격 증명 작업
    ├── AppStore+SubtitleGeneration.swift
    ├── AppStore+MediaImport.swift
    ├── AppStore+MediaComposition.swift
    ├── AppStore+Workspaces.swift
    ├── AppStore+*.swift          # 그 밖의 책임별 AppStore 동작
    ├── HybridBridgeController.swift
    ├── LocalMCPServiceController.swift # 앱 내 MCP HTTP 스위치와 상태
    ├── HybridWebView.swift
    ├── AssetSchemeHandler.swift  # 로컬 이미지, 비디오, 오디오를 WebUI에 안전하게 제공
    └── Resources/WebUI/
        ├── js/automatic-flow.js  # 템플릿, Profile 사전 검사, 작업 공간 계획
        └── …                     # 나머지 HTML/CSS/JavaScript 프런트엔드
├── GenImageASRPoC/
    └── main.swift                # 독립 ASR 검증 도구
└── GenImageMCPServer/
    ├── MCPServer.swift           # 독립 stdio JSON-RPC server 코어
    └── MCPHTTPServer.swift       # 앱 스위치용 localhost HTTP transport
Patches/
├── MLX-Swift-LM-Qwen35-Text-Only.patch # Qwen3.5 텍스트 전용 입력 호환 수정
└── manifest.txt                  # 빌드 시 적용되는 의존 패치 목록
scripts/
├── apply-runtime-patches.command  # manifest에 따라 의존 패키지 수정 적용 및 검증
├── install-ltx-runtime.command     # LTX 비디오 Runtime 설치
├── install-minimax-music3-mlx-audio-runtime.command # 전용 MiniMax 4-bit Runtime 설치
└── build-ffmpeg-macos.sh          # 번들 및 재링크 가능한 LGPL FFmpeg 빌드
```

## 현재 상태

앱은 Z-Image Turbo 텍스트→이미지, Qwen3-VL/Qwen3.5/Qwen3.8 멀티모달 이미지→텍스트 및 텍스트→텍스트, Qwen 2511 이미지→이미지, LTX-2.3 MLX 텍스트→비디오 및 이미지→비디오, ACE-Step 1.5 Turbo MLX와 MiniMax Music 3 MLX 8-bit/4-bit 텍스트→음악, Whisper/Paraformer/Parakeet 자막 생성, Core ML Real-ESRGAN 업스케일의 로컬 추론에 연결되어 있습니다. 비디오는 MP4, 음악은 MP3/M4A/AAC/FLAC, 자막은 SRT/WebVTT로 추가되며 언어, 타임라인, 프로필 스냅샷, 계보를 보존합니다.

추가 정보:

- [업데이트 노트](UpdateNote.md)
- [아키텍처](docs/ARCHITECTURE.ko.md)
- [Web Bridge](docs/WEB_BRIDGE.ko.md)
- [로드맵](docs/ROADMAP.ko.md)
- [MCP 인터페이스](docs/MCP.ko.md)
- [ASR 자막 PoC](docs/ASR_POC.ko.md)
- [로컬 모델 테스트 보고서](docs/MODEL_TEST_REPORT.ko.md)

## 라이선스

이 프로젝트는 GPLv3와 상용 라이선스의 이중 라이선스 방식을 사용합니다.

- 오픈 소스 사용은 [GNU General Public License v3.0](LICENSE)에 따라 허가됩니다.
- 비공개 소스 통합, 독점 제품 배포, 맞춤형 상용 조건 등 GPLv3를 준수할 수 없거나 준수하지 않으려는 경우 저작권자에게 별도의 상용 라이선스를 요청하세요.
- 내장 FFmpeg와 LAME은 각각의 LGPL 조건을 유지합니다. 라이선스 본문, 정확한 소스 버전, 빌드 정보는 App의 `Contents/Resources/Licenses/`에 포함됩니다.

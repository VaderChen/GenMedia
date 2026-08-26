# GenImage MCP Server

[繁體中文](MCP.md) | [English](MCP.en.md) | [日本語](MCP.ja.md) | 한국어

GenImage는 동일한 MCP 도구 코어를 프로토콜 버전 `2025-06-18`의 두 JSON-RPC 2.0 transport로 제공합니다. 하나는 독립 stdio server이고 다른 하나는 GenMedia.app 설정 스위치로 제어하는 localhost HTTP API입니다.

## 독립 stdio

```bash
./build.command
MCP_BIN_DIR="$(swift build -c release --show-bin-path)"
"$MCP_BIN_DIR/GenImageMCP"
```

정식 연동에서는 `build.command`가 출력한 절대 경로를 사용하세요. `swift build`만 실행하면 metallib이 복사되지 않습니다.

`swift build -c release --show-bin-path`를 실행해 현재 도구 체인이 사용하는 Release 디렉터리를 확인하세요. 이전 SwiftPM의 `.build/arm64-apple-macosx/release` 경로를 하드코딩하지 마세요.

stdio server는 자체 `InferenceServices`를 보유하고 `GenImageMCP` 프로세스 안에서 추론을 완료하므로 GenMedia.app이 실행 중일 필요가 없습니다.

## App HTTP API

GenMedia.app에서 설정 → MCP 연동 스위치를 켜면 다음 URL이 표시됩니다.

```text
http://127.0.0.1:12181/mcp
```

이 엔드포인트는 localhost에만 바인딩되고 HTTP `POST` JSON-RPC를 받습니다. 스위치를 끄면 즉시 수신을 중단합니다. HTTP transport는 stdio와 같은 `MCPServer` 도구 코어를 직접 호출하며 stdio 자식 프로세스를 시작하거나 다른 서비스로 프록시하지 않습니다. 앱 내장 모드이므로 사용할 때는 GenMedia.app을 계속 실행해야 하며 headless 연동에는 독립 stdio 실행 파일을 사용하세요.

## MCP 메서드

- `initialize`
- `notifications/initialized`
- `ping`
- `tools/list`
- `tools/call`

stdio 메시지는 한 줄에 하나의 UTF-8 JSON-RPC 객체입니다. stdout에는 프로토콜 메시지만 출력하고 진단 정보는 stderr에 기록해야 합니다.

## 도구

- `genimage_models_list`
- `genimage_profiles_list`
- `genimage_upscale_image`
- `genimage_generate_image`
- `genimage_edit_image`
- `genimage_describe_image`
- `genimage_generate_subtitle`

텍스트→이미지, 이미지 편집, 이미지 설명, 선택적 자막 번역은 네이티브 MLX Swift Runtime을 사용하고 자막 인식은 네이티브 Core ML Runtime을 사용합니다. Release 실행 파일 옆에 `mlx.metallib`이 있어야 하며 `build.command`가 자동으로 처리합니다.

모델 루트는 `model_root` 또는 `GENIMAGE_MODEL_ROOT`로 지정할 수 있습니다.

### `genimage_generate_subtitle`

필수 인수는 원본 절대 경로 `input_path`, ASR `model_path`, `model_id`입니다. 지원 ASR 모델 ID:

- `argmaxinc/whisperkit-coreml@large-v3-turbo`: 다국어 Whisper
- `FluidInference/paraformer-large-zh-coreml`: 중국어 Paraformer
- `FluidInference/parakeet-0.6b-ja-coreml`: 일본어 Parakeet

`format`은 `srt` 또는 `vtt`이며 기본값은 `srt`입니다. `language_code`는 원본 언어 또는 `auto`를 지정합니다. `output_path`는 아직 존재하지 않고 형식과 확장자가 일치하는 절대 경로를 사용할 수 있습니다. `target_language_code`를 제공하면 `translation_model_path`와 `translation_model_id`도 필수이며 로컬 Qwen 텍스트 모델이 타임라인을 유지한 채 `zh-Hant`, `zh-Hans`, `en`, `ja`, `ko`로 번역합니다.

이 도구는 `GenImageMCP` 프로세스 안에서 `SubtitleGenerationRouter`와 추론 서비스를 직접 만듭니다. GenMedia.app에 연결하지 않으며 앱 실행도 필요하지 않습니다.

## 설정 예시

컴파일된 실행 파일을 사용하면 MCP Client마다 다른 작업 디렉터리 문제를 피할 수 있습니다.

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

MCP Client는 `build.command`가 생성한 Release 실행 파일을 사용하여 MLX Runtime과 실행 파일이 같은 디렉터리에 있도록 해야 합니다.

## 검증된 동작

- `initialize`와 7개 도구를 반환하는 `tools/list` 스모크 테스트를 통과했습니다.
- App HTTP transport는 localhost의 `tools/list` 요청에 stdio와 같은 7개 도구로 응답합니다.
- 알 수 없는 메서드는 JSON-RPC `-32601`을 반환합니다.
- `genimage_generate_image`는 로컬 Z-Image Turbo Q4를 사용한 256×256, 1-step 엔드투엔드 생성을 완료했습니다.
- `genimage_describe_image`는 로컬 Qwen3-VL 4-bit를 사용한 번체 중국어 이미지 설명을 완료했습니다.
- `genimage_upscale_image`는 로컬 Real-ESRGAN Core ML 모델을 사용한 4배 확대를 완료했습니다.
- 모델 내부 Logger는 stdout에 기록하지 않으므로 stdio는 순수 JSON-RPC를 유지합니다.

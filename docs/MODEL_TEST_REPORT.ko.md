# 로컬 모델 테스트 보고서

[繁體中文](MODEL_TEST_REPORT.md) | [English](MODEL_TEST_REPORT.en.md) | [日本語](MODEL_TEST_REPORT.ja.md) | 한국어

테스트 날짜: 2026-08-03
테스트 환경: Apple M4, macOS, Xcode 26.6, Swift 6.3

모델 루트: 로컬 테스트 디렉터리, 버전 관리에 포함되지 않음

## 텍스트→이미지

모델: `z-image-turbo-q4`

- 크기: 약 7.8 GB.
- Diffusers `ZImagePipeline` 디렉터리 구조가 완전합니다.
- Transformer와 text encoder는 4-bit affine, group size 32를 사용합니다.
- `Z-Image.swift` commit `28bfcf3148c041a554629247170eb54d9ac46830`을 성공적으로 불러왔습니다.
- 256×256, 1-step, Seed 1 테스트를 통과했습니다.
- 256×256 RGBA PNG를 생성했습니다.
- 모델 로드, Prompt encoding, denoising, VAE decoding, 파일 저장을 모두 완료했습니다.
- MCP `genimage_generate_image`는 256×256, 1-step, Seed 7 엔드투엔드 테스트를 통과했으며 8-bit RGBA PNG를 출력했습니다.
- 앱의 `ZImageTextToImageService`는 네이티브 `ZImagePipeline`을 직접 감싸며 더 이상 외부 CLI를 호출하지 않습니다.

SwiftPM 실행에는 `mlx.metallib`이 필요합니다. 호환 파일은 `RuntimeSupport`에 저장되어 있으며 `build.command`가 Release 실행 파일 옆으로 복사하므로 런타임에 다른 앱에 의존하지 않습니다.

## 업스케일

모델:

- `upscale/realesrgan512.mlmodel`
- `upscale/realesrganAnime512.mlmodel`

두 모델 모두 `coremlcompiler compile`을 통과했습니다.

- 입력: RGB image, 512×512.
- 출력: RGB image, 2048×2048.
- 확대 배율: 4배.

## 이미지→텍스트

모델: `Qwen3-VL-4B-Instruct-4bit`

- 모델 아키텍처: `Qwen3VLForConditionalGeneration`.
- `model_type`: `qwen3_vl`.
- 양자화: 4-bit affine, group size 64.
- 모델, tokenizer, chat template, preprocessor 설정이 완전합니다.

`mlx-swift-lm` revision `7da33441c7c08b010ff1aa8da9dc3d82277272f5`의 Qwen 멀티모달 Runtime을 연동했습니다.

- 로컬 4-bit 모델을 성공적으로 불러왔습니다.
- Z-Image가 생성한 PNG를 입력으로 바로 사용할 수 있습니다.
- 96-token 번체 중국어 이미지 설명 테스트를 통과했습니다.
- 앱은 설명을 텍스트→이미지 Prompt에 넣어 수정하거나 바로 생성할 수 있게 합니다.

## MCP 검증

표준 JSON-RPC 2.0 stdio MCP server에서 다음 스모크 테스트와 엔드투엔드 테스트를 완료했습니다.

- `initialize`는 프로토콜 버전 `2025-06-18`을 반환합니다.
- `tools/list`는 모델, 프로필, 업스케일, 텍스트→이미지, 이미지 편집, 이미지→텍스트, 자막 생성의 7개 도구를 표시합니다.
- `genimage_generate_image`는 네이티브 Z-Image Runtime을 직접 호출하고 출력 PNG의 크기와 형식을 확인했습니다.
- `genimage_describe_image`는 네이티브 Qwen3-VL Runtime을 직접 호출하고 번체 중국어 설명을 출력했습니다.
- `genimage_upscale_image`는 로컬 Real-ESRGAN Core ML 모델로 4배 출력을 생성했습니다.
- stdout에는 줄 단위 JSON-RPC 응답만 포함되며 모델 Logger는 포함되지 않습니다.

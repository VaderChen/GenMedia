# 개발 로드맵

[繁體中文](ROADMAP.md) | [English](ROADMAP.en.md) | [日本語](ROADMAP.ja.md) | 한국어

## 완료: 핵심 기능

- Swift Package, macOS 14 이상 App, Apple Silicon 네이티브 MLX/Core ML 추론, 하이브리드 Web UI.
- 텍스트→이미지, 이미지→텍스트, 이미지→이미지, 텍스트→비디오, 이미지→비디오, 텍스트→음악, 4배 업스케일 독립 서비스와 에셋 lineage.
- Z-Image Turbo, Qwen3-VL, Qwen 2511, LTX-2.3, ACE-Step 1.5, MiniMax Music 3, Real-ESRGAN 프로필.
- 음악 Prompt, 선택적 가사, 일반적인 음악 스타일, 5~300초 설정, MP3/M4A/AAC/FLAC 출력.
- 모델 센터 다운로드, 일시 정지, 이어받기, 디스크 사전 검사, 복구, 삭제, 프로필 종속성 검사, 설치 후 자동 정렬.
- 작업 대기열, 취소, 진행률, 예상 남은 시간, 생성 시간, 모델 캐시, 수동 메모리 해제.
- 각 작업 공간 탭을 영구 생성 프로젝트로 취급하고 재실행 후 에셋, 작업, 선택 상태, 프로필 스냅샷 복원.
- 생성 패널 부분 렌더링, 커서와 IME 보호, 오디오/비디오 연속 재생.
- `Image-YYYYMMDD-HHmm`, `Video-YYYYMMDD-HHmm`, `Music-YYYYMMDD-HHmm` 출력 이름과 같은 분 충돌 방지 일련번호.
- Release App bundle, MLX metallib, 네이티브 MCP 추론 도구, 독립 DMG 서명/공증 흐름.

## 현재 단계: 안정화와 검증

1. 16 GB, 24 GB, 32 GB 환경에서 ACE-Step 장시간 오디오의 메모리, 발열, 취소, 복구 테스트를 완료합니다.
2. 다운로드 중단, 해시 불일치, 디스크 부족, 재시작 복구 테스트 범위를 확장합니다.
3. 앱 비정상 종료, 에셋 누락, 인덱스 복구 상황에서 다중 탭 프로젝트 일관성을 검증합니다.
4. Runtime, 모델, LoRA의 라이선스 화면과 배포 manifest를 완성합니다.

## 이후

1. 프로필 가져오기/내보내기, 버전 마이그레이션, 호환성 검사를 추가합니다.
2. Apple Silicon 네이티브 MLX 미디어 생성 엔진을 확장합니다.
3. MCP에 비디오, 음악, 프로젝트 작업 흐름 도구를 추가합니다.
4. 재현 가능한 모델 및 Runtime 성능 벤치마크를 구축합니다.

# Roadmap

[繁體中文](ROADMAP.md) | English | [日本語](ROADMAP.ja.md) | [한국어](ROADMAP.ko.md)

## Completed: Core Capabilities

- Swift Package, macOS 14+ app, Apple Silicon-native MLX/Core ML inference, and a hybrid Web UI.
- Independent text-to-image, image-to-text, image-to-image, text-to-video, image-to-video, text-to-music, and 4× upscale services with asset lineage.
- Profiles for Z-Image Turbo, Qwen3-VL, Qwen 2511, LTX-2.3, ACE-Step 1.5, MiniMax Music 3, and Real-ESRGAN.
- Music Prompt, optional lyrics, common music styles, 5–300 second settings, and MP3/M4A/AAC/FLAC output.
- Model Center download, pause, resume, disk preflight, repair, removal, profile dependency checks, and automatic post-install sorting.
- Job queue, cancellation, progress, estimated remaining time, generation time, model caching, and manual memory release.
- Each workspace tab acts as a persistent generation project and restores assets, operations, selection, and profile snapshots after relaunch.
- Regional creation-panel rendering, caret and IME protection, and uninterrupted audio/video playback.
- `Image-YYYYMMDD-HHmm`, `Video-YYYYMMDD-HHmm`, and `Music-YYYYMMDD-HHmm` output naming with same-minute collision suffixes.
- Release app bundle, MLX metallib, native MCP inference tools, and a separate DMG signing/notarization workflow.

## Current Phase: Stability and Validation

1. Complete ACE-Step long-audio memory, thermal, cancellation, and recovery tests on 16 GB, 24 GB, and 32 GB systems.
2. Expand interrupted download, hash mismatch, insufficient disk, and restart recovery coverage.
3. Validate multi-tab project consistency after abnormal app termination, missing assets, and index repair.
4. Complete license pages and distribution manifests for runtimes, models, and LoRAs.

## Later

1. Add profile import/export, version migration, and compatibility checks.
2. Expand Apple Silicon-native MLX media-generation engines.
3. Extend MCP with video, music, and project-workflow tools.
4. Establish reproducible model and runtime performance benchmarks.

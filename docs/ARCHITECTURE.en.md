# GenMedia Architecture

[繁體中文](ARCHITECTURE.md) | English | [日本語](ARCHITECTURE.ja.md) | [한국어](ARCHITECTURE.ko.md)

## Design Goals

1. Text-to-image, image-to-text, image-to-image, video generation, music generation, subtitle generation, and upscaling are independent capabilities with no mutual dependency.
2. Image, video, audio, and subtitle outputs can form workflows and branches.
3. The UI does not directly depend on MLX, Core ML, or any specific model.
4. Model versions and inference architectures are switched through profiles when models are updated.
5. Existing works preserve profile snapshots and are unaffected by later profile changes.

## Layers

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

The Web UI can access local capabilities only through the bridge. It cannot directly read arbitrary files, model directories, or system APIs.

### Web UI Update Strategy

- While a Prompt, negative Prompt, or lyrics field is focused, incoming Swift state preserves local edits and defers unnecessary full rendering so the caret, selection, and IME composition are not reset.
- Generation type and Prompt, Lyrics, and output-setting tabs use an independent creation-panel renderer and do not replace the preview, players, Inspector, or sidebar DOM.
- Workspace-tab schema v3 stores each tab's task type, Profile reference, Prompt, output settings, and automatic-flow step. Tab activation restores the native creation state through one Bridge command.
- When global state genuinely requires a full render, playing `<audio>`, `<video>`, and audio-visualizer nodes are detached and reattached at the matching asset position, preserving playback progress and the Web Audio connection.

## Internal Structure Refactor

This refactor changes code boundaries and ownership only; existing generation capabilities, user flows, and the Web Bridge protocol remain unchanged:

- `ApplicationSupport` is the single definition for the Application Support paths of `Models`, `Runtime`, `Workspace`, `Pasted`, and `Generated`, and adopts legacy workspace data from `GenMedia` at launch.
- `OutputGeometry` owns dimension limits, 16-pixel alignment, aspect-ratio conversion, and image-to-image generation plans. Web UI `js/geometry.js` mirrors it so Native, MCP, and UI calculations stay consistent.
- `AppStore` now keeps only type declarations, stored properties, and initialization. Persistence, Paths, Selection, Profiles, OutputSettings, Assets, ImageGeneration, MediaGeneration, Jobs, and ModelInstallation are split into `AppStore+*.swift` extensions by responsibility.
- The Web UI separates sidebar and routing, workspace tabs, geometry, and full-render preservation into `chrome.js`, `workspace-tabs.js`, `geometry.js`, and `render-preservation.js`; `app.js` retains bridge and application coordination.
- `SubprocessRuntime` centralizes environment setup, logs, progress, stall detection, cancellation, and termination semantics for external workers, video and music CLIs, and FFmpeg.
- `MediaCompatibilityService` is the sole lookup and probing entry point for `ffmpeg` and `ffprobe`. Production app bundles prefer `Contents/Resources/bin/`; development runs may fall back to environment overrides, Homebrew, and `PATH`.
- `MediaSourceCompatibilityService` classifies every imported video or audio file through `ffprobe`. Directly playable sources stay untouched, H.264/HEVC is remuxed when possible, and only remaining cases receive a VideoToolbox H.264/AAC or M4A AAC playback proxy. `AppStore+MediaImport` wraps imports and conversions in cancellable jobs with FFmpeg progress, while transcode bitrate scales with image area.
- `AssetSchemeHandler` responds to HTTP Range for timed media and uses background `FileHandle` reads in 512 KiB chunks. A stopped-task set prevents callbacks after WebKit cancels a request; images remain one-shot responses. At startup, `ApplicationSupport.orphanMediaCacheFiles` finds and removes UUID cache files without matching assets.
- `MediaCompositionService` implements model-free image-loop video and media-merge jobs using the shared FFmpeg subprocess, progress, cancellation, and output naming. Results re-enter the existing lineage as `generatedVideo` assets and `WorkflowOperation` records.
- `automatic-flow.js` preflights Profiles from declarative templates and creates the workspace and tabs. Steps link source tabs and asset IDs; the first Simple MV flow contains key visuals, background music, image looping, and media merging.
- Dependency source patches are described by `Patches/manifest.txt` and applied and verified by `scripts/apply-runtime-patches.command`. A pin mismatch or failed patch stops the build instead of silently continuing with unpatched sources.
- `GenImageMCP` remains a standalone stdio server that owns its `InferenceServices`. The app's `LocalMCPServiceController` manages only the localhost HTTP transport lifecycle; both call the same `MCPServer` tool core directly, with no stdio-to-HTTP proxy layer.
- Production ACE-Step stages no longer use PoC names. The diagnostic-only DiT probe is kept separate from the production generation stages so experimental code is not confused with the application path.

## Profiles

`InferenceProfile` contains:

- Capability type.
- Model ID.
- Model revision.
- Inference architecture: MLX Swift, Core ML, local service, or external CLI.
- Capability defaults.
- Profile revision.

When a job runs, `WorkflowOperation.profileSnapshot` stores the complete value instead of only the profile ID.

Built-in profiles are not modified directly. Users duplicate a profile before saving changes as a new revision.

## Assets and Workflows

`MediaAsset.parentAssetID` identifies the source media. Assets without a parent are root nodes of independent jobs:

- Standalone text-to-image: the generated image has no parent.
- Standalone image-to-text: a root image is imported first, and the description output is written to the recipe.
- Standalone upscale: a root image is imported first, and the upscaled result uses the original image as its parent.
- Chained generation: the generated result uses the selected image as its parent.
- Standalone text-to-video: the MP4 asset has no parent.
- Image-to-video: the MP4 asset uses the source image as its parent.
- Standalone text-to-music: the MP3, M4A, AAC, or FLAC asset has no parent and records its actual duration, sample rate, and channel count.
- Subtitle generation: the source is imported as `importedVideo` or `importedAudio`; the SRT/WebVTT result is stored as `generatedSubtitle` with the source media as its parent.

`WorkflowGraph` provides lineage and children queries, so the UI does not need to infer asset relationships.

Open workspace tabs define generation-project lifetimes. Swift atomically stores `Project`, `MediaAsset`, `WorkflowOperation`, and selection state as a JSON snapshot under Application Support; ordinary app termination does not clear it. The Web UI keeps tab metadata in WebKit localStorage. Closing a tab notifies the native layer through the bridge to remove that tab's asset and lineage indexes without deleting exported media files.

Named workspaces sit above tabs, and each workspace owns its own tab collection. Create and delete commands enter `AppStore+Workspaces` through the bridge, with confirmation required before deletion. Switching workspaces changes only the corresponding tabs and selection state; it does not rebuild runtimes or media players.

## Inference Runtimes

Text-to-image uses `Z-Image.swift` commit `28bfcf3148c041a554629247170eb54d9ac46830`:

- macOS 14+ and Swift 6.
- `ZImageGenerationRequest` supports prompts, negative prompts, dimensions, steps, seeds, models, and runtime options.
- `ZImageTextToImageService` wraps `ZImagePipeline.generate` and reports progress by stage.
- The denoising loop checks Swift task cancellation.
- Model unloading, LoRA unloading, cancellation, and memory cache cleanup are supported.

The multimodal image-to-text and text-to-text paths use `mlx-swift-lm` revision `7da33441c7c08b010ff1aa8da9dc3d82277272f5`:

- `QwenVLImageDescriptionService` loads local Qwen3-VL models through `VLMModelFactory`.
- `QwenTextGenerationService` uses text-only input through the same multimodal container. Qwen3-VL, Qwen3.5, and Qwen3.8 descriptors declare both `.imageToText` and `.textToText`, with a separate profile for each capability.
- `Patches/MLX-Swift-LM-Qwen35-Text-Only.patch` applies the upstream compatibility fix for Qwen3.5 text-only input.
- Managed downloads fetch and validate `processor_config.json`, image/video preprocessing configuration, tokenizer data, chat templates, weights, and weight indexes so a multimodal model cannot be marked installed with text weights alone.
- The model container is cached for the lifetime of the service to avoid reloading the same profile.
- Output prompts support Traditional Chinese, English, Japanese, and Korean.

Upscaling is handled by `CoreMLUpscaleService` with Real-ESRGAN 512 tiles and 4× stitching.

Video generation is handled by `LTXVideoGenerationService`, which launches the bundled `GenImageLTXVideoWorker` Swift subprocess:

- Text-to-video and image-to-video share `VideoGenerating` and `VideoGenerationRequest`.
- Swift validates the profile, model path, dimensions, frame count, FPS, and output count.
- LTX-2.3 additionally requires frame counts in the form `8n+1`.
- JSON requests and stage-specific progress events run through the existing `RuntimeProcess`, which supports task cancellation, log-based error reporting, and percentage progress extraction.
- LTX LoRA control videos are created through the shared FFmpeg layer with VideoToolbox H.264, without GPL `libx264`.
- MP4 outputs are added to the workspace as `generatedVideo` assets and played with the native Web UI `<video>` element.
- The Worker is copied into the app bundle's `Contents/Helpers`; development builds may override it with `GENIMAGE_LTX_WORKER`, while production does not depend on an external runtime.

Music generation is dispatched by `MusicGenerationRouter` through `MusicRuntimeAdapter.supports`, keeping model IDs out of centralized router logic:

- Text-to-music uses `MusicGenerating`, `MusicGenerationRequest`, and `MusicGenerationOptions`.
- `ACEStepMusicGenerationService` uses a `.mlxSwift` profile and calls `ACEStepSwiftRuntime` directly for Qwen3 embedding, condition encoding, Turbo DiT, Euler sampling, and Oobleck VAE decoding, without starting an external service or process.
- ACE-Step supports 10–300 seconds, 1–20 steps, optional lyrics, and instrumental generation. Latent length is derived from the VAE sample rate and hop length; long audio uses overlap-discard tiled decoding and streamed PCM writing to limit peak memory. Its code and model use the MIT License.
- Music profiles expose duration bounds and target/maximum semantics through `ProfileMusicConfiguration`, so the Web UI does not branch on model IDs.
- `MiniMaxMusic3GenerationService` uses an `.externalCLI` profile to launch the bundled `GenImageMiniMaxMusic3Worker`. The app writes a JSON request, and the Worker executes the Swift/MLX pipeline with the fixed bfloat16 production path. The 5–300 second parameter is a maximum duration, and the model may stop naturally when it emits the audio end token.
- The 8-bit and `mlx-community/MiniMax-Music3-4bit` checkpoints share the same Worker. It reports autoregressive progress per frame, denoising progress per chunk×step, and vocoder progress per chunk. The app retains `RuntimeProcess` cancellation and forced termination, then uses bundled FFmpeg to convert the completed WAV. MiniMax Music 3 no longer requires a Python runtime.
- `Mothersuperior/minimax-music3-composer-5.7b-distilled` is managed by Model Center as a music component, with the `lr-6e-5` weights selected by default. It is not an independent profile and is never sent to the music service by itself; it requires a compatible Composer-override runtime before it can be applied.
- Both adapters obtain a temporary WAV and use `AudioOutputEncoder` through bundled FFmpeg to produce MP3 at 320 kbps, M4A AAC at 256 kbps, ADTS AAC at 256 kbps, or lossless FLAC.
- Temporary files are cleaned after success, failure, or cancellation. Only completed compressed audio is retained as a `generatedAudio` asset with actual duration, sample rate, and channel count.
- ACE-Step weights are managed by Model Center, while the native runtime is compiled into the app and uses no separate installation path or service environment variables.

Media assets retain both `fileURL` and an optional `playbackURL`. The former always points to the user's original file for subtitle output, lineage, and explicit deletion; the latter points only to a compatibility proxy under `Application Support/GenImage/MediaCache` for WebKit preview. Removing an asset or closing a project cleans the proxy without rewriting the original media.

Subtitle generation sits between media import and text generation:

1. `MediaAudioPreparer` verifies the audio track with bundled `ffprobe`, then uses bundled `ffmpeg` to normalize it into a 16 kHz mono PCM WAV and temporary paths.
2. `SubtitleGenerationRouter` evaluates `MediaTranscribing.supports(profile:)` in order and selects the first matching adapter.
3. Whisper Large v3 Turbo handles multilingual ASR, Paraformer Large handles Chinese, and Parakeet 0.6B handles Japanese; all three run through local Core ML paths.
4. Optional `QwenTextGenerationService` translation uses Qwen3.5/Qwen3.8 MLX in batches without changing segment start or end times.
5. `SubtitleDocument` renders SRT/WebVTT and stores the result in the current workspace as a `generatedSubtitle` asset.

`GenImageASRPoC` is a standalone validation executable for WhisperKit media decoding, language recognition, and timestamp output. It does not write to app workspaces and is not an alternate production path.

`scripts/build-ffmpeg-macos.sh` creates an Apple Silicon, LGPL-only, dynamically linked, replaceable FFmpeg distribution. `build.command` rejects GPL/nonfree encoders, copies `ffmpeg`, `ffprobe`, dylibs, and license files, rewrites install names to `@rpath`, and signs dylibs, tools, and the app in order before the existing DMG notarization workflow. Developer ID testing confirms that FFmpeg dylibs and tools do not need a separate library-validation entitlement when they use the same Team ID; local ad-hoc builds therefore leave FFmpeg without hardened runtime. Prebuilt binaries live under Git-ignored `third_party/ffmpeg/` and are excluded from GitHub source archives.

The MLX metallib is copied from `RuntimeSupport/mlx.metallib` into the release executable directory by `build.command`. Model license review and 16/24/32 GB stress testing remain release requirements.

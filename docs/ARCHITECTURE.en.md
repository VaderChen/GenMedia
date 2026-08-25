# GenMedia Architecture

[繁體中文](ARCHITECTURE.md) | English | [日本語](ARCHITECTURE.ja.md) | [한국어](ARCHITECTURE.ko.md)

## Design Goals

1. Text-to-image, image-to-text, image-to-image, video generation, music generation, and upscaling are independent capabilities with no mutual dependency.
2. Image, video, and audio outputs can form workflows and branches.
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
        └── ImageUpscaling
                    │
                    ▼
       MLX Swift / Core ML / Local REST Service / External CLI
```

The Web UI can access local capabilities only through the bridge. It cannot directly read arbitrary files, model directories, or system APIs.

### Web UI Update Strategy

- While a Prompt, negative Prompt, or lyrics field is focused, incoming Swift state preserves local edits and defers unnecessary full rendering so the caret, selection, and IME composition are not reset.
- Generation type and Prompt, Lyrics, and output-setting tabs use an independent creation-panel renderer and do not replace the preview, players, Inspector, or sidebar DOM.
- When global state genuinely requires a full render, playing `<audio>`, `<video>`, and audio-visualizer nodes are detached and reattached at the matching asset position, preserving playback progress and the Web Audio connection.

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

`ImageAsset.parentAssetID` identifies the source asset. Assets without a parent are root nodes of independent jobs:

- Standalone text-to-image: the generated image has no parent.
- Standalone image-to-text: a root image is imported first, and the description output is written to the recipe.
- Standalone upscale: a root image is imported first, and the upscaled result uses the original image as its parent.
- Chained generation: the generated result uses the selected image as its parent.
- Standalone text-to-video: the MP4 asset has no parent.
- Image-to-video: the MP4 asset uses the source image as its parent.
- Standalone text-to-music: the MP3, M4A, AAC, or FLAC asset has no parent and records its actual duration, sample rate, and channel count.

`WorkflowGraph` provides lineage and children queries, so the UI does not need to infer asset relationships.

Open workspace tabs define generation-project lifetimes. Swift atomically stores `Project`, `ImageAsset`, `WorkflowOperation`, and selection state as a JSON snapshot under Application Support; ordinary app termination does not clear it. The Web UI keeps tab metadata in WebKit localStorage. Closing a tab notifies the native layer through the bridge to remove that tab's asset and lineage indexes without deleting exported media files.

## Inference Runtimes

Text-to-image uses `Z-Image.swift` commit `28bfcf3148c041a554629247170eb54d9ac46830`:

- macOS 14+ and Swift 6.
- `ZImageGenerationRequest` supports prompts, negative prompts, dimensions, steps, seeds, models, and runtime options.
- `ZImageTextToImageService` wraps `ZImagePipeline.generate` and reports progress by stage.
- The denoising loop checks Swift task cancellation.
- Model unloading, LoRA unloading, cancellation, and memory cache cleanup are supported.

Image-to-text uses `mlx-swift-lm 2.30.6`:

- `QwenVLImageDescriptionService` loads local Qwen3-VL models through `VLMModelFactory`.
- The model container is cached for the lifetime of the service to avoid reloading the same profile.
- Output prompts support Traditional Chinese, English, Japanese, and Korean.

Upscaling is handled by `CoreMLUpscaleService` with Real-ESRGAN 512 tiles and 4× stitching.

Video generation is handled by `LTXVideoGenerationService`, which invokes `ltx-2-mlx generate`:

- Text-to-video and image-to-video share `VideoGenerating` and `VideoGenerationRequest`.
- Swift validates the profile, model path, dimensions, frame count, FPS, and output count.
- LTX-2.3 additionally requires frame counts in the form `8n+1`.
- The external process supports task cancellation, log-based error reporting, and percentage progress extraction.
- MP4 outputs are added to the workspace as `generatedVideo` assets and played with the native Web UI `<video>` element.
- The runtime can be replaced through `GENIMAGE_LTX_RUNTIME` or standard installation locations without coupling the Python implementation to the UI or `AppStore`.

Music generation is dispatched by `MusicGenerationRouter` through `MusicRuntimeAdapter.supports`, keeping model IDs out of centralized router logic:

- Text-to-music uses `MusicGenerating`, `MusicGenerationRequest`, and `MusicGenerationOptions`.
- `ACEStepMusicGenerationService` uses a `.mlxSwift` profile and calls `ACEStepSwiftRuntime` directly for Qwen3 embedding, condition encoding, Turbo DiT, Euler sampling, and Oobleck VAE decoding, without starting an external service or process.
- ACE-Step supports 10–300 seconds, 1–20 steps, optional lyrics, and instrumental generation. Latent length is derived from the VAE sample rate and hop length; long audio uses overlap-discard tiled decoding and streamed PCM writing to limit peak memory. Its code and model use the MIT License.
- Music profiles expose duration bounds and target/maximum semantics through `ProfileMusicConfiguration`, so the Web UI does not branch on model IDs.
- `MiniMaxMusic3GenerationService` uses an `.externalCLI` profile to invoke `mlx-minimax-music3 generate`. Its 5–300 second parameter is a maximum duration, and the model may stop naturally when it emits the audio end token. Its model retains its separate Community License.
- Both adapters obtain a temporary WAV and use `AudioOutputEncoder` to produce MP3 at 320 kbps, M4A AAC at 256 kbps, ADTS AAC at 256 kbps, or lossless FLAC.
- Temporary files are cleaned after success, failure, or cancellation. Only completed compressed audio is retained as a `generatedAudio` asset with actual duration, sample rate, and channel count.
- ACE-Step weights are managed by Model Center, while the native runtime is compiled into the app and uses no separate installation path or service environment variables.

The MLX metallib is copied from `RuntimeSupport/mlx.metallib` into the release executable directory by `build.command`. Before distribution, model license review, 16/24/32 GB stress testing, app bundling, signing, and notarization still need to be completed.

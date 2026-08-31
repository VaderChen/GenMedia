# GenMedia

[繁體中文](README.md) | English | [日本語](README.ja.md) | [한국어](README.ko.md)

GenMedia is a local AI media generation app with **native Apple Silicon support**. The project provides a compilable hybrid application with the following capabilities:

- Swift handles models, profiles, job queues, files, and MLX/Core ML inference.
- `WKWebView` embeds the HTML, CSS, and JavaScript UI without requiring a network connection or npm runtime.
- Text-to-image, image-to-text, image-to-image, text-to-video, image-to-video, text-to-music, subtitle generation, and upscaling can run independently or be chained through asset lineage.
- Automatic flows create workspace tabs with independent settings. The Simple MV template prepares linked key-visual, background-music, image-loop, and media-merge steps.
- Every operation preserves a profile snapshot, making the model and architecture revision traceable after updates.
- A dedicated settings page supports Traditional Chinese, English, Japanese, Korean, and six persistent color themes.
- Settings provides a switch for a localhost-only MCP HTTP API, while a standalone JSON-RPC 2.0 stdio server remains available when the app is not running.

## Preview

![GenMedia intelligent media generation interface](images/cap001.jpg)

## Run

Requirements: macOS 14+, Apple Silicon, and Xcode 16+.

```bash
./build.command
./run.command
```

`build.command` creates the release executables and a standard `GenMedia.app` in `dist/`. The app contains WebUI resources, the MLX Metal runtime, the MCP server, model diagnostic tools, and shared LGPL `ffmpeg`/`ffprobe` binaries as its media compatibility layer. On the first app-bundle build, `build.command` automatically downloads the sources and prepares the bundled FFmpeg distribution; neither a manual FFmpeg preparation step nor `pkg-config` is required.

```bash
# Build the release executables and app
./build.command

# Incremental release build without creating the app bundle
./build.command --no-app

# Set the version and bundle identifier
GENIMAGE_VERSION=1.1.0 GENIMAGE_BUNDLE_ID=com.example.genimage ./build.command
```

`run.command` automatically uses `--no-app`, so normal development runs do not repeatedly create the app bundle. Release DMGs are handled by a separate local workflow with Developer ID Application signing, Apple notarization, stapling, and Gatekeeper verification.

### FFmpeg Build Troubleshooting

- The first `./build.command` run needs network access to download the FFmpeg and LAME sources. Later builds reuse the cached sources and `third_party/ffmpeg`; a missing or incomplete distribution is rebuilt automatically.
- Homebrew `pkg-config` is not required. The project supplies a fallback used only for LAME configuration and runs it from a space-free temporary path, so project paths containing spaces are supported.
- `._*` AppleDouble sidecars created by external filesystems such as ExFAT are removed before dylib processing so they cannot be mistaken for Mach-O files.
- If the build is interrupted or fails, the previous usable FFmpeg distribution is restored. Fix the network or Xcode issue and rerun `./build.command`. Set `GENMEDIA_FFMPEG_ROOT` to use a different output location.

### Video Runtime

Video generation runs in the bundled `GenImageLTXVideoWorker` Swift subprocess. The Swift app manages profiles, parameter validation, the job queue, cancellation, progress, assets, and video playback without an additional video runtime.

The model-center plan for `dgrauet/ltx-2.3-mlx-q4` also downloads the native MLX INT4 transformer, video/audio VAEs, vocoder, spatial upscaler, and the Gemma 3 12B text encoder from `google/gemma-3-12b-it-qat-q4_0-unquantized`; the complete download is about 42 GiB, and 48 GB or more of memory is recommended.

Development builds can use `GENIMAGE_LTX_WORKER` to select a custom Worker. Release apps use `Contents/Helpers/GenImageLTXVideoWorker`. `GENIMAGE_LTX_GEMMA_MODEL` can override the Gemma directory; when unset, the Worker first uses `gemma-3-12b` inside the LTX model directory.

The LTX Worker communicates with the app through a JSON request and stage-specific progress events; model files are not bundled in the app. Legacy runtime data is not removed automatically; after confirming it is no longer needed, the old directory `~/Library/Application Support/GenImage/Runtime/ltx-2-mlx/` can be removed manually.

### MiniMax H3 Runtime

The app now includes a native Swift/MLX video Runtime for MiniMax H3 GGUF, supporting text-to-video and selected single-image-to-video profiles. The H3 Worker, Qwen 3 VL text encoder, video/audio VAEs, and GGUF weights are managed by each profile's download plan and are not bundled in the app.

MiniMax H3 is currently in a **testing phase; correctness is not guaranteed for generated results or every H3 GGUF variant on every hardware configuration**. It should not be treated as a stable production feature. If an output is incorrect, retain the Profile, model variant, and Runtime log for follow-up fixes.

### Compatible Media Import

Imported video and audio are inspected by bundled `ffprobe` for container, codecs, tracks, duration, and display-oriented dimensions; long conversions run as cancellable jobs with progress. WebKit-playable sources remain untouched; H.264 or HEVC with only container or audio incompatibilities is remuxed losslessly to MP4; other video receives a VideoToolbox H.264/AAC playback proxy, while incompatible audio receives an M4A AAC proxy. Playback proxies are streamed in background chunks through `AssetSchemeHandler` with HTTP Range support for progressive playback and seeking. The original path remains the subtitle source, so sidecar subtitles still use the source directory and filename. Startup removes MediaCache orphans without matching assets, while closing a project or removing an asset cleans only app-managed compatibility cache files.

### Automatic Flows and Media Composition

Automatic flows use declarative steps to create a new workspace and its tabs. Every tab preserves its own task type, Profile assignment, Prompt, and image, video, music, or media-processing settings, so switching tabs or relaunching the app does not overwrite another draft. The first Simple MV template links text-to-image, text-to-music, image-loop video, and media merge.

Image-loop and media-merge tasks do not require an AI model. `MediaCompositionService` runs them through the bundled FFmpeg. Image loops support multiple images, per-image timing, total duration, resolution, frame rate, and Cover or Contain fitting. Media merge supports replacing or mixing the original audio, volume control, and duration policies. Automatic-flow dependencies pass asset IDs rather than guessing filenames.

### Subtitle Generation

The subtitle workflow accepts video or audio, uses the bundled `ffprobe` to inspect its tracks, and normalizes the selected audio through the bundled `ffmpeg` to 16 kHz mono PCM before `SubtitleGenerationRouter` selects a native Core ML ASR adapter. It preserves segment timing and exports SRT or WebVTT. The faster, lower-memory multilingual Whisper Small is recommended by default; Whisper Large v3 Turbo remains available when higher recognition quality is preferred, alongside Chinese Paraformer Large and Japanese Parakeet 0.6B. The source language can be detected automatically or selected by the profile.

Whisper Small and Large v3 Turbo can both be downloaded from Model Center using `argmaxinc/whisperkit-coreml@small` and `argmaxinc/whisperkit-coreml@large-v3-turbo`. Small requires about 216 MB of model data and 8 GB of memory; Large requires about 954 MB and 16 GB. Both use the same native WhisperKit/Core ML runtime.

After transcription, an optional local Qwen3.5 or Qwen3.8 MLX text model can translate the unchanged timeline into 25 target languages: Traditional Chinese, Simplified Chinese, English, Japanese, Korean, Spanish, French, German, Italian, Portuguese, Russian, Arabic, Hindi, Bengali, Indonesian, Vietnamese, Thai, Turkish, Polish, Dutch, Swedish, Czech, Ukrainian, Malay, and Filipino. The result is stored in the active workspace as a `generatedSubtitle` asset whose parent is the source video or audio asset.

`GenImageASRPoC` is a standalone WhisperKit validation utility for checking media decoding, language recognition, and timestamps without modifying the main app workspace. The production app follows the same Swift/Core ML boundary. See [ASR Subtitle PoC](docs/ASR_POC.en.md).

Qwen3-VL, Qwen3.5, and Qwen3.8 are multimodal models, so Model Center classifies each under both image-to-text and text-to-text and provides a profile for each capability. Managed downloads include `processor_config.json`, image/video preprocessing configuration, tokenizer data, chat templates, and the complete weight index; installation is reported complete only after required-file validation succeeds.

### Civitai LoRA

Model Center also includes several Civitai LoRAs based on `ZImageTurbo` (Asian Beauties, Turbo Lightning, Flat AnimeStyle, and Diorama). Paste an API token into the **Civitai LoRA** card in Settings and save it. The token is stored only in macOS Keychain and is never written to model manifests, workspaces, or logs. Model Center then downloads the file directly over HTTPS with an `Authorization: Bearer` header instead of opening the Civitai website for manual download.

For legacy headless launch scripts, `CIVITAI_TOKEN` remains supported as a compatibility fallback.

### Profiles, Jobs, and Memory

- Profiles are ordered as active, ready, downloading, and unavailable. A subtle green outline marks profiles whose model and LoRA dependencies are complete, and the list is reordered as soon as a download finishes.
- Cancellation enters `cancelling` first, then changes to `cancelled` and unlocks all generation and memory controls when the runtime task exits. A numeric ETA appears after 35% progress and 15 seconds; overall elapsed time is used as a fallback when stable samples are not yet available.
- The Z-Image MLX compatibility layer handles `quantize_config.json`, affine/mxfp4 modes, packed pad tokens, and FP16-to-BF16 loading. The andrevp Z-Image Turbo MLX 4-bit profile has been validated with a real generation run.
- Source patches for dependencies are listed in `Patches/manifest.txt` and applied by `scripts/apply-runtime-patches.command` after Swift Package resolution; `build.command` invokes it. A dependency pinned to a version the manifest was not written against, a missing patch file, a failed application, or a missing marker afterwards all abort the build rather than letting it continue against unpatched sources. Run `scripts/apply-runtime-patches.command --verify` to check without changing anything.
- Text-to-image completion keeps model weights and warm buffers resident. Reusable MLX buffers are trimmed after five idle minutes without unloading the model. Models are unloaded only by the sidebar Release Memory action, a model switch, or the over-90% RAM protection applied while switching profiles.
- Downloads retain their upstream filenames. Generated outputs use `Image-YYYYMMDD-HHmm`, `Video-YYYYMMDD-HHmm`, or `Music-YYYYMMDD-HHmm`; a numeric suffix prevents collisions within the same minute. The output directory is configurable in Settings.
- Each open workspace tab is treated as a generation project. Assets and lineage are atomically stored under Application Support and restored after the app relaunches. Explicitly closing a tab removes that project's workspace index while keeping exported media files on disk.
- All app data lives under `~/Library/Application Support/GenImage/` (`Models`, `Runtime`, `Workspace`, `Pasted`, `Generated`), defined in one place by `GenImageCore/ApplicationSupport.swift`. The workspace index used to be written under `GenMedia/`; it is adopted into the current root at launch, and an entry that already exists is kept rather than overwritten or merged. The directory keeps the name `GenImage` rather than matching the app's `GenMedia` to preserve compatibility with existing models and legacy runtime data.
- Prompt and lyrics editors preserve the caret, selection, and IME composition while native state updates arrive. Generation type, Prompt, Lyrics, and output-setting tabs rerender only the creation panel. Unavoidable full updates reuse playing audio and video nodes instead of interrupting playback.
- The workspace filmstrip provides an image import button and supports dropping one or more PNG, JPEG, WebP, GIF, TIFF, HEIC, or HEIF files from Finder. Image import is disabled during music generation to keep media sources separate. In image generation mode, selecting a source image automatically routes the main button to the image-to-image profile; without a source image it uses the text-to-image profile.
- Image and video aspect-ratio choices are dropdowns. Image-to-image shows `Original Resolution` only after a source image is selected, using source dimensions quantized to Runtime-compatible multiples of 16.
- Image-to-image width and height are passed to the Qwen Image Edit Runtime. When the requested aspect ratio differs from the source, the source keeps its own resolution and is edge-extended onto a canvas of the output aspect before generation, preserving the complete source content.
- The conditioning image is encoded at the generation resolution so the conditioning grid and the output grid share identical RoPE positions. Pinning the conditioning grid to a fixed 1024²-area size would align the model over the centre of the source only, and the output would be a cropped, magnified region.
- Generation resolution is decoupled from output resolution. When the requested area is below 1024², the Runtime generates at 1024² area in the requested aspect — the same ~4096 latent tokens the diffusers reference uses — and Lanczos-resamples down to the requested size. Far below the trained token count the DiT's denoise degrades and then collapses into striping, so low resolutions such as `128 × 192` are produced by resampling and look markedly better.
- Requests at or above 1024² area are generated at the requested size with no resampling; higher resolutions raise Runtime memory use and generation time accordingly. Outputs below 1024² still generate at 1024² area, so generation time does not drop as the output gets smaller.
- Starting an image-to-image run whose output area is below `512 × 512` opens a warning dialog first, offering Cancel or Generate Anyway. Confirming once suppresses the warning for the rest of the session.

### Music Runtime

Text-to-music is dispatched by `MusicGenerationRouter` to an ACE-Step 1.5 or MiniMax Music 3 adapter according to the active profile. The Swift app consistently manages music style, an optional prompt, optional lyrics, steps, seed, cancellation, time estimation, and audio metadata. A blank prompt uses the selected style, while blank lyrics generate instrumental music. Temporary WAV output from either runtime is converted by the shared FFmpeg layer to MP3, M4A, AAC, or FLAC.

- **ACE-Step 1.5 Turbo MLX**: the recommended profile, generating 10–300 second songs or instrumentals through the app's built-in Apple Silicon-native Swift/MLX runtime with no additional service installation. Its code and model use the commercially usable MIT License. Long audio uses overlap-discard tiled VAE decoding to control memory use.

- **MiniMax Music 3 MLX 8-bit**: runs through the independent Swift Worker shipped with the app. Its 5–300 second setting is a maximum duration; the model may end naturally earlier based on the song structure, and the app reports the actual output duration after completion. Its model remains subject to the Community License.
- **MiniMax Music 3 MLX 4-bit**: runs through the independent Swift Worker shipped with the app and uses the complete affine 4-bit MLX checkpoint. Its 5–300 second setting is a maximum duration; the model may still end naturally earlier based on the song structure. The model remains subject to the MiniMax Music 3 Community License.
- **MiniMax Music 3 Composer 5.7B Distilled**: is available in Model Center as an optional Composer acceleration component. The installer selects the `lr-6e-5` weights; it requires a complete Music 3 checkpoint and a compatible runtime, and cannot generate music by itself.

The native ACE-Step runtime, MiniMax Music 3 Swift Worker, and LGPL FFmpeg compatibility layer ship with the app. MiniMax Music 3 models remain optional components.

## Validation

```bash
swift test

for file in Sources/GenImageApp/Resources/WebUI/js/*.js; do
  node --check "$file"
done
```

Diagnose local models and automatically generated profiles:

```bash
swift run GenImageDoctor

# Or specify a custom model directory
GENIMAGE_MODEL_ROOT="/path/to/models" swift run GenImageDoctor
```

Start the standard MCP stdio server:

```bash
MCP_BIN_DIR="$(swift build -c release --show-bin-path)"
"$MCP_BIN_DIR/GenImageMCP"
```

The MCP server supports `initialize`, `ping`, `tools/list`, and `tools/call`. Available tools cover local models and profiles, native Z-Image text-to-image generation, Qwen image editing and description, Core ML upscaling, and standalone subtitle generation.

For app-managed use, enable the switch under Settings → MCP Integration. The app then exposes `http://127.0.0.1:12181/mcp`, a localhost-only HTTP POST JSON-RPC endpoint. HTTP and stdio share the same MCP tool core, while the stdio executable continues to work independently with GenMedia.app closed.

End-to-end MCP validation has been completed: `genimage_generate_image` outputs PNG files with a local Z-Image Turbo Q4 model, `genimage_describe_image` produces Traditional Chinese descriptions with Qwen3-VL, and `genimage_upscale_image` performs 4× upscaling with a local Real-ESRGAN Core ML model.

## Project Structure

This internal refactor changes code boundaries and ownership only; existing generation capabilities, user flows, and the Web Bridge protocol remain unchanged.

```text
Sources/
├── GenImageCore/
│   ├── ApplicationSupport.swift  # The one definition of where app data lives
│   ├── CivitaiTokenStore.swift   # macOS Keychain storage for the Civitai token
│   ├── DomainModels.swift        # Assets, recipes, jobs, models, and profiles
│   ├── InferenceServices.swift   # Image, text, video, music, and subtitle interfaces
│   ├── ModelCatalog.swift        # Built-in models and profiles
│   ├── OutputFileNaming.swift    # Image, video, music, and subtitle output names
│   ├── OutputGeometry.swift      # The one definition of output-size arithmetic (mirrored by js/geometry.js)
│   ├── ProjectWorkspacePersistence.swift # Open generation-project persistence
│   ├── SubtitleDocument.swift    # SRT and WebVTT rendering
│   └── WorkflowGraph.swift       # Asset lineage and branch relationships
├── GenImageRuntime/
│   ├── ZImageTextToImageService.swift
│   ├── QwenVLImageDescriptionService.swift
│   ├── Qwen2511ImageToImageService.swift
│   ├── LTXVideoGenerationService.swift
│   ├── MusicGenerationRouter.swift
│   ├── ACEStepMusicGenerationService.swift
│   ├── MiniMaxMusic3GenerationService.swift
│   ├── MediaCompatibilityService.swift # Bundled FFmpeg/ffprobe lookup, probing, transcoding
│   ├── MediaSourceCompatibilityService.swift # Direct source playback, remuxing, playback proxies
│   ├── MediaCompositionService.swift # Image-loop video and media merging
│   ├── MediaAudioPreparer.swift
│   ├── WhisperSubtitleTranscriber.swift
│   ├── ParaformerChineseSubtitleTranscriber.swift
│   ├── ParakeetJapaneseSubtitleTranscriber.swift
│   ├── SubtitleGenerationRouter.swift
│   ├── QwenTextGenerationService.swift
│   ├── SubprocessRuntime.swift   # Shared subprocess plumbing for external runtimes
│   ├── AudioOutputEncoder.swift
│   └── CoreMLUpscaleService.swift
├── GenImageApp/
    ├── AppStore.swift            # Type declaration, stored properties, init
    ├── AppStore+Credentials.swift # Settings credential operations
    ├── AppStore+SubtitleGeneration.swift
    ├── AppStore+MediaImport.swift
    ├── AppStore+MediaComposition.swift
    ├── AppStore+Workspaces.swift
    ├── AppStore+*.swift          # Other responsibility-based AppStore behavior
    ├── HybridBridgeController.swift
    ├── LocalMCPServiceController.swift # In-app MCP HTTP switch and state
    ├── HybridWebView.swift
    ├── AssetSchemeHandler.swift  # Secure local image, video, and audio delivery to WebUI
    └── Resources/WebUI/
        ├── js/automatic-flow.js  # Templates, Profile preflight, workspace plans
        └── …                     # Remaining HTML/CSS/JavaScript frontend
├── GenImageASRPoC/
    └── main.swift                # Standalone ASR validation utility
└── GenImageMCPServer/
    ├── MCPServer.swift           # Standalone stdio JSON-RPC server core
    └── MCPHTTPServer.swift       # Localhost HTTP transport used by the app switch
Patches/
├── MLX-Swift-LM-Qwen35-Text-Only.patch # Qwen3.5 text-only input compatibility
└── manifest.txt                  # Dependency patch manifest applied during builds
scripts/
├── apply-runtime-patches.command  # Apply and verify dependency patches from the manifest
└── build-ffmpeg-macos.sh          # Build a bundle-ready, relinkable LGPL FFmpeg distribution
```

## Current Status

The app is connected to local inference for Z-Image Turbo text-to-image, Qwen3-VL/Qwen3.5/Qwen3.8 multimodal image-to-text and text-to-text, Qwen 2511 image-to-image, LTX-2.3 MLX text-to-video and image-to-video, MiniMax H3 GGUF text-to-video and selected image-to-video profiles (testing phase), ACE-Step 1.5 Turbo MLX and MiniMax Music 3 MLX 8-bit/4-bit text-to-music, Whisper/Paraformer/Parakeet subtitle generation, and Core ML Real-ESRGAN upscaling. Videos are added as MP4 assets; music exports as MP3, M4A, AAC, or FLAC; subtitles export as SRT or WebVTT with language, timing, profile snapshots, and lineage.

More information:

- [Update Notes](UpdateNote.md)
- [Architecture](docs/ARCHITECTURE.en.md)
- [Web Bridge](docs/WEB_BRIDGE.en.md)
- [Roadmap](docs/ROADMAP.en.md)
- [MCP Interface](docs/MCP.en.md)
- [ASR Subtitle PoC](docs/ASR_POC.en.md)
- [Local Model Test Report](docs/MODEL_TEST_REPORT.en.md)

## License

This project uses a dual GPLv3 and commercial licensing model:

- Open-source use is licensed under the [GNU General Public License v3.0](LICENSE).
- If you cannot or do not want to comply with GPLv3, such as for closed-source integration, proprietary distribution, or customized commercial terms, contact the copyright holder to obtain a separate commercial license.
- Bundled FFmpeg and LAME remain under their respective LGPL terms. License texts, exact source versions, and build information are included under `Contents/Resources/Licenses/` in the app.

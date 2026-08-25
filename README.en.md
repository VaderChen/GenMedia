# GenMedia

[繁體中文](README.md) | English | [日本語](README.ja.md) | [한국어](README.ko.md)

GenMedia is a local AI media generation app with **native Apple Silicon support**. The project provides a compilable hybrid application with the following capabilities:

- Swift handles models, profiles, job queues, files, and MLX/Core ML inference.
- `WKWebView` embeds the HTML, CSS, and JavaScript UI without requiring a network connection or npm runtime.
- Text-to-image, image-to-text, image-to-image, text-to-video, image-to-video, text-to-music, and upscaling can run independently or be chained through asset lineage.
- Every operation preserves a profile snapshot, making the model and architecture revision traceable after updates.
- A dedicated settings page supports Traditional Chinese, English, Japanese, Korean, and six persistent color themes.
- A standard JSON-RPC 2.0 stdio MCP server is available to agents and automation tools.

## Preview

![GenMedia intelligent media generation interface](images/cap001.jpg)

## Run

Requirements: macOS 14+, Apple Silicon, and Xcode 16+.

```bash
./build.command
./run.command
```

`build.command` creates the release executables and a standard `GenMedia.app` in `dist/`. The app contains WebUI resources, the MLX Metal runtime, the MCP server, and model diagnostic tools.

```bash
# Build the release executables and app
./build.command

# Incremental release build without creating the app bundle
./build.command --no-app

# Set the version and bundle identifier
GENIMAGE_VERSION=1.1.0 GENIMAGE_BUNDLE_ID=com.example.genimage ./build.command
```

`run.command` automatically uses `--no-app`, so normal development runs do not repeatedly create the app bundle. Release DMGs are handled by a separate local workflow with Developer ID Application signing, Apple notarization, stapling, and Gatekeeper verification.

### Video Runtime

Video generation uses a replaceable external `ltx-2-mlx` runtime. The Swift app manages profiles, parameter validation, the job queue, cancellation, progress, assets, and video playback. Install the CLI and FFmpeg before first use:

```bash
brew install uv ffmpeg
./scripts/install-ltx-runtime.command
```

The app searches `GENIMAGE_LTX_RUNTIME`, `GENIMAGE_LTX_RUNTIME_ROOT/.venv/bin/ltx-2-mlx`, app helpers, `~/.local/bin/ltx-2-mlx`, common Homebrew paths, and `PATH` in order. To use a custom executable location:

```bash
GENIMAGE_LTX_RUNTIME="/absolute/path/to/ltx-2-mlx" ./run.command
```

`ltx-2-mlx` uses its Gemma text encoder configuration by default. If a local Gemma model is available, set `GENIMAGE_LTX_GEMMA_MODEL` to its directory or Hugging Face ID. The app DMG currently does not bundle the Python runtime, Gemma weights, or FFmpeg. Treat these as optional external components and review their runtime and model licenses separately before distribution.

### Profiles, Jobs, and Memory

- Profiles are ordered as active, ready, downloading, and unavailable. A subtle green outline marks profiles whose model and LoRA dependencies are complete, and the list is reordered as soon as a download finishes.
- Cancellation enters `cancelling` first, then changes to `cancelled` and unlocks all generation and memory controls when the runtime task exits. A numeric ETA appears after 35% progress and 15 seconds; overall elapsed time is used as a fallback when stable samples are not yet available.
- The Z-Image MLX compatibility layer handles `quantize_config.json`, affine/mxfp4 modes, packed pad tokens, and FP16-to-BF16 loading. `build.command` reapplies the patches under `Patches/` after Swift Package resolution. The andrevp Z-Image Turbo MLX 4-bit profile has been validated with a real generation run.
- Text-to-image completion keeps model weights and warm buffers resident. Reusable MLX buffers are trimmed after five idle minutes without unloading the model. Models are unloaded only by the sidebar Release Memory action, a model switch, or the over-90% RAM protection applied while switching profiles.
- Downloads retain their upstream filenames. Generated outputs use `Image-YYYYMMDD-HHmm`, `Video-YYYYMMDD-HHmm`, or `Music-YYYYMMDD-HHmm`; a numeric suffix prevents collisions within the same minute. The output directory is configurable in Settings.
- Each open workspace tab is treated as a generation project. Assets and lineage are atomically stored under Application Support and restored after the app relaunches. Explicitly closing a tab removes that project's workspace index while keeping exported media files on disk.
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

- **MiniMax Music 3 MLX 8-bit**: runs through the external `mlx-minimax-music3` CLI. Its 5–300 second setting is a maximum duration; the model may end naturally earlier based on the song structure, and the app reports the actual output duration after completion. Its model remains subject to the Community License.

The native ACE-Step runtime ships with the app. Models, the MiniMax Music 3 runtime, and FFmpeg remain optional external components.

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
.build/arm64-apple-macosx/release/GenImageMCP
```

The MCP server supports `initialize`, `ping`, `tools/list`, and `tools/call`. Available tools cover local models and profiles, native Z-Image text-to-image generation, Qwen3-VL image description, and Core ML upscaling.

End-to-end MCP validation has been completed: `genimage_generate_image` outputs PNG files with a local Z-Image Turbo Q4 model, `genimage_describe_image` produces Traditional Chinese descriptions with Qwen3-VL, and `genimage_upscale_image` performs 4× upscaling with a local Real-ESRGAN Core ML model.

## Project Structure

```text
Sources/
├── GenImageCore/
│   ├── DomainModels.swift        # Assets, recipes, jobs, models, and profiles
│   ├── InferenceServices.swift   # Image, text, video, and music inference interfaces
│   ├── ModelCatalog.swift        # Built-in models and profiles
│   ├── OutputFileNaming.swift    # Image, video, and music output names
│   ├── ProjectWorkspacePersistence.swift # Open generation-project persistence
│   └── WorkflowGraph.swift       # Asset lineage and branch relationships
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
    ├── AppStore.swift            # Application state and job coordination
    ├── HybridBridgeController.swift
    ├── HybridWebView.swift
    ├── AssetSchemeHandler.swift  # Secure local image, video, and audio delivery to WebUI
    └── Resources/WebUI/          # HTML/CSS/JavaScript frontend
Patches/                           # Z-Image MLX compatibility patches applied during builds
```

## Current Status

The app is connected to local inference for Z-Image Turbo text-to-image, Qwen3-VL image-to-text, Qwen 2511 image-to-image, LTX-2.3 MLX text-to-video and image-to-video, ACE-Step 1.5 Turbo MLX and MiniMax Music 3 MLX 8-bit text-to-music, and Core ML Real-ESRGAN upscaling. Videos are added as MP4 assets; music can be exported as MP3, M4A, AAC, or FLAC with actual duration, sample rate, channel count, profile snapshots, and lineage preserved.

More information:

- [Update Notes](UpdateNote.md)
- [Architecture](docs/ARCHITECTURE.en.md)
- [Web Bridge](docs/WEB_BRIDGE.en.md)
- [Roadmap](docs/ROADMAP.en.md)
- [MCP Interface](docs/MCP.en.md)
- [Local Model Test Report](docs/MODEL_TEST_REPORT.en.md)

## License

This project uses a dual GPLv3 and commercial licensing model:

- Open-source use is licensed under the [GNU General Public License v3.0](LICENSE).
- If you cannot or do not want to comply with GPLv3, such as for closed-source integration, proprietary distribution, or customized commercial terms, contact the copyright holder to obtain a separate commercial license.

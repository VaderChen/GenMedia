# GenImage

[繁體中文](README.md) | English | [日本語](README.ja.md) | [한국어](README.ko.md)

GenImage is a local AI media generation app with **native Apple Silicon support**. The project provides a compilable hybrid application with the following capabilities:

- Swift handles models, profiles, job queues, files, and MLX/Core ML inference.
- `WKWebView` embeds the HTML, CSS, and JavaScript UI without requiring a network connection or npm runtime.
- Text-to-image, image-to-text, image-to-image, text-to-video, image-to-video, and upscaling can run independently or be chained through asset lineage.
- Every operation preserves a profile snapshot, making the model and architecture revision traceable after updates.
- A dedicated settings page supports Traditional Chinese, English, Japanese, Korean, and six persistent color themes.
- A standard JSON-RPC 2.0 stdio MCP server is available to agents and automation tools.

## Run

Requirements: macOS 14+, Apple Silicon, and Xcode 16+.

```bash
./build.command
./run.command
```

`build.command` creates the release executables and a standard `GenImage.app` by default. Use `--dmg` explicitly when a disk image is needed. The DMG contains an Applications shortcut, WebUI resources, the MLX Metal runtime, the MCP server, and model diagnostic tools.

```bash
# Build the App (default; no DMG)
./build.command

# Release executables only, without an App bundle
./build.command --no-dmg

# Build the App and package a DMG
./build.command --dmg

# Set the version and bundle identifier
GENIMAGE_VERSION=1.1.0 GENIMAGE_BUNDLE_ID=com.example.genimage ./build.command
```

`run.command` automatically uses `--no-dmg`, so normal development runs only update the Release executables. The DMG is an optional, unnotarized local test package. Public distribution still requires Developer ID signing and Apple notarization.

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
- Downloads retain their upstream filenames. Generated outputs use `Image-MMDD-HHmmss` or `Video-MMDD-HHmmss`, and the output directory is configurable in Settings.

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
│   ├── InferenceServices.swift   # Image, text, and video inference interfaces
│   ├── ModelCatalog.swift        # Built-in models and profiles
│   ├── OutputFileNaming.swift    # Image and video output names
│   └── WorkflowGraph.swift       # Asset lineage and branch relationships
├── GenImageRuntime/
│   ├── ZImageTextToImageService.swift
│   ├── QwenVLImageDescriptionService.swift
│   ├── Qwen2511ImageToImageService.swift
│   ├── LTXVideoGenerationService.swift
│   └── CoreMLUpscaleService.swift
└── GenImageApp/
    ├── AppStore.swift            # Application state and job coordination
    ├── HybridBridgeController.swift
    ├── HybridWebView.swift
    ├── AssetSchemeHandler.swift  # Secure local image and video delivery to WebUI
    └── Resources/WebUI/          # HTML/CSS/JavaScript frontend
Patches/                           # Z-Image MLX compatibility patches applied during builds
```

## Current Status

The app is connected to local inference for Z-Image Turbo text-to-image, Qwen3-VL image-to-text, Qwen 2511 image-to-image, LTX-2.3 MLX text-to-video and image-to-video, and Core ML Real-ESRGAN upscaling. The video runtime runs through the external `ltx-2-mlx` CLI; completed MP4 files are added to the workspace as assets with profile snapshots and lineage preserved.

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

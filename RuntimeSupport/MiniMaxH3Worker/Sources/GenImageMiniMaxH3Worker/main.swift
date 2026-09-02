import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXRandom
import MiniMaxH3SwiftRuntime

private final class RequestEventEmitter: @unchecked Sendable {
    private let lock = NSLock()

    func emit(_ event: MiniMaxH3RequestProtocol.Event) {
        guard let data = try? JSONEncoder().encode(event) else { return }
        lock.lock()
        defer { lock.unlock() }
        FileHandle.standardOutput.write(data)
        FileHandle.standardOutput.write(Data("\n".utf8))
    }
}

/// MiniMax H3 worker.
///
/// The worker exposes checkpoint inspection, text-encoder diagnostics, and
/// end-to-end text/image-to-video/audio generation for the MiniMax H3 runtime.
private enum GenImageMiniMaxH3Worker {
    static func main() {
        let arguments = Array(CommandLine.arguments.dropFirst())
        guard let subcommand = arguments.first else {
            usage()
            exit(2)
        }

        // The app drives the worker with --request; the other subcommands are
        // diagnostics used by the parity harness.
        if arguments.contains("--request") {
            runRequest(arguments: arguments)
            return
        }

        switch subcommand {
        case "inspect":
            guard let path = value(for: "--transformer", in: arguments) else {
                fail("inspect 需要 --transformer <path to .gguf>")
            }
            runInspect(path: path, verbose: arguments.contains("--verbose"))
        case "decode-video":
            guard let path = value(for: "--vae", in: arguments) else {
                fail("decode-video 需要 --vae <path to video vae .safetensors>")
            }
            runDecodeVideo(
                vaePath: path,
                latentPath: value(for: "--latent", in: arguments),
                outputPath: value(for: "--output", in: arguments),
                pipelineNormalization: arguments.contains("--pipeline")
            )
        case "decode-audio":
            guard let path = value(for: "--vae", in: arguments) else {
                fail("decode-audio 需要 --vae <path to audio vae .safetensors>")
            }
            runDecodeAudio(
                vaePath: path,
                latentPath: value(for: "--latent", in: arguments),
                outputPath: value(for: "--output", in: arguments)
            )
        case "layout":
            runLayout(
                textLength: intValue(for: "--text", in: arguments) ?? 7,
                frames: intValue(for: "--frames", in: arguments) ?? 3,
                height: intValue(for: "--height", in: arguments) ?? 8,
                width: intValue(for: "--width", in: arguments) ?? 12,
                audioFrames: intValue(for: "--audio", in: arguments) ?? 5,
                keyframeSpec: value(for: "--keyframes", in: arguments),
                outputPath: value(for: "--output", in: arguments)
            )
        case "quant-check":
            guard let path = value(for: "--transformer", in: arguments) else {
                fail("quant-check 需要 --transformer <path to .gguf>")
            }
            runQuantCheck(
                path: path,
                useMetalQuantizer: !arguments.contains("--cpu-quantizer")
            )
        case "refiner-check":
            guard let path = value(for: "--transformer", in: arguments) else {
                fail("refiner-check 需要 --transformer <path to .gguf>")
            }
            runRefinerCheck(
                path: path,
                inputPath: value(for: "--input", in: arguments),
                outputPath: value(for: "--output", in: arguments)
            )
        case "dit-check":
            guard let path = value(for: "--transformer", in: arguments) else {
                fail("dit-check 需要 --transformer <path to .gguf>")
            }
            runDitCheck(
                path: path,
                layers: intValue(for: "--layers", in: arguments) ?? 2,
                inputPath: value(for: "--input", in: arguments),
                keyframeSpec: value(for: "--keyframes", in: arguments),
                outputPath: value(for: "--output", in: arguments)
            )
        case "forward-check":
            guard let path = value(for: "--transformer", in: arguments) else {
                fail("forward-check 需要 --transformer <path to .gguf>")
            }
            runForwardCheck(
                path: path,
                layers: intValue(for: "--layers", in: arguments) ?? 2,
                inputPath: value(for: "--input", in: arguments),
                outputPath: value(for: "--output", in: arguments)
            )
        case "load-check":
            guard let path = value(for: "--transformer", in: arguments) else {
                fail("load-check 需要 --transformer <path to .gguf>")
            }
            runLoadCheck(
                path: path,
                useMetalQuantizer: !arguments.contains("--cpu-quantizer")
            )
        case "text-check":
            guard let encoder = value(for: "--text-encoder", in: arguments),
                  let tokenizer = value(for: "--tokenizer", in: arguments) else {
                fail("text-check 需要 --text-encoder <path to .gguf> / --tokenizer <directory>")
            }
            runTextCheck(
                encoderPath: encoder,
                tokenizerPath: tokenizer,
                prompt: value(for: "--prompt", in: arguments)
                    ?? "一隻狐狸在雪地裡奔跑，a cinematic blue hour.",
                tokenIDs: value(for: "--token-ids", in: arguments),
                layerCount: intValue(for: "--layers", in: arguments),
                dense: arguments.contains("--dense"),
                useMetalQuantizer: !arguments.contains("--cpu-quantizer"),
                outputPath: value(for: "--output", in: arguments)
            )
        case "generate":
            guard let transformer = value(for: "--transformer", in: arguments),
                  let videoVAE = value(for: "--video-vae", in: arguments),
                  let output = value(for: "--output", in: arguments) else {
                fail("generate 需要 --transformer / --video-vae / --output")
            }
            runGenerate(
                transformerPath: transformer,
                videoVAEPath: videoVAE,
                audioVAEPath: value(for: "--audio-vae", in: arguments),
                textEncoderPath: value(for: "--text-encoder", in: arguments),
                tokenizerPath: value(for: "--tokenizer", in: arguments),
                inputPath: value(for: "--input", in: arguments),
                prompt: value(for: "--prompt", in: arguments) ?? "",
                outputPath: output,
                frames: intValue(for: "--latent-frames", in: arguments) ?? 3,
                height: intValue(for: "--latent-height", in: arguments)
                    ?? intValue(for: "--latent-size", in: arguments)
                    ?? 16,
                width: intValue(for: "--latent-width", in: arguments)
                    ?? intValue(for: "--latent-size", in: arguments)
                    ?? 16,
                audioFrames: intValue(for: "--audio-frames", in: arguments) ?? 8,
                frameRate: intValue(for: "--frame-rate", in: arguments) ?? 24,
                steps: intValue(for: "--steps", in: arguments) ?? 8,
                seed: intValue(for: "--seed", in: arguments) ?? 0
            )
        case "encode-image":
            guard let path = value(for: "--vae", in: arguments) else {
                fail("encode-image 需要 --vae <path to video vae .safetensors>")
            }
            runEncodeImage(
                vaePath: path,
                inputPath: value(for: "--input", in: arguments),
                outputPath: value(for: "--output", in: arguments)
            )
        default:
            usage()
            exit(2)
        }
    }

    /// Serve one `--request` invocation.
    private static func runRequest(arguments: [String]) {
        let emitter = RequestEventEmitter()

        do {
            guard let requestPath = value(for: "--request", in: arguments) else {
                throw MiniMaxH3RequestProtocol.RequestError
                    .missingComponent("--request")
            }
            let formatRaw = value(for: "--format", in: arguments) ?? "gguf"
            guard let format = MiniMaxH3RequestProtocol.Format(rawValue: formatRaw) else {
                throw MiniMaxH3RequestProtocol.RequestError.unsupportedFormat(formatRaw)
            }
            _ = format
            let variantRaw = value(for: "--variant", in: arguments) ?? "fl2va"
            guard let variant = MiniMaxH3RequestProtocol.Variant(rawValue: variantRaw) else {
                throw MiniMaxH3RequestProtocol.RequestError.unsupportedVariant(variantRaw)
            }

            let data = try Data(contentsOf: URL(fileURLWithPath: requestPath))
            let request = try JSONDecoder().decode(
                MiniMaxH3RequestProtocol.Request.self, from: data
            )
            // Ref2VA's transformer denoises text-to-video the same way; only
            // the reference-block conditioning is unimplemented. So the variant
            // is rejected when anchors are actually requested, not up front.
            if variant == .ref2va, request.keyframes?.isEmpty == false {
                throw MiniMaxH3RequestProtocol.RequestError.unsupportedVariant(variantRaw)
            }
            let components = try MiniMaxH3RequestProtocol.Components.resolve(
                request: request
            )
            let geometry = try MiniMaxH3RequestProtocol.latentGeometry(
                width: request.width, height: request.height, frames: request.frames
            )

            let started = Date()
            emitter.emit(.progress(stage: "loadingModel", value: 0.01))

            // Keyframe anchors are encoded before the pipeline runs, so the
            // large VAE encoder is not resident alongside the transformer.
            var conditioning: MiniMaxH3Transformer.Conditioning?
            if let keyframes = request.keyframes, !keyframes.isEmpty {
                emitter.emit(.progress(stage: "encodingKeyframes", value: 0.03))
                let encoder = try MiniMaxH3VideoVAEEncoder.load(
                    fileURL: components.videoVAE
                )
                var entries: [MiniMaxH3Transformer.Conditioning.Entry] = []
                for keyframe in keyframes {
                    let pixels = try loadImagePixels(
                        path: keyframe.imagePath,
                        width: request.width,
                        height: request.height
                    )
                    // Negative indices count from the end of the video.
                    let resolved = keyframe.frameIndex < 0
                        ? max(0, request.frames + keyframe.frameIndex)
                        : keyframe.frameIndex
                    entries.append(.init(
                        resolvedFrameIndex: resolved,
                        videoLatent: try encoder.encode(pixels: pixels)
                    ))
                }
                MLX.GPU.clearCache()
                conditioning = MiniMaxH3Transformer.Conditioning(entries: entries)
            }

            let pipeline = MiniMaxH3Pipeline(
                transformerURL: components.transformer,
                videoVAEURL: components.videoVAE,
                audioVAEURL: components.audioVAE,
                textEncoderURL: components.textEncoder,
                tokenizerDirectoryURL: components.tokenizerDirectory
            )
            let pipelineRequest = MiniMaxH3Pipeline.Request(
                latentFrames: geometry.latentFrames,
                latentHeight: geometry.latentHeight,
                latentWidth: geometry.latentWidth,
                audioFrames: MiniMaxH3RequestProtocol.audioLatentFrames(
                    frames: request.frames, frameRate: request.frameRate
                ),
                steps: request.steps,
                seed: request.seed,
                prompt: request.prompt
            )

            let result = try pipeline.run(
                request: pipelineRequest,
                conditioning: conditioning
            ) { stage, fraction in
                // Reserve the tail of the range for encoding the video file.
                emitter.emit(.progress(stage: stage, value: 0.05 + fraction * 0.85))
            }

            emitter.emit(.progress(stage: "encoding", value: 0.92))
            let encodingHeartbeat = DispatchSource.makeTimerSource(
                queue: DispatchQueue.global(qos: .utility)
            )
            encodingHeartbeat.schedule(deadline: .now() + 5, repeating: 5)
            encodingHeartbeat.setEventHandler {
                emitter.emit(.progress(stage: "encoding", value: 0.92))
            }
            encodingHeartbeat.resume()
            defer { encodingHeartbeat.cancel() }

            let images = try MiniMaxH3VideoWriter.images(from: result.pixels)
            let outputURL = URL(fileURLWithPath: request.outputPath)
            let audioConfiguration = MiniMaxH3AudioVAEConfiguration.default
            try MiniMaxH3VideoWriter.writeMP4(
                images,
                to: outputURL,
                frameRate: request.frameRate,
                audio: result.audio.map {
                    MiniMaxH3VideoWriter.Audio(
                        waveform: $0,
                        sampleRate: audioConfiguration.sampleRate
                    )
                },
                progress: { fraction in
                    emitter.emit(
                        .progress(stage: "encoding", value: 0.92 + fraction * 0.07)
                    )
                }
            )

            emitter.emit(.completed(
                durationSeconds: Double(images.count) / Double(max(request.frameRate, 1)),
                sampleRate: result.audio == nil ? 0 : audioConfiguration.sampleRate,
                numFrames: images.count,
                pixelWidth: request.width,
                pixelHeight: request.height
            ))
        } catch {
            emitter.emit(.error(error.localizedDescription))
            exit(1)
        }
    }

    /// Load an image and resize it to the generation geometry, as `[1, 3, 1, H, W]`
    /// in [-1, 1].
    private static func loadImagePixels(
        path: String,
        width: Int,
        height: Int
    ) throws -> MLXArray {
        let url = URL(fileURLWithPath: path)
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw MiniMaxH3RequestProtocol.RequestError.missingComponent(path)
        }
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        guard let context = CGContext(
            data: &rgba, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else {
            throw MiniMaxH3RequestProtocol.RequestError.missingComponent(path)
        }
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        var planar = [Float](repeating: 0, count: 3 * height * width)
        for pixel in 0 ..< (width * height) {
            for channel in 0 ..< 3 {
                planar[channel * height * width + pixel] =
                    Float(rgba[pixel * 4 + channel]) / 255.0 * 2.0 - 1.0
            }
        }
        return MLXArray(planar, [1, 3, 1, height, width])
    }

    /// Encode one frame into a keyframe latent.
    private static func runEncodeImage(
        vaePath: String,
        inputPath: String?,
        outputPath: String?
    ) {
        do {
            let encoder = try MiniMaxH3VideoVAEEncoder.load(
                fileURL: URL(fileURLWithPath: vaePath)
            )
            print("encoder loaded and validated")

            let pixels: MLXArray
            if let inputPath {
                let loaded = try MLX.loadArrays(url: URL(fileURLWithPath: inputPath))
                guard let value = loaded["pixels"] else {
                    fail("input 檔案缺少 \"pixels\" 這個 key")
                }
                pixels = value.asType(.float32)
            } else {
                pixels = MLXRandom.normal([1, 3, 1, 256, 256])
            }
            print("pixels: \(pixels.shape)")

            let started = Date()
            let latent = try encoder.encode(pixels: pixels)
            MLX.eval(latent)
            let elapsed = Date().timeIntervalSince(started)

            print("latent: \(latent.shape)")
            print(String(
                format: "stats: min %.6f max %.6f mean %.6f std %.6f",
                MLX.min(latent).item(Float.self),
                MLX.max(latent).item(Float.self),
                MLX.mean(latent).item(Float.self),
                MLX.sqrt(MLX.variance(latent)).item(Float.self)
            ))
            print(String(format: "time: %.2fs", elapsed))

            if let outputPath {
                try MLX.save(
                    arrays: ["latent": latent],
                    url: URL(fileURLWithPath: outputPath)
                )
                print("wrote \(outputPath)")
            }
        } catch {
            fail(error.localizedDescription)
        }
    }

    private static func runGenerate(
        transformerPath: String,
        videoVAEPath: String,
        audioVAEPath: String?,
        textEncoderPath: String?,
        tokenizerPath: String?,
        inputPath: String?,
        prompt: String,
        outputPath: String,
        frames: Int,
        height: Int,
        width: Int,
        audioFrames: Int,
        frameRate: Int,
        steps: Int,
        seed: Int
    ) {
        do {
            let pipeline = MiniMaxH3Pipeline(
                transformerURL: URL(fileURLWithPath: transformerPath),
                videoVAEURL: URL(fileURLWithPath: videoVAEPath),
                audioVAEURL: audioVAEPath.map { URL(fileURLWithPath: $0) },
                textEncoderURL: textEncoderPath.map { URL(fileURLWithPath: $0) },
                tokenizerDirectoryURL: tokenizerPath.map { URL(fileURLWithPath: $0) }
            )
            let request = MiniMaxH3Pipeline.Request(
                latentFrames: frames,
                latentHeight: height,
                latentWidth: width,
                audioFrames: audioFrames,
                steps: steps,
                seed: UInt64(seed),
                prompt: prompt
            )
            let target = request.outputSize()
            print("target: \(target.width)x\(target.height), \(target.frames) frames, \(steps) steps")
            if prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                print("NOTE: no prompt — running unconditional (zero text states)")
            } else if textEncoderPath == nil || tokenizerPath == nil {
                fail("有 prompt 時必須同時提供 --text-encoder 與 --tokenizer")
            } else {
                print("conditioning: Qwen3-VL raw layer-50 hidden states")
            }

            var conditioning: MiniMaxH3Transformer.Conditioning?
            if let inputPath {
                print("conditioning: encoding image keyframe")
                var encoder: MiniMaxH3VideoVAEEncoder? = try MiniMaxH3VideoVAEEncoder.load(
                    fileURL: URL(fileURLWithPath: videoVAEPath)
                )
                let pixels = try loadImagePixels(
                    at: URL(fileURLWithPath: inputPath),
                    width: target.width,
                    height: target.height
                )
                let latent = try encoder!.encode(pixels: pixels)
                MLX.eval(latent)
                encoder = nil
                MLX.GPU.clearCache()
                conditioning = MiniMaxH3Transformer.Conditioning(
                    entries: [
                        .init(resolvedFrameIndex: 0, videoLatent: latent)
                    ]
                )
                print("conditioning: image latent (latent.shape)")
            }

            let started = Date()
            var lastStage = ""
            let result = try pipeline.run(
                request: request,
                conditioning: conditioning
            ) { stage, fraction in
                if stage != lastStage || fraction == 1 {
                    lastStage = stage
                    print(String(format: "  %@ %.0f%%", stage, fraction * 100))
                }
            }
            let elapsed = Date().timeIntervalSince(started)

            print("pixels: \(result.pixels.shape)")
            print(String(
                format: "pixel range: %.4f .. %.4f  mean %.4f",
                MLX.min(result.pixels).item(Float.self),
                MLX.max(result.pixels).item(Float.self),
                MLX.mean(result.pixels).item(Float.self)
            ))

            let images = try MiniMaxH3VideoWriter.images(from: result.pixels)
            let outputURL = URL(fileURLWithPath: outputPath)
            let directory = outputURL.deletingPathExtension()
            let pngs = try MiniMaxH3VideoWriter.writePNGs(images, to: directory)
            print("wrote \(pngs.count) PNG frame(s) to \(directory.path)")
            try MiniMaxH3VideoWriter.writeMP4(
                images,
                to: outputURL,
                frameRate: frameRate
            )
            let size = (try? FileManager.default.attributesOfItem(
                atPath: outputURL.path
            )[.size] as? Int) ?? 0
            print("wrote \(outputURL.path) (\(size) bytes)")

            if let audio = result.audio {
                print("audio: \(audio.shape) at 32 kHz")
            } else {
                print("audio: not decoded (pass --audio-vae to enable)")
            }
            print(String(format: "total: %.1fs", elapsed))
            let peak = Double(MLX.GPU.peakMemory) / 1_073_741_824.0
            print(String(format: "peak GPU memory: %.2f GB", peak))
        } catch {
            fail(error.localizedDescription)
        }
    }

    private enum ImageInputError: LocalizedError {
        case unreadable(URL)
        case invalidDimensions(Int, Int)
        case contextCreationFailed

        var errorDescription: String? {
            switch self {
            case let .unreadable(url):
                "無法讀取圖片錨點：\(url.path)"
            case let .invalidDimensions(width, height):
                "圖片錨點輸出尺寸無效：\(width)x\(height)。"
            case .contextCreationFailed:
                "無法建立圖片解碼色彩空間。"
            }
        }
    }

    private static func loadImagePixels(
        at url: URL,
        width: Int,
        height: Int
    ) throws -> MLXArray {
        guard width > 0, height > 0 else {
            throw ImageInputError.invalidDimensions(width, height)
        }
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw ImageInputError.unreadable(url)
        }

        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(
                  data: &rgba,
                  width: width,
                  height: height,
                  bitsPerComponent: 8,
                  bytesPerRow: width * 4,
                  space: colorSpace,
                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
              ) else {
            throw ImageInputError.contextCreationFailed
        }

        context.setFillColor(CGColor(
            red: 1, green: 1, blue: 1, alpha: 1
        ))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        context.interpolationQuality = .high
        context.draw(
            image,
            in: CGRect(x: 0, y: 0, width: width, height: height)
        )

        let pixelCount = width * height
        var values = [Float](repeating: 0, count: pixelCount * 3)
        for index in 0 ..< pixelCount {
            let sourceOffset = index * 4
            let normalizedRed = Float(rgba[sourceOffset]) / 127.5 - 1
            let normalizedGreen = Float(rgba[sourceOffset + 1]) / 127.5 - 1
            let normalizedBlue = Float(rgba[sourceOffset + 2]) / 127.5 - 1
            values[index] = normalizedRed
            values[pixelCount + index] = normalizedGreen
            values[pixelCount * 2 + index] = normalizedBlue
        }
        return MLXArray(values, [1, 3, 1, height, width])
    }

    private static func runTextCheck(
        encoderPath: String,
        tokenizerPath: String,
        prompt: String,
        tokenIDs: String?,
        layerCount: Int?,
        dense: Bool,
        useMetalQuantizer: Bool,
        outputPath: String?
    ) {
        do {
            let started = Date()
            let encoder = try MiniMaxH3Qwen3VLTextEncoder.load(
                fileURL: URL(fileURLWithPath: encoderPath),
                tokenizerDirectory: URL(fileURLWithPath: tokenizerPath),
                layerCount: layerCount,
                quantizeLinear: !dense,
                useMetalQuantizer: useMetalQuantizer
            )
            print("source tensors      : \(encoder.report.sourceTensorCount)")
            print("loaded LM tensors   : \(encoder.report.loadedTensorCount)")
            print("INT8 linear modules : \(encoder.report.quantizedModuleCount)")
            print("dense tensors       : \(encoder.report.denseTensorCount)")
            print("vocabulary size     : \(encoder.vocabularySize)")
            print("language layers     : \(encoder.layerCount)")
            let ids: [Int]
            if let tokenIDs {
                guard let parsed = parseTokenIDs(tokenIDs), !parsed.isEmpty else {
                    fail("--token-ids 必須是逗號分隔的非空整數")
                }
                ids = parsed
            } else {
                ids = try encoder.tokenIDs(for: prompt)
            }
            print("prompt              : \(prompt)")
            print("token ids (\(ids.count)): \(ids)")
            let hidden = try encoder.hiddenStates(
                forTokenIDs: ids,
                throughLayerCount: layerCount
            )
            print("hidden              : \(hidden.shape) raw, no final norm")
            print(String(
                format: "stats               : min %.6f max %.6f mean %.6f std %.6f",
                MLX.min(hidden).item(Float.self),
                MLX.max(hidden).item(Float.self),
                MLX.mean(hidden).item(Float.self),
                MLX.sqrt(MLX.variance(hidden)).item(Float.self)
            ))
            if let outputPath {
                try MLX.save(
                    arrays: ["hidden": hidden],
                    url: URL(fileURLWithPath: outputPath)
                )
                print("wrote              : \(outputPath)")
            }
            print(String(format: "elapsed             : %.1fs", Date().timeIntervalSince(started)))
            let peak = Double(MLX.GPU.peakMemory) / 1_073_741_824.0
            print(String(format: "peak GPU memory     : %.2f GB", peak))
        } catch {
            fail(error.localizedDescription)
        }
    }

    /// Load the whole transformer as INT8 and report cost, so a generation run
    /// does not discover an out-of-memory failure late.
    private static func runLoadCheck(path: String, useMetalQuantizer: Bool) {
        do {
            let started = Date()
            let loaded = try MiniMaxH3GGUFQuantizedLoader.load(
                fileURL: URL(fileURLWithPath: path),
                useMetalQuantizer: useMetalQuantizer
            )
            let elapsed = Date().timeIntervalSince(started)
            print("quantized tensors : \(loaded.quantizedCount)")
            print("dense tensors     : \(loaded.denseCount)")
            print("skipped quantize  : \(loaded.skippedQuantizationCount)")
            print("Metal INT8 tensors: \(loaded.metalQuantizedCount)")
            print("CPU INT8 tensors  : \(loaded.cpuQuantizedCount)")
            print("total entries     : \(loaded.tensors.count)")
            print(String(format: "load time         : %.1fs", elapsed))
            let active = Double(MLX.GPU.activeMemory) / 1_073_741_824.0
            let peak = Double(MLX.GPU.peakMemory) / 1_073_741_824.0
            print(String(format: "GPU active/peak   : %.2f GB / %.2f GB", active, peak))

            let configuration = loaded.configuration
            var missing: [String] = []
            for name in configuration.expectedTensorShapes.keys.sorted()
            where loaded.tensors[name] == nil {
                missing.append(name)
            }
            if missing.isEmpty {
                print("all \(configuration.expectedTensorShapes.count) required tensors present")
            } else {
                print("missing \(missing.count): \(missing.prefix(5).joined(separator: ", "))")
            }
        } catch {
            fail(error.localizedDescription)
        }
    }

    /// Full forward through N DiT blocks, including the final layer and the
    /// unpatchify back to latent shape.
    private static func runForwardCheck(
        path: String,
        layers: Int,
        inputPath: String?,
        outputPath: String?
    ) {
        do {
            let inventory = try MiniMaxH3GGUFWeightLoader.inspectTransformer(
                fileURL: URL(fileURLWithPath: path)
            )
            let configuration = try MiniMaxH3Configuration.forInventory(inventory)
            let needed: (String) -> Bool = { name in
                if (!configuration.usesAdalnCurves && name.hasPrefix("time_embedder."))
                    || (configuration.usesAdalnCurves && name == "adaln_t_table")
                    || name == "rope.inv_freq"
                    || name.hasPrefix("video_patch_proj.")
                    || name.hasPrefix("audio_patch_proj.")
                    || name.hasPrefix("final_layer.") {
                    return true
                }
                guard name.hasPrefix("blocks.") else { return false }
                let index = name.dropFirst("blocks.".count).prefix { $0.isNumber }
                return Int(index).map { $0 < layers } ?? false
            }
            let dense = try MiniMaxH3GGUFQuantizedLoader.loadDense(
                fileURL: URL(fileURLWithPath: path), matching: needed
            )
            print("dense tensors loaded: \(dense.count)")
            let transformer = MiniMaxH3Transformer(
                configuration: configuration,
                weights: dense, quantizedPrefixes: [], computeDType: .float32
            )

            let video: MLXArray
            let audio: MLXArray
            let text: MLXArray
            if let inputPath {
                let loaded = try MLX.loadArrays(url: URL(fileURLWithPath: inputPath))
                guard let v = loaded["video"], let a = loaded["audio"],
                      let t = loaded["text"] else {
                    fail("input 檔案需要 video / audio / text 三個 key")
                }
                video = v.asType(.float32)
                audio = a.asType(.float32)
                text = t.asType(.float32)
            } else {
                video = MLXRandom.normal([1, 24, 3, 8, 12])
                audio = MLXRandom.normal([1, 32, 2, 5])
                text = MLXRandom.normal([7, 5376])
            }

            let started = Date()
            let output = try transformer.forward(
                videoLatent: video, audioLatent: audio, textStates: text,
                sigma: 0.7, layerCount: layers
            )
            MLX.eval(output.video, output.audio)
            let elapsed = Date().timeIntervalSince(started)

            print("video out: \(output.video.shape)  audio out: \(output.audio.shape)")
            print(String(
                format: "video stats: min %.6f max %.6f mean %.6f",
                MLX.min(output.video).item(Float.self),
                MLX.max(output.video).item(Float.self),
                MLX.mean(output.video).item(Float.self)
            ))
            print(String(format: "time: %.2fs", elapsed))

            if let outputPath {
                try MLX.save(
                    arrays: ["video": output.video, "audio": output.audio],
                    url: URL(fileURLWithPath: outputPath)
                )
                print("wrote \(outputPath)")
            }
        } catch {
            fail(error.localizedDescription)
        }
    }

    /// Run the packed sequence through the first N DiT blocks with fully
    /// decoded weights, isolating AdaLN/RoPE from quantization error.
    private static func runDitCheck(
        path: String,
        layers: Int,
        inputPath: String?,
        keyframeSpec: String?,
        outputPath: String?
    ) {
        do {
            let inventory = try MiniMaxH3GGUFWeightLoader.inspectTransformer(
                fileURL: URL(fileURLWithPath: path)
            )
            let configuration = try MiniMaxH3Configuration.forInventory(inventory)
            let needed: (String) -> Bool = { name in
                if (!configuration.usesAdalnCurves && name.hasPrefix("time_embedder."))
                    || (configuration.usesAdalnCurves && name == "adaln_t_table")
                    || name == "rope.inv_freq"
                    || name.hasPrefix("video_patch_proj.")
                    || name.hasPrefix("audio_patch_proj.") {
                    return true
                }
                guard name.hasPrefix("blocks.") else { return false }
                let index = name.dropFirst("blocks.".count)
                    .prefix { $0.isNumber }
                return Int(index).map { $0 < layers } ?? false
            }
            let dense = try MiniMaxH3GGUFQuantizedLoader.loadDense(
                fileURL: URL(fileURLWithPath: path), matching: needed
            )
            print("dense tensors loaded: \(dense.count)")
            let transformer = MiniMaxH3Transformer(
                configuration: configuration,
                weights: dense, quantizedPrefixes: [], computeDType: .float32
            )

            let video: MLXArray
            let audio: MLXArray
            let text: MLXArray
            if let inputPath {
                let loaded = try MLX.loadArrays(url: URL(fileURLWithPath: inputPath))
                guard let v = loaded["video"], let a = loaded["audio"],
                      let t = loaded["text"] else {
                    fail("input 檔案需要 video / audio / text 三個 key")
                }
                video = v.asType(.float32)
                audio = a.asType(.float32)
                text = t.asType(.float32)
            } else {
                video = MLXRandom.normal([1, 24, 3, 8, 12])
                audio = MLXRandom.normal([1, 32, 2, 5])
                text = MLXRandom.normal([7, 5376])
            }
            print("video \(video.shape)  audio \(audio.shape)  text \(text.shape)")

            // Conditioning latents ride in the same fixture as cond_video_N.
            var entries: [MiniMaxH3Transformer.Conditioning.Entry] = []
            if let keyframeSpec, let inputPath {
                let loaded = try MLX.loadArrays(url: URL(fileURLWithPath: inputPath))
                for (index, entry) in keyframeSpec.split(separator: ",").enumerated() {
                    let parts = entry.split(separator: ":").map(String.init)
                    guard let frameIndex = parts.first.flatMap(Int.init) else { continue }
                    entries.append(.init(
                        resolvedFrameIndex: frameIndex,
                        videoLatent: loaded["cond_video_\(index)"]?.asType(.float32)
                    ))
                }
                print("conditioning entries: \(entries.count)")
            }
            // aug = 1 disables the noise blend so this matches the reference exactly.
            let conditioning = entries.isEmpty ? nil
                : MiniMaxH3Transformer.Conditioning(
                    entries: entries, visualNoiseAugmentation: 1.0
                )

            let started = Date()
            let hidden = try transformer.forwardPartial(
                videoLatent: video, audioLatent: audio, refinedText: text,
                sigma: 0.7, layerCount: layers, conditioning: conditioning
            )
            MLX.eval(hidden)
            let elapsed = Date().timeIntervalSince(started)

            print("hidden: \(hidden.shape)  after \(layers) block(s)")
            print(String(
                format: "stats: min %.6f max %.6f mean %.6f std %.6f",
                MLX.min(hidden).item(Float.self),
                MLX.max(hidden).item(Float.self),
                MLX.mean(hidden).item(Float.self),
                MLX.sqrt(MLX.variance(hidden)).item(Float.self)
            ))
            print(String(format: "time: %.2fs", elapsed))

            if let outputPath {
                try MLX.save(
                    arrays: ["hidden": hidden],
                    url: URL(fileURLWithPath: outputPath)
                )
                print("wrote \(outputPath)")
            }
        } catch {
            fail(error.localizedDescription)
        }
    }

    /// Run condition_proj + the two token-refiner blocks with fully decoded
    /// weights, so the comparison isolates model logic from quantization.
    private static func runRefinerCheck(
        path: String,
        inputPath: String?,
        outputPath: String?
    ) {
        do {
            let dense = try MiniMaxH3GGUFQuantizedLoader.loadDense(
                fileURL: URL(fileURLWithPath: path),
                matching: { name in
                    name.hasPrefix("condition_proj.")
                        || name.hasPrefix("token_refiner.")
                }
            )
            print("dense tensors loaded: \(dense.count)")

            let transformer = MiniMaxH3Transformer(
                weights: dense,
                quantizedPrefixes: [],
                computeDType: .float32
            )

            let text: MLXArray
            if let inputPath {
                let loaded = try MLX.loadArrays(url: URL(fileURLWithPath: inputPath))
                guard let value = loaded["text"] else {
                    fail("input 檔案缺少 \"text\" 這個 key")
                }
                text = value.asType(.float32)
            } else {
                text = MLXRandom.normal([7, 5120])
            }
            print("text states: \(text.shape)")

            let started = Date()
            let refined = try transformer.refineTextStates(text)
            MLX.eval(refined)
            let elapsed = Date().timeIntervalSince(started)

            print("refined: \(refined.shape)")
            print(String(
                format: "stats: min %.6f max %.6f mean %.6f std %.6f",
                MLX.min(refined).item(Float.self),
                MLX.max(refined).item(Float.self),
                MLX.mean(refined).item(Float.self),
                MLX.sqrt(MLX.variance(refined)).item(Float.self)
            ))
            print(String(format: "time: %.2fs", elapsed))

            if let outputPath {
                try MLX.save(
                    arrays: ["refined": refined],
                    url: URL(fileURLWithPath: outputPath)
                )
                print("wrote \(outputPath)")
            }
        } catch {
            fail(error.localizedDescription)
        }
    }

    /// Compare INT4 and INT8 re-quantization error against the exactly decoded
    /// Q4_0 values, on a few representative tensors.
    private static func runQuantCheck(path: String, useMetalQuantizer: Bool) {
        let names = [
            "blocks.0.attn.qkv_proj.weight",
            "blocks.0.mlp.fc1.weight",
            "blocks.25.attn.out_proj.weight",
            "token_refiner.blocks.0.mlp.fc2.weight"
        ]
        do {
            let dense = try MiniMaxH3GGUFQuantizedLoader.loadDense(
                fileURL: URL(fileURLWithPath: path),
                matching: { names.contains($0) }
            )
            print("quantizer: \(useMetalQuantizer ? "Metal" : "CPU")")
            print("tensor                                    shape            INT4 rel   INT8 rel")
            for name in names {
                guard let reference = dense[name] else {
                    print("  \(name): missing")
                    continue
                }
                let denominator = MLX.max(MLX.abs(reference)).item(Float.self)
                let int8 = try MiniMaxH3GGUFQuantizedLoader.loadQuantizedTensor(
                    fileURL: URL(fileURLWithPath: path),
                    named: name,
                    useMetalQuantizer: useMetalQuantizer
                )
                var line = "  \(name.padding(toLength: 40, withPad: " ", startingAt: 0))"
                line += " \(String(describing: reference.shape).padding(toLength: 16, withPad: " ", startingAt: 0))"
                let int4 = MLX.quantized(reference, groupSize: 64, bits: 4)
                let int4Restored = MLX.dequantized(
                    int4.wq, scales: int4.scales, biases: int4.biases,
                    groupSize: 64, bits: 4
                )
                let int4Error = MLX.max(MLX.abs(int4Restored - reference)).item(Float.self)
                let int8Restored = MLX.dequantized(
                    int8.weights, scales: int8.scales, biases: int8.biases,
                    groupSize: 64, bits: 8
                )
                let int8Error = MLX.max(MLX.abs(int8Restored - reference)).item(Float.self)
                line += String(format: " %10.3e", int4Error / max(denominator, 1e-12))
                line += String(format: " %10.3e", int8Error / max(denominator, 1e-12))
                print(line)
            }
        } catch {
            fail(error.localizedDescription)
        }
    }

    private static func runLayout(
        textLength: Int,
        frames: Int,
        height: Int,
        width: Int,
        audioFrames: Int,
        keyframeSpec: String?,
        outputPath: String?
    ) {
        // "index:videoFrames[:audioFrames]" entries separated by commas.
        let keyframes: [MiniMaxH3Keyframe] = (keyframeSpec ?? "")
            .split(separator: ",")
            .compactMap { entry in
                let parts = entry.split(separator: ":").map(String.init)
                guard let index = parts.first.flatMap(Int.init) else { return nil }
                return MiniMaxH3Keyframe(
                    resolvedFrameIndex: index,
                    videoLatentFrames: parts.count > 1 ? Int(parts[1]) : nil,
                    audioLatentFrames: parts.count > 2 ? Int(parts[2]) : nil
                )
            }
        let layout = MiniMaxH3PackedLayout(
            textLength: textLength,
            latentFrames: frames,
            latentHeight: height,
            latentWidth: width,
            audioFrames: audioFrames,
            keyframes: keyframes
        )
        let described = layout.segments
            .map { "(\($0.start), \($0.end), \($0.kind.rawValue))" }
            .joined(separator: ", ")
        print("segments: [\(described)]")
        print("seq_len: \(layout.sequenceLength)")
        print("position_ids: (\(layout.sequenceLength), 3)")
        if let outputPath {
            let array = MLXArray(
                layout.positionIDs.map { Float($0) },
                [layout.sequenceLength, 3]
            )
            do {
                try MLX.save(
                    arrays: ["positions": array],
                    url: URL(fileURLWithPath: outputPath)
                )
                print("wrote \(outputPath)")
            } catch {
                fail(error.localizedDescription)
            }
        }
    }

    private static func intValue(for flag: String, in arguments: [String]) -> Int? {
        value(for: flag, in: arguments).flatMap(Int.init)
    }

    private static func parseTokenIDs(_ value: String) -> [Int]? {
        let values = value.split(separator: ",", omittingEmptySubsequences: false)
        guard !values.isEmpty else { return nil }
        let parsed = values.compactMap { Int(String($0).trimmingCharacters(in: .whitespaces)) }
        return parsed.count == values.count ? parsed : nil
    }

    private static func runDecodeAudio(
        vaePath: String,
        latentPath: String?,
        outputPath: String?
    ) {
        do {
            let decoder = try MiniMaxH3AudioVAEDecoder.load(
                fileURL: URL(fileURLWithPath: vaePath)
            )
            print("audio decoder loaded and validated")

            let latent: MLXArray
            if let latentPath {
                let loaded = try MLX.loadArrays(url: URL(fileURLWithPath: latentPath))
                guard let value = loaded["latent"] else {
                    fail("latent 檔案缺少 \"latent\" 這個 key")
                }
                latent = value.asType(.float32)
            } else {
                latent = MLXRandom.normal([1, 32, 2, 8])
            }
            print("latent shape: \(latent.shape)")

            let started = Date()
            let waveform = try decoder.decode(latent: latent)
            MLX.eval(waveform)
            let elapsed = Date().timeIntervalSince(started)

            print("output shape: \(waveform.shape)")
            let flat = waveform.asType(.float32)
            print(String(
                format: "output stats: min %.6f max %.6f mean %.6f std %.6f",
                MLX.min(flat).item(Float.self),
                MLX.max(flat).item(Float.self),
                MLX.mean(flat).item(Float.self),
                MLX.sqrt(MLX.variance(flat)).item(Float.self)
            ))
            print(String(format: "decode time: %.2fs", elapsed))

            if let outputPath {
                try MLX.save(
                    arrays: ["audio": flat],
                    url: URL(fileURLWithPath: outputPath)
                )
                print("wrote \(outputPath)")
            }
        } catch {
            fail(error.localizedDescription)
        }
    }

    private static func runDecodeVideo(
        vaePath: String,
        latentPath: String?,
        outputPath: String?,
        pipelineNormalization: Bool = false
    ) {
        do {
            let decoder = try MiniMaxH3VideoVAEDecoder.load(
                fileURL: URL(fileURLWithPath: vaePath)
            )
            print("decoder loaded and validated against MiniMaxH3VideoVAEConfiguration")

            let latent: MLXArray
            if let latentPath {
                let loaded = try MLX.loadArrays(url: URL(fileURLWithPath: latentPath))
                guard let value = loaded["latent"] else {
                    fail("latent 檔案缺少 \"latent\" 這個 key")
                }
                latent = value.asType(.float32)
            } else {
                latent = MLXRandom.normal([1, 24, 1, 16, 16])
            }
            print("latent shape: \(latent.shape)")

            let started = Date()
            // --pipeline matches what MiniMaxH3Pipeline does around the raw
            // decoder: de-normalize the latent going in, map to [0, 1] coming out.
            let input = pipelineNormalization
                ? MiniMaxH3VideoVAEDecoder.denormalizeLatent(latent)
                : latent
            var pixels = try decoder.decode(latent: input)
            if pipelineNormalization {
                pixels = MiniMaxH3VideoVAEDecoder.finalizePixels(pixels)
            }
            MLX.eval(pixels)
            let elapsed = Date().timeIntervalSince(started)

            print("output shape: \(pixels.shape)")
            let flat = pixels.asType(.float32)
            print(String(
                format: "output stats: min %.6f max %.6f mean %.6f std %.6f",
                MLX.min(flat).item(Float.self),
                MLX.max(flat).item(Float.self),
                MLX.mean(flat).item(Float.self),
                MLX.sqrt(MLX.variance(flat)).item(Float.self)
            ))
            print(String(format: "decode time: %.2fs", elapsed))

            if let outputPath {
                try MLX.save(
                    arrays: ["pixels": flat],
                    url: URL(fileURLWithPath: outputPath)
                )
                print("wrote \(outputPath)")
            }
        } catch {
            fail(error.localizedDescription)
        }
    }

    private static func runInspect(path: String, verbose: Bool) {
        let url = URL(fileURLWithPath: path)
        do {
            let inventory = try MiniMaxH3GGUFWeightLoader.inspectTransformer(fileURL: url)
            let configuration = try MiniMaxH3Configuration.forInventory(inventory)

            print("file: \(url.path)")
            print("architecture metadata: \(inventory.architecture ?? "<none>")")
            print("tensors: \(inventory.tensorCount)")
            let counts = inventory.ggmlTypeCounts.sorted { $0.key < $1.key }
            print("ggml types: " + counts.map { "\($0.key)=\($0.value)" }.joined(separator: " "))

            let overridden = inventory.overriddenEntries
            print("tensors needing comfy orig_shape: \(overridden.count)")
            if verbose {
                for entry in overridden.prefix(8) {
                    print("  \(entry.name): stored \(entry.storedShape) -> \(entry.logicalShape)")
                }
                if overridden.count > 8 {
                    print("  … \(overridden.count - 8) more")
                }
            }

            try configuration.validate(against: inventory)
            let expected = configuration.expectedTensorShapes.count
            print("architecture validation: OK (\(expected) required tensors matched)")
            let unreferenced = inventory.tensorCount - expected
            if unreferenced != 0 {
                print("note: \(unreferenced) tensor(s) present but not required by the configuration")
            }
        } catch {
            fail(error.localizedDescription)
        }
    }

    private static func value(for flag: String, in arguments: [String]) -> String? {
        guard let index = arguments.firstIndex(of: flag),
              arguments.index(after: index) < arguments.endIndex else {
            return nil
        }
        return arguments[arguments.index(after: index)]
    }

    private static func usage() {
        let text = """
        GenImageMiniMaxH3Worker inspect --transformer <model.gguf> [--verbose]
        GenImageMiniMaxH3Worker text-check --text-encoder <qwen3vl.gguf> --tokenizer <processor directory> [--prompt <text>] [--token-ids <id,id,...>] [--layers <count>] [--dense] [--cpu-quantizer] [--output out.safetensors]
        GenImageMiniMaxH3Worker quant-check --transformer <model.gguf> [--cpu-quantizer]
        GenImageMiniMaxH3Worker load-check --transformer <model.gguf> [--cpu-quantizer]
        GenImageMiniMaxH3Worker decode-video --vae <video_vae.safetensors> [--latent in.safetensors] [--output out.safetensors]
        GenImageMiniMaxH3Worker generate --transformer <h3.gguf> --video-vae <video_vae.safetensors> --audio-vae <audio_vae.safetensors> --text-encoder <qwen3vl.gguf> --tokenizer <processor directory> --prompt <text> --output out.mp4 [--input image] [--latent-frames 3] [--latent-height 16] [--latent-width 16] [--audio-frames 8] [--frame-rate 24] [--steps 8] [--seed 0]
        """
        FileHandle.standardError.write(Data((text + "\n").utf8))
    }

    private static func fail(_ message: String) -> Never {
        FileHandle.standardError.write(Data(("錯誤：" + message + "\n").utf8))
        exit(1)
    }
}

GenImageMiniMaxH3Worker.main()

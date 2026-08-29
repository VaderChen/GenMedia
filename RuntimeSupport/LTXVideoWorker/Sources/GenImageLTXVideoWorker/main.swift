import Darwin
import Foundation
import LTXVideoSwiftRuntime
import MLX
import Tokenizers

private struct WorkerRequest: Decodable {
    struct LoRA: Decodable {
        let path: String
        let scale: Double
        let conditioningScale: Double?

        enum CodingKeys: String, CodingKey {
            case path
            case scale
            case conditioningScale = "conditioning_scale"
        }
    }

    let modelDirectory: String
    let outputPath: String
    let prompt: String
    let width: Int
    let height: Int
    let frames: Int
    let frameRate: Float
    let seed: UInt64
    let stage1Steps: Int
    let stage2Steps: Int
    let imagePaths: [String]
    let loras: [LoRA]
    let gemmaDirectory: String?

    enum CodingKeys: String, CodingKey {
        case modelDirectory
        case outputPath
        case prompt
        case width
        case height
        case frames
        case frameRate
        case seed
        case stage1Steps
        case stage2Steps
        case imagePaths
        case loras
        case gemmaDirectory
    }

    init(from decoder: Swift.Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        modelDirectory = try values.decode(String.self, forKey: .modelDirectory)
        outputPath = try values.decode(String.self, forKey: .outputPath)
        prompt = try values.decode(String.self, forKey: .prompt)
        width = try values.decode(Int.self, forKey: .width)
        height = try values.decode(Int.self, forKey: .height)
        frames = try values.decode(Int.self, forKey: .frames)
        frameRate = try values.decodeIfPresent(Float.self, forKey: .frameRate) ?? 24
        seed = try values.decode(UInt64.self, forKey: .seed)
        stage1Steps = try values.decodeIfPresent(Int.self, forKey: .stage1Steps) ?? 8
        stage2Steps = try values.decodeIfPresent(Int.self, forKey: .stage2Steps) ?? 3
        imagePaths = try values.decodeIfPresent([String].self, forKey: .imagePaths) ?? []
        loras = try values.decodeIfPresent([LoRA].self, forKey: .loras) ?? []
        gemmaDirectory = try values.decodeIfPresent(String.self, forKey: .gemmaDirectory)
    }
}

private struct WorkerEvent: Encodable {
    let type: String
    let stage: String?
    let value: Double?
    let durationSeconds: Double?
    let sampleRate: Int?
    let numFrames: Int?
    let numChunks: Int?
    let pixelWidth: Int?
    let pixelHeight: Int?
    let message: String?

    static func progress(stage: String, value: Double) -> Self {
        Self(
            type: "progress",
            stage: stage,
            value: min(1, max(0, value)),
            durationSeconds: nil,
            sampleRate: nil,
            numFrames: nil,
            numChunks: nil,
            pixelWidth: nil,
            pixelHeight: nil,
            message: nil
        )
    }

    static func completed(
        durationSeconds: Double,
        sampleRate: Int,
        numFrames: Int,
        pixelWidth: Int,
        pixelHeight: Int
    ) -> Self {
        Self(
            type: "completed",
            stage: nil,
            value: 1,
            durationSeconds: durationSeconds,
            sampleRate: sampleRate,
            numFrames: numFrames,
            numChunks: nil,
            pixelWidth: pixelWidth,
            pixelHeight: pixelHeight,
            message: nil
        )
    }

    static func error(_ message: String) -> Self {
        Self(
            type: "error",
            stage: nil,
            value: nil,
            durationSeconds: nil,
            sampleRate: nil,
            numFrames: nil,
            numChunks: nil,
            pixelWidth: nil,
            pixelHeight: nil,
            message: message
        )
    }
}

private enum WorkerError: LocalizedError {
    case usage
    case invalidRequest(String)
    case modelIncomplete(URL, [String])
    case missingInput(URL)
    case unsupportedImageConditioning
    case unsupportedLoRA
    case missingOutput
    case invalidTensor(String)
    case ffmpegNotFound
    case ffmpegFailed(Int32, String)

    var errorDescription: String? {
        switch self {
        case .usage:
            "用法：GenImageLTXVideoWorker --request <request.json>"
        case let .invalidRequest(message):
            "LTX Worker 請求無效：\(message)"
        case let .modelIncomplete(url, missing):
            "LTX-2.3 原生模型不完整：\(url.path)；缺少 \(missing.joined(separator: "、"))"
        case let .missingInput(url):
            "找不到 LTX Worker 輸入檔案：\(url.path)"
        case .unsupportedImageConditioning:
            "目前原生 LTX Swift Worker 尚未提供 image conditioning；不會退回其他 Runtime。"
        case .unsupportedLoRA:
            "目前原生 LTX Swift Worker 尚未提供 LoRA fusion；不會退回其他 Runtime。"
        case .missingOutput:
            "LTX Swift Worker 完成但沒有產生影片。"
        case let .invalidTensor(message):
            "LTX Swift Worker 產生無效張量：\(message)"
        case .ffmpegNotFound:
            "找不到內建或系統 FFmpeg，無法封裝 LTX 影片。"
        case let .ffmpegFailed(status, message):
            "LTX 影片 FFmpeg 封裝失敗（\(status)）：\(message)"
        }
    }
}

@main
private enum GenImageLTXVideoWorker {
    private static let computeDType: LTXVideoComputeDType = .bfloat16
    private static let gemmaMaxLength = 1024

    static func main() async {
        do {
            let requestURL = try requestURL(from: CommandLine.arguments)
            let request = try JSONDecoder().decode(
                WorkerRequest.self,
                from: Data(contentsOf: requestURL)
            )
            try await run(request)
        } catch {
            emit(.error(error.localizedDescription))
            Darwin.exit(EXIT_FAILURE)
        }
    }

    private static func run(_ request: WorkerRequest) async throws {
        try validate(request)
        let modelDirectory = URL(fileURLWithPath: request.modelDirectory, isDirectory: true)
        let outputURL = URL(fileURLWithPath: request.outputPath)
        let missing = requiredModelFiles().filter {
            !FileManager.default.fileExists(atPath: modelDirectory.appendingPathComponent($0).path)
        }
        guard missing.isEmpty else {
            throw WorkerError.modelIncomplete(modelDirectory, missing)
        }
        for path in request.imagePaths {
            let url = URL(fileURLWithPath: path)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw WorkerError.missingInput(url)
            }
        }
        guard request.imagePaths.isEmpty else {
            throw WorkerError.unsupportedImageConditioning
        }
        guard request.loras.isEmpty else {
            throw WorkerError.unsupportedLoRA
        }

        let gemmaDirectory = try resolveGemmaDirectory(
            request.gemmaDirectory,
            modelDirectory: modelDirectory
        )
        let gemmaMissing = [
            "config.json",
            "tokenizer.json",
            "model.safetensors.index.json"
        ].filter {
            !FileManager.default.fileExists(atPath: gemmaDirectory.appendingPathComponent($0).path)
        }
        guard gemmaMissing.isEmpty else {
            throw WorkerError.modelIncomplete(gemmaDirectory, gemmaMissing)
        }

        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        emit(.progress(stage: "loadingModel", value: 0.01))

        let textEmbeds = try await encodePrompt(
            request.prompt,
            gemmaDirectory: gemmaDirectory,
            modelDirectory: modelDirectory
        )
        emit(.progress(stage: "encodingPrompt", value: 0.08))

        let generated = try runDiffusion(
            textEmbeds: textEmbeds,
            request: request,
            modelDirectory: modelDirectory
        )
        emit(.progress(stage: "denoising", value: 0.76))

        let audio = try decodeAudio(generated.audioLatent, modelDirectory: modelDirectory)
        emit(.progress(stage: "audioDecoding", value: 0.84))
        MLX.eval(audio)

        let temporaryWAV = outputURL
            .deletingLastPathComponent()
            .appendingPathComponent(".ltx-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: temporaryWAV) }
        let audioInfo = try writeWAV(audio, sampleRate: 48_000, to: temporaryWAV)

        let video = try decodeVideo(generated.videoLatent, modelDirectory: modelDirectory)
        emit(.progress(stage: "videoDecoding", value: 0.92))
        MLX.eval(video)
        try muxVideo(
            video,
            audioURL: temporaryWAV,
            frameRate: request.frameRate,
            outputURL: outputURL,
            progress: { value in
                emit(.progress(stage: "encoding", value: 0.92 + value * 0.07))
            }
        )
        let outputFileSize = try? outputURL.resourceValues(forKeys: [.fileSizeKey]).fileSize
        guard FileManager.default.fileExists(atPath: outputURL.path),
              (outputFileSize ?? 0) > 0 else {
            throw WorkerError.missingOutput
        }

        emit(
            .completed(
                durationSeconds: audioInfo.durationSeconds,
                sampleRate: audioInfo.sampleRate,
                numFrames: generated.frames,
                pixelWidth: generated.width,
                pixelHeight: generated.height
            )
        )
    }

    private static func encodePrompt(
        _ prompt: String,
        gemmaDirectory: URL,
        modelDirectory: URL
    ) async throws -> (video: MLXArray, audio: MLXArray) {
        let tokenizer = try await AutoTokenizer.from(modelFolder: gemmaDirectory)
        let tokenIDs = tokenizer.encode(text: prompt.trimmingCharacters(in: .whitespacesAndNewlines))
        let padTokenID = tokenizer.convertTokenToId("<pad>") ?? 0
        let layout = try LTXGemmaFeaturePreparation.leftPad(
            tokenIDs: tokenIDs,
            maxLength: gemmaMaxLength,
            padTokenID: padTokenID
        )
        let tokenArray = MLXArray(layout.tokenIDs.map(Int32.init), [1, gemmaMaxLength])
        let attentionMask = MLXArray(layout.attentionMask.map(Int32.init), [1, gemmaMaxLength])

        let loadedText = try LTXGemma3TextWeightLoader.load(
            from: gemmaDirectory,
            computeDType: computeDType
        )
        let hiddenStates = try loadedText.model.allHiddenStates(
            tokenIDs: tokenArray,
            attentionMask: attentionMask
        )
        let stacked = try LTXGemmaFeaturePreparation.stackForProjection(
            hiddenStates,
            attentionMask: attentionMask
        )
        let connector = LTXGemmaTextEncoderConnector(
            configuration: try LTXGemmaConnectorConfiguration()
        )
        _ = try LTXGemmaConnectorWeightLoader.load(
            connector: connector,
            from: modelDirectory,
            computeDType: computeDType
        )
        let embeds = connector(stacked, attentionMask: attentionMask)
        MLX.eval(embeds.video, embeds.audio)
        return embeds
    }

    private static func runDiffusion(
        textEmbeds: (video: MLXArray, audio: MLXArray),
        request: WorkerRequest,
        modelDirectory: URL
    ) throws -> LTXDistilledGenerationResult {
        let transformerURL = modelDirectory.appendingPathComponent(
            "transformer-distilled-1.1.safetensors"
        )
        let transformer = try LTXTransformerWeightLoader.load(
            from: transformerURL,
            modelDirectory: modelDirectory,
            computeDType: computeDType
        ).model
        let videoStatistics = try LTXVideoVAEWeightLoader.loadEncoderStatistics(
            from: modelDirectory,
            computeDType: computeDType
        )
        let upsamplerURL = modelDirectory.appendingPathComponent(
            "spatial_upscaler_x2_v1_1.safetensors"
        )
        let upsampler = try LTXLatentUpsamplerWeightLoader.load(
            from: upsamplerURL,
            modelDirectory: modelDirectory,
            computeDType: computeDType
        ).model
        let configuration = try LTXDistilledGenerationConfiguration(
            width: request.width,
            height: request.height,
            frames: request.frames,
            frameRate: request.frameRate,
            seed: request.seed,
            stage1Steps: request.stage1Steps,
            stage2Steps: request.stage2Steps,
            computeDType: computeDType
        )
        let pipeline = LTXDistilledGenerationPipeline(
            transformer: transformer,
            videoStatistics: videoStatistics,
            upsampler: upsampler
        )
        return try pipeline.generate(
            videoTextEmbeds: textEmbeds.video,
            audioTextEmbeds: textEmbeds.audio,
            configuration: configuration,
            progress: { event in
                let value: Double
                switch event.stage {
                case .stage1Denoising:
                    value = 0.08 + event.value * 0.40
                case .upscaling:
                    value = 0.50
                case .stage2Denoising:
                    value = 0.50 + event.value * 0.26
                }
                emit(.progress(stage: event.stage.rawValue, value: value))
            }
        )
    }

    private static func decodeAudio(
        _ latent: MLXArray,
        modelDirectory: URL
    ) throws -> MLXArray {
        let decoder = LTXAudioVAEDecoder(configuration: try LTXAudioVAEDecoderConfiguration())
        _ = try LTXAudioVAEWeightLoader.loadDecoder(
            model: decoder,
            from: modelDirectory,
            computeDType: computeDType
        )
        let vocoder = LTXVocoderWithBWE()
        _ = try LTXVocoderWeightLoader.load(
            model: vocoder,
            from: modelDirectory,
            computeDType: .float32
        )
        let mel = try decoder.decode(latent)
        MLX.eval(mel)
        return try vocoder.decode(mel)
    }

    private static func decodeVideo(
        _ latent: MLXArray,
        modelDirectory: URL
    ) throws -> MLXArray {
        let decoder = LTXVideoVAEDecoder(
            configuration: try LTXVideoVAEConfiguration.load(from: modelDirectory)
        )
        _ = try LTXVideoVAEWeightLoader.loadDecoder(
            model: decoder,
            from: modelDirectory,
            computeDType: computeDType
        )
        return try decoder.decode(latent, materializeStages: true)
    }

    private static func writeWAV(
        _ audio: MLXArray,
        sampleRate: Int,
        to url: URL
    ) throws -> (sampleRate: Int, durationSeconds: Double) {
        guard audio.ndim == 3, audio.shape[0] == 1, audio.shape[1] > 0, audio.shape[2] > 0 else {
            throw WorkerError.invalidTensor("音訊應為 [1, channels, samples]，實際為 \(audio.shape)。")
        }
        let channels = audio.shape[1]
        let sampleCount = audio.shape[2]
        let values = audio.asType(.float32).asArray(Float.self)
        guard values.allSatisfy(\.isFinite) else {
            throw WorkerError.invalidTensor("音訊含有 NaN 或 Inf。")
        }
        var pcm = Data(capacity: channels * sampleCount * 2)
        for sample in 0..<sampleCount {
            for channel in 0..<channels {
                let value = min(1, max(-1, values[channel * sampleCount + sample]))
                let integer = Int16((value * 32_767).rounded(.towardZero))
                var littleEndian = integer.littleEndian
                withUnsafeBytes(of: &littleEndian) { pcm.append(contentsOf: $0) }
            }
        }

        var data = Data()
        data.append(contentsOf: Array("RIFF".utf8))
        appendUInt32LE(&data, UInt32(36 + pcm.count))
        data.append(contentsOf: Array("WAVE".utf8))
        data.append(contentsOf: Array("fmt ".utf8))
        appendUInt32LE(&data, 16)
        appendUInt16LE(&data, 1)
        appendUInt16LE(&data, UInt16(channels))
        appendUInt32LE(&data, UInt32(sampleRate))
        appendUInt32LE(&data, UInt32(sampleRate * channels * 2))
        appendUInt16LE(&data, UInt16(channels * 2))
        appendUInt16LE(&data, 16)
        data.append(contentsOf: Array("data".utf8))
        appendUInt32LE(&data, UInt32(pcm.count))
        data.append(pcm)
        try data.write(to: url, options: .atomic)
        return (sampleRate, Double(sampleCount) / Double(sampleRate))
    }

    private static func muxVideo(
        _ video: MLXArray,
        audioURL: URL,
        frameRate: Float,
        outputURL: URL,
        progress: (Double) -> Void
    ) throws {
        guard video.ndim == 5, video.shape[0] == 1, video.shape[1] == 3 else {
            throw WorkerError.invalidTensor("影片應為 [1, 3, frames, height, width]，實際為 \(video.shape)。")
        }
        let executable = try ffmpegExecutable()
        let pipe = Pipe()
        let process = Process()
        process.executableURL = executable
        process.arguments = [
            "-hide_banner",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s:v", "\(video.shape[4])x\(video.shape[3])",
            "-r", String(frameRate),
            "-i", "-",
            "-i", audioURL.path,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-shortest",
            "-y",
            outputURL.path
        ]
        process.standardInput = pipe
        process.standardOutput = FileHandle.standardError
        process.standardError = FileHandle.standardError
        try process.run()

        let frameCount = video.shape[2]
        for index in 0..<frameCount {
            let frame = video[0..., 0..., index, 0..., 0...]
                .transposed(0, 2, 3, 1)
                .asType(.float32)
            let values = frame.asArray(Float.self)
            var bytes = [UInt8](repeating: 0, count: video.shape[3] * video.shape[4] * 3)
            for pixel in 0..<video.shape[3] * video.shape[4] {
                for channel in 0..<3 {
                    let value = values[pixel * 3 + channel]
                    guard value.isFinite else {
                        pipe.fileHandleForWriting.closeFile()
                        process.terminate()
                        throw WorkerError.invalidTensor("影片 frame \(index) 含有 NaN 或 Inf。")
                    }
                    let normalized = min(255, max(0, ((value + 1) * 127.5).rounded()))
                    bytes[pixel * 3 + channel] = UInt8(normalized)
                }
            }
            pipe.fileHandleForWriting.write(Data(bytes))
            progress(Double(index + 1) / Double(frameCount))
        }
        pipe.fileHandleForWriting.closeFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw WorkerError.ffmpegFailed(
                process.terminationStatus,
                "FFmpeg 未能建立 \(outputURL.lastPathComponent)。"
            )
        }
    }

    private static func ffmpegExecutable() throws -> URL {
        let environment = ProcessInfo.processInfo.environment
        var candidates: [URL] = []
        for key in ["GENMEDIA_FFMPEG", "GENIMAGE_FFMPEG"] {
            if let path = environment[key], !path.isEmpty {
                candidates.append(URL(fileURLWithPath: path))
            }
        }
        if let executable = Bundle.main.executableURL {
            let contents = executable
                .deletingLastPathComponent()
                .deletingLastPathComponent()
            candidates.append(contents.appendingPathComponent("Resources/bin/ffmpeg"))
        }
        candidates.append(contentsOf: [
            URL(fileURLWithPath: "/opt/homebrew/bin/ffmpeg"),
            URL(fileURLWithPath: "/usr/local/bin/ffmpeg"),
            URL(fileURLWithPath: "/usr/bin/ffmpeg"),
            URL(fileURLWithPath: "/bin/ffmpeg")
        ])
        if let path = environment["PATH"] {
            candidates.append(contentsOf: path.split(separator: ":").map {
                URL(fileURLWithPath: String($0)).appendingPathComponent("ffmpeg")
            })
        }
        guard let executable = candidates.first(where: {
            FileManager.default.isExecutableFile(atPath: $0.path)
        }) else {
            throw WorkerError.ffmpegNotFound
        }
        return executable
    }

    private static func validate(_ request: WorkerRequest) throws {
        guard !request.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw WorkerError.invalidRequest("prompt 不可為空白。")
        }
        guard request.width >= 64, request.height >= 64 else {
            throw WorkerError.invalidRequest("width 與 height 必須至少為 64。")
        }
        guard request.frames > 0, request.frames % 8 == 1 else {
            throw WorkerError.invalidRequest("frames 必須符合 8n+1。")
        }
        guard request.frameRate.isFinite, request.frameRate > 0 else {
            throw WorkerError.invalidRequest("frameRate 必須是正數。")
        }
        guard request.stage1Steps > 0, request.stage2Steps > 0 else {
            throw WorkerError.invalidRequest("stage1Steps 與 stage2Steps 必須是正整數。")
        }
    }

    private static func requiredModelFiles() -> [String] {
        [
            "config.json",
            "embedded_config.json",
            "quantize_config.json",
            "transformer-distilled-1.1.safetensors",
            "connector.safetensors",
            "vae_decoder.safetensors",
            "vae_encoder.safetensors",
            "audio_vae.safetensors",
            "vocoder.safetensors",
            "spatial_upscaler_x2_v1_1.safetensors",
            "spatial_upscaler_x2_v1_1_config.json"
        ]
    }

    private static func resolveGemmaDirectory(
        _ configuredPath: String?,
        modelDirectory: URL
    ) throws -> URL {
        if let configuredPath, !configuredPath.isEmpty {
            let url = URL(fileURLWithPath: configuredPath, isDirectory: true)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw WorkerError.modelIncomplete(url, ["config.json", "tokenizer.json"])
            }
            return url
        }
        let bundled = modelDirectory.appendingPathComponent("gemma-3-12b", isDirectory: true)
        guard FileManager.default.fileExists(atPath: bundled.path) else {
            throw WorkerError.modelIncomplete(bundled, ["config.json", "tokenizer.json"])
        }
        return bundled
    }

    private static func appendUInt16LE(_ data: inout Data, _ value: UInt16) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
    }

    private static func appendUInt32LE(_ data: inout Data, _ value: UInt32) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
    }

    private static func requestURL(from arguments: [String]) throws -> URL {
        guard arguments.count == 3, arguments[1] == "--request" else {
            throw WorkerError.usage
        }
        return URL(fileURLWithPath: arguments[2])
    }

    private static func emit(_ event: WorkerEvent) {
        guard let data = try? JSONEncoder().encode(event) else { return }
        FileHandle.standardOutput.write(data)
        FileHandle.standardOutput.write(Data([0x0A]))
        try? FileHandle.standardOutput.synchronize()
    }
}

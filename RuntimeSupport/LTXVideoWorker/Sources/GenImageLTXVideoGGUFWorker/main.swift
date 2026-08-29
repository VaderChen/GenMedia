import Darwin
import Foundation
import LTXVideoSwiftRuntime
import MLX
import Hub
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
    case unsupportedModelLayout(URL)

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
        case let .unsupportedModelLayout(url):
            "找不到可識別的 LTX GGUF 模型配置：\(url.path)"
        }
    }
}

@main
private enum GenImageLTXVideoGGUFWorker {
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
        let is096 = FileManager.default.fileExists(
            atPath: modelDirectory.appendingPathComponent(
                "ltxv-2b-0.9.6-distilled-04-25-Q4_K_M.gguf"
            ).path
        )
        let missing = requiredModelFiles(is096: is096).filter {
            !FileManager.default.fileExists(atPath: modelDirectory.appendingPathComponent($0).path)
        }
        guard missing.isEmpty else {
            throw WorkerError.modelIncomplete(
                modelDirectory,
                missing
            )
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

        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        emit(.progress(stage: "loadingModel", value: 0.01))

        if is096 {
            try await run096(request, modelDirectory: modelDirectory, outputURL: outputURL)
            return
        }

        let gemmaDirectory = try resolveGemmaDirectory(
            request.gemmaDirectory,
            modelDirectory: modelDirectory
        )
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
        let outputFileSizeValues = try outputURL.resourceValues(forKeys: [.fileSizeKey])
        let outputFileSize = outputFileSizeValues.fileSize ?? 0
        guard FileManager.default.fileExists(atPath: outputURL.path),
              outputFileSize > 0 else {
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
        let gemmaWeightsURL = modelDirectory.appendingPathComponent(
            "text_encoders/gemma-3-12b-it-qat-UD-Q4_K_XL.gguf"
        )
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

        let loadedText = try LTXGemma3TextWeightLoader.loadGGUF(
            from: gemmaWeightsURL,
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
        _ = try LTXGemmaConnectorWeightLoader.loadGGUF(
            connector: connector,
            mainWeightsURL: gemmaWeightsURL,
            connectorWeightsURL: modelDirectory.appendingPathComponent(
                "text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors"
            ),
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
            "distilled-1.1/ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf"
        )
        let transformer = try LTXTransformerGGUFWeightLoader.load(
            from: transformerURL,
            computeDType: computeDType
        ).model
        let videoStatistics = try LTXVideoVAEWeightLoader.loadEncoderStatistics(
            from: modelDirectory,
            computeDType: computeDType
        )
        let upsamplerURL = modelDirectory.appendingPathComponent(
            "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
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
        let configuration: LTXVideoVAEConfiguration
        if FileManager.default.fileExists(
            atPath: modelDirectory.appendingPathComponent("embedded_config.json").path
        ) {
            configuration = try LTXVideoVAEConfiguration.load(from: modelDirectory)
        } else {
            configuration = try LTXVideoVAEConfiguration()
        }
        let decoder = LTXVideoVAEDecoder(configuration: configuration)
        _ = try LTXVideoVAEWeightLoader.loadDecoder(
            model: decoder,
            from: modelDirectory,
            computeDType: computeDType
        )
        return try decoder.decode(latent, materializeStages: true)
    }

    private static func run096(
        _ request: WorkerRequest,
        modelDirectory: URL,
        outputURL: URL
    ) async throws {
        let t5URL = modelDirectory.appendingPathComponent(
            "text_encoder/t5-v1_1-xxl-encoder-Q4_K_M.gguf"
        )
        let tokenizer = try makeTokenizer(from: t5URL)
        let tokenIDs = tokenizer.encode(
            text: request.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        )
        let limitedTokenIDs = Array(tokenIDs.prefix(128))
        guard !limitedTokenIDs.isEmpty else {
            throw WorkerError.invalidRequest("prompt 無法產生 T5 token。")
        }
        let tokenArray = MLXArray(limitedTokenIDs.map(Int32.init), [1, limitedTokenIDs.count])
        let t5 = try LTXVideo096T5WeightLoader.load(from: t5URL).model
        let textFeatures = try t5(tokenArray)
        MLX.eval(textFeatures)
        emit(.progress(stage: "encodingPrompt", value: 0.08))

        let transformerURL = modelDirectory.appendingPathComponent(
            "ltxv-2b-0.9.6-distilled-04-25-Q4_K_M.gguf"
        )
        let transformer = try LTXVideo096TransformerWeightLoader.load(
            from: transformerURL
        ).model
        let dimensions = (
            frames: (request.frames + 7) / 8,
            height: max(1, request.height / 32),
            width: max(1, request.width / 32)
        )
        let tokenCount = dimensions.frames * dimensions.height * dimensions.width
        var latent = MLXRandom.normal(
            [1, 128, dimensions.frames, dimensions.height, dimensions.width],
            dtype: computeDType.mlxDType,
            key: MLXRandom.key(request.seed)
        )
        let positions = LTXPositionBuilder.video(
            frameCount: dimensions.frames,
            height: dimensions.height,
            width: dimensions.width
        )
        let sigmas = try LTXDiffusionScheduler.schedule(
            steps: request.stage1Steps,
            numTokens: tokenCount
        )
        let totalSteps = sigmas.count - 1
        for step in 0..<totalSteps {
            let sigma = sigmas[step]
            let sigmaNext = sigmas[step + 1]
            let timestep = MLXArray([sigma]).asType(latent.dtype)
            let velocity = transformer(
                hiddenStates: latent
                    .transposed(0, 2, 3, 4, 1)
                    .reshaped(1, tokenCount, 128),
                indicesGrid: positions,
                encoderHiddenStates: textFeatures,
                timestep: timestep
            )
            let predicted = velocity
                .reshaped(1, dimensions.frames, dimensions.height, dimensions.width, 128)
                .transposed(0, 4, 1, 2, 3)
            let denoised = LTXDiffusionScheduler.eulerStep(
                sample: latent,
                denoised: latent - predicted * sigma,
                sigma: sigma,
                sigmaNext: sigmaNext
            )
            latent = denoised
            MLX.eval(latent)
            emit(.progress(
                stage: "denoising",
                value: 0.10 + 0.58 * Double(step + 1) / Double(totalSteps)
            ))
        }

        let decoder = LTXVideo096VAEDecoder()
        _ = try LTXVideo096VAEWeightLoader.load(
            model: decoder,
            from: modelDirectory.appendingPathComponent("LTX-Video-0.9.6-VAE-BF16.safetensors"),
            computeDType: computeDType
        )
        let video = try decoder.decode(latent)
        MLX.eval(video)
        emit(.progress(stage: "videoDecoding", value: 0.86))
        try muxVideoOnly(
            video,
            frameRate: request.frameRate,
            outputURL: outputURL,
            progress: { value in
                emit(.progress(stage: "encoding", value: 0.86 + value * 0.13))
            }
        )
        let outputFileSizeValues = try outputURL.resourceValues(forKeys: [.fileSizeKey])
        let outputFileSize = outputFileSizeValues.fileSize ?? 0
        guard FileManager.default.fileExists(atPath: outputURL.path),
              outputFileSize > 0 else {
            throw WorkerError.missingOutput
        }
        emit(.completed(
            durationSeconds: Double(video.shape[2]) / Double(request.frameRate),
            sampleRate: 0,
            numFrames: video.shape[2],
            pixelWidth: video.shape[4],
            pixelHeight: video.shape[3]
        ))
    }

    private static func makeTokenizer(from weightsURL: URL) throws -> Tokenizer {
        let metadata = try LTXGGUFInspector.metadata(from: weightsURL)
        guard let tokens = metadata.stringArrays["tokenizer.ggml.tokens"], !tokens.isEmpty else {
            throw WorkerError.modelIncomplete(weightsURL, ["tokenizer metadata"])
        }
        let scores = metadata.numberArrays["tokenizer.ggml.scores"] ?? []
        let vocabulary: [[Any]] = tokens.enumerated().map { index, token in
            [token, index < scores.count ? scores[index] : 0]
        }
        let specialNames = ["<pad>", "<eos>", "</s>", "<bos>", "<unk>", "<mask>"]
        let addedTokens: [[String: Any]] = tokens.enumerated().compactMap { index, token in
            guard specialNames.contains(token) else { return nil }
            return [
                "id": index,
                "content": token,
                "special": true,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false
            ]
        }
        let rawData: [String: Any] = [
            "model": [
                "type": "Unigram",
                "vocab": vocabulary,
                "unk_id": tokens.firstIndex(of: "<unk>") ?? 0
            ],
            "added_tokens": addedTokens,
            "normalizer": ["type": "Precompiled"],
            "pre_tokenizer": NSNull(),
            "decoder": ["type": "Metaspace", "replacement": "▁"]
        ]
        var tokenizerConfiguration: [String: Any] = [
            "tokenizer_class": "T5Tokenizer",
            "unk_token": tokens.contains("<unk>") ? "<unk>" : "<unk>",
            "clean_up_tokenization_spaces": false
        ]
        if tokens.contains("<pad>") {
            tokenizerConfiguration["pad_token"] = "<pad>"
        }
        if tokens.contains("</s>") {
            tokenizerConfiguration["eos_token"] = "</s>"
        } else if tokens.contains("<eos>") {
            tokenizerConfiguration["eos_token"] = "<eos>"
        }
        if tokens.contains("<bos>") {
            tokenizerConfiguration["bos_token"] = "<bos>"
        }
        let data = try JSONSerialization.data(withJSONObject: [
            "model": rawData["model"]!,
            "added_tokens": rawData["added_tokens"]!,
            "normalizer": rawData["normalizer"]!,
            "pre_tokenizer": rawData["pre_tokenizer"]!,
            "decoder": rawData["decoder"]!
        ])
        let configData = try JSONSerialization.data(withJSONObject: tokenizerConfiguration)
        let dataObject = try JSONSerialization.jsonObject(with: data) as! [NSString: Any]
        let configObject = try JSONSerialization.jsonObject(with: configData) as! [NSString: Any]
        return try AutoTokenizer.from(
            tokenizerConfig: Config(configObject),
            tokenizerData: Config(dataObject),
            strict: false
        )
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
            "-c:v", "h264_videotoolbox",
            "-profile:v", "high",
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

    private static func muxVideoOnly(
        _ video: MLXArray,
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
            "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s:v", "\(video.shape[4])x\(video.shape[3])",
            "-r", String(frameRate), "-i", "-",
            "-c:v", "h264_videotoolbox", "-profile:v", "high",
            "-pix_fmt", "yuv420p", "-y", outputURL.path
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
                    bytes[pixel * 3 + channel] = UInt8(
                        min(255, max(0, ((value + 1) * 127.5).rounded()))
                    )
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

    private static func requiredModelFiles(is096: Bool) -> [String] {
        if is096 {
            return [
                "ltxv-2b-0.9.6-distilled-04-25-Q4_K_M.gguf",
                "text_encoder/t5-v1_1-xxl-encoder-Q4_K_M.gguf",
                "text_encoder/config.json",
                "tokenizer/spiece.model",
                "LTX-Video-0.9.6-VAE-BF16.safetensors"
            ]
        }
        return [
            "distilled-1.1/ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf",
            "text_encoders/gemma-3-12b-it-qat-UD-Q4_K_XL.gguf",
            "text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors",
            "vae/ltx-2.3-22b-distilled_audio_vae.safetensors",
            "vae/ltx-2.3-22b-distilled_video_vae.safetensors",
            "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
        ]
    }

    private static func resolveGemmaDirectory(
        _ configuredPath: String?,
        modelDirectory: URL
    ) throws -> URL {
        let requiredFiles = [
            "config.json",
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json"
        ]
        if let configuredPath, !configuredPath.isEmpty {
            let url = URL(fileURLWithPath: configuredPath, isDirectory: true)
            let missing = requiredFiles.filter {
                !FileManager.default.fileExists(atPath: url.appendingPathComponent($0).path)
            }
            guard missing.isEmpty else {
                throw WorkerError.modelIncomplete(url, missing)
            }
            return url
        }
        let bundled = modelDirectory.appendingPathComponent("gemma-3-12b", isDirectory: true)
        let missing = requiredFiles.filter {
            !FileManager.default.fileExists(atPath: bundled.appendingPathComponent($0).path)
        }
        guard missing.isEmpty else {
            throw WorkerError.modelIncomplete(bundled, missing)
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

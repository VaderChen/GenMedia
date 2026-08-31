import Foundation
import GenImageCore

/// 主系統使用的 MiniMax H3 GGUF 影片服務。
///
/// H3 的純 Swift 核心仍由獨立 Worker 持有；這個 adapter 只負責把 App 的
/// VideoGenerationRequest 轉成明確的 Worker CLI 參數，並把產出的 MP4 收回主系統。
public final class MiniMaxH3VideoGenerationService: VideoGenerating, Sendable {
    private let outputDirectory: URL

    private static let supportedModelIDs: Set<String> = [
        "unsloth/minimax-h3-gguf@fl2va-pruned-q4_k",
        "abiray/minimax-h3-gguf@fl2va-q4_0",
        "abiray/minimax-h3-gguf@fl2va-q4_k_m",
        "abiray/minimax-h3-gguf@fl2va-q4_k_s",
        "abiray/minimax-h3-gguf@ref2va-q4_0",
        "abiray/minimax-h3-gguf@ref2va-q4_k_m",
        "abiray/minimax-h3-gguf@ref2va-q4_k_s"
    ]

    public init(outputDirectory: URL) {
        self.outputDirectory = outputDirectory
    }

    public static func isH3ModelID(_ modelID: String) -> Bool {
        modelID.lowercased().contains("minimax-h3")
    }

    public static func isSupportedModelID(_ modelID: String) -> Bool {
        supportedModelIDs.contains(modelID.lowercased())
    }

    private static func isRef2VAModelID(_ modelID: String) -> Bool {
        modelID.lowercased().contains("@ref2va-")
    }

    public static func normalizedFrameCount(_ frameCount: Int) -> Int {
        let clamped = min(max(frameCount, 1), 512)
        let lower = max(4, (clamped / 4) * 4)
        let upper = min(512, lower + 4)
        guard upper != lower else { return lower }
        return clamped - lower < upper - clamped ? lower : upper
    }

    public func generate(
        request: VideoGenerationRequest,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> [MediaAsset] {
        progress(0.01)
        try request.options.validate()
        let isImageToVideo = request.profile.capability == .imageToVideo
        guard isImageToVideo || request.profile.capability == .textToVideo else {
            throw MiniMaxH3VideoRuntimeError.unsupportedImageConditioning
        }
        guard request.profile.architecture == .externalCLI else {
            throw MiniMaxH3VideoRuntimeError.unsupportedArchitecture(request.profile.architecture)
        }
        guard Self.isSupportedModelID(request.profile.modelID) else {
            throw MiniMaxH3VideoRuntimeError.unsupportedModel(request.profile.modelID)
        }
        if isImageToVideo && Self.isRef2VAModelID(request.profile.modelID) {
            throw MiniMaxH3VideoRuntimeError.unsupportedImageConditioning
        }

        let inputImageURL: URL?
        if isImageToVideo {
            guard request.sourceAssets.count == 1 else {
                throw MiniMaxH3VideoRuntimeError.unsupportedImageCount(request.sourceAssets.count)
            }
            guard let sourceAsset = request.sourceAssets.first,
                  sourceAsset.kind.isImage,
                  let sourceURL = sourceAsset.fileURL,
                  FileManager.default.fileExists(atPath: sourceURL.path) else {
                throw MiniMaxH3VideoRuntimeError.missingImageInput
            }
            inputImageURL = sourceURL
        } else {
            guard request.sourceAssets.isEmpty else {
                throw MiniMaxH3VideoRuntimeError.unsupportedImageConditioning
            }
            inputImageURL = nil
        }
        guard request.profileLoRAs.isEmpty else {
            throw MiniMaxH3VideoRuntimeError.unsupportedLoRA
        }

        let modelFiles = try Self.modelFiles(
            for: request.profile.modelID,
            modelDirectory: request.modelURL
        )
        let frameCount = Self.normalizedFrameCount(request.options.frameCount)
        let latentFrames = frameCount / 4
        let latentHeight = request.options.height / 16
        let latentWidth = request.options.width / 16
        let executable = try Self.workerExecutable()

        try FileManager.default.createDirectory(
            at: outputDirectory,
            withIntermediateDirectories: true
        )

        var outputs: [MediaAsset] = []
        var generatedOutputURLs: [URL] = []
        var completed = false
        defer {
            if !completed {
                for outputURL in generatedOutputURLs {
                    try? FileManager.default.removeItem(at: outputURL)
                }
            }
        }

        for index in 0 ..< request.options.outputCount {
            try Task.checkCancellation()
            let identifier = UUID().uuidString
            let outputURL = OutputFileNaming.videoURL(in: outputDirectory, pathExtension: "mp4")
            let logURL = outputDirectory.appendingPathComponent("minimax-h3-\(identifier).log")
            generatedOutputURLs.append(outputURL)

            let log = try RuntimeLog(at: logURL)
            defer { log.close() }
            var activity = RuntimeLogActivity(log: log)
            let generationSpan = 1.0 / Double(request.options.outputCount)
            let completedFraction = Double(index) * generationSpan
            var workerArguments = [
                "generate",
                "--transformer", modelFiles.transformer.path,
                "--video-vae", modelFiles.videoVAE.path,
                "--audio-vae", modelFiles.audioVAE.path,
                "--text-encoder", modelFiles.textEncoder.path,
                "--tokenizer", modelFiles.processor.path,
                "--prompt", request.options.prompt,
                "--output", outputURL.path,
                "--latent-frames", String(latentFrames),
                "--latent-height", String(latentHeight),
                "--latent-width", String(latentWidth),
                "--audio-frames", String(max(8, request.options.frameCount / 16)),
                "--frame-rate", String(request.options.frameRate),
                "--steps", String(request.options.steps),
                "--seed", String(request.options.seed &+ UInt64(index))
            ]
            if let inputImageURL {
                workerArguments += ["--input", inputImageURL.path]
            }

            let status = try await RuntimeProcess.run(
                executable: executable,
                arguments: workerArguments,
                environment: RuntimeExecutable.environment(),
                log: log,
                pollInterval: .milliseconds(300)
            ) {
                if !activity.sample(log), activity.idleDuration >= 30 * 60 {
                    throw MiniMaxH3VideoRuntimeError.runtimeStalled(
                        details: log.lastLine(fallback: "H3 Worker 未提供最後狀態。")
                    )
                }
                if let workerProgress = Self.latestProgress(in: log) {
                    progress(completedFraction + workerProgress * generationSpan)
                }
            }

            guard status == 0 else {
                throw MiniMaxH3VideoRuntimeError.runtimeFailed(
                    status: status,
                    message: log.message(maximumBytes: 8_192, fallback: "H3 Worker 執行失敗。")
                )
            }
            guard FileManager.default.fileExists(atPath: outputURL.path),
                  RuntimeLog.fileSize(at: outputURL) > 0 else {
                throw MiniMaxH3VideoRuntimeError.outputMissing(outputURL)
            }

            outputs.append(
                MediaAsset(
                    projectID: request.projectID,
                    parentAssetID: request.sourceAsset?.id,
                    kind: .generatedVideo,
                    title: request.options.outputCount == 1
                        ? "MiniMax H3 生成影片"
                        : "MiniMax H3 生成影片 \(index + 1)",
                    fileURL: outputURL,
                    pixelWidth: request.options.width,
                    pixelHeight: request.options.height,
                    recipeID: request.recipeID
                )
            )
            progress(completedFraction + generationSpan)
        }

        completed = true
        return outputs
    }

    private struct ModelFiles {
        let transformer: URL
        let videoVAE: URL
        let audioVAE: URL
        let textEncoder: URL
        let processor: URL
    }

    private static func modelFiles(
        for modelID: String,
        modelDirectory: URL
    ) throws -> ModelFiles {
        let normalizedID = modelID.lowercased()
        let variant: String
        let usesPrunedLayout = normalizedID ==
            "unsloth/minimax-h3-gguf@fl2va-pruned-q4_k"
        switch normalizedID {
        case "unsloth/minimax-h3-gguf@fl2va-pruned-q4_k":
            variant = "pruned-Q4_K"
        case "abiray/minimax-h3-gguf@fl2va-q4_0":
            variant = "FL2VA-Q4_0"
        case "abiray/minimax-h3-gguf@fl2va-q4_k_m":
            variant = "FL2VA-Q4_K_M"
        case "abiray/minimax-h3-gguf@fl2va-q4_k_s":
            variant = "FL2VA-Q4_K_S"
        case "abiray/minimax-h3-gguf@ref2va-q4_0":
            variant = "Ref2VA-Q4_0"
        case "abiray/minimax-h3-gguf@ref2va-q4_k_m":
            variant = "Ref2VA-Q4_K_M"
        case "abiray/minimax-h3-gguf@ref2va-q4_k_s":
            variant = "Ref2VA-Q4_K_S"
        default:
            throw MiniMaxH3VideoRuntimeError.unsupportedModel(modelID)
        }

        let files = ModelFiles(
            transformer: usesPrunedLayout
                ? modelDirectory.appendingPathComponent(
                    "minimax_h3_fl2va_pruned-Q4_K.gguf"
                )
                : modelDirectory.appendingPathComponent(
                    "unet/MiniMax-H3-\(variant).gguf"
                ),
            videoVAE: modelDirectory.appendingPathComponent(
                "vae/minimax_h3_video_vae_fp16.safetensors"
            ),
            audioVAE: modelDirectory.appendingPathComponent(
                "vae/minimax_h3_audio_vae_fp32.safetensors"
            ),
            textEncoder: modelDirectory.appendingPathComponent(
                usesPrunedLayout
                    ? "qwen3vl_32b_minimax_h3-Q4_K_M.gguf"
                    : "text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf"
            ),
            processor: Self.processorDirectory(in: modelDirectory)
        )
        let required = [
            files.transformer,
            files.videoVAE,
            files.audioVAE,
            files.textEncoder,
            files.processor.appendingPathComponent("tokenizer.json"),
            files.processor.appendingPathComponent("tokenizer_config.json")
        ]
        guard required.allSatisfy({ path in
            var isDirectory: ObjCBool = false
            return FileManager.default.fileExists(atPath: path.path, isDirectory: &isDirectory)
                && !isDirectory.boolValue
                && FileManager.default.isReadableFile(atPath: path.path)
        }) else {
            throw MiniMaxH3VideoRuntimeError.modelNotInstalled(modelDirectory)
        }
        return files
    }

    private static func processorDirectory(in modelDirectory: URL) -> URL {
        let candidates = [
            modelDirectory.appendingPathComponent("upstream/FL2VA/processor"),
            modelDirectory.appendingPathComponent("upstream/FL2VA/tokenizer")
        ]
        return candidates.first { directory in
            FileManager.default.fileExists(
                atPath: directory.appendingPathComponent("tokenizer.json").path
            )
        } ?? candidates[0]
    }

    private static func workerExecutable() throws -> URL {
        let environment = ProcessInfo.processInfo.environment
        var candidates: [URL] = []
        if let configured = environment["GENIMAGE_MINIMAX_H3_WORKER"], !configured.isEmpty {
            candidates.append(URL(fileURLWithPath: configured))
        }
        if let executableDirectory = Bundle.main.executableURL?.deletingLastPathComponent() {
            candidates.append(executableDirectory.appendingPathComponent("GenImageMiniMaxH3Worker"))
            candidates.append(
                executableDirectory
                    .deletingLastPathComponent()
                    .appendingPathComponent("Helpers", isDirectory: true)
                    .appendingPathComponent("GenImageMiniMaxH3Worker")
            )
        }

        let sourceRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let packageRoot = sourceRoot
            .appendingPathComponent("RuntimeSupport", isDirectory: true)
            .appendingPathComponent("MiniMaxH3Worker", isDirectory: true)
        candidates.append(
            packageRoot
                .appendingPathComponent(".build", isDirectory: true)
                .appendingPathComponent("release", isDirectory: true)
                .appendingPathComponent("GenImageMiniMaxH3Worker")
        )
        candidates.append(
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
                .appendingPathComponent(
                    "RuntimeSupport/MiniMaxH3Worker/.build/release/GenImageMiniMaxH3Worker"
                )
        )

        guard let executable = RuntimeExecutable.locate(candidates) else {
            throw MiniMaxH3VideoRuntimeError.workerNotFound(candidates.map(\.path))
        }
        return executable
    }

    private static func latestProgress(in log: RuntimeLog) -> Double? {
        guard let data = log.data(),
              let text = String(data: data, encoding: .utf8) else { return nil }
        for line in text.split(whereSeparator: { $0 == "\n" || $0 == "\r" }).reversed() {
            let parts = line.split(separator: " ")
            guard parts.count >= 2,
                  let token = parts.last,
                  token.hasSuffix("%"),
                  let percent = Double(token.dropLast()) else { continue }
            let fraction = min(max(percent / 100, 0), 1)
            switch parts[parts.count - 2] {
            case "loadingTextEncoder": return fraction * 0.05
            case "loadingTransformer": return 0.05 + fraction * 0.10
            case "denoising": return 0.15 + fraction * 0.65
            case "videoDecoding": return 0.80 + fraction * 0.15
            case "audioDecoding": return 0.95 + fraction * 0.05
            default: return fraction
            }
        }
        return nil
    }
}

/// 根據模型 Profile 把影片生成分派到既有 LTX 或 MiniMax H3 Worker。
public final class VideoGenerationRouter: VideoGenerating, Sendable {
    private let ltxService: LTXVideoGenerationService
    private let h3Service: MiniMaxH3VideoGenerationService

    public init(outputDirectory: URL) {
        ltxService = LTXVideoGenerationService(outputDirectory: outputDirectory)
        h3Service = MiniMaxH3VideoGenerationService(outputDirectory: outputDirectory)
    }

    public func generate(
        request: VideoGenerationRequest,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> [MediaAsset] {
        if MiniMaxH3VideoGenerationService.isH3ModelID(request.profile.modelID) {
            return try await h3Service.generate(request: request, progress: progress)
        }
        return try await ltxService.generate(request: request, progress: progress)
    }
}

public enum MiniMaxH3VideoRuntimeError: LocalizedError, Sendable {
    case unsupportedImageConditioning
    case unsupportedImageCount(Int)
    case missingImageInput
    case unsupportedArchitecture(InferenceArchitecture)
    case unsupportedModel(String)
    case unsupportedLoRA
    case modelNotInstalled(URL)
    case workerNotFound([String])
    case runtimeFailed(status: Int32, message: String)
    case runtimeStalled(details: String)
    case outputMissing(URL)

    public var errorDescription: String? {
        switch self {
        case .unsupportedImageConditioning:
            "MiniMax H3 目前僅支援 FL2VA 文生影與單一圖片圖生影。"
        case let .unsupportedImageCount(count):
            "MiniMax H3 FL2VA 目前只接受一張圖片錨點，實際收到 \(count) 張。"
        case .missingImageInput:
            "MiniMax H3 圖生影找不到可讀取的圖片錨點。"
        case let .unsupportedArchitecture(architecture):
            "MiniMax H3 Worker 不支援此架構：\(architecture.title)。"
        case let .unsupportedModel(modelID):
            "目前 MiniMax H3 Swift Runtime 尚不支援模型：\(modelID)。"
        case .unsupportedLoRA:
            "MiniMax H3 Swift Worker 尚未提供 LoRA fusion。"
        case let .modelNotInstalled(url):
            "MiniMax H3 模型或配套檔案尚未完整安裝：\(url.path)"
        case let .workerNotFound(paths):
            "找不到 MiniMax H3 Runtime Worker；已檢查：\(paths.joined(separator: "、"))"
        case let .runtimeFailed(status, message):
            "MiniMax H3 Worker 結束（\(status)）：\(message)"
        case let .runtimeStalled(details):
            "MiniMax H3 Worker 超過 30 分鐘沒有輸出新進度，已自動停止。最後狀態：\(details)"
        case let .outputMissing(url):
            "MiniMax H3 Worker 完成但沒有產生影片：\(url.path)"
        }
    }
}

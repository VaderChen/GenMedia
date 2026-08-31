import Foundation
import GenImageCore

public final class LTXVideoGenerationService: VideoGenerating, Sendable {
    private let outputDirectory: URL

    public static func isValidFrameCount(_ frameCount: Int) -> Bool {
        (1...512).contains(frameCount) && frameCount % 8 == 1
    }

    public static func normalizedFrameCount(_ frameCount: Int) -> Int {
        let clamped = min(max(frameCount, 1), 512)
        let lower = ((clamped - 1) / 8) * 8 + 1
        let upper = lower + 8 <= 512 ? lower + 8 : lower
        guard upper != lower else { return lower }
        return clamped - lower < upper - clamped ? lower : upper
    }

    public init(outputDirectory: URL) {
        self.outputDirectory = outputDirectory
    }

    public func generate(
        request: VideoGenerationRequest,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> [MediaAsset] {
        progress(0.01)
        try request.options.validate()
        guard request.profile.capability == .textToVideo
                || request.profile.capability == .imageToVideo else {
            throw LTXVideoRuntimeError.incompatibleProfile
        }
        guard request.profile.architecture == .externalCLI else {
            throw LTXVideoRuntimeError.unsupportedArchitecture(request.profile.architecture)
        }
        guard let workerKind = Self.workerKind(for: request.profile.modelID) else {
            throw LTXVideoRuntimeError.unsupportedModel(request.profile.modelID)
        }
        guard Self.isValidFrameCount(request.options.frameCount) else {
            throw LTXVideoRuntimeError.invalidFrameCount(request.options.frameCount)
        }

        let sourceAssets = Self.uniqueSourceAssets(request.sourceAssets)
        let sourcePaths = try Self.sourcePaths(
            sourceAssets,
            capability: request.profile.capability,
            frameCount: request.options.frameCount
        )
        let loras = try Self.validatedLoRAs(request.profileLoRAs)
        guard sourcePaths.isEmpty else {
            throw LTXVideoRuntimeError.unsupportedImageConditioning
        }
        guard loras.isEmpty else {
            throw LTXVideoRuntimeError.unsupportedLoRA
        }

        let manifestURL = request.modelURL.appendingPathComponent("genimage-model.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw LTXVideoRuntimeError.modelNotInstalled(request.modelURL)
        }

        let executable = try Self.workerExecutable(for: workerKind)
        try FileManager.default.createDirectory(
            at: outputDirectory,
            withIntermediateDirectories: true
        )

        var outputs: [MediaAsset] = []
        var generatedOutputURLs: [URL] = []
        var completed = false
        let legacyWorkerOverride = ProcessInfo.processInfo.environment["GENIMAGE_LTX_GGUF_WORKER"]
        var didWarnLegacyWorkerOverride = false
        defer {
            if !completed {
                for outputURL in generatedOutputURLs {
                    try? FileManager.default.removeItem(at: outputURL)
                }
            }
        }

        for index in 0..<request.options.outputCount {
            try Task.checkCancellation()
            let identifier = UUID().uuidString
            let outputURL = OutputFileNaming.videoURL(in: outputDirectory, pathExtension: "mp4")
            let requestURL = outputDirectory.appendingPathComponent(
                "ltx-video-\(identifier)-request.json"
            )
            let logURL = outputDirectory.appendingPathComponent("ltx-video-\(identifier).log")
            generatedOutputURLs.append(outputURL)
            defer {
                try? FileManager.default.removeItem(at: requestURL)
                try? FileManager.default.removeItem(at: logURL)
            }

            let payload = WorkerRequest(
                modelDirectory: request.modelURL.path,
                outputPath: outputURL.path,
                prompt: request.options.prompt,
                width: request.options.width,
                height: request.options.height,
                frames: request.options.frameCount,
                frameRate: request.options.frameRate,
                seed: request.options.seed &+ UInt64(index),
                stage1Steps: request.options.steps,
                stage2Steps: 3,
                imagePaths: sourcePaths,
                loras: loras,
                gemmaDirectory: Self.gemmaDirectory()
            )
            try JSONEncoder().encode(payload).write(to: requestURL, options: .atomic)

            let log = try RuntimeLog(at: logURL)
            defer { log.close() }
            if !didWarnLegacyWorkerOverride,
               let legacyWorkerOverride,
               !legacyWorkerOverride.isEmpty {
                let warning = "[warning] GENIMAGE_LTX_GGUF_WORKER 已失效；合併後請改用 GENIMAGE_LTX_WORKER。已忽略：\(legacyWorkerOverride)\n"
                try? log.handle.write(contentsOf: Data(warning.utf8))
                log.flush()
                didWarnLegacyWorkerOverride = true
            }
            var activity = RuntimeLogActivity(log: log)
            let generationSpan = 1.0 / Double(request.options.outputCount)
            let completedFraction = Double(index) * generationSpan
            let status = try await RuntimeProcess.run(
                executable: executable,
                arguments: [
                    "--request", requestURL.path,
                    "--format", workerKind.formatArgument,
                    "--variant", workerKind.variantArgument
                ],
                environment: RuntimeExecutable.environment(),
                log: log,
                pollInterval: .milliseconds(300)
            ) {
                if !activity.sample(log), activity.idleDuration >= 30 * 60 {
                    throw LTXVideoRuntimeError.runtimeStalled(
                        details: log.lastLine(fallback: "Worker 未提供最後狀態。")
                    )
                }
                if let workerProgress = Self.latestProgress(in: log) {
                    progress(completedFraction + workerProgress * generationSpan)
                }
            }

            guard status == 0 else {
                throw LTXVideoRuntimeError.runtimeFailed(
                    status: status,
                    message: Self.logMessage(in: log)
                )
            }
            guard FileManager.default.fileExists(atPath: outputURL.path),
                  RuntimeLog.fileSize(at: outputURL) > 0 else {
                throw LTXVideoRuntimeError.outputMissing(outputURL)
            }

            let completion = Self.completionMetadata(in: log)

            outputs.append(
                MediaAsset(
                    projectID: request.projectID,
                    parentAssetID: request.sourceAsset?.id,
                    kind: .generatedVideo,
                    title: request.options.outputCount == 1
                        ? "生成影片"
                        : "生成影片 \(index + 1)",
                    fileURL: outputURL,
                    pixelWidth: completion?.pixelWidth ?? request.options.width,
                    pixelHeight: completion?.pixelHeight ?? request.options.height,
                    recipeID: request.recipeID
                )
            )
            progress(completedFraction + generationSpan)
        }
        completed = true
        return outputs
    }

    private static func uniqueSourceAssets(_ assets: [MediaAsset]) -> [MediaAsset] {
        var seen = Set<UUID>()
        return assets.filter { seen.insert($0.id).inserted }
    }

    private static func sourcePaths(
        _ assets: [MediaAsset],
        capability: ModelCapability,
        frameCount: Int
    ) throws -> [String] {
        guard capability == .imageToVideo else { return [] }
        guard !assets.isEmpty else {
            throw LTXVideoRuntimeError.missingInputFile
        }
        guard assets.count <= frameCount else {
            throw LTXVideoRuntimeError.tooManyImageAnchors(
                count: assets.count,
                frameCount: frameCount
            )
        }
        return try assets.map { asset in
            guard let url = asset.fileURL,
                  FileManager.default.fileExists(atPath: url.path) else {
                throw LTXVideoRuntimeError.missingInputFile
            }
            return url.path
        }
    }

    private static func validatedLoRAs(
        _ values: [VideoGenerationLoRA]
    ) throws -> [WorkerRequest.LoRA] {
        try values.map { lora in
            guard FileManager.default.fileExists(atPath: lora.localURL.path) else {
                throw LTXVideoRuntimeError.loraNotInstalled(lora.localURL)
            }
            guard lora.scale.isFinite, (0...1).contains(lora.scale),
                  lora.conditioningScale.isFinite,
                  (0...1).contains(lora.conditioningScale) else {
                throw LTXVideoRuntimeError.invalidLoRAScale(lora.modelID)
            }
            return WorkerRequest.LoRA(
                path: lora.localURL.path,
                scale: lora.scale,
                conditioningScale: lora.conditioningScale
            )
        }
    }

    private static func gemmaDirectory() -> String? {
        guard let value = ProcessInfo.processInfo.environment["GENIMAGE_LTX_GEMMA_MODEL"],
              !value.isEmpty else { return nil }
        return value
    }

    private enum WorkerKind {
        case mlx
        case gguf096
        case gguf23

        var formatArgument: String {
            switch self {
            case .mlx: "mlx"
            case .gguf096, .gguf23: "gguf"
            }
        }

        var variantArgument: String {
            switch self {
            case .mlx, .gguf23: "ltx-2.3"
            case .gguf096: "ltx-0.9.6"
            }
        }
    }

    private static func workerKind(for modelID: String) -> WorkerKind? {
        switch modelID.lowercased() {
        case "dgrauet/ltx-2.3-mlx-q4":
            .mlx
        case "city96/ltx-video-0.9.6-distilled-gguf@q4_k_m":
            .gguf096
        case "unsloth/ltx-2.3-gguf@distilled-1.1-q3_k_m":
            .gguf23
        default:
            nil
        }
    }

    private static func workerExecutable(for kind: WorkerKind) throws -> URL {
        let name = "GenImageLTXVideoWorker"
        let environment = ProcessInfo.processInfo.environment
        var candidates: [URL] = []
        if let configured = environment["GENIMAGE_LTX_WORKER"], !configured.isEmpty {
            candidates.append(URL(fileURLWithPath: configured))
        }
        if let executableDirectory = Bundle.main.executableURL?.deletingLastPathComponent() {
            candidates.append(executableDirectory.appendingPathComponent(name))
            candidates.append(
                executableDirectory
                    .deletingLastPathComponent()
                    .appendingPathComponent("Helpers", isDirectory: true)
                    .appendingPathComponent(name)
            )
        }
        candidates.append(
            Bundle.main.bundleURL
                .appendingPathComponent("Contents", isDirectory: true)
                .appendingPathComponent("Helpers", isDirectory: true)
                .appendingPathComponent(name)
        )

        let sourceRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let packageRoot = sourceRoot
            .appendingPathComponent("RuntimeSupport", isDirectory: true)
            .appendingPathComponent("LTXVideoWorker", isDirectory: true)
        candidates.append(
            packageRoot
                .appendingPathComponent(".build", isDirectory: true)
                .appendingPathComponent("out", isDirectory: true)
                .appendingPathComponent("Products", isDirectory: true)
                .appendingPathComponent("Release", isDirectory: true)
                .appendingPathComponent(name)
        )
        candidates.append(
            packageRoot
                .appendingPathComponent(".build", isDirectory: true)
                .appendingPathComponent("release", isDirectory: true)
                .appendingPathComponent(name)
        )
        candidates.append(
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
                .appendingPathComponent("RuntimeSupport/LTXVideoWorker/.build/release/\(name)")
        )

        guard let executable = RuntimeExecutable.locate(candidates) else {
            throw LTXVideoRuntimeError.workerNotFound(candidates.map(\.path))
        }
        return executable
    }

    private struct WorkerRequest: Encodable {
        struct LoRA: Encodable {
            var path: String
            var scale: Double
            var conditioningScale: Double?

            enum CodingKeys: String, CodingKey {
                case path
                case scale
                case conditioningScale = "conditioning_scale"
            }
        }

        var modelDirectory: String
        var outputPath: String
        var prompt: String
        var width: Int
        var height: Int
        var frames: Int
        var frameRate: Int
        var seed: UInt64
        var stage1Steps: Int
        var stage2Steps: Int
        var imagePaths: [String]
        var loras: [LoRA]
        var gemmaDirectory: String?
    }

    private struct WorkerEvent: Decodable {
        var type: String
        var value: Double?
        var message: String?
        var pixelWidth: Int?
        var pixelHeight: Int?
    }

    private nonisolated static func latestProgress(in log: RuntimeLog) -> Double? {
        guard let data = log.data() else { return nil }
        return data.split(separator: 0x0A).compactMap { line -> Double? in
            guard let event = try? JSONDecoder().decode(WorkerEvent.self, from: Data(line)),
                  event.type == "progress" else { return nil }
            return event.value
        }.last
    }

    private nonisolated static func completionMetadata(
        in log: RuntimeLog
    ) -> (pixelWidth: Int?, pixelHeight: Int?)? {
        guard let data = log.data() else { return nil }
        let events = data.split(separator: 0x0A).compactMap {
            try? JSONDecoder().decode(WorkerEvent.self, from: Data($0))
        }
        guard let event = events.last(where: { $0.type == "completed" }) else {
            return nil
        }
        return (event.pixelWidth, event.pixelHeight)
    }

    private nonisolated static func logMessage(in log: RuntimeLog) -> String {
        guard let data = log.data() else { return "Worker 未提供錯誤訊息。" }
        let events = data.split(separator: 0x0A).compactMap {
            try? JSONDecoder().decode(WorkerEvent.self, from: Data($0))
        }
        if let message = events.last(where: { $0.type == "error" })?.message {
            return message
        }
        return log.message(maximumBytes: 4_096, fallback: "Worker 執行失敗。")
    }
}

public enum LTXVideoRuntimeError: LocalizedError, Sendable {
    case incompatibleProfile
    case unsupportedArchitecture(InferenceArchitecture)
    case unsupportedModel(String)
    case invalidFrameCount(Int)
    case missingInputFile
    case tooManyImageAnchors(count: Int, frameCount: Int)
    case modelNotInstalled(URL)
    case loraNotInstalled(URL)
    case invalidLoRAScale(String)
    case unsupportedImageConditioning
    case unsupportedLoRA
    case workerNotFound([String])
    case runtimeFailed(status: Int32, message: String)
    case runtimeStalled(details: String)
    case outputMissing(URL)

    public var errorDescription: String? {
        switch self {
        case .incompatibleProfile:
            "Profile 不是文生影或圖生影類型。"
        case let .unsupportedArchitecture(architecture):
            "LTX 影片 Worker 不支援此架構：\(architecture.title)。"
        case let .unsupportedModel(modelID):
            "目前 LTX Swift Runtime 尚不支援模型：\(modelID)。"
        case let .invalidFrameCount(frameCount):
            "LTX-2.3 幀數必須符合 8n+1，目前為 \(frameCount)。"
        case .missingInputFile:
            "圖生影需要至少一張可讀取的本機圖片。"
        case let .tooManyImageAnchors(count, frameCount):
            "圖片錨點數量（\(count)）不可超過影片幀數（\(frameCount)）。"
        case let .modelNotInstalled(url):
            "LTX 模型尚未完整安裝：\(url.path)"
        case let .loraNotInstalled(url):
            "Profile 的 LoRA 尚未完整安裝：\(url.path)"
        case let .invalidLoRAScale(modelID):
            "LoRA 權重與控制強度必須介於 0 到 1：\(modelID)"
        case .unsupportedImageConditioning:
            "目前 LTX Swift Worker 尚未提供 image conditioning；未退回其他 Runtime。"
        case .unsupportedLoRA:
            "目前 LTX Swift Worker 尚未提供 LoRA fusion；未退回其他 Runtime。"
        case let .workerNotFound(paths):
            "找不到 LTX Swift Runtime Worker。請重新建置 App，或設定 GENIMAGE_LTX_WORKER；已檢查：\(paths.joined(separator: "、"))"
        case let .runtimeFailed(status, message):
            "LTX Swift Worker 結束（\(status)）：\(message)"
        case let .runtimeStalled(details):
            "LTX Swift Worker 超過 30 分鐘沒有輸出新進度，已自動停止。最後狀態：\(details)"
        case let .outputMissing(url):
            "LTX Swift Worker 完成但沒有產生影片：\(url.path)"
        }
    }
}

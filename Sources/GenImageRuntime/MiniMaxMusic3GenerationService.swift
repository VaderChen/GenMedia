import Foundation
import GenImageCore

public final class MiniMaxMusic3GenerationService: MusicRuntimeAdapter, Sendable {
    private let outputDirectory: URL

    public init(outputDirectory: URL) {
        self.outputDirectory = outputDirectory
    }

    public func supports(_ request: MusicGenerationRequest) -> Bool {
        request.profile.capability == .textToMusic
            && request.profile.architecture == .externalCLI
            && [
                HuggingFaceModelInstaller.miniMaxMusic3MLX8BitModelID,
                HuggingFaceModelInstaller.miniMaxMusic3MLX4BitModelID,
                HuggingFaceModelInstaller.miniMaxMusic3GGUFModelID
            ].contains { request.profile.modelID.caseInsensitiveCompare($0) == .orderedSame }
    }

    public func generate(
        request: MusicGenerationRequest,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> MediaAsset {
        progress(0.001)
        try request.options.validate()
        guard request.profile.capability == .textToMusic else {
            throw MiniMaxMusic3RuntimeError.incompatibleProfile
        }
        guard request.profile.architecture == .externalCLI else {
            throw MiniMaxMusic3RuntimeError.unsupportedArchitecture(request.profile.architecture)
        }
        guard supports(request) else {
            throw MiniMaxMusic3RuntimeError.unsupportedModel(request.profile.modelID)
        }
        try Self.validateModel(at: request.modelURL, modelID: request.profile.modelID)

        let executable = try Self.workerExecutable()
        try FileManager.default.createDirectory(
            at: outputDirectory,
            withIntermediateDirectories: true
        )

        let identifier = UUID().uuidString
        let requestURL = outputDirectory.appendingPathComponent("music-\(identifier)-request.json")
        let waveURL = outputDirectory.appendingPathComponent("music-\(identifier).wav")
        let logURL = outputDirectory.appendingPathComponent("music-\(identifier).log")
        let outputURL = OutputFileNaming.musicURL(
            in: outputDirectory,
            pathExtension: request.options.format.fileExtension
        )
        let temporaryURLs = [requestURL, waveURL, logURL]
        var completed = false
        defer {
            for url in temporaryURLs {
                try? FileManager.default.removeItem(at: url)
            }
            if !completed {
                try? FileManager.default.removeItem(at: outputURL)
            }
        }

        let trimmedLyrics = request.options.lyrics.trimmingCharacters(in: .whitespacesAndNewlines)
        let workerPrompt = trimmedLyrics.isEmpty
            ? "\(request.options.prompt.trimmingCharacters(in: .whitespacesAndNewlines)). Instrumental arrangement, no vocals."
            : request.options.prompt
        let workerLyrics = trimmedLyrics.isEmpty ? "[instrumental]" : request.options.lyrics
        let workerRequest = WorkerRequest(
            modelDirectory: request.modelURL.path,
            outputPath: waveURL.path,
            prompt: workerPrompt,
            lyrics: workerLyrics,
            seed: Int(bitPattern: UInt(request.options.seed)),
            audioDuration: Float(request.options.durationSeconds),
            steps: request.options.steps,
            arCfgScale: 1.5,
            flowCfgScale: 1.7,
            topK: 50
        )
        try JSONEncoder().encode(workerRequest).write(to: requestURL, options: .atomic)
        let log = try RuntimeLog(at: logURL)
        defer { log.close() }

        let startedAt = Date()
        let maximumRuntime = max(30 * 60, Double(request.options.durationSeconds) * 60)
        let status = try await RuntimeProcess.run(
            executable: executable,
            arguments: ["--request", requestURL.path],
            environment: RuntimeExecutable.environment(),
            log: log,
            pollInterval: .milliseconds(500)
        ) {
            let elapsed = Date().timeIntervalSince(startedAt)
            if elapsed >= maximumRuntime {
                throw MiniMaxMusic3RuntimeError.runtimeTimedOut
            }
            if let workerProgress = Self.latestProgress(in: log) {
                progress(workerProgress)
            }
        }

        guard status == 0 else {
            throw MiniMaxMusic3RuntimeError.runtimeFailed(
                status: status,
                message: Self.logMessage(in: log)
            )
        }
        guard FileManager.default.fileExists(atPath: waveURL.path),
              RuntimeLog.fileSize(at: waveURL) > 0 else {
            throw MiniMaxMusic3RuntimeError.waveOutputMissing(waveURL)
        }
        progress(0.995)
        let waveMetadata = try await AudioOutputEncoder.encodeWave(
            inputURL: waveURL,
            outputURL: outputURL,
            format: request.options.format
        )
        guard FileManager.default.fileExists(atPath: outputURL.path),
              RuntimeLog.fileSize(at: outputURL) > 0 else {
            throw MiniMaxMusic3RuntimeError.encodedOutputMissing(outputURL)
        }

        progress(1)
        completed = true
        return MediaAsset(
            projectID: request.projectID,
            kind: .generatedAudio,
            title: "生成音樂",
            fileURL: outputURL,
            pixelWidth: 0,
            pixelHeight: 0,
            mediaDurationSeconds: waveMetadata.durationSeconds,
            sampleRate: waveMetadata.sampleRate,
            channelCount: waveMetadata.channelCount,
            audioFormat: request.options.format,
            recipeID: request.recipeID
        )
    }

    private static func validateModel(at modelURL: URL, modelID: String) throws {
        let requiredPaths: [String]
        if modelID.caseInsensitiveCompare(HuggingFaceModelInstaller.miniMaxMusic3GGUFModelID) == .orderedSame {
            requiredPaths = [
                "config.json",
                "config/condition_encoder.json",
                "config/language_model.json",
                "config/rvq_depth_decoder.json",
                "config/transformer.json",
                "config/vocoder.json",
                "condition_encoder.gguf",
                "language_model_q4_k.gguf",
                "rvq_depth_decoder_q8_0.gguf",
                "tokenizer/tokenizer.json",
                "tokenizer/tokenizer_config.json",
                "transformer_q4_k.gguf",
                "vocoder.gguf"
            ]
        } else if modelID.caseInsensitiveCompare(HuggingFaceModelInstaller.miniMaxMusic3MLX4BitModelID) == .orderedSame {
            requiredPaths = [
                "config.json",
                "model.safetensors.index.json",
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
                "scheduler/scheduler_config.json",
                "tokenizer/chat_template.jinja",
                "tokenizer/tokenizer.json",
                "tokenizer/tokenizer_config.json"
            ]
        } else {
            requiredPaths = [
            "config.json",
            "source_manifest.json",
            "tokenizer/tokenizer.json",
            "language_model/model.safetensors.index.json",
            "language_model/model-00001.safetensors",
            "language_model/model-00002.safetensors",
            "language_model/model-00003.safetensors",
            "language_model/model-00004.safetensors",
            "language_model/model-00005.safetensors",
            "language_model/model-00006.safetensors",
            "language_model/model-00007.safetensors",
            "rvq_depth_decoder/model.safetensors.index.json",
            "rvq_depth_decoder/model-00001.safetensors",
            "condition_encoder/model.safetensors.index.json",
            "condition_encoder/model-00001.safetensors",
            "transformer/model.safetensors.index.json",
            "transformer/model-00001.safetensors",
            "transformer/model-00002.safetensors",
            "vocoder/model.safetensors.index.json",
            "vocoder/model-00001.safetensors"
            ]
        }
        let missing = requiredPaths.filter {
            !FileManager.default.fileExists(atPath: modelURL.appendingPathComponent($0).path)
        }
        guard missing.isEmpty else {
            throw MiniMaxMusic3RuntimeError.modelIncomplete(modelURL, missing)
        }
    }

    private struct WorkerRequest: Encodable {
        var modelDirectory: String
        var outputPath: String
        var prompt: String
        var lyrics: String
        var seed: Int
        var audioDuration: Float
        var steps: Int
        var arCfgScale: Float
        var flowCfgScale: Float
        var topK: Int
    }

    private struct WorkerEvent: Decodable {
        var type: String
        var value: Double?
        var message: String?
    }

    private nonisolated static func workerExecutable() throws -> URL {
        let name = "GenImageMiniMaxMusic3Worker"
        let environment = ProcessInfo.processInfo.environment
        var candidates: [URL] = []
        if let configured = environment["GENIMAGE_MINIMAX_MUSIC3_WORKER"],
           !configured.isEmpty {
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
            .appendingPathComponent("MiniMaxMusic3Worker", isDirectory: true)
        candidates.append(
            packageRoot
                .appendingPathComponent(".build", isDirectory: true)
                .appendingPathComponent("release", isDirectory: true)
                .appendingPathComponent(name)
        )
        candidates.append(
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
                .appendingPathComponent(
                    "RuntimeSupport/MiniMaxMusic3Worker/.build/release/\(name)"
                )
        )
        guard let executable = RuntimeExecutable.locate(candidates) else {
            throw MiniMaxMusic3RuntimeError.workerNotFound(candidates.map(\.path))
        }
        return executable
    }

    private nonisolated static func latestProgress(in log: RuntimeLog) -> Double? {
        guard let data = log.data() else { return nil }
        return data.split(separator: 0x0A).compactMap { line -> Double? in
            guard let event = try? JSONDecoder().decode(WorkerEvent.self, from: Data(line)),
                  event.type == "progress" else { return nil }
            return event.value
        }.max()
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

public enum MiniMaxMusic3RuntimeError: LocalizedError, Sendable {
    case incompatibleProfile
    case unsupportedArchitecture(InferenceArchitecture)
    case unsupportedModel(String)
    case modelIncomplete(URL, [String])
    case workerNotFound([String])
    case runtimeTimedOut
    case runtimeFailed(status: Int32, message: String)
    case waveOutputMissing(URL)
    case encodedOutputMissing(URL)

    public var errorDescription: String? {
        switch self {
        case .incompatibleProfile:
            "Profile 不是文生音樂類型。"
        case let .unsupportedArchitecture(architecture):
            "MiniMax Music 3 Runtime 不支援此架構：\(architecture.title)。"
        case let .unsupportedModel(modelID):
            "MiniMax Music 3 Runtime 不支援模型：\(modelID)。"
        case let .modelIncomplete(url, missing):
            "MiniMax Music 3 模型不完整：\(url.path)；缺少 \(missing.joined(separator: "、"))"
        case let .workerNotFound(paths):
            "找不到 MiniMax Music 3 Swift Runtime Worker。請重新建置 App，或設定 GENIMAGE_MINIMAX_MUSIC3_WORKER；已檢查：\(paths.joined(separator: "、"))"
        case .runtimeTimedOut:
            "MiniMax Music 3 Runtime 超過安全執行時間，已自動停止。"
        case let .runtimeFailed(status, message):
            "MiniMax Music 3 Runtime 結束（\(status)）：\(message)"
        case let .waveOutputMissing(url):
            "MiniMax Music 3 完成但未產生暫存音訊：\(url.path)"
        case let .encodedOutputMissing(url):
            "音訊轉碼完成但找不到輸出檔案：\(url.path)"
        }
    }
}

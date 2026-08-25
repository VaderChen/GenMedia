import Darwin
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
            && request.profile.modelID.caseInsensitiveCompare(
                HuggingFaceModelInstaller.miniMaxMusic3MLX8BitModelID
            ) == .orderedSame
    }

    public func generate(
        request: MusicGenerationRequest,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> ImageAsset {
        progress(0.001)
        try request.options.validate()
        guard request.profile.capability == .textToMusic else {
            throw MiniMaxMusic3RuntimeError.incompatibleProfile
        }
        guard request.profile.architecture == .externalCLI else {
            throw MiniMaxMusic3RuntimeError.unsupportedArchitecture(request.profile.architecture)
        }
        guard request.profile.modelID.lowercased().contains("minimax-music3") else {
            throw MiniMaxMusic3RuntimeError.unsupportedModel(request.profile.modelID)
        }
        try Self.validateModel(at: request.modelURL)

        let runtimeExecutable = try Self.runtimeExecutable()
        try FileManager.default.createDirectory(
            at: outputDirectory,
            withIntermediateDirectories: true
        )

        let identifier = UUID().uuidString
        let promptURL = outputDirectory.appendingPathComponent("music-\(identifier)-prompt.txt")
        let lyricsURL = outputDirectory.appendingPathComponent("music-\(identifier)-lyrics.txt")
        let waveURL = outputDirectory.appendingPathComponent("music-\(identifier).wav")
        let logURL = outputDirectory.appendingPathComponent("music-\(identifier).log")
        let outputURL = OutputFileNaming.musicURL(
            in: outputDirectory,
            pathExtension: request.options.format.fileExtension
        )
        let temporaryURLs = [promptURL, lyricsURL, waveURL, logURL]
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
        let runtimePrompt = trimmedLyrics.isEmpty
            ? "\(request.options.prompt.trimmingCharacters(in: .whitespacesAndNewlines)). Instrumental arrangement, no vocals."
            : request.options.prompt
        let runtimeLyrics = trimmedLyrics.isEmpty ? "[instrumental]" : request.options.lyrics
        try runtimePrompt.write(to: promptURL, atomically: true, encoding: .utf8)
        try runtimeLyrics.write(to: lyricsURL, atomically: true, encoding: .utf8)
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        let logHandle = try FileHandle(forWritingTo: logURL)
        defer { try? logHandle.close() }

        let process = Process()
        process.executableURL = runtimeExecutable
        process.arguments = [
            "generate",
            "--model", request.modelURL.path,
            "--prompt-file", promptURL.path,
            "--lyrics-file", lyricsURL.path,
            "--duration", String(request.options.durationSeconds),
            "--seed", String(request.options.seed),
            "--steps", String(request.options.steps),
            "--output", waveURL.path
        ]
        process.environment = Self.runtimeEnvironment()
        process.standardInput = FileHandle.nullDevice
        process.standardOutput = logHandle
        process.standardError = logHandle

        let startedAt = Date()
        let estimatedRuntime = max(90, Double(request.options.durationSeconds) * 15)
        let maximumRuntime = max(30 * 60, Double(request.options.durationSeconds) * 60)
        do {
            try process.run()
            while process.isRunning {
                try Task.checkCancellation()
                try await Task.sleep(for: .milliseconds(500))
                let elapsed = Date().timeIntervalSince(startedAt)
                if elapsed >= maximumRuntime {
                    Self.forceTerminate(process)
                    throw MiniMaxMusic3RuntimeError.runtimeTimedOut
                }
                let estimatedProgress = min(0.90, max(0.01, elapsed / estimatedRuntime * 0.90))
                progress(estimatedProgress)
            }
        } catch {
            Self.forceTerminate(process)
            throw error
        }
        process.waitUntilExit()
        try? logHandle.synchronize()

        guard process.terminationStatus == 0 else {
            throw MiniMaxMusic3RuntimeError.runtimeFailed(
                status: process.terminationStatus,
                message: Self.logMessage(in: logURL)
            )
        }
        guard FileManager.default.fileExists(atPath: waveURL.path),
              Self.fileSize(at: waveURL) > 0 else {
            throw MiniMaxMusic3RuntimeError.waveOutputMissing(waveURL)
        }
        progress(0.92)
        let waveMetadata = try await AudioOutputEncoder.encodeWave(
            inputURL: waveURL,
            outputURL: outputURL,
            format: request.options.format
        )
        guard FileManager.default.fileExists(atPath: outputURL.path),
              Self.fileSize(at: outputURL) > 0 else {
            throw MiniMaxMusic3RuntimeError.encodedOutputMissing(outputURL)
        }

        progress(1)
        completed = true
        return ImageAsset(
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

    private static func validateModel(at modelURL: URL) throws {
        let requiredPaths = [
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
        let missing = requiredPaths.filter {
            !FileManager.default.fileExists(atPath: modelURL.appendingPathComponent($0).path)
        }
        guard missing.isEmpty else {
            throw MiniMaxMusic3RuntimeError.modelIncomplete(modelURL, missing)
        }
    }

    private static func runtimeExecutable() throws -> URL {
        let fileManager = FileManager.default
        let environment = ProcessInfo.processInfo.environment
        var candidates: [URL] = []
        if let configured = environment["GENIMAGE_MINIMAX_MUSIC3_RUNTIME"],
           !configured.isEmpty {
            candidates.append(URL(fileURLWithPath: configured))
        }
        if let executableDirectory = Bundle.main.executableURL?.deletingLastPathComponent() {
            candidates.append(executableDirectory.appendingPathComponent("mlx-minimax-music3"))
            candidates.append(
                executableDirectory
                    .deletingLastPathComponent()
                    .appendingPathComponent("Helpers/mlx-minimax-music3")
            )
        }
        let home = fileManager.homeDirectoryForCurrentUser
        candidates.append(
            home.appendingPathComponent(
                "Library/Application Support/GenImage/Runtime/minimax-music3/.venv/bin/mlx-minimax-music3"
            )
        )
        candidates.append(home.appendingPathComponent(".local/bin/mlx-minimax-music3"))
        candidates.append(URL(fileURLWithPath: "/opt/homebrew/bin/mlx-minimax-music3"))
        candidates.append(URL(fileURLWithPath: "/usr/local/bin/mlx-minimax-music3"))
        for directory in (environment["PATH"] ?? "").split(separator: ":") {
            candidates.append(
                URL(fileURLWithPath: String(directory), isDirectory: true)
                    .appendingPathComponent("mlx-minimax-music3")
            )
        }
        guard let executable = candidates.first(where: {
            fileManager.isExecutableFile(atPath: $0.path)
        }) else {
            throw MiniMaxMusic3RuntimeError.runtimeNotFound(candidates.map(\.path))
        }
        return executable
    }

    private static func runtimeEnvironment() -> [String: String] {
        var environment = ProcessInfo.processInfo.environment
        let commonPaths = ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin"]
        let currentPaths = (environment["PATH"] ?? "").split(separator: ":").map(String.init)
        environment["PATH"] = (currentPaths + commonPaths)
            .reduce(into: [String]()) { paths, path in
                if !paths.contains(path) { paths.append(path) }
            }
            .joined(separator: ":")
        environment["PYTHONUNBUFFERED"] = "1"
        return environment
    }

    private static func forceTerminate(_ process: Process) {
        guard process.isRunning else { return }
        process.terminate()
        if process.isRunning {
            Darwin.kill(process.processIdentifier, SIGKILL)
        }
        process.waitUntilExit()
    }

    private static func fileSize(at url: URL) -> Int64 {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
              let size = attributes[.size] as? NSNumber else { return 0 }
        return size.int64Value
    }

    private static func logMessage(in logURL: URL) -> String {
        guard let data = try? Data(contentsOf: logURL), !data.isEmpty else {
            return "Runtime 未提供錯誤訊息。"
        }
        return String(data: data.suffix(8_192), encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            ?? "Runtime 執行失敗。"
    }
}

public enum MiniMaxMusic3RuntimeError: LocalizedError, Sendable {
    case incompatibleProfile
    case unsupportedArchitecture(InferenceArchitecture)
    case unsupportedModel(String)
    case modelIncomplete(URL, [String])
    case runtimeNotFound([String])
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
        case let .runtimeNotFound(paths):
            "找不到 mlx-minimax-music3 Runtime。請安裝後設定 GENIMAGE_MINIMAX_MUSIC3_RUNTIME；已檢查：\(paths.joined(separator: "、"))"
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

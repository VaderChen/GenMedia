import Foundation
import GenImageCore
import ImageIO

public actor Qwen2511ImageToImageService: ImageToImageGenerating {
    private var outputDirectory: URL
    private var runningProcess: Process?

    public init(outputDirectory: URL) {
        self.outputDirectory = outputDirectory
    }

    public func setOutputDirectory(_ outputDirectory: URL) {
        self.outputDirectory = outputDirectory
    }

    public func generate(
        request: ImageToImageRequest,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> ImageAsset {
        guard request.profile.capability == .imageToImage else {
            throw Qwen2511RuntimeError.incompatibleProfile
        }
        guard request.profile.architecture == .externalCLI else {
            throw Qwen2511RuntimeError.unsupportedArchitecture(request.profile.architecture)
        }
        guard let inputURL = request.sourceAsset.fileURL,
              FileManager.default.fileExists(atPath: inputURL.path) else {
            throw Qwen2511RuntimeError.missingInputFile
        }
        let manifestURL = request.modelURL.appendingPathComponent("genimage-model.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw Qwen2511RuntimeError.modelNotInstalled(request.modelURL)
        }
        let executable = try Self.workerExecutable()
        try FileManager.default.createDirectory(
            at: outputDirectory,
            withIntermediateDirectories: true
        )

        let identifier = UUID().uuidString
        let outputURL = OutputFileNaming.imageURL(in: outputDirectory, pathExtension: "png")
        let requestURL = outputDirectory.appendingPathComponent("qwen-edit-\(identifier).json")
        let logURL = outputDirectory.appendingPathComponent("qwen-edit-\(identifier).log")
        defer {
            try? FileManager.default.removeItem(at: requestURL)
            try? FileManager.default.removeItem(at: logURL)
        }

        let payload = WorkerRequest(
            modelDirectory: request.modelURL.path,
            quantization: Self.workerQuantization(request.quantization),
            inputPath: inputURL.path,
            outputPath: outputURL.path,
            prompt: request.recipe.prompt,
            negativePrompt: request.recipe.negativePrompt,
            width: request.recipe.width,
            height: request.recipe.height,
            steps: request.recipe.steps,
            seed: request.recipe.seed
        )
        try JSONEncoder().encode(payload).write(to: requestURL, options: .atomic)
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        let logHandle = try FileHandle(forWritingTo: logURL)
        defer { try? logHandle.close() }

        let process = Process()
        process.executableURL = executable
        process.arguments = ["--request", requestURL.path]
        process.standardInput = FileHandle.nullDevice
        process.standardOutput = logHandle
        process.standardError = logHandle
        runningProcess = process
        defer { runningProcess = nil }

        progress(0.01)
        try process.run()
        var lastProgress = 0.01
        do {
            while process.isRunning {
                try Task.checkCancellation()
                try await Task.sleep(for: .milliseconds(250))
                if let value = Self.latestProgress(in: logURL), value > lastProgress {
                    lastProgress = value
                    progress(value)
                }
            }
        } catch is CancellationError {
            process.terminate()
            process.waitUntilExit()
            throw CancellationError()
        }
        process.waitUntilExit()

        guard process.terminationStatus == 0 else {
            throw Qwen2511RuntimeError.workerFailed(
                status: process.terminationStatus,
                message: Self.logMessage(in: logURL)
            )
        }
        guard FileManager.default.fileExists(atPath: outputURL.path),
              let dimensions = Self.imageDimensions(at: outputURL) else {
            throw Qwen2511RuntimeError.outputMissing(outputURL)
        }
        progress(1)
        return ImageAsset(
            projectID: request.projectID,
            parentAssetID: request.sourceAsset.id,
            kind: .edited,
            title: "圖生圖結果",
            fileURL: outputURL,
            pixelWidth: dimensions.width,
            pixelHeight: dimensions.height,
            recipeID: request.recipe.id
        )
    }

    private struct WorkerRequest: Encodable {
        var modelDirectory: String
        var quantization: String
        var inputPath: String
        var outputPath: String
        var prompt: String
        var negativePrompt: String
        var width: Int
        var height: Int
        var steps: Int
        var seed: UInt64
    }

    private struct WorkerEvent: Decodable {
        var type: String
        var value: Double?
        var message: String?
    }

    private nonisolated static func workerQuantization(_ quantization: ModelQuantization) -> String {
        switch quantization {
        case .twoBit: "int4"
        case .bf16: "fp16"
        case .fourBit: "int4"
        case .eightBit: "int8"
        case .fp16: "fp16"
        case .coreML: "fp16"
        case .lora: "fp16"
        }
    }

    private nonisolated static func workerExecutable() throws -> URL {
        let fileManager = FileManager.default
        let name = "GenImageQwen2511Worker"
        var candidates: [URL] = []
        if let configured = ProcessInfo.processInfo.environment["GENIMAGE_QWEN_WORKER"],
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
            .appendingPathComponent("Qwen2511Worker", isDirectory: true)
        candidates.append(
            packageRoot
                .appendingPathComponent(".build", isDirectory: true)
                .appendingPathComponent("release", isDirectory: true)
                .appendingPathComponent(name)
        )
        candidates.append(
            URL(fileURLWithPath: fileManager.currentDirectoryPath, isDirectory: true)
                .appendingPathComponent("RuntimeSupport/Qwen2511Worker/.build/release/\(name)")
        )
        guard let executable = candidates.first(where: {
            fileManager.isExecutableFile(atPath: $0.path)
        }) else {
            throw Qwen2511RuntimeError.workerNotFound(candidates.map(\.path))
        }
        return executable
    }

    private nonisolated static func latestProgress(in logURL: URL) -> Double? {
        guard let data = try? Data(contentsOf: logURL), !data.isEmpty else { return nil }
        return data.split(separator: 0x0A).compactMap { line -> Double? in
            guard let event = try? JSONDecoder().decode(WorkerEvent.self, from: Data(line)),
                  event.type == "progress" else { return nil }
            return event.value
        }.max()
    }

    private nonisolated static func logMessage(in logURL: URL) -> String {
        guard let data = try? Data(contentsOf: logURL), !data.isEmpty else {
            return "Worker 未提供錯誤訊息。"
        }
        let events = data.split(separator: 0x0A).compactMap {
            try? JSONDecoder().decode(WorkerEvent.self, from: Data($0))
        }
        if let message = events.last(where: { $0.type == "error" })?.message {
            return message
        }
        let suffix = data.suffix(4_096)
        return String(data: suffix, encoding: .utf8) ?? "Worker 執行失敗。"
    }

    private nonisolated static func imageDimensions(at url: URL) -> (width: Int, height: Int)? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { return nil }
        return (image.width, image.height)
    }
}

public enum Qwen2511RuntimeError: LocalizedError, Sendable {
    case incompatibleProfile
    case unsupportedArchitecture(InferenceArchitecture)
    case missingInputFile
    case modelNotInstalled(URL)
    case workerNotFound([String])
    case workerFailed(status: Int32, message: String)
    case outputMissing(URL)

    public var errorDescription: String? {
        switch self {
        case .incompatibleProfile:
            "Profile 不是圖生圖類型。"
        case let .unsupportedArchitecture(architecture):
            "Qwen 2511 Runtime 不支援此架構：\(architecture.title)。"
        case .missingInputFile:
            "請先選取一張可讀取的本機圖片。"
        case let .modelNotInstalled(url):
            "Qwen 2511 模型尚未完整安裝：\(url.path)"
        case let .workerNotFound(paths):
            "找不到 Qwen 2511 Runtime Worker。已檢查：\(paths.joined(separator: "、"))"
        case let .workerFailed(status, message):
            "Qwen 2511 Runtime 結束（\(status)）：\(message)"
        case let .outputMissing(url):
            "Qwen 2511 推論完成但沒有產生檔案：\(url.path)"
        }
    }
}

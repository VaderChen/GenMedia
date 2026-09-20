import Foundation
import GenImageCore

public actor ZImageTextToImageService: TextToImageGenerating {
    private nonisolated let outputLocation: OutputDirectoryStorage
    private var outputDirectory: URL { outputLocation.url }
    private let worker = WarmRuntimeWorker()
    private var isGenerating = false
    private let configuredWorker: URL?

    public init(outputDirectory: URL) {
        self.outputLocation = OutputDirectoryStorage(outputDirectory)
        self.configuredWorker = nil
    }

    init(outputDirectory: URL, workerExecutable: URL) {
        self.outputLocation = OutputDirectoryStorage(outputDirectory)
        self.configuredWorker = workerExecutable
    }

    public nonisolated func setOutputDirectory(_ outputDirectory: URL) {
        outputLocation.update(to: outputDirectory)
    }

    public func unload() async {
        await worker.unload()
    }

    public func generate(
        request: TextToImageRequest,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> [MediaAsset] {
        let outputDirectory = self.outputDirectory
        guard !isGenerating else { throw WarmRuntimeWorker.Failure.busy }
        isGenerating = true
        defer { isGenerating = false }
        guard request.profile.capability == .textToImage else {
            throw ZImageRuntimeError.incompatibleProfile
        }
        guard request.profile.architecture == .mlxSwift else {
            throw ZImageRuntimeError.unsupportedArchitecture(request.profile.architecture)
        }

        try request.recipe.validate()
        let modelURL = URL(fileURLWithPath: request.profile.modelID, isDirectory: true)
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: modelURL.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            throw ZImageRuntimeError.modelNotFound(modelURL)
        }

        let loraPath: URL?
        let loraScale: Double?
        if let selection = request.recipe.lora {
            let loraURL = selection.localURL.resolvingSymlinksInPath().standardizedFileURL
            var isLoRADirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: loraURL.path, isDirectory: &isLoRADirectory) else {
                throw ZImageRuntimeError.loraNotFound(loraURL)
            }
            loraPath = isLoRADirectory.boolValue
                ? loraURL
                : try ZImageLoRAAdapterNormalizer.normalize(loraURL)
            loraScale = selection.scale
        } else {
            loraPath = nil
            loraScale = nil
        }

        try FileManager.default.createDirectory(
            at: outputDirectory,
            withIntermediateDirectories: true
        )

        let outputURLs = (0..<request.recipe.outputCount).map { _ in
            OutputFileNaming.imageURL(in: outputDirectory, pathExtension: "png")
        }
        var completed = false
        defer {
            if !completed {
                for outputURL in outputURLs {
                    try? FileManager.default.removeItem(at: outputURL)
                }
            }
        }

        let identifier = UUID().uuidString
        let payload = WorkerRequest(
            requestID: identifier,
            modelDirectory: modelURL.path,
            outputPaths: outputURLs.map(\.path),
            prompt: request.recipe.prompt,
            negativePrompt: request.recipe.negativePrompt,
            width: request.recipe.width,
            height: request.recipe.height,
            steps: request.recipe.steps,
            seed: request.recipe.seed,
            loraPath: loraPath?.path,
            loraScale: loraScale
        )
        progress(0.01)
        _ = try await worker.run(
            executable: configuredWorker ?? Self.workerExecutable(),
            request: JSONEncoder().encode(payload),
            requestID: identifier,
            progress: progress
        )
        try Task.checkCancellation()
        guard outputURLs.allSatisfy({ FileManager.default.fileExists(atPath: $0.path) }) else {
            let missing = outputURLs.first(where: { !FileManager.default.fileExists(atPath: $0.path) })
                ?? outputURLs[0]
            throw ZImageRuntimeError.outputMissing(missing)
        }

        let outputCount = outputURLs.count
        let assets = outputURLs.enumerated().map { index, outputURL in
            MediaAsset(
                projectID: request.projectID,
                parentAssetID: request.sourceAsset?.id,
                kind: .generated,
                title: outputCount == 1 ? "生成結果" : "生成結果 \(index + 1)",
                fileURL: outputURL,
                pixelWidth: request.recipe.width,
                pixelHeight: request.recipe.height,
                recipeID: request.recipe.id
            )
        }

        completed = true
        progress(1)
        return assets
    }

    private struct WorkerRequest: Encodable {
        var requestID: String
        var modelDirectory: String
        var outputPaths: [String]
        var prompt: String
        var negativePrompt: String
        var width: Int
        var height: Int
        var steps: Int
        var seed: UInt64
        var loraPath: String?
        var loraScale: Double?
    }

    private nonisolated static func workerExecutable() throws -> URL {
        let name = "GenImageZImageWorker"
        let environment = ProcessInfo.processInfo.environment
        var candidates: [URL] = []
        if let configured = environment["GENIMAGE_ZIMAGE_WORKER"], !configured.isEmpty {
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
            candidates.append(
                executableDirectory
                    .deletingLastPathComponent()
                    .appendingPathComponent("Helpers/ZImage", isDirectory: true)
                    .appendingPathComponent(name)
            )
        }
        candidates.append(
            Bundle.main.bundleURL
                .appendingPathComponent("Contents/Helpers/ZImage", isDirectory: true)
                .appendingPathComponent(name)
        )

        let sourceRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let packageRoot = sourceRoot
            .appendingPathComponent("RuntimeSupport/ZImageWorker", isDirectory: true)
        candidates.append(
            packageRoot
                .appendingPathComponent(".build/out/Products/Release", isDirectory: true)
                .appendingPathComponent(name)
        )
        candidates.append(
            packageRoot
                .appendingPathComponent(".build/release", isDirectory: true)
                .appendingPathComponent(name)
        )
        candidates.append(
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
                .appendingPathComponent("RuntimeSupport/ZImageWorker/.build/release/\(name)")
        )

        guard let executable = RuntimeExecutable.locate(candidates) else {
            throw ZImageRuntimeError.workerNotFound(candidates.map(\.path))
        }
        return executable
    }


}

public enum ZImageRuntimeError: LocalizedError, Sendable {
    case incompatibleProfile
    case unsupportedArchitecture(InferenceArchitecture)
    case modelNotFound(URL)
    case loraNotFound(URL)
    case unsupportedLoRAFormat(URL)
    case workerNotFound([String])
    case workerFailed(status: Int32, message: String)
    case outputMissing(URL)

    public var errorDescription: String? {
        switch self {
        case .incompatibleProfile:
            "Profile 不是文生圖類型。"
        case let .unsupportedArchitecture(architecture):
            "Z-Image Runtime 不支援此架構：\(architecture.title)。"
        case let .modelNotFound(url):
            "找不到 Z-Image 模型：\(url.path)"
        case let .loraNotFound(url):
            "找不到 LoRA 模型：\(url.path)"
        case let .unsupportedLoRAFormat(url):
            "LoRA 必須是 .safetensors 檔案：\(url.path)"
        case let .workerNotFound(paths):
            "找不到 Z-Image Runtime Worker。已檢查：\(paths.joined(separator: "、"))"
        case let .workerFailed(status, message):
            "Z-Image Runtime 結束（\(status)）：\(message)"
        case let .outputMissing(url):
            "Z-Image 完成推論但沒有產生檔案：\(url.path)"
        }
    }
}

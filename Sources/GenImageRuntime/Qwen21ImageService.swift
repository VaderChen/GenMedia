import Foundation
import GenImageCore
import ImageIO

public actor Qwen21ImageService: TextToImageGenerating, ImageToImageGenerating {
    private nonisolated let outputLocation: OutputDirectoryStorage
    private let configuredWorker: URL?
    private var busy = false

    public init(outputDirectory: URL) {
        outputLocation = OutputDirectoryStorage(outputDirectory); configuredWorker = nil
    }
    init(outputDirectory: URL, workerExecutable: URL) {
        outputLocation = OutputDirectoryStorage(outputDirectory); configuredWorker = workerExecutable
    }
    public nonisolated func setOutputDirectory(_ url: URL) { outputLocation.update(to: url) }

    public func generate(request: TextToImageRequest, progress: @escaping @Sendable (Double) -> Void) async throws -> [MediaAsset] {
        guard request.profile.capability == .textToImage else { throw Failure.invalid("Profile 不是文生圖。") }
        return try await run(projectID: request.projectID, recipe: request.recipe, profile: request.profile,
            modelURL: URL(fileURLWithPath: request.profile.modelID), source: nil,
            parentID: request.sourceAsset?.id, progress: progress)
    }
    public func generate(request: ImageToImageRequest, progress: @escaping @Sendable (Double) -> Void) async throws -> MediaAsset {
        guard request.profile.capability == .imageToImage, request.quantization == .fourBit else {
            throw Failure.invalid("需要 Qwen-Image 2.1 4-bit 圖生圖 Profile。")
        }
        guard let url = request.sourceAsset.fileURL, FileManager.default.fileExists(atPath: url.path) else {
            throw Failure.invalid("請先選取可讀取的本機圖片。")
        }
        return try await run(projectID: request.projectID, recipe: request.recipe, profile: request.profile,
            modelURL: request.modelURL, source: request.sourceAsset,
            parentID: request.sourceAsset.id, progress: progress)[0]
    }

    private func run(projectID: UUID, recipe: GenerationRecipe, profile: InferenceProfile,
                     modelURL: URL, source: MediaAsset?, parentID: UUID?,
                     progress: @escaping @Sendable (Double) -> Void) async throws -> [MediaAsset] {
        let directory = outputLocation.url
        guard !busy else { throw Failure.invalid("已有 Qwen-Image 2.1 工作執行中。") }
        busy = true
        defer { busy = false }
        try recipe.validate()
        guard profile.architecture == .mlxSwift, ImageGenerationRouter.isQwen21(recipe: recipe, profile: profile) else {
            throw Failure.invalid("不相容的模型或 Runtime。")
        }
        guard recipe.lora == nil else { throw Failure.invalid("Qwen-Image 2.1 尚未支援 LoRA。") }
        guard recipe.negativePrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw Failure.invalid("Qwen-Image 2.1 使用 CFG=1，請清空負面提示詞。")
        }
        guard (256...2048).contains(recipe.width), (256...2048).contains(recipe.height),
              recipe.width % 32 == 0, recipe.height % 32 == 0, (1...100).contains(recipe.steps),
              (1...8).contains(recipe.outputCount) else {
            throw Failure.invalid("尺寸須為 256～2048 的 32 倍數；步數 1～100，輸出 1～8 張。")
        }
        guard QwenImage21Model.requiredFiles.allSatisfy({
            (try? modelURL.appendingPathComponent($0).resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]))
                .map { $0.isRegularFile == true && ($0.fileSize ?? 0) > 0 } ?? false
        }) else { throw Failure.invalid("模型尚未完整安裝。") }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let outputs = (0..<(source == nil ? recipe.outputCount : 1)).map { _ in
            OutputFileNaming.imageURL(in: directory, pathExtension: "png")
        }
        let id = UUID().uuidString
        let requestURL = directory.appendingPathComponent("qwen21-\(id).json")
        let logURL = directory.appendingPathComponent("qwen21-\(id).log")
        var completed = false
        defer {
            try? FileManager.default.removeItem(at: requestURL)
            try? FileManager.default.removeItem(at: logURL)
            if !completed { outputs.forEach { try? FileManager.default.removeItem(at: $0) } }
        }
        let payload = Payload(modelDirectory: modelURL.path, outputPaths: outputs.map(\.path),
            inputPath: source?.fileURL?.path, prompt: recipe.prompt, negativePrompt: recipe.negativePrompt,
            width: recipe.width, height: recipe.height, steps: recipe.steps, seed: recipe.seed)
        try JSONEncoder().encode(payload).write(to: requestURL, options: .atomic)
        let log = try RuntimeLog(at: logURL)
        defer { log.close() }
        var last = 0.0
        let status = try await RuntimeProcess.run(executable: configuredWorker ?? Self.workerExecutable(),
            arguments: ["--request", requestURL.path], log: log) {
                if let value = log.latestProgress(useMaximum: true), value > last {
                    last = value; progress(min(value, 0.99))
                }
            }
        try Task.checkCancellation()
        guard status == 0 else {
            throw Failure.invalid(log.message(maximumBytes: 4096, fallback: "Worker 結束（\(status)）。"))
        }
        let assets = try outputs.map { url in
            guard let image = CGImageSourceCreateWithURL(url as CFURL, nil),
                  let properties = CGImageSourceCopyPropertiesAtIndex(image, 0, nil) as? [CFString: Any],
                  properties[kCGImagePropertyPixelWidth] as? Int == recipe.width,
                  properties[kCGImagePropertyPixelHeight] as? Int == recipe.height else {
                throw Failure.invalid("Worker 未產生符合尺寸的 PNG。")
            }
            return MediaAsset(projectID: projectID, parentAssetID: parentID,
                kind: source == nil ? .generated : .edited, title: "Qwen-Image 2.1 結果", fileURL: url,
                pixelWidth: recipe.width, pixelHeight: recipe.height, recipeID: recipe.id)
        }
        completed = true; progress(1)
        return assets
    }

    private struct Payload: Encodable {
        let modelDirectory: String; let outputPaths: [String]; let inputPath: String?
        let prompt: String; let negativePrompt: String; let width: Int; let height: Int; let steps: Int; let seed: UInt64
    }
    enum Failure: LocalizedError {
        case invalid(String)
        var errorDescription: String? { switch self { case .invalid(let message): "Qwen-Image 2.1：\(message)" } }
    }
    private static func workerExecutable() throws -> URL {
        let name = "GenImageQwen21Worker"
        var candidates: [URL] = []
        if let path = ProcessInfo.processInfo.environment["GENIMAGE_QWEN21_WORKER"], !path.isEmpty {
            candidates.append(URL(fileURLWithPath: path))
        }
        if let executable = Bundle.main.executableURL?.deletingLastPathComponent() {
            candidates.append(executable.appendingPathComponent(name))
            candidates.append(executable.deletingLastPathComponent().appendingPathComponent("Helpers/\(name)"))
        }
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        for path in [".build/release", ".build/debug", ".build/out/Products/Release", ".build/out/Products/Debug"] {
            candidates.append(root.appendingPathComponent(path).appendingPathComponent(name))
        }
        guard let url = RuntimeExecutable.locate(candidates) else {
            throw Failure.invalid("找不到 GenImageQwen21Worker；請重新建置專案。")
        }
        return url
    }
}

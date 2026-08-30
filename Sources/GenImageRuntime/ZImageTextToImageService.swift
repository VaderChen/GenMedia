import Foundation
import GenImageCore

public actor ZImageTextToImageService: TextToImageGenerating {
    private var outputDirectory: URL

    public init(outputDirectory: URL) {
        self.outputDirectory = outputDirectory
    }

    public func setOutputDirectory(_ outputDirectory: URL) {
        self.outputDirectory = outputDirectory
    }

    public func unload() {
    }

    public func generate(
        request: TextToImageRequest,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> [MediaAsset] {
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
                : try normalizedLoRAURLIfNeeded(loraURL)
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
        let requestURL = outputDirectory.appendingPathComponent("z-image-\(identifier)-request.json")
        let logURL = outputDirectory.appendingPathComponent("z-image-\(identifier).log")
        defer {
            try? FileManager.default.removeItem(at: requestURL)
            try? FileManager.default.removeItem(at: logURL)
        }
        let payload = WorkerRequest(
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
        try JSONEncoder().encode(payload).write(to: requestURL, options: .atomic)
        let log = try RuntimeLog(at: logURL)
        defer { log.close() }

        let executable = try Self.workerExecutable()
        progress(0.01)
        let status = try await RuntimeProcess.run(
            executable: executable,
            arguments: ["--request", requestURL.path],
            environment: RuntimeExecutable.environment(),
            log: log,
            pollInterval: .milliseconds(300)
        ) {
            if let value = Self.latestProgress(in: log) {
                progress(min(0.99, max(0.01, value)))
            }
        }
        guard status == 0 else {
            throw ZImageRuntimeError.workerFailed(
                status: status,
                message: Self.logMessage(in: log)
            )
        }
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

    private struct WorkerEvent: Decodable {
        var type: String
        var value: Double?
        var message: String?
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

    private nonisolated static func latestProgress(in log: RuntimeLog) -> Double? {
        guard let data = log.data() else { return nil }
        return data.split(separator: 0x0A).compactMap { line -> Double? in
            guard let event = try? JSONDecoder().decode(WorkerEvent.self, from: Data(line)),
                  event.type == "progress" else { return nil }
            return event.value
        }.last
    }

    private nonisolated static func logMessage(in log: RuntimeLog) -> String {
        guard let data = log.data() else { return "Z-Image Worker 未提供錯誤訊息。" }
        let events = data.split(separator: 0x0A).compactMap {
            try? JSONDecoder().decode(WorkerEvent.self, from: Data($0))
        }
        if let message = events.last(where: { $0.type == "error" })?.message {
            return message
        }
        return log.message(maximumBytes: 4_096, fallback: "Z-Image Worker 執行失敗。")
    }

    private func normalizedLoRAURLIfNeeded(_ url: URL) throws -> URL {
        let data = try Data(contentsOf: url, options: [.mappedIfSafe])
        guard data.count >= 8 else { return url }

        var headerLength: UInt64 = 0
        for (index, byte) in data.prefix(8).enumerated() {
            headerLength |= UInt64(byte) << UInt64(index * 8)
        }
        guard headerLength <= UInt64(data.count - 8),
              headerLength <= UInt64(Int.max) else {
            return url
        }

        let oldHeaderLength = Int(headerLength)
        let headerStart = data.index(data.startIndex, offsetBy: 8)
        let headerEnd = data.index(headerStart, offsetBy: oldHeaderLength)
        guard let headerObject = try JSONSerialization.jsonObject(
            with: data[headerStart..<headerEnd]
        ) as? [String: Any] else {
            return url
        }

        var header = headerObject
        var renamed = false
        for key in headerObject.keys {
            let normalizedKey: String
            if key.hasSuffix(".lora_A") {
                normalizedKey = "\(key).weight"
            } else if key.hasSuffix(".lora_B") {
                normalizedKey = "\(key).weight"
            } else {
                continue
            }
            guard header[normalizedKey] == nil,
                  let value = header.removeValue(forKey: key) else { continue }
            header[normalizedKey] = value
            renamed = true
        }
        guard renamed else { return url }

        var normalizedHeader = try JSONSerialization.data(
            withJSONObject: header,
            options: [.sortedKeys]
        )
        let padding = (8 - normalizedHeader.count % 8) % 8
        if padding > 0 {
            normalizedHeader.append(Data(repeating: 0x20, count: padding))
        }

        let cacheDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("GenImage-LoRAAdapters", isDirectory: true)
        try FileManager.default.createDirectory(
            at: cacheDirectory,
            withIntermediateDirectories: true
        )
        let values = try? url.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey])
        let signature = "\(url.path)|\(values?.fileSize ?? 0)|\(values?.contentModificationDate?.timeIntervalSince1970 ?? 0)"
        var hash: UInt64 = 14_695_981_039_346_656_037
        for byte in signature.utf8 {
            hash ^= UInt64(byte)
            hash &*= 1_099_511_628_211
        }
        let baseName = url.deletingPathExtension().lastPathComponent
            .replacingOccurrences(of: ".", with: "-")
        let normalizedURL = cacheDirectory.appendingPathComponent(
            "\(baseName)-\(String(hash, radix: 16)).safetensors"
        )
        if FileManager.default.fileExists(atPath: normalizedURL.path) {
            return normalizedURL
        }

        var normalizedData = Data()
        var encodedHeaderLength = UInt64(normalizedHeader.count)
        for _ in 0..<8 {
            normalizedData.append(UInt8(encodedHeaderLength & 0xff))
            encodedHeaderLength >>= 8
        }
        normalizedData.append(normalizedHeader)
        normalizedData.append(data[headerEnd..<data.endIndex])
        try normalizedData.write(to: normalizedURL, options: .atomic)
        return normalizedURL
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

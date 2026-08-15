import Foundation
import GenImageCore
import Logging
import ZImage

public actor ZImageTextToImageService: TextToImageGenerating {
    private var outputDirectory: URL
    private let pipeline: ZImagePipeline
    private var cacheTrimTask: Task<Void, Never>?

    public init(outputDirectory: URL) {
        self.outputDirectory = outputDirectory
        let logger = Logger(label: "genimage.zimage") { _ in
            SwiftLogNoOpLogHandler()
        }
        pipeline = ZImagePipeline(logger: logger)
    }

    public func setOutputDirectory(_ outputDirectory: URL) {
        self.outputDirectory = outputDirectory
    }

    /// 釋放目前常駐的 MLX 模型，供記憶體壓力保護使用。
    public func unload() {
        cacheTrimTask?.cancel()
        cacheTrimTask = nil
        pipeline.unloadModel()
    }

    public func generate(
        request: TextToImageRequest,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> [ImageAsset] {
        guard request.profile.capability == .textToImage else {
            throw ZImageRuntimeError.incompatibleProfile
        }
        guard request.profile.architecture == .mlxSwift else {
            throw ZImageRuntimeError.unsupportedArchitecture(request.profile.architecture)
        }

        try request.recipe.validate()
        let modelURL = URL(fileURLWithPath: request.profile.modelID, isDirectory: true)
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: modelURL.path, isDirectory: &isDirectory) else {
            throw ZImageRuntimeError.modelNotFound(modelURL)
        }

        cacheTrimTask?.cancel()
        cacheTrimTask = nil
        defer { scheduleWarmCacheTrim() }

        let loraConfiguration: LoRAConfiguration?
        if let selection = request.recipe.lora {
            let loraURL = selection.localURL.resolvingSymlinksInPath().standardizedFileURL
            var isLoRADirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: loraURL.path, isDirectory: &isLoRADirectory) else {
                throw ZImageRuntimeError.loraNotFound(loraURL)
            }
            let compatibleLoRAURL = isLoRADirectory.boolValue
                ? loraURL
                : try normalizedLoRAURLIfNeeded(loraURL)
            loraConfiguration = .local(compatibleLoRAURL, scale: Float(selection.scale))
        } else {
            loraConfiguration = nil
        }

        try FileManager.default.createDirectory(
            at: outputDirectory,
            withIntermediateDirectories: true
        )

        let outputCount = request.recipe.outputCount
        var assets: [ImageAsset] = []
        assets.reserveCapacity(outputCount)

        for index in 0..<outputCount {
            try Task.checkCancellation()
            let seed = request.recipe.seed &+ UInt64(index)
            let outputURL = OutputFileNaming.imageURL(in: outputDirectory, pathExtension: "png")
            let generationRequest = ZImageGenerationRequest(
                prompt: request.recipe.prompt,
                negativePrompt: request.recipe.negativePrompt.isEmpty
                    ? nil
                    : request.recipe.negativePrompt,
                width: request.recipe.width,
                height: request.recipe.height,
                steps: request.recipe.steps,
                guidanceScale: 0,
                seed: seed,
                outputPath: outputURL,
                model: modelURL.path,
                lora: loraConfiguration,
                runtimeOptions: ZImageRuntimeOptions(residencyPolicy: .warm)
            )

            var lastFraction = 0.0
            var lastReportedFraction = -1.0
            var lastReportedAt = Date.distantPast
            do {
                _ = try await pipeline.generate(generationRequest) { update in
                    let currentFraction: Double
                    switch update.stage {
                    case .loadingModel:
                        currentFraction = 0.02
                    case .encodingText:
                        currentFraction = 0.12
                    case .loadingTransformer:
                        currentFraction = 0.18
                    case .loadingLoRA:
                        currentFraction = 0.22
                    case .loadingVAE:
                        currentFraction = 0.24
                    case .denoising:
                        currentFraction = 0.25 + update.fractionCompleted * 0.60
                    case .decoding:
                        currentFraction = 0.90
                    case .saving:
                        currentFraction = 0.98
                    }
                    lastFraction = max(lastFraction, currentFraction)
                    let aggregateFraction = (Double(index) + lastFraction) / Double(outputCount)
                    let now = Date()
                    guard aggregateFraction >= 1
                        || aggregateFraction - lastReportedFraction >= 0.01
                        || now.timeIntervalSince(lastReportedAt) >= 0.1 else {
                        return
                    }
                    lastReportedFraction = aggregateFraction
                    lastReportedAt = now
                    progress(aggregateFraction)
                }
            } catch let error as ZImagePipeline.PipelineError {
                throw ZImageRuntimeError.pipelineFailure(Self.pipelineErrorMessage(error))
            }
            try Task.checkCancellation()
            progress(Double(index + 1) / Double(outputCount))

            guard FileManager.default.fileExists(atPath: outputURL.path) else {
                throw ZImageRuntimeError.outputMissing(outputURL)
            }
            assets.append(
                ImageAsset(
                    projectID: request.projectID,
                    parentAssetID: request.sourceAsset?.id,
                    kind: .generated,
                    title: outputCount == 1 ? "生成結果" : "生成結果 \(index + 1)",
                    fileURL: outputURL,
                    pixelWidth: request.recipe.width,
                    pixelHeight: request.recipe.height,
                    recipeID: request.recipe.id
                )
            )
        }

        progress(1)
        return assets
    }

    private func scheduleWarmCacheTrim() {
        cacheTrimTask?.cancel()
        cacheTrimTask = Task { [weak self] in
            do {
                try await Task.sleep(for: .seconds(300))
            } catch {
                return
            }
            guard !Task.isCancelled else { return }
            await self?.trimWarmCache()
        }
    }

    private func trimWarmCache() {
        pipeline.trimCache()
        cacheTrimTask = nil
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

    private static func pipelineErrorMessage(_ error: ZImagePipeline.PipelineError) -> String {
        switch error {
        case .notImplemented:
            return "Z-Image Runtime 尚未實作此功能。"
        case .tokenizerNotLoaded:
            return "Z-Image tokenizer 尚未載入。"
        case let .invalidDimensions(message), let .invalidModelPath(message), let .weightsMissing(message):
            return message
        case .textEncoderNotLoaded:
            return "Z-Image 文字編碼器尚未載入。"
        case .transformerNotLoaded:
            return "Z-Image Transformer 尚未載入。"
        case .vaeNotLoaded:
            return "Z-Image VAE 尚未載入。"
        case .modelNotLoaded:
            return "Z-Image 模型尚未載入。"
        case let .loraError(error):
            return "LoRA 載入失敗：\(error.localizedDescription)"
        }
    }
}

public enum ZImageRuntimeError: LocalizedError, Sendable {
    case incompatibleProfile
    case unsupportedArchitecture(InferenceArchitecture)
    case modelNotFound(URL)
    case loraNotFound(URL)
    case unsupportedLoRAFormat(URL)
    case pipelineFailure(String)
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
        case let .pipelineFailure(message):
            message
        case let .outputMissing(url):
            "Z-Image 完成推論但沒有產生檔案：\(url.path)"
        }
    }
}

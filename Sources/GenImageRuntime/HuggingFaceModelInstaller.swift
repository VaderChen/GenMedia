import Foundation
import GenImageCore
import GenImageGGUF

public struct ModelInstallProgress: Sendable {
    public var fractionCompleted: Double
    public var downloadedBytes: Int64
    public var totalBytes: Int64

    public init(fractionCompleted: Double, downloadedBytes: Int64, totalBytes: Int64) {
        self.fractionCompleted = fractionCompleted
        self.downloadedBytes = downloadedBytes
        self.totalBytes = totalBytes
    }
}

public actor HuggingFaceModelInstaller {
    public static let int4ModelID = "qwen-image-edit-2511@mlx-int4"
    public static let int8ModelID = "qwen-image-edit-2511@mlx-int8"
    public static let fp16ModelID = "qwen-image-edit-2511@mlx-fp16"
    public static let zImage8BitModelID = "mzbac/z-image-turbo-8bit"
    public static let zImageMLX2BitModelID = "andrevp/Z-Image-Turbo-MLX-2bit"
    public static let zImageMLX4BitModelID = "andrevp/Z-Image-Turbo-MLX-4bit"
    public static let zImageMLX8BitModelID = "andrevp/Z-Image-Turbo-MLX-8bit"
    public static let zImageGiniiki4BitModelID = "Giniiki/Z-Image-Turbo-mlx-4bit"
    public static let zImageFP16ModelID = "Tongyi-MAI/Z-Image-Turbo"
    public static let zImagePixelArtLoRAModelID = "tarn59/pixel_art_style_lora_z_image_turbo"
    public static let zImageRealismLoRAModelID = "suayptalha/Z-Image-Turbo-Realism-LoRA"
    public static let zImageClassicPaintingLoRAModelID = "renderartist/Classic-Painting-Z-Image-Turbo-LoRA"
    public static let zImageColoringBookLoRAModelID = "renderartist/Coloring-Book-Z-Image-Turbo-LoRA"
    public static let civitaiAsianBeautiesLoRAModelID = "civitai/2465401"
    public static let civitaiLightningLoRAModelID = "civitai/2709343"
    public static let civitaiFlatAnimeLoRAModelID = "civitai/2449645"
    public static let civitaiDioramaLoRAModelID = "civitai/2608073"
    public static let ltx23UnionControlLoRAModelID = "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control"
    public static let captionerModelID = "local-captioner-3b@q4"
    public static let nsfwCaptionerModelID = "qwen3-vl-8b-nsfw-caption-v45@mxfp4"
    public static let qwen35Multimodal4BModelID = "lmstudio-community/Qwen3.5-4B-MLX-4bit"
    public static let qwen35Multimodal9BModelID = "lmstudio-community/Qwen3.5-9B-MLX-4bit"
    public static let qwen38Multimodal27BModelID = "lmstudio-community/Qwen3.8-27B-MLX-4bit"
    public static let ltx23DistilledModelID = "Lightricks/LTX-2.3@distilled-1.1"
    public static let ltx23MLXQ4ModelID = "dgrauet/ltx-2.3-mlx-q4"
    public static let ltxVideo096GGUFQ4KMModelID = "city96/LTX-Video-0.9.6-distilled-gguf@Q4_K_M"
    public static let ltxVideo096T5Q4KMModelID = "city96/t5-v1_1-xxl-encoder-gguf@Q4_K_M"
    public static let ltxVideo096VAEModelID = "city96/LTX-Video-0.9.6-VAE@BF16"
    public static let ltx23GGUFDistilledQ3KMModelID = "unsloth/LTX-2.3-GGUF@distilled-1.1-Q3_K_M"
    public static let ltx23GGUFVAEModelID = "unsloth/LTX-2.3-GGUF@distilled-1.1-VAE"
    public static let miniMaxH3MLX8BitModelID = "pipenetwork/MiniMax-H3-MLX-8bit"
    public static let miniMaxH3MLX4BitModelID = "pipenetwork/MiniMax-H3-MLX-4bit"
    public static let miniMaxH3GGUFFL2VAPrunedQ4KModelID = "unsloth/MiniMax-H3-GGUF@fl2va-pruned-Q4_K"
    public static let miniMaxH3GGUFFL2VAQ40ModelID = "Abiray/MiniMax-H3-GGUF@fl2va-Q4_0"
    public static let miniMaxH3GGUFFL2VAQ4KMModelID = "Abiray/MiniMax-H3-GGUF@fl2va-Q4_K_M"
    public static let miniMaxH3GGUFFL2VAQ4KSModelID = "Abiray/MiniMax-H3-GGUF@fl2va-Q4_K_S"
    public static let miniMaxH3GGUFRef2VAQ40ModelID = "Abiray/MiniMax-H3-GGUF@ref2va-Q4_0"
    public static let miniMaxH3GGUFRef2VAQ4KMModelID = "Abiray/MiniMax-H3-GGUF@ref2va-Q4_K_M"
    public static let miniMaxH3GGUFRef2VAQ4KSModelID = "Abiray/MiniMax-H3-GGUF@ref2va-Q4_K_S"
    public static let aceStep15TurboModelID = "ACE-Step/Ace-Step1.5"
    public static let miniMaxMusic3MLX8BitModelID = "vanch007/MiniMax-Music3-MLX-8bit"
    public static let miniMaxMusic3MLX4BitModelID = "mlx-community/MiniMax-Music3-4bit"
    public static let miniMaxMusic3GGUFModelID = "audio-cpp/MiniMax-Music3-GGUF@Q4_K-LM-Q4_K-DiT-Q8_0-depth"
    public static let miniMaxMusic3ComposerModelID = "Mothersuperior/minimax-music3-composer-5.7b-distilled"
    public static let whisperLargeV3TurboCoreMLModelID = "argmaxinc/whisperkit-coreml@large-v3-turbo"
    public static let whisperSmallCoreMLModelID = "argmaxinc/whisperkit-coreml@small"
    public static let paraformerChineseCoreMLModelID = "FluidInference/paraformer-large-zh-coreml"
    public static let parakeetJapaneseCoreMLModelID = "FluidInference/parakeet-0.6b-ja-coreml"
    public static let realESRGAN4xModelID = "realesrgan-x4@coreml"
    public static let realESRGAN2xModelID = "realesrgan-x2@coreml"

    private struct SourcePlan: Sendable {
        var repository: String
        var revision: String = "main"
        var destinationSubdirectory: String
        var prefixes: [String]
        var exactFiles: Set<String>
        var directURL: URL?
        var directFileName: String?
        var directSize: Int64?

        init(
            repository: String,
            revision: String = "main",
            destinationSubdirectory: String,
            prefixes: [String],
            exactFiles: Set<String>,
            directURL: URL? = nil,
            directFileName: String? = nil,
            directSize: Int64? = nil
        ) {
            self.repository = repository
            self.revision = revision
            self.destinationSubdirectory = destinationSubdirectory
            self.prefixes = prefixes
            self.exactFiles = exactFiles
            self.directURL = directURL
            self.directFileName = directFileName
            self.directSize = directSize
        }

        func includes(_ path: String) -> Bool {
            exactFiles.contains(path) || prefixes.contains { path.hasPrefix($0) }
        }
    }

    private struct InstallPlan: Sendable {
        var directoryName: String
        var runtimeRelativePath: String?
        var requiredRuntimeFiles: Set<String>
        var sources: [SourcePlan]

        init(
            directoryName: String,
            runtimeRelativePath: String? = nil,
            requiredRuntimeFiles: Set<String> = [],
            sources: [SourcePlan]
        ) {
            self.directoryName = directoryName
            self.runtimeRelativePath = runtimeRelativePath
            self.requiredRuntimeFiles = requiredRuntimeFiles
            self.sources = sources
        }
    }

    private struct HubTreeEntry: Decodable, Sendable {
        var type: String
        var size: Int64
        var path: String
    }

    private struct ManifestFile: Codable, Sendable {
        var relativePath: String
        var remotePath: String?
        var size: Int64
        var repository: String
        var revision: String
    }

    private struct InstallManifest: Codable, Sendable {
        var schemaVersion: Int
        var modelID: String
        var installedAt: Date
        var files: [ManifestFile]
    }

    private struct QuantizeConfig: Decodable, Sendable {
        var quantization: QuantizeSpecification
    }

    private struct ComponentConfig: Decodable, Sendable {
        var quantization: QuantizeSpecification?
    }

    private struct QuantizeSpecification: Decodable, Sendable {
        var bits: Int
        var groupSize: Int
        var mode: String?

        enum CodingKeys: String, CodingKey {
            case bits
            case groupSize = "group_size"
            case mode
        }
    }

    private struct GeneratedQuantizationManifest: Encodable, Sendable {
        var modelId: String?
        var groupSize: Int
        var bits: Int
        var mode: String
        var layers: [GeneratedQuantizedLayer]

        enum CodingKeys: String, CodingKey {
            case modelId = "model_id"
            case groupSize = "group_size"
            case bits
            case mode
            case layers
        }
    }

    private struct GeneratedQuantizedLayer: Encodable, Sendable {
        var name: String
        var shape: [Int]
        var inDim: Int
        var outDim: Int
        var file: String

        enum CodingKeys: String, CodingKey {
            case name
            case shape
            case inDim = "in_dim"
            case outDim = "out_dim"
            case file
        }
    }

    private struct ResolvedFile: Sendable {
        var repository: String
        var revision: String
        var remotePath: String
        var relativePath: String
        var size: Int64
        var downloadURL: URL?
    }

    public init() {}

    public nonisolated static func supports(modelID: String) -> Bool {
        plan(for: modelID) != nil
    }

    public nonisolated static func installationDirectory(modelID: String, rootURL: URL) -> URL? {
        guard let plan = plan(for: modelID) else { return nil }
        return rootURL.appendingPathComponent(plan.directoryName, isDirectory: true)
    }

    public func install(
        modelID: String,
        rootURL: URL,
        civitaiToken: String? = nil,
        huggingFaceToken: String? = nil,
        progress: @escaping @Sendable (ModelInstallProgress) -> Void
    ) async throws -> URL {
        guard let plan = Self.plan(for: modelID) else {
            throw ModelInstallerError.unsupportedModel(modelID)
        }
        try Task.checkCancellation()
        try FileManager.default.createDirectory(at: rootURL, withIntermediateDirectories: true)

        let destination = rootURL.appendingPathComponent(plan.directoryName, isDirectory: true)
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)

        let files = try await resolveFiles(plan: plan, huggingFaceToken: huggingFaceToken)
        guard !files.isEmpty else { throw ModelInstallerError.emptyRepository(modelID) }
        let totalBytes = files.reduce(Int64(0)) { $0 + $1.size }
        let completedBytes = files.reduce(Int64(0)) { result, file in
            let url = destination.appendingPathComponent(file.relativePath)
            return result + (Self.fileSize(at: url) == file.size ? file.size : 0)
        }
        let progressTracker = DownloadProgressTracker(
            initialBytes: completedBytes,
            totalBytes: totalBytes,
            progress: progress
        )
        progressTracker.emit()

        // Link already available files first. Network downloads are scheduled below
        // with a small concurrency limit so multi-shard models can use more of the
        // available bandwidth without creating an unbounded number of connections.
        var pendingDownloads: [ResolvedFile] = []
        for file in files {
            try Task.checkCancellation()
            let fileURL = destination.appendingPathComponent(file.relativePath)
            if Self.fileSize(at: fileURL) == file.size { continue }

            try FileManager.default.createDirectory(
                at: fileURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )

            if let reusableURL = Self.reusableFile(
                for: file,
                rootURL: rootURL,
                excluding: destination
            ) {
                if FileManager.default.fileExists(atPath: fileURL.path) {
                    try FileManager.default.removeItem(at: fileURL)
                }
                do {
                    try FileManager.default.linkItem(at: reusableURL, to: fileURL)
                    progressTracker.markCompleted(file.relativePath, bytes: file.size)
                    continue
                } catch {
                    if FileManager.default.fileExists(atPath: fileURL.path) {
                        try? FileManager.default.removeItem(at: fileURL)
                    }
                }
            }
            pendingDownloads.append(file)
        }
        // Start the largest shards first so small metadata files do not occupy
        // download slots while the multi-gigabyte weights wait in the queue.
        pendingDownloads.sort { lhs, rhs in
            if lhs.size != rhs.size { return lhs.size > rhs.size }
            return lhs.relativePath.localizedStandardCompare(rhs.relativePath) == .orderedAscending
        }

        var iterator = pendingDownloads.makeIterator()
        let maxConcurrentDownloads = min(4, max(1, pendingDownloads.count))
        try await withThrowingTaskGroup(of: Void.self) { group in
            for _ in 0..<maxConcurrentDownloads {
                guard let file = iterator.next() else { break }
                group.addTask {
                    try await self.download(
                        file,
                        destination: destination,
                        civitaiToken: civitaiToken,
                        huggingFaceToken: huggingFaceToken,
                        progressTracker: progressTracker
                    )
                }
            }

            while try await group.next() != nil {
                try Task.checkCancellation()
                guard let file = iterator.next() else { continue }
                group.addTask {
                    try await self.download(
                        file,
                        destination: destination,
                        civitaiToken: civitaiToken,
                        huggingFaceToken: huggingFaceToken,
                        progressTracker: progressTracker
                    )
                }
            }
        }

        progressTracker.emit()

        try Self.materializeQuantizationManifest(at: destination)
        try Self.validateRequiredRuntimeFiles(for: plan, at: destination)
        try Self.validateGGUFWeights(modelID: modelID, at: destination)

        let manifest = InstallManifest(
            schemaVersion: 2,
            modelID: modelID,
            installedAt: .now,
            files: files.map {
                ManifestFile(
                    relativePath: $0.relativePath,
                    remotePath: $0.remotePath,
                    size: $0.size,
                    repository: $0.repository,
                    revision: $0.revision
                )
            }
        )
        let manifestData = try JSONEncoder.genImageManifest.encode(manifest)
        try manifestData.write(
            to: destination.appendingPathComponent("genimage-model.json"),
            options: .atomic
        )
        return Self.runtimeURL(for: plan, destination: destination)
    }

    private func download(
        _ file: ResolvedFile,
        destination: URL,
        civitaiToken: String?,
        huggingFaceToken: String?,
        progressTracker: DownloadProgressTracker
    ) async throws {
        try Task.checkCancellation()
        let fileURL = destination.appendingPathComponent(file.relativePath)

        // A single multi-gigabyte shard can otherwise be limited by the
        // throughput of one CDN connection. Use HTTP Range requests for large
        // files, and fall back to the regular resumable download if the source
        // does not support byte ranges.
        if file.size >= 256 * 1_048_576 {
            let segmentKeys = Self.segmentKeys(for: file)
            do {
                try await downloadSegmented(
                    file,
                    fileURL: fileURL,
                    civitaiToken: civitaiToken,
                    huggingFaceToken: huggingFaceToken,
                    progressTracker: progressTracker,
                    segmentKeys: segmentKeys
                )
                progressTracker.finalizeSegmented(
                    file.relativePath,
                    segmentKeys: segmentKeys,
                    bytes: file.size
                )
                return
            } catch {
                if Task.isCancelled { throw error }
                progressTracker.discard(segmentKeys)
                try? FileManager.default.removeItem(at: fileURL.appendingPathExtension("segments"))
            }
        }

        let request = try Self.downloadRequest(
            for: file,
            civitaiToken: civitaiToken,
            huggingFaceToken: huggingFaceToken
        )
        let resumeDataURL = fileURL.appendingPathExtension("resume")
        let hasResumeData = (try? Data(contentsOf: resumeDataURL))?.isEmpty == false
        let downloader = FileDownloadDelegate(
            destination: fileURL,
            expectedBytes: file.size,
            progress: { received, _ in
                progressTracker.report(file.relativePath, received: received)
            }
        )
        do {
            try await downloader.start(request: request)
        } catch let error as ModelInstallerError {
            try Task.checkCancellation()
            guard case .httpStatus(401, _) = error else { throw error }
            guard hasResumeData else { throw error }

            try? FileManager.default.removeItem(at: resumeDataURL)
            let retryDownloader = FileDownloadDelegate(
                destination: fileURL,
                expectedBytes: file.size,
                progress: { received, _ in
                    progressTracker.report(file.relativePath, received: received)
                }
            )
            try await retryDownloader.start(request: request)
        } catch {
            try Task.checkCancellation()
            throw error
        }
        let actual = Self.fileSize(at: fileURL)
        guard actual == file.size else {
            throw ModelInstallerError.sizeMismatch(
                path: file.relativePath,
                expected: file.size,
                actual: actual
            )
        }
        progressTracker.markCompleted(file.relativePath, bytes: file.size)
    }

    private func downloadSegmented(
        _ file: ResolvedFile,
        fileURL: URL,
        civitaiToken: String?,
        huggingFaceToken: String?,
        progressTracker: DownloadProgressTracker,
        segmentKeys: [String]
    ) async throws {
        let segmentCount = segmentKeys.count
        let segmentSize = file.size / Int64(segmentCount)
        let stagingDirectory = fileURL.appendingPathExtension("segments")
        try FileManager.default.createDirectory(
            at: stagingDirectory,
            withIntermediateDirectories: true
        )

        var ranges: [(start: Int64, end: Int64)] = []
        for index in 0..<segmentCount {
            let start = Int64(index) * segmentSize
            let end = index == segmentCount - 1
                ? file.size - 1
                : start + segmentSize - 1
            ranges.append((start, end))
        }

        try await withThrowingTaskGroup(of: Void.self) { group in
            for (index, range) in ranges.enumerated() {
                let partURL = stagingDirectory.appendingPathComponent(
                    String(format: "part-%02d", index)
                )
                let expectedBytes = range.end - range.start + 1
                let key = segmentKeys[index]
                group.addTask {
                    try Task.checkCancellation()
                    if Self.fileSize(at: partURL) == expectedBytes {
                        progressTracker.report(key, received: expectedBytes)
                        return
                    }
                    var request = try Self.downloadRequest(
                        for: file,
                        civitaiToken: civitaiToken,
                        huggingFaceToken: huggingFaceToken
                    )
                    request.setValue(
                        "bytes=\(range.start)-\(range.end)",
                        forHTTPHeaderField: "Range"
                    )
                    request.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
                    let downloader = FileDownloadDelegate(
                        destination: partURL,
                        expectedBytes: expectedBytes,
                        progress: { received, _ in
                            progressTracker.report(key, received: received)
                        }
                    )
                    try await downloader.start(request: request)
                    guard Self.fileSize(at: partURL) == expectedBytes else {
                        throw ModelInstallerError.sizeMismatch(
                            path: file.relativePath,
                            expected: expectedBytes,
                            actual: Self.fileSize(at: partURL)
                        )
                    }
                }
            }
            while try await group.next() != nil {
                try Task.checkCancellation()
            }
        }

        let stagedURL = fileURL.appendingPathExtension("staged")
        try? FileManager.default.removeItem(at: stagedURL)
        FileManager.default.createFile(atPath: stagedURL.path, contents: nil)
        let output = try FileHandle(forWritingTo: stagedURL)
        defer { try? output.close() }
        for index in 0..<segmentCount {
            let partURL = stagingDirectory.appendingPathComponent(
                String(format: "part-%02d", index)
            )
            let input = try FileHandle(forReadingFrom: partURL)
            defer { try? input.close() }
            while let data = try input.read(upToCount: 1_048_576), !data.isEmpty {
                try output.write(contentsOf: data)
            }
        }
        try output.close()
        guard Self.fileSize(at: stagedURL) == file.size else {
            throw ModelInstallerError.sizeMismatch(
                path: file.relativePath,
                expected: file.size,
                actual: Self.fileSize(at: stagedURL)
            )
        }
        if FileManager.default.fileExists(atPath: fileURL.path) {
            try FileManager.default.removeItem(at: fileURL)
        }
        try FileManager.default.moveItem(at: stagedURL, to: fileURL)
        try? FileManager.default.removeItem(at: stagingDirectory)
    }

    private nonisolated static func segmentKeys(for file: ResolvedFile) -> [String] {
        let count = min(4, max(2, Int(ceil(Double(file.size) / Double(256 * 1_048_576)))))
        return (0..<count).map { "\(file.relativePath)#segment-\($0)" }
    }

    public nonisolated static func verify(modelID: String, rootURL: URL) throws -> URL {
        guard let plan = plan(for: modelID) else {
            throw ModelInstallerError.unsupportedModel(modelID)
        }
        let destination = rootURL.appendingPathComponent(plan.directoryName, isDirectory: true)
        try materializeQuantizationManifest(at: destination)
        let manifestURL = destination.appendingPathComponent("genimage-model.json")
        guard let data = try? Data(contentsOf: manifestURL),
              let manifest = try? JSONDecoder.genImageManifest.decode(InstallManifest.self, from: data),
              manifest.modelID == modelID,
              !manifest.files.isEmpty else {
            throw ModelInstallerError.invalidManifest(manifestURL)
        }
        for file in manifest.files {
            let url = destination.appendingPathComponent(file.relativePath)
            let actual = fileSize(at: url)
            guard actual == file.size else {
                throw ModelInstallerError.sizeMismatch(
                    path: file.relativePath,
                    expected: file.size,
                    actual: actual
                )
            }
        }
        try validateRequiredRuntimeFiles(for: plan, at: destination)
        try validateGGUFWeights(modelID: modelID, at: destination)
        let runtimeURL = runtimeURL(for: plan, destination: destination)
        guard FileManager.default.fileExists(atPath: runtimeURL.path) else {
            throw ModelInstallerError.runtimeNotFound(runtimeURL)
        }
        return runtimeURL
    }

    public nonisolated static func remove(modelID: String, rootURL: URL) throws {
        guard let destination = installationDirectory(modelID: modelID, rootURL: rootURL) else {
            throw ModelInstallerError.unsupportedModel(modelID)
        }
        guard FileManager.default.fileExists(atPath: destination.path) else { return }
        try FileManager.default.removeItem(at: destination)
    }

    private func resolveFiles(
        plan: InstallPlan,
        huggingFaceToken: String?
    ) async throws -> [ResolvedFile] {
        var result: [ResolvedFile] = []
        for source in plan.sources {
            if let directURL = source.directURL {
                guard let fileName = source.directFileName,
                      let size = source.directSize,
                      size > 0 else {
                    throw ModelInstallerError.noMatchingFiles(source.repository)
                }
                result.append(
                    ResolvedFile(
                        repository: source.repository,
                        revision: source.revision,
                        remotePath: fileName,
                        relativePath: source.destinationSubdirectory.isEmpty
                            ? fileName
                            : source.destinationSubdirectory + "/" + fileName,
                        size: size,
                        downloadURL: directURL
                    )
                )
                continue
            }
            let tree = try await Self.fetchTree(
                repository: source.repository,
                revision: source.revision,
                huggingFaceToken: huggingFaceToken
            )
            let selected = tree.filter {
                $0.type == "file"
                    && $0.size >= 0
                    && source.includes($0.path)
                    && Self.isSafeRelativePath($0.path)
            }
            guard !selected.isEmpty else {
                throw ModelInstallerError.noMatchingFiles(source.repository)
            }
            result.append(contentsOf: selected.map {
                let relativePath = source.destinationSubdirectory.isEmpty
                    ? $0.path
                    : source.destinationSubdirectory + "/" + $0.path
                return ResolvedFile(
                    repository: source.repository,
                    revision: source.revision,
                    remotePath: $0.path,
                    relativePath: relativePath,
                    size: $0.size,
                    downloadURL: nil
                )
            })
        }
        return result.sorted { $0.relativePath.localizedStandardCompare($1.relativePath) == .orderedAscending }
    }

    private nonisolated static func fetchTree(
        repository: String,
        revision: String,
        huggingFaceToken: String?
    ) async throws -> [HubTreeEntry] {
        guard let encodedRevision = revision.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed),
              let url = URL(
                string: "https://huggingface.co/api/models/\(repository)/tree/\(encodedRevision)?recursive=true&expand=false&limit=1000"
              ) else {
            throw ModelInstallerError.invalidRepository(repository)
        }
        var request = URLRequest(url: url)
        request.timeoutInterval = 60
        request.setValue("GenImage/1.0", forHTTPHeaderField: "User-Agent")
        Self.applyAuthorization(to: &request, token: huggingFaceToken)
        let (data, response) = try await URLSession.shared.data(for: request)
        try Self.validate(response: response, data: data)
        return try JSONDecoder().decode([HubTreeEntry].self, from: data)
    }

    private nonisolated static func downloadRequest(
        for file: ResolvedFile,
        civitaiToken: String? = nil,
        huggingFaceToken: String? = nil
    ) throws -> URLRequest {
        if let downloadURL = file.downloadURL {
            var request = URLRequest(url: downloadURL)
            request.timeoutInterval = 60 * 60 * 24
            request.setValue("GenImage/1.0", forHTTPHeaderField: "User-Agent")
            if let token = civitaiToken?.trimmingCharacters(in: .whitespacesAndNewlines),
               !token.isEmpty {
                request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
            } else if let token = CivitaiTokenStore.token() {
                request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
            }
            return request
        }
        let pathSegments = file.remotePath.split(separator: "/").map(String.init)
        var url = URL(string: "https://huggingface.co/\(file.repository)/resolve/\(file.revision)")!
        for segment in pathSegments { url.appendPathComponent(segment) }
        var request = URLRequest(url: url)
        request.timeoutInterval = 60 * 60 * 24
        request.setValue("GenImage/1.0", forHTTPHeaderField: "User-Agent")
        applyAuthorization(to: &request, token: huggingFaceToken)
        return request
    }

    private nonisolated static func applyAuthorization(
        to request: inout URLRequest,
        token: String? = nil
    ) {
        let token = token?.trimmingCharacters(in: .whitespacesAndNewlines)
            ?? huggingFaceToken()
        guard let token, !token.isEmpty else { return }
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
    }

    /// Prefer the token saved by the app, then reuse HF_TOKEN or the token saved
    /// by `huggingface-cli login` / `hf auth login` when available. This keeps
    /// model downloads authenticated so Hugging Face does not apply the
    /// anonymous throughput limit.
    private nonisolated static func huggingFaceToken() -> String? {
        HuggingFaceTokenStore.token()
    }

    private nonisolated static func validate(response: URLResponse, data: Data) throws {
        guard let http = response as? HTTPURLResponse else {
            throw ModelInstallerError.invalidResponse
        }
        guard (200..<300).contains(http.statusCode) else {
            let message = String(data: data.prefix(2_048), encoding: .utf8) ?? ""
            throw ModelInstallerError.httpStatus(http.statusCode, message)
        }
    }

    private nonisolated static func plan(for modelID: String) -> InstallPlan? {
        let officialFiles: Set<String> = ["model_index.json"]
        let officialPrefixes = [
            "processor/", "scheduler/", "text_encoder/", "tokenizer/", "transformer/", "vae/"
        ]
        switch modelID {
        case zImage8BitModelID:
            return InstallPlan(
                directoryName: "z-image-turbo-8bit",
                sources: [
                    SourcePlan(
                        repository: "mzbac/Z-Image-Turbo-8bit",
                        destinationSubdirectory: "",
                        prefixes: [
                            "scheduler/", "text_encoder/", "tokenizer/", "transformer/", "vae/"
                        ],
                        exactFiles: ["model_index.json", "quantization.json"]
                    )
                ]
            )
        case zImageFP16ModelID:
            return InstallPlan(
                directoryName: "z-image-turbo-fp16",
                sources: [
                    SourcePlan(
                        repository: "Tongyi-MAI/Z-Image-Turbo",
                        destinationSubdirectory: "",
                        prefixes: [
                            "scheduler/", "text_encoder/", "tokenizer/", "transformer/", "vae/"
                        ],
                        exactFiles: ["model_index.json"]
                    )
                ]
            )
        case zImageMLX2BitModelID:
            return zImageMLXPlan(
                repository: zImageMLX2BitModelID,
                directoryName: "z-image-turbo-mlx-2bit"
            )
        case zImageMLX4BitModelID:
            return zImageMLXPlan(
                repository: zImageMLX4BitModelID,
                directoryName: "z-image-turbo-mlx-4bit"
            )
        case zImageMLX8BitModelID:
            return zImageMLXPlan(
                repository: zImageMLX8BitModelID,
                directoryName: "z-image-turbo-mlx-8bit"
            )
        case zImageGiniiki4BitModelID:
            return zImageMLXPlan(
                repository: zImageGiniiki4BitModelID,
                directoryName: "z-image-turbo-giniiki-4bit"
            )
        case zImagePixelArtLoRAModelID:
            return InstallPlan(
                directoryName: "loras/z-image-pixel-art",
                runtimeRelativePath: "pixel_art_style_z_image_turbo.safetensors",
                sources: [
                    SourcePlan(
                        repository: "tarn59/pixel_art_style_lora_z_image_turbo",
                        destinationSubdirectory: "",
                        prefixes: [],
                        exactFiles: ["pixel_art_style_z_image_turbo.safetensors"]
                    )
                ]
            )
        case zImageRealismLoRAModelID:
            return InstallPlan(
                directoryName: "loras/z-image-realism",
                runtimeRelativePath: "pytorch_lora_weights.safetensors",
                sources: [
                    SourcePlan(
                        repository: "suayptalha/Z-Image-Turbo-Realism-LoRA",
                        destinationSubdirectory: "",
                        prefixes: [],
                        exactFiles: ["pytorch_lora_weights.safetensors"]
                    )
                ]
            )
        case zImageClassicPaintingLoRAModelID:
            return InstallPlan(
                directoryName: "loras/z-image-classic-painting",
                runtimeRelativePath: "Classic_Painting_Z_Image_Turbo_v1_renderartist_1750.safetensors",
                sources: [
                    SourcePlan(
                        repository: "renderartist/Classic-Painting-Z-Image-Turbo-LoRA",
                        destinationSubdirectory: "",
                        prefixes: [],
                        exactFiles: ["Classic_Painting_Z_Image_Turbo_v1_renderartist_1750.safetensors"]
                    )
                ]
            )
        case zImageColoringBookLoRAModelID:
            return InstallPlan(
                directoryName: "loras/z-image-coloring-book",
                runtimeRelativePath: "Coloring_Book_Z_Image_Turbo_v1_renderartist_2000.safetensors",
                sources: [
                    SourcePlan(
                        repository: "renderartist/Coloring-Book-Z-Image-Turbo-LoRA",
                        destinationSubdirectory: "",
                        prefixes: [],
                        exactFiles: ["Coloring_Book_Z_Image_Turbo_v1_renderartist_2000.safetensors"]
                    )
                ]
            )
        case civitaiAsianBeautiesLoRAModelID:
            return InstallPlan(
                directoryName: "loras/civitai-z-image-asian-beauties",
                runtimeRelativePath: "asian_woman_zit_v1.safetensors",
                sources: [
                    SourcePlan(
                        repository: "civitai",
                        destinationSubdirectory: "",
                        prefixes: [],
                        exactFiles: [],
                        directURL: URL(string: "https://civitai.com/api/download/models/2465401?fileId=2353982"),
                        directFileName: "asian_woman_zit_v1.safetensors",
                        directSize: 170_128_232
                    )
                ]
            )
        case civitaiLightningLoRAModelID:
            return InstallPlan(
                directoryName: "loras/civitai-z-image-lightning",
                runtimeRelativePath: "Zed_Turbo_Lightning.safetensors",
                sources: [
                    SourcePlan(
                        repository: "civitai",
                        destinationSubdirectory: "",
                        prefixes: [],
                        exactFiles: [],
                        directURL: URL(string: "https://civitai.com/api/download/models/2709343?fileId=2595263"),
                        directFileName: "Zed_Turbo_Lightning.safetensors",
                        directSize: 35_180_808
                    )
                ]
            )
        case civitaiFlatAnimeLoRAModelID:
            return InstallPlan(
                directoryName: "loras/civitai-z-image-flat-anime",
                runtimeRelativePath: "UU_000000960.safetensors",
                sources: [
                    SourcePlan(
                        repository: "civitai",
                        destinationSubdirectory: "",
                        prefixes: [],
                        exactFiles: [],
                        directURL: URL(string: "https://civitai.com/api/download/models/2449645?fileId=2340786"),
                        directFileName: "UU_000000960.safetensors",
                        directSize: 170_128_200
                    )
                ]
            )
        case civitaiDioramaLoRAModelID:
            return InstallPlan(
                directoryName: "loras/civitai-z-image-diorama",
                runtimeRelativePath: "loonalone_diorama.safetensors",
                sources: [
                    SourcePlan(
                        repository: "civitai",
                        destinationSubdirectory: "",
                        prefixes: [],
                        exactFiles: [],
                        directURL: URL(string: "https://civitai.com/api/download/models/2608073?fileId=2567496"),
                        directFileName: "loonalone_diorama.safetensors",
                        directSize: 170_127_768
                    )
                ]
            )
        case ltx23UnionControlLoRAModelID:
            return InstallPlan(
                directoryName: "loras/ltx-2.3-union-control",
                runtimeRelativePath: "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
                sources: [
                    SourcePlan(
                        repository: "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
                        destinationSubdirectory: "",
                        prefixes: [],
                        exactFiles: [
                            "LICENSE",
                            "README.md",
                            "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
                        ]
                    )
                ]
            )
        case captionerModelID:
            return InstallPlan(
                directoryName: "Qwen3-VL-4B-Instruct-4bit",
                requiredRuntimeFiles: [
                    "config.json",
                    "model.safetensors",
                    "preprocessor_config.json",
                    "processor_config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "video_preprocessor_config.json"
                ],
                sources: [
                    SourcePlan(
                        repository: "mlx-community/Qwen3-VL-4B-Instruct-4bit",
                        destinationSubdirectory: "",
                        prefixes: [],
                        exactFiles: [
                            "added_tokens.json",
                            "chat_template.jinja",
                            "chat_template.json",
                            "config.json",
                            "generation_config.json",
                            "merges.txt",
                            "model.safetensors",
                            "model.safetensors.index.json",
                            "preprocessor_config.json",
                            "processor_config.json",
                            "special_tokens_map.json",
                            "tokenizer.json",
                            "tokenizer_config.json",
                            "video_preprocessor_config.json",
                            "vocab.json"
                        ]
                    )
                ]
            )
        case nsfwCaptionerModelID:
            return InstallPlan(
                directoryName: "Qwen3-VL-8B-NSFW-Caption-V4.5-mxfp4",
                sources: [
                    SourcePlan(
                        repository: "mlx-community/Qwen3-VL-8B-NSFW-Caption-V4.5-mxfp4",
                        destinationSubdirectory: "",
                        prefixes: [],
                        exactFiles: [
                            "added_tokens.json",
                            "chat_template.jinja",
                            "config.json",
                            "model-00001-of-00002.safetensors",
                            "model-00002-of-00002.safetensors",
                            "model.safetensors.index.json",
                            "preprocessor_config.json",
                            "processor_config.json",
                            "special_tokens_map.json",
                            "tokenizer.json",
                            "tokenizer_config.json",
                            "video_preprocessor_config.json",
                            "vocab.json"
                        ]
                    )
                ]
            )
        case qwen35Multimodal4BModelID:
            return qwenMultimodalPlan(
                repository: qwen35Multimodal4BModelID,
                revision: "c43ee1d65576a5d98de1e8405cac93c371a655c1",
                directoryName: "qwen3.5-4b-mlx-4bit"
            )
        case qwen35Multimodal9BModelID:
            return qwenMultimodalPlan(
                repository: qwen35Multimodal9BModelID,
                revision: "b455506b0f574c74616dbcd56879bde38fafcff3",
                directoryName: "qwen3.5-9b-mlx-4bit"
            )
        case qwen38Multimodal27BModelID:
            return qwenMultimodalPlan(
                repository: qwen38Multimodal27BModelID,
                revision: "6067b15cf581666a4aecf6af3afaba4bb5efc20c",
                directoryName: "qwen3.8-27b-mlx-4bit"
            )
        case realESRGAN4xModelID:
            return realESRGANPlan(directoryName: "realesrgan-coreml-x4")
        case realESRGAN2xModelID:
            return realESRGANPlan(directoryName: "realesrgan-coreml-x2")
        case ltx23DistilledModelID:
            return InstallPlan(
                directoryName: "ltx-2.3-distilled-1.1",
                sources: [
                    SourcePlan(
                        repository: "Lightricks/LTX-2.3",
                        destinationSubdirectory: "",
                        prefixes: [],
                        exactFiles: [
                            "LICENSE",
                            "ltx-2.3-22b-distilled-1.1.safetensors",
                            "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
                        ]
                    ),
                    SourcePlan(
                        repository: "Lightricks/gemma-3-12b-it-qat-q4_0-unquantized",
                        revision: "d62fe4f1995ade703b49a0f3c0d0f161237ef437",
                        destinationSubdirectory: "gemma-3-12b",
                        prefixes: [],
                        exactFiles: [
                            "added_tokens.json",
                            "chat_template.json",
                            "config.json",
                            "generation_config.json",
                            "model-00001-of-00005.safetensors",
                            "model-00002-of-00005.safetensors",
                            "model-00003-of-00005.safetensors",
                            "model-00004-of-00005.safetensors",
                            "model-00005-of-00005.safetensors",
                            "model.safetensors.index.json",
                            "preprocessor_config.json",
                            "processor_config.json",
                            "special_tokens_map.json",
                            "tokenizer.json",
                            "tokenizer.model",
                            "tokenizer_config.json"
                        ]
                    )
                ]
            )
        case ltx23MLXQ4ModelID:
            return InstallPlan(
                directoryName: "ltx-2.3-mlx-q4",
                sources: [
                    SourcePlan(
                        repository: "dgrauet/ltx-2.3-mlx-q4",
                        destinationSubdirectory: "",
                        prefixes: [],
                        exactFiles: [
                            "README.md",
                            "audio_vae.safetensors",
                            "config.json",
                            "connector.safetensors",
                            "embedded_config.json",
                            "quantize_config.json",
                            "spatial_upscaler_x2_v1_1.safetensors",
                            "spatial_upscaler_x2_v1_1_config.json",
                            "split_model.json",
                            "transformer-distilled-1.1.safetensors",
                            "vae_decoder.safetensors",
                            "vae_encoder.safetensors",
                            "vocoder.safetensors"
                        ]
                    ),
                    SourcePlan(
                        repository: "Lightricks/gemma-3-12b-it-qat-q4_0-unquantized",
                        revision: "d62fe4f1995ade703b49a0f3c0d0f161237ef437",
                        destinationSubdirectory: "gemma-3-12b",
                        prefixes: [],
                        exactFiles: [
                            "added_tokens.json",
                            "chat_template.json",
                            "config.json",
                            "generation_config.json",
                            "model-00001-of-00005.safetensors",
                            "model-00002-of-00005.safetensors",
                            "model-00003-of-00005.safetensors",
                            "model-00004-of-00005.safetensors",
                            "model-00005-of-00005.safetensors",
                            "model.safetensors.index.json",
                            "preprocessor_config.json",
                            "processor_config.json",
                            "special_tokens_map.json",
                            "tokenizer.json",
                            "tokenizer.model",
                            "tokenizer_config.json"
                        ]
                    )
                ]
            )
        case ltxVideo096GGUFQ4KMModelID:
            return InstallPlan(
                directoryName: "ltx-video-0.9.6-distilled-gguf",
                requiredRuntimeFiles: [
                    "LTX-Video-0.9.6-VAE-BF16.safetensors",
                    "ltxv-2b-0.9.6-distilled-04-25-Q4_K_M.gguf",
                    "text_encoder/config.json",
                    "text_encoder/t5-v1_1-xxl-encoder-Q4_K_M.gguf",
                    "tokenizer/spiece.model"
                ],
                sources: [
                    SourcePlan(
                        repository: "city96/LTX-Video-0.9.6-distilled-gguf",
                        revision: "f5ccd5ad1821ff03addbb1bc97a9f0829adc1026",
                        destinationSubdirectory: "",
                        prefixes: [],
                        exactFiles: [
                            "LICENSE.md",
                            "README.md",
                            "LTX-Video-0.9.6-VAE-BF16.safetensors",
                            "ltxv-2b-0.9.6-distilled-04-25-Q4_K_M.gguf"
                        ]
                    ),
                    SourcePlan(
                        repository: "city96/t5-v1_1-xxl-encoder-gguf",
                        revision: "005a6ea51a7d0b84d677b3e633bb52a8c85a83d9",
                        destinationSubdirectory: "text_encoder",
                        prefixes: [],
                        exactFiles: ["t5-v1_1-xxl-encoder-Q4_K_M.gguf"]
                    ),
                    SourcePlan(
                        repository: "Lightricks/LTX-Video",
                        revision: "8984fa25007f376c1a299016d0957a37a2f797bb",
                        destinationSubdirectory: "",
                        prefixes: [],
                        exactFiles: [
                            "text_encoder/config.json",
                            "tokenizer/added_tokens.json",
                            "tokenizer/special_tokens_map.json",
                            "tokenizer/spiece.model",
                            "tokenizer/tokenizer_config.json"
                        ]
                    )
                ]
            )
        case ltxVideo096T5Q4KMModelID:
            return InstallPlan(
                directoryName: "ltx-video-0.9.6-t5-q4-k-m",
                requiredRuntimeFiles: [
                    "t5-v1_1-xxl-encoder-Q4_K_M.gguf"
                ],
                sources: [
                    SourcePlan(
                        repository: "city96/t5-v1_1-xxl-encoder-gguf",
                        revision: "005a6ea51a7d0b84d677b3e633bb52a8c85a83d9",
                        destinationSubdirectory: "",
                        prefixes: [],
                        exactFiles: [
                            "README.md",
                            "t5-v1_1-xxl-encoder-Q4_K_M.gguf"
                        ]
                    )
                ]
            )
        case ltxVideo096VAEModelID:
            return InstallPlan(
                directoryName: "ltx-video-0.9.6-vae-bf16",
                requiredRuntimeFiles: [
                    "LTX-Video-0.9.6-VAE-BF16.safetensors"
                ],
                sources: [
                    SourcePlan(
                        repository: "city96/LTX-Video-0.9.6-distilled-gguf",
                        revision: "f5ccd5ad1821ff03addbb1bc97a9f0829adc1026",
                        destinationSubdirectory: "",
                        prefixes: [],
                        exactFiles: [
                            "LTX-Video-0.9.6-VAE-BF16.safetensors"
                        ]
                    )
                ]
            )
        case ltx23GGUFDistilledQ3KMModelID:
            return InstallPlan(
                directoryName: "ltx-2.3-distilled-1.1-gguf-q3-k-m",
                requiredRuntimeFiles: [
                    "distilled-1.1/ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf",
                    "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
                    "text_encoders/gemma-3-12b-it-qat-UD-Q4_K_XL.gguf",
                    "text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors",
                    "vae/ltx-2.3-22b-distilled_audio_vae.safetensors",
                    "vae/ltx-2.3-22b-distilled_video_vae.safetensors",
                    "gemma-3-12b/config.json",
                    "gemma-3-12b/tokenizer.json",
                    "gemma-3-12b/tokenizer.model",
                    "gemma-3-12b/tokenizer_config.json"
                ],
                sources: [
                    SourcePlan(
                        repository: "unsloth/LTX-2.3-GGUF",
                        revision: "96e8ed4925ead3db9ff4d0084f165ef6a74f28d0",
                        destinationSubdirectory: "",
                        prefixes: [],
                        exactFiles: [
                            "LICENSE",
                            "README.md",
                            "distilled-1.1/ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf"
                        ]
                    ),
                    SourcePlan(
                        repository: "unsloth/LTX-2.3-GGUF",
                        revision: "96e8ed4925ead3db9ff4d0084f165ef6a74f28d0",
                        destinationSubdirectory: "",
                        prefixes: [],
                        exactFiles: [
                            "text_encoders/gemma-3-12b-it-qat-UD-Q4_K_XL.gguf",
                            "text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors",
                            "vae/ltx-2.3-22b-distilled_audio_vae.safetensors",
                            "vae/ltx-2.3-22b-distilled_video_vae.safetensors"
                        ]
                    ),
                    SourcePlan(
                        repository: "unsloth/gemma-3-12b-it-qat-GGUF",
                        revision: "858acec7ec0541a46c39985c95d3b52d8f3ab183",
                        destinationSubdirectory: "text_encoders",
                        prefixes: [],
                        exactFiles: ["gemma-3-12b-it-qat-UD-Q4_K_XL.gguf"]
                    ),
                    SourcePlan(
                        repository: "Lightricks/LTX-2.3",
                        destinationSubdirectory: "",
                        prefixes: [],
                        exactFiles: ["ltx-2.3-spatial-upscaler-x2-1.1.safetensors"]
                    ),
                    SourcePlan(
                        repository: "Lightricks/gemma-3-12b-it-qat-q4_0-unquantized",
                        revision: "d62fe4f1995ade703b49a0f3c0d0f161237ef437",
                        destinationSubdirectory: "gemma-3-12b",
                        prefixes: [],
                        exactFiles: [
                            "added_tokens.json",
                            "chat_template.json",
                            "config.json",
                            "special_tokens_map.json",
                            "tokenizer.json",
                            "tokenizer.model",
                            "tokenizer_config.json"
                        ]
                    )
                ]
            )
        case ltx23GGUFVAEModelID:
            return InstallPlan(
                directoryName: "ltx-2.3-distilled-1.1-vae",
                requiredRuntimeFiles: [
                    "vae/ltx-2.3-22b-distilled_audio_vae.safetensors",
                    "vae/ltx-2.3-22b-distilled_video_vae.safetensors"
                ],
                sources: [
                    SourcePlan(
                        repository: "unsloth/LTX-2.3-GGUF",
                        revision: "96e8ed4925ead3db9ff4d0084f165ef6a74f28d0",
                        destinationSubdirectory: "",
                        prefixes: [],
                        exactFiles: [
                            "vae/ltx-2.3-22b-distilled_audio_vae.safetensors",
                            "vae/ltx-2.3-22b-distilled_video_vae.safetensors"
                        ]
                    )
                ]
            )
        case miniMaxH3MLX8BitModelID:
            return miniMaxH3MLXPlan(
                repository: "pipenetwork/MiniMax-H3-MLX-8bit",
                directoryName: "minimax-h3-mlx-8bit"
            )
        case miniMaxH3MLX4BitModelID:
            return miniMaxH3MLXPlan(
                repository: "pipenetwork/MiniMax-H3-MLX-4bit",
                directoryName: "minimax-h3-mlx-4bit"
            )
        case miniMaxH3GGUFFL2VAPrunedQ4KModelID:
            return InstallPlan(
                directoryName: "minimax-h3-gguf-fl2va-pruned-q4-k",
                requiredRuntimeFiles: [
                    "minimax_h3_fl2va_pruned-Q4_K.gguf",
                    "qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
                    "vae/minimax_h3_audio_vae_fp32.safetensors",
                    "vae/minimax_h3_video_vae_fp16.safetensors",
                    "upstream/FL2VA/processor/tokenizer.json",
                    "upstream/FL2VA/processor/tokenizer_config.json"
                ],
                sources: [
                    SourcePlan(
                        repository: "unsloth/MiniMax-H3-GGUF",
                        revision: "d629413c2e5b51b38c453668b75ca3b06ca92703",
                        destinationSubdirectory: "",
                        prefixes: [],
                        exactFiles: [
                            "LICENSE",
                            "README.md",
                            "NOTICE",
                            "minimax_h3_fl2va_pruned-Q4_K.gguf",
                            "qwen3vl_32b_minimax_h3-Q4_K_M.gguf"
                        ]
                    ),
                    SourcePlan(
                        repository: "Comfy-Org/MiniMax-H3",
                        revision: "4cc1d817b6184899b41293954329f576cb5ae86b",
                        destinationSubdirectory: "",
                        prefixes: [],
                        exactFiles: [
                            "vae/minimax_h3_audio_vae_fp32.safetensors",
                            "vae/minimax_h3_video_vae_fp16.safetensors"
                        ]
                    ),
                    // Same gap as `miniMaxH3GGUFPlan`: the ComfyUI-converted
                    // Qwen3-VL encoder ships without tokenizer metadata, so the
                    // tokenizer and vision preprocessor come from upstream.
                    SourcePlan(
                        repository: "MiniMaxAI/MiniMax-H3",
                        destinationSubdirectory: "upstream",
                        prefixes: [
                            "FL2VA/processor/",
                            "FL2VA/tokenizer/"
                        ],
                        exactFiles: ["FL2VA/model_index.json"]
                    )
                ]
            )
        case miniMaxH3GGUFFL2VAQ40ModelID:
            return miniMaxH3GGUFPlan(
                variant: "FL2VA-Q4_0",
                directoryName: "minimax-h3-gguf-fl2va-q4-0"
            )
        case miniMaxH3GGUFFL2VAQ4KMModelID:
            return miniMaxH3GGUFPlan(
                variant: "FL2VA-Q4_K_M",
                directoryName: "minimax-h3-gguf-fl2va-q4-k-m"
            )
        case miniMaxH3GGUFFL2VAQ4KSModelID:
            return miniMaxH3GGUFPlan(
                variant: "FL2VA-Q4_K_S",
                directoryName: "minimax-h3-gguf-fl2va-q4-k-s"
            )
        case miniMaxH3GGUFRef2VAQ40ModelID:
            return miniMaxH3GGUFPlan(
                variant: "Ref2VA-Q4_0",
                directoryName: "minimax-h3-gguf-ref2va-q4-0"
            )
        case miniMaxH3GGUFRef2VAQ4KMModelID:
            return miniMaxH3GGUFPlan(
                variant: "Ref2VA-Q4_K_M",
                directoryName: "minimax-h3-gguf-ref2va-q4-k-m"
            )
        case miniMaxH3GGUFRef2VAQ4KSModelID:
            return miniMaxH3GGUFPlan(
                variant: "Ref2VA-Q4_K_S",
                directoryName: "minimax-h3-gguf-ref2va-q4-k-s"
            )
        case aceStep15TurboModelID:
            return InstallPlan(
                directoryName: "ace-step-1.5-turbo",
                sources: [
                    SourcePlan(
                        repository: "ACE-Step/Ace-Step1.5",
                        revision: "19671f406d603126926c1b7e2adc169acbcade22",
                        destinationSubdirectory: "",
                        prefixes: [
                            "Qwen3-Embedding-0.6B/",
                            "acestep-v15-turbo/",
                            "vae/"
                        ],
                        exactFiles: ["README.md", "config.json"]
                    )
                ]
            )
        case miniMaxMusic3MLX8BitModelID:
            return InstallPlan(
                directoryName: "minimax-music3-mlx-8bit",
                sources: [
                    SourcePlan(
                        repository: "vanch007/MiniMax-Music3-MLX-8bit",
                        revision: "57d87a63181336634a9557fd31aacc2ad6762935",
                        destinationSubdirectory: "",
                        prefixes: [
                            "language_model/",
                            "rvq_depth_decoder/",
                            "condition_encoder/",
                            "transformer/",
                            "vocoder/",
                            "tokenizer/"
                        ],
                        exactFiles: [
                            "LICENSE",
                            "README.md",
                            "config.json",
                            "source_manifest.json"
                        ]
                    )
                ]
            )
        case miniMaxMusic3MLX4BitModelID:
            return InstallPlan(
                directoryName: "minimax-music3-mlx-4bit",
                requiredRuntimeFiles: [
                    "config.json",
                    "model.safetensors.index.json",
                    "model-00001-of-00002.safetensors",
                    "model-00002-of-00002.safetensors",
                    "scheduler/scheduler_config.json",
                    "tokenizer/chat_template.jinja",
                    "tokenizer/tokenizer.json",
                    "tokenizer/tokenizer_config.json"
                ],
                sources: [
                    SourcePlan(
                        repository: "mlx-community/MiniMax-Music3-4bit",
                        revision: "c7ea32923b245fe5afc22d740a1936ad2ac590f3",
                        destinationSubdirectory: "",
                        prefixes: [],
                        exactFiles: [
                            "LICENSE",
                            "README.md",
                            "config.json",
                            "model-00001-of-00002.safetensors",
                            "model-00002-of-00002.safetensors",
                            "model.safetensors.index.json",
                            "scheduler/scheduler_config.json",
                            "tokenizer/chat_template.jinja",
                            "tokenizer/tokenizer.json",
                            "tokenizer/tokenizer_config.json"
                        ]
                    )
                ]
            )
        case miniMaxMusic3GGUFModelID:
            return InstallPlan(
                directoryName: "minimax-music3-gguf-q4",
                sources: [
                    SourcePlan(
                        repository: "audio-cpp/MiniMax-Music3-GGUF",
                        revision: "2a19a42dd84d9ab9411316977ff3c9fd143ab214",
                        destinationSubdirectory: "",
                        prefixes: [],
                        exactFiles: [
                            "LICENSE",
                            "README.md",
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
                    )
                ]
            )
        case miniMaxMusic3ComposerModelID:
            return InstallPlan(
                directoryName: "minimax-music3-composer-5.7b-distilled",
                requiredRuntimeFiles: [
                    "lr-6e-5/config.json",
                    "lr-6e-5/generation_config.json",
                    "lr-6e-5/model.safetensors"
                ],
                sources: [
                    SourcePlan(
                        repository: "Mothersuperior/minimax-music3-composer-5.7b-distilled",
                        revision: "ef3022805574ac77cd64b2f26342b6233e7e86bc",
                        destinationSubdirectory: "",
                        prefixes: [],
                        exactFiles: [
                            "README.md",
                            "lr-6e-5/config.json",
                            "lr-6e-5/generation_config.json",
                            "lr-6e-5/model.safetensors"
                        ]
                    )
                ]
            )
        case whisperLargeV3TurboCoreMLModelID:
            return InstallPlan(
                directoryName: "whisper-large-v3-turbo-coreml",
                runtimeRelativePath: "openai_whisper-large-v3_turbo_954MB",
                sources: [
                    SourcePlan(
                        repository: "argmaxinc/whisperkit-coreml",
                        revision: "0f63a7800b00dd0226abd051b906c246e1907482",
                        destinationSubdirectory: "",
                        prefixes: [
                            "openai_whisper-large-v3_turbo_954MB/MelSpectrogram.mlmodelc/",
                            "openai_whisper-large-v3_turbo_954MB/AudioEncoder.mlmodelc/",
                            "openai_whisper-large-v3_turbo_954MB/TextDecoder.mlmodelc/"
                        ],
                        exactFiles: [
                            "openai_whisper-large-v3_turbo_954MB/config.json",
                            "openai_whisper-large-v3_turbo_954MB/generation_config.json"
                        ]
                    ),
                    SourcePlan(
                        repository: "openai/whisper-large-v3",
                        revision: "06f233fe06e710322aca913c1bc4249a0d71fce1",
                        destinationSubdirectory: "openai_whisper-large-v3_turbo_954MB",
                        prefixes: [],
                        exactFiles: [
                            "added_tokens.json",
                            "merges.txt",
                            "normalizer.json",
                            "preprocessor_config.json",
                            "special_tokens_map.json",
                            "tokenizer.json",
                            "tokenizer_config.json",
                            "vocab.json"
                        ]
                    )
                ]
            )
        case whisperSmallCoreMLModelID:
            return InstallPlan(
                directoryName: "whisper-small-coreml",
                runtimeRelativePath: "openai_whisper-small_216MB",
                sources: [
                    SourcePlan(
                        repository: "argmaxinc/whisperkit-coreml",
                        revision: "0f63a7800b00dd0226abd051b906c246e1907482",
                        destinationSubdirectory: "",
                        prefixes: [
                            "openai_whisper-small_216MB/MelSpectrogram.mlmodelc/",
                            "openai_whisper-small_216MB/AudioEncoder.mlmodelc/",
                            "openai_whisper-small_216MB/TextDecoder.mlmodelc/"
                        ],
                        exactFiles: [
                            "openai_whisper-small_216MB/config.json",
                            "openai_whisper-small_216MB/generation_config.json"
                        ]
                    ),
                    SourcePlan(
                        repository: "openai/whisper-small",
                        revision: "973afd24965f72e36ca33b3055d56a652f456b4d",
                        destinationSubdirectory: "openai_whisper-small_216MB",
                        prefixes: [],
                        exactFiles: [
                            "added_tokens.json",
                            "merges.txt",
                            "normalizer.json",
                            "preprocessor_config.json",
                            "special_tokens_map.json",
                            "tokenizer.json",
                            "tokenizer_config.json",
                            "vocab.json"
                        ]
                    )
                ]
            )
        case paraformerChineseCoreMLModelID:
            return InstallPlan(
                directoryName: "paraformer-large-zh-coreml-int8",
                sources: [
                    SourcePlan(
                        repository: paraformerChineseCoreMLModelID,
                        revision: "5dd557bd06342a3cd07ceccb909d8a45e48b053a",
                        destinationSubdirectory: "",
                        prefixes: [
                            "ParaformerPreprocessor.mlmodelc/",
                            "ParaformerEncoder_int8.mlmodelc/",
                            "ParaformerCifAlphas.mlmodelc/",
                            "ParaformerDecoder_int8.mlmodelc/"
                        ],
                        exactFiles: ["README.md", "vocab.json"]
                    )
                ]
            )
        case parakeetJapaneseCoreMLModelID:
            return InstallPlan(
                directoryName: "parakeet-0.6b-ja-coreml",
                sources: [
                    SourcePlan(
                        repository: parakeetJapaneseCoreMLModelID,
                        revision: "2952296ff1da4a6d6a7aec545e226367db80c612",
                        destinationSubdirectory: "",
                        prefixes: [
                            "Preprocessor.mlmodelc/",
                            "Encoder.mlmodelc/",
                            "Decoderv2.mlmodelc/",
                            "Jointerv2.mlmodelc/"
                        ],
                        exactFiles: ["README.md", "config.json", "metadata.json", "vocab.json"]
                    )
                ]
            )
        case int4ModelID:
            return InstallPlan(
                directoryName: "qwen-image-edit-2511-int4",
                sources: [
                    SourcePlan(
                        repository: "Qwen/Qwen-Image-Edit-2511",
                        destinationSubdirectory: "snapshot",
                        prefixes: ["processor/", "text_encoder/", "vae/"],
                        exactFiles: officialFiles
                    ),
                    SourcePlan(
                        repository: "xocialize/qwen-image-edit-2511-mlx-int4",
                        destinationSubdirectory: "quantized",
                        prefixes: [],
                        exactFiles: [
                            "qie-2511-dit-int4-mod8.safetensors",
                            "qie-2511-vl7b-int4.safetensors"
                        ]
                    )
                ]
            )
        case int8ModelID:
            return InstallPlan(
                directoryName: "qwen-image-edit-2511-int8",
                sources: [
                    SourcePlan(
                        repository: "Qwen/Qwen-Image-Edit-2511",
                        destinationSubdirectory: "snapshot",
                        prefixes: officialPrefixes,
                        exactFiles: officialFiles
                    )
                ]
            )
        case fp16ModelID:
            return InstallPlan(
                directoryName: "qwen-image-edit-2511-fp16",
                sources: [
                    SourcePlan(
                        repository: "Qwen/Qwen-Image-Edit-2511",
                        destinationSubdirectory: "snapshot",
                        prefixes: officialPrefixes,
                        exactFiles: officialFiles
                    )
                ]
            )
        default:
            return nil
        }
    }

    private nonisolated static func miniMaxH3MLXPlan(
        repository: String,
        directoryName: String
    ) -> InstallPlan {
        InstallPlan(
            directoryName: directoryName,
            sources: [
                SourcePlan(
                    repository: repository,
                    destinationSubdirectory: "transformer",
                    prefixes: ["model-"],
                    exactFiles: [
                        "LICENSE",
                        "README.md",
                        "config.json",
                        "model.safetensors.index.json",
                        "quant_config.json"
                    ]
                ),
                SourcePlan(
                    repository: "MiniMaxAI/MiniMax-H3",
                    destinationSubdirectory: "upstream",
                    prefixes: [
                        "FL2VA/audio_vae/",
                        "FL2VA/processor/",
                        "FL2VA/text_encoder/",
                        "FL2VA/tokenizer/",
                        "FL2VA/video_vae/"
                    ],
                    exactFiles: ["FL2VA/model_index.json"]
                )
            ]
        )
    }

    private nonisolated static func miniMaxH3GGUFPlan(
        variant: String,
        directoryName: String
    ) -> InstallPlan {
        let weightPath = "unet/MiniMax-H3-\(variant).gguf"
        let repositoryCompanionPaths = [
            "text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
            "vae/minimax_h3_audio_vae_fp32.safetensors",
            "vae/minimax_h3_video_vae_fp16.safetensors"
        ]
        let processorPaths = [
            "upstream/FL2VA/processor/tokenizer.json",
            "upstream/FL2VA/processor/tokenizer_config.json"
        ]
        return InstallPlan(
            directoryName: directoryName,
            requiredRuntimeFiles: Set(
                [weightPath] + repositoryCompanionPaths + processorPaths
            ),
            sources: [
                SourcePlan(
                    repository: "Abiray/MiniMax-H3-GGUF",
                    revision: "9fc3454d3ebe1be1bade862cd4a5011f325a22cb",
                    destinationSubdirectory: "",
                    prefixes: [],
                    exactFiles: Set([
                        "LICENSE",
                        "README.md",
                        "NOTICE",
                        weightPath
                    ] + repositoryCompanionPaths)
                ),
                // The Qwen3-VL text encoder above is a ComfyUI conversion: it
                // holds weights only and carries no `tokenizer.ggml.*`
                // metadata, unlike a llama.cpp GGUF. Without the upstream
                // tokenizer and vision preprocessor there is no way to turn a
                // prompt into ids the encoder accepts, so the download is not
                // runnable on its own. `miniMaxH3MLXPlan` sources these from
                // the same place.
                SourcePlan(
                    repository: "MiniMaxAI/MiniMax-H3",
                    destinationSubdirectory: "upstream",
                    prefixes: [
                        "FL2VA/processor/",
                        "FL2VA/tokenizer/"
                    ],
                    exactFiles: ["FL2VA/model_index.json"]
                )
            ]
        )
    }

    private nonisolated static func qwenMultimodalPlan(
        repository: String,
        revision: String,
        directoryName: String
    ) -> InstallPlan {
        InstallPlan(
            directoryName: directoryName,
            requiredRuntimeFiles: [
                "chat_template.jinja",
                "config.json",
                "model.safetensors.index.json",
                "preprocessor_config.json",
                "processor_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "video_preprocessor_config.json",
                "vocab.json"
            ],
            sources: [
                SourcePlan(
                    repository: repository,
                    revision: revision,
                    destinationSubdirectory: "",
                    prefixes: ["model"],
                    exactFiles: [
                        "README.md",
                        "chat_template.jinja",
                        "config.json",
                        "generation_config.json",
                        "preprocessor_config.json",
                        "processor_config.json",
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "video_preprocessor_config.json",
                        "vocab.json"
                    ]
                )
            ]
        )
    }

    private nonisolated static func zImageMLXPlan(
        repository: String,
        directoryName: String
    ) -> InstallPlan {
        InstallPlan(
            directoryName: directoryName,
            sources: [
                SourcePlan(
                    repository: repository,
                    destinationSubdirectory: "",
                    prefixes: [
                        "scheduler/",
                        "text_encoder/",
                        "tokenizer/",
                        "transformer/",
                        "vae/"
                    ],
                    exactFiles: ["model_index.json", "quantize_config.json"]
                )
            ]
        )
    }

    private nonisolated static func realESRGANPlan(directoryName: String) -> InstallPlan {
        InstallPlan(
            directoryName: directoryName,
            runtimeRelativePath: "RealESRGAN_x4.mlpackage",
            sources: [
                SourcePlan(
                    repository: "mlboydaisuke/Real-ESRGAN-x4-CoreML",
                    destinationSubdirectory: "",
                    prefixes: ["RealESRGAN_x4.mlpackage/"],
                    exactFiles: []
                )
            ]
        )
    }

    /// 將部分 MLX Diffusers 倉庫使用的 quantize_config.json 轉成
    /// Z-Image.swift 讀取的 quantization.json。量化權重本身已包含
    /// weight/scales/biases，空的 layers 讓 Runtime 套用全域 bits/group_size。
    private nonisolated static func materializeQuantizationManifest(at directory: URL) throws {
        let fileManager = FileManager.default
        let configURL = directory.appendingPathComponent("quantize_config.json")
        let manifestURL = directory.appendingPathComponent("quantization.json")
        guard !fileManager.fileExists(atPath: manifestURL.path) else { return }

        var specification: QuantizeSpecification?
        if fileManager.fileExists(atPath: configURL.path) {
            guard let data = try? Data(contentsOf: configURL),
                  let config = try? JSONDecoder().decode(QuantizeConfig.self, from: data) else {
                throw ModelInstallerError.invalidQuantizationConfig(
                    configURL,
                    "無法解析 quantize_config.json。"
                )
            }
            specification = config.quantization
        } else {
            for component in ["transformer", "text_encoder"] {
                let componentURL = directory
                    .appendingPathComponent(component, isDirectory: true)
                    .appendingPathComponent("config.json")
                guard let data = try? Data(contentsOf: componentURL),
                      let config = try? JSONDecoder().decode(ComponentConfig.self, from: data),
                      let componentSpecification = config.quantization else {
                    continue
                }
                if let existing = specification,
                   existing.bits != componentSpecification.bits
                    || existing.groupSize != componentSpecification.groupSize {
                    throw ModelInstallerError.invalidQuantizationConfig(
                        componentURL,
                        "Transformer 與 text encoder 的量化設定不一致。"
                    )
                }
                specification = componentSpecification
            }
        }

        guard let specification else { return }
        guard [2, 4, 8].contains(specification.bits) else {
            throw ModelInstallerError.invalidQuantizationConfig(
                configURL,
                "不支援的量化位元數：\(specification.bits)。"
            )
        }
        guard [32, 64, 128].contains(specification.groupSize) else {
            throw ModelInstallerError.invalidQuantizationConfig(
                configURL,
                "不支援的 group_size：\(specification.groupSize)。"
            )
        }

        let mode: String
        switch specification.mode?.lowercased() {
        case nil, "affine":
            mode = "affine"
        case "mxfp4":
            mode = "mxfp4"
        default:
            throw ModelInstallerError.invalidQuantizationConfig(
                configURL,
                "不支援的量化模式：\(specification.mode ?? "")。"
            )
        }

        let manifest = GeneratedQuantizationManifest(
            modelId: nil,
            groupSize: specification.groupSize,
            bits: specification.bits,
            mode: mode,
            layers: []
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(manifest).write(to: manifestURL, options: .atomic)
    }

    private nonisolated static func runtimeURL(
        for plan: InstallPlan,
        destination: URL
    ) -> URL {
        guard let relativePath = plan.runtimeRelativePath else { return destination }
        return destination.appendingPathComponent(relativePath)
    }

    private nonisolated static func validateRequiredRuntimeFiles(
        for plan: InstallPlan,
        at destination: URL
    ) throws {
        for relativePath in plan.requiredRuntimeFiles.sorted() {
            let url = destination.appendingPathComponent(relativePath)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw ModelInstallerError.requiredRuntimeFileMissing(relativePath)
            }
        }
    }

    private nonisolated static func validateGGUFWeights(
        modelID: String,
        at directory: URL
    ) throws {
        let weightRelativePath: String
        let expectedSourceType: String
        if let spec = GGUFDiagnosticPlan.all.first(where: { $0.modelID == modelID }) {
            weightRelativePath = spec.weightRelativePath
            expectedSourceType = spec.expectedSourceType
        } else if let spec = miniMaxH3GGUFValidationSpec(for: modelID) {
            weightRelativePath = spec.weightRelativePath
            expectedSourceType = spec.expectedSourceType
        } else {
            return
        }
        let weightURL = directory.appendingPathComponent(weightRelativePath)
        do {
            let inspection = try GGUFModelLoader.inspect(fileURL: weightURL)
            guard inspection.unsupportedTypes.isEmpty else {
                throw ModelInstallerError.invalidGGUF(
                    weightURL,
                    "包含不支援的 tensor type：\(inspection.unsupportedTypes.joined(separator: ", "))。"
                )
            }
            guard inspection.quantizationCounts[expectedSourceType, default: 0] > 0 else {
                throw ModelInstallerError.invalidGGUF(
                    weightURL,
                    "找不到預期的 \(expectedSourceType) 量化 tensor。"
                )
            }
        } catch let error as ModelInstallerError {
            throw error
        } catch {
            throw ModelInstallerError.invalidGGUF(weightURL, error.localizedDescription)
        }
    }

    private nonisolated static func miniMaxH3GGUFValidationSpec(
        for modelID: String
    ) -> (weightRelativePath: String, expectedSourceType: String)? {
        let variants: [String: String] = [
            miniMaxH3GGUFFL2VAQ40ModelID: "FL2VA-Q4_0",
            miniMaxH3GGUFFL2VAQ4KMModelID: "FL2VA-Q4_K_M",
            miniMaxH3GGUFFL2VAQ4KSModelID: "FL2VA-Q4_K_S",
            miniMaxH3GGUFRef2VAQ40ModelID: "Ref2VA-Q4_0",
            miniMaxH3GGUFRef2VAQ4KMModelID: "Ref2VA-Q4_K_M",
            miniMaxH3GGUFRef2VAQ4KSModelID: "Ref2VA-Q4_K_S"
        ]
        guard let variant = variants[modelID] else { return nil }
        let expectedSourceType = variant.contains("Q4_0") ? "Q4_0" : "Q4_K"
        return ("unet/MiniMax-H3-\(variant).gguf", expectedSourceType)
    }

    private nonisolated static func isSafeRelativePath(_ path: String) -> Bool {
        !path.isEmpty
            && !path.hasPrefix("/")
            && !path.split(separator: "/").contains("..")
    }

    private nonisolated static func reusableFile(
        for file: ResolvedFile,
        rootURL: URL,
        excluding destination: URL
    ) -> URL? {
        let fileManager = FileManager.default
        guard let enumerator = fileManager.enumerator(
            at: rootURL,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants]
        ) else { return nil }

        let excludedPath = destination.standardizedFileURL.path
        for case let manifestURL as URL in enumerator
        where manifestURL.lastPathComponent == "genimage-model.json" {
            let installDirectory = manifestURL.deletingLastPathComponent().standardizedFileURL
            if installDirectory.path == excludedPath { continue }
            guard let data = try? Data(contentsOf: manifestURL),
                  let manifest = try? JSONDecoder.genImageManifest.decode(InstallManifest.self, from: data),
                  let match = manifest.files.first(where: {
                      $0.repository == file.repository
                          && $0.revision == file.revision
                          && $0.remotePath == file.remotePath
                          && $0.size == file.size
                          && isSafeRelativePath($0.relativePath)
                  }) else { continue }

            let candidate = installDirectory.appendingPathComponent(match.relativePath)
            if fileSize(at: candidate) == file.size {
                return candidate
            }
        }
        return nil
    }

    private nonisolated static func fileSize(at url: URL) -> Int64 {
        guard let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
              values.isRegularFile == true else { return -1 }
        return Int64(values.fileSize ?? -1)
    }

    private nonisolated static func fraction(_ completed: Int64, _ total: Int64) -> Double {
        guard total > 0 else { return 0 }
        return min(1, max(0, Double(completed) / Double(total)))
    }
}

/// Aggregates progress from concurrent file downloads without touching the
/// installer actor or the WebUI for every network callback.
private final class DownloadProgressTracker: @unchecked Sendable {
    private let initialBytes: Int64
    private let totalBytes: Int64
    private let progress: @Sendable (ModelInstallProgress) -> Void
    private let lock = NSLock()
    private var fileBytes: [String: Int64] = [:]
    private var completedFiles: Set<String> = []

    init(
        initialBytes: Int64,
        totalBytes: Int64,
        progress: @escaping @Sendable (ModelInstallProgress) -> Void
    ) {
        self.initialBytes = initialBytes
        self.totalBytes = totalBytes
        self.progress = progress
    }

    func emit() {
        lock.lock()
        let downloaded = currentBytesLocked()
        lock.unlock()
        progress(
            ModelInstallProgress(
                fractionCompleted: fraction(downloaded, totalBytes),
                downloadedBytes: downloaded,
                totalBytes: totalBytes
            )
        )
    }

    func report(_ file: String, received: Int64) {
        lock.lock()
        guard !completedFiles.contains(file) else {
            lock.unlock()
            return
        }
        fileBytes[file] = max(fileBytes[file] ?? 0, received)
        let downloaded = currentBytesLocked()
        lock.unlock()
        progress(
            ModelInstallProgress(
                fractionCompleted: fraction(downloaded, totalBytes),
                downloadedBytes: downloaded,
                totalBytes: totalBytes
            )
        )
    }

    func markCompleted(_ file: String, bytes: Int64) {
        lock.lock()
        completedFiles.insert(file)
        fileBytes[file] = bytes
        let downloaded = currentBytesLocked()
        lock.unlock()
        progress(
            ModelInstallProgress(
                fractionCompleted: fraction(downloaded, totalBytes),
                downloadedBytes: downloaded,
                totalBytes: totalBytes
            )
        )
    }

    func finalizeSegmented(_ file: String, segmentKeys: [String], bytes: Int64) {
        lock.lock()
        for key in segmentKeys {
            fileBytes.removeValue(forKey: key)
        }
        completedFiles.insert(file)
        fileBytes[file] = bytes
        let downloaded = currentBytesLocked()
        lock.unlock()
        progress(
            ModelInstallProgress(
                fractionCompleted: fraction(downloaded, totalBytes),
                downloadedBytes: downloaded,
                totalBytes: totalBytes
            )
        )
    }

    func discard(_ keys: [String]) {
        lock.lock()
        for key in keys {
            fileBytes.removeValue(forKey: key)
        }
        lock.unlock()
    }

    private func currentBytesLocked() -> Int64 {
        min(totalBytes, initialBytes + fileBytes.values.reduce(0, +))
    }

    private func fraction(_ completed: Int64, _ total: Int64) -> Double {
        guard total > 0 else { return 0 }
        return min(1, max(0, Double(completed) / Double(total)))
    }
}

private final class FileDownloadDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let destination: URL
    private let resumeDataURL: URL
    private let expectedBytes: Int64
    private let progress: @Sendable (Int64, Int64) -> Void
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Void, Error>?
    private var session: URLSession?
    private var task: URLSessionDownloadTask?
    private var movedFile = false
    private var completed = false

    init(
        destination: URL,
        expectedBytes: Int64,
        progress: @escaping @Sendable (Int64, Int64) -> Void
    ) {
        self.destination = destination
        resumeDataURL = destination.appendingPathExtension("resume")
        self.expectedBytes = expectedBytes
        self.progress = progress
    }

    func start(request: URLRequest) async throws {
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                lock.lock()
                self.continuation = continuation
                let configuration = URLSessionConfiguration.default
                configuration.timeoutIntervalForRequest = 60 * 60 * 24
                configuration.timeoutIntervalForResource = 60 * 60 * 24
                configuration.waitsForConnectivity = false
                configuration.httpMaximumConnectionsPerHost = 8
                configuration.httpShouldUsePipelining = true
                configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
                configuration.allowsExpensiveNetworkAccess = true
                configuration.allowsConstrainedNetworkAccess = true
                let session = URLSession(configuration: configuration, delegate: self, delegateQueue: nil)
                self.session = session
                let task: URLSessionDownloadTask
                if let resumeData = try? Data(contentsOf: resumeDataURL), !resumeData.isEmpty {
                    task = session.downloadTask(withResumeData: resumeData)
                } else {
                    task = session.downloadTask(with: request)
                }
                self.task = task
                lock.unlock()
                task.resume()
            }
        } onCancel: {
            self.cancel()
        }
    }

    func cancel() {
        lock.lock()
        let task = task
        lock.unlock()
        task?.cancel { [resumeDataURL] resumeData in
            guard let resumeData, !resumeData.isEmpty else { return }
            try? resumeData.write(to: resumeDataURL, options: .atomic)
        }
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        progress(totalBytesWritten, totalBytesExpectedToWrite)
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        do {
            let fileManager = FileManager.default
            if let response = downloadTask.response as? HTTPURLResponse,
               !(200..<300).contains(response.statusCode) {
                let body = try? Data(contentsOf: location)
                let message = body
                    .flatMap { String(data: $0.prefix(2_048), encoding: .utf8) } ?? ""
                if response.statusCode == 401,
                   let url = downloadTask.originalRequest?.url,
                   url.host?.localizedCaseInsensitiveContains("civitai.com") == true {
                    throw ModelInstallerError.authenticationRequired(url, message)
                }
                throw ModelInstallerError.httpStatus(response.statusCode, message)
            }
            let actual = Int64(
                (try location.resourceValues(forKeys: [.fileSizeKey])).fileSize ?? -1
            )
            guard actual == expectedBytes else {
                throw ModelInstallerError.sizeMismatch(
                    path: destination.lastPathComponent,
                    expected: expectedBytes,
                    actual: actual
                )
            }
            if fileManager.fileExists(atPath: destination.path) {
                try fileManager.removeItem(at: destination)
            }
            try fileManager.moveItem(at: location, to: destination)
            try? fileManager.removeItem(at: resumeDataURL)
            lock.lock()
            movedFile = true
            lock.unlock()
        } catch {
            finish(.failure(error))
        }
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: (any Error)?
    ) {
        if let error {
            finish(.failure(error))
            return
        }
        lock.lock()
        let movedFile = movedFile
        lock.unlock()
        finish(movedFile ? .success(()) : .failure(ModelInstallerError.invalidResponse))
    }

    private func finish(_ result: Result<Void, Error>) {
        lock.lock()
        guard !completed else {
            lock.unlock()
            return
        }
        completed = true
        let continuation = continuation
        self.continuation = nil
        let session = session
        self.session = nil
        task = nil
        lock.unlock()
        session?.finishTasksAndInvalidate()
        continuation?.resume(with: result)
    }
}

public enum ModelInstallerError: LocalizedError, Sendable {
    case unsupportedModel(String)
    case invalidRepository(String)
    case emptyRepository(String)
    case noMatchingFiles(String)
    case invalidResponse
    case httpStatus(Int, String)
    case authenticationRequired(URL, String)
    case invalidManifest(URL)
    case runtimeNotFound(URL)
    case sizeMismatch(path: String, expected: Int64, actual: Int64)
    case invalidQuantizationConfig(URL, String)
    case requiredRuntimeFileMissing(String)
    case invalidGGUF(URL, String)

    public var errorDescription: String? {
        switch self {
        case let .unsupportedModel(id):
            "此模型尚未提供可執行的下載方案：\(id)"
        case let .invalidRepository(repository):
            "Hugging Face 模型來源格式不正確：\(repository)"
        case let .emptyRepository(id):
            "模型下載清單是空的：\(id)"
        case let .noMatchingFiles(repository):
            "模型來源沒有符合 Runtime 的檔案：\(repository)"
        case .invalidResponse:
            "模型伺服器回傳了無效回應。"
        case let .httpStatus(status, message):
            status == 401
                ? "模型下載需要 Hugging Face API Token（HTTP 401），請輸入金鑰後重試。"
                : status == 403
                    && (message.localizedCaseInsensitiveContains("restricted")
                        || message.localizedCaseInsensitiveContains("authorized list"))
                    ? "模型需要 Hugging Face gated 授權（HTTP 403）；Token 已送出，但目前帳號尚未取得此模型權限。請先在模型頁接受授權條款後重試。"
                    : "模型伺服器回傳 HTTP \(status)：\(message)"
        case let .authenticationRequired(_, message):
            "Civitai 下載需要登入或 API Token（HTTP 401）：\(message)"
        case let .invalidManifest(url):
            "模型安裝資訊不存在或已損壞：\(url.path)"
        case let .runtimeNotFound(url):
            "安裝完成但找不到 Runtime 模型：\(url.path)"
        case let .sizeMismatch(path, expected, actual):
            "模型檔案大小不符：\(path)（預期 \(expected)，實際 \(actual) bytes）"
        case let .invalidQuantizationConfig(url, message):
            "量化設定無法轉換：\(url.lastPathComponent)；\(message)"
        case let .requiredRuntimeFileMissing(path):
            "模型缺少 Runtime 必要檔案：\(path)"
        case let .invalidGGUF(url, message):
            "GGUF 權重驗證失敗：\(url.path)；\(message)"
        }
    }
}

private extension JSONEncoder {
    static var genImageManifest: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }
}

private extension JSONDecoder {
    static var genImageManifest: JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }
}

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
        guard request.profile.modelID.lowercased().contains("ltx-2.3-mlx") else {
            throw LTXVideoRuntimeError.unsupportedModel(request.profile.modelID)
        }
        guard Self.isValidFrameCount(request.options.frameCount) else {
            throw LTXVideoRuntimeError.invalidFrameCount(request.options.frameCount)
        }
        let sourceAssets = Self.uniqueSourceAssets(request.sourceAssets)
        if request.profile.capability == .imageToVideo {
            guard !sourceAssets.isEmpty else {
                throw LTXVideoRuntimeError.missingInputFile
            }
            guard sourceAssets.count <= request.options.frameCount else {
                throw LTXVideoRuntimeError.tooManyImageAnchors(
                    count: sourceAssets.count,
                    frameCount: request.options.frameCount
                )
            }
            for asset in sourceAssets {
                guard let inputURL = asset.fileURL,
                      FileManager.default.fileExists(atPath: inputURL.path) else {
                    throw LTXVideoRuntimeError.missingInputFile
                }
            }
        }
        let manifestURL = request.modelURL.appendingPathComponent("genimage-model.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw LTXVideoRuntimeError.modelNotInstalled(request.modelURL)
        }
        for lora in request.profileLoRAs {
            guard FileManager.default.fileExists(atPath: lora.localURL.path) else {
                throw LTXVideoRuntimeError.loraNotInstalled(lora.localURL)
            }
            guard lora.scale.isFinite, (0...1).contains(lora.scale),
                  lora.conditioningScale.isFinite,
                  (0...1).contains(lora.conditioningScale) else {
                throw LTXVideoRuntimeError.invalidLoRAScale(lora.modelID)
            }
        }

        let executable = try Self.runtimeExecutable()
        try FileManager.default.createDirectory(
            at: outputDirectory,
            withIntermediateDirectories: true
        )

        let controlLoRAs = request.profileLoRAs.filter { $0.conditioning != nil }
        let controlPreparationSpan = controlLoRAs.isEmpty ? 0.0 : 0.02
        let controlVideoURL: URL?
        if !controlLoRAs.isEmpty {
            guard let inputURL = request.sourceAsset?.fileURL else {
                throw LTXVideoRuntimeError.missingControlSource
            }
            let url = outputDirectory.appendingPathComponent(
                "ltx-control-\(UUID().uuidString).mp4"
            )
            try await createCannyControlVideo(
                inputURL: inputURL,
                outputURL: url,
                width: request.options.width,
                height: request.options.height,
                frameCount: request.options.frameCount,
                frameRate: request.options.frameRate,
                progress: { value in
                    progress(min(controlPreparationSpan, value * controlPreparationSpan))
                }
            )
            controlVideoURL = url
            progress(controlPreparationSpan)
        } else {
            controlVideoURL = nil
        }
        defer {
            if let controlVideoURL {
                try? FileManager.default.removeItem(at: controlVideoURL)
            }
        }

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
        for index in 0..<request.options.outputCount {
            try Task.checkCancellation()
            let identifier = UUID().uuidString
            let outputURL = OutputFileNaming.videoURL(in: outputDirectory, pathExtension: "mp4")
            generatedOutputURLs.append(outputURL)
            let logURL = outputDirectory.appendingPathComponent("ltx-video-\(identifier).log")
            defer { try? FileManager.default.removeItem(at: logURL) }

            let log = try RuntimeLog(at: logURL)
            defer { log.close() }

            let generationSpan = 1 - controlPreparationSpan
            let perOutputSpan = generationSpan / Double(request.options.outputCount)
            let completedFraction = controlPreparationSpan + Double(index) * perOutputSpan
            progress(completedFraction + 0.01 * perOutputSpan)
            var activity = RuntimeLogActivity(log: log)
            let status = try await RuntimeProcess.run(
                executable: executable,
                arguments: Self.arguments(
                    request: request,
                    outputURL: outputURL,
                    seed: request.options.seed &+ UInt64(index),
                    controlVideoURL: controlVideoURL
                ),
                environment: RuntimeExecutable.environment(),
                log: log,
                pollInterval: .milliseconds(300)
            ) {
                if !activity.sample(log), activity.idleDuration >= 30 * 60 {
                    throw LTXVideoRuntimeError.runtimeStalled(
                        details: log.lastLine(fallback: "Runtime 未提供最後狀態。")
                    )
                }
                let runtimeProgress = min(
                    Self.runtimeProgress(in: log, stage1Steps: request.options.steps) ?? 0.01,
                    0.99
                )
                progress(completedFraction + runtimeProgress * perOutputSpan)
            }

            guard status == 0 else {
                throw LTXVideoRuntimeError.runtimeFailed(
                    status: status,
                    message: log.message(fallback: "Runtime 未提供錯誤訊息。")
                )
            }
            guard FileManager.default.fileExists(atPath: outputURL.path),
                  (try outputURL.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0) > 0 else {
                throw LTXVideoRuntimeError.outputMissing(outputURL)
            }

            outputs.append(
                MediaAsset(
                    projectID: request.projectID,
                    parentAssetID: request.sourceAsset?.id,
                    kind: .generatedVideo,
                    title: request.options.outputCount == 1
                        ? "生成影片"
                        : "生成影片 \(index + 1)",
                    fileURL: outputURL,
                    pixelWidth: request.options.width,
                    pixelHeight: request.options.height,
                    recipeID: request.recipeID
                )
            )
            progress(
                controlPreparationSpan + Double(index + 1) * perOutputSpan
            )
        }
        completed = true
        return outputs
    }

    private func createCannyControlVideo(
        inputURL: URL,
        outputURL: URL,
        width: Int,
        height: Int,
        frameCount: Int,
        frameRate: Int,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws {
        var completed = false
        defer {
            if !completed {
                try? FileManager.default.removeItem(at: outputURL)
            }
        }
        let logURL = outputURL.appendingPathExtension("log")
        defer { try? FileManager.default.removeItem(at: logURL) }
        let log = try RuntimeLog(at: logURL)
        defer { log.close() }

        progress(0.01)
        var activity = RuntimeLogActivity(log: log)
        let status = try await RuntimeProcess.run(
            executable: try Self.ffmpegExecutable(),
            arguments: [
                "-y",
                "-nostdin",
                "-loglevel", "error",
                "-nostats",
                "-progress", "pipe:1",
                "-loop", "1",
                "-i", inputURL.path,
                "-vf", "scale=\(width):\(height):force_original_aspect_ratio=increase,crop=\(width):\(height),edgedetect=mode=canny:low=0.1:high=0.4,format=yuv420p",
                "-frames:v", String(frameCount),
                "-r", String(frameRate),
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "18",
                outputURL.path
            ],
            environment: RuntimeExecutable.environment(),
            log: log,
            pollInterval: .milliseconds(100)
        ) {
            if activity.sample(log) {
                if let frameProgress = Self.ffmpegFrameProgress(in: log, frameCount: frameCount) {
                    progress(frameProgress)
                }
            } else if activity.idleDuration >= 120 {
                throw LTXVideoRuntimeError.controlVideoStalled(
                    details: log.lastLine(fallback: "Runtime 未提供最後狀態。")
                )
            }
        }

        guard status == 0 else {
            throw LTXVideoRuntimeError.controlVideoFailed(
                status: status,
                message: log.message(fallback: "Runtime 未提供錯誤訊息。")
            )
        }
        guard FileManager.default.fileExists(atPath: outputURL.path),
              RuntimeLog.fileSize(at: outputURL) > 0 else {
            throw LTXVideoRuntimeError.controlVideoMissing(outputURL)
        }
        progress(1)
        completed = true
    }

    private static func ffmpegFrameProgress(
        in log: RuntimeLog,
        frameCount: Int
    ) -> Double? {
        guard frameCount > 0, let text = log.tail(16 * 1_024) else {
            return nil
        }
        for line in text.split(whereSeparator: { $0 == "\n" || $0 == "\r" }).reversed() {
            guard line.hasPrefix("frame="),
                  let frame = Int(line.dropFirst("frame=".count)) else { continue }
            return min(1, max(0.01, Double(frame) / Double(frameCount)))
        }
        return nil
    }

    private static func arguments(
        request: VideoGenerationRequest,
        outputURL: URL,
        seed: UInt64,
        controlVideoURL: URL?
    ) -> [String] {
        let usesICLoRA = controlVideoURL != nil
        var arguments = [
            usesICLoRA ? "ic-lora" : "generate",
            "--prompt", request.options.prompt,
            "--output", outputURL.path,
            "--model", request.modelURL.path,
            "--height", String(request.options.height),
            "--width", String(request.options.width),
            "--frames", String(request.options.frameCount),
            "--frame-rate", String(request.options.frameRate),
            "--seed", String(seed),
            "--stage1-steps", String(request.options.steps),
            "--stage2-steps", "3"
        ]
        if !usesICLoRA {
            arguments.append("--distilled")
        }
        for lora in request.profileLoRAs {
            arguments.append(contentsOf: [
                "--lora", lora.localURL.path, String(lora.scale)
            ])
        }
        if let controlVideoURL {
            for lora in request.profileLoRAs where lora.conditioning != nil {
                arguments.append(contentsOf: [
                    "--video-conditioning",
                    controlVideoURL.path,
                    String(lora.conditioningScale)
                ])
            }
            arguments.append(contentsOf: ["--upsample-only", "--refine-steps", "3"])
        }
        if let gemmaModel = ProcessInfo.processInfo.environment["GENIMAGE_LTX_GEMMA_MODEL"],
           !gemmaModel.isEmpty {
            arguments.append(contentsOf: ["--gemma", gemmaModel])
        }
        let sourceAssets = uniqueSourceAssets(request.sourceAssets)
        if sourceAssets.count == 1, let inputURL = sourceAssets[0].fileURL {
            arguments.append(contentsOf: ["--image", inputURL.path, "0", "1.0"])
            if request.options.frameCount > 1 {
                arguments.append(contentsOf: [
                    "--image",
                    inputURL.path,
                    String(request.options.frameCount - 1),
                    "0.65"
                ])
            }
        } else if sourceAssets.count > 1 {
            let finalFrame = request.options.frameCount - 1
            for (index, asset) in sourceAssets.enumerated() {
                guard let inputURL = asset.fileURL else { continue }
                let frameIndex = Int(
                    (Double(index) * Double(finalFrame) / Double(sourceAssets.count - 1)).rounded()
                )
                arguments.append(contentsOf: [
                    "--image", inputURL.path, String(frameIndex), "1.0"
                ])
            }
        }
        let untiledPixelArea = 1_280 * 720
        let outputPixelArea = request.options.width * request.options.height
        let spatialTiles = max(
            1,
            Int(ceil(sqrt(Double(outputPixelArea) / Double(untiledPixelArea))))
        )
        let temporalTiles = max(
            1,
            Int(ceil(Double(request.options.frameCount) / 97.0))
        )
        if spatialTiles > 1 || temporalTiles > 1 {
            arguments.append("--low-ram")
            if spatialTiles > 1 {
                arguments.append(contentsOf: [
                    "--tile-spatial", String(spatialTiles),
                    "--tile-overlap", "4"
                ])
            }
            if temporalTiles > 1 {
                arguments.append(contentsOf: ["--tile-frames", String(temporalTiles)])
            }
        }
        return arguments
    }

    private static func uniqueSourceAssets(_ assets: [MediaAsset]) -> [MediaAsset] {
        var seen = Set<UUID>()
        return assets.filter { seen.insert($0.id).inserted }
    }

    private static func ffmpegExecutable() throws -> URL {
        let fileManager = FileManager.default
        let environment = ProcessInfo.processInfo.environment
        var candidates: [URL] = []
        if let configured = environment["GENIMAGE_FFMPEG"], !configured.isEmpty {
            candidates.append(URL(fileURLWithPath: configured))
        }
        candidates.append(URL(fileURLWithPath: "/opt/homebrew/bin/ffmpeg"))
        candidates.append(URL(fileURLWithPath: "/usr/local/bin/ffmpeg"))
        candidates.append(URL(fileURLWithPath: "/usr/bin/ffmpeg"))
        candidates.append(
            contentsOf: RuntimeExecutable.pathCandidates(for: "ffmpeg", environment: environment)
        )
        guard let executable = RuntimeExecutable.locate(candidates) else {
            throw LTXVideoRuntimeError.ffmpegNotFound(candidates.map(\.path))
        }
        return executable
    }

    private static func runtimeExecutable() throws -> URL {
        let fileManager = FileManager.default
        var candidates: [URL] = []
        let environment = ProcessInfo.processInfo.environment

        if let configured = environment["GENIMAGE_LTX_RUNTIME"], !configured.isEmpty {
            candidates.append(URL(fileURLWithPath: configured))
        }
        if let runtimeRoot = environment["GENIMAGE_LTX_RUNTIME_ROOT"], !runtimeRoot.isEmpty {
            candidates.append(
                URL(fileURLWithPath: runtimeRoot, isDirectory: true)
                    .appendingPathComponent(".venv/bin/ltx-2-mlx")
            )
        }
        if let executableDirectory = Bundle.main.executableURL?.deletingLastPathComponent() {
            candidates.append(executableDirectory.appendingPathComponent("ltx-2-mlx"))
            candidates.append(
                executableDirectory
                    .deletingLastPathComponent()
                    .appendingPathComponent("Helpers/ltx-2-mlx")
            )
        }

        let home = fileManager.homeDirectoryForCurrentUser
        candidates.append(home.appendingPathComponent(".local/bin/ltx-2-mlx"))
        candidates.append(
            ApplicationSupport.directory(.runtime, fileManager: fileManager)
                .appendingPathComponent("ltx-2-mlx/.venv/bin/ltx-2-mlx")
        )
        candidates.append(URL(fileURLWithPath: "/opt/homebrew/bin/ltx-2-mlx"))
        candidates.append(URL(fileURLWithPath: "/usr/local/bin/ltx-2-mlx"))

        candidates.append(
            contentsOf: RuntimeExecutable.pathCandidates(for: "ltx-2-mlx", environment: environment)
        )

        guard let executable = RuntimeExecutable.locate(candidates) else {
            throw LTXVideoRuntimeError.runtimeNotFound(candidates.map(\.path))
        }
        return executable
    }

    private static func runtimeProgress(
        in log: RuntimeLog,
        stage1Steps: Int
    ) -> Double? {
        guard let text = log.tail(64 * 1_024) else { return nil }
        let plan = runtimeProgressPlan(stage1Steps: stage1Steps)
        if text.contains("[Decoding video + audio + muxing] done") { return 0.99 }
        if text.contains("[Decoding video + audio + muxing] ...") {
            return plan.stage2End + plan.decodeSpan * 0.25
        }
        if text.contains("[Loading decoders (VAE + audio + vocoder)] done") {
            return plan.stage2End + plan.decodeSpan * 0.20
        }
        if text.contains("[Loading decoders (VAE + audio + vocoder)] ...") {
            return plan.stage2End + plan.decodeSpan * 0.10
        }

        let pattern = #"Denoising:\s+(\d{1,3})(?:\.\d+)?%"#
        guard let expression = try? NSRegularExpression(pattern: pattern) else { return nil }
        let range = NSRange(text.startIndex..., in: text)
        let percentages: [Double] = expression.matches(in: text, range: range).compactMap { match -> Double? in
            guard let valueRange = Range(match.range(at: 1), in: text),
                  let value = Double(text[valueRange]), value <= 100 else { return nil }
            return value / 100
        }
        if !percentages.isEmpty {
            var stageIndex = 0
            var previous = percentages[0]
            for value in percentages.dropFirst() {
                if previous >= 0.99, value < previous {
                    stageIndex += 1
                }
                previous = value
            }
            if stageIndex >= 1 {
                return plan.stage1End + min(previous, 1) * plan.stage2Span
            }
            return plan.setupEnd + min(previous, 1) * plan.stage1Span
        }

        if text.range(
            of: #"\[Loading transformer \([^\r\n]+\)\] done in"#,
            options: .regularExpression
        ) != nil { return plan.setupEnd }
        if text.contains("[Loading transformer") { return plan.setupEnd * 0.85 }
        if text.contains("[Encoding prompt] done") { return plan.setupEnd * 0.72 }
        if text.contains("[Encoding prompt] ...") { return plan.setupEnd * 0.55 }
        if text.contains("[Loading text encoder (Gemma)] done") { return plan.setupEnd * 0.45 }
        if text.contains("[Loading text encoder (Gemma)] ...") { return plan.setupEnd * 0.15 }
        return nil
    }

    private static func runtimeProgressPlan(
        stage1Steps: Int
    ) -> (
        setupEnd: Double,
        stage1Span: Double,
        stage1End: Double,
        stage2Span: Double,
        stage2End: Double,
        decodeSpan: Double
    ) {
        let setupEnd = 0.04
        let stage1Work = Double(max(stage1Steps, 1))
        let fullResolutionScale = 4.0
        let stage2StepCount = 3.0
        let stage2Work = stage2StepCount * fullResolutionScale
        let decodeWork = fullResolutionScale
        let totalWork = stage1Work + stage2Work + decodeWork
        let measurableSpan = 0.95
        let stage1Span = measurableSpan * stage1Work / totalWork
        let stage2Span = measurableSpan * stage2Work / totalWork
        let decodeSpan = measurableSpan * decodeWork / totalWork
        let stage1End = setupEnd + stage1Span
        let stage2End = stage1End + stage2Span
        return (setupEnd, stage1Span, stage1End, stage2Span, stage2End, decodeSpan)
    }
}

public enum LTXVideoRuntimeError: LocalizedError, Sendable {
    case incompatibleProfile
    case unsupportedArchitecture(InferenceArchitecture)
    case unsupportedModel(String)
    case invalidFrameCount(Int)
    case missingInputFile
    case tooManyImageAnchors(count: Int, frameCount: Int)
    case missingControlSource
    case modelNotInstalled(URL)
    case loraNotInstalled(URL)
    case invalidLoRAScale(String)
    case ffmpegNotFound([String])
    case controlVideoFailed(status: Int32, message: String)
    case controlVideoStalled(details: String)
    case controlVideoMissing(URL)
    case runtimeNotFound([String])
    case runtimeFailed(status: Int32, message: String)
    case runtimeStalled(details: String)
    case outputMissing(URL)

    public var errorDescription: String? {
        switch self {
        case .incompatibleProfile:
            "Profile 不是文生影或圖生影類型。"
        case let .unsupportedArchitecture(architecture):
            "LTX 影片 Runtime 不支援此架構：\(architecture.title)。"
        case let .unsupportedModel(modelID):
            "目前影片 Runtime 尚不支援模型：\(modelID)。"
        case let .invalidFrameCount(frameCount):
            "LTX-2.3 幀數必須符合 8n+1，目前為 \(frameCount)。"
        case .missingInputFile:
            "圖生影需要至少一張可讀取的本機圖片。"
        case let .tooManyImageAnchors(count, frameCount):
            "圖片錨點數量（\(count)）不可超過影片幀數（\(frameCount)）。"
        case .missingControlSource:
            "Profile 的 LoRA 需要來源圖片才能建立控制影片。"
        case let .modelNotInstalled(url):
            "LTX-2.3 模型尚未完整安裝：\(url.path)"
        case let .loraNotInstalled(url):
            "Profile 的 LoRA 尚未完整安裝：\(url.path)"
        case let .invalidLoRAScale(modelID):
            "LoRA 權重與控制強度必須介於 0 到 1：\(modelID)"
        case let .ffmpegNotFound(paths):
            "找不到 ffmpeg，無法建立 LoRA 控制影片；已檢查：\(paths.joined(separator: "、"))"
        case let .controlVideoFailed(status, message):
            "建立 LoRA 控制影片失敗（\(status)）：\(message)"
        case let .controlVideoStalled(details):
            "建立 LoRA 控制影片超過 2 分鐘沒有新進度，已自動停止。最後狀態：\(details)"
        case let .controlVideoMissing(url):
            "ffmpeg 完成但沒有產生 LoRA 控制影片：\(url.path)"
        case let .runtimeNotFound(paths):
            "找不到 ltx-2-mlx Runtime。請安裝後設定 GENIMAGE_LTX_RUNTIME；已檢查：\(paths.joined(separator: "、"))"
        case let .runtimeFailed(status, message):
            "LTX 影片 Runtime 結束（\(status)）：\(message)"
        case let .runtimeStalled(details):
            "LTX 影片 Runtime 超過 30 分鐘沒有輸出新進度，已自動停止。最後狀態：\(details)"
        case let .outputMissing(url):
            "LTX 推論完成但沒有產生影片：\(url.path)"
        }
    }
}

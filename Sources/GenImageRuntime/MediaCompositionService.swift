import Foundation
import GenImageCore

public enum ImageLoopFitMode: String, Codable, CaseIterable, Sendable {
    case contain
    case cover
}

public struct ImageLoopOptions: Codable, Hashable, Sendable {
    public var width: Int
    public var height: Int
    public var frameRate: Int
    public var imageDurationSeconds: Double
    public var totalDurationSeconds: Double
    public var fitMode: ImageLoopFitMode

    public init(
        width: Int = 1280,
        height: Int = 720,
        frameRate: Int = 30,
        imageDurationSeconds: Double = 4,
        totalDurationSeconds: Double = 60,
        fitMode: ImageLoopFitMode = .cover
    ) {
        self.width = width
        self.height = height
        self.frameRate = frameRate
        self.imageDurationSeconds = imageDurationSeconds
        self.totalDurationSeconds = totalDurationSeconds
        self.fitMode = fitMode
    }

    public func validate() throws {
        guard (64...4096).contains(width), (64...4096).contains(height),
              width.isMultiple(of: 2), height.isMultiple(of: 2) else {
            throw MediaCompositionError.invalidDimensions
        }
        guard (1...120).contains(frameRate) else {
            throw MediaCompositionError.invalidFrameRate
        }
        guard imageDurationSeconds.isFinite, (0.1...600).contains(imageDurationSeconds),
              totalDurationSeconds.isFinite, (0.1...86_400).contains(totalDurationSeconds) else {
            throw MediaCompositionError.invalidDuration
        }
    }
}

public enum MediaMergeAudioMode: String, Codable, CaseIterable, Sendable {
    case replace
    case mix
}

public enum MediaMergeDurationMode: String, Codable, CaseIterable, Sendable {
    case shortest
    case video
}

public struct MediaMergeOptions: Codable, Hashable, Sendable {
    public var audioMode: MediaMergeAudioMode
    public var durationMode: MediaMergeDurationMode
    public var audioVolume: Double

    public init(
        audioMode: MediaMergeAudioMode = .replace,
        durationMode: MediaMergeDurationMode = .shortest,
        audioVolume: Double = 1
    ) {
        self.audioMode = audioMode
        self.durationMode = durationMode
        self.audioVolume = audioVolume
    }

    public func validate() throws {
        guard audioVolume.isFinite, (0...2).contains(audioVolume) else {
            throw MediaCompositionError.invalidAudioVolume
        }
    }
}

public final class MediaCompositionService: @unchecked Sendable {
    private let outputDirectory: URL

    public init(outputDirectory: URL) {
        self.outputDirectory = outputDirectory
    }

    public func createImageLoop(
        projectID: UUID,
        sourceAssets: [MediaAsset],
        options: ImageLoopOptions,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> MediaAsset {
        try options.validate()
        let sourceURLs = try sourceAssets.map { asset in
            guard asset.kind.isImage,
                  let url = asset.fileURL,
                  FileManager.default.fileExists(atPath: url.path) else {
                throw MediaCompositionError.invalidImageSource(asset.title)
            }
            return url
        }
        guard !sourceURLs.isEmpty else {
            throw MediaCompositionError.imageSourceRequired
        }

        let fileManager = FileManager.default
        try fileManager.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
        let outputURL = OutputFileNaming.videoURL(in: outputDirectory)
        let temporaryURL = temporaryOutputURL(for: outputURL)
        let concatURL = fileManager.temporaryDirectory
            .appendingPathComponent("genmedia-image-loop-\(UUID().uuidString).ffconcat")
        let logURL = fileManager.temporaryDirectory
            .appendingPathComponent("genmedia-image-loop-\(UUID().uuidString).log")
        defer {
            try? fileManager.removeItem(at: concatURL)
            try? fileManager.removeItem(at: temporaryURL)
            try? fileManager.removeItem(at: logURL)
        }

        try concatDocument(
            sourceURLs: sourceURLs,
            imageDurationSeconds: options.imageDurationSeconds,
            totalDurationSeconds: options.totalDurationSeconds
        ).write(to: concatURL, atomically: true, encoding: .utf8)

        let log = try RuntimeLog(at: logURL)
        defer { log.close() }
        let videoFilter = imageLoopFilter(options: options)
        let status = try await MediaCompatibilityService.runFFmpeg(
            arguments: [
                "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", concatURL.path,
                "-an",
                "-vf", videoFilter,
                "-t", Self.decimal(options.totalDurationSeconds),
                "-c:v", "h264_videotoolbox",
                "-profile:v", "high",
            ] + MediaCompatibilityService.videoBitrateArguments(
                pixelWidth: options.width,
                pixelHeight: options.height
            ) + [
                "-movflags", "+faststart",
                "-progress", "pipe:1", "-nostats",
                "-f", "mp4",
                temporaryURL.path
            ],
            log: log
        ) {
            progress(Self.progressValue(in: log, durationSeconds: options.totalDurationSeconds))
        }
        guard status == 0, RuntimeLog.fileSize(at: temporaryURL) > 0 else {
            throw MediaCompositionError.ffmpegFailed(
                log.message(fallback: "FFmpeg 沒有產生圖片循環影片。")
            )
        }
        try Self.finalizeOutput(temporaryURL: temporaryURL, outputURL: outputURL)
        progress(1)

        return MediaAsset(
            projectID: projectID,
            parentAssetID: sourceAssets.first?.id,
            kind: .generatedVideo,
            title: outputURL.deletingPathExtension().lastPathComponent,
            fileURL: outputURL,
            pixelWidth: options.width,
            pixelHeight: options.height,
            mediaDurationSeconds: options.totalDurationSeconds
        )
    }

    public func mergeMedia(
        projectID: UUID,
        videoAsset: MediaAsset,
        audioAsset: MediaAsset,
        options: MediaMergeOptions,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> MediaAsset {
        try options.validate()
        guard videoAsset.kind == .importedVideo || videoAsset.kind == .generatedVideo,
              let videoURL = videoAsset.playbackURL ?? videoAsset.fileURL,
              FileManager.default.fileExists(atPath: videoURL.path) else {
            throw MediaCompositionError.videoSourceRequired
        }
        guard videoAsset.id != audioAsset.id,
              audioAsset.kind == .importedAudio || audioAsset.kind == .generatedAudio,
              let audioURL = audioAsset.playbackURL ?? audioAsset.fileURL,
              FileManager.default.fileExists(atPath: audioURL.path) else {
            throw MediaCompositionError.audioSourceRequired
        }

        let videoProbe = try await MediaCompatibilityService.probe(sourceURL: videoURL)
        let audioProbe = try await MediaCompatibilityService.probe(sourceURL: audioURL)
        guard videoProbe.hasVideo else { throw MediaCompositionError.videoSourceRequired }
        guard audioProbe.hasAudio else { throw MediaCompositionError.audioSourceRequired }
        let videoDuration = max(0.1, videoProbe.durationSeconds)
        let audioDuration = max(0.1, audioProbe.durationSeconds)
        let outputDuration = options.durationMode == .video
            ? videoDuration
            : min(videoDuration, audioDuration)

        let fileManager = FileManager.default
        try fileManager.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
        let outputURL = OutputFileNaming.videoURL(in: outputDirectory)
        let temporaryURL = temporaryOutputURL(for: outputURL)
        let logURL = fileManager.temporaryDirectory
            .appendingPathComponent("genmedia-media-merge-\(UUID().uuidString).log")
        defer {
            try? fileManager.removeItem(at: temporaryURL)
            try? fileManager.removeItem(at: logURL)
        }

        let log = try RuntimeLog(at: logURL)
        defer { log.close() }
        var arguments = [
            "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-i", videoURL.path,
            "-i", audioURL.path
        ]
        let shouldMix = options.audioMode == .mix && videoProbe.hasAudio
        if shouldMix {
            arguments.append(contentsOf: [
                "-filter_complex",
                "[0:a:0]volume=1[a0];[1:a:0]volume=\(Self.decimal(options.audioVolume))[a1];[a0][a1]amix=inputs=2:duration=longest:dropout_transition=2[a]",
                "-map", "0:v:0",
                "-map", "[a]"
            ])
        } else {
            let audioFilter = options.durationMode == .video
                ? "volume=\(Self.decimal(options.audioVolume)),apad"
                : "volume=\(Self.decimal(options.audioVolume))"
            arguments.append(contentsOf: [
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-filter:a", audioFilter
            ])
        }
        arguments.append(contentsOf: [
            "-t", Self.decimal(outputDuration),
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "256k",
            "-ac", "2",
            "-movflags", "+faststart",
            "-progress", "pipe:1", "-nostats",
            "-f", "mp4",
            temporaryURL.path
        ])

        let status = try await MediaCompatibilityService.runFFmpeg(
            arguments: arguments,
            log: log
        ) {
            progress(Self.progressValue(in: log, durationSeconds: outputDuration))
        }
        guard status == 0, RuntimeLog.fileSize(at: temporaryURL) > 0 else {
            throw MediaCompositionError.ffmpegFailed(
                log.message(fallback: "FFmpeg 沒有產生影音合併檔案。")
            )
        }
        try Self.finalizeOutput(temporaryURL: temporaryURL, outputURL: outputURL)
        progress(1)

        return MediaAsset(
            projectID: projectID,
            parentAssetID: videoAsset.id,
            kind: .generatedVideo,
            title: outputURL.deletingPathExtension().lastPathComponent,
            fileURL: outputURL,
            pixelWidth: videoProbe.pixelWidth,
            pixelHeight: videoProbe.pixelHeight,
            mediaDurationSeconds: outputDuration
        )
    }

    private func temporaryOutputURL(for outputURL: URL) -> URL {
        outputURL.deletingPathExtension()
            .appendingPathExtension("part")
            .appendingPathExtension(outputURL.pathExtension)
    }

    private func imageLoopFilter(options: ImageLoopOptions) -> String {
        let size = "\(options.width):\(options.height)"
        let geometry = options.fitMode == .contain
            ? "scale=\(size):force_original_aspect_ratio=decrease,pad=\(size):(ow-iw)/2:(oh-ih)/2:color=black"
            : "scale=\(size):force_original_aspect_ratio=increase,crop=\(size)"
        return "\(geometry),fps=\(options.frameRate),setsar=1,format=nv12"
    }

    private func concatDocument(
        sourceURLs: [URL],
        imageDurationSeconds: Double,
        totalDurationSeconds: Double
    ) -> String {
        var lines = ["ffconcat version 1.0"]
        var elapsed = 0.0
        var lastURL = sourceURLs[0]
        while elapsed < totalDurationSeconds {
            for sourceURL in sourceURLs where elapsed < totalDurationSeconds {
                let duration = min(imageDurationSeconds, totalDurationSeconds - elapsed)
                lines.append("file '\(Self.escapeConcatPath(sourceURL.path))'")
                lines.append("duration \(Self.decimal(duration))")
                lastURL = sourceURL
                elapsed += duration
            }
        }
        lines.append("file '\(Self.escapeConcatPath(lastURL.path))'")
        return lines.joined(separator: "\n") + "\n"
    }

    private static func escapeConcatPath(_ path: String) -> String {
        path.replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "'", with: "\\'")
    }

    private static func decimal(_ value: Double) -> String {
        String(format: "%.4f", locale: Locale(identifier: "en_US_POSIX"), value)
    }

    private static func progressValue(in log: RuntimeLog, durationSeconds: Double) -> Double {
        guard durationSeconds > 0,
              let text = log.tail(16_384) else { return 0.001 }
        let timestamps = text.split(whereSeparator: { $0 == "\n" || $0 == "\r" })
            .compactMap { line -> Double? in
                guard line.hasPrefix("out_time=") else { return nil }
                let value = line.dropFirst("out_time=".count)
                let components = value.split(separator: ":")
                guard components.count == 3,
                      let hours = Double(components[0]),
                      let minutes = Double(components[1]),
                      let seconds = Double(components[2]) else { return nil }
                return hours * 3_600 + minutes * 60 + seconds
            }
        guard let current = timestamps.last else { return 0.001 }
        return min(0.99, max(0.001, current / durationSeconds))
    }

    private static func finalizeOutput(temporaryURL: URL, outputURL: URL) throws {
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: outputURL.path) {
            try fileManager.removeItem(at: outputURL)
        }
        try fileManager.moveItem(at: temporaryURL, to: outputURL)
    }
}

public enum MediaCompositionError: LocalizedError, Sendable {
    case imageSourceRequired
    case invalidImageSource(String)
    case videoSourceRequired
    case audioSourceRequired
    case invalidDimensions
    case invalidFrameRate
    case invalidDuration
    case invalidAudioVolume
    case ffmpegFailed(String)

    public var errorDescription: String? {
        switch self {
        case .imageSourceRequired: "圖片循環至少需要一張來源圖片。"
        case let .invalidImageSource(name): "來源「\(name)」不是可用的圖片檔案。"
        case .videoSourceRequired: "影音合併需要一個可用的影片來源。"
        case .audioSourceRequired: "影音合併需要一個可用的音訊來源。"
        case .invalidDimensions: "影片寬高必須介於 64 到 4096，並且是 2 的倍數。"
        case .invalidFrameRate: "影片 FPS 必須介於 1 到 120。"
        case .invalidDuration: "圖片顯示秒數或影片總長度無效。"
        case .invalidAudioVolume: "音訊音量必須介於 0% 到 200%。"
        case let .ffmpegFailed(message): "FFmpeg 媒體合成失敗：\(message)"
        }
    }
}

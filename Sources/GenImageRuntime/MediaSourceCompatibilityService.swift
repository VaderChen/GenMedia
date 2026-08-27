import Foundation
import GenImageCore

public enum MediaSourceKind: String, Sendable {
    case video
    case audio
}

public enum MediaPlaybackPreparation: String, Sendable {
    case original
    case remuxed
    case transcoded
}

public struct PreparedMediaSource: Sendable {
    public let sourceURL: URL
    public let playbackURL: URL
    public let kind: MediaSourceKind
    public let preparation: MediaPlaybackPreparation
    public let durationSeconds: Double
    public let pixelWidth: Int
    public let pixelHeight: Int

    public var compatibilityURL: URL? {
        preparation == .original ? nil : playbackURL
    }
}

public enum MediaSourceCompatibilityService {
    public static func prepare(
        sourceURL: URL,
        assetID: UUID,
        cacheDirectory: URL = ApplicationSupport.directory(.mediaCache),
        requiresAudio: Bool = false,
        progress: @escaping @Sendable (Double) -> Void = { _ in }
    ) async throws -> PreparedMediaSource {
        let fileManager = FileManager.default
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: sourceURL.path, isDirectory: &isDirectory),
              !isDirectory.boolValue else {
            throw MediaSourceCompatibilityError.inputMissing(sourceURL)
        }

        progress(0.02)
        let probe = try await MediaCompatibilityService.probe(sourceURL: sourceURL)
        guard probe.hasVideo || probe.hasAudio else {
            throw MediaSourceCompatibilityError.noMediaStreams(sourceURL)
        }
        if requiresAudio, !probe.hasAudio {
            throw MediaSourceCompatibilityError.audioRequired(sourceURL)
        }

        let kind: MediaSourceKind = probe.hasVideo ? .video : .audio
        let plan = playbackPlan(
            sourceExtension: sourceURL.pathExtension,
            probe: probe
        )
        if plan == .original {
            progress(1)
            return PreparedMediaSource(
                sourceURL: sourceURL,
                playbackURL: sourceURL,
                kind: kind,
                preparation: .original,
                durationSeconds: probe.durationSeconds,
                pixelWidth: probe.pixelWidth,
                pixelHeight: probe.pixelHeight
            )
        }

        try fileManager.createDirectory(
            at: cacheDirectory,
            withIntermediateDirectories: true
        )
        let playbackURL: URL
        let preparation: MediaPlaybackPreparation
        switch plan {
        case .original:
            playbackURL = sourceURL
            preparation = .original
        case .audioM4A:
            playbackURL = try await convertAudio(
                sourceURL: sourceURL,
                assetID: assetID,
                cacheDirectory: cacheDirectory,
                durationSeconds: probe.durationSeconds,
                progress: progress
            )
            preparation = .transcoded
        case let .videoMP4(videoCodec, audioCodec, videoTag):
            do {
                playbackURL = try await remuxVideo(
                    sourceURL: sourceURL,
                    assetID: assetID,
                    cacheDirectory: cacheDirectory,
                    durationSeconds: probe.durationSeconds,
                    videoCodec: videoCodec,
                    audioCodec: audioCodec,
                    videoTag: videoTag,
                    progress: progress
                )
                preparation = .remuxed
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                playbackURL = try await transcodeVideo(
                    sourceURL: sourceURL,
                    assetID: assetID,
                    cacheDirectory: cacheDirectory,
                    durationSeconds: probe.durationSeconds,
                    pixelWidth: probe.pixelWidth,
                    pixelHeight: probe.pixelHeight,
                    progress: progress
                )
                preparation = .transcoded
            }
        case .videoH264:
            playbackURL = try await transcodeVideo(
                sourceURL: sourceURL,
                assetID: assetID,
                cacheDirectory: cacheDirectory,
                durationSeconds: probe.durationSeconds,
                pixelWidth: probe.pixelWidth,
                pixelHeight: probe.pixelHeight,
                progress: progress
            )
            preparation = .transcoded
        }

        return PreparedMediaSource(
            sourceURL: sourceURL,
            playbackURL: playbackURL,
            kind: kind,
            preparation: preparation,
            durationSeconds: probe.durationSeconds,
            pixelWidth: probe.pixelWidth,
            pixelHeight: probe.pixelHeight
        )
    }

    static func playbackPlan(
        sourceExtension: String,
        probe: MediaProbeResult
    ) -> MediaPlaybackPlan {
        let pathExtension = sourceExtension.lowercased()
        if !probe.hasVideo {
            return isDirectlyPlayableAudio(
                pathExtension: pathExtension,
                codec: probe.audioCodec
            ) ? .original : .audioM4A
        }

        let videoCodec = probe.videoCodec ?? ""
        let audioCodec = probe.audioCodec ?? ""
        let audioCanCopy = ["", "aac", "mp3", "alac"].contains(audioCodec)
        let isNativeContainer = ["mp4", "m4v", "mov"].contains(pathExtension)

        switch videoCodec {
        case "h264", "avc1":
            if isNativeContainer, audioCanCopy {
                return .original
            }
            return .videoMP4(
                videoCodec: "copy",
                audioCodec: audioCanCopy ? "copy" : "aac",
                videoTag: nil
            )
        case "hevc", "h265":
            if isNativeContainer,
               audioCanCopy,
               pathExtension == "mov" || probe.videoCodecTag == "hvc1" {
                return .original
            }
            return .videoMP4(
                videoCodec: "copy",
                audioCodec: audioCanCopy ? "copy" : "aac",
                videoTag: "hvc1"
            )
        default:
            return .videoH264
        }
    }

    private static func isDirectlyPlayableAudio(
        pathExtension: String,
        codec: String?
    ) -> Bool {
        let codec = codec ?? ""
        switch pathExtension {
        case "mp3":
            return codec == "mp3"
        case "m4a", "mp4":
            return codec == "aac" || codec == "alac"
        case "aac":
            return codec == "aac"
        case "wav", "wave", "aif", "aiff":
            return codec.hasPrefix("pcm_")
        case "flac":
            return codec == "flac"
        default:
            return false
        }
    }

    private static func convertAudio(
        sourceURL: URL,
        assetID: UUID,
        cacheDirectory: URL,
        durationSeconds: Double,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> URL {
        let outputURL = cacheDirectory
            .appendingPathComponent(assetID.uuidString.lowercased())
            .appendingPathExtension("m4a")
        return try await runConversion(
            sourceURL: sourceURL,
            outputURL: outputURL,
            durationSeconds: durationSeconds,
            progress: progress,
            arguments: { temporaryURL in
                [
                    "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
                    "-i", sourceURL.path,
                    "-map", "0:a:0",
                    "-map_metadata", "0",
                    "-vn",
                    "-c:a", "aac",
                    "-b:a", "256k",
                    "-movflags", "+faststart",
                    "-f", "mp4",
                    temporaryURL.path
                ]
            }
        )
    }

    private static func remuxVideo(
        sourceURL: URL,
        assetID: UUID,
        cacheDirectory: URL,
        durationSeconds: Double,
        videoCodec: String,
        audioCodec: String,
        videoTag: String?,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> URL {
        let outputURL = cacheDirectory
            .appendingPathComponent(assetID.uuidString.lowercased())
            .appendingPathExtension("mp4")
        return try await runConversion(
            sourceURL: sourceURL,
            outputURL: outputURL,
            durationSeconds: durationSeconds,
            progress: progress,
            arguments: { temporaryURL in
                var arguments = [
                    "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
                    "-fflags", "+genpts",
                    "-i", sourceURL.path,
                    "-map", "0:v:0",
                    "-map", "0:a:0?",
                    "-map_metadata", "0",
                    "-c:v", videoCodec
                ]
                if let videoTag {
                    arguments.append(contentsOf: ["-tag:v", videoTag])
                }
                arguments.append(contentsOf: ["-c:a", audioCodec])
                if audioCodec == "aac" {
                    arguments.append(contentsOf: ["-b:a", "256k"])
                }
                arguments.append(contentsOf: [
                    "-movflags", "+faststart",
                    "-f", "mp4",
                    temporaryURL.path
                ])
                return arguments
            }
        )
    }

    private static func transcodeVideo(
        sourceURL: URL,
        assetID: UUID,
        cacheDirectory: URL,
        durationSeconds: Double,
        pixelWidth: Int,
        pixelHeight: Int,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> URL {
        let outputURL = cacheDirectory
            .appendingPathComponent(assetID.uuidString.lowercased())
            .appendingPathExtension("mp4")
        return try await runConversion(
            sourceURL: sourceURL,
            outputURL: outputURL,
            durationSeconds: durationSeconds,
            progress: progress,
            arguments: { temporaryURL in
                [
                    "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
                    "-fflags", "+genpts",
                    "-i", sourceURL.path,
                    "-map", "0:v:0",
                    "-map", "0:a:0?",
                    "-map_metadata", "0",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos,format=nv12",
                    "-c:v", "h264_videotoolbox",
                    "-profile:v", "high",
                    "-c:a", "aac",
                    "-b:a", "256k",
                    "-ac", "2",
                ] + MediaCompatibilityService.videoBitrateArguments(
                    pixelWidth: pixelWidth,
                    pixelHeight: pixelHeight
                ) + [
                    "-movflags", "+faststart",
                    "-f", "mp4",
                    temporaryURL.path
                ]
            }
        )
    }

    private static func runConversion(
        sourceURL: URL,
        outputURL: URL,
        durationSeconds: Double,
        progress: @escaping @Sendable (Double) -> Void,
        arguments: (URL) -> [String]
    ) async throws -> URL {
        let fileManager = FileManager.default
        let temporaryURL = outputURL.deletingPathExtension()
            .appendingPathExtension("part")
            .appendingPathExtension(outputURL.pathExtension)
        let logURL = fileManager.temporaryDirectory
            .appendingPathComponent("genmedia-media-compatibility-\(UUID().uuidString).log")
        defer {
            try? fileManager.removeItem(at: temporaryURL)
            try? fileManager.removeItem(at: logURL)
        }
        try? fileManager.removeItem(at: temporaryURL)

        let log = try RuntimeLog(at: logURL)
        defer { log.close() }
        let startedAt = Date()
        let timeout = max(10 * 60, durationSeconds * 4)
        let status = try await MediaCompatibilityService.runFFmpeg(
            arguments: ["-progress", "pipe:1", "-nostats"] + arguments(temporaryURL),
            log: log,
            pollInterval: .milliseconds(200)
        ) {
            if Date().timeIntervalSince(startedAt) >= timeout {
                throw MediaSourceCompatibilityError.conversionTimedOut(sourceURL)
            }
            progress(Self.progressValue(in: log, durationSeconds: durationSeconds))
        }
        guard status == 0 else {
            throw MediaSourceCompatibilityError.conversionFailed(
                sourceURL,
                status: status,
                message: log.message(fallback: "FFmpeg 未提供錯誤訊息。")
            )
        }
        guard RuntimeLog.fileSize(at: temporaryURL) > 0 else {
            throw MediaSourceCompatibilityError.outputMissing(sourceURL)
        }

        try? fileManager.removeItem(at: outputURL)
        do {
            try fileManager.moveItem(at: temporaryURL, to: outputURL)
        } catch {
            throw MediaSourceCompatibilityError.outputMoveFailed(outputURL, error.localizedDescription)
        }
        progress(1)
        return outputURL
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
}

enum MediaPlaybackPlan: Equatable, Sendable {
    case original
    case audioM4A
    case videoMP4(videoCodec: String, audioCodec: String, videoTag: String?)
    case videoH264
}

public enum MediaSourceCompatibilityError: LocalizedError, Sendable {
    case inputMissing(URL)
    case noMediaStreams(URL)
    case audioRequired(URL)
    case conversionTimedOut(URL)
    case conversionFailed(URL, status: Int32, message: String)
    case outputMissing(URL)
    case outputMoveFailed(URL, String)

    public var errorDescription: String? {
        switch self {
        case let .inputMissing(url):
            "找不到來源媒體：\(url.path)"
        case let .noMediaStreams(url):
            "來源沒有可辨識的影片或音訊軌：\(url.lastPathComponent)"
        case let .audioRequired(url):
            "來源媒體沒有可辨識的聲音軌：\(url.lastPathComponent)"
        case let .conversionTimedOut(url):
            "準備相容媒體逾時：\(url.lastPathComponent)"
        case let .conversionFailed(url, status, message):
            "無法準備相容媒體「\(url.lastPathComponent)」（\(status)）：\(message)；硬體編碼器可能不可用，仍可使用原始檔匯出／下載。"
        case let .outputMissing(url):
            "FFmpeg 完成後沒有產生相容媒體：\(url.lastPathComponent)"
        case let .outputMoveFailed(url, reason):
            "無法保存相容媒體「\(url.lastPathComponent)」：\(reason)"
        }
    }
}

import Foundation

struct PreparedMediaAudio: Sendable {
    let sourceURL: URL
    let audioURL: URL
    let durationSeconds: Double
    let temporaryDirectory: URL
}

struct MediaAudioFormat: Equatable, Sendable {
    let sampleRate: Double
    let channelCount: Int

    func outputFrameCount(
        inputFrameCount: Int,
        inputSampleRate: Double
    ) -> Int {
        guard inputFrameCount > 0,
              inputSampleRate.isFinite,
              inputSampleRate > 0,
              sampleRate.isFinite,
              sampleRate > 0 else {
            return 0
        }
        return max(
            0,
            Int((Double(inputFrameCount) * sampleRate / inputSampleRate).rounded())
        )
    }

    func outputSampleCount(
        inputSampleCount: Int,
        inputSampleRate: Double,
        inputChannelCount: Int
    ) -> Int {
        guard inputSampleCount > 0,
              inputChannelCount > 0,
              channelCount > 0 else {
            return 0
        }
        let inputFrames = Double(inputSampleCount) / Double(inputChannelCount)
        let outputFrames = inputFrames * sampleRate / inputSampleRate
        guard outputFrames.isFinite, outputFrames > 0 else { return 0 }
        return max(0, Int((outputFrames * Double(channelCount)).rounded()))
    }
}

struct MediaAudioPreparationPaths: Equatable, Sendable {
    let temporaryDirectory: URL
    let audioURL: URL
}

enum MediaAudioPreparer {
    static let speechRecognitionFormat = MediaAudioFormat(
        sampleRate: 16_000,
        channelCount: 1
    )

    static func paths(
        temporaryRoot: URL,
        identifier: UUID
    ) -> MediaAudioPreparationPaths {
        let temporaryDirectory = temporaryRoot.appendingPathComponent(
            "genmedia-asr-\(identifier.uuidString)",
            isDirectory: true
        )
        return MediaAudioPreparationPaths(
            temporaryDirectory: temporaryDirectory,
            audioURL: temporaryDirectory
                .appendingPathComponent("audio", isDirectory: false)
                .appendingPathExtension("wav")
        )
    }

    static func prepare(sourceURL: URL) async throws -> PreparedMediaAudio {
        guard FileManager.default.fileExists(atPath: sourceURL.path) else {
            throw MediaAudioPreparationError.inputMissing(sourceURL)
        }

        let probe: MediaProbeResult
        do {
            probe = try await MediaCompatibilityService.probe(sourceURL: sourceURL)
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            throw MediaAudioPreparationError.inputUnreadable(sourceURL, error.localizedDescription)
        }
        guard probe.hasAudio else {
            throw MediaAudioPreparationError.noAudioTrack(sourceURL)
        }

        let paths = paths(
            temporaryRoot: FileManager.default.temporaryDirectory,
            identifier: UUID()
        )
        try FileManager.default.createDirectory(
            at: paths.temporaryDirectory,
            withIntermediateDirectories: true
        )

        let logURL = paths.temporaryDirectory.appendingPathComponent("ffmpeg.log")
        defer { try? FileManager.default.removeItem(at: logURL) }
        do {
            let log = try RuntimeLog(at: logURL)
            defer { log.close() }
            let startedAt = Date()
            let timeout = max(5 * 60, probe.durationSeconds * 2)
            let status = try await MediaCompatibilityService.runFFmpeg(
                arguments: [
                    "-y",
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel", "error",
                    "-i", sourceURL.path,
                    "-map", "0:a:0",
                    "-vn",
                    "-ac", String(speechRecognitionFormat.channelCount),
                    "-ar", String(Int(speechRecognitionFormat.sampleRate)),
                    "-c:a", "pcm_s16le",
                    "-f", "wav",
                    paths.audioURL.path
                ],
                log: log,
                pollInterval: .milliseconds(100)
            ) {
                if Date().timeIntervalSince(startedAt) >= timeout {
                    throw MediaAudioPreparationError.exportTimedOut(sourceURL)
                }
            }
            guard status == 0 else {
                throw MediaAudioPreparationError.exportFailed(
                    sourceURL,
                    log.message(fallback: "FFmpeg 未提供錯誤訊息。")
                )
            }
            guard RuntimeLog.fileSize(at: paths.audioURL) > 44 else {
                throw MediaAudioPreparationError.outputMissing(paths.audioURL)
            }
        } catch {
            try? FileManager.default.removeItem(at: paths.temporaryDirectory)
            throw error
        }

        return PreparedMediaAudio(
            sourceURL: sourceURL,
            audioURL: paths.audioURL,
            durationSeconds: probe.durationSeconds,
            temporaryDirectory: paths.temporaryDirectory
        )
    }
}

enum MediaAudioPreparationError: LocalizedError, Sendable {
    case inputMissing(URL)
    case inputUnreadable(URL, String)
    case noAudioTrack(URL)
    case exportTimedOut(URL)
    case exportFailed(URL, String)
    case outputMissing(URL)

    var errorDescription: String? {
        switch self {
        case let .inputMissing(url):
            "找不到來源媒體：\(url.path)"
        case let .inputUnreadable(url, reason):
            "無法讀取媒體「\(url.lastPathComponent)」：\(reason)"
        case let .noAudioTrack(url):
            "媒體沒有可辨識的音訊軌：\(url.lastPathComponent)"
        case let .exportTimedOut(url):
            "準備辨識音訊逾時：\(url.lastPathComponent)"
        case let .exportFailed(url, reason):
            "無法準備「\(url.lastPathComponent)」的辨識音訊：\(reason)"
        case let .outputMissing(url):
            "FFmpeg 完成後沒有產生有效的辨識音訊：\(url.path)"
        }
    }
}

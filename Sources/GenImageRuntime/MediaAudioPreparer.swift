@preconcurrency import AVFoundation
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
                .appendingPathExtension("m4a")
        )
    }

    static func prepare(sourceURL: URL) async throws -> PreparedMediaAudio {
        guard FileManager.default.fileExists(atPath: sourceURL.path) else {
            throw MediaAudioPreparationError.inputMissing(sourceURL)
        }

        let asset = AVURLAsset(url: sourceURL)
        let audioTracks: [AVAssetTrack]
        let duration: CMTime
        do {
            audioTracks = try await asset.loadTracks(withMediaType: .audio)
            duration = try await asset.load(.duration)
        } catch {
            throw MediaAudioPreparationError.inputUnreadable(sourceURL, error.localizedDescription)
        }
        guard !audioTracks.isEmpty else {
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

        guard let exporter = AVAssetExportSession(
            asset: asset,
            presetName: AVAssetExportPresetAppleM4A
        ) else {
            throw MediaAudioPreparationError.exportFailed(
                sourceURL,
                "系統無法建立 Apple M4A 轉換工作。"
            )
        }
        let handle = ExportSessionHandle(exporter)
        handle.session.outputURL = paths.audioURL
        handle.session.outputFileType = .m4a
        handle.session.shouldOptimizeForNetworkUse = false

        do {
            try await withCheckedThrowingContinuation { continuation in
                handle.session.exportAsynchronously {
                    switch handle.session.status {
                    case .completed:
                        continuation.resume()
                    case .cancelled:
                        continuation.resume(throwing: CancellationError())
                    case .failed, .waiting, .exporting, .unknown:
                        continuation.resume(
                            throwing: handle.session.error
                                ?? MediaAudioPreparationError.exportFailed(
                                    sourceURL,
                                    "ExportSession 狀態：\(handle.session.status.rawValue)"
                                )
                        )
                    @unknown default:
                        continuation.resume(
                            throwing: MediaAudioPreparationError.exportFailed(
                                sourceURL,
                                "ExportSession 回傳未知狀態。"
                            )
                        )
                    }
                }
            }
        } catch {
            try? FileManager.default.removeItem(at: paths.temporaryDirectory)
            throw error
        }

        return PreparedMediaAudio(
            sourceURL: sourceURL,
            audioURL: paths.audioURL,
            durationSeconds: duration.isNumeric ? max(0, duration.seconds) : 0,
            temporaryDirectory: paths.temporaryDirectory
        )
    }
}

private final class ExportSessionHandle: @unchecked Sendable {
    let session: AVAssetExportSession

    init(_ session: AVAssetExportSession) {
        self.session = session
    }
}

enum MediaAudioPreparationError: LocalizedError, Sendable {
    case inputMissing(URL)
    case inputUnreadable(URL, String)
    case noAudioTrack(URL)
    case exportFailed(URL, String)

    var errorDescription: String? {
        switch self {
        case let .inputMissing(url):
            "找不到來源媒體：\(url.path)"
        case let .inputUnreadable(url, reason):
            "無法讀取媒體「\(url.lastPathComponent)」：\(reason)"
        case let .noAudioTrack(url):
            "媒體沒有可辨識的音訊軌：\(url.lastPathComponent)"
        case let .exportFailed(url, reason):
            "無法準備「\(url.lastPathComponent)」的辨識音訊：\(reason)"
        }
    }
}

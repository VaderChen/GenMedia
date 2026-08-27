import Foundation
import GenImageCore

struct AudioOutputMetadata: Sendable {
    var durationSeconds: Double
    var sampleRate: Int
    var channelCount: Int
}

enum AudioOutputEncoder {
    static func encodeWave(
        inputURL: URL,
        outputURL: URL,
        format: AudioOutputFormat
    ) async throws -> AudioOutputMetadata {
        let metadata = try waveMetadata(at: inputURL)
        var arguments = [
            "-y",
            "-nostdin",
            "-loglevel", "error",
            "-i", inputURL.path,
            "-map_metadata", "-1",
            "-vn"
        ]
        switch format {
        case .mp3:
            arguments.append(contentsOf: ["-c:a", "libmp3lame", "-b:a", "320k"])
        case .m4a:
            arguments.append(contentsOf: ["-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart"])
        case .aac:
            arguments.append(contentsOf: ["-c:a", "aac", "-b:a", "256k", "-f", "adts"])
        case .flac:
            arguments.append(contentsOf: ["-c:a", "flac", "-compression_level", "8"])
        }
        arguments.append(outputURL.path)

        let logURL = outputURL.appendingPathExtension("log")
        defer { try? FileManager.default.removeItem(at: logURL) }
        let log = try RuntimeLog(at: logURL)
        defer { log.close() }

        let startedAt = Date()
        let status = try await MediaCompatibilityService.runFFmpeg(
            arguments: arguments,
            log: log,
            pollInterval: .milliseconds(100)
        ) {
            if Date().timeIntervalSince(startedAt) >= 5 * 60 {
                throw AudioOutputEncodingError.transcodeTimedOut
            }
        }
        guard status == 0 else {
            throw AudioOutputEncodingError.transcodeFailed(
                status: status,
                message: log.message(fallback: "轉檔未提供錯誤訊息。")
            )
        }
        guard RuntimeLog.fileSize(at: outputURL) > 0 else {
            throw AudioOutputEncodingError.encodedOutputMissing(outputURL)
        }
        return metadata
    }

    static func waveMetadata(at url: URL) throws -> AudioOutputMetadata {
        let data = try Data(contentsOf: url)
        guard data.count >= 44,
              String(data: data[0..<4], encoding: .ascii) == "RIFF",
              String(data: data[8..<12], encoding: .ascii) == "WAVE" else {
            throw AudioOutputEncodingError.invalidWaveOutput(url)
        }
        var offset = 12
        var sampleRate: Int?
        var channelCount: Int?
        var byteRate: Int?
        var audioByteCount: Int?
        while offset + 8 <= data.count {
            let chunkID = String(data: data[offset..<(offset + 4)], encoding: .ascii)
            let chunkSize = Int(littleEndianUInt32(data, at: offset + 4))
            let contentOffset = offset + 8
            guard contentOffset + chunkSize <= data.count else { break }
            if chunkID == "fmt ", chunkSize >= 16 {
                channelCount = Int(littleEndianUInt16(data, at: contentOffset + 2))
                sampleRate = Int(littleEndianUInt32(data, at: contentOffset + 4))
                byteRate = Int(littleEndianUInt32(data, at: contentOffset + 8))
            } else if chunkID == "data" {
                audioByteCount = chunkSize
            }
            offset = contentOffset + chunkSize + (chunkSize % 2)
        }
        guard let sampleRate, sampleRate > 0,
              let channelCount, channelCount > 0,
              let byteRate, byteRate > 0,
              let audioByteCount, audioByteCount > 0 else {
            throw AudioOutputEncodingError.invalidWaveOutput(url)
        }
        return AudioOutputMetadata(
            durationSeconds: Double(audioByteCount) / Double(byteRate),
            sampleRate: sampleRate,
            channelCount: channelCount
        )
    }

    private static func littleEndianUInt16(_ data: Data, at offset: Int) -> UInt16 {
        UInt16(data[offset]) | UInt16(data[offset + 1]) << 8
    }

    private static func littleEndianUInt32(_ data: Data, at offset: Int) -> UInt32 {
        UInt32(data[offset])
            | UInt32(data[offset + 1]) << 8
            | UInt32(data[offset + 2]) << 16
            | UInt32(data[offset + 3]) << 24
    }
}

enum AudioOutputEncodingError: LocalizedError, Sendable {
    case invalidWaveOutput(URL)
    case transcodeTimedOut
    case transcodeFailed(status: Int32, message: String)
    case encodedOutputMissing(URL)

    var errorDescription: String? {
        switch self {
        case let .invalidWaveOutput(url):
            "暫存 WAV 音訊格式無效：\(url.path)"
        case .transcodeTimedOut:
            "FFmpeg 音訊轉碼超過 5 分鐘，已自動停止。"
        case let .transcodeFailed(status, message):
            "音訊轉碼失敗（\(status)）：\(message)"
        case let .encodedOutputMissing(url):
            "音訊轉碼完成但找不到輸出檔案：\(url.path)"
        }
    }
}

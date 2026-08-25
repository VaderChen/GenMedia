import Darwin
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
        let ffmpegExecutable = try ffmpegExecutable()
        let process = Process()
        process.executableURL = ffmpegExecutable
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
        process.arguments = arguments
        process.environment = runtimeEnvironment()
        process.standardInput = FileHandle.nullDevice

        let logURL = outputURL.appendingPathExtension("log")
        defer { try? FileManager.default.removeItem(at: logURL) }
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        let logHandle = try FileHandle(forWritingTo: logURL)
        defer { try? logHandle.close() }
        process.standardOutput = logHandle
        process.standardError = logHandle

        do {
            try process.run()
            let startedAt = Date()
            while process.isRunning {
                try Task.checkCancellation()
                try await Task.sleep(for: .milliseconds(100))
                if Date().timeIntervalSince(startedAt) >= 5 * 60 {
                    forceTerminate(process)
                    throw AudioOutputEncodingError.transcodeTimedOut
                }
            }
        } catch {
            forceTerminate(process)
            throw error
        }
        process.waitUntilExit()
        try? logHandle.synchronize()
        guard process.terminationStatus == 0 else {
            throw AudioOutputEncodingError.transcodeFailed(
                status: process.terminationStatus,
                message: logMessage(in: logURL)
            )
        }
        guard fileSize(at: outputURL) > 0 else {
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

    static func runtimeEnvironment() -> [String: String] {
        var environment = ProcessInfo.processInfo.environment
        let commonPaths = ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin"]
        let currentPaths = (environment["PATH"] ?? "").split(separator: ":").map(String.init)
        environment["PATH"] = (currentPaths + commonPaths)
            .reduce(into: [String]()) { paths, path in
                if !paths.contains(path) { paths.append(path) }
            }
            .joined(separator: ":")
        environment["PYTHONUNBUFFERED"] = "1"
        return environment
    }

    static func forceTerminate(_ process: Process) {
        guard process.isRunning else { return }
        process.terminate()
        if process.isRunning {
            Darwin.kill(process.processIdentifier, SIGKILL)
        }
        process.waitUntilExit()
    }

    static func fileSize(at url: URL) -> Int64 {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
              let size = attributes[.size] as? NSNumber else { return 0 }
        return size.int64Value
    }

    static func logMessage(in logURL: URL) -> String {
        guard let data = try? Data(contentsOf: logURL), !data.isEmpty else {
            return "Runtime 未提供錯誤訊息。"
        }
        return String(data: data.suffix(8_192), encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            ?? "Runtime 執行失敗。"
    }

    private static func ffmpegExecutable() throws -> URL {
        let fileManager = FileManager.default
        let environment = ProcessInfo.processInfo.environment
        var candidates: [URL] = []
        for key in ["GENMEDIA_FFMPEG", "GENIMAGE_FFMPEG"] {
            if let configured = environment[key], !configured.isEmpty {
                candidates.append(URL(fileURLWithPath: configured))
            }
        }
        candidates.append(URL(fileURLWithPath: "/opt/homebrew/bin/ffmpeg"))
        candidates.append(URL(fileURLWithPath: "/usr/local/bin/ffmpeg"))
        candidates.append(URL(fileURLWithPath: "/usr/bin/ffmpeg"))
        for directory in (environment["PATH"] ?? "").split(separator: ":") {
            candidates.append(
                URL(fileURLWithPath: String(directory), isDirectory: true)
                    .appendingPathComponent("ffmpeg")
            )
        }
        guard let executable = candidates.first(where: {
            fileManager.isExecutableFile(atPath: $0.path)
        }) else {
            throw AudioOutputEncodingError.ffmpegNotFound(candidates.map(\.path))
        }
        return executable
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
    case ffmpegNotFound([String])
    case invalidWaveOutput(URL)
    case transcodeTimedOut
    case transcodeFailed(status: Int32, message: String)
    case encodedOutputMissing(URL)

    var errorDescription: String? {
        switch self {
        case let .ffmpegNotFound(paths):
            "找不到 FFmpeg，無法輸出 MP3、M4A、AAC 或 FLAC；已檢查：\(paths.joined(separator: "、"))"
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

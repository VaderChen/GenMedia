// Block 5 will invoke this runtime as an independent Swift subprocess.
// The app integration will replace the Python process boundary without changing this runtime's API.

import Foundation
import MLX

public struct MiniMaxMusic3AudioReport: Sendable {
    public let sampleCount: Int
    public let channelCount: Int
    public let sampleRate: Int

    public var durationSeconds: Double {
        Double(sampleCount) / Double(sampleRate)
    }
}

public enum MiniMaxMusic3AudioError: LocalizedError, Sendable {
    case invalidShape([Int])
    case invalidSampleRate
    case nonFiniteSamples
    case fileTooLarge

    public var errorDescription: String? {
        switch self {
        case let .invalidShape(shape):
            "音訊必須是 [1, 2, samples]，目前為 \(shape)。"
        case .invalidSampleRate:
            "sample rate 必須是正整數。"
        case .nonFiniteSamples:
            "音訊包含 NaN 或 Infinity。"
        case .fileTooLarge:
            "音訊超過 WAV 32-bit 大小限制。"
        }
    }
}

public enum MiniMaxMusic3Audio {
    public static func validate(_ audio: MLXArray, sampleRate: Int) throws -> MiniMaxMusic3AudioReport {
        guard audio.shape.count == 3,
              audio.shape[0] == 1,
              audio.shape[1] == 2,
              audio.shape[2] > 0 else {
            throw MiniMaxMusic3AudioError.invalidShape(audio.shape)
        }
        guard sampleRate > 0 else {
            throw MiniMaxMusic3AudioError.invalidSampleRate
        }
        let values = audio.asType(.float32).asArray(Float.self)
        guard values.allSatisfy(\.isFinite) else {
            throw MiniMaxMusic3AudioError.nonFiniteSamples
        }
        return MiniMaxMusic3AudioReport(
            sampleCount: audio.shape[2],
            channelCount: audio.shape[1],
            sampleRate: sampleRate
        )
    }

    public static func writeWAV(
        _ audio: MLXArray,
        sampleRate: Int,
        to outputURL: URL
    ) throws -> MiniMaxMusic3AudioReport {
        let report = try validate(audio, sampleRate: sampleRate)
        let values = audio.asType(.float32).asArray(Float.self)
        let pcmByteCount = values.count * MemoryLayout<Int16>.size
        guard pcmByteCount <= Int(UInt32.max) - 36 else {
            throw MiniMaxMusic3AudioError.fileTooLarge
        }

        var data = Data(capacity: 44 + pcmByteCount)
        data.append(contentsOf: "RIFF".utf8)
        appendLittleEndian(UInt32(36 + pcmByteCount), to: &data)
        data.append(contentsOf: "WAVEfmt ".utf8)
        appendLittleEndian(UInt32(16), to: &data)
        appendLittleEndian(UInt16(1), to: &data)
        appendLittleEndian(UInt16(report.channelCount), to: &data)
        appendLittleEndian(UInt32(sampleRate), to: &data)
        appendLittleEndian(
            UInt32(sampleRate * report.channelCount * MemoryLayout<Int16>.size),
            to: &data
        )
        appendLittleEndian(UInt16(report.channelCount * MemoryLayout<Int16>.size), to: &data)
        appendLittleEndian(UInt16(16), to: &data)
        data.append(contentsOf: "data".utf8)
        appendLittleEndian(UInt32(pcmByteCount), to: &data)

        for index in 0..<report.sampleCount {
            for channel in 0..<report.channelCount {
                let value = values[channel * report.sampleCount + index]
                let clamped = max(-1, min(1, value))
                let sample: Int16
                if clamped <= -1 {
                    sample = Int16.min
                } else {
                    sample = Int16((clamped * Float(Int16.max)).rounded())
                }
                appendLittleEndian(sample, to: &data)
            }
        }

        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try data.write(to: outputURL, options: .atomic)
        return report
    }

    private static func appendLittleEndian<T: FixedWidthInteger>(
        _ value: T,
        to data: inout Data
    ) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { bytes in
            data.append(contentsOf: bytes)
        }
    }
}

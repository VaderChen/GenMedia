// The app invokes this runtime as an independent Swift subprocess.
// Keep the process boundary stable so the App integration does not need a second audio API.

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
        let report = try validateLayout(audio, sampleRate: sampleRate)
        let values = audio.asType(.float32)
        try withExtendedLifetime(values) {
            let source = values.asData(access: .noCopyIfContiguous)
            try source.data.withUnsafeBytes { raw in
                for offset in stride(from: 0, to: raw.count, by: MemoryLayout<Float>.size) {
                    guard raw.loadUnaligned(fromByteOffset: offset, as: Float.self).isFinite else {
                        throw MiniMaxMusic3AudioError.nonFiniteSamples
                    }
                }
            }
        }
        return report
    }

    private static func validateLayout(
        _ audio: MLXArray, sampleRate: Int
    ) throws -> MiniMaxMusic3AudioReport {
        guard audio.shape.count == 3,
              audio.shape[0] == 1,
              audio.shape[1] == 2,
              audio.shape[2] > 0 else {
            throw MiniMaxMusic3AudioError.invalidShape(audio.shape)
        }
        guard sampleRate > 0 else {
            throw MiniMaxMusic3AudioError.invalidSampleRate
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
        let report = try validateLayout(audio, sampleRate: sampleRate)
        guard audio.size <= (Int(UInt32.max) - 36) / MemoryLayout<Int16>.size else {
            throw MiniMaxMusic3AudioError.fileTooLarge
        }
        let pcmByteCount = audio.size * MemoryLayout<Int16>.size

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

        data.count = 44 + pcmByteCount
        let values = audio.asType(.float32)
        try withExtendedLifetime(values) {
            let source = values.asData(access: .noCopyIfContiguous)
            try source.data.withUnsafeBytes { raw in
                try data.withUnsafeMutableBytes { destination in
                    var offset = 44
                    for index in 0..<report.sampleCount {
                        for channel in 0..<report.channelCount {
                            let value = raw.loadUnaligned(
                                fromByteOffset: (channel * report.sampleCount + index) * MemoryLayout<Float>.size,
                                as: Float.self
                            )
                            guard value.isFinite else { throw MiniMaxMusic3AudioError.nonFiniteSamples }
                            let clamped = max(-1, min(1, value))
                            let sample = clamped <= -1
                                ? Int16.min : Int16((clamped * Float(Int16.max)).rounded())
                            destination.storeBytes(of: sample.littleEndian, toByteOffset: offset, as: Int16.self)
                            offset += 2
                        }
                    }
                }
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

import MLX
import Foundation

public struct PCM16WaveReport: Sendable {
    public let sampleCount: Int
    public let channelCount: Int
    public let sampleRate: Int
    public let sourcePeak: Float
    public let appliedGain: Float

    public var durationSeconds: Double {
        Double(sampleCount) / Double(sampleRate)
    }
}

enum PCM16WaveError: LocalizedError {
    case invalidShape([Int])
    case fileTooLarge
    case temporaryDataCorrupted

    var errorDescription: String? {
        switch self {
        case let .invalidShape(shape):
            "WAV 輸入必須是 [1, samples, channels]，目前為 \(shape)。"
        case .fileTooLarge:
            "PCM 資料超過 RIFF/WAV 32-bit 大小限制。"
        case .temporaryDataCorrupted:
            "PCM 暫存資料長度不完整。"
        }
    }
}

enum PCM16WaveWriter {
    static func write(
        audio: MLXArray,
        sampleRate: Int,
        to outputURL: URL
    ) throws -> PCM16WaveReport {
        guard audio.ndim == 3,
              audio.shape[0] == 1,
              audio.shape[1] > 0,
              audio.shape[2] > 0 else {
            throw PCM16WaveError.invalidShape(audio.shape)
        }
        let sampleCount = audio.shape[1]
        let channelCount = audio.shape[2]
        guard audio.size <= (Int(UInt32.max) - 36) / MemoryLayout<Int16>.size else {
            throw PCM16WaveError.fileTooLarge
        }
        let pcmByteCount = audio.size * MemoryLayout<Int16>.size

        var waveData = waveHeader(
            pcmByteCount: pcmByteCount,
            sampleRate: sampleRate,
            channelCount: channelCount
        )
        waveData.count = 44 + pcmByteCount
        var sourcePeak: Float = 0
        var appliedGain: Float = 1
        let values = audio.asType(.float32)
        withExtendedLifetime(values) {
            let source = values.asData(access: .noCopyIfContiguous)
            source.data.withUnsafeBytes { raw in
                for offset in stride(from: 0, to: raw.count, by: MemoryLayout<Float>.size) {
                    let value = raw.loadUnaligned(fromByteOffset: offset, as: Float.self)
                    if value.isFinite { sourcePeak = max(sourcePeak, abs(value)) }
                }
                appliedGain = sourcePeak > 0 ? min(Float(0.95) / sourcePeak, 1) : 1
                waveData.withUnsafeMutableBytes { destination in
                    for index in 0..<values.size {
                        let value = raw.loadUnaligned(
                            fromByteOffset: index * MemoryLayout<Float>.size, as: Float.self
                        )
                        let scaled = max(-1, min(1, (value.isFinite ? value : 0) * appliedGain))
                        let integer = Int16((scaled * Float(Int16.max)).rounded())
                        destination.storeBytes(
                            of: integer.littleEndian, toByteOffset: 44 + index * 2, as: Int16.self
                        )
                    }
                }
            }
        }

        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try waveData.write(to: outputURL, options: .atomic)
        return PCM16WaveReport(
            sampleCount: sampleCount,
            channelCount: channelCount,
            sampleRate: sampleRate,
            sourcePeak: sourcePeak,
            appliedGain: appliedGain
        )
    }

    static func waveHeader(
        pcmByteCount: Int,
        sampleRate: Int,
        channelCount: Int
    ) -> Data {
        let bitsPerSample = 16
        let blockAlignment = channelCount * bitsPerSample / 8
        let byteRate = sampleRate * blockAlignment
        var data = Data(capacity: 44)
        data.append(contentsOf: "RIFF".utf8)
        data.appendLittleEndian(UInt32(36 + pcmByteCount))
        data.append(contentsOf: "WAVE".utf8)
        data.append(contentsOf: "fmt ".utf8)
        data.appendLittleEndian(UInt32(16))
        data.appendLittleEndian(UInt16(1))
        data.appendLittleEndian(UInt16(channelCount))
        data.appendLittleEndian(UInt32(sampleRate))
        data.appendLittleEndian(UInt32(byteRate))
        data.appendLittleEndian(UInt16(blockAlignment))
        data.appendLittleEndian(UInt16(bitsPerSample))
        data.append(contentsOf: "data".utf8)
        data.appendLittleEndian(UInt32(pcmByteCount))
        return data
    }
}

final class PCM16WaveStreamWriter {
    private let outputURL: URL
    private let temporaryURL: URL
    private let sampleRate: Int
    private let channelCount: Int
    private var temporaryHandle: FileHandle?
    private var sampleCount = 0
    private var sourcePeak: Float = 0

    init(outputURL: URL, sampleRate: Int, channelCount: Int) throws {
        self.outputURL = outputURL
        self.sampleRate = sampleRate
        self.channelCount = channelCount
        self.temporaryURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("ace-step-pcm-\(UUID().uuidString).f32")
        FileManager.default.createFile(atPath: temporaryURL.path, contents: nil)
        self.temporaryHandle = try FileHandle(forWritingTo: temporaryURL)
    }

    deinit {
        try? temporaryHandle?.close()
        try? FileManager.default.removeItem(at: temporaryURL)
    }

    func append(audio: MLXArray) throws {
        guard audio.ndim == 3,
              audio.shape[0] == 1,
              audio.shape[1] > 0,
              audio.shape[2] == channelCount else {
            throw PCM16WaveError.invalidShape(audio.shape)
        }
        // Own the staging bytes before sanitizing; never mutate the source tensor.
        var data = audio.asType(.float32).asData(access: .copy).data
        data.withUnsafeMutableBytes { raw in
            for offset in stride(from: 0, to: raw.count, by: MemoryLayout<Float>.size) {
                let value = raw.loadUnaligned(fromByteOffset: offset, as: Float.self)
                if value.isFinite {
                    sourcePeak = max(sourcePeak, abs(value))
                } else {
                    raw.storeBytes(of: Float(0), toByteOffset: offset, as: Float.self)
                }
            }
        }
        try temporaryHandle?.write(contentsOf: data)
        sampleCount += audio.shape[1]
    }

    func finish() throws -> PCM16WaveReport {
        try temporaryHandle?.synchronize()
        try temporaryHandle?.close()
        temporaryHandle = nil

        let pcmByteCount = sampleCount * channelCount * MemoryLayout<Int16>.size
        guard pcmByteCount <= Int(UInt32.max) - 36 else {
            throw PCM16WaveError.fileTooLarge
        }
        let appliedGain = sourcePeak > 0 ? min(Float(0.95) / sourcePeak, 1) : 1
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        if FileManager.default.fileExists(atPath: outputURL.path) {
            try FileManager.default.removeItem(at: outputURL)
        }
        FileManager.default.createFile(atPath: outputURL.path, contents: nil)

        var completed = false
        defer {
            if !completed {
                try? FileManager.default.removeItem(at: outputURL)
            }
        }
        let inputHandle = try FileHandle(forReadingFrom: temporaryURL)
        defer { try? inputHandle.close() }
        let outputHandle = try FileHandle(forWritingTo: outputURL)
        defer { try? outputHandle.close() }
        try outputHandle.write(
            contentsOf: PCM16WaveWriter.waveHeader(
                pcmByteCount: pcmByteCount,
                sampleRate: sampleRate,
                channelCount: channelCount
            )
        )

        var pending = Data()
        while let block = try inputHandle.read(upToCount: 256 * 1_024), !block.isEmpty {
            pending.append(block)
            let usableByteCount = pending.count - pending.count % MemoryLayout<Float>.size
            guard usableByteCount > 0 else { continue }
            var pcmData = Data(count: usableByteCount / 2)
            pending.withUnsafeBytes { source in
                pcmData.withUnsafeMutableBytes { destination in
                    for offset in stride(from: 0, to: usableByteCount, by: MemoryLayout<Float>.size) {
                        let bits = source.loadUnaligned(fromByteOffset: offset, as: UInt32.self)
                        let value = Float(bitPattern: UInt32(littleEndian: bits))
                        let scaled = max(-1, min(1, value * appliedGain))
                        let sample = Int16((scaled * Float(Int16.max)).rounded())
                        destination.storeBytes(
                            of: sample.littleEndian, toByteOffset: offset / 2, as: Int16.self
                        )
                    }
                }
            }
            try outputHandle.write(contentsOf: pcmData)
            pending.removeFirst(usableByteCount)
        }
        guard pending.isEmpty else {
            throw PCM16WaveError.temporaryDataCorrupted
        }
        try outputHandle.synchronize()
        completed = true
        try? FileManager.default.removeItem(at: temporaryURL)
        return PCM16WaveReport(
            sampleCount: sampleCount,
            channelCount: channelCount,
            sampleRate: sampleRate,
            sourcePeak: sourcePeak,
            appliedGain: appliedGain
        )
    }
}

private extension Data {
    mutating func appendLittleEndian<Value: FixedWidthInteger>(_ value: Value) {
        var littleEndianValue = value.littleEndian
        Swift.withUnsafeBytes(of: &littleEndianValue) { bytes in
            append(contentsOf: bytes)
        }
    }
}

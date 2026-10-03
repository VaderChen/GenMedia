import Foundation
import MLX

/// Byte conversion shared by the audio/video and silent-video output paths.
public enum LTXMediaEncoding {
    public enum EncodingError: LocalizedError {
        case invalidAudioShape([Int]), invalidFrameShape([Int]), invalidWAVFormat
        case nonFiniteAudio, nonFiniteVideo

        public var errorDescription: String? {
            switch self {
            case let .invalidAudioShape(shape): "音訊應為 [1, channels, samples]，實際為 \(shape)。"
            case let .invalidFrameShape(shape): "影格應為 [height, width, 3]，實際為 \(shape)。"
            case .invalidWAVFormat: "音訊尺寸或取樣率超出 PCM WAV 支援範圍。"
            case .nonFiniteAudio: "音訊含有 NaN 或 Inf。"
            case .nonFiniteVideo: "影片含有 NaN 或 Inf。"
            }
        }
    }

    /// Preserve the original PCM16 clipping, truncation and channel interleaving.
    public static func wav(_ audio: MLXArray, sampleRate: Int) throws -> Data {
        guard audio.ndim == 3, audio.dim(0) == 1, audio.dim(1) > 0, audio.dim(2) > 0 else {
            throw EncodingError.invalidAudioShape(audio.shape)
        }
        let channels = audio.dim(1), samples = audio.dim(2)
        guard channels <= Int(UInt16.max) / 2 else { throw EncodingError.invalidWAVFormat }
        let blockAlign = channels * 2
        guard sampleRate > 0, sampleRate <= Int(UInt32.max) / blockAlign,
              samples <= (Int(UInt32.max) - 36) / blockAlign else {
            throw EncodingError.invalidWAVFormat
        }
        let pcmBytes = samples * blockAlign
        var data = Data(capacity: 44 + pcmBytes)
        data.append(contentsOf: "RIFF".utf8)
        appendUInt32LE(&data, UInt32(36 + pcmBytes))
        data.append(contentsOf: "WAVEfmt ".utf8)
        appendUInt32LE(&data, 16)
        appendUInt16LE(&data, 1)
        appendUInt16LE(&data, UInt16(channels))
        appendUInt32LE(&data, UInt32(sampleRate))
        appendUInt32LE(&data, UInt32(sampleRate * blockAlign))
        appendUInt16LE(&data, UInt16(blockAlign))
        appendUInt16LE(&data, 16)
        data.append(contentsOf: "data".utf8)
        appendUInt32LE(&data, UInt32(pcmBytes))
        data.count = 44 + pcmBytes

        let values = audio.asType(.float32)
        // MLX's borrowed Data must not outlive its tensor. Noncontiguous views use
        // the library's contiguous fallback, including strided channel/sample axes.
        try withExtendedLifetime(values) {
            let source = values.asData(access: .noCopyIfContiguous)
            try source.data.withUnsafeBytes { raw in
                let floats = raw.bindMemory(to: Float.self)
                try data.withUnsafeMutableBytes { destination in
                    var offset = 44
                    for sample in 0..<samples {
                        for channel in 0..<channels {
                            let value = floats[channel * samples + sample]
                            guard value.isFinite else { throw EncodingError.nonFiniteAudio }
                            let clipped = min(1, max(-1, value))
                            let integer = Int16((clipped * 32_767).rounded(.towardZero))
                            destination.storeBytes(of: integer.littleEndian, toByteOffset: offset, as: Int16.self)
                            offset += 2
                        }
                    }
                }
            }
        }
        return data
    }

    /// Reuse one frame buffer; the caller must finish writing it before the next call.
    public static func rgb24Frame(_ frame: MLXArray, into bytes: inout Data) throws {
        guard frame.ndim == 3, frame.dim(0) > 0, frame.dim(1) > 0, frame.dim(2) == 3 else {
            throw EncodingError.invalidFrameShape(frame.shape)
        }
        if bytes.count != frame.size { bytes = Data(count: frame.size) }
        // Let MLX pack channel-first/strided views; its CPU Data fallback otherwise
        // walks every strided float before the RGB conversion walks them again.
        let values = frame.asType(.float32).contiguous()
        try withExtendedLifetime(values) {
            let source = values.asData(access: .noCopyIfContiguous)
            try source.data.withUnsafeBytes { raw in
                let floats = raw.bindMemory(to: Float.self)
                try bytes.withUnsafeMutableBytes { destination in
                    let pixels = destination.bindMemory(to: UInt8.self)
                    for index in floats.indices {
                        let value = floats[index]
                        guard value.isFinite else { throw EncodingError.nonFiniteVideo }
                        pixels[index] = UInt8(min(255, max(0, ((value + 1) * 127.5).rounded())))
                    }
                }
            }
        }
    }

    private static func appendUInt16LE(_ data: inout Data, _ value: UInt16) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
    }

    private static func appendUInt32LE(_ data: inout Data, _ value: UInt32) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
    }
}

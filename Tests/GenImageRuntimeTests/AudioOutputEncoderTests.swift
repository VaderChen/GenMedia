import Foundation
import Testing
@testable import GenImageRuntime

struct AudioOutputEncoderTests {
    @Test(arguments: [UInt16(1), UInt16(3)])
    func readsPCMAndFloatWithExtendedFormat(formatCode: UInt16) throws {
        let bits: UInt16 = formatCode == 1 ? 16 : 32
        let byteRate = 48_000 * 2 * UInt32(bits / 8)
        let format = formatPayload(code: formatCode, bits: bits) + Data(repeating: 0, count: 8)
        try withWave(chunks: chunk("fmt ", format) + chunk("data", Data(repeating: 7, count: Int(byteRate)))) { url in
            let metadata = try AudioOutputEncoder.waveMetadata(at: url)
            #expect(metadata.channelCount == 2)
            #expect(metadata.sampleRate == 48_000)
            #expect(metadata.durationSeconds == 1)
        }
    }

    @Test func preservesChunkOrderPaddingAndLastValidValues() throws {
        let chunks = chunk("data", Data([1, 2, 3]))
            + chunk("JUNK", Data([4, 5, 6]))
            + chunk("fmt ", formatPayload())
            + chunk("fmt ", formatPayload(sampleRate: 24_000, channels: 1))
            + chunk("data", Data(repeating: 1, count: 48_000))
            + chunk("fmt ", Data([0]))
        try withWave(chunks: chunks) { url in
            let metadata = try AudioOutputEncoder.waveMetadata(at: url)
            #expect(metadata.channelCount == 1)
            #expect(metadata.sampleRate == 24_000)
            #expect(metadata.durationSeconds == 1)
        }
    }

    @Test func skipsLargeSparsePayloadAndFindsFormatAfterIt() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("wave-\(UUID()).wav")
        defer { try? FileManager.default.removeItem(at: url) }
        let payloadSize: UInt32 = 512 * 1_024 * 1_024
        // The legacy reader uses the actual file length, including when RIFF's size is stale.
        try (riffHeader + Data("data".utf8) + littleEndian(payloadSize)).write(to: url)
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        try handle.seek(toOffset: UInt64(20 + payloadSize))
        try handle.write(contentsOf: chunk("fmt ", formatPayload()))
        let metadata = try AudioOutputEncoder.waveMetadata(at: url)
        #expect(metadata.durationSeconds == Double(payloadSize) / 192_000)
        #expect(metadata.sampleRate == 48_000)
    }

    @Test func ignoresIncompleteTrailingChunkAfterValidMetadata() throws {
        let chunks = chunk("fmt ", formatPayload()) + chunk("data", Data([1, 2, 3, 4]))
            + Data("data".utf8) + littleEndian(UInt32.max)
        try withWave(chunks: chunks) { url in
            let metadata = try AudioOutputEncoder.waveMetadata(at: url)
            #expect(metadata.durationSeconds == 4.0 / 192_000)
        }
    }

    @Test(arguments: ["header", "short", "missing-format", "missing-data", "truncated-data", "truncated-format",
                      "zero-rate", "zero-channels", "zero-byte-rate", "empty-data"])
    func rejectsInvalidWave(reason: String) throws {
        let format = formatPayload()
        var bytes = riffHeader + chunk("fmt ", format) + chunk("data", Data([1, 2, 3, 4]))
        switch reason {
        case "header": bytes[0] = 0
        case "short": bytes = Data(bytes.prefix(43))
        case "missing-format": bytes = riffHeader + chunk("JUNK", Data(repeating: 0, count: 32)) + chunk("data", Data([1]))
        case "missing-data": bytes = riffHeader + chunk("fmt ", format) + chunk("JUNK", Data([1]))
        case "truncated-data": bytes = riffHeader + chunk("fmt ", format) + Data("data".utf8) + littleEndian(UInt32.max)
        case "truncated-format": bytes = riffHeader + chunk("data", Data(repeating: 0, count: 32)) + Data("fmt ".utf8) + littleEndian(UInt32(16))
        case "zero-rate": bytes.replaceSubrange(24..<28, with: littleEndian(UInt32(0)))
        case "zero-channels": bytes.replaceSubrange(22..<24, with: littleEndian(UInt16(0)))
        case "zero-byte-rate": bytes.replaceSubrange(28..<32, with: littleEndian(UInt32(0)))
        case "empty-data": bytes = riffHeader + chunk("fmt ", format) + chunk("data", Data())
        default: break
        }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("wave-\(UUID()).wav")
        defer { try? FileManager.default.removeItem(at: url) }
        try bytes.write(to: url)
        #expect(throws: AudioOutputEncodingError.self) { try AudioOutputEncoder.waveMetadata(at: url) }
    }

    private var riffHeader: Data { Data("RIFF".utf8) + littleEndian(UInt32(0)) + Data("WAVE".utf8) }

    private func withWave(chunks: Data, body: (URL) throws -> Void) throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("wave-\(UUID()).wav")
        defer { try? FileManager.default.removeItem(at: url) }
        try (riffHeader + chunks).write(to: url)
        try body(url)
    }

    private func formatPayload(code: UInt16 = 1, sampleRate: UInt32 = 48_000, channels: UInt16 = 2,
                               bits: UInt16 = 16) -> Data {
        let blockAlign = channels * (bits / 8)
        return littleEndian(code) + littleEndian(channels) + littleEndian(sampleRate)
            + littleEndian(sampleRate * UInt32(blockAlign)) + littleEndian(blockAlign) + littleEndian(bits)
    }

    private func chunk(_ name: String, _ payload: Data) -> Data {
        Data(name.utf8) + littleEndian(UInt32(payload.count)) + payload + Data(repeating: 0, count: payload.count % 2)
    }

    private func littleEndian<T: FixedWidthInteger>(_ value: T) -> Data {
        withUnsafeBytes(of: value.littleEndian) { Data($0) }
    }
}

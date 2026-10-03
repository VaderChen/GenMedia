import Foundation
import MLX
import Testing
@testable import ACEStepSwiftRuntime

@Suite(.serialized)
struct PCM16WaveWriterTests {
    @Test func bulkAndStreamingPreserveSanitizationGainAndSampleOrder() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        // Cross both append and 256 KiB read boundaries, with a peak in the last chunk.
        var values = (0..<140_002).map { Float($0 % 199 - 99) / 100 }
        values.replaceSubrange(0..<8, with: [.nan, .infinity, -.infinity, -0, 0.5, -0.5, 1, -1])
        values[values.count - 1] = -2
        let audio = MLXArray(values, [1, values.count / 2, 2])
        let sourceBits = audio.asArray(Float.self).map(\.bitPattern)
        let bulkURL = directory.appendingPathComponent("bulk.wav")
        let streamURL = directory.appendingPathComponent("stream.wav")
        let bulk = try PCM16WaveWriter.write(audio: audio, sampleRate: 48_000, to: bulkURL)
        let writer = try PCM16WaveStreamWriter(outputURL: streamURL, sampleRate: 48_000, channelCount: 2)
        try writer.append(audio: audio[0..., 0..<3, 0...])
        try writer.append(audio: audio[0..., 3..<32_769, 0...])
        try writer.append(audio: audio[0..., 32_769..., 0...])
        let streamed = try writer.finish()
        let data = try Data(contentsOf: bulkURL)
        #expect(data == (try Data(contentsOf: streamURL)))
        #expect(bulk.sourcePeak == 2 && streamed.sourcePeak == 2)
        #expect(bulk.appliedGain == 0.475 && streamed.appliedGain == bulk.appliedGain)
        #expect(bulk.sampleCount == values.count / 2 && streamed.sampleCount == bulk.sampleCount)
        #expect(audio.asArray(Float.self).map(\.bitPattern) == sourceBits)
        let actual = data.dropFirst(44).withUnsafeBytes { raw in
            stride(from: 0, to: raw.count, by: 2).map { Int16(littleEndian: raw.loadUnaligned(fromByteOffset: $0, as: Int16.self)) }
        }
        let expected = values.map { value -> Int16 in
            let scaled = max(-1, min(1, (value.isFinite ? value : 0) * Float(0.475)))
            return Int16((scaled * 32_767).rounded())
        }
        #expect(actual == expected)
    }

    @Test func stridedInputAndSilenceKeepTheOriginalLayout() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        for audio in [MLXArray([Float(0.1), 0.2, 0.3, -0.1, -0.2, -0.3], [1, 2, 3]).transposed(0, 2, 1),
                      MLXArray.zeros([1, 1, 2])] {
            let url = directory.appendingPathComponent("audio.wav")
            let report = try PCM16WaveWriter.write(audio: audio, sampleRate: 44_100, to: url)
            #expect(report.appliedGain == 1)
            let data = try Data(contentsOf: url)
            #expect(data.count == 44 + audio.size * 2)
            let expected = audio.asArray(Float.self).map { Int16(($0 * 32_767).rounded()) }
            let actual = data.dropFirst(44).withUnsafeBytes { raw in
                stride(from: 0, to: raw.count, by: 2).map { Int16(littleEndian: raw.loadUnaligned(fromByteOffset: $0, as: Int16.self)) }
            }
            #expect(actual == expected)
        }
    }
}

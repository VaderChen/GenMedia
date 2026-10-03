import Foundation
import MLX
import Testing
@testable import MiniMaxMusic3SwiftRuntime

@Suite(.serialized)
struct MiniMaxMusic3FunctionOptimizationTests {
    @Test func preparedRotaryMatchesEveryStepOfTheUncachedTransformer() throws {
        for rotaryDim in [0, 4, 8] {
            let configuration = MiniMaxMusic3FlowTransformerConfiguration(
                inChannels: 4, conditionDim: 6, numLayers: 2,
                numAttentionHeads: 2, attentionHeadDim: 8, ffInnerDim: 24,
                rotaryDim: rotaryDim, fourierEmbeddingDim: 8
            )
            let model = MiniMaxMusic3FlowTransformer(configuration: configuration, quantizationBits: nil)
            for length in [3, 9] {
                let latents = MLXArray((0..<(length * 4)).map { Float($0 % 7 - 3) / 10 }, [1, length, 4])
                let condition = MLXArray((0..<(length * 6)).map { Float($0 % 11 - 5) / 10 }, [1, length, 6])
                let rotary = try model.prepareRotary(sequenceLength: length + 1)
                for sigma: Float in [1, 0.3, 0] {
                    for context in [condition, MLXArray.zeros(like: condition)] {
                        let expected = try model(latents, timestep: MLXArray([sigma]), encoderHiddenStates: context)
                        let actual = try model(latents, timestep: MLXArray([sigma]),
                            encoderHiddenStates: context, preparedRotary: rotary)
                        #expect(actual.asArray(Float.self).map(\.bitPattern) == expected.asArray(Float.self).map(\.bitPattern))
                    }
                }
                let wrongLength = try model.prepareRotary(sequenceLength: length + 2)
                #expect(throws: MiniMaxMusic3FlowTransformerError.self) {
                    try model(latents, timestep: MLXArray([Float(1)]), encoderHiddenStates: condition, preparedRotary: wrongLength)
                }
            }
        }
    }

    @Test func wavKeepsClippingRoundingInterleavingAndHeaderForStridedInput() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let left: [Float] = [-2, -1, -0.5, -0.5 / 32_767, -0, 0, 0.5 / 32_767, 0.5, 1, 2]
        let right = Array(left.reversed())
        let interleaved = zip(left, right).flatMap { [$0, $1] }
        let audio = MLXArray(interleaved, [1, left.count, 2]).transposed(0, 2, 1)
        let url = directory.appendingPathComponent("stereo.wav")
        let report = try MiniMaxMusic3Audio.writeWAV(audio, sampleRate: 44_100, to: url)
        let data = try Data(contentsOf: url)
        #expect(report.sampleCount == left.count && report.channelCount == 2)
        #expect(data.count == 44 + interleaved.count * 2)
        #expect(String(data: data[0..<4], encoding: .utf8) == "RIFF")
        #expect(String(data: data[8..<16], encoding: .utf8) == "WAVEfmt ")
        #expect(data.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: 24, as: UInt32.self)) } == 44_100)
        let actual = data.dropFirst(44).withUnsafeBytes { raw in
            stride(from: 0, to: raw.count, by: 2).map { Int16(littleEndian: raw.loadUnaligned(fromByteOffset: $0, as: Int16.self)) }
        }
        let expected = interleaved.map { value -> Int16 in
            let clamped = max(-1, min(1, value))
            return clamped <= -1 ? Int16.min : Int16((clamped * 32_767).rounded())
        }
        #expect(actual == expected)
        #expect(try MiniMaxMusic3Audio.validate(audio, sampleRate: 44_100).sampleCount == report.sampleCount)
    }

    @Test func invalidAudioDoesNotReplaceAnExistingFile() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".wav")
        let original = Data("existing".utf8)
        try original.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        for value in [Float.nan, .infinity, -.infinity] {
            let audio = MLXArray([Float(0), value], [1, 2, 1])
            #expect(throws: MiniMaxMusic3AudioError.self) { try MiniMaxMusic3Audio.validate(audio, sampleRate: 44_100) }
            #expect(throws: MiniMaxMusic3AudioError.self) { try MiniMaxMusic3Audio.writeWAV(audio, sampleRate: 44_100, to: url) }
            #expect(try Data(contentsOf: url) == original)
        }
        #expect(throws: MiniMaxMusic3AudioError.self) {
            try MiniMaxMusic3Audio.writeWAV(.zeros([1, 2, 1]), sampleRate: 0, to: url)
        }
        #expect(throws: MiniMaxMusic3AudioError.self) {
            try MiniMaxMusic3Audio.writeWAV(.zeros([1, 1, 2]), sampleRate: 44_100, to: url)
        }
    }
}

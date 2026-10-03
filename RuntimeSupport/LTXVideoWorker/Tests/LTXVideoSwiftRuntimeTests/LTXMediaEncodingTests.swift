import Foundation
import MLX
import Testing
@testable import LTXVideoSwiftRuntime

@Suite(.serialized)
struct LTXMediaEncodingTests {
    @Test func wavPreservesPCM16HeaderClippingTruncationAndInterleaving() throws {
        let audio = MLXArray([Float(-2), -1, -0.5, -0.00002, 0, 0.5, 1, 2]).reshaped([1, 2, 4])
        let data = try LTXMediaEncoding.wav(audio, sampleRate: 48_000)
        // Explicit RIFF PCM header, followed by interleaved signed little-endian PCM.
        let header: [UInt8] = [82,73,70,70,52,0,0,0,87,65,86,69,102,109,116,32,
                              16,0,0,0,1,0,2,0,128,187,0,0,0,238,2,0,4,0,16,0,
                              100,97,116,97,16,0,0,0]
        #expect(Array(data.prefix(44)) == header)
        #expect(Array(data.dropFirst(44)) == [1,128,0,0, 1,128,255,63, 1,192,255,127, 0,0,255,127])
    }

    @Test(arguments: [DType.float32, .bfloat16, .float16])
    func stridedAudioMatchesContiguousAudio(dtype: DType) throws {
        let interleaved = MLXArray([Float(-1), 1, 0.5, -0.5, 0, 0.25]).reshaped([1, 3, 2]).asType(dtype)
        let view = interleaved.transposed(0, 2, 1)
        let contiguous = MLXArray(view.asArray(Float.self)).reshaped([1, 2, 3]).asType(dtype)
        #expect(try LTXMediaEncoding.wav(view, sampleRate: 24_000) == LTXMediaEncoding.wav(contiguous, sampleRate: 24_000))
        let mono = try LTXMediaEncoding.wav(MLXArray([Float(0.5)]).reshaped([1, 1, 1]), sampleRate: 16_000)
        #expect(Array(mono.suffix(2)) == [255, 63])
    }

    @Test(arguments: [DType.float32, .bfloat16, .float16])
    func rgbPreservesStridedFrameOrderRoundingAndClipping(dtype: DType) throws {
        // Two frames stored in channel-first order; only the second frame is selected.
        let values: [Float] = [9, 9, -1, 0, 9, 9, 1, -2, 9, 9, 2, 0.5]
        let video = MLXArray(values).reshaped([1, 3, 2, 1, 2]).asType(dtype)
        var bytes = Data()
        try LTXMediaEncoding.rgb24Frame(video[0, 0..., 1, 0..., 0...].transposed(1, 2, 0), into: &bytes)
        #expect(Array(bytes) == [0,255,255,128,0,191])
        let retained = bytes
        try LTXMediaEncoding.rgb24Frame(.zeros([1, 2, 3], dtype: dtype), into: &bytes)
        #expect(Array(bytes) == [128,128,128,128,128,128])
        #expect(Array(retained) == [0,255,255,128,0,191])
        try LTXMediaEncoding.rgb24Frame(.ones([1, 1, 3], dtype: dtype), into: &bytes)
        #expect(Array(bytes) == [255,255,255])
    }

    @Test func rgbPreservesEveryQuantizationBoundary() throws {
        let centers = (0..<256).map { Float($0) / 127.5 - 1 }
        let boundaries = (0..<255).flatMap { i -> [Float] in
            let x = (Float(i) + 0.5) / 127.5 - 1
            return [x.nextDown, x, x.nextUp]
        }
        let values = centers + boundaries + [Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude]
        let triplets = values.flatMap { [$0, $0, $0] }
        var bytes = Data()
        try LTXMediaEncoding.rgb24Frame(MLXArray(triplets).reshaped([1, values.count, 3]), into: &bytes)
        let expected = triplets.map { UInt8(min(255, max(0, (($0 + 1) * 127.5).rounded()))) }
        #expect(Array(bytes) == expected)
    }

    @Test(arguments: [Float.nan, Float.infinity, -Float.infinity])
    func rejectsNonFiniteSamples(value: Float) {
        #expect(throws: LTXMediaEncoding.EncodingError.self) {
            try LTXMediaEncoding.wav(MLXArray([Float(0), value]).reshaped([1, 2, 1]), sampleRate: 48_000)
        }
        #expect(throws: LTXMediaEncoding.EncodingError.self) {
            var bytes = Data()
            try LTXMediaEncoding.rgb24Frame(MLXArray([Float(0), 0, value]).reshaped([1, 1, 3]), into: &bytes)
        }
    }

    @Test func rejectsInvalidShapeAndWAVFields() {
        for shape in [[2, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1]] {
            #expect(throws: LTXMediaEncoding.EncodingError.self) {
                try LTXMediaEncoding.wav(.zeros(shape), sampleRate: 48_000)
            }
        }
        for rate in [0, -1, Int.max] {
            #expect(throws: LTXMediaEncoding.EncodingError.self) {
                try LTXMediaEncoding.wav(.zeros([1, 2, 1]), sampleRate: rate)
            }
        }
        for shape in [[1, 1, 4], [0, 1, 3], [1, 3]] {
            #expect(throws: LTXMediaEncoding.EncodingError.self) {
                var bytes = Data()
                try LTXMediaEncoding.rgb24Frame(.zeros(shape), into: &bytes)
            }
        }
    }
}

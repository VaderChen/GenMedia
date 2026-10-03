import MLX
import Testing
@testable import LTXVideoSwiftRuntime

struct LTXVocoderTests {
    @Test(arguments: [1, 2, 13])
    func sincResamplerPreservesSingletonBatchAndConstantSignals(length: Int) throws {
        let resampler = LTXHannSincResampler(upsampleFactor: 3)
        let input = MLXArray.ones([1, length])
        let mono = try resampler(input)
        #expect(mono.shape == [1, length * 3])
        #expect(max(abs(mono - 1)).item(Float.self) < 0.01)
        let batch = try resampler(concatenated([input, input * -0.5], axis: 0))
        #expect(batch.shape == [2, length * 3])
        #expect(max(abs(batch[0] - 1)).item(Float.self) < 0.01)
        #expect(max(abs(batch[1] + 0.5)).item(Float.self) < 0.01)
    }
}

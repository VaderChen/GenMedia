import Foundation
import Testing

@testable import MiniMaxH3SwiftRuntime

@Suite("MiniMax H3 video VAE")
struct MiniMaxH3VideoVAETests {
    private let configuration = MiniMaxH3VideoVAEConfiguration.default

    @Test("Geometry follows the reference constants")
    func geometry() {
        #expect(configuration.hiddenSize == 2048)
        // prod(space_down) = 16, prod(time_down) = 4
        #expect(configuration.spatialRatio == 16)
        #expect(configuration.temporalRatio == 4)
        // 3 channels * 4 frames * 16 * 16
        #expect(configuration.patchOutputDim == 3072)
        // rope covers 3/4 of each 64-wide head, split into 24 pairs
        #expect(configuration.ropeDim == 48)
        #expect(configuration.ropePairCount == 24)
        // 4 register tokens plus one zero token
        #expect(configuration.suffixTokenCount == 5)
    }

    @Test("Temporal tiling follows the H3 reference plan")
    func temporalTilingPlan() {
        let tiling = MiniMaxH3VideoVAETemporalTiling()
        #expect(tiling.tokensPerChunk(temporalRatio: configuration.temporalRatio) == 5)
        #expect(tiling.tokenOverlap(temporalRatio: configuration.temporalRatio) == 2)
        #expect(tiling.framePrePadding(temporalRatio: configuration.temporalRatio) == 3)
        #expect(tiling.frameOverlap(temporalRatio: configuration.temporalRatio) == 5)

        let plan = tiling.plan(
            latentFrameCount: 12,
            temporalRatio: configuration.temporalRatio
        )
        #expect(plan.paddedTokenCount == 12)
        #expect(plan.padTokenCount == 0)
        #expect(plan.chunkCount == 2)
        #expect(plan.ranges == [0 ..< 7, 5 ..< 12])
    }

    @Test("Temporal tiling preserves reference output frame counts")
    func temporalOutputFrameCount() {
        let tiling = MiniMaxH3VideoVAETemporalTiling()
        #expect(tiling.outputFrameCount(latentFrameCount: 2, temporalRatio: 4) == 5)
        #expect(tiling.outputFrameCount(latentFrameCount: 3, temporalRatio: 4) == 9)
        #expect(tiling.outputFrameCount(latentFrameCount: 7, temporalRatio: 4) == 22)
        #expect(tiling.outputFrameCount(latentFrameCount: 12, temporalRatio: 4) == 39)
    }

    @Test("Pipeline keeps short latent clips on the single-shot path")
    func temporalTilingThreshold() {
        let tiling = MiniMaxH3VideoVAETemporalTiling()
        #expect(!tiling.shouldUse(for: 3))
        #expect(tiling.shouldUse(for: 8))
    }

    @Test("Expected shapes match the real checkpoint layout")
    func expectedShapes() {
        let expected = configuration.expectedDecoderShapes
        #expect(expected["decoder.x_embedder.weight"] == [2048, 24])
        #expect(expected["decoder.transformer_blocks.0.attn.to_qkv.weight"] == [6144, 2048])
        #expect(expected["decoder.transformer_blocks.0.attn.to_out.weight"] == [2048, 2048])
        // w1 emits 2x the inner width because the FFN is SiLU-gated
        #expect(expected["decoder.transformer_blocks.0.ff.w1.weight"] == [16384, 2048])
        #expect(expected["decoder.transformer_blocks.0.ff.w2.weight"] == [2048, 8192])
        #expect(expected["decoder.transformer_blocks.35.scale2"] == [2048])
        #expect(expected["decoder.proj_out.weight"] == [3072, 2048])
        // 36 blocks x 12 tensors + 8 decoder-level + 2 for post_quant_conv
        #expect(expected.count == 36 * 12 + 8 + 2)
        #expect(expected["post_quant_conv.weight"] == [24, 24, 1, 1, 1])
    }

    @Test("Validation rejects a mis-shaped tensor")
    func validationRejectsMisShapedTensor() {
        var shapes = configuration.expectedDecoderShapes
        shapes["decoder.proj_out.weight"] = [2048, 3072]
        #expect(throws: MiniMaxH3WeightError.self) {
            try configuration.validate(against: shapes)
        }
    }

    @Test("Validation rejects a missing tensor")
    func validationRejectsMissingTensor() {
        var shapes = configuration.expectedDecoderShapes
        shapes.removeValue(forKey: "decoder.transformer_blocks.12.norm2.weight")
        #expect(throws: MiniMaxH3WeightError.self) {
            try configuration.validate(against: shapes)
        }
    }

    @Test("Patch centres are row-major and in [-1, 1]")
    func patchCentreLayout() {
        // A single position sits at the centre of the axis.
        #expect(MiniMaxH3VideoVAEDecoder.patchCentres(1) == [0.0])
        // Two positions sit at -0.5 and +0.5, not -1 and +1: the reference
        // uses cell centres (arange(0.5, n)/n), not endpoints.
        let two = MiniMaxH3VideoVAEDecoder.patchCentres(2)
        #expect(abs(two[0] - -0.5) < 1e-6)
        #expect(abs(two[1] - 0.5) < 1e-6)
        let four = MiniMaxH3VideoVAEDecoder.patchCentres(4)
        #expect(abs(four[0] - -0.75) < 1e-6)
        #expect(abs(four[3] - 0.75) < 1e-6)
    }

    @Test("Inverse frequency count sets the rotary pair count")
    func inverseFrequencyCount() {
        // arange(0, 1, 2*3/48) = arange(0, 1, 0.125) -> 8 values per axis,
        // 3 axes -> 24 pairs -> 48 rotated channels.
        let frequencies = MiniMaxH3VideoVAEDecoder.inverseFrequencies(
            ropeDim: configuration.ropeDim,
            theta: configuration.ropeTheta
        )
        #expect(frequencies.count == 8)
        #expect(frequencies.count * 3 == configuration.ropePairCount)
        // First frequency is theta^0 = 1.
        #expect(abs(frequencies[0] - 1.0) < 1e-6)
        #expect(frequencies[1] < frequencies[0])
    }
}

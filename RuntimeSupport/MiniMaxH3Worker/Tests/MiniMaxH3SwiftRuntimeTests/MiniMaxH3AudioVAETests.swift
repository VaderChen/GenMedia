import Foundation
import Testing

@testable import MiniMaxH3SwiftRuntime

@Suite("MiniMax H3 audio VAE")
struct MiniMaxH3AudioVAETests {
    private let configuration = MiniMaxH3AudioVAEConfiguration.default

    @Test("Hop length is the product of the upsample rates")
    func hopLength() {
        // 5*5*2*2*2*2*2 = 800 samples per latent frame at 32 kHz -> 40 fps.
        #expect(configuration.hopLength == 800)
        #expect(configuration.sampleRate / configuration.hopLength == 40)
    }

    @Test("Channel width halves at every upsample stage")
    func channelSchedule() {
        let widths = configuration.upsampleRates.indices.map {
            configuration.channels(afterStage: $0)
        }
        #expect(widths == [512, 256, 128, 64, 32, 16, 8])
        #expect(configuration.finalChannels == 8)
    }

    @Test("Resblock count is stages times kernel sizes")
    func resblockCount() {
        // 7 upsample stages x 3 resblock kernels = 21 AMP blocks,
        // each with 3 convs1 + 3 convs2 and 6 anti-aliased activations.
        #expect(configuration.resblockCount == 21)
        let expected = configuration.expectedDecoderShapes
        let convs1 = expected.keys.filter { $0.contains(".convs1.") && $0.hasSuffix(".weight") }
        #expect(convs1.count == 63)
        let alphas = expected.keys.filter { $0.hasSuffix(".act.alpha") }
        // 21 blocks x 6 activations, plus activation_post
        #expect(alphas.count == 21 * 6 + 1)
    }

    @Test("Expected shapes match the real checkpoint layout")
    func expectedShapes() {
        let expected = configuration.expectedDecoderShapes
        #expect(expected["dec_in_proj.weight"] == [2048, 32, 1])
        #expect(expected["decoder.conv_pre.weight"] == [1024, 2048, 7])
        // conv_post has no bias (use_bias_at_final=False)
        #expect(expected["decoder.conv_post.weight"] == [1, 8, 7])
        #expect(expected["decoder.conv_post.bias"] == nil)
        // ConvTranspose1d weight is [C_in, C_out, K]
        #expect(expected["decoder.ups.0.0.weight"] == [1024, 512, 9])
        #expect(expected["decoder.ups.6.0.weight"] == [16, 8, 4])
        #expect(expected["decoder.resblocks.0.convs1.0.weight"] == [512, 512, 3])
        #expect(expected["decoder.activation_post.act.alpha"] == [8])
        #expect(expected["latents_mean"] == [32])
    }

    @Test("Encode-only tensors are not required for decoding")
    func encodeOnlyTensorsAreNotRequired() {
        let expected = configuration.expectedDecoderShapes
        // The checkpoint also carries encoder/pre_block/mean_proj/logs_proj,
        // which the decode path never touches.
        #expect(expected.keys.first { $0.hasPrefix("encoder.") } == nil)
        #expect(expected.keys.first { $0.hasPrefix("pre_block.") } == nil)
        #expect(expected["mean_proj.weight"] == nil)
        #expect(expected["logs_proj.weight"] == nil)
    }

    @Test("Dilated padding keeps the sequence length")
    func dilatedPadding() {
        // (k*d - d)/2 keeps 'same' length for odd kernels.
        #expect(MiniMaxH3AudioVAEDecoder.padding(kernel: 3, dilation: 1) == 1)
        #expect(MiniMaxH3AudioVAEDecoder.padding(kernel: 3, dilation: 3) == 3)
        #expect(MiniMaxH3AudioVAEDecoder.padding(kernel: 3, dilation: 5) == 5)
        #expect(MiniMaxH3AudioVAEDecoder.padding(kernel: 11, dilation: 5) == 25)
    }

    @Test("Validation rejects a mis-shaped tensor")
    func validationRejectsMisShapedTensor() {
        var shapes = configuration.expectedDecoderShapes
        shapes["decoder.ups.0.0.weight"] = [512, 1024, 9]
        #expect(throws: MiniMaxH3WeightError.self) {
            try configuration.validate(against: shapes)
        }
    }

    @Test("Validation rejects a missing tensor")
    func validationRejectsMissingTensor() {
        var shapes = configuration.expectedDecoderShapes
        shapes.removeValue(forKey: "decoder.resblocks.20.convs2.2.weight")
        #expect(throws: MiniMaxH3WeightError.self) {
            try configuration.validate(against: shapes)
        }
    }
}

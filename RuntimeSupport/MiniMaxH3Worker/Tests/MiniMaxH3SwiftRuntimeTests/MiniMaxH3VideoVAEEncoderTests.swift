import Foundation
import Testing

@testable import MiniMaxH3SwiftRuntime

@Suite("MiniMax H3 video VAE encoder")
struct MiniMaxH3VideoVAEEncoderTests {
    @Test("Channel schedule follows ch * ch_mult")
    func channelSchedule() {
        // ch = 128, ch_mult = (1, 2, 2, 4, 4, 8)
        #expect(MiniMaxH3VideoVAEEncoder.blockChannels == [128, 256, 256, 512, 512, 1024])
        // Each level's first block reads the previous level's width.
        #expect(MiniMaxH3VideoVAEEncoder.blockInputChannels == [128, 128, 256, 256, 512, 512])
    }

    @Test("Downsampling product matches the VAE ratios")
    func downsampleProduct() {
        let space = MiniMaxH3VideoVAEEncoder.spaceDown.reduce(1, *)
        let time = MiniMaxH3VideoVAEEncoder.timeDown.reduce(1, *)
        #expect(space == 16)
        #expect(time == 4)
        #expect(space == MiniMaxH3VideoVAEConfiguration.default.spatialRatio)
        #expect(time == MiniMaxH3VideoVAEConfiguration.default.temporalRatio)
    }

    @Test("Expected shapes match the real checkpoint layout")
    func expectedShapes() {
        let expected = MiniMaxH3VideoVAEEncoder.expectedShapes()
        #expect(expected["encoder.conv_in.weight"] == [128, 3, 3, 3, 3])
        // conv_out emits both halves of the posterior, 2 * 24.
        #expect(expected["encoder.conv_out.weight"] == [48, 1024, 3, 3, 3])
        #expect(expected["quant_conv.weight"] == [48, 48, 1, 1, 1])
        #expect(expected["encoder.down.5.block.0.conv1.weight"] == [1024, 512, 3, 3, 3])
        // Downsamples exist only where space * time > 1, i.e. levels 0...3.
        #expect(expected["encoder.down.0.downsample.conv.weight"] == [128, 128, 3, 3, 3])
        #expect(expected["encoder.down.4.downsample.conv.weight"] == nil)
        #expect(expected["encoder.down.5.downsample.conv.weight"] == nil)
        // nin_shortcut only where the block changes width: levels 1, 3, 5.
        #expect(expected["encoder.down.1.block.0.nin_shortcut.weight"] == [256, 128, 1, 1, 1])
        #expect(expected["encoder.down.0.block.0.nin_shortcut.weight"] == nil)
        #expect(expected["encoder.down.2.block.0.nin_shortcut.weight"] == nil)
    }

    @Test("Tensor count matches the checkpoint")
    func tensorCount() {
        // 116 encoder tensors in the file, plus quant_conv's weight and bias.
        #expect(MiniMaxH3VideoVAEEncoder.expectedShapes().count == 116 + 2)
    }

    @Test("Validation rejects a mis-shaped tensor")
    func validationRejectsMisShapedTensor() {
        var shapes = MiniMaxH3VideoVAEEncoder.expectedShapes()
        shapes["encoder.conv_in.weight"] = [128, 3, 1, 3, 3]
        #expect(throws: MiniMaxH3WeightError.self) {
            try MiniMaxH3VideoVAEEncoder.validate(
                shapes: shapes, configuration: .default
            )
        }
    }

    @Test("Validation rejects a missing tensor")
    func validationRejectsMissingTensor() {
        var shapes = MiniMaxH3VideoVAEEncoder.expectedShapes()
        shapes.removeValue(forKey: "quant_conv.bias")
        #expect(throws: MiniMaxH3WeightError.self) {
            try MiniMaxH3VideoVAEEncoder.validate(
                shapes: shapes, configuration: .default
            )
        }
    }
}

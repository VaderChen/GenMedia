import MLX
import MLXNN

public final class LTXVideoVAEEncoder: Module {
    @ModuleInfo(key: "conv_in") public var convIn: LTXConv3DBlock
    @ModuleInfo(key: "down_blocks") public var downBlocks: [UnaryLayer]
    @ModuleInfo(key: "conv_out") public var convOut: LTXConv3DBlock
    @ModuleInfo(key: "per_channel_statistics")
    public var perChannelStatistics: LTXEncoderPerChannelStatistics

    public let configuration: LTXVideoVAEConfiguration

    public init(configuration: LTXVideoVAEConfiguration) {
        self.configuration = configuration
        let padding = configuration.spatialPaddingMode
        self._convIn = ModuleInfo(wrappedValue: LTXConv3DBlock(
            inputChannels: 48,
            outputChannels: 128,
            causal: true,
            spatialPaddingMode: padding
        ), key: "conv_in")
        self._downBlocks = ModuleInfo(wrappedValue: [
            LTXResBlockStage(channels: 128, count: 4, causal: true, spatialPaddingMode: padding),
            LTXSpaceToDepthDownsample(
                inputChannels: 128,
                outputChannels: 256,
                stride: (1, 2, 2),
                spatialPaddingMode: padding
            ),
            LTXResBlockStage(channels: 256, count: 6, causal: true, spatialPaddingMode: padding),
            LTXSpaceToDepthDownsample(
                inputChannels: 256,
                outputChannels: 512,
                stride: (2, 1, 1),
                spatialPaddingMode: padding
            ),
            LTXResBlockStage(channels: 512, count: 4, causal: true, spatialPaddingMode: padding),
            LTXSpaceToDepthDownsample(
                inputChannels: 512,
                outputChannels: 1024,
                stride: (2, 2, 2),
                spatialPaddingMode: padding
            ),
            LTXResBlockStage(channels: 1024, count: 2, causal: true, spatialPaddingMode: padding),
            LTXSpaceToDepthDownsample(
                inputChannels: 1024,
                outputChannels: 1024,
                stride: (2, 2, 2),
                spatialPaddingMode: padding
            ),
            LTXResBlockStage(channels: 1024, count: 2, causal: true, spatialPaddingMode: padding)
        ], key: "down_blocks")
        self._convOut = ModuleInfo(wrappedValue: LTXConv3DBlock(
            inputChannels: 1024,
            outputChannels: 129,
            causal: true,
            spatialPaddingMode: padding
        ), key: "conv_out")
        self._perChannelStatistics = ModuleInfo(
            wrappedValue: LTXEncoderPerChannelStatistics(channels: 128),
            key: "per_channel_statistics"
        )
        super.init()
    }

    public func normalizeLatent(_ latent: MLXArray) -> MLXArray {
        let mean = perChannelStatistics.meanOfMeans.reshaped(1, 1, 1, 1, -1)
        let standardDeviation = perChannelStatistics.stdOfMeans.reshaped(1, 1, 1, 1, -1)
        return (latent - mean) / standardDeviation
    }

    public func encode(_ pixels: MLXArray) throws -> MLXArray {
        guard pixels.ndim == 5,
              pixels.shape[1] == 3,
              pixels.shape[2] > 0,
              pixels.shape[3] > 0,
              pixels.shape[4] > 0,
              pixels.shape[2] >= 1,
              (pixels.shape[2] - 1).isMultiple(of: 8),
              pixels.shape[3].isMultiple(of: 32),
              pixels.shape[4].isMultiple(of: 32) else {
            throw LTXVideoRuntimeError.invalidConfiguration(
                "影片像素必須是 [B,3,8n+1,H,W]，且 H/W 為 32 的倍數，實際為 \(pixels.shape)。"
            )
        }

        let outputDType = pixels.dtype
        var value = pixels
        if value.dtype != convIn.conv.weight.dtype {
            value = value.asType(convIn.conv.weight.dtype)
        }
        value = value.transposed(0, 2, 3, 4, 1)
        value = ltxPatchifySpatial(value, patchSize: configuration.patchSize)
        value = convIn(value)
        for block in downBlocks {
            value = block(value)
        }
        value = convOut(silu(ltxPixelNorm(value)))
        value = value[0..., 0..., 0..., 0..., ..<128]
        value = normalizeLatent(value)
        return value.transposed(0, 4, 1, 2, 3).asType(outputDType)
    }
}

private func ltxPatchifySpatial(_ input: MLXArray, patchSize: Int) -> MLXArray {
    let shape = input.shape
    precondition(shape[2].isMultiple(of: patchSize))
    precondition(shape[3].isMultiple(of: patchSize))
    return input
        .reshaped(
            shape[0], shape[1], shape[2] / patchSize, patchSize,
            shape[3] / patchSize, patchSize, shape[4]
        )
        .transposed(0, 1, 2, 4, 6, 5, 3)
        .reshaped(shape[0], shape[1], shape[2] / patchSize, shape[3] / patchSize, shape[4] * patchSize * patchSize)
}

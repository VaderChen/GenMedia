import MLX
import MLXNN

public struct LTXAudioVAEDecoderConfiguration: Sendable, Equatable {
    public let latentChannels: Int
    public let latentFrequencyBins: Int
    public let outputChannels: Int
    public let outputFrequencyBins: Int
    public let causal: Bool

    public init(
        latentChannels: Int = 8,
        latentFrequencyBins: Int = 16,
        outputChannels: Int = 2,
        outputFrequencyBins: Int = 64,
        causal: Bool = true
    ) throws {
        guard latentChannels == 8,
              latentFrequencyBins == 16,
              outputChannels == 2,
              outputFrequencyBins == 64 else {
            throw LTXVideoRuntimeError.invalidConfiguration(
                "Audio VAE decoder 僅支援 latent=[8,T,16] 與 mel=[2,T,64]。"
            )
        }
        self.latentChannels = latentChannels
        self.latentFrequencyBins = latentFrequencyBins
        self.outputChannels = outputChannels
        self.outputFrequencyBins = outputFrequencyBins
        self.causal = causal
    }
}

@inline(__always)
private func ltxAudioPixelNorm(_ input: MLXArray, epsilon: Float = 1e-6) -> MLXArray {
    let weight = MLXArray.ones([input.shape.last ?? 1], dtype: input.dtype)
    return MLXFast.rmsNorm(input, weight: weight, eps: epsilon)
}

public final class LTXAudioConv2DBlock: Module, UnaryLayer {
    @ModuleInfo public var conv: Conv2d

    private let kernelSize: (height: Int, width: Int)
    private let padding: (height: Int, width: Int)
    private let causal: Bool

    public init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: (height: Int, width: Int) = (3, 3),
        padding: (height: Int, width: Int) = (1, 1),
        causal: Bool
    ) {
        self.kernelSize = kernelSize
        self.padding = padding
        self.causal = causal
        self._conv = ModuleInfo(
            wrappedValue: Conv2d(
                inputChannels: inputChannels,
                outputChannels: outputChannels,
                kernelSize: .init(kernelSize),
                padding: 0,
                bias: true
            ),
            key: "conv"
        )
        super.init()
    }

    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        let top = causal ? max(0, kernelSize.height - 1) : padding.height
        let bottom = causal ? 0 : padding.height
        let left = padding.width
        let right = padding.width
        let widths = [
            (0, 0),
            (top, bottom),
            (left, right),
            (0, 0)
        ]
        let padded = widths.contains { $0.0 != 0 || $0.1 != 0 }
            ? MLX.padded(input, widths: widths.map { .init($0) })
            : input
        return conv(padded)
    }
}

public final class LTXAudioResBlock: Module, UnaryLayer {
    @ModuleInfo(key: "conv1") public var conv1: LTXAudioConv2DBlock
    @ModuleInfo(key: "conv2") public var conv2: LTXAudioConv2DBlock
    @ModuleInfo(key: "nin_shortcut") public var ninShortcut: LTXAudioConv2DBlock?

    public init(inChannels: Int, outChannels: Int? = nil, causal: Bool) {
        let outChannels = outChannels ?? inChannels
        self._conv1 = ModuleInfo(wrappedValue: LTXAudioConv2DBlock(
            inputChannels: inChannels,
            outputChannels: outChannels,
            causal: causal
        ), key: "conv1")
        self._conv2 = ModuleInfo(wrappedValue: LTXAudioConv2DBlock(
            inputChannels: outChannels,
            outputChannels: outChannels,
            causal: causal
        ), key: "conv2")
        self._ninShortcut = ModuleInfo(wrappedValue: inChannels == outChannels
            ? nil
            : LTXAudioConv2DBlock(
                inputChannels: inChannels,
                outputChannels: outChannels,
                kernelSize: (1, 1),
                padding: (0, 0),
                causal: false
            ), key: "nin_shortcut")
        super.init()
    }

    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        var residual = input
        var value = ltxAudioPixelNorm(input)
        value = silu(value)
        value = conv1(value)
        value = ltxAudioPixelNorm(value)
        value = silu(value)
        value = conv2(value)
        if let ninShortcut {
            residual = ninShortcut(residual)
        }
        return value + residual
    }
}

public final class LTXAudioUpsample: Module, UnaryLayer {
    @ModuleInfo(key: "conv") public var conv: LTXAudioConv2DBlock
    private let causal: Bool

    public init(channels: Int, causal: Bool) {
        self.causal = causal
        self._conv = ModuleInfo(wrappedValue: LTXAudioConv2DBlock(
            inputChannels: channels,
            outputChannels: channels,
            causal: causal
        ), key: "conv")
        super.init()
    }

    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        var value = MLX.repeated(input, count: 2, axis: 1)
        value = MLX.repeated(value, count: 2, axis: 2)
        value = conv(value)
        if causal {
            value = value[0..., 1..<value.shape[1], 0..., 0...]
        }
        return value
    }
}

public final class LTXAudioUpBlock: Module, UnaryLayer {
    @ModuleInfo(key: "block") public var blocks: [LTXAudioResBlock]
    @ModuleInfo(key: "upsample") public var upsample: LTXAudioUpsample?

    public init(
        inChannels: Int,
        outChannels: Int,
        blockCount: Int = 3,
        addUpsample: Bool,
        causal: Bool
    ) {
        self._blocks = ModuleInfo(wrappedValue: (0..<blockCount).map { index in
            LTXAudioResBlock(
                inChannels: index == 0 ? inChannels : outChannels,
                outChannels: outChannels,
                causal: causal
            )
        }, key: "block")
        self._upsample = ModuleInfo(
            wrappedValue: addUpsample ? LTXAudioUpsample(channels: outChannels, causal: causal) : nil,
            key: "upsample"
        )
        super.init()
    }

    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        var value = input
        for block in blocks {
            value = block(value)
        }
        if let upsample {
            value = upsample(value)
        }
        return value
    }
}

public final class LTXAudioMidBlock: Module, UnaryLayer {
    @ModuleInfo(key: "block_1") public var block1: LTXAudioResBlock
    @ModuleInfo(key: "block_2") public var block2: LTXAudioResBlock

    public init(channels: Int, causal: Bool) {
        self._block1 = ModuleInfo(wrappedValue: LTXAudioResBlock(
            inChannels: channels,
            causal: causal
        ), key: "block_1")
        self._block2 = ModuleInfo(wrappedValue: LTXAudioResBlock(
            inChannels: channels,
            causal: causal
        ), key: "block_2")
        super.init()
    }

    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        block2(block1(input))
    }
}

public final class LTXAudioPerChannelStatistics: Module {
    public var meanOfMeans: MLXArray
    public var stdOfMeans: MLXArray

    public init(channels: Int) {
        self.meanOfMeans = MLXArray.zeros([channels])
        self.stdOfMeans = MLXArray.ones([channels])
        super.init()
    }
}

public final class LTXAudioVAEDecoder: Module {
    @ModuleInfo(key: "conv_in") public var convIn: LTXAudioConv2DBlock
    @ModuleInfo(key: "mid") public var mid: LTXAudioMidBlock
    @ModuleInfo(key: "up") public var up: [LTXAudioUpBlock]
    @ModuleInfo(key: "conv_out") public var convOut: LTXAudioConv2DBlock
    @ModuleInfo(key: "per_channel_statistics")
    public var perChannelStatistics: LTXAudioPerChannelStatistics

    public let configuration: LTXAudioVAEDecoderConfiguration

    public init(configuration: LTXAudioVAEDecoderConfiguration) {
        self.configuration = configuration
        self._convIn = ModuleInfo(wrappedValue: LTXAudioConv2DBlock(
            inputChannels: configuration.latentChannels,
            outputChannels: 512,
            causal: configuration.causal
        ), key: "conv_in")
        self._mid = ModuleInfo(wrappedValue: LTXAudioMidBlock(
            channels: 512,
            causal: configuration.causal
        ), key: "mid")
        self._up = ModuleInfo(wrappedValue: [
            LTXAudioUpBlock(
                inChannels: 256,
                outChannels: 128,
                addUpsample: false,
                causal: configuration.causal
            ),
            LTXAudioUpBlock(
                inChannels: 512,
                outChannels: 256,
                addUpsample: true,
                causal: configuration.causal
            ),
            LTXAudioUpBlock(
                inChannels: 512,
                outChannels: 512,
                addUpsample: true,
                causal: configuration.causal
            )
        ], key: "up")
        self._convOut = ModuleInfo(wrappedValue: LTXAudioConv2DBlock(
            inputChannels: 128,
            outputChannels: configuration.outputChannels,
            causal: configuration.causal
        ), key: "conv_out")
        self._perChannelStatistics = ModuleInfo(
            wrappedValue: LTXAudioPerChannelStatistics(channels: 128),
            key: "per_channel_statistics"
        )
        super.init()
    }

    public func decode(_ latent: MLXArray) throws -> MLXArray {
        guard latent.ndim == 4,
              latent.shape[1] == configuration.latentChannels,
              latent.shape[2] > 0,
              latent.shape[3] == configuration.latentFrequencyBins else {
            throw LTXVideoRuntimeError.invalidLatentShape(latent.shape)
        }

        let batch = latent.shape[0]
        let time = latent.shape[2]
        let channels = latent.shape[1]
        let frequencyBins = latent.shape[3]
        let flattened = latent
            .transposed(0, 2, 1, 3)
            .reshaped(batch, time, channels * frequencyBins)
        let mean = perChannelStatistics.meanOfMeans.reshaped(1, 1, -1)
        let standardDeviation = perChannelStatistics.stdOfMeans.reshaped(1, 1, -1)
        var value = flattened * standardDeviation + mean
        value = value
            .reshaped(batch, time, channels, frequencyBins)
            .transposed(0, 1, 3, 2)

        if value.dtype != convIn.conv.weight.dtype {
            value = value.asType(convIn.conv.weight.dtype)
        }
        value = convIn(value)
        value = mid(value)
        for block in up.reversed() {
            value = block(value)
        }
        value = convOut(silu(ltxAudioPixelNorm(value)))
        return value.transposed(0, 3, 1, 2)
    }
}

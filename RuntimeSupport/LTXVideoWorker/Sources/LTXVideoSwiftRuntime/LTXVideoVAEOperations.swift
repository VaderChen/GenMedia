import MLX
import MLXNN

@inline(__always)
func ltxPixelNorm(_ input: MLXArray, epsilon: Float = 1e-8) -> MLXArray {
    let weight = MLXArray.ones([input.shape.last ?? 1], dtype: input.dtype)
    return MLXFast.rmsNorm(input, weight: weight, eps: epsilon)
}

public final class LTXConv3DBlock: Module, UnaryLayer {
    @ModuleInfo public var conv: Conv3d

    private let kernelSize: (Int, Int, Int)
    private let causal: Bool
    private let spatialPadding: (Int, Int)
    private let spatialPaddingMode: String

    public init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: (Int, Int, Int) = (3, 3, 3),
        stride: (Int, Int, Int) = (1, 1, 1),
        padding: (Int, Int, Int) = (1, 1, 1),
        causal: Bool,
        spatialPaddingMode: String
    ) {
        self._conv = ModuleInfo(wrappedValue: Conv3d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: .init(kernelSize),
            stride: .init(stride),
            padding: 0,
            bias: true
        ))
        self.kernelSize = kernelSize
        self.causal = causal
        self.spatialPadding = (padding.1, padding.2)
        self.spatialPaddingMode = spatialPaddingMode
        super.init()
    }

    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        var value = input
        let temporalKernel = kernelSize.0
        if causal {
            if temporalKernel > 1 {
                let first = value[0..., 0..<1, 0..., 0..., 0...]
                let prefix = MLX.repeated(first, count: temporalKernel - 1, axis: 1)
                value = MLX.concatenated([prefix, value], axis: 1)
            }
        } else {
            let amount = (temporalKernel - 1) / 2
            if amount > 0 {
                let first = value[0..., 0..<1, 0..., 0..., 0...]
                let lastIndex = value.shape[1] - 1
                let last = value[0..., lastIndex..<value.shape[1], 0..., 0..., 0...]
                value = MLX.concatenated([
                    MLX.repeated(first, count: amount, axis: 1),
                    value,
                    MLX.repeated(last, count: amount, axis: 1)
                ], axis: 1)
            }
        }

        let (heightPadding, widthPadding) = spatialPadding
        if heightPadding > 0 || widthPadding > 0 {
            if spatialPaddingMode == "reflect" {
                if heightPadding > 0 {
                    value = MLX.concatenated([
                        value[0..., 0..., 1..<(1 + heightPadding), 0..., 0...],
                        value,
                        value[0..., 0..., (value.shape[2] - heightPadding - 1)..<(value.shape[2] - 1), 0..., 0...]
                    ], axis: 2)
                }
                if widthPadding > 0 {
                    value = MLX.concatenated([
                        value[0..., 0..., 0..., 1..<(1 + widthPadding), 0...],
                        value,
                        value[0..., 0..., 0..., (value.shape[3] - widthPadding - 1)..<(value.shape[3] - 1), 0...]
                    ], axis: 3)
                }
            } else {
                value = MLX.padded(
                    value,
                    widths: [
                        .init((0, 0)), .init((0, 0)),
                        .init((heightPadding, heightPadding)),
                        .init((widthPadding, widthPadding)), .init((0, 0))
                    ]
                )
            }
        }
        return conv(value)
    }
}

public final class LTXResBlock3D: Module, UnaryLayer {
    @ModuleInfo public var conv1: LTXConv3DBlock
    @ModuleInfo public var conv2: LTXConv3DBlock

    public init(channels: Int, causal: Bool, spatialPaddingMode: String) {
        self._conv1 = ModuleInfo(wrappedValue: LTXConv3DBlock(
            inputChannels: channels,
            outputChannels: channels,
            causal: causal,
            spatialPaddingMode: spatialPaddingMode
        ))
        self._conv2 = ModuleInfo(wrappedValue: LTXConv3DBlock(
            inputChannels: channels,
            outputChannels: channels,
            causal: causal,
            spatialPaddingMode: spatialPaddingMode
        ))
        super.init()
    }

    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        var value = conv1(silu(ltxPixelNorm(input)))
        value = conv2(silu(ltxPixelNorm(value)))
        return value + input
    }
}

public final class LTXResBlockStage: Module, UnaryLayer {
    @ModuleInfo(key: "res_blocks") public var resBlocks: [LTXResBlock3D]

    public init(channels: Int, count: Int, causal: Bool, spatialPaddingMode: String) {
        self._resBlocks = ModuleInfo(
            wrappedValue: (0..<count).map { _ in
                LTXResBlock3D(
                    channels: channels,
                    causal: causal,
                    spatialPaddingMode: spatialPaddingMode
                )
            },
            key: "res_blocks"
        )
        super.init()
    }

    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        resBlocks.reduce(input) { value, block in block(value) }
    }
}

public final class LTXDepthToSpaceUpsample: Module, UnaryLayer {
    @ModuleInfo public var conv: LTXConv3DBlock

    public init(
        inputChannels: Int,
        outputChannels: Int,
        causal: Bool,
        spatialPaddingMode: String
    ) {
        self._conv = ModuleInfo(wrappedValue: LTXConv3DBlock(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            causal: causal,
            spatialPaddingMode: spatialPaddingMode
        ))
        super.init()
    }

    public func callAsFunction(_ input: MLXArray) -> MLXArray { conv(input) }
}

func ltxPixelShuffle3D(
    _ input: MLXArray,
    spatialFactor: Int,
    temporalFactor: Int
) -> MLXArray {
    let shape = input.shape
    let channels = shape[4] / (spatialFactor * spatialFactor * temporalFactor)
    return input
        .reshaped([
            shape[0], shape[1], shape[2], shape[3], channels,
            temporalFactor, spatialFactor, spatialFactor
        ])
        .transposed(0, 1, 5, 2, 6, 3, 7, 4)
        .reshaped([
            shape[0], shape[1] * temporalFactor,
            shape[2] * spatialFactor, shape[3] * spatialFactor, channels
        ])
}

func ltxUnpatchifySpatial(_ input: MLXArray, patchSize: Int) -> MLXArray {
    let shape = input.shape
    let channels = shape[4] / (patchSize * patchSize)
    return input
        .reshaped([
            shape[0], shape[1], shape[2], shape[3], channels,
            patchSize, patchSize
        ])
        .transposed(0, 1, 2, 6, 3, 5, 4)
        .reshaped([
            shape[0], shape[1], shape[2] * patchSize,
            shape[3] * patchSize, channels
        ])
}

public final class LTXPerChannelStatistics: Module {
    public var mean: MLXArray
    public var std: MLXArray

    public init(channels: Int) {
        self.mean = MLXArray.zeros([channels])
        self.std = MLXArray.ones([channels])
        super.init()
    }
}

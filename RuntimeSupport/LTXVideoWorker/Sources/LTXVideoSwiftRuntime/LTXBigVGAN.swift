import Foundation
import MLX
import MLXNN

public struct LTXBigVGANConfiguration: Sendable, Equatable {
    public let inputChannels: Int
    public let initialChannels: Int
    public let upsampleRates: [Int]
    public let upsampleKernelSizes: [Int]
    public let residualKernelSizes: [Int]
    public let residualDilations: [[Int]]
    public let outputChannels: Int
    public let applyFinalActivation: Bool

    public init(
        inputChannels: Int = 128,
        initialChannels: Int = 1536,
        upsampleRates: [Int] = [5, 2, 2, 2, 2, 2],
        upsampleKernelSizes: [Int] = [11, 4, 4, 4, 4, 4],
        residualKernelSizes: [Int] = [3, 7, 11],
        residualDilations: [[Int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        outputChannels: Int = 2,
        applyFinalActivation: Bool = true
    ) throws {
        guard inputChannels > 0, initialChannels > 0, outputChannels > 0 else {
            throw LTXVideoRuntimeError.invalidConfiguration(
                "BigVGAN 的 channels 必須是正整數。"
            )
        }
        guard !upsampleRates.isEmpty,
              upsampleRates.count == upsampleKernelSizes.count,
              upsampleRates.allSatisfy({ $0 > 0 }),
              upsampleKernelSizes.allSatisfy({ $0 > 0 }) else {
            throw LTXVideoRuntimeError.invalidConfiguration(
                "BigVGAN 的 upsample rate 與 kernel 設定無效。"
            )
        }
        guard !residualKernelSizes.isEmpty,
              residualKernelSizes.count == residualDilations.count,
              residualKernelSizes.allSatisfy({ $0 > 0 }),
              residualDilations.allSatisfy({ !$0.isEmpty && $0.allSatisfy({ $0 > 0 }) }) else {
            throw LTXVideoRuntimeError.invalidConfiguration(
                "BigVGAN 的 residual block 設定無效。"
            )
        }
        var channels = initialChannels
        for _ in upsampleRates {
            guard channels.isMultiple(of: 2) else {
                throw LTXVideoRuntimeError.invalidConfiguration(
                    "BigVGAN initial_channels 必須能逐層除以 2。"
                )
            }
            channels /= 2
        }
        self.inputChannels = inputChannels
        self.initialChannels = initialChannels
        self.upsampleRates = upsampleRates
        self.upsampleKernelSizes = upsampleKernelSizes
        self.residualKernelSizes = residualKernelSizes
        self.residualDilations = residualDilations
        self.outputChannels = outputChannels
        self.applyFinalActivation = applyFinalActivation
    }

    public static let ltxBase: LTXBigVGANConfiguration = try! LTXBigVGANConfiguration()
    public static let ltxBWE: LTXBigVGANConfiguration = try! LTXBigVGANConfiguration(
        initialChannels: 512,
        upsampleRates: [6, 5, 2, 2, 2],
        upsampleKernelSizes: [12, 11, 4, 4, 4],
        applyFinalActivation: false
    )

    public var hopLength: Int {
        upsampleRates.reduce(1, *)
    }

    public func outputLength(for inputLength: Int) throws -> Int {
        guard inputLength > 0 else {
            throw LTXVideoRuntimeError.invalidConfiguration(
                "BigVGAN input length 必須是正整數。"
            )
        }
        return inputLength * hopLength
    }
}

private final class LTXSnakeBeta: Module {
    @ParameterInfo(key: "alpha") var alpha: MLXArray
    @ParameterInfo(key: "beta") var beta: MLXArray

    init(channels: Int) {
        self._alpha.wrappedValue = MLXArray.zeros([channels])
        self._beta.wrappedValue = MLXArray.zeros([channels])
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let alpha = exp(self.alpha).reshaped(1, 1, -1).asType(input.dtype)
        let beta = exp(self.beta).reshaped(1, 1, -1).asType(input.dtype)
        return input + (sin(alpha * input).pow(2) / (beta + 1e-9))
    }
}

private final class LTXLowPassKernel: Module {
    @ParameterInfo(key: "filter") var filter: MLXArray

    init(kernelSize: Int) {
        self._filter.wrappedValue = MLXArray.ones([1, kernelSize, 1])
        super.init()
    }
}

private final class LTXDownSample1d: Module {
    @ModuleInfo(key: "lowpass") var lowpass: LTXLowPassKernel

    init(kernelSize: Int = 12) {
        self._lowpass = ModuleInfo(
            wrappedValue: LTXLowPassKernel(kernelSize: kernelSize),
            key: "lowpass"
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let batch = input.shape[0]
        let length = input.shape[1]
        let channels = input.shape[2]
        var value = input.transposed(0, 2, 1).reshaped(batch * channels, length, 1)
        let kernelSize = lowpass.filter.shape[1]
        let even = kernelSize.isMultiple(of: 2) ? 1 : 0
        let leftCount = kernelSize / 2 - even
        let rightCount = kernelSize / 2
        value = MLX.concatenated([
            MLX.repeated(value[0..., 0..<1, 0...], count: leftCount, axis: 1),
            value,
            MLX.repeated(value[0..., (length - 1)..<length, 0...], count: rightCount, axis: 1)
        ], axis: 1)
        value = conv1d(value, lowpass.filter, stride: 2)
        return value.reshaped(batch, channels, value.shape[1]).transposed(0, 2, 1)
    }
}

private final class LTXUpSample1d: Module {
    @ParameterInfo(key: "filter") var filter: MLXArray

    init(kernelSize: Int = 12) {
        self._filter.wrappedValue = MLXArray.ones([1, kernelSize, 1])
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let batch = input.shape[0]
        let length = input.shape[1]
        let channels = input.shape[2]
        var upsampled = MLXArray.zeros([batch, length * 2, channels], dtype: input.dtype)
        upsampled[0..., .stride(from: 0, to: length * 2, by: 2), 0...] = input
        upsampled = upsampled.transposed(0, 2, 1).reshaped(batch * channels, length * 2, 1)
        let padding = filter.shape[1] / 2
        upsampled = MLX.concatenated([
            MLX.repeated(upsampled[0..., 0..<1, 0...], count: padding, axis: 1),
            upsampled,
            MLX.repeated(upsampled[0..., (length * 2 - 1)..<(length * 2), 0...], count: padding - 1, axis: 1)
        ], axis: 1)
        upsampled = conv1d(upsampled, filter)
        return upsampled
            .reshaped(batch, channels, upsampled.shape[1])
            .transposed(0, 2, 1) * 2
    }
}

private final class LTXActivation1d: Module {
    @ModuleInfo(key: "act") var activation: LTXSnakeBeta
    @ModuleInfo(key: "upsample") var upsample: LTXUpSample1d
    @ModuleInfo(key: "downsample") var downsample: LTXDownSample1d

    init(channels: Int, kernelSize: Int = 12) {
        self._activation = ModuleInfo(
            wrappedValue: LTXSnakeBeta(channels: channels),
            key: "act"
        )
        self._upsample = ModuleInfo(
            wrappedValue: LTXUpSample1d(kernelSize: kernelSize),
            key: "upsample"
        )
        self._downsample = ModuleInfo(
            wrappedValue: LTXDownSample1d(kernelSize: kernelSize),
            key: "downsample"
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        downsample(activation(upsample(input)))
    }
}

private final class LTXAMPBlock1: Module {
    @ModuleInfo(key: "convs1") var convs1: [Conv1d]
    @ModuleInfo(key: "convs2") var convs2: [Conv1d]
    @ModuleInfo(key: "acts1") var acts1: [LTXActivation1d]
    @ModuleInfo(key: "acts2") var acts2: [LTXActivation1d]

    init(channels: Int, kernelSize: Int, dilations: [Int]) {
        self._convs1 = ModuleInfo(wrappedValue: dilations.map { dilation in
            Conv1d(
                inputChannels: channels,
                outputChannels: channels,
                kernelSize: kernelSize,
                padding: (kernelSize * dilation - dilation) / 2,
                dilation: dilation
            )
        }, key: "convs1")
        self._convs2 = ModuleInfo(wrappedValue: dilations.map { _ in
            Conv1d(
                inputChannels: channels,
                outputChannels: channels,
                kernelSize: kernelSize,
                padding: kernelSize / 2
            )
        }, key: "convs2")
        self._acts1 = ModuleInfo(wrappedValue: dilations.map { _ in
            LTXActivation1d(channels: channels)
        }, key: "acts1")
        self._acts2 = ModuleInfo(wrappedValue: dilations.map { _ in
            LTXActivation1d(channels: channels)
        }, key: "acts2")
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var value = input
        for index in convs1.indices {
            let residual = value
            value = convs2[index](acts2[index](convs1[index](acts1[index](value))))
            value = value + residual
        }
        return value
    }
}

public final class LTXBigVGANVocoder: Module {
    @ModuleInfo(key: "conv_pre") var convPre: Conv1d
    @ModuleInfo(key: "ups") var upsamplers: [ConvTransposed1d]
    @ModuleInfo(key: "resblocks") fileprivate var residualBlocks: [LTXAMPBlock1]
    @ModuleInfo(key: "act_post") fileprivate var postActivation: LTXActivation1d
    @ModuleInfo(key: "conv_post") var convPost: Conv1d

    public let configuration: LTXBigVGANConfiguration
    private let kernelCount: Int
    private let upsampleCount: Int

    public init(configuration: LTXBigVGANConfiguration) {
        self.configuration = configuration
        self.kernelCount = configuration.residualKernelSizes.count
        self.upsampleCount = configuration.upsampleRates.count
        self._convPre = ModuleInfo(wrappedValue: Conv1d(
            inputChannels: configuration.inputChannels,
            outputChannels: configuration.initialChannels,
            kernelSize: 7,
            padding: 3
        ), key: "conv_pre")

        var channels = configuration.initialChannels
        var configuredUpsamplers: [ConvTransposed1d] = []
        var configuredResidualBlocks: [LTXAMPBlock1] = []
        for (rate, kernel) in zip(configuration.upsampleRates, configuration.upsampleKernelSizes) {
            let outputChannels = channels / 2
            configuredUpsamplers.append(ConvTransposed1d(
                inputChannels: channels,
                outputChannels: outputChannels,
                kernelSize: kernel,
                stride: rate,
                padding: (kernel - rate) / 2
            ))
            for (residualKernel, residualDilations) in zip(
                configuration.residualKernelSizes,
                configuration.residualDilations
            ) {
                configuredResidualBlocks.append(LTXAMPBlock1(
                    channels: outputChannels,
                    kernelSize: residualKernel,
                    dilations: residualDilations
                ))
            }
            channels = outputChannels
        }
        self._upsamplers = ModuleInfo(wrappedValue: configuredUpsamplers, key: "ups")
        self._residualBlocks = ModuleInfo(wrappedValue: configuredResidualBlocks, key: "resblocks")
        self._postActivation = ModuleInfo(
            wrappedValue: LTXActivation1d(channels: channels),
            key: "act_post"
        )
        self._convPost = ModuleInfo(wrappedValue: Conv1d(
            inputChannels: channels,
            outputChannels: configuration.outputChannels,
            kernelSize: 7,
            padding: 3,
            bias: false
        ), key: "conv_post")
        super.init()
    }

    public func callAsFunction(_ mel: MLXArray) -> MLXArray {
        precondition(mel.ndim == 3 && mel.shape[2] == configuration.inputChannels)
        var value = convPre(mel)
        for index in 0..<upsampleCount {
            value = upsamplers[index](value)
            var averaged: MLXArray?
            for residualIndex in 0..<kernelCount {
                let block = residualBlocks[index * kernelCount + residualIndex](value)
                averaged = averaged.map { $0 + block } ?? block
            }
            value = averaged! / Float(kernelCount)
        }
        value = convPost(postActivation(value))
        return configuration.applyFinalActivation ? tanh(value) : value
    }
}

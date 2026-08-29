import Foundation
import MLX
import MLXNN

final class LTXVideo096VAETimeStage: Module, UnaryLayer {
    @ModuleInfo(key: "time_embedder") var timeEmbedder: LTXVideo096TimestepEmbeddingContainer
    @ModuleInfo(key: "res_blocks") var resBlocks: [LTXVideo096VAEResBlock]

    private let channels: Int

    init(channels: Int, count: Int) {
        self.channels = channels
        self._timeEmbedder = ModuleInfo(
            wrappedValue: LTXVideo096TimestepEmbeddingContainer(
                inputDimension: 256,
                outputDimension: channels * 4
            ),
            key: "time_embedder"
        )
        self._resBlocks = ModuleInfo(
            wrappedValue: (0..<count).map { _ in LTXVideo096VAEResBlock(channels: channels) },
            key: "res_blocks"
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray, timestepEmbedding: MLXArray) -> MLXArray {
        let parameters = timeEmbedder(timestepEmbedding)
        return resBlocks.reduce(input) { value, block in
            block(value, timestepParameters: parameters)
        }
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        input
    }
}

final class LTXVideo096VAEResBlock: Module, UnaryLayer {
    @ModuleInfo(key: "conv1") var conv1: LTXConv3DBlock
    @ModuleInfo(key: "conv2") var conv2: LTXConv3DBlock
    @ParameterInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray

    private let channels: Int

    init(channels: Int) {
        self.channels = channels
        self._conv1 = ModuleInfo(
            wrappedValue: LTXConv3DBlock(
                inputChannels: channels,
                outputChannels: channels,
                causal: false,
                spatialPaddingMode: "zeros"
            ),
            key: "conv1"
        )
        self._conv2 = ModuleInfo(
            wrappedValue: LTXConv3DBlock(
                inputChannels: channels,
                outputChannels: channels,
                causal: false,
                spatialPaddingMode: "zeros"
            ),
            key: "conv2"
        )
        self._scaleShiftTable = ParameterInfo(
            wrappedValue: MLXArray.zeros([4, channels]),
            key: "scale_shift_table"
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray, timestepParameters: MLXArray) -> MLXArray {
        let values = scaleShiftTable[.newAxis, .newAxis, .newAxis, .newAxis, 0..., 0...]
            + timestepParameters.reshaped(input.shape[0], 1, 1, 1, 4, channels)
        let shift1 = values[0..., 0, 0, 0, 0, 0...]
        let scale1 = values[0..., 0, 0, 0, 1, 0...]
        let shift2 = values[0..., 0, 0, 0, 2, 0...]
        let scale2 = values[0..., 0, 0, 0, 3, 0...]

        var value = ltxPixelNorm(input)
        value = value * (1 + scale1) + shift1
        value = conv1(silu(value))
        value = ltxPixelNorm(value)
        value = value * (1 + scale2) + shift2
        value = conv2(silu(value))
        return value + input
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        input
    }
}

final class LTXVideo096VAEUpsample: Module, UnaryLayer {
    @ModuleInfo(key: "conv") var conv: LTXConv3DBlock

    let spatialFactor: Int
    let temporalFactor: Int

    init(inputChannels: Int, outputChannels: Int, spatialFactor: Int, temporalFactor: Int) {
        self.spatialFactor = spatialFactor
        self.temporalFactor = temporalFactor
        self._conv = ModuleInfo(
            wrappedValue: LTXConv3DBlock(
                inputChannels: inputChannels,
                outputChannels: outputChannels,
                causal: false,
                spatialPaddingMode: "zeros"
            ),
            key: "conv"
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let value = ltxPixelShuffle3D(
            conv(input),
            spatialFactor: spatialFactor,
            temporalFactor: temporalFactor
        )
        guard temporalFactor > 1 else { return value }
        return value[0..., 1..<value.shape[1], 0..., 0..., 0...]
    }
}

public final class LTXVideo096VAEDecoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: LTXConv3DBlock
    @ModuleInfo(key: "up_blocks") var upBlocks: [UnaryLayer]
    @ModuleInfo(key: "conv_out") var convOut: LTXConv3DBlock
    @ModuleInfo(key: "last_time_embedder") var lastTimeEmbedder: LTXVideo096TimestepEmbeddingContainer
    @ParameterInfo(key: "last_scale_shift_table") var lastScaleShiftTable: MLXArray
    @ParameterInfo(key: "timestep_scale_multiplier") var timestepScaleMultiplier: MLXArray

    public override init() {
        self._convIn = ModuleInfo(
            wrappedValue: LTXConv3DBlock(
                inputChannels: 128,
                outputChannels: 1024,
                causal: false,
                spatialPaddingMode: "zeros"
            ),
            key: "conv_in"
        )
        self._upBlocks = ModuleInfo(
            wrappedValue: [
                LTXVideo096VAETimeStage(channels: 1024, count: 5),
                LTXVideo096VAEUpsample(
                    inputChannels: 1024, outputChannels: 4096,
                    spatialFactor: 2, temporalFactor: 2
                ),
                LTXVideo096VAETimeStage(channels: 512, count: 5),
                LTXVideo096VAEUpsample(
                    inputChannels: 512, outputChannels: 2048,
                    spatialFactor: 2, temporalFactor: 2
                ),
                LTXVideo096VAETimeStage(channels: 256, count: 5),
                LTXVideo096VAEUpsample(
                    inputChannels: 256, outputChannels: 1024,
                    spatialFactor: 2, temporalFactor: 2
                ),
                LTXVideo096VAETimeStage(channels: 128, count: 5)
            ],
            key: "up_blocks"
        )
        self._convOut = ModuleInfo(
            wrappedValue: LTXConv3DBlock(
                inputChannels: 128,
                outputChannels: 48,
                causal: false,
                spatialPaddingMode: "zeros"
            ),
            key: "conv_out"
        )
        self._lastTimeEmbedder = ModuleInfo(
            wrappedValue: LTXVideo096TimestepEmbeddingContainer(
                inputDimension: 256,
                outputDimension: 256
            ),
            key: "last_time_embedder"
        )
        self._lastScaleShiftTable = ParameterInfo(
            wrappedValue: MLXArray.zeros([2, 128]),
            key: "last_scale_shift_table"
        )
        self._timestepScaleMultiplier = ParameterInfo(
            wrappedValue: MLXArray(1000 as Float),
            key: "timestep_scale_multiplier"
        )
        super.init()
    }

    public func decode(
        _ latent: MLXArray,
        timestep: MLXArray = MLXArray([0.05 as Float])
    ) throws -> MLXArray {
        guard latent.ndim == 5, latent.shape[1] == 128,
              latent.shape[2] > 0, latent.shape[3] > 0, latent.shape[4] > 0 else {
            throw LTXVideoRuntimeError.invalidLatentShape(latent.shape)
        }
        let batch = latent.shape[0]
        let scaled = timestep.asType(latent.dtype) * timestepScaleMultiplier.asType(latent.dtype)
        let timestepEmbedding = LTXVideo096SinusoidalEmbedding.make(
            scaled.reshaped(-1), dimension: 256
        )
        var value = convIn(latent.transposed(0, 2, 3, 4, 1))
        for block in upBlocks {
            if let stage = block as? LTXVideo096VAETimeStage {
                value = stage(value, timestepEmbedding: timestepEmbedding)
            } else {
                value = block(value)
            }
        }
        let lastEmbedding = lastTimeEmbedder(timestepEmbedding)
            .reshaped(batch, 2, 128)
        let values = lastScaleShiftTable[.newAxis, 0..., 0...]
            + lastEmbedding.asType(value.dtype)
        let shift = values[0..., 0, 0...].reshaped(batch, 1, 1, 1, 128)
        let scale = values[0..., 1, 0...].reshaped(batch, 1, 1, 1, 128)
        value = value * (1 + scale) + shift
        value = convOut(silu(ltxPixelNorm(value)))
        return ltxUnpatchifySpatial(value, patchSize: 4)
            .transposed(0, 4, 1, 2, 3)
    }
}

public enum LTXVideo096VAEWeightLoader {
    public static func load(
        model: LTXVideo096VAEDecoder,
        from weightsURL: URL,
        computeDType: LTXVideoComputeDType = .bfloat16
    ) throws -> LTXVideo096GGUFLoadReport {
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw LTXVideo096GGUFError.missingFile(weightsURL)
        }
        let source = try MLX.loadArrays(url: weightsURL)
        let expected = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1) }
        )
        var converted: [String: MLXArray] = [:]
        var sourceTypes: [String: Int] = [:]
        for (sourceKey, sourceValue) in source where sourceKey.hasPrefix("decoder.") {
            let key = String(sourceKey.dropFirst("decoder.".count))
            guard let expectedValue = expected[key] else { continue }
            var value = sourceValue
            if key.hasSuffix(".conv.weight") {
                guard value.ndim == 5 else {
                    throw LTXVideo096GGUFError.invalidTensor(key)
                }
                value = value.transposed(0, 2, 3, 4, 1)
            }
            guard value.shape == expectedValue.shape else {
                throw LTXVideo096GGUFError.weightShapeMismatch(
                    name: key,
                    expected: expectedValue.shape,
                    actual: value.shape
                )
            }
            sourceTypes[String(describing: sourceValue.dtype), default: 0] += 1
            converted[key] = value.asType(computeDType.mlxDType)
        }
        let missing = Set(expected.keys).subtracting(converted.keys).sorted()
        guard missing.isEmpty else { throw LTXVideo096GGUFError.missingWeights(missing) }
        let parameterTree = ModuleParameters.unflattened(
            converted.sorted { $0.key < $1.key }
        )
        try model.update(parameters: parameterTree, verify: .all)
        MLX.eval(model)
        return LTXVideo096GGUFLoadReport(
            sourceTensorCount: source.count,
            loadedParameterCount: converted.count,
            quantizedTensorCount: 0,
            sourceTypes: sourceTypes
        )
    }
}

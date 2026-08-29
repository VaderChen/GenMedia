import Foundation
import MLX
import MLXNN

public enum LTXRoPEType: String, Sendable, Equatable {
    case split
    case interleaved
}

public struct LTXRoPEFrequencies {
    public let cos: MLXArray
    public let sin: MLXArray
    public let type: LTXRoPEType

    public init(cos: MLXArray, sin: MLXArray, type: LTXRoPEType) {
        self.cos = cos
        self.sin = sin
        self.type = type
    }
}

public enum LTXTransformerOps {
    public static func timestepEmbedding(
        _ timesteps: MLXArray,
        dimension: Int,
        flipSinToCos: Bool = true
    ) -> MLXArray {
        let half = dimension / 2
        let exponent = -Foundation.log(10_000 as Float) * MLXArray(0..<half).asType(.float32)
            / Float(half)
        let frequencies = exp(exponent)
        let angles = timesteps[.ellipsis, .newAxis].asType(.float32) * frequencies[.newAxis, .ellipsis]
        let embedding = flipSinToCos
            ? concatenated([cos(angles), sin(angles)], axis: -1)
            : concatenated([sin(angles), cos(angles)], axis: -1)
        return dimension.isMultiple(of: 2)
            ? embedding
            : padded(embedding, widths: [.init((0, 0)), .init((0, 1))])
    }

    public static func rmsNorm(_ input: MLXArray, eps: Float) -> MLXArray {
        let weight = MLXArray.ones([input.shape.last ?? 1], dtype: input.dtype)
        return MLXFast.rmsNorm(input, weight: weight, eps: eps)
    }

    public static func layerNorm(_ input: MLXArray, eps: Float) -> MLXArray {
        MLXFast.layerNorm(input, weight: nil, bias: nil, eps: eps)
    }

    public static func generateFreqGrid(theta: Float, numPositionDimensions: Int, innerDimension: Int) -> MLXArray {
        let count = innerDimension / (2 * numPositionDimensions)
        return theta ** linspace(Float(0), Float(1), count: count) * (Float.pi / 2)
    }

    public static func generateFreqGridValues(
        theta: Float,
        numPositionDimensions: Int,
        innerDimension: Int
    ) -> [Float] {
        let count = innerDimension / (2 * numPositionDimensions)
        guard count > 0 else { return [] }
        let denominator = max(count - 1, 1)
        return (0..<count).map { index in
            let fraction = Float(index) / Float(denominator)
            return Float(Foundation.pow(Double(theta), Double(fraction))) * (Float.pi / 2)
        }
    }

    public static func computeFrequencies(
        _ freqIndices: MLXArray,
        positions: MLXArray,
        maxPositions: [Int]
    ) -> MLXArray {
        let dimensions = positions.shape.last ?? 0
        let normalized = stacked((0..<dimensions).map {
            positions[.ellipsis, $0].asType(.float32) / Float(maxPositions[$0])
        }, axis: -1)
        let scaled = freqIndices * (normalized[.ellipsis, .newAxis] * 2 - 1)
        return scaled.transposed(0, 1, 3, 2).reshaped(
            positions.shape[0], positions.shape[1], -1
        )
    }

    public static func precomputeRoPE(
        positions: MLXArray,
        numHeads: Int,
        headDimension: Int,
        theta: Float,
        maxPositions: [Int],
        type: LTXRoPEType
    ) -> LTXRoPEFrequencies {
        let innerDimension = numHeads * headDimension
        let freqIndices = generateFreqGrid(
            theta: theta,
            numPositionDimensions: positions.shape.last ?? 0,
            innerDimension: innerDimension
        )
        MLX.eval(positions, freqIndices)
        let batch = positions.shape[0]
        let tokens = positions.shape[1]
        let positionDimensions = positions.shape[2]
        let positionValues = positions.asArray(Int32.self)
        let frequencyValues = freqIndices.asArray(Float.self)
        let frequencyCount = frequencyValues.count

        func angle(batch: Int, token: Int, frequency: Int, axis: Int) -> Float {
            let positionIndex = (batch * tokens + token) * positionDimensions + axis
            let coordinate = Float(positionValues[positionIndex])
            let normalized = coordinate / Float(maxPositions[axis])
            return frequencyValues[frequency] * (normalized * 2 - 1)
        }

        if type == .interleaved {
            let rawCount = frequencyCount * positionDimensions
            let padding = innerDimension - rawCount * 2
            var cosineValues: [Float] = []
            var sineValues: [Float] = []
            cosineValues.reserveCapacity(batch * numHeads * tokens * headDimension)
            sineValues.reserveCapacity(batch * numHeads * tokens * headDimension)
            for batchIndex in 0..<batch {
                for head in 0..<numHeads {
                    for token in 0..<tokens {
                        for dimension in 0..<headDimension {
                            let flat = head * headDimension + dimension
                            if flat < padding {
                                cosineValues.append(1)
                                sineValues.append(0)
                            } else {
                                let raw = (flat - padding) / 2
                                let frequency = raw / positionDimensions
                                let axis = raw % positionDimensions
                                let value = angle(
                                    batch: batchIndex,
                                    token: token,
                                    frequency: frequency,
                                    axis: axis
                                )
                                cosineValues.append(Float(Darwin.cos(Double(value))))
                                sineValues.append(Float(Darwin.sin(Double(value))))
                            }
                        }
                    }
                }
            }
            return LTXRoPEFrequencies(
                cos: MLXArray(cosineValues).reshaped(batch, numHeads, tokens, headDimension),
                sin: MLXArray(sineValues).reshaped(batch, numHeads, tokens, headDimension),
                type: type
            )
        }

        let halfDimension = innerDimension / 2
        let padding = halfDimension - frequencyCount * positionDimensions
        let headHalfDimension = innerDimension / (2 * numHeads)
        var cosineValues: [Float] = []
        var sineValues: [Float] = []
        cosineValues.reserveCapacity(batch * numHeads * tokens * headHalfDimension)
        sineValues.reserveCapacity(batch * numHeads * tokens * headHalfDimension)
        for batchIndex in 0..<batch {
            for head in 0..<numHeads {
                for token in 0..<tokens {
                    for dimension in 0..<headHalfDimension {
                        let flat = head * headHalfDimension + dimension
                        if flat < padding {
                            cosineValues.append(1)
                            sineValues.append(0)
                        } else {
                            let raw = flat - padding
                            let frequency = raw / positionDimensions
                            let axis = raw % positionDimensions
                            let value = angle(
                                batch: batchIndex,
                                token: token,
                                frequency: frequency,
                                axis: axis
                            )
                            cosineValues.append(Float(Darwin.cos(Double(value))))
                            sineValues.append(Float(Darwin.sin(Double(value))))
                        }
                    }
                }
            }
        }
        return LTXRoPEFrequencies(
            cos: MLXArray(cosineValues).reshaped(batch, numHeads, tokens, headHalfDimension),
            sin: MLXArray(sineValues).reshaped(batch, numHeads, tokens, headHalfDimension),
            type: type
        )
    }

    public static func applyRoPE(_ input: MLXArray, frequencies: LTXRoPEFrequencies) -> MLXArray {
        let cosines = frequencies.cos.asType(input.dtype)
        let sines = frequencies.sin.asType(input.dtype)
        if frequencies.type == .split {
            let half = (input.shape.last ?? 0) / 2
            let first = input[.ellipsis, ..<half]
            let second = input[.ellipsis, half...]
            return concatenated([
                first * cosines - second * sines,
                first * sines + second * cosines
            ], axis: -1)
        }
        let pairs = input.reshaped(
            input.shape[0], input.shape[1], input.shape[2], (input.shape.last ?? 0) / 2, 2
        )
        let first = pairs[.ellipsis, 0]
        let second = pairs[.ellipsis, 1]
        let rotated = stacked([-second, first], axis: -1).reshaped(input.shape)
        return input * cosines + rotated * sines
    }
}

public final class LTXTimestepEmbedding: Module {
    @ModuleInfo(key: "linear1") public var linear1: Linear
    @ModuleInfo(key: "linear2") public var linear2: Linear

    public init(inputDimension: Int, outputDimension: Int) {
        self._linear1 = ModuleInfo(wrappedValue: Linear(inputDimension, outputDimension), key: "linear1")
        self._linear2 = ModuleInfo(wrappedValue: Linear(outputDimension, outputDimension), key: "linear2")
        super.init()
    }

    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        linear2(silu(linear1(input)))
    }
}

public final class LTXAdaLayerNormSingle: Module {
    @ModuleInfo(key: "emb") public var embedding: LTXTimestepEmbeddingContainer
    @ModuleInfo(key: "linear") public var linear: Linear
    public let parameterCount: Int

    public init(dimension: Int, parameterCount: Int, timestepDimension: Int) {
        self.parameterCount = parameterCount
        self._embedding = ModuleInfo(
            wrappedValue: LTXTimestepEmbeddingContainer(
                inputDimension: timestepDimension,
                outputDimension: dimension
            ),
            key: "emb"
        )
        self._linear = ModuleInfo(
            wrappedValue: Linear(dimension, parameterCount * dimension),
            key: "linear"
        )
        super.init()
    }

    public func callAsFunction(_ input: MLXArray) -> (parameters: MLXArray, embedded: MLXArray) {
        let embedded = embedding(input)
        return (linear(silu(embedded)), embedded)
    }
}

public final class LTXTimestepEmbeddingContainer: Module {
    @ModuleInfo(key: "timestep_embedder") public var timestepEmbedder: LTXTimestepEmbedding

    public init(inputDimension: Int, outputDimension: Int) {
        self._timestepEmbedder = ModuleInfo(
            wrappedValue: LTXTimestepEmbedding(inputDimension: inputDimension, outputDimension: outputDimension),
            key: "timestep_embedder"
        )
        super.init()
    }

    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        timestepEmbedder(input)
    }
}

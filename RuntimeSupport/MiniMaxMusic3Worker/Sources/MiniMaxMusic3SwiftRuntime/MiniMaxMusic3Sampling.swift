import Foundation
import MLX

public enum MiniMaxMusic3SamplingError: LocalizedError, Sendable {
    case invalidLogits([Int])
    case invalidAllowedVocabulary([Int], [Int])
    case invalidTopK(Int)
    case invalidRandomKey([Int])

    public var errorDescription: String? {
        switch self {
        case let .invalidLogits(shape):
            "logits 必須有非空 vocabulary 維度，實際 shape 為 \(shape)。"
        case let .invalidAllowedVocabulary(expected, actual):
            "allowed vocabulary shape 不符：預期 \(expected)，實際 \(actual)。"
        case let .invalidTopK(value):
            "top-k 必須是正整數，實際為 \(value)。"
        case let .invalidRandomKey(shape):
            "random key 必須是 MLX key，實際 shape 為 \(shape)。"
        }
    }
}

public struct MiniMaxMusic3SampleResult {
    public let sample: MLXArray
    public let nextKey: MLXArray

    public init(sample: MLXArray, nextKey: MLXArray) {
        self.sample = sample
        self.nextKey = nextKey
    }
}

public enum MiniMaxMusic3Sampling {
    public static func semanticGuidedLogits(
        logits: [[Float]],
        allowedVocabulary: [Bool],
        cfgScale: Float = 1.5,
        conditionalTopK: Int = 50
    ) throws -> [[Float]] {
        guard logits.count == 2, logits.allSatisfy({ $0.count == logits[0].count }) else {
            throw MiniMaxMusic3SamplingError.invalidLogits(
                [logits.count, logits.first?.count ?? 0]
            )
        }
        guard allowedVocabulary.count == logits[0].count else {
            throw MiniMaxMusic3SamplingError.invalidAllowedVocabulary(
                [logits[0].count],
                [allowedVocabulary.count]
            )
        }
        guard conditionalTopK > 0 else {
            throw MiniMaxMusic3SamplingError.invalidTopK(conditionalTopK)
        }

        let finiteConditional = logits[0].map(finiteLogit)
        let finiteUnconditional = logits[1].map(finiteLogit)
        let guided = zip(finiteConditional, finiteUnconditional).map {
            $1 + ($0 - $1) * cfgScale
        }
        let maskedConditional = zip(finiteConditional, allowedVocabulary).map {
            $1 ? $0 : -.infinity
        }
        let k = min(conditionalTopK, maskedConditional.count)
        let threshold = maskedConditional.sorted(by: >)[k - 1]
        let output = zip(zip(maskedConditional, guided), allowedVocabulary).map {
            $1 && $0.0 >= threshold ? $0.1 : -.infinity
        }
        return [output]
    }

    public static func sampleTopK(
        logits: MLXArray,
        key: MLXArray,
        topK: Int = 50
    ) throws -> MiniMaxMusic3SampleResult {
        guard logits.ndim >= 1, logits.shape.last ?? 0 > 0 else {
            throw MiniMaxMusic3SamplingError.invalidLogits(logits.shape)
        }
        guard topK > 0 else {
            throw MiniMaxMusic3SamplingError.invalidTopK(topK)
        }
        guard key.shape == [2] else {
            throw MiniMaxMusic3SamplingError.invalidRandomKey(key.shape)
        }

        let values = finiteLogits(logits)
        let k = min(topK, values.shape.last!)
        let threshold = top(values, k: k, axis: -1).min(axis: -1, keepDims: true)
        let filtered = MLX.where(values .< threshold, -Float.infinity, values)
        let keys = MLXRandom.split(key: key, into: 2)
        let sample = MLXRandom.categorical(filtered, axis: -1, key: keys[1])
        return MiniMaxMusic3SampleResult(sample: sample, nextKey: keys[0])
    }

    public static func semanticGuidedLogits(
        logits: MLXArray,
        allowedVocabulary: MLXArray,
        cfgScale: Float = 1.5,
        conditionalTopK: Int = 50
    ) throws -> MLXArray {
        guard logits.ndim == 2, logits.shape[0] == 2 else {
            throw MiniMaxMusic3SamplingError.invalidLogits(logits.shape)
        }
        guard allowedVocabulary.shape == [logits.shape[1]] else {
            throw MiniMaxMusic3SamplingError.invalidAllowedVocabulary(
                [logits.shape[1]], allowedVocabulary.shape
            )
        }
        guard conditionalTopK > 0 else {
            throw MiniMaxMusic3SamplingError.invalidTopK(conditionalTopK)
        }

        let values = finiteLogits(logits)
        let conditional = values[0..<1]
        let unconditional = values[1..<2]
        let guided = unconditional + (conditional - unconditional) * cfgScale
        let maskedConditional = MLX.where(
            allowedVocabulary[.newAxis, 0...], conditional, -Float.infinity
        )
        let k = min(conditionalTopK, maskedConditional.shape[1])
        let threshold = top(maskedConditional, k: k, axis: -1).min(axis: -1, keepDims: true)
        let keep = (maskedConditional .>= threshold) & allowedVocabulary[.newAxis, 0...]
        return MLX.where(keep, guided, -Float.infinity)
    }

    private static func finiteLogits(_ logits: MLXArray) -> MLXArray {
        nanToNum(logits.asType(.float32), nan: -1e9, posInf: 1e9, negInf: -1e9)
    }

    private static func finiteLogit(_ value: Float) -> Float {
        if value.isNaN { return -1e9 }
        if value == .infinity { return 1e9 }
        if value == -.infinity { return -1e9 }
        return value
    }
}

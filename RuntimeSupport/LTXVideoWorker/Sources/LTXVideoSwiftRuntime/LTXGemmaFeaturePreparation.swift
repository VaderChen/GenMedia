import Foundation
import MLX

public struct LTXGemmaPromptLayout: Sendable, Equatable {
    public let tokenIDs: [Int]
    public let attentionMask: [Int]

    public init(tokenIDs: [Int], attentionMask: [Int]) {
        self.tokenIDs = tokenIDs
        self.attentionMask = attentionMask
    }
}

public enum LTXGemmaFeaturePreparationError: LocalizedError, Sendable, Equatable {
    case invalidMaxLength(Int)
    case hiddenStateCount(expected: Int, actual: Int)
    case hiddenStateShapeMismatch(expected: [Int], actual: [Int])
    case invalidAttentionMask([Int])

    public var errorDescription: String? {
        switch self {
        case .invalidMaxLength(let value):
            return "Gemma max length 必須大於 0，實際為 \(value)。"
        case .hiddenStateCount(let expected, let actual):
            return "LTX Gemma 需要 \(expected) 層 hidden states，實際為 \(actual) 層。"
        case .hiddenStateShapeMismatch(let expected, let actual):
            return "Gemma hidden state shape 不一致：預期 \(expected)，實際 \(actual)。"
        case .invalidAttentionMask(let shape):
            return "Gemma attention mask 必須是 [B,T]，實際為 \(shape)。"
        }
    }
}

public enum LTXGemmaFeaturePreparation {
    public static let hiddenStateCount = 49

    public static func projectionShape(for hiddenStateShapes: [[Int]]) throws -> [Int] {
        guard hiddenStateShapes.count == hiddenStateCount else {
            throw LTXGemmaFeaturePreparationError.hiddenStateCount(
                expected: hiddenStateCount,
                actual: hiddenStateShapes.count
            )
        }
        guard let first = hiddenStateShapes.first, first.count == 3,
              first.allSatisfy({ $0 > 0 }) else {
            throw LTXGemmaFeaturePreparationError.hiddenStateShapeMismatch(
                expected: [0, 0, 0],
                actual: hiddenStateShapes.first ?? []
            )
        }
        guard hiddenStateShapes.dropFirst().allSatisfy({ $0 == first }) else {
            let actual = hiddenStateShapes.first(where: { $0 != first }) ?? []
            throw LTXGemmaFeaturePreparationError.hiddenStateShapeMismatch(
                expected: first,
                actual: actual
            )
        }
        return [first[0], first[1], first[2] * hiddenStateCount]
    }

    public static func leftPad(
        tokenIDs: [Int],
        maxLength: Int,
        padTokenID: Int
    ) throws -> LTXGemmaPromptLayout {
        guard maxLength > 0 else {
            throw LTXGemmaFeaturePreparationError.invalidMaxLength(maxLength)
        }
        let truncated = tokenIDs.count > maxLength
            ? Array(tokenIDs.suffix(maxLength))
            : tokenIDs
        let paddingCount = maxLength - truncated.count
        return LTXGemmaPromptLayout(
            tokenIDs: Array(repeating: padTokenID, count: paddingCount) + truncated,
            attentionMask: Array(repeating: 0, count: paddingCount)
                + Array(repeating: 1, count: truncated.count)
        )
    }

    public static func stackForProjection(
        _ allHiddenStates: [MLXArray],
        attentionMask: MLXArray? = nil
    ) throws -> MLXArray {
        guard let first = allHiddenStates.first, first.ndim == 3 else {
            throw LTXGemmaFeaturePreparationError.hiddenStateShapeMismatch(
                expected: [0, 0, 0],
                actual: allHiddenStates.first?.shape ?? []
            )
        }
        let expectedShape = first.shape
        _ = try projectionShape(for: allHiddenStates.map(\.shape))

        let encoded = stacked(allHiddenStates, axis: -1)
        let variance = mean(encoded * encoded, axis: 2, keepDims: true)
        let normalized = encoded * rsqrt(variance + 1e-6)
        let stackedStates = normalized.reshaped(
            expectedShape[0], expectedShape[1], expectedShape[2] * hiddenStateCount
        )

        guard let attentionMask else {
            return stackedStates
        }
        guard attentionMask.ndim == 2,
              attentionMask.shape[0] == expectedShape[0],
              attentionMask.shape[1] == expectedShape[1] else {
            throw LTXGemmaFeaturePreparationError.invalidAttentionMask(attentionMask.shape)
        }
        let mask = attentionMask[0..., 0..., .newAxis].asType(stackedStates.dtype)
        return stackedStates * mask
    }
}

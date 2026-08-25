import MLX
import Foundation

enum ACEStepSilenceLatentError: LocalizedError {
    case invalidShape([Int], expectedChannels: Int)
    case requestedFrames(Int, available: Int)
    case nonFiniteValues

    var errorDescription: String? {
        switch self {
        case let .invalidShape(shape, expectedChannels):
            "silence latent shape 不符：\(shape)，預期 [1, \(expectedChannels), T]"
        case let .requestedFrames(requested, available):
            "silence latent 幀數不足：要求 \(requested)，可用 \(available)"
        case .nonFiniteValues:
            "silence latent 含有 NaN 或 Infinity"
        }
    }
}

struct ACEStepSilenceLatent {
    let tensor: MLXArray
    let sourceShape: [Int]
    let availableFrames: Int

    static func load(
        from url: URL,
        expectedChannels: Int,
        frameLimit: Int? = nil,
        dtype: DType = .bfloat16
    ) throws -> ACEStepSilenceLatent {
        let source = try PyTorchZipTensorReader.loadSingleFloatTensor(from: url)
        guard source.shape.count == 3,
              source.shape[0] == 1,
              source.shape[1] == expectedChannels else {
            throw ACEStepSilenceLatentError.invalidShape(
                source.shape,
                expectedChannels: expectedChannels
            )
        }
        guard source.values.allSatisfy(\.isFinite) else {
            throw ACEStepSilenceLatentError.nonFiniteValues
        }

        let availableFrames = source.shape[2]
        let requestedFrames = frameLimit ?? availableFrames
        guard requestedFrames > 0, requestedFrames <= availableFrames else {
            throw ACEStepSilenceLatentError.requestedFrames(
                requestedFrames,
                available: availableFrames
            )
        }
        var tensor = MLXArray(source.values)
            .reshaped(source.shape)
            .transposed(0, 2, 1)
        if requestedFrames < availableFrames {
            tensor = tensor[0..., 0..<requestedFrames, 0...]
        }
        tensor = tensor.asType(dtype)
        MLX.eval(tensor)
        return ACEStepSilenceLatent(
            tensor: tensor,
            sourceShape: source.shape,
            availableFrames: availableFrames
        )
    }
}

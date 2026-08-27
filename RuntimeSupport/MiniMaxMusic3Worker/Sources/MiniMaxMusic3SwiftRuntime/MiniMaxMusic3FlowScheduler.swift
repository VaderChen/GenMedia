import Foundation
import MLX

public enum MiniMaxMusic3FlowSchedulerError: LocalizedError, Sendable {
    case invalidInferenceSteps(Int)
    case shapeMismatch([Int], [Int])
    case invalidOverlap(Int)
    case insufficientOverlap
    case invalidLatentShape([Int])

    public var errorDescription: String? {
        switch self {
        case let .invalidInferenceSteps(steps):
            "去噪步數必須是正整數，實際為 \(steps)。"
        case let .shapeMismatch(sample, velocity):
            "sample 與 velocity shape 不一致：\(sample) 與 \(velocity)。"
        case let .invalidOverlap(overlap):
            "overlap 無效：\(overlap)。"
        case .insufficientOverlap:
            "overlap 超過輸入序列可用長度。"
        case let .invalidLatentShape(shape):
            "latent 必須是至少兩維的序列，實際為 \(shape)。"
        }
    }
}

public enum MiniMaxMusic3FlowScheduler {
    public static let overlapLatentLength = 172

    public static func flowTimestepValues(numInferenceSteps: Int) throws -> [Float] {
        guard numInferenceSteps > 0 else {
            throw MiniMaxMusic3FlowSchedulerError.invalidInferenceSteps(numInferenceSteps)
        }
        return (0..<numInferenceSteps).map {
            Float($0) / Float(numInferenceSteps)
        }
    }

    public static func flowTimesteps(numInferenceSteps: Int) throws -> MLXArray {
        MLXArray(try flowTimestepValues(numInferenceSteps: numInferenceSteps))
    }

    public static func eulerStep(
        sample: [Float],
        velocity: [Float],
        numInferenceSteps: Int
    ) throws -> [Float] {
        guard sample.count == velocity.count else {
            throw MiniMaxMusic3FlowSchedulerError.shapeMismatch(
                [sample.count],
                [velocity.count]
            )
        }
        guard numInferenceSteps > 0 else {
            throw MiniMaxMusic3FlowSchedulerError.invalidInferenceSteps(numInferenceSteps)
        }
        return zip(sample, velocity).map {
            $0 + $1 / Float(numInferenceSteps)
        }
    }

    public static func eulerStep(
        sample: MLXArray,
        velocity: MLXArray,
        numInferenceSteps: Int
    ) throws -> MLXArray {
        guard sample.shape == velocity.shape else {
            throw MiniMaxMusic3FlowSchedulerError.shapeMismatch(
                sample.shape,
                velocity.shape
            )
        }
        guard numInferenceSteps > 0 else {
            throw MiniMaxMusic3FlowSchedulerError.invalidInferenceSteps(numInferenceSteps)
        }
        return (
            sample.asType(.float32) + velocity.asType(.float32) / Float(numInferenceSteps)
        ).asType(velocity.dtype)
    }

    public static func blendOverlap(
        latents: [[Float]],
        noisePrompt: [[Float]],
        previousLatent: [[Float]],
        overlap: Int,
        timestep: Float
    ) throws -> [[Float]] {
        guard overlap >= 0, overlap <= latents.count else {
            throw MiniMaxMusic3FlowSchedulerError.invalidOverlap(overlap)
        }
        guard noisePrompt.count >= overlap, previousLatent.count >= overlap else {
            throw MiniMaxMusic3FlowSchedulerError.insufficientOverlap
        }
        guard latents.allSatisfy({ $0.count == latents.first?.count }),
              noisePrompt.allSatisfy({ $0.count == latents.first?.count }),
              previousLatent.allSatisfy({ $0.count == latents.first?.count }) else {
            throw MiniMaxMusic3FlowSchedulerError.invalidLatentShape([])
        }
        guard overlap > 0 else { return latents }
        let prefix = zip(noisePrompt.prefix(overlap), previousLatent.prefix(overlap)).map {
            zip($0.0, $0.1).map {
                (1.0 - (1.0 - 1e-6) * timestep) * $0.0 + timestep * $0.1
            }
        }
        return prefix + Array(latents.dropFirst(overlap))
    }

    public static func blendOverlap(
        latents: MLXArray,
        noisePrompt: MLXArray,
        previousLatent: MLXArray,
        overlap: Int,
        timestep: MLXArray
    ) throws -> MLXArray {
        guard latents.ndim >= 2 else {
            throw MiniMaxMusic3FlowSchedulerError.invalidLatentShape(latents.shape)
        }
        guard overlap >= 0, overlap <= latents.shape[1] else {
            throw MiniMaxMusic3FlowSchedulerError.invalidOverlap(overlap)
        }
        guard overlap > 0 else { return latents }
        guard noisePrompt.shape.count > 1,
              previousLatent.shape.count > 1,
              noisePrompt.shape[1] >= overlap,
              previousLatent.shape[1] >= overlap else {
            throw MiniMaxMusic3FlowSchedulerError.insufficientOverlap
        }

        let time = timestep.asType(latents.dtype)
        let prefix = (
            (1.0 - (1.0 - 1e-6) * time) * noisePrompt[0..., 0..<overlap, 0...]
            + time * previousLatent[0..., 0..<overlap, 0...]
        )
        return MLX.concatenated(
            [prefix, latents[0..., overlap..<latents.shape[1], 0...]],
            axis: 1
        )
    }

    public static func restoreOverlap(
        latents: [[Float]],
        previousLatent: [[Float]],
        overlap: Int
    ) throws -> [[Float]] {
        guard overlap >= 0,
              overlap <= latents.count,
              previousLatent.count >= overlap else {
            throw MiniMaxMusic3FlowSchedulerError.invalidOverlap(overlap)
        }
        guard overlap > 0 else { return latents }
        return Array(previousLatent.prefix(overlap)) + Array(latents.dropFirst(overlap))
    }

    public static func restoreOverlap(
        latents: MLXArray,
        previousLatent: MLXArray,
        overlap: Int
    ) throws -> MLXArray {
        guard overlap >= 0, overlap <= latents.shape[1] else {
            throw MiniMaxMusic3FlowSchedulerError.invalidOverlap(overlap)
        }
        guard overlap > 0 else { return latents }
        guard previousLatent.ndim > 1, previousLatent.shape[1] >= overlap else {
            throw MiniMaxMusic3FlowSchedulerError.invalidOverlap(overlap)
        }
        return MLX.concatenated(
            [previousLatent[0..., 0..<overlap, 0...], latents[0..., overlap..<latents.shape[1], 0...]],
            axis: 1
        )
    }

    public static func carryWindow(_ latents: MLXArray) throws -> MLXArray {
        guard latents.ndim >= 2 else {
            throw MiniMaxMusic3FlowSchedulerError.invalidLatentShape(latents.shape)
        }
        let start = max(0, latents.shape[1] - 2 * overlapLatentLength)
        let end = max(start, latents.shape[1] - overlapLatentLength)
        return latents[0..., start..<end, 0...]
    }

    public static func carryWindow(_ latents: [[Float]]) throws -> [[Float]] {
        let start = max(0, latents.count - 2 * overlapLatentLength)
        let end = max(start, latents.count - overlapLatentLength)
        return Array(latents[start..<end])
    }
}

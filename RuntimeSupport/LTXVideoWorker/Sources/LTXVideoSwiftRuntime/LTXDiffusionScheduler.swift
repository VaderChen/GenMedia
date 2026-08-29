import Foundation
import MLX

public enum LTXDiffusionSchedulerError: LocalizedError, Equatable, Sendable {
    case invalidStepCount(Int)
    case invalidTokenCount(Int)
    case invalidShiftAnchors
    case invalidTerminal(Float)
    case invalidSigmaSchedule

    public var errorDescription: String? {
        switch self {
        case let .invalidStepCount(value):
            "LTX sigma 排程的步數必須大於 0：\(value)"
        case let .invalidTokenCount(value):
            "LTX sigma 排程的 token 數必須大於 0：\(value)"
        case .invalidShiftAnchors:
            "LTX sigma 排程的 token anchor 必須滿足 maxTokens > baseTokens。"
        case let .invalidTerminal(value):
            "LTX sigma 排程的 terminal 必須介於 0 與 1 之間：\(value)"
        case .invalidSigmaSchedule:
            "LTX sigma 排程無法建立有效的非零 sigma。"
        }
    }
}

public struct LTXDiffusionScheduler: Sendable {
    public static let distilledSigmas: [Float] = [
        1.0, 0.99375, 0.9875, 0.98125, 0.975,
        0.909375, 0.725, 0.421875, 0.0
    ]

    public static let stage2Sigmas: [Float] = [
        0.909375, 0.725, 0.421875, 0.0
    ]

    public init() {}

    public static func schedule(
        steps: Int,
        numTokens: Int = 4096,
        maxShift: Double = 2.05,
        baseShift: Double = 0.95,
        baseTokens: Int = 1024,
        maxTokens: Int = 4096,
        stretch: Bool = true,
        terminal: Double = 0.1
    ) throws -> [Float] {
        guard steps > 0 else { throw LTXDiffusionSchedulerError.invalidStepCount(steps) }
        guard numTokens > 0 else { throw LTXDiffusionSchedulerError.invalidTokenCount(numTokens) }
        guard maxTokens > baseTokens else {
            throw LTXDiffusionSchedulerError.invalidShiftAnchors
        }
        guard (0..<1).contains(terminal) else {
            throw LTXDiffusionSchedulerError.invalidTerminal(Float(terminal))
        }

        let slope = (maxShift - baseShift) / Double(maxTokens - baseTokens)
        let intercept = baseShift - slope * Double(baseTokens)
        let shift = Double(numTokens) * slope + intercept
        let exponentialShift = Foundation.exp(shift)
        var sigmas = (0...steps).map { index -> Double in
            let sigma = 1.0 - Double(index) / Double(steps)
            guard sigma > 0 else { return 0 }
            return exponentialShift / (exponentialShift + (1.0 / sigma - 1.0))
        }

        if stretch {
            guard let lastNonzero = sigmas.last(where: { $0 != 0 }) else {
                throw LTXDiffusionSchedulerError.invalidSigmaSchedule
            }
            let scaleFactor = (1.0 - lastNonzero) / (1.0 - terminal)
            guard scaleFactor != 0, scaleFactor.isFinite else {
                throw LTXDiffusionSchedulerError.invalidSigmaSchedule
            }
            for index in sigmas.indices where sigmas[index] != 0 {
                sigmas[index] = 1.0 - (1.0 - sigmas[index]) / scaleFactor
            }
        }
        return sigmas.map(Float.init)
    }

    public static func sigmaToTimestep(_ sigma: Float, dtype: DType = .bfloat16) -> MLXArray {
        MLXArray([sigma]).asType(dtype)
    }

    public static func eulerStep(
        sample: MLXArray,
        denoised: MLXArray,
        sigma: Float,
        sigmaNext: Float
    ) -> MLXArray {
        guard sigma != 0 else { return denoised }
        let direction = (sample - denoised) / sigma
        return sample + (sigmaNext - sigma) * direction
    }

    public static func eulerStep(
        sample: [Float],
        denoised: [Float],
        sigma: Float,
        sigmaNext: Float
    ) -> [Float] {
        precondition(sample.count == denoised.count)
        guard sigma != 0 else { return denoised }
        let scale = (sigmaNext - sigma) / sigma
        return zip(sample, denoised).map { current, clean in
            current + scale * (current - clean)
        }
    }

    public static func denoise(
        model: LTXX0Model,
        videoLatent: MLXArray,
        audioLatent: MLXArray,
        videoCleanLatent: MLXArray? = nil,
        audioCleanLatent: MLXArray? = nil,
        videoDenoiseMask: MLXArray? = nil,
        audioDenoiseMask: MLXArray? = nil,
        videoTextEmbeds: MLXArray? = nil,
        audioTextEmbeds: MLXArray? = nil,
        videoPositions: MLXArray? = nil,
        audioPositions: MLXArray? = nil,
        videoAttentionMask: MLXArray? = nil,
        audioAttentionMask: MLXArray? = nil,
        sigmas: [Float] = LTXDiffusionScheduler.distilledSigmas,
        progress: ((Int, Int) -> Void)? = nil
    ) throws -> (video: MLXArray, audio: MLXArray) {
        guard sigmas.count >= 2, sigmas.allSatisfy({ $0.isFinite && $0 >= 0 }) else {
            throw LTXDiffusionSchedulerError.invalidSigmaSchedule
        }
        guard videoLatent.ndim == 3, audioLatent.ndim == 3 else {
            throw LTXVideoRuntimeError.invalidConfiguration(
                "LTX 去噪 latent 必須是 [batch, tokens, channels]。"
            )
        }
        let videoClean = videoCleanLatent ?? videoLatent
        let audioClean = audioCleanLatent ?? audioLatent
        let videoMask = videoDenoiseMask ?? MLXArray.ones(
            [videoLatent.shape[0], videoLatent.shape[1], 1], dtype: videoLatent.dtype
        )
        let audioMask = audioDenoiseMask ?? MLXArray.ones(
            [audioLatent.shape[0], audioLatent.shape[1], 1], dtype: audioLatent.dtype
        )
        guard videoClean.shape == videoLatent.shape,
              audioClean.shape == audioLatent.shape,
              videoMask.shape == [videoLatent.shape[0], videoLatent.shape[1], 1],
              audioMask.shape == [audioLatent.shape[0], audioLatent.shape[1], 1] else {
            throw LTXVideoRuntimeError.invalidConfiguration(
                "LTX 去噪 clean latent 或 denoise mask shape 不一致。"
            )
        }

        var video = videoLatent
        var audio = audioLatent
        let totalSteps = sigmas.count - 1
        for step in 0..<totalSteps {
            let sigma = sigmas[step]
            let sigmaNext = sigmas[step + 1]
            let timestep = sigmaToTimestep(sigma, dtype: video.dtype)
            let prediction = model(
                videoLatent: video,
                audioLatent: audio,
                sigma: timestep,
                videoTextEmbeds: videoTextEmbeds,
                audioTextEmbeds: audioTextEmbeds,
                videoPositions: videoPositions,
                audioPositions: audioPositions,
                videoAttentionMask: videoAttentionMask,
                audioAttentionMask: audioAttentionMask
            )
            let videoDenoised = applyDenoiseMask(
                prediction.video, clean: videoClean, mask: videoMask
            )
            let audioDenoised = applyDenoiseMask(
                prediction.audio, clean: audioClean, mask: audioMask
            )
            video = eulerStep(
                sample: video, denoised: videoDenoised,
                sigma: sigma, sigmaNext: sigmaNext
            )
            audio = eulerStep(
                sample: audio, denoised: audioDenoised,
                sigma: sigma, sigmaNext: sigmaNext
            )
            MLX.eval(video, audio)
            progress?(step + 1, totalSteps)
        }
        return (video, audio)
    }

    private static func applyDenoiseMask(
        _ denoised: MLXArray,
        clean: MLXArray,
        mask: MLXArray
    ) -> MLXArray {
        let dtype = denoised.dtype
        return (denoised * mask + clean.asType(.float32) * (1 - mask)).asType(dtype)
    }
}

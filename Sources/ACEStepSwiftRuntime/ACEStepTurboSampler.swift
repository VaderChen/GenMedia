import MLX
import Foundation

enum ACEStepTurboSamplerError: LocalizedError {
    case invalidSteps(Int)
    case invalidShift(Float)
    case invalidCondition(String)
    case nonFiniteLatents

    var errorDescription: String? {
        switch self {
        case let .invalidSteps(steps):
            "Turbo sampler steps 必須為 1 至 20，實際為 \(steps)"
        case let .invalidShift(shift):
            "Turbo sampler shift 必須大於 0，實際為 \(shift)"
        case let .invalidCondition(reason):
            "Turbo sampler 條件無效：\(reason)"
        case .nonFiniteLatents:
            "Turbo sampler 產生的 latent 含有 NaN 或 Infinity"
        }
    }
}

struct ACEStepTurboSamplerResult {
    let latents: MLXArray
    let schedule: [Float]
    let elapsedSeconds: TimeInterval
}

enum ACEStepTurboSampler {
    static func generate(
        decoder: ACEStepDiTDecoder,
        configuration: ACEStepDiTConfiguration,
        encoderHiddenStates: MLXArray,
        contextLatents: MLXArray,
        seed: UInt64,
        inferenceSteps: Int,
        shift: Float,
        progress: ((Double) -> Void)? = nil
    ) throws -> ACEStepTurboSamplerResult {
        guard (1...20).contains(inferenceSteps) else {
            throw ACEStepTurboSamplerError.invalidSteps(inferenceSteps)
        }
        guard shift > 0 else {
            throw ACEStepTurboSamplerError.invalidShift(shift)
        }
        guard encoderHiddenStates.ndim == 3,
              contextLatents.ndim == 3,
              encoderHiddenStates.dim(0) == contextLatents.dim(0),
              contextLatents.dim(2)
                == configuration.inChannels - configuration.audioAcousticHiddenDim else {
            throw ACEStepTurboSamplerError.invalidCondition(
                "encoder=\(encoderHiddenStates.shape)、context=\(contextLatents.shape)"
            )
        }

        let schedule = makeSchedule(inferenceSteps: inferenceSteps, shift: shift)
        let batchSize = contextLatents.dim(0)
        var latents = MLXRandom.normal(
            [batchSize, contextLatents.dim(1), configuration.audioAcousticHiddenDim],
            dtype: .float32,
            key: MLXRandom.key(seed)
        )
        let encoder = encoderHiddenStates.asType(.float32)
        let context = contextLatents.asType(.float32)
        let cache = ACEStepCrossAttentionCache()
        let start = Date()

        for (index, currentTimestep) in schedule.enumerated() {
            try Task.checkCancellation()
            let timestep = MLXArray(
                Array(repeating: currentTimestep, count: batchSize)
            )
            let velocity = try decoder(
                hiddenStates: latents,
                timestep: timestep,
                referenceTimestep: timestep,
                encoderHiddenStates: encoder,
                contextLatents: context,
                cache: cache,
                useCache: true
            )
            if index == schedule.count - 1 {
                latents = latents - velocity * currentTimestep
            } else {
                let delta = currentTimestep - schedule[index + 1]
                latents = latents - velocity * delta
            }
            MLX.eval(latents)
            progress?(Double(index + 1) / Double(schedule.count))
        }

        let values = latents.asArray(Float.self)
        guard values.allSatisfy({ $0.isFinite }) else {
            throw ACEStepTurboSamplerError.nonFiniteLatents
        }
        return ACEStepTurboSamplerResult(
            latents: latents,
            schedule: schedule,
            elapsedSeconds: Date().timeIntervalSince(start)
        )
    }

    static func makeSchedule(inferenceSteps: Int, shift: Float) -> [Float] {
        (0..<inferenceSteps).map { index in
            let timestep = 1 - Float(index) / Float(inferenceSteps)
            guard shift != 1 else { return timestep }
            return shift * timestep / (1 + (shift - 1) * timestep)
        }
    }
}

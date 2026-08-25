import MLX
import Foundation

public struct GeneratedAudioPoCReport {
    public let latentShape: [Int]
    public let schedule: [Float]
    public let diffusionSeconds: TimeInterval
    public let wave: PCM16WaveReport
    public let outputURL: URL
}

public enum GeneratedAudioPoC {
    public static func run(
        modelRoot: URL,
        conditionURL: URL,
        outputURL: URL,
        seed: UInt64,
        inferenceSteps: Int,
        shift: Float,
        progress: ((Double) -> Void)? = nil
    ) throws -> GeneratedAudioPoCReport {
        let configuration = try ACEStepDiTConfiguration.load(
            from: modelRoot.appendingPathComponent("acestep-v15-turbo/config.json")
        )
        let condition = try MLX.loadArrays(url: conditionURL)
        guard let encoderHiddenStates = condition["encoder_hidden_states"],
              let contextLatents = condition["context_latents"] else {
            throw ACEStepTurboSamplerError.invalidCondition(
                "condition safetensors 缺少 encoder_hidden_states 或 context_latents"
            )
        }
        let sampled = try sampleLatents(
            modelRoot: modelRoot,
            configuration: configuration,
            encoderHiddenStates: encoderHiddenStates,
            contextLatents: contextLatents,
            seed: seed,
            inferenceSteps: inferenceSteps,
            shift: shift,
            progress: { progress?(0.78 * $0) }
        )
        Memory.clearCache()
        try Task.checkCancellation()
        let decoded = try VAEDecodePoC.run(
            modelRoot: modelRoot,
            outputURL: outputURL,
            latents: sampled.latents,
            progress: { progress?(0.78 + 0.22 * $0) }
        )
        progress?(1)
        return GeneratedAudioPoCReport(
            latentShape: sampled.latents.shape,
            schedule: sampled.schedule,
            diffusionSeconds: sampled.elapsedSeconds,
            wave: decoded.wave,
            outputURL: outputURL
        )
    }

    private static func sampleLatents(
        modelRoot: URL,
        configuration: ACEStepDiTConfiguration,
        encoderHiddenStates: MLXArray,
        contextLatents: MLXArray,
        seed: UInt64,
        inferenceSteps: Int,
        shift: Float,
        progress: ((Double) -> Void)?
    ) throws -> ACEStepTurboSamplerResult {
        let decoder = try ACEStepDiTDecoder(configuration: configuration)
        try ACEStepDiTWeightLoader.load(
            model: decoder,
            from: modelRoot.appendingPathComponent("acestep-v15-turbo/model.safetensors"),
            dtype: .float32
        )
        return try ACEStepTurboSampler.generate(
            decoder: decoder,
            configuration: configuration,
            encoderHiddenStates: encoderHiddenStates,
            contextLatents: contextLatents,
            seed: seed,
            inferenceSteps: inferenceSteps,
            shift: shift,
            progress: progress
        )
    }
}

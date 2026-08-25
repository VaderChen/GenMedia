import MLX
import Foundation

public struct DiTForwardPoCReport {
    public let inputShape: [Int]
    public let outputShape: [Int]
    public let meanAbsoluteValue: Float
    public let outputURL: URL
}

public enum DiTForwardPoC {
    public static func run(
        modelRoot: URL,
        conditionURL: URL,
        outputURL: URL,
        seed: UInt64
    ) throws -> DiTForwardPoCReport {
        let configuration = try ACEStepDiTConfiguration.load(
            from: modelRoot.appendingPathComponent("acestep-v15-turbo/config.json")
        )
        let condition = try MLX.loadArrays(url: conditionURL)
        guard let encoderHiddenStates = condition["encoder_hidden_states"],
              let contextLatents = condition["context_latents"] else {
            throw ACEStepDiTError.invalidInput(
                "condition safetensors 缺少 encoder_hidden_states 或 context_latents"
            )
        }
        let batchSize = contextLatents.dim(0)
        let frameCount = contextLatents.dim(1)
        let noise = MLXRandom.normal(
            [batchSize, frameCount, configuration.audioAcousticHiddenDim],
            dtype: .float32,
            key: MLXRandom.key(seed)
        )
        let timestep = MLXArray.ones([batchSize], dtype: .float32)

        let decoder = try ACEStepDiTDecoder(configuration: configuration)
        try ACEStepDiTWeightLoader.load(
            model: decoder,
            from: modelRoot.appendingPathComponent("acestep-v15-turbo/model.safetensors"),
            dtype: .float32
        )
        let output = try decoder(
            hiddenStates: noise,
            timestep: timestep,
            referenceTimestep: timestep,
            encoderHiddenStates: encoderHiddenStates.asType(.float32),
            contextLatents: contextLatents.asType(.float32)
        )
        MLX.eval(noise, output)
        let values = output.asType(.float32).asArray(Float.self)
        guard values.allSatisfy({ $0.isFinite }) else {
            throw ACEStepDiTError.nonFiniteOutput
        }
        let meanAbsoluteValue = values.reduce(Float(0)) { $0 + abs($1) }
            / Float(max(1, values.count))

        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try MLX.save(
            arrays: [
                "noise": noise,
                "velocity": output
            ],
            url: outputURL
        )
        return DiTForwardPoCReport(
            inputShape: noise.shape,
            outputShape: output.shape,
            meanAbsoluteValue: meanAbsoluteValue,
            outputURL: outputURL
        )
    }
}

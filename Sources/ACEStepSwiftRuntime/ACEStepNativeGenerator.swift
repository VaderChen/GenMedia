import MLX
import Foundation

public struct ACEStepNativeGenerationReport: Sendable {
    public let conditionFrames: Int
    public let latentShape: [Int]
    public let diffusionSeconds: TimeInterval
    public let wave: PCM16WaveReport
    public let outputURL: URL
}

public enum ACEStepNativeGeneratorError: LocalizedError, Sendable {
    case invalidDuration(Int)
    case invalidSteps(Int)
    case invalidShift(Float)
    case incompleteModel(URL, [String])

    public var errorDescription: String? {
        switch self {
        case let .invalidDuration(seconds):
            "ACE-Step 音樂長度必須介於 10 到 300 秒，目前為 \(seconds) 秒。"
        case let .invalidSteps(steps):
            "ACE-Step Turbo 推論步數必須介於 1 到 20，目前為 \(steps)。"
        case let .invalidShift(shift):
            "ACE-Step Turbo timestep shift 必須大於 0，目前為 \(shift)。"
        case let .incompleteModel(url, missingPaths):
            "ACE-Step 1.5 模型不完整：\(url.path)；缺少 \(missingPaths.joined(separator: "、"))"
        }
    }
}

public enum ACEStepNativeGenerator {
    public static func generate(
        modelRoot: URL,
        prompt: String,
        lyrics: String,
        language: String = "en",
        durationSeconds: Int,
        inferenceSteps: Int,
        seed: UInt64,
        shift: Float = 3,
        outputURL: URL,
        progress: @escaping @Sendable (Double) -> Void
    ) throws -> ACEStepNativeGenerationReport {
        guard (10...300).contains(durationSeconds) else {
            throw ACEStepNativeGeneratorError.invalidDuration(durationSeconds)
        }
        guard (1...20).contains(inferenceSteps) else {
            throw ACEStepNativeGeneratorError.invalidSteps(inferenceSteps)
        }
        guard shift > 0 else {
            throw ACEStepNativeGeneratorError.invalidShift(shift)
        }
        try validateModel(at: modelRoot)
        try Task.checkCancellation()

        let vaeConfiguration = try OobleckVAEConfiguration.load(
            from: modelRoot.appendingPathComponent("vae/config.json")
        )
        let framesPerSecond = Double(vaeConfiguration.samplingRate)
            / Double(vaeConfiguration.hopLength)
        let conditionFrames = max(
            128,
            Int(ceil(Double(durationSeconds) * framesPerSecond))
        )
        let temporaryDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("ace-step-native-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: temporaryDirectory,
            withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: temporaryDirectory) }
        let conditionURL = temporaryDirectory.appendingPathComponent("condition.safetensors")

        progress(0.02)
        _ = try ACEStepConditioningStage.run(
            modelRoot: modelRoot,
            prompt: prompt,
            lyrics: lyrics,
            language: language,
            conditionFrames: conditionFrames,
            embeddingOutputURL: nil,
            outputURL: conditionURL
        )
        Memory.clearCache()
        try Task.checkCancellation()
        progress(0.30)

        let generated = try ACEStepAudioGenerationStage.run(
            modelRoot: modelRoot,
            conditionURL: conditionURL,
            outputURL: outputURL,
            seed: seed,
            inferenceSteps: inferenceSteps,
            shift: shift,
            progress: { progress(0.30 + 0.70 * $0) }
        )
        progress(1)
        return ACEStepNativeGenerationReport(
            conditionFrames: conditionFrames,
            latentShape: generated.latentShape,
            diffusionSeconds: generated.diffusionSeconds,
            wave: generated.wave,
            outputURL: outputURL
        )
    }

    private static func validateModel(at modelRoot: URL) throws {
        let requiredPaths = [
            "config.json",
            "Qwen3-Embedding-0.6B/config.json",
            "Qwen3-Embedding-0.6B/model.safetensors",
            "Qwen3-Embedding-0.6B/tokenizer.json",
            "acestep-v15-turbo/config.json",
            "acestep-v15-turbo/model.safetensors",
            "acestep-v15-turbo/silence_latent.pt",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors"
        ]
        let missingPaths = requiredPaths.filter {
            !FileManager.default.fileExists(
                atPath: modelRoot.appendingPathComponent($0).path
            )
        }
        guard missingPaths.isEmpty else {
            throw ACEStepNativeGeneratorError.incompleteModel(modelRoot, missingPaths)
        }
    }
}

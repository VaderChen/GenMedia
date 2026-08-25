// ACE-Step 原生管線的條件編碼階段：Prompt／歌詞 → Qwen 文字嵌入 → ACE-Step condition encoder。
//
// 這個檔案原本叫 ConditioningPoC —— 名字說是實驗，實際上音樂生成的正式路徑就是走這裡
// （ACEStepMusicGenerationService → ACEStepNativeGenerator → 這裡）。名字不改，任何一次清理
// 都可能把它當成實驗殘留刪掉。

import MLX
import Foundation

enum ACEStepConditioningError: LocalizedError {
    case nonFiniteOutput(String)

    var errorDescription: String? {
        switch self {
        case let .nonFiniteOutput(name):
            "\(name) 含有 NaN 或 Infinity"
        }
    }
}

public struct ACEStepConditioningReport {
    public let textTokenCount: Int
    public let lyricTokenCount: Int
    public let sourceSilenceShape: [Int]
    public let availableSilenceFrames: Int
    public let encoderHiddenShape: [Int]
    public let encoderMaskShape: [Int]
    public let contextLatentShape: [Int]
    public let nullConditionShape: [Int]
    public let encoderMeanAbsoluteValue: Float
    public let outputURL: URL
}

public enum ACEStepConditioningStage {
    public static func run(
        modelRoot: URL,
        prompt: String,
        lyrics: String,
        language: String,
        conditionFrames: Int,
        embeddingOutputURL: URL?,
        outputURL: URL
    ) throws -> ACEStepConditioningReport {
        let configuration = try ACEStepDiTConfiguration.load(
            from: modelRoot.appendingPathComponent("acestep-v15-turbo/config.json")
        )
        let embeddings = try ACEStepTextEmbedder.encode(
            modelRoot: modelRoot,
            prompt: prompt,
            lyrics: lyrics,
            language: language
        )
        if let embeddingOutputURL {
            try ACEStepTextEmbedder.save(embeddings, to: embeddingOutputURL)
        }

        let requiredSilenceFrames = max(configuration.timbreFixFrame, conditionFrames)
        let silence = try ACEStepSilenceLatent.load(
            from: modelRoot.appendingPathComponent("acestep-v15-turbo/silence_latent.pt"),
            expectedChannels: configuration.timbreHiddenDim,
            frameLimit: requiredSilenceFrames
        )
        let timbreInput = silence.tensor[0..., 0..<configuration.timbreFixFrame, 0...]
        let sourceLatents = silence.tensor[0..., 0..<conditionFrames, 0...]

        let encoder = try ACEStepConditionEncoder(configuration: configuration)
        let nullCondition = try ACEStepConditionWeightLoader.load(
            model: encoder,
            from: modelRoot.appendingPathComponent("acestep-v15-turbo/model.safetensors")
        )
        let encoded = try encoder.encode(
            textHiddenStates: embeddings.textHiddenStates,
            textAttentionMask: embeddings.textAttentionMask,
            lyricHiddenStates: embeddings.lyricHiddenStates,
            lyricAttentionMask: embeddings.lyricAttentionMask,
            timbreHiddenStates: timbreInput,
            timbreReferenceOrder: MLXArray([Int32(0)])
        )
        let chunkMask = MLXArray.ones(
            [1, conditionFrames, configuration.timbreHiddenDim],
            dtype: sourceLatents.dtype
        )
        let contextLatents = MLX.concatenated([sourceLatents, chunkMask], axis: -1)
        MLX.eval(encoded.hiddenStates, encoded.attentionMask, contextLatents, nullCondition)

        let encoderValues = encoded.hiddenStates.asType(.float32).asArray(Float.self)
        guard encoderValues.allSatisfy(\.isFinite) else {
            throw ACEStepConditioningError.nonFiniteOutput("encoder_hidden_states")
        }
        let contextValues = contextLatents.asType(.float32).asArray(Float.self)
        guard contextValues.allSatisfy(\.isFinite) else {
            throw ACEStepConditioningError.nonFiniteOutput("context_latents")
        }
        let meanAbsoluteValue = encoderValues.reduce(Float(0)) { $0 + abs($1) }
            / Float(max(1, encoderValues.count))

        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try MLX.save(
            arrays: [
                "encoder_hidden_states": encoded.hiddenStates,
                "encoder_attention_mask": encoded.attentionMask,
                "context_latents": contextLatents,
                "null_condition_emb": nullCondition
            ],
            url: outputURL
        )
        return ACEStepConditioningReport(
            textTokenCount: embeddings.textTokenCount,
            lyricTokenCount: embeddings.lyricTokenCount,
            sourceSilenceShape: silence.sourceShape,
            availableSilenceFrames: silence.availableFrames,
            encoderHiddenShape: encoded.hiddenStates.shape,
            encoderMaskShape: encoded.attentionMask.shape,
            contextLatentShape: contextLatents.shape,
            nullConditionShape: nullCondition.shape,
            encoderMeanAbsoluteValue: meanAbsoluteValue,
            outputURL: outputURL
        )
    }
}

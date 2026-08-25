// ACE-Step 條件編碼所需的 Qwen 文字嵌入。由條件編碼階段呼叫，屬於正式路徑。

import MLX
import MLXNN
import ZImage
import Foundation

struct QwenEmbeddingConfiguration: Decodable {
    let vocabSize: Int
    let hiddenSize: Int
    let numHiddenLayers: Int
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let intermediateSize: Int
    let ropeTheta: Float
    let maxPositionEmbeddings: Int
    let rmsNormEps: Float
    let headDim: Int

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case intermediateSize = "intermediate_size"
        case ropeTheta = "rope_theta"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case headDim = "head_dim"
    }

    static func load(from url: URL) throws -> QwenEmbeddingConfiguration {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(QwenEmbeddingConfiguration.self, from: data)
    }

    var mlxConfiguration: QwenTextEncoderConfiguration {
        QwenTextEncoderConfiguration(
            vocabSize: vocabSize,
            hiddenSize: hiddenSize,
            numHiddenLayers: numHiddenLayers,
            numAttentionHeads: numAttentionHeads,
            numKeyValueHeads: numKeyValueHeads,
            intermediateSize: intermediateSize,
            ropeTheta: ropeTheta,
            maxPositionEmbeddings: maxPositionEmbeddings,
            rmsNormEps: rmsNormEps,
            promptDropIndex: 0,
            headDim: headDim
        )
    }
}

enum ACEStepTextEmbeddingError: LocalizedError {
    case emptyTokens(String)
    case parameterMismatch(missing: [String], unexpected: [String])
    case nonFiniteOutput

    var errorDescription: String? {
        switch self {
        case let .emptyTokens(kind):
            "\(kind) tokenizer 沒有產生有效 Token。"
        case let .parameterMismatch(missing, unexpected):
            "Qwen 權重與 Swift 模型不一致。缺少：\(missing)，多餘：\(unexpected)"
        case .nonFiniteOutput:
            "Qwen Embedding 輸出含有 NaN 或 Infinity。"
        }
    }
}

public struct ACEStepTextEmbeddingReport {
    public let promptText: String
    public let lyricsText: String
    public let textTokenCount: Int
    public let lyricTokenCount: Int
    public let tokenPreview: [Int32]
    public let textHiddenShape: [Int]
    public let lyricHiddenShape: [Int]
    public let textMeanAbsoluteValue: Float
    public let outputURL: URL?
}

struct QwenConditionEmbeddings {
    let textHiddenStates: MLXArray
    let textAttentionMask: MLXArray
    let lyricHiddenStates: MLXArray
    let lyricAttentionMask: MLXArray
    let promptText: String
    let lyricsText: String
    let textTokenCount: Int
    let lyricTokenCount: Int
    let tokenPreview: [Int32]
    let textMeanAbsoluteValue: Float
}

public enum ACEStepTextEmbedder {
    public static func run(
        modelRoot: URL,
        prompt: String,
        lyrics: String,
        language: String,
        outputURL: URL?
    ) throws -> ACEStepTextEmbeddingReport {
        let embeddings = try encode(
            modelRoot: modelRoot,
            prompt: prompt,
            lyrics: lyrics,
            language: language
        )
        if let outputURL {
            try save(embeddings, to: outputURL)
        }
        return ACEStepTextEmbeddingReport(
            promptText: embeddings.promptText,
            lyricsText: embeddings.lyricsText,
            textTokenCount: embeddings.textTokenCount,
            lyricTokenCount: embeddings.lyricTokenCount,
            tokenPreview: embeddings.tokenPreview,
            textHiddenShape: embeddings.textHiddenStates.shape,
            lyricHiddenShape: embeddings.lyricHiddenStates.shape,
            textMeanAbsoluteValue: embeddings.textMeanAbsoluteValue,
            outputURL: outputURL
        )
    }

    static func encode(
        modelRoot: URL,
        prompt: String,
        lyrics: String,
        language: String
    ) throws -> QwenConditionEmbeddings {
        let embeddingRoot = modelRoot.appendingPathComponent(
            "Qwen3-Embedding-0.6B",
            isDirectory: true
        )
        let configuration = try QwenEmbeddingConfiguration.load(
            from: embeddingRoot.appendingPathComponent("config.json")
        )
        let tokenizer = try QwenTokenizer.load(
            from: embeddingRoot,
            maxLengthOverride: 2_048
        )
        let promptText = formattedPrompt(prompt)
        let lyricsText = formattedLyrics(lyrics, language: language)
        let textBatch = try compact(
            tokenizer.encodePlain(prompts: [promptText], maxLength: 256),
            kind: "Prompt"
        )
        let lyricBatch = try compact(
            tokenizer.encodePlain(prompts: [lyricsText], maxLength: 2_048),
            kind: "歌詞"
        )

        let encoder = QwenEncoder(configuration: configuration.mlxConfiguration)
        try loadWeights(
            into: encoder,
            from: embeddingRoot.appendingPathComponent("model.safetensors")
        )
        let textHiddenStates = encoder.forward(
            inputIds: textBatch.inputIds,
            attentionMask: textBatch.attentionMask
        ).lastHiddenState
        let lyricHiddenStates = encoder.embed(inputIds: lyricBatch.inputIds)
        MLX.eval(textHiddenStates, lyricHiddenStates)

        let textValues = textHiddenStates.asType(.float32).asArray(Float.self)
        guard textValues.allSatisfy(\.isFinite) else {
            throw ACEStepTextEmbeddingError.nonFiniteOutput
        }
        let meanAbsoluteValue = textValues.reduce(Float(0)) { $0 + abs($1) }
            / Float(max(1, textValues.count))

        return QwenConditionEmbeddings(
            textHiddenStates: textHiddenStates,
            textAttentionMask: textBatch.attentionMask,
            lyricHiddenStates: lyricHiddenStates,
            lyricAttentionMask: lyricBatch.attentionMask,
            promptText: promptText,
            lyricsText: lyricsText,
            textTokenCount: textBatch.validLength,
            lyricTokenCount: lyricBatch.validLength,
            tokenPreview: Array(
                textBatch.inputIds.asType(.int32).asArray(Int32.self).prefix(12)
            ),
            textMeanAbsoluteValue: meanAbsoluteValue
        )
    }

    static func save(_ embeddings: QwenConditionEmbeddings, to outputURL: URL) throws {
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try MLX.save(
            arrays: [
                "text_hidden_states": embeddings.textHiddenStates,
                "text_attention_mask": embeddings.textAttentionMask,
                "lyric_hidden_states": embeddings.lyricHiddenStates,
                "lyric_attention_mask": embeddings.lyricAttentionMask
            ],
            url: outputURL
        )
    }

    private struct CompactTokenBatch {
        let inputIds: MLXArray
        let attentionMask: MLXArray
        let validLength: Int
    }

    private static func compact(
        _ batch: QwenTokenBatch,
        kind: String
    ) throws -> CompactTokenBatch {
        let maskValues = batch.attentionMask.asType(.int32).asArray(Int32.self)
        let validLength = maskValues.reduce(0) { $0 + Int($1) }
        guard validLength > 0 else {
            throw ACEStepTextEmbeddingError.emptyTokens(kind)
        }
        return CompactTokenBatch(
            inputIds: batch.inputIds[0..., 0..<validLength],
            attentionMask: batch.attentionMask[0..., 0..<validLength],
            validLength: validLength
        )
    }

    private static func loadWeights(
        into encoder: QwenEncoder,
        from url: URL
    ) throws {
        let source = try MLX.loadArrays(url: url)
        let expectedKeys = Set(encoder.parameters().flattened().map(\.0))
        let sourceKeys = Set(source.keys)
        let missing = expectedKeys.subtracting(sourceKeys).sorted()
        let unexpected = sourceKeys.subtracting(expectedKeys).sorted()
        guard missing.isEmpty, unexpected.isEmpty else {
            throw ACEStepTextEmbeddingError.parameterMismatch(
                missing: missing,
                unexpected: unexpected
            )
        }
        try encoder.update(
            parameters: ModuleParameters.unflattened(source),
            verify: .all
        )
        MLX.eval(encoder)
    }

    private static func formattedPrompt(_ prompt: String) -> String {
        """
        # Instruction
        Fill the audio semantic mask based on the given conditions:

        # Caption
        \(prompt)

        # Metas
        {}<|endoftext|>
        """
    }

    private static func formattedLyrics(_ lyrics: String, language: String) -> String {
        """
        # Languages
        \(language)

        # Lyric
        \(lyrics)<|endoftext|>
        """
    }
}

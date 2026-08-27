import Foundation
import MLX

// MiniMax Music 3 自迴歸取樣的 language logits 必須使用 float32。
// bfloat16 的單步差異會在 greedy top-k=1 時造成 token 分岔，float32 可維持 parity。
public enum MiniMaxMusic3LanguageInputDType: String, Equatable, Sendable {
    case bfloat16
    case float32

    fileprivate var mlxDType: DType {
        switch self {
        case .bfloat16:
            .bfloat16
        case .float32:
            .float32
        }
    }
}

public struct MiniMaxMusic3ModelConfiguration: Equatable, Sendable {
    public let frameRate: Float
    public let semanticVocabularySize: Int
    public let audioVocabularySize: Int
    public let audioCodeOffset: Int
    public let audioCFGTokenID: Int
    public let audioEndTokenID: Int
    public let maxPromptTokens: Int
    public let maxAudioFrames: Int
    public let numberOfCodebooks: Int

    public init(
        frameRate: Float = 25,
        semanticVocabularySize: Int = 16_384,
        audioVocabularySize: Int = 1_024,
        audioCodeOffset: Int = 151_675,
        audioCFGTokenID: Int = 151_654,
        audioEndTokenID: Int = 151_670,
        maxPromptTokens: Int = 5_000,
        maxAudioFrames: Int = 9_000,
        numberOfCodebooks: Int = 8
    ) {
        self.frameRate = frameRate
        self.semanticVocabularySize = semanticVocabularySize
        self.audioVocabularySize = audioVocabularySize
        self.audioCodeOffset = audioCodeOffset
        self.audioCFGTokenID = audioCFGTokenID
        self.audioEndTokenID = audioEndTokenID
        self.maxPromptTokens = maxPromptTokens
        self.maxAudioFrames = maxAudioFrames
        self.numberOfCodebooks = numberOfCodebooks
    }

    public static let music3 = Self()

    public static func load(from modelDirectory: URL) throws -> Self {
        let url = modelDirectory.appendingPathComponent("config.json")
        do {
            let data = try Data(contentsOf: url)
            let source = try JSONDecoder().decode(ModelConfigurationFile.self, from: data)
            return Self(
                frameRate: source.frameRate,
                semanticVocabularySize: source.semanticVocabularySize,
                audioVocabularySize: source.audioVocabularySize,
                audioCodeOffset: source.audioCodeOffset,
                audioCFGTokenID: source.audioCFGTokenID,
                audioEndTokenID: source.audioEndTokenID,
                maxPromptTokens: source.maxPromptTokens,
                maxAudioFrames: source.maxAudioFrames,
                numberOfCodebooks: source.numberOfCodebooks
            )
        } catch {
            throw MiniMaxMusic3AutoregressiveError.invalidConfiguration(url)
        }
    }

    public func validate() throws {
        guard frameRate > 0,
              frameRate.isFinite,
              semanticVocabularySize > 0,
              audioVocabularySize > 0,
              audioCodeOffset >= 0,
              audioEndTokenID >= 0,
              maxPromptTokens > 0,
              maxAudioFrames > 0,
              numberOfCodebooks > 1 else {
            throw MiniMaxMusic3AutoregressiveError.invalidConfiguration(
                URL(fileURLWithPath: "model configuration")
            )
        }
    }

    private struct ModelConfigurationFile: Decodable {
        let frameRate: Float
        let semanticVocabularySize: Int
        let audioVocabularySize: Int
        let audioCodeOffset: Int
        let audioCFGTokenID: Int
        let audioEndTokenID: Int
        let maxPromptTokens: Int
        let maxAudioFrames: Int
        let numberOfCodebooks: Int

        enum CodingKeys: String, CodingKey {
            case frameRate = "frame_rate"
            case semanticVocabularySize = "semantic_vocab_size"
            case audioVocabularySize = "audio_vocab_size"
            case audioCodeOffset = "audio_code_offset"
            case audioCFGTokenID = "audio_cfg_token_id"
            case audioEndTokenID = "audio_end_token_id"
            case maxPromptTokens = "max_prompt_tokens"
            case maxAudioFrames = "max_audio_frames"
            case numberOfCodebooks = "num_codebooks"
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            frameRate = try container.decodeIfPresent(Float.self, forKey: .frameRate) ?? 25
            semanticVocabularySize = try container.decodeIfPresent(
                Int.self,
                forKey: .semanticVocabularySize
            ) ?? 16_384
            audioVocabularySize = try container.decodeIfPresent(
                Int.self,
                forKey: .audioVocabularySize
            ) ?? 1_024
            audioCodeOffset = try container.decodeIfPresent(
                Int.self,
                forKey: .audioCodeOffset
            ) ?? 151_675
            audioCFGTokenID = try container.decodeIfPresent(
                Int.self,
                forKey: .audioCFGTokenID
            ) ?? 151_654
            audioEndTokenID = try container.decodeIfPresent(
                Int.self,
                forKey: .audioEndTokenID
            ) ?? 151_670
            maxPromptTokens = try container.decodeIfPresent(
                Int.self,
                forKey: .maxPromptTokens
            ) ?? 5_000
            maxAudioFrames = try container.decodeIfPresent(
                Int.self,
                forKey: .maxAudioFrames
            ) ?? 9_000
            numberOfCodebooks = try container.decodeIfPresent(
                Int.self,
                forKey: .numberOfCodebooks
            ) ?? 8
        }
    }
}

public struct MiniMaxMusic3GenerationConfiguration: Equatable, Sendable {
    public let audioDuration: Float
    public let seed: Int
    public let numInferenceSteps: Int
    public let arCFGScale: Float
    public let flowCFGScale: Float
    public let topK: Int
    public let inputDType: MiniMaxMusic3LanguageInputDType

    public init(
        audioDuration: Float = 60,
        seed: Int = 0,
        numInferenceSteps: Int = 30,
        arCFGScale: Float = 1.5,
        flowCFGScale: Float = 1.7,
        topK: Int = 50,
        inputDType: MiniMaxMusic3LanguageInputDType = .bfloat16
    ) {
        self.audioDuration = audioDuration
        self.seed = seed
        self.numInferenceSteps = numInferenceSteps
        self.arCFGScale = arCFGScale
        self.flowCFGScale = flowCFGScale
        self.topK = topK
        self.inputDType = inputDType
    }

    public func maxFrames(using model: MiniMaxMusic3ModelConfiguration) throws -> Int {
        guard audioDuration > 0, audioDuration.isFinite else {
            throw MiniMaxMusic3AutoregressiveError.invalidGeneration(
                "audio duration 必須是正數。"
            )
        }
        let frames = Int(audioDuration * model.frameRate)
        guard frames > 0 else {
            throw MiniMaxMusic3AutoregressiveError.invalidGeneration(
                "audio duration 短於一個 audio frame。"
            )
        }
        return min(frames, model.maxAudioFrames)
    }

    public func validate() throws {
        guard numInferenceSteps > 0,
              arCFGScale.isFinite,
              flowCFGScale.isFinite,
              topK > 0 else {
            throw MiniMaxMusic3AutoregressiveError.invalidGeneration(
                "去噪步數、CFG scale 必須有效且 top-k 必須是正整數。"
            )
        }
    }
}

public enum MiniMaxMusic3AutoregressiveError: LocalizedError, Sendable {
    case invalidConfiguration(URL)
    case invalidGeneration(String)
    case invalidInput(String)
    case zeroFrames

    public var errorDescription: String? {
        switch self {
        case let .invalidConfiguration(url):
            "無法解析 MiniMax Music 3 設定：\(url.path)"
        case let .invalidGeneration(message):
            "MiniMax Music 3 自迴歸設定無效：\(message)"
        case let .invalidInput(message):
            "MiniMax Music 3 自迴歸輸入無效：\(message)"
        case .zeroFrames:
            "MiniMax Music 3 沒有生成任何 audio frame。"
        }
    }
}

public struct MiniMaxMusic3DepthGenerationResult {
    public let frameCodes: MLXArray
    public let hiddenStates: MLXArray
    public let guidedLogits: MLXArray
    public let nextKey: MLXArray

    public init(
        frameCodes: MLXArray,
        hiddenStates: MLXArray,
        guidedLogits: MLXArray,
        nextKey: MLXArray
    ) {
        self.frameCodes = frameCodes
        self.hiddenStates = hiddenStates
        self.guidedLogits = guidedLogits
        self.nextKey = nextKey
    }
}

public func miniMaxMusic3GenerateDepthCodes(
    languageModel: MiniMaxMusic3LanguageModel,
    decoder: MiniMaxMusic3RVQDepthDecoder,
    lastHidden: MLXArray,
    semanticCode: MLXArray,
    key: MLXArray,
    modelConfiguration: MiniMaxMusic3ModelConfiguration = .music3,
    arCFGScale: Float = 1.5,
    topK: Int = 50,
    inputDType: MiniMaxMusic3LanguageInputDType = .bfloat16
) throws -> MiniMaxMusic3DepthGenerationResult {
    guard lastHidden.shape == [2, decoder.configuration.hiddenSize],
          semanticCode.shape == [2] else {
        throw MiniMaxMusic3AutoregressiveError.invalidInput(
            "last hidden 必須是 [2, hidden]，semantic code 必須是 [2]。"
        )
    }
    guard modelConfiguration.numberOfCodebooks == decoder.configuration.numberOfCodebooks else {
        throw MiniMaxMusic3AutoregressiveError.invalidInput(
            "model 與 RVQ decoder 的 codebook 數量不一致。"
        )
    }

    let semanticEmbeddings = try languageModel.tokenEmbeddings(
        for: semanticCode + Int32(modelConfiguration.audioCodeOffset)
    ).asType(inputDType.mlxDType).reshaped(2, 1, decoder.configuration.hiddenSize)
    var sequence = [
        try decoder.project(lastHidden)[0..., .newAxis, 0...],
        try decoder.project(semanticEmbeddings)
    ]
    var codes = [semanticCode]
    var hiddenParts: [MLXArray] = []
    var guidedLogits: [MLXArray] = []
    var nextKey = key

    for index in 1..<decoder.configuration.numberOfCodebooks {
        let depthInput = MLX.concatenated(sequence, axis: 1)
        let hidden = try decoder(depthInput)
        let lastDepthHidden = hidden[0..., -1, 0...]
        hiddenParts.append(lastDepthHidden[0..<1, 0...])
        let logits = try decoder.logits(for: hidden)
        let depthLogits = logits[index - 1][0..., -1, 0...]
        let conditional = depthLogits[0..<1, 0...]
        let unconditional = depthLogits[1..<2, 0...]
        let guided = unconditional + (conditional - unconditional) * arCFGScale
        guidedLogits.append(guided[0])
        let sampled = try MiniMaxMusic3Sampling.sampleTopK(
            logits: guided,
            key: nextKey,
            topK: topK
        )
        nextKey = sampled.nextKey
        let code = MLX.repeated(sampled.sample, count: 2, axis: 0).asType(.int32)
        codes.append(code)
        if index < decoder.configuration.numberOfCodebooks - 1 {
            let embedding = try decoder.residualEmbedding(
                for: code,
                codebookIndex: index - 1
            )
            sequence.append(try decoder.project(embedding)[0..., .newAxis, 0...])
        }
    }

    return MiniMaxMusic3DepthGenerationResult(
        frameCodes: MLX.stacked(codes, axis: 1),
        hiddenStates: MLX.concatenated(hiddenParts, axis: 1),
        guidedLogits: MLX.stacked(guidedLogits, axis: 0),
        nextKey: nextKey
    )
}

public struct MiniMaxMusic3FrameHiddenGenerationResult {
    public let initialHidden: MLXArray
    public let frameHiddens: MLXArray
    public let semanticCodes: MLXArray
    public let frameCodes: MLXArray
    public let depthGuidedLogits: MLXArray
    public let nextKey: MLXArray
    public let iterations: Int
    public let stoppedByEndToken: Bool

    public init(
        initialHidden: MLXArray,
        frameHiddens: MLXArray,
        semanticCodes: MLXArray,
        frameCodes: MLXArray,
        depthGuidedLogits: MLXArray,
        nextKey: MLXArray,
        iterations: Int,
        stoppedByEndToken: Bool
    ) {
        self.initialHidden = initialHidden
        self.frameHiddens = frameHiddens
        self.semanticCodes = semanticCodes
        self.frameCodes = frameCodes
        self.depthGuidedLogits = depthGuidedLogits
        self.nextKey = nextKey
        self.iterations = iterations
        self.stoppedByEndToken = stoppedByEndToken
    }
}

public enum MiniMaxMusic3AutoregressiveGenerator {
    public static func generateFrameHiddens(
        textIDs: MLXArray,
        languageModel: MiniMaxMusic3LanguageModel,
        decoder: MiniMaxMusic3RVQDepthDecoder,
        modelConfiguration: MiniMaxMusic3ModelConfiguration = .music3,
        generation: MiniMaxMusic3GenerationConfiguration = .init(),
        key: MLXArray,
        inputDType: MiniMaxMusic3LanguageInputDType = .bfloat16,
        progress: (@Sendable (_ completedFrames: Int, _ maximumFrames: Int) -> Void)? = nil
    ) throws -> MiniMaxMusic3FrameHiddenGenerationResult {
        try modelConfiguration.validate()
        try generation.validate()
        guard textIDs.ndim == 2,
              textIDs.shape[0] == 2,
              textIDs.shape[1] > 0 else {
            throw MiniMaxMusic3AutoregressiveError.invalidInput(
                "text IDs 必須是 [2, sequence]。"
            )
        }
        guard languageModel.hiddenSize == decoder.configuration.hiddenSize else {
            throw MiniMaxMusic3AutoregressiveError.invalidInput(
                "language model 與 RVQ decoder hidden size 不一致。"
            )
        }
        let maxFrames = try generation.maxFrames(using: modelConfiguration)
        let cache = languageModel.newCache()
        let initialHidden: MLXArray
        if inputDType == .float32 {
            let initialEmbeddings = try languageModel.tokenEmbeddings(for: textIDs)
                .asType(inputDType.mlxDType)
            initialHidden = try languageModel.hiddenStates(
                inputEmbeddings: initialEmbeddings,
                cache: cache
            )
        } else {
            initialHidden = try languageModel.hiddenStates(textIDs, cache: cache)
        }
        var lastHidden = initialHidden[0..., -1, 0...]
        let vocabulary = MLXArray.arange(languageModel.vocabularySize, dtype: .int32)
        let semanticStart = Int32(modelConfiguration.audioCodeOffset)
        let semanticEnd = semanticStart + Int32(modelConfiguration.semanticVocabularySize)
        let allowedVocabulary = (
            (vocabulary .>= semanticStart) & (vocabulary .< semanticEnd)
        ) | (vocabulary .== Int32(modelConfiguration.audioEndTokenID))

        var nextKey = key
        var frameHiddens: [MLXArray] = []
        var semanticCodes: [MLXArray] = []
        var frameCodesTrace: [MLXArray] = []
        var depthGuidedLogits: [MLXArray] = []
        var iterations = 0
        var stoppedByEndToken = false

        for frameIndex in 0...maxFrames {
            iterations += 1
            let logits = try languageModel.logits(forHiddenStates: lastHidden)
            let guided = try MiniMaxMusic3Sampling.semanticGuidedLogits(
                logits: logits,
                allowedVocabulary: allowedVocabulary,
                cfgScale: generation.arCFGScale,
                conditionalTopK: generation.topK
            )
            let sampled = try MiniMaxMusic3Sampling.sampleTopK(
                logits: guided,
                key: nextKey,
                topK: generation.topK
            )
            nextKey = sampled.nextKey
            MLX.eval(sampled.sample)
            let sampledValue = sampled.sample.asArray(Int32.self).first
            guard let sampledValue else {
                throw MiniMaxMusic3AutoregressiveError.invalidInput(
                    "semantic sample 沒有回傳 token。"
                )
            }
            if sampledValue == Int32(modelConfiguration.audioEndTokenID) {
                stoppedByEndToken = true
                progress?(min(frameHiddens.count, maxFrames), maxFrames)
                break
            }

            semanticCodes.append(sampled.sample)
            let semanticCode = (sampled.sample - semanticStart).asType(.int32)
            let semanticPair = MLX.repeated(semanticCode, count: 2, axis: 0)
            let depth = try miniMaxMusic3GenerateDepthCodes(
                languageModel: languageModel,
                decoder: decoder,
                lastHidden: lastHidden,
                semanticCode: semanticPair,
                key: nextKey,
                modelConfiguration: modelConfiguration,
                arCFGScale: generation.arCFGScale,
                topK: generation.topK,
                inputDType: inputDType
            )
            nextKey = depth.nextKey
            if frameIndex < maxFrames {
                frameCodesTrace.append(depth.frameCodes)
                depthGuidedLogits.append(depth.guidedLogits)
            }
            if frameIndex > 0 {
                frameHiddens.append(
                    MLX.concatenated(
                        [lastHidden[0..<1, 0...], depth.hiddenStates],
                        axis: 1
                    )
                )
                if frameHiddens.count >= maxFrames {
                    break
                }
            }

            let semanticEmbedding = try languageModel.tokenEmbeddings(
                for: semanticPair + Int32(modelConfiguration.audioCodeOffset)
            ).asType(inputDType.mlxDType).reshaped(2, 1, languageModel.hiddenSize)
            var feedback = try decoder.embedAudioFrame(
                semanticEmbedding: semanticEmbedding,
                residualCodes: depth.frameCodes[0..., 1...]
            )
            if inputDType == .float32 {
                feedback = feedback.asType(.float32)
            }
            let hidden = try languageModel.hiddenStates(
                inputEmbeddings: feedback,
                cache: cache
            )
            lastHidden = hidden[0..., -1, 0...]
            MLX.eval(lastHidden)
            progress?(min(frameIndex, maxFrames), maxFrames)
        }

        guard !frameHiddens.isEmpty else {
            throw MiniMaxMusic3AutoregressiveError.zeroFrames
        }
        let output = MLX.stacked(frameHiddens, axis: 1)
        let semanticCodeOutput = MLX.concatenated(semanticCodes, axis: 0)
        let frameCodeOutput = MLX.stacked(frameCodesTrace, axis: 0)
        let depthGuidedLogitsOutput = MLX.stacked(depthGuidedLogits, axis: 0)
        MLX.eval(
            initialHidden,
            output,
            semanticCodeOutput,
            frameCodeOutput,
            depthGuidedLogitsOutput
        )
        return MiniMaxMusic3FrameHiddenGenerationResult(
            initialHidden: initialHidden,
            frameHiddens: output,
            semanticCodes: semanticCodeOutput,
            frameCodes: frameCodeOutput,
            depthGuidedLogits: depthGuidedLogitsOutput,
            nextKey: nextKey,
            iterations: iterations,
            stoppedByEndToken: stoppedByEndToken
        )
    }
}

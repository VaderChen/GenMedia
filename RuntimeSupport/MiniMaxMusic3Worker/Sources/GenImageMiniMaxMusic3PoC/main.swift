import Darwin
import Dispatch
import Foundation
import MLX
import MLXNN
import MiniMaxMusic3SwiftRuntime
import Tokenizers

private enum PoCError: LocalizedError {
    case usage(String)
    case missingArgument(String)
    case missingTensor(URL, String)
    case shapeMismatch([Int], [Int])

    var errorDescription: String? {
        switch self {
        case let .usage(message):
            message
        case let .missingArgument(argument):
            "參數 \(argument) 後方需要值。"
        case let .missingTensor(url, key):
            "\(url.path) 缺少 Tensor：\(key)"
        case let .shapeMismatch(reference, actual):
            "shape 不一致：reference=\(reference)，actual=\(actual)"
        }
    }
}

private func comparisonThreshold(
    referenceDType: DType,
    actualDType: DType,
    declaredInputDType: String?
) -> (value: Double, label: String) {
    switch declaredInputDType {
    case "bfloat16":
        return (1e-2, "bf16 input")
    case "float32":
        return (1e-4, "fp32 input")
    default:
        if referenceDType == .bfloat16 || actualDType == .bfloat16 {
            return (1e-2, "bf16 tensor")
        }
        return (1e-4, "fp32/other tensor")
    }
}

// Final audio is judged by error energy rather than the single worst sample.
// The relative max diff remains the primary metric for deterministic tensors.
private func audioSNRThreshold(
    declaredInputDType: String?
) -> (value: Double, label: String)? {
    switch declaredInputDType {
    case "bfloat16":
        return (30.0, "bf16 compute")
    case "float32":
        return (80.0, "fp32 compute")
    default:
        return nil
    }
}

private struct Arguments {
    enum Command: String {
        case decode
        case decodeChunks = "decode-chunks"
        case forward
        case denoise
        case generate
        case compare
        case level1
        case languageForward = "language-forward"
        case attentionProbe = "attention-probe"
        case conditionForward = "condition-forward"
        case rvqForward = "rvq-forward"
        case rvqEmbeddings = "rvq-embeddings"
        case frameHiddens = "frame-hiddens"
    }

    var command: Command
    var modelDirectory: URL?
    var inputURL: URL?
    var outputURL: URL?
    var wavOutputURL: URL?
    var referenceURL: URL?
    var actualURL: URL?
    var key = "audio"
    var prompt = "sunset over a quiet ocean"
    var lyrics = "[Verse] gentle waves"
    var seed = 7
    var tokenizerDirectory: URL?
    var textIDs = "151644,77091,151645,151643,151643"
    var inputDType = "bfloat16"
    var topK = 4
    var arCFG: Float = 1.5
    var flowCFG: Float = 1.7
    var steps = 3
    var cfg: Float = 1.7
    var overlap = 0
    var audioDuration: Float = 0.2

    static func parse(_ values: [String]) throws -> Arguments {
        guard let commandValue = values.first,
              let command = Command(rawValue: commandValue) else {
            throw PoCError.usage(Self.usage)
        }
        var result = Arguments(command: command)
        var index = 1
        while index < values.count {
            let argument = values[index]
            index += 1
            guard index < values.count else {
                throw PoCError.missingArgument(argument)
            }
            let value = values[index]
            index += 1
            let url = URL(fileURLWithPath: (value as NSString).expandingTildeInPath)
            switch argument {
            case "--model-dir":
                result.modelDirectory = url
            case "--input":
                result.inputURL = url
            case "--output":
                result.outputURL = url
            case "--wav-output":
                result.wavOutputURL = url
            case "--reference":
                result.referenceURL = url
            case "--actual":
                result.actualURL = url
            case "--key":
                result.key = value
            case "--prompt":
                result.prompt = value
            case "--lyrics":
                result.lyrics = value
            case "--seed":
                guard let seed = Int(value) else {
                    throw PoCError.usage("--seed 必須是整數。")
                }
                result.seed = seed
            case "--tokenizer-dir":
                result.tokenizerDirectory = url
            case "--text-ids":
                result.textIDs = value
            case "--input-dtype":
                guard value == "bfloat16" || value == "float32" else {
                    throw PoCError.usage("--input-dtype 必須是 bfloat16 或 float32。")
                }
                result.inputDType = value
            case "--top-k":
                guard let topK = Int(value) else {
                    throw PoCError.usage("--top-k 必須是整數。")
                }
                result.topK = topK
            case "--ar-cfg":
                guard let arCFG = Float(value) else {
                    throw PoCError.usage("--ar-cfg 必須是數字。")
                }
                result.arCFG = arCFG
            case "--flow-cfg":
                guard let flowCFG = Float(value) else {
                    throw PoCError.usage("--flow-cfg 必須是數字。")
                }
                result.flowCFG = flowCFG
            case "--steps":
                guard let steps = Int(value) else {
                    throw PoCError.usage("--steps 必須是整數。")
                }
                result.steps = steps
            case "--cfg":
                guard let cfg = Float(value) else {
                    throw PoCError.usage("--cfg 必須是數字。")
                }
                result.cfg = cfg
            case "--overlap":
                guard let overlap = Int(value) else {
                    throw PoCError.usage("--overlap 必須是整數。")
                }
                result.overlap = overlap
            case "--audio-duration":
                guard let audioDuration = Float(value) else {
                    throw PoCError.usage("--audio-duration 必須是數字。")
                }
                result.audioDuration = audioDuration
            default:
                throw PoCError.usage("未知參數：\(argument)\n\n\(Self.usage)")
            }
        }
        return result
    }

    static let usage = """
    用法：
      GenImageMiniMaxMusic3PoC decode --model-dir <模型目錄> --input <reference.safetensors> --output <actual.safetensors> [--wav-output <audio.wav>]
      GenImageMiniMaxMusic3PoC decode-chunks --model-dir <模型目錄> --input <reference.safetensors> --output <actual.safetensors>
      GenImageMiniMaxMusic3PoC forward --model-dir <模型目錄> --input <reference.safetensors> --output <actual.safetensors>
      GenImageMiniMaxMusic3PoC denoise --model-dir <模型目錄> --input <reference.safetensors> --output <actual.safetensors> [--steps <N>] [--cfg <scale>] [--overlap <N>]
      GenImageMiniMaxMusic3PoC generate --model-dir <模型目錄> --output <actual.safetensors> [--wav-output <audio.wav>] [--prompt <文字>] [--lyrics <歌詞>] [--seed <N>] [--audio-duration <seconds>] [--steps <N>] [--ar-cfg <scale>] [--flow-cfg <scale>] [--top-k <N>] [--input-dtype bfloat16|float32]
      GenImageMiniMaxMusic3PoC level1 --model-dir <模型目錄> --input <reference.safetensors> --output <actual.safetensors> [--tokenizer-dir <目錄>]
      GenImageMiniMaxMusic3PoC language-forward --model-dir <模型目錄> --input <reference.safetensors> --output <actual.safetensors> [--input-dtype bfloat16|float32]
      GenImageMiniMaxMusic3PoC attention-probe --input <reference.safetensors> --output <actual.safetensors>
      GenImageMiniMaxMusic3PoC condition-forward --model-dir <模型目錄> --input <reference.safetensors> --output <actual.safetensors>
      GenImageMiniMaxMusic3PoC rvq-forward --model-dir <模型目錄> --input <reference.safetensors> --output <actual.safetensors>
      GenImageMiniMaxMusic3PoC rvq-embeddings --model-dir <模型目錄> --input <reference.safetensors> --output <actual.safetensors>
      GenImageMiniMaxMusic3PoC frame-hiddens --model-dir <模型目錄> --input <reference.safetensors> --output <actual.safetensors> [--input-dtype bfloat16|float32] [--audio-duration <seconds>] [--top-k <N>] [--ar-cfg <scale>]
      GenImageMiniMaxMusic3PoC compare --reference <reference.safetensors> --actual <actual.safetensors> [--key audio]
    """
}

private func tensor(_ key: String, from arrays: [String: MLXArray], url: URL) throws -> MLXArray {
    guard let value = arrays[key] else {
        throw PoCError.missingTensor(url, key)
    }
    return value
}

private func namedModule(_ path: String, in root: Module, inputURL: URL) throws -> Module {
    guard let module = root.namedModules().first(where: { $0.0 == path })?.1 else {
        throw PoCError.usage("模型缺少模組：\(path)，輸入為 \(inputURL.path)。")
    }
    return module
}

private func runDecode(_ arguments: Arguments) throws {
    guard let modelDirectory = arguments.modelDirectory else {
        throw PoCError.missingArgument("--model-dir")
    }
    guard let inputURL = arguments.inputURL else {
        throw PoCError.missingArgument("--input")
    }
    guard let outputURL = arguments.outputURL else {
        throw PoCError.missingArgument("--output")
    }
    let inputArrays = try MLX.loadArrays(url: inputURL)
    let latent = try tensor("latent", from: inputArrays, url: inputURL)
    let decoder = try MiniMaxMusic3Decoder(modelDirectory: modelDirectory)
    let audio = try decoder.decodeChunk(latent)
    try FileManager.default.createDirectory(
        at: outputURL.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    try MLX.save(
        arrays: ["latent": latent, "audio": audio],
        metadata: [
            "stage": "decode_chunks",
            "sampling_rate": String(decoder.vocoder.configuration.samplingRate),
            "latent_layout": "BCL",
            "audio_layout": "BCS"
        ],
        url: outputURL
    )
    if let wavOutputURL = arguments.wavOutputURL {
        let report = try MiniMaxMusic3Audio.writeWAV(
            audio,
            sampleRate: decoder.vocoder.configuration.samplingRate,
            to: wavOutputURL
        )
        print("wav=\(wavOutputURL.path) duration=\(report.durationSeconds)s")
    }
    print("actual=\(outputURL.path)")
    print("weights=\(decoder.weightReport?.tensorCount ?? 0) tensors")
    print("latent shape=\(latent.shape) dtype=\(latent.dtype)")
    print("audio shape=\(audio.shape) dtype=\(audio.dtype)")
}

private struct ChunkDecodeDiagnostic: Codable {
    let latentFrames: Int
    let waveformSamples: Int
    let cropLeft: Int
    let cropRight: Int
    let rawEnd: Int
    let retainedSamples: Int

    enum CodingKeys: String, CodingKey {
        case latentFrames = "latent_frames"
        case waveformSamples = "waveform_samples"
        case cropLeft = "crop_left"
        case cropRight = "crop_right"
        case rawEnd = "raw_end"
        case retainedSamples = "retained_samples"
    }
}

private func runDecodeChunks(_ arguments: Arguments) throws {
    guard let modelDirectory = arguments.modelDirectory else {
        throw PoCError.missingArgument("--model-dir")
    }
    guard let inputURL = arguments.inputURL else {
        throw PoCError.missingArgument("--input")
    }
    guard let outputURL = arguments.outputURL else {
        throw PoCError.missingArgument("--output")
    }
    let (inputArrays, inputMetadata) = try MLX.loadArraysAndMetadata(url: inputURL)
    guard let chunkCountValue = inputMetadata["chunk_count"],
          let chunkCount = Int(chunkCountValue),
          chunkCount > 0 else {
        throw PoCError.usage("decode-chunks fixture 必須包含至少一個 chunk。")
    }
    var latentChunks = try (0..<chunkCount).map { index in
        try tensor("latent_chunk_\(index)", from: inputArrays, url: inputURL)
    }
    let decoder = try MiniMaxMusic3Decoder(modelDirectory: modelDirectory)
    if inputMetadata["input_dtype"] == "float32" {
        let converted = Dictionary(uniqueKeysWithValues:
            decoder.vocoder.parameters().flattened().map { key, value in
                (key, value.asType(.float32))
            }
        )
        try decoder.vocoder.update(
            parameters: ModuleParameters.unflattened(converted),
            verify: .all
        )
        latentChunks = latentChunks.map { $0.asType(.float32) }
        MLX.eval(decoder.vocoder, latentChunks)
    }
    var diagnostics: [ChunkDecodeDiagnostic] = []
    for (index, latentChunk) in latentChunks.enumerated() {
        let waveform = try decoder.decodeChunk(latentChunk)
        let crop = try MiniMaxMusic3ChunkLayout.waveformCrop(
            chunkIndex: index,
            chunkCount: chunkCount,
            hopLength: decoder.vocoder.configuration.hopLength
        )
        let rawEnd = waveform.shape[2] - crop.right
        let retainedSamples = rawEnd <= crop.left
            ? waveform.shape[2]
            : rawEnd - crop.left
        diagnostics.append(ChunkDecodeDiagnostic(
            latentFrames: latentChunk.shape[2],
            waveformSamples: waveform.shape[2],
            cropLeft: crop.left,
            cropRight: crop.right,
            rawEnd: rawEnd,
            retainedSamples: retainedSamples
        ))
    }

    let audio = try decoder.decodeChunks(latentChunks)
    MLX.eval(audio)
    try FileManager.default.createDirectory(
        at: outputURL.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    var outputArrays = ["audio": audio]
    for (index, latentChunk) in latentChunks.enumerated() {
        outputArrays["latent_chunk_\(index)"] = latentChunk
    }
    var outputMetadata = inputMetadata
    outputMetadata["runtime"] = "swift-mlx-0.31.6"
    outputMetadata["swift_chunk_diagnostics"] = String(
        decoding: try JSONEncoder().encode(diagnostics),
        as: UTF8.self
    )
    try MLX.save(arrays: outputArrays, metadata: outputMetadata, url: outputURL)

    print("actual=\(outputURL.path)")
    print("weights=\(decoder.weightReport?.tensorCount ?? 0) tensors")
    print("chunks=\(chunkCount)")
    for (index, diagnostic) in diagnostics.enumerated() {
        print(
            "chunk[\(index)] latent=\(diagnostic.latentFrames) "
                + "waveform=\(diagnostic.waveformSamples) "
                + "crop=(\(diagnostic.cropLeft),\(diagnostic.cropRight)) "
                + "raw_end=\(diagnostic.rawEnd) "
                + "retained=\(diagnostic.retainedSamples)"
        )
    }
    print("audio shape=\(audio.shape) dtype=\(audio.dtype)")
}

private func loadFlowTransformer(
    modelDirectory: URL
) throws -> (model: MiniMaxMusic3FlowTransformer, report: MiniMaxMusic3FlowTransformerWeightLoadReport) {
    let configuration = try MiniMaxMusic3FlowTransformerConfiguration.load(from: modelDirectory)
    let model = MiniMaxMusic3FlowTransformer(configuration: configuration)
    let report = try MiniMaxMusic3FlowTransformerWeightLoader.load(
        model: model,
        from: modelDirectory
    )
    return (model, report)
}

private func runForward(_ arguments: Arguments) throws {
    guard let modelDirectory = arguments.modelDirectory else {
        throw PoCError.missingArgument("--model-dir")
    }
    guard let inputURL = arguments.inputURL else {
        throw PoCError.missingArgument("--input")
    }
    guard let outputURL = arguments.outputURL else {
        throw PoCError.missingArgument("--output")
    }
    let arrays = try MLX.loadArrays(url: inputURL)
    let latents = try tensor("latents", from: arrays, url: inputURL)
    let timestep = try tensor("timestep", from: arrays, url: inputURL)
    let condition = try tensor("condition", from: arrays, url: inputURL)
    let loaded = try loadFlowTransformer(modelDirectory: modelDirectory)
    let velocity = try loaded.model(latents, timestep: timestep, encoderHiddenStates: condition)
    MLX.eval(latents, timestep, condition, velocity)
    try FileManager.default.createDirectory(
        at: outputURL.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    try MLX.save(
        arrays: [
            "latents": latents,
            "timestep": timestep,
            "condition": condition,
            "velocity": velocity
        ],
        metadata: [
            "stage": "transformer_forward",
            "input_dtype": String(describing: latents.dtype)
        ],
        url: outputURL
    )
    print("actual=\(outputURL.path)")
    print("weights=\(loaded.report.tensorCount) tensors")
    print("quantized modules=\(loaded.report.quantizedModuleCount)")
    print("velocity shape=\(velocity.shape) dtype=\(velocity.dtype)")
}

private func runLevel1(_ arguments: Arguments) async throws {
    guard let modelDirectory = arguments.modelDirectory else {
        throw PoCError.missingArgument("--model-dir")
    }
    guard let inputURL = arguments.inputURL else {
        throw PoCError.missingArgument("--input")
    }
    guard let outputURL = arguments.outputURL else {
        throw PoCError.missingArgument("--output")
    }
    let tokenizerDirectory = arguments.tokenizerDirectory
        ?? modelDirectory.appendingPathComponent("tokenizer", isDirectory: true)
    let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerDirectory)
    let (referenceArrays, referenceMetadata) = try MLX.loadArraysAndMetadata(url: inputURL)
    let prompt = referenceMetadata["prompt"] ?? arguments.prompt
    let lyrics = referenceMetadata["lyrics"] ?? arguments.lyrics
    let tokenIDs = try MiniMaxMusic3Prompt.buildCFGTokenIDs(
        prompt: prompt,
        lyrics: lyrics,
        encode: { tokenizer.encode(text: $0) }
    )
    guard tokenIDs.count == 2,
          let sequenceLength = tokenIDs.first?.count,
          sequenceLength > 0 else {
        throw PoCError.usage("Level 1 token 序列無效。")
    }

    let logits = try tensor("logits", from: referenceArrays, url: inputURL)
    let allowedVocabulary = try tensor("allowed_vocabulary", from: referenceArrays, url: inputURL)
    let initialKey = try tensor("initial_key", from: referenceArrays, url: inputURL)
    let guidedLogits = try MiniMaxMusic3Sampling.semanticGuidedLogits(
        logits: logits,
        allowedVocabulary: allowedVocabulary,
        cfgScale: Float(referenceMetadata["ar_cfg"] ?? "1.5") ?? arguments.arCFG,
        conditionalTopK: Int(referenceMetadata["top_k"] ?? "4") ?? arguments.topK
    )
    let sampled = try MiniMaxMusic3Sampling.sampleTopK(
        logits: guidedLogits,
        key: initialKey,
        topK: Int(referenceMetadata["top_k"] ?? "4") ?? arguments.topK
    )
    let tokenIDArray = MLXArray(
        tokenIDs.flatMap { $0 },
        [2, sequenceLength]
    )
    MLX.eval(tokenIDArray, guidedLogits, sampled.sample, sampled.nextKey)
    try FileManager.default.createDirectory(
        at: outputURL.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    try MLX.save(
        arrays: [
            "conditional_token_ids": tokenIDArray[0],
            "unconditional_token_ids": tokenIDArray[1],
            "logits": logits,
            "allowed_vocabulary": allowedVocabulary,
            "guided_logits": guidedLogits,
            "initial_key": initialKey,
            "sampled": sampled.sample,
            "next_key": sampled.nextKey
        ],
        metadata: [
            "stage": "level1",
            "prompt": prompt,
            "lyrics": lyrics,
            "prompt_text": try MiniMaxMusic3Prompt.buildPromptText(
                prompt: prompt,
                lyrics: lyrics
            )
        ],
        url: outputURL
    )
    let referenceConditional = try tensor(
        "conditional_token_ids", from: referenceArrays, url: inputURL
    )
    let referenceUnconditional = try tensor(
        "unconditional_token_ids", from: referenceArrays, url: inputURL
    )
    MLX.eval(referenceConditional, referenceUnconditional)
    let tokensMatch = referenceConditional.asArray(Int32.self) == tokenIDArray[0].asArray(Int32.self)
        && referenceUnconditional.asArray(Int32.self) == tokenIDArray[1].asArray(Int32.self)
    print("actual=\(outputURL.path)")
    print("prompt token parity: \(tokensMatch ? "PASS" : "FAIL")")
    print("guided logits shape=\(guidedLogits.shape) dtype=\(guidedLogits.dtype)")
    print("sampled=\(sampled.sample) next_key=\(sampled.nextKey)")
}

private func runLanguageForward(_ arguments: Arguments) throws {
    guard let modelDirectory = arguments.modelDirectory else {
        throw PoCError.missingArgument("--model-dir")
    }
    guard let inputURL = arguments.inputURL else {
        throw PoCError.missingArgument("--input")
    }
    guard let outputURL = arguments.outputURL else {
        throw PoCError.missingArgument("--output")
    }
    let arrays = try MLX.loadArrays(url: inputURL)
    let textIDs = try tensor("text_ids", from: arrays, url: inputURL)
    let language = try MiniMaxMusic3LanguageModel(modelDirectory: modelDirectory)
    let hiddenStates: MLXArray
    let logits: MLXArray
    if arguments.inputDType == "float32" {
        let inputEmbeddings = try language.tokenEmbeddings(for: textIDs).asType(.float32)
        hiddenStates = try language.hiddenStates(inputEmbeddings: inputEmbeddings)
        logits = try language.logits(forHiddenStates: hiddenStates)
    } else {
        hiddenStates = try language.hiddenStates(textIDs)
        logits = try language.logits(for: textIDs)
    }
    MLX.eval(textIDs, hiddenStates, logits)
    try FileManager.default.createDirectory(
        at: outputURL.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    try MLX.save(
        arrays: [
            "text_ids": textIDs,
            "hidden_states": hiddenStates,
            "logits": logits
        ],
        metadata: ["stage": "language_forward", "input_dtype": arguments.inputDType],
        url: outputURL
    )
    print("actual=\(outputURL.path)")
    print("weights=\(language.report.tensorCount) tensors")
    print("quantized modules=\(language.report.quantizedModuleCount)")
    print("hidden states shape=\(hiddenStates.shape) dtype=\(hiddenStates.dtype)")
    print("logits shape=\(logits.shape) dtype=\(logits.dtype)")
}

private func runAttentionProbe(_ arguments: Arguments) throws {
    guard let inputURL = arguments.inputURL else {
        throw PoCError.missingArgument("--input")
    }
    guard let outputURL = arguments.outputURL else {
        throw PoCError.missingArgument("--output")
    }
    let arrays = try MLX.loadArrays(url: inputURL)
    if let modelDirectory = arguments.modelDirectory {
        let textIDs = try tensor("text_ids", from: arrays, url: inputURL)
        let language = try MiniMaxMusic3LanguageModel(modelDirectory: modelDirectory)
        let moduleRoot = language.model
        guard let embedding = try namedModule("model.embed_tokens", in: moduleRoot, inputURL: inputURL) as? Embedding,
              let inputLayerNorm = try namedModule("model.layers.0.input_layernorm", in: moduleRoot, inputURL: inputURL) as? RMSNorm,
              let queryProjection = try namedModule("model.layers.0.self_attn.q_proj", in: moduleRoot, inputURL: inputURL) as? Linear,
              let keyProjection = try namedModule("model.layers.0.self_attn.k_proj", in: moduleRoot, inputURL: inputURL) as? Linear,
              let valueProjection = try namedModule("model.layers.0.self_attn.v_proj", in: moduleRoot, inputURL: inputURL) as? Linear,
              let queryNorm = try namedModule("model.layers.0.self_attn.q_norm", in: moduleRoot, inputURL: inputURL) as? RMSNorm,
              let keyNorm = try namedModule("model.layers.0.self_attn.k_norm", in: moduleRoot, inputURL: inputURL) as? RMSNorm,
              let rope = try namedModule("model.layers.0.self_attn.rope", in: moduleRoot, inputURL: inputURL) as? RoPE,
              let outputProjection = try namedModule("model.layers.0.self_attn.o_proj", in: moduleRoot, inputURL: inputURL) as? Linear,
              let postAttentionLayerNorm = try namedModule("model.layers.0.post_attention_layernorm", in: moduleRoot, inputURL: inputURL) as? RMSNorm,
              let gateProjection = try namedModule("model.layers.0.mlp.gate_proj", in: moduleRoot, inputURL: inputURL) as? Linear,
              let upProjection = try namedModule("model.layers.0.mlp.up_proj", in: moduleRoot, inputURL: inputURL) as? Linear,
              let downProjection = try namedModule("model.layers.0.mlp.down_proj", in: moduleRoot, inputURL: inputURL) as? Linear else {
            throw PoCError.usage("找不到 Qwen3 第一層的必要模組。")
        }
        let embedded = embedding(textIDs)
        let normalized = inputLayerNorm(embedded)
        let batch = normalized.shape[0]
        let length = normalized.shape[1]
        let queriesBeforeRope = queryNorm(
            queryProjection(normalized).reshaped(batch, length, 32, 128)
        ).transposed(0, 2, 1, 3)
        let keysBeforeRope = keyNorm(
            keyProjection(normalized).reshaped(batch, length, 8, 128)
        ).transposed(0, 2, 1, 3)
        let values = valueProjection(normalized).reshaped(batch, length, 8, 128)
            .transposed(0, 2, 1, 3)
        let queries = rope(queriesBeforeRope, offset: 0)
        let keys = rope(keysBeforeRope, offset: 0)
        let sdpaOutput = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: pow(Float(128), -0.5),
            mask: .causal
        )
        let attentionOutput = outputProjection(
            sdpaOutput.transposed(0, 2, 1, 3).reshaped(batch, length, 32 * 128)
        )
        let referenceSDPAOutput = arrays["sdpa_output"] ?? sdpaOutput
        let attentionOnReferenceSDPA = outputProjection(
            referenceSDPAOutput.transposed(0, 2, 1, 3).reshaped(batch, length, 32 * 128)
        )
        let attentionResidual = embedded + attentionOutput
        let mlpInput = postAttentionLayerNorm(attentionResidual)
        let gateOutput = gateProjection(mlpInput)
        let upOutput = upProjection(mlpInput)
        let activatedOutput = silu(gateOutput) * upOutput
        let mlpOutput = downProjection(activatedOutput)
        let referenceMLPInput = arrays["mlp_input"] ?? mlpInput
        let gateOnReferenceInput = gateProjection(referenceMLPInput)
        let upOnReferenceInput = upProjection(referenceMLPInput)
        let activatedOnReferenceInput = silu(gateOnReferenceInput) * upOnReferenceInput
        let mlpOnReferenceInput = downProjection(activatedOnReferenceInput)
        let layerOutput = attentionResidual + mlpOutput
        let probeArrays: [String: MLXArray] = [
            "embedded": embedded,
            "normalized": normalized,
            "queries_before_rope": queriesBeforeRope,
            "keys_before_rope": keysBeforeRope,
            "queries": queries,
            "keys": keys,
            "values": values,
            "sdpa_output": sdpaOutput,
            "attention_output": attentionOutput,
            "attention_on_reference_sdpa": attentionOnReferenceSDPA,
            "attention_residual": attentionResidual,
            "mlp_input": mlpInput,
            "gate_output": gateOutput,
            "up_output": upOutput,
            "activated_output": activatedOutput,
            "mlp_output": mlpOutput,
            "gate_on_reference_input": gateOnReferenceInput,
            "up_on_reference_input": upOnReferenceInput,
            "activated_on_reference_input": activatedOnReferenceInput,
            "mlp_on_reference_input": mlpOnReferenceInput,
            "layer_output": layerOutput
        ]
        MLX.eval(probeArrays.values)
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try MLX.save(arrays: probeArrays, metadata: ["stage": "attention_probe"], url: outputURL)
        print("actual=\(outputURL.path)")
        print("model layer=0 input dtype=\(embedded.dtype)")
        print("layer output shape=\(layerOutput.shape) dtype=\(layerOutput.dtype)")
        return
    }
    let queries = try tensor("queries", from: arrays, url: inputURL)
    let keys = try tensor("keys", from: arrays, url: inputURL)
    let values = try tensor("values", from: arrays, url: inputURL)
    guard queries.ndim == 4, keys.ndim == 4, values.ndim == 4,
          queries.shape[0] == keys.shape[0], keys.shape == values.shape,
          queries.shape[2] == keys.shape[2], queries.shape[3] == keys.shape[3] else {
        throw PoCError.usage("attention probe 的 Q/K/V shape 不相容。")
    }
    let output = MLXFast.scaledDotProductAttention(
        queries: queries,
        keys: keys,
        values: values,
        scale: pow(Float(queries.shape[3]), -0.5),
        mask: .causal
    )
    let sequenceLength = queries.shape[2]
    let rowIndices = MLXArray.arange(sequenceLength).reshaped(sequenceLength, 1)
    let columnIndices = MLXArray.arange(sequenceLength).reshaped(1, sequenceLength)
    let causalMask = rowIndices .>= columnIndices
    let headRepeatCount = queries.shape[1] / keys.shape[1]
    let expandedKeys = headRepeatCount == 1
        ? keys
        : MLX.repeated(keys, count: headRepeatCount, axis: 1)
    let expandedValues = headRepeatCount == 1
        ? values
        : MLX.repeated(values, count: headRepeatCount, axis: 1)
    let scores = matmul(
        queries * pow(Float(queries.shape[3]), -0.5),
        expandedKeys.transposed(0, 1, 3, 2)
    )
    let maskedScores = MLX.where(causalMask, scores, -Float.infinity)
    let probabilities = softmax(maskedScores.asType(.float32), axis: -1).asType(values.dtype)
    let basicOutput = matmul(probabilities, expandedValues)
    MLX.eval(queries, keys, values, output)
    MLX.eval(basicOutput)
    try FileManager.default.createDirectory(
        at: outputURL.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    try MLX.save(
        arrays: [
            "queries": queries,
            "keys": keys,
            "values": values,
            "output": output,
            "basic_output": basicOutput
        ],
        metadata: ["stage": "attention_probe"],
        url: outputURL
    )
    print("actual=\(outputURL.path)")
    print("input dtype=\(queries.dtype)")
    print("output shape=\(output.shape) dtype=\(output.dtype)")
}

private func loadConditionEncoder(
    modelDirectory: URL
) throws -> (model: MiniMaxMusic3ConditionEncoder, report: MiniMaxMusic3ConditionEncoderWeightLoadReport) {
    let configuration = try MiniMaxMusic3ConditionEncoderConfiguration.load(from: modelDirectory)
    let model = MiniMaxMusic3ConditionEncoder(configuration: configuration)
    let report = try MiniMaxMusic3ConditionEncoderWeightLoader.load(
        model: model,
        from: modelDirectory
    )
    return (model, report)
}

private func loadRVQDecoder(
    modelDirectory: URL
) throws -> (model: MiniMaxMusic3RVQDepthDecoder, report: MiniMaxMusic3RVQDecoderWeightLoadReport) {
    let configuration = try MiniMaxMusic3RVQDecoderConfiguration.load(from: modelDirectory)
    let model = MiniMaxMusic3RVQDepthDecoder(configuration: configuration)
    let report = try MiniMaxMusic3RVQDecoderWeightLoader.load(
        model: model,
        from: modelDirectory
    )
    return (model, report)
}

private func runConditionForward(_ arguments: Arguments) throws {
    guard let modelDirectory = arguments.modelDirectory else {
        throw PoCError.missingArgument("--model-dir")
    }
    guard let inputURL = arguments.inputURL else {
        throw PoCError.missingArgument("--input")
    }
    guard let outputURL = arguments.outputURL else {
        throw PoCError.missingArgument("--output")
    }
    let arrays = try MLX.loadArrays(url: inputURL)
    let hiddenStates = try tensor("hidden_states", from: arrays, url: inputURL)
    let loaded = try loadConditionEncoder(modelDirectory: modelDirectory)
    let condition = try loaded.model(hiddenStates)
    MLX.eval(hiddenStates, condition)
    try FileManager.default.createDirectory(
        at: outputURL.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    try MLX.save(
        arrays: ["hidden_states": hiddenStates, "condition": condition],
        metadata: [
            "stage": "condition_encoder",
            "input_dtype": String(describing: hiddenStates.dtype)
        ],
        url: outputURL
    )
    print("actual=\(outputURL.path)")
    print("weights=\(loaded.report.tensorCount) tensors")
    print("hidden states shape=\(hiddenStates.shape) dtype=\(hiddenStates.dtype)")
    print("condition shape=\(condition.shape) dtype=\(condition.dtype)")
}

private func runRVQForward(_ arguments: Arguments) throws {
    guard let modelDirectory = arguments.modelDirectory else {
        throw PoCError.missingArgument("--model-dir")
    }
    guard let inputURL = arguments.inputURL else {
        throw PoCError.missingArgument("--input")
    }
    guard let outputURL = arguments.outputURL else {
        throw PoCError.missingArgument("--output")
    }
    let arrays = try MLX.loadArrays(url: inputURL)
    let inputsEmbeds = try tensor("inputs_embeds", from: arrays, url: inputURL)
    let loaded = try loadRVQDecoder(modelDirectory: modelDirectory)
    let hiddenStates = try loaded.model(inputsEmbeds)
    MLX.eval(inputsEmbeds, hiddenStates)
    try FileManager.default.createDirectory(
        at: outputURL.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    try MLX.save(
        arrays: ["inputs_embeds": inputsEmbeds, "hidden_states": hiddenStates],
        metadata: [
            "stage": "rvq_decoder_forward",
            "input_dtype": String(describing: inputsEmbeds.dtype)
        ],
        url: outputURL
    )
    print("actual=\(outputURL.path)")
    print("weights=\(loaded.report.tensorCount) tensors")
    print("quantized modules=\(loaded.report.quantizedModuleCount)")
    print("inputs embeds shape=\(inputsEmbeds.shape) dtype=\(inputsEmbeds.dtype)")
    print("hidden states shape=\(hiddenStates.shape) dtype=\(hiddenStates.dtype)")
}

private func runRVQEmbeddings(_ arguments: Arguments) throws {
    guard let modelDirectory = arguments.modelDirectory else {
        throw PoCError.missingArgument("--model-dir")
    }
    guard let inputURL = arguments.inputURL else {
        throw PoCError.missingArgument("--input")
    }
    guard let outputURL = arguments.outputURL else {
        throw PoCError.missingArgument("--output")
    }
    let arrays = try MLX.loadArrays(url: inputURL)
    let semanticEmbedding = try tensor("semantic_embedding", from: arrays, url: inputURL)
    let residualCodes = try tensor("residual_codes", from: arrays, url: inputURL)
    let loaded = try loadRVQDecoder(modelDirectory: modelDirectory)
    let residualEmbeddings = try loaded.model.residualEmbeddings(for: residualCodes)
    let combined = try loaded.model.embedAudioFrame(
        semanticEmbedding: semanticEmbedding,
        residualCodes: residualCodes
    )
    MLX.eval(semanticEmbedding, residualCodes, residualEmbeddings, combined)
    try FileManager.default.createDirectory(
        at: outputURL.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    try MLX.save(
        arrays: [
            "semantic_embedding": semanticEmbedding,
            "residual_codes": residualCodes,
            "residual_embeddings": residualEmbeddings,
            "combined": combined
        ],
        metadata: ["stage": "rvq_audio_embedding"],
        url: outputURL
    )
    print("actual=\(outputURL.path)")
    print("weights=\(loaded.report.tensorCount) tensors")
    print("residual embeddings shape=\(residualEmbeddings.shape) dtype=\(residualEmbeddings.dtype)")
    print("combined shape=\(combined.shape) dtype=\(combined.dtype)")
}

private func runFrameHiddens(_ arguments: Arguments) throws {
    guard let modelDirectory = arguments.modelDirectory else {
        throw PoCError.missingArgument("--model-dir")
    }
    guard let inputURL = arguments.inputURL else {
        throw PoCError.missingArgument("--input")
    }
    guard let outputURL = arguments.outputURL else {
        throw PoCError.missingArgument("--output")
    }
    let (arrays, inputMetadata) = try MLX.loadArraysAndMetadata(url: inputURL)
    let textIDs = try tensor("text_ids", from: arrays, url: inputURL)
    let initialKey = try tensor("initial_key", from: arrays, url: inputURL)
    let modelConfiguration = try MiniMaxMusic3ModelConfiguration.load(
        from: modelDirectory
    )
    let language = try MiniMaxMusic3LanguageModel(modelDirectory: modelDirectory)
    let loadedDecoder = try loadRVQDecoder(modelDirectory: modelDirectory)
    let decoder = loadedDecoder.model
    let generation = MiniMaxMusic3GenerationConfiguration(
        audioDuration: Float(inputMetadata["audio_duration"] ?? "") ?? arguments.audioDuration,
        seed: Int(inputMetadata["seed"] ?? "0") ?? 0,
        arCFGScale: Float(inputMetadata["ar_cfg"] ?? "") ?? arguments.arCFG,
        topK: Int(inputMetadata["top_k"] ?? "") ?? arguments.topK
    )
    let languageInputDType: MiniMaxMusic3LanguageInputDType = arguments.inputDType == "float32"
        ? .float32
        : .bfloat16
    let result = try MiniMaxMusic3AutoregressiveGenerator.generateFrameHiddens(
        textIDs: textIDs,
        languageModel: language,
        decoder: decoder,
        modelConfiguration: modelConfiguration,
        generation: generation,
        key: initialKey,
        inputDType: languageInputDType
    )
    MLX.eval(
        textIDs,
        initialKey,
        result.initialHidden,
        result.frameHiddens,
        result.semanticCodes,
        result.frameCodes,
        result.depthGuidedLogits,
        result.nextKey
    )
    try FileManager.default.createDirectory(
        at: outputURL.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    try MLX.save(
        arrays: [
            "text_ids": textIDs,
            "initial_key": initialKey,
            "initial_hidden": result.initialHidden,
            "frame_hiddens": result.frameHiddens,
            "semantic_codes": result.semanticCodes,
            "frame_codes": result.frameCodes,
            "depth_guided_logits": result.depthGuidedLogits,
            "next_key": result.nextKey
        ],
        metadata: [
            "stage": "generate_frame_hiddens",
            "audio_duration": String(generation.audioDuration),
            "ar_cfg": String(generation.arCFGScale),
            "top_k": String(generation.topK),
            "input_dtype": arguments.inputDType,
            "iterations": String(result.iterations),
            "stopped_by_end_token": String(result.stoppedByEndToken)
        ],
        url: outputURL
    )
    print("actual=\(outputURL.path)")
    print("language weights=\(language.report.tensorCount) tensors")
    print("RVQ weights=\(loadedDecoder.report.tensorCount) tensors")
    print("language input dtype=\(arguments.inputDType)")
    print("iterations=\(result.iterations) stopped_by_end_token=\(result.stoppedByEndToken)")
    print("frame hiddens shape=\(result.frameHiddens.shape) dtype=\(result.frameHiddens.dtype)")
    let semanticCodes = result.semanticCodes.asArray(Int32.self)
    let frameCodes = result.frameCodes.asArray(Int32.self)
    let frameStride = 2 * modelConfiguration.numberOfCodebooks
    for index in semanticCodes.indices {
        let start = index * frameStride
        let end = min(start + modelConfiguration.numberOfCodebooks, frameCodes.count)
        let depthCodes = Array(frameCodes[(start + 0)..<end])
        print("token-step[\(index + 1)] semantic=\(semanticCodes[index]) depth=\(depthCodes)")
    }
}

private func peakMemoryBytes() -> Int64 {
    var usage = rusage()
    guard getrusage(RUSAGE_SELF, &usage) == 0 else { return 0 }
    return Int64(usage.ru_maxrss)
}

private func runGenerate(_ arguments: Arguments) async throws {
    guard let modelDirectory = arguments.modelDirectory else {
        throw PoCError.missingArgument("--model-dir")
    }
    guard let outputURL = arguments.outputURL else {
        throw PoCError.missingArgument("--output")
    }
    let startedAt = Date()
    let tokenizerDirectory = arguments.tokenizerDirectory
        ?? modelDirectory.appendingPathComponent("tokenizer", isDirectory: true)
    let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerDirectory)
    let modelConfiguration = try MiniMaxMusic3ModelConfiguration.load(from: modelDirectory)
    let language = try MiniMaxMusic3LanguageModel(modelDirectory: modelDirectory)
    let loadedRVQ = try loadRVQDecoder(modelDirectory: modelDirectory)
    let loadedCondition = try loadConditionEncoder(modelDirectory: modelDirectory)
    let loadedTransformer = try loadFlowTransformer(modelDirectory: modelDirectory)
    let decoder = try MiniMaxMusic3Decoder(modelDirectory: modelDirectory)
    let inputDType: MiniMaxMusic3LanguageInputDType = arguments.inputDType == "float32"
        ? .float32
        : .bfloat16
    let generation = MiniMaxMusic3GenerationConfiguration(
        audioDuration: arguments.audioDuration,
        seed: arguments.seed,
        numInferenceSteps: arguments.steps,
        arCFGScale: arguments.arCFG,
        flowCFGScale: arguments.flowCFG,
        topK: arguments.topK,
        inputDType: inputDType
    )
    let pipeline = try MiniMaxMusic3Pipeline(
        modelConfiguration: modelConfiguration,
        languageModel: language,
        rvqDepthDecoder: loadedRVQ.model,
        conditionEncoder: loadedCondition.model,
        transformer: loadedTransformer.model,
        vocoder: decoder.vocoder,
        tokenEncoder: { tokenizer.encode(text: $0) }
    )
    let result = try pipeline.generate(
        prompt: arguments.prompt,
        lyrics: arguments.lyrics,
        generation: generation
    )
    let elapsed = Date().timeIntervalSince(startedAt)
    MLX.eval(result.audio)
    let audioReport = try MiniMaxMusic3Audio.validate(
        result.audio,
        sampleRate: result.samplingRate
    )
    let audioValues = result.audio.asType(.float32).asArray(Float.self)
    let audioPeak = audioValues.map(abs).max() ?? 0
    let clipped = audioValues.contains { abs($0) >= 1 }
    try FileManager.default.createDirectory(
        at: outputURL.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    let outputArrays: [String: MLXArray] = ["audio": result.audio]
    let outputMetadata: [String: String] = [
            "stage": "generate",
            "prompt": arguments.prompt,
            "lyrics": arguments.lyrics,
            "seed": String(generation.seed),
            "audio_duration": String(generation.audioDuration),
            "steps": String(generation.numInferenceSteps),
            "ar_cfg": String(generation.arCFGScale),
            "flow_cfg": String(generation.flowCFGScale),
            "top_k": String(generation.topK),
            "input_dtype": arguments.inputDType,
            "sampling_rate": String(result.samplingRate),
            "num_frames": String(result.numFrames),
            "num_chunks": String(result.numChunks),
            "iterations": String(result.iterations),
            "stopped_by_end_token": String(result.stoppedByEndToken)
    ]
    try MLX.save(arrays: outputArrays, metadata: outputMetadata, url: outputURL)
    if let wavOutputURL = arguments.wavOutputURL {
        let report = try MiniMaxMusic3Audio.writeWAV(
            result.audio,
            sampleRate: result.samplingRate,
            to: wavOutputURL
        )
        print("wav=\(wavOutputURL.path) duration=\(report.durationSeconds)s")
    }
    print("actual=\(outputURL.path)")
    print("language weights=\(language.report.tensorCount) tensors")
    print("RVQ weights=\(loadedRVQ.report.tensorCount) tensors")
    print("condition weights=\(loadedCondition.report.tensorCount) tensors")
    print("transformer weights=\(loadedTransformer.report.tensorCount) tensors")
    print("vocoder weights=\(decoder.weightReport?.tensorCount ?? 0) tensors")
    print("input dtype=\(arguments.inputDType)")
    print("audio shape=\(result.audio.shape) dtype=\(result.audio.dtype)")
    print("sampling rate=\(result.samplingRate)")
    print("audio finite=true duration=\(audioReport.durationSeconds)s")
    print(String(format: "audio peak=%.10g clipped=%@", audioPeak, clipped ? "true" : "false"))
    print("num frames=\(result.numFrames) num chunks=\(result.numChunks)")
    print("iterations=\(result.iterations) stopped_by_end_token=\(result.stoppedByEndToken)")
    print(String(format: "elapsed seconds=%.3f", elapsed))
    print(String(format: "peak memory MB=%.1f", Double(peakMemoryBytes()) / 1_048_576.0))
}

private func runDenoise(_ arguments: Arguments) throws {
    guard let modelDirectory = arguments.modelDirectory else {
        throw PoCError.missingArgument("--model-dir")
    }
    guard let inputURL = arguments.inputURL else {
        throw PoCError.missingArgument("--input")
    }
    guard let outputURL = arguments.outputURL else {
        throw PoCError.missingArgument("--output")
    }
    guard arguments.steps > 0 else {
        throw PoCError.usage("--steps 必須是正整數。")
    }
    let arrays = try MLX.loadArrays(url: inputURL)
    let initialLatents = try tensor("initial_latents", from: arrays, url: inputURL)
    let condition = try tensor("condition", from: arrays, url: inputURL)
    guard initialLatents.ndim == 3,
          initialLatents.shape[0] > 0,
          initialLatents.shape[1] > 0,
          initialLatents.shape[2] == 128 else {
        throw PoCError.usage("initial_latents 必須是 [batch, frames, 128]。")
    }
    guard arguments.overlap >= 0,
          arguments.overlap <= initialLatents.shape[1] else {
        throw PoCError.usage("--overlap 不得超過 latent 長度。")
    }
    let previousLatent: MLXArray?
    let noisePrompt: MLXArray?
    if arguments.overlap > 0 {
        previousLatent = try tensor("previous_latent", from: arrays, url: inputURL)
        noisePrompt = try tensor("noise_prompt", from: arrays, url: inputURL)
    } else {
        previousLatent = nil
        noisePrompt = nil
    }
    let loaded = try loadFlowTransformer(modelDirectory: modelDirectory)
    let timesteps = try MiniMaxMusic3FlowScheduler.flowTimesteps(
        numInferenceSteps: arguments.steps
    )
    MLX.eval(timesteps)
    let timestepValues = timesteps.asArray(Float.self)
    var latents = initialLatents
    for timestepValue in timestepValues {
        let timestep = MLXArray(
            Array(repeating: timestepValue, count: latents.shape[0])
        )
        if arguments.overlap > 0 {
            latents = try MiniMaxMusic3FlowScheduler.blendOverlap(
                latents: latents,
                noisePrompt: noisePrompt!,
                previousLatent: previousLatent!,
                overlap: arguments.overlap,
                timestep: timestep
            )
        }
        let conditional = try loaded.model(
            latents,
            timestep: timestep,
            encoderHiddenStates: condition
        )
        let unconditional = try loaded.model(
            latents,
            timestep: timestep,
            encoderHiddenStates: MLXArray.zeros(like: condition)
        )
        let velocity = unconditional + arguments.cfg * (conditional - unconditional)
        latents = try MiniMaxMusic3FlowScheduler.eulerStep(
            sample: latents,
            velocity: velocity,
            numInferenceSteps: arguments.steps
        )
        MLX.eval(latents)
    }
    if arguments.overlap > 0 {
        latents = try MiniMaxMusic3FlowScheduler.restoreOverlap(
            latents: latents,
            previousLatent: previousLatent!,
            overlap: arguments.overlap
        )
    }
    MLX.eval(latents)
    try FileManager.default.createDirectory(
        at: outputURL.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    try MLX.save(
        arrays: [
            "initial_latents": initialLatents,
            "condition": condition,
            "final_latents": latents
        ],
        metadata: [
            "stage": "denoise_chunks",
            "steps": String(arguments.steps),
            "cfg": String(arguments.cfg),
            "overlap": String(arguments.overlap)
        ],
        url: outputURL
    )
    print("actual=\(outputURL.path)")
    print("weights=\(loaded.report.tensorCount) tensors")
    print("denoise steps=\(arguments.steps) overlap=\(arguments.overlap)")
    print("final latents shape=\(latents.shape) dtype=\(latents.dtype)")
}

private func runCompare(_ arguments: Arguments) throws {
    guard let referenceURL = arguments.referenceURL else {
        throw PoCError.missingArgument("--reference")
    }
    guard let actualURL = arguments.actualURL else {
        throw PoCError.missingArgument("--actual")
    }
    let (referenceArrays, referenceMetadata) = try MLX.loadArraysAndMetadata(url: referenceURL)
    let (actualArrays, _) = try MLX.loadArraysAndMetadata(url: actualURL)
    let reference = try tensor(arguments.key, from: referenceArrays, url: referenceURL)
    let actual = try tensor(arguments.key, from: actualArrays, url: actualURL)
    guard reference.shape == actual.shape else {
        print("shape: FAIL")
        print("reference shape=\(reference.shape) actual shape=\(actual.shape)")
        throw PoCError.shapeMismatch(reference.shape, actual.shape)
    }
    let referenceFloat = reference.asType(.float32)
    let actualFloat = actual.asType(.float32)
    MLX.eval(referenceFloat, actualFloat)
    let referenceValues = referenceFloat.asArray(Float.self)
    let actualValues = actualFloat.asArray(Float.self)

    var maxAbsDiff = 0.0
    var sumAbsDiff = 0.0
    var maxReferenceAbs = 0.0
    var errorSquaredSum = 0.0
    var dot = 0.0
    var referenceSquaredSum = 0.0
    var actualSquaredSum = 0.0
    var finiteCount = 0
    var nonFiniteMismatchCount = 0
    for (referenceValue, actualValue) in zip(referenceValues, actualValues) {
        let referenceDouble = Double(referenceValue)
        let actualDouble = Double(actualValue)
        if !referenceDouble.isFinite || !actualDouble.isFinite {
            let sameNaN = referenceDouble.isNaN && actualDouble.isNaN
            let sameInfinity = referenceDouble.isInfinite
                && actualDouble.isInfinite
                && referenceDouble.sign == actualDouble.sign
            if !sameNaN && !sameInfinity {
                nonFiniteMismatchCount += 1
            }
            continue
        }
        let absoluteDifference = abs(referenceDouble - actualDouble)
        maxAbsDiff = max(maxAbsDiff, absoluteDifference)
        sumAbsDiff += absoluteDifference
        maxReferenceAbs = max(maxReferenceAbs, abs(referenceDouble))
        errorSquaredSum += absoluteDifference * absoluteDifference
        dot += referenceDouble * actualDouble
        referenceSquaredSum += referenceDouble * referenceDouble
        actualSquaredSum += actualDouble * actualDouble
        finiteCount += 1
    }

    let relativeMaxDiff: Double
    if maxReferenceAbs == 0 {
        relativeMaxDiff = maxAbsDiff == 0 ? 0 : .infinity
    } else {
        relativeMaxDiff = maxAbsDiff / maxReferenceAbs
    }
    let referenceNorm = sqrt(referenceSquaredSum)
    let actualNorm = sqrt(actualSquaredSum)
    let meanAbsDiff = finiteCount == 0 ? 0 : sumAbsDiff / Double(finiteCount)
    let declaredInputDType = referenceMetadata["input_dtype"]
    let threshold = comparisonThreshold(
        referenceDType: reference.dtype,
        actualDType: actual.dtype,
        declaredInputDType: declaredInputDType
    )
    let finalAudio = arguments.key == "audio" || arguments.key == "waveform"
    let snrThreshold = finalAudio
        ? audioSNRThreshold(declaredInputDType: declaredInputDType)
        : nil
    let snrDB: Double
    if nonFiniteMismatchCount > 0 {
        snrDB = -.infinity
    } else if errorSquaredSum == 0 {
        snrDB = .infinity
    } else if referenceSquaredSum == 0 {
        snrDB = -.infinity
    } else {
        // 20*log10(RMS(reference)/RMS(error)); sample count cancels.
        snrDB = 10.0 * log10(referenceSquaredSum / errorSquaredSum)
    }
    let cosine: Double
    if finiteCount == 0 {
        cosine = nonFiniteMismatchCount == 0 ? 1 : 0
    } else if referenceNorm == 0 || actualNorm == 0 {
        cosine = referenceNorm == actualNorm ? 1 : 0
    } else {
        cosine = min(1, max(-1, dot / (referenceNorm * actualNorm)))
    }

    print("shape: PASS \(reference.shape)")
    print("reference dtype=\(reference.dtype) actual dtype=\(actual.dtype)")
    if finalAudio {
        print(String(format: "SNR: %.10g dB (primary)", snrDB))
    }
    print(String(format: "relative max diff: %.10g", relativeMaxDiff))
    print(String(format: "max abs diff: %.10g", maxAbsDiff))
    print(String(format: "mean abs diff: %.10g", meanAbsDiff))
    print(String(format: "cosine similarity: %.10g", cosine))
    print("non-finite mismatches: \(nonFiniteMismatchCount)")
    if finalAudio {
        if let snrThreshold {
            print(String(
                format: "SNR threshold: %.1f dB [%@] (%@)",
                snrThreshold.value,
                snrThreshold.label,
                nonFiniteMismatchCount == 0 && snrDB >= snrThreshold.value ? "PASS" : "FAIL"
            ))
        } else {
            print("SNR threshold: unavailable [input_dtype metadata missing or unsupported] (UNDETERMINED)")
        }
        print("relative max diff: auxiliary for final audio")
    } else {
        print(String(
            format: "relative max diff threshold: %.10g [%@] (%@)",
            threshold.value,
            threshold.label,
            nonFiniteMismatchCount == 0 && relativeMaxDiff <= threshold.value ? "PASS" : "FAIL"
        ))
    }
}

@main
private enum GenImageMiniMaxMusic3PoC {
    static func main() async {
        do {
            let arguments = try Arguments.parse(Array(CommandLine.arguments.dropFirst()))
            switch arguments.command {
            case .decode:
                try runDecode(arguments)
            case .decodeChunks:
                try runDecodeChunks(arguments)
            case .forward:
                try runForward(arguments)
            case .denoise:
                try runDenoise(arguments)
            case .generate:
                try await runGenerate(arguments)
            case .compare:
                try runCompare(arguments)
            case .level1:
                try await runLevel1(arguments)
            case .languageForward:
                try runLanguageForward(arguments)
            case .attentionProbe:
                try runAttentionProbe(arguments)
            case .conditionForward:
                try runConditionForward(arguments)
            case .rvqForward:
                try runRVQForward(arguments)
            case .rvqEmbeddings:
                try runRVQEmbeddings(arguments)
            case .frameHiddens:
                try runFrameHiddens(arguments)
            }
        } catch {
            fputs("錯誤：\(error.localizedDescription)\n", stderr)
            exit(1)
        }
    }
}

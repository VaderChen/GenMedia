import Darwin
import Foundation
import LTXVideoSwiftRuntime
import MLX

private enum PoCError: LocalizedError {
    case usage(String)
    case missingArgument(String)
    case missingTensor(URL, String)
    case shapeMismatch([Int], [Int])

    var errorDescription: String? {
        switch self {
        case let .usage(message): message
        case let .missingArgument(argument): "參數 \(argument) 後方需要值。"
        case let .missingTensor(url, key): "\(url.path) 缺少 Tensor：\(key)"
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
    case "bfloat16": (1e-2, "bf16 input")
    case "float32": (1e-4, "fp32 input")
    default:
        if referenceDType == .bfloat16 || actualDType == .bfloat16 {
            (1e-2, "bf16 tensor")
        } else {
            (1e-4, "fp32/other tensor")
        }
    }
}

private struct Arguments {
    enum Command: String {
        case decode
        case encode
        case audioDecode = "audio-decode"
        case transformer
        case transformerTiled = "transformer-tiled"
        case gemmaHidden = "gemma-hidden"
        case gemmaFeatures = "gemma-features"
        case upsampler
        case compare
    }

    var command: Command
    var modelDirectory: URL?
    var inputURL: URL?
    var outputURL: URL?
    var referenceURL: URL?
    var actualURL: URL?
    var gemmaDirectory: URL?
    var weightsName: String?
    var key = "frames"
    var latentFrames = 2

    static func parse(_ values: [String]) throws -> Arguments {
        guard let commandValue = values.first,
              let command = Command(rawValue: commandValue) else {
            throw PoCError.usage(usage)
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
            case "--model-dir": result.modelDirectory = url
            case "--input": result.inputURL = url
            case "--output": result.outputURL = url
            case "--reference": result.referenceURL = url
            case "--actual": result.actualURL = url
            case "--gemma-dir": result.gemmaDirectory = url
            case "--weights": result.weightsName = value
            case "--key": result.key = value
            case "--latent-frames":
                guard let latentFrames = Int(value), latentFrames > 0 else {
                    throw PoCError.usage("--latent-frames 必須是正整數。")
                }
                result.latentFrames = latentFrames
            default: throw PoCError.usage("未知參數：\(argument)\n\n\(usage)")
            }
        }
        return result
    }

    static let usage = """
    用法：
      GenImageLTXVideoPoC decode --model-dir <模型目錄> --input <reference.safetensors> --output <actual.safetensors>
      GenImageLTXVideoPoC encode --model-dir <模型目錄> --input <reference.safetensors> --output <actual.safetensors>
      GenImageLTXVideoPoC audio-decode --model-dir <模型目錄> --input <reference.safetensors> --output <actual.safetensors>
      GenImageLTXVideoPoC transformer --model-dir <模型目錄> --weights <相對模型目錄的 transformer.safetensors> --input <reference.safetensors> --output <actual.safetensors>
      GenImageLTXVideoPoC transformer-tiled --model-dir <模型目錄> --weights <相對模型目錄的 transformer.safetensors> --input <reference.safetensors> --output <actual.safetensors>
      GenImageLTXVideoPoC gemma-hidden --gemma-dir <Gemma模型目錄> --input <reference.safetensors> --output <actual.safetensors>
      GenImageLTXVideoPoC gemma-features --model-dir <LTX模型目錄> --input <gemma-hidden.safetensors> --output <actual.safetensors>
      GenImageLTXVideoPoC upsampler --model-dir <模型目錄> --weights <相對模型目錄的 upscaler.safetensors> --input <reference.safetensors> --output <actual.safetensors>
      GenImageLTXVideoPoC compare --reference <reference.safetensors> --actual <actual.safetensors> [--key frames]
    """
}

private func tensor(
    _ key: String,
    from arrays: [String: MLXArray],
    url: URL
) throws -> MLXArray {
    guard let value = arrays[key] else {
        throw PoCError.missingTensor(url, key)
    }
    return value
}

private func intMetadata(_ key: String, _ metadata: [String: String]) throws -> Int {
    guard let value = metadata[key], let integer = Int(value) else {
        throw PoCError.usage("reference metadata 缺少整數 \(key)。")
    }
    return integer
}

private func resolveWeightsURL(_ weightsName: String, relativeTo modelDirectory: URL) throws -> URL {
    guard !weightsName.hasPrefix("/") else {
        throw PoCError.usage("--weights 必須是相對於 --model-dir 的檔名，不可使用完整路徑。")
    }
    let modelRoot = modelDirectory.standardizedFileURL
    let resolved = URL(fileURLWithPath: weightsName, relativeTo: modelRoot).standardizedFileURL
    let rootPath = modelRoot.path.hasSuffix("/") ? modelRoot.path : modelRoot.path + "/"
    guard resolved.path.hasPrefix(rootPath) else {
        throw PoCError.usage("--weights 不可離開 --model-dir 目錄。")
    }
    return resolved
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
    let (arrays, metadata) = try MLX.loadArraysAndMetadata(url: inputURL)
    let latent = try tensor("latent", from: arrays, url: inputURL)
    let declaredDType = metadata["input_dtype"] ?? "bfloat16"
    guard let computeDType = LTXVideoComputeDType(rawValue: declaredDType) else {
        throw PoCError.usage("input_dtype 必須是 bfloat16 或 float32。")
    }
    let tilingMode = metadata["tiling_mode"] ?? "none"
    let tiling: LTXTilingConfiguration?
    switch tilingMode {
    case "none":
        tiling = nil
    case "single", "multi":
        tiling = LTXTilingConfiguration(temporal: try LTXTemporalTilingConfiguration(
            tileSizeInFrames: try intMetadata("tile_size_in_frames", metadata),
            tileOverlapInFrames: try intMetadata("tile_overlap_in_frames", metadata)
        ))
    default:
        throw PoCError.usage("未知 tiling_mode：\(tilingMode)")
    }

    let configuration = try LTXVideoVAEConfiguration.load(from: modelDirectory)
    let decoder = LTXVideoVAEDecoder(configuration: configuration)
    let report = try LTXVideoVAEWeightLoader.loadDecoder(
        model: decoder,
        from: modelDirectory,
        computeDType: computeDType
    )
    let tiles = try LTXVideoTiling.prepareDecoderTiles(
        latentShape: latent.shape,
        configuration: tiling
    )
    let frames = try decoder.decodeTiled(latent, configuration: tiling)
    MLX.eval(latent, frames)
    try FileManager.default.createDirectory(
        at: outputURL.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    var outputMetadata = metadata
    outputMetadata["runtime"] = "swift-mlx-0.31.6"
    outputMetadata["weight_tensors"] = String(report.tensorCount)
    outputMetadata["quantized_modules"] = String(report.quantizedModuleCount)
    outputMetadata["tile_count"] = String(tiles.count)
    try MLX.save(
        arrays: ["latent": latent, "frames": frames],
        metadata: outputMetadata,
        url: outputURL
    )
    print("actual=\(outputURL.path)")
    print("weights=\(report.tensorCount) tensors")
    print("quantized modules=\(report.quantizedModuleCount)")
    print("source dtypes=\(report.sourceDTypes.joined(separator: ","))")
    print("input dtype=\(declaredDType)")
    print("tiling mode=\(tilingMode) tile count=\(tiles.count)")
    print("latent shape=\(latent.shape) dtype=\(latent.dtype)")
    print("frames shape=\(frames.shape) dtype=\(frames.dtype)")
}

private func runEncode(_ arguments: Arguments) throws {
    guard let modelDirectory = arguments.modelDirectory else {
        throw PoCError.missingArgument("--model-dir")
    }
    guard let inputURL = arguments.inputURL else {
        throw PoCError.missingArgument("--input")
    }
    guard let outputURL = arguments.outputURL else {
        throw PoCError.missingArgument("--output")
    }
    let (arrays, metadata) = try MLX.loadArraysAndMetadata(url: inputURL)
    let pixels = try tensor("pixels", from: arrays, url: inputURL)
    let declaredDType = metadata["input_dtype"] ?? "bfloat16"
    guard let computeDType = LTXVideoComputeDType(rawValue: declaredDType) else {
        throw PoCError.usage("input_dtype 必須是 bfloat16 或 float32。")
    }

    let configuration = try LTXVideoVAEConfiguration.load(from: modelDirectory)
    let encoder = LTXVideoVAEEncoder(configuration: configuration)
    let report = try LTXVideoVAEWeightLoader.loadEncoder(
        model: encoder,
        from: modelDirectory,
        computeDType: computeDType
    )
    let latent = try encoder.encode(pixels)
    MLX.eval(pixels, latent)
    try FileManager.default.createDirectory(
        at: outputURL.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    var outputMetadata = metadata
    outputMetadata["runtime"] = "swift-mlx-0.31.6"
    outputMetadata["stage"] = "video_vae_encode"
    outputMetadata["weight_tensors"] = String(report.tensorCount)
    outputMetadata["quantized_modules"] = String(report.quantizedModuleCount)
    try MLX.save(
        arrays: ["pixels": pixels, "latent": latent],
        metadata: outputMetadata,
        url: outputURL
    )
    print("actual=\(outputURL.path)")
    print("weights=\(report.tensorCount) tensors")
    print("quantized modules=\(report.quantizedModuleCount)")
    print("source dtypes=\(report.sourceDTypes.joined(separator: ","))")
    print("input dtype=\(declaredDType)")
    print("pixels shape=\(pixels.shape) dtype=\(pixels.dtype)")
    print("latent shape=\(latent.shape) dtype=\(latent.dtype)")
}

private func runAudioDecode(_ arguments: Arguments) throws {
    guard let modelDirectory = arguments.modelDirectory else {
        throw PoCError.missingArgument("--model-dir")
    }
    guard let inputURL = arguments.inputURL else {
        throw PoCError.missingArgument("--input")
    }
    guard let outputURL = arguments.outputURL else {
        throw PoCError.missingArgument("--output")
    }
    let (arrays, metadata) = try MLX.loadArraysAndMetadata(url: inputURL)
    let latent = try tensor("latent", from: arrays, url: inputURL)
    let declaredDType = metadata["input_dtype"] ?? "bfloat16"
    guard let computeDType = LTXVideoComputeDType(rawValue: declaredDType) else {
        throw PoCError.usage("input_dtype 必須是 bfloat16 或 float32。")
    }
    let configuration = try LTXAudioVAEDecoderConfiguration()
    let decoder = LTXAudioVAEDecoder(configuration: configuration)
    let report = try LTXAudioVAEWeightLoader.loadDecoder(
        model: decoder,
        from: modelDirectory,
        computeDType: computeDType
    )
    let mel = try decoder.decode(latent)
    MLX.eval(latent, mel)
    try FileManager.default.createDirectory(
        at: outputURL.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    var outputMetadata = metadata
    outputMetadata["runtime"] = "swift-mlx-0.31.6"
    outputMetadata["stage"] = "audio_vae_decode"
    outputMetadata["weight_tensors"] = String(report.tensorCount)
    try MLX.save(
        arrays: ["latent": latent, "mel": mel],
        metadata: outputMetadata,
        url: outputURL
    )
    print("actual=\(outputURL.path)")
    print("weights=\(report.tensorCount) tensors")
    print("source dtypes=\(report.sourceDTypes.joined(separator: ","))")
    print("input dtype=\(declaredDType)")
    print("latent shape=\(latent.shape) dtype=\(latent.dtype)")
    print("mel shape=\(mel.shape) dtype=\(mel.dtype)")
}

private func runTransformer(_ arguments: Arguments) throws {
    guard let modelDirectory = arguments.modelDirectory else {
        throw PoCError.missingArgument("--model-dir")
    }
    guard let weightsName = arguments.weightsName else {
        throw PoCError.missingArgument("--weights")
    }
    guard let inputURL = arguments.inputURL else {
        throw PoCError.missingArgument("--input")
    }
    guard let outputURL = arguments.outputURL else {
        throw PoCError.missingArgument("--output")
    }
    let (arrays, metadata) = try MLX.loadArraysAndMetadata(url: inputURL)
    let videoLatent = try tensor("video_latent", from: arrays, url: inputURL)
    let audioLatent = try tensor("audio_latent", from: arrays, url: inputURL)
    let timestep = try tensor("timestep", from: arrays, url: inputURL)
    let videoTextEmbeds = try tensor("video_text_embeds", from: arrays, url: inputURL)
    let audioTextEmbeds = try tensor("audio_text_embeds", from: arrays, url: inputURL)
    let videoPositions = try tensor("video_positions", from: arrays, url: inputURL)
    let audioPositions = try tensor("audio_positions", from: arrays, url: inputURL)
    let declaredDType = metadata["input_dtype"] ?? "bfloat16"
    guard let computeDType = LTXVideoComputeDType(rawValue: declaredDType) else {
        throw PoCError.usage("input_dtype 必須是 bfloat16 或 float32。")
    }

    let report = try LTXTransformerWeightLoader.load(
        from: try resolveWeightsURL(weightsName, relativeTo: modelDirectory),
        modelDirectory: modelDirectory,
        computeDType: computeDType
    )
    let output = report.model(
        videoLatent: videoLatent,
        audioLatent: audioLatent,
        timestep: timestep,
        videoTextEmbeds: videoTextEmbeds,
        audioTextEmbeds: audioTextEmbeds,
        videoPositions: videoPositions,
        audioPositions: audioPositions
    )
    MLX.eval(videoLatent, audioLatent, timestep, videoTextEmbeds, audioTextEmbeds,
             videoPositions, audioPositions, output.video, output.audio)
    let videoRoPE = LTXTransformerOps.precomputeRoPE(
        positions: videoPositions,
        numHeads: 32,
        headDimension: 128,
        theta: 10000,
        maxPositions: [20, 2048, 2048],
        type: .split
    )
    let videoFreqGrid = LTXTransformerOps.generateFreqGrid(
        theta: 10000,
        numPositionDimensions: 3,
        innerDimension: 32 * 128
    )
    MLX.eval(videoRoPE.cos, videoRoPE.sin)
    MLX.eval(videoFreqGrid)

    try FileManager.default.createDirectory(
        at: outputURL.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    var outputMetadata = metadata
    outputMetadata["runtime"] = "swift-mlx-0.31.6"
    outputMetadata["stage"] = "transformer_forward"
    outputMetadata["effective_input_dtype"] = "bfloat16"
    outputMetadata["weight_tensors"] = String(report.report.tensorCount)
    outputMetadata["quantized_modules"] = String(report.report.quantizedModuleCount)
    outputMetadata["source_dtypes"] = report.report.sourceDTypes.joined(separator: ",")
    if let bits = report.report.bits {
        outputMetadata["quantization_bits"] = String(bits)
    }
    if let groupSize = report.report.groupSize {
        outputMetadata["quantization_group_size"] = String(groupSize)
    }
    try MLX.save(
        arrays: [
            "video_latent": videoLatent,
            "audio_latent": audioLatent,
            "timestep": timestep,
            "video_text_embeds": videoTextEmbeds,
            "audio_text_embeds": audioTextEmbeds,
            "video_positions": videoPositions,
            "audio_positions": audioPositions,
            "video_rope_cos": videoRoPE.cos,
            "video_rope_sin": videoRoPE.sin,
            "video_freq_grid": videoFreqGrid,
            "video_velocity": output.video,
            "audio_velocity": output.audio,
        ],
        metadata: outputMetadata,
        url: outputURL
    )
    print("actual=\(outputURL.path)")
    print("weights=\(report.report.tensorCount) tensors")
    print("quantized modules=\(report.report.quantizedModuleCount)")
    print("source dtypes=\(report.report.sourceDTypes.joined(separator: ","))")
    print("video velocity shape=\(output.video.shape) dtype=\(output.video.dtype)")
    print("audio velocity shape=\(output.audio.shape) dtype=\(output.audio.dtype)")
}

private func runUpsampler(_ arguments: Arguments) throws {
    guard let modelDirectory = arguments.modelDirectory else {
        throw PoCError.missingArgument("--model-dir")
    }
    guard let weightsName = arguments.weightsName else {
        throw PoCError.missingArgument("--weights")
    }
    guard let inputURL = arguments.inputURL else {
        throw PoCError.missingArgument("--input")
    }
    guard let outputURL = arguments.outputURL else {
        throw PoCError.missingArgument("--output")
    }
    let (arrays, metadata) = try MLX.loadArraysAndMetadata(url: inputURL)
    let latent = try tensor("latent", from: arrays, url: inputURL)
    let declaredDType = metadata["input_dtype"] ?? "bfloat16"
    guard let computeDType = LTXVideoComputeDType(rawValue: declaredDType) else {
        throw PoCError.usage("input_dtype 必須是 bfloat16 或 float32。")
    }
    let loaded = try LTXLatentUpsamplerWeightLoader.load(
        from: try resolveWeightsURL(weightsName, relativeTo: modelDirectory),
        modelDirectory: modelDirectory,
        computeDType: computeDType
    )
    let upscaled = try loaded.model.upsample(latent)
    MLX.eval(latent, upscaled)
    try FileManager.default.createDirectory(
        at: outputURL.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    var outputMetadata = metadata
    outputMetadata["runtime"] = "swift-mlx-0.31.6"
    outputMetadata["stage"] = "latent_upsampler"
    outputMetadata["effective_input_dtype"] = declaredDType
    outputMetadata["weight_tensors"] = String(loaded.report.tensorCount)
    outputMetadata["quantized_modules"] = String(loaded.report.quantizedModuleCount)
    outputMetadata["source_dtypes"] = loaded.report.sourceDTypes.joined(separator: ",")
    try MLX.save(
        arrays: ["latent": latent, "upscaled_latent": upscaled],
        metadata: outputMetadata,
        url: outputURL
    )
    print("actual=\(outputURL.path)")
    print("weights=\(loaded.report.tensorCount) tensors")
    print("quantized modules=\(loaded.report.quantizedModuleCount)")
    print("source dtypes=\(loaded.report.sourceDTypes.joined(separator: ","))")
    print("input dtype=\(declaredDType)")
    print("latent shape=\(latent.shape) dtype=\(latent.dtype)")
    print("upscaled latent shape=\(upscaled.shape) dtype=\(upscaled.dtype)")
}

private func runTransformerTiled(_ arguments: Arguments) throws {
    guard let modelDirectory = arguments.modelDirectory else {
        throw PoCError.missingArgument("--model-dir")
    }
    guard let weightsName = arguments.weightsName else {
        throw PoCError.missingArgument("--weights")
    }
    guard let inputURL = arguments.inputURL else {
        throw PoCError.missingArgument("--input")
    }
    guard let outputURL = arguments.outputURL else {
        throw PoCError.missingArgument("--output")
    }
    let (arrays, metadata) = try MLX.loadArraysAndMetadata(url: inputURL)
    let videoLatent = try tensor("video_latent", from: arrays, url: inputURL)
    let audioLatent = try tensor("audio_latent", from: arrays, url: inputURL)
    let timestep = try tensor("timestep", from: arrays, url: inputURL)
    let videoTextEmbeds = try tensor("video_text_embeds", from: arrays, url: inputURL)
    let audioTextEmbeds = try tensor("audio_text_embeds", from: arrays, url: inputURL)
    let videoPositions = try tensor("video_positions", from: arrays, url: inputURL)
    let audioPositions = try tensor("audio_positions", from: arrays, url: inputURL)
    let declaredDType = metadata["input_dtype"] ?? "bfloat16"
    guard LTXVideoComputeDType(rawValue: declaredDType) != nil else {
        throw PoCError.usage("input_dtype 必須是 bfloat16 或 float32。")
    }
    let videoFrames = try intMetadata("video_frames", metadata)
    let videoHeight = try intMetadata("video_height", metadata)
    let videoWidth = try intMetadata("video_width", metadata)
    let tileFrames = try intMetadata("tile_frames", metadata)
    let tileFrameOverlap = try intMetadata("tile_frame_overlap", metadata)
    let tileHeight = try intMetadata("tile_height", metadata)
    let tileHeightOverlap = try intMetadata("tile_height_overlap", metadata)
    let tileWidth = try intMetadata("tile_width", metadata)
    let tileWidthOverlap = try intMetadata("tile_width_overlap", metadata)
    let configuration = LTXModalityTileConfiguration(
        frames: try LTXModalityDimensionConfiguration(
            numTiles: tileFrames,
            overlap: tileFrameOverlap
        ),
        height: try LTXModalityDimensionConfiguration(
            numTiles: tileHeight,
            overlap: tileHeightOverlap
        ),
        width: try LTXModalityDimensionConfiguration(
            numTiles: tileWidth,
            overlap: tileWidthOverlap
        )
    )
    let tiles = try LTXModalityTiling.makeTiles(
        frameCount: videoFrames,
        height: videoHeight,
        width: videoWidth,
        configuration: configuration
    )
    let report = try LTXTransformerWeightLoader.load(
        from: try resolveWeightsURL(weightsName, relativeTo: modelDirectory),
        modelDirectory: modelDirectory,
        computeDType: LTXVideoComputeDType(rawValue: declaredDType) ?? .bfloat16
    )

    var videoOutput: MLXArray?
    var audioOutputs: [MLXArray] = []
    for tile in tiles {
        let indices = LTXModalityTiling.tokenIndices(
            tile: tile,
            gridShape: (videoFrames, videoHeight, videoWidth)
        )
        let indexArray = MLXArray(indices.map(Int32.init))
        let tileVideoLatent = videoLatent.take(indexArray, axis: 1)
        let tileVideoPositions = videoPositions[0..., indexArray, 0...]
        let result = report.model(
            videoLatent: tileVideoLatent,
            audioLatent: audioLatent,
            timestep: timestep,
            videoTextEmbeds: videoTextEmbeds,
            audioTextEmbeds: audioTextEmbeds,
            videoPositions: tileVideoPositions,
            audioPositions: audioPositions
        )
        videoOutput = try LTXModalityTiling.blend(
            tileOutput: result.video,
            tile: tile,
            gridShape: (videoFrames, videoHeight, videoWidth),
            output: videoOutput
        )
        audioOutputs.append(result.audio)
    }
    guard let videoOutput, let firstAudio = audioOutputs.first else {
        throw PoCError.usage("沒有可供輸出的 modality tile。")
    }
    let audioOutput = audioOutputs.count == 1
        ? firstAudio
        : MLX.stacked(audioOutputs).mean(axis: 0)
    MLX.eval(videoLatent, audioLatent, timestep, videoTextEmbeds, audioTextEmbeds,
             videoPositions, audioPositions, videoOutput, audioOutput)

    try FileManager.default.createDirectory(
        at: outputURL.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    var outputMetadata = metadata
    outputMetadata["runtime"] = "swift-mlx-0.31.6"
    outputMetadata["stage"] = "transformer_tiled"
    outputMetadata["effective_input_dtype"] = "bfloat16"
    outputMetadata["tile_count"] = String(tiles.count)
    outputMetadata["weight_tensors"] = String(report.report.tensorCount)
    outputMetadata["quantized_modules"] = String(report.report.quantizedModuleCount)
    try MLX.save(
        arrays: [
            "video_latent": videoLatent,
            "audio_latent": audioLatent,
            "timestep": timestep,
            "video_text_embeds": videoTextEmbeds,
            "audio_text_embeds": audioTextEmbeds,
            "video_positions": videoPositions,
            "audio_positions": audioPositions,
            "video_velocity": videoOutput,
            "audio_velocity": audioOutput,
        ],
        metadata: outputMetadata,
        url: outputURL
    )
    print("actual=\(outputURL.path)")
    print("weights=\(report.report.tensorCount) tensors")
    print("quantized modules=\(report.report.quantizedModuleCount)")
    print("input dtype=\(declaredDType) effective dtype=bfloat16")
    print("video patchified token count=\(videoFrames * videoHeight * videoWidth)")
    print("video token range=0..<\(videoFrames * videoHeight * videoWidth)")
    print("audio token range=0..<\(audioLatent.shape[1])")
    print("video RoPE position range=frame 0..<\(videoFrames), height 0..<\(videoHeight), width 0..<\(videoWidth)")
    print("tile count=\(tiles.count)")
    print("video velocity shape=\(videoOutput.shape) dtype=\(videoOutput.dtype)")
    print("audio velocity shape=\(audioOutput.shape) dtype=\(audioOutput.dtype)")
}

private func runGemmaHidden(_ arguments: Arguments) throws {
    guard let gemmaDirectory = arguments.gemmaDirectory else {
        throw PoCError.missingArgument("--gemma-dir")
    }
    guard let inputURL = arguments.inputURL else {
        throw PoCError.missingArgument("--input")
    }
    guard let outputURL = arguments.outputURL else {
        throw PoCError.missingArgument("--output")
    }
    let (arrays, metadata) = try MLX.loadArraysAndMetadata(url: inputURL)
    let tokenIDs = try tensor("token_ids", from: arrays, url: inputURL)
    let attentionMask = arrays["attention_mask"]
    let declaredDType = metadata["input_dtype"] ?? "bfloat16"
    guard let computeDType = LTXVideoComputeDType(rawValue: declaredDType) else {
        throw PoCError.usage("input_dtype 必須是 bfloat16 或 float32。")
    }
    let loaded = try LTXGemma3TextWeightLoader.load(
        from: gemmaDirectory,
        computeDType: computeDType
    )
    let states = try loaded.model.allHiddenStates(
        tokenIDs: tokenIDs,
        attentionMask: attentionMask
    )
    MLX.eval(tokenIDs, states)
    try FileManager.default.createDirectory(
        at: outputURL.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    var outputMetadata = metadata
    outputMetadata["runtime"] = "swift-mlx-0.31.6"
    outputMetadata["stage"] = "gemma3_all_hidden_states"
    outputMetadata["hidden_state_count"] = String(states.count)
    outputMetadata["hidden_size"] = String(loaded.model.configuration.hiddenSize)
    outputMetadata["weight_tensors"] = String(loaded.report.tensorCount)
    outputMetadata["quantized_modules"] = String(loaded.report.quantizedModuleCount)
    outputMetadata["source_dtypes"] = loaded.report.sourceDTypes.joined(separator: ",")
    outputMetadata["effective_input_dtype"] = declaredDType
    var outputArrays: [String: MLXArray] = ["token_ids": tokenIDs]
    if let attentionMask {
        outputArrays["attention_mask"] = attentionMask
    }
    for (index, state) in states.enumerated() {
        outputArrays["hidden_\(index)"] = state
    }
    try MLX.save(arrays: outputArrays, metadata: outputMetadata, url: outputURL)
    print("actual=\(outputURL.path)")
    print("weights=\(loaded.report.tensorCount) tensors")
    print("quantized modules=\(loaded.report.quantizedModuleCount)")
    print("input dtype=\(declaredDType)")
    print("hidden states=\(states.count) hidden size=\(loaded.model.configuration.hiddenSize)")
    print("token ids shape=\(tokenIDs.shape) dtype=\(tokenIDs.dtype)")
}

private func runGemmaFeatures(_ arguments: Arguments) throws {
    guard let modelDirectory = arguments.modelDirectory else {
        throw PoCError.missingArgument("--model-dir")
    }
    guard let inputURL = arguments.inputURL else {
        throw PoCError.missingArgument("--input")
    }
    guard let outputURL = arguments.outputURL else {
        throw PoCError.missingArgument("--output")
    }
    let (arrays, metadata) = try MLX.loadArraysAndMetadata(url: inputURL)
    var hiddenStates = [MLXArray]()
    hiddenStates.reserveCapacity(LTXGemmaFeaturePreparation.hiddenStateCount)
    for index in 0..<LTXGemmaFeaturePreparation.hiddenStateCount {
        hiddenStates.append(try tensor("hidden_\(index)", from: arrays, url: inputURL))
    }
    let attentionMask = arrays["attention_mask"]
    let stacked = try LTXGemmaFeaturePreparation.stackForProjection(
        hiddenStates,
        attentionMask: attentionMask
    )
    let connector = LTXGemmaTextEncoderConnector()
    let report = try LTXGemmaConnectorWeightLoader.load(
        connector: connector,
        from: modelDirectory,
        computeDType: LTXVideoComputeDType(rawValue: metadata["input_dtype"] ?? "bfloat16") ?? .bfloat16
    )
    let features = connector(stacked, attentionMask: attentionMask)
    MLX.eval(stacked, features.video, features.audio)
    try FileManager.default.createDirectory(
        at: outputURL.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    var outputMetadata = metadata
    outputMetadata["runtime"] = "swift-mlx-0.31.6"
    outputMetadata["stage"] = "gemma3_features"
    outputMetadata["weight_tensors"] = String(report.tensorCount)
    outputMetadata["quantized_modules"] = "0"
    outputMetadata["source_dtypes"] = report.sourceDTypes.joined(separator: ",")
    let effectiveDType = metadata["input_dtype"] ?? "bfloat16"
    guard LTXVideoComputeDType(rawValue: effectiveDType) != nil else {
        throw PoCError.usage("input_dtype 必須是 bfloat16 或 float32。")
    }
    outputMetadata["effective_input_dtype"] = effectiveDType
    try MLX.save(
        arrays: [
            "video_text_embeds": features.video,
            "audio_text_embeds": features.audio,
            "stacked_hidden_states": stacked,
        ],
        metadata: outputMetadata,
        url: outputURL
    )
    print("actual=\(outputURL.path)")
    print("weights=\(report.tensorCount) tensors")
    print("quantized modules=0")
    print("video text embeds shape=\(features.video.shape) dtype=\(features.video.dtype)")
    print("audio text embeds shape=\(features.audio.shape) dtype=\(features.audio.dtype)")
}

// Kept line-for-line equivalent to the MiniMax Music 3 PoC comparison path:
// Double accumulation and relative max diff are the acceptance signal.
private func runCompare(_ arguments: Arguments) throws {
    guard let referenceURL = arguments.referenceURL else {
        throw PoCError.missingArgument("--reference")
    }
    guard let actualURL = arguments.actualURL else {
        throw PoCError.missingArgument("--actual")
    }
    let (referenceArrays, referenceMetadata) = try MLX.loadArraysAndMetadata(url: referenceURL)
    let (actualArrays, actualMetadata) = try MLX.loadArraysAndMetadata(url: actualURL)
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
    let declaredInputDType = referenceMetadata["effective_input_dtype"]
        ?? actualMetadata["effective_input_dtype"]
        ?? referenceMetadata["input_dtype"]
        ?? actualMetadata["input_dtype"]
    let threshold = comparisonThreshold(
        referenceDType: reference.dtype,
        actualDType: actual.dtype,
        declaredInputDType: declaredInputDType
    )
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
    print(String(format: "relative max diff: %.10g", relativeMaxDiff))
    print(String(format: "max abs diff: %.10g", maxAbsDiff))
    print(String(format: "mean abs diff: %.10g", meanAbsDiff))
    print(String(format: "cosine similarity: %.10g", cosine))
    print("non-finite mismatches: \(nonFiniteMismatchCount)")
    print(String(
        format: "relative max diff threshold: %.10g [%@] (%@)",
        threshold.value,
        threshold.label,
        nonFiniteMismatchCount == 0 && relativeMaxDiff <= threshold.value ? "PASS" : "FAIL"
    ))
}

@main
private enum GenImageLTXVideoPoC {
    static func main() {
        do {
            let arguments = try Arguments.parse(Array(CommandLine.arguments.dropFirst()))
            switch arguments.command {
            case .decode: try runDecode(arguments)
            case .encode: try runEncode(arguments)
            case .audioDecode: try runAudioDecode(arguments)
            case .transformer: try runTransformer(arguments)
            case .transformerTiled: try runTransformerTiled(arguments)
            case .gemmaHidden: try runGemmaHidden(arguments)
            case .gemmaFeatures: try runGemmaFeatures(arguments)
            case .upsampler: try runUpsampler(arguments)
            case .compare: try runCompare(arguments)
            }
        } catch {
            fputs("錯誤：\(error.localizedDescription)\n", stderr)
            exit(1)
        }
    }
}

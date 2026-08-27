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
        case compare
    }

    var command: Command
    var modelDirectory: URL?
    var inputURL: URL?
    var outputURL: URL?
    var referenceURL: URL?
    var actualURL: URL?
    var key = "frames"

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
            case "--key": result.key = value
            default: throw PoCError.usage("未知參數：\(argument)\n\n\(usage)")
            }
        }
        return result
    }

    static let usage = """
    用法：
      GenImageLTXVideoPoC decode --model-dir <模型目錄> --input <reference.safetensors> --output <actual.safetensors>
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
    let declaredInputDType = referenceMetadata["input_dtype"] ?? actualMetadata["input_dtype"]
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
            case .compare: try runCompare(arguments)
            }
        } catch {
            fputs("錯誤：\(error.localizedDescription)\n", stderr)
            exit(1)
        }
    }
}

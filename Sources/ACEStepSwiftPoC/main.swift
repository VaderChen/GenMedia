import Darwin
import ACEStepSwiftRuntime
import Foundation
import MLX
import MLXNN

private enum PoCError: LocalizedError {
    case invalidArguments(String)
    case modelRootNotFound([URL])
    case missingFile(URL)
    case invalidJSON(URL)
    case invalidSafeTensors(URL, String)
    case missingTensor(URL, String)

    var errorDescription: String? {
        switch self {
        case let .invalidArguments(message):
            message
        case let .modelRootNotFound(candidates):
            "找不到 ACE-Step 模型目錄。已檢查：\n" + candidates.map(\.path).joined(separator: "\n")
        case let .missingFile(url):
            "缺少檔案：\(url.path)"
        case let .invalidJSON(url):
            "JSON 格式無法解析：\(url.path)"
        case let .invalidSafeTensors(url, reason):
            "safetensors 格式無法解析：\(url.path)（\(reason)）"
        case let .missingTensor(url, name):
            "權重缺少必要 Tensor：\(name)（\(url.path)）"
        }
    }
}

private enum ModelComponent: String, CaseIterable {
    case dit
    case embedding
    case vae
    case languageModel = "language-model"

    var displayName: String {
        switch self {
        case .dit: "ACE-Step Turbo DiT"
        case .embedding: "Qwen3 Embedding"
        case .vae: "Oobleck VAE"
        case .languageModel: "ACE-Step 5Hz Language Model"
        }
    }

    var relativeWeightPath: String {
        switch self {
        case .dit: "acestep-v15-turbo/model.safetensors"
        case .embedding: "Qwen3-Embedding-0.6B/model.safetensors"
        case .vae: "vae/diffusion_pytorch_model.safetensors"
        case .languageModel: "acestep-5Hz-lm-1.7B/model.safetensors"
        }
    }

    var relativeConfigPath: String {
        switch self {
        case .dit: "acestep-v15-turbo/config.json"
        case .embedding: "Qwen3-Embedding-0.6B/config.json"
        case .vae: "vae/config.json"
        case .languageModel: "acestep-5Hz-lm-1.7B/config.json"
        }
    }

    var isRequiredForDirectConditioning: Bool {
        self != .languageModel
    }

    var representativeTensorNames: [String] {
        switch self {
        case .dit:
            [
                "decoder.proj_in.1.weight",
                "decoder.proj_out.1.weight",
                "decoder.layers.0.self_attn.q_proj.weight"
            ]
        case .embedding:
            [
                "embed_tokens.weight",
                "layers.0.self_attn.q_proj.weight",
                "norm.weight"
            ]
        case .vae:
            [
                "decoder.conv1.weight_g",
                "decoder.conv1.weight_v",
                "decoder.block.0.conv_t1.weight_g",
                "decoder.block.0.conv_t1.weight_v"
            ]
        case .languageModel:
            []
        }
    }
}

private struct Arguments {
    var explicitModelRoot: URL?
    var loadComponent: ModelComponent?
    var vaeOutputURL: URL?
    var latentFrames = 8
    var prompt: String?
    var lyrics = ""
    var language = "en"
    var embeddingOutputURL: URL?
    var conditionOutputURL: URL?
    var conditionInputURL: URL?
    var conditionFrames = 128
    var ditOutputURL: URL?
    var generatedAudioURL: URL?
    var seed: UInt64 = 42
    var inferenceSteps = 8
    var shift: Float = 3
    var listedTensorCount = 0
    var showHelp = false

    static func parse(_ values: [String]) throws -> Arguments {
        var result = Arguments()
        var index = 0
        while index < values.count {
            switch values[index] {
            case "--model-root":
                index += 1
                guard index < values.count else {
                    throw PoCError.invalidArguments("--model-root 後方需要目錄路徑。")
                }
                result.explicitModelRoot = URL(
                    fileURLWithPath: (values[index] as NSString).expandingTildeInPath,
                    isDirectory: true
                )
            case "--load-component":
                index += 1
                guard index < values.count,
                      let component = ModelComponent(rawValue: values[index]) else {
                    let choices = ModelComponent.allCases.map(\.rawValue).joined(separator: ", ")
                    throw PoCError.invalidArguments("--load-component 可用值：\(choices)")
                }
                result.loadComponent = component
            case "--list-tensors":
                index += 1
                guard index < values.count,
                      let count = Int(values[index]),
                      count >= 0 else {
                    throw PoCError.invalidArguments("--list-tensors 後方需要零或正整數。")
                }
                result.listedTensorCount = count
            case "--decode-vae":
                index += 1
                guard index < values.count else {
                    throw PoCError.invalidArguments("--decode-vae 後方需要 WAV 輸出路徑。")
                }
                result.vaeOutputURL = URL(
                    fileURLWithPath: (values[index] as NSString).expandingTildeInPath
                )
            case "--latent-frames":
                index += 1
                guard index < values.count,
                      let count = Int(values[index]),
                      (1...250).contains(count) else {
                    throw PoCError.invalidArguments("--latent-frames 需要 1 至 250。")
                }
                result.latentFrames = count
            case "--encode-prompt":
                index += 1
                guard index < values.count, !values[index].isEmpty else {
                    throw PoCError.invalidArguments("--encode-prompt 後方需要 Prompt。")
                }
                result.prompt = values[index]
            case "--lyrics":
                index += 1
                guard index < values.count else {
                    throw PoCError.invalidArguments("--lyrics 後方需要歌詞，可傳入空字串。")
                }
                result.lyrics = values[index]
            case "--language":
                index += 1
                guard index < values.count, !values[index].isEmpty else {
                    throw PoCError.invalidArguments("--language 後方需要語言代碼。")
                }
                result.language = values[index]
            case "--embedding-output":
                index += 1
                guard index < values.count else {
                    throw PoCError.invalidArguments("--embedding-output 後方需要 safetensors 路徑。")
                }
                result.embeddingOutputURL = URL(
                    fileURLWithPath: (values[index] as NSString).expandingTildeInPath
                )
            case "--encode-condition":
                index += 1
                guard index < values.count else {
                    throw PoCError.invalidArguments("--encode-condition 後方需要 safetensors 路徑。")
                }
                result.conditionOutputURL = URL(
                    fileURLWithPath: (values[index] as NSString).expandingTildeInPath
                )
            case "--condition-frames":
                index += 1
                guard index < values.count,
                      let count = Int(values[index]),
                      (128...15_000).contains(count) else {
                    throw PoCError.invalidArguments("--condition-frames 需要 128 至 15000。")
                }
                result.conditionFrames = count
            case "--condition-input":
                index += 1
                guard index < values.count else {
                    throw PoCError.invalidArguments("--condition-input 後方需要 safetensors 路徑。")
                }
                result.conditionInputURL = URL(
                    fileURLWithPath: (values[index] as NSString).expandingTildeInPath
                )
            case "--dit-forward":
                index += 1
                guard index < values.count else {
                    throw PoCError.invalidArguments("--dit-forward 後方需要 safetensors 路徑。")
                }
                result.ditOutputURL = URL(
                    fileURLWithPath: (values[index] as NSString).expandingTildeInPath
                )
            case "--seed":
                index += 1
                guard index < values.count, let seed = UInt64(values[index]) else {
                    throw PoCError.invalidArguments("--seed 後方需要零或正整數。")
                }
                result.seed = seed
            case "--generate-audio":
                index += 1
                guard index < values.count else {
                    throw PoCError.invalidArguments("--generate-audio 後方需要 WAV 路徑。")
                }
                result.generatedAudioURL = URL(
                    fileURLWithPath: (values[index] as NSString).expandingTildeInPath
                )
            case "--inference-steps":
                index += 1
                guard index < values.count,
                      let steps = Int(values[index]),
                      (1...20).contains(steps) else {
                    throw PoCError.invalidArguments("--inference-steps 需要 1 至 20。")
                }
                result.inferenceSteps = steps
            case "--shift":
                index += 1
                guard index < values.count,
                      let shift = Float(values[index]),
                      shift > 0 else {
                    throw PoCError.invalidArguments("--shift 需要大於 0。")
                }
                result.shift = shift
            case "--help", "-h":
                result.showHelp = true
            default:
                throw PoCError.invalidArguments("不支援的參數：\(values[index])")
            }
            index += 1
        }
        if result.conditionOutputURL != nil, result.prompt == nil {
            throw PoCError.invalidArguments("--encode-condition 必須搭配 --encode-prompt。")
        }
        if result.ditOutputURL != nil,
           result.conditionInputURL == nil,
           result.conditionOutputURL == nil {
            throw PoCError.invalidArguments(
                "--dit-forward 必須搭配 --condition-input 或 --encode-condition。"
            )
        }
        if result.generatedAudioURL != nil,
           result.conditionInputURL == nil,
           result.conditionOutputURL == nil {
            throw PoCError.invalidArguments(
                "--generate-audio 必須搭配 --condition-input 或 --encode-condition。"
            )
        }
        return result
    }
}

private struct JSONDocument {
    let values: [String: Any]

    static func load(from url: URL) throws -> JSONDocument {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw PoCError.missingFile(url)
        }
        let data = try Data(contentsOf: url)
        guard let values = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw PoCError.invalidJSON(url)
        }
        return JSONDocument(values: values)
    }

    func string(_ key: String) -> String? {
        values[key] as? String
    }

    func integer(_ key: String) -> Int? {
        (values[key] as? NSNumber)?.intValue
    }

    func boolean(_ key: String) -> Bool? {
        (values[key] as? NSNumber)?.boolValue
    }
}

private struct SafeTensorDescriptor {
    let name: String
    let dtype: String
    let shape: [Int]
    let dataOffsets: [UInt64]

    var byteCount: UInt64 {
        dataOffsets[1] - dataOffsets[0]
    }
}

private struct SafeTensorHeader {
    let fileURL: URL
    let tensors: [SafeTensorDescriptor]

    static func load(from url: URL) throws -> SafeTensorHeader {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw PoCError.missingFile(url)
        }
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        guard let prefix = try handle.read(upToCount: 8), prefix.count == 8 else {
            throw PoCError.invalidSafeTensors(url, "缺少 8-byte header length")
        }
        let headerLength = prefix.enumerated().reduce(UInt64(0)) { partial, item in
            partial | (UInt64(item.element) << UInt64(item.offset * 8))
        }
        guard headerLength > 0, headerLength <= 64 * 1_024 * 1_024 else {
            throw PoCError.invalidSafeTensors(url, "header length 不合理：\(headerLength)")
        }
        guard let headerData = try handle.read(upToCount: Int(headerLength)),
              headerData.count == Int(headerLength),
              let object = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            throw PoCError.invalidSafeTensors(url, "header JSON 不完整")
        }

        var tensors: [SafeTensorDescriptor] = []
        for (name, rawValue) in object where name != "__metadata__" {
            guard let descriptor = rawValue as? [String: Any],
                  let dtype = descriptor["dtype"] as? String,
                  let shapeValues = descriptor["shape"] as? [NSNumber],
                  let offsetValues = descriptor["data_offsets"] as? [NSNumber],
                  offsetValues.count == 2 else {
                throw PoCError.invalidSafeTensors(url, "Tensor \(name) 描述不完整")
            }
            let offsets = offsetValues.map(\.uint64Value)
            guard offsets[0] <= offsets[1] else {
                throw PoCError.invalidSafeTensors(url, "Tensor \(name) offset 顛倒")
            }
            tensors.append(
                SafeTensorDescriptor(
                    name: name,
                    dtype: dtype,
                    shape: shapeValues.map(\.intValue),
                    dataOffsets: offsets
                )
            )
        }
        guard !tensors.isEmpty else {
            throw PoCError.invalidSafeTensors(url, "沒有 Tensor")
        }
        return SafeTensorHeader(fileURL: url, tensors: tensors.sorted { $0.name < $1.name })
    }

    var totalTensorBytes: UInt64 {
        tensors.reduce(UInt64(0)) { $0 + $1.byteCount }
    }

    func contains(_ name: String) -> Bool {
        tensors.contains { $0.name == name }
    }
}

private enum ModelRootResolver {
    static func resolve(explicit: URL?) throws -> URL {
        if let explicit {
            return explicit.standardizedFileURL
        }
        let fileManager = FileManager.default
        let environment = ProcessInfo.processInfo.environment
        let home = fileManager.homeDirectoryForCurrentUser
        var candidates: [URL] = []
        for key in ["GENMEDIA_ACESTEP_MODEL_ROOT", "GENIMAGE_ACESTEP_MODEL_ROOT"] {
            if let value = environment[key], !value.isEmpty {
                candidates.append(
                    URL(
                        fileURLWithPath: (value as NSString).expandingTildeInPath,
                        isDirectory: true
                    )
                )
            }
        }
        if let value = environment["GENIMAGE_MODEL_ROOT"], !value.isEmpty {
            candidates.append(
                URL(
                    fileURLWithPath: (value as NSString).expandingTildeInPath,
                    isDirectory: true
                ).appendingPathComponent("ace-step-1.5-turbo", isDirectory: true)
            )
        }
        candidates.append(
            home.appendingPathComponent("AI Modes/GenImage/ace-step-1.5-turbo", isDirectory: true)
        )
        candidates.append(
            home.appendingPathComponent(
                "Library/Application Support/GenImage/Models/ace-step-1.5-turbo",
                isDirectory: true
            )
        )
        for candidate in candidates {
            let configURL = candidate.appendingPathComponent("config.json")
            if fileManager.fileExists(atPath: configURL.path) {
                return candidate.standardizedFileURL
            }
        }
        throw PoCError.modelRootNotFound(candidates)
    }
}

private enum TextOutput {
    static func section(_ title: String) {
        print("\n=== \(title) ===")
    }

    static func success(_ message: String) {
        print("[通過] \(message)")
    }

    static func notice(_ message: String) {
        print("[資訊] \(message)")
    }

    static func warning(_ message: String) {
        print("[注意] \(message)")
    }

    static func bytes(_ count: UInt64) -> String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useKB, .useMB, .useGB]
        formatter.countStyle = .file
        return formatter.string(fromByteCount: Int64(clamping: count))
    }
}

private struct CompatibilityResult {
    let headers: [ModelComponent: SafeTensorHeader]
}

private enum ACEStepCompatibilityProbe {
    static func run(modelRoot: URL, listedTensorCount: Int) throws -> CompatibilityResult {
        TextOutput.section("執行環境")
        #if arch(arm64)
        TextOutput.success("目前為 Apple Silicon arm64")
        #else
        TextOutput.warning("目前不是 arm64，ACE-Step MLX 正式整合不應使用此架構")
        #endif
        TextOutput.notice("模型根目錄：\(modelRoot.path)")

        let rootConfig = try JSONDocument.load(
            from: modelRoot.appendingPathComponent("config.json")
        )
        TextOutput.section("模型設定")
        TextOutput.success(
            "主模型：type=\(rootConfig.string("model_type") ?? "未知")、"
                + "version=\(rootConfig.string("model_version") ?? "未知")、"
                + "dtype=\(rootConfig.string("dtype") ?? "未知")"
        )
        TextOutput.notice(
            "DiT：hidden=\(rootConfig.integer("hidden_size") ?? -1)、"
                + "layers=\(rootConfig.integer("num_hidden_layers") ?? -1)、"
                + "heads=\(rootConfig.integer("num_attention_heads") ?? -1)、"
                + "turbo=\(rootConfig.boolean("is_turbo") ?? false)"
        )

        let embeddingRoot = modelRoot.appendingPathComponent(
            "Qwen3-Embedding-0.6B",
            isDirectory: true
        )
        let tokenizerConfig = try JSONDocument.load(
            from: embeddingRoot.appendingPathComponent("tokenizer_config.json")
        )
        _ = try JSONDocument.load(from: embeddingRoot.appendingPathComponent("tokenizer.json"))
        TextOutput.success(
            "Qwen tokenizer 資產可解析：\(tokenizerConfig.string("tokenizer_class") ?? "未標示類別")"
        )

        var headers: [ModelComponent: SafeTensorHeader] = [:]
        TextOutput.section("safetensors 索引")
        for component in ModelComponent.allCases {
            let configURL = modelRoot.appendingPathComponent(component.relativeConfigPath)
            let weightURL = modelRoot.appendingPathComponent(component.relativeWeightPath)
            let configExists = FileManager.default.fileExists(atPath: configURL.path)
            let weightExists = FileManager.default.fileExists(atPath: weightURL.path)
            if !configExists || !weightExists {
                if component.isRequiredForDirectConditioning {
                    throw PoCError.missingFile(configExists ? weightURL : configURL)
                }
                TextOutput.notice("略過選配的 \(component.displayName)")
                continue
            }
            _ = try JSONDocument.load(from: configURL)
            let header = try SafeTensorHeader.load(from: weightURL)
            try validateRepresentativeTensors(component: component, header: header)
            headers[component] = header
            let dtypeSummary = Dictionary(grouping: header.tensors, by: \.dtype)
                .map { "\($0.key):\($0.value.count)" }
                .sorted()
                .joined(separator: ", ")
            TextOutput.success(
                "\(component.displayName)：\(header.tensors.count) tensors、"
                    + "\(TextOutput.bytes(header.totalTensorBytes))、\(dtypeSummary)"
            )
            if listedTensorCount > 0 {
                for tensor in header.tensors.prefix(listedTensorCount) {
                    let shape = tensor.shape.map(String.init).joined(separator: "×")
                    print("  - \(tensor.name) [\(shape)] \(tensor.dtype)")
                }
            }
        }

        reportConversionRequirements(headers: headers)
        return CompatibilityResult(headers: headers)
    }

    private static func validateRepresentativeTensors(
        component: ModelComponent,
        header: SafeTensorHeader
    ) throws {
        for name in component.representativeTensorNames where !header.contains(name) {
            throw PoCError.missingTensor(header.fileURL, name)
        }
    }

    private static func reportConversionRequirements(
        headers: [ModelComponent: SafeTensorHeader]
    ) {
        TextOutput.section("Swift 權重轉換")
        if let dit = headers[.dit] {
            let decoderCount = dit.tensors.filter { $0.name.hasPrefix("decoder.") }.count
            TextOutput.success("DiT decoder 可分離：\(decoderCount) tensors")
            TextOutput.notice("proj_in 需 [out,in,k] → [out,k,in]")
            TextOutput.notice("proj_out 需 [in,out,k] → [out,k,in]")
        }
        if let vae = headers[.vae] {
            let scaleCount = vae.tensors.filter { $0.name.hasSuffix(".weight_g") }.count
            let vectorCount = vae.tensors.filter { $0.name.hasSuffix(".weight_v") }.count
            if scaleCount == vectorCount {
                TextOutput.success("VAE weight normalization 配對完整：\(scaleCount) 組")
            } else {
                TextOutput.warning("VAE weight_g / weight_v 數量不一致：\(scaleCount) / \(vectorCount)")
            }
        }
        if let embedding = headers[.embedding] {
            let layerIndexes = Set(embedding.tensors.compactMap { tensor -> Int? in
                let parts = tensor.name.split(separator: ".")
                guard parts.count > 2, parts[0] == "layers" else { return nil }
                return Int(parts[1])
            })
            TextOutput.success("Qwen3 Embedding 可辨識 \(layerIndexes.count) 層 Transformer")
        }
    }
}

private enum MLXOperatorProbe {
    static func run() throws {
        TextOutput.section("MLX Swift 算子")
        let signal = MLXArray.zeros([1, 32, 4], dtype: .float32)
        let convolution = Conv1d(
            inputChannels: 4,
            outputChannels: 8,
            kernelSize: 3,
            padding: 1
        )
        let transposedConvolution = ConvTransposed1d(
            inputChannels: 8,
            outputChannels: 4,
            kernelSize: 4,
            stride: 2,
            padding: 1
        )
        let encoded = convolution(signal)
        let decoded = transposedConvolution(encoded)

        let queries = MLXArray.zeros([1, 2, 8, 16], dtype: .float32)
        let keys = MLXArray.zeros([1, 2, 8, 16], dtype: .float32)
        let values = MLXArray.zeros([1, 2, 8, 16], dtype: .float32)
        let attention = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: 1 / sqrt(Float(16)),
            mask: nil
        )
        let normalized = RMSNorm(dimensions: 16, eps: 1e-6)(attention)
        MLX.eval(decoded, normalized)

        guard encoded.shape == [1, 32, 8], decoded.shape == [1, 64, 4] else {
            throw PoCError.invalidArguments(
                "Conv1D shape 異常：encoded=\(encoded.shape)、decoded=\(decoded.shape)"
            )
        }
        guard normalized.shape == [1, 2, 8, 16] else {
            throw PoCError.invalidArguments("Attention shape 異常：\(normalized.shape)")
        }
        TextOutput.success("Conv1D：\(signal.shape) → \(encoded.shape)")
        TextOutput.success("ConvTransposed1D：\(encoded.shape) → \(decoded.shape)")
        TextOutput.success("Scaled Dot Product Attention + RMSNorm：\(normalized.shape)")
    }
}

private enum MLXWeightProbe {
    static func run(component: ModelComponent, header: SafeTensorHeader) throws {
        TextOutput.section("MLX 實體權重載入")
        TextOutput.warning("正在以 MLX 直接開啟大型權重；此流程完全不啟動 Python。")
        let arrays = try MLX.loadArrays(url: header.fileURL)
        switch component {
        case .dit:
            try materializeDiT(arrays: arrays, sourceURL: header.fileURL)
        case .vae:
            try materializeVAE(arrays: arrays, sourceURL: header.fileURL)
        case .embedding, .languageModel:
            try materializeSmallTensor(arrays: arrays, header: header)
        }
    }

    private static func materializeDiT(
        arrays: [String: MLXArray],
        sourceURL: URL
    ) throws {
        let inputName = "decoder.proj_in.1.weight"
        let outputName = "decoder.proj_out.1.weight"
        guard let inputWeight = arrays[inputName] else {
            throw PoCError.missingTensor(sourceURL, inputName)
        }
        guard let outputWeight = arrays[outputName] else {
            throw PoCError.missingTensor(sourceURL, outputName)
        }
        let convertedInput = inputWeight.transposed(0, 2, 1)
        let convertedOutput = outputWeight.transposed(1, 2, 0)
        MLX.eval(convertedInput, convertedOutput)
        TextOutput.success("proj_in：\(inputWeight.shape) → \(convertedInput.shape)")
        TextOutput.success("proj_out：\(outputWeight.shape) → \(convertedOutput.shape)")
    }

    private static func materializeVAE(
        arrays: [String: MLXArray],
        sourceURL: URL
    ) throws {
        let regular = try fusedWeight(
            scaleName: "decoder.conv1.weight_g",
            vectorName: "decoder.conv1.weight_v",
            arrays: arrays,
            sourceURL: sourceURL,
            transposedConvolution: false
        )
        let transposed = try fusedWeight(
            scaleName: "decoder.block.0.conv_t1.weight_g",
            vectorName: "decoder.block.0.conv_t1.weight_v",
            arrays: arrays,
            sourceURL: sourceURL,
            transposedConvolution: true
        )
        MLX.eval(regular, transposed)
        TextOutput.success("VAE Conv1D fused weight：\(regular.shape)")
        TextOutput.success("VAE ConvTransposed1D fused weight：\(transposed.shape)")
    }

    private static func fusedWeight(
        scaleName: String,
        vectorName: String,
        arrays: [String: MLXArray],
        sourceURL: URL,
        transposedConvolution: Bool
    ) throws -> MLXArray {
        guard let scale = arrays[scaleName] else {
            throw PoCError.missingTensor(sourceURL, scaleName)
        }
        guard let vector = arrays[vectorName] else {
            throw PoCError.missingTensor(sourceURL, vectorName)
        }
        let floatScale = scale.asType(.float32)
        let floatVector = vector.asType(.float32)
        let axes = Array(1..<floatVector.ndim)
        let norm = floatVector.square().sum(axes: axes, keepDims: true).sqrt()
        let fused = floatScale * floatVector / (norm + 1e-9)
        if transposedConvolution {
            return fused.transposed(1, 2, 0)
        }
        return fused.transposed(0, 2, 1)
    }

    private static func materializeSmallTensor(
        arrays: [String: MLXArray],
        header: SafeTensorHeader
    ) throws {
        guard let descriptor = header.tensors.min(by: { $0.byteCount < $1.byteCount }),
              let tensor = arrays[descriptor.name] else {
            throw PoCError.invalidSafeTensors(header.fileURL, "無法選取代表 Tensor")
        }
        MLX.eval(tensor)
        TextOutput.success(
            "已載入 \(descriptor.name)：shape=\(tensor.shape)、dtype=\(tensor.dtype)"
        )
    }
}

@main
private enum ACEStepSwiftPoC {
    static func main() {
        do {
            let arguments = try Arguments.parse(Array(CommandLine.arguments.dropFirst()))
            if arguments.showHelp {
                printUsage()
                return
            }
            let modelRoot = try ModelRootResolver.resolve(explicit: arguments.explicitModelRoot)
            let result = try ACEStepCompatibilityProbe.run(
                modelRoot: modelRoot,
                listedTensorCount: arguments.listedTensorCount
            )
            try MLXOperatorProbe.run()
            if let component = arguments.loadComponent {
                guard let header = result.headers[component] else {
                    throw PoCError.missingFile(
                        modelRoot.appendingPathComponent(component.relativeWeightPath)
                    )
                }
                try MLXWeightProbe.run(component: component, header: header)
            }
            if let outputURL = arguments.vaeOutputURL {
                TextOutput.section("純 Swift VAE 解碼")
                TextOutput.warning("載入並轉換 Oobleck VAE 完整 decoder 權重")
                let report = try ACEStepVAEDecodeStage.run(
                    modelRoot: modelRoot,
                    outputURL: outputURL,
                    latentFrames: arguments.latentFrames
                )
                TextOutput.success("latent：\(report.latentShape) → audio：\(report.audioShape)")
                TextOutput.success(
                    String(
                        format: "WAV：%.3f 秒、%d Hz、%d 聲道",
                        report.wave.durationSeconds,
                        report.wave.sampleRate,
                        report.wave.channelCount
                    )
                )
                TextOutput.notice(
                    String(
                        format: "來源峰值 %.6f、輸出增益 %.6f",
                        report.wave.sourcePeak,
                        report.wave.appliedGain
                    )
                )
                TextOutput.success("輸出：\(report.outputURL.path)")
            }
            if let prompt = arguments.prompt,
               let conditionOutputURL = arguments.conditionOutputURL {
                TextOutput.section("純 Swift ACE-Step 條件編碼")
                TextOutput.warning("依序載入 Qwen3、silence latent 與 ACE condition encoder 完整權重")
                let report = try ACEStepConditioningStage.run(
                    modelRoot: modelRoot,
                    prompt: prompt,
                    lyrics: arguments.lyrics,
                    language: arguments.language,
                    conditionFrames: arguments.conditionFrames,
                    embeddingOutputURL: arguments.embeddingOutputURL,
                    outputURL: conditionOutputURL
                )
                TextOutput.success(
                    "Prompt tokens：\(report.textTokenCount)、歌詞 tokens：\(report.lyricTokenCount)"
                )
                TextOutput.success(
                    "silence latent：\(report.sourceSilenceShape) → 可用 \(report.availableSilenceFrames) 幀"
                )
                TextOutput.success(
                    "encoder：\(report.encoderHiddenShape)、mask：\(report.encoderMaskShape)"
                )
                TextOutput.success(
                    "context：\(report.contextLatentShape)、null：\(report.nullConditionShape)"
                )
                TextOutput.notice(
                    String(
                        format: "Encoder hidden mean(abs)=%.6f",
                        report.encoderMeanAbsoluteValue
                    )
                )
                TextOutput.success("輸出：\(report.outputURL.path)")
            } else if let prompt = arguments.prompt {
                TextOutput.section("純 Swift Qwen3 Embedding")
                TextOutput.warning("載入 Qwen3-Embedding-0.6B 完整權重")
                let report = try ACEStepTextEmbedder.run(
                    modelRoot: modelRoot,
                    prompt: prompt,
                    lyrics: arguments.lyrics,
                    language: arguments.language,
                    outputURL: arguments.embeddingOutputURL
                )
                TextOutput.success(
                    "Prompt tokens：\(report.textTokenCount)、hidden：\(report.textHiddenShape)"
                )
                TextOutput.success(
                    "歌詞 tokens：\(report.lyricTokenCount)、embedding：\(report.lyricHiddenShape)"
                )
                TextOutput.notice("Token 預覽：\(report.tokenPreview)")
                TextOutput.notice(
                    String(
                        format: "Prompt hidden mean(abs)=%.6f",
                        report.textMeanAbsoluteValue
                    )
                )
                if let outputURL = report.outputURL {
                    TextOutput.success("輸出：\(outputURL.path)")
                }
            }
            if let ditOutputURL = arguments.ditOutputURL,
               let conditionInputURL = arguments.conditionInputURL
                    ?? arguments.conditionOutputURL {
                TextOutput.section("純 Swift ACE-Step DiT 前向")
                TextOutput.warning("載入 ACE-Step Turbo DiT 完整 24 層權重")
                let report = try ACEStepDiTForwardProbe.run(
                    modelRoot: modelRoot,
                    conditionURL: conditionInputURL,
                    outputURL: ditOutputURL,
                    seed: arguments.seed
                )
                TextOutput.success("noise：\(report.inputShape) → velocity：\(report.outputShape)")
                TextOutput.notice(
                    String(
                        format: "Velocity mean(abs)=%.6f",
                        report.meanAbsoluteValue
                    )
                )
                TextOutput.success("輸出：\(report.outputURL.path)")
            }
            if let generatedAudioURL = arguments.generatedAudioURL,
               let conditionInputURL = arguments.conditionInputURL
                    ?? arguments.conditionOutputURL {
                TextOutput.section("純 Swift ACE-Step 端到端生成")
                TextOutput.warning("執行 Turbo diffusion 並以 Oobleck VAE 解碼")
                let report = try ACEStepAudioGenerationStage.run(
                    modelRoot: modelRoot,
                    conditionURL: conditionInputURL,
                    outputURL: generatedAudioURL,
                    seed: arguments.seed,
                    inferenceSteps: arguments.inferenceSteps,
                    shift: arguments.shift
                )
                TextOutput.success(
                    "Turbo steps：\(report.schedule.count)、latent：\(report.latentShape)"
                )
                TextOutput.notice(
                    String(format: "Diffusion：%.3f 秒", report.diffusionSeconds)
                )
                TextOutput.success(
                    String(
                        format: "WAV：%.3f 秒、%d Hz、%d 聲道",
                        report.wave.durationSeconds,
                        report.wave.sampleRate,
                        report.wave.channelCount
                    )
                )
                TextOutput.success("輸出：\(report.outputURL.path)")
            }
            TextOutput.section("PoC 結論")
            TextOutput.success("Swift 可直接解析 ACE-Step 設定與 safetensors")
            TextOutput.success("MLX Swift 已具備核心 Conv1D、反卷積、Attention 與 RMSNorm 算子")
            if arguments.vaeOutputURL != nil {
                TextOutput.success("Oobleck VAE 已可由純 Swift 載入權重並解碼 PCM 音訊")
            }
            if arguments.prompt != nil {
                TextOutput.success("Prompt 與歌詞條件已可由純 Swift 產生 MLX Hidden States")
            }
            if arguments.conditionOutputURL != nil {
                TextOutput.success("ACE condition encoder 與 DiT context latents 已可由純 Swift 產生")
            }
            if arguments.ditOutputURL != nil {
                TextOutput.success("ACE-Step Turbo DiT 24 層前向已可由純 Swift 執行")
            }
            if arguments.generatedAudioURL != nil {
                TextOutput.success("Prompt 條件、Turbo DiT、VAE 與 WAV 已完成純 Swift 端到端串接")
            }
        } catch {
            let message = "ACEStepSwiftPoC 失敗：\(error.localizedDescription)\n"
            FileHandle.standardError.write(Data(message.utf8))
            exit(EXIT_FAILURE)
        }
    }

    private static func printUsage() {
        print(
            """
            用法：
              swift run ACEStepSwiftPoC [選項]

            選項：
              --model-root PATH       指定 ACE-Step 模型根目錄
              --load-component NAME   以 MLX 實際載入並轉換代表權重
                                      NAME: dit, embedding, vae, language-model
              --list-tensors COUNT    每個元件列出前 COUNT 個 Tensor
              --decode-vae PATH       以純 Swift VAE 解碼固定 latent 並輸出 WAV
              --latent-frames COUNT   VAE 測試 latent 長度，範圍 1 至 250，預設 8
              --encode-prompt TEXT    以 Qwen3 Embedding 編碼 ACE-Step Prompt
              --lyrics TEXT           選填歌詞，預設空字串
              --language CODE         歌詞語言代碼，預設 en
              --embedding-output PATH 將條件 Hidden States 保存成 safetensors
              --encode-condition PATH 執行 ACE condition encoder 並保存 DiT 條件
              --condition-frames N    context latent 幀數，範圍 128 至 15000，預設 128
              --condition-input PATH  讀取已產生的 DiT 條件 safetensors
              --dit-forward PATH      執行一次 DiT 前向並保存 noise 與 velocity
              --seed N                DiT noise seed，預設 42
              --generate-audio PATH   執行 Turbo diffusion 並輸出 WAV
              --inference-steps N     Turbo 取樣步數，範圍 1 至 20，預設 8
              --shift VALUE           Turbo timestep shift，預設 3
              --help                  顯示說明

            未指定 --load-component 時，只讀取 safetensors header，不載入數 GB 權重。
            """
        )
    }
}

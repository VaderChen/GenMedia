import Foundation

public struct DiscoveredModelCatalog: Sendable {
    public var models: [ModelDescriptor]
    public var profiles: [InferenceProfile]
    public var loras: [LoRADescriptor]

    public init(
        models: [ModelDescriptor] = [],
        profiles: [InferenceProfile] = [],
        loras: [LoRADescriptor] = []
    ) {
        self.models = models
        self.profiles = profiles
        self.loras = loras
    }
}

public enum LocalModelDiscovery {
    public static func discover(at root: URL, fileManager: FileManager = .default) -> DiscoveredModelCatalog {
        var result = DiscoveredModelCatalog()

        discoverZImage(root: root, fileManager: fileManager, result: &result)
        discoverCaptioner(root: root, fileManager: fileManager, result: &result)
        discoverNSFWCaptioner(root: root, fileManager: fileManager, result: &result)
        discoverQwenImageEdit(root: root, fileManager: fileManager, result: &result)
        discoverLTX23(root: root, fileManager: fileManager, result: &result)
        discoverLTX23MLXQ4(root: root, fileManager: fileManager, result: &result)
        discoverMiniMaxH3MLX(root: root, fileManager: fileManager, result: &result)
        discoverUpscalers(root: root, fileManager: fileManager, result: &result)
        discoverLoRAs(root: root, fileManager: fileManager, result: &result)

        return result
    }

    private struct ManagedModelManifest: Decodable {
        var modelID: String
    }

    private static func discoverLTX23(
        root: URL,
        fileManager: FileManager,
        result: inout DiscoveredModelCatalog
    ) {
        let modelID = "Lightricks/LTX-2.3@distilled-1.1"
        let directory = root.appendingPathComponent("ltx-2.3-distilled-1.1", isDirectory: true)
        let manifestURL = directory.appendingPathComponent("genimage-model.json")
        let requiredPaths = [
            "ltx-2.3-22b-distilled-1.1.safetensors",
            "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
            "gemma-3-12b/config.json",
            "gemma-3-12b/model-00001-of-00005.safetensors",
            "gemma-3-12b/model-00002-of-00005.safetensors",
            "gemma-3-12b/model-00003-of-00005.safetensors",
            "gemma-3-12b/model-00004-of-00005.safetensors",
            "gemma-3-12b/model-00005-of-00005.safetensors",
            "gemma-3-12b/model.safetensors.index.json",
            "gemma-3-12b/tokenizer.json"
        ]
        guard let manifestData = try? Data(contentsOf: manifestURL),
              let manifest = try? JSONDecoder().decode(ManagedModelManifest.self, from: manifestData),
              manifest.modelID == modelID,
              requiredPaths.allSatisfy({
                  fileManager.fileExists(atPath: directory.appendingPathComponent($0).path)
              }) else { return }

        result.models.append(
            ModelDescriptor(
                id: modelID,
                displayName: "LTX-2.3 Distilled 1.1",
                publisher: "Lightricks",
                summary: "已安裝 LTX-2.3 Distilled、空間升頻器與 Gemma 3 12B 文字編碼器。",
                capabilities: [.imageToVideo],
                quantization: .bf16,
                approximateDownloadGB: sizeInGB(of: directory, fileManager: fileManager),
                recommendedMemoryGB: 96,
                licenseName: "LTX-2 Community License / Gemma Terms",
                sourceURL: URL(string: "https://huggingface.co/Lightricks/LTX-2.3"),
                localURL: directory,
                isRecommended: true
            )
        )
        result.profiles.append(
            InferenceProfile(
                name: "圖生影 · LTX-2.3 Distilled",
                capability: .imageToVideo,
                modelID: modelID,
                modelRevision: "distilled-1.1",
                architecture: .externalCLI,
                defaults: ProfileDefaults(
                    width: 1280,
                    height: 720,
                    steps: 8,
                    outputCount: 1,
                    frameCount: 121,
                    frameRate: 24
                ),
                notes: "從 \(directory.path) 自動偵測；推論需官方 LTX-2 Python Runtime。",
                isBuiltIn: true
            )
        )
    }

    private static func discoverMiniMaxH3MLX(
        root: URL,
        fileManager: FileManager,
        result: inout DiscoveredModelCatalog
    ) {
        let candidates: [(
            directory: String,
            id: String,
            name: String,
            quantization: ModelQuantization,
            memory: Int,
            transformerShardCount: Int
        )] = [
            (
                "minimax-h3-mlx-8bit",
                "pipenetwork/MiniMax-H3-MLX-8bit",
                "MiniMax H3 MLX Q8",
                .eightBit,
                128,
                7
            ),
            (
                "minimax-h3-mlx-4bit",
                "pipenetwork/MiniMax-H3-MLX-4bit",
                "MiniMax H3 MLX Q4",
                .fourBit,
                96,
                5
            )
        ]
        let sharedRequiredPaths = [
            "transformer/config.json",
            "transformer/model.safetensors.index.json",
            "transformer/quant_config.json",
            "upstream/FL2VA/model_index.json",
            "upstream/FL2VA/processor/tokenizer.json",
            "upstream/FL2VA/text_encoder/config.json",
            "upstream/FL2VA/text_encoder/model.safetensors.index.json",
            "upstream/FL2VA/video_vae/config.json",
            "upstream/FL2VA/video_vae/source/model.safetensors",
            "upstream/FL2VA/audio_vae/config.json",
            "upstream/FL2VA/audio_vae/model.safetensors"
        ]

        for candidate in candidates {
            let directory = root.appendingPathComponent(candidate.directory, isDirectory: true)
            let manifestURL = directory.appendingPathComponent("genimage-model.json")
            let transformerShards = (1...candidate.transformerShardCount).map {
                String(
                    format: "transformer/model-%05d-of-%05d.safetensors",
                    $0,
                    candidate.transformerShardCount
                )
            }
            let textEncoderShards = (1...14).map {
                String(format: "upstream/FL2VA/text_encoder/model-%05d-of-00014.safetensors", $0)
            }
            let requiredPaths = sharedRequiredPaths + transformerShards + textEncoderShards
            guard let manifestData = try? Data(contentsOf: manifestURL),
                  let manifest = try? JSONDecoder().decode(ManagedModelManifest.self, from: manifestData),
                  manifest.modelID == candidate.id,
                  requiredPaths.allSatisfy({
                      fileManager.fileExists(atPath: directory.appendingPathComponent($0).path)
                  }) else { continue }

            result.models.append(
                ModelDescriptor(
                    id: candidate.id,
                    displayName: candidate.name,
                    publisher: "PipeNetwork / MiniMaxAI",
                    summary: "已安裝 MiniMax H3 MLX 量化 Transformer 與完整 FL2VA 文字編碼器、Video/Audio VAE、Tokenizer。",
                    capabilities: [.imageToVideo],
                    quantization: candidate.quantization,
                    approximateDownloadGB: sizeInGB(of: directory, fileManager: fileManager),
                    recommendedMemoryGB: candidate.memory,
                    licenseName: "MiniMax H3 Community License",
                    sourceURL: URL(string: "https://huggingface.co/\(candidate.id)"),
                    localURL: directory,
                    isRecommended: candidate.quantization == .fourBit
                )
            )
            result.profiles.append(
                InferenceProfile(
                    name: "圖生影 · \(candidate.name)",
                    capability: .imageToVideo,
                    modelID: candidate.id,
                    modelRevision: "main",
                    architecture: .externalCLI,
                    defaults: ProfileDefaults(
                        width: 1280,
                        height: 720,
                        steps: 16,
                        outputCount: 1,
                        frameCount: 124,
                        frameRate: 24
                    ),
                    notes: "從 \(directory.path) 自動偵測；推論需 pipenetwork/minimax-h3-mlx Python Runtime。",
                    isBuiltIn: true
                )
            )
        }
    }

    private static func discoverLTX23MLXQ4(
        root: URL,
        fileManager: FileManager,
        result: inout DiscoveredModelCatalog
    ) {
        let modelID = "dgrauet/ltx-2.3-mlx-q4"
        let directory = root.appendingPathComponent("ltx-2.3-mlx-q4", isDirectory: true)
        let requiredPaths = [
            "config.json",
            "embedded_config.json",
            "quantize_config.json",
            "split_model.json",
            "connector.safetensors",
            "transformer-distilled-1.1.safetensors",
            "spatial_upscaler_x2_v1_1.safetensors",
            "spatial_upscaler_x2_v1_1_config.json",
            "vae_decoder.safetensors",
            "vae_encoder.safetensors",
            "audio_vae.safetensors",
            "vocoder.safetensors"
        ]
        let manifestURL = directory.appendingPathComponent("genimage-model.json")
        guard let manifestData = try? Data(contentsOf: manifestURL),
              let manifest = try? JSONDecoder().decode(ManagedModelManifest.self, from: manifestData),
              manifest.modelID == modelID,
              requiredPaths.allSatisfy({
                  fileManager.fileExists(atPath: directory.appendingPathComponent($0).path)
              }) else { return }

        result.models.append(
            ModelDescriptor(
                id: modelID,
                displayName: "LTX-2.3 MLX Q4",
                publisher: "dgrauet / LTX-2 MLX",
                summary: "已安裝原生 MLX INT4 Transformer、Video/Audio VAE、vocoder 與空間升頻器。",
                capabilities: [.imageToVideo, .textToVideo],
                quantization: .fourBit,
                approximateDownloadGB: sizeInGB(of: directory, fileManager: fileManager),
                recommendedMemoryGB: 24,
                licenseName: "LTX-2 Community License / MLX Port MIT",
                sourceURL: URL(string: "https://huggingface.co/\(modelID)"),
                localURL: directory,
                isRecommended: true
            )
        )
        result.profiles.append(
            InferenceProfile(
                name: "圖生影 · LTX-2.3 MLX Q4",
                capability: .imageToVideo,
                modelID: modelID,
                modelRevision: "main",
                architecture: .externalCLI,
                defaults: ProfileDefaults(
                    width: 1280,
                    height: 720,
                    steps: 8,
                    outputCount: 1,
                    frameCount: 97,
                    frameRate: 24
                ),
                loras: [
                    ProfileLoRAConfiguration(
                        modelID: "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
                        scale: 1,
                        conditioning: .sourceImageCanny,
                        conditioningScale: 1
                    )
                ],
                notes: "從 \(directory.path) 自動偵測；預設使用 Union Control IC-LoRA 與來源圖片 Canny 控制影片，由 ltx-2-mlx 使用 Apple Silicon Metal 執行。",
                isBuiltIn: true
            )
        )
        result.profiles.append(
            InferenceProfile(
                name: "文生影 · LTX-2.3 MLX Q4",
                capability: .textToVideo,
                modelID: modelID,
                modelRevision: "main",
                architecture: .externalCLI,
                defaults: ProfileDefaults(
                    width: 1280,
                    height: 720,
                    steps: 8,
                    outputCount: 1,
                    frameCount: 97,
                    frameRate: 24
                ),
                notes: "從 \(directory.path) 自動偵測；由 ltx-2-mlx 使用 Apple Silicon Metal 執行文生影。",
                isBuiltIn: true
            )
        )
    }

    private static func discoverQwenImageEdit(
        root: URL,
        fileManager: FileManager,
        result: inout DiscoveredModelCatalog
    ) {
        let candidates: [(
            directory: String,
            id: String,
            name: String,
            summary: String,
            quantization: ModelQuantization,
            memory: Int,
            requiredPaths: [String]
        )] = [
            (
                "qwen-image-edit-2511-int4",
                "qwen-image-edit-2511@mlx-int4",
                "Qwen Image Edit 2511 INT4",
                "已安裝官方 2511 基礎檔案與 Swift/MLX 預量化 INT4 權重。",
                .fourBit,
                32,
                [
                    "snapshot/processor/tokenizer.json",
                    "snapshot/text_encoder/config.json",
                    "snapshot/vae/config.json",
                    "quantized/qie-2511-dit-int4-mod8.safetensors",
                    "quantized/qie-2511-vl7b-int4.safetensors"
                ]
            ),
            (
                "qwen-image-edit-2511-int8",
                "qwen-image-edit-2511@mlx-int8",
                "Qwen Image Edit 2511 INT8",
                "已安裝官方 2511 權重；Runtime 會在首次使用時建立並保存 MLX INT8。",
                .eightBit,
                48,
                [
                    "snapshot/processor/tokenizer.json",
                    "snapshot/text_encoder/config.json",
                    "snapshot/transformer/config.json",
                    "snapshot/vae/config.json"
                ]
            ),
            (
                "qwen-image-edit-2511-fp16",
                "qwen-image-edit-2511@mlx-fp16",
                "Qwen Image Edit 2511 FP16",
                "已安裝官方 Qwen Image Edit 2511 BF16/FP16 權重。",
                .fp16,
                64,
                [
                    "snapshot/processor/tokenizer.json",
                    "snapshot/text_encoder/config.json",
                    "snapshot/transformer/config.json",
                    "snapshot/vae/config.json"
                ]
            )
        ]

        for candidate in candidates {
            let directory = root.appendingPathComponent(candidate.directory, isDirectory: true)
            let manifestURL = directory.appendingPathComponent("genimage-model.json")
            guard let data = try? Data(contentsOf: manifestURL),
                  let manifest = try? JSONDecoder().decode(ManagedModelManifest.self, from: data),
                  manifest.modelID == candidate.id,
                  candidate.requiredPaths.allSatisfy({
                      fileManager.fileExists(atPath: directory.appendingPathComponent($0).path)
                  }) else { continue }

            result.models.append(
                ModelDescriptor(
                    id: candidate.id,
                    displayName: candidate.name,
                    publisher: "Qwen",
                    summary: candidate.summary,
                    capabilities: [.imageToImage],
                    quantization: candidate.quantization,
                    approximateDownloadGB: sizeInGB(of: directory, fileManager: fileManager),
                    recommendedMemoryGB: candidate.memory,
                    licenseName: "Apache-2.0",
                    sourceURL: URL(string: "https://huggingface.co/Qwen/Qwen-Image-Edit-2511"),
                    localURL: directory,
                    isRecommended: candidate.quantization == .fourBit
                )
            )
        }
    }

    private static func discoverLoRAs(
        root: URL,
        fileManager: FileManager,
        result: inout DiscoveredModelCatalog
    ) {
        guard let enumerator = fileManager.enumerator(
            at: root,
            includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants]
        ) else { return }

        var discovered: [String: LoRADescriptor] = [:]
        var managedModels: [String: ModelDescriptor] = [:]
        for case let fileURL as URL in enumerator {
            guard fileURL.pathExtension.lowercased() == "safetensors",
                  fileURL.deletingLastPathComponent().pathComponents.contains(where: {
                      $0.localizedCaseInsensitiveContains("lora")
                  }),
                  let values = try? fileURL.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
                  values.isRegularFile == true else { continue }

            let normalizedURL = fileURL.resolvingSymlinksInPath().standardizedFileURL
            let id = normalizedURL.path
            discovered[id] = LoRADescriptor(
                id: id,
                displayName: normalizedURL.deletingPathExtension().lastPathComponent,
                localURL: normalizedURL,
                fileSizeMB: Double(values.fileSize ?? 0) / 1_048_576
            )
            let manifestURL = normalizedURL.deletingLastPathComponent()
                .appendingPathComponent("genimage-model.json")
            guard let manifestData = try? Data(contentsOf: manifestURL),
                  let manifest = try? JSONDecoder().decode(
                      ManagedModelManifest.self,
                      from: manifestData
                  ) else {
                // Keep locally discovered LoRAs visible in Model Center as
                // installed entries even when they were not downloaded
                // through GenImage and therefore have no manifest file.
                managedModels[id] = ModelDescriptor(
                    id: id,
                    displayName: normalizedURL.deletingPathExtension().lastPathComponent,
                    publisher: "Local LoRA",
                    summary: "從本機 LoRA 目錄偵測的 .safetensors 檔案，可搭配 Z-Image Turbo 文生圖 Profile 使用。",
                    capabilities: [.lora],
                    quantization: .lora,
                    approximateDownloadGB: sizeInGB(of: normalizedURL, fileManager: fileManager),
                    recommendedMemoryGB: 16,
                    licenseName: "本機檔案（請自行確認授權）",
                    localURL: normalizedURL
                )
                continue
            }
            switch manifest.modelID {
            case "tarn59/pixel_art_style_lora_z_image_turbo":
                managedModels[manifest.modelID] = ModelDescriptor(
                    id: manifest.modelID,
                    displayName: "Z-Image Turbo Pixel Art LoRA",
                    publisher: "tarn59",
                    summary: "已偵測到可搭配 Z-Image Turbo 使用的像素藝術風格 LoRA。",
                    capabilities: [.lora],
                    quantization: .lora,
                    approximateDownloadGB: sizeInGB(of: normalizedURL, fileManager: fileManager),
                    recommendedMemoryGB: 16,
                    licenseName: "Apache-2.0",
                    sourceURL: URL(string: "https://huggingface.co/tarn59/pixel_art_style_lora_z_image_turbo"),
                    localURL: normalizedURL,
                    isRecommended: true
                )
            case "suayptalha/Z-Image-Turbo-Realism-LoRA":
                managedModels[manifest.modelID] = ModelDescriptor(
                    id: manifest.modelID,
                    displayName: "Z-Image Turbo Realism LoRA",
                    publisher: "suayptalha",
                    summary: "已偵測到可搭配 Z-Image Turbo 使用的寫實風格 LoRA。",
                    capabilities: [.lora],
                    quantization: .lora,
                    approximateDownloadGB: sizeInGB(of: normalizedURL, fileManager: fileManager),
                    recommendedMemoryGB: 16,
                    licenseName: "Apache-2.0",
                    sourceURL: URL(string: "https://huggingface.co/suayptalha/Z-Image-Turbo-Realism-LoRA"),
                    localURL: normalizedURL
                )
            case "renderartist/Classic-Painting-Z-Image-Turbo-LoRA":
                managedModels[manifest.modelID] = ModelDescriptor(
                    id: manifest.modelID,
                    displayName: "Z-Image Turbo Classic Painting LoRA",
                    publisher: "renderartist",
                    summary: "已偵測到可搭配 Z-Image Turbo 使用的古典油畫風格 LoRA。",
                    capabilities: [.lora],
                    quantization: .lora,
                    approximateDownloadGB: sizeInGB(of: normalizedURL, fileManager: fileManager),
                    recommendedMemoryGB: 16,
                    licenseName: "Apache-2.0",
                    sourceURL: URL(string: "https://huggingface.co/renderartist/Classic-Painting-Z-Image-Turbo-LoRA"),
                    localURL: normalizedURL
                )
            case "renderartist/Coloring-Book-Z-Image-Turbo-LoRA":
                managedModels[manifest.modelID] = ModelDescriptor(
                    id: manifest.modelID,
                    displayName: "Z-Image Turbo Coloring Book LoRA",
                    publisher: "renderartist",
                    summary: "已偵測到可搭配 Z-Image Turbo 使用的著色書線稿風格 LoRA。",
                    capabilities: [.lora],
                    quantization: .lora,
                    approximateDownloadGB: sizeInGB(of: normalizedURL, fileManager: fileManager),
                    recommendedMemoryGB: 16,
                    licenseName: "Apache-2.0",
                    sourceURL: URL(string: "https://huggingface.co/renderartist/Coloring-Book-Z-Image-Turbo-LoRA"),
                    localURL: normalizedURL
                )
            case "civitai/2465401":
                managedModels[manifest.modelID] = ModelDescriptor(
                    id: manifest.modelID,
                    displayName: "Civitai · Z-Image Asian Beauties",
                    publisher: "DeViLDoNia / Civitai",
                    summary: "已偵測到 Civitai ZImageTurbo 人像 LoRA。",
                    capabilities: [.lora],
                    quantization: .lora,
                    approximateDownloadGB: sizeInGB(of: normalizedURL, fileManager: fileManager),
                    recommendedMemoryGB: 16,
                    licenseName: "Civitai：Image / RentCivit",
                    sourceURL: URL(string: "https://civitai.com/models/785643"),
                    localURL: normalizedURL
                )
            case "civitai/2709343":
                managedModels[manifest.modelID] = ModelDescriptor(
                    id: manifest.modelID,
                    displayName: "Civitai · Z-Image Turbo Lightning",
                    publisher: "Felldude / Civitai",
                    summary: "已偵測到 Civitai ZImageTurbo Lightning LoRA。",
                    capabilities: [.lora],
                    quantization: .lora,
                    approximateDownloadGB: sizeInGB(of: normalizedURL, fileManager: fileManager),
                    recommendedMemoryGB: 16,
                    licenseName: "Civitai：Image / RentCivit",
                    sourceURL: URL(string: "https://civitai.com/models/2409672"),
                    localURL: normalizedURL
                )
            case "civitai/2449645":
                managedModels[manifest.modelID] = ModelDescriptor(
                    id: manifest.modelID,
                    displayName: "Civitai · Z-Image Flat AnimeStyle",
                    publisher: "MenRiVy1 / Civitai",
                    summary: "已偵測到 Civitai ZImageTurbo 平面動畫風格 LoRA。",
                    capabilities: [.lora],
                    quantization: .lora,
                    approximateDownloadGB: sizeInGB(of: normalizedURL, fileManager: fileManager),
                    recommendedMemoryGB: 16,
                    licenseName: "Civitai：Image / RentCivit / Rent / Sell",
                    sourceURL: URL(string: "https://civitai.com/models/2175307"),
                    localURL: normalizedURL
                )
            case "civitai/2608073":
                managedModels[manifest.modelID] = ModelDescriptor(
                    id: manifest.modelID,
                    displayName: "Civitai · Z-Image Diorama",
                    publisher: "loonalone / Civitai",
                    summary: "已偵測到 Civitai ZImageTurbo Diorama 風格 LoRA。",
                    capabilities: [.lora],
                    quantization: .lora,
                    approximateDownloadGB: sizeInGB(of: normalizedURL, fileManager: fileManager),
                    recommendedMemoryGB: 16,
                    licenseName: "Civitai：RentCivit / Rent",
                    sourceURL: URL(string: "https://civitai.com/models/2318236"),
                    localURL: normalizedURL
                )
            case "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control":
                discovered[id]?.compatibleCapabilities = [.imageToVideo]
                managedModels[manifest.modelID] = ModelDescriptor(
                    id: manifest.modelID,
                    displayName: "LTX-2.3 IC-LoRA Union Control",
                    publisher: "Lightricks",
                    summary: "已安裝 LTX-2.3 Union Control LoRA，可供圖生影 Profile 使用 Canny 逐幀控制。",
                    capabilities: [.lora],
                    quantization: .lora,
                    approximateDownloadGB: sizeInGB(of: normalizedURL, fileManager: fileManager),
                    recommendedMemoryGB: 24,
                    licenseName: "LTX-2 Community License",
                    sourceURL: URL(string: "https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control"),
                    localURL: normalizedURL,
                    isRecommended: true
                )
            default:
                break
            }
        }
        result.loras = discovered.values.sorted {
            $0.displayName.localizedStandardCompare($1.displayName) == .orderedAscending
        }
        result.models.append(contentsOf: managedModels.values)
    }

    private static func discoverZImage(
        root: URL,
        fileManager: FileManager,
        result: inout DiscoveredModelCatalog
    ) {
        let directory = root.appendingPathComponent("z-image-turbo-q4", isDirectory: true)
        let requiredFiles = [
            "model_index.json",
            "quantization.json",
            "text_encoder/model.safetensors",
            "transformer/model-00001-of-00002.safetensors",
            "transformer/model-00002-of-00002.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
            "tokenizer/tokenizer.json"
        ]
        if requiredFiles.allSatisfy({
            fileManager.fileExists(atPath: directory.appendingPathComponent($0).path)
        }) {
            let model = ModelDescriptor(
                id: directory.path,
                displayName: "Z-Image Turbo Q4",
                publisher: "Tongyi-MAI",
                summary: "已偵測到完整 Diffusers 格式與 4-bit quantization manifest。",
                capabilities: [.textToImage],
                quantization: .fourBit,
                approximateDownloadGB: sizeInGB(of: directory, fileManager: fileManager),
                recommendedMemoryGB: 16,
                licenseName: "依來源模型授權",
                localURL: directory,
                isRecommended: true
            )
            result.models.append(model)
            result.profiles.append(
                InferenceProfile(
                    name: "文生圖 · Z-Image Q4",
                    capability: .textToImage,
                    modelID: directory.path,
                    modelRevision: "q4-g32-local",
                    architecture: .mlxSwift,
                    defaults: ProfileDefaults(width: 1024, height: 1024, steps: 9, outputCount: 1),
                    notes: "從 \(directory.path) 自動偵測。",
                    isBuiltIn: true
                )
            )
        }

        let managedCandidates: [(
            directory: String,
            id: String,
            name: String,
            summary: String,
            quantization: ModelQuantization,
            memory: Int,
            requiredFiles: [String]
        )] = [
            (
                "z-image-turbo-8bit",
                "mzbac/z-image-turbo-8bit",
                "Z-Image Turbo 8-bit",
                "已安裝 mzbac 公開 8-bit Diffusers 模型。",
                .eightBit,
                16,
                [
                    "genimage-model.json", "model_index.json", "quantization.json",
                    "scheduler/scheduler_config.json", "text_encoder/config.json",
                    "tokenizer/tokenizer.json", "transformer/config.json", "vae/config.json"
                ]
            ),
            (
                "z-image-turbo-mlx-2bit",
                "andrevp/Z-Image-Turbo-MLX-2bit",
                "Z-Image Turbo MLX 2-bit",
                "已安裝 andrevp 公開 MLX 2-bit Diffusers 模型。",
                .twoBit,
                12,
                [
                    "genimage-model.json", "model_index.json", "quantize_config.json", "quantization.json",
                    "scheduler/scheduler_config.json", "text_encoder/config.json",
                    "tokenizer/tokenizer.json", "transformer/config.json", "vae/config.json"
                ]
            ),
            (
                "z-image-turbo-mlx-4bit",
                "andrevp/Z-Image-Turbo-MLX-4bit",
                "Z-Image Turbo MLX 4-bit",
                "已安裝 andrevp 公開 MLX 4-bit Diffusers 模型。",
                .fourBit,
                16,
                [
                    "genimage-model.json", "model_index.json", "quantize_config.json", "quantization.json",
                    "scheduler/scheduler_config.json", "text_encoder/config.json",
                    "tokenizer/tokenizer.json", "transformer/config.json", "vae/config.json"
                ]
            ),
            (
                "z-image-turbo-mlx-8bit",
                "andrevp/Z-Image-Turbo-MLX-8bit",
                "Z-Image Turbo MLX 8-bit",
                "已安裝 andrevp 公開 MLX 8-bit Diffusers 模型。",
                .eightBit,
                24,
                [
                    "genimage-model.json", "model_index.json", "quantize_config.json", "quantization.json",
                    "scheduler/scheduler_config.json", "text_encoder/config.json",
                    "tokenizer/tokenizer.json", "transformer/config.json", "vae/config.json"
                ]
            ),
            (
                "z-image-turbo-giniiki-4bit",
                "Giniiki/Z-Image-Turbo-mlx-4bit",
                "Z-Image Turbo MLX 4-bit · Giniiki",
                "已安裝 Giniiki 公開 MLX 4-bit Diffusers 模型。",
                .fourBit,
                16,
                [
                    "genimage-model.json", "model_index.json",
                    "scheduler/scheduler_config.json", "text_encoder/config.json",
                    "tokenizer/tokenizer.json", "transformer/config.json", "vae/config.json"
                ]
            ),
            (
                "z-image-turbo-fp16",
                "Tongyi-MAI/Z-Image-Turbo",
                "Z-Image Turbo FP16",
                "已安裝 Tongyi-MAI 公開 FP16 Diffusers 模型。",
                .fp16,
                32,
                [
                    "genimage-model.json", "model_index.json", "scheduler/scheduler_config.json",
                    "text_encoder/config.json", "tokenizer/tokenizer.json",
                    "transformer/config.json", "vae/config.json"
                ]
            )
        ]

        for candidate in managedCandidates {
            let managedDirectory = root.appendingPathComponent(candidate.directory, isDirectory: true)
            let manifestURL = managedDirectory.appendingPathComponent("genimage-model.json")
            guard candidate.requiredFiles.allSatisfy({
                fileManager.fileExists(atPath: managedDirectory.appendingPathComponent($0).path)
            }),
                  let manifestData = try? Data(contentsOf: manifestURL),
                  let manifest = try? JSONDecoder().decode(ManagedModelManifest.self, from: manifestData),
                  manifest.modelID == candidate.id else { continue }

            result.models.append(
                ModelDescriptor(
                    id: candidate.id,
                    displayName: candidate.name,
                    publisher: "Z-Image",
                    summary: candidate.summary,
                    capabilities: [.textToImage],
                    quantization: candidate.quantization,
                    approximateDownloadGB: sizeInGB(of: managedDirectory, fileManager: fileManager),
                    recommendedMemoryGB: candidate.memory,
                    licenseName: "Apache-2.0",
                    sourceURL: URL(string: "https://huggingface.co/\(candidate.id)"),
                    localURL: managedDirectory,
                    isRecommended: candidate.quantization == .eightBit
                )
            )
            result.profiles.append(
                InferenceProfile(
                    name: candidate.name,
                    capability: .textToImage,
                    modelID: candidate.id,
                    modelRevision: "main",
                    architecture: .mlxSwift,
                    defaults: ProfileDefaults(width: 1024, height: 1024, steps: 9, outputCount: 1),
                    notes: "從 \(managedDirectory.path) 自動偵測。",
                    isBuiltIn: true
                )
            )

        }
    }

    private static func discoverCaptioner(
        root: URL,
        fileManager: FileManager,
        result: inout DiscoveredModelCatalog
    ) {
        let directory = root.appendingPathComponent("Qwen3-VL-4B-Instruct-4bit", isDirectory: true)
        let requiredFiles = ["config.json", "model.safetensors", "tokenizer.json", "preprocessor_config.json"]
        guard requiredFiles.allSatisfy({ fileManager.fileExists(atPath: directory.appendingPathComponent($0).path) }) else {
            return
        }

        let manifestURL = directory.appendingPathComponent("genimage-model.json")
        let managedModelID: String? = (try? Data(contentsOf: manifestURL))
            .flatMap { try? JSONDecoder().decode(ManagedModelManifest.self, from: $0).modelID }
        let isManagedInstall = managedModelID == "local-captioner-3b@q4"
        let modelID = isManagedInstall ? "local-captioner-3b@q4" : directory.path
        let model = ModelDescriptor(
            id: modelID,
            displayName: "Qwen3-VL 4B 4-bit",
            publisher: isManagedInstall ? "MLX Community" : "Qwen",
            summary: "已偵測到 Qwen3-VL 圖像理解模型，可供圖生文 Profile 使用。",
            capabilities: [.imageToText],
            quantization: .fourBit,
            approximateDownloadGB: sizeInGB(of: directory, fileManager: fileManager),
            recommendedMemoryGB: 16,
            licenseName: "Apache-2.0",
            sourceURL: URL(string: "https://huggingface.co/mlx-community/Qwen3-VL-4B-Instruct-4bit"),
            localURL: directory,
            isRecommended: true
        )
        result.models.append(model)
        result.profiles.append(
            InferenceProfile(
                name: "圖生文 · Qwen3-VL 4-bit",
                capability: .imageToText,
                modelID: modelID,
                modelRevision: "4bit-local",
                architecture: .mlxSwift,
                defaults: ProfileDefaults(maxTokens: 512, languageCode: "zh-Hant"),
                notes: "從 \(directory.path) 自動偵測。",
                isBuiltIn: true
            )
        )
    }

    private static func discoverNSFWCaptioner(
        root: URL,
        fileManager: FileManager,
        result: inout DiscoveredModelCatalog
    ) {
        let modelID = "qwen3-vl-8b-nsfw-caption-v45@mxfp4"
        let directory = root.appendingPathComponent(
            "Qwen3-VL-8B-NSFW-Caption-V4.5-mxfp4",
            isDirectory: true
        )
        let requiredFiles = [
            "config.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "model.safetensors.index.json",
            "tokenizer.json",
            "preprocessor_config.json",
            "processor_config.json"
        ]
        guard requiredFiles.allSatisfy({
            fileManager.fileExists(atPath: directory.appendingPathComponent($0).path)
        }) else { return }

        let manifestURL = directory.appendingPathComponent("genimage-model.json")
        let managedModelID: String? = (try? Data(contentsOf: manifestURL))
            .flatMap { try? JSONDecoder().decode(ManagedModelManifest.self, from: $0).modelID }
        let isManagedInstall = managedModelID == modelID
        let discoveredModelID = isManagedInstall ? modelID : directory.path
        result.models.append(
            ModelDescriptor(
                id: discoveredModelID,
                displayName: "Qwen3-VL 8B NSFW Caption V4.5 mxfp4",
                publisher: isManagedInstall ? "MLX Community" : "Qwen",
                summary: "已偵測到 Qwen3-VL NSFW Caption MLX 模型，可供圖生文 Profile 使用。",
                capabilities: [.imageToText],
                quantization: .fourBit,
                approximateDownloadGB: sizeInGB(of: directory, fileManager: fileManager),
                recommendedMemoryGB: 24,
                licenseName: "Apache-2.0",
                sourceURL: URL(string: "https://huggingface.co/mlx-community/Qwen3-VL-8B-NSFW-Caption-V4.5-mxfp4"),
                localURL: directory,
                isRecommended: false
            )
        )
        result.profiles.append(
            InferenceProfile(
                name: "NSFW 圖生文 · Qwen3-VL 8B mxfp4",
                capability: .imageToText,
                modelID: discoveredModelID,
                modelRevision: "V4.5-mxfp4-local",
                architecture: .mlxSwift,
                defaults: ProfileDefaults(maxTokens: 512, languageCode: "zh-Hant"),
                notes: "從 (directory.path) 自動偵測；內容標記為 Not-For-All-Audiences。",
                isBuiltIn: true
            )
        )
    }

    private static func discoverUpscalers(
        root: URL,
        fileManager: FileManager,
        result: inout DiscoveredModelCatalog
    ) {
        let candidates: [(file: String, name: String, notes: String)] = [
            ("realesrgan512.mlmodel", "照片放大 · Real-ESRGAN 4×", "一般照片與寫實圖片"),
            ("realesrganAnime512.mlmodel", "動漫放大 · Real-ESRGAN 4×", "插畫與動漫圖片")
        ]
        let directory = root.appendingPathComponent("upscale", isDirectory: true)

        for candidate in candidates {
            let modelURL = directory.appendingPathComponent(candidate.file)
            guard fileManager.fileExists(atPath: modelURL.path) else { continue }

            let model = ModelDescriptor(
                id: modelURL.path,
                displayName: candidate.name,
                publisher: "Real-ESRGAN",
                summary: "已偵測到可由 Core ML 編譯的 Upscale 模型。",
                capabilities: [.upscale],
                quantization: .coreML,
                approximateDownloadGB: sizeInGB(of: modelURL, fileManager: fileManager),
                recommendedMemoryGB: 8,
                licenseName: "依來源模型授權",
                localURL: modelURL,
                isRecommended: candidate.file == "realesrgan512.mlmodel"
            )
            result.models.append(model)
            result.profiles.append(
                InferenceProfile(
                    name: candidate.name,
                    capability: .upscale,
                    modelID: modelURL.path,
                    modelRevision: "coreml-local",
                    architecture: .coreML,
                    defaults: ProfileDefaults(upscaleScale: 4, tileSize: 512),
                    notes: "\(candidate.notes)；從本機自動偵測。",
                    isBuiltIn: true
                )
            )

            if candidate.file == "realesrgan512.mlmodel" {
                let twoXID = "\(modelURL.path)#2x"
                result.models.append(
                    ModelDescriptor(
                        id: twoXID,
                        displayName: "照片放大 · Real-ESRGAN 2×",
                        publisher: "Real-ESRGAN",
                        summary: "使用本機 4× 模型修復後縮放為 2×。",
                        capabilities: [.upscale],
                        quantization: .coreML,
                        approximateDownloadGB: sizeInGB(of: modelURL, fileManager: fileManager),
                        recommendedMemoryGB: 8,
                        licenseName: "依來源模型授權",
                        localURL: modelURL,
                        isRecommended: false
                    )
                )
                result.profiles.append(
                    InferenceProfile(
                        name: "照片放大 · Real-ESRGAN 2×",
                        capability: .upscale,
                        modelID: twoXID,
                        modelRevision: "coreml-local",
                        architecture: .coreML,
                        defaults: ProfileDefaults(upscaleScale: 2, tileSize: 512),
                        notes: "使用本機 4× 模型修復後縮放為 2×。",
                        isBuiltIn: true
                    )
                )
            }
        }

        let managedCandidates: [(directory: String, id: String, scale: Int, name: String)] = [
            ("realesrgan-coreml-x4", "realesrgan-x4@coreml", 4, "一般照片放大 · Real-ESRGAN 4×"),
            ("realesrgan-coreml-x2", "realesrgan-x2@coreml", 2, "一般照片放大 · Real-ESRGAN 2×")
        ]
        for candidate in managedCandidates {
            let managedDirectory = root.appendingPathComponent(candidate.directory, isDirectory: true)
            let modelURL = managedDirectory.appendingPathComponent("RealESRGAN_x4.mlpackage", isDirectory: true)
            let manifestURL = managedDirectory.appendingPathComponent("genimage-model.json")
            guard fileManager.fileExists(atPath: modelURL.path),
                  fileManager.fileExists(atPath: manifestURL.path),
                  (try? Data(contentsOf: manifestURL))
                    .flatMap({ try? JSONDecoder().decode(ManagedModelManifest.self, from: $0).modelID }) == candidate.id else {
                continue
            }

            result.models.append(
                ModelDescriptor(
                    id: candidate.id,
                    displayName: candidate.name,
                    publisher: "Real-ESRGAN",
                    summary: candidate.scale == 4
                        ? "已安裝可由 Core ML 編譯的 Real-ESRGAN 模型。"
                        : "使用 4× Core ML 模型修復後縮放為 2×。",
                    capabilities: [.upscale],
                    quantization: .coreML,
                    approximateDownloadGB: sizeInGB(of: modelURL, fileManager: fileManager),
                    recommendedMemoryGB: 8,
                    licenseName: "BSD-3-Clause",
                    sourceURL: URL(string: "https://huggingface.co/mlboydaisuke/Real-ESRGAN-x4-CoreML"),
                    localURL: modelURL,
                    isRecommended: candidate.scale == 4
                )
            )
            result.profiles.append(
                InferenceProfile(
                    name: candidate.name,
                    capability: .upscale,
                    modelID: candidate.id,
                    modelRevision: "coreml-local",
                    architecture: .coreML,
                    defaults: ProfileDefaults(upscaleScale: candidate.scale, tileSize: 512),
                    notes: "從 \(modelURL.path) 自動偵測。",
                    isBuiltIn: true
                )
            )
        }
    }

    private static func sizeInGB(of url: URL, fileManager: FileManager) -> Double {
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory) else { return 0 }

        if !isDirectory.boolValue {
            let values = try? url.resourceValues(forKeys: [.fileSizeKey])
            return Double(values?.fileSize ?? 0) / 1_073_741_824
        }

        guard let enumerator = fileManager.enumerator(
            at: url,
            includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else { return 0 }

        var totalBytes: Int64 = 0
        for case let fileURL as URL in enumerator {
            guard let values = try? fileURL.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
                  values.isRegularFile == true else { continue }
            totalBytes += Int64(values.fileSize ?? 0)
        }
        return Double(totalBytes) / 1_073_741_824
    }
}

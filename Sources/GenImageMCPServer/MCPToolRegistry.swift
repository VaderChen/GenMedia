import Foundation
import GenImageCore
import GenImageRuntime

public struct MCPToolRegistry {
    public init() {}

    public var definitions: [[String: Any]] {
        [
            tool(
                name: "genimage_models_list",
                description: "Discover local GenImage-compatible models and their capabilities.",
                properties: [
                    "model_root": stringProperty("Optional local model root path.")
                ]
            ),
            tool(
                name: "genimage_profiles_list",
                description: "List inferred Profiles for text-to-image, image-to-text, image editing, and Upscale runtimes.",
                properties: [
                    "model_root": stringProperty("Optional local model root path.")
                ]
            ),
            tool(
                name: "genimage_upscale_image",
                description: "Upscale a local image 4x with a Core ML Real-ESRGAN model.",
                properties: [
                    "input_path": stringProperty("Absolute input image path."),
                    "model_path": stringProperty("Absolute .mlmodel or .mlmodelc path."),
                    "output_directory": stringProperty("Optional absolute output directory.")
                ],
                required: ["input_path", "model_path"]
            ),
            tool(
                name: "genimage_generate_image",
                description: "Generate an image with the native Z-Image MLX runtime and a local model path.",
                properties: [
                    "prompt": stringProperty("Text prompt."),
                    "model_path": stringProperty("Absolute Z-Image Diffusers model directory."),
                    "output_path": stringProperty("Optional absolute output PNG path."),
                    "width": integerProperty("Output width, divisible by 16.", defaultValue: 1024),
                    "height": integerProperty("Output height, divisible by 16.", defaultValue: 1024),
                    "steps": integerProperty("Denoising steps.", defaultValue: 9),
                    "seed": integerProperty("Random seed.", defaultValue: 42),
                    "lora_path": stringProperty("Optional absolute path to a local .safetensors LoRA file."),
                    "lora_scale": numberProperty("Optional LoRA weight from 0.0 to 1.0.", defaultValue: 1)
                ],
                required: ["prompt", "model_path"]
            ),
            tool(
                name: "genimage_edit_image",
                description: "Edit a local image with Qwen Image Edit 2511 on Apple Silicon.",
                properties: [
                    "input_path": stringProperty("Absolute input image path."),
                    "model_path": stringProperty("Absolute GenImage Qwen 2511 installation directory."),
                    "prompt": stringProperty("Image editing instruction."),
                    "negative_prompt": stringProperty("Optional negative prompt."),
                    "quantization": [
                        "type": "string",
                        "description": "Installed Qwen 2511 tier.",
                        "enum": ["int4", "int8", "fp16"]
                    ],
                    "output_path": stringProperty("Optional absolute output PNG path."),
                    "steps": integerProperty("Denoising steps.", defaultValue: 20),
                    "seed": integerProperty("Random seed.", defaultValue: 42)
                ],
                required: ["input_path", "model_path", "prompt", "quantization"]
            ),
            tool(
                name: "genimage_describe_image",
                description: "Describe a local image with the native Qwen3-VL MLX runtime.",
                properties: [
                    "input_path": stringProperty("Absolute input image path."),
                    "model_path": stringProperty("Absolute Qwen3-VL model directory."),
                    "language_code": stringProperty("Output language code such as zh-Hant, en, ja, or ko."),
                    "max_tokens": integerProperty("Maximum generated tokens.", defaultValue: 512)
                ],
                required: ["input_path", "model_path"]
            )
        ]
    }

    public func call(name: String, arguments: [String: Any]) async -> [String: Any] {
        do {
            switch name {
            case "genimage_models_list":
                return try listModels(arguments: arguments)
            case "genimage_profiles_list":
                return try listProfiles(arguments: arguments)
            case "genimage_upscale_image":
                return try await upscale(arguments: arguments)
            case "genimage_generate_image":
                return try await generate(arguments: arguments)
            case "genimage_edit_image":
                return try await edit(arguments: arguments)
            case "genimage_describe_image":
                return try await describe(arguments: arguments)
            default:
                return toolError("Unknown tool: \(name)")
            }
        } catch {
            return toolError(error.localizedDescription)
        }
    }

    private func listModels(arguments: [String: Any]) throws -> [String: Any] {
        let root = modelRoot(arguments: arguments)
        let catalog = LocalModelDiscovery.discover(at: root)
        let models: [[String: Any]] = catalog.models.map { model in
            [
                "id": model.id,
                "name": model.displayName,
                "path": model.localURL?.path ?? "",
                "capabilities": model.capabilities.map(\.rawValue).sorted(),
                "quantization": model.quantization.rawValue,
                "size_gb": model.approximateDownloadGB
            ]
        }
        return toolSuccess(
            text: "Discovered \(models.count) local models under \(root.path).",
            structured: ["model_root": root.path, "models": models]
        )
    }

    private func listProfiles(arguments: [String: Any]) throws -> [String: Any] {
        let root = modelRoot(arguments: arguments)
        let catalog = LocalModelDiscovery.discover(at: root)
        let localModelIDs = Set(catalog.models.map(\.id))
        let inferredProfiles = catalog.profiles + ModelCatalog.builtInProfiles.filter { builtIn in
            localModelIDs.contains(builtIn.modelID)
                && !catalog.profiles.contains(where: { local in
                    local.modelID == builtIn.modelID && local.capability == builtIn.capability
                })
        }
        let profiles: [[String: Any]] = inferredProfiles.map { profile in
            [
                "id": profile.id.uuidString,
                "name": profile.name,
                "capability": profile.capability.rawValue,
                "model_id": profile.modelID,
                "model_revision": profile.modelRevision,
                "architecture": profile.architecture.rawValue,
                "profile_revision": profile.profileRevision
            ]
        }
        return toolSuccess(
            text: "Discovered \(profiles.count) Profiles.",
            structured: ["model_root": root.path, "profiles": profiles]
        )
    }

    private func upscale(arguments: [String: Any]) async throws -> [String: Any] {
        let inputURL = try requiredFileURL("input_path", arguments: arguments)
        let modelURL = try requiredFileURL("model_path", arguments: arguments)
        let outputDirectory = (arguments["output_directory"] as? String).map {
            URL(fileURLWithPath: $0, isDirectory: true)
        } ?? FileManager.default.temporaryDirectory.appendingPathComponent("GenImageMCP", isDirectory: true)

        let asset = MediaAsset(
            projectID: UUID(),
            kind: .imported,
            title: inputURL.deletingPathExtension().lastPathComponent,
            fileURL: inputURL,
            pixelWidth: 0,
            pixelHeight: 0
        )
        let profile = InferenceProfile(
            name: "MCP Core ML Upscale",
            capability: .upscale,
            modelID: modelURL.path,
            modelRevision: "mcp",
            architecture: .coreML,
            defaults: ProfileDefaults(upscaleScale: 4, tileSize: 512)
        )
        let service = CoreMLUpscaleService(outputDirectory: outputDirectory)
        let result = try await service.upscale(
            request: UpscaleRequest(asset: asset, profile: profile, scale: 4),
            progress: { _ in }
        )
        guard let outputURL = result.fileURL else {
            throw MCPToolError.runtime("Upscale completed without an output path.")
        }
        return toolSuccess(
            text: "Upscaled image written to \(outputURL.path)",
            structured: [
                "output_path": outputURL.path,
                "width": result.pixelWidth,
                "height": result.pixelHeight
            ]
        )
    }

    private func generate(arguments: [String: Any]) async throws -> [String: Any] {
        guard let prompt = arguments["prompt"] as? String,
              !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw MCPToolError.invalidArgument("prompt is required")
        }
        let modelURL = try requiredFileURL("model_path", arguments: arguments)

        let width = integer(arguments["width"], defaultValue: 1024)
        let height = integer(arguments["height"], defaultValue: 1024)
        let steps = integer(arguments["steps"], defaultValue: 9)
        let seed = integer(arguments["seed"], defaultValue: 42)
        guard OutputGeometry.isSupported(width: width, height: height) else {
            throw MCPToolError.invalidArgument(
                "width and height must be between \(OutputGeometry.minimumDimension) and "
                    + "\(OutputGeometry.maximumDimension) and divisible by "
                    + "\(OutputGeometry.alignment)"
            )
        }
        guard (1...100).contains(steps) else {
            throw MCPToolError.invalidArgument("steps must be between 1 and 100")
        }

        let loraSelection: LoRASelection?
        if let loraPath = arguments["lora_path"] as? String, !loraPath.isEmpty {
            guard NSString(string: loraPath).isAbsolutePath else {
                throw MCPToolError.invalidArgument("lora_path must be absolute")
            }
            let loraURL = try requiredFileURL("lora_path", arguments: arguments)
                .resolvingSymlinksInPath()
                .standardizedFileURL
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: loraURL.path, isDirectory: &isDirectory),
                  !isDirectory.boolValue,
                  loraURL.pathExtension.lowercased() == "safetensors" else {
                throw MCPToolError.invalidArgument("lora_path must be a .safetensors file")
            }
            let scale = double(arguments["lora_scale"], defaultValue: 1)
            guard scale.isFinite, (0...1).contains(scale) else {
                throw MCPToolError.invalidArgument("lora_scale must be between 0.0 and 1.0")
            }
            loraSelection = LoRASelection(adapterID: loraURL.path, localURL: loraURL, scale: scale)
        } else {
            guard arguments["lora_scale"] == nil else {
                throw MCPToolError.invalidArgument("lora_scale requires lora_path")
            }
            loraSelection = nil
        }

        let requestedOutputURL: URL?
        let outputDirectory: URL
        if let path = arguments["output_path"] as? String {
            requestedOutputURL = URL(fileURLWithPath: path)
            outputDirectory = requestedOutputURL!.deletingLastPathComponent()
        } else {
            requestedOutputURL = nil
            outputDirectory = FileManager.default.temporaryDirectory
                .appendingPathComponent("GenImageMCP", isDirectory: true)
        }
        try FileManager.default.createDirectory(
            at: outputDirectory,
            withIntermediateDirectories: true
        )

        let projectID = UUID()
        let profile = InferenceProfile(
            name: "MCP Z-Image",
            capability: .textToImage,
            modelID: modelURL.path,
            modelRevision: "mcp",
            architecture: .mlxSwift,
            defaults: ProfileDefaults(width: width, height: height, steps: steps, outputCount: 1)
        )
        let recipe = GenerationRecipe(
            name: "MCP Generation",
            prompt: prompt,
            modelID: modelURL.path,
            profileID: profile.id,
            width: width,
            height: height,
            steps: steps,
            outputCount: 1,
            seed: UInt64(max(seed, 0)),
            lora: loraSelection
        )
        let service = ZImageTextToImageService(outputDirectory: outputDirectory)
        let results = try await service.generate(
            request: TextToImageRequest(
                projectID: projectID,
                recipe: recipe,
                profile: profile
            ),
            progress: { _ in }
        )
        guard var outputURL = results.first?.fileURL else {
            throw MCPToolError.runtime("Z-Image completed without an output path.")
        }

        if let requestedOutputURL, outputURL.standardizedFileURL != requestedOutputURL.standardizedFileURL {
            guard !FileManager.default.fileExists(atPath: requestedOutputURL.path) else {
                throw MCPToolError.runtime("Output path already exists: \(requestedOutputURL.path)")
            }
            try FileManager.default.moveItem(at: outputURL, to: requestedOutputURL)
            outputURL = requestedOutputURL
        }
        var structured: [String: Any] = [
            "output_path": outputURL.path,
            "width": width,
            "height": height,
            "seed": seed
        ]
        if let loraSelection {
            structured["lora_path"] = loraSelection.localURL.path
            structured["lora_scale"] = loraSelection.scale
        }
        return toolSuccess(
            text: "Generated image written to \(outputURL.path)",
            structured: structured
        )
    }

    private func describe(arguments: [String: Any]) async throws -> [String: Any] {
        let inputURL = try requiredFileURL("input_path", arguments: arguments)
        let modelURL = try requiredFileURL("model_path", arguments: arguments)
        let languageCode = arguments["language_code"] as? String ?? "zh-Hant"
        let maxTokens = max(32, min(integer(arguments["max_tokens"], defaultValue: 512), 2_048))
        let asset = MediaAsset(
            projectID: UUID(),
            kind: .imported,
            title: inputURL.deletingPathExtension().lastPathComponent,
            fileURL: inputURL,
            pixelWidth: 0,
            pixelHeight: 0
        )
        let profile = InferenceProfile(
            name: "MCP Qwen3-VL",
            capability: .imageToText,
            modelID: modelURL.path,
            modelRevision: "mcp",
            architecture: .mlxSwift,
            defaults: ProfileDefaults(maxTokens: maxTokens, languageCode: languageCode)
        )
        let description = try await QwenVLImageDescriptionService().describe(
            request: ImageDescriptionRequest(
                asset: asset,
                profile: profile,
                languageCode: languageCode
            ),
            progress: { _ in }
        )
        return toolSuccess(
            text: description,
            structured: [
                "input_path": inputURL.path,
                "language_code": languageCode,
                "description": description
            ]
        )
    }

    private func edit(arguments: [String: Any]) async throws -> [String: Any] {
        guard let prompt = arguments["prompt"] as? String,
              !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw MCPToolError.invalidArgument("prompt is required")
        }
        let inputURL = try requiredFileURL("input_path", arguments: arguments)
        let modelURL = try requiredFileURL("model_path", arguments: arguments)
        let quantization: ModelQuantization
        switch arguments["quantization"] as? String {
        case "int4": quantization = .fourBit
        case "int8": quantization = .eightBit
        case "fp16": quantization = .fp16
        default:
            throw MCPToolError.invalidArgument("quantization must be int4, int8, or fp16")
        }
        let steps = integer(arguments["steps"], defaultValue: 20)
        let seed = integer(arguments["seed"], defaultValue: 42)
        guard (1...100).contains(steps) else {
            throw MCPToolError.invalidArgument("steps must be between 1 and 100")
        }

        let requestedOutputURL: URL?
        let outputDirectory: URL
        if let path = arguments["output_path"] as? String {
            guard NSString(string: path).isAbsolutePath else {
                throw MCPToolError.invalidArgument("output_path must be absolute")
            }
            requestedOutputURL = URL(fileURLWithPath: path)
            outputDirectory = requestedOutputURL!.deletingLastPathComponent()
        } else {
            requestedOutputURL = nil
            outputDirectory = FileManager.default.temporaryDirectory
                .appendingPathComponent("GenImageMCP", isDirectory: true)
        }
        try FileManager.default.createDirectory(
            at: outputDirectory,
            withIntermediateDirectories: true
        )

        let projectID = UUID()
        let asset = MediaAsset(
            projectID: projectID,
            kind: .imported,
            title: inputURL.deletingPathExtension().lastPathComponent,
            fileURL: inputURL,
            pixelWidth: 0,
            pixelHeight: 0
        )
        let profile = InferenceProfile(
            name: "MCP Qwen Image Edit 2511",
            capability: .imageToImage,
            modelID: modelURL.path,
            modelRevision: "2511",
            architecture: .externalCLI,
            defaults: ProfileDefaults(width: 1024, height: 1024, steps: steps, outputCount: 1)
        )
        let recipe = GenerationRecipe(
            name: "MCP Image Edit",
            prompt: prompt,
            negativePrompt: arguments["negative_prompt"] as? String ?? "",
            modelID: modelURL.path,
            profileID: profile.id,
            width: 1024,
            height: 1024,
            steps: steps,
            outputCount: 1,
            seed: UInt64(max(seed, 0))
        )
        let result = try await Qwen2511ImageToImageService(outputDirectory: outputDirectory).generate(
            request: ImageToImageRequest(
                projectID: projectID,
                sourceAsset: asset,
                recipe: recipe,
                profile: profile,
                modelURL: modelURL,
                quantization: quantization
            ),
            progress: { _ in }
        )
        guard var outputURL = result.fileURL else {
            throw MCPToolError.runtime("Qwen 2511 completed without an output path.")
        }
        if let requestedOutputURL, outputURL.standardizedFileURL != requestedOutputURL.standardizedFileURL {
            guard !FileManager.default.fileExists(atPath: requestedOutputURL.path) else {
                throw MCPToolError.runtime("Output path already exists: \(requestedOutputURL.path)")
            }
            try FileManager.default.moveItem(at: outputURL, to: requestedOutputURL)
            outputURL = requestedOutputURL
        }
        return toolSuccess(
            text: "Edited image written to \(outputURL.path)",
            structured: [
                "output_path": outputURL.path,
                "width": result.pixelWidth,
                "height": result.pixelHeight,
                "seed": seed,
                "quantization": arguments["quantization"] as? String ?? ""
            ]
        )
    }

    private func modelRoot(arguments: [String: Any]) -> URL {
        ModelStorage.rootURL(explicitPath: arguments["model_root"] as? String)
    }

    private func requiredFileURL(_ key: String, arguments: [String: Any]) throws -> URL {
        guard let path = arguments[key] as? String, !path.isEmpty else {
            throw MCPToolError.invalidArgument("\(key) is required")
        }
        let url = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw MCPToolError.invalidArgument("Path does not exist: \(url.path)")
        }
        return url
    }

    private func integer(_ value: Any?, defaultValue: Int) -> Int {
        if let number = value as? NSNumber { return number.intValue }
        if let string = value as? String, let number = Int(string) { return number }
        return defaultValue
    }

    private func double(_ value: Any?, defaultValue: Double) -> Double {
        if let number = value as? NSNumber { return number.doubleValue }
        if let string = value as? String, let number = Double(string) { return number }
        return defaultValue
    }

    private func toolSuccess(text: String, structured: [String: Any]) -> [String: Any] {
        [
            "content": [["type": "text", "text": text]],
            "structuredContent": structured,
            "isError": false
        ]
    }

    private func toolError(_ message: String) -> [String: Any] {
        [
            "content": [["type": "text", "text": message]],
            "isError": true
        ]
    }

    private func tool(
        name: String,
        description: String,
        properties: [String: Any],
        required: [String] = []
    ) -> [String: Any] {
        var schema: [String: Any] = [
            "type": "object",
            "properties": properties,
            "additionalProperties": false
        ]
        if !required.isEmpty { schema["required"] = required }
        return [
            "name": name,
            "description": description,
            "inputSchema": schema
        ]
    }

    private func stringProperty(_ description: String) -> [String: Any] {
        ["type": "string", "description": description]
    }

    private func integerProperty(_ description: String, defaultValue: Int) -> [String: Any] {
        ["type": "integer", "description": description, "default": defaultValue]
    }

    private func numberProperty(_ description: String, defaultValue: Double) -> [String: Any] {
        ["type": "number", "description": description, "default": defaultValue]
    }
}

private enum MCPToolError: LocalizedError {
    case invalidArgument(String)
    case runtime(String)

    var errorDescription: String? {
        switch self {
        case let .invalidArgument(message): "Invalid argument: \(message)"
        case let .runtime(message): message
        }
    }
}

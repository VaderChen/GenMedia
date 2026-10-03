import Foundation

/// Verified, revision-pinned LoRAs shared by Model Center, installation and discovery.
public enum LoRACatalog {
    public struct Entry: Sendable {
        public let id: String
        public let directoryName: String
        public let filename: String
        public let revision: String
        public let displayName: String
        public let summary: String
        public let promptHint: String
        public let capability: ModelCapability
        public let sizeBytes: Int64
        public var h3Acceleration: MiniMaxH3Acceleration? = nil

        public var repository: String { String(id.split(separator: "@")[0]) }
        public func supports(modelID: String, capability: ModelCapability) -> Bool {
            guard self.capability == capability else { return false }
            guard capability == .textToVideo else { return true }
            let compatible = h3Acceleration?.compatibleModelIDs ?? LoRACatalog.ltxModelIDs
            return compatible.contains { $0.lowercased() == modelID.lowercased() }
        }

        public var descriptor: ModelDescriptor {
            ModelDescriptor(id: id, displayName: displayName,
                publisher: String(id.split(separator: "/")[0]), summary: summary,
                capabilities: [.lora], quantization: .lora,
                approximateDownloadGB: Double(sizeBytes) / 1_000_000_000,
                recommendedMemoryGB: h3Acceleration == .larryV4 ? 64 : (capability == .textToVideo ? 48 : 16),
                licenseName: capability == .textToVideo && h3Acceleration == nil ? "LTX-2 Community License" : "Apache-2.0",
                sourceURL: URL(string: "https://huggingface.co/" + repository))
        }
    }

    public static let ltxModelIDs = [
        "dgrauet/ltx-2.3-mlx-q4",
        "unsloth/LTX-2.3-GGUF@distilled-1.1-Q3_K_M"
    ]

    public static let entries: [Entry] = [
        Entry(id: "Ttio2/Z-Image-Turbo-pencil-sketch", directoryName: "loras/z-image-pencil-sketch",
            filename: "Zimage_pencil_sketch.safetensors", revision: "a840a0b18bbe544d25cbfff27c71e0f4c0c3bcfe",
            displayName: "Z-Image · 鉛筆素描",
            summary: "鉛筆與彩色鉛筆素描；提示詞可加入 pencil sketch 或 color pencil sketch。", promptHint: "pencil sketch",
            capability: .textToImage, sizeBytes: 170128296),
        Entry(id: "Ttio2/Z-Image-Turbo-Ghibli-Style", directoryName: "loras/z-image-ghibli-style",
            filename: "ghibli_zimage_finetune.safetensors", revision: "57fb8460fa3accadfcdcbb4379d38403f41c5676",
            displayName: "Z-Image · 吉卜力風格",
            summary: "手繪動畫風格；提示詞加入 Ghibli style。", promptHint: "Ghibli style",
            capability: .textToImage, sizeBytes: 170128296),
        Entry(id: "renderartist/Saturday-Morning-Z-Image-Turbo", directoryName: "loras/z-image-saturday-morning",
            filename: "Saturday_Morning_Z_Image_Turbo_v1_renderartist_1250.safetensors", revision: "4d23d61855455aeafeb63d95ad7a67e78ae34109",
            displayName: "Z-Image · 週六早晨卡通",
            summary: "復古美式卡通與平塗插畫；觸發詞 saturd4ym0rning。", promptHint: "saturd4ym0rning",
            capability: .textToImage, sizeBytes: 74465728),
        Entry(id: "renderartist/Technically-Color-Z-Image-Turbo", directoryName: "loras/z-image-technically-color",
            filename: "Technically_Color_Z_Image_Turbo_v1_renderartist_2000.safetensors", revision: "193838d6d10d552514d51105fa1c07e4a7bbd7e6",
            displayName: "Z-Image · 復古彩色電影",
            summary: "復古電影色彩與鮮明飽和質感；觸發詞 t3chnic4lly。", promptHint: "t3chnic4lly",
            capability: .textToImage, sizeBytes: 255161456),
        Entry(id: "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In", directoryName: "loras/ltx-camera-dolly-in",
            filename: "ltx-2-19b-lora-camera-control-dolly-in.safetensors", revision: "cc976355a158a07fc70ec45ca961fa30661e80f9",
            displayName: "LTX · 鏡頭推近",
            summary: "Lightricks 官方 Dolly In 鏡頭運動 LoRA；原為 LTX-2 19B 訓練，LTX-2.3 為實驗性相容，影片效果待驗證。", promptHint: "The camera dollies in toward the subject.",
            capability: .textToVideo, sizeBytes: 327309208),
        Entry(id: "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out", directoryName: "loras/ltx-camera-dolly-out",
            filename: "ltx-2-19b-lora-camera-control-dolly-out.safetensors", revision: "8e3560fdc7004a6d7a2dd29559fe73200357fbcc",
            displayName: "LTX · 鏡頭拉遠",
            summary: "Lightricks 官方 Dolly Out 鏡頭運動 LoRA；原為 LTX-2 19B 訓練，LTX-2.3 為實驗性相容，影片效果待驗證。", promptHint: "The camera dollies out, revealing the surroundings.",
            capability: .textToVideo, sizeBytes: 327309208),
        Entry(id: "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left", directoryName: "loras/ltx-camera-dolly-left",
            filename: "ltx-2-19b-lora-camera-control-dolly-left.safetensors", revision: "75cdba2244db6e2934d06095dd8bb6efa33006f6",
            displayName: "LTX · 鏡頭左移",
            summary: "Lightricks 官方 Dolly Left 鏡頭運動 LoRA；原為 LTX-2 19B 訓練，LTX-2.3 為實驗性相容，影片效果待驗證。", promptHint: "The camera dollies left.",
            capability: .textToVideo, sizeBytes: 327309208),
        Entry(id: "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right", directoryName: "loras/ltx-camera-dolly-right",
            filename: "ltx-2-19b-lora-camera-control-dolly-right.safetensors", revision: "0c0a2f9fd886da63f5954b220de0d6f034f4056d",
            displayName: "LTX · 鏡頭右移",
            summary: "Lightricks 官方 Dolly Right 鏡頭運動 LoRA；原為 LTX-2 19B 訓練，LTX-2.3 為實驗性相容，影片效果待驗證。", promptHint: "The camera dollies right.",
            capability: .textToVideo, sizeBytes: 327309208),
    ] + MiniMaxH3Acceleration.allCases.map(\.entry)

    public static func entry(for id: String) -> Entry? { entries.first { $0.id == id } }

    public static let videoProfiles: [InferenceProfile] = entries.filter { $0.capability == .textToVideo && $0.h3Acceleration == nil }.flatMap { entry in
        ltxModelIDs.map { modelID in
            InferenceProfile(name: "文生影 · " + entry.displayName + (modelID.contains("GGUF") ? " · GGUF Q3" : " · MLX Q4"),
                capability: .textToVideo, modelID: modelID, modelRevision: "main", architecture: .externalCLI,
                defaults: ProfileDefaults(width: 1280, height: 720, steps: 8, outputCount: 1, frameCount: 97, frameRate: 24),
                loras: [ProfileLoRAConfiguration(modelID: entry.id, scale: 0.8)],
                notes: "LTX-2 19B LoRA 的 LTX-2.3 實驗性套用。鏡頭 LoRA 權重預設 0.8，可複製 Profile 後調整。提示詞建議：" + entry.promptHint,
                isBuiltIn: true)
        }
    } + MiniMaxH3Acceleration.allCases.flatMap(\.profiles)
}

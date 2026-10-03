import Foundation

/// A distilled adapter and its sampling schedule are one inseparable preset.
public enum QwenImage21Acceleration: String, Codable, Sendable {
    case viggleV03 = "viggle-v0.3-6step-r128"

    public static let modelID = "Viggle/Qwen-Image-2.1-viggle-turbo@v0.3-6step-r128"
    public static let filename = "Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r128.safetensors"
    public var steps: Int { 6 }
    public var rawSigmas: [Double] { [1, 0.9375, 0.875, 0.75, 0.5, 0.25] }

    public static var entry: LoRACatalog.Entry {
        .init(id: modelID, directoryName: "loras/qwen21-viggle-turbo-v0.3-r128",
            filename: filename, revision: "009a44a895ef85f7e643c80fdca9543795248867",
            displayName: "Qwen 2.1 · Viggle Turbo v0.3（6 步）",
            summary: "6 步文生圖與單圖編輯；需 Qwen-Image 2.1，固定權重 1。細字與複雜編輯可能不如原模型。限非商業研究／評估。",
            promptHint: "", capability: .textToImage, sizeBytes: 679604800,
            qwenAcceleration: .viggleV03)
    }

    public static var profiles: [InferenceProfile] {
        [ModelCapability.textToImage, .imageToImage].map { capability in
            .init(name: "\(capability.title) · Qwen 2.1 Viggle Turbo（6 步）",
                capability: capability, modelID: QwenImage21Model.id,
                modelRevision: QwenImage21Model.revision, architecture: .mlxSwift,
                defaults: .init(width: 1024, height: 1024, steps: 6, outputCount: 1),
                loras: [.init(modelID: modelID, scale: 1)],
                notes: "純 Swift／MLX，固定 6 步、LoRA 權重 1、CFG=1；請清空負面提示詞。限非商業研究／評估。",
                isBuiltIn: true)
        }
    }
}

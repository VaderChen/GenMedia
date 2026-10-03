import Foundation

public enum QwenImage21PromptEnhancer: String, CaseIterable, Sendable {
    case textToImage = "T2I"
    case imageToImage = "I2I"

    public var capability: ModelCapability { self == .textToImage ? .textToImage : .imageToImage }
    public var repository: String { "prithivMLmods/Qwen-Image-2.1-PE-\(rawValue)-MLX" }
    public var modelID: String { repository + "@4bit" }
    public var officialRepository: String { "Qwen/Qwen-Image-2.1-PE-\(rawValue)" }
    public var revision: String {
        self == .textToImage ? "2d26ac502b16396fdd4b0cc21cc667831a7f6454" : "75172ef3e7b681eae8f82b9ca20f81885557018d"
    }
    public var officialRevision: String {
        self == .textToImage ? "f3ed7985c788ad75b3ab7223e0c4c51e2a43545b" : "72927bc08afc99b7888ceb7d7d51a12db3700bbd"
    }
    public var directoryName: String { "qwen-image-2.1-pe-\(rawValue.lowercased())-mlx-4bit" }
    public static let requiredFiles: Set<String> = [
        "config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
        "model.safetensors.index.json", "processor_config.json", "chat_template.jinja",
        "tokenizer.json", "tokenizer_config.json", "generation_config.json", "system_prompt.txt"
    ]
    public static func model(for id: String) -> Self? { allCases.first { $0.modelID == id } }

    public var descriptor: ModelDescriptor {
        .init(id: modelID, displayName: "Qwen 2.1 提示詞增強 · \(rawValue) 4-bit",
            publisher: "Qwen / prithivMLmods",
            summary: "\(capability.title)的提示詞增強模型，將簡短指令展開為完整描述；保留指定尺寸，會增加前處理時間。限非商業研究／評估。",
            capabilities: [.textToText], quantization: .fourBit, approximateDownloadGB: 5.97,
            recommendedMemoryGB: 16, licenseName: "Qwen Research License",
            sourceURL: URL(string: "https://huggingface.co/\(officialRepository)"))
    }

    public static var profiles: [InferenceProfile] {
        allCases.flatMap { enhancer in
            (QwenImage21Model.profiles + QwenImage21Acceleration.profiles)
                .filter { $0.capability == enhancer.capability }.map { base in
                    var profile = base
                    profile.name += "＋提示詞增強"
                    profile.promptEnhancerModelID = enhancer.modelID
                    profile.notes += " 先以 Qwen PE 展開提示詞；保留原始指令與指定尺寸，會增加前處理時間。"
                    return profile
                }
        }
    }
}

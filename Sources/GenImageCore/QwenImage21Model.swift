import Foundation

public enum QwenImage21Model {
    public static let id = "mlx-community/Qwen-Image-2.1-MLX-4bit"
    public static let revision = "4db4e8c0c0e7a1debf0320415bec8388e888494c"
    public static let directoryName = "qwen-image-2.1-mlx-4bit"
    public static let requiredFiles: Set<String> = [
        "model_index.json", "transformer/config.json", "transformer/model.safetensors",
        "text_encoder/config.json", "text_encoder/model.safetensors",
        "vae/config.json", "vae/model.safetensors", "scheduler/scheduler_config.json",
        "processor/tokenizer.json", "processor/tokenizer_config.json", "processor/preprocessor_config.json"
    ]
    public static func isModelDirectory(_ url: URL) -> Bool {
        struct Index: Decodable { let _class_name: String }
        let path = url.appendingPathComponent("model_index.json")
        guard let size = try? path.resourceValues(forKeys: [.fileSizeKey]).fileSize,
              size < 65_536, let data = try? Data(contentsOf: path),
              let index = try? JSONDecoder().decode(Index.self, from: data) else { return false }
        return index._class_name == "QwenImage21Pipeline"
    }
    public static var descriptor: ModelDescriptor {
        ModelDescriptor(id: id, displayName: "Qwen-Image 2.1 MLX 4-bit", publisher: "Qwen / MLX Community",
            summary: "純 Swift／MLX 文生圖與單圖編輯，原生 RGBA 輸出；新移植 Runtime，完整模型品質尚待驗證。",
            capabilities: [.textToImage, .imageToImage], quantization: .fourBit,
            approximateDownloadGB: 10.5, recommendedMemoryGB: 32, licenseName: "Qwen Research License",
            sourceURL: URL(string: "https://huggingface.co/Qwen/Qwen-Image-2.1"))
    }
    public static var profiles: [InferenceProfile] {
        [ModelCapability.textToImage, .imageToImage].map { capability in
            InferenceProfile(name: "\(capability == .textToImage ? "文生圖" : "圖生圖") · Qwen-Image 2.1 4-bit",
                capability: capability, modelID: id, modelRevision: revision, architecture: .mlxSwift,
                defaults: ProfileDefaults(width: 1024, height: 1024, steps: 40, outputCount: 1),
                notes: "Swift／MLX 實驗性 Runtime；尺寸須為 32 倍數，CFG=1，不使用負面提示詞與 LoRA。建議 32GB 以上。",
                isBuiltIn: true)
        }
    }
}

import Foundation

public enum LTX25Model {
    public static let repository = "ddalcu/LTX-2.5-MLX-Serve-4bit"
    public static let id = repository + "@distilled"
    public static let revision = "e8f8c97cd6e4ff9c997e0d382c027ad9b61776bb"
    public static let directoryName = "ltx-2.5-mlx-q4"
    public static let requiredFiles: Set<String> = [
        "config.json", "embedded_config.json", "quantize_config.json", "transformer-distilled.safetensors",
        "connector.safetensors", "vae_encoder.safetensors", "vae_decoder.safetensors",
        "audio_vae.safetensors", "vocoder.safetensors", "spatial_upscaler_x2_v1_1.safetensors",
        "spatial_upscaler_x2_v1_1_config.json", "gemma4-12b-ltx-v1/config.json",
        "gemma4-12b-ltx-v1/model.safetensors", "gemma4-12b-ltx-v1/tokenizer.json",
        "gemma4-12b-ltx-v1/tokenizer_config.json"
    ]
    public static var descriptor: ModelDescriptor {
        .init(id: id, displayName: "LTX-2.5 Distilled MLX Q4", publisher: "Lightricks / ddalcu",
            summary: "8 步文生影與同步立體聲音訊，支援放大細化。純 Swift／MLX；4-bit 為記憶體與品質折衷，屬實驗性支援。",
            capabilities: [.textToVideo], quantization: .fourBit, approximateDownloadGB: 27.21,
            recommendedMemoryGB: 32, licenseName: "LTX-2.x Community License",
            sourceURL: URL(string: "https://huggingface.co/\(repository)"))
    }
    public static var profile: InferenceProfile {
        .init(name: "文生影 · LTX-2.5 Distilled Q4（8 步）", capability: .textToVideo,
            modelID: id, modelRevision: revision, architecture: .externalCLI,
            defaults: .init(width: 512, height: 320, steps: 8, outputCount: 1, frameCount: 49, frameRate: 24),
            notes: "純 Swift／MLX 實驗性 Runtime；主階段固定 8 步，放大後細化 3 步，CFG=1；尺寸須為 64 倍數、幀數為 8n+1。建議 32GB 以上。目前提供文生影，不混用 2.3 LoRA。",
            isBuiltIn: true)
    }
}

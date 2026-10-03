import Foundation

/// Sampling contracts shared by the catalog, app service and native H3 worker.
public enum MiniMaxH3Acceleration: String, Codable, CaseIterable, Sendable {
    case lightX2V4Step768 = "lightx2v-fl2va-4step-v1-768p"
    case lightX2V8Step768 = "lightx2v-fl2va-8step-v1-768p"
    case larryV4 = "larry-h3-turbo-v4-600"

    public static let fullModelIDs = [
        "Abiray/MiniMax-H3-GGUF@fl2va-Q4_0",
        "Abiray/MiniMax-H3-GGUF@fl2va-Q4_K_M",
        "Abiray/MiniMax-H3-GGUF@fl2va-Q4_K_S"
    ]
    public static let prunedModelID = "unsloth/MiniMax-H3-GGUF@fl2va-pruned-Q4_K"

    public var compatibleModelIDs: [String] {
        self == .larryV4 ? Self.fullModelIDs : [Self.prunedModelID] + Self.fullModelIDs
    }
    public var allowedSteps: ClosedRange<Int> {
        switch self {
        case .lightX2V4Step768: 4...4
        case .lightX2V8Step768: 8...8
        case .larryV4: 4...8
        }
    }
    public var defaultSteps: Int { self == .larryV4 ? 6 : allowedSteps.lowerBound }
    public var videoShift: Float { self == .larryV4 ? 12 : 6 }
    public var audioShift: Float { 3 }
    public var supportsPrunedBase: Bool { self != .larryV4 }

    public var modelID: String {
        switch self {
        case .lightX2V4Step768: "lightx2v/Minimax-h3-Turbo@fl2v-4step-v1.0-768p"
        case .lightX2V8Step768: "lightx2v/Minimax-h3-Turbo@fl2v-8step-v1.0-768p"
        case .larryV4: "larryvrh/MiniMax-H3-Turbo-Lora@v4-step600-ema"
        }
    }

    public var entry: LoRACatalog.Entry {
        let isLarry = self == .larryV4
        let name = isLarry ? "H3 · Turbo v4（4–8 步）" : "H3 · LightX2V \(defaultSteps) 步 768p"
        let filename = isLarry ? "minimax_h3_turbo_v4_step600_ema.safetensors"
            : "minimax_h3_fl2v_turbo_\(defaultSteps)step_v1.0_768p_comfyui_bf16.safetensors"
        return LoRACatalog.Entry(
            id: modelID, directoryName: "loras/" + rawValue, filename: filename,
            revision: isLarry ? "43a74557ac3f6539db8e0f2a959d03feb7a81480" : "3ec17a324ced54151364f24f8b5fb6bf7e26414f",
            displayName: name,
            summary: isLarry
                ? "低步數文生影；預設 6 步，可使用 4–8 步。僅支援完整 FL2VA GGUF，Pruned 版不相容；目前為實驗性功能。"
                : "FL2VA 文生影加速 LoRA；固定 \(defaultSteps) 步，搭配對應的影片／音訊採樣設定。支援完整及 Pruned GGUF；目前為實驗性功能。",
            promptHint: "", capability: .textToVideo,
            sizeBytes: isLarry ? 779849816 : (self == .lightX2V4Step768 ? 1956192992 : 1956193000),
            h3Acceleration: self
        )
    }

    public var profiles: [InferenceProfile] {
        compatibleModelIDs.map { modelID in
            let suffix = String(modelID.split(separator: "@").last ?? "")
            return InferenceProfile(
                name: "文生影 · \(entry.displayName) · \(suffix)",
                capability: .textToVideo, modelID: modelID,
                modelRevision: "main", architecture: .externalCLI,
                defaults: ProfileDefaults(width: 1344, height: 768, steps: defaultSteps,
                    outputCount: 1, frameCount: 124, frameRate: 24),
                loras: [.init(modelID: self.modelID, scale: 1)],
                notes: "低步數加速 Profile（實驗性），建議 LoRA 權重 1。" +
                    (self == .larryV4 ? "預設 6 步；4 步快速動作可能有殘影，可提高至 6–8 步。" : "請維持 \(defaultSteps) 步。") +
                    "加速只減少去噪次數，主模型的記憶體需求不變。",
                isBuiltIn: true)
        }
    }
}

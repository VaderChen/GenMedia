import Foundation
import GenImageCore

public enum Qwen21Error: LocalizedError {
    case invalid(String)
    public var errorDescription: String? {
        switch self { case .invalid(let message): "Qwen-Image 2.1：\(message)" }
    }
}

public struct Qwen21Request: Codable, Sendable {
    public var modelDirectory: String
    public var outputPaths: [String]
    public var inputPath: String?
    public var prompt: String
    public var negativePrompt: String
    public var width: Int
    public var height: Int
    public var steps: Int
    public var seed: UInt64
    public var loraPath: String?
    public var loraScale: Double?
    public var acceleration: QwenImage21Acceleration?

    public init(modelDirectory: String, outputPaths: [String], inputPath: String? = nil,
                prompt: String, negativePrompt: String = "", width: Int, height: Int,
                steps: Int, seed: UInt64, loraPath: String? = nil, loraScale: Double? = nil,
                acceleration: QwenImage21Acceleration? = nil) {
        self.modelDirectory = modelDirectory; self.outputPaths = outputPaths; self.inputPath = inputPath
        self.prompt = prompt; self.negativePrompt = negativePrompt
        self.width = width; self.height = height; self.steps = steps; self.seed = seed
        self.loraPath = loraPath; self.loraScale = loraScale; self.acceleration = acceleration
    }

    public func validate() throws {
        guard (loraPath != nil) == (acceleration != nil),
              (loraPath != nil) == (loraScale != nil),
              acceleration == nil || (steps == acceleration?.steps && loraScale == 1 && loraPath?.isEmpty == false) else {
            throw Qwen21Error.invalid("Viggle Turbo 需要完整 LoRA、固定 6 步與權重 1。")
        }
        guard (256...2048).contains(width), (256...2048).contains(height),
              width % 32 == 0, height % 32 == 0 else {
            throw Qwen21Error.invalid("寬高須為 256～2048 之間的 32 倍數。")
        }
        guard (1...100).contains(steps), (1...8).contains(outputPaths.count),
              Set(outputPaths).count == outputPaths.count,
              !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw Qwen21Error.invalid("請提供提示詞、1～100 步與 1～8 個不同的輸出路徑。")
        }
        guard negativePrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw Qwen21Error.invalid("目前使用 guidance-free 推論，請清空負面提示詞。")
        }
    }
}

struct Qwen21TransformerConfiguration: Decodable {
    let in_channels: Int
    let out_channels: Int
    let context_in_dim: Int
    let num_attention_heads: Int
    let attention_head_dim: Int
    let num_layers: Int
    let axes_dims_rope: [Int]
    let eps: Float
    let quantization: Quantization
    struct Quantization: Decodable { let group_size: Int; let bits: Int; let mode: String }
    var hidden: Int { num_attention_heads * attention_head_dim }
}

struct Qwen21VAEConfiguration: Decodable {
    let base_dim: Int
    let decoder_base_dim: Int
    let dim_mult: [Int]
    let num_res_blocks: Int
    let z_dim: Int
    let latents_mean: [Float]
    let latents_std: [Float]
}

struct Qwen21Scheduler: Decodable {
    let base_image_seq_len: Int
    let max_image_seq_len: Int
    let base_shift: Double
    let max_shift: Double
    let shift_terminal: Double?

    func sigmas(steps: Int, imageTokens: Int, acceleration: QwenImage21Acceleration? = nil) throws -> [Float] {
        guard steps > 0, max_image_seq_len > base_image_seq_len,
              imageTokens > 0, base_shift.isFinite, max_shift.isFinite,
              shift_terminal.map({ $0 > 0 && $0 < 1 }) ?? true,
              acceleration == nil || steps == acceleration?.steps else {
            throw Qwen21Error.invalid("無效的取樣器參數。")
        }
        let mu = base_shift + Double(imageTokens - base_image_seq_len)
            * (max_shift - base_shift) / Double(max_image_seq_len - base_image_seq_len)
        let e = exp(mu)
        let raw = acceleration?.rawSigmas ?? (0..<steps).map { 1 - Double($0) / Double(steps) }
        var values = raw.map { e / (e + 1 / $0 - 1) }
        // FlowMatchEuler's stretch_shift_to_terminal is undefined for a one-step schedule.
        // Viggle's distilled schedule explicitly disables terminal stretching.
        if acceleration == nil, steps > 1, let shift_terminal {
            let scale = (1 - values[steps - 1]) / (1 - shift_terminal)
            values = values.map { 1 - (1 - $0) / scale }
        }
        return values.map(Float.init) + [0]
    }
}

func readJSON<T: Decodable>(_ type: T.Type, at url: URL) throws -> T {
    try JSONDecoder().decode(type, from: Data(contentsOf: url))
}

import CoreImage
import Foundation
import Hub
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM
import Tokenizers

struct Qwen21Conditioning {
    let hidden: MLXArray
    let imageSlots: [Bool]
}

final class Qwen21TextEncoder {
    let model: Qwen3VL
    let tokenizer: any Tokenizers.Tokenizer
    let processorConfiguration: Qwen3VLProcessorConfiguration
    static let system = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"

    init(directory: URL) throws {
        let text = directory.appendingPathComponent("text_encoder")
        let configuration = try readJSON(Qwen3VLConfiguration.self, at: text.appendingPathComponent("config.json"))
        model = Qwen3VL(configuration)
        var weights = try MLX.loadArrays(url: text.appendingPathComponent("model.safetensors"))
        // mlx-vlm names are already mapped, but the pinned conversion retains OITHW
        // for the vision Conv3d. The model sanitizer checks its layout before transposing
        // and removes the unused LM head when the text configuration ties embeddings.
        weights = model.sanitize(weights: weights)
        quantize(model: model, groupSize: 64, bits: 4, filter: { path, _ in weights[path + ".scales"] != nil })
        try model.update(parameters: .unflattened(weights), verify: .all)
        weights.removeAll()
        model.train(false)
        let p = directory.appendingPathComponent("processor")
        tokenizer = try AutoTokenizer.from(
            tokenizerConfig: readJSON(Config.self, at: p.appendingPathComponent("tokenizer_config.json")),
            tokenizerData: readJSON(Config.self, at: p.appendingPathComponent("tokenizer.json")))
        processorConfiguration = try readJSON(Qwen3VLProcessorConfiguration.self,
            at: p.appendingPathComponent("preprocessor_config.json"))
    }

    func encode(prompt: String, image: CIImage?) throws -> Qwen21Conditioning {
        var prefix = "", processed: LMInput.ProcessedImage?
        var slots = 0
        if let image {
            let (pixels, grid) = try Qwen21ImageProcessing.visionPixels(image, configuration: processorConfiguration)
            slots = grid.product / 4
            guard grid.h * 16 == Int(image.extent.height), grid.w * 16 == Int(image.extent.width) else {
                throw Qwen21Error.invalid("視覺前處理改變了參考影像尺寸。")
            }
            processed = .init(pixels: pixels, frames: [grid])
            prefix = "<image1><|vision_start|>" + String(repeating: "<|image_pad|>", count: slots) + "<|vision_end|>"
        }
        let template = Self.system + "<|im_start|>user\n" + prefix + prompt + "<|im_end|>\n<|im_start|>assistant\n"
        let ids = tokenizer.encode(text: template, addSpecialTokens: false)
        let drop = tokenizer.encode(text: Self.system, addSpecialTokens: false).count
        guard ids.count <= 8192, ids.count > drop else {
            throw Qwen21Error.invalid("提示詞與影像合計超出 8192 token 上限。")
        }
        let input = LMInput(text: .init(tokens: MLXArray(ids).expandedDimensions(axis: 0)), image: processed)
        let hidden = try model.encodeImageConditioning(input)[0..., drop..., 0...].asType(.bfloat16)
        eval(hidden)
        let mask = ids.dropFirst(drop).map { $0 == 151655 }
        guard mask.filter({ $0 }).count == slots else { throw Qwen21Error.invalid("圖片 token 數量不符。") }
        return Qwen21Conditioning(hidden: hidden, imageSlots: mask)
    }
}

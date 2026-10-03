import Foundation
import GenImageCore
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXVLM
import Tokenizers

enum QwenPromptEnhancementService {
    /// No retained model container: release the 9B enhancer before the image worker loads.
    static func enhance(prompt: String, imageURL: URL?, modelURL: URL,
                        kind: QwenImage21PromptEnhancer, seed: UInt64,
                        responseObserver: (@Sendable (String) -> Void)? = nil,
                        progress: @escaping @Sendable (Double) -> Void) async throws -> String {
        guard (kind == .imageToImage) == (imageURL != nil),
              QwenImage21PromptEnhancer.requiredFiles.allSatisfy({ file in
                  (try? modelURL.appendingPathComponent(file).resourceValues(forKeys: [.fileSizeKey]))
                      .map { ($0.fileSize ?? 0) > 0 } ?? false
              }) else { throw Failure.invalid("提示詞增強模型尚未完整安裝，或來源圖片不符。") }
        let system = try String(contentsOf: modelURL.appendingPathComponent("system_prompt.txt"), encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !system.isEmpty else { throw Failure.invalid("提示詞增強模型缺少系統提示詞。") }
        // Transformers 5 stores image preprocessing under image_processor; MLX 3.31
        // expects the flat configuration. Keep this adapter local to the PE factory.
        let processors = ProcessorTypeRegistry(creators: ["Qwen3VLProcessor": { data, tokenizer in
            struct Root: Decodable { let image_processor: Qwen3VLProcessorConfiguration }
            let configuration = try JSONDecoder().decode(Root.self, from: data).image_processor
            return Qwen3VLProcessor(configuration, tokenizer: tokenizer)
        }])
        let factory = VLMModelFactory(typeRegistry: VLMTypeRegistry.shared,
            processorRegistry: processors, modelRegistry: VLMRegistry.shared)
        let container = try await factory.loadContainer(from: modelURL, using: #huggingFaceTokenizerLoader())
        try Task.checkCancellation()
        progress(0.3)
        let input = UserInput(chat: [.system(system), .user(prompt, images: imageURL.map { [.url($0)] } ?? [])],
            processing: .init(minPixels: 65_536, maxPixels: 1_048_576),
            additionalContext: ["enable_thinking": true])
        let prepared = try await container.prepare(input: input)
        progress(0.5)
        let parameters = GenerateParameters(maxTokens: kind == .textToImage ? 16_256 : 24_000,
            temperature: 1, topP: 0.95, topK: 20, seed: seed)
        let stream = try await container.generate(input: prepared, parameters: parameters)
        var output = "", chunks = 0
        var stopReason: GenerateStopReason?
        for await event in stream {
            try Task.checkCancellation()
            switch event {
            case let .chunk(text):
                output += text; chunks += 1
                progress(0.5 + 0.49 * Double(chunks) / Double(chunks + 512))
            case let .info(info): stopReason = info.stopReason
            case .toolCall: throw Failure.invalid("提示詞增強回傳了無效的工具呼叫。")
            }
        }
        try Task.checkCancellation()
        responseObserver?(output)
        guard stopReason == .stop else { throw Failure.invalid("提示詞增強未完整結束，請縮短指令後重試。") }
        let rewritten = try parse(output, kind: kind)
        progress(1)
        return rewritten
    }

    static func parse(_ output: String, kind: QwenImage21PromptEnhancer) throws -> String {
        struct Answer: Decodable {
            let rewritten_prompt: String
            let wh_ratio: String
            let ratio_follow: String?
        }
        // Do not extract arbitrary braces from incomplete thinking or truncated responses.
        var text = output.trimmingCharacters(in: .whitespacesAndNewlines)
        if !text.hasPrefix("{"), let end = text.range(of: "</think>") {
            text = String(text[end.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        // The 4-bit I2I model can omit the opening quote before its image reference.
        // Normalize only that exact final metadata field, then require a complete JSON object.
        if kind == .imageToImage,
           let field = text.range(of: #",\s*"ratio_follow"\s*:\s*<image1>"\s*\}$"#, options: .regularExpression) {
            text.replaceSubrange(field, with: #", "ratio_follow": "<image1>"}"#)
        }
        guard let answer = try? JSONDecoder().decode(Answer.self, from: Data(text.utf8)) else {
            throw Failure.invalid("提示詞增強未回傳完整 JSON，未開始生成圖片。")
        }
        let prompt = answer.rewritten_prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let ratio = answer.wh_ratio.split(separator: ":", omittingEmptySubsequences: false)
        let validRatio = ratio.count == 2 && ratio.allSatisfy { Int($0).map { $0 > 0 } ?? false }
        let followsImage = answer.wh_ratio.isEmpty && answer.ratio_follow == "<image1>"
        let explicitRatio = validRatio && (answer.ratio_follow ?? "").isEmpty
        guard !prompt.isEmpty, prompt.count <= 32_768,
              explicitRatio || (kind == .imageToImage && followsImage) else {
            throw Failure.invalid("提示詞增強的描述或圖片比例格式無效。")
        }
        // The enhancer's aspect-ratio advice never overrides the user's chosen output size.
        return prompt
    }

    enum Failure: LocalizedError {
        case invalid(String)
        var errorDescription: String? { switch self { case .invalid(let message): message } }
    }
}

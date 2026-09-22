import Foundation
import CoreImage
import ImageIO
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
import MLXVLM
import Testing
@testable import QwenImage21Runtime

@Suite(.serialized)
struct Qwen21Tests {
    @Test func encoderBypassesFinalNormWithoutChangingOrdinaryLogits() throws {
        let json = #"""
        {"model_type":"qwen3_vl","text_config":{"model_type":"qwen3_vl_text","hidden_size":8,"intermediate_size":16,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":8,"max_position_embeddings":128,"vocab_size":32,"rope_scaling":{"mrope_interleaved":true,"mrope_section":[1,1,2]}},"vision_config":{"model_type":"qwen3_vl","depth":1,"hidden_size":8,"intermediate_size":16,"out_hidden_size":8,"num_heads":1,"patch_size":2,"spatial_merge_size":2,"temporal_patch_size":2,"num_position_embeddings":16}}
        """#
        let configuration = try JSONDecoder().decode(Qwen3VLConfiguration.self, from: Data(json.utf8))
        let model = Qwen3VL(configuration)
        model.update(parameters: .unflattened(["language_model.model.norm.weight": MLXArray.ones([8]) * 3]))
        let input = LMInput(text: .init(tokens: MLXArray([1, 2, 3]).reshaped([1, 3])))
        let hidden = try model.encodeImageConditioning(input)
        let weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        let norm = try #require(weights["language_model.model.norm.weight"])
        let embedding = try #require(weights["language_model.model.embed_tokens.weight"])
        let expected = matmul(MLXFast.rmsNorm(hidden, weight: norm, eps: 1e-6), embedding.T)
        let ordinary = model(input.text, cache: nil, state: nil).logits
        #expect(hidden.shape == [1, 3, 8])
        #expect(max(abs(ordinary - expected)).item(Float.self) < 1e-4)
        #expect(max(abs(ordinary - matmul(hidden, embedding.T))).item(Float.self) > 0.01)
    }

    @Test func visionNormalizesSRGBPixelsAndPacksTemporalPatches() throws {
        let config = #"{"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"merge_size":2,"patch_size":2,"temporal_patch_size":2,"image_processor_type":"Qwen2VLImageProcessorFast"}"#
        let configuration = try JSONDecoder().decode(Qwen3VLProcessorConfiguration.self, from: Data(config.utf8))
        // Mid-gray is deliberately chosen: linear-light normalization would be about -0.57, not 0.
        let data = Data(Array(repeating: [UInt8(128), 128, 128, 255], count: 16).flatMap { $0 })
        let image = CIImage(bitmapData: data, bytesPerRow: 16, size: CGSize(width: 4, height: 4),
                            format: .RGBA8, colorSpace: CGColorSpace(name: CGColorSpace.sRGB))
        let (pixels, grid) = try Qwen21ImageProcessing.visionPixels(image, configuration: configuration)
        #expect(grid.h == 2 && grid.w == 2 && grid.t == 1)
        #expect(pixels.shape == [4, 24])
        #expect(max(abs(pixels - (Float(128) / 255 * 2 - 1))).item(Float.self) < 0.002)
    }

    @Test func schedulerMatchesFlowMatchTerminal() throws {
        let scheduler = Qwen21Scheduler(base_image_seq_len: 256, max_image_seq_len: 8192,
            base_shift: 0.5, max_shift: 0.9, shift_terminal: 0.02)
        let values = try scheduler.sigmas(steps: 40, imageTokens: 4096)
        #expect(values.count == 41)
        #expect(values[0] == 1)
        #expect(abs(values[39] - 0.02) < 1e-6)
        #expect(values.last == 0)
        #expect(zip(values, values.dropFirst()).allSatisfy { $0 > $1 })
        #expect(try scheduler.sigmas(steps: 1, imageTokens: 256) == [1, 0])
    }

    @Test func layoutExpandsVisionSlotsAndKeepsTextCausal() throws {
        let layout = try Qwen21Layout(imageSlots: [false, true, false, false],
            referenceHeight: 2, referenceWidth: 2, height: 2, width: 2)
        #expect(layout.targetStart == 7)
        #expect(layout.textIndices == [0, 2, 3])
        #expect(layout.imageIndices == [1, 2, 3, 4, 7, 8, 9, 10])
        #expect(layout.positions[1] == [1, -1, -1])
        #expect(layout.positions[5] == [3, 3, 3])
        #expect(layout.positions[7] == [5, -1, -1])
        #expect(throws: Qwen21Error.self) {
            try Qwen21Layout(imageSlots: [true], referenceHeight: 4, referenceWidth: 4, height: 2, width: 2)
        }
    }

    @Test func requestRejectsUnsupportedOptionsBeforeLoading() throws {
        var request = Qwen21Request(modelDirectory: "/missing", outputPaths: ["/a.png"],
            prompt: "貓", width: 512, height: 512, steps: 40, seed: 1)
        try request.validate()
        request.negativePrompt = "blur"
        #expect(throws: Qwen21Error.self) { try request.validate() }
        request.negativePrompt = ""; request.width = 513
        #expect(throws: Qwen21Error.self) { try request.validate() }
    }

    @Test func quantizedLinearMatchesDequantizedMatrix() throws {
        let w = MLXArray((0..<128).map { sin(Float($0)) }).reshaped([2, 64])
        let (packed, scales, biases) = quantized(w, groupSize: 64, bits: 4)
        let weights = Qwen21Weights(tensors: ["layer.weight": packed, "layer.scales": scales, "layer.biases": biases!])
        let x = MLXArray((0..<64).map { Float($0) / 64 }).reshaped([1, 64])
        let actual = try weights.linear(x, "layer")
        let expected = matmul(x, dequantized(packed, scales: scales, biases: biases, groupSize: 64, bits: 4).T)
        #expect(max(abs(actual - expected)).item(Float.self) < 1e-4)
    }

    @Test func vaeShortcutsRespectCausalTemporalPadding() {
        let x = MLXArray((0..<16).map(Float.init)).reshaped([1, 2, 2, 4])
        let averaged = Qwen21VAE.averageShortcut(x, outChannels: 8, spatial: 2, temporal: 2)
        #expect(averaged.shape == [1, 1, 1, 8])
        // Each channel's zero temporal frame precedes its spatial samples.
        #expect(averaged.asArray(Float.self) == [0, 6, 0, 7, 0, 8, 0, 9])
        let up = Qwen21VAE.duplicateShortcut(MLXArray([Float(2), 4]).reshaped([1, 1, 1, 2]), outChannels: 2, temporal: 2)
        #expect(up.shape == [1, 2, 2, 2])
        #expect(up.asArray(Float.self) == [2, 4, 2, 4, 2, 4, 2, 4])
    }

    @Test func pngPreservesAlpha() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".png")
        defer { try? FileManager.default.removeItem(at: url) }
        try Qwen21Pipeline.writePNG(MLXArray([Float(1), 0, 0, 0.25]).reshaped([1, 1, 1, 4]), to: url, width: 1, height: 1)
        #expect(throws: Qwen21Error.self) {
            try Qwen21Pipeline.writePNG(MLXArray([Float.nan, 0, 0, 1]).reshaped([1, 1, 1, 4]), to: url, width: 1, height: 1)
        }
        let source = try #require(CGImageSourceCreateWithURL(url as CFURL, nil))
        let image = try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
        #expect(image.alphaInfo != .none)
        let properties = try #require(CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any])
        #expect(properties[kCGImagePropertyHasAlpha] as? Bool == true)
    }

    @Test func transformerMatchesIndependentDenseAttentionOracle() throws {
        struct Tensor: Decodable { let shape: [Int]; let values: [Float] }
        struct Case: Decodable { let slots: [Bool]; let conditioning: [Float]; let latents: [Float]; let expected: [Float] }
        struct Fixture: Decodable { let weights: [String: Tensor]; let cases: [Case] }
        let url = try #require(Bundle.module.url(forResource: "transformer-oracle", withExtension: "json", subdirectory: "Fixtures"))
        let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let config = #"{"in_channels":64,"out_channels":64,"context_in_dim":8,"num_attention_heads":1,"attention_head_dim":8,"num_layers":1,"axes_dims_rope":[2,2,4],"eps":0.000001,"quantization":{"bits":4,"group_size":64,"mode":"affine"}}"#
        try Data(config.utf8).write(to: directory.appendingPathComponent("config.json"))
        try MLX.save(arrays: fixture.weights.mapValues { MLXArray($0.values).reshaped($0.shape) },
                     url: directory.appendingPathComponent("model.safetensors"))
        let model = try Qwen21Transformer(directory: directory)
        for item in fixture.cases {
            let edit = item.slots.contains(true)
            let layout = try Qwen21Layout(imageSlots: item.slots, referenceHeight: edit ? 2 : nil,
                referenceWidth: edit ? 2 : nil, height: 2, width: 2)
            let actual = try model(latents: MLXArray(item.latents).reshaped([1, edit ? 8 : 4, 64]),
                conditioning: MLXArray(item.conditioning).reshaped([1, 4, 8]), sigma: 0.7, layout: layout)
            let expected = MLXArray(item.expected).reshaped([1, 4, 64])
            // Production uses BF16 in the joint stream; the independent oracle uses fp32/fp64.
            #expect(max(abs(actual.asType(.float32) - expected)).item(Float.self) < 0.02)
        }
    }
}

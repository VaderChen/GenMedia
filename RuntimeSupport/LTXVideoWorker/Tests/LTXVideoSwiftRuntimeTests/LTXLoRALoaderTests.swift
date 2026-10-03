import Foundation
import MLX
import MLXNN
import Testing
@testable import LTXVideoSwiftRuntime

@Suite(.serialized)
struct LTXLoRALoaderTests {
    private final class Model: Module {
        @ModuleInfo var projection: Linear
        init(quantized: Bool) {
            let weight = MLXArray((0..<(32 * 64)).map { Float($0 % 19) / 100 }).reshaped([32, 64])
            let linear = Linear(weight: weight, bias: MLXArray.ones([32]))
            self._projection = ModuleInfo(wrappedValue: quantized ? QuantizedLinear(linear, groupSize: 64, bits: 4) : linear)
            super.init()
        }
    }

    @Test(arguments: [false, true])
    func residualMatchesIndependentFormulaAndPreservesBase(quantized: Bool) throws {
        let model = Model(quantized: quantized)
        let original = model.projection
        let originalFloatWeight = quantized ? [] : original.weight.asArray(Float.self)
        let originalPackedWeight = quantized ? original.weight.asArray(UInt32.self) : []
        let input = MLXArray.ones([1, 3, 64])
        let baseline = original(input)
        eval(baseline)
        let a = MLXArray((0..<128).map { Float($0 % 7) / 100 }).reshaped([2, 64])
        let b = MLXArray((0..<64).map { Float($0 % 5) / 100 }).reshaped([32, 2])
        let file = try fixture(["diffusion_model.projection.lora_A.weight": a,
                                "diffusion_model.projection.lora_B.weight": b,
                                "diffusion_model.projection.alpha": MLXArray(Float(1))])
        defer { try? FileManager.default.removeItem(at: file) }
        let count = try LTXLoRALoader.apply([.init(url: file, scale: 0.6), .init(url: file, scale: 0.2)], to: model)
        #expect(count == 1)
        let expected = baseline + matmul(matmul(input, a.T), b.T) * Float(0.4)
        let actual = model.projection(input)
        #expect(max(abs(expected - actual)).item(Float.self) < 0.0001)
        #expect(max(abs(actual - baseline)).item(Float.self) > 0)
        if quantized {
            #expect(original.weight.asArray(UInt32.self) == originalPackedWeight)
        } else {
            #expect(original.weight.asArray(Float.self) == originalFloatWeight)
        }
        #expect((original is QuantizedLinear) == quantized)
    }

    @Test func zeroScaleIsAnExactNoOp() throws {
        let model = Model(quantized: true)
        let original = model.projection
        let count = try LTXLoRALoader.apply([.init(url: URL(fileURLWithPath: "/unused.safetensors"), scale: 0)], to: model)
        #expect(count == 0)
        #expect(model.projection === original)
    }

    @Test(arguments: ["missing-pair", "shape", "unknown-layer", "duplicate", "alpha", "unsupported", "scale"])
    func rejectsInvalidAdaptersWithoutPartiallyReplacingLayers(reason: String) throws {
        let model = Model(quantized: true)
        let original = model.projection
        var tensors = ["projection.lora_A.weight": MLXArray.ones([2, 64]),
                       "projection.lora_B.weight": MLXArray.ones([32, 2])]
        switch reason {
        case "missing-pair": tensors.removeValue(forKey: "projection.lora_B.weight")
        case "shape": tensors["projection.lora_B.weight"] = MLXArray.ones([16, 2])
        case "unknown-layer": tensors["unknown.lora_A.weight"] = MLXArray.ones([2, 64]); tensors["unknown.lora_B.weight"] = MLXArray.ones([32, 2])
        case "duplicate": tensors["diffusion_model.projection.lora_A.weight"] = MLXArray.ones([2, 64])
        case "alpha": tensors["projection.alpha"] = MLXArray([Float.nan])
        case "unsupported": tensors["projection.lora_magnitude_vector"] = MLXArray.ones([32])
        default: break
        }
        let file = try fixture(tensors)
        defer { try? FileManager.default.removeItem(at: file) }
        #expect(throws: LTXLoRAError.self) {
            try LTXLoRALoader.apply([.init(url: file, scale: reason == "scale" ? .nan : 0.8)], to: model)
        }
        #expect(model.projection === original)
    }

    @Test func officialPrefixesAndFeedForwardPathsAreMapped() {
        #expect(LTXLoRALoader.normalize("diffusion_model.transformer_blocks.0.attn1.to_out.0") == "transformer_blocks.0.attn1.to_out")
        #expect(LTXLoRALoader.normalize("model.diffusion_model.transformer_blocks.0.ff.net.0.proj") == "transformer_blocks.0.ff.proj_in")
        #expect(LTXLoRALoader.normalize("transformer.transformer_blocks.0.audio_ff.net.2") == "transformer_blocks.0.audio_ff.proj_out")
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["GENIMAGE_LTX_LORA_MODEL"] != nil))
    func installedAdapterMapsAllLayersAndChangesRealQuantizedProjection() throws {
        let environment = ProcessInfo.processInfo.environment
        let root = URL(fileURLWithPath: try #require(environment["GENIMAGE_LTX_LORA_MODEL"]))
        let adapter = URL(fileURLWithPath: try #require(environment["GENIMAGE_LTX_LORA_FILE"]))
        // Verify the full model layout while evaluating only one real projection.
        // Loading the entire 22B model is unnecessary for this adapter compatibility check.
        let model = LTXTransformer(configuration: try LTXTransformerConfiguration.load(from: root))
        let weights = try MLX.loadArrays(url: root.appendingPathComponent("transformer-distilled-1.1.safetensors"))
        let prefix = "transformer.transformer_blocks.0.attn1.to_q."
        let original = QuantizedLinear(
            weight: try #require(weights[prefix + "weight"]),
            bias: weights[prefix + "bias"]?.asType(.bfloat16),
            scales: try #require(weights[prefix + "scales"]).asType(.bfloat16),
            biases: weights[prefix + "biases"]?.asType(.bfloat16),
            groupSize: 64, bits: 4)
        try model.update(modules: ModuleChildren.unflattened([
            ("transformer_blocks.0.attn1.to_q", original as Module)
        ]), verify: .noUnusedKeys)
        let input = MLXArray.ones([1, 2, model.configuration.videoDim], dtype: .bfloat16)
        let baseline = original(input)
        eval(baseline)
        let count = try LTXLoRALoader.apply([.init(url: adapter, scale: 0.8)], to: model)
        #expect(count == 480)
        let actual = model.transformerBlocks[0].attn1.toQ(input)
        #expect(all(isFinite(actual)).item(Bool.self))
        let difference = max(abs(actual.asType(.float32) - baseline.asType(.float32))).item(Float.self)
        #expect(difference > 0)
        #expect(original.weight.dtype == .uint32)
        #expect(original.bits == 4)
    }

    private func fixture(_ tensors: [String: MLXArray]) throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".safetensors")
        try MLX.save(arrays: tensors, url: url)
        return url
    }
}

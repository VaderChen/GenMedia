import Foundation
import MLX
import MLXNN
import Testing
import ZImage

@Suite(.serialized)
struct LoRACompatibilityTests {
    private final class Model: Module {
        @ModuleInfo var projection: Linear
        init(mode: QuantizationMode) {
            let weights = MLXArray((0..<(32 * 64)).map { Float($0 % 19) / 100 }).reshaped([32, 64])
            let (packed, scales, biases) = MLX.quantized(weights, groupSize: 32, bits: 4, mode: mode)
            self._projection = ModuleInfo(wrappedValue: QuantizedLinear(
                weight: packed, scales: scales, biases: biases, groupSize: 32, bits: 4, mode: mode))
            super.init()
        }
    }

    @Test(arguments: ["affine", "mxfp4"])
    func quantizedAdapterChangesOutputAndCanBeCleared(mode: String) throws {
        let quantizationMode: QuantizationMode = mode == "affine" ? .affine : .mxfp4
        let model = Model(mode: quantizationMode)
        let original = try #require(model.projection as? QuantizedLinear)
        let packed = original.weight.asArray(UInt32.self)
        let input = MLXArray.ones([1, 2, 64])
        let baseline = original(input)
        eval(baseline)
        let down = MLXArray.ones([2, 64]) * Float(0.01)
        let up = MLXArray.ones([32, 2]) * Float(0.02)
        let adapter = LoRAWeights(weights: ["projection.weight": (down: down, up: up)], rank: 2)
        #expect(LoRAApplicator.applyDynamically(to: model, loraWeights: adapter, scale: 0.8) == 1)
        let wrapped = try #require(model.projection as? LoRAQuantizedLinear)
        #expect(wrapped.mode == quantizationMode)
        #expect(wrapped.weight.asArray(UInt32.self) == packed)
        let expected = baseline + matmul(matmul(input, down.T), up.T) * Float(0.8)
        #expect(max(abs(model.projection(input) - expected)).item(Float.self) < 0.0001)
        #expect(max(abs(model.projection(input) - baseline)).item(Float.self) > 0)
        LoRAApplicator.clearDynamicLoRA(from: model)
        #expect(max(abs(model.projection(input) - baseline)).item(Float.self) == 0)
    }

    @Test func loaderRetainsAdaLNWeightsAndRejectsUnknownTargets() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        let prefix = "diffusion_model.layers.0.adaLN_modulation.0"
        try MLX.save(arrays: [prefix + ".lora_A.weight": MLXArray.ones([2, 256]),
                              prefix + ".lora_B.weight": MLXArray.ones([15360, 2])], url: url)
        let adapter = try LoRAWeightLoader.load(from: url)
        #expect(adapter.layerCount == 1)
        #expect(adapter.weights["layers.0.adaLN_modulation.0.weight"] != nil)
        let model = Model(mode: .affine)
        #expect(LoRAApplicator.applyDynamically(to: model, loraWeights: adapter, scale: 0.8) == 0)
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["GENIMAGE_ZIMAGE_LORA_MODEL"] != nil))
    func installedStyleMatchesAll240TransformerLayers() throws {
        let environment = ProcessInfo.processInfo.environment
        let root = URL(fileURLWithPath: try #require(environment["GENIMAGE_ZIMAGE_LORA_MODEL"]))
        let file = URL(fileURLWithPath: try #require(environment["GENIMAGE_ZIMAGE_LORA_FILE"]))
        let configuration = try JSONDecoder().decode(ZImageTransformerConfig.self,
            from: Data(contentsOf: root.appendingPathComponent("transformer/config.json")))
        let model = ZImageTransformer2DModel(configuration: configuration)
        // Check all real adapter keys without evaluating random base weights.
        let adapter = try LoRAWeightLoader.load(from: file)
        #expect(adapter.layerCount == 240)
        #expect(LoRAApplicator.applyDynamically(to: model, loraWeights: adapter, scale: 0.8) == 240)
    }
}

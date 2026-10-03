import Foundation
import GenImageCore
import MLX
import Testing
@testable import QwenImage21Runtime

@Suite(.serialized)
struct Qwen21AccelerationTests {
    @Test func turboUsesShiftedSixPointScheduleWithoutTerminalStretch() throws {
        let scheduler = Qwen21Scheduler(base_image_seq_len: 256, max_image_seq_len: 8192,
            base_shift: 0.5, max_shift: 0.9, shift_terminal: 0.02)
        let actual = try scheduler.sigmas(steps: 6, imageTokens: 256, acceleration: .viggleV03)
        // Independent scalar values from FlowMatchEulerDiscreteScheduler with sigmas supplied
        // by Viggle v0.3, mu=0.5 and shift_terminal=None.
        let expected: [Float] = [1, 0.961136737, 0.920262329, 0.831824344, 0.622459331, 0.354661244, 0]
        #expect(zip(actual, expected).allSatisfy { abs($0 - $1) < 1e-6 })
        let larger = try scheduler.sigmas(steps: 6, imageTokens: 4096, acceleration: .viggleV03)
        #expect(larger[5] > actual[5])
        #expect(throws: Qwen21Error.self) { try scheduler.sigmas(steps: 8, imageTokens: 256, acceleration: .viggleV03) }
    }

    @Test func requestKeepsOldProtocolAndRejectsPartialTurboConfiguration() throws {
        let json = #"{"modelDirectory":"/model","outputPaths":["/a.png"],"prompt":"cat","negativePrompt":"","width":256,"height":256,"steps":40,"seed":42}"#
        var request = try JSONDecoder().decode(Qwen21Request.self, from: Data(json.utf8))
        try request.validate()
        request.acceleration = .viggleV03
        #expect(throws: Qwen21Error.self) { try request.validate() }
        request.steps = 6; request.loraPath = "/lora.safetensors"; request.loraScale = 1
        try request.validate()
        request.loraScale = 0.8
        #expect(throws: Qwen21Error.self) { try request.validate() }
    }

    @Test func unmergedLoRAMatchesDenseOracleAndLeavesQuantizedBaseUnchanged() throws {
        let w = MLXArray((0..<128).map { sin(Float($0)) }).reshaped([2, 64])
        let (packed, scales, biases) = quantized(w, groupSize: 64, bits: 4)
        let base = Qwen21Weights(tensors: ["modulation.0.weight": packed,
            "modulation.0.scales": scales, "modulation.0.biases": biases!])
        let a = MLXArray((0..<128).map { Float($0) / 1024 }).reshaped([2, 64])
        let b = MLXArray([Float(0.1), -0.2, 0.3, 0.4]).reshaped([2, 2])
        let x = MLXArray((0..<64).map { cos(Float($0)) }).reshaped([1, 64])
        let before = try base.linear(x, "modulation.0").asArray(Float.self)
        var adapted = base
        adapted.lora = try Qwen21LoRAAdapter(tensors: ["transformer.modulation.1.lora_A.weight": a,
            "transformer.modulation.1.lora_B.weight": b, "transformer.modulation.1.alpha": MLXArray(Float(1))], base: base)
        let expectedWeight = dequantized(packed, scales: scales, biases: biases, groupSize: 64, bits: 4) + matmul(b, a) * 0.5
        let expected = matmul(x, expectedWeight.T)
        let actual = try adapted.linear(x, "modulation.0")
        #expect(max(abs(actual - expected)).item(Float.self) < 1e-4)
        #expect(try base.linear(x, "modulation.0").asArray(Float.self) == before)
        #expect(max(abs(actual - MLXArray(before).reshaped([1, 2]))).item(Float.self) > 0.001)
    }

    @Test func rejectsMissingPairsUnknownLayersAndMismatchedDimensions() throws {
        let base = Qwen21Weights(tensors: ["layer.weight": MLXArray.zeros([2, 64])])
        let a = MLXArray.zeros([1, 64]), b = MLXArray.zeros([2, 1])
        for invalid in [[:], ["layer.lora_A.weight": a],
                        ["other.lora_A.weight": a, "other.lora_B.weight": b],
                        ["layer.lora_A.weight": a, "layer.lora_B.weight": MLXArray.zeros([3, 1])],
                        ["layer.lora_A.weight": a, "layer.lora_B.weight": b, "junk": a]] {
            #expect(throws: Qwen21Error.self) { try Qwen21LoRAAdapter(tensors: invalid, base: base) }
        }
        #expect(Qwen21LoRAAdapter.normalized("transformer.time_text_embed.timestep_embedder.linear_1") == "time_text_embed.linear_1")
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["GENIMAGE_QWEN21_LORA"] != nil))
    func installedViggleAdapterMatchesEveryPackedBaseLayer() throws {
        let path = try #require(ProcessInfo.processInfo.environment["GENIMAGE_QWEN21_LORA"])
        let root = try #require(ProcessInfo.processInfo.environment["GENIMAGE_QWEN21_MODEL"])
        let base = try Qwen21Weights(directory: URL(fileURLWithPath: root).appendingPathComponent("transformer"))
        let adapter = try Qwen21LoRAAdapter(url: URL(fileURLWithPath: path), base: base)
        #expect(adapter.pairs.count == 227)
        #expect(adapter.pairs.values.allSatisfy { $0.a.dim(0) == 128 && $0.scale == 1 })
        let layer = "time_text_embed.linear_1"
        let pair = try #require(adapter.pairs[layer])
        let delta = try #require(adapter.delta(MLXArray.ones([1, pair.a.dim(1)], dtype: .bfloat16), layer: layer))
        #expect(all(isFinite(delta)).item(Bool.self))
        #expect(max(abs(delta)).item(Float.self) > 0)
    }
}

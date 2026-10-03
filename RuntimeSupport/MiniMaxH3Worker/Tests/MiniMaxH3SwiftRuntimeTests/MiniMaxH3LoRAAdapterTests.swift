import Foundation
import GenImageCore
import MLX
import Testing
@testable import MiniMaxH3SwiftRuntime

@Suite(.serialized)
struct MiniMaxH3LoRAAdapterTests {
    @Test(arguments: [false, true])
    func transformerResidualHonorsAlphaAndClearsWithoutChangingBase(quantized: Bool) throws {
        let prefix = "blocks.0.attn.out_proj"
        let weight = MLXArray((0..<(32 * 64)).map { Float($0 % 19) / 100 }).reshaped([32, 64])
        var weights = [prefix + ".weight": weight, prefix + ".bias": MLXArray.ones([32])]
        if quantized {
            let packed = MLX.quantized(weight, groupSize: 64, bits: 8)
            weights[prefix + ".weight"] = packed.wq
            weights[prefix + ".scales"] = packed.scales
            weights[prefix + ".biases"] = packed.biases
        }
        let originalPacked = quantized ? weights[prefix + ".weight"]!.asArray(UInt32.self) : []
        let originalDense = quantized ? [] : weight.asArray(Float.self)
        let transformer = MiniMaxH3Transformer(weights: weights, quantizedPrefixes: quantized ? [prefix] : [])
        let input = MLXArray.ones([1, 2, 64])
        let baseline = try transformer.linear(input, prefix, bias: true)
        eval(baseline)
        let a = MLXArray((0..<128).map { Float($0 % 7) / 100 }).reshaped([2, 64])
        let b = MLXArray((0..<64).map { Float($0 % 5) / 100 }).reshaped([32, 2])
        // LightX2V 8-step uses alpha/rank = 1/16, including fused QKV.
        let file = try fixture(["diffusion_model." + prefix + ".lora_A.weight": a,
                                "diffusion_model." + prefix + ".lora_B.weight": b,
                                "diffusion_model." + prefix + ".alpha": MLXArray(Float(0.125))])
        defer { try? FileManager.default.removeItem(at: file) }
        #expect(try transformer.loadLoRAs([.init(path: file.path, scale: 0.8)]) == 1)
        let expected = baseline + matmul(matmul(input, a.T), b.T) * Float(0.05)
        let actual = try transformer.linear(input, prefix, bias: true)
        #expect(max(abs(expected - actual)).item(Float.self) < 0.0001)
        #expect(max(abs(actual - baseline)).item(Float.self) > 0)
        if quantized {
            #expect(weights[prefix + ".weight"]!.asArray(UInt32.self) == originalPacked)
        } else {
            #expect(weight.asArray(Float.self) == originalDense)
        }
        #expect(try transformer.loadLoRAs([]) == 0)
        #expect(try transformer.linear(input, prefix, bias: true).asArray(Float.self) == baseline.asArray(Float.self))
        #expect(try transformer.loadLoRAs([.init(path: "/missing.safetensors", scale: 0)]) == 0)
    }

    @Test func larryStyleWithoutAlphaUsesTheRequestedStrength() throws {
        let a = MLXArray.ones([2, 4])
        let b = MLXArray.ones([3, 2])
        let file = try fixture(["blocks.0.adaln_proj.linear.lora_A.weight": a,
                                "blocks.0.adaln_proj.linear.lora_B.weight": b])
        defer { try? FileManager.default.removeItem(at: file) }
        let adapter = try MiniMaxH3LoRAAdapter.load([.init(path: file.path, scale: 0.5)],
            shapes: ["blocks.0.adaln_proj.linear": [3, 4]])
        let output = adapter.apply(to: .zeros([1, 3]), input: .ones([1, 4]), path: "blocks.0.adaln_proj.linear")
        #expect(output.asArray(Float.self) == [4, 4, 4])
    }

    @Test(arguments: ["missing-pair", "rank", "shape", "unknown-layer", "duplicate", "alpha", "alpha-vector", "unsupported", "scale"])
    func rejectsMalformedAdaptersWithoutChangingTheActiveAdapter(reason: String) throws {
        let prefix = "projection"
        let transformer = MiniMaxH3Transformer(weights: [prefix + ".weight": .zeros([3, 4])])
        let valid = [prefix + ".lora_A.weight": MLXArray.ones([2, 4]),
                     prefix + ".lora_B.weight": MLXArray.ones([3, 2])]
        let previous = try fixture(valid)
        defer { try? FileManager.default.removeItem(at: previous) }
        try transformer.loadLoRAs([.init(path: previous.path, scale: 0.5)])
        let input = MLXArray.ones([1, 4])
        let baseline = try transformer.linear(input, prefix).asArray(Float.self)
        var tensors = valid
        switch reason {
        case "missing-pair": tensors.removeValue(forKey: prefix + ".lora_B.weight")
        case "rank": tensors[prefix + ".lora_B.weight"] = .ones([3, 1])
        case "shape": tensors[prefix + ".lora_B.weight"] = .ones([4, 2])
        case "unknown-layer": tensors["unknown.lora_A.weight"] = .ones([2, 4]); tensors["unknown.lora_B.weight"] = .ones([3, 2])
        case "duplicate": tensors["diffusion_model." + prefix + ".lora_A.weight"] = .ones([2, 4])
        case "alpha": tensors[prefix + ".alpha"] = MLXArray(Float.nan)
        case "alpha-vector": tensors[prefix + ".alpha"] = .ones([2])
        case "unsupported": tensors[prefix + ".lora_magnitude_vector"] = .ones([3])
        default: break
        }
        let file = try fixture(tensors)
        defer { try? FileManager.default.removeItem(at: file) }
        #expect(throws: MiniMaxH3LoRAError.self) {
            try transformer.loadLoRAs([.init(path: file.path, scale: reason == "scale" ? .nan : 0.8)])
        }
        #expect(try transformer.linear(input, prefix).asArray(Float.self) == baseline)
    }

    @Test func workerContractRejectsMissingPresetWrongStepsAndRef2VA() throws {
        let file = try fixture(["test": MLXArray(Float(1))])
        defer { try? FileManager.default.removeItem(at: file) }
        let loras = [MiniMaxH3LoRAConfiguration(path: file.path, scale: 1)]
        try MiniMaxH3LoRAAdapter.validateRequest(loras: [], acceleration: nil, steps: 30)
        try MiniMaxH3LoRAAdapter.validateRequest(loras: loras, acceleration: .lightX2V4Step768, steps: 4)
        #expect(throws: MiniMaxH3LoRAError.self) {
            try MiniMaxH3LoRAAdapter.validateRequest(loras: loras, acceleration: nil, steps: 4)
        }
        #expect(throws: MiniMaxH3LoRAError.self) {
            try MiniMaxH3LoRAAdapter.validateRequest(loras: [], acceleration: .lightX2V4Step768, steps: 4)
        }
        #expect(throws: MiniMaxH3LoRAError.self) {
            try MiniMaxH3LoRAAdapter.validateRequest(loras: loras, acceleration: .lightX2V4Step768, steps: 8)
        }
        #expect(throws: MiniMaxH3LoRAError.self) {
            try MiniMaxH3LoRAAdapter.validateRequest(loras: loras, acceleration: .larryV4, steps: 6, isRef2VA: true)
        }
    }

    @Test func accelerationSelectsItsVideoAndAudioSchedules() {
        let unused = URL(fileURLWithPath: "/unused")
        for acceleration in MiniMaxH3Acceleration.allCases {
            let pipeline = MiniMaxH3Pipeline(transformerURL: unused, videoVAEURL: unused, acceleration: acceleration)
            let schedule = pipeline.scheduler
            #expect(schedule.videoShift == (acceleration == .larryV4 ? 12 : 6))
            #expect(schedule.audioShift == 3)
            let sigmas = schedule.sigmas(steps: acceleration.defaultSteps)
            #expect(sigmas.count == acceleration.defaultSteps + 1)
            #expect(sigmas.first == 1)
            #expect(sigmas.last == 0)
            for index in 0..<acceleration.defaultSteps {
                let baseTime = 1 - Double(index) / Double(acceleration.defaultSteps)
                let expectedAudio = 3 * baseTime / (1 + 2 * baseTime)
                #expect(abs(schedule.audioSigma(videoSigma: sigmas[index]) - expectedAudio) < 1e-10)
            }
        }
        let schedule = MiniMaxH3Pipeline(transformerURL: unused, videoVAEURL: unused, acceleration: .lightX2V4Step768).scheduler
        let reference = [1.0, 18.0 / 19, 6.0 / 7, 2.0 / 3, 0.0]
        for (actual, expected) in zip(schedule.sigmas(steps: 4), reference) {
            #expect(abs(actual - expected) < 1e-12)
        }
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["GENIMAGE_H3_LORA_MODEL_ROOT"] != nil))
    func realAdaptersMatchTheBaseLayoutsAndProduceFiniteResiduals() throws {
        let root = URL(fileURLWithPath: try #require(ProcessInfo.processInfo.environment["GENIMAGE_H3_LORA_MODEL_ROOT"]))
        for acceleration in [MiniMaxH3Acceleration.lightX2V8Step768, .larryV4] {
            try checkInstalledAdapter(acceleration, root: root)
            MLX.Memory.clearCache()
        }
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["GENIMAGE_H3_LORA_HEADER_ROOT"] != nil))
    func publishedHeadersMatchEverySupportedBaseLayout() throws {
        let root = URL(fileURLWithPath: try #require(ProcessInfo.processInfo.environment["GENIMAGE_H3_LORA_HEADER_ROOT"]))
        for acceleration in MiniMaxH3Acceleration.allCases {
            let file = root.appendingPathComponent(acceleration.entry.filename + ".header.json")
            let header = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any])
            let keys = header.keys.filter { $0.hasSuffix(".lora_A.weight") }
            #expect(keys.count == (acceleration == .larryV4 ? 259 : 208))
            let bases: [MiniMaxH3Configuration] = acceleration.supportsPrunedBase ? [.fl2va, .fl2vaPruned] : [.fl2va]
            for base in bases {
                for key in keys {
                    let stem = String(key.dropLast(".lora_A.weight".count))
                    let a = try #require((header[key] as? [String: Any])?["shape"] as? [Int])
                    let b = try #require((header[stem + ".lora_B.weight"] as? [String: Any])?["shape"] as? [Int])
                    #expect(a.count == 2 && b.count == 2)
                    #expect(a[0] == b[1])
                    let path = MiniMaxH3LoRAAdapter.normalize(stem)
                    #expect(base.expectedTensorShapes[path + ".weight"] == [b[0], a[1]])
                }
            }
        }
    }

    private func checkInstalledAdapter(_ acceleration: MiniMaxH3Acceleration, root: URL) throws {
        let entry = acceleration.entry
        let file = root.appendingPathComponent(entry.directoryName).appendingPathComponent(entry.filename)
        let base = acceleration.supportsPrunedBase ? MiniMaxH3Configuration.fl2vaPruned : .fl2va
        let shapes = Dictionary(uniqueKeysWithValues: base.expectedTensorShapes.compactMap { key, shape in
            key.hasSuffix(".weight") && shape.count == 2 ? (String(key.dropLast(7)), shape) : nil
        })
        let adapter = try MiniMaxH3LoRAAdapter.load([.init(path: file.path, scale: 1)], shapes: shapes)
        #expect(adapter.layerCount == (acceleration == .larryV4 ? 259 : 208))
        let prefix = "blocks.0.attn.out_proj"
        let output = adapter.apply(to: .zeros([1, 2, base.hiddenSize]),
            input: .ones([1, 2, base.attentionInnerDim]), path: prefix)
        #expect(all(isFinite(output)).item(Bool.self))
        #expect(max(abs(output)).item(Float.self) > 0)
        print("H3 adapter \(acceleration.rawValue): \(adapter.layerCount) layers; finite nonzero residual")
    }

    private func fixture(_ tensors: [String: MLXArray]) throws -> URL {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".safetensors")
        try MLX.save(arrays: tensors, url: file)
        return file
    }
}

import Foundation
import MLX
import Testing
@testable import LTXVideoSwiftRuntime

struct LTXLatentUpsamplerWeightLoaderTests {
    @Test(arguments: [false, true])
    func numericConvolutionKeyLoadsAsAModule(temporal: Bool) throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let configuration: [String: Any] = [
            "in_channels": 2, "mid_channels": 32, "num_blocks_per_stage": 0,
            "dims": 3, "spatial_upsample": !temporal, "temporal_upsample": temporal,
            "spatial_scale": 2, "rational_resampler": false
        ]
        try JSONSerialization.data(withJSONObject: ["config": configuration])
            .write(to: directory.appendingPathComponent("upscale_config.json"))
        // Use the checkpoint's numeric module path, independent of model.parameters().
        let weights: [String: MLXArray] = [
            "initial_conv.weight": .zeros([32, 3, 3, 3, 2]),
            "initial_conv.bias": .zeros([32]),
            "initial_norm.weight": .ones([32]),
            "initial_norm.bias": .zeros([32]),
            "upsampler.0.weight": .zeros(temporal ? [64, 3, 3, 3, 32] : [128, 3, 3, 32]),
            "upsampler.0.bias": .zeros([temporal ? 64 : 128]),
            "final_conv.weight": .zeros([2, 3, 3, 3, 32]),
            "final_conv.bias": MLXArray([Float(0.25), -0.5])
        ]
        let url = directory.appendingPathComponent("upscale.safetensors")
        try MLX.save(arrays: weights, url: url)
        let loaded = try LTXLatentUpsamplerWeightLoader.load(from: url, modelDirectory: directory)
        #expect(loaded.report.tensorCount == weights.count)
        let output = try loaded.model.upsample(.ones([1, 2, 2, 2, 2]))
        #expect(output.shape == (temporal ? [1, 2, 3, 2, 2] : [1, 2, 2, 4, 4]))
        #expect(all(output[0, 0] .== Float(0.25)).item(Bool.self))
        #expect(all(output[0, 1] .== Float(-0.5)).item(Bool.self))
    }
}

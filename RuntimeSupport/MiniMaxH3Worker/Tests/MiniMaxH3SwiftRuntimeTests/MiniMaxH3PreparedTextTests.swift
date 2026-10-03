import Foundation
import MLX
import Testing
@testable import MiniMaxH3SwiftRuntime

@Suite(.serialized)
struct MiniMaxH3PreparedTextTests {
    @Test func preparedTextPreservesVideoAndAudioAcrossStepsAndLoRA() throws {
        for pruned in [false, true] {
            var configuration = MiniMaxH3Configuration.fl2va
            configuration.hiddenSize = 16
            configuration.conditionInputDim = 12
            configuration.layerCount = 1
            configuration.tokenRefinerLayerCount = 2
            configuration.attentionHeadCount = 2
            configuration.attentionHeadDim = 8
            configuration.ffnHiddenSize = 24
            configuration.timeEmbedDim = 8
            configuration.ropeInvFreqLength = 1
            configuration.videoLatentChannels = 2
            configuration.audioLatentChannels = 2
            configuration.adalnCurveGrid = pruned ? 9 : 0
            var weights: [String: MLXArray] = [:]
            for (name, shape) in configuration.expectedTensorShapes {
                let count = shape.reduce(1, *)
                let norm = name.contains("norm")
                weights[name] = MLXArray((0..<count).map { norm ? Float(1) : Float($0 % 23 - 11) / 100 }, shape)
            }
            let transformer = MiniMaxH3Transformer(configuration: configuration, weights: weights)
            let text = MLXArray((0..<36).map { Float($0 % 13) / 10 }, [3, 12])
            let video = MLXArray((0..<8).map { Float($0) / 10 }, [1, 2, 1, 2, 2])
            let audio = MLXArray((0..<8).map { Float($0) / 20 }, [1, 2, 2, 2])
            let adapterURL = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".safetensors")
            defer { try? FileManager.default.removeItem(at: adapterURL) }
            try MLX.save(arrays: [
                "condition_proj.lora_A.weight": MLXArray.ones([2, 12]) * Float(0.01),
                "condition_proj.lora_B.weight": MLXArray.ones([16, 2]) * Float(0.02),
            ], url: adapterURL)
            for withLoRA in [false, true] {
                try transformer.loadLoRAs(withLoRA ? [.init(path: adapterURL.path, scale: 0.8)] : [])
                let refined = try transformer.refineTextStates(text)
                MLX.eval(refined)
                let conditioning = MiniMaxH3Transformer.Conditioning(entries: [
                    .init(resolvedFrameIndex: 0, videoLatent: video, audioLatent: audio)
                ], visualNoise: MLXArray.zeros([1, 8]))
                for context in [nil, conditioning] {
                    for sigma: Float in [1, 0.75, 0.3, 0] {
                        let expected = try transformer.forward(videoLatent: video, audioLatent: audio,
                            textStates: text, sigma: sigma, conditioning: context)
                        let actual = try transformer.forward(videoLatent: video, audioLatent: audio,
                            textStates: refined, sigma: sigma, conditioning: context)
                        #expect(actual.video.asArray(Float.self).map(\.bitPattern) == expected.video.asArray(Float.self).map(\.bitPattern))
                        #expect(actual.audio.asArray(Float.self).map(\.bitPattern) == expected.audio.asArray(Float.self).map(\.bitPattern))
                    }
                }
            }
        }
    }
}

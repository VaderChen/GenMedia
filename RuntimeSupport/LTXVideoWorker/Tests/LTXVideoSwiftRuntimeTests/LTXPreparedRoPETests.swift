import Foundation
import MLX
import MLXNN
import Testing
@testable import LTXVideoSwiftRuntime

@Suite(.serialized)
struct LTXPreparedRoPETests {
    @Test(arguments: [LTXRoPEType.split, .interleaved], [false, true])
    func preparedTablesPreserveVelocityAndStageChanges(type: LTXRoPEType, keyframes: Bool) throws {
        let config = try LTXTransformerConfiguration(numLayers: 2, videoDim: 16, audioDim: 8,
            videoNumHeads: 2, audioNumHeads: 2, videoHeadDim: 8, audioHeadDim: 4,
            avCrossNumHeads: 2, avCrossHeadDim: 4, videoPatchChannels: 8, audioPatchChannels: 4,
            timestepEmbeddingDim: 8, ropeType: type, videoFFBias: !keyframes,
            useKeyframesEmbedding: keyframes)
        let transformer = LTXTransformer(configuration: config)
        let weights = Dictionary(uniqueKeysWithValues: transformer.parameters().flattened().enumerated().map { index, pair in
            let values = (0..<pair.1.size).map { Float(($0 + index) % 17 - 8) * 0.01 }
            return (pair.0, MLXArray(values, pair.1.shape).asType(.bfloat16))
        })
        try transformer.update(parameters: .unflattened(weights), verify: .all)
        eval(transformer)
        let audio = MLXArray((0..<12).map { Float($0) / 32 }, [1, 3, 4]).asType(.bfloat16)
        let videoText = MLXArray.ones([1, 3, 16], dtype: .bfloat16) * 0.25
        let audioText = MLXArray.ones([1, 2, 8], dtype: .bfloat16) * 0.125
        for tokens in [4, 8, 4] {
            let video = MLXArray((0..<(tokens * 8)).map { Float($0 % 19) / 32 }, [1, tokens, 8]).asType(.bfloat16)
            let videoPositions = LTXDistilledPositionBuilder.video(frameCount: 1, height: 2, width: tokens / 2, frameRate: 24)
            let audioPositions = LTXDistilledPositionBuilder.audio(tokenCount: 3)
            let prepared = transformer.prepareRoPE(videoPositions: videoPositions, audioPositions: audioPositions)
            #expect(prepared.video?.cos.shape[2] == tokens)
            let mask = keyframes ? MLXArray.ones([1, tokens, 1]) : nil
            for sigma: Float in [1, 0.75, 0.25, 0] {
                let time = MLXArray([sigma]).asType(.bfloat16)
                let expected = transformer(videoLatent: video, audioLatent: audio, timestep: time,
                    videoTextEmbeds: videoText, audioTextEmbeds: audioText,
                    videoPositions: videoPositions, audioPositions: audioPositions, videoKeyframeMask: mask)
                let actual = transformer(videoLatent: video, audioLatent: audio, timestep: time,
                    videoTextEmbeds: videoText, audioTextEmbeds: audioText,
                    videoPositions: videoPositions, audioPositions: audioPositions, videoKeyframeMask: mask,
                    preparedRoPE: prepared)
                for (left, right) in [(expected.video, actual.video), (expected.audio, actual.audio)] {
                    #expect(left.asType(.float32).asArray(Float.self).map(\.bitPattern)
                        == right.asType(.float32).asArray(Float.self).map(\.bitPattern))
                    #expect(all(isFinite(right)).item(Bool.self))
                }
            }
        }
        let empty = transformer.prepareRoPE(videoPositions: nil, audioPositions: nil)
        #expect(empty.video == nil && empty.audio == nil && empty.videoCross == nil && empty.audioCross == nil)
    }
}

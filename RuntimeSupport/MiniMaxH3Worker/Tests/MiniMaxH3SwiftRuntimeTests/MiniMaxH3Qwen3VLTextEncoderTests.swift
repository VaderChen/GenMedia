import Testing
@testable import MiniMaxH3SwiftRuntime

@Suite("MiniMax H3 Qwen3-VL text encoder")
struct MiniMaxH3Qwen3VLTextEncoderTests {
    @Test("Configuration matches the real checkpoint")
    func configurationMatchesCheckpoint() {
        let configuration = MiniMaxH3Qwen3VLTextEncoder.Configuration.minimaxH3
        #expect(configuration.hiddenSize == 5120)
        #expect(configuration.vocabularySize == 151936)
        #expect(configuration.layerCount == 50)
        #expect(configuration.intermediateSize == 25600)
        #expect(configuration.attentionHeadCount == 64)
        #expect(configuration.keyValueHeadCount == 8)
        #expect(configuration.headDimension == 128)
        #expect(configuration.ropeTheta == 5_000_000)
        #expect(configuration.rmsNormEps == 1e-6)
    }

    @Test("Error descriptions preserve diagnostic values")
    func errorDescriptionsPreserveDiagnosticValues() {
        #expect(
            MiniMaxH3Qwen3VLTextEncoder.Error.invalidCheckpoint("bad header")
                .errorDescription == "MiniMax H3 Qwen3-VL checkpoint 無效：bad header"
        )
        #expect(
            MiniMaxH3Qwen3VLTextEncoder.Error.missingTensor("model.embed_tokens.weight")
                .errorDescription == "MiniMax H3 Qwen3-VL 缺少權重：model.embed_tokens.weight"
        )
    }
}

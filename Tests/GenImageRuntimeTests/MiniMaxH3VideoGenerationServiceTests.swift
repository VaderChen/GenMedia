import Testing

@testable import GenImageRuntime

struct MiniMaxH3VideoGenerationServiceTests {
    @Test func allCatalogGGUFVariantsAreSupported() {
        let supportedModelIDs = [
            "unsloth/MiniMax-H3-GGUF@fl2va-pruned-Q4_K",
            "Abiray/MiniMax-H3-GGUF@fl2va-Q4_0",
            "Abiray/MiniMax-H3-GGUF@fl2va-Q4_K_M",
            "Abiray/MiniMax-H3-GGUF@fl2va-Q4_K_S",
            "Abiray/MiniMax-H3-GGUF@ref2va-Q4_0",
            "Abiray/MiniMax-H3-GGUF@ref2va-Q4_K_M",
            "Abiray/MiniMax-H3-GGUF@ref2va-Q4_K_S"
        ]
        #expect(supportedModelIDs.allSatisfy {
            MiniMaxH3VideoGenerationService.isSupportedModelID($0)
        })
        #expect(
            !MiniMaxH3VideoGenerationService.isSupportedModelID(
                "pipenetwork/MiniMax-H3-MLX-4bit"
            )
        )
    }

    @Test func H3OutputFramesAreNormalizedToTheVAETemporalRatio() {
        #expect(MiniMaxH3VideoGenerationService.normalizedFrameCount(1) == 4)
        #expect(MiniMaxH3VideoGenerationService.normalizedFrameCount(5) == 4)
        #expect(MiniMaxH3VideoGenerationService.normalizedFrameCount(6) == 8)
        #expect(MiniMaxH3VideoGenerationService.normalizedFrameCount(124) == 124)
        #expect(MiniMaxH3VideoGenerationService.normalizedFrameCount(512) == 512)
    }

    @Test func H3SpatialDimensionsAreNormalizedToTransformerPatchGrid() {
        let normalized = MiniMaxH3VideoGenerationService.normalizedSpatialDimensions(
            width: 1280,
            height: 720
        )
        #expect(normalized.width == 1280)
        #expect(normalized.height == 704)
        #expect(
            MiniMaxH3VideoGenerationService.normalizedSpatialDimension(640) == 640
        )
    }
}

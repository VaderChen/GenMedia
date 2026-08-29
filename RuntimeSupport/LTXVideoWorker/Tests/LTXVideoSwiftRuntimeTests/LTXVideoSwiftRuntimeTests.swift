import Foundation
import MLX
import Testing

@testable import LTXVideoSwiftRuntime

struct LTXVideoSwiftRuntimeTests {
    @Test func distilledConfigurationSnapsProfileDimensions() throws {
        let configuration = try LTXDistilledGenerationConfiguration(
            width: 1280,
            height: 720,
            frames: 97,
            frameRate: 24,
            seed: 1
        )

        #expect(configuration.snappedDimensions.width == 1280)
        #expect(configuration.snappedDimensions.height == 704)
    }

    @Test func distilledAndStage2SigmaTablesMatchReference() {
        #expect(LTXDiffusionScheduler.distilledSigmas == [
            1.0, 0.99375, 0.9875, 0.98125, 0.975,
            0.909375, 0.725, 0.421875, 0.0
        ])
        #expect(LTXDiffusionScheduler.stage2Sigmas == [
            0.909375, 0.725, 0.421875, 0.0
        ])
    }

    @Test func dynamicSigmaScheduleMatchesPythonReference() throws {
        let actual = try LTXDiffusionScheduler.schedule(steps: 4, numTokens: 4096)
        let expected: [Float] = [1.0, 0.867083205818, 0.631568571232, 0.1, 0.0]
        #expect(actual.count == expected.count)
        for (left, right) in zip(actual, expected) {
            #expect(abs(left - right) < 1e-6)
        }
    }

    @Test func dynamicSigmaScheduleRespondsToTokenCount() throws {
        let short = try LTXDiffusionScheduler.schedule(steps: 4, numTokens: 1024)
        let long = try LTXDiffusionScheduler.schedule(steps: 4, numTokens: 4096)
        #expect(short[1] < long[1])
        #expect(short[2] < long[2])
        #expect(short.last == 0)
        #expect(long.last == 0)
    }

    @Test func eulerStepMatchesX0PredictionFormula() {
        let sample: [Float] = [4, 8]
        let denoised: [Float] = [1, 2]
        let actual = LTXDiffusionScheduler.eulerStep(
            sample: sample, denoised: denoised, sigma: 1.0, sigmaNext: 0.25
        )
        #expect(actual == [1.75, 3.5])
    }

    @Test func gemma3ConfigurationReadsFlatAndNestedLayouts() throws {
        let flat = """
        {
          "model_type": "gemma3_text",
          "hidden_size": 3840,
          "num_hidden_layers": 48,
          "intermediate_size": 15360,
          "num_attention_heads": 32,
          "head_dim": 128,
          "rms_norm_eps": 0.000001,
          "vocab_size": 262208,
          "num_key_value_heads": 16,
          "rope_theta": 1000000,
          "rope_local_base_freq": 10000,
          "query_pre_attn_scalar": 256,
          "sliding_window": 512,
          "sliding_window_pattern": 6,
          "max_position_embeddings": 32768
        }
        """
        let nested = """
        {
          "text_config": \(flat)
        }
        """
        let flatConfiguration = try JSONDecoder().decode(
            LTXGemma3TextConfiguration.self,
            from: Data(flat.utf8)
        )
        let nestedConfiguration = try JSONDecoder().decode(
            LTXGemma3TextConfiguration.self,
            from: Data(nested.utf8)
        )
        #expect(flatConfiguration == nestedConfiguration)
        #expect(flatConfiguration.hiddenLayers == 48)
        #expect(flatConfiguration.attentionHeads == 32)
        #expect(flatConfiguration.keyValueHeads == 16)
        #expect(flatConfiguration.ropeTheta == 1_000_000)
    }

    @Test func gemma3ConfigurationRejectsNonDivisibleGQA() {
        #expect(throws: LTXGemma3TextEncoderError.invalidConfiguration(
            "Gemma3 維度、head 設定或位置設定無效。"
        )) {
            try LTXGemma3TextConfiguration(
                hiddenSize: 32,
                hiddenLayers: 2,
                intermediateSize: 64,
                attentionHeads: 3,
                headDimension: 8,
                rmsNormEpsilon: 1e-6,
                vocabularySize: 128,
                keyValueHeads: 2
            )
        }
    }

    @Test func gemmaLeftPaddingKeepsLatestTokens() throws {
        let layout = try LTXGemmaFeaturePreparation.leftPad(
            tokenIDs: [1, 2, 3, 4, 5],
            maxLength: 4,
            padTokenID: 0
        )

        #expect(layout.tokenIDs == [2, 3, 4, 5])
        #expect(layout.attentionMask == [1, 1, 1, 1])

        let padded = try LTXGemmaFeaturePreparation.leftPad(
            tokenIDs: [7, 8],
            maxLength: 4,
            padTokenID: 99
        )
        #expect(padded.tokenIDs == [99, 99, 7, 8])
        #expect(padded.attentionMask == [0, 0, 1, 1])
    }

    @Test func gemmaProjectionShapeKeepsDimensionInterleavingContract() throws {
        let shapes = Array(
            repeating: [1, 2, 3840],
            count: LTXGemmaFeaturePreparation.hiddenStateCount
        )
        #expect(try LTXGemmaFeaturePreparation.projectionShape(for: shapes) == [1, 2, 188160])
    }

    @Test func gemmaProjectionStackRejectsWrongLayerCount() {
        let shapes = [[1, 2, 3840]]
        #expect(throws: LTXGemmaFeaturePreparationError.hiddenStateCount(
            expected: 49,
            actual: 1
        )) {
            try LTXGemmaFeaturePreparation.projectionShape(for: shapes)
        }
    }

    @Test func gemmaConnectorConfigurationMatchesLTX23Contract() throws {
        let configuration = try LTXGemmaConnectorConfiguration()
        #expect(configuration.gemmaLayerCount == 49)
        #expect(configuration.projectionInputDimension == 188160)
        #expect(configuration.videoDimension == 4096)
        #expect(configuration.audioDimension == 2048)
        #expect(configuration.layerCount == 8)
        #expect(configuration.registerCount == 128)
        #expect(configuration.maxPosition == 4096)
    }

    @Test func gemmaConnectorRejectsInvalidHeadDimensions() {
        #expect(throws: LTXGemmaConnectorRuntimeError.invalidConfiguration(
            "Gemma connector 維度、layer、register 或位置設定無效。"
        )) {
            try LTXGemmaConnectorConfiguration(
                videoDimension: 4097
            )
        }
    }

    @Test func videoPatchifyUsesFrameHeightWidthOrder() throws {
        let tokenShape = try LTXVideoLatentPatchifier.tokenShape(for: [1, 2, 1, 2, 2])
        #expect(tokenShape == [1, 4, 2])
        #expect(try LTXVideoLatentPatchifier.restoredShape(
            for: tokenShape,
            dimensions: [1, 2, 2]
        ) == [1, 2, 1, 2, 2])
    }

    @Test func audioPatchifyFlattensEightBySixteenLatentChannels() throws {
        let tokenShape = try LTXAudioLatentPatchifier.tokenShape(for: [1, 8, 2, 16])
        #expect(tokenShape == [1, 2, 128])
        #expect(try LTXAudioLatentPatchifier.restoredShape(for: tokenShape) == [1, 8, 2, 16])
    }

    @Test func modalitySplitByCountMatchesReferenceCoordinates() throws {
        let intervals = try LTXModalityTiling.splitByCount(
            dimension: 10,
            numTiles: 3,
            overlap: 1
        )

        #expect(intervals.starts == [0, 3, 6])
        #expect(intervals.ends == [4, 7, 10])
        #expect(intervals.leftRamps == [0, 1, 1])
        #expect(intervals.rightRamps == [1, 1, 0])
    }

    @Test func modalityTilesUseCartesianFrameHeightWidthOrder() throws {
        let configuration = LTXModalityTileConfiguration(
            frames: try LTXModalityDimensionConfiguration(numTiles: 2, overlap: 1),
            width: try LTXModalityDimensionConfiguration(numTiles: 2, overlap: 1)
        )
        let tiles = try LTXModalityTiling.makeTiles(
            frameCount: 4,
            height: 2,
            width: 3,
            configuration: configuration
        )

        #expect(tiles.count == 4)
        #expect(tiles.map { $0.input } == [
            [LTXVideoSlice(0, 3), LTXVideoSlice(0, 2), LTXVideoSlice(0, 2)],
            [LTXVideoSlice(0, 3), LTXVideoSlice(0, 2), LTXVideoSlice(1, 3)],
            [LTXVideoSlice(2, 4), LTXVideoSlice(0, 2), LTXVideoSlice(0, 2)],
            [LTXVideoSlice(2, 4), LTXVideoSlice(0, 2), LTXVideoSlice(1, 3)],
        ])
        #expect(LTXModalityTiling.tokenIndices(
            tile: tiles[0], gridShape: (frames: 4, height: 2, width: 3)
        ) == [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16])
    }

    @Test func ropePositionBuilderMatchesFlatTokenOrder() {
        #expect(LTXPositionBuilder.videoCoordinates(frameCount: 2, height: 2, width: 2) == [
            0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1,
            1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1,
        ])
        #expect(LTXPositionBuilder.audioCoordinates(tokenCount: 4) == [0, 1, 2, 3])
    }

    @Test func ropeFrequencyGridUsesReferenceLogSpacing() {
        let frequencies = LTXTransformerOps.generateFreqGridValues(
            theta: 100,
            numPositionDimensions: 1,
            innerDimension: 8
        )
        let expected = [
            Float.pi / 2,
            Float(Foundation.pow(100.0, 1.0 / 3.0)) * Float.pi / 2,
            Float(Foundation.pow(100.0, 2.0 / 3.0)) * Float.pi / 2,
            100 * Float.pi / 2,
        ]
        #expect(frequencies.count == expected.count)
        for (actual, reference) in zip(frequencies, expected) {
            #expect(abs(actual - reference) < 1e-5)
        }
    }

    @Test func encoderConfigurationKeepsLTXLatentContract() throws {
        let configuration = try LTXVideoVAEConfiguration()
        #expect(configuration.latentChannels == 128)
        #expect(configuration.patchSize == 4)
        #expect(configuration.causalDecoder == false)
        #expect(configuration.spatialPaddingMode == "zeros")
    }

    @Test func audioDecoderConfigurationKeepsLTXLatentContract() throws {
        let configuration = try LTXAudioVAEDecoderConfiguration()
        #expect(configuration.latentChannels == 8)
        #expect(configuration.latentFrequencyBins == 16)
        #expect(configuration.outputChannels == 2)
        #expect(configuration.outputFrequencyBins == 64)
        #expect(configuration.causal)
    }

    @Test func latentUpsamplerSpatial2xPreservesLatentContract() throws {
        let configuration = try LTXLatentUpsamplerConfiguration(
            inChannels: 8,
            midChannels: 32,
            numBlocksPerStage: 1
        )
        #expect(try configuration.outputShape(for: [1, 8, 2, 3, 4]) == [1, 8, 2, 6, 8])
    }

    @Test func latentUpsamplerRational15xUsesBlurDownsample() throws {
        let configuration = try LTXLatentUpsamplerConfiguration(
            inChannels: 8,
            midChannels: 32,
            numBlocksPerStage: 1,
            spatialScale: 1.5,
            rationalResampler: true
        )
        #expect(try configuration.outputShape(for: [1, 8, 2, 4, 6]) == [1, 8, 2, 6, 9])
    }

    @Test func latentUpsamplerTemporal2xDropsCausalFrame() throws {
        let configuration = try LTXLatentUpsamplerConfiguration(
            inChannels: 8,
            midChannels: 32,
            numBlocksPerStage: 1,
            spatialUpsample: false,
            temporalUpsample: true,
            spatialScale: 1.0,
            rationalResampler: true
        )
        #expect(try configuration.outputShape(for: [1, 8, 4, 3, 4]) == [1, 8, 7, 3, 4])
    }

    @Test func latentUpsamplerRejectsInvalidVariant() {
        var didThrow = false
        do {
            _ = try LTXLatentUpsamplerConfiguration(
                spatialUpsample: true,
                temporalUpsample: true
            )
        } catch {
            didThrow = true
        }
        #expect(didThrow)
    }

    @Test func temporalSplitMatchesPythonReference() {
        let intervals = LTXVideoTiling.splitTemporalLatents(
            dimension: 4,
            size: 2,
            overlap: 1
        )

        #expect(intervals.starts == [0, 0, 1])
        #expect(intervals.ends == [2, 3, 4])
        #expect(intervals.leftRamps == [0, 2, 2])
        #expect(intervals.rightRamps == [1, 1, 0])
    }

    @Test func trapezoidalMasksMatchPythonReference() throws {
        let ordinary = try LTXVideoTiling.trapezoidalMask(
            length: 5,
            leftRamp: 2,
            rightRamp: 2
        )
        let causal = try LTXVideoTiling.trapezoidalMask(
            length: 5,
            leftRamp: 2,
            rightRamp: 2,
            leftStartsFromZero: true
        )

        #expect(abs(ordinary[0] - 1.0 / 3.0) < 1e-6)
        #expect(abs(ordinary[1] - 2.0 / 3.0) < 1e-6)
        #expect(ordinary[2] == 1)
        #expect(abs(ordinary[3] - 2.0 / 3.0) < 1e-6)
        #expect(abs(ordinary[4] - 1.0 / 3.0) < 1e-6)
        #expect(causal[0] == 0)
        #expect(causal[1] == 0.5)
        #expect(causal[2] == 1)
    }

    @Test func singleTemporalTileCoordinatesMatchPythonReference() throws {
        let configuration = LTXTilingConfiguration(
            temporal: try LTXTemporalTilingConfiguration(
                tileSizeInFrames: 16,
                tileOverlapInFrames: 8
            )
        )
        let tiles = try LTXVideoTiling.prepareDecoderTiles(
            latentShape: [1, 128, 2, 1, 1],
            configuration: configuration
        )

        #expect(tiles.count == 1)
        #expect(tiles[0].input[2] == LTXVideoSlice(0, 2))
        #expect(tiles[0].output[2] == LTXVideoSlice(0, 9))
        #expect(tiles[0].output[3] == LTXVideoSlice(0, 32))
        #expect(tiles[0].output[4] == LTXVideoSlice(0, 32))
    }

    @Test func multiTemporalTileCoordinatesExerciseStitching() throws {
        let configuration = LTXTilingConfiguration(
            temporal: try LTXTemporalTilingConfiguration(
                tileSizeInFrames: 16,
                tileOverlapInFrames: 8
            )
        )
        let tiles = try LTXVideoTiling.prepareDecoderTiles(
            latentShape: [1, 128, 4, 1, 1],
            configuration: configuration
        )

        #expect(tiles.count == 3)
        #expect(tiles.map { $0.input[2] } == [
            LTXVideoSlice(0, 2), LTXVideoSlice(0, 3), LTXVideoSlice(1, 4)
        ])
        #expect(tiles.map { $0.output[2] } == [
            LTXVideoSlice(0, 9), LTXVideoSlice(0, 17), LTXVideoSlice(8, 25)
        ])
        #expect(tiles[1].masks[2]?.values.first == 0)
        #expect(tiles[2].masks[2]?.values.first == 0)
    }

    @Test func spatialCoordinatesCreateMultipleRealTiles() throws {
        let configuration = LTXTilingConfiguration(
            spatial: try LTXSpatialTilingConfiguration(
                tileSizeInPixels: 64,
                tileOverlapInPixels: 32
            )
        )
        let tiles = try LTXVideoTiling.prepareDecoderTiles(
            latentShape: [1, 128, 1, 3, 2],
            configuration: configuration
        )

        #expect(tiles.count == 2)
        #expect(tiles.map { $0.input[3] } == [LTXVideoSlice(0, 2), LTXVideoSlice(1, 3)])
        #expect(tiles.map { $0.output[3] } == [LTXVideoSlice(0, 64), LTXVideoSlice(32, 96)])
    }
}

import Testing

@testable import MiniMaxMusic3SwiftRuntime

struct MiniMaxMusic3SwiftRuntimeTests {
    @Test func ggufQuantizationStrategyMatchesMLXPolicy() {
        #expect(
            MiniMaxMusic3GGUFQuantizationStrategy.groupStrategy
                == MiniMaxMusic3GGUFGroupStrategy(groupSize: 64, mode: .affine)
        )
        #expect(MiniMaxMusic3GGUFQuantizationStrategy.groupSize == 64)
        #expect(MiniMaxMusic3GGUFQuantizationStrategy.mode == .affine)
        #expect(MiniMaxMusic3GGUFQuantizationStrategy.targetBits(for: "Q2_K") == 4)
        #expect(MiniMaxMusic3GGUFQuantizationStrategy.targetBits(for: "Q4_K") == 4)
        #expect(MiniMaxMusic3GGUFQuantizationStrategy.targetBits(for: "Q5_K") == 8)
        #expect(MiniMaxMusic3GGUFQuantizationStrategy.targetBits(for: "Q8_0") == 8)
        #expect(MiniMaxMusic3GGUFQuantizationStrategy.targetBits(for: "BF16") == nil)
        #expect(MiniMaxMusic3GGUFQuantizationStrategy.isSourcePrecision("F16"))
        #expect(MiniMaxMusic3GGUFQuantizationStrategy.isSourcePrecision("BF16"))
        #expect(MiniMaxMusic3GGUFQuantizationStrategy.isSourcePrecision("F32"))
    }

    @Test func conditionEncoderConfigurationMatchesCheckpoint() throws {
        let configuration = MiniMaxMusic3ConditionEncoderConfiguration.music3

        try configuration.validate()
        #expect(try configuration.outputLength(for: 1) == 3)
        #expect(try configuration.outputLength(for: 4) == 13)
        #expect(try configuration.outputLength(for: 200) == 689)
    }

    @Test func rvqResidualEmbeddingIDsUseCodebookOffsets() throws {
        #expect(
            try MiniMaxMusic3RVQCodebookLayout.residualEmbeddingIDs(
                for: [[10, 1, 2, 3], [20, 4, 5, 6]],
                audioVocabularySize: 8
            ) == [[1, 10, 19], [4, 13, 22]]
        )
    }

    @Test func music3VocoderConfigurationMatchesCheckpoint() throws {
        let configuration = MiniMaxMusic3VocoderConfiguration.music3

        try configuration.validate()
        #expect(configuration.latentChannels == 128)
        #expect(configuration.hopLength == 512)
        #expect(configuration.samplingRate == 44_100)
    }

    @Test func chunkStartsFollowMusic3Overlap() throws {
        #expect(try MiniMaxMusic3ChunkLayout.starts(for: 200) == [0])
        #expect(try MiniMaxMusic3ChunkLayout.starts(for: 201) == [0, 100])
        #expect(try MiniMaxMusic3ChunkLayout.starts(for: 401) == [0, 100, 200, 300])
    }

    @Test func waveformCropMatchesReference() throws {
        #expect(
            try MiniMaxMusic3ChunkLayout.waveformCrop(chunkIndex: 0, chunkCount: 3)
                == (left: 0, right: 258 * 512)
        )
        #expect(
            try MiniMaxMusic3ChunkLayout.waveformCrop(chunkIndex: 1, chunkCount: 3)
                == (left: 86 * 512, right: 258 * 512)
        )
        #expect(
            try MiniMaxMusic3ChunkLayout.waveformCrop(chunkIndex: 2, chunkCount: 3)
                == (left: 86 * 512, right: 0)
        )
    }
    @Test func flowTimestepsMatchPythonReference() throws {
        #expect(try MiniMaxMusic3FlowScheduler.flowTimestepValues(numInferenceSteps: 5) == [
            0.0,
            0.2,
            0.4,
            0.6,
            0.8
        ])
    }

    @Test func eulerStepMatchesPythonReference() throws {
        #expect(try MiniMaxMusic3FlowScheduler.eulerStep(
            sample: [1, 2, 3, 4],
            velocity: [2, 4, 6, 8],
            numInferenceSteps: 4
        ) == [1.5, 3.0, 4.5, 6.0])
    }

    @Test func blendOverlapMatchesPythonReference() throws {
        let blended = try MiniMaxMusic3FlowScheduler.blendOverlap(
            latents: [[10, 11], [12, 13], [14, 15], [16, 17]],
            noisePrompt: [[1, 2], [3, 4]],
            previousLatent: [[20, 21], [22, 23]],
            overlap: 2,
            timestep: 0.25
        )
        let expected: [[Float]] = [
            [5.75, 6.750000476837158],
            [7.750000953674316, 8.750000953674316],
            [14.0, 15.0],
            [16.0, 17.0]
        ]
        #expect(blended.count == expected.count)
        #expect(blended.enumerated().allSatisfy { row, values in
            values.enumerated().allSatisfy { column, value in
                abs(value - expected[row][column]) < 1e-5
            }
        })
    }

    @Test func restoreOverlapMatchesPythonReference() throws {
        #expect(try MiniMaxMusic3FlowScheduler.restoreOverlap(
            latents: [[10, 11], [12, 13], [14, 15], [16, 17]],
            previousLatent: [[20, 21], [22, 23]],
            overlap: 2
        ) == [[20, 21], [22, 23], [14, 15], [16, 17]])
    }

    @Test func carryWindowMatchesPythonReference() throws {
        let latents = (0..<400).map { [Float($0 * 2), Float($0 * 2 + 1)] }
        let carried = try MiniMaxMusic3FlowScheduler.carryWindow(latents)

        #expect(carried.count == 172)
        #expect(carried.first == [112, 113])
        #expect(carried.last == [454, 455])
    }

    @Test func captionCleaningMatchesPythonReference() throws {
        let caption = "<|foo bar|>\n### **hello**\n- *world*\n\n---\n• item    "
        #expect(try MiniMaxMusic3Prompt.cleanCaption(caption) == "foo is bar\nhello\nworld\nitem")
    }

    @Test func lyricsNormalizationMatchesPythonReference() throws {
        #expect(
            try MiniMaxMusic3Prompt.normalizeLyrics("[Verse] hello [Chorus] world ^ next")
                == "[start]\n[verse]"
        )
        #expect(
            try MiniMaxMusic3Prompt.normalizeLyrics("[Verse][Chorus]line\nplain")
                == "[start]\n[verse][chorus]\nplain"
        )
    }

    @Test func promptAssemblyMatchesPythonReference() throws {
        #expect(
            try MiniMaxMusic3Prompt.buildPromptText(
                prompt: "sunset [tag] <|tempo fast|>",
                lyrics: "[Verse]hello"
            )
                == "<|im_start|><|caption_start|>sunset [tag] tempo is fast<|caption_end|><|lyrics_start|>[start]\n[verse]<|lyrics_end|><|im_end|><|audio_start|>"
        )
    }

    @Test func cfgTokenIDsMaskOnlyTheConditionalBody() throws {
        let result = try MiniMaxMusic3Prompt.buildCFGTokenIDs(
            prompt: "prompt",
            lyrics: "lyrics",
            encode: { _ in [10, 11, 12, 13, 14] }
        )
        #expect(result == [
            [10, 11, 12, 13, 14],
            [10, 151_654, 151_654, 13, 14]
        ])
    }

    @Test func semanticGuidedLogitsMatchesPythonReference() throws {
        let logits: [[Float]] = [
            [0.0, 4.0, 2.0, .nan, -3.0, 1.0],
            [0.0, 1.0, 1.0, 2.0, 3.0, 4.0]
        ]
        let allowed = [true, false, true, true, false, false]
        let guided = try MiniMaxMusic3Sampling.semanticGuidedLogits(
            logits: logits,
            allowedVocabulary: allowed,
            cfgScale: 1.5,
            conditionalTopK: 2
        )
        let values = guided[0]
        #expect(values.count == 6)
        #expect(values[0] == 0)
        #expect(values[2] == 2.5)
        #expect(values[1].isInfinite && values[1] < 0)
        #expect(values[3].isInfinite && values[3] < 0)
        #expect(values[4].isInfinite && values[4] < 0)
        #expect(values[5].isInfinite && values[5] < 0)
    }

    @Test func autoregressiveFrameLimitUsesMusicFrameRate() throws {
        let generation = MiniMaxMusic3GenerationConfiguration(audioDuration: 0.2)

        #expect(try generation.maxFrames(using: .music3) == 5)
        #expect(
            try MiniMaxMusic3GenerationConfiguration(audioDuration: 1_000)
                .maxFrames(using: .music3) == 9_000
        )
    }

    @Test func autoregressiveConfigurationRejectsSubFrameDuration() {
        let generation = MiniMaxMusic3GenerationConfiguration(audioDuration: 0.01)

        #expect(throws: MiniMaxMusic3AutoregressiveError.self) {
            _ = try generation.maxFrames(using: .music3)
        }
    }

    @Test func autoregressiveConfigurationHandlesDurationBoundaries() throws {
        #expect(
            try MiniMaxMusic3GenerationConfiguration(audioDuration: 0.04)
                .maxFrames(using: .music3) == 1
        )
        #expect(throws: MiniMaxMusic3AutoregressiveError.self) {
            _ = try MiniMaxMusic3GenerationConfiguration(audioDuration: 0.039)
                .maxFrames(using: .music3)
        }
    }

    @Test func autoregressiveConfigurationRejectsNonFiniteDuration() {
        for duration in [Float.nan, Float.infinity, -Float.infinity] {
            let generation = MiniMaxMusic3GenerationConfiguration(audioDuration: duration)
            #expect(throws: MiniMaxMusic3AutoregressiveError.self) {
                _ = try generation.maxFrames(using: .music3)
            }
        }
    }

    @Test func generationConfigurationValidatesDenoiseSettings() {
        #expect(throws: MiniMaxMusic3AutoregressiveError.self) {
            try MiniMaxMusic3GenerationConfiguration(numInferenceSteps: 0).validate()
        }
        #expect(throws: MiniMaxMusic3AutoregressiveError.self) {
            try MiniMaxMusic3GenerationConfiguration(flowCFGScale: .nan).validate()
        }
        #expect(throws: MiniMaxMusic3AutoregressiveError.self) {
            try MiniMaxMusic3GenerationConfiguration(topK: 0).validate()
        }
    }
}

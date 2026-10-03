import Foundation
import MLX
import MLXNN
import Testing
@testable import LTXVideoSwiftRuntime

@Suite(.serialized)
struct LTX25Tests {
    @Test func gemma4StatesMatchIndependentScalarOracle() throws {
        struct Tensor: Decodable { let shape: [Int]; let values: [Float] }
        struct Fixture: Decodable {
            let configuration: LTXGemma4TextEncoder.Configuration
            let tokens: [Int32]
            let weights: [String: Tensor]
            let states: [[Float]]
        }
        let url = try #require(Bundle.module.url(forResource: "gemma4-tiny", withExtension: "json", subdirectory: "Fixtures"))
        let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
        let encoder = try LTXGemma4TextEncoder(configuration: fixture.configuration,
            tensors: fixture.weights.mapValues { MLXArray($0.values).reshaped($0.shape) })
        let actual = try encoder.allHiddenStates(tokenIDs: MLXArray(fixture.tokens).reshaped([1, 3]), attentionMask: MLXArray.ones([1, 3]))
        #expect(actual.count == 3)
        for (value, expected) in zip(actual, fixture.states) {
            #expect(max(abs(value.flattened() - MLXArray(expected))).item(Float.self) < 3e-5)
        }
        // Padding must not alter any valid-token hidden state (RoPE's relative positions).
        let padded = try encoder.allHiddenStates(tokenIDs: MLXArray([Int32(0), 2, 5, 7]).reshaped([1, 4]),
            attentionMask: MLXArray([Int32(0), 1, 1, 1]).reshaped([1, 4]))
        for (value, expected) in zip(padded, actual) {
            #expect(max(abs(value[0..., 1..., 0...] - expected)).item(Float.self) < 3e-5)
        }
    }

    @Test func gemma4AddsBOSAndPreservesPromptHeadWhenTruncating() throws {
        #expect(try LTXGemma4TextEncoder.promptLayout(tokenIDs: [10, 11, 12, 13], maxLength: 4).tokenIDs == [2, 10, 11, 12])
        let layout = try LTXGemma4TextEncoder.promptLayout(tokenIDs: [2, 10], maxLength: 4)
        #expect(layout.tokenIDs == [0, 0, 2, 10])
        #expect(layout.attentionMask == [0, 0, 1, 1])
    }

    @Test func ltx25KeepsAudioFFBiasAndAddsOnlyTheKeyframeParameter() throws {
        let config = try LTXTransformerConfiguration(numLayers: 1, videoDim: 8, audioDim: 4,
            videoNumHeads: 2, audioNumHeads: 2, videoHeadDim: 4, audioHeadDim: 2,
            avCrossNumHeads: 2, avCrossHeadDim: 2, videoFFBias: false, useKeyframesEmbedding: true)
        let model = LTXTransformer(configuration: config)
        let params = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        #expect(params["transformer_blocks.0.ff.proj_in.bias"] == nil)
        #expect(params["transformer_blocks.0.audio_ff.proj_in.bias"] != nil)
        #expect(params["keyframes_abs_pos_embedding"]?.shape == [1, 8])
        #expect(throws: LTXVideoRuntimeError.self) {
            try LTXDiffusionScheduler.denoise(model: LTXX0Model(transformer: model),
                videoLatent: MLXArray.zeros([1, 2, 128]), audioLatent: MLXArray.zeros([1, 2, 128]))
        }
        let old = LTXTransformer(configuration: try LTXTransformerConfiguration(numLayers: 1, videoDim: 8, audioDim: 4,
            videoNumHeads: 2, audioNumHeads: 2, videoHeadDim: 4, audioHeadDim: 2, avCrossNumHeads: 2, avCrossHeadDim: 2))
        #expect(old.keyframesAbsPosEmbedding == nil)
    }

    @Test func ancestralStepMatchesRectifiedFlowOracleAndEndsCleanly() {
        let result = LTXDiffusionScheduler.ancestralStep(sample: MLXArray([Float(2), -1]), denoised: MLXArray([Float(0.5), 1]),
            sigma: 1, sigmaNext: 0.75, noise: MLXArray([Float(0.25), -0.5]))
        // Scalar FP64 reference with sigmaDown=0.5625, alphaRatio=4/7.
        let noiseScale = Float(Foundation.sqrt(0.5625 - 0.31640625 * 16 / 49))
        #expect(abs(result[0].item(Float.self) - (Float(43.0/56.0) + noiseScale * 0.25)) < 1e-6)
        #expect(abs(result[1].item(Float.self) - (-Float(1.0/14.0) - noiseScale * 0.5)) < 1e-6)
        let terminal = LTXDiffusionScheduler.ancestralStep(sample: result, denoised: MLXArray([Float(3), 4]),
            sigma: 0.1, sigmaNext: 0, noise: MLXArray([Float.nan, .nan]))
        #expect(terminal.asArray(Float.self) == [3, 4])
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["GENIMAGE_LTX25_MODEL"] != nil))
    func installedDecodersProduceFiniteStereoAndNineVideoFrames() throws {
        let root = URL(fileURLWithPath: try #require(ProcessInfo.processInfo.environment["GENIMAGE_LTX25_MODEL"]))
        func decodeAudio() throws -> MLXArray {
            let decoder = LTXAudioVAEDecoder(configuration: try LTXAudioVAEDecoderConfiguration())
            _ = try LTXAudioVAEWeightLoader.loadDecoder(model: decoder, from: root)
            let mel = try decoder.decode(.zeros([1, 8, 9, 16], dtype: .bfloat16))
            eval(mel)
            let vocoder = LTXVocoderWithBWE()
            _ = try LTXVocoderWeightLoader.load(model: vocoder, from: root, computeDType: .float32)
            let audio = try vocoder.decode(mel)
            eval(audio)
            return audio
        }
        let audio = try decodeAudio()
        #expect(audio.ndim == 3 && audio.shape[0] == 1 && audio.shape[1] == 2 && audio.shape[2] > 0)
        #expect(all(isFinite(audio)).item(Bool.self))
        Memory.clearCache()
        let decoder = LTXVideoVAEDecoder(configuration: try LTXVideoVAEConfiguration())
        _ = try LTXVideoVAEWeightLoader.loadDecoder(model: decoder, from: root)
        let video = try decoder.decode(.zeros([1, 128, 2, 8, 8], dtype: .bfloat16), materializeStages: true)
        #expect(video.shape == [1, 3, 9, 256, 256])
        #expect(all(isFinite(video)).item(Bool.self))
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["GENIMAGE_LTX25_WORKER"] != nil), .timeLimit(.minutes(1)))
    func workerKeepsNineFramesWhenDecodedAudioIsShorter() async throws {
        let environment = ProcessInfo.processInfo.environment
        let model = try #require(environment["GENIMAGE_LTX25_MODEL"])
        let worker = try #require(environment["GENIMAGE_LTX25_WORKER"])
        let ffmpeg = try #require(environment["GENIMAGE_LTX25_FFMPEG"])
        let ffprobe = try #require(environment["GENIMAGE_LTX25_FFPROBE"])
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let latentURL = directory.appendingPathComponent("latents.safetensors")
        try MLX.save(arrays: ["video": .zeros([1, 128, 2, 8, 8], dtype: .bfloat16),
                             "audio": .zeros([1, 8, 9, 16], dtype: .bfloat16)], url: latentURL)
        let output = directory.appendingPathComponent("video.mp4")
        let request = directory.appendingPathComponent("request.json")
        try JSONSerialization.data(withJSONObject: ["modelDirectory": model, "outputPath": output.path,
            "prompt": "Decoder regression", "width": 256, "height": 256, "frames": 9,
            "frameRate": 24, "seed": 42, "stage1Steps": 8, "stage2Steps": 3])
            .write(to: request)
        func run(_ executable: String, _ arguments: [String], _ environment: [String: String]) async throws -> Data {
            let process = Process(), pipe = Pipe()
            process.executableURL = URL(fileURLWithPath: executable)
            process.arguments = arguments; process.environment = environment
            process.standardOutput = pipe; process.standardError = pipe
            defer { if process.isRunning { process.terminate() } }
            try process.run()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            while process.isRunning { try await Task.sleep(for: .milliseconds(10)) }
            try #require(process.terminationStatus == 0, "\(String(decoding: data, as: UTF8.self))")
            return data
        }
        var workerEnvironment = environment
        workerEnvironment["GENMEDIA_FFMPEG"] = ffmpeg
        workerEnvironment["GENIMAGE_LTX_DEBUG_LATENTS"] = nil
        workerEnvironment["GENIMAGE_LTX_DEBUG_LATENTS_INPUT"] = latentURL.path
        _ = try await run(worker, ["--request", request.path, "--format", "mlx", "--variant", "ltx-2.5"], workerEnvironment)
        let metadata = try await run(ffprobe, ["-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", output.path], environment)
        let object = try #require(JSONSerialization.jsonObject(with: metadata) as? [String: Any])
        let streams = try #require(object["streams"] as? [[String: Any]])
        let video = try #require(streams.first { $0["codec_type"] as? String == "video" })
        let audio = try #require(streams.first { $0["codec_type"] as? String == "audio" })
        #expect(video["nb_read_frames"] as? String == "9")
        #expect(video["width"] as? Int == 256 && video["height"] as? Int == 256)
        #expect(audio["channels"] as? Int == 2 && audio["sample_rate"] as? String == "48000")
        let format = try #require(object["format"] as? [String: Any])
        let duration = try #require((format["duration"] as? String).flatMap(Double.init))
        #expect(abs(duration - 0.375) < 0.001)
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["GENIMAGE_LTX25_MODEL"] != nil))
    func installedGemma4LoadsEveryLayerAndEncodesTokens() throws {
        let root = URL(fileURLWithPath: try #require(ProcessInfo.processInfo.environment["GENIMAGE_LTX25_MODEL"]))
        let encoder = try LTXGemma4TextEncoder(directory: root.appendingPathComponent("gemma4-12b-ltx-v1"))
        let states = try encoder.allHiddenStates(tokenIDs: MLXArray([Int32(2), 42, 71]).reshaped([1, 3]), attentionMask: MLXArray.ones([1, 3]))
        #expect(states.count == 49)
        #expect(states.allSatisfy { $0.shape == [1, 3, 3840] && all(isFinite($0)).item(Bool.self) })
        #expect(max(abs(states[48])).item(Float.self) > 0)
    }
}

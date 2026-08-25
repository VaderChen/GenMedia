import MLX
import Foundation

public struct VAEDecodePoCReport {
    public let outputURL: URL
    public let latentShape: [Int]
    public let audioShape: [Int]
    public let wave: PCM16WaveReport
}

enum VAEDecodePoCError: LocalizedError {
    case invalidOutputShape(expectedSamples: Int, actualShape: [Int])
    case invalidTiledSampleCount(expected: Int, actual: Int)

    var errorDescription: String? {
        switch self {
        case let .invalidOutputShape(expectedSamples, actualShape):
            "VAE 輸出形狀錯誤，預期 samples=\(expectedSamples)，實際為 \(actualShape)。"
        case let .invalidTiledSampleCount(expected, actual):
            "VAE 分塊輸出長度錯誤，預期 samples=\(expected)，實際為 \(actual)。"
        }
    }
}

public enum VAEDecodePoC {
    public static func run(
        modelRoot: URL,
        outputURL: URL,
        latentFrames: Int
    ) throws -> VAEDecodePoCReport {
        let configuration = try loadConfiguration(modelRoot: modelRoot)
        let latentValueCount = latentFrames * configuration.decoderInputChannels
        let latentValues = (0..<latentValueCount).map { index in
            sin(Float(index) * 0.173) * 0.5
        }
        let latentShape = [1, latentFrames, configuration.decoderInputChannels]
        let latents = MLXArray(latentValues, latentShape)
        return try run(modelRoot: modelRoot, outputURL: outputURL, latents: latents)
    }

    static func run(
        modelRoot: URL,
        outputURL: URL,
        latents: MLXArray,
        progress: ((Double) -> Void)? = nil
    ) throws -> VAEDecodePoCReport {
        let vaeRoot = modelRoot.appendingPathComponent("vae", isDirectory: true)
        let configuration = try loadConfiguration(modelRoot: modelRoot)
        guard latents.ndim == 3,
              latents.dim(0) == 1,
              latents.dim(1) > 0,
              latents.dim(2) == configuration.decoderInputChannels else {
            throw VAEDecodePoCError.invalidOutputShape(
                expectedSamples: latents.ndim > 1 ? latents.dim(1) * configuration.hopLength : 0,
                actualShape: latents.shape
            )
        }
        let model = OobleckVAEDecoder(configuration: configuration)
        try OobleckVAEWeightLoader.load(
            model: model,
            from: vaeRoot.appendingPathComponent("diffusion_pytorch_model.safetensors")
        )
        if latents.dim(1) > 512 {
            return try decodeTiled(
                model: model,
                configuration: configuration,
                latents: latents,
                outputURL: outputURL,
                progress: progress
            )
        }
        let modelLatents = latents.asType(model.dtype)
        let audio = model.decode(modelLatents)
        MLX.eval(audio)

        let expectedSamples = latents.dim(1) * configuration.hopLength
        guard audio.shape == [1, expectedSamples, configuration.audioChannels] else {
            throw VAEDecodePoCError.invalidOutputShape(
                expectedSamples: expectedSamples,
                actualShape: audio.shape
            )
        }
        progress?(1)
        let wave = try PCM16WaveWriter.write(
            audio: audio,
            sampleRate: configuration.samplingRate,
            to: outputURL
        )
        return VAEDecodePoCReport(
            outputURL: outputURL,
            latentShape: latents.shape,
            audioShape: audio.shape,
            wave: wave
        )
    }

    private static func decodeTiled(
        model: OobleckVAEDecoder,
        configuration: OobleckVAEConfiguration,
        latents: MLXArray,
        outputURL: URL,
        progress: ((Double) -> Void)?
    ) throws -> VAEDecodePoCReport {
        let chunkSize = 512
        let overlap = 64
        let strideLength = chunkSize - 2 * overlap
        let latentFrames = latents.dim(1)
        let stepCount = Int(ceil(Double(latentFrames) / Double(strideLength)))
        let writer = try PCM16WaveStreamWriter(
            outputURL: outputURL,
            sampleRate: configuration.samplingRate,
            channelCount: configuration.audioChannels
        )

        for index in 0..<stepCount {
            try Task.checkCancellation()
            let coreStart = index * strideLength
            let coreEnd = min(coreStart + strideLength, latentFrames)
            let windowStart = max(0, coreStart - overlap)
            let windowEnd = min(latentFrames, coreEnd + overlap)
            let chunk = latents[0..., windowStart..<windowEnd, 0...].asType(model.dtype)
            let decoded = model.decode(chunk)
            MLX.eval(decoded)

            let trimStart = (coreStart - windowStart) * configuration.hopLength
            let trimEnd = (windowEnd - coreEnd) * configuration.hopLength
            let endIndex = decoded.dim(1) - trimEnd
            let trimmed = decoded[0..., trimStart..<endIndex, 0...]
            MLX.eval(trimmed)
            try writer.append(audio: trimmed)
            progress?(Double(index + 1) / Double(stepCount))
            Memory.clearCache()
        }

        let wave = try writer.finish()
        let expectedSamples = latentFrames * configuration.hopLength
        guard wave.sampleCount == expectedSamples else {
            throw VAEDecodePoCError.invalidTiledSampleCount(
                expected: expectedSamples,
                actual: wave.sampleCount
            )
        }
        return VAEDecodePoCReport(
            outputURL: outputURL,
            latentShape: latents.shape,
            audioShape: [1, expectedSamples, configuration.audioChannels],
            wave: wave
        )
    }

    private static func loadConfiguration(modelRoot: URL) throws -> OobleckVAEConfiguration {
        try OobleckVAEConfiguration.load(
            from: modelRoot.appendingPathComponent("vae/config.json")
        )
    }
}

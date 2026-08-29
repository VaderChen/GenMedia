import Foundation
import MLX

public enum MiniMaxMusic3ChunkLayout {
    // CHUNK_FRAMES = 200 is the window size in autoregressive frame_hiddens,
    // before the condition encoder. It is not a latent-frame limit: the
    // condition encoder expands a full 200-frame window to 689 latent frames,
    // while the final partial window in the fixtures expands to 378 latent
    // frames. Therefore latent chunk lengths are intentionally much larger
    // than CHUNK_FRAMES.
    public static let chunkFrames = 200
    public static let chunkHop = 100
    public static let latentHopLength = 512
    public static let cropLeftLatent = 86
    public static let cropRightLatent = 258

    public static func starts(for frameCount: Int) throws -> [Int] {
        guard frameCount > 0 else {
            throw MiniMaxMusic3VocoderError.invalidChunk(frameCount)
        }
        if frameCount <= chunkFrames {
            return [0]
        }
        return Array(stride(from: 0, to: frameCount - chunkHop, by: chunkHop))
    }

    public static func waveformCrop(
        chunkIndex: Int,
        chunkCount: Int,
        hopLength: Int = latentHopLength
    ) throws -> (left: Int, right: Int) {
        guard chunkCount > 0, chunkIndex >= 0, chunkIndex < chunkCount else {
            throw MiniMaxMusic3VocoderError.invalidChunk(chunkIndex)
        }
        guard hopLength > 0 else {
            throw MiniMaxMusic3VocoderError.invalidConfiguration(
                "waveform hop length 必須是正整數。"
            )
        }
        return (
            chunkIndex == 0 ? 0 : cropLeftLatent * hopLength,
            chunkIndex == chunkCount - 1 ? 0 : cropRightLatent * hopLength
        )
    }
}

public final class MiniMaxMusic3Decoder {
    public let vocoder: MiniMaxMusic3Vocoder
    public let weightReport: MiniMaxMusic3VocoderWeightLoadReport?

    public init(vocoder: MiniMaxMusic3Vocoder) {
        self.vocoder = vocoder
        self.weightReport = nil
    }

    public init(modelDirectory: URL) throws {
        let configuration = try MiniMaxMusic3VocoderConfiguration.load(from: modelDirectory)
        let vocoder = MiniMaxMusic3Vocoder(configuration: configuration)
        let report = try MiniMaxMusic3VocoderWeightLoader.load(
            model: vocoder,
            from: modelDirectory
        )
        self.vocoder = vocoder
        self.weightReport = report
    }

    public func decodeChunk(_ latents: MLXArray) throws -> MLXArray {
        guard latents.ndim == 3,
              latents.shape[0] > 0,
              latents.shape[1] == vocoder.configuration.latentChannels,
              latents.shape[2] > 0 else {
            throw MiniMaxMusic3VocoderError.invalidLatentShape(latents.shape)
        }
        let channelLastLatents = latents.transposed(0, 2, 1)
        let waveform = vocoder(channelLastLatents).asType(.float32)
        let clipped = MLX.clip(waveform, min: Float(-1), max: Float(1))
        MLX.eval(clipped)
        return clipped
    }

    public func decodeChunks(_ latentChunks: [MLXArray]) throws -> MLXArray {
        guard !latentChunks.isEmpty else {
            throw MiniMaxMusic3VocoderError.emptyChunks
        }
        var waveforms: [MLXArray] = []
        for (index, latentChunk) in latentChunks.enumerated() {
            let waveform = try decodeChunk(latentChunk)
            let crop = try MiniMaxMusic3ChunkLayout.waveformCrop(
                chunkIndex: index,
                chunkCount: latentChunks.count,
                hopLength: vocoder.configuration.hopLength
            )
            let end = waveform.shape[2] - crop.right
            if end <= crop.left {
                waveforms.append(waveform)
            } else {
                waveforms.append(waveform[0..., 0..., crop.left..<end])
            }
        }
        let audio = MLX.concatenated(waveforms, axis: 2)
        let clipped = MLX.clip(audio.asType(.float32), min: Float(-1), max: Float(1))
        MLX.eval(clipped)
        return clipped
    }
}

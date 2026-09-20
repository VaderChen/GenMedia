import AVFoundation
import CoreGraphics
import Foundation
import MLX
import Testing

@testable import MiniMaxH3SwiftRuntime

@Suite("MiniMax H3 video writer")
struct MiniMaxH3VideoWriterTests {
    @Test("Audio length targets the video duration exactly")
    func sampleCountMatchesVideoDuration() {
        // 8 frames at 24 fps is 1/3 s; at 32 kHz that is 10666.67 -> 10667.
        #expect(MiniMaxH3VideoWriter.sampleCount(
            frameCount: 8, frameRate: 24, sampleRate: 32000
        ) == 10667)
        // A whole second lands exactly on the sample rate.
        #expect(MiniMaxH3VideoWriter.sampleCount(
            frameCount: 24, frameRate: 24, sampleRate: 32000
        ) == 32000)
        #expect(MiniMaxH3VideoWriter.sampleCount(
            frameCount: 48, frameRate: 24, sampleRate: 32000
        ) == 64000)
    }

    @Test("Audio length differs from the VAE's native output")
    func vaeOutputNeedsResizing() {
        // The audio VAE emits whole latent frames of 800 samples, so its output
        // rarely equals the video duration — hence the trim/pad.
        let audio = MiniMaxH3AudioVAEConfiguration.default
        let wanted = MiniMaxH3VideoWriter.sampleCount(
            frameCount: 8, frameRate: 24, sampleRate: audio.sampleRate
        )
        let latentFrames = 13
        #expect(latentFrames * audio.hopLength != wanted)
        #expect(abs(latentFrames * audio.hopLength - wanted) < audio.hopLength)
    }

    @Test("Degenerate inputs do not produce negative lengths")
    func degenerateInputs() {
        #expect(MiniMaxH3VideoWriter.sampleCount(
            frameCount: 0, frameRate: 24, sampleRate: 32000
        ) == 0)
        // A zero frame rate is clamped rather than dividing by zero.
        #expect(MiniMaxH3VideoWriter.sampleCount(
            frameCount: 8, frameRate: 0, sampleRate: 32000
        ) > 0)
    }

    @Test("Long video audio is submitted before video backpressure")
    func longVideoWithAudioCompletes() throws {
        let provider = try #require(CGDataProvider(
            data: Data(repeating: 0, count: 16) as CFData
        ))
        let image = try #require(CGImage(
            width: 2,
            height: 2,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: 8,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        ))
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("h3-long-video-\(UUID().uuidString).mp4")
        defer { try? FileManager.default.removeItem(at: outputURL) }

        let frameCount = 152
        let sampleCount = MiniMaxH3VideoWriter.sampleCount(
            frameCount: frameCount, frameRate: 15, sampleRate: 32_000
        )
        let waveform = MLXArray(
            [Float](repeating: 0, count: sampleCount * 2),
            [1, 2, sampleCount]
        )
        var progressValues: [Double] = []
        try MiniMaxH3VideoWriter.writeMP4(
            Array(repeating: image, count: frameCount),
            to: outputURL,
            frameRate: 15,
            audio: .init(waveform: waveform, sampleRate: 32_000),
            progress: { progressValues.append($0) }
        )

        #expect(FileManager.default.fileExists(atPath: outputURL.path))
        #expect(progressValues.last == 1)
    }
    @Test("Streaming tensor output retains every frame and its dimensions")
    func streamingPixelsProducesCompleteVideo() async throws {
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("h3-stream-\(UUID()).mp4")
        defer { try? FileManager.default.removeItem(at: outputURL) }
        let frameCount = 24
        // Distinct grayscale frames also detect accidental reuse of one slice.
        let channel = (0..<frameCount).flatMap { frame in
            [Float](repeating: Float(frame + 1) / Float(frameCount + 1), count: 16 * 32)
        }
        let pixels = MLXArray(channel + channel + channel, [1, 3, frameCount, 16, 32])
        try MiniMaxH3VideoWriter.writeMP4(pixels: pixels, to: outputURL, frameRate: 12)
        let asset = AVURLAsset(url: outputURL)
        let tracks = try await asset.loadTracks(withMediaType: .video)
        let track = try #require(tracks.first)
        let size = try await track.load(.naturalSize)
        #expect(size == CGSize(width: 32, height: 16))
        let duration = try await asset.load(.duration)
        #expect(abs(duration.seconds - 2) < 0.01)
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ])
        reader.add(output)
        #expect(reader.startReading())
        var grayValues: [Int] = []
        while let sample = output.copyNextSampleBuffer() {
            let buffer = try #require(CMSampleBufferGetImageBuffer(sample))
            CVPixelBufferLockBaseAddress(buffer, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
            let bytes = try #require(CVPixelBufferGetBaseAddress(buffer))
                .assumingMemoryBound(to: UInt8.self)
            grayValues.append(Int(bytes[0]))
        }
        #expect(reader.status == .completed)
        #expect(grayValues.count == frameCount)
        for (index, value) in grayValues.enumerated() {
            let expected = Int(Float(index + 1) / Float(frameCount + 1) * 255)
            #expect(abs(value - expected) <= 8)
        }
    }

}

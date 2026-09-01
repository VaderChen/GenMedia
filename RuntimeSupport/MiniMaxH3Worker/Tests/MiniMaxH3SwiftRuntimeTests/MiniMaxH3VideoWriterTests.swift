import Foundation
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
}

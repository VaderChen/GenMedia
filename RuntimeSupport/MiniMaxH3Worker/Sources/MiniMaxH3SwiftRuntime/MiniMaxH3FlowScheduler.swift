import Foundation
import MLX

/// Flow-matching schedule for MiniMax H3.
///
/// Ported from `ModelSamplingDiscreteFlow` / `ModelSamplingAV` in the ComfyUI
/// reference (`model_sampling.py`).
///
/// Video and audio denoise on different shifted schedules. Rather than running
/// two integrations, the reference *carries* the audio latent scaled onto the
/// video schedule, so the pack behaves as one ordinary flow latent whose audio
/// target is scaled by ``audioScale``.
public struct MiniMaxH3FlowScheduler: Sendable, Equatable {
    public var videoShift: Float
    public var audioShift: Float

    public init(videoShift: Float = 12.0, audioShift: Float = 3.0) {
        self.videoShift = videoShift
        self.audioShift = audioShift
    }

    /// `alpha * t / (1 + (alpha - 1) * t)`
    public static func timeSNRShift(alpha: Double, _ t: Double) -> Double {
        alpha == 1.0 ? t : alpha * t / (1.0 + (alpha - 1.0) * t)
    }

    /// Re-express a sigma from one shift schedule on another.
    ///
    /// Inverts the shift back to the base grid, then re-applies the other one.
    public static func timeShiftSigma(_ sigma: Double, from: Double, to: Double) -> Double {
        let base = sigma / (from + sigma * (1.0 - from))
        return to * base / (1.0 + (to - 1.0) * base)
    }

    /// How much the audio target is scaled when carried on the video schedule.
    public var audioScale: Double { Double(videoShift) / Double(audioShift) }

    /// Descending sigma schedule with a trailing zero, length `steps + 1`.
    ///
    /// Uniform in the base time grid, then shifted by ``videoShift`` — the flow
    /// equivalent of a "simple" scheduler.
    public func sigmas(steps: Int) -> [Double] {
        precondition(steps > 0, "steps must be positive")
        var result = (0 ..< steps).map { index -> Double in
            let t = 1.0 - Double(index) / Double(steps)
            return Self.timeSNRShift(alpha: Double(videoShift), t)
        }
        result.append(0.0)
        return result
    }

    /// The audio stream's sigma for a given video sigma.
    public func audioSigma(videoSigma: Double) -> Double {
        Self.timeShiftSigma(
            videoSigma, from: Double(videoShift), to: Double(audioShift)
        )
    }

    /// Factor applied to the audio latent to carry it onto the video schedule.
    public func carry(videoSigma: Double) -> Double {
        let clamped = max(videoSigma, 1e-6)
        return audioSigma(videoSigma: clamped) / clamped
    }

    /// One Euler step: `x + (next - current) * velocity`.
    public static func eulerStep(
        _ latent: MLXArray,
        velocity: MLXArray,
        sigma: Double,
        nextSigma: Double
    ) -> MLXArray {
        latent + velocity * Float(nextSigma - sigma)
    }

    /// Undo the audio carry on the model's velocity output.
    ///
    /// Mirrors the reference's
    /// `out[1] = (1 - scale) * carried + (1 + (scale - 1) * sigma_a) * out[1]`.
    public func uncarryAudioVelocity(
        _ velocity: MLXArray,
        carriedAudio: MLXArray,
        videoSigma: Double
    ) -> MLXArray {
        let scale = audioScale
        guard scale != 1.0 else { return velocity }
        let sigmaAudio = audioSigma(videoSigma: max(videoSigma, 1e-6))
        return carriedAudio * Float(1.0 - scale)
            + velocity * Float(1.0 + (scale - 1.0) * sigmaAudio)
    }
}

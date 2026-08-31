import Foundation
import Testing

@testable import MiniMaxH3SwiftRuntime

@Suite("MiniMax H3 flow scheduler")
struct MiniMaxH3FlowSchedulerTests {
    private let scheduler = MiniMaxH3FlowScheduler()

    @Test("Audio scale is the ratio of the two shifts")
    func audioScale() {
        // ModelSamplingAV.audio_scale = shift / audio_shift = 12 / 3.
        #expect(scheduler.audioScale == 4.0)
    }

    @Test("Time SNR shift is identity at alpha 1 and fixes the endpoints")
    func timeSNRShift() {
        #expect(MiniMaxH3FlowScheduler.timeSNRShift(alpha: 1.0, 0.37) == 0.37)
        #expect(MiniMaxH3FlowScheduler.timeSNRShift(alpha: 12.0, 0.0) == 0.0)
        #expect(MiniMaxH3FlowScheduler.timeSNRShift(alpha: 12.0, 1.0) == 1.0)
        // A shift above 1 pushes mass toward the noisy end.
        #expect(MiniMaxH3FlowScheduler.timeSNRShift(alpha: 12.0, 0.5) > 0.5)
    }

    @Test("Sigma schedule descends from 1 to 0 with the right length")
    func sigmaSchedule() {
        let sigmas = scheduler.sigmas(steps: 8)
        #expect(sigmas.count == 9)
        #expect(sigmas[0] == 1.0)
        #expect(sigmas.last == 0.0)
        for index in 1 ..< sigmas.count {
            #expect(sigmas[index] < sigmas[index - 1])
        }
    }

    @Test("Sigma schedule matches the reference values")
    func sigmaScheduleMatchesReference() {
        // Produced by the ComfyUI reference's time_snr_shift(12.0, t) over a
        // uniform base grid; see scripts/minimax-h3-parity.
        let expected: [Double] = [
            1.0, 0.988235294118, 0.972972972973, 0.952380952381,
            0.923076923077, 0.878048780488, 0.800000000000, 0.631578947368, 0.0
        ]
        let sigmas = scheduler.sigmas(steps: 8)
        #expect(sigmas.count == expected.count)
        for (actual, reference) in zip(sigmas, expected) {
            #expect(abs(actual - reference) < 1e-11)
        }
    }

    @Test("Audio sigma matches the reference shift")
    func audioSigmaMatchesReference() {
        // time_shift_sigma(0.5, 12 -> 3) == 0.2 in the reference.
        #expect(abs(scheduler.audioSigma(videoSigma: 0.5) - 0.2) < 1e-9)
        #expect(abs(scheduler.audioSigma(videoSigma: 0.9) - 0.692307692308) < 1e-9)
    }

    @Test("Shift round-trips between the two schedules")
    func shiftRoundTrip() {
        // Moving a sigma onto the audio grid and back must be the identity.
        let videoSigma = 0.63
        let audio = MiniMaxH3FlowScheduler.timeShiftSigma(
            videoSigma, from: 12.0, to: 3.0
        )
        let back = MiniMaxH3FlowScheduler.timeShiftSigma(audio, from: 3.0, to: 12.0)
        #expect(abs(back - videoSigma) < 1e-12)
    }

    @Test("Audio sigma trails the video sigma under a smaller shift")
    func audioSigmaOrdering() {
        // audio_shift < video_shift, so the audio stream sits lower on the
        // schedule at the same step.
        for sigma in [0.1, 0.5, 0.9] {
            #expect(scheduler.audioSigma(videoSigma: sigma) < sigma)
        }
    }

    @Test("Carry factor is finite at sigma zero")
    func carryIsFiniteAtZero() {
        // The reference clamps sigma before dividing; without that this is 0/0.
        let carry = scheduler.carry(videoSigma: 0.0)
        #expect(carry.isFinite)
        #expect(carry > 0)
    }
}

import Foundation
import Testing

@testable import MiniMaxH3SwiftRuntime

@Suite("MiniMax H3 packed layout")
struct MiniMaxH3PackedLayoutTests {
    /// Matches the reference `PackedLayout(7, 3, 8, 12, 5)`.
    private let layout = MiniMaxH3PackedLayout(
        textLength: 7, latentFrames: 3, latentHeight: 8, latentWidth: 12, audioFrames: 5
    )

    @Test("Segments are text, then audio, then video")
    func segmentOrder() {
        // The reference is explicit that the two target streams are last, audio
        // before video.
        #expect(layout.segments.map(\.kind) == [.text, .audio, .video])
        #expect(layout.segments.map(\.start) == [0, 7, 17])
        #expect(layout.segments.map(\.end) == [7, 17, 89])
        #expect(layout.sequenceLength == 89)
    }

    @Test("Segment sizes follow the patch and stereo layout")
    func segmentSizes() {
        // audio is channel-major stereo: 5 frames x 2 channels
        #expect(layout.segment(.audio)?.count == 10)
        // video is one row per 2x2 patch: 3 frames x (8/2 * 12/2)
        #expect(layout.segment(.video)?.count == 72)
    }

    @Test("Modality tags match the AdaLN row convention")
    func modalityTags() {
        // video 0, text 1, audio 2 — the three AdaLN modalities.
        #expect(MiniMaxH3PackedLayout.Kind.video.modalityTag == 0)
        #expect(MiniMaxH3PackedLayout.Kind.text.modalityTag == 1)
        #expect(MiniMaxH3PackedLayout.Kind.audio.modalityTag == 2)
    }

    @Test("Text positions count up on the time axis only")
    func textPositions() {
        for index in 0 ..< 7 {
            #expect(layout.positionIDs[index * 3] == Double(index))
            #expect(layout.positionIDs[index * 3 + 1] == 0)
            #expect(layout.positionIDs[index * 3 + 2] == 0)
        }
    }

    @Test("Video time grid is an exclusive cumsum of rescaled frame spans")
    func videoTimeGrid() {
        // framesPerToken cycles (1, 4, 4, 4, 4) and each is scaled by 5/3, so
        // the first token spans 5/3 and the rest 20/3.
        let grid = MiniMaxH3PackedLayout.videoTimeGrid(count: 4, origin: 7)
        #expect(abs(grid[0] - 7.0) < 1e-12)
        #expect(abs(grid[1] - (7.0 + 5.0 / 3.0)) < 1e-12)
        #expect(abs(grid[2] - (7.0 + 5.0 / 3.0 + 20.0 / 3.0)) < 1e-12)
        #expect(abs(grid[3] - (7.0 + 5.0 / 3.0 + 40.0 / 3.0)) < 1e-12)
    }

    @Test("Audio rows pin each stereo channel to a width extreme")
    func audioPositions() {
        let (_, widthAxis) = MiniMaxH3PackedLayout.frameGrid(height: 8, width: 12)
        let audio = layout.segment(.audio)!
        let low = layout.positionIDs[audio.start * 3 + 2]
        let high = layout.positionIDs[(audio.start + 5) * 3 + 2]
        #expect(abs(low - widthAxis.first!) < 1e-12)
        #expect(abs(high - widthAxis.last!) < 1e-12)
        // Time advances per latent frame within each channel, and restarts for
        // the second channel.
        #expect(layout.positionIDs[audio.start * 3] == 7)
        #expect(layout.positionIDs[(audio.start + 5) * 3] == 7)
    }

    @Test("Keyframes insert conditioning rows between text and the targets")
    func keyframeSegments() {
        // Verified against the reference PackedLayout(7, 3, 8, 12, 5,
        // keyframes=[{0, 1 latent frame}, {7, 1 latent frame}]).
        let anchored = MiniMaxH3PackedLayout(
            textLength: 7, latentFrames: 3, latentHeight: 8, latentWidth: 12,
            audioFrames: 5,
            keyframes: [
                MiniMaxH3Keyframe(resolvedFrameIndex: 0, videoLatentFrames: 1),
                MiniMaxH3Keyframe(resolvedFrameIndex: 7, videoLatentFrames: 1)
            ]
        )
        #expect(anchored.segments.map(\.kind) == [.text, .cond, .cond, .audio, .video])
        #expect(anchored.segments.map(\.start) == [0, 7, 31, 55, 65])
        #expect(anchored.segments.map(\.end) == [7, 31, 55, 65, 137])
        #expect(anchored.sequenceLength == 137)
        // Each cond block is one latent frame of the target spatial grid.
        #expect(anchored.frameRows == 24)
    }

    @Test("Conditioning rows carry the modality tag of their stream")
    func conditioningTags() {
        #expect(MiniMaxH3PackedLayout.Kind.cond.modalityTag == 0)
        #expect(MiniMaxH3PackedLayout.Kind.condAudio.modalityTag == 2)
        // Only the two target streams are denoised.
        #expect(MiniMaxH3PackedLayout.Kind.video.isTarget)
        #expect(MiniMaxH3PackedLayout.Kind.audio.isTarget)
        #expect(!MiniMaxH3PackedLayout.Kind.cond.isTarget)
        #expect(!MiniMaxH3PackedLayout.Kind.condAudio.isTarget)
    }

    @Test("Keyframe time anchor is rescaled per pixel frame")
    func keyframeTimeAnchor() {
        let anchored = MiniMaxH3PackedLayout(
            textLength: 7, latentFrames: 3, latentHeight: 8, latentWidth: 12,
            audioFrames: 5,
            keyframes: [MiniMaxH3Keyframe(resolvedFrameIndex: 7,
                                          videoLatentFrames: 1)]
        )
        let cond = anchored.segment(.cond)!
        // cursor (= textLength) + 5/3 * frameIndex
        #expect(abs(anchored.positionIDs[cond.start * 3] - (7.0 + 5.0 / 3.0 * 7.0)) < 1e-12)
    }

    @Test("Axis coordinates are area-normalized and start below zero when wide")
    func axisNormalization() {
        // width 12 > sqrt(96), so the ratio exceeds 1 and the axis starts negative.
        let area = (8.0 * 12.0).squareRoot()
        let axis = MiniMaxH3PackedLayout.axisFromSqrtArea(
            dimension: 12, patch: 2, sqrtArea: area
        )
        #expect(axis.count == 6)
        #expect(axis[0] < 0)
        // Verified against the reference layout dump.
        #expect(abs(axis[0] - -3.5959179422654266) < 1e-9)
        #expect(abs(axis[5] - 29.063945294843617) < 1e-9)
    }
}

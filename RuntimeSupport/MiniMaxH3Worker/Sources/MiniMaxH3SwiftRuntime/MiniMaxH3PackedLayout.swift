import Foundation

/// A frame the video is anchored to.
///
/// FL2VA supplies encoded latents for specific frames — typically the first and
/// last — which enter the packed sequence as conditioning rows that are never
/// denoised.
public struct MiniMaxH3Keyframe: Sendable, Equatable {
    /// Frame index in the output video, resolved to a non-negative value.
    public var resolvedFrameIndex: Int
    /// Latent temporal extent of the anchored video, or nil for audio-only.
    public var videoLatentFrames: Int?
    /// Latent temporal extent of the anchored audio, or nil for video-only.
    public var audioLatentFrames: Int?

    public init(
        resolvedFrameIndex: Int,
        videoLatentFrames: Int? = nil,
        audioLatentFrames: Int? = nil
    ) {
        self.resolvedFrameIndex = resolvedFrameIndex
        self.videoLatentFrames = videoLatentFrames
        self.audioLatentFrames = audioLatentFrames
    }
}

/// Packed-sequence structure for one H3 forward pass.
///
/// Ported from `PackedLayout` in the ComfyUI reference (`model.py`). The
/// sequence is `[text, (cond | cond_audio)*, audio, video]`: keyframe
/// conditioning rows sit right after the text, and the two target streams are
/// always the last two segments, audio before video.
///
/// Reference blocks (Ref2VA) are not modelled.
public struct MiniMaxH3PackedLayout: Sendable, Equatable {
    public enum Kind: String, Sendable {
        case text
        /// Keyframe video conditioning; not denoised.
        case cond
        /// Keyframe audio conditioning; not denoised.
        case condAudio
        case audio
        case video

        /// AdaLN modality tag: video 0, text 1, audio 2. Conditioning rows
        /// carry the tag of the stream they condition.
        public var modalityTag: Int {
            switch self {
            case .video, .cond: 0
            case .text: 1
            case .audio, .condAudio: 2
            }
        }

        /// True for the streams the sampler updates.
        public var isTarget: Bool {
            self == .video || self == .audio
        }
    }

    public struct Segment: Sendable, Equatable {
        public let start: Int
        public let end: Int
        public let kind: Kind

        public var count: Int { end - start }
    }

    /// Time-axis rescale applied per pixel frame.
    public static let frameRescale = 5.0 / 3.0
    /// Frames represented by each video latent token, cycling with period 5.
    public static let framesPerToken = [1.0, 4.0, 4.0, 4.0, 4.0]
    /// Timestep the visual conditioning rows are pinned near.
    public static let visualCondTimestep = 0.999
    /// Timestep the audio conditioning rows are pinned at.
    public static let audioCondTimestep = 1.0

    public let textLength: Int
    public let latentFrames: Int
    public let latentHeight: Int
    public let latentWidth: Int
    public let audioFrames: Int
    public let keyframes: [MiniMaxH3Keyframe]
    public let segments: [Segment]
    /// Row-major `[sequenceLength * 3]` positions as (t, h, w).
    public let positionIDs: [Double]

    public var sequenceLength: Int { segments.last?.end ?? 0 }

    public func segment(_ kind: Kind) -> Segment? {
        segments.first { $0.kind == kind }
    }

    /// Rows per latent frame, i.e. one per 2x2 patch.
    public var frameRows: Int { (latentHeight / 2) * (latentWidth / 2) }

    /// Coordinates of one axis, normalized by the frame's sqrt area.
    static func axisFromSqrtArea(dimension: Int, patch: Int, sqrtArea: Double) -> [Double] {
        let ratio = Double(dimension) / sqrtArea
        let count = dimension / patch
        guard count > 0 else { return [] }
        return (0 ..< count).map { index in
            (Double(index) * (ratio / Double(count)) + (1.0 - ratio) / 2.0) * 32.0
        }
    }

    /// Per-patch (h, w) coordinates of one latent frame, plus the width axis.
    static func frameGrid(
        height: Int,
        width: Int
    ) -> (rows: [(h: Double, w: Double)], widthAxis: [Double]) {
        let area = (Double(height) * Double(width)).squareRoot()
        let hAxis = axisFromSqrtArea(dimension: height, patch: 2, sqrtArea: area)
        let wAxis = axisFromSqrtArea(dimension: width, patch: 2, sqrtArea: area)
        var rows: [(h: Double, w: Double)] = []
        rows.reserveCapacity(hAxis.count * wAxis.count)
        for h in hAxis {
            for w in wAxis {
                rows.append((h: h, w: w))
            }
        }
        return (rows, wAxis)
    }

    /// Video token time coordinates: origin plus an exclusive cumulative sum of
    /// per-token frame spans.
    static func videoTimeGrid(count: Int, origin: Double) -> [Double] {
        var result = [Double]()
        result.reserveCapacity(count)
        var accumulated = 0.0
        for index in 0 ..< count {
            result.append(origin + accumulated)
            accumulated += frameRescale * framesPerToken[index % framesPerToken.count]
        }
        return result
    }

    public init(
        textLength: Int,
        latentFrames: Int,
        latentHeight: Int,
        latentWidth: Int,
        audioFrames: Int,
        keyframes: [MiniMaxH3Keyframe] = []
    ) {
        self.textLength = textLength
        self.latentFrames = latentFrames
        self.latentHeight = latentHeight
        self.latentWidth = latentWidth
        self.audioFrames = audioFrames
        self.keyframes = keyframes

        let (frame, widthAxis) = Self.frameGrid(height: latentHeight, width: latentWidth)
        let widthLow = widthAxis.first ?? 0
        let widthHigh = widthAxis.last ?? 0
        var positions = [Double]()
        var built: [Segment] = []
        var offset = 0

        func append(kind: Kind, rows: [(t: Double, h: Double, w: Double)]) {
            for row in rows {
                positions.append(row.t)
                positions.append(row.h)
                positions.append(row.w)
            }
            built.append(Segment(start: offset, end: offset + rows.count, kind: kind))
            offset += rows.count
        }

        /// Video rows: the frame's spatial grid repeated over its time grid.
        func videoRows(frames: Int, origin: Double) -> [(t: Double, h: Double, w: Double)] {
            Self.videoTimeGrid(count: frames, origin: origin).flatMap { time in
                frame.map { (t: time, h: $0.h, w: $0.w) }
            }
        }

        /// Audio rows: channel-major stereo, each channel pinned to a width
        /// extreme, h fixed at zero.
        func audioRows(frames: Int, origin: Double) -> [(t: Double, h: Double, w: Double)] {
            (0 ..< 2).flatMap { channel in
                (0 ..< frames).map { index in
                    (t: origin + Double(index), h: 0.0,
                     w: channel == 0 ? widthLow : widthHigh)
                }
            }
        }

        // Text: t counts up from zero, h and w pinned to zero.
        append(kind: .text, rows: (0 ..< textLength).map {
            (t: Double($0), h: 0.0, w: 0.0)
        })

        // With no reference blocks the target timeline starts right after text.
        let cursor = Double(textLength)

        // Keyframe conditioning shares the target spatial grid; its time anchor
        // counts from the target origin, rescaled per pixel frame.
        for keyframe in keyframes {
            let condTime = cursor
                + Self.frameRescale * Double(keyframe.resolvedFrameIndex)
            if let frames = keyframe.videoLatentFrames {
                append(kind: .cond, rows: videoRows(frames: frames, origin: condTime))
            }
            if let frames = keyframe.audioLatentFrames {
                append(kind: .condAudio, rows: audioRows(frames: frames, origin: condTime))
            }
        }

        // Targets are always last, audio before video.
        append(kind: .audio, rows: audioRows(frames: audioFrames, origin: cursor))
        append(kind: .video, rows: videoRows(frames: latentFrames, origin: cursor))

        self.segments = built
        self.positionIDs = positions
    }
}

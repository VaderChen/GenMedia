import Foundation

/// Temporal chunking parameters used by the MiniMax H3 video VAE decoder.
///
/// The values mirror the reference VAE: a 17-frame clip, three dropped tail
/// tokens, and a four-frame latent-to-pixel ratio. The decoder keeps the
/// existing single-shot path for short inputs and uses this plan only when a
/// caller explicitly selects temporal tiling.
public struct MiniMaxH3VideoVAETemporalTiling: Sendable, Equatable {
    public var clipLength: Int
    public var tokenDrop: Int
    public var minimumLatentFrames: Int

    public init(
        clipLength: Int = 17,
        tokenDrop: Int = 3,
        minimumLatentFrames: Int = 8
    ) {
        self.clipLength = max(1, clipLength)
        self.tokenDrop = max(0, tokenDrop)
        self.minimumLatentFrames = max(2, minimumLatentFrames)
    }

    /// Latent tokens in each non-overlapping temporal chunk.
    public func tokensPerChunk(temporalRatio: Int) -> Int {
        let ratio = max(1, temporalRatio)
        return max(1, (clipLength + ratio - 1) / ratio)
    }

    /// Latent-token overlap between adjacent chunks.
    public func tokenOverlap(temporalRatio: Int) -> Int {
        let chunkSize = tokensPerChunk(temporalRatio: temporalRatio)
        let remainder = tokenDrop % chunkSize
        return remainder == 0 ? 0 : chunkSize - remainder
    }

    /// Causal front padding removed from each decoded chunk.
    public func framePrePadding(temporalRatio: Int) -> Int {
        let ratio = max(1, temporalRatio)
        let remainder = clipLength % ratio
        return remainder == 0 ? 0 : ratio - remainder
    }

    /// Number of overlapping output frames blended at a chunk boundary.
    public func frameOverlap(temporalRatio: Int) -> Int {
        max(
            tokenOverlap(temporalRatio: temporalRatio) * max(1, temporalRatio)
                - framePrePadding(temporalRatio: temporalRatio),
            0
        )
    }

    public func shouldUse(for latentFrameCount: Int) -> Bool {
        latentFrameCount >= minimumLatentFrames
    }

    /// The temporal slices decoded after tail-token padding.
    public func plan(
        latentFrameCount: Int,
        temporalRatio: Int
    ) -> TemporalPlan {
        let ratio = max(1, temporalRatio)
        let chunkSize = tokensPerChunk(temporalRatio: ratio)
        let overlap = tokenOverlap(temporalRatio: ratio)
        let pseudoTotal = max(0, latentFrameCount) + tokenDrop
        var padTokens = (chunkSize - pseudoTotal % chunkSize) % chunkSize
        var paddedTotal = pseudoTotal + padTokens
        var chunkCount = paddedTotal / chunkSize - (tokenDrop > 0 ? 1 : 0)

        if chunkCount < 1 {
            padTokens += chunkSize
            paddedTotal += chunkSize
            chunkCount = 1
        }

        var ranges: [Range<Int>] = []
        ranges.reserveCapacity(chunkCount)
        for index in 0 ..< chunkCount {
            let start = index * chunkSize
            let end = min(start + chunkSize + overlap, paddedTotal)
            ranges.append(start ..< end)
        }

        return TemporalPlan(
            paddedTokenCount: paddedTotal,
            padTokenCount: padTokens,
            chunkCount: chunkCount,
            ranges: ranges
        )
    }

    /// Output frame count produced by the reference temporal stitching plan.
    public func outputFrameCount(
        latentFrameCount: Int,
        temporalRatio: Int
    ) -> Int {
        let ratio = max(1, temporalRatio)
        guard latentFrameCount > 1 else { return ratio }

        let plan = plan(
            latentFrameCount: latentFrameCount,
            temporalRatio: ratio
        )
        let chunkDecodedFrames = tokensPerChunk(temporalRatio: ratio) * ratio
        let splitCount = tokenDrop > 0 ? 2 : 1
        let prePadding = framePrePadding(temporalRatio: ratio)
        var totalFrames = 0
        var finalOverlapFrames = 0

        for range in plan.ranges {
            let clipTokenCount = range.count
            let clipFrameCount = clipTokenCount * ratio
            for split in 0 ..< splitCount {
                let start = split * chunkDecodedFrames
                let end = min(start + chunkDecodedFrames, clipFrameCount)
                let frames = max(0, end - start - prePadding)
                if split == 0 {
                    totalFrames += frames
                } else {
                    finalOverlapFrames = frames
                }
            }
        }

        totalFrames += finalOverlapFrames
        totalFrames -= temporalPadFrames(
            paddedTokenCount: plan.paddedTokenCount,
            padTokenCount: plan.padTokenCount,
            temporalRatio: ratio
        )
        return max(0, totalFrames)
    }

    private func temporalPadFrames(
        paddedTokenCount: Int,
        padTokenCount: Int,
        temporalRatio: Int
    ) -> Int {
        guard padTokenCount > 0 else { return 0 }
        let ratio = max(1, temporalRatio)
        let intraTail = clipLength % ratio
        if intraTail == 0 {
            return padTokenCount * ratio
        }

        let originalTokenCount = paddedTokenCount - padTokenCount
        let chunkSize = tokensPerChunk(temporalRatio: ratio)
        var frames = 0
        for offset in 0 ..< padTokenCount {
            frames += (originalTokenCount + offset) % chunkSize == 0
                ? intraTail : ratio
        }
        return frames
    }

    public struct TemporalPlan: Sendable, Equatable {
        public let paddedTokenCount: Int
        public let padTokenCount: Int
        public let chunkCount: Int
        public let ranges: [Range<Int>]

        public init(
            paddedTokenCount: Int,
            padTokenCount: Int,
            chunkCount: Int,
            ranges: [Range<Int>]
        ) {
            self.paddedTokenCount = paddedTokenCount
            self.padTokenCount = padTokenCount
            self.chunkCount = chunkCount
            self.ranges = ranges
        }
    }
}

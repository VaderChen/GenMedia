import Foundation
import MLX

public struct LTXSpatialTilingConfiguration: Sendable, Equatable {
    public let tileSizeInPixels: Int
    public let tileOverlapInPixels: Int

    public init(tileSizeInPixels: Int, tileOverlapInPixels: Int = 0) throws {
        guard tileSizeInPixels >= 64, tileSizeInPixels.isMultiple(of: 32) else {
            throw LTXVideoRuntimeError.invalidTileConfiguration(
                "spatial tile size 必須至少為 64 且可被 32 整除。"
            )
        }
        guard tileOverlapInPixels >= 0,
              tileOverlapInPixels.isMultiple(of: 32),
              tileOverlapInPixels < tileSizeInPixels else {
            throw LTXVideoRuntimeError.invalidTileConfiguration(
                "spatial overlap 必須為非負、可被 32 整除，且小於 tile size。"
            )
        }
        self.tileSizeInPixels = tileSizeInPixels
        self.tileOverlapInPixels = tileOverlapInPixels
    }
}

public struct LTXTemporalTilingConfiguration: Sendable, Equatable {
    public let tileSizeInFrames: Int
    public let tileOverlapInFrames: Int

    public init(tileSizeInFrames: Int, tileOverlapInFrames: Int = 0) throws {
        guard tileSizeInFrames >= 16, tileSizeInFrames.isMultiple(of: 8) else {
            throw LTXVideoRuntimeError.invalidTileConfiguration(
                "temporal tile size 必須至少為 16 且可被 8 整除。"
            )
        }
        guard tileOverlapInFrames >= 0,
              tileOverlapInFrames.isMultiple(of: 8),
              tileOverlapInFrames < tileSizeInFrames else {
            throw LTXVideoRuntimeError.invalidTileConfiguration(
                "temporal overlap 必須為非負、可被 8 整除，且小於 tile size。"
            )
        }
        self.tileSizeInFrames = tileSizeInFrames
        self.tileOverlapInFrames = tileOverlapInFrames
    }
}

public struct LTXTilingConfiguration: Sendable, Equatable {
    public let spatial: LTXSpatialTilingConfiguration?
    public let temporal: LTXTemporalTilingConfiguration?

    public init(
        spatial: LTXSpatialTilingConfiguration? = nil,
        temporal: LTXTemporalTilingConfiguration? = nil
    ) {
        self.spatial = spatial
        self.temporal = temporal
    }
}

public struct LTXVideoSlice: Sendable, Equatable {
    public let start: Int
    public let end: Int

    public init(_ start: Int, _ end: Int) {
        self.start = start
        self.end = end
    }

    public var count: Int { end - start }
}

public struct LTXDimensionIntervals: Sendable, Equatable {
    public let starts: [Int]
    public let ends: [Int]
    public let leftRamps: [Int]
    public let rightRamps: [Int]

    public init(starts: [Int], ends: [Int], leftRamps: [Int], rightRamps: [Int]) {
        precondition(starts.count == ends.count)
        precondition(starts.count == leftRamps.count)
        precondition(starts.count == rightRamps.count)
        self.starts = starts
        self.ends = ends
        self.leftRamps = leftRamps
        self.rightRamps = rightRamps
    }
}

public struct LTXBlendMask1D: Sendable, Equatable {
    public let values: [Float]

    public init(values: [Float]) {
        self.values = values
    }
}

public struct LTXVideoTile: Sendable, Equatable {
    public let input: [LTXVideoSlice]
    public let output: [LTXVideoSlice]
    public let masks: [LTXBlendMask1D?]

    public init(
        input: [LTXVideoSlice],
        output: [LTXVideoSlice],
        masks: [LTXBlendMask1D?]
    ) {
        self.input = input
        self.output = output
        self.masks = masks
    }

    public func blendMask(dtype: DType) -> MLXArray {
        var result = MLXArray.ones(Array(repeating: 1, count: output.count), dtype: dtype)
        for (axis, mask) in masks.enumerated() {
            guard let mask else { continue }
            var shape = Array(repeating: 1, count: output.count)
            shape[axis] = mask.values.count
            result = result * MLXArray(mask.values).asType(dtype).reshaped(shape)
        }
        return result
    }
}

public enum LTXVideoTiling {
    public static let temporalScale = 8
    public static let spatialScale = 32

    public static func trapezoidalMask(
        length: Int,
        leftRamp: Int,
        rightRamp: Int,
        leftStartsFromZero: Bool = false
    ) throws -> [Float] {
        guard length > 0 else {
            throw LTXVideoRuntimeError.invalidTileConfiguration("mask length 必須大於 0。")
        }
        let left = max(0, min(leftRamp, length))
        let right = max(0, min(rightRamp, length))
        var values = Array(repeating: Float(1), count: length)
        if left > 0 {
            for index in 0..<left {
                let fade: Float = leftStartsFromZero
                    ? Float(index) / Float(left)
                    : Float(index + 1) / Float(left + 1)
                values[index] *= fade
            }
        }
        if right > 0 {
            for index in 0..<right {
                values[length - right + index] *= Float(right - index) / Float(right + 1)
            }
        }
        return values.map { min(1, max(0, $0)) }
    }

    public static func splitWithSymmetricOverlaps(
        dimension: Int,
        size: Int,
        overlap: Int
    ) -> LTXDimensionIntervals {
        if dimension <= size {
            return LTXDimensionIntervals(
                starts: [0], ends: [dimension], leftRamps: [0], rightRamps: [0]
            )
        }
        let amount = (dimension + size - 2 * overlap - 1) / (size - overlap)
        let starts = (0..<amount).map { $0 * (size - overlap) }
        var ends = starts.map { $0 + size }
        ends[ends.count - 1] = dimension
        return LTXDimensionIntervals(
            starts: starts,
            ends: ends,
            leftRamps: [0] + Array(repeating: overlap, count: amount - 1),
            rightRamps: Array(repeating: overlap, count: amount - 1) + [0]
        )
    }

    public static func splitTemporalLatents(
        dimension: Int,
        size: Int,
        overlap: Int
    ) -> LTXDimensionIntervals {
        if dimension <= size {
            return LTXDimensionIntervals(
                starts: [0], ends: [dimension], leftRamps: [0], rightRamps: [0]
            )
        }
        let base = splitWithSymmetricOverlaps(
            dimension: dimension,
            size: size,
            overlap: overlap
        )
        var starts = base.starts
        var leftRamps = base.leftRamps
        for index in 1..<starts.count {
            starts[index] -= 1
            leftRamps[index] += 1
        }
        return LTXDimensionIntervals(
            starts: starts,
            ends: base.ends,
            leftRamps: leftRamps,
            rightRamps: base.rightRamps
        )
    }

    public static func prepareDecoderTiles(
        latentShape: [Int],
        configuration: LTXTilingConfiguration?
    ) throws -> [LTXVideoTile] {
        guard latentShape.count == 5,
              latentShape[0] > 0,
              latentShape[1] == 128,
              latentShape[2] > 0,
              latentShape[3] > 0,
              latentShape[4] > 0 else {
            throw LTXVideoRuntimeError.invalidLatentShape(latentShape)
        }

        var intervals = latentShape.map {
            LTXDimensionIntervals(starts: [0], ends: [$0], leftRamps: [0], rightRamps: [0])
        }
        var mappings: [[MappedInterval]] = latentShape.indices.map { axis in
            let outputEnd: Int
            switch axis {
            case 2:
                outputEnd = 1 + (latentShape[axis] - 1) * temporalScale
            case 3, 4:
                outputEnd = latentShape[axis] * spatialScale
            default:
                outputEnd = latentShape[axis]
            }
            return [MappedInterval(
                output: LTXVideoSlice(0, outputEnd),
                mask: nil
            )]
        }

        if let spatial = configuration?.spatial {
            let longSide = max(latentShape[3], latentShape[4])
            for axis in [3, 4] {
                let tileSize = spatial.tileSizeInPixels / spatialScale
                let overlap = spatial.tileOverlapInPixels / spatialScale
                let lowerThreshold = max(2, overlap + 1)
                let scaled = Double(tileSize * latentShape[axis]) / Double(longSide)
                let adjustedSize = max(
                    lowerThreshold,
                    Int(scaled.rounded(.toNearestOrEven))
                )
                let split = splitWithSymmetricOverlaps(
                    dimension: latentShape[axis],
                    size: adjustedSize,
                    overlap: overlap
                )
                intervals[axis] = split
                mappings[axis] = try zip4(split).map { begin, end, left, right in
                    MappedInterval(
                        output: LTXVideoSlice(begin * spatialScale, end * spatialScale),
                        mask: LTXBlendMask1D(values: try trapezoidalMask(
                            length: (end - begin) * spatialScale,
                            leftRamp: left * spatialScale,
                            rightRamp: right * spatialScale
                        ))
                    )
                }
            }
        }

        if let temporal = configuration?.temporal {
            let split = splitTemporalLatents(
                dimension: latentShape[2],
                size: temporal.tileSizeInFrames / temporalScale,
                overlap: temporal.tileOverlapInFrames / temporalScale
            )
            intervals[2] = split
            mappings[2] = try zip4(split).map { begin, end, left, right in
                let start = begin * temporalScale
                let endFrame = 1 + (end - 1) * temporalScale
                let leftFrames = left == 0 ? 0 : 1 + (left - 1) * temporalScale
                let rightFrames = right * temporalScale
                return MappedInterval(
                    output: LTXVideoSlice(start, endFrame),
                    mask: LTXBlendMask1D(values: try trapezoidalMask(
                        length: endFrame - start,
                        leftRamp: leftFrames,
                        rightRamp: rightFrames,
                        leftStartsFromZero: true
                    ))
                )
            }
        }

        var tiles: [LTXVideoTile] = []
        var selectedIndices = Array(repeating: 0, count: latentShape.count)
        func appendTiles(axis: Int) {
            if axis == latentShape.count {
                var inputSlices: [LTXVideoSlice] = []
                var outputSlices: [LTXVideoSlice] = []
                var masks: [LTXBlendMask1D?] = []
                for currentAxis in latentShape.indices {
                    let index = selectedIndices[currentAxis]
                    inputSlices.append(LTXVideoSlice(
                        intervals[currentAxis].starts[index],
                        intervals[currentAxis].ends[index]
                    ))
                    outputSlices.append(mappings[currentAxis][index].output)
                    masks.append(mappings[currentAxis][index].mask)
                }
                tiles.append(LTXVideoTile(
                    input: inputSlices,
                    output: outputSlices,
                    masks: masks
                ))
                return
            }
            for index in intervals[axis].starts.indices {
                selectedIndices[axis] = index
                appendTiles(axis: axis + 1)
            }
        }
        appendTiles(axis: 0)
        return tiles
    }

    private struct MappedInterval {
        let output: LTXVideoSlice
        let mask: LTXBlendMask1D?
    }

    private static func zip4(
        _ intervals: LTXDimensionIntervals
    ) -> [(Int, Int, Int, Int)] {
        intervals.starts.indices.map {
            (
                intervals.starts[$0], intervals.ends[$0],
                intervals.leftRamps[$0], intervals.rightRamps[$0]
            )
        }
    }
}

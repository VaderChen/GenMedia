import Foundation
import MLX

public struct LTXModalityDimensionConfiguration: Sendable, Equatable {
    public let numTiles: Int
    public let overlap: Int

    public init(numTiles: Int = 1, overlap: Int = 0) throws {
        guard numTiles > 0, overlap >= 0 else {
            throw LTXVideoRuntimeError.invalidTileConfiguration(
                "modality tile 數量必須大於 0，overlap 不可為負數。"
            )
        }
        self.numTiles = numTiles
        self.overlap = overlap
    }
}

public struct LTXModalityTileConfiguration: Sendable, Equatable {
    public let frames: LTXModalityDimensionConfiguration
    public let height: LTXModalityDimensionConfiguration
    public let width: LTXModalityDimensionConfiguration

    public init(
        frames: LTXModalityDimensionConfiguration = try! .init(),
        height: LTXModalityDimensionConfiguration = try! .init(),
        width: LTXModalityDimensionConfiguration = try! .init()
    ) {
        self.frames = frames
        self.height = height
        self.width = width
    }
}

public enum LTXModalityTiling {
    public static func splitByCount(
        dimension: Int,
        numTiles: Int,
        overlap: Int
    ) throws -> LTXDimensionIntervals {
        guard dimension > 0 else {
            throw LTXVideoRuntimeError.invalidTileConfiguration(
                "modality dimension 必須大於 0。"
            )
        }
        guard numTiles > 0, numTiles <= dimension else {
            throw LTXVideoRuntimeError.invalidTileConfiguration(
                "modality tile 數量不可超過 dimension。"
            )
        }
        guard overlap >= 0 else {
            throw LTXVideoRuntimeError.invalidTileConfiguration(
                "modality overlap 不可為負數。"
            )
        }
        if numTiles == 1 {
            return LTXDimensionIntervals(
                starts: [0], ends: [dimension], leftRamps: [0], rightRamps: [0]
            )
        }

        let total = dimension + overlap * (numTiles - 1)
        let tileSize = total / numTiles
        let remainder = total % numTiles
        guard tileSize > overlap else {
            throw LTXVideoRuntimeError.invalidTileConfiguration(
                "modality overlap 必須小於每個 tile 的有效尺寸。"
            )
        }

        let base = LTXVideoTiling.splitWithSymmetricOverlaps(
            dimension: dimension - remainder,
            size: tileSize,
            overlap: overlap
        )
        var starts: [Int] = []
        var ends: [Int] = []
        starts.reserveCapacity(numTiles)
        ends.reserveCapacity(numTiles)
        for index in 0..<numTiles {
            let shift = min(index, remainder)
            let grow = index < remainder ? 1 : 0
            starts.append(base.starts[index] + shift)
            ends.append(base.ends[index] + shift + grow)
        }
        return LTXDimensionIntervals(
            starts: starts,
            ends: ends,
            leftRamps: base.leftRamps,
            rightRamps: base.rightRamps
        )
    }

    public static func makeTiles(
        frameCount: Int,
        height: Int,
        width: Int,
        configuration: LTXModalityTileConfiguration
    ) throws -> [LTXVideoTile] {
        guard frameCount > 0, height > 0, width > 0 else {
            throw LTXVideoRuntimeError.invalidTileConfiguration(
                "modality token grid 的三個 dimension 必須大於 0。"
            )
        }
        let intervals = try [
            splitByCount(
                dimension: frameCount,
                numTiles: configuration.frames.numTiles,
                overlap: configuration.frames.overlap
            ),
            splitByCount(
                dimension: height,
                numTiles: configuration.height.numTiles,
                overlap: configuration.height.overlap
            ),
            splitByCount(
                dimension: width,
                numTiles: configuration.width.numTiles,
                overlap: configuration.width.overlap
            )
        ]

        var tiles: [LTXVideoTile] = []
        var selections = Array(repeating: 0, count: intervals.count)
        func append(axis: Int) throws {
            if axis == intervals.count {
                var input: [LTXVideoSlice] = []
                var output: [LTXVideoSlice] = []
                var masks: [LTXBlendMask1D?] = []
                for dimension in intervals.indices {
                    let index = selections[dimension]
                    let start = intervals[dimension].starts[index]
                    let end = intervals[dimension].ends[index]
                    input.append(LTXVideoSlice(start, end))
                    output.append(LTXVideoSlice(start, end))
                    let left = intervals[dimension].leftRamps[index]
                    let right = intervals[dimension].rightRamps[index]
                    masks.append(
                        LTXBlendMask1D(values: try LTXVideoTiling.trapezoidalMask(
                            length: end - start,
                            leftRamp: left,
                            rightRamp: right
                        ))
                    )
                }
                tiles.append(LTXVideoTile(input: input, output: output, masks: masks))
                return
            }
            for index in intervals[axis].starts.indices {
                selections[axis] = index
                try append(axis: axis + 1)
            }
        }
        try append(axis: 0)
        return tiles
    }

    public static func tokenIndices(
        tile: LTXVideoTile,
        gridShape: (frames: Int, height: Int, width: Int)
    ) -> [Int] {
        let frame = tile.input[0]
        let row = tile.input[1]
        let column = tile.input[2]
        var indices: [Int] = []
        indices.reserveCapacity(frame.count * row.count * column.count)
        for frameIndex in frame.start..<frame.end {
            for rowIndex in row.start..<row.end {
                for columnIndex in column.start..<column.end {
                    indices.append(frameIndex * gridShape.height * gridShape.width
                        + rowIndex * gridShape.width + columnIndex)
                }
            }
        }
        return indices
    }

    public static func blendMaskValues(_ tile: LTXVideoTile) -> [Float] {
        let lengths = tile.input.map(\.count)
        var values = Array(repeating: Float(1), count: lengths.reduce(1, *))
        var cursor = 0
        for frame in 0..<lengths[0] {
            for row in 0..<lengths[1] {
                for column in 0..<lengths[2] {
                    let index = cursor
                    let masks = [frame, row, column]
                    for axis in 0..<3 {
                        if let mask = tile.masks[axis]?.values[masks[axis]] {
                            values[index] *= mask
                        }
                    }
                    cursor += 1
                }
            }
        }
        return values
    }

    public static func blend(
        tileOutput: MLXArray,
        tile: LTXVideoTile,
        gridShape: (frames: Int, height: Int, width: Int),
        output: MLXArray? = nil
    ) throws -> MLXArray {
        guard tileOutput.ndim == 3 else {
            throw LTXVideoRuntimeError.invalidConfiguration(
                "Modality tile output 必須是 [batch, tokens, channels]。"
            )
        }
        let indices = tokenIndices(tile: tile, gridShape: gridShape)
        guard tileOutput.shape[1] == indices.count else {
            throw LTXVideoRuntimeError.invalidConfiguration(
                "Modality tile output 的 token 數與 tile 座標不一致。"
            )
        }
        let batch = tileOutput.shape[0]
        let channels = tileOutput.shape[2]
        let totalTokens = gridShape.frames * gridShape.height * gridShape.width
        let result = output ?? MLXArray.zeros(
            [batch, totalTokens, channels], dtype: tileOutput.dtype
        )
        guard result.shape == [batch, totalTokens, channels] else {
            throw LTXVideoRuntimeError.invalidConfiguration(
                "Modality blend output 的 shape 不一致。"
            )
        }
        let indexArray = MLXArray(indices.map(Int32.init))
        let selected = result[0..., indexArray, 0...]
        let mask = MLXArray(blendMaskValues(tile))
            .asType(tileOutput.dtype)
            .reshaped(1, indices.count, 1)
        result[0..., indexArray, 0...] = selected + tileOutput * mask
        return result
    }
}

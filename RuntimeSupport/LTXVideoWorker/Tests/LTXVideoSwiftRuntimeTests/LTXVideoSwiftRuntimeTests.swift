import MLX
import Testing

@testable import LTXVideoSwiftRuntime

struct LTXVideoSwiftRuntimeTests {
    @Test func temporalSplitMatchesPythonReference() {
        let intervals = LTXVideoTiling.splitTemporalLatents(
            dimension: 4,
            size: 2,
            overlap: 1
        )

        #expect(intervals.starts == [0, 0, 1])
        #expect(intervals.ends == [2, 3, 4])
        #expect(intervals.leftRamps == [0, 2, 2])
        #expect(intervals.rightRamps == [1, 1, 0])
    }

    @Test func trapezoidalMasksMatchPythonReference() throws {
        let ordinary = try LTXVideoTiling.trapezoidalMask(
            length: 5,
            leftRamp: 2,
            rightRamp: 2
        )
        let causal = try LTXVideoTiling.trapezoidalMask(
            length: 5,
            leftRamp: 2,
            rightRamp: 2,
            leftStartsFromZero: true
        )

        #expect(abs(ordinary[0] - 1.0 / 3.0) < 1e-6)
        #expect(abs(ordinary[1] - 2.0 / 3.0) < 1e-6)
        #expect(ordinary[2] == 1)
        #expect(abs(ordinary[3] - 2.0 / 3.0) < 1e-6)
        #expect(abs(ordinary[4] - 1.0 / 3.0) < 1e-6)
        #expect(causal[0] == 0)
        #expect(causal[1] == 0.5)
        #expect(causal[2] == 1)
    }

    @Test func singleTemporalTileCoordinatesMatchPythonReference() throws {
        let configuration = LTXTilingConfiguration(
            temporal: try LTXTemporalTilingConfiguration(
                tileSizeInFrames: 16,
                tileOverlapInFrames: 8
            )
        )
        let tiles = try LTXVideoTiling.prepareDecoderTiles(
            latentShape: [1, 128, 2, 1, 1],
            configuration: configuration
        )

        #expect(tiles.count == 1)
        #expect(tiles[0].input[2] == LTXVideoSlice(0, 2))
        #expect(tiles[0].output[2] == LTXVideoSlice(0, 9))
        #expect(tiles[0].output[3] == LTXVideoSlice(0, 32))
        #expect(tiles[0].output[4] == LTXVideoSlice(0, 32))
    }

    @Test func multiTemporalTileCoordinatesExerciseStitching() throws {
        let configuration = LTXTilingConfiguration(
            temporal: try LTXTemporalTilingConfiguration(
                tileSizeInFrames: 16,
                tileOverlapInFrames: 8
            )
        )
        let tiles = try LTXVideoTiling.prepareDecoderTiles(
            latentShape: [1, 128, 4, 1, 1],
            configuration: configuration
        )

        #expect(tiles.count == 3)
        #expect(tiles.map { $0.input[2] } == [
            LTXVideoSlice(0, 2), LTXVideoSlice(0, 3), LTXVideoSlice(1, 4)
        ])
        #expect(tiles.map { $0.output[2] } == [
            LTXVideoSlice(0, 9), LTXVideoSlice(0, 17), LTXVideoSlice(8, 25)
        ])
        #expect(tiles[1].masks[2]?.values.first == 0)
        #expect(tiles[2].masks[2]?.values.first == 0)
    }

    @Test func spatialCoordinatesCreateMultipleRealTiles() throws {
        let configuration = LTXTilingConfiguration(
            spatial: try LTXSpatialTilingConfiguration(
                tileSizeInPixels: 64,
                tileOverlapInPixels: 32
            )
        )
        let tiles = try LTXVideoTiling.prepareDecoderTiles(
            latentShape: [1, 128, 1, 3, 2],
            configuration: configuration
        )

        #expect(tiles.count == 2)
        #expect(tiles.map { $0.input[3] } == [LTXVideoSlice(0, 2), LTXVideoSlice(1, 3)])
        #expect(tiles.map { $0.output[3] } == [LTXVideoSlice(0, 64), LTXVideoSlice(32, 96)])
    }
}

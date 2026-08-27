import Foundation
import MLX
import MLXNN

public final class LTXVideoVAEDecoder: Module {
    @ModuleInfo(key: "conv_in") public var convIn: LTXConv3DBlock
    @ModuleInfo(key: "up_blocks") public var upBlocks: [UnaryLayer]
    @ModuleInfo(key: "conv_out") public var convOut: LTXConv3DBlock
    @ModuleInfo(key: "per_channel_statistics")
    public var perChannelStatistics: LTXPerChannelStatistics

    public let configuration: LTXVideoVAEConfiguration
    private let upsampleFactors: [(spatial: Int, temporal: Int)] = [
        (2, 2), (2, 2), (1, 2), (2, 1)
    ]
    // 與 MiniMax Music 3 的 scheduler／輸出策略一致：神經網路主幹保留
    // BF16，但跨 tile 的加權、累加及除法固定在 FP32，避免先在 BF16
    // 乘上 ramp mask 後才寫入 FP32 buffer，造成不可逆的捨入誤差。
    private static let accumulationDType: DType = .float32

    public init(configuration: LTXVideoVAEConfiguration) {
        self.configuration = configuration
        let causal = configuration.causalDecoder
        let padding = configuration.spatialPaddingMode
        self._convIn = ModuleInfo(
            wrappedValue: LTXConv3DBlock(
                inputChannels: 128,
                outputChannels: 1024,
                causal: causal,
                spatialPaddingMode: padding
            ),
            key: "conv_in"
        )
        self._upBlocks = ModuleInfo(
            wrappedValue: [
                LTXResBlockStage(channels: 1024, count: 2, causal: causal, spatialPaddingMode: padding),
                LTXDepthToSpaceUpsample(inputChannels: 1024, outputChannels: 4096, causal: causal, spatialPaddingMode: padding),
                LTXResBlockStage(channels: 512, count: 2, causal: causal, spatialPaddingMode: padding),
                LTXDepthToSpaceUpsample(inputChannels: 512, outputChannels: 4096, causal: causal, spatialPaddingMode: padding),
                LTXResBlockStage(channels: 512, count: 4, causal: causal, spatialPaddingMode: padding),
                LTXDepthToSpaceUpsample(inputChannels: 512, outputChannels: 512, causal: causal, spatialPaddingMode: padding),
                LTXResBlockStage(channels: 256, count: 6, causal: causal, spatialPaddingMode: padding),
                LTXDepthToSpaceUpsample(inputChannels: 256, outputChannels: 512, causal: causal, spatialPaddingMode: padding),
                LTXResBlockStage(channels: 128, count: 4, causal: causal, spatialPaddingMode: padding)
            ],
            key: "up_blocks"
        )
        self._convOut = ModuleInfo(
            wrappedValue: LTXConv3DBlock(
                inputChannels: 128,
                outputChannels: 48,
                causal: causal,
                spatialPaddingMode: padding
            ),
            key: "conv_out"
        )
        self._perChannelStatistics = ModuleInfo(
            wrappedValue: LTXPerChannelStatistics(channels: 128),
            key: "per_channel_statistics"
        )
        super.init()
    }

    public func decode(
        _ latent: MLXArray,
        materializeStages: Bool = false
    ) throws -> MLXArray {
        try validate(latent)
        let outputDType = latent.dtype
        var value = latent
        if value.dtype != convIn.conv.weight.dtype {
            value = value.asType(convIn.conv.weight.dtype)
        }

        value = value.transposed(0, 2, 3, 4, 1)
        value = denormalize(value)
        value = convIn(value)

        var upsampleIndex = 0
        for (index, block) in upBlocks.enumerated() {
            value = block(value)
            if index % 2 == 1 {
                let factors = upsampleFactors[upsampleIndex]
                value = ltxPixelShuffle3D(
                    value,
                    spatialFactor: factors.spatial,
                    temporalFactor: factors.temporal
                )
                if factors.temporal > 1 {
                    value = value[0..., 1..<value.shape[1], 0..., 0..., 0...]
                }
                upsampleIndex += 1
                if materializeStages {
                    MLX.eval(value)
                    Memory.clearCache()
                }
            }
        }

        value = convOut(silu(ltxPixelNorm(value)))
        value = ltxUnpatchifySpatial(value, patchSize: configuration.patchSize)
        return value.transposed(0, 4, 1, 2, 3).asType(outputDType)
    }

    public func decodeTiled(
        _ latent: MLXArray,
        configuration tiling: LTXTilingConfiguration?
    ) throws -> MLXArray {
        var chunks: [MLXArray] = []
        _ = try decodeTiled(latent, configuration: tiling) { chunk in
            chunks.append(chunk)
        }
        guard let first = chunks.first else {
            throw LTXVideoRuntimeError.invalidTileConfiguration("VAE decode 沒有輸出 chunk。")
        }
        return chunks.dropFirst().reduce(first) { result, chunk in
            MLX.concatenated([result, chunk], axis: 2)
        }
    }

    @discardableResult
    public func decodeTiled(
        _ latent: MLXArray,
        configuration tiling: LTXTilingConfiguration?,
        consume: (MLXArray) throws -> Void
    ) throws -> Int {
        try validate(latent)
        guard let tiling else {
            let output = try decode(latent)
            MLX.eval(output)
            try consume(output)
            return 1
        }

        let tiles = try LTXVideoTiling.prepareDecoderTiles(
            latentShape: latent.shape,
            configuration: tiling
        )
        let groups = temporalGroups(tiles)
        let outputHeight = latent.shape[3] * LTXVideoTiling.spatialScale
        let outputWidth = latent.shape[4] * LTXVideoTiling.spatialScale

        var previousChunk: MLXArray?
        var previousWeights: MLXArray?
        var previousTemporalSlice: LTXVideoSlice?
        var emittedChunks = 0

        for group in groups {
            guard let firstTile = group.first else { continue }
            let currentTemporalSlice = firstTile.output[2]
            let temporalLength = currentTemporalSlice.count
            var buffer = MLXArray.zeros(
                [latent.shape[0], 3, temporalLength, outputHeight, outputWidth],
                dtype: Self.accumulationDType
            )
            var weights = MLXArray.zeros(like: buffer)

            for tile in group {
                let latentTile = latent[indices(for: tile.input)]
                let decodedTile = try decode(latentTile, materializeStages: true)
                MLX.eval(decodedTile)
                Memory.clearCache()

                let mask = tile.blendMask(dtype: Self.accumulationDType)
                let temporalOffset = tile.output[2].start - currentTemporalSlice.start
                let expectedTemporalLength = tile.output[2].count
                let actualTemporalLength = min(
                    expectedTemporalLength,
                    decodedTile.shape[2],
                    buffer.shape[2] - temporalOffset
                )
                let targetSlices = [
                    LTXVideoSlice(0, latent.shape[0]),
                    LTXVideoSlice(0, 3),
                    LTXVideoSlice(temporalOffset, temporalOffset + actualTemporalLength),
                    tile.output[3],
                    tile.output[4]
                ]
                let sourceSlices = [
                    LTXVideoSlice(0, decodedTile.shape[0]),
                    LTXVideoSlice(0, decodedTile.shape[1]),
                    LTXVideoSlice(0, actualTemporalLength),
                    LTXVideoSlice(0, decodedTile.shape[3]),
                    LTXVideoSlice(0, decodedTile.shape[4])
                ]
                let decodedSlice = decodedTile[indices(for: sourceSlices)]
                    .asType(Self.accumulationDType)
                let maskSlice: MLXArray
                if mask.shape[2] > 1 {
                    maskSlice = mask[indices(for: [
                        LTXVideoSlice(0, mask.shape[0]),
                        LTXVideoSlice(0, mask.shape[1]),
                        LTXVideoSlice(0, actualTemporalLength),
                        LTXVideoSlice(0, mask.shape[3]),
                        LTXVideoSlice(0, mask.shape[4])
                    ])]
                } else {
                    maskSlice = mask
                }
                buffer = add(decodedSlice * maskSlice, to: buffer, at: targetSlices)
                weights = add(maskSlice, to: weights, at: targetSlices)
                MLX.eval(buffer, weights)
                Memory.clearCache()
            }

            if var previousChunkValue = previousChunk,
               var previousWeightsValue = previousWeights,
               let previousSlice = previousTemporalSlice {
                if previousSlice.end > currentTemporalSlice.start {
                    let overlapLength = previousSlice.end - currentTemporalSlice.start
                    let previousOverlapStart = currentTemporalSlice.start - previousSlice.start
                    let previousPrefix = previousChunkValue[
                        0..., 0..., 0..<previousOverlapStart, 0..., 0...
                    ]
                    let previousWeightsPrefix = previousWeightsValue[
                        0..., 0..., 0..<previousOverlapStart, 0..., 0...
                    ]
                    let merged = previousChunkValue[
                        0..., 0..., previousOverlapStart..<previousChunkValue.shape[2], 0..., 0...
                    ] + buffer[0..., 0..., 0..<overlapLength, 0..., 0...]
                    let mergedWeights = previousWeightsValue[
                        0..., 0..., previousOverlapStart..<previousWeightsValue.shape[2], 0..., 0...
                    ] + weights[0..., 0..., 0..<overlapLength, 0..., 0...]
                    previousChunkValue = MLX.concatenated([previousPrefix, merged], axis: 2)
                    previousWeightsValue = MLX.concatenated(
                        [previousWeightsPrefix, mergedWeights], axis: 2
                    )
                    buffer = MLX.concatenated([
                        merged,
                        buffer[0..., 0..., overlapLength..<buffer.shape[2], 0..., 0...]
                    ], axis: 2)
                    weights = MLX.concatenated([
                        mergedWeights,
                        weights[0..., 0..., overlapLength..<weights.shape[2], 0..., 0...]
                    ], axis: 2)
                }

                let emitLength = currentTemporalSlice.start - previousSlice.start
                if emitLength > 0 {
                    let safeWeights = MLX.maximum(previousWeightsValue, 1e-8)
                    let chunk = (previousChunkValue / safeWeights)[
                        0..., 0..., 0..<emitLength, 0..., 0...
                    ]
                    MLX.eval(chunk)
                    try consume(chunk)
                    emittedChunks += 1
                }
            }

            previousChunk = buffer
            previousWeights = weights
            previousTemporalSlice = currentTemporalSlice
        }

        if let previousChunk, let previousWeights {
            let chunk = previousChunk / MLX.maximum(previousWeights, 1e-8)
            MLX.eval(chunk)
            try consume(chunk)
            emittedChunks += 1
        }
        return emittedChunks
    }

    private func validate(_ latent: MLXArray) throws {
        guard latent.ndim == 5,
              latent.shape[0] > 0,
              latent.shape[1] == configuration.latentChannels,
              latent.shape[2] > 0,
              latent.shape[3] > 0,
              latent.shape[4] > 0 else {
            throw LTXVideoRuntimeError.invalidLatentShape(latent.shape)
        }
    }

    private func denormalize(_ latent: MLXArray) -> MLXArray {
        let mean = perChannelStatistics.mean.reshaped([1, 1, 1, 1, -1])
        let standardDeviation = perChannelStatistics.std.reshaped([1, 1, 1, 1, -1])
        return latent * standardDeviation + mean
    }

    private func temporalGroups(_ tiles: [LTXVideoTile]) -> [[LTXVideoTile]] {
        var groups: [[LTXVideoTile]] = []
        for tile in tiles {
            if let last = groups.last, last.first?.output[2] == tile.output[2] {
                groups[groups.count - 1].append(tile)
            } else {
                groups.append([tile])
            }
        }
        return groups
    }

    private func indices(for slices: [LTXVideoSlice]) -> [any MLXArrayIndex] {
        slices.map { $0.start..<$0.end }
    }

    private func add(
        _ values: MLXArray,
        to buffer: MLXArray,
        at slices: [LTXVideoSlice]
    ) -> MLXArray {
        let result = buffer
        let arrayIndices = indices(for: slices)
        result[arrayIndices] = result[arrayIndices] + values
        return result
    }
}

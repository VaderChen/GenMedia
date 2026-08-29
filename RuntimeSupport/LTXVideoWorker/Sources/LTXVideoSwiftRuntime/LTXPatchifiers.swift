import MLX

public struct LTXVideoLatentPatchifier: Sendable {
    public init() {}

    public static func tokenShape(for latentShape: [Int]) throws -> [Int] {
        guard latentShape.count == 5, latentShape.allSatisfy({ $0 > 0 }) else {
            throw LTXVideoRuntimeError.invalidLatentShape(latentShape)
        }
        return [latentShape[0], latentShape[2] * latentShape[3] * latentShape[4], latentShape[1]]
    }

    public static func restoredShape(
        for tokenShape: [Int],
        dimensions: [Int]
    ) throws -> [Int] {
        guard tokenShape.count == 3, dimensions.count == 3,
              tokenShape.allSatisfy({ $0 > 0 }), dimensions.allSatisfy({ $0 > 0 }),
              tokenShape[1] == dimensions.reduce(1, *) else {
            throw LTXVideoRuntimeError.invalidConfiguration(
                "Video token shape 或 dimensions 無效。"
            )
        }
        return [tokenShape[0], tokenShape[2], dimensions[0], dimensions[1], dimensions[2]]
    }

    public func patchify(_ latent: MLXArray) throws -> (tokens: MLXArray, dimensions: [Int]) {
        guard latent.ndim == 5 else {
            throw LTXVideoRuntimeError.invalidLatentShape(latent.shape)
        }
        let batch = latent.shape[0]
        let dimensions = Array(latent.shape[2...])
        let tokens = latent.transposed(0, 2, 3, 4, 1).reshaped(
            batch,
            dimensions.reduce(1, *) ,
            latent.shape[1]
        )
        return (tokens, dimensions)
    }

    public func unpatchify(_ tokens: MLXArray, dimensions: [Int]) throws -> MLXArray {
        guard tokens.ndim == 3, dimensions.count == 3 else {
            throw LTXVideoRuntimeError.invalidConfiguration("Video patchifier 輸入或 dimensions 無效。")
        }
        let batch = tokens.shape[0]
        let channels = tokens.shape[2]
        let tokenCount = dimensions.reduce(1, *)
        guard tokens.shape[1] == tokenCount else {
            throw LTXVideoRuntimeError.invalidConfiguration("Video token 數量與 latent dimensions 不一致。")
        }
        return tokens.reshaped(batch, dimensions[0], dimensions[1], dimensions[2], channels)
            .transposed(0, 4, 1, 2, 3)
    }
}

public struct LTXAudioLatentPatchifier: Sendable {
    public init() {}

    public static func tokenShape(for latentShape: [Int]) throws -> [Int] {
        guard latentShape.count == 4, latentShape[1] == 8,
              latentShape[2] > 0, latentShape[3] == 16 else {
            throw LTXVideoRuntimeError.invalidConfiguration(
                "Audio latent shape 必須是 [B,8,T,16]。"
            )
        }
        return [latentShape[0], latentShape[2], 128]
    }

    public static func restoredShape(for tokenShape: [Int]) throws -> [Int] {
        guard tokenShape.count == 3, tokenShape.allSatisfy({ $0 > 0 }), tokenShape[2] == 128 else {
            throw LTXVideoRuntimeError.invalidConfiguration(
                "Audio token shape 必須是 [B,T,128]。"
            )
        }
        return [tokenShape[0], 8, tokenShape[1], 16]
    }

    public func patchify(_ latent: MLXArray) throws -> MLXArray {
        guard latent.ndim == 4, latent.shape[1] * latent.shape[3] == 128 else {
            throw LTXVideoRuntimeError.invalidConfiguration("Audio latent 必須是 [B,8,T,16]。")
        }
        return latent.transposed(0, 2, 1, 3).reshaped(
            latent.shape[0], latent.shape[2], latent.shape[1] * latent.shape[3]
        )
    }

    public func unpatchify(_ tokens: MLXArray) throws -> MLXArray {
        guard tokens.ndim == 3, tokens.shape[2] == 128 else {
            throw LTXVideoRuntimeError.invalidConfiguration("Audio tokens 必須是 [B,T,128]。")
        }
        return tokens.reshaped(tokens.shape[0], tokens.shape[1], 8, 16)
            .transposed(0, 2, 1, 3)
    }
}

public enum LTXPositionBuilder {
    public static func videoCoordinates(frameCount: Int, height: Int, width: Int) -> [Int32] {
        var values: [Int32] = []
        values.reserveCapacity(frameCount * height * width * 3)
        for frame in 0..<frameCount {
            for y in 0..<height {
                for x in 0..<width {
                    values.append(Int32(frame))
                    values.append(Int32(y))
                    values.append(Int32(x))
                }
            }
        }
        return values
    }

    public static func video(frameCount: Int, height: Int, width: Int) -> MLXArray {
        MLXArray(videoCoordinates(frameCount: frameCount, height: height, width: width))
            .reshaped(1, frameCount * height * width, 3)
    }

    public static func audioCoordinates(tokenCount: Int) -> [Int32] {
        Array(0..<tokenCount).map(Int32.init)
    }

    public static func audio(tokenCount: Int) -> MLXArray {
        MLXArray(audioCoordinates(tokenCount: tokenCount)).reshaped(1, tokenCount, 1)
    }
}

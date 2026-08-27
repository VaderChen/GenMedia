import Foundation
import MLX

public enum LTXVideoRuntimeError: LocalizedError, Sendable {
    case missingFile(URL)
    case invalidConfiguration(String)
    case invalidLatentShape([Int])
    case invalidTileConfiguration(String)
    case missingWeights([String])
    case unexpectedWeights([String])
    case duplicateWeight(String)
    case weightShapeMismatch(name: String, expected: [Int], actual: [Int])

    public var errorDescription: String? {
        switch self {
        case let .missingFile(url):
            "找不到檔案：\(url.path)"
        case let .invalidConfiguration(message):
            "LTX VAE 設定無效：\(message)"
        case let .invalidLatentShape(shape):
            "LTX VAE latent 必須是 [B,128,F,H,W]，實際為 \(shape)。"
        case let .invalidTileConfiguration(message):
            "LTX VAE tile 設定無效：\(message)"
        case let .missingWeights(keys):
            "LTX VAE 缺少權重：\(keys.prefix(12).joined(separator: "、"))"
        case let .unexpectedWeights(keys):
            "LTX VAE 出現未預期權重：\(keys.prefix(12).joined(separator: "、"))"
        case let .duplicateWeight(key):
            "LTX VAE 權重重複：\(key)"
        case let .weightShapeMismatch(name, expected, actual):
            "LTX VAE 權重 \(name) shape 不一致：預期 \(expected)，實際 \(actual)。"
        }
    }
}

public struct LTXVideoVAEConfiguration: Sendable, Equatable {
    public let latentChannels: Int
    public let patchSize: Int
    public let causalDecoder: Bool
    public let spatialPaddingMode: String

    public init(
        latentChannels: Int = 128,
        patchSize: Int = 4,
        causalDecoder: Bool = false,
        spatialPaddingMode: String = "zeros"
    ) throws {
        guard latentChannels == 128 else {
            throw LTXVideoRuntimeError.invalidConfiguration(
                "Block 1 僅支援 LTX-2.3 的 128 latent channels，實際為 \(latentChannels)。"
            )
        }
        guard patchSize == 4 else {
            throw LTXVideoRuntimeError.invalidConfiguration(
                "Block 1 僅支援 patch_size=4，實際為 \(patchSize)。"
            )
        }
        guard spatialPaddingMode == "zeros" || spatialPaddingMode == "reflect" else {
            throw LTXVideoRuntimeError.invalidConfiguration(
                "spatial_padding_mode 必須是 zeros 或 reflect。"
            )
        }
        self.latentChannels = latentChannels
        self.patchSize = patchSize
        self.causalDecoder = causalDecoder
        self.spatialPaddingMode = spatialPaddingMode
    }

    public static func load(from modelDirectory: URL) throws -> Self {
        let url = modelDirectory.appendingPathComponent("embedded_config.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw LTXVideoRuntimeError.missingFile(url)
        }
        let data = try Data(contentsOf: url)
        let root = try JSONDecoder().decode(EmbeddedConfiguration.self, from: data)
        return try Self(
            latentChannels: root.vae.latentChannels,
            patchSize: root.vae.patchSize,
            causalDecoder: root.vae.causalDecoder,
            spatialPaddingMode: root.vae.spatialPaddingMode
        )
    }

    private struct EmbeddedConfiguration: Decodable {
        let vae: VAE

        struct VAE: Decodable {
            let latentChannels: Int
            let patchSize: Int
            let causalDecoder: Bool
            let spatialPaddingMode: String

            enum CodingKeys: String, CodingKey {
                case latentChannels = "latent_channels"
                case patchSize = "patch_size"
                case causalDecoder = "causal_decoder"
                case spatialPaddingMode = "spatial_padding_mode"
            }
        }
    }
}

public enum LTXVideoComputeDType: String, Sendable {
    case bfloat16
    case float32

    public var mlxDType: DType {
        switch self {
        case .bfloat16: .bfloat16
        case .float32: .float32
        }
    }
}

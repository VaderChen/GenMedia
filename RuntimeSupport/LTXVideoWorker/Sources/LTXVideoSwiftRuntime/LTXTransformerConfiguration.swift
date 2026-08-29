import Foundation
import MLX

public struct LTXTransformerConfiguration: Sendable, Equatable {
    public let numLayers: Int
    public let videoDim: Int
    public let audioDim: Int
    public let videoNumHeads: Int
    public let audioNumHeads: Int
    public let videoHeadDim: Int
    public let audioHeadDim: Int
    public let avCrossNumHeads: Int
    public let avCrossHeadDim: Int
    public let videoPatchChannels: Int
    public let audioPatchChannels: Int
    public let ffMult: Float
    public let timestepEmbeddingDim: Int
    public let timestepScaleMultiplier: Float
    public let avCATimestepScaleMultiplier: Float
    public let ropeTheta: Float
    public let ropeType: LTXRoPEType
    public let positionalEmbeddingMaxPos: [Int]
    public let audioPositionalEmbeddingMaxPos: [Int]
    public let normEps: Float

    public init(
        numLayers: Int = 48,
        videoDim: Int = 4096,
        audioDim: Int = 2048,
        videoNumHeads: Int = 32,
        audioNumHeads: Int = 32,
        videoHeadDim: Int = 128,
        audioHeadDim: Int = 64,
        avCrossNumHeads: Int = 32,
        avCrossHeadDim: Int = 64,
        videoPatchChannels: Int = 128,
        audioPatchChannels: Int = 128,
        ffMult: Float = 4,
        timestepEmbeddingDim: Int = 256,
        timestepScaleMultiplier: Float = 1000,
        avCATimestepScaleMultiplier: Float = 1000,
        ropeTheta: Float = 10000,
        ropeType: LTXRoPEType = .split,
        positionalEmbeddingMaxPos: [Int] = [20, 2048, 2048],
        audioPositionalEmbeddingMaxPos: [Int] = [20],
        normEps: Float = 1e-6
    ) throws {
        guard numLayers > 0,
              videoDim > 0,
              audioDim > 0,
              videoNumHeads > 0,
              audioNumHeads > 0,
              videoHeadDim > 0,
              audioHeadDim > 0,
              avCrossNumHeads > 0,
              avCrossHeadDim > 0,
              videoPatchChannels > 0,
              audioPatchChannels > 0,
              timestepEmbeddingDim > 0,
              positionalEmbeddingMaxPos.count == 3,
              audioPositionalEmbeddingMaxPos.count == 1 else {
            throw LTXVideoRuntimeError.invalidConfiguration("Transformer 維度或位置設定無效。")
        }
        let videoFrequencyCount = videoNumHeads * videoHeadDim
            / (2 * positionalEmbeddingMaxPos.count)
        let audioFrequencyCount = audioNumHeads * audioHeadDim
            / (2 * audioPositionalEmbeddingMaxPos.count)
        guard videoFrequencyCount > 0,
              audioFrequencyCount > 0,
              avCrossNumHeads * avCrossHeadDim % 2 == 0 else {
            throw LTXVideoRuntimeError.invalidConfiguration("Transformer attention 維度無法建立 RoPE。")
        }
        guard normEps > 0, ropeTheta > 1, ffMult > 0 else {
            throw LTXVideoRuntimeError.invalidConfiguration("Transformer norm、RoPE 或 FFN 設定無效。")
        }

        self.numLayers = numLayers
        self.videoDim = videoDim
        self.audioDim = audioDim
        self.videoNumHeads = videoNumHeads
        self.audioNumHeads = audioNumHeads
        self.videoHeadDim = videoHeadDim
        self.audioHeadDim = audioHeadDim
        self.avCrossNumHeads = avCrossNumHeads
        self.avCrossHeadDim = avCrossHeadDim
        self.videoPatchChannels = videoPatchChannels
        self.audioPatchChannels = audioPatchChannels
        self.ffMult = ffMult
        self.timestepEmbeddingDim = timestepEmbeddingDim
        self.timestepScaleMultiplier = timestepScaleMultiplier
        self.avCATimestepScaleMultiplier = avCATimestepScaleMultiplier
        self.ropeTheta = ropeTheta
        self.ropeType = ropeType
        self.positionalEmbeddingMaxPos = positionalEmbeddingMaxPos
        self.audioPositionalEmbeddingMaxPos = audioPositionalEmbeddingMaxPos
        self.normEps = normEps
    }

    public static func load(from modelDirectory: URL) throws -> Self {
        let candidates = [
            modelDirectory.appendingPathComponent("embedded_config.json"),
            modelDirectory.appendingPathComponent("config.json")
        ]
        guard let url = candidates.first(where: { FileManager.default.fileExists(atPath: $0.path) }) else {
            throw LTXVideoRuntimeError.missingFile(candidates[0])
        }
        let root = try JSONDecoder().decode(JSONValue.self, from: Data(contentsOf: url))
        let transformer = root["transformer"] ?? root
        return try Self(
            numLayers: transformer.int("num_layers", default: 48),
            videoDim: transformer.int("cross_attention_dim", default: 4096),
            audioDim: transformer.int("audio_cross_attention_dim", default: 2048),
            videoNumHeads: transformer.int("num_attention_heads", default: 32),
            audioNumHeads: transformer.int("audio_num_attention_heads", default: 32),
            videoHeadDim: transformer.int("attention_head_dim", default: 128),
            audioHeadDim: transformer.int("audio_attention_head_dim", default: 64),
            avCrossNumHeads: transformer.int("audio_num_attention_heads", default: 32),
            avCrossHeadDim: transformer.int("audio_attention_head_dim", default: 64),
            videoPatchChannels: transformer.int("in_channels", default: 128),
            audioPatchChannels: transformer.int("audio_in_channels", default: 128),
            timestepScaleMultiplier: transformer.float("timestep_scale_multiplier", default: 1000),
            avCATimestepScaleMultiplier: transformer.float("av_ca_timestep_scale_multiplier", default: 1000),
            ropeTheta: transformer.float("positional_embedding_theta", default: 10000),
            ropeType: LTXRoPEType(rawValue: transformer.string("rope_type", default: "split")) ?? .split,
            positionalEmbeddingMaxPos: transformer.ints("positional_embedding_max_pos", default: [20, 2048, 2048]),
            audioPositionalEmbeddingMaxPos: transformer.ints("audio_positional_embedding_max_pos", default: [20]),
            normEps: transformer.float("norm_eps", default: 1e-6)
        )
    }
}

private indirect enum JSONValue: Decodable {
    case object([String: JSONValue])
    case array([JSONValue])
    case string(String)
    case number(Double)
    case bool(Bool)
    case null

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() { self = .null }
        else if let value = try? container.decode([String: JSONValue].self) { self = .object(value) }
        else if let value = try? container.decode([JSONValue].self) { self = .array(value) }
        else if let value = try? container.decode(String.self) { self = .string(value) }
        else if let value = try? container.decode(Double.self) { self = .number(value) }
        else { self = .bool(try container.decode(Bool.self)) }
    }

    subscript(key: String) -> JSONValue? {
        guard case let .object(values) = self else { return nil }
        return values[key]
    }

    func int(_ key: String, default value: Int) -> Int {
        guard case let .number(number) = self[key] else { return value }
        return Int(number)
    }

    func float(_ key: String, default value: Float) -> Float {
        guard case let .number(number) = self[key] else { return value }
        return Float(number)
    }

    func string(_ key: String, default value: String) -> String {
        guard case let .string(string) = self[key] else { return value }
        return string
    }

    func ints(_ key: String, default value: [Int]) -> [Int] {
        guard case let .array(values) = self[key] else { return value }
        return values.compactMap { value in
            guard case let .number(number) = value else { return nil }
            return Int(number)
        }
    }
}

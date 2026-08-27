import Foundation
import MLX
import MLXNN

public struct MiniMaxMusic3ConditionEncoderConfiguration: Equatable, Sendable {
    public let conditionHiddenDimension: Int
    public let numberOfConditionLayers: Int
    public let outputDimension: Int
    public let inputSamplingRate: Int
    public let inputHopLength: Int
    public let outputSamplingRate: Int
    public let outputHopLength: Int

    public init(
        conditionHiddenDimension: Int = 4_096,
        numberOfConditionLayers: Int = 8,
        outputDimension: Int = 2_048,
        inputSamplingRate: Int = 24_000,
        inputHopLength: Int = 960,
        outputSamplingRate: Int = 44_100,
        outputHopLength: Int = 512
    ) {
        self.conditionHiddenDimension = conditionHiddenDimension
        self.numberOfConditionLayers = numberOfConditionLayers
        self.outputDimension = outputDimension
        self.inputSamplingRate = inputSamplingRate
        self.inputHopLength = inputHopLength
        self.outputSamplingRate = outputSamplingRate
        self.outputHopLength = outputHopLength
    }

    public static let music3 = Self()

    public func outputLength(for numberOfFrames: Int) throws -> Int {
        guard numberOfFrames > 0 else {
            throw MiniMaxMusic3ConditionEncoderError.invalidInput(
                "輸入 frame 數必須是正整數。"
            )
        }
        let scaled = Double(numberOfFrames)
            * Double(outputSamplingRate) / Double(inputSamplingRate)
            * Double(inputHopLength) / Double(outputHopLength)
        return max(1, Int(scaled))
    }

    public func validate() throws {
        guard conditionHiddenDimension > 0,
              numberOfConditionLayers > 0,
              outputDimension > 0,
              inputSamplingRate > 0,
              inputHopLength > 0,
              outputSamplingRate > 0,
              outputHopLength > 0 else {
            throw MiniMaxMusic3ConditionEncoderError.invalidConfiguration
        }
    }

    public static func load(from modelDirectory: URL) throws -> Self {
        let url = modelDirectory.appendingPathComponent("config.json")
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw MiniMaxMusic3ConditionEncoderError.missingFile(url)
        }
        do {
            let source = try JSONDecoder().decode(ModelConfigurationFile.self, from: data)
            let configuration = Self(
                conditionHiddenDimension: source.conditionHiddenDimension,
                numberOfConditionLayers: source.numberOfConditionLayers,
                outputDimension: source.outputDimension,
                inputSamplingRate: source.inputSamplingRate,
                inputHopLength: source.inputHopLength,
                outputSamplingRate: source.outputSamplingRate,
                outputHopLength: source.outputHopLength
            )
            try configuration.validate()
            return configuration
        } catch let error as MiniMaxMusic3ConditionEncoderError {
            throw error
        } catch {
            throw MiniMaxMusic3ConditionEncoderError.invalidConfigurationFile(url)
        }
    }

    private struct ModelConfigurationFile: Decodable {
        let conditionHiddenDimension: Int
        let numberOfConditionLayers: Int
        let outputDimension: Int
        let inputSamplingRate: Int
        let inputHopLength: Int
        let outputSamplingRate: Int
        let outputHopLength: Int

        enum CodingKeys: String, CodingKey {
            case conditionHiddenDimension = "hidden_size"
            case numberOfConditionLayers = "num_condition_layers"
            case outputDimension = "condition_out_dim"
            case inputSamplingRate = "input_sampling_rate"
            case inputHopLength = "input_hop_length"
            case outputSamplingRate = "output_sampling_rate"
            case outputHopLength = "output_hop_length"
        }
    }
}

public enum MiniMaxMusic3ConditionEncoderError: LocalizedError, Sendable {
    case invalidConfiguration
    case invalidConfigurationFile(URL)
    case missingFile(URL)
    case invalidInput(String)
    case missingWeights([String])
    case unexpectedWeights([String])
    case weightShapeMismatch(name: String, expected: [Int], actual: [Int])

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration:
            "MiniMax Music 3 condition encoder 設定無效。"
        case let .invalidConfigurationFile(url):
            "無法解析 condition encoder 設定：\(url.path)"
        case let .missingFile(url):
            "缺少 condition encoder 必要檔案：\(url.path)"
        case let .invalidInput(message):
            "condition encoder 輸入無效：\(message)"
        case let .missingWeights(names):
            "condition encoder 缺少權重：\(names.joined(separator: ", "))"
        case let .unexpectedWeights(names):
            "condition encoder 收到重複或未使用權重：\(names.joined(separator: ", "))"
        case let .weightShapeMismatch(name, expected, actual):
            "condition encoder 權重 \(name) 形狀不符，預期 \(expected)，實際 \(actual)。"
        }
    }
}

public func miniMaxMusic3NearestResize1D(
    _ hiddenStates: MLXArray,
    outputLength: Int
) throws -> MLXArray {
    guard hiddenStates.ndim == 3 else {
        throw MiniMaxMusic3ConditionEncoderError.invalidInput(
            "hidden states 必須是 [batch, length, channels]。"
        )
    }
    let inputLength = hiddenStates.shape[1]
    guard inputLength > 0, outputLength > 0 else {
        throw MiniMaxMusic3ConditionEncoderError.invalidInput(
            "輸入與輸出 length 必須是正整數。"
        )
    }
    guard inputLength != outputLength else { return hiddenStates }

    let indices = floor(
        MLXArray.arange(outputLength, dtype: .float32)
            * Float(inputLength) / Float(outputLength)
    ).asType(.int32)
    return hiddenStates.take(indices, axis: 1)
}

public final class MiniMaxMusic3ConditionEncoder: Module {
    public let configuration: MiniMaxMusic3ConditionEncoderConfiguration

    @ParameterInfo(key: "layer_weight_logits") private var layerWeightLogits: MLXArray
    @ParameterInfo(key: "layer_scale") private var layerScale: MLXArray
    @ModuleInfo(key: "proj") private var projection: Conv1d

    public init(
        configuration: MiniMaxMusic3ConditionEncoderConfiguration = .music3
    ) {
        precondition((try? configuration.validate()) != nil)
        self.configuration = configuration
        self._layerWeightLogits.wrappedValue = MLXArray.zeros(
            [configuration.numberOfConditionLayers]
        )
        self._layerScale.wrappedValue = MLXArray.ones([1])
        self._projection.wrappedValue = Conv1d(
            inputChannels: configuration.conditionHiddenDimension,
            outputChannels: configuration.outputDimension,
            kernelSize: 3,
            padding: 1
        )
        super.init()
    }

    public func callAsFunction(_ hiddenStates: MLXArray) throws -> MLXArray {
        guard hiddenStates.ndim == 3 else {
            throw MiniMaxMusic3ConditionEncoderError.invalidInput(
                "hidden states 必須是 [batch, frames, features]。"
            )
        }
        let batch = hiddenStates.shape[0]
        let frames = hiddenStates.shape[1]
        let features = hiddenStates.shape[2]
        let expectedFeatures = configuration.numberOfConditionLayers
            * configuration.conditionHiddenDimension
        guard features == expectedFeatures else {
            throw MiniMaxMusic3ConditionEncoderError.invalidInput(
                "預期 \(expectedFeatures) 個 hidden features，實際為 \(features)。"
            )
        }

        let grouped = hiddenStates.reshaped(
            batch,
            frames,
            configuration.numberOfConditionLayers,
            configuration.conditionHiddenDimension
        )
        let weights = softmax(layerWeightLogits, axis: 0).asType(grouped.dtype)
        let mixed = (grouped * weights[.newAxis, .newAxis, 0..., .newAxis])
            .sum(axis: 2)
        let projected = projection(mixed * layerScale.asType(mixed.dtype))
        let outputLength = try configuration.outputLength(for: frames)
        return try miniMaxMusic3NearestResize1D(projected, outputLength: outputLength)
    }
}

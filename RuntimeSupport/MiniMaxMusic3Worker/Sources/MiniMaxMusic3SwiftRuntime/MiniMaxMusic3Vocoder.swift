import Foundation
import MLX
import MLXNN

public struct MiniMaxMusic3VocoderConfiguration: Equatable, Sendable {
    public let latentChannels: Int
    public let decoderInputDimension: Int
    public let decoderHiddenDimension: Int
    public let upsamplingRatios: [Int]
    public let samplingRate: Int

    public init(
        latentChannels: Int,
        decoderInputDimension: Int,
        decoderHiddenDimension: Int,
        upsamplingRatios: [Int],
        samplingRate: Int
    ) {
        self.latentChannels = latentChannels
        self.decoderInputDimension = decoderInputDimension
        self.decoderHiddenDimension = decoderHiddenDimension
        self.upsamplingRatios = upsamplingRatios
        self.samplingRate = samplingRate
    }

    public static let music3 = MiniMaxMusic3VocoderConfiguration(
        latentChannels: 128,
        decoderInputDimension: 1024,
        decoderHiddenDimension: 1536,
        upsamplingRatios: [8, 8, 4, 2],
        samplingRate: 44_100
    )

    public var hopLength: Int {
        upsamplingRatios.reduce(1, *)
    }

    public func validate() throws {
        guard latentChannels >= 2, latentChannels.isMultiple(of: 2) else {
            throw MiniMaxMusic3VocoderError.invalidConfiguration(
                "latentChannels 必須是大於 1 的偶數。"
            )
        }
        guard decoderInputDimension > 0, decoderHiddenDimension > 0 else {
            throw MiniMaxMusic3VocoderError.invalidConfiguration(
                "Vocoder 維度必須是正整數。"
            )
        }
        guard !upsamplingRatios.isEmpty, upsamplingRatios.allSatisfy({ $0 > 0 }) else {
            throw MiniMaxMusic3VocoderError.invalidConfiguration(
                "upsamplingRatios 必須至少包含一個正整數。"
            )
        }
        let divisor = 1 << upsamplingRatios.count
        guard decoderHiddenDimension.isMultiple(of: divisor) else {
            throw MiniMaxMusic3VocoderError.invalidConfiguration(
                "decoderHiddenDimension 必須可被上採樣區塊數的 2 次方整除。"
            )
        }
        guard samplingRate > 0 else {
            throw MiniMaxMusic3VocoderError.invalidConfiguration(
                "samplingRate 必須是正整數。"
            )
        }
    }

    public static func load(from modelDirectory: URL) throws -> Self {
        let configURL = modelDirectory.appendingPathComponent("config.json")
        let data: Data
        do {
            data = try Data(contentsOf: configURL)
        } catch {
            throw MiniMaxMusic3VocoderError.missingFile(configURL)
        }
        let source: ModelConfigurationFile
        do {
            source = try JSONDecoder().decode(ModelConfigurationFile.self, from: data)
        } catch {
            throw MiniMaxMusic3VocoderError.invalidConfigurationFile(configURL)
        }
        let configuration = Self(
            latentChannels: source.latentChannels,
            decoderInputDimension: source.decoderInputDimension,
            decoderHiddenDimension: source.decoderHiddenDimension,
            upsamplingRatios: source.upsamplingRatios,
            samplingRate: source.outputSamplingRate ?? source.sampleRate ?? 44_100
        )
        try configuration.validate()
        return configuration
    }

    private struct ModelConfigurationFile: Decodable {
        let latentChannels: Int
        let decoderInputDimension: Int
        let decoderHiddenDimension: Int
        let upsamplingRatios: [Int]
        let outputSamplingRate: Int?
        let sampleRate: Int?

        enum CodingKeys: String, CodingKey {
            case latentChannels = "dit_in_channels"
            case decoderInputDimension = "vocoder_input_dim"
            case decoderHiddenDimension = "vocoder_hidden_dim"
            case upsamplingRatios = "vocoder_upsampling_ratios"
            case outputSamplingRate = "output_sampling_rate"
            case sampleRate = "sample_rate"
        }
    }
}

public enum MiniMaxMusic3VocoderError: LocalizedError, Sendable {
    case invalidConfiguration(String)
    case invalidConfigurationFile(URL)
    case missingFile(URL)
    case invalidLatentShape([Int])
    case emptyChunks
    case invalidChunk(Int)
    case missingWeights([String])
    case unexpectedWeights([String])
    case weightShapeMismatch(name: String, expected: [Int], actual: [Int])

    public var errorDescription: String? {
        switch self {
        case let .invalidConfiguration(message):
            "MiniMax Music 3 Vocoder 設定無效：\(message)"
        case let .invalidConfigurationFile(url):
            "無法解析模型設定：\(url.path)"
        case let .missingFile(url):
            "缺少必要檔案：\(url.path)"
        case let .invalidLatentShape(shape):
            "latent 必須是 [batch, 128, length]，目前為 \(shape)。"
        case .emptyChunks:
            "latent chunks 不可為空。"
        case let .invalidChunk(index):
            "latent chunk 索引無效：\(index)。"
        case let .missingWeights(names):
            "Vocoder 缺少權重：\(names.joined(separator: ", "))"
        case let .unexpectedWeights(names):
            "Vocoder 收到未使用的權重：\(names.joined(separator: ", "))"
        case let .weightShapeMismatch(name, expected, actual):
            "Vocoder 權重 \(name) 形狀不符，預期 \(expected)，實際 \(actual)。"
        }
    }
}

private final class MiniMaxMusic3Snake1d: Module {
    @ParameterInfo(key: "alpha") private var alpha: MLXArray

    init(channelCount: Int) {
        self._alpha.wrappedValue = MLXArray.ones([1, 1, channelCount])
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let alpha = self.alpha.asType(input.dtype)
        return input + (alpha * input).sin().square() / (alpha + 1e-9)
    }
}

private final class MiniMaxMusic3VocoderResidualUnit: Module {
    @ModuleInfo(key: "snake1") private var snake1: MiniMaxMusic3Snake1d
    @ModuleInfo(key: "conv1") private var conv1: Conv1d
    @ModuleInfo(key: "snake2") private var snake2: MiniMaxMusic3Snake1d
    @ModuleInfo(key: "conv2") private var conv2: Conv1d

    init(channelCount: Int, dilation: Int) {
        self._snake1.wrappedValue = MiniMaxMusic3Snake1d(channelCount: channelCount)
        self._conv1.wrappedValue = Conv1d(
            inputChannels: channelCount,
            outputChannels: channelCount,
            kernelSize: 7,
            padding: (7 - 1) * dilation / 2,
            dilation: dilation
        )
        self._snake2.wrappedValue = MiniMaxMusic3Snake1d(channelCount: channelCount)
        self._conv2.wrappedValue = Conv1d(
            inputChannels: channelCount,
            outputChannels: channelCount,
            kernelSize: 1
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var output = conv1(snake1(input))
        output = conv2(snake2(output))
        return input + output
    }
}

private final class MiniMaxMusic3VocoderBlock: Module {
    @ModuleInfo(key: "snake1") private var snake1: MiniMaxMusic3Snake1d
    @ModuleInfo(key: "conv_t1") private var convT1: ConvTransposed1d
    @ModuleInfo(key: "res_unit1") private var residualUnit1: MiniMaxMusic3VocoderResidualUnit
    @ModuleInfo(key: "res_unit2") private var residualUnit2: MiniMaxMusic3VocoderResidualUnit
    @ModuleInfo(key: "res_unit3") private var residualUnit3: MiniMaxMusic3VocoderResidualUnit

    init(inputDimension: Int, outputDimension: Int, stride: Int) {
        self._snake1.wrappedValue = MiniMaxMusic3Snake1d(channelCount: inputDimension)
        self._convT1.wrappedValue = ConvTransposed1d(
            inputChannels: inputDimension,
            outputChannels: outputDimension,
            kernelSize: 2 * stride,
            stride: stride,
            padding: Int(ceil(Double(stride) / 2))
        )
        self._residualUnit1.wrappedValue = MiniMaxMusic3VocoderResidualUnit(
            channelCount: outputDimension,
            dilation: 1
        )
        self._residualUnit2.wrappedValue = MiniMaxMusic3VocoderResidualUnit(
            channelCount: outputDimension,
            dilation: 3
        )
        self._residualUnit3.wrappedValue = MiniMaxMusic3VocoderResidualUnit(
            channelCount: outputDimension,
            dilation: 9
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var output = convT1(snake1(input))
        output = residualUnit1(output)
        output = residualUnit2(output)
        return residualUnit3(output)
    }
}

public final class MiniMaxMusic3Vocoder: Module {
    public let configuration: MiniMaxMusic3VocoderConfiguration

    @ModuleInfo(key: "dec_in_proj") private var inputProjection: Conv1d
    @ModuleInfo(key: "conv_in") private var inputConvolution: Conv1d
    @ModuleInfo(key: "blocks") private var blocks: [MiniMaxMusic3VocoderBlock]
    @ModuleInfo(key: "snake_out") private var outputSnake: MiniMaxMusic3Snake1d
    @ModuleInfo(key: "conv_out") private var outputConvolution: Conv1d

    public init(configuration: MiniMaxMusic3VocoderConfiguration) {
        precondition((try? configuration.validate()) != nil)
        self.configuration = configuration
        self._inputProjection.wrappedValue = Conv1d(
            inputChannels: configuration.latentChannels / 2,
            outputChannels: configuration.decoderInputDimension,
            kernelSize: 1
        )
        self._inputConvolution.wrappedValue = Conv1d(
            inputChannels: configuration.decoderInputDimension,
            outputChannels: configuration.decoderHiddenDimension,
            kernelSize: 7,
            padding: 3
        )
        var outputDimension = configuration.decoderHiddenDimension
        var configuredBlocks: [MiniMaxMusic3VocoderBlock] = []
        for (index, stride) in configuration.upsamplingRatios.enumerated() {
            let inputDimension = configuration.decoderHiddenDimension / (1 << index)
            outputDimension = configuration.decoderHiddenDimension / (1 << (index + 1))
            configuredBlocks.append(
                MiniMaxMusic3VocoderBlock(
                    inputDimension: inputDimension,
                    outputDimension: outputDimension,
                    stride: stride
                )
            )
        }
        self._blocks.wrappedValue = configuredBlocks
        self._outputSnake.wrappedValue = MiniMaxMusic3Snake1d(channelCount: outputDimension)
        self._outputConvolution.wrappedValue = Conv1d(
            inputChannels: outputDimension,
            outputChannels: 1,
            kernelSize: 7,
            padding: 3
        )
        super.init()
    }

    public func callAsFunction(_ latents: MLXArray) -> MLXArray {
        precondition(
            latents.ndim == 3,
            "MiniMaxMusic3Vocoder 輸入必須是三維 NLC latent。"
        )
        let batch = latents.shape[0]
        let length = latents.shape[1]
        precondition(
            latents.shape[2] == configuration.latentChannels,
            "MiniMaxMusic3Vocoder latent channel 數不符。"
        )
        let halfChannels = configuration.latentChannels / 2
        var hidden = latents
            .reshaped(batch, length, 2, halfChannels)
            .transposed(0, 2, 1, 3)
            .reshaped(batch * 2, length, halfChannels)
        hidden = inputConvolution(inputProjection(hidden))
        for block in blocks {
            hidden = block(hidden)
        }
        hidden = MLX.tanh(outputConvolution(outputSnake(hidden)))
        return hidden.reshaped(batch, 2, hidden.shape[1])
    }
}

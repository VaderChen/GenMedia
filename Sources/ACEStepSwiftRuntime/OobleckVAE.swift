import MLX
import MLXNN
import Foundation

struct OobleckVAEConfiguration: Decodable {
    let audioChannels: Int
    let channelMultiples: [Int]
    let decoderChannels: Int
    let decoderInputChannels: Int
    let downsamplingRatios: [Int]
    let encoderHiddenSize: Int
    let samplingRate: Int

    enum CodingKeys: String, CodingKey {
        case audioChannels = "audio_channels"
        case channelMultiples = "channel_multiples"
        case decoderChannels = "decoder_channels"
        case decoderInputChannels = "decoder_input_channels"
        case downsamplingRatios = "downsampling_ratios"
        case encoderHiddenSize = "encoder_hidden_size"
        case samplingRate = "sampling_rate"
    }

    static func load(from url: URL) throws -> OobleckVAEConfiguration {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(OobleckVAEConfiguration.self, from: data)
    }

    var hopLength: Int {
        downsamplingRatios.reduce(1, *)
    }
}

final class OobleckSnake1D: Module {
    @ParameterInfo(key: "alpha") private var alpha: MLXArray
    @ParameterInfo(key: "beta") private var beta: MLXArray

    private let usesLogScale: Bool

    init(channelCount: Int, usesLogScale: Bool = true) {
        self._alpha.wrappedValue = MLXArray.zeros([channelCount])
        self._beta.wrappedValue = MLXArray.zeros([channelCount])
        self.usesLogScale = usesLogScale
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let inputFloat = input.asType(.float32)
        let alphaFloat = alpha.asType(.float32)
        let betaFloat = beta.asType(.float32)
        let alphaScale = usesLogScale ? alphaFloat.exp() : alphaFloat
        let betaScale = usesLogScale ? betaFloat.exp() : betaFloat
        let periodic = (alphaScale * inputFloat).sin().square()
        let output = inputFloat + (betaScale + 1e-9).reciprocal() * periodic
        return output.asType(input.dtype)
    }
}

final class OobleckResidualUnit: Module {
    @ModuleInfo(key: "snake1") private var snake1: OobleckSnake1D
    @ModuleInfo(key: "conv1") private var conv1: Conv1d
    @ModuleInfo(key: "snake2") private var snake2: OobleckSnake1D
    @ModuleInfo(key: "conv2") private var conv2: Conv1d

    init(channelCount: Int, dilation: Int) {
        let padding = ((7 - 1) * dilation) / 2
        self._snake1.wrappedValue = OobleckSnake1D(channelCount: channelCount)
        self._conv1.wrappedValue = Conv1d(
            inputChannels: channelCount,
            outputChannels: channelCount,
            kernelSize: 7,
            padding: padding,
            dilation: dilation
        )
        self._snake2.wrappedValue = OobleckSnake1D(channelCount: channelCount)
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
        let lengthDifference = input.shape[1] - output.shape[1]
        guard lengthDifference > 0 else { return input + output }
        let leadingTrim = lengthDifference / 2
        let trailingIndex = input.shape[1] - (lengthDifference - leadingTrim)
        return input[0..., leadingTrim..<trailingIndex, 0...] + output
    }
}

final class OobleckDecoderBlock: Module {
    @ModuleInfo(key: "snake1") private var snake1: OobleckSnake1D
    @ModuleInfo(key: "conv_t1") private var transposedConvolution: ConvTransposed1d
    @ModuleInfo(key: "res_unit1") private var residualUnit1: OobleckResidualUnit
    @ModuleInfo(key: "res_unit2") private var residualUnit2: OobleckResidualUnit
    @ModuleInfo(key: "res_unit3") private var residualUnit3: OobleckResidualUnit

    init(inputChannels: Int, outputChannels: Int, stride: Int) {
        self._snake1.wrappedValue = OobleckSnake1D(channelCount: inputChannels)
        self._transposedConvolution.wrappedValue = ConvTransposed1d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: 2 * stride,
            stride: stride,
            padding: Int(ceil(Double(stride) / 2))
        )
        self._residualUnit1.wrappedValue = OobleckResidualUnit(
            channelCount: outputChannels,
            dilation: 1
        )
        self._residualUnit2.wrappedValue = OobleckResidualUnit(
            channelCount: outputChannels,
            dilation: 3
        )
        self._residualUnit3.wrappedValue = OobleckResidualUnit(
            channelCount: outputChannels,
            dilation: 9
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var hiddenState = transposedConvolution(snake1(input))
        hiddenState = residualUnit1(hiddenState)
        hiddenState = residualUnit2(hiddenState)
        hiddenState = residualUnit3(hiddenState)
        return hiddenState
    }
}

final class OobleckDecoder: Module {
    @ModuleInfo(key: "conv1") private var inputConvolution: Conv1d
    @ModuleInfo(key: "block") private var blocks: [OobleckDecoderBlock]
    @ModuleInfo(key: "snake1") private var outputActivation: OobleckSnake1D
    @ModuleInfo(key: "conv2") private var outputConvolution: Conv1d

    init(configuration: OobleckVAEConfiguration) {
        let channelMultiples = [1] + configuration.channelMultiples
        let strides = configuration.downsamplingRatios.reversed()
        self._inputConvolution.wrappedValue = Conv1d(
            inputChannels: configuration.decoderInputChannels,
            outputChannels: configuration.decoderChannels * channelMultiples.last!,
            kernelSize: 7,
            padding: 3
        )
        self._blocks.wrappedValue = strides.enumerated().map { index, stride in
            OobleckDecoderBlock(
                inputChannels: configuration.decoderChannels
                    * channelMultiples[configuration.downsamplingRatios.count - index],
                outputChannels: configuration.decoderChannels
                    * channelMultiples[configuration.downsamplingRatios.count - index - 1],
                stride: stride
            )
        }
        self._outputActivation.wrappedValue = OobleckSnake1D(
            channelCount: configuration.decoderChannels
        )
        self._outputConvolution.wrappedValue = Conv1d(
            inputChannels: configuration.decoderChannels,
            outputChannels: configuration.audioChannels,
            kernelSize: 7,
            padding: 3,
            bias: false
        )
        super.init()
    }

    var dtype: DType {
        inputConvolution.weight.dtype
    }

    func callAsFunction(_ latents: MLXArray) -> MLXArray {
        var hiddenState = inputConvolution(latents)
        for block in blocks {
            hiddenState = block(hiddenState)
        }
        return outputConvolution(outputActivation(hiddenState))
    }
}

final class OobleckVAEDecoder: Module {
    @ModuleInfo(key: "decoder") private var decoder: OobleckDecoder

    init(configuration: OobleckVAEConfiguration) {
        self._decoder.wrappedValue = OobleckDecoder(configuration: configuration)
        super.init()
    }

    var dtype: DType {
        decoder.dtype
    }

    func decode(_ latents: MLXArray) -> MLXArray {
        decoder(latents)
    }
}

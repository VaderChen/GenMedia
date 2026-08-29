import Foundation
import MLX
import MLXNN

public final class LTXHannSincResampler {
    public let upsampleFactor: Int
    public let kernel: MLXArray

    private let pad: Int
    private let padLeft: Int
    private let padRight: Int

    public init(upsampleFactor: Int = 3) {
        precondition(upsampleFactor > 0)
        self.upsampleFactor = upsampleFactor
        let rolloff = 0.99
        let lowpassFilterWidth = 6
        let width = Int(Foundation.ceil(Double(lowpassFilterWidth) / rolloff))
        let kernelSize = 2 * width * upsampleFactor + 1
        let values = (0..<kernelSize).map { index -> Float in
            let time = (Double(index) / Double(upsampleFactor) - Double(width)) * rolloff
            let clamped = min(Double(lowpassFilterWidth), max(-Double(lowpassFilterWidth), time))
            let window = Foundation.pow(
                Foundation.cos(clamped * Double.pi / Double(lowpassFilterWidth) / 2),
                2
            )
            let sinc = abs(time) < 1e-12 ? 1 : Foundation.sin(Double.pi * time) / (Double.pi * time)
            return Float(sinc * window * rolloff / Double(upsampleFactor))
        }
        self.kernel = MLXArray(values).reshaped(kernelSize, 1)
        self.pad = width
        self.padLeft = 2 * width * upsampleFactor
        self.padRight = kernelSize - upsampleFactor
    }

    public func callAsFunction(_ input: MLXArray) throws -> MLXArray {
        guard input.ndim == 2, input.shape[1] > 0 else {
            throw LTXVideoRuntimeError.invalidLatentShape(input.shape)
        }
        let batch = input.shape[0]
        let length = input.shape[1]
        let first = MLX.repeated(input[0..., 0..<1], count: pad, axis: 1)
        let last = MLX.repeated(input[0..., (length - 1)..<length], count: pad, axis: 1)
        let padded = MLX.concatenated([first, input, last], axis: 1)
        let paddedLength = padded.shape[1]
        let zeroInsertedLength = (paddedLength - 1) * upsampleFactor + 1
        var zeroInserted = MLXArray.zeros([batch, zeroInsertedLength], dtype: input.dtype)
        zeroInserted[0..., .stride(from: 0, to: zeroInsertedLength, by: upsampleFactor)] = padded
        var value = zeroInserted.reshaped(batch, zeroInsertedLength, 1)
        let filter = kernel.reshaped(1, kernel.shape[0], 1).asType(input.dtype)
        value = MLX.padded(
            value,
            widths: [(0, 0), (kernel.shape[0] - 1, kernel.shape[0] - 1), (0, 0)].map { .init($0) }
        )
        value = MLX.conv1d(value, filter).squeezed()
        value = value * Float(upsampleFactor)
        return value[0..., padLeft..<(value.shape[1] - padRight)][0..., 0..<(length * upsampleFactor)]
    }
}

public final class LTXSTFTFunction: Module {
    @ParameterInfo(key: "forward_basis") public var forwardBasis: MLXArray
    @ParameterInfo(key: "inverse_basis") public var inverseBasis: MLXArray

    public let fftSize: Int

    public init(fftSize: Int = 512) {
        self.fftSize = fftSize
        self._forwardBasis = ParameterInfo(
            wrappedValue: MLXArray.zeros([fftSize + 2, fftSize, 1], dtype: .float32),
            key: "forward_basis"
        )
        self._inverseBasis = ParameterInfo(
            wrappedValue: MLXArray.zeros([fftSize + 2, fftSize, 1], dtype: .float32),
            key: "inverse_basis"
        )
        super.init()
    }
}

public final class LTXMelSTFT: Module {
    @ParameterInfo(key: "mel_basis") public var melBasis: MLXArray
    @ModuleInfo(key: "stft_fn") public var stft: LTXSTFTFunction

    public let fftSize: Int
    public let hopLength: Int
    public let melCount: Int

    public init(fftSize: Int = 512, hopLength: Int = 80, melCount: Int = 64) {
        self.fftSize = fftSize
        self.hopLength = hopLength
        self.melCount = melCount
        self._melBasis = ParameterInfo(
            wrappedValue: MLXArray.zeros([melCount, fftSize / 2 + 1], dtype: .float32),
            key: "mel_basis"
        )
        self._stft = ModuleInfo(wrappedValue: LTXSTFTFunction(fftSize: fftSize), key: "stft_fn")
        super.init()
    }

    public func callAsFunction(_ waveform: MLXArray) throws -> MLXArray {
        guard waveform.ndim == 2, waveform.shape[1] > 0 else {
            throw LTXVideoRuntimeError.invalidLatentShape(waveform.shape)
        }
        let leftPadding = max(0, fftSize - hopLength)
        let input = MLX.padded(
            waveform[0..., 0..., .newAxis],
            widths: [(0, 0), (leftPadding, 0), (0, 0)].map { .init($0) }
        )
        let stft = MLX.conv1d(input, stft.forwardBasis, stride: hopLength)
        let frequencyCount = fftSize / 2 + 1
        let real = stft[0..., 0..., 0..<frequencyCount]
        let imaginary = stft[0..., 0..., frequencyCount..<(frequencyCount * 2)]
        let magnitude = MLX.sqrt(real * real + imaginary * imaginary + 1e-9)
        let mel = matmul(magnitude, melBasis.transposed())
        return MLX.log(MLX.maximum(mel, 1e-5))
    }
}

public final class LTXVocoderWithBWE: Module {
    @ModuleInfo(key: "base_vocoder") public var baseVocoder: LTXBigVGANVocoder
    @ModuleInfo(key: "bwe_generator") public var bweGenerator: LTXBigVGANVocoder
    @ModuleInfo(key: "mel_stft") public var melSTFT: LTXMelSTFT

    public let samplingRate = 48_000
    public let baseSamplingRate = 16_000
    public let stereoChannels = 2
    private let resampler: LTXHannSincResampler

    public override init() {
        self._baseVocoder = ModuleInfo(
            wrappedValue: LTXBigVGANVocoder(configuration: .ltxBase),
            key: "base_vocoder"
        )
        self._bweGenerator = ModuleInfo(
            wrappedValue: LTXBigVGANVocoder(configuration: .ltxBWE),
            key: "bwe_generator"
        )
        self._melSTFT = ModuleInfo(wrappedValue: LTXMelSTFT(), key: "mel_stft")
        self.resampler = LTXHannSincResampler(upsampleFactor: 3)
        super.init()
    }

    public func decode(_ mel: MLXArray) throws -> MLXArray {
        guard mel.ndim == 4,
              mel.shape[1] == stereoChannels,
              mel.shape[2] > 0,
              mel.shape[3] == melSTFT.melCount else {
            throw LTXVideoRuntimeError.invalidLatentShape(mel.shape)
        }
        let input = mel.asType(.float32)
        let batch = input.shape[0]
        let time = input.shape[2]
        let melChannels = input.shape[1]
        let flattened = input
            .transposed(0, 1, 3, 2)
            .reshaped(batch, melChannels * melSTFT.melCount, time)
            .transposed(0, 2, 1)
        let base = baseVocoder(flattened).transposed(0, 2, 1)
        let baseLength = base.shape[2]
        let targetLength = baseLength * resampler.upsampleFactor
        let paddedLength = ((baseLength + melSTFT.hopLength - 1) / melSTFT.hopLength) * melSTFT.hopLength
        let padded = paddedLength == baseLength
            ? base
            : MLX.padded(
                base,
                widths: [(0, 0), (0, 0), (0, paddedLength - baseLength)].map { .init($0) }
            )
        let flatWaveform = padded.reshaped(batch * stereoChannels, padded.shape[2])
        let bweMel = try melSTFT(flatWaveform)
            .reshaped(batch, stereoChannels, -1, melSTFT.melCount)
        let bweInput = bweMel
            .transposed(0, 1, 3, 2)
            .reshaped(batch, stereoChannels * melSTFT.melCount, bweMel.shape[2])
            .transposed(0, 2, 1)
        let residual = bweGenerator(bweInput).transposed(0, 2, 1)
        let resampled = try MLX.concatenated((0..<stereoChannels).map { channel in
                try resampler(base[0..., channel, 0...])
        }, axis: 1)
        let count = min(resampled.shape[2], residual.shape[2])
        let output = MLX.clip(
            resampled[0..., 0..., 0..<count] + residual[0..., 0..., 0..<count],
            min: Float(-1),
            max: Float(1)
        )
        return output[0..., 0..., 0..<min(targetLength, output.shape[2])]
    }
}

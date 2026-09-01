import Foundation
import GenImageGGUF
import MLX

/// Loads H3 transformer weights from GGUF into MLX's quantized format.
///
/// Follows the conservative ("auto") strategy from the user's LlamaLoader
/// project: a low-bit GGUF source is decoded and **re-quantized to INT8**
/// rather than reusing its 4-bit block layout as MLX's execution layout.
///
/// Q4_0 does map onto MLX's affine int4 format exactly as *storage*
/// (`bias = -8 * scale`), so reusing the block is tempting. LlamaLoader
/// documents why not to: compressing a low-bit source into MLX int4 amounts to
/// a second low-bit quantization at execution time, and has been observed to
/// make some architectures emit invalid tokens. INT8 keeps the decoded values
/// intact at the cost of roughly double the resident memory.
public enum MiniMaxH3GGUFQuantizedLoader {
    /// Target width for re-quantized low-bit sources.
    public static let targetBits = 8
    /// MLX group size used when re-quantizing.
    public static let groupSize = 64

    public static var defaultUseMetalQuantizer: Bool {
        MiniMaxH3GGUFMetalQuantizer.isEnabledByDefault
    }

    public enum QuantizationBackend: String, Sendable {
        case metal
        case cpu
    }

    public struct QuantizedTensor {
        public let weights: MLXArray
        public let scales: MLXArray
        public let biases: MLXArray
        public let backend: QuantizationBackend
    }

    public struct Loaded {
        public let configuration: MiniMaxH3Configuration
        /// Dense tensors plus, for each quantized linear, `name` (packed
        /// weights), `prefix.scales` and `prefix.biases`.
        public let tensors: [String: MLXArray]
        /// Parameter prefixes stored in quantized form.
        public let quantizedPrefixes: Set<String>
        public let quantizedCount: Int
        public let denseCount: Int
        /// Tensors whose shape could not be re-quantized and stayed dense.
        public let skippedQuantizationCount: Int
        public let metalQuantizedCount: Int
        public let cpuQuantizedCount: Int
    }

    public static func load(fileURL: URL) throws -> Loaded {
        try load(
            fileURL: fileURL,
            useMetalQuantizer: MiniMaxH3GGUFMetalQuantizer.isEnabledByDefault
        )
    }

    public static func load(
        fileURL: URL,
        useMetalQuantizer: Bool
    ) throws -> Loaded {
        let inspection = try GGUFModelLoader.inspect(fileURL: fileURL)
        let inventory = try MiniMaxH3GGUFWeightLoader.inspectTransformer(fileURL: fileURL)
        let configuration = try MiniMaxH3Configuration.forInventory(inventory)
        let overrides = try MiniMaxH3GGUFWeightLoader.comfyShapeOverrides(in: inspection)
        let data = try Data(contentsOf: fileURL, options: .mappedIfSafe)

        var tensors: [String: MLXArray] = [:]
        var quantizedPrefixes: Set<String> = []
        var quantizedCount = 0
        var denseCount = 0
        var skipped = 0
        var metalQuantizedCount = 0
        var cpuQuantizedCount = 0

        for descriptor in inspection.tensors {
            let shape = overrides[descriptor.name] ?? descriptor.shape
            guard let byteSize = descriptor.byteSize else {
                throw MiniMaxH3WeightError.missingTensor(descriptor.name)
            }
            let start = inspection.dataOffset + Int(descriptor.offset)
            let end = start + Int(byteSize)
            guard start >= 0, end <= data.count else {
                throw MiniMaxH3WeightError.missingTensor(descriptor.name)
            }
            let raw = data.subdata(in: start ..< end)

            let isQuantized = GGUFDequantizer.isQuantized(typeName: descriptor.type)
            guard isQuantized, canQuantize(shape: shape) else {
                if isQuantized {
                    skipped += 1
                }
                let value = try GGUFDequantizer.array(
                    raw: raw,
                    typeCode: descriptor.typeCode,
                    shape: shape,
                    name: descriptor.name
                )
                MLX.eval(value)
                tensors[descriptor.name] = value
                denseCount += 1
                continue
            }

            let quantized = try quantize(
                raw: raw,
                descriptor: descriptor,
                shape: shape,
                useMetalQuantizer: useMetalQuantizer
            )
            let prefix = parameterPrefix(descriptor.name)
            tensors[descriptor.name] = quantized.weights
            tensors["\(prefix).scales"] = quantized.scales
            tensors["\(prefix).biases"] = quantized.biases
            switch quantized.backend {
            case .metal:
                metalQuantizedCount += 1
            case .cpu:
                cpuQuantizedCount += 1
            }
            quantizedPrefixes.insert(prefix)
            quantizedCount += 1
        }

        return Loaded(
            configuration: configuration,
            tensors: tensors,
            quantizedPrefixes: quantizedPrefixes,
            quantizedCount: quantizedCount,
            denseCount: denseCount,
            skippedQuantizationCount: skipped,
            metalQuantizedCount: metalQuantizedCount,
            cpuQuantizedCount: cpuQuantizedCount
        )
    }

    public static func loadQuantizedTensor(
        fileURL: URL,
        named name: String,
        useMetalQuantizer: Bool = defaultUseMetalQuantizer
    ) throws -> QuantizedTensor {
        let inspection = try GGUFModelLoader.inspect(fileURL: fileURL)
        guard let descriptor = inspection.tensors.first(where: { $0.name == name }),
              let byteSize = descriptor.byteSize else {
            throw MiniMaxH3WeightError.missingTensor(name)
        }
        let overrides = try MiniMaxH3GGUFWeightLoader.comfyShapeOverrides(in: inspection)
        let shape = overrides[name] ?? descriptor.shape
        guard GGUFDequantizer.isQuantized(typeName: descriptor.type),
              canQuantize(shape: shape) else {
            throw MiniMaxH3WeightError.unquantizableShape(name, shape: shape)
        }
        let data = try Data(contentsOf: fileURL, options: .mappedIfSafe)
        let start = inspection.dataOffset + Int(descriptor.offset)
        let end = start + Int(byteSize)
        guard start >= 0, end <= data.count else {
            throw MiniMaxH3WeightError.missingTensor(name)
        }
        return try quantize(
            raw: data.subdata(in: start ..< end),
            descriptor: descriptor,
            shape: shape,
            useMetalQuantizer: useMetalQuantizer
        )
    }

    private static func quantize(
        raw: Data,
        descriptor: GGUFTensorDescriptor,
        shape: [Int],
        useMetalQuantizer: Bool
    ) throws -> QuantizedTensor {
        if useMetalQuantizer,
           MiniMaxH3GGUFMetalQuantizer.supports(typeCode: descriptor.typeCode) {
            let quantized = try MiniMaxH3GGUFMetalQuantizer.quantize(
                raw: raw,
                sourceType: descriptor.typeCode,
                sourceShape: shape,
                name: descriptor.name
            )
            return QuantizedTensor(
                weights: quantized.weights,
                scales: quantized.scales,
                biases: quantized.biases,
                backend: .metal
            )
        }

        let value = try GGUFDequantizer.array(
            raw: raw,
            typeCode: descriptor.typeCode,
            shape: shape,
            name: descriptor.name
        ).asType(.float32)
        let quantized = MLX.quantized(value, groupSize: groupSize, bits: targetBits)
        guard let biases = quantized.biases else {
            throw MiniMaxH3WeightError.unsupportedTensorType(
                descriptor.name,
                type: descriptor.type
            )
        }
        MLX.eval(quantized.wq, quantized.scales, biases)
        return QuantizedTensor(
            weights: quantized.wq,
            scales: quantized.scales,
            biases: biases,
            backend: .cpu
        )
    }

    /// Load every tensor fully decoded, with no re-quantization.
    ///
    /// Only usable on a subset of the checkpoint — the full transformer would
    /// need roughly 74 GB as float32. Intended for parity work, where the goal
    /// is to isolate model logic from quantization error.
    public static func loadDense(
        fileURL: URL,
        matching predicate: (String) -> Bool
    ) throws -> [String: MLXArray] {
        let inspection = try GGUFModelLoader.inspect(fileURL: fileURL)
        let overrides = try MiniMaxH3GGUFWeightLoader.comfyShapeOverrides(in: inspection)
        let data = try Data(contentsOf: fileURL, options: .mappedIfSafe)

        var tensors: [String: MLXArray] = [:]
        for descriptor in inspection.tensors where predicate(descriptor.name) {
            guard let byteSize = descriptor.byteSize else {
                throw MiniMaxH3WeightError.missingTensor(descriptor.name)
            }
            let shape = overrides[descriptor.name] ?? descriptor.shape
            let start = inspection.dataOffset + Int(descriptor.offset)
            let end = start + Int(byteSize)
            guard start >= 0, end <= data.count else {
                throw MiniMaxH3WeightError.missingTensor(descriptor.name)
            }
            let value = try GGUFDequantizer.array(
                raw: data.subdata(in: start ..< end),
                typeCode: descriptor.typeCode,
                shape: shape,
                name: descriptor.name
            ).asType(.float32)
            MLX.eval(value)
            tensors[descriptor.name] = value
        }
        return tensors
    }

    static func parameterPrefix(_ name: String) -> String {
        name.hasSuffix(".weight") ? String(name.dropLast(".weight".count)) : name
    }

    /// MLX affine quantization needs a 2-D tensor whose inner dimension is a
    /// multiple of the group size.
    static func canQuantize(shape: [Int]) -> Bool {
        shape.count == 2
            && shape[0] > 0
            && shape[1] > 0
            && shape[0] % 32 == 0
            && shape[1] % groupSize == 0
    }
}

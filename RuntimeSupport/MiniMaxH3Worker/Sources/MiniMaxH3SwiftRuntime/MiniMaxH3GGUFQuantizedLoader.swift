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
    }

    public static func load(fileURL: URL) throws -> Loaded {
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

            var value = try GGUFDequantizer.array(
                raw: raw,
                typeCode: descriptor.typeCode,
                shape: shape,
                name: descriptor.name
            )

            guard GGUFDequantizer.isQuantized(typeName: descriptor.type),
                  canQuantize(shape: shape) else {
                if GGUFDequantizer.isQuantized(typeName: descriptor.type) {
                    skipped += 1
                }
                MLX.eval(value)
                tensors[descriptor.name] = value
                denseCount += 1
                continue
            }

            value = value.asType(.float32)
            let quantized = MLX.quantized(value, groupSize: groupSize, bits: targetBits)
            let prefix = parameterPrefix(descriptor.name)
            tensors[descriptor.name] = quantized.wq
            tensors["\(prefix).scales"] = quantized.scales
            if let biases = quantized.biases {
                tensors["\(prefix).biases"] = biases
                MLX.eval(quantized.wq, quantized.scales, biases)
            } else {
                MLX.eval(quantized.wq, quantized.scales)
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
            skippedQuantizationCount: skipped
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

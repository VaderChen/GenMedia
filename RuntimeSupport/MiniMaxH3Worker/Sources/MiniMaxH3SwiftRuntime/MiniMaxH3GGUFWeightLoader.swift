import Foundation
import GenImageGGUF
import MLX

/// Loads the MiniMax H3 transformer from a ComfyUI-converted GGUF file.
///
/// ComfyUI reshapes some tensors to a 2-D layout before quantizing them and
/// records the true shape under `comfy.gguf.orig_shape.<tensor name>`. The
/// stored dimensions keep the correct element count, so loading them verbatim
/// produces a silently mis-shaped weight instead of an error. This loader
/// recovers the real shapes before handing anything to MLX.
public enum MiniMaxH3GGUFWeightLoader {
    public static let origShapeMetadataPrefix = "comfy.gguf.orig_shape."

    /// Shapes recorded by the ComfyUI GGUF converter, keyed by tensor name.
    ///
    /// Tensors without an entry already carry their true shape in the GGUF
    /// dimensions (which `GGUFModelLoader` reverses into row-major order).
    public static func comfyShapeOverrides(
        in inspection: GGUFInspection
    ) throws -> [String: [Int]] {
        var overrides: [String: [Int]] = [:]
        for (key, value) in inspection.metadata
        where key.hasPrefix(origShapeMetadataPrefix) {
            let name = String(key.dropFirst(origShapeMetadataPrefix.count))
            guard let elements = value.arrayValue else {
                throw MiniMaxH3WeightError.malformedOrigShape(name)
            }
            let shape = try elements.map { element -> Int in
                guard let dimension = element.integerValue, dimension > 0 else {
                    throw MiniMaxH3WeightError.malformedOrigShape(name)
                }
                return dimension
            }
            guard !shape.isEmpty else {
                throw MiniMaxH3WeightError.malformedOrigShape(name)
            }
            overrides[name] = shape
        }
        return overrides
    }

    /// The logical shape every tensor should have once overrides are applied.
    public static func resolvedShapes(
        in inspection: GGUFInspection
    ) throws -> [String: [Int]] {
        let overrides = try comfyShapeOverrides(in: inspection)
        var resolved: [String: [Int]] = [:]
        for tensor in inspection.tensors {
            resolved[tensor.name] = overrides[tensor.name] ?? tensor.shape
        }
        return resolved
    }

    /// Inspect a transformer GGUF and report what the loader would produce,
    /// without materializing 18 GB of weights.
    public static func inspectTransformer(
        fileURL: URL
    ) throws -> MiniMaxH3WeightInventory {
        let inspection = try GGUFModelLoader.inspect(fileURL: fileURL)
        let overrides = try comfyShapeOverrides(in: inspection)
        var entries: [MiniMaxH3WeightInventory.Entry] = []
        entries.reserveCapacity(inspection.tensors.count)
        for tensor in inspection.tensors {
            let override = overrides[tensor.name]
            entries.append(
                MiniMaxH3WeightInventory.Entry(
                    name: tensor.name,
                    storedShape: tensor.shape,
                    logicalShape: override ?? tensor.shape,
                    ggmlType: tensor.type,
                    usedOrigShape: override != nil
                )
            )
        }
        return MiniMaxH3WeightInventory(
            architecture: inspection.metadata["general.architecture"]?.stringValue,
            entries: entries.sorted { $0.name < $1.name }
        )
    }

    /// Load the transformer weights with ComfyUI shape overrides applied.
    ///
    /// Materializes the whole file, so expect this to cost roughly the size of
    /// the GGUF in resident memory.
    public static func loadTransformer(
        fileURL: URL,
        options: GGUFLoadOptions = GGUFLoadOptions()
    ) throws -> GGUFLoadedWeights {
        let inspection = try GGUFModelLoader.inspect(fileURL: fileURL)
        var options = options
        options.shapeOverrides = try comfyShapeOverrides(in: inspection)
        return try GGUFModelLoader.loadWeights(fileURL: fileURL, options: options)
    }
}

/// What a transformer GGUF contains, with ComfyUI shape overrides resolved.
public struct MiniMaxH3WeightInventory: Sendable {
    public struct Entry: Sendable, Hashable {
        public let name: String
        public let storedShape: [Int]
        public let logicalShape: [Int]
        public let ggmlType: String
        public let usedOrigShape: Bool

        public init(
            name: String,
            storedShape: [Int],
            logicalShape: [Int],
            ggmlType: String,
            usedOrigShape: Bool
        ) {
            self.name = name
            self.storedShape = storedShape
            self.logicalShape = logicalShape
            self.ggmlType = ggmlType
            self.usedOrigShape = usedOrigShape
        }

        /// True when the recorded shape disagrees with the stored dimensions,
        /// i.e. loading without the override would silently mis-shape this
        /// tensor rather than fail.
        public var overrideChangesShape: Bool {
            usedOrigShape && storedShape != logicalShape
        }
    }

    public let architecture: String?
    public let entries: [Entry]

    public init(architecture: String?, entries: [Entry]) {
        self.architecture = architecture
        self.entries = entries
    }

    public var tensorCount: Int { entries.count }

    public func entry(named name: String) -> Entry? {
        entries.first { $0.name == name }
    }

    public func shape(of name: String) -> [Int]? {
        entry(named: name)?.logicalShape
    }

    /// Tensors whose shape would be wrong without the ComfyUI metadata.
    public var overriddenEntries: [Entry] {
        entries.filter(\.overrideChangesShape)
    }

    public var ggmlTypeCounts: [String: Int] {
        entries.reduce(into: [:]) { counts, entry in
            counts[entry.ggmlType, default: 0] += 1
        }
    }
}

public enum MiniMaxH3WeightError: LocalizedError, Sendable {
    case malformedOrigShape(String)
    case missingTensor(String)
    case unexpectedShape(String, expected: [Int], actual: [Int])
    case architectureMismatch(expected: [Int], actual: [Int], detail: String)
    case unsupportedTensorType(String, type: String)
    case unquantizableShape(String, shape: [Int])

    public var errorDescription: String? {
        switch self {
        case let .malformedOrigShape(name):
            "GGUF 權重「\(name)」的 comfy.gguf.orig_shape metadata 格式不正確。"
        case let .missingTensor(name):
            "H3 GGUF 缺少必要權重：\(name)。"
        case let .unexpectedShape(name, expected, actual):
            "H3 權重「\(name)」形狀不符：預期 \(expected)，實際 \(actual)。"
        case let .architectureMismatch(expected, actual, detail):
            "H3 架構參數不符（\(detail)）：預期 \(expected)，實際 \(actual)。"
        case let .unsupportedTensorType(name, type):
            "H3 權重「\(name)」使用未支援的 GGUF 型別 \(type)。"
        case let .unquantizableShape(name, shape):
            "H3 權重「\(name)」形狀 \(shape) 無法對齊 Q4_0 的 32 元素區塊。"
        }
    }
}

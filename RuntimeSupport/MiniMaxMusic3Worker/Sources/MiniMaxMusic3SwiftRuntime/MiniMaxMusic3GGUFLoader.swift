import Foundation
import MLX
import MLXNN

public struct MiniMaxMusic3GGUFLoadReport: Sendable {
    public let sourceTensorCount: Int
    public let parameterCount: Int
    public let quantizedTensorCount: Int
    public let sourceTypes: [String: Int]

    public init(
        sourceTensorCount: Int,
        parameterCount: Int,
        quantizedTensorCount: Int,
        sourceTypes: [String: Int]
    ) {
        self.sourceTensorCount = sourceTensorCount
        self.parameterCount = parameterCount
        self.quantizedTensorCount = quantizedTensorCount
        self.sourceTypes = sourceTypes
    }
}

public struct MiniMaxMusic3GGUFRawArrays {
    public let arrays: [String: MLXArray]
    public let sourceTensorCount: Int
    public let sourceTypes: [String: Int]

    public init(
        arrays: [String: MLXArray],
        sourceTensorCount: Int,
        sourceTypes: [String: Int]
    ) {
        self.arrays = arrays
        self.sourceTensorCount = sourceTensorCount
        self.sourceTypes = sourceTypes
    }
}

/// MLX quantized linear 使用的分組策略。
///
/// `groupSize` 沿權重矩陣的輸入維度分組；每組各自保存 scale 與 bias，
/// 而不是讓整個張量共用一組量化參數。Music3 的 MLX checkpoint 使用
/// 64 個輸入元素一組的 affine 量化。
public struct MiniMaxMusic3GGUFGroupStrategy: Sendable, Equatable {
    public let groupSize: Int
    public let mode: QuantizationMode

    public init(groupSize: Int = 64, mode: QuantizationMode = .affine) {
        precondition(groupSize > 0, "GGUF group size 必須大於 0")
        self.groupSize = groupSize
        self.mode = mode
    }
}

/// MiniMax Music 3 的 GGUF 量化對應 MLX 的 affine quantized linear。
///
/// GGUF 的 Q4_K/Q8_0 是來源儲存格式；載入後先還原成 dense 權重，再依
/// `groupStrategy` 重新建立 INT4/INT8 權重。非量化來源保留 F16、BF16 或
/// F32，不把整個模型升成 FP32。
public enum MiniMaxMusic3GGUFQuantizationStrategy {
    public static let groupStrategy = MiniMaxMusic3GGUFGroupStrategy(
        groupSize: 64,
        mode: .affine
    )

    public static var groupSize: Int { groupStrategy.groupSize }
    public static var mode: QuantizationMode { groupStrategy.mode }

    public static func targetBits(for sourceType: String) -> Int? {
        switch sourceType.uppercased() {
        case "Q1_0", "Q2_0", "Q2_K", "Q3_K", "Q4_0", "Q4_1", "Q4_K", "MXFP4":
            return 4
        case "Q5_0", "Q5_1", "Q5_K", "Q6_K", "Q8_0", "Q8_1":
            return 8
        default:
            return nil
        }
    }

    public static func isSourcePrecision(_ sourceType: String) -> Bool {
        switch sourceType.uppercased() {
        case "F16", "BF16", "F32":
            return true
        default:
            return false
        }
    }
}

func miniMaxMusic3Linear(
    _ inputDimensions: Int,
    _ outputDimensions: Int,
    bias: Bool = true,
    quantizationBits: Int?
) -> Linear {
    guard let bits = quantizationBits else {
        return Linear(inputDimensions, outputDimensions, bias: bias)
    }
    precondition(bits == 4 || bits == 8)
    return QuantizedLinear(
        inputDimensions,
        outputDimensions,
        bias: bias,
        groupSize: MiniMaxMusic3GGUFQuantizationStrategy.groupSize,
        bits: bits,
        mode: MiniMaxMusic3GGUFQuantizationStrategy.mode
    )
}

public enum MiniMaxMusic3GGUFError: LocalizedError, Sendable {
    case missingFile(URL)
    case invalidMagic
    case unsupportedVersion(UInt32)
    case truncated
    case invalidSize
    case invalidText
    case invalidAlignment
    case unsupportedMetadataType(UInt32)
    case unsupportedTensorType(UInt32, String)
    case invalidTensor(String)
    case duplicateWeight(String)
    case missingWeights([String])
    case weightShapeMismatch(name: String, expected: [Int], actual: [Int])
    case missingComponent(String, URL)
    case ambiguousComponent(String, [String])
    case mixedQuantization(String)

    public var errorDescription: String? {
        switch self {
        case let .missingFile(url):
            "找不到 Music3 GGUF 權重：\(url.path)"
        case .invalidMagic:
            "Music3 GGUF 標頭不正確。"
        case let .unsupportedVersion(version):
            "Music3 GGUF 版本不支援：\(version)。"
        case .truncated:
            "Music3 GGUF 檔案內容不完整。"
        case .invalidSize:
            "Music3 GGUF 尺寸超出目前平台可處理範圍。"
        case .invalidText:
            "Music3 GGUF 包含無法解碼的文字欄位。"
        case .invalidAlignment:
            "Music3 GGUF 資料對齊設定不正確。"
        case let .unsupportedMetadataType(type):
            "Music3 GGUF metadata 型別不支援：\(type)。"
        case let .unsupportedTensorType(type, name):
            "Music3 GGUF 權重「\(name)」使用不支援的型別：\(type)。"
        case let .invalidTensor(name):
            "Music3 GGUF 權重「\(name)」的形狀或資料範圍不正確。"
        case let .duplicateWeight(name):
            "Music3 GGUF 權重名稱重複：\(name)。"
        case let .missingWeights(names):
            "Music3 GGUF 缺少模型權重：\(names.joined(separator: ", "))"
        case let .weightShapeMismatch(name, expected, actual):
            "Music3 GGUF 權重 \(name) 形狀不符，預期 \(expected)，實際 \(actual)。"
        case let .missingComponent(component, url):
            "Music3 GGUF 缺少元件 \(component)：\(url.path)"
        case let .ambiguousComponent(component, names):
            "Music3 GGUF 元件 \(component) 不唯一：\(names.joined(separator: ", "))"
        case let .mixedQuantization(component):
            "Music3 GGUF 元件 \(component) 同時包含不同量化位元，無法套用單一 MLX Linear 策略。"
        }
    }
}

public enum MiniMaxMusic3GGUFWeightLoader {
    public static func loadRawArrays(
        from fileURL: URL
    ) throws -> MiniMaxMusic3GGUFRawArrays {
        let file = try MiniMaxMusic3GGUFFile(url: fileURL)
        var arrays: [String: MLXArray] = [:]
        var sourceTypes: [String: Int] = [:]
        for tensor in file.tensors {
            guard arrays[tensor.name] == nil else {
                throw MiniMaxMusic3GGUFError.duplicateWeight(tensor.name)
            }
            arrays[tensor.name] = try file.array(for: tensor)
            sourceTypes[tensor.typeName, default: 0] += 1
        }
        return MiniMaxMusic3GGUFRawArrays(
            arrays: arrays,
            sourceTensorCount: file.tensors.count,
            sourceTypes: sourceTypes
        )
    }

    public static func load(
        model: Module,
        from fileURL: URL,
        valueTransform: (String, MLXArray) throws -> MLXArray = { _, value in value }
    ) throws -> MiniMaxMusic3GGUFLoadReport {
        let file = try MiniMaxMusic3GGUFFile(url: fileURL)
        let expected = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1) }
        )
        var converted: [String: MLXArray] = [:]
        var sourceTypes: [String: Int] = [:]
        var quantizedTensorCount = 0

        for tensor in file.tensors {
            let typeName = tensor.typeName
            sourceTypes[typeName, default: 0] += 1
            guard expected[tensor.name] != nil else { continue }
            var value = try valueTransform(tensor.name, file.array(for: tensor))

            if let bits = MiniMaxMusic3GGUFQuantizationStrategy.targetBits(
                for: tensor.typeName
            ) {
                guard tensor.shape.count == 2,
                      tensor.shape[0] % 32 == 0,
                      tensor.shape[1] % MiniMaxMusic3GGUFQuantizationStrategy.groupSize == 0
                else {
                    throw MiniMaxMusic3GGUFError.invalidTensor(tensor.name)
                }
                value = value.asType(.float32)
                let quantized = MLX.quantized(
                    value,
                    groupSize: MiniMaxMusic3GGUFQuantizationStrategy.groupSize,
                    bits: bits,
                    mode: MiniMaxMusic3GGUFQuantizationStrategy.mode
                )
                if let biases = quantized.biases {
                    MLX.eval(quantized.wq, quantized.scales, biases)
                } else {
                    MLX.eval(quantized.wq, quantized.scales)
                }
                try insert(quantized.wq, name: tensor.name, expected: expected, into: &converted)
                let prefix = parameterPrefix(tensor.name)
                try insert(
                    quantized.scales,
                    name: prefix + ".scales",
                    expected: expected,
                    into: &converted
                )
                if let biases = quantized.biases {
                    try insert(
                        biases,
                        name: prefix + ".biases",
                        expected: expected,
                        into: &converted
                    )
                }
                quantizedTensorCount += 1
            } else {
                guard value.shape == expected[tensor.name]!.shape else {
                    throw MiniMaxMusic3GGUFError.weightShapeMismatch(
                        name: tensor.name,
                        expected: expected[tensor.name]!.shape,
                        actual: value.shape
                    )
                }
                MLX.eval(value)
                try insert(value, name: tensor.name, expected: expected, into: &converted)
            }
        }

        let missing = Set(expected.keys).subtracting(converted.keys).sorted()
        guard missing.isEmpty else {
            throw MiniMaxMusic3GGUFError.missingWeights(missing)
        }
        try model.update(
            parameters: ModuleParameters.unflattened(converted),
            verify: .all
        )
        MLX.eval(model)
        return MiniMaxMusic3GGUFLoadReport(
            sourceTensorCount: file.tensors.count,
            parameterCount: converted.count,
            quantizedTensorCount: quantizedTensorCount,
            sourceTypes: sourceTypes
        )
    }

    private static func insert(
        _ value: MLXArray,
        name: String,
        expected: [String: MLXArray],
        into converted: inout [String: MLXArray]
    ) throws {
        guard expected[name] != nil else { return }
        guard converted[name] == nil else {
            throw MiniMaxMusic3GGUFError.duplicateWeight(name)
        }
        guard value.shape == expected[name]!.shape else {
            throw MiniMaxMusic3GGUFError.weightShapeMismatch(
                name: name,
                expected: expected[name]!.shape,
                actual: value.shape
            )
        }
        converted[name] = value
    }

    private static func parameterPrefix(_ name: String) -> String {
        name.hasSuffix(".weight")
            ? String(name.dropLast(".weight".count))
            : name
    }
}

public enum MiniMaxMusic3GGUFModel {
    public enum Component: String, CaseIterable, Sendable {
        case conditionEncoder = "condition_encoder"
        case languageModel = "language_model"
        case rvqDepthDecoder = "rvq_depth_decoder"
        case transformer
        case vocoder

        var preferredFileName: String {
            switch self {
            case .conditionEncoder: "condition_encoder.gguf"
            case .languageModel: "language_model_q4_k.gguf"
            case .rvqDepthDecoder: "rvq_depth_decoder_q8_0.gguf"
            case .transformer: "transformer_q4_k.gguf"
            case .vocoder: "vocoder.gguf"
            }
        }
    }

    public static func isAvailable(at directoryURL: URL) -> Bool {
        Component.allCases.allSatisfy { component in
            (try? componentURL(for: component, at: directoryURL)) != nil
        }
    }

    public static func componentURL(
        for component: Component,
        at directoryURL: URL
    ) throws -> URL {
        let preferred = directoryURL.appendingPathComponent(component.preferredFileName)
        if FileManager.default.fileExists(atPath: preferred.path) {
            return preferred
        }
        let candidates = try FileManager.default.contentsOfDirectory(
            at: directoryURL,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ).filter { url in
            url.pathExtension.lowercased() == "gguf"
                && url.lastPathComponent.hasPrefix(component.rawValue)
                && (try? url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true
        }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard candidates.count == 1, let candidate = candidates.first else {
            if candidates.isEmpty {
                throw MiniMaxMusic3GGUFError.missingComponent(component.rawValue, preferred)
            }
            throw MiniMaxMusic3GGUFError.ambiguousComponent(
                component.rawValue,
                candidates.map(\.lastPathComponent)
            )
        }
        return candidate
    }

    public static func quantizationBits(
        for fileURL: URL,
        component: Component
    ) throws -> Int? {
        let file = try MiniMaxMusic3GGUFFile(url: fileURL)
        let bitValues = Set(
            file.tensors.compactMap {
                MiniMaxMusic3GGUFQuantizationStrategy.targetBits(for: $0.typeName)
            }
        )
        guard bitValues.count <= 1 else {
            throw MiniMaxMusic3GGUFError.mixedQuantization(component.rawValue)
        }
        return bitValues.first
    }
}

private struct MiniMaxMusic3GGUFTensor {
    let name: String
    let shape: [Int]
    let typeCode: UInt32
    let offset: UInt64

    var typeName: String {
        switch typeCode {
        case 0: "F32"
        case 1: "F16"
        case 8: "Q8_0"
        case 12: "Q4_K"
        case 30: "BF16"
        default: "TYPE_\(typeCode)"
        }
    }

    var quantizedBits: Int? {
        MiniMaxMusic3GGUFQuantizationStrategy.targetBits(for: typeName)
    }
}

private struct MiniMaxMusic3GGUFFile {
    let data: Data
    let tensorDataOffset: Int
    let tensors: [MiniMaxMusic3GGUFTensor]

    init(url: URL) throws {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw MiniMaxMusic3GGUFError.missingFile(url)
        }
        do {
            data = try Data(contentsOf: url, options: [.mappedIfSafe])
        } catch {
            throw MiniMaxMusic3GGUFError.truncated
        }
        var reader = MiniMaxMusic3GGUFReader(data: data)
        guard try reader.readUInt32() == 0x4655_4747 else {
            throw MiniMaxMusic3GGUFError.invalidMagic
        }
        let version = try reader.readUInt32()
        guard version == 2 || version == 3 else {
            throw MiniMaxMusic3GGUFError.unsupportedVersion(version)
        }
        let tensorCount = try reader.readCount()
        let metadataCount = try reader.readCount()
        var alignment = 32
        for _ in 0..<metadataCount {
            let key = try reader.readString()
            let type = try reader.readUInt32()
            if key == "general.alignment" {
                switch type {
                case 4:
                    alignment = try reader.readCount()
                case 10:
                    let value = try reader.readUInt64()
                    guard value <= UInt64(Int.max) else {
                        throw MiniMaxMusic3GGUFError.invalidAlignment
                    }
                    alignment = Int(value)
                default:
                    try reader.skipMetadataValue(type: type)
                }
            } else {
                try reader.skipMetadataValue(type: type)
            }
        }
        guard alignment > 0 else { throw MiniMaxMusic3GGUFError.invalidAlignment }

        var descriptors: [MiniMaxMusic3GGUFTensor] = []
        descriptors.reserveCapacity(tensorCount)
        for _ in 0..<tensorCount {
            let name = try reader.readString()
            let dimensionCount = try reader.readUInt32()
            guard dimensionCount > 0, dimensionCount <= 8 else {
                throw MiniMaxMusic3GGUFError.invalidSize
            }
            var dimensions: [Int] = []
            dimensions.reserveCapacity(Int(dimensionCount))
            for _ in 0..<dimensionCount {
                dimensions.append(try reader.readCount())
            }
            descriptors.append(
                MiniMaxMusic3GGUFTensor(
                    name: name,
                    shape: dimensions.reversed(),
                    typeCode: try reader.readUInt32(),
                    offset: try reader.readUInt64()
                )
            )
        }
        let remainder = reader.offset % alignment
        if remainder != 0 {
            try reader.skip(bytes: alignment - remainder)
        }
        tensorDataOffset = reader.offset
        tensors = descriptors
    }

    func array(for tensor: MiniMaxMusic3GGUFTensor) throws -> MLXArray {
        let elementCount = try product(tensor.shape, name: tensor.name)
        let bytes = try byteCount(typeCode: tensor.typeCode, elementCount: elementCount)
        guard tensor.offset <= UInt64(data.count - tensorDataOffset),
              tensor.offset <= UInt64(Int.max) else {
            throw MiniMaxMusic3GGUFError.invalidTensor(tensor.name)
        }
        let start = tensorDataOffset + Int(tensor.offset)
        guard start >= 0, start <= data.count, bytes <= data.count - start else {
            throw MiniMaxMusic3GGUFError.invalidTensor(tensor.name)
        }
        let raw = data.subdata(in: start..<(start + bytes))
        switch tensor.typeCode {
        case 0:
            return MLXArray(raw, tensor.shape, dtype: .float32)
        case 1:
            return MLXArray(raw, tensor.shape, dtype: .float16)
        case 30:
            return MLXArray(raw, tensor.shape, dtype: .bfloat16)
        case 8, 12:
            guard let values = try decodeQuantized(raw, typeCode: tensor.typeCode, count: elementCount) else {
                throw MiniMaxMusic3GGUFError.unsupportedTensorType(tensor.typeCode, tensor.name)
            }
            return MLXArray(
                Data(bytes: values, count: values.count * MemoryLayout<Float>.stride),
                tensor.shape,
                dtype: .float32
            )
        default:
            throw MiniMaxMusic3GGUFError.unsupportedTensorType(tensor.typeCode, tensor.name)
        }
    }

    private func decodeQuantized(
        _ raw: Data,
        typeCode: UInt32,
        count: Int
    ) throws -> [Float]? {
        var values = [Float](repeating: 0, count: count)
        switch typeCode {
        case 8:
            guard count % 32 == 0, raw.count == count / 32 * 34 else {
                throw MiniMaxMusic3GGUFError.invalidSize
            }
            for block in 0..<(count / 32) {
                let offset = block * 34
                let scale = float16(raw, at: offset)
                for index in 0..<32 {
                    values[block * 32 + index] =
                        Float(Int8(bitPattern: raw[offset + 2 + index])) * scale
                }
            }
        case 12:
            guard count % 256 == 0, raw.count == count / 256 * 144 else {
                throw MiniMaxMusic3GGUFError.invalidSize
            }
            for block in 0..<(count / 256) {
                decodeQ4K(raw, offset: block * 144, output: &values, outputOffset: block * 256)
            }
        default:
            return nil
        }
        return values
    }

    private func decodeQ4K(
        _ raw: Data,
        offset: Int,
        output: inout [Float],
        outputOffset: Int
    ) {
        let scale = float16(raw, at: offset)
        let minimumScale = float16(raw, at: offset + 2)
        let scalesOffset = offset + 4
        let quantizedOffset = offset + 16
        var scaleIndex = 0
        for segment in 0..<4 {
            let firstScale = kScaleAndMin(index: scaleIndex, raw: raw, offset: scalesOffset)
            let secondScale = kScaleAndMin(index: scaleIndex + 1, raw: raw, offset: scalesOffset)
            let segmentOffset = quantizedOffset + segment * 32
            let firstOutput = outputOffset + segment * 64
            for index in 0..<32 {
                let quantized = raw[segmentOffset + index]
                output[firstOutput + index] =
                    scale * Float(firstScale.scale) * Float(quantized & 0x0f)
                    - minimumScale * Float(firstScale.minimum)
                output[firstOutput + 32 + index] =
                    scale * Float(secondScale.scale) * Float(quantized >> 4)
                    - minimumScale * Float(secondScale.minimum)
            }
            scaleIndex += 2
        }
    }

    private func kScaleAndMin(
        index: Int,
        raw: Data,
        offset: Int
    ) -> (scale: UInt8, minimum: UInt8) {
        if index < 4 {
            return (raw[offset + index] & 63, raw[offset + index + 4] & 63)
        }
        let scale = (raw[offset + index + 4] & 0x0f)
            | ((raw[offset + index - 4] >> 6) << 4)
        let minimum = (raw[offset + index + 4] >> 4)
            | ((raw[offset + index] >> 6) << 4)
        return (scale, minimum)
    }

    private func float16(_ data: Data, at offset: Int) -> Float {
        Float(Float16(bitPattern: uint16(data, at: offset)))
    }

    private func uint16(_ data: Data, at offset: Int) -> UInt16 {
        UInt16(data[offset]) | UInt16(data[offset + 1]) << 8
    }

    private func byteCount(typeCode: UInt32, elementCount: Int) throws -> Int {
        switch typeCode {
        case 0, 26:
            return try multiplied(elementCount, by: 4)
        case 1, 30:
            return try multiplied(elementCount, by: 2)
        case 8:
            return try blockByteCount(elementCount, elementsPerBlock: 32, bytesPerBlock: 34)
        case 12:
            return try blockByteCount(elementCount, elementsPerBlock: 256, bytesPerBlock: 144)
        default:
            throw MiniMaxMusic3GGUFError.unsupportedTensorType(typeCode, "")
        }
    }

    private func product(_ values: [Int], name: String) throws -> Int {
        guard !values.isEmpty else { throw MiniMaxMusic3GGUFError.invalidTensor(name) }
        var result = 1
        for value in values {
            guard value > 0 else { throw MiniMaxMusic3GGUFError.invalidTensor(name) }
            let multiplication = result.multipliedReportingOverflow(by: value)
            guard !multiplication.overflow else { throw MiniMaxMusic3GGUFError.invalidSize }
            result = multiplication.partialValue
        }
        return result
    }

    private func multiplied(_ value: Int, by factor: Int) throws -> Int {
        let result = value.multipliedReportingOverflow(by: factor)
        guard !result.overflow else { throw MiniMaxMusic3GGUFError.invalidSize }
        return result.partialValue
    }

    private func blockByteCount(
        _ elementCount: Int,
        elementsPerBlock: Int,
        bytesPerBlock: Int
    ) throws -> Int {
        guard elementCount % elementsPerBlock == 0 else {
            throw MiniMaxMusic3GGUFError.invalidSize
        }
        return try multiplied(elementCount / elementsPerBlock, by: bytesPerBlock)
    }
}

private struct MiniMaxMusic3GGUFReader {
    let data: Data
    var offset = 0

    mutating func readUInt8() throws -> UInt8 {
        guard offset < data.count else { throw MiniMaxMusic3GGUFError.truncated }
        defer { offset += 1 }
        return data[offset]
    }

    mutating func readUInt32() throws -> UInt32 {
        UInt32(try readUInt8())
            | UInt32(try readUInt8()) << 8
            | UInt32(try readUInt8()) << 16
            | UInt32(try readUInt8()) << 24
    }

    mutating func readUInt64() throws -> UInt64 {
        var value: UInt64 = 0
        for index in 0..<8 {
            value |= UInt64(try readUInt8()) << UInt64(index * 8)
        }
        return value
    }

    mutating func readCount() throws -> Int {
        let value = try readUInt64()
        guard value <= UInt64(Int.max) else { throw MiniMaxMusic3GGUFError.invalidSize }
        return Int(value)
    }

    mutating func readString() throws -> String {
        let length = try readCount()
        guard length <= data.count - offset else { throw MiniMaxMusic3GGUFError.truncated }
        let value = data.subdata(in: offset..<(offset + length))
        offset += length
        guard let string = String(data: value, encoding: .utf8) else {
            throw MiniMaxMusic3GGUFError.invalidText
        }
        return string
    }

    mutating func skip(bytes: Int) throws {
        guard bytes >= 0, bytes <= data.count - offset else {
            throw MiniMaxMusic3GGUFError.truncated
        }
        offset += bytes
    }

    mutating func skipMetadataValue(type: UInt32) throws {
        switch type {
        case 0, 1, 7:
            try skip(bytes: 1)
        case 2, 3:
            try skip(bytes: 2)
        case 4, 5, 6:
            try skip(bytes: 4)
        case 8:
            _ = try readString()
        case 9:
            let elementType = try readUInt32()
            let count = try readCount()
            for _ in 0..<count {
                try skipMetadataValue(type: elementType)
            }
        case 10, 11, 12:
            try skip(bytes: 8)
        default:
            throw MiniMaxMusic3GGUFError.unsupportedMetadataType(type)
        }
    }
}

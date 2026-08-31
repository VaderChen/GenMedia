import Foundation
import MLX

public enum GGUFStorageType: String, Codable, Hashable, Sendable {
    case int4 = "INT4"
    case int8 = "INT8"
    case int16 = "INT16"
    case int32 = "INT32"
    case int64 = "INT64"
    case float16 = "FP16"
    case bfloat16 = "BF16"
    case float32 = "FP32"
    case float64 = "FP64"
}

public enum GGUFMaterialization: String, Codable, Hashable, Sendable {
    case dense
    case mlxQuantized
}

public enum GGUFQuantizationProfile: String, Codable, Hashable, Sendable {
    case quality
    case speed
}

public enum GGUFComputeDType: String, Codable, Hashable, Sendable {
    case source
    case float16
    case bfloat16
    case float32

    var mlxDType: DType? {
        switch self {
        case .source: nil
        case .float16: .float16
        case .bfloat16: .bfloat16
        case .float32: .float32
        }
    }
}

public enum GGUFMetadataValue: Equatable, Sendable {
    case uint8(UInt8)
    case int8(Int8)
    case uint16(UInt16)
    case int16(Int16)
    case uint32(UInt32)
    case int32(Int32)
    case float32(Float)
    case boolean(Bool)
    case string(String)
    case array([GGUFMetadataValue])
    case uint64(UInt64)
    case int64(Int64)
    case float64(Double)

    public var integerValue: Int? {
        switch self {
        case let .uint8(value): Int(value)
        case let .int8(value): Int(value)
        case let .uint16(value): Int(value)
        case let .int16(value): Int(value)
        case let .uint32(value): Int(exactly: value)
        case let .int32(value): Int(value)
        case let .uint64(value): Int(exactly: value)
        case let .int64(value): Int(exactly: value)
        case let .float32(value): Int(exactly: value)
        case let .float64(value): Int(exactly: value)
        default: nil
        }
    }

    public var stringValue: String? {
        guard case let .string(value) = self else { return nil }
        return value
    }

    public var arrayValue: [GGUFMetadataValue]? {
        guard case let .array(value) = self else { return nil }
        return value
    }
}

public struct GGUFTensorDescriptor: Codable, Hashable, Sendable {
    public let name: String
    public let shape: [Int]
    public let typeCode: UInt32
    public let type: String
    public let offset: UInt64
    public let byteSize: UInt64?
    public let storageType: GGUFStorageType?
    public let isMaterializable: Bool
    public let requiresRequantization: Bool

    public init(
        name: String,
        shape: [Int],
        typeCode: UInt32,
        type: String,
        offset: UInt64,
        byteSize: UInt64?,
        storageType: GGUFStorageType?,
        isMaterializable: Bool,
        requiresRequantization: Bool
    ) {
        self.name = name
        self.shape = shape
        self.typeCode = typeCode
        self.type = type
        self.offset = offset
        self.byteSize = byteSize
        self.storageType = storageType
        self.isMaterializable = isMaterializable
        self.requiresRequantization = requiresRequantization
    }
}

public struct GGUFInspection: Sendable {
    public let version: UInt32
    public let alignment: Int
    public let dataOffset: Int
    public let fileSize: UInt64
    public let metadata: [String: GGUFMetadataValue]
    public let tensors: [GGUFTensorDescriptor]
    public let quantizationCounts: [String: Int]
    public let unsupportedTypes: [String]

    public var tensorCount: Int { tensors.count }

    public var storageTypeCounts: [GGUFStorageType: Int] {
        tensors.reduce(into: [:]) { counts, tensor in
            if let storageType = tensor.storageType {
                counts[storageType, default: 0] += 1
            }
        }
    }
}

public struct GGUFLoadOptions: Sendable {
    public var materialization: GGUFMaterialization
    public var computeDType: GGUFComputeDType
    public var quantizationProfile: GGUFQuantizationProfile
    public var groupSize: Int
    /// Logical shapes that override the shape implied by the GGUF tensor
    /// dimensions, keyed by tensor name.
    ///
    /// Some converters (notably ComfyUI, via `comfy.gguf.orig_shape.*`
    /// metadata) reshape a tensor before quantizing it and record the original
    /// shape separately. The stored dimensions then have the correct element
    /// count but the wrong rank/extents, so loading them verbatim yields a
    /// silently mis-shaped weight rather than an error. Supplying the true
    /// shape here restores it. An override must preserve the element count.
    ///
    /// Empty by default, so existing callers are unaffected.
    public var shapeOverrides: [String: [Int]]

    public init(
        materialization: GGUFMaterialization = .mlxQuantized,
        computeDType: GGUFComputeDType = .source,
        quantizationProfile: GGUFQuantizationProfile = .quality,
        groupSize: Int = 64,
        shapeOverrides: [String: [Int]] = [:]
    ) {
        self.materialization = materialization
        self.computeDType = computeDType
        self.quantizationProfile = quantizationProfile
        self.groupSize = groupSize
        self.shapeOverrides = shapeOverrides
    }
}

public struct GGUFLoadedWeights {
    public let tensors: [String: MLXArray]
    public let sourceTensorCount: Int
    public let quantizedTensorCount: Int
    public let denseTensorCount: Int
    public let skippedQuantizationTensorCount: Int
    public let sourceQuantizationCounts: [String: Int]

    public init(
        tensors: [String: MLXArray],
        sourceTensorCount: Int,
        quantizedTensorCount: Int,
        denseTensorCount: Int,
        skippedQuantizationTensorCount: Int,
        sourceQuantizationCounts: [String: Int]
    ) {
        self.tensors = tensors
        self.sourceTensorCount = sourceTensorCount
        self.quantizedTensorCount = quantizedTensorCount
        self.denseTensorCount = denseTensorCount
        self.skippedQuantizationTensorCount = skippedQuantizationTensorCount
        self.sourceQuantizationCounts = sourceQuantizationCounts
    }
}

public struct GGUFLoadedTensor {
    public let descriptor: GGUFTensorDescriptor
    public let value: MLXArray

    public init(descriptor: GGUFTensorDescriptor, value: MLXArray) {
        self.descriptor = descriptor
        self.value = value
    }
}

public enum GGUFStoragePolicy {
    private static let int4Types: Set<String> = [
        "MXFP4", "Q1_0", "Q2_0", "Q2_K", "Q3_K", "Q4_0", "Q4_1", "Q4_K"
    ]
    private static let int8Types: Set<String> = [
        "Q5_0", "Q5_1", "Q5_K", "Q6_K", "Q8_0", "Q8_1"
    ]

    public static func storageType(
        for sourceType: String,
        profile: GGUFQuantizationProfile = .quality
    ) -> GGUFStorageType? {
        let normalized = sourceType.uppercased()
        if int4Types.contains(normalized) { return .int4 }
        if int8Types.contains(normalized) {
            if profile == .speed, normalized == "Q5_K" || normalized == "Q6_K" {
                return .int4
            }
            return .int8
        }
        switch normalized {
        case "F16": return .float16
        case "BF16": return .bfloat16
        case "F32": return .float32
        case "F64": return .float64
        case "I8": return .int8
        case "I16": return .int16
        case "I32": return .int32
        case "I64": return .int64
        default: return nil
        }
    }

    public static func targetBits(
        for sourceType: String,
        profile: GGUFQuantizationProfile = .quality
    ) -> Int? {
        switch storageType(for: sourceType, profile: profile) {
        case .int4: 4
        case .int8 where GGUFDequantizer.isQuantized(typeName: sourceType): 8
        default: nil
        }
    }

    public static func isMaterializable(_ sourceType: String) -> Bool {
        GGUFDequantizer.isMaterializable(typeName: sourceType)
    }
}

public enum GGUFLoaderError: LocalizedError, Sendable {
    case fileNotFound(URL)
    case ambiguousWeights(URL, [String])
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
    case invalidGroupSize(Int)
    case shapeOverrideMismatch(String, stored: [Int], override: [Int])

    public var errorDescription: String? {
        switch self {
        case let .fileNotFound(url):
            "找不到 GGUF 檔案：\(url.path)"
        case let .ambiguousWeights(url, names):
            "GGUF 模型目錄中的主權重不唯一：\(url.path)（\(names.joined(separator: ", "))）"
        case .invalidMagic:
            "GGUF 檔案標頭不正確。"
        case let .unsupportedVersion(version):
            "不支援 GGUF 版本：\(version)。"
        case .truncated:
            "GGUF 檔案內容不完整。"
        case .invalidSize:
            "GGUF 檔案尺寸超出目前平台可處理範圍。"
        case .invalidText:
            "GGUF 檔案包含無法解碼的文字欄位。"
        case .invalidAlignment:
            "GGUF 檔案的資料對齊設定不正確。"
        case let .unsupportedMetadataType(type):
            "GGUF metadata 型別 \(type) 不支援。"
        case let .unsupportedTensorType(type, name):
            "GGUF 權重「\(name)」使用目前未支援的型別 \(type)。"
        case let .invalidTensor(name):
            "GGUF 權重「\(name)」的形狀或資料範圍不正確。"
        case let .duplicateWeight(name):
            "GGUF 權重名稱重複：\(name)。"
        case let .invalidGroupSize(groupSize):
            "GGUF MLX 量化 group size 只支援 32 或 64，實際為 \(groupSize)。"
        case let .shapeOverrideMismatch(name, stored, override):
            """
            GGUF 權重「\(name)」的 shape override 元素數與實際資料不符：\
            儲存形狀 \(stored)，override \(override)。
            """
        }
    }
}

public enum GGUFModelLoader {
    public static func locate(
        in directoryURL: URL,
        preferredFileName: String? = nil
    ) throws -> URL {
        let fileManager = FileManager.default
        if let preferredFileName {
            let root = directoryURL.standardizedFileURL
            let candidate = root.appendingPathComponent(preferredFileName).standardizedFileURL
            guard candidate.path.hasPrefix(root.path + "/"),
                  candidate.pathExtension.lowercased() == "gguf",
                  fileManager.fileExists(atPath: candidate.path) else {
                throw GGUFLoaderError.fileNotFound(candidate)
            }
            return candidate
        }

        guard let enumerator = fileManager.enumerator(
            at: directoryURL,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            throw GGUFLoaderError.fileNotFound(directoryURL)
        }
        let candidates = enumerator.compactMap { item -> URL? in
            guard let url = item as? URL,
                  url.pathExtension.lowercased() == "gguf",
                  !url.lastPathComponent.lowercased().contains("mmproj"),
                  (try? url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true
            else { return nil }
            return url
        }.sorted { $0.path < $1.path }
        guard candidates.count == 1, let candidate = candidates.first else {
            if candidates.isEmpty { throw GGUFLoaderError.fileNotFound(directoryURL) }
            throw GGUFLoaderError.ambiguousWeights(
                directoryURL,
                candidates.map(\.lastPathComponent)
            )
        }
        return candidate
    }

    public static func inspect(fileURL: URL) throws -> GGUFInspection {
        let data = try mappedData(fileURL: fileURL)
        let layout = try parse(data)
        let descriptors = layout.tensors.map { tensor in
            let shape = Array(tensor.dimensions.reversed())
            let elementCount = try? checkedProduct(shape)
            let byteSize = elementCount.flatMap {
                try? GGUFDequantizer.byteCount(typeCode: tensor.typeCode, elementCount: $0)
            }
            let typeName = GGUFDequantizer.typeName(typeCode: tensor.typeCode)
            let materializable = byteSize != nil
                && GGUFDequantizer.isMaterializable(typeName: typeName)
            return GGUFTensorDescriptor(
                name: tensor.name,
                shape: shape,
                typeCode: tensor.typeCode,
                type: typeName,
                offset: tensor.offset,
                byteSize: byteSize.map(UInt64.init),
                storageType: GGUFStoragePolicy.storageType(for: typeName),
                isMaterializable: materializable,
                requiresRequantization: GGUFDequantizer.isQuantized(typeName: typeName)
            )
        }
        let counts = descriptors.reduce(into: [String: Int]()) { result, descriptor in
            result[descriptor.type, default: 0] += 1
        }
        let unsupported = Set(
            descriptors.filter { !$0.isMaterializable }.map(\.type)
        ).sorted()
        return GGUFInspection(
            version: layout.version,
            alignment: layout.alignment,
            dataOffset: layout.tensorDataOffset,
            fileSize: UInt64(data.count),
            metadata: layout.metadata,
            tensors: descriptors,
            quantizationCounts: counts,
            unsupportedTypes: unsupported
        )
    }

    public static func loadTensor(
        fileURL: URL,
        named tensorName: String,
        computeDType: GGUFComputeDType = .source
    ) throws -> GGUFLoadedTensor {
        let data = try mappedData(fileURL: fileURL)
        let layout = try parse(data)
        guard let tensor = layout.tensors.first(where: { $0.name == tensorName }) else {
            throw GGUFLoaderError.invalidTensor(tensorName)
        }
        let shape = Array(tensor.dimensions.reversed())
        let elementCount = try checkedProduct(shape)
        let byteCount = try GGUFDequantizer.byteCount(
            typeCode: tensor.typeCode,
            elementCount: elementCount
        )
        let raw = try dataSlice(
            data,
            tensor: tensor,
            dataOffset: layout.tensorDataOffset,
            byteCount: byteCount
        )
        var value = try GGUFDequantizer.array(
            raw: raw,
            typeCode: tensor.typeCode,
            shape: shape,
            name: tensor.name
        )
        if let dtype = computeDType.mlxDType, value.dtype.isFloatingPoint {
            value = value.asType(dtype)
        }
        let descriptor = GGUFTensorDescriptor(
            name: tensor.name,
            shape: shape,
            typeCode: tensor.typeCode,
            type: GGUFDequantizer.typeName(typeCode: tensor.typeCode),
            offset: tensor.offset,
            byteSize: UInt64(byteCount),
            storageType: GGUFStoragePolicy.storageType(
                for: GGUFDequantizer.typeName(typeCode: tensor.typeCode)
            ),
            isMaterializable: true,
            requiresRequantization: GGUFDequantizer.isQuantized(
                typeName: GGUFDequantizer.typeName(typeCode: tensor.typeCode)
            )
        )
        return GGUFLoadedTensor(descriptor: descriptor, value: value)
    }

    public static func loadWeights(
        fileURL: URL,
        options: GGUFLoadOptions = GGUFLoadOptions()
    ) throws -> GGUFLoadedWeights {
        guard options.groupSize == 32 || options.groupSize == 64 else {
            throw GGUFLoaderError.invalidGroupSize(options.groupSize)
        }
        let data = try mappedData(fileURL: fileURL)
        let layout = try parse(data)
        var weights = [String: MLXArray]()
        var quantizedTensorCount = 0
        var denseTensorCount = 0
        var skippedQuantizationTensorCount = 0
        var sourceQuantizationCounts = [String: Int]()

        for tensor in layout.tensors {
            let storedShape = Array(tensor.dimensions.reversed())
            let storedElementCount = try checkedProduct(storedShape)
            let shape: [Int]
            if let override = options.shapeOverrides[tensor.name] {
                let overrideElementCount = try checkedProduct(override)
                guard overrideElementCount == storedElementCount else {
                    throw GGUFLoaderError.shapeOverrideMismatch(
                        tensor.name,
                        stored: storedShape,
                        override: override
                    )
                }
                shape = override
            } else {
                shape = storedShape
            }
            let elementCount = storedElementCount
            let typeName = GGUFDequantizer.typeName(typeCode: tensor.typeCode)
            sourceQuantizationCounts[typeName, default: 0] += 1
            let byteCount: Int
            do {
                byteCount = try GGUFDequantizer.byteCount(
                    typeCode: tensor.typeCode,
                    elementCount: elementCount
                )
            } catch {
                throw GGUFLoaderError.unsupportedTensorType(tensor.typeCode, tensor.name)
            }
            let raw = try dataSlice(
                data,
                tensor: tensor,
                dataOffset: layout.tensorDataOffset,
                byteCount: byteCount
            )
            var value = try GGUFDequantizer.array(
                raw: raw,
                typeCode: tensor.typeCode,
                shape: shape,
                name: tensor.name
            )

            let targetBits = GGUFStoragePolicy.targetBits(
                for: typeName,
                profile: options.quantizationProfile
            )
            if options.materialization == .mlxQuantized,
               let targetBits,
               canQuantize(shape: shape, groupSize: options.groupSize) {
                value = value.asType(.float32)
                let quantized = MLX.quantized(
                    value,
                    groupSize: options.groupSize,
                    bits: targetBits
                )
                if let biases = quantized.biases {
                    MLX.eval(quantized.wq, quantized.scales, biases)
                } else {
                    MLX.eval(quantized.wq, quantized.scales)
                }
                try insert(quantized.wq, name: tensor.name, into: &weights)
                let prefix = parameterPrefix(tensor.name)
                try insert(quantized.scales, name: prefix + ".scales", into: &weights)
                if let biases = quantized.biases {
                    try insert(biases, name: prefix + ".biases", into: &weights)
                }
                quantizedTensorCount += 1
            } else {
                if targetBits != nil, options.materialization == .mlxQuantized {
                    skippedQuantizationTensorCount += 1
                }
                if value.dtype.isFloatingPoint,
                   let computeDType = options.computeDType.mlxDType {
                    value = value.asType(computeDType)
                }
                MLX.eval(value)
                try insert(value, name: tensor.name, into: &weights)
                denseTensorCount += 1
            }
        }

        return GGUFLoadedWeights(
            tensors: weights,
            sourceTensorCount: layout.tensors.count,
            quantizedTensorCount: quantizedTensorCount,
            denseTensorCount: denseTensorCount,
            skippedQuantizationTensorCount: skippedQuantizationTensorCount,
            sourceQuantizationCounts: sourceQuantizationCounts
        )
    }

    private static func canQuantize(shape: [Int], groupSize: Int) -> Bool {
        shape.count == 2
            && shape[0] > 0
            && shape[1] > 0
            && shape[0] % 32 == 0
            && shape[1] % groupSize == 0
    }

    private static func parameterPrefix(_ name: String) -> String {
        name.hasSuffix(".weight")
            ? String(name.dropLast(".weight".count))
            : name
    }

    private static func mappedData(fileURL: URL) throws -> Data {
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            throw GGUFLoaderError.fileNotFound(fileURL)
        }
        do {
            return try Data(contentsOf: fileURL, options: [.mappedIfSafe])
        } catch {
            throw GGUFLoaderError.truncated
        }
    }

    private static func insert(
        _ array: MLXArray,
        name: String,
        into weights: inout [String: MLXArray]
    ) throws {
        guard weights[name] == nil else { throw GGUFLoaderError.duplicateWeight(name) }
        weights[name] = array
    }

    private static func dataSlice(
        _ data: Data,
        tensor: GGUFTensorInfo,
        dataOffset: Int,
        byteCount: Int
    ) throws -> Data {
        guard dataOffset >= 0,
              dataOffset <= data.count,
              byteCount >= 0,
              tensor.offset <= UInt64(data.count - dataOffset),
              tensor.offset <= UInt64(Int.max) else {
            throw GGUFLoaderError.invalidTensor(tensor.name)
        }
        let (start, overflow) = dataOffset.addingReportingOverflow(Int(tensor.offset))
        guard !overflow,
              start <= data.count,
              byteCount <= data.count - start else {
            throw GGUFLoaderError.invalidTensor(tensor.name)
        }
        return data.subdata(in: start..<(start + byteCount))
    }

    private static func checkedProduct(_ values: [Int]) throws -> Int {
        var product = 1
        for value in values {
            guard value > 0 else { throw GGUFLoaderError.invalidSize }
            let result = product.multipliedReportingOverflow(by: value)
            guard !result.overflow else { throw GGUFLoaderError.invalidSize }
            product = result.partialValue
        }
        return product
    }

    private static func parse(_ data: Data) throws -> GGUFFileLayout {
        var reader = GGUFReader(data: data)
        guard try reader.readUInt32() == 0x4655_4747 else {
            throw GGUFLoaderError.invalidMagic
        }
        let version = try reader.readUInt32()
        guard version == 2 || version == 3 else {
            throw GGUFLoaderError.unsupportedVersion(version)
        }
        let tensorCount = try reader.readCount()
        let metadataCount = try reader.readCount()
        var alignment = 32
        var metadata = [String: GGUFMetadataValue]()
        metadata.reserveCapacity(metadataCount)

        for _ in 0..<metadataCount {
            let key = try reader.readString()
            let valueType = try reader.readUInt32()
            let value = try reader.readMetadataValue(type: valueType)
            metadata[key] = value
            if key == "general.alignment", let configuredAlignment = value.integerValue {
                alignment = configuredAlignment
            }
        }
        guard alignment > 0 else { throw GGUFLoaderError.invalidAlignment }

        var tensors = [GGUFTensorInfo]()
        tensors.reserveCapacity(tensorCount)
        for _ in 0..<tensorCount {
            let name = try reader.readString()
            let dimensionCount = try reader.readUInt32()
            guard dimensionCount > 0, dimensionCount <= 8 else {
                throw GGUFLoaderError.invalidSize
            }
            var dimensions = [Int]()
            dimensions.reserveCapacity(Int(dimensionCount))
            for _ in 0..<dimensionCount {
                dimensions.append(try reader.readCount())
            }
            tensors.append(GGUFTensorInfo(
                name: name,
                dimensions: dimensions,
                typeCode: try reader.readUInt32(),
                offset: try reader.readUInt64()
            ))
        }

        let remainder = reader.offset % alignment
        if remainder != 0 {
            try reader.skip(bytes: alignment - remainder)
        }
        return GGUFFileLayout(
            version: version,
            alignment: alignment,
            tensorDataOffset: reader.offset,
            metadata: metadata,
            tensors: tensors
        )
    }
}

private struct GGUFTensorInfo {
    let name: String
    let dimensions: [Int]
    let typeCode: UInt32
    let offset: UInt64
}

private struct GGUFFileLayout {
    let version: UInt32
    let alignment: Int
    let tensorDataOffset: Int
    let metadata: [String: GGUFMetadataValue]
    let tensors: [GGUFTensorInfo]
}

private struct GGUFReader {
    let data: Data
    var offset = 0

    mutating func readUInt8() throws -> UInt8 {
        guard offset < data.count else { throw GGUFLoaderError.truncated }
        defer { offset += 1 }
        return data[offset]
    }

    mutating func readUInt16() throws -> UInt16 {
        UInt16(try readUInt8()) | UInt16(try readUInt8()) << 8
    }

    mutating func readUInt32() throws -> UInt32 {
        var value: UInt32 = 0
        for byteIndex in 0..<4 {
            value |= UInt32(try readUInt8()) << UInt32(byteIndex * 8)
        }
        return value
    }

    mutating func readUInt64() throws -> UInt64 {
        var value: UInt64 = 0
        for byteIndex in 0..<8 {
            value |= UInt64(try readUInt8()) << UInt64(byteIndex * 8)
        }
        return value
    }

    mutating func readCount() throws -> Int {
        let value = try readUInt64()
        guard let count = Int(exactly: value) else { throw GGUFLoaderError.invalidSize }
        return count
    }

    mutating func readString() throws -> String {
        let length = try readCount()
        guard length >= 0, length <= data.count - offset else {
            throw GGUFLoaderError.truncated
        }
        let end = offset + length
        let value = String(data: data[offset..<end], encoding: .utf8)
        offset = end
        guard let value else { throw GGUFLoaderError.invalidText }
        return value
    }

    mutating func skip(bytes: Int) throws {
        guard bytes >= 0, bytes <= data.count - offset else {
            throw GGUFLoaderError.truncated
        }
        offset += bytes
    }

    mutating func readMetadataValue(type: UInt32) throws -> GGUFMetadataValue {
        switch type {
        case 0: return .uint8(try readUInt8())
        case 1: return .int8(Int8(bitPattern: try readUInt8()))
        case 2: return .uint16(try readUInt16())
        case 3: return .int16(Int16(bitPattern: try readUInt16()))
        case 4: return .uint32(try readUInt32())
        case 5: return .int32(Int32(bitPattern: try readUInt32()))
        case 6: return .float32(Float(bitPattern: try readUInt32()))
        case 7: return .boolean(try readUInt8() != 0)
        case 8: return .string(try readString())
        case 9:
            let elementType = try readUInt32()
            let count = try readCount()
            var values = [GGUFMetadataValue]()
            values.reserveCapacity(count)
            for _ in 0..<count {
                values.append(try readMetadataValue(type: elementType))
            }
            return .array(values)
        case 10: return .uint64(try readUInt64())
        case 11: return .int64(Int64(bitPattern: try readUInt64()))
        case 12: return .float64(Double(bitPattern: try readUInt64()))
        default: throw GGUFLoaderError.unsupportedMetadataType(type)
        }
    }
}

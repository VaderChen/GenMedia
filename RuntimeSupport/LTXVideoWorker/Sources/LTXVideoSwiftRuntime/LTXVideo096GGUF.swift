import Foundation
import MLX
import MLXNN

public struct LTXVideo096GGUFLoadReport: Sendable {
    public let sourceTensorCount: Int
    public let loadedParameterCount: Int
    public let quantizedTensorCount: Int
    public let sourceTypes: [String: Int]

    public init(
        sourceTensorCount: Int,
        loadedParameterCount: Int,
        quantizedTensorCount: Int,
        sourceTypes: [String: Int]
    ) {
        self.sourceTensorCount = sourceTensorCount
        self.loadedParameterCount = loadedParameterCount
        self.quantizedTensorCount = quantizedTensorCount
        self.sourceTypes = sourceTypes
    }
}

public struct LTXGGUFMetadata: Sendable {
    public let strings: [String: String]
    public let numbers: [String: Double]
    public let stringArrays: [String: [String]]
    public let numberArrays: [String: [Double]]

    public init(
        strings: [String: String],
        numbers: [String: Double],
        stringArrays: [String: [String]],
        numberArrays: [String: [Double]]
    ) {
        self.strings = strings
        self.numbers = numbers
        self.stringArrays = stringArrays
        self.numberArrays = numberArrays
    }
}

public enum LTXVideo096GGUFError: LocalizedError, Sendable {
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
    case invalidGroupSize(Int)

    public var errorDescription: String? {
        switch self {
        case let .missingFile(url): "找不到 LTX-Video 0.9.6 GGUF：\(url.path)"
        case .invalidMagic: "LTX-Video 0.9.6 GGUF 標頭不正確。"
        case let .unsupportedVersion(version): "LTX GGUF 版本不支援：\(version)"
        case .truncated: "LTX GGUF 檔案內容不完整。"
        case .invalidSize: "LTX GGUF tensor 尺寸無效。"
        case .invalidText: "LTX GGUF 包含無法解碼的文字欄位。"
        case .invalidAlignment: "LTX GGUF 對齊設定無效。"
        case let .unsupportedMetadataType(type): "LTX GGUF metadata 型別不支援：\(type)"
        case let .unsupportedTensorType(type, name): "LTX GGUF tensor \(name) 型別不支援：\(type)"
        case let .invalidTensor(name): "LTX GGUF tensor 無效：\(name)"
        case let .duplicateWeight(name): "LTX GGUF 權重重複：\(name)"
        case let .missingWeights(names): "LTX Transformer 缺少權重：\(names.prefix(12).joined(separator: "、"))"
        case let .weightShapeMismatch(name, expected, actual):
            "LTX 權重 \(name) shape 不一致：預期 \(expected)，實際 \(actual)"
        case let .invalidGroupSize(size): "LTX GGUF group size 不支援：\(size)"
        }
    }
}

public enum LTXVideo096GGUFQuantization {
    public static let groupSize = 64

    public static func targetBits(for typeName: String) -> Int? {
        switch typeName {
        case "Q3_K", "Q4_K": 4
        case "Q5_K", "Q6_K", "IQ4_XS": 8
        default: nil
        }
    }
}

func ltxQuantizeSorted(
    model: Module,
    groupSize: Int,
    bitsForPath: (String, Module) -> Int?
) {
    let updates = model
        .leafModules()
        .flattened()
        .compactMap { path, module -> (String, Module)? in
            guard let bits = bitsForPath(path, module),
                  let quantized = quantizeSingle(
                      layer: module,
                      groupSize: groupSize,
                      bits: bits,
                      mode: .affine
                  ) else {
                return nil
            }
            return (path, quantized)
        }
        .sorted { $0.0 < $1.0 }
    let moduleTree = ModuleChildren.unflattened(updates)
    model.update(modules: moduleTree)
}

public enum LTXVideo096TransformerWeightLoader {
    public static func load(
        from weightsURL: URL,
        configuration: LTXVideo096TransformerConfiguration = try! .init()
    ) throws -> (model: LTXVideo096Transformer, report: LTXVideo096GGUFLoadReport) {
        let file = try LTXVideo096GGUFFile(url: weightsURL)
        let layerCount = file.tensors.compactMap { tensor -> Int? in
            let components = tensor.name.split(separator: ".")
            guard components.count > 1,
                  components[0] == "transformer_blocks",
                  let index = Int(components[1]) else { return nil }
            return index
        }.max().map { $0 + 1 } ?? configuration.numLayers
        let configuration = try LTXVideo096TransformerConfiguration(
            numLayers: layerCount,
            dimension: configuration.dimension,
            numHeads: configuration.numHeads,
            headDimension: configuration.headDimension,
            latentChannels: configuration.latentChannels,
            captionChannels: configuration.captionChannels,
            timestepDimension: configuration.timestepDimension,
            timestepScaleMultiplier: configuration.timestepScaleMultiplier,
            ropeTheta: configuration.ropeTheta,
            ropeMaxPositions: configuration.ropeMaxPositions,
            normEps: configuration.normEps
        )
        let model = LTXVideo096Transformer(configuration: configuration)
        let q4Names = Set(file.tensors.filter { ["Q3_K", "Q4_K"].contains($0.typeName) }
            .map { quantizationPath($0.name) })
        let q8Names = Set(file.tensors.filter { ["Q5_K", "Q6_K", "IQ4_XS"].contains($0.typeName) }
            .map { quantizationPath($0.name) })
        if !q4Names.isEmpty || !q8Names.isEmpty {
            ltxQuantizeSorted(
                model: model,
                groupSize: LTXVideo096GGUFQuantization.groupSize
            ) { path, module in
                guard module is Linear else { return nil }
                if q4Names.contains(path) { return 4 }
                if q8Names.contains(path) { return 8 }
                return nil
            }
        }

        let expected = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1) }
        )
        var converted: [String: MLXArray] = [:]
        var sourceTypes: [String: Int] = [:]
        var quantizedCount = 0
        for tensor in file.tensors {
            sourceTypes[tensor.typeName, default: 0] += 1
            let key = canonicalKey(tensor.name)
            guard expected[key] != nil else { continue }
            let source = try file.array(for: tensor)
            if let bits = LTXVideo096GGUFQuantization.targetBits(for: tensor.typeName) {
                guard tensor.shape.count == 2,
                      tensor.shape[0] % 32 == 0,
                      tensor.shape[1] % LTXVideo096GGUFQuantization.groupSize == 0 else {
                    throw LTXVideo096GGUFError.invalidTensor(tensor.name)
                }
                let quantized = MLX.quantized(
                    source.asType(.float32),
                    groupSize: LTXVideo096GGUFQuantization.groupSize,
                    bits: bits,
                    mode: .affine
                )
                try insert(quantized.wq, name: key, expected: expected, into: &converted)
                let prefix = String(key.dropLast(".weight".count))
                try insert(quantized.scales, name: prefix + ".scales", expected: expected, into: &converted)
                if let biases = quantized.biases {
                    try insert(biases, name: prefix + ".biases", expected: expected, into: &converted)
                }
                quantizedCount += 1
            } else {
                guard source.shape == expected[key]!.shape else {
                    throw LTXVideo096GGUFError.weightShapeMismatch(
                        name: key,
                        expected: expected[key]!.shape,
                        actual: source.shape
                    )
                }
                let value = source.dtype == .uint32
                    ? source
                    : source.asType(.bfloat16)
                try insert(value, name: key, expected: expected, into: &converted)
            }
        }

        let missing = Set(expected.keys).subtracting(converted.keys).sorted()
        guard missing.isEmpty else { throw LTXVideo096GGUFError.missingWeights(missing) }
        let parameterTree = ModuleParameters.unflattened(
            converted.sorted { $0.key < $1.key }
        )
        try model.update(parameters: parameterTree, verify: .all)
        MLX.eval(model)
        return (
            model,
            LTXVideo096GGUFLoadReport(
                sourceTensorCount: file.tensors.count,
                loadedParameterCount: converted.count,
                quantizedTensorCount: quantizedCount,
                sourceTypes: sourceTypes
            )
        )
    }

    private static func insert(
        _ value: MLXArray,
        name: String,
        expected: [String: MLXArray],
        into converted: inout [String: MLXArray]
    ) throws {
        guard expected[name] != nil else { return }
        guard converted[name] == nil else { throw LTXVideo096GGUFError.duplicateWeight(name) }
        guard value.shape == expected[name]!.shape else {
            throw LTXVideo096GGUFError.weightShapeMismatch(
                name: name,
                expected: expected[name]!.shape,
                actual: value.shape
            )
        }
        converted[name] = value
    }

    private static func parameterPath(_ name: String) -> String {
        name.hasSuffix(".weight")
            ? String(name.dropLast(".weight".count))
            : name
    }

    private static func quantizationPath(_ name: String) -> String {
        parameterPath(canonicalKey(name))
    }

    private static func canonicalKey(_ name: String) -> String {
        name
            .replacingOccurrences(of: ".to_out.0.", with: ".to_out.")
            .replacingOccurrences(of: ".ff.net.0.proj.", with: ".ff.proj_in.")
            .replacingOccurrences(of: ".ff.net.2.", with: ".ff.proj_out.")
    }
}

struct LTXVideo096GGUFTensor {
    let name: String
    let shape: [Int]
    let typeCode: UInt32
    let offset: UInt64

    var typeName: String {
        switch typeCode {
        case 0: "F32"
        case 1: "F16"
        case 11: "Q3_K"
        case 12: "Q4_K"
        case 13: "Q5_K"
        case 14: "Q6_K"
        case 23: "IQ4_XS"
        case 30: "BF16"
        default: "TYPE_\(typeCode)"
        }
    }
}

final class LTXVideo096GGUFFile {
    let data: Data
    let tensorDataOffset: Int
    let tensors: [LTXVideo096GGUFTensor]
    let stringMetadata: [String: String]
    let numberMetadata: [String: Double]
    let stringArrayMetadata: [String: [String]]
    let numberArrayMetadata: [String: [Double]]

    init(url: URL) throws {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw LTXVideo096GGUFError.missingFile(url)
        }
        do {
            data = try Data(contentsOf: url, options: [.mappedIfSafe])
        } catch {
            throw LTXVideo096GGUFError.truncated
        }
        var reader = LTXVideo096GGUFReader(data: data)
        guard try reader.readUInt32() == 0x4655_4747 else {
            throw LTXVideo096GGUFError.invalidMagic
        }
        let version = try reader.readUInt32()
        guard version == 2 || version == 3 else {
            throw LTXVideo096GGUFError.unsupportedVersion(version)
        }
        let tensorCount = try reader.readCount()
        let metadataCount = try reader.readCount()
        var alignment = 32
        var stringMetadata: [String: String] = [:]
        var numberMetadata: [String: Double] = [:]
        var stringArrayMetadata: [String: [String]] = [:]
        var numberArrayMetadata: [String: [Double]] = [:]
        for _ in 0..<metadataCount {
            let key = try reader.readString()
            let type = try reader.readUInt32()
            switch type {
            case 4:
                let value = try reader.readUInt32()
                numberMetadata[key] = Double(value)
                if key == "general.alignment" { alignment = Int(value) }
            case 5:
                numberMetadata[key] = Double(try reader.readInt32())
            case 6:
                numberMetadata[key] = Double(try reader.readFloat32())
            case 8:
                stringMetadata[key] = try reader.readString()
            case 10:
                let value = try reader.readUInt64()
                numberMetadata[key] = Double(value)
                if key == "general.alignment" {
                    guard value <= UInt64(Int.max) else { throw LTXVideo096GGUFError.invalidAlignment }
                    alignment = Int(value)
                }
            case 11:
                numberMetadata[key] = Double(try reader.readInt64())
            case 12:
                numberMetadata[key] = try reader.readFloat64()
            case 9:
                let elementType = try reader.readUInt32()
                let count = try reader.readCount()
                if elementType == 8 {
                    var values: [String] = []
                    values.reserveCapacity(count)
                    for _ in 0..<count { values.append(try reader.readString()) }
                    stringArrayMetadata[key] = values
                } else if [4, 5, 6, 10, 11, 12].contains(elementType) {
                    var values: [Double] = []
                    values.reserveCapacity(count)
                    for _ in 0..<count {
                        switch elementType {
                        case 4: values.append(Double(try reader.readUInt32()))
                        case 5: values.append(Double(try reader.readInt32()))
                        case 6: values.append(Double(try reader.readFloat32()))
                        case 10: values.append(Double(try reader.readUInt64()))
                        case 11: values.append(Double(try reader.readInt64()))
                        case 12: values.append(try reader.readFloat64())
                        default: break
                        }
                    }
                    numberArrayMetadata[key] = values
                } else {
                    for _ in 0..<count { try reader.skipMetadataValue(type: elementType) }
                }
            default:
                try reader.skipMetadataValue(type: type)
            }
        }
        guard alignment > 0 else { throw LTXVideo096GGUFError.invalidAlignment }

        var parsed: [LTXVideo096GGUFTensor] = []
        parsed.reserveCapacity(tensorCount)
        for _ in 0..<tensorCount {
            let name = try reader.readString()
            let dimensionCount = try reader.readUInt32()
            guard dimensionCount > 0, dimensionCount <= 8 else { throw LTXVideo096GGUFError.invalidSize }
            var dimensions: [Int] = []
            for _ in 0..<dimensionCount { dimensions.append(try reader.readCount()) }
            parsed.append(LTXVideo096GGUFTensor(
                name: name,
                shape: dimensions.reversed(),
                typeCode: try reader.readUInt32(),
                offset: try reader.readUInt64()
            ))
        }
        let remainder = reader.offset % alignment
        if remainder != 0 { try reader.skip(bytes: alignment - remainder) }
        tensorDataOffset = reader.offset
        tensors = parsed
        self.stringMetadata = stringMetadata
        self.numberMetadata = numberMetadata
        self.stringArrayMetadata = stringArrayMetadata
        self.numberArrayMetadata = numberArrayMetadata
    }

    func array(for tensor: LTXVideo096GGUFTensor) throws -> MLXArray {
        let count = try product(tensor.shape, name: tensor.name)
        let bytes = try byteCount(typeCode: tensor.typeCode, count: count)
        guard tensor.offset <= UInt64(data.count - tensorDataOffset), tensor.offset <= UInt64(Int.max) else {
            throw LTXVideo096GGUFError.invalidTensor(tensor.name)
        }
        let start = tensorDataOffset + Int(tensor.offset)
        guard start >= 0, start <= data.count, bytes <= data.count - start else {
            throw LTXVideo096GGUFError.invalidTensor(tensor.name)
        }
        let raw = data.subdata(in: start..<(start + bytes))
        switch tensor.typeCode {
        case 0: return MLXArray(raw, tensor.shape, dtype: .float32)
        case 1: return MLXArray(raw, tensor.shape, dtype: .float16)
        case 30: return MLXArray(raw, tensor.shape, dtype: .bfloat16)
        case 11, 12, 13, 14, 23:
            let values = try decodeQuantized(raw, typeCode: tensor.typeCode, count: count)
            return MLXArray(
                Data(bytes: values, count: values.count * MemoryLayout<Float>.stride),
                tensor.shape,
                dtype: .float32
            )
        default: throw LTXVideo096GGUFError.unsupportedTensorType(tensor.typeCode, tensor.name)
        }
    }

    private func decodeQuantized(_ raw: Data, typeCode: UInt32, count: Int) throws -> [Float] {
        let blockSize: Int
        switch typeCode {
        case 11: blockSize = 110
        case 12: blockSize = 144
        case 13: blockSize = 176
        case 14: blockSize = 210
        case 23: blockSize = 136
        default: throw LTXVideo096GGUFError.unsupportedTensorType(typeCode, "")
        }
        guard count % 256 == 0, raw.count == count / 256 * blockSize else {
            throw LTXVideo096GGUFError.invalidSize
        }
        var values = [Float](repeating: 0, count: count)
        for block in 0..<(count / 256) {
            if typeCode == 11 {
                decodeQ3K(raw, offset: block * 110, output: &values, outputOffset: block * 256)
            } else if typeCode == 12 {
                decodeQ4K(raw, offset: block * 144, output: &values, outputOffset: block * 256)
            } else if typeCode == 13 {
                decodeQ5K(raw, offset: block * 176, output: &values, outputOffset: block * 256)
            } else if typeCode == 14 {
                decodeQ6K(raw, offset: block * 210, output: &values, outputOffset: block * 256)
            } else {
                decodeIQ4XS(raw, offset: block * 136, output: &values, outputOffset: block * 256)
            }
        }
        return values
    }

    private func decodeQ4K(_ raw: Data, offset: Int, output: inout [Float], outputOffset: Int) {
        let scale = float16(raw, at: offset)
        let minimumScale = float16(raw, at: offset + 2)
        var scaleIndex = 0
        for segment in 0..<4 {
            let first = kScaleAndMin(index: scaleIndex, raw: raw, offset: offset + 4)
            let second = kScaleAndMin(index: scaleIndex + 1, raw: raw, offset: offset + 4)
            let sourceOffset = offset + 16 + segment * 32
            let destinationOffset = outputOffset + segment * 64
            for index in 0..<32 {
                let packed = raw[sourceOffset + index]
                output[destinationOffset + index] = scale * Float(first.scale) * Float(packed & 15)
                    - minimumScale * Float(first.minimum)
                output[destinationOffset + 32 + index] = scale * Float(second.scale) * Float(packed >> 4)
                    - minimumScale * Float(second.minimum)
            }
            scaleIndex += 2
        }
    }

    private func decodeQ3K(_ raw: Data, offset: Int, output: inout [Float], outputOffset: Int) {
        let scale = float16(raw, at: offset + 108)
        let scalesOffset = offset + 96
        let quantizedOffset = offset + 32
        var auxiliary = [UInt32](repeating: 0, count: 4)
        for index in 0..<3 {
            auxiliary[index] = uint32(raw, at: scalesOffset + index * 4)
        }
        let mask1: UInt32 = 0x03030303
        let mask2: UInt32 = 0x0f0f0f0f
        let temporary = auxiliary[2]
        auxiliary[2] = ((auxiliary[0] >> 4) & mask2) | (((temporary >> 4) & mask1) << 4)
        auxiliary[3] = ((auxiliary[1] >> 4) & mask2) | (((temporary >> 6) & mask1) << 4)
        auxiliary[0] = (auxiliary[0] & mask2) | (((temporary >> 0) & mask1) << 4)
        auxiliary[1] = (auxiliary[1] & mask2) | (((temporary >> 2) & mask1) << 4)

        var scaleIndex = 0
        var highBitMask: UInt8 = 1
        for half in 0..<2 {
            let quantizedHalfOffset = quantizedOffset + half * 32
            let outputHalfOffset = outputOffset + half * 128
            var shift = 0
            for _ in 0..<4 {
                let outputChunkOffset = outputHalfOffset + (shift / 2) * 32
                let firstScale = Int8(bitPattern: UInt8(truncatingIfNeeded:
                    auxiliary[scaleIndex / 4] >> ((scaleIndex % 4) * 8)))
                scaleIndex += 1
                let first = scale * (Float(firstScale) - 32)
                for index in 0..<16 {
                    let low = (raw[quantizedHalfOffset + index] >> shift) & 3
                    let high = (raw[offset + index] & highBitMask) == 0 ? 4 : 0
                    output[outputChunkOffset + index] = first * Float(Int(low) - high)
                }
                let secondScale = Int8(bitPattern: UInt8(truncatingIfNeeded:
                    auxiliary[scaleIndex / 4] >> ((scaleIndex % 4) * 8)))
                scaleIndex += 1
                let second = scale * (Float(secondScale) - 32)
                for index in 0..<16 {
                    let low = (raw[quantizedHalfOffset + index + 16] >> shift) & 3
                    let high = (raw[offset + index + 16] & highBitMask) == 0 ? 4 : 0
                    output[outputChunkOffset + 16 + index] = second * Float(Int(low) - high)
                }
                shift += 2
                highBitMask <<= 1
            }
        }
    }

    private func decodeQ5K(_ raw: Data, offset: Int, output: inout [Float], outputOffset: Int) {
        let scale = float16(raw, at: offset)
        let minimumScale = float16(raw, at: offset + 2)
        let scalesOffset = offset + 4
        let highBitsOffset = offset + 16
        let quantizedOffset = offset + 48
        var scaleIndex = 0

        for segment in 0..<4 {
            let firstScale = kScaleAndMin(index: scaleIndex, raw: raw, offset: scalesOffset)
            let secondScale = kScaleAndMin(index: scaleIndex + 1, raw: raw, offset: scalesOffset)
            let segmentOffset = quantizedOffset + segment * 32
            let highBit1 = UInt8(1 << (segment * 2))
            let highBit2 = UInt8(2 << (segment * 2))
            let firstOutput = outputOffset + segment * 64
            for index in 0..<32 {
                let high = raw[highBitsOffset + index]
                let firstQuantized = Int(raw[segmentOffset + index] & 0x0f)
                    + ((high & highBit1) == 0 ? 0 : 16)
                let secondQuantized = Int(raw[segmentOffset + index] >> 4)
                    + ((high & highBit2) == 0 ? 0 : 16)
                output[firstOutput + index] = scale * Float(firstScale.scale) * Float(firstQuantized)
                    - minimumScale * Float(firstScale.minimum)
                output[firstOutput + 32 + index] = scale * Float(secondScale.scale) * Float(secondQuantized)
                    - minimumScale * Float(secondScale.minimum)
            }
            scaleIndex += 2
        }
    }

    private func decodeQ6K(_ raw: Data, offset: Int, output: inout [Float], outputOffset: Int) {
        let scale = float16(raw, at: offset + 208)
        for half in 0..<2 {
            let destination = outputOffset + half * 128
            let lowOffset = offset + half * 64
            let highOffset = offset + 128 + half * 32
            let scaleOffset = offset + 192 + half * 8
            for index in 0..<32 {
                let high = raw[highOffset + index]
                let first = Int((raw[lowOffset + index] & 15) | ((high & 3) << 4)) - 32
                let second = Int((raw[lowOffset + 32 + index] & 15) | (((high >> 2) & 3) << 4)) - 32
                let third = Int((raw[lowOffset + index] >> 4) | (((high >> 4) & 3) << 4)) - 32
                let fourth = Int((raw[lowOffset + 32 + index] >> 4) | (((high >> 6) & 3) << 4)) - 32
                output[destination + index] = scale * Float(Int8(bitPattern: raw[scaleOffset + index / 16])) * Float(first)
                output[destination + 32 + index] = scale * Float(Int8(bitPattern: raw[scaleOffset + 2 + index / 16])) * Float(second)
                output[destination + 64 + index] = scale * Float(Int8(bitPattern: raw[scaleOffset + 4 + index / 16])) * Float(third)
                output[destination + 96 + index] = scale * Float(Int8(bitPattern: raw[scaleOffset + 6 + index / 16])) * Float(fourth)
            }
        }
    }

    private func decodeIQ4XS(_ raw: Data, offset: Int, output: inout [Float], outputOffset: Int) {
        let scale = float16(raw, at: offset)
        let scalesHigh = uint16(raw, at: offset + 2)
        let scalesLowOffset = offset + 4
        let quantizedOffset = offset + 8
        let codebook: [Int] = [
            -127, -104, -83, -65, -49, -35, -22, -10,
            1, 13, 25, 38, 53, 69, 89, 113
        ]
        for index in 0..<8 {
            let low = (raw[scalesLowOffset + index / 2] >> UInt8(4 * (index % 2))) & 0x0f
            let high = (scalesHigh >> UInt16(2 * index)) & 0x03
            let scaleValue = Float(Int(UInt16(low) | (high << 4)) - 32)
            let dequantizedScale = scale * scaleValue
            let destination = outputOffset + index * 32
            let source = quantizedOffset + index * 16
            for element in 0..<16 {
                let packed = raw[source + element]
                output[destination + element] = dequantizedScale * Float(codebook[Int(packed & 0x0f)])
                output[destination + 16 + element] = dequantizedScale * Float(codebook[Int(packed >> 4)])
            }
        }
    }

    private func kScaleAndMin(index: Int, raw: Data, offset: Int) -> (scale: UInt8, minimum: UInt8) {
        if index < 4 { return (raw[offset + index] & 63, raw[offset + index + 4] & 63) }
        return (
            (raw[offset + index + 4] & 15) | ((raw[offset + index - 4] >> 6) << 4),
            (raw[offset + index + 4] >> 4) | ((raw[offset + index] >> 6) << 4)
        )
    }

    private func float16(_ data: Data, at offset: Int) -> Float {
        Float(Float16(bitPattern: UInt16(data[offset]) | UInt16(data[offset + 1]) << 8))
    }

    private func byteCount(typeCode: UInt32, count: Int) throws -> Int {
        switch typeCode {
        case 0: return try multiplied(count, by: 4)
        case 1, 30: return try multiplied(count, by: 2)
        case 11: return try blockBytes(count, size: 110)
        case 12: return try blockBytes(count, size: 144)
        case 13: return try blockBytes(count, size: 176)
        case 14: return try blockBytes(count, size: 210)
        case 23: return try blockBytes(count, size: 136)
        default: throw LTXVideo096GGUFError.unsupportedTensorType(typeCode, "")
        }
    }

    private func product(_ values: [Int], name: String) throws -> Int {
        guard !values.isEmpty else { throw LTXVideo096GGUFError.invalidTensor(name) }
        var result = 1
        for value in values {
            guard value > 0 else { throw LTXVideo096GGUFError.invalidTensor(name) }
            let multiplication = result.multipliedReportingOverflow(by: value)
            guard !multiplication.overflow else { throw LTXVideo096GGUFError.invalidSize }
            result = multiplication.partialValue
        }
        return result
    }

    private func multiplied(_ value: Int, by factor: Int) throws -> Int {
        let result = value.multipliedReportingOverflow(by: factor)
        guard !result.overflow else { throw LTXVideo096GGUFError.invalidSize }
        return result.partialValue
    }

    private func blockBytes(_ count: Int, size: Int) throws -> Int {
        guard count % 256 == 0 else { throw LTXVideo096GGUFError.invalidSize }
        return try multiplied(count / 256, by: size)
    }

    private func uint32(_ data: Data, at offset: Int) -> UInt32 {
        UInt32(data[offset]) | UInt32(data[offset + 1]) << 8
            | UInt32(data[offset + 2]) << 16 | UInt32(data[offset + 3]) << 24
    }

    private func uint16(_ data: Data, at offset: Int) -> UInt16 {
        UInt16(data[offset]) | UInt16(data[offset + 1]) << 8
    }
}

public enum LTXGGUFInspector {
    public static func metadata(from url: URL) throws -> LTXGGUFMetadata {
        let file = try LTXVideo096GGUFFile(url: url)
        return LTXGGUFMetadata(
            strings: file.stringMetadata,
            numbers: file.numberMetadata,
            stringArrays: file.stringArrayMetadata,
            numberArrays: file.numberArrayMetadata
        )
    }
}

private struct LTXVideo096GGUFReader {
    let data: Data
    var offset = 0

    mutating func readUInt8() throws -> UInt8 {
        guard offset < data.count else { throw LTXVideo096GGUFError.truncated }
        defer { offset += 1 }
        return data[offset]
    }

    mutating func readUInt32() throws -> UInt32 {
        UInt32(try readUInt8()) | UInt32(try readUInt8()) << 8
            | UInt32(try readUInt8()) << 16 | UInt32(try readUInt8()) << 24
    }

    mutating func readUInt64() throws -> UInt64 {
        var value: UInt64 = 0
        for index in 0..<8 { value |= UInt64(try readUInt8()) << UInt64(index * 8) }
        return value
    }

    mutating func readInt32() throws -> Int32 {
        Int32(bitPattern: try readUInt32())
    }

    mutating func readInt64() throws -> Int64 {
        Int64(bitPattern: try readUInt64())
    }

    mutating func readFloat32() throws -> Float {
        Float(bitPattern: try readUInt32())
    }

    mutating func readFloat64() throws -> Double {
        Double(bitPattern: try readUInt64())
    }

    mutating func readCount() throws -> Int {
        let value = try readUInt64()
        guard value <= UInt64(Int.max) else { throw LTXVideo096GGUFError.invalidSize }
        return Int(value)
    }

    mutating func readString() throws -> String {
        let length = try readCount()
        guard length <= data.count - offset else { throw LTXVideo096GGUFError.truncated }
        let value = data.subdata(in: offset..<(offset + length))
        offset += length
        guard let string = String(data: value, encoding: .utf8) else {
            throw LTXVideo096GGUFError.invalidText
        }
        return string
    }

    mutating func skip(bytes: Int) throws {
        guard bytes >= 0, bytes <= data.count - offset else { throw LTXVideo096GGUFError.truncated }
        offset += bytes
    }

    mutating func skipMetadataValue(type: UInt32) throws {
        switch type {
        case 0, 1, 7: try skip(bytes: 1)
        case 2, 3: try skip(bytes: 2)
        case 4, 5, 6: try skip(bytes: 4)
        case 8: _ = try readString()
        case 9:
            let elementType = try readUInt32()
            let count = try readCount()
            for _ in 0..<count { try skipMetadataValue(type: elementType) }
        case 10, 11, 12: try skip(bytes: 8)
        default: throw LTXVideo096GGUFError.unsupportedMetadataType(type)
        }
    }
}

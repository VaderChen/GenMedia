import Foundation
import MLX

public enum GGUFDequantizer {
    public static func typeName(typeCode: UInt32) -> String {
        switch typeCode {
        case 0: "F32"
        case 1: "F16"
        case 2: "Q4_0"
        case 3: "Q4_1"
        case 4: "Q4_2"
        case 5: "Q4_3"
        case 6: "Q5_0"
        case 7: "Q5_1"
        case 8: "Q8_0"
        case 9: "Q8_1"
        case 10: "Q2_K"
        case 11: "Q3_K"
        case 12: "Q4_K"
        case 13: "Q5_K"
        case 14: "Q6_K"
        case 24: "I8"
        case 25: "I16"
        case 26: "I32"
        case 27: "I64"
        case 28: "F64"
        case 30: "BF16"
        case 39: "MXFP4"
        case 40: "NVFP4"
        case 41: "Q1_0"
        case 42: "Q2_0"
        default: "TYPE_\(typeCode)"
        }
    }

    public static func isQuantized(typeName: String) -> Bool {
        switch typeName.uppercased() {
        case "Q1_0", "Q2_0", "Q2_K", "Q3_K", "Q4_0", "Q4_1", "Q4_K",
             "Q5_0", "Q5_1", "Q5_K", "Q6_K", "Q8_0", "Q8_1", "MXFP4", "NVFP4":
            true
        default:
            false
        }
    }

    public static func isMaterializable(typeName: String) -> Bool {
        switch typeName.uppercased() {
        case "F32", "F16", "BF16", "F64", "I8", "I16", "I32", "I64",
             "Q1_0", "Q2_0", "Q2_K", "Q3_K", "Q4_0", "Q4_1", "Q4_K",
             "MXFP4", "Q5_0", "Q5_1", "Q5_K", "Q6_K", "Q8_0", "Q8_1":
            true
        default:
            false
        }
    }

    public static func byteCount(typeCode: UInt32, elementCount: Int) throws -> Int {
        guard elementCount > 0 else { throw GGUFLoaderError.invalidSize }
        switch typeCode {
        case 0: return try multiplied(elementCount, by: 4)
        case 1, 25, 30: return try multiplied(elementCount, by: 2)
        case 24: return elementCount
        case 26: return try multiplied(elementCount, by: 4)
        case 27, 28: return try multiplied(elementCount, by: 8)
        case 2: return try blockByteCount(elementCount, elementsPerBlock: 32, bytesPerBlock: 18)
        case 3: return try blockByteCount(elementCount, elementsPerBlock: 32, bytesPerBlock: 20)
        case 6: return try blockByteCount(elementCount, elementsPerBlock: 32, bytesPerBlock: 22)
        case 7: return try blockByteCount(elementCount, elementsPerBlock: 32, bytesPerBlock: 24)
        case 8: return try blockByteCount(elementCount, elementsPerBlock: 32, bytesPerBlock: 34)
        case 9: return try blockByteCount(elementCount, elementsPerBlock: 32, bytesPerBlock: 36)
        case 10: return try blockByteCount(elementCount, elementsPerBlock: 256, bytesPerBlock: 84)
        case 11: return try blockByteCount(elementCount, elementsPerBlock: 256, bytesPerBlock: 110)
        case 12: return try blockByteCount(elementCount, elementsPerBlock: 256, bytesPerBlock: 144)
        case 13: return try blockByteCount(elementCount, elementsPerBlock: 256, bytesPerBlock: 176)
        case 14: return try blockByteCount(elementCount, elementsPerBlock: 256, bytesPerBlock: 210)
        case 39: return try blockByteCount(elementCount, elementsPerBlock: 32, bytesPerBlock: 17)
        case 41: return try blockByteCount(elementCount, elementsPerBlock: 128, bytesPerBlock: 18)
        case 42: return try blockByteCount(elementCount, elementsPerBlock: 64, bytesPerBlock: 18)
        default: throw GGUFLoaderError.unsupportedTensorType(typeCode, "")
        }
    }

    public static func array(
        raw: Data,
        typeCode: UInt32,
        shape: [Int],
        name: String
    ) throws -> MLXArray {
        let elementCount = try product(shape, name: name)
        let expectedByteCount: Int
        do {
            expectedByteCount = try byteCount(typeCode: typeCode, elementCount: elementCount)
        } catch let error as GGUFLoaderError {
            switch error {
            case .unsupportedTensorType:
                throw GGUFLoaderError.unsupportedTensorType(typeCode, name)
            default:
                throw error
            }
        }
        guard raw.count == expectedByteCount else {
            throw GGUFLoaderError.invalidTensor(name)
        }

        switch typeCode {
        case 0: return MLXArray(raw, shape, dtype: .float32)
        case 1: return MLXArray(raw, shape, dtype: .float16)
        case 24: return MLXArray(raw, shape, dtype: .int8)
        case 25: return MLXArray(raw, shape, dtype: .int16)
        case 26: return MLXArray(raw, shape, dtype: .int32)
        case 27: return MLXArray(raw, shape, dtype: .int64)
        case 28: return MLXArray(raw, shape, dtype: .float64)
        case 30: return MLXArray(raw, shape, dtype: .bfloat16)
        default:
            let values = try decodeQuantized(
                raw: raw,
                typeCode: typeCode,
                shape: shape,
                elementCount: elementCount,
                name: name
            )
            return MLXArray(
                Data(bytes: values, count: values.count * MemoryLayout<Float>.stride),
                shape,
                dtype: .float32
            )
        }
    }

    private static func decodeQuantized(
        raw: Data,
        typeCode: UInt32,
        shape: [Int],
        elementCount: Int,
        name: String
    ) throws -> [Float] {
        let elementsPerBlock: Int
        switch typeCode {
        case 2, 3, 6, 7, 8, 9, 39: elementsPerBlock = 32
        case 10, 11, 12, 13, 14: elementsPerBlock = 256
        case 41: elementsPerBlock = 128
        case 42: elementsPerBlock = 64
        default: throw GGUFLoaderError.unsupportedTensorType(typeCode, name)
        }
        guard let lastDimension = shape.last,
              lastDimension > 0,
              lastDimension % elementsPerBlock == 0,
              elementCount % elementsPerBlock == 0 else {
            throw GGUFLoaderError.invalidTensor(name)
        }

        let blockCount = elementCount / elementsPerBlock
        var values = [Float](repeating: 0, count: elementCount)
        for block in 0..<blockCount {
            let rawOffset: Int
            switch typeCode {
            case 2: rawOffset = block * 18
            case 3: rawOffset = block * 20
            case 6: rawOffset = block * 22
            case 7: rawOffset = block * 24
            case 8: rawOffset = block * 34
            case 9: rawOffset = block * 36
            case 10: rawOffset = block * 84
            case 11: rawOffset = block * 110
            case 12: rawOffset = block * 144
            case 13: rawOffset = block * 176
            case 14: rawOffset = block * 210
            case 39: rawOffset = block * 17
            case 41: rawOffset = block * 18
            case 42: rawOffset = block * 18
            default: throw GGUFLoaderError.unsupportedTensorType(typeCode, name)
            }
            switch typeCode {
            case 2: decodeQ4_0(raw, offset: rawOffset, output: &values, outputOffset: block * 32)
            case 3: decodeQ4_1(raw, offset: rawOffset, output: &values, outputOffset: block * 32)
            case 6: decodeQ5_0(raw, offset: rawOffset, output: &values, outputOffset: block * 32)
            case 7: decodeQ5_1(raw, offset: rawOffset, output: &values, outputOffset: block * 32)
            case 8: decodeQ8_0(raw, offset: rawOffset, output: &values, outputOffset: block * 32)
            case 9: decodeQ8_1(raw, offset: rawOffset, output: &values, outputOffset: block * 32)
            case 10: decodeQ2K(raw, offset: rawOffset, output: &values, outputOffset: block * 256)
            case 11: decodeQ3K(raw, offset: rawOffset, output: &values, outputOffset: block * 256)
            case 12: decodeQ4K(raw, offset: rawOffset, output: &values, outputOffset: block * 256)
            case 13: decodeQ5K(raw, offset: rawOffset, output: &values, outputOffset: block * 256)
            case 14: decodeQ6K(raw, offset: rawOffset, output: &values, outputOffset: block * 256)
            case 39: decodeMXFP4(raw, offset: rawOffset, output: &values, outputOffset: block * 32)
            case 41: decodeQ1_0(raw, offset: rawOffset, output: &values, outputOffset: block * 128)
            case 42: decodeQ2_0(raw, offset: rawOffset, output: &values, outputOffset: block * 64)
            default: throw GGUFLoaderError.unsupportedTensorType(typeCode, name)
            }
        }
        return values
    }

    private static func decodeQ4_0(_ raw: Data, offset: Int, output: inout [Float], outputOffset: Int) {
        let scale = float16(raw, at: offset)
        for index in 0..<16 {
            let packed = raw[offset + 2 + index]
            output[outputOffset + index] = Float(Int(packed & 0x0f) - 8) * scale
            output[outputOffset + 16 + index] = Float(Int(packed >> 4) - 8) * scale
        }
    }

    private static func decodeQ4_1(_ raw: Data, offset: Int, output: inout [Float], outputOffset: Int) {
        let scale = float16(raw, at: offset)
        let minimum = float16(raw, at: offset + 2)
        for index in 0..<16 {
            let packed = raw[offset + 4 + index]
            output[outputOffset + index] = Float(packed & 0x0f) * scale + minimum
            output[outputOffset + 16 + index] = Float(packed >> 4) * scale + minimum
        }
    }

    private static func decodeQ5_0(_ raw: Data, offset: Int, output: inout [Float], outputOffset: Int) {
        let scale = float16(raw, at: offset)
        let highBits = uint32(raw, at: offset + 2)
        for index in 0..<16 {
            let low = Int(raw[offset + 6 + index] & 0x0f)
            let high = Int(raw[offset + 6 + index] >> 4)
            output[outputOffset + index] = Float((low | (Int((highBits >> UInt32(index)) & 1) << 4)) - 16) * scale
            output[outputOffset + 16 + index] = Float((high | (Int((highBits >> UInt32(index + 16)) & 1) << 4)) - 16) * scale
        }
    }

    private static func decodeQ5_1(_ raw: Data, offset: Int, output: inout [Float], outputOffset: Int) {
        let scale = float16(raw, at: offset)
        let minimum = float16(raw, at: offset + 2)
        let highBits = uint32(raw, at: offset + 4)
        for index in 0..<16 {
            let low = Int(raw[offset + 8 + index] & 0x0f)
            let high = Int(raw[offset + 8 + index] >> 4)
            output[outputOffset + index] = Float(low | (Int((highBits >> UInt32(index)) & 1) << 4)) * scale + minimum
            output[outputOffset + 16 + index] = Float(high | (Int((highBits >> UInt32(index + 16)) & 1) << 4)) * scale + minimum
        }
    }

    private static func decodeQ8_0(_ raw: Data, offset: Int, output: inout [Float], outputOffset: Int) {
        let scale = float16(raw, at: offset)
        for index in 0..<32 {
            output[outputOffset + index] = Float(Int8(bitPattern: raw[offset + 2 + index])) * scale
        }
    }

    private static func decodeQ8_1(_ raw: Data, offset: Int, output: inout [Float], outputOffset: Int) {
        let scale = float16(raw, at: offset)
        for index in 0..<32 {
            output[outputOffset + index] = Float(Int8(bitPattern: raw[offset + 4 + index])) * scale
        }
    }

    private static func decodeMXFP4(_ raw: Data, offset: Int, output: inout [Float], outputOffset: Int) {
        let scale = Foundation.pow(2 as Float, Float(Int(raw[offset]) - 127))
        let lookup: [Float] = [
            0, 0.5, 1, 1.5, 2, 3, 4, 6,
            -0, -0.5, -1, -1.5, -2, -3, -4, -6
        ]
        for index in 0..<16 {
            let packed = raw[offset + 1 + index]
            output[outputOffset + index] = scale * lookup[Int(packed & 0x0f)]
            output[outputOffset + 16 + index] = scale * lookup[Int(packed >> 4)]
        }
    }

    private static func decodeQ1_0(_ raw: Data, offset: Int, output: inout [Float], outputOffset: Int) {
        let scale = float16(raw, at: offset)
        for index in 0..<128 {
            let bit = (raw[offset + 2 + index / 8] >> (index % 8)) & 1
            output[outputOffset + index] = bit == 0 ? -scale : scale
        }
    }

    private static func decodeQ2_0(_ raw: Data, offset: Int, output: inout [Float], outputOffset: Int) {
        let scale = float16(raw, at: offset)
        for index in 0..<64 {
            let quantized = (raw[offset + 2 + index / 4] >> ((index % 4) * 2)) & 3
            output[outputOffset + index] = Float(Int(quantized) - 1) * scale
        }
    }

    private static func decodeQ2K(_ raw: Data, offset: Int, output: inout [Float], outputOffset: Int) {
        let scale = float16(raw, at: offset)
        let minimumScale = float16(raw, at: offset + 2)
        let scalesOffset = offset + 4
        let quantizedOffset = offset + 20
        var scaleIndex = 0

        for half in 0..<2 {
            let quantizedHalfOffset = quantizedOffset + half * 32
            let outputHalfOffset = outputOffset + half * 128
            var shift = 0
            for _ in 0..<4 {
                let outputChunkOffset = outputHalfOffset + (shift / 2) * 32
                let firstScale = raw[scalesOffset + scaleIndex]
                scaleIndex += 1
                let first = scale * Float(firstScale & 0x0f)
                let firstMinimum = minimumScale * Float(firstScale >> 4)
                for index in 0..<16 {
                    let quantized = (raw[quantizedHalfOffset + index] >> shift) & 3
                    output[outputChunkOffset + index] = first * Float(quantized) - firstMinimum
                }
                let secondScale = raw[scalesOffset + scaleIndex]
                scaleIndex += 1
                let second = scale * Float(secondScale & 0x0f)
                let secondMinimum = minimumScale * Float(secondScale >> 4)
                for index in 0..<16 {
                    let quantized = (raw[quantizedHalfOffset + index + 16] >> shift) & 3
                    output[outputChunkOffset + 16 + index] = second * Float(quantized) - secondMinimum
                }
                shift += 2
            }
        }
    }

    private static func decodeQ3K(_ raw: Data, offset: Int, output: inout [Float], outputOffset: Int) {
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
                let firstScale = Int8(bitPattern: UInt8(truncatingIfNeeded: auxiliary[scaleIndex / 4] >> ((scaleIndex % 4) * 8)))
                scaleIndex += 1
                let first = scale * (Float(firstScale) - 32)
                for index in 0..<16 {
                    let low = (raw[quantizedHalfOffset + index] >> shift) & 3
                    let high = (raw[offset + index] & highBitMask) == 0 ? 4 : 0
                    output[outputChunkOffset + index] = first * Float(Int(low) - high)
                }
                let secondScale = Int8(bitPattern: UInt8(truncatingIfNeeded: auxiliary[scaleIndex / 4] >> ((scaleIndex % 4) * 8)))
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

    private static func decodeQ4K(_ raw: Data, offset: Int, output: inout [Float], outputOffset: Int) {
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
                output[firstOutput + index] = scale * Float(firstScale.scale) * Float(quantized & 0x0f)
                    - minimumScale * Float(firstScale.minimum)
                output[firstOutput + 32 + index] = scale * Float(secondScale.scale) * Float(quantized >> 4)
                    - minimumScale * Float(secondScale.minimum)
            }
            scaleIndex += 2
        }
    }

    private static func decodeQ5K(_ raw: Data, offset: Int, output: inout [Float], outputOffset: Int) {
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

    private static func decodeQ6K(_ raw: Data, offset: Int, output: inout [Float], outputOffset: Int) {
        let lowBitsOffset = offset
        let highBitsOffset = offset + 128
        let scalesOffset = offset + 192
        let scale = float16(raw, at: offset + 208)

        for half in 0..<2 {
            let valueOffset = outputOffset + half * 128
            let lowOffset = lowBitsOffset + half * 64
            let highOffset = highBitsOffset + half * 32
            let scaleOffset = scalesOffset + half * 8
            for index in 0..<32 {
                let high = raw[highOffset + index]
                let first = Int((raw[lowOffset + index] & 0x0f) | ((high & 0x03) << 4)) - 32
                let second = Int((raw[lowOffset + 32 + index] & 0x0f) | (((high >> 2) & 0x03) << 4)) - 32
                let third = Int((raw[lowOffset + index] >> 4) | (((high >> 4) & 0x03) << 4)) - 32
                let fourth = Int((raw[lowOffset + 32 + index] >> 4) | (((high >> 6) & 0x03) << 4)) - 32
                output[valueOffset + index] = scale * Float(Int8(bitPattern: raw[scaleOffset + index / 16])) * Float(first)
                output[valueOffset + 32 + index] = scale * Float(Int8(bitPattern: raw[scaleOffset + 2 + index / 16])) * Float(second)
                output[valueOffset + 64 + index] = scale * Float(Int8(bitPattern: raw[scaleOffset + 4 + index / 16])) * Float(third)
                output[valueOffset + 96 + index] = scale * Float(Int8(bitPattern: raw[scaleOffset + 6 + index / 16])) * Float(fourth)
            }
        }
    }

    private static func kScaleAndMin(index: Int, raw: Data, offset: Int) -> (scale: UInt8, minimum: UInt8) {
        if index < 4 {
            return (raw[offset + index] & 63, raw[offset + index + 4] & 63)
        }
        let scale = (raw[offset + index + 4] & 0x0f) | ((raw[offset + index - 4] >> 6) << 4)
        let minimum = (raw[offset + index + 4] >> 4) | ((raw[offset + index] >> 6) << 4)
        return (scale, minimum)
    }

    private static func float16(_ data: Data, at offset: Int) -> Float {
        Float(Float16(bitPattern: uint16(data, at: offset)))
    }

    private static func uint16(_ data: Data, at offset: Int) -> UInt16 {
        UInt16(data[offset]) | UInt16(data[offset + 1]) << 8
    }

    private static func uint32(_ data: Data, at offset: Int) -> UInt32 {
        UInt32(data[offset])
            | UInt32(data[offset + 1]) << 8
            | UInt32(data[offset + 2]) << 16
            | UInt32(data[offset + 3]) << 24
    }

    private static func product(_ values: [Int], name: String) throws -> Int {
        guard !values.isEmpty else { throw GGUFLoaderError.invalidTensor(name) }
        var result = 1
        for value in values {
            guard value > 0 else { throw GGUFLoaderError.invalidTensor(name) }
            let multiplication = result.multipliedReportingOverflow(by: value)
            guard !multiplication.overflow else { throw GGUFLoaderError.invalidSize }
            result = multiplication.partialValue
        }
        return result
    }

    private static func multiplied(_ value: Int, by factor: Int) throws -> Int {
        let result = value.multipliedReportingOverflow(by: factor)
        guard !result.overflow else { throw GGUFLoaderError.invalidSize }
        return result.partialValue
    }

    private static func blockByteCount(
        _ elementCount: Int,
        elementsPerBlock: Int,
        bytesPerBlock: Int
    ) throws -> Int {
        guard elementCount % elementsPerBlock == 0 else {
            throw GGUFLoaderError.invalidSize
        }
        return try multiplied(elementCount / elementsPerBlock, by: bytesPerBlock)
    }
}

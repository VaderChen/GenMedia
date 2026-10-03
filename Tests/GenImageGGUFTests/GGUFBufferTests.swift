import Foundation
import MLX
import Testing
@testable import GenImageGGUF

struct GGUFBufferTests {
    static let types: [UInt32] = [2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 39, 41, 42]

    @Test(arguments: types)
    func blockBoundariesAndUnalignedSlices(type: UInt32) throws {
        let elements = type == 41 ? 128 : type == 42 ? 64 : (10...14).contains(type) ? 256 : 32
        let bytes = try GGUFDequantizer.byteCount(typeCode: type, elementCount: elements)
        let blocks = (0..<3).map { block in
            Data((0..<bytes).map { UInt8(($0 * 13 + block * 7 + 3) % 64) })
        }
        let payload = blocks.reduce(into: Data()) { $0.append($1) }
        // A slice has a nonzero Data index and its base address need not be aligned.
        let unaligned = (Data([0xfe, 0xfd, 0xfc]) + payload).dropFirst(3)
        try Device.withDefaultDevice(.cpu) {
            let expected = try blocks.flatMap {
                try GGUFDequantizer.array(raw: $0, typeCode: type, shape: [elements], name: "block")
                    .asArray(Float.self).map(\.bitPattern)
            }
            for raw in [payload, unaligned] {
                let actual = try GGUFDequantizer.array(raw: raw, typeCode: type,
                    shape: [3, elements], name: "blocks")
                #expect(actual.shape == [3, elements])
                #expect(actual.asArray(Float.self).map(\.bitPattern) == expected)
            }
            for invalid in [Data(payload.dropLast()), payload + Data([0])] {
                #expect(throws: GGUFLoaderError.self) {
                    try GGUFDequantizer.array(raw: invalid, typeCode: type,
                        shape: [3, elements], name: "invalid")
                }
            }
        }
    }
}

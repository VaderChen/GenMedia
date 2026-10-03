import Foundation
import MLX
import Testing
@testable import LTXVideoSwiftRuntime

@Suite(.serialized)
struct LTXGemma4RotaryTests {
    @Test(arguments: [Float(1), 0.25], [DType.float32, .bfloat16, .float16])
    func preparedPositionsMatchScalarReference(fraction: Float, dtype: DType) {
        let dimension = 8, half = 4, count = 5
        let rotation = LTXGemma4TextEncoder.PreparedRotary(count: count, dimension: dimension,
            base: 10_000, fraction: fraction)
        // Q and K can have different head counts but share the same positions.
        for heads in [3, 1] {
            let input = MLXArray((0..<(2 * heads * count * dimension)).map { sin(Float($0)) })
                .reshaped([2, heads, count, dimension]).asType(dtype)
            let values = input.asArray(Float.self)
            var expected = values
            for row in 0..<(2 * heads * count) {
                let position = row % count, offset = row * dimension
                for i in 0..<Int(Float(half) * fraction) {
                    let angle = Float(position) * pow(Float(10_000), -2 * Float(i) / Float(dimension))
                    let a = values[offset + i], b = values[offset + half + i]
                    expected[offset + i] = a * cos(angle) - b * sin(angle)
                    expected[offset + half + i] = b * cos(angle) + a * sin(angle)
                }
            }
            let actual = rotation(input)
            let reference = MLXArray(expected).reshaped(input.shape).asType(dtype)
            #expect(actual.dtype == dtype && actual.shape == input.shape)
            #expect(max(abs(actual.asType(.float32) - reference.asType(.float32))).item(Float.self) < 3e-5)
            #expect(rotation(input).asArray(Float.self) == actual.asArray(Float.self))
        }
    }
}

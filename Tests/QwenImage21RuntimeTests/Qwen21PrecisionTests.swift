import MLX
import MLXFast
import Testing
@testable import QwenImage21Runtime

struct Qwen21PrecisionTests {
    @Test func zeroCenteredNormPreservesFP32ScaleUntilFinalCast() {
        let x = MLXArray((1...8).map(Float.init)).reshaped([1, 1, 8]).asType(.bfloat16)
        // Near -1 verifies small signed scales. Positive half-ULP values expose premature BF16 rounding.
        let weight = MLXArray([Float(-0.9921875), -1.0078125, 0.00390625, 0.01171875,
                               0.19140625, 0, -0.5, -0.75]).asType(.bfloat16)
        let fp32 = x.asType(.float32)
        let expected = (fp32 * rsqrt(mean(fp32 * fp32, axis: -1, keepDims: true) + 1e-6)
                        * (weight.asType(.float32) + 1)).asType(x.dtype)
        let actual = Qwen21Transformer.zeroCenteredRMSNorm(x, weight: weight, eps: 1e-6)
        let prematurelyRounded = MLXFast.rmsNorm(x, weight: weight + 1, eps: 1e-6)

        #expect(actual.dtype == .bfloat16)
        #expect(max(abs(actual.asType(.float32) - expected.asType(.float32))).item(Float.self) < 1e-6)
        #expect(max(abs(prematurelyRounded.asType(.float32) - expected.asType(.float32))).item(Float.self) > 0.001)
    }
}

import Foundation
import GenImageGGUF
import MLX

enum MiniMaxH3GGUFMetalQuantizer {
    static let environmentKey = "GENMEDIA_H3_USE_METAL_QUANTIZER"
    static let targetBits = 8
    static let targetGroupSize = 64
    static let supportedTypeCodes: Set<UInt32> = [2, 12, 13, 14]

    struct QuantizedTensor {
        let weights: MLXArray
        let scales: MLXArray
        let biases: MLXArray
    }

    static var isEnabledByDefault: Bool {
        guard let value = ProcessInfo.processInfo.environment[environmentKey]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        else {
            return true
        }
        return !["0", "false", "no", "off", "cpu"].contains(value)
    }

    static func supports(typeCode: UInt32) -> Bool {
        supportedTypeCodes.contains(typeCode)
    }

    private static let kernel = MLXFast.metalKernel(
        name: "genmedia_h3_gguf_direct_int8_quantize",
        inputNames: ["raw"],
        outputNames: ["wq", "scales", "biases"],
        source: """
            uint group_index = thread_position_in_grid.x;
            uint group_start = group_index * 64u;
            float values[64];
            values[0] = source_value(raw, sourceType, group_start);
            float minimum = values[0];
            float maximum = values[0];
            for (uint index = 1; index < 64u; ++index) {
                values[index] = source_value(raw, sourceType, group_start + index);
                minimum = min(minimum, values[index]);
                maximum = max(maximum, values[index]);
            }

            float scale = max((maximum - minimum) / 255.0f, 1e-7f);
            bool use_minimum_as_edge = abs(minimum) > abs(maximum);
            scale = use_minimum_as_edge ? scale : -scale;
            float edge = use_minimum_as_edge ? minimum : maximum;
            float initial_quantized_edge = round(edge / scale);
            float bias = initial_quantized_edge == 0.0f ? 0.0f : edge;
            scale = initial_quantized_edge == 0.0f
                ? scale
                : edge / initial_quantized_edge;

            scales[group_index] = scale;
            biases[group_index] = bias;
            for (uint word = 0; word < 16u; ++word) {
                uint result = 0u;
                uint start = word * 4u;
                for (uint index = 0; index < 4u; ++index) {
                    float quantized = round((values[start + index] - bias) / scale);
                    quantized = min(max(quantized, 0.0f), 255.0f);
                    result |= uint(quantized) << (index * 8u);
                }
                wq[group_index * 16u + word] = result;
            }
        """,
        header: """
            uint load_u16(device const uchar *raw, uint offset) {
                return uint(raw[offset]) | (uint(raw[offset + 1]) << 8);
            }

            float load_f16(device const uchar *raw, uint offset) {
                return float(as_type<half>((ushort) load_u16(raw, offset)));
            }

            int signed_byte(uchar value) {
                int result = int(value);
                return result >= 128 ? result - 256 : result;
            }

            uint qk_scale(device const uchar *raw, uint offset, uint index) {
                if (index < 4) {
                    return uint(raw[offset + index] & 63);
                }
                return uint((raw[offset + index + 4] & 15)
                    | ((raw[offset + index - 4] >> 6) << 4));
            }

            uint qk_minimum(device const uchar *raw, uint offset, uint index) {
                if (index < 4) {
                    return uint(raw[offset + index + 4] & 63);
                }
                return uint((raw[offset + index + 4] >> 4)
                    | ((raw[offset + index] >> 6) << 4));
            }

            float source_value(
                device const uchar *raw,
                uint source_type,
                uint element
            ) {
                if (source_type == 2) {
                    uint block = element / 32u;
                    uint source_index = element % 32u;
                    uint block_offset = block * 18u;
                    uchar packed = raw[block_offset + 2u + source_index % 16u];
                    uint quantized = source_index < 16u
                        ? uint(packed & 15)
                        : uint(packed >> 4);
                    return load_f16(raw, block_offset) * float(int(quantized) - 8);
                }

                uint block = element / 256u;
                uint source_index = element % 256u;
                uint block_offset = block * (source_type == 12 ? 144u
                    : source_type == 13 ? 176u : 210u);
                uint block_half = source_index / 128u;
                uint half_index = source_index % 128u;

                if (source_type == 12 || source_type == 13) {
                    uint segment = source_index / 64u;
                    bool upper = (source_index % 64u) >= 32u;
                    uint position = source_index % 32u;
                    uint scale_index = segment * 2u + (upper ? 1u : 0u);
                    uint quantized;
                    if (source_type == 12) {
                        quantized = upper
                            ? uint(raw[block_offset + 16u + segment * 32u + position] >> 4)
                            : uint(raw[block_offset + 16u + segment * 32u + position] & 15);
                    } else {
                        uchar high = raw[block_offset + 16u + position];
                        uint high_bit = upper ? 2u << (segment * 2u) : 1u << (segment * 2u);
                        quantized = upper
                            ? uint(raw[block_offset + 48u + segment * 32u + position] >> 4)
                            : uint(raw[block_offset + 48u + segment * 32u + position] & 15);
                        quantized += (high & high_bit) == 0 ? 0u : 16u;
                    }
                    uint source_scale = qk_scale(raw, block_offset + 4u, scale_index);
                    uint source_minimum = qk_minimum(raw, block_offset + 4u, scale_index);
                    return load_f16(raw, block_offset) * float(source_scale)
                        * float(quantized)
                        - load_f16(raw, block_offset + 2u) * float(source_minimum);
                }

                uint plane = half_index / 32u;
                uint position = half_index % 32u;
                uint low_offset = block_offset + block_half * 64u;
                uchar high = raw[block_offset + 128u + block_half * 32u + position];
                uint quantized;
                switch (plane) {
                case 0:
                    quantized = uint(raw[low_offset + position] & 15)
                        | (uint(high & 3) << 4);
                    break;
                case 1:
                    quantized = uint(raw[low_offset + position + 32u] & 15)
                        | (uint((high >> 2) & 3) << 4);
                    break;
                case 2:
                    quantized = uint(raw[low_offset + position] >> 4)
                        | (uint((high >> 4) & 3) << 4);
                    break;
                default:
                    quantized = uint(raw[low_offset + position + 32u] >> 4)
                        | (uint((high >> 6) & 3) << 4);
                    break;
                }
                int source_scale = signed_byte(
                    raw[block_offset + 192u + block_half * 8u + plane * 2u + position / 16u]
                );
                return load_f16(raw, block_offset + 208u) * float(source_scale)
                    * float(int(quantized) - 32);
            }
        """
    )

    static func quantize(
        raw: Data,
        sourceType: UInt32,
        sourceShape: [Int],
        name: String
    ) throws -> QuantizedTensor {
        guard supports(typeCode: sourceType) else {
            throw MiniMaxH3WeightError.unsupportedTensorType(
                name,
                type: GGUFDequantizer.typeName(typeCode: sourceType)
            )
        }
        guard sourceShape.count == 2,
              sourceShape[0] > 0,
              sourceShape[1] > 0,
              sourceShape[1] % targetGroupSize == 0 else {
            throw MiniMaxH3WeightError.unquantizableShape(name, shape: sourceShape)
        }
        let elementCount = sourceShape.reduce(1, *)
        let expectedByteCount: Int
        do {
            expectedByteCount = try GGUFDequantizer.byteCount(
                typeCode: sourceType,
                elementCount: elementCount
            )
        } catch {
            throw MiniMaxH3WeightError.unsupportedTensorType(
                name,
                type: GGUFDequantizer.typeName(typeCode: sourceType)
            )
        }
        guard raw.count == expectedByteCount else {
            throw MiniMaxH3WeightError.missingTensor(name)
        }

        let rowCount = sourceShape[0]
        let columnCount = sourceShape[1]
        let groupCount = elementCount / targetGroupSize
        let rawArray = MLXArray(raw, [raw.count], dtype: .uint8)
        let output = kernel(
            [rawArray],
            template: [("sourceType", Int(sourceType))],
            grid: (groupCount, 1, 1),
            threadGroup: (min(groupCount, 64), 1, 1),
            outputShapes: [
                [rowCount, columnCount * targetBits / 32],
                [rowCount, columnCount / targetGroupSize],
                [rowCount, columnCount / targetGroupSize]
            ],
            outputDTypes: [.uint32, .float32, .float32]
        )
        MLX.eval(output[0], output[1], output[2])
        return QuantizedTensor(
            weights: output[0],
            scales: output[1],
            biases: output[2]
        )
    }
}

import Foundation
import MLX

/// The pinned MLX checkpoint already uses OHWI kernels and affine packed linear weights.
struct Qwen21Weights {
    let tensors: [String: MLXArray]
    var groupSize = 64
    var bits = 4

    init(directory: URL, groupSize: Int = 64, bits: Int = 4) throws {
        tensors = try MLX.loadArrays(url: directory.appendingPathComponent("model.safetensors"))
        self.groupSize = groupSize; self.bits = bits
    }

    init(tensors: [String: MLXArray], groupSize: Int = 64, bits: Int = 4) {
        self.tensors = tensors; self.groupSize = groupSize; self.bits = bits
    }

    func tensor(_ name: String) throws -> MLXArray {
        guard let value = tensors[name] else { throw Qwen21Error.invalid("權重缺少 \(name)") }
        return value
    }

    func linear(_ x: MLXArray, _ name: String) throws -> MLXArray {
        let weight = try tensor(name + ".weight")
        var y: MLXArray
        if let scales = tensors[name + ".scales"] {
            guard weight.dtype == .uint32, let biases = tensors[name + ".biases"],
                  weight.ndim == 2, weight.dim(1) * (32 / bits) == x.dim(-1) else {
                throw Qwen21Error.invalid("量化權重格式不符：\(name)")
            }
            y = quantizedMatmul(x, weight, scales: scales, biases: biases,
                                transpose: true, groupSize: groupSize, bits: bits)
        } else {
            guard weight.ndim == 2, weight.dim(1) == x.dim(-1) else {
                throw Qwen21Error.invalid("線性層尺寸不符：\(name)")
            }
            y = matmul(x, weight.T.asType(x.dtype))
        }
        if let bias = tensors[name + ".bias"] { y = y + bias }
        return y
    }
}

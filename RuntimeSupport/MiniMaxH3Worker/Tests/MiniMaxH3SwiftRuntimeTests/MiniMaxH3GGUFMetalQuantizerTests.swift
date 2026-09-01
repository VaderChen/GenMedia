import Foundation
import GenImageGGUF
import MLX
import Testing

@testable import MiniMaxH3SwiftRuntime

@Suite("MiniMax H3 GPU GGUF quantizer")
struct MiniMaxH3GGUFMetalQuantizerTests {
    init() {
        let packageURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let testBundleURL = packageURL
            .appendingPathComponent(".build/out/Products/Debug/MiniMaxH3SwiftRuntimeTests.xctest")
        let metallibURL = testBundleURL
            .appendingPathComponent("Contents/MacOS/mlx.metallib")
        guard !FileManager.default.fileExists(atPath: metallibURL.path) else { return }
        let candidates = [
            packageURL.appendingPathComponent(".build/out/Products/Release/mlx.metallib"),
            testBundleURL.appendingPathComponent(
                "Contents/Resources/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"
            )
        ]
        for candidate in candidates where FileManager.default.fileExists(atPath: candidate.path) {
            try? FileManager.default.copyItem(at: candidate, to: metallibURL)
            return
        }
    }

    @Test("Q4_0 GPU quantization matches the CPU reference")
    func q40MatchesCPUReference() throws {
        let raw = q40Data(blockCount: 2)
        try assertMatchesCPU(
            raw: raw,
            typeCode: 2,
            shape: [1, 64],
            name: "q40.weight"
        )
    }

    @Test("K-quant GPU decoders match the CPU reference")
    func kQuantMatchesCPUReference() throws {
        let cases: [(UInt32, Int, String)] = [
            (12, 144, "q4k.weight"),
            (13, 176, "q5k.weight"),
            (14, 210, "q6k.weight")
        ]
        for (typeCode, bytesPerBlock, name) in cases {
            try assertMatchesCPU(
                raw: kQuantData(typeCode: typeCode, bytesPerBlock: bytesPerBlock),
                typeCode: typeCode,
                shape: [1, 256],
                name: name
            )
        }
    }

    @Test("Only H3 source formats use the GPU quantizer")
    func supportedSourceTypes() {
        #expect(MiniMaxH3GGUFMetalQuantizer.supports(typeCode: 2))
        #expect(MiniMaxH3GGUFMetalQuantizer.supports(typeCode: 12))
        #expect(MiniMaxH3GGUFMetalQuantizer.supports(typeCode: 13))
        #expect(MiniMaxH3GGUFMetalQuantizer.supports(typeCode: 14))
        #expect(!MiniMaxH3GGUFMetalQuantizer.supports(typeCode: 0))
        #expect(!MiniMaxH3GGUFMetalQuantizer.supports(typeCode: 1))
    }

    private func assertMatchesCPU(
        raw: Data,
        typeCode: UInt32,
        shape: [Int],
        name: String
    ) throws {
        let dense = try GGUFDequantizer.array(
            raw: raw,
            typeCode: typeCode,
            shape: shape,
            name: name
        ).asType(.float32)
        let cpu = MLX.quantized(dense, groupSize: 64, bits: 8)
        guard let cpuBiases = cpu.biases else {
            Issue.record("CPU reference did not return affine biases")
            return
        }
        MLX.eval(cpu.wq, cpu.scales, cpuBiases)

        let gpu = try MiniMaxH3GGUFMetalQuantizer.quantize(
            raw: raw,
            sourceType: typeCode,
            sourceShape: shape,
            name: name
        )
        #expect(gpu.weights.shape == cpu.wq.shape)
        #expect(gpu.scales.shape == cpu.scales.shape)
        #expect(gpu.biases.shape == cpuBiases.shape)

        let gpuRestored = MLX.dequantized(
            gpu.weights,
            scales: gpu.scales,
            biases: gpu.biases,
            groupSize: 64,
            bits: 8,
            dtype: .float32
        )
        let cpuRestored = MLX.dequantized(
            cpu.wq,
            scales: cpu.scales,
            biases: cpuBiases,
            groupSize: 64,
            bits: 8,
            dtype: .float32
        )
        MLX.eval(gpuRestored, cpuRestored)
        let maximumDifference = MLX.max(MLX.abs(gpuRestored - cpuRestored))
            .item(Float.self)
        #expect(maximumDifference < 1e-5)
    }

    private func q40Data(blockCount: Int) -> Data {
        var bytes = Data()
        for block in 0..<blockCount {
            appendUInt16(Float16(0.25 + Float(block) * 0.125).bitPattern, to: &bytes)
            for index in 0..<16 {
                let low = UInt8((index * 3 + block) % 16)
                let high = UInt8((index * 5 + 2 * block + 1) % 16)
                bytes.append(low | (high << 4))
            }
        }
        return bytes
    }

    private func kQuantData(typeCode: UInt32, bytesPerBlock: Int) -> Data {
        var bytes = Data(repeating: 0, count: bytesPerBlock)
        if typeCode == 14 {
            appendUInt16(Float16(0.125).bitPattern, to: &bytes, at: 208)
            for index in 0..<16 {
                bytes[192 + index] = UInt8((index * 3 + 1) % 256)
            }
        } else {
            appendUInt16(Float16(0.25).bitPattern, to: &bytes, at: 0)
            appendUInt16(Float16(0.125).bitPattern, to: &bytes, at: 2)
            for index in 0..<12 {
                bytes[4 + index] = UInt8((index * 7 + 3) % 64)
            }
        }
        for index in 16..<bytesPerBlock where typeCode != 14 {
            bytes[index] = UInt8((index * 11 + Int(typeCode)) % 256)
        }
        if typeCode == 14 {
            for index in 0..<192 {
                bytes[index] = UInt8((index * 11 + 5) % 256)
            }
        }
        return bytes
    }

    private func appendUInt16(
        _ value: UInt16,
        to data: inout Data,
        at offset: Int? = nil
    ) {
        let bytes = [UInt8(value & 0xff), UInt8(value >> 8)]
        if let offset {
            data.replaceSubrange(offset ..< offset + 2, with: bytes)
        } else {
            data.append(contentsOf: bytes)
        }
    }
}

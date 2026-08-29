import Foundation
import MLX
import Testing

@testable import GenImageGGUF

struct GGUFModelLoaderTests {
    @Test func inspectsF32ShapeAndMetadata() throws {
        let url = try Fixture.make(
            typeCode: 0,
            storedDimensions: [3, 2],
            payload: Fixture.float32Data([1, 2, 3, 4, 5, 6]),
            metadata: [("general.alignment", .uint32(64))]
        )
        defer { try? FileManager.default.removeItem(at: url) }

        let inspection = try GGUFModelLoader.inspect(fileURL: url)
        #expect(inspection.version == 3)
        #expect(inspection.alignment == 64)
        #expect(inspection.tensorCount == 1)
        #expect(inspection.tensors[0].shape == [2, 3])
        #expect(inspection.tensors[0].type == "F32")
        #expect(inspection.tensors[0].byteSize == 24)
        #expect(inspection.unsupportedTypes.isEmpty)
    }

    @Test func loadsF32ValuesWithoutChangingShape() throws {
        let values: [Float] = [1, -2, 3.5, 4, 5, -6]
        let url = try Fixture.make(
            typeCode: 0,
            storedDimensions: [3, 2],
            payload: Fixture.float32Data(values)
        )
        defer { try? FileManager.default.removeItem(at: url) }

        let (loaded, actual) = try Device.withDefaultDevice(.cpu) {
            let loaded = try GGUFModelLoader.loadWeights(
                fileURL: url,
                options: GGUFLoadOptions(materialization: .dense, computeDType: .float32)
            )
            return (loaded, loaded.tensors["test.weight"]!.asArray(Float.self))
        }
        #expect(actual == values)
        #expect(loaded.sourceTensorCount == 1)
        #expect(loaded.denseTensorCount == 1)
        #expect(loaded.quantizedTensorCount == 0)
    }

    @Test func decodesQ4_0InDenseMode() throws {
        var payload = Data()
        Fixture.appendUInt16(Float16(0.5).bitPattern, to: &payload)
        payload.append(contentsOf: repeatElement(0x89, count: 16))
        let url = try Fixture.make(
            typeCode: 2,
            storedDimensions: [32],
            payload: payload
        )
        defer { try? FileManager.default.removeItem(at: url) }

        let (loaded, actual) = try Device.withDefaultDevice(.cpu) {
            let loaded = try GGUFModelLoader.loadWeights(
                fileURL: url,
                options: GGUFLoadOptions(materialization: .dense, computeDType: .float32)
            )
            return (loaded, loaded.tensors["test.weight"]!.asArray(Float.self))
        }
        #expect(actual.count == 32)
        #expect(actual[0] == 0.5)
        #expect(actual[16] == 0)
        #expect(loaded.sourceQuantizationCounts["Q4_0"] == 1)
    }

    @Test func requantizesQ4_0IntoMLXQuantizedWeights() throws {
        let rows = 32
        let columns = 64
        var payload = Data()
        for _ in 0..<(rows * columns / 32) {
            Fixture.appendUInt16(Float16(0.5).bitPattern, to: &payload)
            payload.append(contentsOf: repeatElement(0x89, count: 16))
        }
        let url = try Fixture.make(
            typeCode: 2,
            storedDimensions: [UInt64(columns), UInt64(rows)],
            payload: payload
        )
        defer { try? FileManager.default.removeItem(at: url) }

        let loaded = try Device.withDefaultDevice(.cpu) {
            try GGUFModelLoader.loadWeights(fileURL: url)
        }
        #expect(loaded.quantizedTensorCount == 1)
        #expect(loaded.tensors["test.weight"] != nil)
        #expect(loaded.tensors["test.scales"] != nil)
        #expect(loaded.tensors["test.biases"] != nil)

        let dequantized = Device.withDefaultDevice(.cpu) {
            let dequantized = MLX.dequantized(
                loaded.tensors["test.weight"]!,
                scales: loaded.tensors["test.scales"]!,
                biases: loaded.tensors["test.biases"]!,
                groupSize: 64,
                bits: 4,
                dtype: .float32
            )
            MLX.eval(dequantized)
            return dequantized
        }
        #expect(dequantized.shape == [rows, columns])
        let values = Device.withDefaultDevice(.cpu) {
            dequantized.asArray(Float.self)
        }
        #expect(values.allSatisfy { $0.isFinite })
    }

    @Test func decodesMXFP4ScaleAndNibbles() throws {
        var payload = Data([127])
        payload.append(contentsOf: repeatElement(0x21, count: 16))
        let url = try Fixture.make(
            typeCode: 39,
            storedDimensions: [32],
            payload: payload
        )
        defer { try? FileManager.default.removeItem(at: url) }

        let (loaded, actual) = try Device.withDefaultDevice(.cpu) {
            let loaded = try GGUFModelLoader.loadWeights(
                fileURL: url,
                options: GGUFLoadOptions(materialization: .dense, computeDType: .float32)
            )
            return (loaded, loaded.tensors["test.weight"]!.asArray(Float.self))
        }
        #expect(loaded.denseTensorCount == 1)
        #expect(actual[0] == 0.5)
        #expect(actual[16] == 1)
    }

    @Test func storagePolicyMapsQualityAndSpeedProfiles() {
        #expect(GGUFDequantizer.typeName(typeCode: 6) == "Q5_0")
        #expect(GGUFDequantizer.typeName(typeCode: 8) == "Q8_0")
        #expect(GGUFLoadOptions().computeDType == .source)
        #expect(GGUFStoragePolicy.storageType(for: "Q4_K") == .int4)
        #expect(GGUFStoragePolicy.storageType(for: "Q5_K") == .int8)
        #expect(GGUFStoragePolicy.storageType(for: "Q5_K", profile: .speed) == .int4)
        #expect(GGUFStoragePolicy.targetBits(for: "Q8_0") == 8)
        #expect(GGUFStoragePolicy.targetBits(for: "F32") == nil)
    }

    @Test func diagnosticPlanCoversAllRequestedModels() {
        #expect(GGUFDiagnosticPlan.all.count == 3)
        #expect(GGUFDiagnosticPlan.spec(for: .ltxVideo096).expectedSourceType == "Q4_K")
        #expect(GGUFDiagnosticPlan.spec(for: .ltx23Distilled).expectedSourceType == "Q3_K")
        #expect(GGUFDiagnosticPlan.spec(for: .miniMaxH3).expectedSourceType == "Q4_K")
        #expect(GGUFStoragePolicy.targetBits(for: "Q3_K") == 4)
        #expect(GGUFStoragePolicy.targetBits(for: "Q4_K") == 4)
    }

    @Test func rejectsInvalidMagic() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("invalid-\(UUID().uuidString).gguf")
        try Data([0, 1, 2, 3]).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        do {
            _ = try GGUFModelLoader.inspect(fileURL: url)
            Issue.record("預期 invalid magic 錯誤")
        } catch GGUFLoaderError.invalidMagic {
        }
    }

    @Test func locateRejectsAmbiguousMainWeights() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("gguf-locate-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try Data().write(to: root.appendingPathComponent("a.gguf"))
        try Data().write(to: root.appendingPathComponent("b.gguf"))

        do {
            _ = try GGUFModelLoader.locate(in: root)
            Issue.record("預期 ambiguous weights 錯誤")
        } catch GGUFLoaderError.ambiguousWeights {
        }
    }

    private enum Fixture {
        enum Metadata {
            case uint32(UInt32)

            var typeCode: UInt32 { 4 }
        }

        static func make(
            typeCode: UInt32,
            storedDimensions: [UInt64],
            payload: Data,
            metadata: [(String, Metadata)] = []
        ) throws -> URL {
            var data = Data()
            appendUInt32(0x4655_4747, to: &data)
            appendUInt32(3, to: &data)
            appendUInt64(1, to: &data)
            appendUInt64(UInt64(metadata.count), to: &data)
            for (key, value) in metadata {
                appendString(key, to: &data)
                appendUInt32(value.typeCode, to: &data)
                if case let .uint32(number) = value {
                    appendUInt32(number, to: &data)
                }
            }
            appendString("test.weight", to: &data)
            appendUInt32(UInt32(storedDimensions.count), to: &data)
            storedDimensions.forEach { appendUInt64($0, to: &data) }
            appendUInt32(typeCode, to: &data)
            appendUInt64(0, to: &data)
            while data.count % 32 != 0 { data.append(0) }
            data.append(payload)

            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent("fixture-\(UUID().uuidString).gguf")
            try data.write(to: url, options: .atomic)
            return url
        }

        static func float32Data(_ values: [Float]) -> Data {
            Data(bytes: values, count: values.count * MemoryLayout<Float>.stride)
        }

        static func appendString(_ value: String, to data: inout Data) {
            let bytes = Array(value.utf8)
            appendUInt64(UInt64(bytes.count), to: &data)
            data.append(contentsOf: bytes)
        }

        static func appendUInt16(_ value: UInt16, to data: inout Data) {
            data.append(UInt8(truncatingIfNeeded: value))
            data.append(UInt8(truncatingIfNeeded: value >> 8))
        }

        static func appendUInt32(_ value: UInt32, to data: inout Data) {
            for index in 0..<4 {
                data.append(UInt8(truncatingIfNeeded: value >> UInt32(index * 8)))
            }
        }

        static func appendUInt64(_ value: UInt64, to data: inout Data) {
            for index in 0..<8 {
                data.append(UInt8(truncatingIfNeeded: value >> UInt64(index * 8)))
            }
        }
    }
}

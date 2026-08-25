import MLX
import Foundation

enum PyTorchZipTensorError: LocalizedError {
    case invalidArchive(String)
    case unsupportedCompression(String, UInt16)
    case unsupportedStorage(String)
    case invalidTensorMetadata(String)
    case invalidTensorData(expectedBytes: Int, actualBytes: Int)

    var errorDescription: String? {
        switch self {
        case let .invalidArchive(reason):
            "PyTorch ZIP 格式無法解析：\(reason)"
        case let .unsupportedCompression(name, method):
            "PyTorch ZIP 項目使用不支援的壓縮格式：\(name)（method=\(method)）"
        case let .unsupportedStorage(storage):
            "PyTorch Tensor storage 不支援：\(storage)"
        case let .invalidTensorMetadata(reason):
            "PyTorch Tensor metadata 無法解析：\(reason)"
        case let .invalidTensorData(expectedBytes, actualBytes):
            "PyTorch Tensor 資料長度不符：預期至少 \(expectedBytes) bytes，實際 \(actualBytes) bytes"
        }
    }
}

struct PyTorchZipTensor {
    let values: [Float]
    let shape: [Int]
}

enum PyTorchZipTensorReader {
    static func loadSingleFloatTensor(from url: URL) throws -> PyTorchZipTensor {
        let archive = try StoredZipArchive(url: url)
        guard let pickleName = archive.entryNames.first(where: { $0.hasSuffix("/data.pkl") }) else {
            throw PyTorchZipTensorError.invalidArchive("缺少 data.pkl")
        }
        guard let storageName = archive.entryNames.first(where: {
            $0.range(of: #"/data/\d+$"#, options: .regularExpression) != nil
        }) else {
            throw PyTorchZipTensorError.invalidArchive("缺少 data storage")
        }

        let pickle = try archive.data(for: pickleName)
        let metadata = try TensorPickleMetadata.parse(pickle)
        guard metadata.storageType == "FloatStorage" else {
            throw PyTorchZipTensorError.unsupportedStorage(metadata.storageType)
        }

        let storage = try archive.data(for: storageName)
        let elementCount = metadata.shape.reduce(1, *)
        let expectedBytes = elementCount * MemoryLayout<UInt32>.size
        guard storage.count >= expectedBytes else {
            throw PyTorchZipTensorError.invalidTensorData(
                expectedBytes: expectedBytes,
                actualBytes: storage.count
            )
        }

        let byteOrderName = archive.entryNames.first(where: { $0.hasSuffix("/byteorder") })
        let byteOrder = try byteOrderName.map { try archive.data(for: $0) }
            .flatMap { String(data: $0, encoding: .utf8) }?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? "little"
        guard byteOrder == "little" || byteOrder == "big" else {
            throw PyTorchZipTensorError.invalidArchive("未知 byteorder：\(byteOrder)")
        }

        var values = [Float]()
        values.reserveCapacity(elementCount)
        for index in 0..<elementCount {
            let offset = index * 4
            let bits: UInt32
            if byteOrder == "little" {
                bits = UInt32(storage[offset])
                    | UInt32(storage[offset + 1]) << 8
                    | UInt32(storage[offset + 2]) << 16
                    | UInt32(storage[offset + 3]) << 24
            } else {
                bits = UInt32(storage[offset]) << 24
                    | UInt32(storage[offset + 1]) << 16
                    | UInt32(storage[offset + 2]) << 8
                    | UInt32(storage[offset + 3])
            }
            values.append(Float(bitPattern: bits))
        }
        return PyTorchZipTensor(values: values, shape: metadata.shape)
    }
}

private struct TensorPickleMetadata {
    let storageType: String
    let shape: [Int]

    static func parse(_ data: Data) throws -> TensorPickleMetadata {
        guard let text = String(data: data, encoding: .isoLatin1) else {
            throw PyTorchZipTensorError.invalidTensorMetadata("pickle 不是可掃描的 binary protocol")
        }
        let supportedStorages = ["FloatStorage", "HalfStorage", "BFloat16Storage"]
        guard let storageType = supportedStorages.first(where: { text.contains($0) }) else {
            throw PyTorchZipTensorError.invalidTensorMetadata("找不到 storage 類型")
        }
        guard let persistentID = data.firstIndex(of: 0x51) else {
            throw PyTorchZipTensorError.invalidTensorMetadata("找不到 BINPERSID")
        }

        var cursor = data.index(after: persistentID)
        _ = try readInteger(from: data, cursor: &cursor)
        let shape = try readIntegerTuple(from: data, cursor: &cursor)
        guard !shape.isEmpty, shape.allSatisfy({ $0 > 0 }) else {
            throw PyTorchZipTensorError.invalidTensorMetadata("Tensor shape 無效：\(shape)")
        }
        return TensorPickleMetadata(storageType: storageType, shape: shape)
    }

    private static func readIntegerTuple(from data: Data, cursor: inout Data.Index) throws -> [Int] {
        guard cursor < data.endIndex else {
            throw PyTorchZipTensorError.invalidTensorMetadata("shape tuple 已達資料尾端")
        }
        if data[cursor] == 0x28 {
            cursor = data.index(after: cursor)
            var values: [Int] = []
            while cursor < data.endIndex, data[cursor] != 0x74 {
                values.append(try readInteger(from: data, cursor: &cursor))
            }
            guard cursor < data.endIndex else {
                throw PyTorchZipTensorError.invalidTensorMetadata("shape tuple 未結束")
            }
            cursor = data.index(after: cursor)
            return values
        }

        var values: [Int] = []
        while cursor < data.endIndex {
            let opcode = data[cursor]
            if opcode == 0x85 || opcode == 0x86 || opcode == 0x87 {
                let count = Int(opcode - 0x84)
                guard values.count == count else {
                    throw PyTorchZipTensorError.invalidTensorMetadata(
                        "shape tuple 元素數不符：\(values.count) / \(count)"
                    )
                }
                cursor = data.index(after: cursor)
                return values
            }
            values.append(try readInteger(from: data, cursor: &cursor))
        }
        throw PyTorchZipTensorError.invalidTensorMetadata("找不到 shape tuple opcode")
    }

    private static func readInteger(from data: Data, cursor: inout Data.Index) throws -> Int {
        guard cursor < data.endIndex else {
            throw PyTorchZipTensorError.invalidTensorMetadata("整數 opcode 已達資料尾端")
        }
        let opcode = data[cursor]
        cursor = data.index(after: cursor)
        switch opcode {
        case 0x4b:
            return Int(try readByte(from: data, cursor: &cursor))
        case 0x4d:
            let low = UInt16(try readByte(from: data, cursor: &cursor))
            let high = UInt16(try readByte(from: data, cursor: &cursor))
            return Int(low | high << 8)
        case 0x4a:
            var value: UInt32 = 0
            for shift in stride(from: 0, through: 24, by: 8) {
                value |= UInt32(try readByte(from: data, cursor: &cursor)) << UInt32(shift)
            }
            return Int(Int32(bitPattern: value))
        default:
            throw PyTorchZipTensorError.invalidTensorMetadata(
                String(format: "不支援的整數 opcode：0x%02x", opcode)
            )
        }
    }

    private static func readByte(from data: Data, cursor: inout Data.Index) throws -> UInt8 {
        guard cursor < data.endIndex else {
            throw PyTorchZipTensorError.invalidTensorMetadata("整數資料已達尾端")
        }
        let value = data[cursor]
        cursor = data.index(after: cursor)
        return value
    }
}

private struct StoredZipArchive {
    private struct Entry {
        let name: String
        let compressionMethod: UInt16
        let compressedSize: Int
        let uncompressedSize: Int
        let localHeaderOffset: Int
    }

    private let archiveData: Data
    private let entries: [String: Entry]

    var entryNames: [String] {
        entries.keys.sorted()
    }

    init(url: URL) throws {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        self.archiveData = data
        self.entries = try Self.readCentralDirectory(data)
    }

    func data(for name: String) throws -> Data {
        guard let entry = entries[name] else {
            throw PyTorchZipTensorError.invalidArchive("缺少項目：\(name)")
        }
        guard entry.compressionMethod == 0 else {
            throw PyTorchZipTensorError.unsupportedCompression(name, entry.compressionMethod)
        }
        let offset = entry.localHeaderOffset
        guard try archiveData.uint32LE(at: offset) == 0x04034b50 else {
            throw PyTorchZipTensorError.invalidArchive("local header signature 錯誤：\(name)")
        }
        let fileNameLength = Int(try archiveData.uint16LE(at: offset + 26))
        let extraLength = Int(try archiveData.uint16LE(at: offset + 28))
        let dataOffset = offset + 30 + fileNameLength + extraLength
        let dataEnd = dataOffset + entry.compressedSize
        guard dataOffset >= 0, dataEnd <= archiveData.count else {
            throw PyTorchZipTensorError.invalidArchive("項目超出檔案範圍：\(name)")
        }
        let result = archiveData.subdata(in: dataOffset..<dataEnd)
        guard result.count == entry.uncompressedSize else {
            throw PyTorchZipTensorError.invalidArchive("項目長度不符：\(name)")
        }
        return result
    }

    private static func readCentralDirectory(_ data: Data) throws -> [String: Entry] {
        let minimumEOCDSize = 22
        guard data.count >= minimumEOCDSize else {
            throw PyTorchZipTensorError.invalidArchive("檔案過短")
        }
        let searchStart = max(0, data.count - minimumEOCDSize - 65_535)
        var eocdOffset: Int?
        for offset in stride(from: data.count - minimumEOCDSize, through: searchStart, by: -1) {
            if try data.uint32LE(at: offset) == 0x06054b50 {
                eocdOffset = offset
                break
            }
        }
        guard let eocdOffset else {
            throw PyTorchZipTensorError.invalidArchive("找不到 end-of-central-directory")
        }

        let entryCount = Int(try data.uint16LE(at: eocdOffset + 10))
        var cursor = Int(try data.uint32LE(at: eocdOffset + 16))
        var entries: [String: Entry] = [:]
        for _ in 0..<entryCount {
            guard try data.uint32LE(at: cursor) == 0x02014b50 else {
                throw PyTorchZipTensorError.invalidArchive("central directory signature 錯誤")
            }
            let compressionMethod = try data.uint16LE(at: cursor + 10)
            let compressedSize = Int(try data.uint32LE(at: cursor + 20))
            let uncompressedSize = Int(try data.uint32LE(at: cursor + 24))
            let fileNameLength = Int(try data.uint16LE(at: cursor + 28))
            let extraLength = Int(try data.uint16LE(at: cursor + 30))
            let commentLength = Int(try data.uint16LE(at: cursor + 32))
            let localHeaderOffset = Int(try data.uint32LE(at: cursor + 42))
            let nameStart = cursor + 46
            let nameEnd = nameStart + fileNameLength
            guard nameEnd <= data.count,
                  let name = String(data: data.subdata(in: nameStart..<nameEnd), encoding: .utf8) else {
                throw PyTorchZipTensorError.invalidArchive("ZIP 項目名稱無效")
            }
            entries[name] = Entry(
                name: name,
                compressionMethod: compressionMethod,
                compressedSize: compressedSize,
                uncompressedSize: uncompressedSize,
                localHeaderOffset: localHeaderOffset
            )
            cursor = nameEnd + extraLength + commentLength
        }
        return entries
    }
}

private extension Data {
    func uint16LE(at offset: Int) throws -> UInt16 {
        guard offset >= 0, offset + 2 <= count else {
            throw PyTorchZipTensorError.invalidArchive("UInt16 超出檔案範圍")
        }
        return UInt16(self[offset]) | UInt16(self[offset + 1]) << 8
    }

    func uint32LE(at offset: Int) throws -> UInt32 {
        guard offset >= 0, offset + 4 <= count else {
            throw PyTorchZipTensorError.invalidArchive("UInt32 超出檔案範圍")
        }
        return UInt32(self[offset])
            | UInt32(self[offset + 1]) << 8
            | UInt32(self[offset + 2]) << 16
            | UInt32(self[offset + 3]) << 24
    }
}

import Foundation
import Testing

@testable import GenImageCore

struct ApplicationSupportTests {
    private func makeSandbox() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("genimage-support-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    private func write(_ contents: String, to url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try contents.write(to: url, atomically: true, encoding: .utf8)
    }

    @Test func everySubdirectoryHangsOffOneRoot() {
        let root = ApplicationSupport.rootURL()
        for subdirectory in ApplicationSupport.Subdirectory.allCases {
            #expect(
                ApplicationSupport.directory(subdirectory).path
                    == root.appendingPathComponent(subdirectory.rawValue).path
            )
        }
        #expect(root.lastPathComponent == ApplicationSupport.directoryName)
    }

    @Test func managedFileCheckAcceptsOnlyPathsInsideTheRoot() {
        let root = ApplicationSupport.rootURL()

        #expect(ApplicationSupport.managesFile(at: root.appendingPathComponent("Generated/a.png")))
        #expect(ApplicationSupport.managesFile(at: ApplicationSupport.directory(.workspace)))
        #expect(!ApplicationSupport.managesFile(at: URL(fileURLWithPath: "/tmp/a.png")))
        // 根目錄本身不算「裡面的檔案」，避免誤刪整個資料目錄。
        #expect(!ApplicationSupport.managesFile(at: root))
        // 前綴相同但不是同一個目錄。
        #expect(!ApplicationSupport.managesFile(at: root.deletingLastPathComponent()
            .appendingPathComponent(ApplicationSupport.directoryName + "Other/a.png")))
    }

    @Test func legacyContentIsAdoptedIntoTheCurrentRoot() throws {
        let sandbox = try makeSandbox()
        defer { try? FileManager.default.removeItem(at: sandbox) }
        let root = sandbox.appendingPathComponent("GenImage")
        let legacy = sandbox.appendingPathComponent("GenMedia")
        try write("{}", to: legacy.appendingPathComponent("Workspace/open-projects.json"))

        let adopted = ApplicationSupport.adoptLegacyDirectories(root: root, legacyRoots: [legacy])

        #expect(adopted == ["Workspace"])
        #expect(FileManager.default.fileExists(
            atPath: root.appendingPathComponent("Workspace/open-projects.json").path
        ))
        // 舊根目錄搬空後一併移除。
        #expect(!FileManager.default.fileExists(atPath: legacy.path))
    }

    // 兩邊都有同名項目時保留現有的：Runtime 裡的 Python venv 內嵌絕對路徑，覆蓋會弄壞它。
    @Test func existingContentIsNeverOverwritten() throws {
        let sandbox = try makeSandbox()
        defer { try? FileManager.default.removeItem(at: sandbox) }
        let root = sandbox.appendingPathComponent("GenImage")
        let legacy = sandbox.appendingPathComponent("GenMedia")
        try write("current", to: root.appendingPathComponent("Runtime/marker.txt"))
        try write("legacy", to: legacy.appendingPathComponent("Runtime/marker.txt"))

        let adopted = ApplicationSupport.adoptLegacyDirectories(root: root, legacyRoots: [legacy])

        #expect(adopted.isEmpty)
        let survivor = try String(
            contentsOf: root.appendingPathComponent("Runtime/marker.txt"),
            encoding: .utf8
        )
        #expect(survivor == "current")
        #expect(FileManager.default.fileExists(
            atPath: legacy.appendingPathComponent("Runtime/marker.txt").path
        ))
    }

    @Test func emptyLegacyDirectoriesAreCleanedUp() throws {
        let sandbox = try makeSandbox()
        defer { try? FileManager.default.removeItem(at: sandbox) }
        let root = sandbox.appendingPathComponent("GenImage")
        let legacy = sandbox.appendingPathComponent("GenMedia")
        try FileManager.default.createDirectory(
            at: root.appendingPathComponent("Runtime"),
            withIntermediateDirectories: true
        )
        try FileManager.default.createDirectory(
            at: legacy.appendingPathComponent("Runtime"),
            withIntermediateDirectories: true
        )

        ApplicationSupport.adoptLegacyDirectories(root: root, legacyRoots: [legacy])

        #expect(!FileManager.default.fileExists(atPath: legacy.path))
        #expect(FileManager.default.fileExists(atPath: root.appendingPathComponent("Runtime").path))
    }

    @Test func adoptionIsIdempotentAndSurvivesAMissingLegacyRoot() throws {
        let sandbox = try makeSandbox()
        defer { try? FileManager.default.removeItem(at: sandbox) }
        let root = sandbox.appendingPathComponent("GenImage")
        let legacy = sandbox.appendingPathComponent("GenMedia")
        try write("{}", to: legacy.appendingPathComponent("Workspace/open-projects.json"))

        #expect(ApplicationSupport.adoptLegacyDirectories(root: root, legacyRoots: [legacy]) == ["Workspace"])
        #expect(ApplicationSupport.adoptLegacyDirectories(root: root, legacyRoots: [legacy]).isEmpty)
        #expect(ApplicationSupport.adoptLegacyDirectories(
            root: root,
            legacyRoots: [sandbox.appendingPathComponent("NeverExisted")]
        ).isEmpty)
        #expect(FileManager.default.fileExists(
            atPath: root.appendingPathComponent("Workspace/open-projects.json").path
        ))
    }

    // 這台機器上實際的分裂狀態：舊根目錄有工作區索引與一個空的 Runtime，新根目錄已經有安裝好的
    // Runtime 與模型。搬完之後舊根目錄應該完全消失，而已安裝的內容一動也不能動。
    @Test func adoptionHandlesTheShippedSplitLayout() throws {
        let sandbox = try makeSandbox()
        defer { try? FileManager.default.removeItem(at: sandbox) }
        let root = sandbox.appendingPathComponent("GenImage")
        let legacy = sandbox.appendingPathComponent("GenMedia")
        for existing in ["Models", "Runtime", "Pasted", "PoC"] {
            try FileManager.default.createDirectory(
                at: root.appendingPathComponent(existing),
                withIntermediateDirectories: true
            )
        }
        try write("venv", to: root.appendingPathComponent("Runtime/ltx-2-mlx/.venv/bin/ltx-2-mlx"))
        try write("{\"projects\":[]}", to: legacy.appendingPathComponent("Workspace/open-projects.json"))
        try FileManager.default.createDirectory(
            at: legacy.appendingPathComponent("Runtime"),
            withIntermediateDirectories: true
        )

        let adopted = ApplicationSupport.adoptLegacyDirectories(root: root, legacyRoots: [legacy])

        #expect(adopted == ["Workspace"])
        #expect(!FileManager.default.fileExists(atPath: legacy.path))
        #expect(FileManager.default.fileExists(
            atPath: root.appendingPathComponent("Workspace/open-projects.json").path
        ))
        // 已安裝的 Runtime 內容原封不動 —— venv 的絕對路徑禁不起搬移。
        let venv = try String(
            contentsOf: root.appendingPathComponent("Runtime/ltx-2-mlx/.venv/bin/ltx-2-mlx"),
            encoding: .utf8
        )
        #expect(venv == "venv")
        for existing in ["Models", "Pasted", "PoC"] {
            #expect(FileManager.default.fileExists(atPath: root.appendingPathComponent(existing).path))
        }
    }

    @Test func adoptionIgnoresALegacyRootThatIsTheCurrentRoot() throws {
        let sandbox = try makeSandbox()
        defer { try? FileManager.default.removeItem(at: sandbox) }
        let root = sandbox.appendingPathComponent("GenImage")
        try write("{}", to: root.appendingPathComponent("Workspace/open-projects.json"))

        #expect(ApplicationSupport.adoptLegacyDirectories(root: root, legacyRoots: [root]).isEmpty)
        #expect(FileManager.default.fileExists(
            atPath: root.appendingPathComponent("Workspace/open-projects.json").path
        ))
    }

    @Test func orphanMediaCacheFilesOnlyReturnsUnreferencedUUIDFiles() throws {
        let sandbox = try makeSandbox()
        defer { try? FileManager.default.removeItem(at: sandbox) }
        let cache = sandbox.appendingPathComponent("MediaCache", isDirectory: true)
        try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)

        let referencedID = UUID(uuidString: "11111111-2222-3333-4444-555555555555")!
        let orphanID = UUID(uuidString: "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE")!
        try write("keep", to: cache.appendingPathComponent("\(referencedID.uuidString).mp4"))
        try write("remove", to: cache.appendingPathComponent("\(orphanID.uuidString).m4a"))
        try write("ignore", to: cache.appendingPathComponent("not-a-cache-file.log"))
        try write("ignore", to: cache.appendingPathComponent("\(orphanID.uuidString).part.mp4"))
        try FileManager.default.createDirectory(
            at: cache.appendingPathComponent("\(orphanID.uuidString).directory"),
            withIntermediateDirectories: true
        )

        let files = ApplicationSupport.orphanMediaCacheFiles(
            in: cache,
            referencedAssetIDs: [referencedID]
        )

        #expect(files.map(\.standardizedFileURL) == [cache
            .appendingPathComponent("\(orphanID.uuidString).m4a")
            .standardizedFileURL])
    }
}

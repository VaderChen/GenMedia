import Foundation
import Testing
@testable import GenImageCore

@MainActor
struct ModelDiscoveryControllerTests {
    @Test func lateResultsCannotReplaceANewerDirectory() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let first = root.appendingPathComponent("first")
        let second = root.appendingPathComponent("second")
        for url in [first, second] { try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true) }
        defer { try? FileManager.default.removeItem(at: root) }
        let releaseFirst = DispatchSemaphore(value: 0)
        let (started, continuation) = AsyncStream<Void>.makeStream()
        let controller = ModelDiscoveryController { url in
            #expect(!Thread.isMainThread)
            if url == first {
                continuation.yield(())
                _ = releaseFirst.wait(timeout: .now() + 5)
            }
            return DiscoveredModelCatalog(profiles: [InferenceProfile(name: url.lastPathComponent,
                capability: .textToImage, modelID: url.path, architecture: .mlxSwift)])
        }
        var applied: [String] = []
        let old = controller.load(at: first) { result in
            if case let .success(catalog) = result { applied += catalog.profiles.map(\.name) }
        }
        for await _ in started { break }
        continuation.finish()
        let current = controller.load(at: second) { result in
            if case let .success(catalog) = result { applied += catalog.profiles.map(\.name) }
        }
        await current.value
        #expect(applied == ["second"])
        #expect(!controller.isLoading)
        releaseFirst.signal()
        await old.value
        #expect(applied == ["second"])
    }

    @Test func cancelledScanDoesNotPublishAndMissingRootIsExplicit() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let controller = ModelDiscoveryController()
        var failed = false
        await controller.load(at: root) { result in
            if case .failure = result { failed = true }
        }.value
        #expect(failed)
        var empty = false
        await controller.load(at: root, allowMissingRoot: true) { result in
            if case let .success(catalog) = result { empty = catalog.models.isEmpty }
        }.value
        #expect(empty)
        var delivered = false
        let task = controller.load(at: root, allowMissingRoot: true) { _ in delivered = true }
        controller.cancel()
        await task.value
        #expect(!delivered)
        #expect(!controller.isLoading)
    }
}

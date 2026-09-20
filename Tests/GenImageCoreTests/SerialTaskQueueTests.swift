import Foundation
import Testing
@testable import GenImageCore

@MainActor
struct SerialTaskQueueTests {
    @Test func replacementWaitsForCancelledFileWorkAndOtherModelsRemainIndependent() async throws {
        let queue = SerialTaskQueue<String>()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent("weights")
        let gate = DispatchSemaphore(value: 0)
        let (started, continuation) = AsyncStream<Void>.makeStream()
        var events: [String] = []
        queue.replace(for: "model") {
            try? await BackgroundTask.run {
                continuation.yield(())
                _ = gate.wait(timeout: .now() + 5)
                // Simulate a filesystem call that cannot be interrupted immediately.
                try Data("old cleanup".utf8).write(to: file)
            }
            events.append("old stopped")
        }
        for await _ in started { break }
        continuation.finish()
        let replacement = queue.replace(for: "model") {
            #expect((try? String(contentsOf: file, encoding: .utf8)) == "old cleanup")
            try? FileManager.default.removeItem(at: file)
            events.append("removed")
        }
        await queue.replace(for: "another model") { events.append("independent") }.value
        #expect(events == ["independent"])
        gate.signal()
        await replacement.value
        #expect(events == ["independent", "old stopped", "removed"])
        #expect(!FileManager.default.fileExists(atPath: file.path))
    }

    @Test func cancelledQueuedOperationsStillFinishAndDrainWaitsForCleanup() async {
        let queue = SerialTaskQueue<String>()
        let (started, startedContinuation) = AsyncStream<Void>.makeStream()
        let (release, releaseContinuation) = AsyncStream<Void>.makeStream()
        var ranCancelled = false
        var finishedCancelled = false
        var oldFinished = false
        queue.replace(for: "model") {
            startedContinuation.yield(())
            // A detached await deliberately ignores the parent's cancellation.
            await Task.detached { for await _ in release { break } }.value
            oldFinished = true
        }
        for await _ in started { break }
        startedContinuation.finish()
        queue.replace(for: "model", operation: { ranCancelled = true }, onFinish: { finishedCancelled = true })
        queue.cancelAll()
        #expect(!finishedCancelled)
        releaseContinuation.yield(())
        releaseContinuation.finish()
        await queue.waitForAll()
        #expect(oldFinished)
        #expect(!ranCancelled)
        #expect(finishedCancelled)
    }

    @Test func discoveryWaitsForPreservedRemovalToFinish() async throws {
        let queue = SerialTaskQueue<String>()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent("weights")
        try Data().write(to: file)
        let (started, startContinuation) = AsyncStream<Void>.makeStream()
        let (release, releaseContinuation) = AsyncStream<Void>.makeStream()
        let (scanning, scanContinuation) = AsyncStream<Void>.makeStream()
        defer { startContinuation.finish(); releaseContinuation.finish(); scanContinuation.finish() }
        queue.replace(for: "removal") {
            startContinuation.yield(())
            for await _ in release { break }
            #expect(!Task.isCancelled)
            try? FileManager.default.removeItem(at: file)
        }
        for await _ in started { break }
        queue.cancelAll(except: ["removal"])
        let controller = ModelDiscoveryController { _ in
            #expect(!FileManager.default.fileExists(atPath: file.path))
            return DiscoveredModelCatalog()
        }
        var published = false
        let scan = controller.load(at: root, beforeDiscovery: {
            scanContinuation.yield(())
            await queue.waitForAll()
        }) { _ in published = true }
        for await _ in scanning { break }
        #expect(!published)
        #expect(controller.isLoading)
        releaseContinuation.yield(())
        await scan.value
        #expect(published)
        #expect(!controller.isLoading)
    }

}

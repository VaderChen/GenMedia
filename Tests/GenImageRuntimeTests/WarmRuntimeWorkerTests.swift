import Darwin
import Foundation
import Testing
@testable import GenImageRuntime

@Suite(.timeLimit(.minutes(1)))
struct WarmRuntimeWorkerTests {
    private func fixture() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("warm-test-\(UUID())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let script = directory.appendingPathComponent("worker")
        let body = #"""
        #!/bin/sh
        while IFS= read -r request; do
            case "$request" in
                '"wait"')
                    printf '{"type":"progress","requestID":"wait","value":0.2}\n'
                    IFS= read -r ignored
                    ;;
                '"error"')
                    printf '{"type":"error","requestID":"error","message":"fixture failure"}\n'
                    ;;
                *) printf '{"type":"completed","requestID":%s,"pid":%d}\n' "$request" "$$" ;;
            esac
        done
        """#
        try body.write(to: script, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: script.path)
        return script
    }

    private func run(_ worker: WarmRuntimeWorker, executable: URL, id: String) async throws -> Int {
        let data = try await worker.run(executable: executable,
            request: JSONEncoder().encode(id), requestID: id, progress: { _ in })
        let object = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        return try #require(object["pid"] as? Int)
    }

    @Test func reusesProcessAndUnloadsAfterIdle() async throws {
        let script = try fixture()
        defer { try? FileManager.default.removeItem(at: script.deletingLastPathComponent()) }
        // Process reuse must not race a 400 ms expiry or system-wide pressure
        // events caused by unrelated tests/builds. Verify expiry independently.
        let worker = WarmRuntimeWorker(observesMemoryPressure: false)
        let first = try await run(worker, executable: script, id: "first")
        let second = try await run(worker, executable: script, id: "second")
        #expect(first == second)
        await worker.unload()

        let expiring = WarmRuntimeWorker(idleSeconds: 0.1, observesMemoryPressure: false)
        let beforeExpiry = try await run(expiring, executable: script, id: "before-expiry")
        let deadline = ContinuousClock.now.advanced(by: .seconds(3))
        while Darwin.kill(Int32(beforeExpiry), 0) == 0, ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(20))
        }
        #expect(Darwin.kill(Int32(beforeExpiry), 0) != 0)
        let afterExpiry = try await run(expiring, executable: script, id: "after-expiry")
        #expect(beforeExpiry != afterExpiry)
        await expiring.unload()
    }

    @Test func cancellationAndErrorsDiscardTheSession() async throws {
        let script = try fixture()
        defer { try? FileManager.default.removeItem(at: script.deletingLastPathComponent()) }
        let worker = WarmRuntimeWorker()
        let first = try await run(worker, executable: script, id: "first")
        let (started, continuation) = AsyncStream<Void>.makeStream()
        let task = Task {
            try await worker.run(executable: script, request: JSONEncoder().encode("wait"),
                requestID: "wait", progress: { _ in continuation.yield(()) })
        }
        for await _ in started { break }
        continuation.finish()
        task.cancel()
        await #expect(throws: CancellationError.self) { try await task.value }
        let afterCancel = try await run(worker, executable: script, id: "afterCancel")
        #expect(afterCancel != first)
        await #expect(throws: WarmRuntimeWorker.Failure.self) {
            try await run(worker, executable: script, id: "error")
        }
        let afterError = try await run(worker, executable: script, id: "afterError")
        #expect(afterError != afterCancel)
        await worker.unload()
    }

    @Test func cancellationInterruptsAFullInputPipe() async throws {
        let script = try fixture()
        defer { try? FileManager.default.removeItem(at: script.deletingLastPathComponent()) }
        // The helper deliberately never reads stdin. A synchronous 512 KiB
        // write blocks until this process exits, regardless of task cancellation.
        try "#!/bin/sh\nexec /bin/sleep 5\n".write(to: script, atomically: false, encoding: .utf8)
        let worker = WarmRuntimeWorker()
        let task = Task {
            try await worker.run(executable: script, request: Data(repeating: 65, count: 512 * 1_024),
                requestID: "blocked", progress: { _ in })
        }
        try await Task.sleep(for: .milliseconds(150))
        let cancelled = ContinuousClock.now
        task.cancel()
        await #expect(throws: CancellationError.self) { try await task.value }
        #expect(cancelled.duration(to: .now) < .seconds(2))
        await worker.unload()
    }

    @Test(arguments: [true, false])
    func completionAfterLargeLogsSurvivesWorkerExit(endsWithNewline: Bool) async throws {
        let script = try fixture()
        defer { try? FileManager.default.removeItem(at: script.deletingLastPathComponent()) }
        let body = #"""
        #!/bin/sh
        IFS= read -r request
        /bin/dd if=/dev/zero bs=1048576 count=3 2>/dev/null
        printf '\n{"type":"completed","requestID":%s,"pid":%d}' "$request" "$$"
        """# + (endsWithNewline ? "\nprintf '\\n'\n" : "\n")
        try body.write(to: script, atomically: false, encoding: .utf8)
        let worker = WarmRuntimeWorker()
        let pid = try await run(worker, executable: script, id: "large-log")
        #expect(pid > 0)
        await worker.unload()
    }

}

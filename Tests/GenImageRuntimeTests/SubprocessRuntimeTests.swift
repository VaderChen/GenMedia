import Foundation
import Testing

@testable import GenImageRuntime

// 這一層是三個 Runtime 服務共用的子行程執行流程，剛好也是整個 Runtime 目錄裡少數不需要模型
// 權重就能實際執行的部分 —— 用 /bin/sh 就能把成功、失敗、逾時與取消四條路徑都跑過一次。
struct SubprocessRuntimeTests {
    private func temporaryLogURL() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("genimage-subprocess-test-\(UUID().uuidString).log")
    }

    @Test func successfulRunReturnsZeroAndCapturesOutput() async throws {
        let logURL = temporaryLogURL()
        defer { try? FileManager.default.removeItem(at: logURL) }
        let log = try RuntimeLog(at: logURL)
        defer { log.close() }

        let status = try await RuntimeProcess.run(
            executable: URL(fileURLWithPath: "/bin/sh"),
            arguments: ["-c", "echo genimage-ok"],
            log: log,
            pollInterval: .milliseconds(20)
        )

        #expect(status == 0)
        #expect(log.message(fallback: "").contains("genimage-ok"))
    }

    @Test func failureReturnsTheExitCodeAndStderrEndsUpInTheLog() async throws {
        let logURL = temporaryLogURL()
        defer { try? FileManager.default.removeItem(at: logURL) }
        let log = try RuntimeLog(at: logURL)
        defer { log.close() }

        let status = try await RuntimeProcess.run(
            executable: URL(fileURLWithPath: "/bin/sh"),
            arguments: ["-c", "echo genimage-boom >&2; exit 3"],
            log: log,
            pollInterval: .milliseconds(20)
        )

        #expect(status == 3)
        #expect(log.message(fallback: "").contains("genimage-boom"))
    }

    @Test func emptyLogFallsBackToTheSuppliedMessage() throws {
        let logURL = temporaryLogURL()
        defer { try? FileManager.default.removeItem(at: logURL) }
        let log = try RuntimeLog(at: logURL)
        defer { log.close() }

        #expect(log.message(fallback: "沒有輸出") == "沒有輸出")
        #expect(log.lastLine(fallback: "沒有輸出") == "沒有輸出")
        #expect(log.data() == nil)
    }

    private struct PollFailure: Error {}

    // onPoll 丟出錯誤（逾時、停滯）必須確實終止子行程，而不是讓它變成孤兒繼續跑。
    @Test func aThrowingPollTerminatesTheProcessInsteadOfWaitingForIt() async throws {
        let logURL = temporaryLogURL()
        defer { try? FileManager.default.removeItem(at: logURL) }
        let log = try RuntimeLog(at: logURL)
        defer { log.close() }

        let startedAt = Date()
        await #expect(throws: PollFailure.self) {
            try await RuntimeProcess.run(
                executable: URL(fileURLWithPath: "/bin/sh"),
                arguments: ["-c", "sleep 30"],
                log: log,
                pollInterval: .milliseconds(20)
            ) {
                throw PollFailure()
            }
        }
        #expect(Date().timeIntervalSince(startedAt) < 5)
    }

    @Test func cancellationTerminatesTheProcessInsteadOfWaitingForIt() async throws {
        let logURL = temporaryLogURL()
        defer { try? FileManager.default.removeItem(at: logURL) }
        let log = try RuntimeLog(at: logURL)
        defer { log.close() }

        let startedAt = Date()
        let task = Task {
            try await RuntimeProcess.run(
                executable: URL(fileURLWithPath: "/bin/sh"),
                arguments: ["-c", "sleep 30"],
                log: log,
                pollInterval: .milliseconds(20)
            )
        }
        try await Task.sleep(for: .milliseconds(200))
        task.cancel()

        await #expect(throws: CancellationError.self) { try await task.value }
        #expect(Date().timeIntervalSince(startedAt) < 5)
    }

    @Test func logActivityDistinguishesGrowthFromIdleness() async throws {
        let logURL = temporaryLogURL()
        defer { try? FileManager.default.removeItem(at: logURL) }
        let log = try RuntimeLog(at: logURL)
        defer { log.close() }

        var activity = RuntimeLogActivity(log: log)
        let beforeAnyOutput = activity.sample(log)
        #expect(!beforeAnyOutput)

        try "keeps growing\n".write(to: logURL, atomically: false, encoding: .utf8)
        let afterOutput = activity.sample(log)
        #expect(afterOutput)
        #expect(activity.idleDuration < 1)
        let withoutFurtherOutput = activity.sample(log)
        #expect(!withoutFurtherOutput)
    }

    @Test func executableLookupTakesTheFirstRunnableCandidate() {
        let missing = URL(fileURLWithPath: "/genimage/does/not/exist")
        let shell = URL(fileURLWithPath: "/bin/sh")

        #expect(RuntimeExecutable.locate([missing, shell]) == shell)
        #expect(RuntimeExecutable.locate([missing]) == nil)
        // 目錄不是可執行檔，不能被當成候選。
        #expect(RuntimeExecutable.locate([URL(fileURLWithPath: "/bin")]) == nil)
    }

    @Test func pathCandidatesExpandEveryPathEntry() {
        let candidates = RuntimeExecutable.pathCandidates(
            for: "ffmpeg",
            environment: ["PATH": "/opt/homebrew/bin:/usr/local/bin"]
        )

        #expect(candidates.map(\.path) == ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"])
        #expect(RuntimeExecutable.pathCandidates(for: "ffmpeg", environment: [:]).isEmpty)
    }

    @Test func runtimeEnvironmentAppendsCommonPathsWithoutDuplicates() {
        let environment = RuntimeExecutable.environment()
        let paths = (environment["PATH"] ?? "").split(separator: ":").map(String.init)

        #expect(paths.contains("/usr/bin"))
        #expect(Set(paths).count == paths.count)
    }
}

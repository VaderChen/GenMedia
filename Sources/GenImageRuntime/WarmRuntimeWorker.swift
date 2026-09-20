import Darwin
import Foundation

/// A sequential JSON-line session; the helper owns all model allocations.
actor WarmRuntimeWorker {
    enum Failure: LocalizedError {
        case busy, requestTooLarge, exited(Int32, String), failed(String), stalled
        var errorDescription: String? {
            switch self {
            case .busy: "模型已有執行中的生成任務。"
            case .requestTooLarge: "生成請求超過大小上限。"
            case let .exited(code, message): "Worker 結束（\(code)）：\(message)"
            case let .failed(message): message
            case .stalled: "Worker 長時間未回應。"
            }
        }
    }

    private final class Session: @unchecked Sendable {
        let process = Process()
        let input = Pipe()
        let log: RuntimeLog
        let reader: IncrementalLogReader
        let executable: URL
        private var stopped = false

        init(executable: URL, environment: [String: String]) throws {
            self.executable = executable
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent("genimage-warm-\(UUID()).log")
            log = try RuntimeLog(at: url)
            reader = IncrementalLogReader(url: url)
            process.executableURL = executable
            process.arguments = ["--serve"]
            process.environment = environment
            process.standardInput = input
            process.standardOutput = log.handle
            process.standardError = log.handle
            _ = fcntl(input.fileHandleForWriting.fileDescriptor, F_SETNOSIGPIPE, 1)
            let descriptor = input.fileHandleForWriting.fileDescriptor
            let flags = fcntl(descriptor, F_GETFL)
            guard flags != -1, fcntl(descriptor, F_SETFL, flags | O_NONBLOCK) != -1 else {
                throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
            }
            try process.run()
            try? input.fileHandleForReading.close()
        }

        func writeRequest(_ data: Data) async throws {
            let descriptor = input.fileHandleForWriting.fileDescriptor
            let deadline = ContinuousClock.now.advanced(by: .seconds(30))
            var offset = 0
            while offset < data.count {
                try Task.checkCancellation()
                guard ContinuousClock.now < deadline else { throw Failure.stalled }
                let written = data.withUnsafeBytes { bytes in
                    Darwin.write(descriptor, bytes.baseAddress!.advanced(by: offset), data.count - offset)
                }
                if written > 0 {
                    offset += written
                } else if written < 0 && errno == EINTR {
                    continue
                } else if written == 0 || errno == EAGAIN || errno == EWOULDBLOCK {
                    try await Task.sleep(for: .milliseconds(10))
                } else {
                    throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
                }
            }
        }

        func stop() {
            guard !stopped else { return }
            stopped = true
            try? input.fileHandleForWriting.close()
            RuntimeProcess.forceTerminate(process)
            log.close()
            try? FileManager.default.removeItem(at: log.url)
        }
        deinit { stop() }
    }

    private var session: Session?
    private var busy = false
    private var unloadWhenIdle = false
    private var idleTask: Task<Void, Never>?
    private var memoryPressure: (any DispatchSourceMemoryPressure)?
    private let idleSeconds: Double
    private let observesMemoryPressure: Bool

    init(idleSeconds: Double = 300, observesMemoryPressure: Bool = true) {
        self.idleSeconds = idleSeconds
        self.observesMemoryPressure = observesMemoryPressure
    }

    func unload() {
        idleTask?.cancel()
        idleTask = nil
        if busy { unloadWhenIdle = true; return }
        session?.stop()
        session = nil
        unloadWhenIdle = false
    }

    func run(
        executable: URL,
        request: Data,
        requestID: String,
        environment: [String: String] = RuntimeExecutable.environment(),
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> Data {
        try Task.checkCancellation()
        guard !busy else { throw Failure.busy }
        guard request.count <= 1_024 * 1_024 else { throw Failure.requestTooLarge }
        busy = true
        idleTask?.cancel()
        idleTask = nil
        defer {
            busy = false
            if unloadWhenIdle { unload() }
            else { scheduleIdleExpiry() }
        }
        if observesMemoryPressure, memoryPressure == nil {
            let source = DispatchSource.makeMemoryPressureSource(eventMask: [.warning, .critical],
                queue: .global(qos: .utility))
            source.setEventHandler { [weak self] in
                Task { await self?.unload() }
            }
            source.resume()
            memoryPressure = source
        }
        do {
            if session?.process.isRunning != true || session?.executable != executable {
                session?.stop()
                session = try Session(executable: executable, environment: environment)
            }
            guard let session else { throw Failure.failed("無法啟動 Worker。") }
            try Task.checkCancellation()
            try await session.writeRequest(request + Data([10]))
            var lastActivity = Date()
            var lastProgress: Double?
            while true {
                try Task.checkCancellation()
                // Sample before reading: if the process exits during this read,
                // the next iteration must still consume its final output.
                let isRunning = session.process.isRunning
                let previousOffset = session.reader.offset
                let batch = try session.reader.readLines(finalize: !isRunning)
                if session.reader.offset != previousOffset { lastActivity = Date() }
                for line in batch.lines {
                    guard let event = try? JSONDecoder().decode(Event.self, from: line),
                          event.requestID == requestID else { continue }
                    switch event.type {
                    case "progress":
                        if let value = event.value, value.isFinite, value != lastProgress {
                            lastProgress = value
                            progress(min(0.99, max(0.01, value)))
                        }
                    case "completed": return line
                    case "error": throw Failure.failed(event.message ?? "Worker 執行失敗。")
                    default: break
                    }
                }
                if batch.hasMore {
                    await Task.yield()
                    continue
                }
                guard isRunning else {
                    throw Failure.exited(session.process.terminationStatus,
                        session.log.message(fallback: "Worker 未提供錯誤訊息。"))
                }
                guard Date().timeIntervalSince(lastActivity) < 30 * 60 else { throw Failure.stalled }
                try await Task.sleep(for: .milliseconds(200))
            }
        } catch {
            session?.stop()
            session = nil
            throw error
        }
    }

    private func scheduleIdleExpiry() {
        guard session != nil else { return }
        idleTask = Task { [weak self, idleSeconds] in
            do { try await Task.sleep(for: .seconds(idleSeconds)) }
            catch { return }
            await self?.unload()
        }
    }

    private struct Event: Decodable {
        let requestID: String?
        let type: String
        let value: Double?
        let message: String?
    }

    deinit {
        idleTask?.cancel()
        memoryPressure?.cancel()
        session?.stop()
    }
}

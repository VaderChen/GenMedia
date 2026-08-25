import Darwin
import Foundation

// 以外部行程執行的 Runtime 共用的部分。
//
// Qwen Image Edit Worker、LTX 影片與 MiniMax 音樂三個服務是同一個形狀：找出可執行檔 → 把
// stdout 與 stderr 導進同一個 log 檔 → 邊跑邊從 log 取進度 → 結束後檢查 exit code，失敗時把 log
// 尾端當成錯誤訊息。這些步驟原本在三個檔案裡各寫一份，連 forceTerminate、fileSize、logMessage
// 都是逐字重複；真正的問題是重複之外的分歧 —— 有的服務只在 CancellationError 時終止子行程，
// 其他錯誤會讓子行程變成孤兒。集中之後，這些語意只有一種。

// MARK: - 可執行檔

enum RuntimeExecutable {
    /// 依序回傳第一個確實可執行的候選路徑。
    ///
    /// 目錄也帶著執行權限，`isExecutableFile` 會對它回答 true —— PATH 裡剛好有個同名目錄時，
    /// 就會挑到一個永遠啟動不了的「可執行檔」，錯誤訊息還完全看不出原因，所以這裡排除目錄。
    static func locate(_ candidates: [URL]) -> URL? {
        let fileManager = FileManager.default
        return candidates.first { candidate in
            var isDirectory: ObjCBool = false
            guard fileManager.fileExists(atPath: candidate.path, isDirectory: &isDirectory),
                  !isDirectory.boolValue
            else { return false }
            return fileManager.isExecutableFile(atPath: candidate.path)
        }
    }

    /// 把 PATH 內的每個目錄展開成 `name` 的候選路徑。
    static func pathCandidates(
        for name: String,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> [URL] {
        (environment["PATH"] ?? "").split(separator: ":").map {
            URL(fileURLWithPath: String($0), isDirectory: true).appendingPathComponent(name)
        }
    }

    /// 子行程的環境：PATH 補上常見安裝位置，並關閉 Python 的輸出緩衝，讓進度能即時進 log。
    static func environment() -> [String: String] {
        var environment = ProcessInfo.processInfo.environment
        let commonPaths = ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin"]
        let currentPaths = (environment["PATH"] ?? "").split(separator: ":").map(String.init)
        environment["PATH"] = (currentPaths + commonPaths)
            .reduce(into: [String]()) { paths, path in
                if !paths.contains(path) { paths.append(path) }
            }
            .joined(separator: ":")
        environment["PYTHONUNBUFFERED"] = "1"
        return environment
    }
}

// MARK: - Log

/// 子行程的 log 檔：stdout 與 stderr 都寫進這裡，同時作為進度來源與錯誤訊息來源。
struct RuntimeLog {
    let url: URL
    let handle: FileHandle

    init(at url: URL) throws {
        FileManager.default.createFile(atPath: url.path, contents: nil)
        self.handle = try FileHandle(forWritingTo: url)
        self.url = url
    }

    func flush() {
        try? handle.synchronize()
    }

    func close() {
        flush()
        try? handle.close()
    }

    var size: Int64 { Self.fileSize(at: url) }

    func data() -> Data? {
        guard let data = try? Data(contentsOf: url), !data.isEmpty else { return nil }
        return data
    }

    func tail(_ maximumBytes: Int) -> String? {
        guard let data = data() else { return nil }
        return String(data: data.suffix(maximumBytes), encoding: .utf8)
    }

    /// 失敗時給人看的訊息：log 尾端，取不到就用 fallback。
    func message(maximumBytes: Int = 8_192, fallback: String) -> String {
        guard let text = tail(maximumBytes)?.trimmingCharacters(in: .whitespacesAndNewlines),
              !text.isEmpty
        else { return fallback }
        return text
    }

    /// 停滯時給人看的訊息：log 的最後一行。
    func lastLine(maximumBytes: Int = 4_096, fallback: String) -> String {
        guard let text = tail(maximumBytes) else { return fallback }
        return text.split(whereSeparator: { $0 == "\n" || $0 == "\r" })
            .last?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            ?? fallback
    }

    static func fileSize(at url: URL) -> Int64 {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
              let size = attributes[.size] as? NSNumber else { return 0 }
        return size.int64Value
    }
}

/// 以 log 是否還在長大判斷子行程是否卡住 —— 沒有進度回報的 Runtime 只剩這個訊號。
struct RuntimeLogActivity {
    private var observedSize: Int64
    private var lastChangedAt: Date

    init(log: RuntimeLog) {
        observedSize = log.size
        lastChangedAt = Date()
    }

    /// 回傳這次輪詢是否有新輸出。
    @discardableResult
    mutating func sample(_ log: RuntimeLog) -> Bool {
        let size = log.size
        guard size != observedSize else { return false }
        observedSize = size
        lastChangedAt = Date()
        return true
    }

    var idleDuration: TimeInterval { Date().timeIntervalSince(lastChangedAt) }
}

// MARK: - 執行

enum RuntimeProcess {
    /// 執行子行程直到結束，回傳 exit code。
    ///
    /// 每隔 `pollInterval` 呼叫一次 `onPoll`：回報進度、或丟出錯誤來中止（逾時、停滯）。取消與
    /// 任何 `onPoll` 丟出的錯誤都會確實終止子行程再往外拋，不會留下孤兒行程。
    @discardableResult
    static func run(
        executable: URL,
        arguments: [String],
        environment: [String: String]? = nil,
        log: RuntimeLog,
        pollInterval: Duration = .milliseconds(250),
        onPoll: () throws -> Void = {}
    ) async throws -> Int32 {
        let process = Process()
        process.executableURL = executable
        process.arguments = arguments
        if let environment { process.environment = environment }
        process.standardInput = FileHandle.nullDevice
        process.standardOutput = log.handle
        process.standardError = log.handle

        do {
            try process.run()
            while process.isRunning {
                try Task.checkCancellation()
                try await Task.sleep(for: pollInterval)
                try onPoll()
            }
        } catch {
            forceTerminate(process)
            throw error
        }
        process.waitUntilExit()
        log.flush()
        return process.terminationStatus
    }

    /// 先 terminate，還活著就 SIGKILL，並等它真的結束。
    static func forceTerminate(_ process: Process) {
        guard process.isRunning else { return }
        process.terminate()
        if process.isRunning {
            Darwin.kill(process.processIdentifier, SIGKILL)
        }
        process.waitUntilExit()
    }
}

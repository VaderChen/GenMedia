import Foundation

final class FileDownloadDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let destination: URL
    private let resumeDataURL: URL
    private let expectedBytes: Int64
    private let progress: @Sendable (Int64, Int64) -> Void
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Void, Error>?
    private var session: URLSession?
    private var task: URLSessionDownloadTask?
    private var movedFile = false
    private var completed = false
    private var cancellationRequested = false
    private var resumeDataPending = false
    private var pendingResult: Result<Void, Error>?

    init(
        destination: URL,
        expectedBytes: Int64,
        progress: @escaping @Sendable (Int64, Int64) -> Void
    ) {
        self.destination = destination
        resumeDataURL = destination.appendingPathExtension("resume")
        self.expectedBytes = expectedBytes
        self.progress = progress
    }

    func start(request: URLRequest) async throws {
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
                lock.lock()
                guard !cancellationRequested else {
                    lock.unlock()
                    continuation.resume(throwing: CancellationError())
                    return
                }
                guard !completed, self.continuation == nil else {
                    lock.unlock()
                    continuation.resume(throwing: ModelInstallerError.invalidResponse)
                    return
                }
                self.continuation = continuation
                let configuration = URLSessionConfiguration.default
                configuration.timeoutIntervalForRequest = 60 * 60 * 24
                configuration.timeoutIntervalForResource = 60 * 60 * 24
                configuration.waitsForConnectivity = false
                configuration.httpMaximumConnectionsPerHost = 8
                configuration.httpShouldUsePipelining = true
                configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
                configuration.allowsExpensiveNetworkAccess = true
                configuration.allowsConstrainedNetworkAccess = true
                let session = URLSession(configuration: configuration, delegate: self, delegateQueue: nil)
                self.session = session
                let task: URLSessionDownloadTask
                if let resumeData = try? Data(contentsOf: resumeDataURL), !resumeData.isEmpty {
                    task = session.downloadTask(withResumeData: resumeData)
                } else {
                    task = session.downloadTask(with: request)
                }
                self.task = task
                lock.unlock()
                task.resume()
            }
        } onCancel: {
            self.cancel()
        }
    }

    func cancel() {
        lock.lock()
        guard !completed, !cancellationRequested else { lock.unlock(); return }
        cancellationRequested = true
        let task = task
        resumeDataPending = task != nil
        lock.unlock()
        task?.cancel { [self] resumeData in
            if let resumeData, !resumeData.isEmpty {
                try? resumeData.write(to: resumeDataURL, options: .atomic)
            }
            lock.lock()
            resumeDataPending = false
            let result = pendingResult
            lock.unlock()
            if let result { finish(result) }
        }
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        lock.lock()
        let cancelled = cancellationRequested || completed
        lock.unlock()
        guard !cancelled else { return }
        progress(totalBytesWritten, totalBytesExpectedToWrite)
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        lock.lock()
        let cancelled = cancellationRequested || completed
        lock.unlock()
        guard !cancelled else { return }
        do {
            let fileManager = FileManager.default
            if let response = downloadTask.response as? HTTPURLResponse,
               !(200..<300).contains(response.statusCode) {
                let handle = try? FileHandle(forReadingFrom: location)
                defer { try? handle?.close() }
                let body = try? handle?.read(upToCount: 2_048)
                let message = body
                    .flatMap { String(data: $0.prefix(2_048), encoding: .utf8) } ?? ""
                if response.statusCode == 401,
                   let url = downloadTask.originalRequest?.url,
                   url.host?.localizedCaseInsensitiveContains("civitai.com") == true {
                    throw ModelInstallerError.authenticationRequired(url, message)
                }
                throw ModelInstallerError.httpStatus(response.statusCode, message)
            }
            let actual = Int64(
                (try location.resourceValues(forKeys: [.fileSizeKey])).fileSize ?? -1
            )
            guard actual == expectedBytes else {
                throw ModelInstallerError.sizeMismatch(
                    path: destination.lastPathComponent,
                    expected: expectedBytes,
                    actual: actual
                )
            }
            try ModelFileReplacement.replace(at: destination) { staging in
                try fileManager.moveItem(at: location, to: staging)
            }
            try? fileManager.removeItem(at: resumeDataURL)
            lock.lock()
            movedFile = true
            lock.unlock()
        } catch {
            finish(.failure(error))
        }
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: (any Error)?
    ) {
        if let error {
            finish(.failure(error))
            return
        }
        lock.lock()
        let movedFile = movedFile
        lock.unlock()
        finish(movedFile ? .success(()) : .failure(ModelInstallerError.invalidResponse))
    }

    private func finish(_ result: Result<Void, Error>) {
        lock.lock()
        guard !completed else {
            lock.unlock()
            return
        }
        if pendingResult == nil { pendingResult = result }
        // Returning from start() is a cleanup barrier: resume metadata must not
        // appear later and overwrite a resumed download or a model removal.
        guard !resumeDataPending else { lock.unlock(); return }
        let finalResult: Result<Void, Error> = cancellationRequested
            ? .failure(CancellationError()) : (pendingResult ?? result)
        completed = true
        let continuation = continuation
        self.continuation = nil
        let session = session
        self.session = nil
        task = nil
        lock.unlock()
        session?.finishTasksAndInvalidate()
        continuation?.resume(with: finalResult)
    }
}

import Foundation
import Network
import Testing
@testable import GenImageRuntime

@Suite(.timeLimit(.minutes(1)))
struct FileDownloadDelegateTests {
    @Test func cancellationBeforeStartDoesNotCreateADownload() async throws {
        let destination = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let downloader = FileDownloadDelegate(destination: destination, expectedBytes: 5, progress: { _, _ in })
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            try await downloader.start(request: URLRequest(url: URL(string: "unsupported://download")!))
        }
        await #expect(throws: CancellationError.self) { try await task.value }
        #expect(!FileManager.default.fileExists(atPath: destination.path))
        #expect(!FileManager.default.fileExists(atPath: destination.appendingPathExtension("resume").path))
    }

    @Test(arguments: ["success", "sizeMismatch", "directory"])
    func onlyValidCompleteDownloadsReplaceTheDestination(mode: String) async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let destination = root.appendingPathComponent("weights")
        if mode == "directory" {
            try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
            try Data("keep".utf8).write(to: destination.appendingPathComponent("sentinel"))
        } else {
            try Data("old".utf8).write(to: destination)
        }
        let server = try DownloadFixtureServer(stall: false)
        let url = try await server.start()
        defer { server.stop() }
        let downloader = FileDownloadDelegate(destination: destination,
            expectedBytes: mode == "sizeMismatch" ? 99 : 5, progress: { _, _ in })
        if mode == "success" {
            try await downloader.start(request: URLRequest(url: url))
            #expect(try Data(contentsOf: destination) == Data("model".utf8))
        } else {
            await #expect(throws: (any Error).self) { try await downloader.start(request: URLRequest(url: url)) }
            if mode == "directory" {
                #expect(try Data(contentsOf: destination.appendingPathComponent("sentinel")) == Data("keep".utf8))
            } else {
                #expect(try Data(contentsOf: destination) == Data("old".utf8))
            }
        }
        #expect(try FileManager.default.contentsOfDirectory(atPath: root.path) == ["weights"])
    }

    @Test func cancellationReturnsAfterResumeMetadataHasFinishedWriting() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let destination = root.appendingPathComponent("weights")
        let server = try DownloadFixtureServer(stall: true)
        let url = try await server.start()
        defer { server.stop() }
        let downloader = FileDownloadDelegate(destination: destination, expectedBytes: 1_048_576, progress: { _, _ in })
        let task = Task { try await downloader.start(request: URLRequest(url: url)) }
        for await _ in server.requests { break }
        try await Task.sleep(for: .milliseconds(30))
        task.cancel()
        await #expect(throws: CancellationError.self) { try await task.value }
        let resume = destination.appendingPathExtension("resume")
        let marker = Data("new operation owns this file".utf8)
        try marker.write(to: resume)
        try await Task.sleep(for: .milliseconds(150))
        #expect(try Data(contentsOf: resume) == marker)
        #expect(!FileManager.default.fileExists(atPath: destination.path))
    }
}

private final class DownloadFixtureServer: @unchecked Sendable {
    private let listener: NWListener
    private let queue = DispatchQueue(label: "download-test")
    private let lock = NSLock()
    private var connections: [NWConnection] = []
    private let stall: Bool
    let requests: AsyncStream<Void>
    private let continuation: AsyncStream<Void>.Continuation

    init(stall: Bool) throws {
        self.stall = stall
        let parameters = NWParameters.tcp
        parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: .any)
        listener = try NWListener(using: parameters)
        (requests, continuation) = AsyncStream.makeStream()
    }

    func start() async throws -> URL {
        listener.newConnectionHandler = { [weak self] connection in
            guard let self else { connection.cancel(); return }
            lock.lock(); connections.append(connection); lock.unlock()
            connection.start(queue: queue)
            receive(connection, buffer: Data())
        }
        return try await withCheckedThrowingContinuation { continuation in
            listener.stateUpdateHandler = { [weak self] state in
                guard let self else { return }
                switch state {
                case .ready:
                    listener.stateUpdateHandler = nil
                    continuation.resume(returning: URL(string: "http://127.0.0.1:\(listener.port!.rawValue)/weights")!)
                case let .failed(error):
                    listener.stateUpdateHandler = nil
                    continuation.resume(throwing: error)
                default: break
                }
            }
            listener.start(queue: queue)
        }
    }

    private func receive(_ connection: NWConnection, buffer: Data) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 8_192) { [weak self] data, _, done, error in
            guard let self else { return }
            var buffer = buffer
            if let data { buffer.append(data) }
            guard buffer.range(of: Data("\r\n\r\n".utf8)) != nil else {
                if !done && error == nil { receive(connection, buffer: buffer) }
                return
            }
            let length = stall ? 1_048_576 : 5
            var response = Data("HTTP/1.1 200 OK\r\nContent-Length: \(length)\r\nAccept-Ranges: bytes\r\nETag: \"fixture\"\r\nConnection: close\r\n\r\n".utf8)
            response.append(stall ? Data(repeating: 65, count: 65_536) : Data("model".utf8))
            connection.send(content: response, isComplete: !stall, completion: .contentProcessed { _ in })
            continuation.yield(())
        }
    }

    func stop() {
        listener.cancel()
        lock.lock(); let pending = connections; connections.removeAll(); lock.unlock()
        pending.forEach { $0.cancel() }
        continuation.finish()
    }
}

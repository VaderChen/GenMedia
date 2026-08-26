import Foundation
import Network

public final class MCPHTTPServer: @unchecked Sendable {
    public enum State: Equatable, Sendable {
        case stopped
        case starting(URL)
        case ready(URL)
        case failed(String)
    }

    public static let defaultHost = "127.0.0.1"
    public static let defaultPort: UInt16 = 12_181
    public static let endpointPath = "/mcp"

    public var stateHandler: (@Sendable (State) -> Void)?

    private let dispatcher: MCPHTTPDispatcher
    private let queue = DispatchQueue(label: "GenImage.MCP.HTTP")
    private let lock = NSLock()
    private var listener: NWListener?
    private var connections: [ObjectIdentifier: MCPHTTPConnection] = [:]
    private var endpointURL: URL?

    public init(server: MCPServer = MCPServer()) {
        dispatcher = MCPHTTPDispatcher(server: server)
    }

    @discardableResult
    public func start(
        host: String = MCPHTTPServer.defaultHost,
        port rawPort: UInt16 = MCPHTTPServer.defaultPort
    ) throws -> URL {
        lock.lock()
        if let listener, listener.state != .cancelled, let endpointURL {
            lock.unlock()
            return endpointURL
        }
        lock.unlock()

        guard let port = NWEndpoint.Port(rawValue: rawPort),
              let endpointURL = URL(string: "http://\(host):\(rawPort)\(Self.endpointPath)") else {
            throw MCPHTTPServerError.invalidEndpoint(host: host, port: rawPort)
        }

        let parameters = NWParameters.tcp
        parameters.requiredLocalEndpoint = .hostPort(host: NWEndpoint.Host(host), port: port)
        let listener = try NWListener(using: parameters)
        listener.newConnectionHandler = { [weak self] connection in
            self?.accept(connection)
        }
        listener.stateUpdateHandler = { [weak self, weak listener] state in
            guard let self, let listener else { return }
            self.handleListenerState(state, listener: listener, endpointURL: endpointURL)
        }

        lock.lock()
        self.listener = listener
        self.endpointURL = endpointURL
        lock.unlock()

        emit(.starting(endpointURL))
        listener.start(queue: queue)
        return endpointURL
    }

    public func stop() {
        let listener: NWListener?
        let activeConnections: [MCPHTTPConnection]
        lock.lock()
        listener = self.listener
        self.listener = nil
        endpointURL = nil
        activeConnections = Array(connections.values)
        connections.removeAll()
        lock.unlock()

        listener?.cancel()
        activeConnections.forEach { $0.cancel() }
        emit(.stopped)
    }

    private func accept(_ connection: NWConnection) {
        let identifier = ObjectIdentifier(connection)
        let session = MCPHTTPConnection(
            connection: connection,
            dispatcher: dispatcher,
            endpointPath: Self.endpointPath
        ) { [weak self] in
            self?.removeConnection(identifier)
        }
        lock.lock()
        connections[identifier] = session
        lock.unlock()
        session.start(on: queue)
    }

    private func removeConnection(_ identifier: ObjectIdentifier) {
        lock.lock()
        connections.removeValue(forKey: identifier)
        lock.unlock()
    }

    private func handleListenerState(
        _ state: NWListener.State,
        listener: NWListener,
        endpointURL: URL
    ) {
        switch state {
        case .ready:
            emit(.ready(endpointURL))
        case let .failed(error):
            lock.lock()
            if self.listener === listener {
                self.listener = nil
                self.endpointURL = nil
            }
            lock.unlock()
            listener.cancel()
            emit(.failed(error.localizedDescription))
        case .cancelled:
            lock.lock()
            let isCurrent = self.listener === listener
            if isCurrent {
                self.listener = nil
                self.endpointURL = nil
            }
            lock.unlock()
            if isCurrent { emit(.stopped) }
        default:
            break
        }
    }

    private func emit(_ state: State) {
        stateHandler?(state)
    }
}

private actor MCPHTTPDispatcher {
    private let server: MCPServer

    init(server: MCPServer) {
        self.server = server
    }

    func response(for requestData: Data) async -> MCPHTTPPayloadResponse {
        do {
            guard let request = try JSONSerialization.jsonObject(with: requestData) as? [String: Any] else {
                return .json(status: 400, object: Self.errorObject(code: -32700, message: "Parse error"))
            }
            guard let response = await server.handle(request: request) else {
                return MCPHTTPPayloadResponse(status: 202, reason: "Accepted", body: Data())
            }
            return .json(status: 200, object: response)
        } catch {
            return .json(status: 400, object: Self.errorObject(code: -32700, message: "Parse error"))
        }
    }

    private static func errorObject(code: Int, message: String) -> [String: Any] {
        [
            "jsonrpc": "2.0",
            "id": NSNull(),
            "error": ["code": code, "message": message]
        ]
    }
}

private struct MCPHTTPPayloadResponse: Sendable {
    var status: Int
    var reason: String
    var body: Data

    static func json(status: Int, object: [String: Any]) -> Self {
        let body = (try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])) ?? Data()
        return Self(status: status, reason: reasonPhrase(for: status), body: body)
    }

    private static func reasonPhrase(for status: Int) -> String {
        switch status {
        case 200: "OK"
        case 202: "Accepted"
        case 204: "No Content"
        case 400: "Bad Request"
        case 404: "Not Found"
        case 405: "Method Not Allowed"
        case 413: "Payload Too Large"
        default: "Error"
        }
    }
}

private final class MCPHTTPConnection: @unchecked Sendable {
    private static let maximumRequestBytes = 4 * 1_024 * 1_024

    private let connection: NWConnection
    private let dispatcher: MCPHTTPDispatcher
    private let endpointPath: String
    private let completion: @Sendable () -> Void
    private var buffer = Data()
    private var isFinished = false

    init(
        connection: NWConnection,
        dispatcher: MCPHTTPDispatcher,
        endpointPath: String,
        completion: @escaping @Sendable () -> Void
    ) {
        self.connection = connection
        self.dispatcher = dispatcher
        self.endpointPath = endpointPath
        self.completion = completion
    }

    func start(on queue: DispatchQueue) {
        connection.stateUpdateHandler = { [weak self] state in
            switch state {
            case .failed, .cancelled:
                self?.finish()
            default:
                break
            }
        }
        connection.start(queue: queue)
        receive()
    }

    func cancel() {
        finish()
    }

    private func receive() {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 65_536) {
            [weak self] data, _, isComplete, error in
            guard let self, !self.isFinished else { return }
            if let data { self.buffer.append(data) }
            if self.buffer.count > Self.maximumRequestBytes {
                self.send(MCPHTTPPayloadResponse(status: 413, reason: "Payload Too Large", body: Data()))
                return
            }
            switch self.parseRequest() {
            case .incomplete:
                if isComplete || error != nil {
                    self.send(.json(status: 400, object: Self.errorObject("Incomplete HTTP request")))
                } else {
                    self.receive()
                }
            case let .failure(status, message):
                self.send(.json(status: status, object: Self.errorObject(message)))
            case let .request(method, path, body):
                self.handle(method: method, path: path, body: body)
            }
        }
    }

    private func handle(method: String, path: String, body: Data) {
        guard path == endpointPath else {
            send(.json(status: 404, object: Self.errorObject("Not found")))
            return
        }
        if method == "OPTIONS" {
            send(MCPHTTPPayloadResponse(status: 204, reason: "No Content", body: Data()))
            return
        }
        guard method == "POST" else {
            send(.json(status: 405, object: Self.errorObject("Use POST for JSON-RPC requests")))
            return
        }
        Task { [weak self, dispatcher] in
            let response = await dispatcher.response(for: body)
            self?.send(response)
        }
    }

    private func parseRequest() -> HTTPParseResult {
        let separator = Data("\r\n\r\n".utf8)
        guard let headerRange = buffer.range(of: separator),
              let headerText = String(data: buffer[..<headerRange.lowerBound], encoding: .utf8) else {
            return .incomplete
        }
        let lines = headerText.components(separatedBy: "\r\n")
        guard let requestLine = lines.first else {
            return .failure(400, "Invalid HTTP request")
        }
        let requestParts = requestLine.split(separator: " ", omittingEmptySubsequences: true)
        guard requestParts.count >= 2 else {
            return .failure(400, "Invalid HTTP request line")
        }
        let method = String(requestParts[0]).uppercased()
        let path = String(requestParts[1]).split(separator: "?", maxSplits: 1).first.map(String.init) ?? ""
        var headers: [String: String] = [:]
        for line in lines.dropFirst() {
            guard let separatorIndex = line.firstIndex(of: ":") else {
                return .failure(400, "Invalid HTTP header")
            }
            let name = line[..<separatorIndex].trimmingCharacters(in: .whitespaces).lowercased()
            let value = line[line.index(after: separatorIndex)...].trimmingCharacters(in: .whitespaces)
            guard !name.isEmpty else {
                return .failure(400, "Invalid HTTP header")
            }
            if name == "content-length", headers[name] != nil {
                return .failure(400, "Duplicate Content-Length header")
            }
            if let existing = headers[name] {
                headers[name] = "\(existing), \(value)"
            } else {
                headers[name] = value
            }
        }
        let contentLength = Int(headers["content-length"] ?? "0") ?? 0
        guard contentLength >= 0, contentLength <= Self.maximumRequestBytes else {
            return .failure(413, "Payload too large")
        }
        let bodyStart = headerRange.upperBound
        guard buffer.count - bodyStart >= contentLength else { return .incomplete }
        return .request(
            method: method,
            path: path,
            body: buffer.subdata(in: bodyStart..<(bodyStart + contentLength))
        )
    }

    private func send(_ response: MCPHTTPPayloadResponse) {
        guard !isFinished else { return }
        let contentType = response.body.isEmpty ? "text/plain" : "application/json"
        let header = """
        HTTP/1.1 \(response.status) \(response.reason)\r
        Content-Type: \(contentType)\r
        Content-Length: \(response.body.count)\r
        Cache-Control: no-store\r
        Access-Control-Allow-Origin: *\r
        Access-Control-Allow-Headers: Content-Type, Accept, MCP-Protocol-Version\r
        Access-Control-Allow-Methods: POST, OPTIONS\r
        Connection: close\r
        \r

        """
        var data = Data(header.utf8)
        data.append(response.body)
        connection.send(content: data, completion: .contentProcessed { [weak self] _ in
            self?.finish()
        })
    }

    private func finish() {
        guard !isFinished else { return }
        isFinished = true
        connection.cancel()
        completion()
    }

    private static func errorObject(_ message: String) -> [String: Any] {
        ["error": message]
    }
}

private enum HTTPParseResult {
    case incomplete
    case failure(Int, String)
    case request(method: String, path: String, body: Data)
}

public enum MCPHTTPServerError: LocalizedError, Sendable {
    case invalidEndpoint(host: String, port: UInt16)

    public var errorDescription: String? {
        switch self {
        case let .invalidEndpoint(host, port):
            "無效的 MCP HTTP 端點：\(host):\(port)"
        }
    }
}

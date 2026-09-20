import Foundation
import GenImageMCPServer
import Testing

@Suite(.serialized, .timeLimit(.minutes(1)))
struct MCPHTTPAccessTests {
    @Test func rejectsNonLoopbackBindings() throws {
        for host in ["0.0.0.0", "192.168.1.1", "example.com"] {
            #expect(throws: MCPHTTPServerError.self) { try MCPHTTPServer().start(host: host) }
        }
    }

    @Test func onlyNativeLocalJSONRequestsReachTheDispatcher() async throws {
        let server = MCPHTTPServer()
        let (states, continuation) = AsyncStream<MCPHTTPServer.State>.makeStream()
        server.stateHandler = { continuation.yield($0) }
        let endpoint = try server.start(port: 24_584)
        defer { server.stop(); continuation.finish() }
        for await state in states {
            if case .ready = state { break }
            if case let .failed(message) = state { throw NSError(domain: message, code: 1) }
        }
        let cases: [(String, String?, String?, String?, Int)] = [
            ("POST", nil, nil, "application/json", 200),
            ("POST", nil, "localhost:24584", "application/json; charset=utf-8", 200),
            ("POST", "https://untrusted.invalid", nil, "application/json", 403),
            ("POST", "null", nil, "application/json", 403),
            ("POST", endpoint.absoluteString, nil, "application/json", 403),
            ("POST", nil, "untrusted.invalid:24584", "application/json", 403),
            ("POST", nil, nil, "text/plain", 415),
            ("POST", nil, nil, nil, 415),
            ("OPTIONS", "https://untrusted.invalid", nil, nil, 403),
            ("OPTIONS", nil, nil, nil, 405)
        ]
        for (method, origin, host, contentType, status) in cases {
            var request = URLRequest(url: endpoint, timeoutInterval: 5)
            request.httpMethod = method
            request.setValue(origin, forHTTPHeaderField: "Origin")
            request.setValue(host, forHTTPHeaderField: "Host")
            request.setValue(contentType, forHTTPHeaderField: "Content-Type")
            request.httpBody = Data(#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#.utf8)
            let (data, response) = try await URLSession.shared.data(for: request)
            let http = try #require(response as? HTTPURLResponse)
            #expect(http.statusCode == status, "\(method), origin=\(origin ?? "none"), host=\(host ?? "default")")
            #expect(http.value(forHTTPHeaderField: "Access-Control-Allow-Origin") == nil)
            if status == 200 {
                let object = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
                #expect(object["result"] != nil)
            }
        }
    }
}

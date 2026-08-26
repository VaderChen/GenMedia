import Foundation
import GenImageMCPServer
import Testing

struct MCPServerTests {
    @Test func initializeNegotiatesStandardProtocol() async throws {
        let response = await MCPServer().handle(
            request: [
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": ["protocolVersion": "2025-06-18"]
            ]
        )
        let result = response?["result"] as? [String: Any]

        #expect(result?["protocolVersion"] as? String == "2025-06-18")
        #expect(result?["serverInfo"] != nil)
    }

    @Test func toolsListExposesStableGenImageTools() async throws {
        let response = await MCPServer().handle(
            request: [
                "jsonrpc": "2.0",
                "id": "tools",
                "method": "tools/list"
            ]
        )
        let result = response?["result"] as? [String: Any]
        let tools = result?["tools"] as? [[String: Any]] ?? []
        let names = Set(tools.compactMap { $0["name"] as? String })

        #expect(names.contains("genimage_models_list"))
        #expect(names.contains("genimage_profiles_list"))
        #expect(names.contains("genimage_upscale_image"))
        #expect(names.contains("genimage_generate_image"))
        #expect(names.contains("genimage_edit_image"))
        #expect(names.contains("genimage_describe_image"))
        #expect(names.contains("genimage_generate_subtitle"))
        #expect(names.count == 7)
    }

    @Test func unknownMethodsReturnJSONRPCError() async throws {
        let response = await MCPServer().handle(
            request: [
                "jsonrpc": "2.0",
                "id": 3,
                "method": "missing/method"
            ]
        )
        let error = response?["error"] as? [String: Any]

        #expect(error?["code"] as? Int == -32601)
    }

    @Test func httpTransportRespondsWithoutAppBridge() async throws {
        let server = MCPHTTPServer()
        let port: UInt16 = 12_381
        let endpoint = try server.start(port: port)
        defer { server.stop() }

        try await Task.sleep(for: .milliseconds(150))
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: [
            "jsonrpc": "2.0",
            "id": "http-tools",
            "method": "tools/list"
        ])

        let (data, response) = try await URLSession.shared.data(for: request)
        let httpResponse = try #require(response as? HTTPURLResponse)
        let object = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let result = try #require(object["result"] as? [String: Any])
        let tools = result["tools"] as? [[String: Any]] ?? []

        #expect(httpResponse.statusCode == 200)
        #expect(tools.count == 7)
    }
}

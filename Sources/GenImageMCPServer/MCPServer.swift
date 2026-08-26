import Foundation

public final class MCPServer: @unchecked Sendable {
    public static let protocolVersion = "2025-06-18"
    private let tools: MCPToolRegistry

    public init(tools: MCPToolRegistry = MCPToolRegistry()) {
        self.tools = tools
    }

    public func runStdio() async {
        while let line = readLine(strippingNewline: true) {
            guard !line.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { continue }

            do {
                guard let data = line.data(using: .utf8),
                      let request = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                    write(response: errorResponse(id: nil, code: -32700, message: "Parse error"))
                    continue
                }

                if let response = await handle(request: request) {
                    write(response: response)
                }
            } catch {
                write(response: errorResponse(id: nil, code: -32700, message: "Parse error"))
            }
        }
    }

    public func handle(request: [String: Any]) async -> [String: Any]? {
        let id = request["id"]
        guard request["jsonrpc"] as? String == "2.0",
              let method = request["method"] as? String else {
            return errorResponse(id: id, code: -32600, message: "Invalid Request")
        }
        let params = request["params"] as? [String: Any] ?? [:]

        switch method {
        case "initialize":
            let requestedVersion = params["protocolVersion"] as? String
            let negotiatedVersion = requestedVersion == Self.protocolVersion
                ? Self.protocolVersion
                : Self.protocolVersion
            return successResponse(
                id: id,
                result: [
                    "protocolVersion": negotiatedVersion,
                    "capabilities": [
                        "tools": ["listChanged": false]
                    ],
                    "serverInfo": [
                        "name": "genimage",
                        "version": "0.1.0"
                    ],
                    "instructions": "Local image generation model, Profile, diagnostics, and Upscale tools. File paths remain local."
                ]
            )

        case "notifications/initialized", "notifications/cancelled":
            return nil

        case "ping":
            return successResponse(id: id, result: [:])

        case "tools/list":
            return successResponse(id: id, result: ["tools": tools.definitions])

        case "tools/call":
            guard let name = params["name"] as? String else {
                return errorResponse(id: id, code: -32602, message: "Missing tool name")
            }
            let arguments = params["arguments"] as? [String: Any] ?? [:]
            let result = await tools.call(name: name, arguments: arguments)
            return successResponse(id: id, result: result)

        default:
            return errorResponse(id: id, code: -32601, message: "Method not found: \(method)")
        }
    }

    private func successResponse(id: Any?, result: [String: Any]) -> [String: Any] {
        [
            "jsonrpc": "2.0",
            "id": id ?? NSNull(),
            "result": result
        ]
    }

    private func errorResponse(id: Any?, code: Int, message: String) -> [String: Any] {
        [
            "jsonrpc": "2.0",
            "id": id ?? NSNull(),
            "error": [
                "code": code,
                "message": message
            ]
        ]
    }

    private func write(response: [String: Any]) {
        guard JSONSerialization.isValidJSONObject(response),
              let data = try? JSONSerialization.data(withJSONObject: response, options: [.sortedKeys]) else {
            return
        }
        FileHandle.standardOutput.write(data)
        FileHandle.standardOutput.write(Data("\n".utf8))
    }
}

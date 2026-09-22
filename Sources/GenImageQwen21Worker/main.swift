import Foundation
import MLX
import QwenImage21Runtime

@main
struct GenImageQwen21Worker {
    static func event(_ type: String, value: Double? = nil, message: String? = nil) {
        var payload: [String: Any] = ["type": type]
        if let value { payload["value"] = value }
        if let message { payload["message"] = message }
        guard let data = try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys]) else { return }
        FileHandle.standardOutput.write(data + Data([10]))
    }
    static func main() {
        do {
            let arguments = CommandLine.arguments
            guard arguments.count == 3, arguments[1] == "--request" else {
                throw Qwen21Error.invalid("用法：GenImageQwen21Worker --request request.json")
            }
            let request = try JSONDecoder().decode(Qwen21Request.self,
                from: Data(contentsOf: URL(fileURLWithPath: arguments[2])))
            // Limit the allocator cache, not working tensors; component lifetimes are staged.
            Memory.cacheLimit = 256 * 1024 * 1024
            try Qwen21Pipeline.generate(request) { event("progress", value: $0) }
            event("completed", value: 1)
        } catch {
            event("error", message: error.localizedDescription)
            exit(1)
        }
    }
}

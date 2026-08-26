import Combine
import Foundation
import GenImageMCPServer

@MainActor
final class LocalMCPServiceController: ObservableObject {
    @Published private(set) var isEnabled = false
    @Published private(set) var isRunning = false
    @Published private(set) var endpointURL: String?
    @Published private(set) var errorMessage: String?

    private static let enabledDefaultsKey = "GenImage.mcpHTTPEnabled.v1"
    private let server: MCPHTTPServer

    init(server: MCPHTTPServer = MCPHTTPServer()) {
        self.server = server
        server.stateHandler = { [weak self] state in
            Task { @MainActor in
                self?.receive(state)
            }
        }
        if UserDefaults.standard.bool(forKey: Self.enabledDefaultsKey) {
            setEnabled(true)
        }
    }

    func setEnabled(_ enabled: Bool) {
        isEnabled = enabled
        errorMessage = nil
        UserDefaults.standard.set(enabled, forKey: Self.enabledDefaultsKey)
        if enabled {
            do {
                endpointURL = try server.start().absoluteString
            } catch {
                isEnabled = false
                isRunning = false
                endpointURL = nil
                errorMessage = error.localizedDescription
                UserDefaults.standard.set(false, forKey: Self.enabledDefaultsKey)
            }
        } else {
            isRunning = false
            endpointURL = nil
            server.stop()
        }
    }

    private func receive(_ state: MCPHTTPServer.State) {
        switch state {
        case .stopped:
            isRunning = false
            if !isEnabled { endpointURL = nil }
        case let .starting(url):
            isRunning = false
            endpointURL = url.absoluteString
        case let .ready(url):
            isRunning = true
            endpointURL = url.absoluteString
            errorMessage = nil
        case let .failed(message):
            isEnabled = false
            isRunning = false
            endpointURL = nil
            errorMessage = message
            UserDefaults.standard.set(false, forKey: Self.enabledDefaultsKey)
        }
    }
}

import Foundation
import LocalAuthentication
import Security

public enum HuggingFaceTokenStore {
    private static let service = "com.genimage.huggingface"
    private static let account = "api-token"
    private static let configuredMarkerKey = "com.genimage.huggingface.token-configured"

    public static func token() -> String? {
        let authenticationContext = LAContext()
        authenticationContext.interactionNotAllowed = true
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
            kSecUseAuthenticationContext as String: authenticationContext
        ]
        var result: CFTypeRef?
        if SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess,
           let data = result as? Data,
           let storedToken = String(data: data, encoding: .utf8),
           !storedToken.isEmpty {
            UserDefaults.standard.set(true, forKey: configuredMarkerKey)
            return storedToken
        }
        if let environmentToken = ProcessInfo.processInfo.environment["HF_TOKEN"]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !environmentToken.isEmpty {
            return environmentToken
        }
        let tokenURL = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/token")
        guard let token = try? String(contentsOf: tokenURL, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines),
              !token.isEmpty else { return nil }
        return token
    }

    public static func isConfigured() -> Bool {
        if let environmentToken = ProcessInfo.processInfo.environment["HF_TOKEN"]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !environmentToken.isEmpty {
            return true
        }
        let tokenURL = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/token")
        if let attributes = try? FileManager.default.attributesOfItem(atPath: tokenURL.path),
           let fileSize = attributes[.size] as? NSNumber,
           fileSize.int64Value > 0 {
            return true
        }

        let authenticationContext = LAContext()
        authenticationContext.interactionNotAllowed = true
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnAttributes as String: true,
            kSecReturnData as String: false,
            kSecMatchLimit as String: kSecMatchLimitOne,
            kSecUseAuthenticationContext as String: authenticationContext
        ]
        var result: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        if status == errSecSuccess {
            UserDefaults.standard.set(true, forKey: configuredMarkerKey)
            return true
        }
        return false
    }

    public static func save(_ rawToken: String) throws {
        let token = rawToken.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !token.isEmpty else {
            throw HuggingFaceTokenStoreError.emptyToken
        }
        let data = Data(token.utf8)
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account
        ]
        let attributes: [String: Any] = [
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
        ]
        let status = SecItemUpdate(query as CFDictionary, attributes as CFDictionary)
        if status == errSecItemNotFound {
            var item = query
            item.merge(attributes) { _, new in new }
            let addStatus = SecItemAdd(item as CFDictionary, nil)
            guard addStatus == errSecSuccess else {
                throw HuggingFaceTokenStoreError.keychainStatus(addStatus)
            }
        } else if status != errSecSuccess {
            throw HuggingFaceTokenStoreError.keychainStatus(status)
        }
        UserDefaults.standard.set(true, forKey: configuredMarkerKey)
    }

    public static func delete() throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account
        ]
        let status = SecItemDelete(query as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw HuggingFaceTokenStoreError.keychainStatus(status)
        }
        UserDefaults.standard.removeObject(forKey: configuredMarkerKey)
    }
}

public enum HuggingFaceTokenStoreError: LocalizedError, Sendable {
    case emptyToken
    case keychainStatus(OSStatus)

    public var errorDescription: String? {
        switch self {
        case .emptyToken:
            "Hugging Face API Token 不可為空白。"
        case let .keychainStatus(status):
            "無法存取 macOS Keychain（狀態碼 \(status)）。"
        }
    }
}

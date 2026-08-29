import Foundation
import LocalAuthentication
import Security

/// Secure storage for the Civitai API token used by model downloads.
///
/// The environment variable remains supported for command-line and legacy
/// setups, while tokens entered in the app are stored only in Keychain.
public enum CivitaiTokenStore {
    private static let service = "com.genimage.civitai"
    private static let account = "api-token"
    private static let configuredMarkerKey = "com.genimage.civitai.token-configured"

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
        guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess,
              let data = result as? Data,
              let storedToken = String(data: data, encoding: .utf8),
              !storedToken.isEmpty else {
            if let environmentToken = ProcessInfo.processInfo.environment["CIVITAI_TOKEN"]?
                .trimmingCharacters(in: .whitespacesAndNewlines),
               !environmentToken.isEmpty {
                return environmentToken
            }
            return nil
        }
        UserDefaults.standard.set(true, forKey: configuredMarkerKey)
        return storedToken
    }

    public static func isConfigured() -> Bool {
        if UserDefaults.standard.bool(forKey: configuredMarkerKey) {
            return true
        }
        if let environmentToken = ProcessInfo.processInfo.environment["CIVITAI_TOKEN"]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !environmentToken.isEmpty {
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
            throw CivitaiTokenStoreError.emptyToken
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
                throw CivitaiTokenStoreError.keychainStatus(addStatus)
            }
        } else if status != errSecSuccess {
            throw CivitaiTokenStoreError.keychainStatus(status)
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
            throw CivitaiTokenStoreError.keychainStatus(status)
        }
        UserDefaults.standard.removeObject(forKey: configuredMarkerKey)
    }
}

public enum CivitaiTokenStoreError: LocalizedError, Sendable {
    case emptyToken
    case keychainStatus(OSStatus)

    public var errorDescription: String? {
        switch self {
        case .emptyToken:
            "Civitai API Token 不可為空白。"
        case let .keychainStatus(status):
            "無法存取 macOS Keychain（狀態碼 \(status)）。"
        }
    }
}

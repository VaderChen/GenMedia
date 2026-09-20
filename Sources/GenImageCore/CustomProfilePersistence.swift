import Foundation

public enum CustomProfilePersistence {
    public static let key = "GenImage.customProfiles.v1"

    public static func load(from defaults: UserDefaults = .standard) throws -> [InferenceProfile] {
        guard let data = defaults.data(forKey: key) else { return [] }
        let profiles = try JSONDecoder().decode([InferenceProfile].self, from: data)
        guard profiles.allSatisfy({ !$0.isBuiltIn }),
              Set(profiles.map(\.id)).count == profiles.count else {
            throw CocoaError(.coderReadCorrupt)
        }
        return profiles
    }

    public static func save(_ profiles: [InferenceProfile], to defaults: UserDefaults = .standard) throws {
        let data = try JSONEncoder().encode(profiles.filter { !$0.isBuiltIn })
        defaults.set(data, forKey: key)
    }
}

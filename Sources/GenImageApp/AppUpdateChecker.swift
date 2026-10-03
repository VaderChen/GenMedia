import Foundation

struct AppUpdateInfo: Codable, Hashable, Sendable {
    let currentVersion: String
    let latestVersion: String
    let releaseName: String
    let releaseURL: URL
}

enum AppUpdateChecker {
    private static let defaultRepository = "VaderChen/GenMedia"

    static func availableUpdate() async -> AppUpdateInfo? {
        guard let currentVersion = currentApplicationVersion(),
              let repository = updateRepository(),
              let apiURL = URL(
                string: "https://api.github.com/repos/\(repository)/releases/latest"
              ) else {
            return nil
        }

        var request = URLRequest(url: apiURL)
        request.timeoutInterval = 15
        request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        request.setValue("GenImage/\(currentVersion)", forHTTPHeaderField: "User-Agent")
        request.setValue("2022-11-28", forHTTPHeaderField: "X-GitHub-Api-Version")

        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200 else {
                return nil
            }
            let release = try JSONDecoder().decode(GitHubRelease.self, from: data)
            guard !release.draft,
                  !release.prerelease,
                  let latestVersion = normalizedVersion(release.tagName),
                  isNewer(latestVersion, than: currentVersion),
                  let releaseURL = URL(string: release.htmlURL),
                  releaseURL.scheme == "https",
                  releaseURL.host == "github.com" else {
                return nil
            }
            let releaseName = release.name.map {
                $0.trimmingCharacters(in: .whitespacesAndNewlines)
            }
            return AppUpdateInfo(
                currentVersion: currentVersion,
                latestVersion: latestVersion,
                releaseName: releaseName?.nonEmpty ?? release.tagName,
                releaseURL: releaseURL
            )
        } catch {
            return nil
        }
    }

    private static func currentApplicationVersion() -> String? {
        let environment = ProcessInfo.processInfo.environment
        if let version = environment["GENIMAGE_VERSION"] {
            return applicationVersion(version, build: environment["GENIMAGE_BUILD_NUMBER"])
        }
        let bundleVersion = Bundle.main.object(
            forInfoDictionaryKey: "CFBundleShortVersionString"
        ) as? String
        let bundleBuild = Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String
        return applicationVersion(bundleVersion ?? "", build: environment["GENIMAGE_BUILD_NUMBER"] ?? bundleBuild)
    }

    static func applicationVersion(_ version: String, build: String?) -> String? {
        guard let normalized = normalizedVersion(version) else { return nil }
        guard ParsedVersion(normalized)?.build == nil,
              let build, ParsedVersion.validBuild(build) else { return normalized }
        return "\(normalized) build \(build)"
    }

    private static func updateRepository() -> String? {
        let configured = (ProcessInfo.processInfo.environment["GENIMAGE_UPDATE_REPOSITORY"] ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let repository = configured.nonEmpty ?? defaultRepository
        let parts = repository.split(separator: "/", omittingEmptySubsequences: true)
        guard parts.count == 2,
              parts.allSatisfy({ part in
                  part.allSatisfy { character in
                      character.isLetter
                          || character.isNumber
                          || "-_.".contains(character)
                  }
              }) else {
            return nil
        }
        return parts.joined(separator: "/")
    }

    static func normalizedVersion(_ rawValue: String) -> String? {
        let trimmed = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let firstDigit = trimmed.firstIndex(where: { $0.isNumber }) else { return nil }
        let version = String(trimmed[firstDigit...])
        let core = version.split(separator: "+", maxSplits: 1).first.map(String.init) ?? version
        guard ParsedVersion(core) != nil else { return nil }
        return core.replacingOccurrences(of: "-build.", with: " build ")
    }

    static func isNewer(_ candidate: String, than current: String) -> Bool {
        guard let candidateVersion = ParsedVersion(candidate),
              let currentVersion = ParsedVersion(current) else {
            return false
        }
        return candidateVersion > currentVersion
    }

    private struct GitHubRelease: Decodable {
        let tagName: String
        let name: String?
        let htmlURL: String
        let draft: Bool
        let prerelease: Bool

        enum CodingKeys: String, CodingKey {
            case tagName = "tag_name"
            case name
            case htmlURL = "html_url"
            case draft
            case prerelease
        }
    }

    private struct ParsedVersion: Comparable {
        let numbers: [Int]
        let prerelease: String?
        let build: Int?

        init?(_ value: String) {
            let buildParts = value.replacingOccurrences(of: " build ", with: "-build.")
                .components(separatedBy: "-build.")
            guard buildParts.count <= 2 else { return nil }
            if buildParts.count == 2 {
                guard Self.validBuild(buildParts[1]) else { return nil }
                build = Int(buildParts[1])
            } else {
                build = nil
            }
            let pieces = buildParts[0].split(separator: "-", maxSplits: 1, omittingEmptySubsequences: false)
            let numberParts = pieces[0].split(separator: ".", omittingEmptySubsequences: false)
            guard !numberParts.isEmpty,
                  numberParts.allSatisfy({ part in
                      !part.isEmpty && part.allSatisfy { $0.isNumber }
                  }) else {
                return nil
            }
            numbers = numberParts.compactMap { Int($0) }
            guard numbers.count == numberParts.count else { return nil }
            prerelease = pieces.count == 2 && !pieces[1].isEmpty ? String(pieces[1]) : nil
        }

        static func validBuild(_ value: String) -> Bool {
            guard value.utf8.count == 4, value.utf8.allSatisfy({ (48...57).contains($0) }),
                  let hours = Int(value.prefix(2)), let minutes = Int(value.suffix(2)) else { return false }
            return hours < 24 && minutes < 60
        }

        static func < (lhs: ParsedVersion, rhs: ParsedVersion) -> Bool {
            let count = max(lhs.numbers.count, rhs.numbers.count)
            for index in 0..<count {
                let left = index < lhs.numbers.count ? lhs.numbers[index] : 0
                let right = index < rhs.numbers.count ? rhs.numbers[index] : 0
                if left != right { return left < right }
            }
            switch (lhs.prerelease, rhs.prerelease) {
            case (nil, nil): return (lhs.build ?? -1) < (rhs.build ?? -1)
            case (.some, nil): return true
            case (nil, .some): return false
            case let (.some(left), .some(right)):
                if left == right { return (lhs.build ?? -1) < (rhs.build ?? -1) }
                return left.compare(right, options: [.numeric, .caseInsensitive]) == .orderedAscending
            }
        }
    }
}

private extension String {
    var nonEmpty: String? { isEmpty ? nil : self }
}

import Foundation

/// Tracks the same Unicode-scalar quality rules across streaming chunk boundaries.
struct DescriptionTextStatistics {
    private static let punctuationAndSymbols = CharacterSet.punctuationCharacters.union(.symbols)
    private var scalarCount = 0
    private var meaningfulCount = 0
    private var punctuationCount = 0
    private var previousScalar: Unicode.Scalar?
    private var currentRun = 0
    private var longestRun = 0

    mutating func append(_ text: String) {
        for scalar in text.unicodeScalars where !CharacterSet.whitespacesAndNewlines.contains(scalar) {
            scalarCount += 1
            if CharacterSet.letters.contains(scalar) || CharacterSet.decimalDigits.contains(scalar) {
                meaningfulCount += 1
            }
            if Self.punctuationAndSymbols.contains(scalar) {
                punctuationCount += 1
            }
            currentRun = scalar == previousScalar ? currentRun + 1 : 1
            longestRun = max(longestRun, currentRun)
            previousScalar = scalar
        }
    }

    var isClearlyDegenerate: Bool {
        scalarCount >= 16 && (longestRun >= 10
            || (scalarCount >= 40 && Double(punctuationCount) / Double(scalarCount) > 0.5))
    }

    var isUsable: Bool {
        scalarCount >= 16 && !isClearlyDegenerate && meaningfulCount >= 8
            && Double(meaningfulCount) / Double(scalarCount) >= 0.3
    }
}

import Foundation

public struct MiniMaxMusic3PromptConfiguration: Equatable, Sendable {
    public let audioCfgTokenID: Int
    public let maxPromptTokens: Int

    public init(audioCfgTokenID: Int = 151_654, maxPromptTokens: Int = 5_000) {
        self.audioCfgTokenID = audioCfgTokenID
        self.maxPromptTokens = maxPromptTokens
    }
}

public enum MiniMaxMusic3PromptError: LocalizedError, Sendable {
    case emptyCaption
    case emptyLyrics
    case invalidTokenSequence([Int])
    case promptTooLong(actual: Int, maximum: Int)

    public var errorDescription: String? {
        switch self {
        case .emptyCaption:
            "prompt 必須是非空白字串。"
        case .emptyLyrics:
            "lyrics 必須是非空白字串。"
        case let .invalidTokenSequence(ids):
            "tokenizer 必須回傳至少三個 token，實際為 \(ids.count) 個。"
        case let .promptTooLong(actual, maximum):
            "組合後的 prompt 有 \(actual) 個 token，超過上限 \(maximum)。"
        }
    }
}

public enum MiniMaxMusic3Prompt {
    public static let imStart = "<|im_start|>"
    public static let imEnd = "<|im_end|>"
    public static let captionStart = "<|caption_start|>"
    public static let captionEnd = "<|caption_end|>"
    public static let lyricsStart = "<|lyrics_start|>"
    public static let lyricsEnd = "<|lyrics_end|>"
    public static let audioStart = "<|audio_start|>"

    public static func cleanCaption(_ caption: String) throws -> String {
        guard !caption.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw MiniMaxMusic3PromptError.emptyCaption
        }

        var text = replaceMatches(
            in: caption,
            pattern: #"<\|([^|]*)\|>"#
        ) { match, source in
            let inner = capture(match, index: 1, in: source)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let parts = inner.split(maxSplits: 1, whereSeparator: { $0.isWhitespace })
            guard parts.count == 2 else { return inner }
            return "\(parts[0]) is \(parts[1])"
        }

        var lines = splitLines(text)
        lines = lines.map { line in
            var value = replaceMatches(in: line, pattern: #"^\s{0,3}#{1,6}\s+"#) { _, _ in "" }
            value = replaceMatches(in: value, pattern: #"^\s*[*+-]\s+"#) { _, _ in "" }
            while value.contains("**") {
                let updated = replaceMatches(in: value, pattern: #"\*\*([^*]+)\*\*"#) { match, source in
                    capture(match, index: 1, in: source)
                }
                if updated == value { break }
                value = updated
            }
            value = replaceMatches(
                in: value,
                pattern: #"(?<!\*)\*([^*\n]+)\*(?!\*)"#
            ) { match, source in
                capture(match, index: 1, in: source)
            }
            return trailingWhitespaceRemoved(value)
        }

        lines = lines.map { line in
            let stripped = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard stripped.count >= 3,
                  stripped.allSatisfy({ $0 == "-" || $0 == "*" || $0 == "_" }) else {
                return line
            }
            return ""
        }
        text = lines.joined(separator: "\n")
        text = text.replacingOccurrences(of: "• ", with: "")
        text = text.replacingOccurrences(of: "    ", with: "")
        while text.contains("\n\n") {
            text = text.replacingOccurrences(of: "\n\n", with: "\n")
        }
        return text
    }

    public static func normalizeLyrics(_ lyrics: String) throws -> String {
        guard !lyrics.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw MiniMaxMusic3PromptError.emptyLyrics
        }

        let lines = splitLinesPreservingTrailingEmpty(lyrics).map { line in
            let match = firstMatch(in: line, pattern: #"^[ \t]*((?:\[[^\]]+\][ \t]*)+)"#)
            guard let match else { return line }
            let prefix = capture(match, index: 1, in: line)
            return prefix.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        var text = lines.joined(separator: "\n")
        text = text.replacingOccurrences(of: "] ", with: "]\n")
        text = text.replacingOccurrences(of: " [", with: "\n[")
        text = text.replacingOccurrences(of: " ^ ", with: "\n")
        text = replaceMatches(in: text, pattern: #"\[([^\]]+)\]"#) { match, source in
            "[\(capture(match, index: 1, in: source).lowercased())]"
        }
        return "[start]\n\(text)"
    }

    public static func buildPromptText(prompt: String, lyrics: String) throws -> String {
        "\(imStart)\(captionStart)\(try cleanCaption(prompt))\(captionEnd)\(lyricsStart)\(try normalizeLyrics(lyrics))\(lyricsEnd)\(imEnd)\(audioStart)"
    }

    public static func buildCFGTokenIDs(
        prompt: String,
        lyrics: String,
        encode: (String) -> [Int],
        configuration: MiniMaxMusic3PromptConfiguration = .init()
    ) throws -> [[Int]] {
        let text = try buildPromptText(prompt: prompt, lyrics: lyrics)
        let conditional = encode(text)
        guard conditional.count <= configuration.maxPromptTokens else {
            throw MiniMaxMusic3PromptError.promptTooLong(
                actual: conditional.count,
                maximum: configuration.maxPromptTokens
            )
        }
        guard conditional.count >= 3 else {
            throw MiniMaxMusic3PromptError.invalidTokenSequence(conditional)
        }
        var unconditional = conditional
        unconditional.replaceSubrange(
            unconditional.index(after: unconditional.startIndex)..<unconditional.index(
                unconditional.endIndex,
                offsetBy: -2
            ),
            with: repeatElement(configuration.audioCfgTokenID, count: unconditional.count - 3)
        )
        return [conditional, unconditional]
    }

    private static func splitLines(_ text: String) -> [String] {
        var normalized = text.replacingOccurrences(of: "\r\n", with: "\n")
        normalized = normalized.replacingOccurrences(of: "\r", with: "\n")
        var lines = normalized.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        if normalized.hasSuffix("\n") { lines.removeLast() }
        return lines
    }

    private static func splitLinesPreservingTrailingEmpty(_ text: String) -> [String] {
        let normalized = text.replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
        return normalized.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
    }

    private static func trailingWhitespaceRemoved(_ text: String) -> String {
        var result = text
        while result.last?.isWhitespace == true {
            result.removeLast()
        }
        return result
    }

    private static func firstMatch(in text: String, pattern: String) -> NSTextCheckingResult? {
        guard let expression = try? NSRegularExpression(pattern: pattern) else { return nil }
        return expression.firstMatch(in: text, range: NSRange(text.startIndex..., in: text))
    }

    private static func replaceMatches(
        in text: String,
        pattern: String,
        replacement: (NSTextCheckingResult, String) -> String
    ) -> String {
        guard let expression = try? NSRegularExpression(pattern: pattern) else { return text }
        let source = text
        let range = NSRange(source.startIndex..., in: source)
        let matches = expression.matches(in: source, range: range)
        var result = ""
        var cursor = source.startIndex
        for match in matches {
            guard let matchRange = Range(match.range, in: source) else { continue }
            result += source[cursor..<matchRange.lowerBound]
            result += replacement(match, source)
            cursor = matchRange.upperBound
        }
        result += source[cursor...]
        return result
    }

    private static func capture(_ match: NSTextCheckingResult, index: Int, in source: String) -> String {
        guard match.numberOfRanges > index,
              let range = Range(match.range(at: index), in: source) else {
            return ""
        }
        return String(source[range])
    }
}

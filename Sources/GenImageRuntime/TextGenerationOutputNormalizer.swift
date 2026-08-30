import Foundation

enum TextGenerationOutputNormalizer {
    static func finalAnswer(_ output: String) -> String {
        var text = output.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return "" }

        if let finalMarker = text.range(
            of: "Final answer:",
            options: [.caseInsensitive, .diacriticInsensitive]
        ) {
            text = String(text[finalMarker.upperBound...])
        }

        let openingTags = ["<think>", "<|thinking|>", "<|assistant_thinking|>"]
        let closingTags = ["</think>", "<|/thinking|>", "<|end_thinking|>"]

        while let opening = Self.firstRange(of: openingTags, in: text) {
            guard let closing = Self.firstRange(
                of: closingTags,
                in: text,
                from: opening.upperBound
            ) else {
                text = String(text[..<opening.lowerBound])
                break
            }
            text.removeSubrange(opening.lowerBound..<closing.upperBound)
        }

        if let closing = Self.lastRange(of: closingTags, in: text) {
            text = String(text[closing.upperBound...])
        }

        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func firstRange(
        of candidates: [String],
        in text: String,
        from start: String.Index? = nil
    ) -> Range<String.Index>? {
        let lowerBound = start ?? text.startIndex
        return candidates.compactMap { candidate in
            text.range(
                of: candidate,
                options: [.caseInsensitive, .diacriticInsensitive],
                range: lowerBound..<text.endIndex
            )
        }.min { lhs, rhs in
            lhs.lowerBound < rhs.lowerBound
        }
    }

    private static func lastRange(
        of candidates: [String],
        in text: String
    ) -> Range<String.Index>? {
        candidates.compactMap { candidate in
            text.range(
                of: candidate,
                options: [.caseInsensitive, .diacriticInsensitive],
                range: text.startIndex..<text.endIndex
            )
        }.max { lhs, rhs in
            lhs.lowerBound < rhs.lowerBound
        }
    }
}

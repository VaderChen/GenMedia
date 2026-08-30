import Foundation

struct SubtitleTranslationItem: Codable, Sendable {
    let index: Int
    let text: String
}

enum SubtitleTranslationResponseParser {
    static func decode(_ output: String) throws -> [SubtitleTranslationItem] {
        let normalized = TextGenerationOutputNormalizer.finalAnswer(output)
        let closingTags = ["</think>", "<|/thinking|>", "<|end_thinking|>"]
        let openingTags = ["<think>", "<|thinking|>", "<|assistant_thinking|>"]
        let hasUnexpectedClosingMarker = Self.containsAny(closingTags, in: output)
            && !Self.containsAny(openingTags, in: output)
        if normalized.isEmpty, hasUnexpectedClosingMarker {
            throw SubtitleTranslationError.unexpectedThinkingMarker(
                rawOutputSample: String(output.prefix(200))
            )
        }

        if normalized.isEmpty, !output.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            throw SubtitleTranslationError.reasoningOnly(
                rawOutputSample: String(output.prefix(200))
            )
        }

        var candidates = [normalized]
        let balancedCandidates = balancedJSONCandidates(in: normalized)
        candidates.append(contentsOf: balancedCandidates)

        var seen = Set<String>()
        for candidate in candidates where seen.insert(candidate).inserted {
            if let items = decodeItems(from: candidate) {
                return items
            }
        }
        if hasUnexpectedClosingMarker {
            throw SubtitleTranslationError.unexpectedThinkingMarker(
                rawOutputSample: String(output.prefix(200))
            )
        }
        if (normalized.contains("[") || normalized.contains("{")),
           balancedCandidates.isEmpty {
            throw SubtitleTranslationError.truncatedJSON(
                rawOutputSample: String(output.prefix(200))
            )
        }
        if !normalized.contains("[") && !normalized.contains("{") {
            throw SubtitleTranslationError.reasoningOnly(
                rawOutputSample: String(output.prefix(200))
            )
        }
        throw SubtitleTranslationError.invalidResponse(
            rawOutputSample: String(output.prefix(200))
        )
    }

    private static func containsAny(_ candidates: [String], in text: String) -> Bool {
        candidates.contains { text.range(of: $0, options: [.caseInsensitive]) != nil }
    }

    private static func decodeItems(
        from candidate: String,
        depth: Int = 0
    ) -> [SubtitleTranslationItem]? {
        guard depth < 4,
              let data = candidate.data(using: .utf8) else {
            return nil
        }

        if let items = try? JSONDecoder().decode(
            [SubtitleTranslationItem].self,
            from: data
        ) {
            return items
        }

        if let nestedJSON = try? JSONDecoder().decode(String.self, from: data) {
            return decodeItems(from: nestedJSON, depth: depth + 1)
        }

        guard let object = try? JSONSerialization.jsonObject(with: data),
              let dictionary = object as? [String: Any] else {
            return nil
        }

        let preferredKeys = [
            "translations", "translated_items", "items", "results", "data", "output", "content", "response"
        ]
        for key in preferredKeys {
            guard let value = dictionary[key] else {
                continue
            }
            if let nestedString = value as? String,
               let items = decodeItems(from: nestedString, depth: depth + 1) {
                return items
            }
            guard JSONSerialization.isValidJSONObject(value),
                  let nestedData = try? JSONSerialization.data(withJSONObject: value) else {
                continue
            }
            let nested = String(decoding: nestedData, as: UTF8.self)
            if let items = decodeItems(from: nested, depth: depth + 1) {
                return items
            }
        }
        return nil
    }

    private static func balancedJSONCandidates(in text: String) -> [String] {
        let characters = Array(text)
        var candidates: [String] = []
        for start in characters.indices where characters[start] == "[" || characters[start] == "{" {
            var stack: [Character] = []
            var inString = false
            var escaped = false

            for index in start..<characters.endIndex {
                let character = characters[index]
                if inString {
                    if escaped {
                        escaped = false
                    } else if character == "\\" {
                        escaped = true
                    } else if character == "\"" {
                        inString = false
                    }
                    continue
                }

                if character == "\"" {
                    inString = true
                    continue
                }
                if character == "[" || character == "{" {
                    stack.append(character)
                    continue
                }
                if character == "]" || character == "}" {
                    guard let opening = stack.popLast(),
                          (opening == "[" && character == "]")
                            || (opening == "{" && character == "}") else {
                        break
                    }
                    if stack.isEmpty {
                        candidates.append(String(characters[start...index]))
                        break
                    }
                }
            }
        }
        return candidates
    }
}

enum SubtitleTranslationError: LocalizedError, Sendable {
    case invalidResponse(rawOutputSample: String)
    case truncatedJSON(rawOutputSample: String)
    case reasoningOnly(rawOutputSample: String)
    case unexpectedThinkingMarker(rawOutputSample: String)
    case incompleteResponse(receivedCount: Int, expectedCount: Int)

    var errorDescription: String? {
        switch self {
        case let .invalidResponse(rawOutputSample):
            Self.description("字幕翻譯模型未回傳有效的 JSON 結果。", sample: rawOutputSample)
        case let .truncatedJSON(rawOutputSample):
            Self.description("字幕翻譯模型回傳的 JSON 似乎被截斷。", sample: rawOutputSample)
        case let .reasoningOnly(rawOutputSample):
            Self.description("字幕翻譯模型只回傳推理文字，沒有 JSON 結果。", sample: rawOutputSample)
        case let .unexpectedThinkingMarker(rawOutputSample):
            Self.description("字幕翻譯模型回傳了不完整的 thinking 標記。", sample: rawOutputSample)
        case let .incompleteResponse(receivedCount, expectedCount):
            "字幕翻譯結果不完整（收到 \(receivedCount) 筆，預期 \(expectedCount) 筆）。"
        }
    }

    private static func description(_ message: String, sample: String) -> String {
        guard !sample.isEmpty else { return message }
        let escapedSample = sample
            .replacingOccurrences(of: "\n", with: "\\n")
            .replacingOccurrences(of: "\r", with: "\\r")
        return "\(message)（模型輸出開頭：\(escapedSample)）"
    }
}

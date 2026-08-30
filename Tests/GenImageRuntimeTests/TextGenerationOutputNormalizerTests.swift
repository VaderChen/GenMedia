import Testing

@testable import GenImageRuntime

struct TextGenerationOutputNormalizerTests {
    @Test func removesThinkingAndMarkdownFenceBeforeDecodingTranslation() throws {
        let output = """
        <think>先確認每個片段的語氣。</think>
        ```json
        [{"index":0,"text":"Hello"},{"index":1,"text":"World"}]
        ```
        """

        let items = try SubtitleTranslationResponseParser.decode(output)

        #expect(items.map(\.index) == [0, 1])
        #expect(items.map(\.text) == ["Hello", "World"])
    }

    @Test func decodesWrappedAndEscapedJSON() throws {
        let output = #"說明文字 {"translations":"[{\"index\":0,\"text\":\"翻譯\"}]"} 完成。"#

        let items = try SubtitleTranslationResponseParser.decode(output)

        #expect(items.count == 1)
        #expect(items[0].index == 0)
        #expect(items[0].text == "翻譯")
    }

    @Test func supportsLlamaLoaderReasoningMarkers() {
        let output = "Thinking process: reason\nFinal answer: translated text"

        #expect(TextGenerationOutputNormalizer.finalAnswer(output) == "translated text")
    }

    @Test func removesUnclosedThinkingTail() {
        let output = "usable answer\n<think>unfinished reasoning"

        #expect(TextGenerationOutputNormalizer.finalAnswer(output) == "usable answer")
    }

    @Test func identifiesTruncatedJSON() {
        let output = "[{\"index\":0,\"text\":\"未完成"

        do {
            _ = try SubtitleTranslationResponseParser.decode(output)
            Issue.record("預期截斷 JSON 錯誤")
        } catch let error as SubtitleTranslationError {
            guard case let .truncatedJSON(sample) = error else {
                Issue.record("收到非預期的 parser 錯誤：\(error)")
                return
            }
            #expect(sample == output)
        } catch {
            Issue.record("收到非預期的錯誤型別：\(error)")
        }
    }

    @Test func identifiesReasoningWithoutJSON() {
        let output = "<think>我先分析字幕內容，但還沒完成翻譯。</think>"

        do {
            _ = try SubtitleTranslationResponseParser.decode(output)
            Issue.record("預期只有推理文字錯誤")
        } catch let error as SubtitleTranslationError {
            guard case .reasoningOnly = error else {
                Issue.record("收到非預期的 parser 錯誤：\(error)")
                return
            }
        } catch {
            Issue.record("收到非預期的錯誤型別：\(error)")
        }
    }

    @Test func identifiesClosingThinkingMarkerWithoutOpeningMarker() {
        let output = "</think>"

        do {
            _ = try SubtitleTranslationResponseParser.decode(output)
            Issue.record("預期不完整 thinking 標記錯誤")
        } catch let error as SubtitleTranslationError {
            guard case .unexpectedThinkingMarker = error else {
                Issue.record("收到非預期的 parser 錯誤：\(error)")
                return
            }
        } catch {
            Issue.record("收到非預期的錯誤型別：\(error)")
        }
    }

    @Test func invalidResponseIncludesTheFirstTwoHundredCharacters() {
        let repeatedText = String(repeating: "x", count: 240)
        let output = "{\"unexpected\":\"\(repeatedText)\"}"

        do {
            _ = try SubtitleTranslationResponseParser.decode(output)
            Issue.record("預期 invalidResponse 錯誤")
        } catch let error as SubtitleTranslationError {
            guard case let .invalidResponse(sample) = error else {
                Issue.record("收到非預期的 parser 錯誤：\(error)")
                return
            }
            #expect(sample == String(output.prefix(200)))
            #expect(error.errorDescription?.contains(sample) == true)
        } catch {
            Issue.record("收到非預期的錯誤型別：\(error)")
        }
    }

    @Test func closingThinkingMarkerWithoutOpeningStillAllowsFollowingJSON() throws {
        let output = """
        推理內容
        </think>
        [{"index":0,"text":"翻譯完成"}]
        """

        let items = try SubtitleTranslationResponseParser.decode(output)

        #expect(items.map(\.text) == ["翻譯完成"])
    }
}

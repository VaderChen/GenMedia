import Testing

@testable import GenImageCore

struct SubtitleDocumentTests {
    @Test func srtFormatsHoursAndZeroPaddedMilliseconds() {
        let document = SubtitleDocument.render(
            format: .srt,
            segments: [
                TimedTranscriptSegment(
                    start: 3_661.007,
                    end: 3_662.09,
                    text: "跨小時"
                )
            ]
        )

        #expect(document == "1\n01:01:01,007 --> 01:01:02,090\n跨小時\n")
    }

    @Test func srtNumbersSegmentsInOrder() {
        let document = SubtitleDocument.render(
            format: .srt,
            segments: [
                TimedTranscriptSegment(start: 0, end: 1, text: "第一段"),
                TimedTranscriptSegment(start: 1, end: 2, text: "第二段"),
                TimedTranscriptSegment(start: 2, end: 3, text: "第三段")
            ]
        )

        #expect(document.contains("1\n00:00:00,000 --> 00:00:01,000\n第一段"))
        #expect(document.contains("2\n00:00:01,000 --> 00:00:02,000\n第二段"))
        #expect(document.contains("3\n00:00:02,000 --> 00:00:03,000\n第三段"))
    }

    @Test func srtKeepsAnEmptyCueAfterTrimmingWhitespace() {
        let document = SubtitleDocument.render(
            format: .srt,
            segments: [
                TimedTranscriptSegment(start: 0, end: 1, text: " \n\t ")
            ]
        )

        #expect(document == "1\n00:00:00,000 --> 00:00:01,000\n\n")
        #expect(SubtitleDocument.render(format: .srt, segments: []).isEmpty)
    }

    @Test func srtClampsNegativeTimeAndCarriesRoundedMilliseconds() {
        let document = SubtitleDocument.render(
            format: .srt,
            segments: [
                TimedTranscriptSegment(start: -2.5, end: 0, text: "邊界"),
                TimedTranscriptSegment(start: 0.9996, end: 1.0004, text: "進位")
            ]
        )

        #expect(document.contains("00:00:00,000 --> 00:00:00,000"))
        #expect(document.contains("00:00:01,000 --> 00:00:01,000"))
    }
}

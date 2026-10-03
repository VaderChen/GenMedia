import Foundation
import Testing
@testable import GenImageRuntime

struct DescriptionTextStatisticsTests {
    @Test func everyChunkBoundaryMatchesTheOriginalUnicodeRules() {
        let samples = [
            "", " \n\t\u{00a0}", "一張在陽光下拍攝的照片，前景是一棵綠色的樹。",
            "abcdefghijklmnop", String(repeating: "a ", count: 10) + "bcdefg",
            "abcdefg123456789" + String(repeating: "!?", count: 20),
            String(repeating: "!?", count: 20), String(repeating: "e\u{301}🌲中9 ", count: 12),
            "一二三四五六七八九十甲乙丙丁戊己庚辛壬癸" + String(repeating: "!?", count: 10)
        ]
        for sample in samples {
            let scalars = Array(sample.unicodeScalars)
            for chunkSize in [1, 2, 3, 7, 16, 40] {
                var statistics = DescriptionTextStatistics()
                var accumulated = ""
                for start in stride(from: 0, to: scalars.count, by: chunkSize) {
                    let chunk = String(String.UnicodeScalarView(scalars[start..<min(start + chunkSize, scalars.count)]))
                    accumulated += chunk
                    statistics.append(chunk)
                    let expected = reference(accumulated)
                    #expect(statistics.isClearlyDegenerate == expected.degenerate)
                    #expect(statistics.isUsable == expected.usable)
                }
                statistics.append("")
                let expected = reference(sample)
                #expect(statistics.isClearlyDegenerate == expected.degenerate)
                #expect(statistics.isUsable == expected.usable)
            }
        }
    }

    // Independent, full-text oracle retains the rules used before incremental scanning.
    private func reference(_ text: String) -> (degenerate: Bool, usable: Bool) {
        let scalars = Array(text.unicodeScalars.filter { !CharacterSet.whitespacesAndNewlines.contains($0) })
        guard scalars.count >= 16 else { return (false, false) }
        var longest = 1, run = 1
        for index in 1..<scalars.count {
            run = scalars[index] == scalars[index - 1] ? run + 1 : 1
            longest = max(longest, run)
        }
        let punctuation = CharacterSet.punctuationCharacters.union(.symbols)
        let degenerate = longest >= 10 || (scalars.count >= 40
            && Double(scalars.filter { punctuation.contains($0) }.count) / Double(scalars.count) > 0.5)
        let meaningful = scalars.filter { CharacterSet.letters.contains($0) || CharacterSet.decimalDigits.contains($0) }.count
        return (degenerate, !degenerate && meaningful >= 8 && Double(meaningful) / Double(scalars.count) >= 0.3)
    }
}

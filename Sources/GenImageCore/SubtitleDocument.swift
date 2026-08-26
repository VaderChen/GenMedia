import Foundation

public enum SubtitleDocument {
    public static func render(
        format: SubtitleFormat,
        segments: [TimedTranscriptSegment]
    ) -> String {
        switch format {
        case .srt:
            segments.enumerated().map { index, segment in
                "\(index + 1)\n"
                    + "\(timestamp(segment.start, separator: ",")) --> "
                    + "\(timestamp(segment.end, separator: ","))\n"
                    + "\(segment.text.trimmingCharacters(in: .whitespacesAndNewlines))\n"
            }.joined(separator: "\n")
        case .vtt:
            "WEBVTT\n\n" + segments.map { segment in
                "\(timestamp(segment.start, separator: ".")) --> "
                    + "\(timestamp(segment.end, separator: "."))\n"
                    + "\(segment.text.trimmingCharacters(in: .whitespacesAndNewlines))\n"
            }.joined(separator: "\n")
        }
    }

    private static func timestamp(_ value: Double, separator: String) -> String {
        let milliseconds = max(0, Int((value * 1_000).rounded()))
        let hours = milliseconds / 3_600_000
        let minutes = (milliseconds / 60_000) % 60
        let seconds = (milliseconds / 1_000) % 60
        let remainder = milliseconds % 1_000
        return String(
            format: "%02d:%02d:%02d%@%03d",
            hours,
            minutes,
            seconds,
            separator,
            remainder
        )
    }
}

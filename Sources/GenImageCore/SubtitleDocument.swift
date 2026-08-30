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

    public static func previewText(
        format: SubtitleFormat,
        content: String
    ) -> String? {
        let normalized = content
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
        let lines = normalized.components(separatedBy: "\n")
        var previews: [String] = []
        var index = 0

        while index < lines.count {
            let line = lines[index].trimmingCharacters(in: .whitespacesAndNewlines)
            if line.isEmpty || (format == .vtt && line.uppercased() == "WEBVTT") {
                index += 1
                continue
            }

            let timingIndex: Int
            if line.contains("-->") {
                timingIndex = index
            } else if index + 1 < lines.count,
                      lines[index + 1].contains("-->") {
                timingIndex = index + 1
            } else {
                index += 1
                continue
            }

            guard let start = startTime(from: lines[timingIndex]) else {
                index = timingIndex + 1
                continue
            }
            index = timingIndex + 1
            var textLines: [String] = []
            while index < lines.count {
                let textLine = lines[index].trimmingCharacters(in: .whitespacesAndNewlines)
                if textLine.isEmpty { break }
                textLines.append(textLine)
                index += 1
            }
            let text = textLines.joined(separator: " ")
            previews.append("[\(previewTimestamp(start))] \(text)")
            index += 1
        }

        return previews.isEmpty ? nil : previews.joined(separator: "\n")
    }

    private static func startTime(from timingLine: String) -> Double? {
        guard let arrowRange = timingLine.range(of: "-->") else { return nil }
        let startText = timingLine[..<arrowRange.lowerBound]
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let components = startText.split(separator: ":")
        guard components.count == 2 || components.count == 3 else { return nil }

        let hours: Double
        let minutes: Double
        let secondsText: Substring
        if components.count == 3 {
            guard let parsedHours = Double(components[0]),
                  let parsedMinutes = Double(components[1]) else { return nil }
            hours = parsedHours
            minutes = parsedMinutes
            secondsText = components[2]
        } else {
            guard let parsedMinutes = Double(components[0]) else { return nil }
            hours = 0
            minutes = parsedMinutes
            secondsText = components[1]
        }
        guard let seconds = Double(
            secondsText.replacingOccurrences(of: ",", with: ".")
        ), hours >= 0, minutes >= 0, seconds >= 0 else {
            return nil
        }
        return hours * 3_600 + minutes * 60 + seconds
    }

    private static func previewTimestamp(_ value: Double) -> String {
        let milliseconds = max(0, Int((value * 1_000).rounded()))
        let hours = milliseconds / 3_600_000
        let minutes = (milliseconds / 60_000) % 60
        let seconds = (milliseconds / 1_000) % 60
        let remainder = milliseconds % 1_000
        return String(
            format: "%02d:%02d:%02d.%03d",
            hours,
            minutes,
            seconds,
            remainder
        )
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

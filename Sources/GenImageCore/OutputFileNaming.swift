import Foundation

/// Generates stable, human-readable names for files produced by inference.
///
/// The timestamp is minute-based for sorting. A fresh UUID distinguishes
/// unwritten batch outputs and concurrent App / MCP processes without reserving
/// empty files that a runtime might mistake for completed outputs.
public enum OutputFileNaming {
    public static func imageURL(
        in directory: URL,
        pathExtension: String = "png",
        date: Date = .now,
        fileManager: FileManager = .default
    ) -> URL {
        uniqueURL(
            prefix: "Image",
            in: directory,
            pathExtension: pathExtension,
            date: date,
            fileManager: fileManager
        )
    }

    public static func videoURL(
        in directory: URL,
        pathExtension: String = "mp4",
        date: Date = .now,
        fileManager: FileManager = .default
    ) -> URL {
        uniqueURL(
            prefix: "Video",
            in: directory,
            pathExtension: pathExtension,
            date: date,
            fileManager: fileManager
        )
    }

    public static func musicURL(
        in directory: URL,
        pathExtension: String,
        date: Date = .now,
        fileManager: FileManager = .default
    ) -> URL {
        uniqueURL(
            prefix: "Music",
            in: directory,
            pathExtension: pathExtension,
            date: date,
            fileManager: fileManager
        )
    }

    public static func subtitleURL(
        in directory: URL,
        pathExtension: String,
        date: Date = .now,
        fileManager: FileManager = .default
    ) -> URL {
        uniqueURL(
            prefix: "Subtitle",
            in: directory,
            pathExtension: pathExtension,
            date: date,
            fileManager: fileManager
        )
    }

    private static func uniqueURL(
        prefix: String,
        in directory: URL,
        pathExtension: String,
        date: Date,
        fileManager: FileManager
    ) -> URL {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.dateFormat = "yyyyMMdd-HHmm"

        let sanitizedExtension = pathExtension.trimmingCharacters(in: CharacterSet(charactersIn: ". "))
        let baseName = "\(prefix)-\(formatter.string(from: date))"
        var candidate: URL
        repeat {
            candidate = directory
                .appendingPathComponent("\(baseName)-\(UUID().uuidString.lowercased())")
                .appendingPathExtension(sanitizedExtension)
        } while fileManager.fileExists(atPath: candidate.path)
        return candidate
    }
}

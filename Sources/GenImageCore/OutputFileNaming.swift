import Foundation

/// Generates stable, human-readable names for files produced by inference.
///
/// The timestamp is intentionally minute-based for easy sorting and sharing. If
/// more than one output is created during the same minute, a numeric suffix is
/// appended to avoid overwriting the previous file.
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
        var candidate = directory.appendingPathComponent(baseName).appendingPathExtension(sanitizedExtension)
        var suffix = 1
        while fileManager.fileExists(atPath: candidate.path) {
            candidate = directory
                .appendingPathComponent("\(baseName)-\(suffix)")
                .appendingPathExtension(sanitizedExtension)
            suffix += 1
        }
        return candidate
    }
}

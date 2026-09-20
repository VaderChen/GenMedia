import Foundation

/// Updating a path does not wait for a busy inference actor. Each inference
/// takes one snapshot before starting so an in-flight job keeps its destination.
final class OutputDirectoryStorage: @unchecked Sendable {
    private let lock = NSLock()
    private var value: URL

    init(_ url: URL) { value = url }

    var url: URL {
        lock.lock()
        defer { lock.unlock() }
        return value
    }

    func update(to url: URL) {
        lock.lock()
        value = url
        lock.unlock()
    }
}

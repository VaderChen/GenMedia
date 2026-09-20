import Foundation

/// Replacing a task cancels it, but waits for all of its cleanup before starting
/// the replacement. Different keys may run concurrently.
@MainActor
public final class SerialTaskQueue<Key: Hashable & Sendable> {
    private struct Entry: Sendable {
        let id: UUID
        let task: Task<Void, Never>
    }
    private var entries: [Key: Entry] = [:]

    public init() {}

    @discardableResult
    public func replace(
        for key: Key,
        operation: @escaping @MainActor () async -> Void,
        onFinish: @escaping @MainActor () -> Void = {}
    ) -> Task<Void, Never> {
        let previous = entries[key]?.task
        previous?.cancel()
        let id = UUID()
        let task = Task { [weak self] in
            if let previous { await previous.value }
            if !Task.isCancelled { await operation() }
            if self?.entries[key]?.id == id { self?.entries[key] = nil }
            onFinish()
        }
        entries[key] = Entry(id: id, task: task)
        return task
    }

    public func cancel(_ key: Key) { entries[key]?.task.cancel() }
    public func cancelAll(except preservedKeys: Set<Key> = []) {
        for (key, entry) in entries where !preservedKeys.contains(key) { entry.task.cancel() }
    }

    public func waitForAll() async {
        let tasks = entries.values.map(\.task)
        for task in tasks { await task.value }
    }

    deinit { entries.values.forEach { $0.task.cancel() } }
}

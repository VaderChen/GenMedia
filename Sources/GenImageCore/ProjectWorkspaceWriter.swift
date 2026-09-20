import Foundation

/// Serializes and coalesces workspace writes without doing filesystem work on
/// the UI executor. All mutable state is confined to `queue`.
public final class ProjectWorkspaceWriter: @unchecked Sendable {
    private let queue = DispatchQueue(label: "genimage.workspace-writer", qos: .utility)
    private let url: URL
    private let delay: TimeInterval
    private var pending: ProjectWorkspaceSnapshot?
    private var work: DispatchWorkItem?

    public init(url: URL, delay: TimeInterval = 0.3) {
        self.url = url
        self.delay = delay
    }

    public func schedule(
        _ snapshot: ProjectWorkspaceSnapshot,
        onError: @escaping @Sendable (String) -> Void = { _ in }
    ) {
        queue.async { [self] in
            pending = snapshot
            work?.cancel()
            let next = DispatchWorkItem { [weak self] in
                guard let self else { return }
                do { try self.savePending() }
                catch { onError(error.localizedDescription) }
            }
            work = next
            queue.asyncAfter(deadline: .now() + delay, execute: next)
        }
    }

    /// Wait for previously queued changes and save the newest snapshot before
    /// application termination. This is intentionally synchronous at shutdown.
    public func flush() throws {
        try queue.sync {
            work?.cancel()
            work = nil
            try savePending()
        }
    }

    private func savePending() throws {
        guard let pending else { return }
        try ProjectWorkspacePersistence.save(pending, to: url)
        self.pending = nil
    }
}

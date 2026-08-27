import Foundation
import GenImageCore
import UniformTypeIdentifiers
@preconcurrency import WebKit

final class AssetSchemeHandler: NSObject, WKURLSchemeHandler, @unchecked Sendable {
    private struct AssetReference: Sendable {
        let fileURL: URL
        let isTimedMedia: Bool
    }

    private final class State: @unchecked Sendable {
        let lock = NSLock()
        var assetReferences: [String: AssetReference] = [:]
        var stoppedTaskIDs = Set<ObjectIdentifier>()
    }

    private final class URLSchemeTaskReference: @unchecked Sendable {
        let task: any WKURLSchemeTask

        init(_ task: any WKURLSchemeTask) {
            self.task = task
        }
    }

    private let state = State()

    func updateAssets(_ assets: [GenImageCore.MediaAsset]) {
        state.lock.lock()
        state.assetReferences = Dictionary(
            uniqueKeysWithValues: assets.compactMap { asset in
                (asset.playbackURL ?? asset.fileURL).map {
                    (
                        asset.id.uuidString.lowercased(),
                        AssetReference(fileURL: $0, isTimedMedia: asset.kind.isTimedMedia)
                    )
                }
            }
        )
        state.lock.unlock()
    }

    func webView(_ webView: WKWebView, start urlSchemeTask: any WKURLSchemeTask) {
        guard let requestURL = urlSchemeTask.request.url,
              let host = requestURL.host?.lowercased() else {
            urlSchemeTask.didFailWithError(URLError(.badURL))
            return
        }

        state.lock.lock()
        let reference = state.assetReferences[host]
        let taskID = ObjectIdentifier(urlSchemeTask as AnyObject)
        state.stoppedTaskIDs.remove(taskID)
        state.lock.unlock()

        guard let reference else {
            urlSchemeTask.didFailWithError(URLError(.fileDoesNotExist))
            return
        }

        let taskReference = URLSchemeTaskReference(urlSchemeTask)
        DispatchQueue.global(qos: .userInitiated).async { [weak self, taskReference] in
            self?.serve(
                reference: reference,
                requestURL: requestURL,
                task: taskReference.task,
                taskID: taskID
            )
        }
    }

    func webView(_ webView: WKWebView, stop urlSchemeTask: any WKURLSchemeTask) {
        let taskID = ObjectIdentifier(urlSchemeTask as AnyObject)
        state.lock.lock()
        state.stoppedTaskIDs.insert(taskID)
        state.lock.unlock()
    }

    nonisolated private func serve(
        reference: AssetReference,
        requestURL: URL,
        task: any WKURLSchemeTask,
        taskID: ObjectIdentifier
    ) {
        defer {
            state.lock.lock()
            state.stoppedTaskIDs.remove(taskID)
            state.lock.unlock()
        }

        do {
            let didAccess = reference.fileURL.startAccessingSecurityScopedResource()
            defer {
                if didAccess { reference.fileURL.stopAccessingSecurityScopedResource() }
            }

            guard !isStopped(taskID) else { return }
            let fileSize = try fileSize(of: reference.fileURL)
            if reference.isTimedMedia {
                try serveMedia(
                    reference: reference,
                    requestURL: requestURL,
                    fileSize: fileSize,
                    task: task,
                    taskID: taskID
                )
            } else {
                let data = try Data(contentsOf: reference.fileURL)
                guard !isStopped(taskID) else { return }
                let contentType = contentType(for: reference.fileURL)
                let response = URLResponse(
                    url: requestURL,
                    mimeType: contentType,
                    expectedContentLength: data.count,
                    textEncodingName: nil
                )
                task.didReceive(response)
                guard !isStopped(taskID) else { return }
                task.didReceive(data)
                guard !isStopped(taskID) else { return }
                task.didFinish()
            }
        } catch {
            guard !isStopped(taskID) else { return }
            task.didFailWithError(error)
        }
    }

    nonisolated private func serveMedia(
        reference: AssetReference,
        requestURL: URL,
        fileSize: Int64,
        task: any WKURLSchemeTask,
        taskID: ObjectIdentifier
    ) throws {
        let byteRange = try requestedRange(
            from: task.request.value(forHTTPHeaderField: "Range"),
            fileSize: fileSize
        )
        let range = byteRange ?? 0..<fileSize
        let contentLength = range.upperBound - range.lowerBound
        let contentType = contentType(for: reference.fileURL)
        var headers = [
            "Accept-Ranges": "bytes",
            "Content-Length": String(contentLength),
            "Content-Type": contentType
        ]
        let statusCode: Int
        if byteRange == nil {
            statusCode = 200
        } else {
            statusCode = 206
            headers["Content-Range"] = "bytes \(range.lowerBound)-\(range.upperBound - 1)/\(fileSize)"
        }
        let response = HTTPURLResponse(
            url: requestURL,
            statusCode: statusCode,
            httpVersion: "HTTP/1.1",
            headerFields: headers
        )!
        guard !isStopped(taskID) else { return }
        task.didReceive(response)

        guard contentLength > 0 else {
            guard !isStopped(taskID) else { return }
            task.didFinish()
            return
        }

        let handle = try FileHandle(forReadingFrom: reference.fileURL)
        defer { try? handle.close() }
        try handle.seek(toOffset: UInt64(range.lowerBound))
        var remaining = contentLength
        while remaining > 0 {
            guard !isStopped(taskID) else { return }
            let chunkSize = Int(min(Int64(512 * 1024), remaining))
            guard let data = try handle.read(upToCount: chunkSize), !data.isEmpty else {
                throw URLError(.cannotLoadFromNetwork)
            }
            task.didReceive(data)
            remaining -= Int64(data.count)
        }
        guard !isStopped(taskID) else { return }
        task.didFinish()
    }

    nonisolated private func requestedRange(
        from header: String?,
        fileSize: Int64
    ) throws -> Range<Int64>? {
        guard let header, !header.isEmpty else { return nil }
        guard fileSize > 0,
              header.lowercased().hasPrefix("bytes=") else {
            throw URLError(.cannotParseResponse)
        }
        let value = header.dropFirst("bytes=".count)
            .split(separator: ",", maxSplits: 1, omittingEmptySubsequences: true)[0]
        let components = value.split(separator: "-", maxSplits: 1, omittingEmptySubsequences: false)
        guard components.count == 2,
              let start = Int64(components[0].trimmingCharacters(in: .whitespaces)),
              start >= 0,
              start < fileSize else {
            throw URLError(.cannotParseResponse)
        }
        let end: Int64
        if components[1].trimmingCharacters(in: .whitespaces).isEmpty {
            end = fileSize - 1
        } else if let requestedEnd = Int64(components[1].trimmingCharacters(in: .whitespaces)) {
            end = min(fileSize - 1, requestedEnd)
        } else {
            throw URLError(.cannotParseResponse)
        }
        guard end >= start else { throw URLError(.cannotParseResponse) }
        return start..<end + 1
    }

    nonisolated private func fileSize(of url: URL) throws -> Int64 {
        guard let size = try FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber else {
            throw CocoaError(.fileNoSuchFile)
        }
        return size.int64Value
    }

    nonisolated private func contentType(for url: URL) -> String {
        UTType(filenameExtension: url.pathExtension)?.preferredMIMEType
            ?? "application/octet-stream"
    }

    nonisolated private func isStopped(_ taskID: ObjectIdentifier) -> Bool {
        state.lock.lock()
        defer { state.lock.unlock() }
        return state.stoppedTaskIDs.contains(taskID)
    }
}

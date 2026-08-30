import Foundation
import GenImageCore
import UniformTypeIdentifiers
@preconcurrency import WebKit

final class AssetSchemeHandler: NSObject, WKURLSchemeHandler, @unchecked Sendable {
    private struct AssetReference: Sendable {
        let fileURL: URL
        let isTimedMedia: Bool
        let subtitleSidecar: SubtitleSidecar?
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
                        AssetReference(
                            fileURL: $0,
                            isTimedMedia: asset.kind.isTimedMedia,
                            subtitleSidecar: Self.subtitleSidecar(for: asset, among: assets)
                        )
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
        guard requestURL.path.isEmpty || requestURL.path == "/" || requestURL.path == "/subtitle" else {
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
                task: taskReference,
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
        task: URLSchemeTaskReference,
        taskID: ObjectIdentifier
    ) {
        defer {
            state.lock.lock()
            state.stoppedTaskIDs.remove(taskID)
            state.lock.unlock()
        }

        do {
            let resourceURL: URL
            if requestURL.path == "/subtitle" {
                guard let subtitleSidecar = reference.subtitleSidecar else {
                    throw URLError(.fileDoesNotExist)
                }
                resourceURL = subtitleSidecar.fileURL
            } else {
                resourceURL = reference.fileURL
            }
            let didAccess = resourceURL.startAccessingSecurityScopedResource()
            defer {
                if didAccess { resourceURL.stopAccessingSecurityScopedResource() }
            }

            guard !isStopped(taskID) else { return }
            if requestURL.path == "/subtitle" {
                try serveSubtitle(
                    sidecar: reference.subtitleSidecar!,
                    requestURL: requestURL,
                    task: task,
                    taskID: taskID
                )
            } else {
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
                    guard deliverIfActive(taskID, task: task, {
                        $0.didReceive(response)
                    }) else { return }
                    guard deliverIfActive(taskID, task: task, {
                        $0.didReceive(data)
                    }) else { return }
                    _ = deliverIfActive(taskID, task: task, {
                        $0.didFinish()
                    })
                }
            }
        } catch {
            _ = deliverIfActive(taskID, task: task, {
                $0.didFailWithError(error)
            })
        }
    }

    nonisolated private func serveSubtitle(
        sidecar: SubtitleSidecar,
        requestURL: URL,
        task: URLSchemeTaskReference,
        taskID: ObjectIdentifier
    ) throws {
        let sourceData = try Data(contentsOf: sidecar.fileURL)
        guard let data = SubtitleSidecarResolver.webVTTData(
            from: sourceData,
            format: sidecar.format
        ) else {
            throw URLError(.cannotDecodeContentData)
        }
        let response = HTTPURLResponse(
            url: requestURL,
            statusCode: 200,
            httpVersion: "HTTP/1.1",
            headerFields: [
                "Content-Length": String(data.count),
                "Content-Type": "text/vtt; charset=utf-8"
            ]
        )!
        guard deliverIfActive(taskID, task: task, {
            $0.didReceive(response)
        }) else { return }
        guard deliverIfActive(taskID, task: task, {
            $0.didReceive(data)
        }) else { return }
        _ = deliverIfActive(taskID, task: task, {
            $0.didFinish()
        })
    }

    nonisolated private static func subtitleSidecar(
        for asset: GenImageCore.MediaAsset,
        among assets: [GenImageCore.MediaAsset]
    ) -> SubtitleSidecar? {
        guard asset.kind == .importedVideo
            || asset.kind == .generatedVideo
            || asset.kind == .generatedSubtitle else {
            return nil
        }
        let mediaURLs = [asset.fileURL, asset.playbackURL].compactMap { $0 }
        let didAccess = mediaURLs.first?.startAccessingSecurityScopedResource() == true
        defer {
            if didAccess, let mediaURL = mediaURLs.first {
                mediaURL.stopAccessingSecurityScopedResource()
            }
        }
        if asset.kind == .generatedSubtitle {
            return SubtitleSidecarResolver.subtitle(for: asset)
        }
        return SubtitleSidecarResolver.locate(for: asset, among: assets)
    }

    nonisolated private func serveMedia(
        reference: AssetReference,
        requestURL: URL,
        fileSize: Int64,
        task: URLSchemeTaskReference,
        taskID: ObjectIdentifier
    ) throws {
        let byteRange = try requestedRange(
            from: task.task.request.value(forHTTPHeaderField: "Range"),
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
        guard deliverIfActive(taskID, task: task, {
            $0.didReceive(response)
        }) else { return }

        guard contentLength > 0 else {
            _ = deliverIfActive(taskID, task: task, {
                $0.didFinish()
            })
            return
        }

        let handle = try FileHandle(forReadingFrom: reference.fileURL)
        defer { try? handle.close() }
        try handle.seek(toOffset: UInt64(range.lowerBound))
        var remaining = contentLength
        while remaining > 0 {
            let chunkSize = Int(min(Int64(512 * 1024), remaining))
            guard let data = try handle.read(upToCount: chunkSize), !data.isEmpty else {
                throw URLError(.cannotLoadFromNetwork)
            }
            guard deliverIfActive(taskID, task: task, {
                $0.didReceive(data)
            }) else { return }
            remaining -= Int64(data.count)
        }
        _ = deliverIfActive(taskID, task: task, {
            $0.didFinish()
        })
    }

    @discardableResult
    nonisolated private func deliverIfActive(
        _ taskID: ObjectIdentifier,
        task: URLSchemeTaskReference,
        _ delivery: @escaping @Sendable (any WKURLSchemeTask) -> Void
    ) -> Bool {
        guard !isStopped(taskID) else { return false }
        DispatchQueue.main.async { [weak self, task, delivery] in
            guard let self, !self.isStopped(taskID) else { return }
            delivery(task.task)
        }
        return true
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

import Foundation
import GenImageCore
import UniformTypeIdentifiers
import WebKit

final class AssetSchemeHandler: NSObject, WKURLSchemeHandler, @unchecked Sendable {
    private let lock = NSLock()
    private var assetURLs: [String: URL] = [:]

    func updateAssets(_ assets: [GenImageCore.MediaAsset]) {
        lock.lock()
        assetURLs = Dictionary(
            uniqueKeysWithValues: assets.compactMap { asset in
                asset.fileURL.map { (asset.id.uuidString.lowercased(), $0) }
            }
        )
        lock.unlock()
    }

    func webView(_ webView: WKWebView, start urlSchemeTask: any WKURLSchemeTask) {
        guard let requestURL = urlSchemeTask.request.url,
              let host = requestURL.host?.lowercased() else {
            urlSchemeTask.didFailWithError(URLError(.badURL))
            return
        }

        lock.lock()
        let fileURL = assetURLs[host]
        lock.unlock()

        guard let fileURL else {
            urlSchemeTask.didFailWithError(URLError(.fileDoesNotExist))
            return
        }

        do {
            let didAccess = fileURL.startAccessingSecurityScopedResource()
            defer {
                if didAccess { fileURL.stopAccessingSecurityScopedResource() }
            }

            let data = try Data(contentsOf: fileURL)
            let contentType = UTType(filenameExtension: fileURL.pathExtension)?.preferredMIMEType ?? "application/octet-stream"
            let response = URLResponse(
                url: requestURL,
                mimeType: contentType,
                expectedContentLength: data.count,
                textEncodingName: nil
            )
            urlSchemeTask.didReceive(response)
            urlSchemeTask.didReceive(data)
            urlSchemeTask.didFinish()
        } catch {
            urlSchemeTask.didFailWithError(error)
        }
    }

    func webView(_ webView: WKWebView, stop urlSchemeTask: any WKURLSchemeTask) {}
}

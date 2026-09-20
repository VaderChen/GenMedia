import AppKit
import Foundation
import GenImageCore
import Testing
@preconcurrency import WebKit
@testable import GenImageApp

@Suite(.serialized, .timeLimit(.minutes(1)))
@MainActor
struct AssetSchemeHandlerTests {
    @Test func stoppingLargeMediaAfterResponsePreventsEverySubsequentCallback() async throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("asset-\(UUID()).mp4")
        FileManager.default.createFile(atPath: file.path, contents: nil)
        defer { try? FileManager.default.removeItem(at: file) }
        let handle = try FileHandle(forWritingTo: file)
        try handle.truncate(atOffset: 512 * 1_024 * 1_024)
        try handle.close()
        _ = NSApplication.shared
        let webView = WKWebView()
        let handler = AssetSchemeHandler()
        let asset = MediaAsset(projectID: UUID(), kind: .generatedVideo, title: "Test", fileURL: file,
            pixelWidth: 16, pixelHeight: 16)
        handler.updateAssets([asset])
        let task = RecordingSchemeTask(url: URL(string: "genimage-asset://\(asset.id.uuidString.lowercased())")!)
        task.onResponse = {
            // A slow main thread used to let the reader enqueue the entire movie.
            Thread.sleep(forTimeInterval: 0.08)
            task.stopped = true
            handler.webView(webView, stop: task)
        }
        defer { task.onResponse = nil }
        handler.webView(webView, start: task)
        for _ in 0..<200 where !task.stopped { try await Task.sleep(for: .milliseconds(10)) }
        #expect(task.stopped)
        try await Task.sleep(for: .milliseconds(150))
        #expect(task.callbacksAfterStop == 0)
        #expect(task.body.isEmpty)
        #expect(!task.finished)
    }

    @Test(arguments: ["bytes=3-7", "bytes="])
    func byteRangeAndMalformedRangeCompleteWithoutCrashing(range: String) async throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("asset-\(UUID()).mp4")
        try Data("0123456789".utf8).write(to: file)
        defer { try? FileManager.default.removeItem(at: file) }
        _ = NSApplication.shared
        let webView = WKWebView()
        let handler = AssetSchemeHandler()
        let asset = MediaAsset(projectID: UUID(), kind: .generatedVideo, title: "Test", fileURL: file,
            pixelWidth: 16, pixelHeight: 16)
        handler.updateAssets([asset])
        let task = RecordingSchemeTask(url: URL(string: "genimage-asset://\(asset.id.uuidString.lowercased())")!, range: range)
        handler.webView(webView, start: task)
        for _ in 0..<200 where !task.finished && task.error == nil {
            try await Task.sleep(for: .milliseconds(10))
        }
        if range == "bytes=" {
            #expect(task.error != nil)
            #expect(task.body.isEmpty)
        } else {
            #expect(task.error == nil)
            #expect(task.finished)
            #expect((task.response as? HTTPURLResponse)?.statusCode == 206)
            #expect(task.body == Data("34567".utf8))
        }
    }
}

// Handler callbacks and the test read these properties on the main thread.
private final class RecordingSchemeTask: NSObject, WKURLSchemeTask, @unchecked Sendable {
    let request: URLRequest
    var stopped = false
    var callbacksAfterStop = 0
    var onResponse: (() -> Void)?
    var response: URLResponse?
    var body = Data()
    var finished = false
    var error: (any Error)?

    init(url: URL, range: String? = nil) {
        var request = URLRequest(url: url)
        request.setValue(range, forHTTPHeaderField: "Range")
        self.request = request
    }

    func didReceive(_ response: URLResponse) {
        if stopped { callbacksAfterStop += 1 }
        self.response = response
        onResponse?()
    }
    func didReceive(_ data: Data) {
        if stopped { callbacksAfterStop += 1 }
        body.append(data)
    }
    func didFinish() {
        if stopped { callbacksAfterStop += 1 }
        finished = true
    }
    func didFailWithError(_ error: any Error) {
        if stopped { callbacksAfterStop += 1 }
        self.error = error
    }
}

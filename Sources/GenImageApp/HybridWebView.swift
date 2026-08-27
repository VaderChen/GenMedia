import AppKit
import SwiftUI
import WebKit

@MainActor
private final class ClipboardAwareWebView: WKWebView {
    var pasteboardImageHandler: (() -> Bool)?

    override var acceptsFirstResponder: Bool { true }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        guard let window else { return }
        NSApp.activate(ignoringOtherApps: true)
        window.makeKeyAndOrderFront(nil)
        DispatchQueue.main.async { [weak self, weak window] in
            guard let self, let window, window.isKeyWindow else { return }
            window.makeFirstResponder(self)
        }
    }

    override func mouseDown(with event: NSEvent) {
        NSApp.activate(ignoringOtherApps: true)
        window?.makeKey()
        super.mouseDown(with: event)
    }

    override func performKeyEquivalent(with event: NSEvent) -> Bool {
        let modifiers = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
        let isPasteShortcut = event.type == .keyDown
            && event.charactersIgnoringModifiers?.lowercased() == "v"
            && (modifiers.contains(.command) || modifiers.contains(.control))

        if isPasteShortcut, pasteboardImageHandler?() == true {
            return true
        }
        return super.performKeyEquivalent(with: event)
    }
}

struct HybridWebView: NSViewRepresentable {
    let controller: HybridBridgeController

    func makeNSView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.preferences.setValue(true, forKey: "developerExtrasEnabled")
        configuration.userContentController.add(controller, name: "genimage")
        configuration.setURLSchemeHandler(controller.assetSchemeHandler, forURLScheme: "genimage-asset")
        configuration.setURLSchemeHandler(
            controller.webUISchemeHandler,
            forURLScheme: WebUISchemeHandler.scheme
        )

        let webView = ClipboardAwareWebView(frame: .zero, configuration: configuration)
        webView.pasteboardImageHandler = { [weak controller] in
            controller?.pasteImageFromSystemClipboard() ?? false
        }
        webView.navigationDelegate = controller
        webView.uiDelegate = controller
        webView.setValue(false, forKey: "drawsBackground")
        controller.attach(webView: webView)

        if controller.webUISchemeHandler.canServeResources,
           let indexURL = URL(string: "\(WebUISchemeHandler.scheme)://\(WebUISchemeHandler.host)/index.html") {
            webView.load(URLRequest(url: indexURL))
        } else {
            webView.loadHTMLString(
                "<h1>GenImage WebUI resource missing</h1>",
                baseURL: nil
            )
        }

        return webView
    }

    func updateNSView(_ webView: WKWebView, context: Context) {}

    static func dismantleNSView(_ webView: WKWebView, coordinator: Void) {
        (webView as? ClipboardAwareWebView)?.pasteboardImageHandler = nil
        webView.configuration.userContentController.removeScriptMessageHandler(forName: "genimage")
        webView.navigationDelegate = nil
        webView.uiDelegate = nil
    }
}

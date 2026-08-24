import AppKit
import SwiftUI

@MainActor
private final class GenImageAppDelegate: NSObject, NSApplicationDelegate {
    func applicationWillFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.activate(ignoringOtherApps: true)
        DispatchQueue.main.async {
            NSApp.windows.first?.makeKeyAndOrderFront(nil)
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}

@main
@MainActor
struct GenImageApp: App {
    @NSApplicationDelegateAdaptor(GenImageAppDelegate.self) private var appDelegate
    @StateObject private var bridge = HybridBridgeController()

    var body: some Scene {
        WindowGroup {
            HybridWebView(controller: bridge)
                .frame(minWidth: 1_080, minHeight: 700)
        }
        .defaultSize(width: 1_440, height: 900)
    }
}

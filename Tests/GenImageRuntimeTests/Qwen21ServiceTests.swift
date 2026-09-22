import CoreGraphics
import Foundation
import GenImageCore
import ImageIO
import Testing
import UniformTypeIdentifiers
@testable import GenImageRuntime

struct Qwen21ServiceTests {
    @Test(arguments: [false, true])
    func workerProtocolPreservesLinksAndCleansFailedBatches(fail: Bool) async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let output = root.appendingPathComponent("output")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        for file in QwenImage21Model.requiredFiles {
            let url = root.appendingPathComponent(file)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data([1]).write(to: url)
        }
        let context = try #require(CGContext(data: nil, width: 256, height: 256, bitsPerComponent: 8,
            bytesPerRow: 256 * 4, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        let image = try #require(context.makeImage())
        let fixture = root.appendingPathComponent("fixture.png")
        let destination = try #require(CGImageDestinationCreateWithURL(fixture as CFURL, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        #expect(CGImageDestinationFinalize(destination))
        let script = root.appendingPathComponent("worker")
        // Protocol fixture only; production inference is the native Swift Worker.
        let body = #"""
        #!/usr/bin/env python3
        import json,sys,shutil
        from pathlib import Path
        request=json.loads(Path(sys.argv[2]).read_text())
        assert request.get('inputPath') is None
        for path in request['outputPaths']:
            shutil.copyfile(Path(__file__).parent/'fixture.png',path)
            if request['prompt']=='fail':
                print(json.dumps({'type':'error','message':'fixture failure'}),flush=True)
                sys.exit(1)
        print(json.dumps({'type':'completed','value':1}),flush=True)
        """#
        try body.write(to: script, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: script.path)
        let service = Qwen21ImageService(outputDirectory: output, workerExecutable: script)
        var profile = QwenImage21Model.profiles[0]; profile.modelID = root.path
        let source = MediaAsset(projectID: UUID(), kind: .imported, title: "Linked source", fileURL: fixture, pixelWidth: 256, pixelHeight: 256)
        let request = TextToImageRequest(projectID: source.projectID,
            recipe: GenerationRecipe(prompt: fail ? "fail" : "test", modelID: QwenImage21Model.id,
                width: 256, height: 256, steps: 1, outputCount: 2), profile: profile, sourceAsset: source)
        if fail {
            await #expect(throws: Qwen21ImageService.Failure.self) {
                try await service.generate(request: request, progress: { _ in })
            }
            #expect(try FileManager.default.contentsOfDirectory(atPath: output.path).isEmpty)
        } else {
            let assets = try await service.generate(request: request, progress: { _ in })
            #expect(assets.count == 2)
            #expect(assets.allSatisfy { $0.parentAssetID == source.id && $0.kind == .generated })
            #expect(Set(assets.compactMap(\.fileURL)).count == 2)
            #expect(try FileManager.default.contentsOfDirectory(atPath: output.path).count == 2)
        }
    }
}

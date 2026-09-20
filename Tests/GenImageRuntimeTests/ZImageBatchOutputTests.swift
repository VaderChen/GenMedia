import Foundation
import GenImageCore
import Testing
@testable import GenImageRuntime

@Suite(.timeLimit(.minutes(1)))
struct ZImageBatchOutputTests {
    @Test(arguments: [false, true])
    func batchPreservesEveryImageAndRemovesPartialFailures(fail: Bool) async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("zimage-test-\(UUID())")
        let output = root.appendingPathComponent("output")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let script = root.appendingPathComponent("worker")
        let body = #"""
        #!/usr/bin/env python3
        import json, sys
        from pathlib import Path
        for line in sys.stdin:
            request = json.loads(line)
            for index, path in enumerate(request['outputPaths']):
                Path(path).write_text('frame-' + str(index))
                if request['prompt'] == 'fail':
                    break
            failed = request['prompt'] == 'fail'
            print(json.dumps({'requestID': request['requestID'],
                'type': 'error' if failed else 'completed', 'message': 'fixture failure'}), flush=True)
        """#
        try body.write(to: script, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: script.path)
        let service = ZImageTextToImageService(outputDirectory: output, workerExecutable: script)
        let profile = InferenceProfile(name: "Test", capability: .textToImage,
            modelID: root.path, architecture: .mlxSwift)
        let request = TextToImageRequest(projectID: UUID(),
            recipe: GenerationRecipe(prompt: fail ? "fail" : "batch", modelID: root.path, outputCount: 4),
            profile: profile)
        if fail {
            await #expect(throws: WarmRuntimeWorker.Failure.self) {
                try await service.generate(request: request, progress: { _ in })
            }
            #expect(try FileManager.default.contentsOfDirectory(atPath: output.path).isEmpty)
        } else {
            let assets = try await service.generate(request: request, progress: { _ in })
            let urls = assets.compactMap(\.fileURL)
            #expect(assets.count == 4)
            #expect(Set(urls).count == 4)
            #expect(try urls.map { try String(contentsOf: $0, encoding: .utf8) }
                == ["frame-0", "frame-1", "frame-2", "frame-3"])
        }
        await service.unload()
    }
}

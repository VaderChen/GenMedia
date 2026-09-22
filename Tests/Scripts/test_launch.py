"""Exercise the real build/launch scripts with disposable Swift outputs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
PRODUCTS = ["GenImage", "GenImageMCP", "GenImageDoctor", "GenImageQwen21Worker"]


@unittest.skipUnless(sys.platform == "darwin", "macOS build and launch scripts")
class LaunchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="genimage-launch-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "Project with spaces"
        self.root.mkdir()
        self.bin = self.root / ".build/Release with spaces"
        self.bin.mkdir(parents=True)
        self.mocks = self.root / "mocks"
        self.mocks.mkdir()
        for name in ["build.command", "run.command"]:
            shutil.copy2(ROOT / name, self.root / name)
        self.env = {**os.environ, "PATH": str(self.mocks) + ":" + os.environ["PATH"],
                    "MOCK_ROOT": str(self.root), "MOCK_BIN": str(self.bin),
                    "SKIP_METAL_TOOLCHAIN_DOWNLOAD": "1"}
        self.executable(self.mocks / "swift", '''#!/usr/bin/python3
import json, os, pathlib, sys
root = pathlib.Path(os.environ["MOCK_ROOT"])
output = pathlib.Path(os.environ["MOCK_BIN"])
args = sys.argv[1:]
with (root / "swift-calls.jsonl").open("a") as log:
    log.write(json.dumps(args) + "\\n")
if "--show-bin-path" in args:
    print(output)
    sys.exit(0)
if args[:1] == ["package"]:
    sys.exit(0)
if "--package-path" in args:
    sys.exit(75)  # Stop after testing root products, before any Worker/Metal work.
products = [args[i + 1] for i, item in enumerate(args) if item == "--product"]
product = products[-1]
if product == os.environ.get("FAIL_PRODUCT"):
    sys.exit(87)
if product != os.environ.get("MISSING_PRODUCT"):
    binary = output / product
    binary.write_text("#!/bin/sh\\nexit 0\\n")
    binary.chmod(0o755)
''')

    @staticmethod
    def executable(path, text):
        path.write_text(text)
        path.chmod(0o755)

    def run_script(self, name, *args, **env):
        return subprocess.run(["/bin/zsh", str(self.root / name), *args], cwd=self.root,
                              env={**self.env, **env}, capture_output=True,
                              text=True, timeout=30)

    def prepare_build(self):
        (self.root / "Sources/GenImageApp/Resources/WebUI").mkdir(parents=True)
        (self.root / "scripts").mkdir()
        self.executable(self.root / "scripts/apply-runtime-patches.command", "#!/bin/sh\nexit 0\n")
        for worker in [None, "Qwen2511Worker", "MiniMaxMusic3Worker", "LTXVideoWorker",
                       "MiniMaxH3Worker", "ZImageWorker"]:
            package = self.root if worker is None else self.root / "RuntimeSupport" / worker
            package.mkdir(parents=True, exist_ok=True)
            version = "0.30.6" if worker == "ZImageWorker" else "0.31.6"
            (package / "Package.resolved").write_text(json.dumps(
                {"pins": [{"identity": "mlx-swift", "state": {"version": version}}]},
                indent=2, separators=(",", " : ")))

    def root_products(self):
        calls = [json.loads(line) for line in (self.root / "swift-calls.jsonl").read_text().splitlines()]
        return [args[args.index("--product") + 1] for args in calls
                if "--product" in args and "--package-path" not in args]

    def test_build_produces_every_shipping_executable(self):
        self.prepare_build()
        result = self.run_script("build.command", "--no-app")
        self.assertEqual(result.returncode, 75, result.stdout + result.stderr)
        self.assertEqual(self.root_products(), PRODUCTS)
        for product in PRODUCTS:
            self.assertTrue(os.access(self.bin / product, os.X_OK), product)

    def test_build_stops_when_a_product_fails(self):
        self.prepare_build()
        result = self.run_script("build.command", "--no-app", FAIL_PRODUCT="GenImageDoctor")
        self.assertEqual(result.returncode, 87, result.stdout + result.stderr)
        self.assertEqual(self.root_products(), PRODUCTS[:3])

    def test_build_rejects_success_without_an_executable(self):
        self.prepare_build()
        result = self.run_script("build.command", "--no-app", MISSING_PRODUCT="GenImage")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("找不到可執行產品", result.stderr)
        self.assertIn(str(self.bin / "GenImage"), result.stderr)
        self.assertEqual(self.root_products(), ["GenImage"])

    def test_run_launches_current_output_and_preserves_arguments(self):
        self.executable(self.root / "build.command", "#!/bin/sh\nexit 0\n")
        self.executable(self.bin / "GenImage", '''#!/usr/bin/python3
import json, os, pathlib, sys
pathlib.Path(os.environ["MOCK_ROOT"], "launched.json").write_text(json.dumps({
    "args": sys.argv[1:], "worker": os.environ["GENIMAGE_QWEN_WORKER"]}))
''')
        result = self.run_script("run.command", "argument with spaces", "literal $value")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        launched = json.loads((self.root / "launched.json").read_text())
        self.assertEqual(launched["args"], ["argument with spaces", "literal $value"])
        self.assertEqual(launched["worker"], str(self.bin / "GenImageQwen2511Worker"))

    def test_run_reports_missing_main_executable(self):
        self.executable(self.root / "build.command", "#!/bin/sh\nexit 0\n")
        result = self.run_script("run.command")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("找不到 GenImage 主程式", result.stderr)
        self.assertNotIn("正在啟動", result.stdout)


if __name__ == "__main__":
    unittest.main()

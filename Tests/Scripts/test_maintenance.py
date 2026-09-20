"""Run with: python3 -m unittest discover -s Tests/Scripts -v.

Execute the actual maintenance scripts against disposable projects. No models,
installed FFmpeg files, or user backups are modified.
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[2]


class MaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="genimage-maintenance-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "Project with spaces"
        (self.root / "Sources").mkdir(parents=True)
        (self.root / "Package.swift").write_text("// fixture\n")
        (self.root / "scripts").mkdir()
        for name in ["backup.command", "clean.command", "build.command",
                     "scripts/project-packages.sh", "scripts/build-ffmpeg-macos.sh",
                     "scripts/pkg-config-fallback.sh"]:
            shutil.copy2(ROOT / name, self.root / name)

    def run_script(self, name, env=None):
        return subprocess.run(["/bin/zsh", str(self.root / name)], cwd=self.root,
                              env={**os.environ, **(env or {})}, capture_output=True,
                              text=True, timeout=15)

    def packages(self):
        packages = [self.root]
        for worker in ["Qwen2511Worker", "ZImageWorker", "LTXVideoWorker",
                       "MiniMaxMusic3Worker", "MiniMaxH3Worker", "FutureWorker"]:
            package = self.root / "RuntimeSupport" / worker
            package.mkdir(parents=True)
            (package / "Package.swift").write_text("// fixture\n")
            packages.append(package)
        for package in packages:
            for cache in [".build", ".swiftpm"]:
                (package / cache).mkdir()
                (package / cache / "cache.bin").write_bytes(b"cache")
        return packages

    def test_backup_excludes_all_package_caches_and_backups(self):
        self.packages()
        (self.root / "Sources/code.swift").write_text("// keep\n")
        (self.root / "source.bak").write_text("backup")
        (self.root / "refactor.bak").mkdir()
        (self.root / "refactor.bak/secret.swift").write_text("backup")
        result = self.run_script("backup.command")
        self.assertEqual(result.returncode, 0, result.stderr)
        archive, = (self.root / "Backups").glob("*.zip")
        with zipfile.ZipFile(archive) as zipped:
            files = [name for name in zipped.namelist() if not name.endswith("/")]
        self.assertTrue(any(name.endswith("Sources/code.swift") for name in files))
        self.assertFalse(any("/.build/" in name or "/.swiftpm/" in name for name in files), files)
        self.assertFalse(any(".bak" in name or "/Backups/" in name for name in files), files)

    def test_clean_preserves_backups_and_user_data(self):
        packages = self.packages()
        protected = ["Backups/important.bak", "Backups/.DS_Store", "source.bak",
                     "refactor.bak/code.swift", "Models/.DS_Store", "Outputs/movie.mp4",
                     "Generated/image.png", "dist/.DS_Store", "third_party/.DS_Store"]
        for name in protected + ["Sources/.DS_Store", ".DS_Store"]:
            file = self.root / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(b"keep")
        result = self.run_script("clean.command")
        self.assertEqual(result.returncode, 0, result.stderr)
        for package in packages:
            self.assertFalse((package / ".build").exists())
            self.assertTrue((package / "Package.swift").exists())
        for name in protected:
            self.assertEqual((self.root / name).read_bytes(), b"keep", name)
        self.assertFalse((self.root / "Sources/.DS_Store").exists())
        self.assertFalse((self.root / ".DS_Store").exists())

    def ffmpeg_fixture(self, early_failure=False, configure="exit 7\n"):
        prefix = self.root / "ffmpeg"
        prefix.mkdir()
        (prefix / "installed-version").write_text("original")
        source = self.root / "source"
        source.mkdir()
        if not early_failure:
            (source / "ffmpeg-fixture").mkdir()
            lame = source / "lame-fixture"
            lame.mkdir()
            script = lame / "configure"
            script.write_text("#!/bin/sh\n" + configure)
            script.chmod(0o755)
        mocks = self.root / "mocks"
        mocks.mkdir()
        curl = mocks / "curl"
        curl.write_text("#!/bin/sh\nexit 22\n")
        curl.chmod(0o755)
        env = {"PATH": str(mocks) + ":" + os.environ["PATH"],
               "FFMPEG_PREFIX": str(prefix), "FFMPEG_SOURCE_ROOT": str(source),
               "FFMPEG_VERSION": "fixture", "LAME_VERSION": "fixture"}
        return prefix, env

    def test_ffmpeg_download_failure_preserves_existing_installation(self):
        prefix, env = self.ffmpeg_fixture(early_failure=True)
        result = self.run_script("scripts/build-ffmpeg-macos.sh", env)
        self.assertEqual(result.returncode, 22, result.stderr)
        self.assertEqual((prefix / "installed-version").read_text(), "original")

    def test_ffmpeg_configure_failure_restores_existing_installation(self):
        prefix, env = self.ffmpeg_fixture()
        result = self.run_script("scripts/build-ffmpeg-macos.sh", env)
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertEqual((prefix / "installed-version").read_text(), "original")
        self.assertFalse(prefix.with_suffix(".bak").exists())

    def test_ffmpeg_signal_restores_installation_only_once(self):
        prefix, env = self.ffmpeg_fixture(configure='kill -TERM "$PPID"\nexit 0\n')
        result = self.run_script("scripts/build-ffmpeg-macos.sh", env)
        self.assertEqual(result.returncode, 143, result.stderr)
        self.assertEqual((prefix / "installed-version").read_text(), "original")
        self.assertFalse(prefix.with_suffix(".bak").exists())

    def test_ffmpeg_preexisting_backup_preserves_both_versions(self):
        prefix, env = self.ffmpeg_fixture()
        backup = prefix.with_suffix(".bak")
        backup.mkdir()
        (backup / "old-version").write_text("backup")
        result = self.run_script("scripts/build-ffmpeg-macos.sh", env)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((prefix / "installed-version").read_text(), "original")
        self.assertEqual((backup / "old-version").read_text(), "backup")

    def test_missing_license_fails_before_build_or_download(self):
        result = self.run_script("build.command")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("LICENSE.md", result.stderr)
        self.assertFalse((self.root / ".build").exists())


if __name__ == "__main__":
    unittest.main()

"""The privacy hooks must be installed, executable and configured.

They exist because two leaks reached the remote while every test was green: the
tests scanned the tracked tree only and ran before `git add`, so a file created
by a cluster job and swept in by `git add -A` was never scanned, and nothing
looked at author, committer or tagger fields. scripts/prepush_audit.sh closes
both holes; this test keeps the wiring from rotting.

  python -m unittest tests.test_hooks
"""
import os
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def git(*args: str) -> str:
    return subprocess.run(["git"] + list(args), cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()


class TestHooksInstalled(unittest.TestCase):
    def test_hooks_path_is_configured(self):
        self.assertEqual(git("config", "--get", "core.hooksPath"), ".githooks",
                         "core.hooksPath must point at .githooks")

    def test_hook_files_are_executable(self):
        for name in ("pre-commit", "pre-push"):
            path = ROOT / ".githooks" / name
            self.assertTrue(path.exists(), f"missing {path}")
            self.assertTrue(os.access(path, os.X_OK),
                            f"{path} is not executable (chmod +x needed)")

    def test_audit_scripts_exist_and_are_executable(self):
        for name in ("prepush_audit.sh", "prepush_audit.py"):
            path = ROOT / "scripts" / name
            self.assertTrue(path.exists(), f"missing {path}")
            self.assertTrue(os.access(path, os.X_OK), f"{path} is not executable")

    def test_ci_runs_the_audit(self):
        ci = (ROOT / ".github" / "workflows" / "tests.yml").read_text()
        self.assertIn("prepush_audit", ci,
                      "the CI workflow must run the prepush audit")

    def test_audit_runs_clean_on_head(self):
        """The audit itself, over the current tree and the last commit."""
        out = subprocess.run(["sh", "scripts/prepush_audit.sh", "--range", "HEAD"],
                             cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(out.returncode, 0,
                         "prepush_audit failed on HEAD:\n" + out.stderr)


if __name__ == "__main__":
    unittest.main()

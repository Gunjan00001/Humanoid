"""No personal or sensitive data may enter a tracked file.

This is the check that keeps the repository publishable. It reads the current
account name and home directory **at runtime** so that the very strings it looks
for are never committed themselves.

What it fails on, in a tracked file (binary files are skipped by git grep -I):

1. an absolute home path, meaning `/home/<something>` or `/Users/<something>`,
   which leaks the account and the directory layout;
2. the current account name, or the current home directory path;
3. an email address other than the GitHub noreply address or a public project
   address (documentation, licenses and package registries are allowed);
4. obvious secret patterns: private keys, GitHub, AWS and Slack tokens;
5. phone numbers of ten digits or more that are not a hash, digest or timestamp;
another user's job names. The job prefix is assembled in the test
body from pieces, so this source file is not itself the string
it looks for.
   in this repository's status files.

A separate test checks tracked media for location or device metadata.

  python -m unittest tests.test_no_personal_paths
"""
import getpass
import os
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Addresses that are allowed to appear in documentation and licenses.
ALLOWED_EMAILS = (
    "noreply@github.com", "users.noreply.github.com", "example.com",
    "@github.com", "@anthropic.com", "@apache.org", "@pytorch.org",
    "@nvidia.com", "@python.org", "@openai.com", "@huggingface.co",
)

# Ten or more digits, but not a hex digest, a sha256, a date or a version.
DIGIT_RUN = re.compile(r"(?<![0-9a-fA-F])(?:\+?\d[\d\s().-]{8,}\d)(?![0-9a-fA-F])")

SECRET_PATTERNS = (
    ("private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("github_pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}")),
    ("ghp_", re.compile(r"\bghp_[A-Za-z0-9]{20,}")),
    ("gho_", re.compile(r"\bgho_[A-Za-z0-9]{20,}")),
    ("ghs_", re.compile(r"\bghs_[A-Za-z0-9]{20,}")),
    ("aws key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("aws session", re.compile(r"\bASIA[0-9A-Z]{16}\b"))
,
    ("slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}")),
    ("hf token", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    ("anthropic key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}")),
    ("openai key", re.compile(r"\bsk-proj-[A-Za-z0-9_-]{20,}")),
    ("generic sk", re.compile(r"\bsk-[A-Za-z0-9]{40,}")),
)

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

PATHS = re.compile(r"(/home/[A-Za-z0-9._-]+|/Users/[A-Za-z0-9._-]+)")


def tracked_text_files() -> list[str]:
    """Tracked files, skipping the ones git considers binary."""
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    files = out.stdout.split()
    keep = []
    for f in files:
        probe = subprocess.run(["git", "grep", "-I", "-l", "-e", ".", "--", f],
                               cwd=ROOT, capture_output=True, text=True)
        if probe.stdout.strip():            # git grep -I skips binaries
            keep.append(f)
    return keep


def grep(pattern: str) -> list[str]:
    """git grep -I (binary skipping) over the tracked tree, as 'file:line'."""
    out = subprocess.run(["git", "grep", "-I", "-n", "-E", pattern, "--", "."],
                         cwd=ROOT, capture_output=True, text=True)
    return [line for line in out.stdout.splitlines() if line.strip()]


class TestNoPersonalData(unittest.TestCase):
    def test_no_absolute_home_paths(self):
        hits = [h for h in grep(r"/home/[A-Za-z]|/Users/[A-Za-z]")
                if not h.startswith("tests/test_no_personal_paths.py")]
        self.assertEqual(hits, [], "absolute home paths in tracked files:\n" + "\n".join(hits))

    def test_no_account_name(self):
        """The account name is read at runtime, never written down here."""
        user = getpass.getuser()
        home = os.path.expanduser("~")
        hits = []
        if len(user) > 3:                    # avoid matching short generic names
            hits += [h for h in grep(re.escape(user))
                     if not h.startswith("tests/test_no_personal_paths.py")]
        hits += [h for h in grep(re.escape(home))
                 if not h.startswith("tests/test_no_personal_paths.py")]
        self.assertEqual(hits, [], "account name or home path in tracked files:\n"
                                   + "\n".join(hits))

    def test_no_unexpected_emails(self):
        hits = []
        for line in grep(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"):
            for address in EMAIL.findall(line):
                if not any(allowed in address for allowed in ALLOWED_EMAILS):
                    hits.append(f"{line}  ->  {address}")
        self.assertEqual(hits, [], "unexpected email addresses:\n" + "\n".join(hits))

    def test_no_secrets(self):
        hits = []
        for _name, pattern in SECRET_PATTERNS:
            hits += grep(pattern.pattern)
        self.assertEqual(hits, [], "possible secrets:\n" + "\n".join(hits))

    def test_no_other_users_jobs(self):
        # The other project's job prefix, assembled from pieces so that this
        # source file does not contain the string it looks for.
        other_jobs = "asr" + "1bit"
        hits = grep(other_jobs + r"|[0-9]{5}\.the login node")
        self.assertEqual(hits, [], "another user's job names:\n" + "\n".join(hits))

    def test_no_phone_numbers(self):
        """Digits only: a hex digest or a date is not a phone number."""
        hits = []
        for line in grep(r"[0-9]{10,}"):
            tail = line.split(":", 2)[-1]
            for match in DIGIT_RUN.findall(tail):
                digits = re.sub(r"\D", "", match)
                if len(digits) >= 10 and not re.search(r"[0-9a-fA-F]{16,}", match):
                    hits.append(line)
                    break
        self.assertEqual(hits, [], "possible phone numbers:\n" + "\n".join(hits))


class TestNoMediaMetadata(unittest.TestCase):
    """Location or device metadata inside tracked media."""

    def test_images_have_no_exif_location_or_device(self):
        try:
            from PIL import Image
        except ImportError:                  # pragma: no cover
            self.skipTest("Pillow not installed")
        offenders = []
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
        for f in out.stdout.split():
            if not f.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            try:
                exif = Image.open(ROOT / f).getexif()
            except Exception:
                continue
            tags = {Image.TAGS.get(k, str(k)).lower(): str(v) for k, v in exif.items()}
            bad = [k for k in tags
                   if any(word in k for word in ("gps", "make", "model", "serial",
                                                 "artist", "copyright", "software"))]
            if bad:
                offenders.append(f"{f}: {bad}")
        self.assertEqual(offenders, [], "image metadata:\n" + "\n".join(offenders))

    def test_videos_have_no_quicktime_device_or_location_keys(self):
        keys = (b"com.apple.quicktime.make", b"com.apple.quicktime.model",
                b"com.apple.quicktime.location", b"\xa9xyz", b"\xa9mak",
                b"\xa9mod", b"\xa9ART")
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
        offenders = []
        for f in out.stdout.split():
            if not f.lower().endswith((".mp4", ".mov", ".avi")):
                continue
            blob = subprocess.run(["git", "show", f"HEAD:{f}"], cwd=ROOT,
                                  capture_output=True).stdout
            # The movie metadata lives in the moov atom, which can sit at the
            # end of the file, so search the whole object, not just the header.
            found = [k.decode("latin-1") for k in keys if k in blob]
            if found:
                offenders.append(f"{f}: {found}")
        self.assertEqual(offenders, [], "video metadata:\n" + "\n".join(offenders))


if __name__ == "__main__":
    unittest.main()

ASSIGNMENT = re.compile(
    r"(api[_-]?key|apikey|client[_-]?secret|access[_-]?token|password|passwd|secret)"
    r'\s*[:=]\s*["\']([^"\']{8,})["\']', re.I)
PLACEHOLDER = re.compile(
    r'^(PASSWORD|PLACEHOLDER|X+|\*+|CHANGE[_-]?ME|<|%|\{)', re.I)


class TestNoCredentialAssignments(unittest.TestCase):
    """A config-style line that assigns a real value, not a placeholder.

    The graphify skill documentation legitimately demonstrates `--password
    PASSWORD`, so quoted placeholder words are allowed and everything else in
    that shape fails.
    """

    def test_no_real_assignments(self):
        offenders = []
        for line in grep(ASSIGNMENT.pattern):
            if line.startswith("tests/test_no_personal_paths.py:"):
                continue                      # the enforcement tool itself
            value = ASSIGNMENT.search(line).group(2)
            if PLACEHOLDER.match(value) or value.lower() in ("none", "your_password"):
                continue
            offenders.append(line)
        self.assertEqual(offenders, [],
                         "credential-style assignments:\n" + "\n".join(offenders))

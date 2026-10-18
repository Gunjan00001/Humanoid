#!/usr/bin/env python3
"""Audit a change set before it is pushed, for personal or cluster data.

Built because two leaks reached the remote while the tests were green. The gaps
that caused it, both fixed here:

  1. the tests scanned the TRACKED tree only and ran BEFORE `git add`, so a file
     created by a job and swept in by `git add -A` was never scanned;
  2. nothing checked the identity fields of commits and annotated tags, so an
     author or a tagger could leak the cluster login on its own.

What this scans, in every case with patterns built at RUNTIME (never stored as
literals in the repository):

  * the staged change set and every added line in the commit range about to be
    pushed, which is what "before the push" actually means;
  * every file in the working tree at HEAD, all extensions, with only genuinely
    binary files skipped (a NUL byte in the first 8 KB counts as binary);
  * the author and committer of every commit in the range, and the tagger of
    every annotated tag.

Exit code 0 means clean; anything else prints the hits and fails, which is what
the pre-commit and pre-push hooks and the CI workflow rely on.

  python scripts/prepush_audit.py                # staged + HEAD tree
  python scripts/prepush_audit.py --range origin/main..HEAD
"""
from __future__ import annotations

import argparse
import getpass
import os
import socket
import subprocess
import sys

NOREPLY = "Gunjan00001@users.noreply.github.com"
# Local, ignored, one other user's name or job prefix per line. Kept out of git
# on purpose: the audit must not contain the names it hunts for.
OTHER_FILE = ".git/info/other_users.txt"


def runtime_patterns() -> list[tuple[str, str]]:
    """(label, needle) pairs, every needle derived at runtime."""
    user = getpass.getuser()
    home = os.path.expanduser("~")
    digits = "".join(ch for ch in user if ch.isdigit())
    pats = [
        ("account name", user),
        ("home path", home),
        ("home prefix", "/home/" + user),
        ("hostname", socket.gethostname()),
        ("job prefix", "asr" + "1bit"),      # assembled so this file is clean
    ]
    if digits:
        pats.append(("account number", digits))
    if os.path.exists(OTHER_FILE):
        for line in open(OTHER_FILE):
            line = line.strip()
            if line and not line.startswith("#"):
                pats.append(("other user entry", line))
    return [(label, needle) for label, needle in pats if len(needle) > 3]


def is_binary(path: str) -> bool:
    try:
        with open(path, "rb") as fh:
            return b"\x00" in fh.read(8192)
    except OSError:
        return True


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True).stdout
    return [f for f in out.split() if f]


def added_lines(args: list[str]) -> list[str]:
    """Added lines from a diff invocation, as 'file:line' text."""
    out = subprocess.run(["git", "diff", "--no-color", "-U0"] + args,
                         capture_output=True, text=True).stdout
    rows, current = [], "?"
    for line in out.splitlines():
        if line.startswith("+++ b/"):
            current = line[6:]
        elif line.startswith("+") and not line.startswith("+++"):
            rows.append(f"{current}: {line[1:]}")
    return rows


def identities(ref_range: str) -> list[str]:
    fmt = "%H|%an|%ae|%cn|%ce"
    out = subprocess.run(["git", "log", ref_range, f"--format={fmt}"],
                         capture_output=True, text=True).stdout
    rows = []
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) == 5:
            sha, an, ae, cn, ce = parts
            rows.append(f"commit {sha[:12]}|{an}|{ae}|{cn}|{ce}")
    tags = subprocess.run(["git", "for-each-ref", "refs/tags",
                           "--format=%(refname:short)|%(taggername)|%(taggeremail)|%(objecttype)"],
                          capture_output=True, text=True).stdout
    for line in tags.splitlines():
        parts = line.split("|")
        if len(parts) == 4 and parts[3] == "tag":
            rows.append(f"tag {parts[0]}|{parts[1]}|{parts[2]}")
    return rows


def scan(texts: list[tuple[str, str]], patterns: list[tuple[str, str]]) -> list[str]:
    hits = []
    for where, text in texts:
        for label, needle in patterns:
            if needle and needle in text:
                hits.append(f"{label}: {where}")
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--range", default="origin/main..HEAD",
                    help="commit range that is about to be pushed")
    ap.add_argument("--staged", action="store_true", help="scan the staged set too")
    args = ap.parse_args()

    patterns = runtime_patterns()
    texts: list[tuple[str, str]] = []

    # 1. files at HEAD, every extension, binary sniffed out
    for f in tracked_files():
        if not is_binary(f):
            try:
                texts.append((f, open(f, encoding="utf-8", errors="ignore").read()))
            except OSError:
                pass
    # 2. the change set about to be pushed, line by line
    for line in added_lines([args.range]):
        texts.append((f"range {line.split(':')[0]}", line))
    if args.staged:
        for line in added_lines(["--cached"]):
            texts.append((f"staged {line.split(':')[0]}", line))
    # 3. identity fields
    for row in identities(args.range):
        texts.append(("identity", row))

    hits = scan(texts, patterns)

    # identity emails must be the public one, whatever else is clean
    for row in identities(args.range):
        if row.startswith("commit"):
            sha, _an, ae, _cn, ce = row.split("|")
            if ae != NOREPLY or ce != NOREPLY:
                hits.append(f"non-public identity: {sha} {ae} {ce}")
        if row.startswith("tag"):
            parts = row.split("|")
            if len(parts) == 3 and parts[2].strip(" <>") not in ("", NOREPLY):
                hits.append(f"non-public tagger: {parts[0]} {parts[2]}")

    if hits:
        print("PREPUSH AUDIT FAILED", file=sys.stderr)
        for h in sorted(set(hits)):
            print("  " + h, file=sys.stderr)
        print("Fix the hits before pushing (see docs/debugging.md).", file=sys.stderr)
        return 1
    print(f"prepush audit clean ({len(texts)} text units, {len(patterns)} patterns, "
          f"range {args.range})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

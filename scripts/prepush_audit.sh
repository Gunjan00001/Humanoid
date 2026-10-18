#!/bin/sh
# Audit before every commit and every push. Thin wrapper around the Python
# implementation so the logic lives in one place and can be tested.
#
# Installed as .githooks/pre-commit and .githooks/pre-push; enable once per
# clone with:  git config core.hooksPath .githooks
#
# Usage: scripts/prepush_audit.sh [--range origin/main..HEAD]

set -e
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

PY="${SPATIALDIFF_PYTHON:-$HOME/miniconda3/envs/spatialdiff/bin/python}"
if [ ! -x "$PY" ]; then
  PY="$(command -v python3)"
fi

exec "$PY" scripts/prepush_audit.py --staged "$@"

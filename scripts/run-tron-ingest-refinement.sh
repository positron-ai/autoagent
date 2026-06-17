#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTOAGENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CALLER_CWD="$PWD"

exec uv run --project "$AUTOAGENT_DIR" --directory "$CALLER_CWD" ingest --create-worktree "$@"

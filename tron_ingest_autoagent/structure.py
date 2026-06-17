"""Analyze Tron ingest structural artifacts.

This module scores the "EqSat structural" gate. It is deliberately tolerant of
which artifacts are available: full IR dumps are best, but generated Bulk
Python and generated C++ are enough to detect whether the key hardware-oriented
patterns survived into generated code.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TEXT_SUFFIXES = {
    ".bulk",
    ".bulk-py",
    ".cpp",
    ".egraph",
    ".h",
    ".hpp",
    ".json",
    ".loopy",
    ".py",
    ".rewritefx",
    ".rewritten",
    ".trace",
    ".tron",
    ".typedfx",
    ".txt",
}

PATTERNS: dict[str, tuple[str, ...]] = {
    "tron_sdpa": (
        r"\bTronSDPA\b",
        r"\bapply_tron_sdpa\b",
        r"\bself_attention_t\b",
        r"\brun_attention_worker\b",
        r"\bsdpa_channel\b",
        r"\bsdpa(?:_\d+)?\b",
    ),
    "tron_rope": (
        r"\bTronRope\b",
        r"\bapply_tron_rope\b",
        r"\bkernel_rope(?:_\d+)?\b",
        r"\brope_cfg\b",
        r"\brope_config\b",
    ),
    "rms_norm": (
        r"\bRmsNormMul\b",
        r"\brms_norm\b",
        r"\brmsnorm\b",
        r"\bkernel_.*rmsnorm\b",
    ),
    "swishmul": (
        r"\bSwishMul\b",
        r"\bswish_mul\b",
        r"\bswishmul\b",
    ),
}


@dataclass(frozen=True)
class PatternHit:
    pattern: str
    count: int
    files: tuple[str, ...]


def discover_files(roots: list[Path], explicit_files: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    files: list[Path] = []

    def add(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen or not path.is_file():
            return
        if path.suffix not in TEXT_SUFFIXES and not any(
            str(path).endswith(suffix) for suffix in TEXT_SUFFIXES
        ):
            return
        seen.add(resolved)
        files.append(path)

    for path in explicit_files:
        add(path)

    for root in roots:
        if root.is_file():
            add(root)
        elif root.is_dir():
            for path in root.rglob("*"):
                add(path)

    return sorted(files)


def read_text(path: Path, max_bytes: int) -> str:
    data = path.read_bytes()
    if len(data) > max_bytes:
        data = data[:max_bytes]
    return data.decode("utf-8", errors="replace")


def infer_expected_patterns(files: list[Path], max_bytes: int) -> list[str]:
    """Infer expected structural patterns from model metadata and weights."""
    text = "\n".join(read_text(path, max_bytes) for path in files if path.name == "metadata.json")
    expected: set[str] = set()

    if re.search(r"self_attn\.(?:q_proj|k_proj|v_proj)", text):
        expected.add("tron_sdpa")
        expected.add("tron_rope")
    if "input_layernorm" in text or "post_attention_layernorm" in text or "rms" in text.lower():
        expected.add("rms_norm")
    if "mlp.gate_proj" in text and "mlp.up_proj" in text:
        expected.add("swishmul")

    return sorted(expected)


def analyze_patterns(
    *,
    roots: list[Path],
    files: list[Path] | None = None,
    expected_patterns: list[str] | None = None,
    max_bytes_per_file: int = 5_000_000,
) -> dict[str, Any]:
    files = discover_files(roots, files or [])
    if expected_patterns is None:
        expected_patterns = infer_expected_patterns(files, max_bytes_per_file)
    expected_patterns = sorted(dict.fromkeys(expected_patterns))

    unknown = [name for name in expected_patterns if name not in PATTERNS]
    if unknown:
        raise ValueError(f"unknown structural patterns: {', '.join(unknown)}")

    counts: dict[str, int] = {name: 0 for name in expected_patterns}
    hits_by_file: dict[str, list[str]] = defaultdict(list)
    files_by_pattern: dict[str, set[str]] = {name: set() for name in expected_patterns}

    for path in files:
        text = read_text(path, max_bytes_per_file)
        rel = str(path)
        for name in expected_patterns:
            count = 0
            for regex in PATTERNS[name]:
                count += len(re.findall(regex, text))
            if count:
                counts[name] += count
                files_by_pattern[name].add(rel)
                hits_by_file[rel].append(name)

    missing = [name for name in expected_patterns if counts.get(name, 0) == 0]
    matched = [name for name in expected_patterns if counts.get(name, 0) > 0]

    if expected_patterns:
        score = len(matched) / len(expected_patterns)
        passed = not missing
    else:
        score = 0.0
        passed = False

    return {
        "passed": passed,
        "score": score,
        "expected_patterns": expected_patterns,
        "matched_patterns": matched,
        "missing_patterns": missing,
        "counts": counts,
        "files": [str(path) for path in files],
        "files_by_pattern": {
            name: sorted(files_by_pattern[name]) for name in expected_patterns
        },
        "patterns_by_file": {name: sorted(patterns) for name, patterns in hits_by_file.items()},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        action="append",
        default=[],
        type=Path,
        help="Directory or file to scan. Can be passed more than once.",
    )
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        type=Path,
        help="Specific file to scan. Can be passed more than once.",
    )
    parser.add_argument(
        "--expected-pattern",
        action="append",
        default=[],
        choices=sorted(PATTERNS),
        help="Expected structural pattern. If omitted, infer from metadata.json.",
    )
    parser.add_argument(
        "--max-bytes-per-file",
        type=int,
        default=5_000_000,
        help="Maximum bytes to read from each artifact.",
    )
    parser.add_argument("--output-json", type=Path, help="Path for structural result JSON")
    parser.add_argument("--print-json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    expected = args.expected_pattern or None
    result = analyze_patterns(
        roots=args.artifact_root,
        files=args.file,
        expected_patterns=expected,
        max_bytes_per_file=args.max_bytes_per_file,
    )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    if args.print_json:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

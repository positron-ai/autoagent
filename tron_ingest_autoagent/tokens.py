"""Compute token-sequence agreement for Tron ingest verification."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _as_sequence(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, int, float)):
        text = str(value).strip()
        if not text:
            return []
        if re.fullmatch(r"[-+]?\d+(?:\s+[-+]?\d+)*", text):
            return text.split()
        return list(text)
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    raise TypeError(f"cannot convert {type(value).__name__} to token sequence")


def _load_any(path: Path) -> Any:
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        stripped = text.strip()
        if re.fullmatch(r"[-+]?\d+(?:\s+[-+]?\d+)*", stripped):
            return stripped.split()
        return stripped


def _extract_sequence(payload: Any, keys: tuple[str, ...]) -> list[str]:
    if isinstance(payload, dict):
        for key in keys:
            if key in payload:
                return _as_sequence(payload[key])
        for key in ("tokens", "token_ids", "output_tokens", "text", "output"):
            if key in payload:
                return _as_sequence(payload[key])
    return _as_sequence(payload)


def common_prefix_len(a: list[str], b: list[str]) -> int:
    count = 0
    for left, right in zip(a, b):
        if left != right:
            break
        count += 1
    return count


def levenshtein_distance(a: list[str], b: list[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, item_a in enumerate(a, start=1):
        current = [i]
        for j, item_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (0 if item_a == item_b else 1),
                )
            )
        previous = current
    return previous[-1]


def compare_sequences(name: str, reference: list[str], candidate: list[str]) -> dict[str, Any]:
    max_len = max(len(reference), len(candidate))
    min_len = min(len(reference), len(candidate))
    positional_matches = sum(
        1 for left, right in zip(reference, candidate) if left == right
    )
    exact_fraction = positional_matches / max_len if max_len else 1.0
    prefix = common_prefix_len(reference, candidate)
    prefix_fraction = prefix / len(reference) if reference else (1.0 if not candidate else 0.0)
    edit_distance = levenshtein_distance(reference, candidate)
    edit_similarity = 1.0 - (edit_distance / max_len) if max_len else 1.0
    exact_match = reference == candidate

    # This mirrors the scorer's token weighting so the artifact is directly
    # usable as tau while still exposing its raw components.
    score = (
        0.40 * (1.0 if exact_match else exact_fraction)
        + 0.25 * prefix_fraction
        + 0.20 * edit_similarity
        + 0.15 * exact_fraction
    )

    return {
        "name": name,
        "reference_length": len(reference),
        "candidate_length": len(candidate),
        "compared_length": min_len,
        "exact_match": exact_match,
        "exact_fraction": exact_fraction,
        "prefix_match": prefix_fraction,
        "edit_distance": edit_distance,
        "edit_similarity": edit_similarity,
        "top1_agreement": exact_fraction,
        "score": max(0.0, min(1.0, score)),
    }


def compare_payloads(reference_payload: Any, candidate_payload: Any) -> dict[str, Any]:
    if isinstance(reference_payload, dict) and isinstance(candidate_payload, dict):
        ref_cases = reference_payload.get("cases")
        cand_cases = candidate_payload.get("cases")
        if isinstance(ref_cases, list) and isinstance(cand_cases, list):
            by_name = {
                str(case.get("name", index)): case
                for index, case in enumerate(cand_cases)
                if isinstance(case, dict)
            }
            cases = []
            for index, ref_case in enumerate(ref_cases):
                if not isinstance(ref_case, dict):
                    continue
                name = str(ref_case.get("name", index))
                cand_case = by_name.get(name, {})
                cases.append(
                    compare_sequences(
                        name,
                        _extract_sequence(ref_case, ("reference_tokens", "tokens", "text")),
                        _extract_sequence(cand_case, ("candidate_tokens", "tokens", "text")),
                    )
                )
            return aggregate_cases(cases)

    case = compare_sequences(
        "default",
        _extract_sequence(reference_payload, ("reference_tokens", "tokens", "text")),
        _extract_sequence(candidate_payload, ("candidate_tokens", "tokens", "text")),
    )
    return aggregate_cases([case])


def aggregate_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not cases:
        return {"score": 0.0, "tau": 0.0, "cases": []}
    score = sum(float(case["score"]) for case in cases) / len(cases)
    exact_fraction = sum(float(case["exact_fraction"]) for case in cases) / len(cases)
    prefix = sum(float(case["prefix_match"]) for case in cases) / len(cases)
    edit = sum(float(case["edit_similarity"]) for case in cases) / len(cases)
    return {
        "score": score,
        "tau": score,
        "exact_fraction": exact_fraction,
        "prefix_match": prefix,
        "edit_similarity": edit,
        "top1_agreement": exact_fraction,
        "cases": cases,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--print-json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = compare_payloads(_load_any(args.reference), _load_any(args.candidate))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    if args.print_json:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

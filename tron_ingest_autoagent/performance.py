"""Compute performance score against speed-of-light throughput."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


THROUGHPUT_PATTERNS = (
    re.compile(r"Throughput\s+([0-9]+(?:\.[0-9]+)?)\s+new\b", re.I),
    re.compile(r"Throughput\s+([0-9]+(?:\.[0-9]+)?)\s+(?:new\s+)?(?:tok/s|tokens/s)", re.I),
    re.compile(r"at\s+([0-9]+(?:\.[0-9]+)?)\s+average\s+(?:tok/s|tokens/s)", re.I),
    re.compile(r"([0-9]+(?:\.[0-9]+)?)\s+(?:tok/s|tokens/s)", re.I),
)

MEASURED_KEYS = (
    "measured_tokens_per_second",
    "measured_tps",
    "tokens_per_second",
    "throughput_tps",
    "throughput",
    "throughput_mean",
    "Gen tok/s",
)

LIMIT_KEYS = (
    "speed_of_light_tokens_per_second",
    "speed_of_light_tps",
    "theoretical_tokens_per_second",
    "theoretical_tps",
    "speed_of_light",
    "limit_tps",
)


def _numbers_from_text(text: str) -> list[float]:
    for pattern in THROUGHPUT_PATTERNS:
        values = [float(match.group(1)) for match in pattern.finditer(text)]
        if values:
            return values
    return []


def _find_numeric(payload: Any, keys: tuple[str, ...]) -> float | None:
    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    pass
        for value in payload.values():
            found = _find_numeric(value, keys)
            if found is not None:
                return found
    elif isinstance(payload, list):
        values = [_find_numeric(item, keys) for item in payload]
        numbers = [value for value in values if value is not None]
        if numbers:
            return sum(numbers) / len(numbers)
    elif isinstance(payload, (int, float)):
        return float(payload)
    return None


def _load_payload(path: Path) -> Any:
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"text": text, "throughputs": _numbers_from_text(text)}


def extract_measured(path: Path) -> tuple[float | None, dict[str, Any]]:
    payload = _load_payload(path)
    if isinstance(payload, dict) and isinstance(payload.get("throughputs"), list):
        values = [float(value) for value in payload["throughputs"]]
        if values:
            return max(values), {
                "source": str(path),
                "values": values,
                "method": "log_regex_preferred",
            }
    numeric = _find_numeric(payload, MEASURED_KEYS)
    return numeric, {"source": str(path), "payload": payload, "method": "json_keys"}


def extract_limit(path: Path) -> tuple[float | None, dict[str, Any]]:
    payload = _load_payload(path)
    numeric = _find_numeric(payload, LIMIT_KEYS)
    if numeric is None and isinstance(payload, dict) and isinstance(payload.get("throughputs"), list):
        values = [float(value) for value in payload["throughputs"]]
        if values:
            numeric = max(values)
    return numeric, {"source": str(path), "payload": payload}


def score_performance(measured_tps: float | None, speed_of_light_tps: float | None) -> dict[str, Any]:
    if measured_tps is None or speed_of_light_tps is None or speed_of_light_tps <= 0:
        return {
            "passed": False,
            "score": 0.0,
            "delta": 0.0,
            "measured_tokens_per_second": measured_tps,
            "speed_of_light_tokens_per_second": speed_of_light_tps,
            "error": "missing measured throughput or speed-of-light throughput",
        }
    delta = max(0.0, min(1.0, measured_tps / speed_of_light_tps))
    return {
        "passed": delta >= 1.0,
        "score": delta,
        "delta": delta,
        "measured_tokens_per_second": measured_tps,
        "speed_of_light_tokens_per_second": speed_of_light_tps,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measured", required=True, type=Path, help="Measured throughput JSON or log")
    parser.add_argument("--speed-of-light", required=True, type=Path, help="Speed-of-light JSON or log")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--print-json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    measured, measured_detail = extract_measured(args.measured)
    limit, limit_detail = extract_limit(args.speed_of_light)
    result = score_performance(measured, limit)
    result["measured_detail"] = measured_detail
    result["speed_of_light_detail"] = limit_detail
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    if args.print_json:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

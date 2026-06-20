"""Compute performance score against speed-of-light throughput."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


THROUGHPUT_PATTERNS = (
    re.compile(r"Throughput\s+([0-9]+(?:\.[0-9]+)?)\s+new\b", re.I),
    re.compile(
        r"Throughput\s+([0-9]+(?:\.[0-9]+)?)\s+(?:new\s+)?(?:tok/s|tokens/s)", re.I
    ),
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

WORKLOAD_KEYS = (
    "workload",
    "workload_type",
    "benchmark_category",
    "category",
)

WORKLOAD_COLLECTION_KEYS = (
    "targets",
    "workloads",
    "benchmarks",
    "performance_targets",
    "results",
)

SCORING_WORKLOADS = {
    "independent_decode",
    "long_prefill",
}

WORKLOAD_ALIASES = {
    "decode": "independent_decode",
    "generation": "independent_decode",
    "independent_generation": "independent_decode",
    "independent-generate": "independent_decode",
    "independent-generate-text": "independent_decode",
    "long_prompt": "long_prefill",
    "prefill": "long_prefill",
    "long-prompt-prefill": "long_prefill",
    "prefix_reuse": "prefix_cache_reuse",
    "prefix-cache": "prefix_cache_reuse",
    "prefix-cache-reuse": "prefix_cache_reuse",
}


def normalize_workload(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return WORKLOAD_ALIASES.get(normalized, normalized)


def is_scoring_workload(value: Any) -> bool:
    workload = normalize_workload(value)
    return workload is None or workload in SCORING_WORKLOADS


def _payload_workload(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in WORKLOAD_KEYS:
        workload = normalize_workload(payload.get(key))
        if workload is not None:
            return workload
    return None


def _select_workload_payload(
    payload: Any,
    workload: str | None,
) -> tuple[Any, str | None, dict[str, Any]]:
    required = normalize_workload(workload)
    detail: dict[str, Any] = {}
    if required is not None:
        detail["required_workload"] = required
    if required is None or not isinstance(payload, dict):
        return payload, _payload_workload(payload), detail

    direct = _payload_workload(payload)
    if direct is not None:
        detail["workload"] = direct
        if direct == required:
            return payload, direct, detail
        detail["error"] = (
            f"payload workload {direct!r} does not match required workload "
            f"{required!r}"
        )
        return None, direct, detail

    collections_seen: list[str] = []
    available_workloads: set[str] = set()
    for key in WORKLOAD_COLLECTION_KEYS:
        collection = payload.get(key)
        if isinstance(collection, dict):
            collections_seen.append(key)
            for name, value in collection.items():
                name_workload = normalize_workload(name)
                if name_workload is not None:
                    available_workloads.add(name_workload)
                if name_workload == required:
                    detail["collection"] = key
                    detail["workload"] = required
                    return value, required, detail
        if isinstance(collection, list):
            collections_seen.append(key)
            for value in collection:
                value_workload = _payload_workload(value)
                if value_workload is not None:
                    available_workloads.add(value_workload)
                if isinstance(value, dict) and value_workload == required:
                    detail["collection"] = key
                    detail["workload"] = required
                    return value, required, detail

    if collections_seen:
        detail["collections"] = collections_seen
        detail["available_workloads"] = sorted(available_workloads)
        detail["error"] = (
            f"required workload {required!r} was not found in structured "
            f"workload collections"
        )
        return None, None, detail

    return payload, None, detail


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


def extract_measured(
    path: Path,
    *,
    workload: str | None = None,
) -> tuple[float | None, dict[str, Any]]:
    payload = _load_payload(path)
    payload, measured_workload, detail = _select_workload_payload(payload, workload)
    if payload is None:
        return None, {"source": str(path), **detail}
    if isinstance(payload, dict) and isinstance(payload.get("throughputs"), list):
        values = [float(value) for value in payload["throughputs"]]
        if values:
            return max(values), {
                "source": str(path),
                "workload": measured_workload,
                "values": values,
                "method": "log_regex_preferred",
                **detail,
            }
    numeric = _find_numeric(payload, MEASURED_KEYS)
    return numeric, {
        "source": str(path),
        "workload": measured_workload,
        "payload": payload,
        "method": "json_keys",
        **detail,
    }


def extract_limit(
    path: Path,
    *,
    workload: str | None = None,
) -> tuple[float | None, dict[str, Any]]:
    payload = _load_payload(path)
    payload, limit_workload, detail = _select_workload_payload(payload, workload)
    if payload is None:
        return None, {"source": str(path), **detail}
    numeric = _find_numeric(payload, LIMIT_KEYS)
    if (
        numeric is None
        and isinstance(payload, dict)
        and isinstance(payload.get("throughputs"), list)
    ):
        values = [float(value) for value in payload["throughputs"]]
        if values:
            numeric = max(values)
    return numeric, {
        "source": str(path),
        "workload": limit_workload,
        "payload": payload,
        **detail,
    }


def score_performance(
    measured_tps: float | None,
    speed_of_light_tps: float | None,
    *,
    measured_workload: str | None = None,
    required_workload: str | None = None,
) -> dict[str, Any]:
    required = normalize_workload(required_workload)
    measured = normalize_workload(measured_workload)
    workload = required or measured
    if measured is not None and required is not None and measured != required:
        return {
            "passed": False,
            "score": 0.0,
            "delta": 0.0,
            "workload": workload,
            "measured_workload": measured,
            "required_workload": required,
            "measured_tokens_per_second": measured_tps,
            "speed_of_light_tokens_per_second": speed_of_light_tps,
            "error": (
                f"measured workload {measured!r} does not match required "
                f"workload {required!r}"
            ),
        }
    if not is_scoring_workload(workload):
        return {
            "passed": False,
            "score": 0.0,
            "delta": 0.0,
            "workload": workload,
            "measured_workload": measured,
            "required_workload": required,
            "measured_tokens_per_second": measured_tps,
            "speed_of_light_tokens_per_second": speed_of_light_tps,
            "error": f"workload {workload!r} is diagnostic-only and cannot score delta",
        }
    if measured_tps is None or speed_of_light_tps is None or speed_of_light_tps <= 0:
        return {
            "passed": False,
            "score": 0.0,
            "delta": 0.0,
            "workload": workload,
            "measured_workload": measured,
            "required_workload": required,
            "measured_tokens_per_second": measured_tps,
            "speed_of_light_tokens_per_second": speed_of_light_tps,
            "error": "missing measured throughput or speed-of-light throughput",
        }
    delta = max(0.0, min(1.0, measured_tps / speed_of_light_tps))
    return {
        "passed": delta >= 1.0,
        "score": delta,
        "delta": delta,
        "workload": workload,
        "measured_workload": measured,
        "required_workload": required,
        "measured_tokens_per_second": measured_tps,
        "speed_of_light_tokens_per_second": speed_of_light_tps,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--measured", required=True, type=Path, help="Measured throughput JSON or log"
    )
    parser.add_argument(
        "--speed-of-light", required=True, type=Path, help="Speed-of-light JSON or log"
    )
    parser.add_argument(
        "--workload",
        help="Required workload category, e.g. independent_decode or long_prefill",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--print-json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    measured, measured_detail = extract_measured(args.measured, workload=args.workload)
    limit, limit_detail = extract_limit(args.speed_of_light, workload=args.workload)
    result = score_performance(
        measured,
        limit,
        measured_workload=measured_detail.get("workload"),
        required_workload=args.workload,
    )
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

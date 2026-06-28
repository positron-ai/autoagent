"""Shared Ares AutoAgent gate helpers."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_payload(path: Path) -> Any:
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def run_command(
    command: str,
    *,
    cwd: Path,
    log: Path,
    timeout: int,
    env: Mapping[str, str] | None = None,
) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as out:
        out.write("+ " + command + "\n")
        out.flush()
        proc = subprocess.run(
            ["sh", "-lc", command],
            cwd=cwd,
            env=dict(env) if env is not None else None,
            stdout=out,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    return proc.returncode


def _float_from_match(match: re.Match[str]) -> float:
    if match.groupdict():
        for value in match.groupdict().values():
            if value is not None:
                return float(value)
    for value in match.groups():
        if value is not None:
            return float(value)
    return float(match.group(0))


def _aggregate(values: list[float], mode: str) -> float:
    if mode == "min":
        return min(values)
    if mode == "mean":
        return sum(values) / len(values)
    if mode == "first":
        return values[0]
    if mode == "last":
        return values[-1]
    return max(values)


def evaluate_command_gate(
    command_gate: dict[str, Any], *, rc: int, log: Path
) -> dict[str, Any]:
    text = log.read_text(errors="replace") if log.exists() else ""
    passed = rc == 0
    detail: dict[str, Any] = {
        "returncode": rc,
        "log": str(log),
        "checks": {
            "returncode": rc == 0,
            "fail_regexes": [],
            "missing_pass_regexes": [],
            "numeric_checks": [],
            "errors": [],
        },
    }

    for pattern in command_gate.get("fail_regexes", []):
        try:
            match = re.search(pattern, text, flags=re.MULTILINE)
        except re.error as exc:
            detail["checks"]["errors"].append({"pattern": pattern, "error": str(exc)})
            passed = False
            continue
        if match:
            detail["checks"]["fail_regexes"].append(
                {"pattern": pattern, "match": match.group(0)}
            )
            passed = False

    pass_results = []
    for pattern in command_gate.get("pass_regexes", []):
        try:
            match = re.search(pattern, text, flags=re.MULTILINE)
        except re.error as exc:
            detail["checks"]["errors"].append({"pattern": pattern, "error": str(exc)})
            passed = False
            continue
        found = match is not None
        pass_results.append(found)
        if not found:
            detail["checks"]["missing_pass_regexes"].append(pattern)

    if pass_results:
        if command_gate.get("pass_regex_mode", "all") == "any":
            passed = passed and any(pass_results)
        else:
            passed = passed and all(pass_results)

    for check in command_gate.get("numeric_checks", []):
        check_detail: dict[str, Any] = {
            "name": check.get("name", "numeric"),
            "regex": check["regex"],
            "aggregate": check.get("aggregate", "max"),
            "values": [],
            "passed": True,
        }
        try:
            values = [
                _float_from_match(match)
                for match in re.finditer(check["regex"], text, flags=re.MULTILINE)
            ]
        except (re.error, ValueError) as exc:
            check_detail["error"] = str(exc)
            check_detail["passed"] = False
            detail["checks"]["numeric_checks"].append(check_detail)
            passed = False
            continue

        check_detail["values"] = values
        if not values:
            check_detail["passed"] = not bool(check.get("required", True))
            if not check_detail["passed"]:
                check_detail["error"] = "no matches"
                passed = False
            detail["checks"]["numeric_checks"].append(check_detail)
            continue

        value = _aggregate(values, check_detail["aggregate"])
        check_detail["value"] = value
        if "min" in check:
            check_detail["min"] = float(check["min"])
            check_detail["passed"] = check_detail["passed"] and value >= float(
                check["min"]
            )
        if "max" in check:
            check_detail["max"] = float(check["max"])
            check_detail["passed"] = check_detail["passed"] and value <= float(
                check["max"]
            )
        if not check_detail["passed"]:
            passed = False
        detail["checks"]["numeric_checks"].append(check_detail)

    detail["passed"] = passed
    detail["score"] = 1.0 if passed else 0.0
    return detail


def file_gate(path: Path, *, label: str, require_json: bool = False) -> dict[str, Any]:
    detail: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    passed = path.is_file()
    if passed and require_json:
        try:
            json.loads(path.read_text())
            detail["json"] = True
        except json.JSONDecodeError as exc:
            detail["json"] = False
            detail["error"] = str(exc)
            passed = False
    detail["passed"] = passed
    detail["score"] = 1.0 if passed else 0.0
    detail["label"] = label
    return detail

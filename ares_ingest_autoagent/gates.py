"""Shared Ares AutoAgent gate helpers."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

SHORTCUT_SCAN_IGNORED_DIRS = frozenset(
    {
        ".autoagent",
        ".git",
        ".hg",
        ".lake",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "logs",
        "logs-rinzler-side-by-side",
        "node_modules",
        "target",
        "third_party",
        "venv",
    }
)

RUNTIME_SIDECAR_MARKERS = (
    "generated-plan-runtime-shortcut",
    "generated_plan_runtime_shortcut",
    "runtime-generated",
    "runtime_generated",
    "runtime-created",
    "runtime_created",
    "runtime-sidecar",
    "runtime_sidecar",
    "shortcut-plan",
    "shortcut_plan",
    "static-runtime-sidecar",
    "static_runtime_sidecar",
)

FORBIDDEN_MODEL_PLUGIN_DIRS = frozenset(
    {
        "hand-authored-models",
        "hand_authored_models",
        "model-plugins",
        "model_plugins",
        "runtime-model-plugins",
        "runtime_model_plugins",
    }
)

MODEL_FAMILY_PLUGIN_STEMS = (
    "gemma",
    "glm",
    "gpt",
    "granite",
    "llama",
    "mistral",
    "mixtral",
    "phi",
    "qwen",
)


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


def shortcut_scan_gate(
    ares_repo: Path,
    *,
    label: str = "shortcut/static runtime sidecar scan",
) -> dict[str, Any]:
    """Reject source-tree shortcuts that cannot count as Ares ingest evidence."""

    root = ares_repo.resolve()
    errors: list[str] = []
    hits: list[dict[str, str]] = []
    scanned_files = 0

    if not root.exists():
        errors.append("Ares repository root is missing")
    elif not root.is_dir():
        errors.append("Ares repository root is not a directory")
    else:
        for path in _iter_shortcut_scan_files(root):
            scanned_files += 1
            hits.extend(_shortcut_scan_hits(path, root))

    passed = not errors and not hits
    detail = {
        "root": str(root),
        "scanned_files": scanned_files,
        "ignored_directories": sorted(SHORTCUT_SCAN_IGNORED_DIRS),
        "forbidden_hits": hits,
        "checks": {
            "hand_authored_rust_model_plugins": not any(
                hit["kind"] == "hand_authored_rust_model_plugin" for hit in hits
            ),
            "runtime_generated_plan_sidecars": not any(
                hit["kind"] == "runtime_generated_plan_sidecar" for hit in hits
            ),
        },
    }
    gate: dict[str, Any] = {
        "label": label,
        "artifact_validator": "shortcut_scan",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "detail": detail,
    }
    if errors:
        gate["errors"] = errors
    if hits:
        gate["errors"] = [
            "forbidden shortcut/static-sidecar evidence was found in the Ares tree"
        ]
    return gate


def _iter_shortcut_scan_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for current, dirs, names in os.walk(root):
        dirs[:] = [
            name
            for name in dirs
            if name not in SHORTCUT_SCAN_IGNORED_DIRS and not name.endswith(".egg-info")
        ]
        current_path = Path(current)
        for name in names:
            path = current_path / name
            if path.is_file():
                files.append(path)
    return files


def _shortcut_scan_hits(path: Path, root: Path) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    rel = path.relative_to(root).as_posix()
    if reason := _forbidden_model_plugin_reason(path, root):
        hits.append(
            {
                "kind": "hand_authored_rust_model_plugin",
                "path": rel,
                "reason": reason,
            }
        )
    if marker := _runtime_sidecar_marker(path, root):
        hits.append(
            {
                "kind": "runtime_generated_plan_sidecar",
                "path": rel,
                "reason": f"path contains forbidden marker {marker!r}",
            }
        )
    return hits


def _forbidden_model_plugin_reason(path: Path, root: Path) -> str | None:
    if path.suffix != ".rs":
        return None
    rel_parts = tuple(part.lower() for part in path.relative_to(root).parts)
    if not rel_parts or rel_parts[0] not in {"backend", "runtime"}:
        return None
    if any(part in FORBIDDEN_MODEL_PLUGIN_DIRS for part in rel_parts):
        return "Rust source is under a forbidden model-plugin directory"
    if "plugins" not in rel_parts:
        return None
    stem = path.stem.lower()
    if any(
        stem == family or stem.startswith(f"{family}_")
        for family in MODEL_FAMILY_PLUGIN_STEMS
    ):
        return "Rust model-family plugin source is under a plugins directory"
    if any(stem.startswith(f"{family}-") for family in MODEL_FAMILY_PLUGIN_STEMS):
        return "Rust model-family plugin source is under a plugins directory"
    return None


def _runtime_sidecar_marker(path: Path, root: Path) -> str | None:
    if path.suffix not in {".json", ".jsonl"}:
        return None
    rel = path.relative_to(root).as_posix().lower()
    if not any(
        plan in rel for plan in ("aresplan", "ares_plan", "targetplan", "target_plan")
    ):
        return None
    for marker in RUNTIME_SIDECAR_MARKERS:
        if marker in rel:
            return marker
    return None

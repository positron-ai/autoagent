#!/usr/bin/env python3
"""Run the CPU-side Tron ingest verifier for a Harbor task."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from tron_ingest_autoagent.architecture import analyze_architecture
from tron_ingest_autoagent.performance import (
    extract_limit,
    extract_measured,
    score_performance,
)
from tron_ingest_autoagent.structure import analyze_patterns
from tron_ingest_autoagent.tokens import compare_payloads


def run(
    cmd: list[str],
    *,
    cwd: Path,
    log: Path,
    timeout: int,
    pythonpath_roots: list[Path] | None = None,
) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if os.environ.get("TRON_ALLOW_HF_TRANSFER") != "1":
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    if libstdcxx_path := env.get("TRON_INGEST_LIBSTDCXX_PATH"):
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            f"{libstdcxx_path}:{existing}" if existing else libstdcxx_path
        )
    if pythonpath_roots:
        existing = env.get("PYTHONPATH", "")
        roots = [str(path) for path in pythonpath_roots]
        env["PYTHONPATH"] = ":".join([*roots, existing] if existing else roots)
    with log.open("w") as f:
        f.write("+ " + " ".join(cmd) + "\n")
        f.flush()
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        return proc.returncode


def resolve_cwd(
    name: str, *, tron_repo: Path, ingest_dir: Path, work_dir: Path
) -> Path:
    if name == "tron_repo":
        return tron_repo
    if name == "ingest":
        return ingest_dir
    if name == "work_dir":
        return work_dir
    path = Path(name)
    if path.is_absolute():
        return path
    return tron_repo / path


def runtime_python(ingest_dir: Path) -> list[str]:
    if override := os.environ.get("TRON_INGEST_PYTHON"):
        return shlex.split(override)
    uv = os.environ.get("UV", "uv")
    return [
        uv,
        "run",
        "--frozen",
        "--project",
        str(ingest_dir / "runtime"),
        "--extra",
        "test",
        "python",
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _float_from_match(match: re.Match[str]) -> float:
    if match.groupdict():
        for value in match.groupdict().values():
            if value is not None:
                return float(value)
    if match.groups():
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

    pass_mode = command_gate.get("pass_regex_mode", "all")
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
        if pass_mode == "any":
            passed = passed and any(pass_results)
        else:
            passed = passed and all(pass_results)

    for check in command_gate.get("numeric_checks", []):
        name = check.get("name", "numeric")
        pattern = check["regex"]
        aggregate = check.get("aggregate", "max")
        required = bool(check.get("required", True))
        check_detail: dict[str, Any] = {
            "name": name,
            "regex": pattern,
            "aggregate": aggregate,
            "values": [],
            "passed": True,
        }
        try:
            matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
            values = [_float_from_match(match) for match in matches]
        except (re.error, ValueError) as exc:
            check_detail["error"] = str(exc)
            check_detail["passed"] = False
            detail["checks"]["numeric_checks"].append(check_detail)
            passed = False
            continue

        check_detail["values"] = values
        if not values:
            check_detail["passed"] = not required
            if required:
                check_detail["error"] = "no matches"
                passed = False
            detail["checks"]["numeric_checks"].append(check_detail)
            continue

        value = _aggregate(values, aggregate)
        check_detail["value"] = value
        if "min" in check and value < float(check["min"]):
            check_detail["passed"] = False
        if "max" in check and value > float(check["max"]):
            check_detail["passed"] = False
        if "min" in check:
            check_detail["min"] = float(check["min"])
        if "max" in check:
            check_detail["max"] = float(check["max"])
        if not check_detail["passed"]:
            passed = False
        detail["checks"]["numeric_checks"].append(check_detail)

    detail["passed"] = passed
    detail["score"] = 1.0 if passed else 0.0
    return detail


def read_payload(path: Path) -> Any:
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def resolve_path(
    path: str, *, task_files: Path, tron_repo: Path, work_dir: Path
) -> Path:
    raw = Path(path)
    if raw.is_absolute():
        return raw
    for base in (work_dir, task_files, tron_repo):
        candidate = base / raw
        if candidate.exists():
            return candidate
    return work_dir / raw


def main() -> int:
    task_files = Path(os.environ.get("TASK_FILES_DIR", "/task/files"))
    spec_path = Path(os.environ.get("MODEL_SPEC", task_files / "model_spec.json"))
    spec = json.loads(spec_path.read_text())

    tron_repo = Path(os.environ.get("TRON_REPO", "/tron"))
    ingest_dir = tron_repo / "ingest"
    work_dir = Path(spec.get("work_dir", "/tmp/tron-ingest"))
    logs_dir = Path(os.environ.get("VERIFIER_LOG_DIR", "/logs/verifier"))
    gates_path = work_dir / "gates.json"

    gates: dict[str, Any] = dict(spec.get("explicit_gates", {}))
    write_json(gates_path, {"gates": gates})

    if not ingest_dir.exists():
        gates["fx_export"] = {
            "passed": False,
            "score": 0.0,
            "error": f"missing {ingest_dir}",
        }
        write_json(gates_path, {"gates": gates})
        return 0

    py = runtime_python(ingest_dir)
    common = [
        "--model",
        spec["hf_model"],
        "--device",
        spec.get("device", "cpu"),
        "--dtype",
        spec.get("dtype", "float32"),
    ]

    if spec.get("run_typedfx", True):
        typedfx_dir = work_dir / "typedfx"
        rc = run(
            [
                *py,
                str(ingest_dir / "runtime/scripts/logit_equivalence_test.py"),
                *common,
                "--output-dir",
                str(typedfx_dir),
            ],
            cwd=ingest_dir,
            log=logs_dir / "typedfx.log",
            timeout=int(spec.get("typedfx_timeout_sec", 3600)),
            pythonpath_roots=[ingest_dir / "runtime"],
        )
        typedfx_results = typedfx_dir / "logit_results.json"
        gates["typedfx_logits"] = {
            "passed": rc == 0 and typedfx_results.exists(),
            "score": 1.0 if rc == 0 and typedfx_results.exists() else 0.0,
            "returncode": rc,
            "results": str(typedfx_results),
        }
        if typedfx_results.exists():
            gates.setdefault("fx_export", True)
            gates.setdefault("typedfx_parse", True)
        write_json(gates_path, {"gates": gates})

    if spec.get("run_bulk", True):
        bulk_dir = work_dir / "bulk"
        rc = run(
            [
                *py,
                str(ingest_dir / "runtime/scripts/bulk_logit_equivalence_test.py"),
                *common,
                "--output-dir",
                str(bulk_dir),
            ],
            cwd=ingest_dir,
            log=logs_dir / "bulk.log",
            timeout=int(spec.get("bulk_timeout_sec", 3600)),
            pythonpath_roots=[ingest_dir / "runtime"],
        )
        bulk_results = bulk_dir / "bulk_logit_results.json"
        gates["bulk_logits"] = {
            "passed": rc == 0 and bulk_results.exists(),
            "score": 1.0 if rc == 0 and bulk_results.exists() else 0.0,
            "returncode": rc,
            "results": str(bulk_results),
        }
        write_json(gates_path, {"gates": gates})

    architecture_path = work_dir / "architecture.json"
    if work_dir.exists():
        architecture = analyze_architecture(
            work_dir,
            expected=spec.get("expected_architecture"),
        )
        write_json(architecture_path, architecture)
        gates["architecture"] = {
            "passed": architecture["passed"],
            "score": architecture["score"],
            "results": str(architecture_path),
        }
        write_json(gates_path, {"gates": gates})

    structure_path = work_dir / "eqsat_structure.json"
    artifact_roots = [
        path
        for path in (
            work_dir / "typedfx",
            work_dir / "bulk",
        )
        if path.exists()
    ]
    if artifact_roots:
        structure = analyze_patterns(
            roots=artifact_roots,
            expected_patterns=spec.get("expected_eqsat_patterns"),
        )
        write_json(structure_path, structure)
        gates["eqsat_structure"] = {
            "passed": structure["passed"],
            "score": structure["score"],
            "results": str(structure_path),
            "missing_patterns": structure["missing_patterns"],
        }
        write_json(gates_path, {"gates": gates})

    for command_gate in spec.get("command_gates", []):
        name = command_gate["name"]
        command = command_gate["command"]
        cwd = resolve_cwd(
            command_gate.get("cwd", "tron_repo"),
            tron_repo=tron_repo,
            ingest_dir=ingest_dir,
            work_dir=work_dir,
        )
        rc = run(
            ["sh", "-lc", command],
            cwd=cwd,
            log=logs_dir / f"{name}.log",
            timeout=int(command_gate.get("timeout_sec", 3600)),
        )
        gate = evaluate_command_gate(command_gate, rc=rc, log=logs_dir / f"{name}.log")
        gate["command"] = command
        gate["cwd"] = str(cwd)
        gates[name] = gate
        write_json(gates_path, {"gates": gates})

    token_results_path = None
    if token_spec := spec.get("token_comparison"):
        reference = resolve_path(
            token_spec["reference"],
            task_files=task_files,
            tron_repo=tron_repo,
            work_dir=work_dir,
        )
        candidate = resolve_path(
            token_spec["candidate"],
            task_files=task_files,
            tron_repo=tron_repo,
            work_dir=work_dir,
        )
        token_results_path = work_dir / token_spec.get("output", "tokens.json")
        token_result = compare_payloads(
            read_payload(reference), read_payload(candidate)
        )
        token_result["reference"] = str(reference)
        token_result["candidate"] = str(candidate)
        write_json(token_results_path, token_result)

    performance_results_path = None
    if performance_spec := spec.get("performance_comparison"):
        measured_path = resolve_path(
            performance_spec["measured"],
            task_files=task_files,
            tron_repo=tron_repo,
            work_dir=work_dir,
        )
        speed_path = resolve_path(
            performance_spec["speed_of_light"],
            task_files=task_files,
            tron_repo=tron_repo,
            work_dir=work_dir,
        )
        workload = performance_spec.get("workload") or performance_spec.get(
            "required_workload"
        )
        measured, measured_detail = extract_measured(measured_path, workload=workload)
        limit, limit_detail = extract_limit(speed_path, workload=workload)
        performance_result = score_performance(
            measured,
            limit,
            measured_workload=measured_detail.get("workload"),
            required_workload=workload,
        )
        performance_result["measured_detail"] = measured_detail
        performance_result["speed_of_light_detail"] = limit_detail
        performance_results_path = work_dir / performance_spec.get(
            "output", "performance.json"
        )
        write_json(performance_results_path, performance_result)

    args = [
        sys.executable,
        "-m",
        "tron_ingest_autoagent.score",
        "--gates",
        str(gates_path),
        "--output-json",
        os.environ.get("REWARD_JSON", "/logs/verifier/reward.json"),
        "--output-txt",
        os.environ.get("REWARD_TXT", "/logs/reward.txt"),
    ]

    if structure_path.exists():
        args.extend(["--eqsat-structure", str(structure_path)])

    if architecture_path.exists():
        args.extend(["--architecture", str(architecture_path)])

    typedfx_results = work_dir / "typedfx/logit_results.json"
    if typedfx_results.exists():
        args.extend(["--typedfx-logits", str(typedfx_results)])

    bulk_results = work_dir / "bulk/bulk_logit_results.json"
    if bulk_results.exists():
        args.extend(["--bulk-logits", str(bulk_results)])

    if token_results_path is not None:
        args.extend(["--tokens", str(token_results_path)])
    elif token_json := spec.get("token_results_json"):
        args.extend(["--tokens", token_json])

    if performance_results_path is not None:
        args.extend(["--performance", str(performance_results_path)])
    elif perf_json := spec.get("performance_results_json"):
        args.extend(["--performance", perf_json])

    required = spec.get("required_gates") or []
    if required:
        args.append("--required-gates")
        args.extend(required)

    subprocess.run(args, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

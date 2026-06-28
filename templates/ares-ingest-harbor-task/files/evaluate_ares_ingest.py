#!/usr/bin/env python3
"""Run the CPU-side Ares ingest verifier for a Harbor task."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from ares_ingest_autoagent.artifacts import ares_plan_gate, target_plan_gate
from ares_ingest_autoagent.gates import (
    evaluate_command_gate,
    read_payload,
    run_command,
    write_json,
)
from tron_ingest_autoagent.performance import (
    extract_limit,
    extract_measured,
    score_performance,
)
from tron_ingest_autoagent.tokens import compare_payloads


def resolve_cwd(
    name: str, *, ares_repo: Path, work_dir: Path, task_files: Path
) -> Path:
    if name == "ares_repo":
        return ares_repo
    if name == "work_dir":
        return work_dir
    if name == "task_files":
        return task_files
    path = Path(name)
    if path.is_absolute():
        return path
    return ares_repo / path


def resolve_path(
    path: str, *, task_files: Path, ares_repo: Path, work_dir: Path
) -> Path:
    raw = Path(path)
    if raw.is_absolute():
        return raw
    for base in (work_dir, task_files, ares_repo):
        candidate = base / raw
        if candidate.exists():
            return candidate
    return work_dir / raw


def read_json_or_jsonl(path: Path) -> Any:
    text = path.read_text()
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def main() -> int:
    task_files = Path(os.environ.get("TASK_FILES_DIR", "/task/files"))
    spec_path = Path(os.environ.get("MODEL_SPEC", task_files / "model_spec.json"))
    spec = json.loads(spec_path.read_text())

    ares_repo = Path(os.environ.get("ARES_REPO", "/ares"))
    work_dir = Path(spec.get("work_dir", "/tmp/ares-ingest"))
    logs_dir = Path(os.environ.get("VERIFIER_LOG_DIR", "/logs/verifier"))
    reward_json = os.environ.get("REWARD_JSON", "/logs/verifier/reward.json")
    reward_txt = os.environ.get("REWARD_TXT", "/logs/reward.txt")
    gates_path = work_dir / "gates.json"
    work_dir.mkdir(parents=True, exist_ok=True)

    gates: dict[str, Any] = dict(spec.get("explicit_gates", {}))
    write_json(gates_path, {"gates": gates})

    if oracle_spec := spec.get("oracle_records"):
        oracle_path = resolve_path(
            oracle_spec, task_files=task_files, ares_repo=ares_repo, work_dir=work_dir
        )
        oracle_payload = read_json_or_jsonl(oracle_path)
        oracle_summary = work_dir / "oracle.json"
        write_json(oracle_summary, oracle_payload)
    else:
        oracle_summary = None

    if ares_plan := spec.get("ares_plan"):
        gates["aresplan_valid"] = ares_plan_gate(
            resolve_path(
                ares_plan, task_files=task_files, ares_repo=ares_repo, work_dir=work_dir
            )
        )

    if target_plan := spec.get("target_plan"):
        gates["targetplan_valid"] = target_plan_gate(
            resolve_path(
                target_plan,
                task_files=task_files,
                ares_repo=ares_repo,
                work_dir=work_dir,
            )
        )

    for command_gate in spec.get("command_gates", []):
        name = command_gate["name"]
        command = command_gate["command"]
        cwd = resolve_cwd(
            command_gate.get("cwd", "ares_repo"),
            ares_repo=ares_repo,
            work_dir=work_dir,
            task_files=task_files,
        )
        log = logs_dir / f"{name}.log"
        rc = run_command(
            command,
            cwd=cwd,
            log=log,
            timeout=int(command_gate.get("timeout_sec", 3600)),
        )
        gate = evaluate_command_gate(command_gate, rc=rc, log=log)
        gate["command"] = command
        gate["cwd"] = str(cwd)
        gates[name] = gate
        write_json(gates_path, {"gates": gates})

    token_results_path = None
    if token_spec := spec.get("token_comparison"):
        reference = resolve_path(
            token_spec["reference"],
            task_files=task_files,
            ares_repo=ares_repo,
            work_dir=work_dir,
        )
        candidate = resolve_path(
            token_spec["candidate"],
            task_files=task_files,
            ares_repo=ares_repo,
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
            ares_repo=ares_repo,
            work_dir=work_dir,
        )
        speed_path = resolve_path(
            performance_spec["speed_of_light"],
            task_files=task_files,
            ares_repo=ares_repo,
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

    write_json(gates_path, {"gates": gates})

    args = [
        sys.executable,
        "-m",
        "ares_ingest_autoagent.score",
        "--gates",
        str(gates_path),
        "--output-json",
        reward_json,
        "--output-txt",
        reward_txt,
    ]
    if oracle_summary is not None:
        args.extend(["--oracle", str(oracle_summary)])
    if one_token := spec.get("one_token_results_json"):
        args.extend(
            [
                "--one-token",
                str(
                    resolve_path(
                        one_token,
                        task_files=task_files,
                        ares_repo=ares_repo,
                        work_dir=work_dir,
                    )
                ),
            ]
        )
    if token_results_path is not None:
        args.extend(["--tokens", str(token_results_path)])
    if performance_results_path is not None:
        args.extend(["--performance", str(performance_results_path)])
    if required := spec.get("required_gates"):
        args.append("--required-gates")
        args.extend(required)

    subprocess.run(args, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

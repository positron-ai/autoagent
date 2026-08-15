#!/usr/bin/env python3
"""Run the CPU-side Ares ingest verifier from Harbor's trusted tests tree."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from ares_ingest_autoagent.artifacts import (
    ares_plan_gate,
    artifact_consistency_gate,
    backend_open_gate,
    build_greedy_token_evidence,
    cpp_tvd_gate,
    depth_performance_gate,
    load_hf_cpu_oracle_evidence_files,
    load_hf_cpu_oracle_authority_file,
    mmlu_pro_gate,
    one_token_logits_gate,
    target_plan_gate,
    token_agreement_gate,
)
from ares_ingest_autoagent.commands import build_command_wrapper_plan
from ares_ingest_autoagent.gates import (
    evaluate_command_gate,
    read_payload,
    run_command,
    shortcut_scan_gate,
    write_json,
)
from ares_ingest_autoagent.score import compute_reward
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


def mmlu_pro_expectations(spec: Mapping[str, Any]) -> dict[str, Any]:
    mmlu_cfg = spec.get("mmlu_pro")
    mmlu_cfg = mmlu_cfg if isinstance(mmlu_cfg, Mapping) else {}
    expected_model = mmlu_cfg.get("model") or spec.get("mmlu_model") or spec.get("model")
    coverage = (
        mmlu_cfg.get("required_coverage_percent")
        or mmlu_cfg.get("coverage_percent")
        or mmlu_cfg.get("coverage")
        or spec.get("mmlu_required_coverage_percent")
        or spec.get("mmlu_coverage")
    )
    return {
        "expected_model": expected_model if isinstance(expected_model, str) else None,
        "expected_backend": spec.get("backend")
        if isinstance(spec.get("backend"), str)
        else None,
        "required_coverage_percent": float(coverage)
        if isinstance(coverage, int | float)
        else None,
    }


def generated_payload(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    if isinstance(payload.get("generated_token_ids"), list):
        return {"tokens": payload["generated_token_ids"]}
    generation = payload.get("generation")
    if isinstance(generation, dict) and isinstance(
        generation.get("generated_token_ids"), list
    ):
        return {"tokens": generation["generated_token_ids"]}
    return payload


def main() -> int:
    task_files = Path(os.environ.get("TASK_FILES_DIR", "/task/files"))
    authority_file = os.environ.get("ORACLE_AUTHORITY_FILE")
    oracle_authority = (
        load_hf_cpu_oracle_authority_file(Path(authority_file))
        if authority_file
        else None
    )
    spec_path = Path(os.environ.get("MODEL_SPEC", task_files / "model_spec.json"))
    spec = json.loads(spec_path.read_text())

    ares_repo = Path(os.environ.get("ARES_REPO", "/ares"))
    work_dir = Path(spec.get("work_dir", "/tmp/ares-ingest"))
    logs_dir = Path(os.environ.get("VERIFIER_LOG_DIR", "/logs/verifier"))
    reward_json = Path(os.environ.get("REWARD_JSON", "/logs/verifier/reward.json"))
    reward_txt = Path(os.environ.get("REWARD_TXT", "/logs/reward.txt"))
    gates_path = work_dir / "gates.json"
    validated_gates_path = work_dir / "validated_gates.json"
    work_dir.mkdir(parents=True, exist_ok=True)

    gates: dict[str, Any] = dict(spec.get("explicit_gates", {}))
    validated_gates: dict[str, Any] = {}
    write_json(gates_path, {"gates": gates})

    required_gates = spec.get("required_gates") or []
    command_wrapper_plan = build_command_wrapper_plan(
        spec,
        run_dir=work_dir,
        ares_repo=ares_repo,
    )
    if command_wrapper_plan["wrappers"]:
        write_json(work_dir / "command_wrappers.json", command_wrapper_plan)
    command_gates = list(spec.get("command_gates", []))
    if command_wrapper_plan["command_gates"]:
        command_gates.extend(command_wrapper_plan["command_gates"])

    # Commands may create or mutate candidate artifacts, so every promotion-
    # bearing validator runs only after the final command has completed.
    for command_gate in command_gates:
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

    if "shortcut_scan" in required_gates or spec.get("shortcut_scan"):
        validated_gates["shortcut_scan"] = shortcut_scan_gate(ares_repo)
        gates["shortcut_scan"] = validated_gates["shortcut_scan"]
        write_json(gates_path, {"gates": gates})

    if oracle_spec := spec.get("oracle_records"):
        oracle_path = resolve_path(
            oracle_spec, task_files=task_files, ares_repo=ares_repo, work_dir=work_dir
        )
    else:
        oracle_path = None
    if oracle_dense_spec := spec.get("oracle_dense_logits"):
        oracle_dense_path = resolve_path(
            oracle_dense_spec,
            task_files=task_files,
            ares_repo=ares_repo,
            work_dir=work_dir,
        )
    else:
        oracle_dense_path = None
    oracle_evidence = (
        load_hf_cpu_oracle_evidence_files(
            oracle_path,
            oracle_dense_path,
            expected_authority=oracle_authority,
        )
        if oracle_path is not None and oracle_dense_path is not None
        else None
    )

    ares_plan_path = None
    if ares_plan := spec.get("ares_plan"):
        ares_plan_path = resolve_path(
            ares_plan, task_files=task_files, ares_repo=ares_repo, work_dir=work_dir
        )
        validated_gates["aresplan_valid"] = ares_plan_gate(
            ares_plan_path
        )
        gates["aresplan_valid"] = validated_gates["aresplan_valid"]

    target_plan_path = None
    if target_plan := spec.get("target_plan"):
        target_plan_path = resolve_path(
            target_plan,
            task_files=task_files,
            ares_repo=ares_repo,
            work_dir=work_dir,
        )
        validated_gates["targetplan_valid"] = target_plan_gate(
            target_plan_path
        )
        gates["targetplan_valid"] = validated_gates["targetplan_valid"]

    if "artifact_consistency" in required_gates:
        validated_gates["artifact_consistency"] = artifact_consistency_gate(
            spec,
            oracle_payload=(
                oracle_evidence.oracle_records
                if oracle_evidence is not None
                else None
            ),
            oracle_transaction_digest=(
                oracle_evidence.validation.detail.get("transaction_digest")
                if oracle_evidence is not None and oracle_evidence.validation.passed
                else None
            ),
            validated_gates=validated_gates,
        )
        gates["artifact_consistency"] = validated_gates["artifact_consistency"]

    if backend_open := spec.get("backend_open_evidence"):
        validated_gates["backend_open"] = backend_open_gate(
            resolve_path(
                backend_open,
                task_files=task_files,
                ares_repo=ares_repo,
                work_dir=work_dir,
            )
        )
        gates["backend_open"] = validated_gates["backend_open"]

    one_token_results_path = None
    if one_token := spec.get("one_token_logits_evidence") or spec.get(
        "one_token_results_json"
    ):
        validated_gates["one_token_logits"] = one_token_logits_gate(
            resolve_path(
                one_token,
                task_files=task_files,
                ares_repo=ares_repo,
                work_dir=work_dir,
            )
        )
        gates["one_token_logits"] = validated_gates["one_token_logits"]
        one_token_results_path = work_dir / "one-token-logits.json"
        write_json(one_token_results_path, gates["one_token_logits"])

    if cpp_tvd := spec.get("cpp_tvd_evidence"):
        validated_gates["cpp_tvd"] = cpp_tvd_gate(
            resolve_path(
                cpp_tvd,
                task_files=task_files,
                ares_repo=ares_repo,
                work_dir=work_dir,
            )
        )
        gates["cpp_tvd"] = validated_gates["cpp_tvd"]

    if depth_performance := spec.get("depth_performance_evidence"):
        validated_gates["depth_performance"] = depth_performance_gate(
            resolve_path(
                depth_performance,
                task_files=task_files,
                ares_repo=ares_repo,
                work_dir=work_dir,
            )
        )
        gates["depth_performance"] = validated_gates["depth_performance"]

    if mmlu_pro := spec.get("mmlu_pro_evidence"):
        validated_gates["mmlu_pro"] = mmlu_pro_gate(
            resolve_path(
                mmlu_pro,
                task_files=task_files,
                ares_repo=ares_repo,
                work_dir=work_dir,
            ),
            **mmlu_pro_expectations(spec),
        )
        gates["mmlu_pro"] = validated_gates["mmlu_pro"]

    token_results_path = None
    token_payload = None
    if eight_token := spec.get("eight_token_greedy_evidence") or spec.get(
        "token_results_json"
    ):
        token_results_path = resolve_path(
            eight_token,
            task_files=task_files,
            ares_repo=ares_repo,
            work_dir=work_dir,
        )
        validated_gates["eight_token_greedy"] = token_agreement_gate(token_results_path)
        gates["eight_token_greedy"] = validated_gates["eight_token_greedy"]
        token_payload = read_payload(token_results_path)

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
        reference_payload = read_payload(reference)
        candidate_payload = read_payload(candidate)
        token_result = compare_payloads(
            generated_payload(reference_payload),
            generated_payload(candidate_payload),
        )
        token_evidence = build_greedy_token_evidence(
            token_result,
            reference=reference,
            candidate=candidate,
            reference_payload=reference_payload,
            candidate_payload=candidate_payload,
            expected_generated_tokens=int(
                token_spec.get("expected_generated_tokens", 8)
            ),
        )
        token_payload = token_evidence
        write_json(token_results_path, token_evidence)
        validated_gates["eight_token_greedy"] = token_agreement_gate(token_results_path)
        gates["eight_token_greedy"] = validated_gates["eight_token_greedy"]

    performance_payload = None
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
        performance_payload = performance_result

    write_json(gates_path, {"gates": gates})
    write_json(validated_gates_path, {"gates": validated_gates})

    # The JSON files above are diagnostic outputs. Promotion authority remains
    # in this process so a candidate cannot replace a validator receipt between
    # validation and reward computation.
    reward = compute_reward(
        gates_payload={"gates": gates},
        validated_gates_payload={"gates": validated_gates},
        oracle_path=oracle_path,
        oracle_dense_path=oracle_dense_path,
        artifact_spec=spec,
        ares_plan_path=ares_plan_path,
        target_plan_path=target_plan_path,
        authority_root=ares_repo,
        expected_oracle_authority=oracle_authority,
        token_payload=token_payload,
        token_payload_base_dir=(
            token_results_path.parent if token_results_path is not None else None
        ),
        performance_payload=performance_payload,
        required_gates=tuple(required_gates),
        promotion_authority=True,
    )
    reward_json.parent.mkdir(parents=True, exist_ok=True)
    reward_json.write_text(json.dumps(reward, indent=2) + "\n")
    reward_txt.parent.mkdir(parents=True, exist_ok=True)
    reward_txt.write_text(f"{reward['score']:.12g}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

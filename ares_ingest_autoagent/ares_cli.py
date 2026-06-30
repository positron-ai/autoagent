"""Create and manage Ares AutoAgent model-ingest run directories."""

from __future__ import annotations

import argparse
import json
import os
import re
import selectors
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ares_ingest_autoagent.artifacts import (
    ares_plan_gate,
    artifact_consistency_gate,
    backend_open_gate,
    build_greedy_token_evidence,
    cpp_tvd_gate,
    depth_performance_gate,
    introspection_ladder_gate,
    mmlu_pro_gate,
    one_token_logits_gate,
    target_plan_gate,
    token_agreement_gate,
    trace_report_gate,
)
from ares_ingest_autoagent.commands import build_command_wrapper_plan
from ares_ingest_autoagent.gates import read_payload, shortcut_scan_gate, write_json
from ares_ingest_autoagent.score import (
    GATE_PROFILES,
    compute_reward,
    required_gates_for_profile,
)
from tron_ingest_autoagent.tokens import compare_payloads


DEFAULT_REFINER_COMMAND = (
    "codex exec --dangerously-bypass-approvals-and-sandbox "
    '-C "$ARES_REPO" --add-dir "$AUTOAGENT_REPO" - < "$REFINEMENT_PROMPT"'
)
CLAUDE_REFINER_COMMAND = 'claude -p "$(cat "$REFINEMENT_PROMPT")"'
DRIVER_COMMANDS = {
    "codex": DEFAULT_REFINER_COMMAND,
    "claude": CLAUDE_REFINER_COMMAND,
    "custom": None,
}


@dataclass
class AresIngestConfig:
    model_slug: str
    safe_model: str
    ares_repo: Path
    autoagent_root: Path
    run_dir: Path
    logs_dir: Path
    state_path: Path
    model_spec_path: Path
    target_score: float
    max_iterations: int | None
    stall_patience: int
    min_improvement: float
    refinement_command: str | None
    driver: str
    gate_profile: str
    setup_only: bool
    cockpit: bool
    stream_refiner_output: bool


class AresIngestError(RuntimeError):
    """Expected Ares AutoAgent command failure."""


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower())
    slug = re.sub(r"-+", "-", slug).strip("-")
    if not slug:
        raise ValueError(f"cannot derive safe model name from {text!r}")
    return slug


def default_run_dir(ares_repo: Path, safe_model: str) -> Path:
    return ares_repo / ".autoagent" / "ares-ingest" / safe_model / utc_stamp()


def build_model_spec(cfg: AresIngestConfig) -> dict[str, Any]:
    return {
        "model": cfg.model_slug,
        "expected_model_ids": [cfg.model_slug],
        "safe_model": cfg.safe_model,
        "ares_repo": str(cfg.ares_repo),
        "frontend": "fx",
        "backend": "tron",
        "gate_profile": cfg.gate_profile,
        "required_gates": list(required_gates_for_profile(cfg.gate_profile)),
        "explicit_gates": {
            "model_spec": {
                "passed": True,
                "score": 1.0,
                "source": "ares-ingest-agent setup",
            }
        },
        "oracle": {
            "required_source": "huggingface_transformers_pytorch_cpu",
            "required_record_kind": "hf_cpu_oracle_capture",
        },
        "policy": {
            "no_hand_authored_rust_model_plugins": True,
            "no_runtime_generated_execution_sidecars": True,
            "runtime_path": "frontend -> Lean ingest -> AresPlan -> TargetPlan -> backend provider",
        },
    }


def reward_fingerprint(reward: dict[str, Any]) -> dict[str, Any]:
    gates = reward.get("gates", {})
    return {
        "score": reward.get("score"),
        "alpha_execution": reward.get("alpha_execution"),
        "tau_tokens": reward.get("tau_tokens"),
        "delta_inference": reward.get("delta_inference"),
        "stage_cap": reward.get("stage_cap"),
        "first_failed_gate": reward.get("first_failed_gate"),
        "gates": {
            name: {
                "passed": detail.get("passed"),
                "score": detail.get("score"),
            }
            for name, detail in gates.items()
            if isinstance(detail, dict)
        },
    }


def trace_report_summary_from_spec(spec: Mapping[str, Any]) -> dict[str, Any] | None:
    gates = spec.get("validated_gates")
    if not isinstance(gates, Mapping):
        gates = spec.get("explicit_gates")
    if not isinstance(gates, Mapping):
        return None
    gate = gates.get("trace_report")
    if not isinstance(gate, Mapping):
        return None
    detail = gate.get("detail")
    if not isinstance(detail, Mapping):
        detail = {}

    summary: dict[str, Any] = {
        "path": gate.get("path"),
        "sha256": detail.get("sha256"),
        "passed": gate.get("passed"),
        "artifact_validator": gate.get("artifact_validator"),
        "schema_version": detail.get("schema_version"),
        "title": detail.get("title"),
        "metadata": detail.get("metadata"),
        "trace": detail.get("trace"),
        "preflight_status": detail.get("preflight_status"),
        "report_grade": detail.get("report_grade"),
        "proof_grade_status": detail.get("proof_grade_status"),
        "answerability_count": detail.get("answerability_count"),
        "answerability_status_counts": detail.get("answerability_status_counts"),
        "unsupported_claim_count": detail.get("unsupported_claim_count"),
        "next_measurement_count": detail.get("next_measurement_count"),
        "report_json_section_count": detail.get("report_json_section_count"),
        "report_json_section_kind_counts": detail.get(
            "report_json_section_kind_counts"
        ),
        "trace_config_count": detail.get("trace_config_count"),
        "trace_config_status_counts": detail.get("trace_config_status_counts"),
        "introspection_capability_count": detail.get("introspection_capability_count"),
        "introspection_capability_status_counts": detail.get(
            "introspection_capability_status_counts"
        ),
        "introspection_artifact_summary_count": detail.get(
            "introspection_artifact_summary_count"
        ),
        "introspection_artifact_summary_status_counts": detail.get(
            "introspection_artifact_summary_status_counts"
        ),
        "next_measurement_samples": detail.get("next_measurement_samples"),
        "unsupported_claim_samples": detail.get("unsupported_claim_samples"),
        "analysis_command_samples": detail.get("analysis_command_samples"),
        "report_json_section_samples": detail.get("report_json_section_samples"),
        "trace_config_samples": detail.get("trace_config_samples"),
        "introspection_capability_samples": detail.get(
            "introspection_capability_samples"
        ),
        "introspection_artifact_summary_samples": detail.get(
            "introspection_artifact_summary_samples"
        ),
        "section_names": detail.get("section_names"),
    }
    errors = gate.get("errors")
    if isinstance(errors, list) and errors:
        summary["errors"] = errors
    return {
        key: value for key, value in summary.items() if value not in (None, "", [], {})
    }


def render_trace_report_lines(summary: Mapping[str, Any]) -> list[str]:
    lines = [
        f"- Path: `{summary.get('path', 'unknown')}`",
        f"- SHA-256: `{summary.get('sha256', 'unknown')}`",
        f"- Validator passed: `{summary.get('passed', 'unknown')}`",
    ]
    grade_parts = []
    for key in ("report_grade", "proof_grade_status", "preflight_status"):
        value = summary.get(key)
        if value:
            grade_parts.append(f"{key}={value}")
    if grade_parts:
        lines.append("- Trace grade: " + ", ".join(f"`{part}`" for part in grade_parts))
    if summary.get("answerability_status_counts"):
        lines.append(
            "- Answerability: "
            f"`{json.dumps(summary['answerability_status_counts'], sort_keys=True)}`"
        )
    if summary.get("report_json_section_kind_counts"):
        lines.append(
            "- Report JSON sections: "
            "`"
            + json.dumps(summary["report_json_section_kind_counts"], sort_keys=True)
            + "`"
        )
    if summary.get("trace_config_status_counts"):
        lines.append(
            "- Trace config: "
            f"`{json.dumps(summary['trace_config_status_counts'], sort_keys=True)}`"
        )
    if summary.get("introspection_capability_status_counts"):
        lines.append(
            "- Introspection capabilities: "
            "`"
            + json.dumps(
                summary["introspection_capability_status_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("introspection_artifact_summary_status_counts"):
        lines.append(
            "- Introspection artifacts: "
            "`"
            + json.dumps(
                summary["introspection_artifact_summary_status_counts"],
                sort_keys=True,
            )
            + "`"
        )
    for sample in summary.get("report_json_section_samples", [])[:6]:
        if not isinstance(sample, Mapping):
            continue
        json_path = sample.get("json_path") or sample.get("json_section")
        section_kind = sample.get("section_kind")
        boundary = sample.get("claim_boundary")
        if json_path:
            parts = []
            if section_kind:
                parts.append(f"kind={section_kind}")
            if boundary:
                parts.append(f"boundary={boundary}")
            line = f"- Report JSON section: {json_path}"
            if parts:
                line += " " + " ".join(parts)
            lines.append(line)
    for sample in summary.get("trace_config_samples", [])[:3]:
        if not isinstance(sample, Mapping):
            continue
        status = sample.get("config_status")
        requested = sample.get("requested_sidecar_controls")
        recorded = sample.get("recorded_sidecar_capabilities")
        level = sample.get("introspection_level")
        compile_feature = sample.get("compile_feature_trace_introspection")
        deep_effective = sample.get("deep_introspection_effective")
        next_action = sample.get("next_action")
        if status:
            parts = []
            if requested:
                parts.append(f"requested={requested}")
            if recorded:
                parts.append(f"recorded={recorded}")
            if level:
                parts.append(f"level={level}")
            if compile_feature:
                parts.append(f"compile_feature={compile_feature}")
            if deep_effective:
                parts.append(f"deep_effective={deep_effective}")
            line = f"- Trace config: {status}"
            if parts:
                line += " " + " ".join(parts)
            lines.append(line)
        if next_action:
            lines.append(f"  Next action: `{next_action}`")
    for sample in summary.get("introspection_capability_samples", [])[:3]:
        if not isinstance(sample, Mapping):
            continue
        capability = sample.get("capture_capability")
        status = sample.get("capability_status")
        artifact_count = sample.get("matching_artifact_count")
        boundary = sample.get("claim_boundary")
        next_action = sample.get("next_action")
        if capability:
            parts = [f"status={status}" if status else ""]
            if artifact_count:
                parts.append(f"artifacts={artifact_count}")
            if boundary:
                parts.append(f"boundary={boundary}")
            line = f"- Introspection capability: {capability}"
            if any(parts):
                line += " " + " ".join(part for part in parts if part)
            lines.append(line)
        if next_action:
            lines.append(f"  Next action: `{next_action}`")
    for sample in summary.get("introspection_artifact_summary_samples", [])[:3]:
        if not isinstance(sample, Mapping):
            continue
        artifact_kind = sample.get("artifact_kind")
        status = sample.get("summary_status")
        row_count = sample.get("row_count_total")
        sections = sample.get("report_sections")
        if artifact_kind:
            parts = [f"status={status}" if status else ""]
            if row_count:
                parts.append(f"rows={row_count}")
            if sections:
                parts.append(f"sections={sections}")
            line = f"- Introspection artifact: {artifact_kind}"
            if any(parts):
                line += " " + " ".join(part for part in parts if part)
            lines.append(line)
    for sample in summary.get("next_measurement_samples", [])[:3]:
        if not isinstance(sample, Mapping):
            continue
        measurement = sample.get("next_measurement") or sample.get("priority")
        reason = sample.get("reason")
        command = sample.get("command_hint")
        if measurement:
            suffix = f" - {reason}" if reason else ""
            lines.append(f"- Next measurement: {measurement}{suffix}")
        if command:
            lines.append(f"  Command hint: `{command}`")
    for sample in summary.get("unsupported_claim_samples", [])[:3]:
        if not isinstance(sample, Mapping):
            continue
        claim = sample.get("claim")
        reason = sample.get("reason")
        if claim:
            suffix = f" - {reason}" if reason else ""
            lines.append(f"- Unsupported claim: {claim}{suffix}")
    errors = summary.get("errors")
    if isinstance(errors, list) and errors:
        lines.append("- Validator errors: " + "; ".join(str(error) for error in errors))
    return lines


def trace_report_handoff_section(state: Mapping[str, Any] | None) -> str:
    if not isinstance(state, Mapping):
        return ""
    summary = state.get("trace_report")
    if not isinstance(summary, Mapping):
        return ""
    lines = ["## Trace Report", "", *render_trace_report_lines(summary), ""]
    return "\n".join(lines)


def trace_report_prompt_section(spec: Mapping[str, Any]) -> list[str]:
    summary = trace_report_summary_from_spec(spec)
    if not summary:
        return []
    return [
        "## Trace Report Summary",
        "",
        *render_trace_report_lines(summary),
        "",
        "Prefer the report's `sections.answerability`, `sections.unsupported_claims`,",
        "and `sections.next_measurements` before ad hoc log parsing. Use",
        "`sections.report_json_section_inventory` to discover available JSON",
        "sections. Then read `sections.trace_config_rows` first to distinguish",
        "requested controls from recorded sidecars, then use",
        "`sections.introspection_capability_rows` and",
        "`sections.introspection_artifact_summary_rows` to decide which tracing",
        "sidecars exist before opening heavier sidecar-specific sections.",
        "If the current trace cannot answer the failing gate, run the named",
        "next-measurement query or capture command before editing production code.",
        "",
    ]


def empty_steering() -> dict[str, Any]:
    return {
        "schema": "ares.autoagent.operator_steering.v1",
        "notes": [],
        "resources": [],
        "driver_commands": [],
    }


def steering_json_path(cfg: AresIngestConfig) -> Path:
    return cfg.run_dir / "steering.json"


def steering_markdown_path(cfg: AresIngestConfig) -> Path:
    return cfg.run_dir / "steering.md"


def load_steering(cfg: AresIngestConfig) -> dict[str, Any]:
    path = steering_json_path(cfg)
    if not path.exists():
        return empty_steering()
    payload = load_json(path)
    steering = empty_steering()
    for key in ("notes", "resources", "driver_commands"):
        if isinstance(payload.get(key), list):
            steering[key] = payload[key]
    return steering


def write_steering(cfg: AresIngestConfig, steering: Mapping[str, Any]) -> None:
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    write_json(steering_json_path(cfg), dict(steering))
    steering_markdown_path(cfg).write_text(render_steering_markdown(steering))


def ensure_steering_files(cfg: AresIngestConfig) -> None:
    if not steering_json_path(cfg).exists() or not steering_markdown_path(cfg).exists():
        write_steering(cfg, load_steering(cfg))


def render_steering_markdown(steering: Mapping[str, Any]) -> str:
    lines = ["# Ares AutoAgent Operator Steering", ""]
    notes = steering.get("notes", [])
    resources = steering.get("resources", [])
    drivers = steering.get("driver_commands", [])
    lines.append("## Notes")
    if isinstance(notes, list) and notes:
        for note in notes:
            lines.append(
                f"- iteration `{note.get('iteration', 'unknown')}` "
                f"{note.get('created_at', '')}: {note.get('text', '')}"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Resources"])
    if isinstance(resources, list) and resources:
        for resource in resources:
            label = resource.get("value", "")
            note = resource.get("note", "")
            suffix = f" - {note}" if note else ""
            lines.append(
                f"- iteration `{resource.get('iteration', 'unknown')}` "
                f"{resource.get('created_at', '')}: `{label}`{suffix}"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Driver Commands"])
    if isinstance(drivers, list) and drivers:
        for driver in drivers:
            lines.append(
                f"- iteration `{driver.get('iteration', 'unknown')}` "
                f"{driver.get('created_at', '')}: `{driver.get('command', '')}`"
            )
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def append_steering_note(
    cfg: AresIngestConfig, *, iteration: int, text: str
) -> dict[str, Any]:
    steering = load_steering(cfg)
    steering.setdefault("notes", [])
    steering["notes"].append(
        {
            "iteration": iteration,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "text": text,
        }
    )
    write_steering(cfg, steering)
    return steering


def append_steering_resource(
    cfg: AresIngestConfig,
    *,
    iteration: int,
    value: str,
    note: str = "",
) -> dict[str, Any]:
    steering = load_steering(cfg)
    steering.setdefault("resources", [])
    steering["resources"].append(
        {
            "iteration": iteration,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "value": value,
            "note": note,
        }
    )
    write_steering(cfg, steering)
    return steering


def append_driver_command(
    cfg: AresIngestConfig, *, iteration: int, command: str
) -> dict[str, Any]:
    steering = load_steering(cfg)
    steering.setdefault("driver_commands", [])
    steering["driver_commands"].append(
        {
            "iteration": iteration,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "driver": cfg.driver,
            "command": command,
        }
    )
    write_steering(cfg, steering)
    return steering


def steering_prompt_lines(cfg: AresIngestConfig) -> list[str]:
    steering = load_steering(cfg)
    lines = [
        f"- Steering JSON: `{steering_json_path(cfg)}`",
        f"- Steering notes: `{steering_markdown_path(cfg)}`",
    ]
    notes = steering.get("notes", [])
    resources = steering.get("resources", [])
    drivers = steering.get("driver_commands", [])
    if isinstance(notes, list) and notes:
        lines.append("- Operator notes:")
        for note in notes[-5:]:
            lines.append(
                f"  - iteration `{note.get('iteration', 'unknown')}`: {note.get('text', '')}"
            )
    else:
        lines.append("- Operator notes: none recorded.")
    if isinstance(resources, list) and resources:
        lines.append("- Operator resources:")
        for resource in resources[-5:]:
            suffix = f" - {resource.get('note', '')}" if resource.get("note") else ""
            lines.append(f"  - `{resource.get('value', '')}`{suffix}")
    else:
        lines.append("- Operator resources: none recorded.")
    if isinstance(drivers, list) and drivers:
        lines.append(f"- Latest driver override: `{drivers[-1].get('command', '')}`")
    return lines


def verifier_command(cfg: AresIngestConfig) -> str:
    return (
        "ares-ingest-agent "
        f"{cfg.model_slug} "
        f"--ares-repo {cfg.ares_repo} "
        f"--run-dir {cfg.run_dir} "
        "--no-refiner"
    )


def selected_workflow_skills(
    cfg: AresIngestConfig,
    *,
    first_failed_gate: str,
) -> list[dict[str, Any]]:
    verify = verifier_command(cfg)
    skills: list[dict[str, Any]] = [
        {
            "name": "command-wiggum",
            "gate": "all",
            "why": "Continue one gate at a time while maintaining durable run state.",
            "allowed_scope": [
                str(cfg.run_dir),
                str(cfg.model_spec_path),
                str(cfg.run_dir / "handoff.md"),
            ],
            "verification_commands": [verify],
        },
        {
            "name": "ares-evidence",
            "gate": "all",
            "why": "Classify oracle, system-under-test, comparison, and performance evidence correctly.",
            "allowed_scope": [
                str(cfg.run_dir / "artifacts"),
                str(cfg.run_dir / "reward.json"),
                str(cfg.model_spec_path),
            ],
            "verification_commands": [verify],
        },
    ]
    gate_skills: dict[str, dict[str, Any]] = {
        "model_spec": {
            "name": "ares-model-port",
            "gate": "model_spec",
            "why": (
                "Build or repair the model row inventory before artifact gates run, "
                "including prior-art inspection across HuggingFace Transformers plus "
                "resolved or cloned vLLM, llama.cpp, and MLX checkouts. Use explicit "
                "checkout paths when supplied; otherwise create/use "
                "${ARES_PRIOR_ART_ROOT:-$HOME/db} as the cache root and clone any "
                "missing official upstream repos there: "
                "https://github.com/vllm-project/vllm.git, "
                "https://github.com/ggml-org/llama.cpp.git, and "
                "https://github.com/ml-explore/mlx.git."
            ),
            "allowed_scope": [
                str(cfg.run_dir),
                str(cfg.model_spec_path),
                str(cfg.run_dir / "handoff.md"),
                "prior-art checkout paths recorded in model_spec.json or handoff.md",
                "${ARES_PRIOR_ART_ROOT:-$HOME/db}/vllm",
                "${ARES_PRIOR_ART_ROOT:-$HOME/db}/llama.cpp",
                "${ARES_PRIOR_ART_ROOT:-$HOME/db}/mlx",
            ],
            "verification_commands": [verify],
        },
        "hf_cpu_oracle": {
            "name": "ares-python",
            "gate": "hf_cpu_oracle",
            "why": "Capture or validate HF Transformers + PyTorch CPU oracle records.",
            "allowed_scope": [
                "tools/oracles/hf-cpu/",
                str(cfg.run_dir / "artifacts"),
                str(cfg.model_spec_path),
            ],
            "verification_commands": [
                "tools/oracles/hf-cpu/capture_hf_cpu_oracle.py validate-jsonl <oracle.jsonl>",
                verify,
            ],
        },
        "frontend_export": {
            "name": "ares-python",
            "gate": "frontend_export",
            "why": "Work on frontend export artifacts before Lean ingest.",
            "allowed_scope": [
                "frontend/hf-export/",
                str(cfg.run_dir / "artifacts"),
                str(cfg.model_spec_path),
            ],
            "verification_commands": [verify],
        },
        "lean_ingest": {
            "name": "ares-lean",
            "gate": "lean_ingest",
            "why": "Run Lean ingest on frontend artifacts without moving planning into Rust.",
            "allowed_scope": [
                "ingest/lean/",
                str(cfg.run_dir / "artifacts"),
                str(cfg.model_spec_path),
            ],
            "verification_commands": [verify],
        },
        "aresplan_valid": {
            "name": "ares-lean",
            "gate": "aresplan_valid",
            "why": "Produce and validate Lean-emitted AresPlan artifacts.",
            "allowed_scope": [
                "ingest/lean/",
                str(cfg.run_dir / "artifacts"),
                str(cfg.model_spec_path),
            ],
            "verification_commands": [verify],
        },
        "targetplan_valid": {
            "name": "ares-targetplan",
            "gate": "targetplan_valid",
            "why": "Coordinate Lean-emitted TargetPlan artifacts with Rust validation and runtime handoff.",
            "allowed_scope": [
                "ingest/lean/",
                "backend/",
                "runtime/",
                str(cfg.run_dir / "artifacts"),
                str(cfg.model_spec_path),
            ],
            "verification_commands": [verify],
        },
        "artifact_consistency": {
            "name": "ares-evidence",
            "gate": "artifact_consistency",
            "why": "Check that oracle and generated artifacts describe the same model row.",
            "allowed_scope": [
                str(cfg.run_dir / "artifacts"),
                str(cfg.model_spec_path),
            ],
            "verification_commands": [verify],
        },
        "shortcut_scan": {
            "name": "ares-rust",
            "gate": "shortcut_scan",
            "why": "Remove forbidden Rust model plugins or runtime-created plan sidecars.",
            "allowed_scope": [
                "runtime/",
                "backend/",
                str(cfg.model_spec_path),
            ],
            "verification_commands": [verify],
        },
        "backend_open": {
            "name": "ares-rust",
            "gate": "backend_open",
            "why": "Debug backend-provider open evidence for generated AresPlan and TargetPlan artifacts.",
            "allowed_scope": [
                "runtime/",
                "backend/",
                str(cfg.run_dir / "artifacts"),
                str(cfg.model_spec_path),
            ],
            "verification_commands": [verify],
        },
        "one_token_logits": {
            "name": "ares-introspection",
            "gate": "one_token_logits",
            "why": (
                "Use the semantic-localization ladder before assigning "
                "one-token logit drift to a backend/runtime owner."
            ),
            "allowed_scope": [
                "frontend/hf-export/",
                "ingest/lean/",
                "bin/ares-introspect",
                "runtime/",
                "backend/",
                str(cfg.run_dir / "artifacts"),
                str(cfg.model_spec_path),
            ],
            "verification_commands": [verify],
        },
        "eight_token_greedy": {
            "name": "ares-introspection",
            "gate": "eight_token_greedy",
            "why": (
                "Localize token drift across source, Lean, AresPlan, "
                "TargetPlan, and backend stages before widening decode depth."
            ),
            "allowed_scope": [
                "frontend/hf-export/",
                "ingest/lean/",
                "bin/ares-introspect",
                "runtime/",
                "backend/",
                str(cfg.run_dir / "artifacts"),
                str(cfg.model_spec_path),
            ],
            "verification_commands": [verify],
        },
        "cpp_tvd": {
            "name": "ares-evidence",
            "gate": "cpp_tvd",
            "why": "Keep C++ Tron/Rinzler classified as comparison and rollback evidence.",
            "allowed_scope": [
                str(cfg.run_dir / "artifacts"),
                str(cfg.model_spec_path),
            ],
            "verification_commands": [verify],
        },
        "depth_performance": {
            "name": "ares-perfetto",
            "gate": "depth_performance",
            "why": "Analyze profiling evidence only after token correctness gates remain green.",
            "allowed_scope": [
                str(cfg.run_dir / "artifacts"),
                str(cfg.run_dir / "logs"),
                str(cfg.model_spec_path),
            ],
            "verification_commands": [verify],
        },
        "mmlu_pro": {
            "name": "ares-mmlu-pro",
            "gate": "mmlu_pro",
            "why": "Run and classify MMLU Pro benchmark evidence for the selected Ares model/backend before production readiness.",
            "allowed_scope": [
                "third_party/systems_test/",
                str(cfg.run_dir / "artifacts"),
                str(cfg.model_spec_path),
            ],
            "verification_commands": [
                "curl $OPENAI_HOST/models",
                "cd third_party/systems_test && uv run mmlu_pro",
                verify,
            ],
        },
    }
    gate_skill = gate_skills.get(first_failed_gate)
    if gate_skill:
        skills.append(gate_skill)
    if first_failed_gate != "complete":
        skills.append(
            {
                "name": "command-fess",
                "gate": "post_commit",
                "why": "Use only when the gate work produced an implementation commit that needs claim audit.",
                "allowed_scope": [
                    "the commit range changed by this gate, if any",
                    str(cfg.run_dir / "handoff.md"),
                    str(cfg.run_dir / "reward.json"),
                ],
                "verification_commands": [
                    "git show --stat",
                    verify,
                ],
            }
        )
    return skills


def workflow_skill_lines(skills: list[Mapping[str, Any]]) -> list[str]:
    lines: list[str] = []
    for skill in skills:
        name = str(skill.get("name", "unknown"))
        gate = str(skill.get("gate", ""))
        why = str(skill.get("why", ""))
        label = f"`{name}`"
        if gate and gate != "all":
            label += f" for `{gate}`"
        lines.append(f"- {label}: {why}")
        allowed = skill.get("allowed_scope", [])
        if isinstance(allowed, list) and allowed:
            lines.append(
                "  Allowed scope: " + ", ".join(f"`{item}`" for item in allowed)
            )
        commands = skill.get("verification_commands", [])
        if isinstance(commands, list) and commands:
            lines.append(
                "  Verification: " + ", ".join(f"`{item}`" for item in commands)
            )
    return lines


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise AresIngestError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AresIngestError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AresIngestError(f"{path} must contain a JSON object")
    return payload


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"history": []}
    return load_json(path)


def load_state_or_recover(path: Path) -> dict[str, Any]:
    try:
        return load_state(path)
    except AresIngestError as state_error:
        return {
            "history": [],
            "previous_state_error": str(state_error),
        }


def resolve_run_path(path: str, cfg: AresIngestConfig) -> Path:
    raw = Path(path)
    if raw.is_absolute():
        return raw
    for base in (cfg.run_dir, cfg.ares_repo):
        candidate = base / raw
        if candidate.exists():
            return candidate
    return cfg.run_dir / raw


def read_json_or_jsonl(path: Path) -> Any:
    try:
        text = path.read_text()
    except FileNotFoundError as exc:
        raise AresIngestError(f"missing JSON file: {path}") from exc
    if path.suffix == ".jsonl":
        rows = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise AresIngestError(
                    f"invalid JSONL in {path}:{lineno}: {exc}"
                ) from exc
        return rows
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise AresIngestError(f"invalid JSON in {path}: {exc}") from exc


def mmlu_pro_expectations(spec: Mapping[str, Any]) -> dict[str, Any]:
    mmlu_cfg = spec.get("mmlu_pro")
    mmlu_cfg = mmlu_cfg if isinstance(mmlu_cfg, Mapping) else {}
    expected_model = (
        mmlu_cfg.get("model") or spec.get("mmlu_model") or spec.get("model")
    )
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
    if not isinstance(payload, Mapping):
        return payload
    if isinstance(payload.get("generated_token_ids"), list):
        return {"tokens": payload["generated_token_ids"]}
    generation = payload.get("generation")
    if isinstance(generation, Mapping) and isinstance(
        generation.get("generated_token_ids"), list
    ):
        return {"tokens": generation["generated_token_ids"]}
    return payload


def introspection_ladder_prompt_lines(spec: Mapping[str, Any]) -> list[str]:
    validated_gates = spec.get("validated_gates")
    if not isinstance(validated_gates, Mapping):
        return []
    gate = validated_gates.get("introspection_ladder")
    if not isinstance(gate, Mapping):
        return []
    detail = gate.get("detail")
    if not isinstance(detail, Mapping):
        detail = {}
    lines = ["- Validated introspection ladder:"]
    if path := gate.get("path"):
        lines.append(f"  - report path: `{path}`")
    if digest := gate.get("sha256"):
        lines.append(f"  - report sha256: `{digest}`")
    if evidence_class := detail.get("evidence_class"):
        lines.append(f"  - evidence class: `{evidence_class}`")
    if first_stage := detail.get("first_failing_stage"):
        lines.append(f"  - first failing introspection stage: `{first_stage}`")
    if next_owner := detail.get("next_owner"):
        lines.append(f"  - next owner: `{next_owner}`")
    trace_context = detail.get("trace_context")
    if isinstance(trace_context, Mapping):
        trace_labels = trace_context.get("trace_labels")
        if isinstance(trace_labels, list) and trace_labels:
            rendered = ", ".join(f"`{label}`" for label in trace_labels[:5])
            suffix = " ..." if len(trace_labels) > 5 else ""
            lines.append(f"  - trace labels: {rendered}{suffix}")
        for field, label in (
            ("backend_events", "backend events"),
            ("perfetto_traces", "Perfetto traces"),
            ("stage_event_summaries", "stage-event summaries"),
            ("perfetto_summaries", "Perfetto summaries"),
        ):
            refs = trace_context.get(field)
            if isinstance(refs, list) and refs:
                paths = [
                    ref.get("path")
                    for ref in refs
                    if isinstance(ref, Mapping) and isinstance(ref.get("path"), str)
                ]
                if paths:
                    rendered = ", ".join(f"`{path}`" for path in paths[:3])
                    suffix = " ..." if len(paths) > 3 else ""
                    lines.append(f"  - {label}: {rendered}{suffix}")
    return lines


def write_handoff(
    cfg: AresIngestConfig,
    reward: dict[str, Any],
    *,
    state: Mapping[str, Any] | None = None,
) -> Path:
    path = cfg.run_dir / "handoff.md"
    history_count = len(state.get("history", [])) if isinstance(state, Mapping) else 0
    status = (
        state.get("status") if isinstance(state, Mapping) else "initialized_setup_only"
    )
    latest_prompt = (
        state.get("latest_refinement_prompt") if isinstance(state, Mapping) else None
    )
    skills = (
        state.get("workflow_skills")
        if isinstance(state, Mapping) and isinstance(state.get("workflow_skills"), list)
        else selected_workflow_skills(
            cfg,
            first_failed_gate=str(reward.get("first_failed_gate", "unknown")),
        )
    )
    skill_text = "\n".join(workflow_skill_lines(skills))
    text = f"""# Ares AutoAgent Run Handoff

## Objective

Bring `{cfg.model_slug}` into Ares through the generated pipeline.

## Current Reward

```json
{json.dumps(reward_fingerprint(reward), indent=2)}
```

Status: `{status}`

History entries: `{history_count}`

Latest refinement prompt: `{latest_prompt or "none"}`

## Workflow Skills

{skill_text}

{trace_report_handoff_section(state)}
## Operator Steering

{chr(10).join(steering_prompt_lines(cfg))}

## Rules

- HF Transformers on PyTorch CPU is the model-correctness oracle.
- Ares/Rust output is system-under-test evidence.
- C++ Tron/Rinzler is comparison and rollback evidence only.
- Runtime execution must flow through frontend artifacts, Lean ingest,
  generated AresPlan, Lean TargetPlan, and a backend provider.
- Do not add hand-authored Rust model plugins or runtime-generated plan
  sidecars.

## Next Action

Work the first failing gate: `{reward["first_failed_gate"]}`.
"""
    path.write_text(text)
    return path


def run_command(
    command: str,
    *,
    cfg: AresIngestConfig,
    env: Mapping[str, str],
    log: Path,
    timeout: int | None = None,
    stream_output: bool = False,
) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    merged_env.update({key: str(value) for key, value in env.items()})
    with log.open("w", encoding="utf-8") as out:
        out.write("+ " + command + "\n")
        out.flush()
        if stream_output:
            print(f"ares-ingest-agent driver log: {log}", flush=True)
            proc = subprocess.Popen(
                ["bash", "-lc", command],
                cwd=cfg.ares_repo,
                env=merged_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            start = time.monotonic()
            assert proc.stdout is not None
            selector = selectors.DefaultSelector()
            selector.register(proc.stdout, selectors.EVENT_READ)
            while True:
                if timeout is not None and time.monotonic() - start > timeout:
                    proc.kill()
                    proc.wait()
                    out.write(f"\nTIMEOUT after {timeout} seconds\n")
                    selector.close()
                    proc.stdout.close()
                    raise AresIngestError(f"refiner timed out; see {log}")
                events = selector.select(timeout=0.1)
                for key, _ in events:
                    line = key.fileobj.readline()
                    if line:
                        out.write(line)
                        out.flush()
                        print(line, end="", flush=True)
                if proc.poll() is not None:
                    break
            for line in proc.stdout:
                out.write(line)
                out.flush()
                print(line, end="", flush=True)
            selector.close()
            proc.stdout.close()
            return int(proc.returncode or 0)
        try:
            proc = subprocess.run(
                ["bash", "-lc", command],
                cwd=cfg.ares_repo,
                env=merged_env,
                stdout=out,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            out.write(f"\nTIMEOUT after {timeout} seconds\n")
            raise AresIngestError(f"refiner timed out; see {log}") from exc
    return proc.returncode


def resolve_refinement_command(args: argparse.Namespace) -> str | None:
    if args.no_refiner:
        return None
    command = args.refinement_command or args.driver_command
    if command:
        return str(command)
    driver_command = DRIVER_COMMANDS[args.driver]
    if driver_command is None:
        raise AresIngestError("--driver custom requires --driver-command")
    return driver_command


def config_from_args(args: argparse.Namespace) -> AresIngestConfig:
    ares_repo = Path(args.ares_repo).resolve()
    safe_model = slugify(args.model)
    run_dir = (
        Path(args.run_dir).resolve()
        if args.run_dir
        else default_run_dir(ares_repo, safe_model)
    )
    return AresIngestConfig(
        model_slug=args.model,
        safe_model=safe_model,
        ares_repo=ares_repo,
        autoagent_root=Path(__file__).resolve().parents[1],
        run_dir=run_dir,
        logs_dir=run_dir / "logs",
        state_path=run_dir / "state.json",
        model_spec_path=run_dir / "model_spec.json",
        target_score=args.target_score,
        max_iterations=None
        if args.max_iterations is None or args.max_iterations <= 0
        else args.max_iterations,
        stall_patience=max(1, args.stall_patience),
        min_improvement=max(0.0, args.min_improvement),
        refinement_command=resolve_refinement_command(args),
        driver=args.driver,
        gate_profile=args.gate_profile,
        setup_only=args.setup_only,
        cockpit=args.cockpit,
        stream_refiner_output=args.stream_refiner_output or args.cockpit,
    )


def evaluate_run(cfg: AresIngestConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    ensure_steering_files(cfg)
    spec = (
        load_json(cfg.model_spec_path)
        if cfg.model_spec_path.exists()
        else build_model_spec(cfg)
    )
    spec.setdefault("model", cfg.model_slug)
    spec.setdefault("safe_model", cfg.safe_model)
    spec.setdefault("ares_repo", str(cfg.ares_repo))
    spec.setdefault("frontend", "fx")
    spec.setdefault("backend", "tron")
    spec.setdefault("gate_profile", cfg.gate_profile)
    spec.setdefault(
        "required_gates", list(required_gates_for_profile(cfg.gate_profile))
    )
    if not isinstance(spec["required_gates"], list) or not all(
        isinstance(gate, str) for gate in spec["required_gates"]
    ):
        raise AresIngestError("model_spec required_gates must be a list of strings")
    explicit_gates = spec.setdefault("explicit_gates", {})
    if not isinstance(explicit_gates, dict):
        raise AresIngestError("model_spec explicit_gates must be a JSON object")
    explicit_gates.setdefault(
        "model_spec",
        {
            "passed": True,
            "score": 1.0,
            "source": "ares-ingest-agent setup",
        },
    )
    validated_gates: dict[str, Any] = {}
    validated_gates["shortcut_scan"] = shortcut_scan_gate(cfg.ares_repo)

    oracle_payload = None
    if oracle_spec := spec.get("oracle_records"):
        oracle_payload = read_json_or_jsonl(resolve_run_path(str(oracle_spec), cfg))
    if ares_plan := spec.get("ares_plan"):
        validated_gates["aresplan_valid"] = ares_plan_gate(
            resolve_run_path(str(ares_plan), cfg)
        )
    if target_plan := spec.get("target_plan"):
        validated_gates["targetplan_valid"] = target_plan_gate(
            resolve_run_path(str(target_plan), cfg)
        )
    if "artifact_consistency" in spec["required_gates"]:
        validated_gates["artifact_consistency"] = artifact_consistency_gate(
            spec,
            oracle_payload=oracle_payload,
            validated_gates=validated_gates,
        )
    if backend_open := spec.get("backend_open_evidence"):
        validated_gates["backend_open"] = backend_open_gate(
            resolve_run_path(str(backend_open), cfg)
        )
    if one_token := spec.get("one_token_logits_evidence") or spec.get(
        "one_token_results_json"
    ):
        validated_gates["one_token_logits"] = one_token_logits_gate(
            resolve_run_path(str(one_token), cfg)
        )
    token_payload = None
    if eight_token := spec.get("eight_token_greedy_evidence") or spec.get(
        "token_results_json"
    ):
        eight_token_path = resolve_run_path(str(eight_token), cfg)
        validated_gates["eight_token_greedy"] = token_agreement_gate(eight_token_path)
        token_payload = read_json_or_jsonl(eight_token_path)
    if token_spec := spec.get("token_comparison"):
        if not isinstance(token_spec, Mapping):
            raise AresIngestError("model_spec token_comparison must be a JSON object")
        reference = resolve_run_path(str(token_spec["reference"]), cfg)
        candidate = resolve_run_path(str(token_spec["candidate"]), cfg)
        reference_payload = read_payload(reference)
        candidate_payload = read_payload(candidate)
        token_result = compare_payloads(
            generated_payload(reference_payload),
            generated_payload(candidate_payload),
        )
        token_payload = build_greedy_token_evidence(
            token_result,
            reference=reference,
            candidate=candidate,
            reference_payload=reference_payload,
            candidate_payload=candidate_payload,
            expected_generated_tokens=int(
                token_spec.get("expected_generated_tokens", 8)
            ),
        )
        token_results_path = cfg.run_dir / str(token_spec.get("output", "tokens.json"))
        write_json(token_results_path, token_payload)
        spec["eight_token_greedy_evidence"] = token_results_path.name
        validated_gates["eight_token_greedy"] = token_agreement_gate(token_results_path)
    if cpp_tvd := spec.get("cpp_tvd_evidence"):
        validated_gates["cpp_tvd"] = cpp_tvd_gate(resolve_run_path(str(cpp_tvd), cfg))
    if depth_performance := spec.get("depth_performance_evidence"):
        validated_gates["depth_performance"] = depth_performance_gate(
            resolve_run_path(str(depth_performance), cfg)
        )
    if mmlu_pro := spec.get("mmlu_pro_evidence"):
        validated_gates["mmlu_pro"] = mmlu_pro_gate(
            resolve_run_path(str(mmlu_pro), cfg),
            **mmlu_pro_expectations(spec),
        )
    if introspection_ladder := spec.get("introspection_ladder"):
        validated_gates["introspection_ladder"] = introspection_ladder_gate(
            resolve_run_path(str(introspection_ladder), cfg)
        )
    trace_report_spec = spec.get("trace_report_json") or spec.get("trace_report")
    if trace_report_spec:
        if not isinstance(trace_report_spec, str):
            raise AresIngestError("model_spec trace_report_json must be a string path")
        validated_gates["trace_report"] = trace_report_gate(
            resolve_run_path(trace_report_spec, cfg)
        )
    command_wrapper_plan = build_command_wrapper_plan(
        spec,
        run_dir=cfg.run_dir,
        ares_repo=cfg.ares_repo,
    )
    if command_wrapper_plan["wrappers"]:
        command_wrapper_path = cfg.run_dir / "command_wrappers.json"
        write_json(command_wrapper_path, command_wrapper_plan)
        spec["command_wrapper_plan"] = command_wrapper_path.name
    else:
        spec.pop("command_wrapper_plan", None)
    spec["validated_gates"] = validated_gates
    spec["explicit_gates"].update(validated_gates)

    write_json(cfg.model_spec_path, spec)
    reward = compute_reward(
        gates_payload={"gates": spec["explicit_gates"]},
        validated_gates_payload={"gates": validated_gates},
        oracle_payload=oracle_payload,
        token_payload=token_payload,
        required_gates=tuple(spec["required_gates"]),
    )
    write_json(cfg.run_dir / "reward.json", reward)
    (cfg.run_dir / "reward.txt").write_text(f"{reward['score']:.12g}\n")
    return spec, reward


def initialize_run(cfg: AresIngestConfig) -> dict[str, Any]:
    spec, reward = evaluate_run(cfg)
    workflow_skills = selected_workflow_skills(
        cfg,
        first_failed_gate=str(reward.get("first_failed_gate", "unknown")),
    )
    trace_report = trace_report_summary_from_spec(spec)
    state = {
        "status": "initialized_setup_only",
        "model": cfg.model_slug,
        "safe_model": cfg.safe_model,
        "target_score": cfg.target_score,
        "max_iterations": cfg.max_iterations,
        "driver": cfg.driver,
        "refinement_command": cfg.refinement_command,
        "gate_profile": cfg.gate_profile,
        "refinement_loop": "setup_only",
        "cockpit": {
            "enabled": cfg.cockpit,
            "stream_refiner_output": cfg.stream_refiner_output,
            "event_log": str(cfg.run_dir / "cockpit.jsonl"),
            "steering_json": str(steering_json_path(cfg)),
            "steering_md": str(steering_markdown_path(cfg)),
        },
        "workflow_skills": workflow_skills,
        "reward": reward_fingerprint(reward),
        "history": [],
    }
    if trace_report:
        state["trace_report"] = trace_report
    write_json(cfg.state_path, state)
    write_handoff(cfg, reward, state=state)
    return state


def gate_guidance(
    first_failed_gate: str,
    *,
    cfg: AresIngestConfig,
    spec: Mapping[str, Any],
    iteration: int,
) -> list[str]:
    common = [
        f"- Model spec: `{cfg.model_spec_path}`",
        f"- Reward JSON: `{cfg.run_dir / 'reward.json'}`",
        f"- Run handoff: `{cfg.run_dir / 'handoff.md'}`",
        f"- Verifier/refiner logs: `{cfg.logs_dir}`",
        f"- This iteration's refiner log: `{cfg.logs_dir / f'{iteration:02d}-refiner.log'}`",
    ]
    for field in ("oracle_records", "ares_plan", "target_plan", "introspection_ladder"):
        if value := spec.get(field):
            common.append(
                f"- `{field}` artifact: `{resolve_run_path(str(value), cfg)}`"
            )
    if value := spec.get("trace_report_json") or spec.get("trace_report"):
        common.extend(
            [
                f"- Trace report JSON: `{resolve_run_path(str(value), cfg)}`",
                "- Inspect `sections.answerability`, `sections.unsupported_claims`, and `sections.next_measurements` before ad hoc parsing.",
                "- Read `sections.report_json_section_inventory` to discover available report sections, then read `sections.trace_config_rows` before choosing sidecar-specific report sections.",
            ]
        )
    if value := spec.get("command_wrapper_plan"):
        common.append(f"- Command wrapper plan: `{resolve_run_path(str(value), cfg)}`")
    common.extend(introspection_ladder_prompt_lines(spec))
    specific: dict[str, list[str]] = {
        "hf_cpu_oracle": [
            "- Capture or attach a real HF Transformers + PyTorch CPU oracle record.",
            "- Do not use Ares/Rust, C++ Tron, mocks, or generated fixtures as the oracle.",
            "- Record model/tokenizer revisions, prompt tokens, generated ids, and logit slices.",
        ],
        "frontend_export": [
            "- Produce or select the frontend artifact declared by the model spec.",
            "- Keep `frontend/fx` as the default route unless this task explicitly asks for HF export.",
        ],
        "lean_ingest": [
            "- Run the Lean ingest path on the frontend artifact and keep logs in this run directory.",
            "- Do not move ahead-of-time planning into Rust.",
        ],
        "aresplan_valid": [
            "- Attach a generated AresPlan JSON artifact and set `ares_plan` in `model_spec.json`.",
            "- The AresPlan must carry Lean ingest provenance and a non-empty generated body.",
        ],
        "targetplan_valid": [
            "- Attach a Lean-emitted TargetPlan JSON artifact and set `target_plan` in `model_spec.json`.",
            "- Missing TargetPlan artifacts must fail before backend execution.",
        ],
        "artifact_consistency": [
            "- Make the HF CPU oracle and TargetPlan identify the same model row.",
            "- Set `expected_model_ids` in `model_spec.json` when a registry row, local checkpoint path, and HF model id are legitimate aliases.",
            "- Do not pass this gate by mixing a real oracle with unrelated fixture plans.",
        ],
        "shortcut_scan": [
            "- Remove any hand-authored Rust model-family plugin or runtime-generated plan sidecar.",
            "- Do not mark this gate manually; it is refreshed from the Ares source tree.",
        ],
        "backend_open": [
            "- Open the selected backend only with generated AresPlan and TargetPlan artifacts.",
            "- Record backend id, hardware gate variables, and event/log paths.",
        ],
        "one_token_logits": [
            "- Compare one-token dense logits against the HF CPU oracle with replay metadata.",
            "- Ares/Rust output is system-under-test evidence, never the oracle.",
            (
                "- If HF CPU oracle and stage snapshots are available, run "
                "`bin/ares-introspect ladder --graph ...` and attach "
                "`introspection_ladder` in `model_spec.json` before generic "
                "backend debugging."
            ),
        ],
        "eight_token_greedy": [
            "- Prove 8-token greedy identity before considering 64-token or 512-token gates.",
            (
                "- If tokens drift after a valid HF CPU oracle, inspect or "
                "produce `ares.introspection.ladder.v1` and work only the "
                "first failing introspection stage."
            ),
        ],
        "cpp_tvd": [
            "- Treat C++ Tron/Rinzler as comparison/rollback evidence, not correctness oracle.",
            "- Require matching replay context and dense-logit TVD artifacts.",
        ],
        "depth_performance": [
            "- Follow the 8 -> 64 -> 512 ladder and keep correctness gates green before speed claims.",
            "- Record throughput, latency, TTFT, memory, and artifact hashes.",
        ],
        "mmlu_pro": [
            "- Run MMLU Pro through `third_party/systems_test` against an OpenAI-compatible Ares endpoint for the selected backend.",
            "- Verify `/v1/models` first; set `MMLU_MODEL` to the exact API-facing model id, and add a matching `scripts/mmlu_pro.py` config entry if the id is new.",
            "- Attach `mmlu_pro_evidence` with schema `ares.benchmark.mmlu_pro.v1`, score, threshold, coverage, model/backend binding, systems_test commit, AresPlan/TargetPlan hashes, and raw artifact hashes.",
            "- Treat the benchmark as Ares system-under-test evidence, not an HF CPU oracle or C++ comparison substitute.",
        ],
    }
    return common + specific.get(first_failed_gate, [])


def allowed_write_scope_lines(first_failed_gate: str) -> list[str]:
    if first_failed_gate == "model_spec":
        return [
            "- Keep Ares source and run-state changes inside the Ares repo and this run directory.",
            "- For model prior-art inspection, use explicit checkout paths recorded in `model_spec.json` or `handoff.md`; otherwise create/use `${ARES_PRIOR_ART_ROOT:-$HOME/db}` and read or clone official vLLM, llama.cpp, and MLX checkouts from `https://github.com/vllm-project/vllm.git`, `https://github.com/ggml-org/llama.cpp.git`, and `https://github.com/ml-explore/mlx.git`.",
            "- Do not vendor those prior-art repositories into Ares or add them as submodules unless the user explicitly asks.",
            "- Keep AutoAgent source changes out of model-ingest refinement unless the failing gate is an AutoAgent tool bug.",
            "- Keep changes focused on the first failing gate; leave later gates for later iterations.",
        ]
    return [
        "- Work inside the Ares repo and this run directory only.",
        "- Keep AutoAgent source changes out of model-ingest refinement unless the failing gate is an AutoAgent tool bug.",
        "- Keep changes focused on the first failing gate; leave later gates for later iterations.",
    ]


def write_refinement_prompt(
    cfg: AresIngestConfig,
    *,
    iteration: int,
    spec: Mapping[str, Any],
    reward: dict[str, Any],
) -> Path:
    prompt_dir = cfg.run_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    path = prompt_dir / f"refinement-{iteration:02d}.md"
    first_failed_gate = str(reward.get("first_failed_gate", "unknown"))
    verify_command = verifier_command(cfg)
    workflow_skills = selected_workflow_skills(
        cfg,
        first_failed_gate=first_failed_gate,
    )
    prompt = "\n".join(
        [
            f"# Ares Ingest Refinement {iteration}",
            "",
            "## Objective",
            "",
            f"Bring `{cfg.model_slug}` into Ares through the generated pipeline.",
            "",
            "## Current Gate",
            "",
            f"Work only the first failing gate: `{first_failed_gate}`.",
            "",
            "Current reward:",
            "",
            "```json",
            json.dumps(reward_fingerprint(reward), indent=2, sort_keys=True),
            "```",
            "",
            "## Paths",
            "",
            f"- Ares repo: `{cfg.ares_repo}`",
            f"- AutoAgent repo: `{cfg.autoagent_root}`",
            f"- Run directory: `{cfg.run_dir}`",
            f"- Model spec: `{cfg.model_spec_path}`",
            f"- Reward JSON: `{cfg.run_dir / 'reward.json'}`",
            f"- Handoff: `{cfg.run_dir / 'handoff.md'}`",
            f"- Logs: `{cfg.logs_dir}`",
            "",
            "## Operator Steering",
            "",
            *steering_prompt_lines(cfg),
            "",
            "## Expected Workflow Skills",
            "",
            *workflow_skill_lines(workflow_skills),
            "",
            "## Relevant Artifacts To Inspect",
            "",
            *gate_guidance(
                first_failed_gate,
                cfg=cfg,
                spec=spec,
                iteration=iteration,
            ),
            "",
            *trace_report_prompt_section(spec),
            "## Ares Rules",
            "",
            "- HF Transformers on PyTorch CPU is the only model-correctness oracle.",
            "- C++ Tron/Rinzler is comparison, compliance, performance, and rollback evidence only.",
            "- Ares/Rust output is system-under-test evidence.",
            "- Runtime execution must flow through frontend artifacts, Lean ingest, generated AresPlan, Lean TargetPlan, and a backend provider.",
            "- Do not add hand-authored Rust model plugins or runtime-generated AresPlan/TargetPlan sidecars.",
            "- Do not use generated-plan runtime shortcuts as production-readiness evidence.",
            "",
            "## Allowed Write Scope",
            "",
            *allowed_write_scope_lines(first_failed_gate),
            "",
            "## Required Verification Before Returning",
            "",
            "- Rerun the cheapest command that proves the first failing gate changed.",
            f"- Rerun `{verify_command}` or update `model_spec.json` with the new artifact paths so the next verifier pass can score it.",
            "- Update `handoff.md` or another run-local note if you discover durable state the next agent needs.",
            "",
        ]
    )
    path.write_text(prompt)
    return path


def run_refiner(
    cfg: AresIngestConfig,
    *,
    iteration: int,
    spec: Mapping[str, Any],
    reward: dict[str, Any],
) -> Path:
    if not cfg.refinement_command:
        raise AresIngestError("no refinement command configured")
    prompt_path = write_refinement_prompt(
        cfg,
        iteration=iteration,
        spec=spec,
        reward=reward,
    )
    env = {
        "ARES_REPO": str(cfg.ares_repo),
        "AUTOAGENT_REPO": str(cfg.autoagent_root),
        "RUN_DIR": str(cfg.run_dir),
        "MODEL_SPEC": str(cfg.model_spec_path),
        "REWARD_JSON": str(cfg.run_dir / "reward.json"),
        "REFINEMENT_PROMPT": str(prompt_path),
        "FIRST_FAILED_GATE": str(reward.get("first_failed_gate", "unknown")),
        "ITERATION": str(iteration),
        "MODEL_SLUG": cfg.model_slug,
        "MODEL_SAFE": cfg.safe_model,
        "STEERING_JSON": str(steering_json_path(cfg)),
        "STEERING_MD": str(steering_markdown_path(cfg)),
        "DRIVER": cfg.driver,
        "DRIVER_COMMAND": cfg.refinement_command or "",
    }
    log = cfg.logs_dir / f"{iteration:02d}-refiner.log"
    if cfg.cockpit:
        print(
            f"ares-ingest-agent cockpit iteration={iteration} driver={cfg.driver} log={log}",
            flush=True,
        )
    rc = run_command(
        cfg.refinement_command,
        cfg=cfg,
        env=env,
        log=log,
        timeout=24 * 3600,
        stream_output=cfg.stream_refiner_output,
    )
    if rc != 0:
        raise AresIngestError(f"refiner failed with exit {rc}; see {log}")
    return prompt_path


def previous_score(state: Mapping[str, Any]) -> float | None:
    history = state.get("history", [])
    if not isinstance(history, list) or not history:
        return None
    score = history[-1].get("score")
    return float(score) if isinstance(score, (int, float)) else None


def write_cockpit_event(
    cfg: AresIngestConfig,
    *,
    iteration: int,
    reward: Mapping[str, Any],
    status: str,
    decision: str | None = None,
) -> None:
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    event = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "iteration": iteration,
        "status": status,
        "decision": decision,
        "driver": cfg.driver,
        "refinement_command": cfg.refinement_command,
        "reward": reward_fingerprint(dict(reward)),
        "run_dir": str(cfg.run_dir),
        "steering_json": str(steering_json_path(cfg)),
        "steering_md": str(steering_markdown_path(cfg)),
    }
    with (cfg.run_dir / "cockpit.jsonl").open("a", encoding="utf-8") as out:
        out.write(json.dumps(event, sort_keys=True) + "\n")


def print_cockpit_dashboard(
    cfg: AresIngestConfig,
    *,
    iteration: int,
    reward: Mapping[str, Any],
    state: Mapping[str, Any],
    best_score: float,
    stall_count: int,
) -> None:
    score = float(reward.get("score", 0.0) or 0.0)
    prior = previous_score(state)
    delta = "n/a" if prior is None else f"{score - prior:+.6f}"
    print("", flush=True)
    print("Ares ingest cockpit", flush=True)
    print(f"  iteration: {iteration}", flush=True)
    print(
        "  quality: "
        f"score={score:.6f} "
        f"alpha={float(reward.get('alpha_execution', 0.0) or 0.0):.6f} "
        f"tau={float(reward.get('tau_tokens', 0.0) or 0.0):.6f} "
        f"delta={float(reward.get('delta_inference', 0.0) or 0.0):.6f} "
        f"change={delta}",
        flush=True,
    )
    print(
        "  control: "
        f"target={cfg.target_score:.6f} best={best_score:.6f} "
        f"stall={stall_count}/{cfg.stall_patience}",
        flush=True,
    )
    print(
        f"  gate: {reward.get('first_failed_gate')} stage_cap={reward.get('stage_cap')}",
        flush=True,
    )
    print(f"  driver: {cfg.driver}", flush=True)
    print(f"  run_dir: {cfg.run_dir}", flush=True)
    print(f"  handoff: {cfg.run_dir / 'handoff.md'}", flush=True)
    print(f"  steering: {steering_markdown_path(cfg)}", flush=True)


def print_cockpit_help() -> None:
    print(
        "cockpit commands: continue | stop | note TEXT | "
        "resource PATH_OR_URL [-- NOTE] | driver SHELL_COMMAND | show | help",
        flush=True,
    )


def cockpit_checkpoint(
    cfg: AresIngestConfig,
    *,
    iteration: int,
    reward: Mapping[str, Any],
    state: Mapping[str, Any],
    best_score: float,
    stall_count: int,
) -> bool:
    if not cfg.cockpit:
        return True
    if not sys.stdin.isatty():
        print(
            "ares-ingest-agent cockpit: stdin is not a TTY; continuing "
            "without interactive steering.",
            flush=True,
        )
        write_cockpit_event(
            cfg,
            iteration=iteration,
            reward=reward,
            status="continue_noninteractive",
            decision="continue",
        )
        return True

    print_cockpit_help()
    while True:
        try:
            line = input("cockpit> ").strip()
        except EOFError:
            line = "continue"
        if not line or line in {"c", "continue"}:
            write_cockpit_event(
                cfg,
                iteration=iteration,
                reward=reward,
                status="continue",
                decision="continue",
            )
            return True
        if line in {"s", "stop", "q", "quit"}:
            write_cockpit_event(
                cfg,
                iteration=iteration,
                reward=reward,
                status="stopped_by_operator",
                decision="stop",
            )
            return False
        if line in {"h", "help", "?"}:
            print_cockpit_help()
            continue
        if line == "show":
            print(f"run_dir={cfg.run_dir}", flush=True)
            print(f"state={cfg.state_path}", flush=True)
            print(f"reward={cfg.run_dir / 'reward.json'}", flush=True)
            print(f"handoff={cfg.run_dir / 'handoff.md'}", flush=True)
            print(f"steering={steering_markdown_path(cfg)}", flush=True)
            continue
        if line.startswith("note "):
            append_steering_note(cfg, iteration=iteration, text=line[5:].strip())
            print(f"recorded note in {steering_markdown_path(cfg)}", flush=True)
            continue
        if line.startswith("resource "):
            value_note = line[len("resource ") :].strip()
            value, sep, note = value_note.partition(" -- ")
            append_steering_resource(
                cfg,
                iteration=iteration,
                value=value.strip(),
                note=note.strip() if sep else "",
            )
            print(f"recorded resource in {steering_markdown_path(cfg)}", flush=True)
            continue
        if line.startswith("driver "):
            command = line[len("driver ") :].strip()
            if not command:
                print("driver command cannot be empty", flush=True)
                continue
            cfg.refinement_command = command
            append_driver_command(cfg, iteration=iteration, command=command)
            print("driver command updated for this run", flush=True)
            continue
        print(f"unknown cockpit command: {line}", flush=True)
        print_cockpit_help()


def append_history(
    state: dict[str, Any],
    *,
    cfg: AresIngestConfig,
    iteration: int,
    reward: dict[str, Any],
    status: str,
    prompt_path: Path | None = None,
) -> None:
    state.setdefault("history", [])
    state.setdefault("model", cfg.model_slug)
    state.setdefault("safe_model", cfg.safe_model)
    state.setdefault("run_dir", str(cfg.run_dir))
    state.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    state["status"] = status
    state["target_score"] = cfg.target_score
    state["max_iterations"] = cfg.max_iterations
    state["driver"] = cfg.driver
    state["refinement_command"] = cfg.refinement_command
    state["gate_profile"] = cfg.gate_profile
    state["refinement_loop"] = "one_failing_gate"
    state["cockpit"] = {
        "enabled": cfg.cockpit,
        "stream_refiner_output": cfg.stream_refiner_output,
        "event_log": str(cfg.run_dir / "cockpit.jsonl"),
        "steering_json": str(steering_json_path(cfg)),
        "steering_md": str(steering_markdown_path(cfg)),
    }
    state["workflow_skills"] = selected_workflow_skills(
        cfg,
        first_failed_gate=str(reward.get("first_failed_gate", "unknown")),
    )
    state["reward"] = reward_fingerprint(reward)
    trace_report = None
    if cfg.model_spec_path.exists():
        trace_report = trace_report_summary_from_spec(load_json(cfg.model_spec_path))
    if trace_report:
        state["trace_report"] = trace_report
    else:
        state.pop("trace_report", None)
    if prompt_path is not None:
        state["latest_refinement_prompt"] = str(prompt_path)
    history_entry = {
        "iteration": iteration,
        "status": status,
        "model_spec": str(cfg.model_spec_path),
        "reward_json": str(cfg.run_dir / "reward.json"),
        "logs_dir": str(cfg.logs_dir),
        **reward_fingerprint(reward),
    }
    if trace_report:
        history_entry["trace_report"] = trace_report
    state["history"].append(history_entry)
    write_json(cfg.state_path, state)
    write_handoff(cfg, reward, state=state)


def write_failure_state(cfg: AresIngestConfig, error: BaseException) -> None:
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    state = load_state_or_recover(cfg.state_path)
    state["status"] = "failed"
    state["error"] = str(error)
    state.setdefault("model", cfg.model_slug)
    state.setdefault("safe_model", cfg.safe_model)
    state.setdefault("run_dir", str(cfg.run_dir))
    state.setdefault("gate_profile", cfg.gate_profile)
    state.setdefault("refinement_loop", "one_failing_gate")
    state.setdefault("driver", cfg.driver)
    state.setdefault(
        "cockpit",
        {
            "enabled": cfg.cockpit,
            "stream_refiner_output": cfg.stream_refiner_output,
            "event_log": str(cfg.run_dir / "cockpit.jsonl"),
            "steering_json": str(steering_json_path(cfg)),
            "steering_md": str(steering_markdown_path(cfg)),
        },
    )
    reward_path = cfg.run_dir / "reward.json"
    reward_payload = load_json(reward_path) if reward_path.exists() else None
    first_failed_gate = (
        str(reward_payload.get("first_failed_gate", "failed"))
        if reward_payload is not None
        else "failed"
    )
    state["workflow_skills"] = selected_workflow_skills(
        cfg,
        first_failed_gate=first_failed_gate,
    )
    if cfg.model_spec_path.exists():
        try:
            trace_report = trace_report_summary_from_spec(
                load_json(cfg.model_spec_path)
            )
        except AresIngestError:
            trace_report = None
        if trace_report:
            state["trace_report"] = trace_report
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(cfg.state_path, state)
    if reward_payload is not None:
        write_handoff(cfg, reward_payload, state=state)


def run_loop(cfg: AresIngestConfig) -> int:
    if cfg.setup_only:
        initialize_run(cfg)
        return 0

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    state = load_state_or_recover(cfg.state_path)
    best_score = max(
        [float(item.get("score", 0.0) or 0.0) for item in state.get("history", [])]
        or [0.0]
    )
    stall_count = 0
    previous_fingerprint: dict[str, Any] | None = None
    iteration = 1

    while cfg.max_iterations is None or iteration <= cfg.max_iterations:
        spec, reward = evaluate_run(cfg)
        score = float(reward.get("score", 0.0) or 0.0)
        fingerprint = reward_fingerprint(reward)

        if score >= cfg.target_score:
            if cfg.cockpit:
                write_cockpit_event(
                    cfg,
                    iteration=iteration,
                    reward=reward,
                    status="complete",
                )
                print_cockpit_dashboard(
                    cfg,
                    iteration=iteration,
                    reward=reward,
                    state=state,
                    best_score=max(best_score, score),
                    stall_count=stall_count,
                )
            append_history(
                state,
                cfg=cfg,
                iteration=iteration,
                reward=reward,
                status="complete",
            )
            print_summary(cfg, reward, status="complete")
            return 0

        if score > best_score + cfg.min_improvement:
            best_score = score
            stall_count = 0
        else:
            stall_count += 1
        if previous_fingerprint == fingerprint:
            stall_count += 1
        previous_fingerprint = fingerprint

        if cfg.cockpit:
            write_cockpit_event(
                cfg,
                iteration=iteration,
                reward=reward,
                status="evaluated",
            )
            print_cockpit_dashboard(
                cfg,
                iteration=iteration,
                reward=reward,
                state=state,
                best_score=best_score,
                stall_count=stall_count,
            )

        if not cfg.refinement_command:
            append_history(
                state,
                cfg=cfg,
                iteration=iteration,
                reward=reward,
                status="blocked_no_refiner",
            )
            print_summary(cfg, reward, status="blocked_no_refiner")
            return 3

        if stall_count >= cfg.stall_patience:
            append_history(
                state,
                cfg=cfg,
                iteration=iteration,
                reward=reward,
                status="stalled",
            )
            print_summary(cfg, reward, status="stalled")
            return 2

        if cfg.max_iterations is not None and iteration >= cfg.max_iterations:
            append_history(
                state,
                cfg=cfg,
                iteration=iteration,
                reward=reward,
                status="max_iterations",
            )
            print_summary(cfg, reward, status="max_iterations")
            return 2

        if not cockpit_checkpoint(
            cfg,
            iteration=iteration,
            reward=reward,
            state=state,
            best_score=best_score,
            stall_count=stall_count,
        ):
            append_history(
                state,
                cfg=cfg,
                iteration=iteration,
                reward=reward,
                status="stopped_by_operator",
            )
            print_summary(cfg, reward, status="stopped_by_operator")
            return 4

        prompt_path = run_refiner(cfg, iteration=iteration, spec=spec, reward=reward)
        append_history(
            state,
            cfg=cfg,
            iteration=iteration,
            reward=reward,
            status="refiner_ran",
            prompt_path=prompt_path,
        )
        iteration += 1

    _, reward = evaluate_run(cfg)
    append_history(
        state,
        cfg=cfg,
        iteration=iteration - 1,
        reward=reward,
        status="max_iterations",
    )
    print_summary(cfg, reward, status="max_iterations")
    return 2


def print_summary(
    cfg: AresIngestConfig, reward: Mapping[str, Any], *, status: str
) -> None:
    print(
        "ares-ingest-agent "
        f"status={status} "
        f"score={reward.get('score')} "
        f"alpha_execution={reward.get('alpha_execution')} "
        f"tau_tokens={reward.get('tau_tokens')} "
        f"delta_inference={reward.get('delta_inference')} "
        f"first_failed_gate={reward.get('first_failed_gate')}"
    )
    print(f"ares_repo={cfg.ares_repo}")
    print(f"run_dir={cfg.run_dir}")
    print(f"state={cfg.state_path}")
    print(f"reward={cfg.run_dir / 'reward.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="HuggingFace model id or local model label")
    parser.add_argument("--ares-repo", default=".", help="Ares repository root")
    parser.add_argument("--run-dir", help="Explicit run directory")
    parser.add_argument("--target-score", type=float, default=1.0)
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=0,
        help="Safety cap on verifier/refiner iterations; 0 means no cap",
    )
    parser.add_argument("--stall-patience", type=int, default=2)
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.001,
        help="Minimum score improvement that resets stall detection",
    )
    parser.add_argument(
        "--no-refiner",
        action="store_true",
        help="Run evaluation only; stop below target with blocked_no_refiner",
    )
    parser.add_argument(
        "--driver",
        choices=sorted(DRIVER_COMMANDS),
        default="codex",
        help="Agentic CLI driver used for refinement when no explicit command is supplied",
    )
    parser.add_argument(
        "--driver-command",
        help="Arbitrary shell command to run as the refinement driver",
    )
    parser.add_argument(
        "--refinement-command",
        help="Backward-compatible alias for --driver-command",
    )
    parser.add_argument(
        "--gate-profile",
        choices=sorted(GATE_PROFILES),
        default="cpu-only",
        help="Gate profile for the generated model_spec.json",
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Create run state and exit without invoking the refiner loop",
    )
    parser.add_argument(
        "--cockpit",
        action="store_true",
        help="Show iteration score dashboards and prompt for steering between refiner passes",
    )
    parser.add_argument(
        "--stream-refiner-output",
        action="store_true",
        help="Tee refiner stdout/stderr to the terminal while preserving logs",
    )
    parser.add_argument("--print-json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg: AresIngestConfig | None = None
    try:
        cfg = config_from_args(args)
        rc = run_loop(cfg)
        if args.print_json:
            print(json.dumps(load_state(cfg.state_path), indent=2))
        return rc
    except AresIngestError as exc:
        failure_note = ""
        if cfg is not None:
            try:
                write_failure_state(cfg, exc)
            except Exception as state_exc:
                failure_note = f" (also failed to write failure state: {state_exc})"
        parser.exit(1, f"ares-ingest-agent: error: {exc}{failure_note}\n")
    except KeyboardInterrupt:
        parser.exit(130, "ares-ingest-agent: interrupted\n")


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ares_ingest_autoagent.artifacts import (
    artifact_consistency_gate,
    backend_open_gate,
    build_greedy_token_evidence,
    cpp_tvd_gate,
    depth_performance_gate,
    introspection_ladder_gate,
    mmlu_pro_gate,
    one_token_logits_gate,
    trace_report_gate,
    token_agreement_gate,
    validate_cpp_tvd_evidence,
    validate_introspection_ladder_report,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write_json_artifact(path: Path, payload: dict) -> dict[str, str]:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return {"path": path.name, "sha256": sha256_file(path)}


def write_prefixed_json_artifact(path: Path, payload: dict) -> dict[str, str]:
    artifact = write_json_artifact(path, payload)
    return {"path": artifact["path"], "sha256": "sha256:" + artifact["sha256"]}


def introspection_ladder_payload(root: Path, *, evidence_class: str) -> dict:
    source_run = write_prefixed_json_artifact(
        root / "source.run.json",
        {"schema": "ares.introspection.run.v1"},
    )
    compare = write_prefixed_json_artifact(
        root / "hf-vs-source.compare.json",
        {"schema": "ares.introspection.compare.v1"},
    )
    backend_events = write_prefixed_json_artifact(
        root / "backend-events.jsonl",
        {"event": "forward_executed"},
    )
    backend_events["role"] = "backend_events"
    perfetto_trace = write_prefixed_json_artifact(
        root / "trace.perfetto-trace",
        {"synthetic": "trace-bytes"},
    )
    perfetto_trace["role"] = "perfetto_trace"
    trace_metadata = write_prefixed_json_artifact(
        root / "run.trace-meta.json",
        {
            "schema_version": 1,
            "trace_run_id": "trace-run-001",
            "introspection_artifacts": [],
        },
    )
    trace_metadata["schema"] = "ares.trace.metadata.v1"
    trace_metadata["role"] = "ares_trace_metadata"
    return {
        "schema": "ares.introspection.ladder.v1",
        "status": "failed",
        "evidence_class": evidence_class,
        "producer": {"tool": "unit-test"},
        "model": {"id": "synthetic/model", "checkpoint": "unit-test"},
        "stage_order": ["hf_cpu_oracle", "source_hf_hypergraph"],
        "graphs": [
            {
                "stage": "source_hf_hypergraph",
                "schema": "ares.introspection.graph.v1",
                "sha256": "sha256:" + SHA_A,
            }
        ],
        "runs": [
            {
                "stage": "source_hf_hypergraph",
                "evidence_role": "candidate",
                "schema": "ares.introspection.run.v1",
                "sha256": source_run["sha256"],
                "status": "passed",
                "path": source_run["path"],
            }
        ],
        "comparisons": [
            {
                "from_stage": "hf_cpu_oracle",
                "to_stage": "source_hf_hypergraph",
                "schema": "ares.introspection.compare.v1",
                "sha256": compare["sha256"],
                "status": "failed",
                "result": "diverged",
                "path": compare["path"],
                "first_mismatch": {
                    "id": "logits",
                    "producer_generator": "linear_0",
                    "max_abs_error": 0.25,
                },
            }
        ],
        "first_failing_stage": "source_hf_hypergraph",
        "next_owner": "ares-python",
        "recommendation": "Investigate HF-export source replay.",
        "trace_context": {
            "autoagent_run_id": "unit-test",
            "request_id": "request-1",
            "trace_labels": ["targetplan.stmt.00001.matmul"],
            "backend_events": [backend_events],
            "perfetto_traces": [perfetto_trace],
            "trace_metadata": [trace_metadata],
        },
    }


def mmlu_pro_payload(
    root: Path,
    *,
    model: str = "synthetic/model",
    backend: str = "fpga",
    openai_host: str = "http://127.0.0.1:8000/v1",
    coverage_percent: float = 10,
    score_percent: float = 72.0,
    required_score_percent: float = 70.0,
    systems_test_dirty: bool = False,
    include_endpoint_models: bool = True,
    include_systems_test_config: bool = True,
) -> dict:
    report = root / "report.txt"
    report.write_text(f"Total, {score_percent:.0f}/100, {score_percent:.2f}%\n")
    systems_test: dict[str, object] = {
        "path": "third_party/systems_test",
        "commit": "1" * 40,
        "dirty": systems_test_dirty,
        "config_model": model,
        "command": (
            f"OPENAI_HOST={openai_host} SKIP_PROVISION=1 MMLU_MODEL={model} uv run mmlu_pro"
        ),
    }
    if include_systems_test_config:
        systems_test["config"] = {
            **write_json_artifact(
                root / "systems-test-config-row.json",
                {
                    "name": "synthetic_model",
                    "sample_name": "synthetic_model",
                    "model": model,
                    "nominal_users": 1,
                    "user_sets": [],
                },
            ),
            "source_path": "scripts/mmlu_pro.py",
            "model": model,
            "nominal_users": 1,
        }
    payload = {
        "schema": "ares.benchmark.mmlu_pro.v1",
        "evidence_class": "system_under_test",
        "status": "passed",
        "model": model,
        "backend": backend,
        "openai_host": openai_host,
        "coverage_percent": coverage_percent,
        "score_percent": score_percent,
        "required_score_percent": required_score_percent,
        "subjects": [
            {
                "subject": "total",
                "correct": score_percent,
                "wrong": 100 - score_percent,
                "score_percent": score_percent,
            }
        ],
        "systems_test": systems_test,
        "ares": {
            "commit": "2" * 40,
            "dirty": False,
            "backend": backend,
            "runtime_generated_sidecars": False,
            "ares_plan_sha256": SHA_A,
            "target_plan_sha256": SHA_B,
        },
        "artifacts": [{"path": "report.txt", "sha256": sha256_file(report)}],
    }
    if include_endpoint_models:
        payload["endpoint_models"] = {
            **write_json_artifact(
                root / "endpoint-models.json",
                {"data": [{"id": model, "object": "model"}]},
            ),
            "openai_host": openai_host,
            "models": [model],
        }
    return payload


def replay_context() -> dict:
    return {
        "context_tokens": [1, 2],
        "context_tokens_role": "prompt",
        "context_count": 2,
        "new_count": 1,
        "runtime_request_token_count": 2,
        "context_prefix_token_count": 0,
        "last_token": 2,
    }


class AresIngestArtifactTest(unittest.TestCase):
    def test_artifact_consistency_accepts_matching_model_ids(self) -> None:
        gate = artifact_consistency_gate(
            {"model": "synthetic/model"},
            oracle_payload=[{"model": {"model_id": "synthetic/model"}}],
            validated_gates={
                "targetplan_valid": {
                    "detail": {"model_id": "synthetic/model"},
                }
            },
        )

        self.assertTrue(gate["passed"])
        self.assertEqual(gate["artifact_validator"], "artifact_consistency")

    def test_artifact_consistency_rejects_target_plan_model_mismatch(self) -> None:
        gate = artifact_consistency_gate(
            {"model": "hf/model"},
            oracle_payload=[{"model": {"model_id": "hf/model"}}],
            validated_gates={
                "targetplan_valid": {
                    "detail": {"model_id": "fixture/model"},
                }
            },
        )

        self.assertFalse(gate["passed"])
        self.assertIn("TargetPlan model_id", " ".join(gate["errors"]))

    def test_artifact_consistency_accepts_explicit_model_aliases(self) -> None:
        gate = artifact_consistency_gate(
            {
                "model": "registry/model",
                "expected_model_ids": ["registry/model", "hf/model"],
            },
            oracle_payload=[{"model": {"model_id": "hf/model"}}],
            validated_gates={
                "targetplan_valid": {
                    "detail": {"model_id": "registry/model"},
                }
            },
        )

        self.assertTrue(gate["passed"])

    def test_backend_open_gate_accepts_jsonl_event_evidence(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "backend.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "event": "backend_open",
                        "status": "opened",
                        "backend_id": "tron",
                        "ares_plan_sha256": SHA_A,
                        "target_plan_sha256": SHA_B,
                        "target_plan_backend": "tron",
                        "runtime_generated_sidecars": False,
                    }
                )
                + "\n"
            )

            gate = backend_open_gate(path)

            self.assertTrue(gate["passed"])
            self.assertEqual(gate["artifact_validator"], "backend_open")
            self.assertEqual(gate["detail"]["backend_id"], "tron")

    def test_backend_open_gate_rejects_runtime_generated_sidecars(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "backend.json"
            path.write_text(
                json.dumps(
                    {
                        "status": "opened",
                        "backend_id": "tron",
                        "runtime_generated_sidecars": True,
                        "ares_plan": {"sha256": SHA_A},
                        "target_plan": {"sha256": SHA_B},
                    }
                )
            )

            gate = backend_open_gate(path)

            self.assertFalse(gate["passed"])
            self.assertIn("runtime-generated", " ".join(gate["errors"]))

    def test_backend_open_gate_requires_explicit_no_runtime_sidecars(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "backend.json"
            path.write_text(
                json.dumps(
                    {
                        "status": "opened",
                        "backend_id": "tron",
                        "ares_plan": {"sha256": SHA_A},
                        "target_plan": {"sha256": SHA_B, "backend_id": "tron"},
                    }
                )
            )

            gate = backend_open_gate(path)

            self.assertFalse(gate["passed"])
            self.assertIn(
                "runtime_generated_sidecars=false",
                " ".join(gate["errors"]),
            )

    def test_backend_open_gate_requires_explicit_target_backend(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "backend.json"
            path.write_text(
                json.dumps(
                    {
                        "status": "opened",
                        "backend_id": "tron",
                        "runtime_generated_sidecars": False,
                        "ares_plan": {"sha256": SHA_A},
                        "target_plan": {"sha256": SHA_B},
                    }
                )
            )

            gate = backend_open_gate(path)

            self.assertFalse(gate["passed"])
            self.assertIn("TargetPlan backend", " ".join(gate["errors"]))

    def test_backend_open_gate_rejects_nested_target_backend_mismatch(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "backend.json"
            path.write_text(
                json.dumps(
                    {
                        "status": "opened",
                        "backend_id": "tron",
                        "runtime_generated_sidecars": False,
                        "ares_plan": {"sha256": SHA_A},
                        "target_plan": {"sha256": SHA_B, "backend_id": "cpu"},
                    }
                )
            )

            gate = backend_open_gate(path)

            self.assertFalse(gate["passed"])
            self.assertIn("backend must match", " ".join(gate["errors"]))

    def test_trace_report_gate_accepts_machine_readable_report_json(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace-report.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "title": "Synthetic Ares Trace Report",
                        "summary": "diagnostic trace report",
                        "inputs": {
                            "metadata": "run.trace-meta.json",
                            "trace": "run.chrome.json",
                        },
                        "sections": {
                            "preflight": [{"status": "pass", "ok": 2, "warn": 0}],
                            "analysis_commands": [
                                {
                                    "purpose": "report",
                                    "command": "bin/ares-trace-report --format json",
                                }
                            ],
                            "report_grade": [
                                {
                                    "report_grade": "diagnostic",
                                    "proof_grade_status": "not_established_by_report",
                                }
                            ],
                            "report_triage": [
                                {
                                    "triage_status": "needs_measurement",
                                    "report_grade": "diagnostic",
                                    "proof_grade_status": ("not_established_by_report"),
                                    "first_blocked_gate": "hf_cpu_oracle",
                                    "first_blocked_gate_status": "blocked",
                                    "first_blocked_gate_basis": (
                                        "HF CPU oracle not recorded"
                                    ),
                                    "first_next_measurement_priority": (
                                        "backend_jsonl"
                                    ),
                                    "first_next_measurement_reason": (
                                        "backend JSONL evidence not present"
                                    ),
                                    "first_next_measurement": "Capture backend JSONL",
                                    "first_next_measurement_command_hint": (
                                        "set ARES_BACKEND_EVENT_ARTIFACT_DIR"
                                    ),
                                    "first_answerable_question": "",
                                    "first_unsupported_claim": (
                                        "backend JSONL evidence is unsupported"
                                    ),
                                    "first_useful_section": (
                                        "sections.next_measurements"
                                    ),
                                    "first_action": "Capture backend JSONL",
                                    "claim_boundary": (
                                        "diagnostic_routing_not_evidence"
                                    ),
                                }
                            ],
                            "supported_claims": [
                                {
                                    "claim": "trace preflight is answerable",
                                    "basis": "preflight section is present",
                                    "evidence_grade": "diagnostic",
                                }
                            ],
                            "answerability": [
                                {
                                    "question": "backend JSONL evidence",
                                    "status": "not_present",
                                    "basis": "missing backend_event_artifacts",
                                }
                            ],
                            "unsupported_claims": [
                                {
                                    "claim": "backend JSONL evidence is unsupported",
                                    "reason": "not_present",
                                }
                            ],
                            "next_measurements": [
                                {
                                    "priority": "backend_jsonl",
                                    "next_measurement": "Capture backend JSONL",
                                    "command_hint": "set ARES_BACKEND_EVENT_ARTIFACT_DIR",
                                }
                            ],
                            "correctness_evidence": [
                                {
                                    "evidence": "hf_cpu_oracle_tokens_logits",
                                    "status": "not_recorded",
                                    "evidence_role": "oracle_required_for_correctness",
                                    "proof_grade_status": "not_established_by_report",
                                    "basis": "no HF CPU oracle artifact attached",
                                    "next_gate": "attach HF CPU oracle evidence",
                                }
                            ],
                            "evidence_artifact_checks": [
                                {
                                    "check": "metadata.evidence_artifacts",
                                    "evidence_artifact": "<all>",
                                    "status": "ok",
                                    "detail": "0 artifact(s)",
                                }
                            ],
                            "promotion_gate_summary": [
                                {
                                    "gate": "capture_preflight",
                                    "status": "passed",
                                    "proof_grade_status": "not_established_by_report",
                                    "basis": "preflight passed for report inputs",
                                    "next_gate": "attach oracle evidence before promotion",
                                }
                            ],
                            "trace_mode_guardrails": [
                                {
                                    "trace_mode": "timeline-lite",
                                    "trace_sinks": "jsonl",
                                    "role": "report",
                                    "overhead_boundary": "low_overhead",
                                    "claim_guardrail": (
                                        "do not treat trace timing as promotion proof"
                                    ),
                                }
                            ],
                            "ab_provenance": [
                                {
                                    "role": "baseline",
                                    "binary": "target/debug/runares",
                                    "git_sha": SHA_A[:40],
                                    "worktree_dirty": "False",
                                    "source_state": "clean",
                                    "artifact_hash_count": 2,
                                    "artifact_hashes": "matched",
                                    "artifact_hash_status": "matched",
                                    "hardware_card_count": 1,
                                    "hardware_cards": "0000:01:00.0",
                                    "provenance_status": "clean_provenance",
                                    "proof_grade_status": ("not_established_by_report"),
                                    "basis": (
                                        "baseline binary, clean source state, "
                                        "artifact hashes, and hardware identity are recorded"
                                    ),
                                },
                                {
                                    "role": "candidate",
                                    "binary": "target/debug/runares",
                                    "git_sha": SHA_A[:40],
                                    "worktree_dirty": "False",
                                    "source_state": "clean",
                                    "artifact_hash_count": 2,
                                    "artifact_hashes": "matched",
                                    "artifact_hash_status": "matched",
                                    "hardware_card_count": 1,
                                    "hardware_cards": "0000:01:00.0",
                                    "provenance_status": "clean_provenance",
                                    "proof_grade_status": ("not_established_by_report"),
                                    "basis": (
                                        "candidate binary, clean source state, "
                                        "artifact hashes, and hardware identity are recorded"
                                    ),
                                },
                                {
                                    "role": "comparison",
                                    "binary": "",
                                    "git_sha": "",
                                    "worktree_dirty": "",
                                    "source_state": "",
                                    "artifact_hash_count": "",
                                    "artifact_hashes": "matched",
                                    "artifact_hash_status": "matched",
                                    "hardware_card_count": "",
                                    "hardware_cards": "matched",
                                    "provenance_status": "clean_provenance",
                                    "proof_grade_status": ("not_established_by_report"),
                                    "basis": (
                                        "baseline/candidate binary, clean source "
                                        "state, artifact hashes, and hardware "
                                        "identity are recorded"
                                    ),
                                },
                            ],
                            "ab_comparability": [
                                {
                                    "status": "comparison-grade",
                                    "basis": (
                                        "baseline/candidate preflight passed "
                                        "and metadata comparability checks passed"
                                    ),
                                    "promotion_gate": (
                                        "Still check correctness, provenance, "
                                        "repeatability, and external gates before "
                                        "proof-grade claims."
                                    ),
                                }
                            ],
                            "ab_coverage": [
                                {
                                    "align": "targetplan-op",
                                    "coverage_status": "partial_overlap",
                                    "total_rows": 5,
                                    "matched_rows": 4,
                                    "baseline_only_rows": 1,
                                    "candidate_only_rows": 0,
                                    "warnings": "",
                                    "basis": (
                                        "matched rows are present, but baseline-only "
                                        "or candidate-only rows must be reviewed as "
                                        "sample-count changes before interpreting deltas."
                                    ),
                                }
                            ],
                            "ab_repeatability": [
                                {
                                    "status": "insufficient_for_proof",
                                    "align": "targetplan-op",
                                    "baseline_runs": 1,
                                    "candidate_runs": 1,
                                    "required_matched_runs_for_hardware_proof": 3,
                                    "matched_rows": 4,
                                    "proof_grade_status": ("not_established_by_report"),
                                    "basis": (
                                        "hardware A/B proof requires at least three matched runs"
                                    ),
                                }
                            ],
                            "capture": [
                                {
                                    "metadata": "run.trace-meta.json",
                                    "trace": "run.chrome.json",
                                    "trace_run_id": "trace-run-001",
                                    "created_utc": "2026-07-01T00:00:00Z",
                                    "process_kind": "runares",
                                    "trace_mode": "timeline-lite",
                                    "trace_sinks": "jsonl",
                                    "model_id": "trace-model",
                                    "backend_id": "fpga",
                                    "target_plan_sha256": SHA_B,
                                    "worktree_dirty": "False",
                                }
                            ],
                            "run_provenance": [
                                {
                                    "binary": "target/debug/runares",
                                    "git_sha": SHA_A[:40],
                                    "worktree_dirty": "False",
                                    "source_state": "clean",
                                    "hardware_card_count": 0,
                                    "hardware_cards": "",
                                    "provenance_boundary": (
                                        "clean promotion evidence requires "
                                        "matched artifacts and hardware identity"
                                    ),
                                }
                            ],
                            "artifact_identities": [
                                {
                                    "artifact": "ares_plan",
                                    "path": "plan.ares.json",
                                    "sha256": SHA_A,
                                    "load_status": "loaded",
                                    "status_source": "trace_metadata",
                                },
                                {
                                    "artifact": "target_plan",
                                    "path": "plan.target.json",
                                    "sha256": SHA_B,
                                    "load_status": "loaded",
                                    "status_source": "backend_event_jsonl",
                                },
                            ],
                            "artifact_identity_checks": [
                                {
                                    "artifact": "<all>",
                                    "check": "metadata.artifacts",
                                    "status": "ok",
                                    "detail": "2 artifact(s)",
                                }
                            ],
                            "capture_capabilities": [
                                {"capability": "token_quality", "present": True},
                                {
                                    "capability": "device_result_digests",
                                    "present": False,
                                },
                            ],
                            "report_json_section_inventory": [
                                {
                                    "heading": "Trace Config Rows",
                                    "json_path": "sections.trace_config_rows",
                                    "json_section": "trace_config_rows",
                                    "section_kind": "capture_configuration",
                                    "claim_boundary": (
                                        "requested_controls_not_recorded_evidence"
                                    ),
                                },
                                {
                                    "heading": "Report Triage",
                                    "json_path": "sections.report_triage",
                                    "json_section": "report_triage",
                                    "section_kind": "measurement_guidance",
                                    "claim_boundary": (
                                        "diagnostic_routing_not_evidence"
                                    ),
                                },
                                {
                                    "heading": "A/B Provenance",
                                    "json_path": "sections.ab_provenance",
                                    "json_section": "ab_provenance",
                                    "section_kind": "comparison",
                                    "claim_boundary": "comparison_provenance",
                                },
                                {
                                    "heading": "A/B Comparability",
                                    "json_path": "sections.ab_comparability",
                                    "json_section": "ab_comparability",
                                    "section_kind": "comparison",
                                    "claim_boundary": "comparison_gate",
                                },
                                {
                                    "heading": "A/B Coverage",
                                    "json_path": "sections.ab_coverage",
                                    "json_section": "ab_coverage",
                                    "section_kind": "comparison",
                                    "claim_boundary": "comparison_coverage",
                                },
                                {
                                    "heading": "A/B Repeatability",
                                    "json_path": "sections.ab_repeatability",
                                    "json_section": "ab_repeatability",
                                    "section_kind": "comparison",
                                    "claim_boundary": "comparison_repeatability",
                                },
                                {
                                    "heading": "Introspection Capability Rows",
                                    "json_path": (
                                        "sections.introspection_capability_rows"
                                    ),
                                    "json_section": "introspection_capability_rows",
                                    "section_kind": "introspection",
                                    "claim_boundary": (
                                        "capability_presence_not_payload_evidence"
                                    ),
                                },
                                {
                                    "heading": "Introspection Artifacts",
                                    "json_path": "sections.introspection_artifacts",
                                    "json_section": "introspection_artifacts",
                                    "section_kind": "introspection",
                                    "claim_boundary": ("system_under_test_diagnostic"),
                                },
                                {
                                    "heading": "Introspection Section Inventory",
                                    "json_path": (
                                        "sections.introspection_section_inventory"
                                    ),
                                    "json_section": "introspection_section_inventory",
                                    "section_kind": "introspection_inventory",
                                    "claim_boundary": "introspection_section_discovery",
                                },
                                {
                                    "heading": (
                                        "Provider Payload Boundary Inventory Rows"
                                    ),
                                    "json_path": (
                                        "sections.provider_payload_boundary_inventory_rows"
                                    ),
                                    "json_section": (
                                        "provider_payload_boundary_inventory_rows"
                                    ),
                                    "section_kind": "introspection_inventory",
                                    "claim_boundary": (
                                        "payload_boundary_inventory_not_evidence"
                                    ),
                                },
                                {
                                    "heading": "Backend Event Artifacts",
                                    "json_path": "sections.backend_event_artifacts",
                                    "json_section": "backend_event_artifacts",
                                    "section_kind": "backend_diagnostic",
                                    "claim_boundary": "system_under_test_diagnostic",
                                },
                                {
                                    "heading": "Backend Event Rows",
                                    "json_path": "sections.backend_event_rows",
                                    "json_section": "backend_event_rows",
                                    "section_kind": "backend_diagnostic",
                                    "claim_boundary": "system_under_test_diagnostic",
                                },
                                {
                                    "heading": "Backend Provider Boundaries",
                                    "json_path": "sections.backend_provider_boundaries",
                                    "json_section": "backend_provider_boundaries",
                                    "section_kind": "backend_diagnostic",
                                    "claim_boundary": (
                                        "system_under_test_backend_provider_boundary"
                                    ),
                                },
                                {
                                    "heading": "Backend Fail-Closed Root Causes",
                                    "json_path": (
                                        "sections.backend_fail_closed_root_causes"
                                    ),
                                    "json_section": ("backend_fail_closed_root_causes"),
                                    "section_kind": "backend_diagnostic",
                                    "claim_boundary": (
                                        "system_under_test_backend_fail_closed_diagnostic"
                                    ),
                                },
                                {
                                    "heading": "Debug Payload Artifact Summary Rows",
                                    "json_path": (
                                        "sections.debug_payload_artifact_summary_rows"
                                    ),
                                    "json_section": (
                                        "debug_payload_artifact_summary_rows"
                                    ),
                                    "section_kind": "debug_payload_diagnostic",
                                    "claim_boundary": (
                                        "debug_payloads_can_perturb_timing"
                                    ),
                                },
                                {
                                    "heading": "Token Quality Summary Rows",
                                    "json_path": (
                                        "sections.token_quality_summary_rows"
                                    ),
                                    "json_section": "token_quality_summary_rows",
                                    "section_kind": "sidecar",
                                    "claim_boundary": (
                                        "system_under_test_diagnostic_not_oracle"
                                    ),
                                },
                                {
                                    "heading": "Oracle Reference Summary Rows",
                                    "json_path": (
                                        "sections.oracle_reference_summary_rows"
                                    ),
                                    "json_section": "oracle_reference_summary_rows",
                                    "section_kind": "sidecar",
                                    "claim_boundary": (
                                        "external_oracle_reference_anchor_not_sut_oracle_evidence"
                                    ),
                                },
                                {
                                    "heading": "Planning Decision Sidecar Rows",
                                    "json_path": (
                                        "sections.planning_decision_sidecar_rows"
                                    ),
                                    "json_section": "planning_decision_sidecar_rows",
                                    "section_kind": "sidecar",
                                    "claim_boundary": (
                                        "planning_decision_diagnostic_not_model_evidence"
                                    ),
                                },
                                {
                                    "heading": "Token Quality Sidecar Rows",
                                    "json_path": (
                                        "sections.token_quality_sidecar_rows"
                                    ),
                                    "json_section": "token_quality_sidecar_rows",
                                    "section_kind": "sidecar",
                                    "claim_boundary": (
                                        "system_under_test_diagnostic_not_oracle"
                                    ),
                                },
                                {
                                    "heading": "Top-K Token Sidecar Rows",
                                    "json_path": "sections.topk_token_sidecar_rows",
                                    "json_section": "topk_token_sidecar_rows",
                                    "section_kind": "sidecar",
                                    "claim_boundary": (
                                        "system_under_test_topk_diagnostic_not_oracle"
                                    ),
                                },
                                {
                                    "heading": "Tensor Payload Sidecar Rows",
                                    "json_path": "sections.tensor_payload_sidecar_rows",
                                    "json_section": "tensor_payload_sidecar_rows",
                                    "section_kind": "sidecar",
                                    "claim_boundary": (
                                        "system_under_test_payload_diagnostic"
                                    ),
                                },
                                {
                                    "heading": "K/V Payload Digest Sidecar Rows",
                                    "json_path": (
                                        "sections.kv_payload_digest_sidecar_rows"
                                    ),
                                    "json_section": "kv_payload_digest_sidecar_rows",
                                    "section_kind": "sidecar",
                                    "claim_boundary": (
                                        "system_under_test_scheduler_kv_payload_diagnostic"
                                    ),
                                },
                                {
                                    "heading": "Logit Slice Sidecar Rows",
                                    "json_path": "sections.logit_slice_sidecar_rows",
                                    "json_section": "logit_slice_sidecar_rows",
                                    "section_kind": "sidecar",
                                    "claim_boundary": (
                                        "system_under_test_final_logit_diagnostic"
                                    ),
                                },
                                {
                                    "heading": "Activation Digest Sidecar Rows",
                                    "json_path": (
                                        "sections.activation_digest_sidecar_rows"
                                    ),
                                    "json_section": "activation_digest_sidecar_rows",
                                    "section_kind": "sidecar",
                                    "claim_boundary": (
                                        "system_under_test_activation_digest_diagnostic"
                                    ),
                                },
                                {
                                    "heading": "Scheduler Packet Lineage Sidecar Rows",
                                    "json_path": (
                                        "sections.scheduler_packet_lineage_sidecar_rows"
                                    ),
                                    "json_section": (
                                        "scheduler_packet_lineage_sidecar_rows"
                                    ),
                                    "section_kind": "sidecar",
                                    "claim_boundary": (
                                        "system_under_test_scheduler_packet_diagnostic"
                                    ),
                                },
                                {
                                    "heading": (
                                        "Scheduler K/V Shard Lifecycle Sidecar Rows"
                                    ),
                                    "json_path": (
                                        "sections.scheduler_kv_shard_lifecycle_sidecar_rows"
                                    ),
                                    "json_section": (
                                        "scheduler_kv_shard_lifecycle_sidecar_rows"
                                    ),
                                    "section_kind": "sidecar",
                                    "claim_boundary": (
                                        "system_under_test_scheduler_kv_lifecycle_diagnostic"
                                    ),
                                },
                                {
                                    "heading": (
                                        "Scheduler Listener Sparse Logit Sidecar Rows"
                                    ),
                                    "json_path": (
                                        "sections.scheduler_listener_sparse_logit_sidecar_rows"
                                    ),
                                    "json_section": (
                                        "scheduler_listener_sparse_logit_sidecar_rows"
                                    ),
                                    "section_kind": "sidecar",
                                    "claim_boundary": (
                                        "system_under_test_scheduler_diagnostic"
                                    ),
                                },
                                {
                                    "heading": "Device DMA Lifecycle Sidecar Rows",
                                    "json_path": (
                                        "sections.device_dma_lifecycle_sidecar_rows"
                                    ),
                                    "json_section": (
                                        "device_dma_lifecycle_sidecar_rows"
                                    ),
                                    "section_kind": "sidecar",
                                    "claim_boundary": (
                                        "system_under_test_device_dma_diagnostic"
                                    ),
                                },
                                {
                                    "heading": "Attention Page Trace Sidecar Rows",
                                    "json_path": (
                                        "sections.attention_page_trace_sidecar_rows"
                                    ),
                                    "json_section": (
                                        "attention_page_trace_sidecar_rows"
                                    ),
                                    "section_kind": "sidecar",
                                    "claim_boundary": (
                                        "system_under_test_numeric_localization_diagnostic"
                                    ),
                                },
                                {
                                    "heading": "Next Measurements",
                                    "json_path": "sections.next_measurements",
                                    "json_section": "next_measurements",
                                    "section_kind": "measurement_guidance",
                                    "claim_boundary": "next_action_not_evidence",
                                },
                            ],
                            "trace_config_rows": [
                                {
                                    "config_status": "requested_and_recorded",
                                    "requested_sidecar_controls": "tensor_payloads",
                                    "recorded_sidecar_capabilities": "tensor_payloads",
                                    "missing_requested_sidecar_controls": "",
                                    "introspection_level": "payload",
                                    "compile_feature_trace_introspection": True,
                                    "deep_introspection_effective": True,
                                    "next_action": (
                                        "inspect_matching_introspection_report_sections"
                                    ),
                                },
                                {
                                    "config_status": (
                                        "requested_with_missing_sidecars"
                                    ),
                                    "requested_sidecar_controls": (
                                        "tensor_payloads,device_result_digests"
                                    ),
                                    "recorded_sidecar_capabilities": (
                                        "tensor_payloads"
                                    ),
                                    "missing_requested_sidecar_controls": (
                                        "device_result_digests"
                                    ),
                                    "introspection_level": "deep",
                                    "compile_feature_trace_introspection": True,
                                    "deep_introspection_effective": True,
                                    "next_action": ("enable_missing_sidecar_controls"),
                                },
                            ],
                            "provider_payload_boundary_inventory_rows": [
                                {
                                    "provider_id": "fpga",
                                    "payload_lane": "kv_payload_digests",
                                    "capture_status": "recorded_artifact",
                                    "artifact_count": 2,
                                    "matching_provider_artifact_count": 2,
                                    "artifact_kind_recorded_count": 2,
                                    "artifact_kind_recorded_backend_count": 1,
                                    "artifact_kind_recorded_backend_ids": "fpga",
                                    "report_section": "kv_payload_digest_sidecar_rows",
                                    "boundary_status": (
                                        "available_from_scheduler_protocol_boundary"
                                    ),
                                    "producer_status": "provider_callback_present",
                                    "producer_contract": (
                                        "fpga_scheduler_batch_dispatch_returns_"
                                        "completed_targetplan_listener_logits"
                                    ),
                                    "payload_record_policy": (
                                        "sha256_digest_plus_bounded_f32_sample"
                                    ),
                                    "payload_sensitivity": ("scheduler_kv_save_values"),
                                    "claim_boundary": (
                                        "system_under_test_scheduler_kv_payload_diagnostic"
                                    ),
                                    "next_action": "inspect_report_section",
                                },
                                {
                                    "provider_id": "generic",
                                    "payload_lane": "device_result_digests",
                                    "capture_status": (
                                        "route_available_no_provider_producer"
                                    ),
                                    "capture_capability": "device_result_digests",
                                    "artifact_kind": "device_result_digests",
                                    "capture_control": (
                                        "ARES_TRACE_RECORD_DEVICE_RESULTS=1"
                                    ),
                                    "artifact_count": 0,
                                    "matching_provider_artifact_count": 0,
                                    "artifact_kind_recorded_count": 1,
                                    "artifact_kind_recorded_backend_count": 1,
                                    "artifact_kind_recorded_backend_ids": "fpga",
                                    "report_section": (
                                        "device_result_digest_sidecar_rows"
                                    ),
                                    "boundary_status": (
                                        "route_available_no_provider_producer_yet"
                                    ),
                                    "producer_status": (
                                        "runtime_route_only_no_provider_producer"
                                    ),
                                    "producer_contract": "runtime_sidecar_route_only",
                                    "payload_record_policy": (
                                        "sha256_digest_plus_bounded_f32_sample"
                                    ),
                                    "payload_sensitivity": "device_result_values",
                                    "claim_boundary": (
                                        "system_under_test_device_result_digest_diagnostic"
                                    ),
                                    "next_action": (
                                        "wait_for_explicit_provider_payload_boundary"
                                    ),
                                },
                            ],
                            "trace_event_artifacts": [
                                {
                                    "index": 0,
                                    "path": "trace-events.jsonl",
                                    "sha256": SHA_C,
                                    "row_count": 3,
                                    "matching_trace_run_id_rows": 3,
                                    "event_kinds": "span_start,span_end",
                                    "status": "ok",
                                }
                            ],
                            "backend_event_artifacts": [
                                {
                                    "index": 0,
                                    "path": "backend-events.jsonl",
                                    "sha256": SHA_B,
                                    "row_count": 2,
                                    "matching_trace_run_id_rows": 2,
                                    "event_kinds": "backend_selected,forward_failed",
                                    "status": "ok",
                                }
                            ],
                            "backend_event_rows": [
                                {
                                    "artifact_index": 0,
                                    "row_index": 0,
                                    "timestamp_ns": "1000",
                                    "backend_id": "fpga",
                                    "event_kind": "backend_selected",
                                    "model_id": "trace-model",
                                    "request_id": "",
                                    "generation_id": "",
                                    "targetplan_op_id": "",
                                    "artifact": "backend-events.jsonl",
                                    "message": "backend selected",
                                    "metadata_keys": (
                                        "provider_stage,target_plan_validation_status"
                                    ),
                                },
                                {
                                    "artifact_index": 0,
                                    "row_index": 1,
                                    "timestamp_ns": "2000",
                                    "backend_id": "fpga",
                                    "event_kind": "forward_failed",
                                    "model_id": "trace-model",
                                    "request_id": "boundary-req-0",
                                    "generation_id": "boundary-gen-0",
                                    "targetplan_op_id": "tp.boundary.0",
                                    "artifact": "backend-events.jsonl",
                                    "message": "fpga target plan rejected",
                                    "metadata_keys": (
                                        "provider_stage,root_cause_stage,root_cause"
                                    ),
                                },
                            ],
                            "backend_provider_boundaries": [
                                {
                                    "artifact_index": 0,
                                    "row_index": 0,
                                    "timestamp_ns": "1000",
                                    "backend_id": "fpga",
                                    "event_kind": "backend_selected",
                                    "provider_stage": "session_open",
                                    "boundary_status": "ok",
                                    "root_cause_stage": "",
                                    "root_cause": "",
                                    "model_id": "trace-model",
                                    "request_id": "",
                                    "generation_id": "",
                                    "targetplan_op_id": "",
                                    "failure_reason": "",
                                    "message": "backend selected",
                                    "plan_artifact_status": "loaded",
                                    "target_plan_artifact_status": "loaded",
                                    "target_plan_validation_status": "accepted",
                                    "runtime_binding_status": "static_bindings_ok",
                                    "backend_descriptor_status": "supports_forward",
                                    "hardware_gate_status": "not_requested",
                                    "device_binding_status": "not_requested",
                                    "weight_policy_status": "packed_int4_gptq",
                                    "scheduler_targetplan_execution_step_bridge_status": "",
                                },
                                {
                                    "artifact_index": 0,
                                    "row_index": 1,
                                    "timestamp_ns": "2000",
                                    "backend_id": "fpga",
                                    "event_kind": "forward_failed",
                                    "provider_stage": "forward",
                                    "boundary_status": "fail_closed",
                                    "root_cause_stage": "targetplan_validation",
                                    "root_cause": (
                                        "target_plan_validation_status="
                                        "rejected_scheduler_runtime_table_missing"
                                    ),
                                    "model_id": "trace-model",
                                    "request_id": "boundary-req-0",
                                    "generation_id": "boundary-gen-0",
                                    "targetplan_op_id": "tp.boundary.0",
                                    "failure_reason": ("targetplan_validation_failed"),
                                    "message": "fpga target plan rejected",
                                    "plan_artifact_status": "loaded",
                                    "target_plan_artifact_status": "loaded",
                                    "target_plan_validation_status": (
                                        "rejected_scheduler_runtime_table_missing"
                                    ),
                                    "runtime_binding_status": "static_bindings_ok",
                                    "backend_descriptor_status": "supports_forward",
                                    "hardware_gate_status": "not_requested",
                                    "device_binding_status": "not_requested",
                                    "weight_policy_status": "packed_int4_gptq",
                                    "scheduler_targetplan_execution_step_bridge_status": "",
                                },
                            ],
                            "backend_fail_closed_root_causes": [
                                {
                                    "backend_id": "fpga",
                                    "provider_stage": "forward",
                                    "root_cause_stage": "targetplan_validation",
                                    "root_cause": (
                                        "target_plan_validation_status="
                                        "rejected_scheduler_runtime_table_missing"
                                    ),
                                    "event_kind": "forward_failed",
                                    "failure_count": 1,
                                    "example_model_id": "trace-model",
                                    "example_request_id": "boundary-req-0",
                                    "example_generation_id": "boundary-gen-0",
                                    "example_targetplan_op_id": "tp.boundary.0",
                                    "example_failure_reason": (
                                        "targetplan_validation_failed"
                                    ),
                                }
                            ],
                            "debug_payload_artifact_summary_rows": [
                                {
                                    "artifact_kind": "attention_page_trace",
                                    "payload_summary_status": "recorded",
                                    "row_count": "1",
                                    "byte_count": "918",
                                    "sampling_policy": (
                                        "selected attention page summaries"
                                    ),
                                    "token_window": "attention-page:7004",
                                    "sensitivity": "local-only",
                                    "compile_features": "trace-introspection",
                                    "report_section": (
                                        "attention_page_trace_sidecar_rows"
                                    ),
                                    "debug_payload_boundary": (
                                        "debug_payloads_can_perturb_timing"
                                    ),
                                    "claim_boundary": (
                                        "system_under_test_numeric_localization_diagnostic"
                                    ),
                                }
                            ],
                            "token_quality_summary_rows": [
                                {
                                    "status": "present",
                                    "evidence_role": "system_under_test",
                                    "request_id": "7001",
                                    "generation_id": "rinzler-7001",
                                    "token_index": 0,
                                    "selected_token_id": "42",
                                    "selected_topk_status": "selected_is_top1",
                                    "score_kind": "logprob",
                                    "top1_token_id": "42",
                                    "top1_score": "-0.1",
                                    "runner_up_token_id": "7",
                                    "runner_up_score": "-1.5",
                                    "top1_margin": "1.4",
                                    "temperature": "0.7",
                                    "top_p": "0.9",
                                    "top_k": "8",
                                    "num_logprobs": "2",
                                    "tokens_reused": "2",
                                    "runtime_request_token_count": "4",
                                    "oracle_reference": "external_hf_cpu_reference",
                                    "oracle_artifact_sha256": SHA_A,
                                    "claim_boundary": (
                                        "external_oracle_reference_present; row_remains_system_under_test"
                                    ),
                                }
                            ],
                            "oracle_reference_summary_rows": [
                                {
                                    "status": "present",
                                    "evidence_role": "system_under_test",
                                    "request_id": "7001",
                                    "generation_id": "rinzler-7001",
                                    "token_index": 0,
                                    "selected_token_id": "42",
                                    "oracle_reference_role": (
                                        "external_hf_cpu_reference"
                                    ),
                                    "hf_cpu_oracle_artifact_path": (
                                        "correctness_hf_cpu_oracle.txt"
                                    ),
                                    "hf_cpu_oracle_sha256": SHA_A,
                                    "expected_oracle_source": (
                                        "hf_transformers_pytorch_cpu"
                                    ),
                                    "oracle_reference_status": (
                                        "external_reference_hash_recorded"
                                    ),
                                    "sut_classification": "system_under_test",
                                    "correctness_claim_status": ("not_oracle_evidence"),
                                    "claim_boundary": (
                                        "external_hf_cpu_reference_anchor_only; "
                                        "token_quality_row_remains_system_under_test"
                                    ),
                                }
                            ],
                            "planning_decision_sidecar_rows": [
                                {
                                    "row_kind": "lean_planning_phase",
                                    "status": "present",
                                    "process_kind": "lean_ingest",
                                    "frontend": "hf-export",
                                    "target_backend": "fpga",
                                    "selection_source": "cli",
                                    "source": "ares-ingest-planning-introspection",
                                    "logical_command": "lake env lean",
                                    "dispatch_command": "bin/ares-ingest",
                                    "runner_count": "1",
                                    "exit_code": "0",
                                    "duration_us": "1200",
                                    "artifact_role": "targetplan",
                                    "artifact_kind": "target_plan_json",
                                    "artifact_path": "plans/fpga.target_plan.json",
                                    "artifact_sha256": SHA_B,
                                    "artifact_byte_count": "4096",
                                    "planning_phase": "lean.target_plan_lower",
                                    "event_name": "targetplan_lowered",
                                    "category": "planning",
                                    "start_ms": "1.5",
                                    "duration_ms": "2.5",
                                    "planning_output_bytes": "4096",
                                    "targetplan_op_count": "4",
                                    "claim_boundary": (
                                        "planning_decision_diagnostic_not_model_evidence"
                                    ),
                                }
                            ],
                            "token_quality_sidecar_rows": [
                                {
                                    "status": "present",
                                    "evidence_role": "system_under_test",
                                    "request_id": "7001",
                                    "generation_id": "rinzler-7001",
                                    "token_index": "0",
                                    "selected_token_id": "42",
                                    "topk_count": "2",
                                    "temperature": "0.7",
                                    "top_p": "0.9",
                                    "top_k": "8",
                                    "eos_policy": "stop_on_stop_token",
                                    "finish_reason": "stop",
                                    "oracle_reference": "external_hf_cpu_reference",
                                }
                            ],
                            "topk_token_sidecar_rows": [
                                {
                                    "status": "present",
                                    "evidence_role": "system_under_test",
                                    "request_id": "7001",
                                    "generation_id": "rinzler-7001",
                                    "token_index": "0",
                                    "selected_token_id": "42",
                                    "candidate_token_id": "42",
                                    "candidate_rank": "0",
                                    "candidate_score": "-0.1",
                                    "score_kind": "logprob",
                                    "selected_candidate_status": "selected_token",
                                    "temperature": "0.7",
                                    "top_p": "0.9",
                                    "top_k": "8",
                                    "oracle_reference": "external_hf_cpu_reference",
                                    "claim_boundary": (
                                        "external_oracle_reference_present; row_remains_system_under_test"
                                    ),
                                }
                            ],
                            "tensor_payload_sidecar_rows": [
                                {
                                    "status": "ok",
                                    "evidence_role": "system_under_test",
                                    "backend_id": "fpga",
                                    "request_id": "7005",
                                    "generation_id": "rinzler-7005",
                                    "token_index": 0,
                                    "targetplan_op_id": "tp.generic.0",
                                    "targetplan_action": "provider_payload",
                                    "layer": "31",
                                    "tensor_payload_kind": "tensor_payload",
                                    "tensor_name": "provider_payload",
                                    "tensor_role": "provider_device_payload",
                                    "element_type": "f32",
                                    "shape": "[2]",
                                    "element_count": "2",
                                    "digest_sha256": SHA_A,
                                    "sample_value_count": "2",
                                    "sample_min": "9.0",
                                    "sample_max": "10.0",
                                    "sample_nan_count": "0",
                                    "sample_pos_inf_count": "0",
                                    "sample_neg_inf_count": "0",
                                    "sample_values": "[9.0, 10.0]",
                                }
                            ],
                            "kv_payload_digest_sidecar_rows": [
                                {
                                    "status": "ok",
                                    "evidence_role": "system_under_test",
                                    "backend_id": "fpga",
                                    "request_id": "7006",
                                    "generation_id": "rinzler-7006",
                                    "token_index": 0,
                                    "targetplan_action": "kv_cache",
                                    "layer": "0",
                                    "tensor_payload_kind": "kv_payload_digest",
                                    "tensor_name": "scheduler_kv_save.layer_0.buf_1.k",
                                    "tensor_role": "kv_key",
                                    "element_type": "f32",
                                    "shape": "[16]",
                                    "element_count": "16",
                                    "digest_sha256": SHA_B,
                                    "sample_value_count": "4",
                                    "sample_min": "0.125",
                                    "sample_max": "0.5",
                                    "sample_nan_count": "0",
                                    "sample_pos_inf_count": "0",
                                    "sample_neg_inf_count": "0",
                                    "sample_values": "[0.125, 0.25, 0.375, 0.5]",
                                }
                            ],
                            "logit_slice_sidecar_rows": [
                                {
                                    "status": "ok",
                                    "evidence_role": "system_under_test",
                                    "backend_id": "fpga",
                                    "request_id": "7005",
                                    "generation_id": "rinzler-7005",
                                    "token_index": "0",
                                    "targetplan_op_id": "tp.logits.0",
                                    "targetplan_action": "final_logits",
                                    "layer": "31",
                                    "intrinsic": "topk",
                                    "tensor_payload_kind": "logit_slice",
                                    "tensor_name": "ares_logits",
                                    "tensor_role": "logits",
                                    "element_type": "f32",
                                    "shape": "[1, 32000]",
                                    "element_count": "32000",
                                    "digest_sha256": SHA_A,
                                    "sample_start": "7",
                                    "sample_stride": "1",
                                    "sample_value_count": "4",
                                    "sample_min": "-0.25",
                                    "sample_max": "0.5",
                                    "sample_nan_count": "1",
                                    "sample_pos_inf_count": "0",
                                    "sample_neg_inf_count": "1",
                                    "sample_values": (
                                        '[0.5, "-Infinity", "NaN", -0.25]'
                                    ),
                                }
                            ],
                            "activation_digest_sidecar_rows": [
                                {
                                    "status": "ok",
                                    "evidence_role": "system_under_test",
                                    "backend_id": "fpga",
                                    "request_id": "7007",
                                    "generation_id": "rinzler-7007",
                                    "token_index": "0",
                                    "targetplan_op_id": "tp.activation.0",
                                    "targetplan_action": "activation",
                                    "layer": "0",
                                    "intrinsic": "rmsnorm",
                                    "tensor_payload_kind": "activation_digest",
                                    "tensor_name": ("layer_0.mlp.down_proj.activation"),
                                    "tensor_role": "activation",
                                    "element_type": "f32",
                                    "shape": "[1, 4096]",
                                    "element_count": "4096",
                                    "digest_sha256": SHA_B,
                                    "sample_start": "0",
                                    "sample_stride": "8",
                                    "sample_value_count": "4",
                                    "sample_min": "-0.125",
                                    "sample_max": "0.5",
                                    "sample_nan_count": "0",
                                    "sample_pos_inf_count": "0",
                                    "sample_neg_inf_count": "0",
                                    "sample_values": "[-0.125, 0.0, 0.25, 0.5]",
                                }
                            ],
                            "device_result_digest_sidecar_rows": [
                                {
                                    "status": "ok",
                                    "evidence_role": "system_under_test",
                                    "backend_id": "fpga",
                                    "request_id": "7009",
                                    "generation_id": "rinzler-7009",
                                    "token_index": "0",
                                    "targetplan_op_id": "tp.device.result.0",
                                    "targetplan_action": "device_result",
                                    "layer": "2",
                                    "intrinsic": "fpga.device_result_digest",
                                    "tensor_payload_kind": "device_result_digest",
                                    "tensor_name": "fpga_device_result",
                                    "tensor_role": "device_result",
                                    "element_type": "bf16",
                                    "shape": "[4]",
                                    "element_count": "4",
                                    "digest_sha256": SHA_C,
                                    "sample_start": "0",
                                    "sample_stride": "1",
                                    "sample_value_count": "2",
                                    "sample_min": "1.25",
                                    "sample_max": "2.5",
                                    "sample_nan_count": "0",
                                    "sample_pos_inf_count": "0",
                                    "sample_neg_inf_count": "0",
                                    "sample_values": "[1.25, 2.5]",
                                }
                            ],
                            "scheduler_packet_lineage_sidecar_rows": [
                                {
                                    "status": "ok",
                                    "evidence_role": "system_under_test",
                                    "request_id": "7002",
                                    "generation_id": "rinzler-7002",
                                    "location_id": "4",
                                    "parent_location_id": "3",
                                    "executor_shape": (
                                        "fullscheduler_forward_batch_v1"
                                    ),
                                    "executor_status": (
                                        "executed_fullscheduler_forward_batch_v1"
                                    ),
                                    "attention_mode": "software_attention",
                                    "token_job_count": "2",
                                    "runtime_request_token_count": "2",
                                    "tokens_reused": "5",
                                    "visible_token_slots": "2",
                                    "kv_context_rows": "64",
                                    "kv_save_rows": "64",
                                    "kv_page_count": "1",
                                    "hw_shard_allocation_requests": "1",
                                    "hw_gof_page_infos": "1",
                                    "prior_host_gof_staging_status": (
                                        "page_info_published"
                                    ),
                                    "prior_host_gof_dma_completions": "1",
                                    "listener_sparse_rows": "1",
                                    "listener_sparse_tokens": "3",
                                }
                            ],
                            "scheduler_kv_shard_lifecycle_sidecar_rows": [
                                {
                                    "status": "ok",
                                    "evidence_role": "system_under_test",
                                    "request_id": "7002",
                                    "generation_id": "rinzler-7002",
                                    "location_id": "4",
                                    "parent_location_id": "3",
                                    "executor_shape": (
                                        "fullscheduler_forward_batch_v1"
                                    ),
                                    "executor_status": (
                                        "executed_fullscheduler_forward_batch_v1"
                                    ),
                                    "attention_mode": "software_attention",
                                    "kv_lifecycle_status": "observed",
                                    "token_job_count": "2",
                                    "kv_job_count": "1",
                                    "runtime_request_token_count": "2",
                                    "visible_token_slots": "2",
                                    "kv_context_rows": "64",
                                    "kv_save_rows": "64",
                                    "kv_page_count": "1",
                                    "hw_shard_allocation_requests": "1",
                                    "hw_gof_page_infos": "1",
                                    "prior_host_gof_staging_status": (
                                        "page_info_published"
                                    ),
                                    "prior_host_gof_dma_completions": "1",
                                }
                            ],
                            "scheduler_listener_sparse_logit_sidecar_rows": [
                                {
                                    "status": "ok",
                                    "listener_sparse_status": "observed",
                                    "evidence_role": "system_under_test",
                                    "request_id": "7002",
                                    "generation_id": "rinzler-7002",
                                    "location_id": "4",
                                    "executor_shape": (
                                        "fullscheduler_forward_batch_v1"
                                    ),
                                    "executor_status": (
                                        "executed_fullscheduler_forward_batch_v1"
                                    ),
                                    "attention_mode": "software_attention",
                                    "listener_sparse_rows": "1",
                                    "listener_sparse_tokens": "3",
                                    "sparse_topk_rows": "1",
                                    "sparse_topk_token_count": "3",
                                    "token_job_count": "2",
                                    "minibatch_count": "1",
                                    "runtime_request_token_count": "2",
                                    "tokens_reused": "5",
                                }
                            ],
                            "device_dma_lifecycle_sidecar_rows": [
                                {
                                    "status": "ok",
                                    "evidence_role": "system_under_test",
                                    "backend_id": "fpga",
                                    "request_id": "7002",
                                    "generation_id": "rinzler-7002",
                                    "targetplan_op_id": "tp.device.load_weights.0",
                                    "targetplan_action": "device_load",
                                    "location_id": "4",
                                    "device_stage": "dma_completion",
                                    "queue_id": "load_weight_q",
                                    "device_index": "0",
                                    "card_bus": "0xe1",
                                    "dma_direction": "host_to_device",
                                    "descriptor_count": "5",
                                    "byte_count": "65536",
                                    "counter_name": "rx_completions",
                                    "counter_value_delta": "5",
                                    "cacheblock_dma_shard_id": "9",
                                    "cacheblock_dma_gof_start_in_shard": "12",
                                    "cacheblock_dma_k_transfer_count": "4",
                                    "cacheblock_dma_v_transfer_count": "6",
                                    "cacheblock_dma_transfer_byte_count": "8192",
                                    "cacheblock_gof_start_position": "48",
                                    "cacheblock_k_word_checksum": "123456",
                                    "cacheblock_v_word_checksum": "654321",
                                    "queue_depth_before": "5",
                                    "queue_depth_after": "0",
                                }
                            ],
                            "attention_page_trace_sidecar_rows": [
                                {
                                    "status": "ok",
                                    "evidence_role": "system_under_test",
                                    "backend_id": "fpga",
                                    "request_id": "7004",
                                    "generation_id": "rinzler-7004",
                                    "targetplan_op_id": "tp.attention.0",
                                    "targetplan_action": "attention",
                                    "layer": "2",
                                    "head": "17",
                                    "kv_head": "4",
                                    "attention_row_index": "7",
                                    "batch": "8",
                                    "visible_tokens": "65",
                                    "page_start": "64",
                                    "page_count": "1",
                                    "page_v_count": "1",
                                    "scaled_score_count": "2",
                                    "exp_score_count": "2",
                                    "v_star_count": "1",
                                    "m_star": "3.0",
                                    "s_star": "5.0",
                                    "was_valid": "True",
                                }
                            ],
                            "introspection_capability_rows": [
                                {
                                    "capture_capability": "token_quality",
                                    "capability_status": "recorded",
                                    "matching_artifact_count": 1,
                                    "claim_boundary": (
                                        "system_under_test_diagnostic_not_oracle"
                                    ),
                                    "next_action": "inspect_token_quality_rows",
                                }
                            ],
                            "introspection_section_inventory": [
                                {
                                    "capture_capability": "token_quality",
                                    "artifact_kind": "token_quality",
                                    "heading": "Token Quality Summary Rows",
                                    "json_section": "token_quality_summary_rows",
                                    "capability_present": True,
                                    "artifact_count": 1,
                                    "section_status": "available",
                                    "claim_boundary": (
                                        "system_under_test_diagnostic_not_oracle"
                                    ),
                                },
                                {
                                    "capture_capability": "deep_introspection",
                                    "artifact_kind": "planning_decisions",
                                    "heading": "Planning Decision Sidecar Rows",
                                    "json_section": "planning_decision_sidecar_rows",
                                    "capability_present": True,
                                    "artifact_count": 1,
                                    "section_status": "available",
                                    "claim_boundary": (
                                        "planning_decision_diagnostic_not_model_evidence"
                                    ),
                                },
                                {
                                    "capture_capability": "token_quality",
                                    "artifact_kind": "token_quality",
                                    "heading": "Token Quality Sidecar Rows",
                                    "json_section": "token_quality_sidecar_rows",
                                    "capability_present": True,
                                    "artifact_count": 1,
                                    "section_status": "available",
                                    "claim_boundary": (
                                        "system_under_test_diagnostic_not_oracle"
                                    ),
                                },
                                {
                                    "capture_capability": "topk_rows",
                                    "artifact_kind": "token_quality",
                                    "heading": "Top-K Token Sidecar Rows",
                                    "json_section": "topk_token_sidecar_rows",
                                    "capability_present": True,
                                    "artifact_count": 1,
                                    "section_status": "available",
                                    "claim_boundary": (
                                        "system_under_test_topk_diagnostic_not_oracle"
                                    ),
                                },
                                {
                                    "capture_capability": "logit_slices",
                                    "artifact_kind": "logit_slices",
                                    "heading": "Logit Slice Sidecar Rows",
                                    "json_section": "logit_slice_sidecar_rows",
                                    "capability_present": True,
                                    "artifact_count": 1,
                                    "section_status": "available",
                                    "claim_boundary": (
                                        "system_under_test_final_logit_diagnostic"
                                    ),
                                },
                                {
                                    "capture_capability": "activation_digests",
                                    "artifact_kind": "activation_digests",
                                    "heading": "Activation Digest Sidecar Rows",
                                    "json_section": "activation_digest_sidecar_rows",
                                    "capability_present": True,
                                    "artifact_count": 1,
                                    "section_status": "available",
                                    "claim_boundary": (
                                        "system_under_test_activation_digest_diagnostic"
                                    ),
                                },
                                {
                                    "capture_capability": "scheduler_packet_lineage",
                                    "artifact_kind": "scheduler_packet_lineage",
                                    "heading": (
                                        "Scheduler Listener Sparse Logit Sidecar Rows"
                                    ),
                                    "json_section": (
                                        "scheduler_listener_sparse_logit_sidecar_rows"
                                    ),
                                    "capability_present": True,
                                    "artifact_count": 1,
                                    "section_status": "available",
                                    "claim_boundary": (
                                        "system_under_test_scheduler_diagnostic"
                                    ),
                                },
                            ],
                            "timeline_query_summary": [
                                {
                                    "section": "Join Key Coverage",
                                    "query": "join-key-coverage",
                                    "row_count": 3,
                                    "rendered_rows": 3,
                                    "native_sql": True,
                                    "status": "rendered",
                                    "portable_command": (
                                        "bin/ares-trace-query --query join-key-coverage"
                                    ),
                                    "native_sql_command": (
                                        "bin/ares-trace-sql --query join-key-coverage"
                                    ),
                                }
                            ],
                            "introspection_artifacts": [
                                {
                                    "index": 0,
                                    "kind": "token_quality",
                                    "artifact_kind": "token_quality_jsonl",
                                    "path": "introspection_token_quality.jsonl",
                                    "sha256": SHA_A,
                                    "status": "recorded",
                                    "row_count": "3",
                                    "byte_count": "120",
                                    "token_window": "0:3",
                                    "sampling_policy": (
                                        "selected token rows; top-k rows only when requested"
                                    ),
                                    "sensitivity": "local-only",
                                    "compile_features": (
                                        "trace-introspection,deep-trace"
                                    ),
                                },
                                {
                                    "index": 1,
                                    "kind": "tensor_payload",
                                    "artifact_kind": "tensor_payload_jsonl",
                                    "path": "introspection_tensor_payload.jsonl",
                                    "sha256": SHA_B,
                                    "status": "missing",
                                    "row_count": 2,
                                    "byte_count": 80,
                                    "token_window": "tensor-payload:7008",
                                    "sampling_policy": (
                                        "bounded tensor payload summaries only"
                                    ),
                                    "sensitivity": "tensor_digest",
                                    "compile_features": "trace-introspection",
                                },
                            ],
                            "introspection_artifact_summary_rows": [
                                {
                                    "artifact_kind": "token_quality",
                                    "summary_status": "recorded_and_locally_present",
                                    "artifact_count": 1,
                                    "local_present_count": 1,
                                    "local_missing_count": 0,
                                    "row_count_total": 1,
                                    "report_sections": (
                                        "oracle_reference_summary_rows,"
                                        "token_quality_sidecar_rows,"
                                        "token_quality_summary_rows,"
                                        "topk_token_sidecar_rows"
                                    ),
                                    "claim_boundaries": (
                                        "external_oracle_reference_anchor_not_"
                                        "sut_oracle_evidence,"
                                        "system_under_test_diagnostic_not_oracle,"
                                        "system_under_test_topk_diagnostic_not_oracle"
                                    ),
                                }
                            ],
                        },
                    }
                )
            )

            gate = trace_report_gate(path)

            self.assertTrue(gate["passed"])
            self.assertEqual(gate["artifact_validator"], "trace_report")
            self.assertEqual(gate["detail"]["report_grade"], "diagnostic")
            self.assertEqual(
                gate["detail"]["report_triage_status_counts"],
                {"needs_measurement": 1},
            )
            self.assertEqual(
                gate["detail"]["report_triage_samples"][0]["first_useful_section"],
                "sections.next_measurements",
            )
            self.assertEqual(
                gate["detail"]["report_triage_samples"][0]["claim_boundary"],
                "diagnostic_routing_not_evidence",
            )
            self.assertEqual(gate["detail"]["preflight_status"], "pass")
            self.assertEqual(gate["detail"]["supported_claim_count"], 1)
            self.assertEqual(gate["detail"]["unsupported_claim_count"], 1)
            self.assertEqual(gate["detail"]["next_measurement_count"], 1)
            self.assertEqual(
                gate["detail"]["correctness_evidence_status_counts"],
                {"not_recorded": 1},
            )
            self.assertEqual(
                gate["detail"]["correctness_evidence_proof_grade_status_counts"],
                {"not_established_by_report": 1},
            )
            self.assertEqual(
                gate["detail"]["evidence_artifact_check_status_counts"],
                {"ok": 1},
            )
            self.assertEqual(
                gate["detail"]["promotion_gate_summary_status_counts"],
                {"passed": 1},
            )
            self.assertEqual(
                gate["detail"]["promotion_gate_summary_proof_grade_status_counts"],
                {"not_established_by_report": 1},
            )
            self.assertEqual(
                gate["detail"]["trace_mode_guardrail_mode_counts"],
                {"timeline-lite": 1},
            )
            self.assertEqual(
                gate["detail"]["trace_mode_guardrail_overhead_counts"],
                {"low_overhead": 1},
            )
            self.assertEqual(gate["detail"]["ab_provenance_count"], 3)
            self.assertEqual(
                gate["detail"]["ab_provenance_status_counts"],
                {"clean_provenance": 3},
            )
            self.assertEqual(
                gate["detail"]["ab_provenance_source_state_counts"],
                {"clean": 2},
            )
            self.assertEqual(
                gate["detail"]["ab_provenance_artifact_hash_status_counts"],
                {"matched": 3},
            )
            self.assertEqual(
                gate["detail"]["ab_provenance_proof_grade_status_counts"],
                {"not_established_by_report": 3},
            )
            self.assertEqual(
                gate["detail"]["ab_comparability_status_counts"],
                {"comparison-grade": 1},
            )
            self.assertEqual(
                gate["detail"]["ab_coverage_status_counts"],
                {"partial_overlap": 1},
            )
            self.assertEqual(gate["detail"]["ab_coverage_total_rows"], 5)
            self.assertEqual(gate["detail"]["ab_coverage_matched_rows"], 4)
            self.assertEqual(gate["detail"]["ab_coverage_baseline_only_rows"], 1)
            self.assertEqual(gate["detail"]["ab_coverage_candidate_only_rows"], 0)
            self.assertEqual(
                gate["detail"]["ab_repeatability_status_counts"],
                {"insufficient_for_proof": 1},
            )
            self.assertEqual(
                gate["detail"]["ab_repeatability_proof_grade_status_counts"],
                {"not_established_by_report": 1},
            )
            self.assertEqual(gate["detail"]["ab_repeatability_baseline_runs"], 1)
            self.assertEqual(gate["detail"]["ab_repeatability_candidate_runs"], 1)
            self.assertEqual(
                gate["detail"][
                    "ab_repeatability_required_matched_runs_for_hardware_proof"
                ],
                3,
            )
            self.assertEqual(gate["detail"]["ab_repeatability_matched_rows"], 4)
            self.assertEqual(
                gate["detail"]["capture_process_kind_counts"],
                {"runares": 1},
            )
            self.assertEqual(
                gate["detail"]["capture_backend_counts"],
                {"fpga": 1},
            )
            self.assertEqual(
                gate["detail"]["capture_trace_mode_counts"],
                {"timeline-lite": 1},
            )
            self.assertEqual(
                gate["detail"]["run_provenance_source_state_counts"],
                {"clean": 1},
            )
            self.assertEqual(
                gate["detail"]["artifact_identity_artifact_counts"],
                {"ares_plan": 1, "target_plan": 1},
            )
            self.assertEqual(
                gate["detail"]["artifact_identity_load_status_counts"],
                {"loaded": 2},
            )
            self.assertEqual(
                gate["detail"]["artifact_identity_check_status_counts"],
                {"ok": 1},
            )
            self.assertEqual(
                gate["detail"]["capture_capability_present_counts"],
                {"False": 1, "True": 1},
            )
            self.assertEqual(
                gate["detail"]["introspection_capability_status_counts"],
                {"recorded": 1},
            )
            self.assertEqual(gate["detail"]["introspection_artifact_count"], 2)
            self.assertEqual(
                gate["detail"]["introspection_artifact_status_counts"],
                {"missing": 1, "recorded": 1},
            )
            self.assertEqual(
                gate["detail"]["introspection_artifact_kind_counts"],
                {"tensor_payload": 1, "token_quality": 1},
            )
            self.assertEqual(
                gate["detail"]["introspection_artifact_format_counts"],
                {"tensor_payload_jsonl": 1, "token_quality_jsonl": 1},
            )
            self.assertEqual(
                gate["detail"]["introspection_artifact_sensitivity_counts"],
                {"local-only": 1, "tensor_digest": 1},
            )
            self.assertEqual(
                gate["detail"]["introspection_artifact_compile_feature_counts"],
                {"deep-trace": 1, "trace-introspection": 2},
            )
            self.assertEqual(
                gate["detail"]["introspection_artifact_row_count_total"],
                5,
            )
            self.assertEqual(
                gate["detail"]["introspection_artifact_byte_count_total"],
                200,
            )
            self.assertEqual(
                gate["detail"]["introspection_artifact_samples"][0]["kind"],
                "token_quality",
            )
            self.assertEqual(
                gate["detail"]["introspection_artifact_samples"][0]["path"],
                "introspection_token_quality.jsonl",
            )
            self.assertEqual(
                gate["detail"]["introspection_artifact_samples"][1]["byte_count"],
                "80",
            )
            self.assertEqual(
                gate["detail"]["introspection_artifact_summary_status_counts"],
                {"recorded_and_locally_present": 1},
            )
            self.assertEqual(
                gate["detail"]["introspection_section_inventory_status_counts"],
                {"available": 7},
            )
            self.assertEqual(
                gate["detail"]["introspection_section_inventory_capability_counts"],
                {
                    "activation_digests": 1,
                    "deep_introspection": 1,
                    "logit_slices": 1,
                    "scheduler_packet_lineage": 1,
                    "token_quality": 2,
                    "topk_rows": 1,
                },
            )
            self.assertEqual(
                gate["detail"]["trace_config_status_counts"],
                {
                    "requested_and_recorded": 1,
                    "requested_with_missing_sidecars": 1,
                },
            )
            self.assertEqual(
                gate["detail"]["trace_config_missing_requested_sidecar_counts"],
                {"device_result_digests": 1},
            )
            self.assertEqual(
                gate["detail"]["provider_payload_boundary_status_counts"],
                {
                    "recorded_artifact": 1,
                    "route_available_no_provider_producer": 1,
                },
            )
            self.assertEqual(
                gate["detail"]["provider_payload_boundary_route_only_count"],
                1,
            )
            self.assertEqual(
                gate["detail"]["provider_payload_boundary_route_only_lanes"],
                ["generic/device_result_digests"],
            )
            self.assertEqual(
                gate["detail"]["provider_payload_boundary_recorded_count"],
                1,
            )
            self.assertEqual(
                gate["detail"]["provider_payload_boundary_recorded_lanes"],
                ["fpga/kv_payload_digests"],
            )
            self.assertEqual(
                gate["detail"]["trace_event_artifact_status_counts"],
                {"ok": 1},
            )
            self.assertEqual(
                gate["detail"]["trace_event_artifact_event_kind_counts"],
                {"span_end": 1, "span_start": 1},
            )
            self.assertEqual(
                gate["detail"]["backend_event_artifact_status_counts"],
                {"ok": 1},
            )
            self.assertEqual(
                gate["detail"]["backend_event_artifact_event_kind_counts"],
                {"backend_selected": 1, "forward_failed": 1},
            )
            self.assertEqual(
                gate["detail"]["backend_event_row_event_kind_counts"],
                {"backend_selected": 1, "forward_failed": 1},
            )
            self.assertEqual(
                gate["detail"]["backend_event_row_backend_counts"],
                {"fpga": 2},
            )
            self.assertEqual(
                gate["detail"]["backend_provider_boundary_status_counts"],
                {"fail_closed": 1, "ok": 1},
            )
            self.assertEqual(
                gate["detail"]["backend_provider_boundary_stage_counts"],
                {"forward": 1, "session_open": 1},
            )
            self.assertEqual(
                gate["detail"]["backend_provider_boundary_root_stage_counts"],
                {"targetplan_validation": 1},
            )
            self.assertEqual(
                gate["detail"]["timeline_query_summary_status_counts"],
                {"rendered": 1},
            )
            self.assertEqual(
                gate["detail"]["capture_samples"][0]["trace_run_id"],
                "trace-run-001",
            )
            self.assertEqual(
                gate["detail"]["run_provenance_samples"][0]["source_state"],
                "clean",
            )
            self.assertEqual(
                gate["detail"]["artifact_identity_samples"][1]["artifact"],
                "target_plan",
            )
            self.assertEqual(
                gate["detail"]["trace_event_artifact_samples"][0]["event_kinds"],
                "span_start,span_end",
            )
            self.assertEqual(
                gate["detail"]["timeline_query_summary_samples"][0]["query"],
                "join-key-coverage",
            )
            self.assertEqual(
                gate["detail"]["ab_provenance_samples"][2]["role"],
                "comparison",
            )
            self.assertEqual(
                gate["detail"]["ab_comparability_samples"][0]["status"],
                "comparison-grade",
            )
            self.assertEqual(
                gate["detail"]["ab_coverage_samples"][0]["align"],
                "targetplan-op",
            )
            self.assertEqual(
                gate["detail"]["ab_repeatability_samples"][0]["proof_grade_status"],
                "not_established_by_report",
            )
            self.assertEqual(
                gate["detail"]["backend_fail_closed_root_cause_backend_counts"],
                {"fpga": 1},
            )
            self.assertEqual(
                gate["detail"]["backend_fail_closed_root_cause_stage_counts"],
                {"forward": 1},
            )
            self.assertEqual(
                gate["detail"]["backend_fail_closed_root_cause_root_stage_counts"],
                {"targetplan_validation": 1},
            )
            self.assertEqual(
                gate["detail"]["debug_payload_artifact_summary_status_counts"],
                {"recorded": 1},
            )
            self.assertEqual(
                gate["detail"]["token_quality_summary_status_counts"],
                {"present": 1},
            )
            self.assertEqual(
                gate["detail"]["token_quality_summary_topk_status_counts"],
                {"selected_is_top1": 1},
            )
            self.assertEqual(
                gate["detail"]["oracle_reference_summary_status_counts"],
                {"external_reference_hash_recorded": 1},
            )
            self.assertEqual(
                gate["detail"]["oracle_reference_summary_correctness_counts"],
                {"not_oracle_evidence": 1},
            )
            self.assertEqual(
                gate["detail"]["planning_decision_sidecar_row_kind_counts"],
                {"lean_planning_phase": 1},
            )
            self.assertEqual(
                gate["detail"]["planning_decision_sidecar_phase_counts"],
                {"lean.target_plan_lower": 1},
            )
            self.assertEqual(
                gate["detail"]["token_quality_sidecar_status_counts"],
                {"present": 1},
            )
            self.assertEqual(
                gate["detail"]["token_quality_sidecar_finish_reason_counts"],
                {"stop": 1},
            )
            self.assertEqual(
                gate["detail"]["topk_token_sidecar_selected_status_counts"],
                {"selected_token": 1},
            )
            self.assertEqual(
                gate["detail"]["topk_token_sidecar_score_kind_counts"],
                {"logprob": 1},
            )
            self.assertEqual(
                gate["detail"]["tensor_payload_sidecar_status_counts"],
                {"ok": 1},
            )
            self.assertEqual(
                gate["detail"]["tensor_payload_sidecar_kind_counts"],
                {"tensor_payload": 1},
            )
            self.assertEqual(
                gate["detail"]["tensor_payload_sidecar_role_counts"],
                {"provider_device_payload": 1},
            )
            self.assertEqual(
                gate["detail"]["kv_payload_digest_sidecar_status_counts"],
                {"ok": 1},
            )
            self.assertEqual(
                gate["detail"]["kv_payload_digest_sidecar_role_counts"],
                {"kv_key": 1},
            )
            self.assertEqual(
                gate["detail"]["logit_slice_sidecar_role_counts"],
                {"logits": 1},
            )
            self.assertEqual(
                gate["detail"]["logit_slice_sidecar_action_counts"],
                {"final_logits": 1},
            )
            self.assertEqual(
                gate["detail"]["activation_digest_sidecar_role_counts"],
                {"activation": 1},
            )
            self.assertEqual(
                gate["detail"]["activation_digest_sidecar_intrinsic_counts"],
                {"rmsnorm": 1},
            )
            self.assertEqual(
                gate["detail"]["device_result_digest_sidecar_status_counts"],
                {"ok": 1},
            )
            self.assertEqual(
                gate["detail"]["device_result_digest_sidecar_role_counts"],
                {"device_result": 1},
            )
            self.assertEqual(
                gate["detail"]["device_result_digest_sidecar_action_counts"],
                {"device_result": 1},
            )
            self.assertEqual(
                gate["detail"]["device_result_digest_sidecar_intrinsic_counts"],
                {"fpga.device_result_digest": 1},
            )
            self.assertEqual(
                gate["detail"]["scheduler_packet_lineage_sidecar_status_counts"],
                {"ok": 1},
            )
            self.assertEqual(
                gate["detail"]["scheduler_packet_lineage_sidecar_executor_counts"],
                {"executed_fullscheduler_forward_batch_v1": 1},
            )
            self.assertEqual(
                gate["detail"]["scheduler_kv_shard_lifecycle_sidecar_status_counts"],
                {"ok": 1},
            )
            self.assertEqual(
                gate["detail"]["scheduler_kv_shard_lifecycle_sidecar_lifecycle_counts"],
                {"observed": 1},
            )
            self.assertEqual(
                gate["detail"][
                    "scheduler_listener_sparse_logit_sidecar_listener_status_counts"
                ],
                {"observed": 1},
            )
            self.assertEqual(
                gate["detail"][
                    "scheduler_listener_sparse_logit_sidecar_executor_counts"
                ],
                {"executed_fullscheduler_forward_batch_v1": 1},
            )
            self.assertEqual(
                gate["detail"]["device_dma_lifecycle_sidecar_status_counts"],
                {"ok": 1},
            )
            self.assertEqual(
                gate["detail"]["device_dma_lifecycle_sidecar_stage_counts"],
                {"dma_completion": 1},
            )
            self.assertEqual(
                gate["detail"]["device_dma_lifecycle_sidecar_queue_counts"],
                {"load_weight_q": 1},
            )
            self.assertEqual(
                gate["detail"]["attention_page_trace_sidecar_status_counts"],
                {"ok": 1},
            )
            self.assertEqual(
                gate["detail"]["attention_page_trace_sidecar_action_counts"],
                {"attention": 1},
            )
            self.assertEqual(gate["detail"]["report_json_section_count"], 30)
            self.assertEqual(
                gate["detail"]["report_json_section_kind_counts"],
                {
                    "backend_diagnostic": 4,
                    "capture_configuration": 1,
                    "comparison": 4,
                    "debug_payload_diagnostic": 1,
                    "introspection": 2,
                    "introspection_inventory": 2,
                    "measurement_guidance": 2,
                    "sidecar": 14,
                },
            )
            self.assertEqual(
                gate["detail"]["report_json_section_samples"][0]["json_path"],
                "sections.trace_config_rows",
            )
            self.assertEqual(
                gate["detail"]["trace_config_samples"][0]["requested_sidecar_controls"],
                "tensor_payloads",
            )
            self.assertEqual(
                gate["detail"]["trace_config_samples"][1][
                    "missing_requested_sidecar_controls"
                ],
                "device_result_digests",
            )
            self.assertEqual(
                gate["detail"]["provider_payload_boundary_samples"][0]["provider_id"],
                "fpga",
            )
            self.assertEqual(
                gate["detail"]["provider_payload_boundary_samples"][0][
                    "matching_provider_artifact_count"
                ],
                "2",
            )
            self.assertEqual(
                gate["detail"]["provider_payload_boundary_samples"][0][
                    "artifact_kind_recorded_backend_ids"
                ],
                "fpga",
            )
            self.assertEqual(
                gate["detail"]["provider_payload_boundary_samples"][0][
                    "producer_status"
                ],
                "provider_callback_present",
            )
            self.assertEqual(
                gate["detail"]["provider_payload_boundary_samples"][0][
                    "payload_record_policy"
                ],
                "sha256_digest_plus_bounded_f32_sample",
            )
            self.assertEqual(
                gate["detail"]["provider_payload_boundary_samples"][0][
                    "payload_sensitivity"
                ],
                "scheduler_kv_save_values",
            )
            self.assertEqual(
                gate["detail"]["provider_payload_boundary_route_only_samples"][0][
                    "payload_lane"
                ],
                "device_result_digests",
            )
            self.assertEqual(
                gate["detail"]["provider_payload_boundary_route_only_samples"][0][
                    "producer_contract"
                ],
                "runtime_sidecar_route_only",
            )
            self.assertEqual(
                gate["detail"]["provider_payload_boundary_route_only_samples"][0][
                    "capture_control"
                ],
                "ARES_TRACE_RECORD_DEVICE_RESULTS=1",
            )
            self.assertEqual(
                gate["detail"]["provider_payload_boundary_route_only_samples"][0][
                    "matching_provider_artifact_count"
                ],
                "0",
            )
            self.assertEqual(
                gate["detail"]["provider_payload_boundary_recorded_samples"][0][
                    "payload_lane"
                ],
                "kv_payload_digests",
            )
            self.assertEqual(
                gate["detail"]["provider_payload_boundary_recorded_samples"][0][
                    "producer_contract"
                ],
                "fpga_scheduler_batch_dispatch_returns_completed_targetplan_listener_logits",
            )
            self.assertEqual(
                gate["detail"]["provider_payload_boundary_recorded_samples"][0][
                    "matching_provider_artifact_count"
                ],
                "2",
            )
            self.assertEqual(
                gate["detail"]["backend_event_artifact_samples"][0]["path"],
                "backend-events.jsonl",
            )
            self.assertEqual(
                gate["detail"]["backend_event_artifact_samples"][0][
                    "matching_trace_run_id_rows"
                ],
                "2",
            )
            self.assertEqual(
                gate["detail"]["backend_event_samples"][1]["event_kind"],
                "forward_failed",
            )
            self.assertEqual(
                gate["detail"]["backend_event_samples"][1]["metadata_keys"],
                "provider_stage,root_cause_stage,root_cause",
            )
            self.assertEqual(
                gate["detail"]["backend_provider_boundary_samples"][1][
                    "boundary_status"
                ],
                "fail_closed",
            )
            self.assertEqual(
                gate["detail"]["backend_provider_boundary_samples"][1][
                    "root_cause_stage"
                ],
                "targetplan_validation",
            )
            self.assertEqual(
                gate["detail"]["backend_provider_boundary_samples"][1][
                    "targetplan_op_id"
                ],
                "tp.boundary.0",
            )
            self.assertEqual(
                gate["detail"]["backend_fail_closed_root_cause_samples"][0][
                    "root_cause"
                ],
                "target_plan_validation_status=rejected_scheduler_runtime_table_missing",
            )
            self.assertEqual(
                gate["detail"]["backend_fail_closed_root_cause_samples"][0][
                    "example_failure_reason"
                ],
                "targetplan_validation_failed",
            )
            self.assertEqual(
                gate["detail"]["debug_payload_artifact_summary_samples"][0][
                    "artifact_kind"
                ],
                "attention_page_trace",
            )
            self.assertEqual(
                gate["detail"]["debug_payload_artifact_summary_samples"][0][
                    "debug_payload_boundary"
                ],
                "debug_payloads_can_perturb_timing",
            )
            self.assertEqual(
                gate["detail"]["token_quality_summary_samples"][0][
                    "selected_topk_status"
                ],
                "selected_is_top1",
            )
            self.assertEqual(
                gate["detail"]["token_quality_summary_samples"][0]["claim_boundary"],
                "external_oracle_reference_present; row_remains_system_under_test",
            )
            self.assertEqual(
                gate["detail"]["oracle_reference_summary_samples"][0][
                    "correctness_claim_status"
                ],
                "not_oracle_evidence",
            )
            self.assertEqual(
                gate["detail"]["oracle_reference_summary_samples"][0]["claim_boundary"],
                (
                    "external_hf_cpu_reference_anchor_only; "
                    "token_quality_row_remains_system_under_test"
                ),
            )
            self.assertEqual(
                gate["detail"]["planning_decision_sidecar_samples"][0][
                    "planning_phase"
                ],
                "lean.target_plan_lower",
            )
            self.assertEqual(
                gate["detail"]["planning_decision_sidecar_samples"][0][
                    "targetplan_op_count"
                ],
                "4",
            )
            self.assertEqual(
                gate["detail"]["token_quality_sidecar_samples"][0]["finish_reason"],
                "stop",
            )
            self.assertEqual(
                gate["detail"]["topk_token_sidecar_samples"][0][
                    "selected_candidate_status"
                ],
                "selected_token",
            )
            self.assertEqual(
                gate["detail"]["tensor_payload_sidecar_samples"][0][
                    "tensor_payload_kind"
                ],
                "tensor_payload",
            )
            self.assertEqual(
                gate["detail"]["tensor_payload_sidecar_samples"][0]["tensor_role"],
                "provider_device_payload",
            )
            self.assertEqual(
                gate["detail"]["kv_payload_digest_sidecar_samples"][0]["tensor_role"],
                "kv_key",
            )
            self.assertEqual(
                gate["detail"]["kv_payload_digest_sidecar_samples"][0]["digest_sha256"],
                SHA_B,
            )
            self.assertEqual(
                gate["detail"]["logit_slice_sidecar_samples"][0]["sample_nan_count"],
                "1",
            )
            self.assertEqual(
                gate["detail"]["activation_digest_sidecar_samples"][0]["intrinsic"],
                "rmsnorm",
            )
            self.assertEqual(
                gate["detail"]["device_result_digest_sidecar_samples"][0]["request_id"],
                "7009",
            )
            self.assertEqual(
                gate["detail"]["device_result_digest_sidecar_samples"][0][
                    "tensor_name"
                ],
                "fpga_device_result",
            )
            self.assertEqual(
                gate["detail"]["device_result_digest_sidecar_samples"][0][
                    "digest_sha256"
                ],
                SHA_C,
            )
            self.assertEqual(
                gate["detail"]["device_result_digest_sidecar_samples"][0]["sample_min"],
                "1.25",
            )
            self.assertEqual(
                gate["detail"]["scheduler_packet_lineage_sidecar_samples"][0][
                    "executor_status"
                ],
                "executed_fullscheduler_forward_batch_v1",
            )
            self.assertEqual(
                gate["detail"]["scheduler_packet_lineage_sidecar_samples"][0][
                    "listener_sparse_rows"
                ],
                "1",
            )
            self.assertEqual(
                gate["detail"]["scheduler_kv_shard_lifecycle_sidecar_samples"][0][
                    "kv_lifecycle_status"
                ],
                "observed",
            )
            self.assertEqual(
                gate["detail"]["scheduler_kv_shard_lifecycle_sidecar_samples"][0][
                    "kv_context_rows"
                ],
                "64",
            )
            self.assertEqual(
                gate["detail"]["scheduler_listener_sparse_logit_sidecar_samples"][0][
                    "listener_sparse_tokens"
                ],
                "3",
            )
            self.assertEqual(
                gate["detail"]["device_dma_lifecycle_sidecar_samples"][0][
                    "device_stage"
                ],
                "dma_completion",
            )
            self.assertEqual(
                gate["detail"]["device_dma_lifecycle_sidecar_samples"][0][
                    "cacheblock_dma_transfer_byte_count"
                ],
                "8192",
            )
            self.assertEqual(
                gate["detail"]["attention_page_trace_sidecar_samples"][0][
                    "attention_row_index"
                ],
                "7",
            )
            self.assertEqual(
                gate["detail"]["attention_page_trace_sidecar_samples"][0][
                    "visible_tokens"
                ],
                "65",
            )
            self.assertEqual(
                gate["detail"]["introspection_capability_samples"][0][
                    "matching_artifact_count"
                ],
                "1",
            )
            self.assertEqual(
                gate["detail"]["introspection_section_inventory_samples"][0][
                    "json_section"
                ],
                "token_quality_summary_rows",
            )
            self.assertEqual(
                gate["detail"]["introspection_section_inventory_samples"][0][
                    "section_status"
                ],
                "available",
            )
            self.assertEqual(
                gate["detail"]["introspection_section_inventory_samples"][1][
                    "section_status"
                ],
                "available",
            )
            self.assertEqual(len(gate["detail"]["sha256"]), 64)

    def test_trace_report_gate_accepts_real_ares_trace_report_fixture(self) -> None:
        path = FIXTURE_DIR / "ares_trace_report_introspection_real.json"

        gate = trace_report_gate(path)

        self.assertTrue(gate["passed"], gate.get("errors"))
        self.assertEqual(gate["artifact_validator"], "trace_report")
        self.assertEqual(gate["detail"]["report_grade"], "diagnostic")
        self.assertIn(
            "preflight passed; no complete baseline/candidate comparison inputs",
            gate["detail"]["report_grade_basis"],
        )
        self.assertIn(
            "Capture comparable baseline/candidate artifacts",
            gate["detail"]["report_grade_promotion_gate"],
        )
        self.assertEqual(gate["detail"]["preflight_status"], "pass")
        self.assertEqual(gate["detail"]["preflight_ok_count"], 41)
        self.assertEqual(gate["detail"]["preflight_warn_count"], 1)
        self.assertEqual(gate["detail"]["preflight_fail_count"], 0)
        self.assertEqual(gate["detail"]["section_count"], 53)
        self.assertIn("preflight", gate["detail"]["section_names"])
        self.assertIn("report_grade", gate["detail"]["section_names"])
        self.assertIn("report_triage", gate["detail"]["section_names"])
        self.assertEqual(
            gate["detail"]["report_triage_status_counts"],
            {"needs_measurement": 1},
        )
        self.assertEqual(
            gate["detail"]["report_triage_samples"][0][
                "first_next_measurement_priority"
            ],
            "timeline_capture",
        )
        self.assertEqual(
            gate["detail"]["report_triage_samples"][0]["first_useful_section"],
            "sections.next_measurements",
        )
        self.assertEqual(
            gate["detail"]["answerability_samples"][0]["question"],
            "metadata artifact identity",
        )
        self.assertEqual(
            gate["detail"]["answerability_samples"][0]["status"],
            "not_present",
        )
        self.assertIn(
            "metadata.artifacts: 0 row(s)",
            gate["detail"]["answerability_samples"][0]["basis"],
        )
        self.assertEqual(gate["detail"]["report_json_section_count"], 53)
        report_json_paths = {
            sample["json_path"]
            for sample in gate["detail"]["report_json_section_samples"]
        }
        self.assertIn("sections.preflight", report_json_paths)
        self.assertIn("sections.analysis_commands", report_json_paths)
        self.assertIn("sections.report_grade", report_json_paths)
        self.assertIn("sections.report_triage", report_json_paths)
        self.assertIn("sections.report_json_section_inventory", report_json_paths)
        self.assertEqual(gate["detail"]["preflight_finding_count"], 1)
        self.assertEqual(
            gate["detail"]["preflight_finding_kind_counts"],
            {"warn": 1},
        )
        self.assertIn(
            "metadata.device_counters",
            gate["detail"]["preflight_finding_samples"][0],
        )
        self.assertEqual(gate["detail"]["evidence_classification_count"], 1)
        self.assertEqual(
            gate["detail"]["evidence_classification_kind_counts"],
            {"diagnostic": 1},
        )
        self.assertIn(
            "diagnostic: preflight passed",
            gate["detail"]["evidence_classification_samples"][0],
        )
        self.assertEqual(gate["detail"]["report_section_inventory_count"], 66)
        self.assertEqual(
            gate["detail"]["report_section_inventory_native_sql_counts"],
            {"True": 66},
        )
        self.assertEqual(
            gate["detail"]["report_section_inventory_samples"][0]["heading"],
            "Run Summary",
        )
        self.assertEqual(
            gate["detail"]["report_section_inventory_samples"][0]["query"],
            "run-summary",
        )
        self.assertEqual(
            gate["detail"]["report_section_inventory_samples"][0]["native_sql"],
            "True",
        )
        self.assertEqual(
            gate["detail"]["capture_process_kind_counts"],
            {"test": 1},
        )
        self.assertEqual(
            gate["detail"]["capture_backend_counts"],
            {"fpga": 1},
        )
        self.assertEqual(
            gate["detail"]["capture_trace_mode_counts"],
            {"debug-heavy": 1},
        )
        self.assertEqual(
            gate["detail"]["run_provenance_source_state_counts"],
            {"unknown": 1},
        )
        self.assertEqual(
            gate["detail"]["artifact_identity_check_status_counts"],
            {"ok": 1},
        )
        self.assertEqual(
            gate["detail"]["capture_capability_present_counts"],
            {"False": 2, "True": 15},
        )
        self.assertEqual(gate["detail"]["introspection_artifact_count"], 9)
        self.assertEqual(
            gate["detail"]["introspection_artifact_status_counts"],
            {"recorded": 9},
        )
        self.assertEqual(
            gate["detail"]["introspection_artifact_kind_counts"],
            {
                "activation_digests": 1,
                "attention_page_trace": 1,
                "device_dma_lifecycle": 1,
                "device_result_digests": 1,
                "kv_payload_digests": 1,
                "logit_slices": 1,
                "scheduler_packet_lineage": 1,
                "tensor_payload": 1,
                "token_quality": 1,
            },
        )
        self.assertEqual(
            gate["detail"]["introspection_artifact_format_counts"],
            {
                "attention_page_trace_jsonl": 1,
                "device_dma_lifecycle_jsonl": 1,
                "scheduler_packet_lineage_jsonl": 1,
                "tensor_payload_jsonl": 5,
                "token_quality_jsonl": 1,
            },
        )
        self.assertEqual(
            gate["detail"]["introspection_artifact_sensitivity_counts"],
            {"local-only": 9},
        )
        self.assertEqual(
            gate["detail"]["introspection_artifact_compile_feature_counts"],
            {"trace-introspection": 9},
        )
        self.assertEqual(
            gate["detail"]["introspection_artifact_row_count_total"],
            11,
        )
        self.assertEqual(
            gate["detail"]["introspection_artifact_byte_count_total"],
            10686,
        )
        self.assertEqual(
            gate["detail"]["introspection_artifact_samples"][0]["kind"],
            "token_quality",
        )
        self.assertEqual(
            gate["detail"]["introspection_artifact_samples"][0]["path"],
            "introspection_token_quality_sidecar.jsonl",
        )
        self.assertEqual(
            gate["detail"]["introspection_artifact_samples"][0]["sha256"],
            "4255461c82b837e9c6196aba804042ac967ee637602826962b70fe4a9576534e",
        )
        self.assertEqual(
            gate["detail"]["introspection_artifact_samples"][0]["token_window"],
            "0:1",
        )
        self.assertEqual(
            gate["detail"]["introspection_artifact_samples"][0]["sensitivity"],
            "local-only",
        )
        self.assertEqual(
            gate["detail"]["introspection_artifact_samples"][0]["compile_features"],
            "trace-introspection",
        )
        self.assertEqual(
            gate["detail"]["ab_provenance_status_counts"],
            {"not_applicable": 1},
        )
        self.assertEqual(
            gate["detail"]["ab_provenance_proof_grade_status_counts"],
            {"not_established_by_report": 1},
        )
        self.assertEqual(
            gate["detail"]["ab_comparability_status_counts"],
            {"not_applicable": 1},
        )
        self.assertEqual(
            gate["detail"]["ab_coverage_status_counts"],
            {"not_applicable": 1},
        )
        self.assertEqual(gate["detail"]["ab_coverage_total_rows"], 0)
        self.assertEqual(gate["detail"]["ab_coverage_matched_rows"], 0)
        self.assertEqual(
            gate["detail"]["ab_repeatability_status_counts"],
            {"not_applicable": 1},
        )
        self.assertEqual(
            gate["detail"]["ab_repeatability_proof_grade_status_counts"],
            {"not_established_by_report": 1},
        )
        self.assertEqual(
            gate["detail"]["ab_repeatability_required_matched_runs_for_hardware_proof"],
            3,
        )
        self.assertEqual(
            gate["detail"]["ab_provenance_samples"][0]["role"],
            "comparison",
        )
        self.assertEqual(
            gate["detail"]["timeline_query_summary_status_counts"],
            {"not_available": 1},
        )
        self.assertEqual(
            gate["detail"]["capture_samples"][0]["trace_run_id"],
            "ares-fixture-introspection",
        )
        self.assertEqual(
            gate["detail"]["run_provenance_samples"][0]["source_state"],
            "unknown",
        )
        self.assertEqual(
            gate["detail"]["artifact_identity_check_samples"][0]["detail"],
            "0 artifact(s)",
        )
        self.assertEqual(
            gate["detail"]["capture_capability_samples"][0]["capability"],
            "activation_digests",
        )
        self.assertEqual(
            gate["detail"]["timeline_query_summary_samples"][0]["status"],
            "not_available",
        )
        self.assertEqual(
            gate["detail"]["trace_config_status_counts"],
            {"requested_and_recorded": 1},
        )
        self.assertEqual(
            gate["detail"]["trace_config_missing_requested_sidecar_counts"],
            {},
        )
        self.assertEqual(
            gate["detail"]["provider_payload_boundary_status_counts"],
            {
                "blocked_no_supported_boundary": 1,
                "capability_without_matching_provider_artifact": 5,
                "recorded_artifact": 5,
                "route_available_no_provider_producer": 2,
            },
        )
        self.assertEqual(
            gate["detail"]["provider_payload_boundary_route_only_count"],
            2,
        )
        self.assertEqual(
            gate["detail"]["provider_payload_boundary_route_only_lanes"],
            ["generic/device_result_digests", "generic/tensor_payloads"],
        )
        self.assertEqual(
            gate["detail"]["provider_payload_boundary_recorded_count"],
            5,
        )
        self.assertEqual(
            gate["detail"]["provider_payload_boundary_recorded_lanes"],
            [
                "fpga/attention_page_trace",
                "fpga/device_dma_lifecycle",
                "fpga/device_result_digests",
                "fpga/kv_payload_digests",
                "fpga/logit_slices",
            ],
        )
        route_only_samples = {
            sample["payload_lane"]: sample
            for sample in gate["detail"]["provider_payload_boundary_route_only_samples"]
        }
        self.assertEqual(
            route_only_samples["device_result_digests"]["producer_status"],
            "runtime_route_only_no_provider_producer",
        )
        self.assertEqual(
            route_only_samples["device_result_digests"]["producer_contract"],
            "runtime_sidecar_route_only",
        )
        self.assertEqual(
            route_only_samples["device_result_digests"]["capture_control"],
            "ARES_TRACE_RECORD_DEVICE_RESULTS=1",
        )
        self.assertEqual(
            route_only_samples["device_result_digests"][
                "matching_provider_artifact_count"
            ],
            "0",
        )
        self.assertEqual(
            route_only_samples["device_result_digests"]["report_section"],
            "device_result_digest_sidecar_rows",
        )
        recorded_samples = {
            sample["payload_lane"]: sample
            for sample in gate["detail"]["provider_payload_boundary_recorded_samples"]
        }
        self.assertEqual(
            recorded_samples["device_result_digests"]["producer_status"],
            "provider_callback_present",
        )
        self.assertEqual(
            recorded_samples["device_result_digests"]["producer_contract"],
            "fpga_scheduler_batch_dispatch_emits_device_result_digest",
        )
        self.assertEqual(
            recorded_samples["device_result_digests"]["capture_control"],
            "ARES_TRACE_RECORD_DEVICE_RESULTS=1",
        )
        self.assertEqual(
            recorded_samples["device_result_digests"][
                "matching_provider_artifact_count"
            ],
            "1",
        )
        self.assertEqual(
            recorded_samples["device_result_digests"]["report_section"],
            "device_result_digest_sidecar_rows",
        )
        self.assertEqual(
            gate["detail"]["introspection_capability_status_counts"],
            {
                "capability_without_artifact": 1,
                "compiled": 1,
                "recorded": 11,
            },
        )
        self.assertEqual(
            gate["detail"]["introspection_artifact_summary_status_counts"],
            {"recorded_and_locally_present": 9},
        )
        self.assertEqual(
            gate["detail"]["introspection_section_inventory_status_counts"],
            {"available": 15, "capability_without_artifact": 1},
        )
        self.assertEqual(
            gate["detail"]["introspection_section_inventory_capability_counts"],
            {
                "activation_digests": 1,
                "attention_page_trace": 1,
                "debug_payloads": 1,
                "deep_introspection": 1,
                "device_dma_lifecycle": 1,
                "device_result_digests": 1,
                "kv_payload_digests": 1,
                "logit_slices": 1,
                "scheduler_packet_lineage": 3,
                "tensor_payloads": 1,
                "token_quality": 3,
                "topk_rows": 1,
            },
        )
        self.assertEqual(
            gate["detail"]["debug_payload_artifact_summary_status_counts"],
            {"recorded": 1},
        )
        self.assertEqual(
            gate["detail"]["token_quality_summary_status_counts"],
            {"present": 1},
        )
        self.assertEqual(
            gate["detail"]["token_quality_summary_topk_status_counts"],
            {"selected_is_top1": 1},
        )
        self.assertEqual(
            gate["detail"]["oracle_reference_summary_status_counts"],
            {"external_reference_hash_recorded": 1},
        )
        self.assertEqual(
            gate["detail"]["oracle_reference_summary_correctness_counts"],
            {"not_oracle_evidence": 1},
        )
        self.assertEqual(gate["detail"]["planning_decision_sidecar_count"], 0)
        self.assertEqual(
            gate["detail"]["token_quality_sidecar_status_counts"],
            {"present": 1},
        )
        self.assertEqual(
            gate["detail"]["token_quality_sidecar_finish_reason_counts"],
            {"stop": 1},
        )
        self.assertEqual(
            gate["detail"]["topk_token_sidecar_selected_status_counts"],
            {"selected_token": 1},
        )
        self.assertEqual(
            gate["detail"]["topk_token_sidecar_score_kind_counts"],
            {"logprob": 1},
        )
        self.assertEqual(
            gate["detail"]["tensor_payload_sidecar_status_counts"],
            {"ok": 1},
        )
        self.assertEqual(
            gate["detail"]["tensor_payload_sidecar_kind_counts"],
            {"logit_slice": 1},
        )
        self.assertEqual(
            gate["detail"]["tensor_payload_sidecar_role_counts"],
            {"logits": 1},
        )
        self.assertEqual(
            gate["detail"]["kv_payload_digest_sidecar_status_counts"],
            {"ok": 1},
        )
        self.assertEqual(
            gate["detail"]["kv_payload_digest_sidecar_role_counts"],
            {"kv_key": 1},
        )
        self.assertEqual(
            gate["detail"]["logit_slice_sidecar_role_counts"],
            {"logits": 1},
        )
        self.assertEqual(
            gate["detail"]["logit_slice_sidecar_action_counts"],
            {"final_logits": 1},
        )
        self.assertEqual(
            gate["detail"]["activation_digest_sidecar_role_counts"],
            {"activation": 1},
        )
        self.assertEqual(
            gate["detail"]["activation_digest_sidecar_intrinsic_counts"],
            {"rmsnorm": 1},
        )
        self.assertEqual(
            gate["detail"]["device_result_digest_sidecar_status_counts"],
            {"ok": 1},
        )
        self.assertEqual(
            gate["detail"]["device_result_digest_sidecar_role_counts"],
            {"device_result": 1},
        )
        self.assertEqual(
            gate["detail"]["device_result_digest_sidecar_action_counts"],
            {"device_result": 1},
        )
        self.assertEqual(
            gate["detail"]["device_result_digest_sidecar_intrinsic_counts"],
            {"fpga.device_result_digest": 1},
        )
        self.assertEqual(
            gate["detail"]["scheduler_packet_lineage_sidecar_status_counts"],
            {"ok": 1},
        )
        self.assertEqual(
            gate["detail"]["scheduler_packet_lineage_sidecar_executor_counts"],
            {"executed_fullscheduler_forward_batch_v1": 1},
        )
        self.assertEqual(
            gate["detail"]["scheduler_kv_shard_lifecycle_sidecar_status_counts"],
            {"ok": 1},
        )
        self.assertEqual(
            gate["detail"]["scheduler_kv_shard_lifecycle_sidecar_lifecycle_counts"],
            {"observed": 1},
        )
        self.assertEqual(
            gate["detail"][
                "scheduler_listener_sparse_logit_sidecar_listener_status_counts"
            ],
            {"observed": 1},
        )
        self.assertEqual(
            gate["detail"]["scheduler_listener_sparse_logit_sidecar_executor_counts"],
            {"executed_fullscheduler_forward_batch_v1": 1},
        )
        self.assertEqual(
            gate["detail"]["device_dma_lifecycle_sidecar_status_counts"],
            {"ok": 1},
        )
        self.assertEqual(
            gate["detail"]["device_dma_lifecycle_sidecar_stage_counts"],
            {"dma_completion": 1},
        )
        self.assertEqual(
            gate["detail"]["device_dma_lifecycle_sidecar_queue_counts"],
            {"load_weight_q": 1},
        )
        self.assertEqual(
            gate["detail"]["attention_page_trace_sidecar_status_counts"],
            {"ok": 1},
        )
        self.assertEqual(
            gate["detail"]["attention_page_trace_sidecar_action_counts"],
            {"attention": 1},
        )
        section_paths = {
            sample["json_path"]
            for sample in gate["detail"]["report_json_section_samples"]
        }
        self.assertIn("sections.trace_config_rows", section_paths)
        self.assertIn("sections.debug_payload_artifact_summary_rows", section_paths)
        self.assertIn("sections.token_quality_summary_rows", section_paths)
        self.assertIn("sections.oracle_reference_summary_rows", section_paths)
        self.assertIn("sections.planning_decision_sidecar_rows", section_paths)
        self.assertIn("sections.token_quality_sidecar_rows", section_paths)
        self.assertIn("sections.topk_token_sidecar_rows", section_paths)
        self.assertIn("sections.tensor_payload_sidecar_rows", section_paths)
        self.assertIn("sections.kv_payload_digest_sidecar_rows", section_paths)
        self.assertIn("sections.logit_slice_sidecar_rows", section_paths)
        self.assertIn("sections.activation_digest_sidecar_rows", section_paths)
        self.assertIn("sections.scheduler_packet_lineage_sidecar_rows", section_paths)
        self.assertIn(
            "sections.scheduler_kv_shard_lifecycle_sidecar_rows",
            section_paths,
        )
        self.assertIn(
            "sections.scheduler_listener_sparse_logit_sidecar_rows",
            section_paths,
        )
        self.assertIn("sections.device_dma_lifecycle_sidecar_rows", section_paths)
        self.assertIn("sections.attention_page_trace_sidecar_rows", section_paths)
        self.assertIn("sections.introspection_capability_rows", section_paths)
        self.assertIn("sections.introspection_artifacts", section_paths)
        self.assertIn("sections.introspection_artifact_summary_rows", section_paths)
        self.assertIn("sections.introspection_section_inventory", section_paths)
        self.assertIn("sections.report_section_inventory", section_paths)
        self.assertIn("sections.preflight_findings", section_paths)
        self.assertIn("sections.evidence_classification", section_paths)
        self.assertIn(
            "device_result_digest_sidecar_rows",
            gate["detail"]["section_names"],
        )
        self.assertIn(
            "provider_payload_boundary_inventory_rows",
            gate["detail"]["section_names"],
        )
        self.assertEqual(
            gate["detail"]["trace_config_samples"][0]["requested_sidecar_controls"],
            "topk,logit_slices,tensor_payloads,activation_digests,kv_payload_digests,device_result_digests",
        )
        self.assertNotIn(
            "missing_requested_sidecar_controls",
            gate["detail"]["trace_config_samples"][0],
        )
        self.assertEqual(
            gate["detail"]["provider_payload_boundary_samples"][0]["capture_status"],
            "capability_without_matching_provider_artifact",
        )
        self.assertEqual(
            gate["detail"]["provider_payload_boundary_samples"][0][
                "matching_provider_artifact_count"
            ],
            "0",
        )
        self.assertEqual(
            gate["detail"]["provider_payload_boundary_samples"][0][
                "artifact_kind_recorded_count"
            ],
            "1",
        )
        self.assertEqual(
            gate["detail"]["provider_payload_boundary_samples"][0][
                "artifact_kind_recorded_backend_ids"
            ],
            "fpga",
        )
        self.assertEqual(
            gate["detail"]["provider_payload_boundary_samples"][0]["producer_status"],
            "provider_callback_present",
        )
        self.assertEqual(
            gate["detail"]["provider_payload_boundary_samples"][0][
                "payload_record_policy"
            ],
            "sha256_digest_plus_bounded_f32_sample",
        )
        self.assertEqual(
            gate["detail"]["provider_payload_boundary_samples"][0][
                "payload_sensitivity"
            ],
            "final_logits",
        )
        self.assertEqual(
            gate["detail"]["introspection_artifact_summary_samples"][0][
                "artifact_kind"
            ],
            "token_quality",
        )
        self.assertEqual(
            gate["detail"]["debug_payload_artifact_summary_samples"][0][
                "artifact_kind"
            ],
            "attention_page_trace",
        )
        self.assertEqual(
            gate["detail"]["debug_payload_artifact_summary_samples"][0]["sensitivity"],
            "local-only",
        )
        self.assertEqual(
            gate["detail"]["token_quality_summary_samples"][0]["selected_topk_status"],
            "selected_is_top1",
        )
        self.assertEqual(
            gate["detail"]["token_quality_summary_samples"][0]["oracle_reference"],
            "external_hf_cpu_reference",
        )
        self.assertEqual(
            gate["detail"]["oracle_reference_summary_samples"][0][
                "oracle_reference_status"
            ],
            "external_reference_hash_recorded",
        )
        self.assertEqual(
            gate["detail"]["oracle_reference_summary_samples"][0][
                "correctness_claim_status"
            ],
            "not_oracle_evidence",
        )
        self.assertEqual(
            gate["detail"]["token_quality_sidecar_samples"][0]["finish_reason"],
            "stop",
        )
        self.assertEqual(
            gate["detail"]["token_quality_sidecar_samples"][0]["selected_token_id"],
            "42",
        )
        self.assertEqual(
            gate["detail"]["topk_token_sidecar_samples"][0][
                "selected_candidate_status"
            ],
            "selected_token",
        )
        self.assertEqual(
            gate["detail"]["topk_token_sidecar_samples"][0]["candidate_token_id"],
            "42",
        )
        self.assertEqual(
            gate["detail"]["tensor_payload_sidecar_samples"][0]["tensor_payload_kind"],
            "logit_slice",
        )
        self.assertEqual(
            gate["detail"]["tensor_payload_sidecar_samples"][0]["tensor_role"],
            "logits",
        )
        self.assertEqual(
            gate["detail"]["kv_payload_digest_sidecar_samples"][0]["tensor_role"],
            "kv_key",
        )
        self.assertEqual(
            gate["detail"]["logit_slice_sidecar_samples"][0]["targetplan_action"],
            "final_logits",
        )
        self.assertEqual(
            gate["detail"]["logit_slice_sidecar_samples"][0]["sample_nan_count"],
            "1",
        )
        self.assertEqual(
            gate["detail"]["activation_digest_sidecar_samples"][0]["tensor_role"],
            "activation",
        )
        self.assertEqual(
            gate["detail"]["activation_digest_sidecar_samples"][0]["intrinsic"],
            "rmsnorm",
        )
        self.assertEqual(
            gate["detail"]["device_result_digest_sidecar_samples"][0]["request_id"],
            "7009",
        )
        self.assertEqual(
            gate["detail"]["device_result_digest_sidecar_samples"][0]["tensor_name"],
            "fpga_device_result",
        )
        self.assertEqual(
            gate["detail"]["device_result_digest_sidecar_samples"][0]["digest_sha256"],
            "e" * 64,
        )
        self.assertEqual(
            gate["detail"]["device_result_digest_sidecar_samples"][0]["sample_max"],
            "2.5",
        )
        self.assertEqual(
            gate["detail"]["scheduler_packet_lineage_sidecar_samples"][0][
                "executor_status"
            ],
            "executed_fullscheduler_forward_batch_v1",
        )
        self.assertEqual(
            gate["detail"]["scheduler_packet_lineage_sidecar_samples"][0][
                "listener_sparse_tokens"
            ],
            "3",
        )
        self.assertEqual(
            gate["detail"]["scheduler_kv_shard_lifecycle_sidecar_samples"][0][
                "kv_lifecycle_status"
            ],
            "observed",
        )
        self.assertEqual(
            gate["detail"]["scheduler_kv_shard_lifecycle_sidecar_samples"][0][
                "prior_host_gof_staging_status"
            ],
            "page_info_published",
        )
        self.assertEqual(
            gate["detail"]["scheduler_listener_sparse_logit_sidecar_samples"][0][
                "listener_sparse_tokens"
            ],
            "3",
        )
        self.assertEqual(
            gate["detail"]["scheduler_listener_sparse_logit_sidecar_samples"][0][
                "listener_sparse_status"
            ],
            "observed",
        )
        self.assertEqual(
            gate["detail"]["device_dma_lifecycle_sidecar_samples"][0]["queue_id"],
            "load_weight_q",
        )
        self.assertEqual(
            gate["detail"]["device_dma_lifecycle_sidecar_samples"][0][
                "counter_value_delta"
            ],
            "5",
        )
        self.assertEqual(
            gate["detail"]["attention_page_trace_sidecar_samples"][0]["head"],
            "17",
        )
        self.assertEqual(
            gate["detail"]["attention_page_trace_sidecar_samples"][0]["m_star"],
            "3.0",
        )
        self.assertEqual(
            gate["detail"]["introspection_section_inventory_samples"][0][
                "json_section"
            ],
            "planning_decision_sidecar_rows",
        )
        self.assertEqual(
            gate["detail"]["introspection_section_inventory_samples"][0][
                "section_status"
            ],
            "capability_without_artifact",
        )
        self.assertEqual(
            gate["detail"]["introspection_section_inventory_samples"][1][
                "json_section"
            ],
            "token_quality_sidecar_rows",
        )
        self.assertEqual(
            gate["detail"]["introspection_section_inventory_samples"][1][
                "section_status"
            ],
            "available",
        )
        introspection_sections = {
            sample["json_section"]
            for sample in gate["detail"]["introspection_section_inventory_samples"]
        }
        self.assertIn(
            "scheduler_kv_shard_lifecycle_sidecar_rows", introspection_sections
        )
        self.assertIn("token_quality_sidecar_rows", introspection_sections)
        self.assertIn("topk_token_sidecar_rows", introspection_sections)
        self.assertIn("logit_slice_sidecar_rows", introspection_sections)
        self.assertIn("activation_digest_sidecar_rows", introspection_sections)
        self.assertIn(
            "scheduler_listener_sparse_logit_sidecar_rows", introspection_sections
        )
        self.assertIn("device_dma_lifecycle_sidecar_rows", introspection_sections)
        self.assertIn("attention_page_trace_sidecar_rows", introspection_sections)
        self.assertIn("kv_payload_digest_sidecar_rows", introspection_sections)

    def test_trace_report_gate_rejects_malformed_introspection_artifact_counts(
        self,
    ) -> None:
        fixture_path = FIXTURE_DIR / "ares_trace_report_introspection_real.json"
        payload = json.loads(fixture_path.read_text())
        payload["sections"]["introspection_artifacts"][0]["row_count"] = "not-int"
        payload["sections"]["introspection_artifacts"][1]["byte_count"] = -1

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace-report.json"
            path.write_text(json.dumps(payload))

            gate = trace_report_gate(path)

            self.assertFalse(gate["passed"])
            joined = " ".join(gate["errors"])
            self.assertIn(
                "trace report sections.introspection_artifacts[0].row_count "
                "must be a non-negative integer",
                joined,
            )
            self.assertIn(
                "trace report sections.introspection_artifacts[1].byte_count "
                "must be a non-negative integer",
                joined,
            )

    def test_trace_report_gate_rejects_missing_report_triage(self) -> None:
        fixture_path = FIXTURE_DIR / "ares_trace_report_introspection_real.json"
        payload = json.loads(fixture_path.read_text())
        del payload["sections"]["report_triage"]

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace-report.json"
            path.write_text(json.dumps(payload))

            gate = trace_report_gate(path)

            self.assertFalse(gate["passed"])
            joined = " ".join(gate["errors"])
            self.assertIn(
                "trace report sections missing required section: report_triage",
                joined,
            )
            self.assertIn(
                "trace report sections.report_triage must be a list",
                joined,
            )

    def test_trace_report_gate_rejects_empty_report_triage(self) -> None:
        fixture_path = FIXTURE_DIR / "ares_trace_report_introspection_real.json"
        payload = json.loads(fixture_path.read_text())
        payload["sections"]["report_triage"] = []

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace-report.json"
            path.write_text(json.dumps(payload))

            gate = trace_report_gate(path)

            self.assertFalse(gate["passed"])
            self.assertIn(
                "trace report sections.report_triage must contain at least one row",
                " ".join(gate["errors"]),
            )

    def test_trace_report_gate_rejects_malformed_report_triage(self) -> None:
        fixture_path = FIXTURE_DIR / "ares_trace_report_introspection_real.json"
        payload = json.loads(fixture_path.read_text())
        row = payload["sections"]["report_triage"][0]
        del row["report_grade"]
        row["triage_status"] = "proof_ready"
        row["proof_grade_status"] = "proof_ready"
        row["first_useful_section"] = "next_measurements"
        row["first_action"] = ""
        row["claim_boundary"] = "promotion_evidence"
        row["first_answerable_question"] = 42

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace-report.json"
            path.write_text(json.dumps(payload))

            gate = trace_report_gate(path)

            self.assertFalse(gate["passed"])
            joined = " ".join(gate["errors"])
            self.assertIn(
                "trace report sections.report_triage[0] missing required "
                "field(s): report_grade",
                joined,
            )
            self.assertIn(
                "trace report sections.report_triage[0].triage_status must be one of",
                joined,
            )
            self.assertIn(
                "trace report sections.report_triage[0].proof_grade_status "
                "must be not_established_by_report",
                joined,
            )
            self.assertIn(
                "trace report sections.report_triage[0].first_useful_section "
                "must match sections.<name>",
                joined,
            )
            self.assertIn(
                "trace report sections.report_triage[0].first_action "
                "must be a non-empty string",
                joined,
            )
            self.assertIn(
                "trace report sections.report_triage[0].claim_boundary "
                "must be diagnostic_routing_not_evidence",
                joined,
            )
            self.assertIn(
                "trace report sections.report_triage[0].first_answerable_question "
                "must be a string",
                joined,
            )

    def test_trace_report_gate_rejects_malformed_string_and_inventory_sections(
        self,
    ) -> None:
        fixture_path = FIXTURE_DIR / "ares_trace_report_introspection_real.json"
        payload = json.loads(fixture_path.read_text())
        payload["sections"]["preflight_findings"][0] = {"warning": "not a string"}
        payload["sections"]["evidence_classification"][0] = 42
        payload["sections"]["report_section_inventory"][0] = "not an object"

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace-report.json"
            path.write_text(json.dumps(payload))

            gate = trace_report_gate(path)

            self.assertFalse(gate["passed"])
            joined = " ".join(gate["errors"])
            self.assertIn(
                "trace report sections.preflight_findings[0] must be a string",
                joined,
            )
            self.assertIn(
                "trace report sections.evidence_classification[0] must be a string",
                joined,
            )
            self.assertIn(
                "trace report sections.report_section_inventory[0] must be an object",
                joined,
            )

    def test_trace_report_gate_rejects_malformed_preflight_counts(self) -> None:
        fixture_path = FIXTURE_DIR / "ares_trace_report_introspection_real.json"
        payload = json.loads(fixture_path.read_text())
        payload["sections"]["preflight"][0]["ok"] = "not-int"
        payload["sections"]["preflight"][0]["warn"] = -3
        payload["sections"]["preflight"][0]["fail"] = {"count": 1}

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace-report.json"
            path.write_text(json.dumps(payload))

            gate = trace_report_gate(path)

            self.assertFalse(gate["passed"])
            joined = " ".join(gate["errors"])
            self.assertIn(
                "trace report sections.preflight[0].ok must be a non-negative integer",
                joined,
            )
            self.assertIn(
                "trace report sections.preflight[0].warn must be a non-negative integer",
                joined,
            )
            self.assertIn(
                "trace report sections.preflight[0].fail must be a non-negative integer",
                joined,
            )

    def test_trace_report_gate_rejects_missing_file(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing-trace-report.json"

            gate = trace_report_gate(path)

            self.assertFalse(gate["passed"])
            self.assertEqual(gate["artifact_validator"], "trace_report")
            self.assertIn("missing", " ".join(gate["errors"]))

    def test_trace_report_gate_rejects_invalid_json_with_hash(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace-report.json"
            path.write_text("{not valid json")

            gate = trace_report_gate(path)

            self.assertFalse(gate["passed"])
            self.assertEqual(gate["artifact_validator"], "trace_report")
            self.assertIn("invalid JSON", " ".join(gate["errors"]))
            self.assertEqual(len(gate["detail"]["sha256"]), 64)

    def test_trace_report_gate_rejects_missing_required_sections(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace-report.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "title": "Incomplete Ares Trace Report",
                        "inputs": {"metadata": "run.trace-meta.json"},
                        "sections": {
                            "preflight": [{"status": "pass"}],
                            "analysis_commands": [{"purpose": "report"}],
                        },
                    }
                )
            )

            gate = trace_report_gate(path)

            self.assertFalse(gate["passed"])
            errors = " ".join(gate["errors"])
            self.assertIn("report_grade", errors)
            self.assertIn("answerability", errors)
            self.assertIn("next_measurements", errors)

    def test_one_token_logits_gate_requires_hf_cpu_oracle_and_replay_context(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "one-token.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "ares.runtime.one_token_logits.v1",
                        "evidence_class": "system_under_test",
                        "oracle": "huggingface_transformers_pytorch_cpu",
                        "candidate": "ares",
                        "tvd": 0.001,
                        "tvd_threshold": 0.01,
                        "top1_agreement": 1.0,
                        "same_argmax": True,
                        "replay_context": replay_context(),
                    }
                )
            )

            gate = one_token_logits_gate(path)

            self.assertTrue(gate["passed"])
            self.assertEqual(gate["artifact_validator"], "one_token_logits")

    def test_one_token_logits_gate_rejects_partial_top1_agreement(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "one-token.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "ares.runtime.one_token_logits.v1",
                        "evidence_class": "system_under_test",
                        "oracle": "huggingface_transformers_pytorch_cpu",
                        "candidate": "ares",
                        "tvd": 0.001,
                        "tvd_threshold": 0.01,
                        "top1_agreement": 0.5,
                        "same_argmax": True,
                        "replay_context": replay_context(),
                    }
                )
            )

            gate = one_token_logits_gate(path)

            self.assertFalse(gate["passed"])
            self.assertIn("top1_agreement must be 1.0", " ".join(gate["errors"]))

    def test_one_token_logits_gate_rejects_non_ares_candidate(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "one-token.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "ares.runtime.one_token_logits.v1",
                        "evidence_class": "system_under_test",
                        "oracle": "huggingface_transformers_pytorch_cpu",
                        "candidate": "cpp_tron_rinzler",
                        "tvd": 0.001,
                        "tvd_threshold": 0.01,
                        "top1_agreement": 1.0,
                        "same_argmax": True,
                        "replay_context": replay_context(),
                    }
                )
            )

            gate = one_token_logits_gate(path)

            self.assertFalse(gate["passed"])
            self.assertIn("candidate must identify Ares", " ".join(gate["errors"]))

    def test_cpp_tvd_rejects_cpp_as_oracle(self) -> None:
        validation = validate_cpp_tvd_evidence(
            {
                "schema": "ares.comparison.cpp_tvd.v1",
                "evidence_class": "comparison",
                "comparison_source": "cpp_tron_rinzler",
                "oracle": "cpp_tron_rinzler",
                "tvd": 0.001,
                "tvd_threshold": 0.01,
                "replay_context": replay_context(),
            }
        )

        self.assertFalse(validation.passed)
        self.assertTrue(
            any(
                "must not be labeled as correctness oracle" in error
                for error in validation.errors
            )
        )

    def test_cpp_tvd_gate_accepts_comparison_evidence(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "cpp-tvd.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "ares.comparison.cpp_tvd.v1",
                        "evidence_class": "comparison",
                        "comparison_source": "cpp_tron_rinzler",
                        "candidate": "ares",
                        "tvd": 0.001,
                        "tvd_threshold": 0.01,
                        "replay_context": replay_context(),
                    }
                )
            )

            gate = cpp_tvd_gate(path)

            self.assertTrue(gate["passed"])
            self.assertEqual(gate["artifact_validator"], "cpp_tvd")

    def test_introspection_ladder_gate_accepts_semantic_localization(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "ladder.json"
            path.write_text(
                json.dumps(
                    introspection_ladder_payload(
                        root,
                        evidence_class="semantic_localization",
                    )
                )
            )

            gate = introspection_ladder_gate(path)

            self.assertTrue(gate["passed"], gate.get("errors"))
            self.assertEqual(gate["artifact_validator"], "introspection_ladder")
            self.assertEqual(gate["sha256"], sha256_file(path))
            self.assertEqual(
                gate["detail"]["first_failing_stage"],
                "source_hf_hypergraph",
            )
            self.assertEqual(
                gate["detail"]["comparisons"][0]["from_stage"],
                "hf_cpu_oracle",
            )
            self.assertEqual(
                gate["detail"]["comparisons"][0]["first_mismatch"]["id"],
                "logits",
            )
            self.assertEqual(
                gate["detail"]["first_failed_comparison"]["to_stage"],
                "source_hf_hypergraph",
            )
            self.assertEqual(gate["detail"]["first_mismatch"]["id"], "logits")
            self.assertEqual(
                gate["detail"]["first_mismatch"]["producer_generator"],
                "linear_0",
            )
            self.assertEqual(
                gate["detail"]["first_mismatch"]["max_abs_error"],
                0.25,
            )
            self.assertEqual(
                gate["detail"]["trace_context"]["trace_labels"],
                ["targetplan.stmt.00001.matmul"],
            )
            self.assertEqual(
                gate["detail"]["trace_context"]["backend_events"][0]["path"],
                "backend-events.jsonl",
            )
            self.assertEqual(
                gate["detail"]["trace_context"]["backend_events"][0]["role"],
                "backend_events",
            )
            self.assertEqual(
                gate["detail"]["trace_context"]["perfetto_traces"][0]["role"],
                "perfetto_trace",
            )
            self.assertEqual(
                gate["detail"]["trace_context"]["trace_metadata"][0]["path"],
                "run.trace-meta.json",
            )
            self.assertEqual(
                gate["detail"]["trace_context"]["trace_metadata"][0]["schema"],
                "ares.trace.metadata.v1",
            )
            self.assertEqual(
                gate["detail"]["trace_context"]["trace_metadata"][0]["role"],
                "ares_trace_metadata",
            )

    def test_introspection_ladder_rejects_promotion_claim(self) -> None:
        with TemporaryDirectory() as tmp:
            validation = validate_introspection_ladder_report(
                introspection_ladder_payload(
                    Path(tmp),
                    evidence_class="promotion",
                ),
                base_dir=Path(tmp),
            )

            self.assertFalse(validation.passed)
            self.assertIn("semantic_localization", " ".join(validation.errors))

    def test_introspection_ladder_rejects_bad_trace_context_hash(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = introspection_ladder_payload(
                root,
                evidence_class="semantic_localization",
            )
            payload["trace_context"]["backend_events"][0]["sha256"] = (
                "sha256:" + "b" * 64
            )

            validation = validate_introspection_ladder_report(
                payload,
                base_dir=root,
            )

            self.assertFalse(validation.passed)
            self.assertIn(
                "trace_context.backend_events[0].sha256 does not match referenced file",
                validation.errors,
            )

    def test_token_agreement_gate_accepts_eight_token_evidence(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference.json"
            candidate = root / "candidate.json"
            path = root / "tokens.json"
            reference_payload = {
                "token_ids": [99],
                "generated_token_ids": list(range(8)),
            }
            candidate_payload = {
                "token_ids": [99],
                "generated_token_ids": list(range(8)),
            }
            reference.write_text(json.dumps(reference_payload))
            candidate.write_text(json.dumps(candidate_payload))
            path.write_text(
                json.dumps(
                    build_greedy_token_evidence(
                        {
                            "score": 1.0,
                            "exact_match": True,
                            "exact_fraction": 1.0,
                            "top1_agreement": 1.0,
                            "cases": [
                                {
                                    "name": "default",
                                    "exact_match": True,
                                    "candidate_length": 8,
                                }
                            ],
                        },
                        reference=reference,
                        candidate=candidate,
                        reference_payload=reference_payload,
                        candidate_payload=candidate_payload,
                    )
                )
            )

            gate = token_agreement_gate(path)

            self.assertTrue(gate["passed"])
            self.assertEqual(gate["artifact_validator"], "eight_token_greedy")

    def test_token_agreement_gate_rejects_short_match(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference.json"
            candidate = root / "candidate.json"
            path = root / "tokens.json"
            reference_payload = {"generated_token_ids": [1, 2, 3]}
            candidate_payload = {"generated_token_ids": [1, 2, 3]}
            reference.write_text(json.dumps(reference_payload))
            candidate.write_text(json.dumps(candidate_payload))
            path.write_text(
                json.dumps(
                    build_greedy_token_evidence(
                        {
                            "score": 1.0,
                            "exact_match": True,
                            "exact_fraction": 1.0,
                            "top1_agreement": 1.0,
                            "cases": [
                                {
                                    "name": "default",
                                    "exact_match": True,
                                    "candidate_length": 3,
                                }
                            ],
                        },
                        reference=reference,
                        candidate=candidate,
                        reference_payload=reference_payload,
                        candidate_payload=candidate_payload,
                    )
                )
            )

            gate = token_agreement_gate(path)

            self.assertFalse(gate["passed"])
            self.assertIn("expected at least 8", " ".join(gate["errors"]))

    def test_token_agreement_gate_rejects_stale_source_digest(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference.json"
            candidate = root / "candidate.json"
            path = root / "tokens.json"
            reference_payload = {"generated_token_ids": list(range(8))}
            candidate_payload = {"generated_token_ids": list(range(8))}
            reference.write_text(json.dumps(reference_payload))
            candidate.write_text(json.dumps(candidate_payload))
            path.write_text(
                json.dumps(
                    build_greedy_token_evidence(
                        {
                            "score": 1.0,
                            "exact_match": True,
                            "exact_fraction": 1.0,
                            "top1_agreement": 1.0,
                            "cases": [
                                {
                                    "name": "default",
                                    "exact_match": True,
                                    "candidate_length": 8,
                                }
                            ],
                        },
                        reference=reference,
                        candidate=candidate,
                        reference_payload=reference_payload,
                        candidate_payload=candidate_payload,
                    )
                )
            )
            candidate.write_text(json.dumps({"generated_token_ids": [7] * 8}))

            gate = token_agreement_gate(path)

            self.assertFalse(gate["passed"])
            self.assertIn("sha256 does not match", " ".join(gate["errors"]))

    def test_depth_performance_gate_requires_full_ladder(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "depth.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "ares.performance.depth_ladder.v1",
                        "evidence_class": "system_under_test",
                        "workload": "independent_decode",
                        "correctness_gates_green": True,
                        "depths": [
                            {
                                "generated_tokens": 8,
                                "tokens_match": True,
                                "throughput_tokens_per_second": 80.0,
                            },
                            {
                                "generated_tokens": 64,
                                "tokens_match": True,
                                "throughput_tokens_per_second": 70.0,
                            },
                            {
                                "generated_tokens": 512,
                                "tokens_match": True,
                                "throughput_tokens_per_second": 60.0,
                            },
                        ],
                    }
                )
            )

            gate = depth_performance_gate(path)

            self.assertTrue(gate["passed"])
            self.assertEqual(gate["detail"]["depths"], [8, 64, 512])

    def test_depth_performance_gate_rejects_reversed_ladder(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "depth.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "ares.performance.depth_ladder.v1",
                        "evidence_class": "system_under_test",
                        "workload": "independent_decode",
                        "correctness_gates_green": True,
                        "depths": [
                            {
                                "generated_tokens": 512,
                                "tokens_match": True,
                                "throughput_tokens_per_second": 60.0,
                            },
                            {
                                "generated_tokens": 64,
                                "tokens_match": True,
                                "throughput_tokens_per_second": 70.0,
                            },
                            {
                                "generated_tokens": 8,
                                "tokens_match": True,
                                "throughput_tokens_per_second": 80.0,
                            },
                        ],
                    }
                )
            )

            gate = depth_performance_gate(path)

            self.assertFalse(gate["passed"])
            self.assertIn("ordered 8 -> 64 -> 512", " ".join(gate["errors"]))

    def test_mmlu_pro_gate_accepts_threshold_passing_evidence(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "mmlu-pro.json"
            path.write_text(json.dumps(mmlu_pro_payload(root)))

            gate = mmlu_pro_gate(
                path,
                expected_model="synthetic/model",
                expected_backend="fpga",
                required_coverage_percent=10,
            )

            self.assertTrue(gate["passed"], gate.get("errors"))
            self.assertEqual(gate["artifact_validator"], "mmlu_pro")
            self.assertEqual(gate["detail"]["score_percent"], 72.0)

    def test_mmlu_pro_gate_rejects_low_score_and_dirty_sources(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "mmlu-pro.json"
            path.write_text(
                json.dumps(
                    mmlu_pro_payload(
                        root,
                        score_percent=60.0,
                        systems_test_dirty=True,
                    )
                )
            )

            gate = mmlu_pro_gate(path)

            self.assertFalse(gate["passed"])
            self.assertIn("score_percent must meet", " ".join(gate["errors"]))
            self.assertIn("systems_test.dirty must be false", " ".join(gate["errors"]))

    def test_mmlu_pro_gate_rejects_wrong_model_or_backend_for_spec(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "mmlu-pro.json"
            path.write_text(
                json.dumps(
                    mmlu_pro_payload(
                        root,
                        model="wrong/model",
                        backend="wrong-backend",
                    )
                )
            )

            gate = mmlu_pro_gate(
                path,
                expected_model="synthetic/model",
                expected_backend="fpga",
            )

            self.assertFalse(gate["passed"])
            self.assertIn("model must match model_spec", " ".join(gate["errors"]))
            self.assertIn("backend must match model_spec", " ".join(gate["errors"]))

    def test_mmlu_pro_gate_rejects_missing_endpoint_and_config_proofs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "mmlu-pro.json"
            path.write_text(
                json.dumps(
                    mmlu_pro_payload(
                        root,
                        include_endpoint_models=False,
                        include_systems_test_config=False,
                    )
                )
            )

            gate = mmlu_pro_gate(path)

            self.assertFalse(gate["passed"])
            self.assertIn("endpoint_models must be an object", " ".join(gate["errors"]))
            self.assertIn(
                "systems_test.config must be an object", " ".join(gate["errors"])
            )

    def test_mmlu_pro_gate_rejects_undercoverage_for_spec(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "mmlu-pro.json"
            path.write_text(json.dumps(mmlu_pro_payload(root, coverage_percent=1)))

            gate = mmlu_pro_gate(path, required_coverage_percent=10)

            self.assertFalse(gate["passed"])
            self.assertIn("coverage_percent must meet", " ".join(gate["errors"]))


if __name__ == "__main__":
    unittest.main()

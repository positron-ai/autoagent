from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from ares_ingest_autoagent.ares_cli import (
    CLAUDE_REFINER_COMMAND,
    DEFAULT_REFINER_COMMAND,
    AresIngestError,
    append_history,
    append_steering_note,
    append_steering_resource,
    build_parser,
    config_from_args,
    cockpit_checkpoint,
    evaluate_run,
    initialize_run,
    main,
    render_trace_report_lines,
    selected_workflow_skills,
    slugify,
    trace_report_summary_from_spec,
    write_handoff,
    write_failure_state,
    write_refinement_prompt,
)
from ares_ingest_autoagent.artifacts import trace_report_gate


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json_artifact(path: Path, payload: dict) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return {"path": path.name, "sha256": "sha256:" + sha256_file(path)}


def write_introspection_ladder(root: Path) -> Path:
    source_run = write_json_artifact(
        root / "source.run.json",
        {"schema": "ares.introspection.run.v1"},
    )
    compare = write_json_artifact(
        root / "source.compare.json",
        {"schema": "ares.introspection.compare.v1"},
    )
    backend_events = write_json_artifact(
        root / "backend-events.jsonl",
        {"event": "forward_executed"},
    )
    trace_metadata = write_json_artifact(
        root / "run.trace-meta.json",
        {
            "schema_version": 1,
            "trace_run_id": "trace-run-001",
            "introspection_artifacts": [],
        },
    )
    trace_metadata["schema"] = "ares.trace.metadata.v1"
    trace_metadata["role"] = "ares_trace_metadata"
    ladder = root / "ladder.json"
    ladder.write_text(
        json.dumps(
            {
                "schema": "ares.introspection.ladder.v1",
                "status": "failed",
                "evidence_class": "semantic_localization",
                "producer": {"tool": "unit-test"},
                "model": {"id": "Provider/Model", "checkpoint": "unit-test"},
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
                            "trace_label": "targetplan.stmt.00001.matmul",
                        },
                    }
                ],
                "first_failing_stage": "source_hf_hypergraph",
                "next_owner": "ares-python",
                "recommendation": "Investigate HF-export source replay.",
                "trace_context": {
                    "trace_labels": ["targetplan.stmt.00001.matmul"],
                    "backend_events": [backend_events],
                    "trace_metadata": [trace_metadata],
                },
            },
            sort_keys=True,
        )
        + "\n"
    )
    return ladder


def trace_report_payload() -> dict:
    return {
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
                    "proof_grade_status": "not_established_by_report",
                    "first_blocked_gate": "hf_cpu_oracle",
                    "first_blocked_gate_status": "blocked",
                    "first_blocked_gate_basis": "HF CPU oracle not recorded",
                    "first_next_measurement_priority": "backend_jsonl",
                    "first_next_measurement_reason": "backend JSONL evidence not present",
                    "first_next_measurement": "Capture backend event JSONL",
                    "first_next_measurement_command_hint": (
                        "set ARES_BACKEND_EVENT_ARTIFACT_DIR"
                    ),
                    "first_answerable_question": "",
                    "first_unsupported_claim": (
                        "backend JSONL evidence is unsupported"
                    ),
                    "first_useful_section": "sections.next_measurements",
                    "first_action": "Capture backend event JSONL",
                    "claim_boundary": "diagnostic_routing_not_evidence",
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
                    "next_measurement": "Capture backend event JSONL",
                    "reason": "backend JSONL evidence not present",
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
                    "claim_guardrail": "do not treat trace timing as promotion proof",
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
                    "proof_grade_status": "not_established_by_report",
                    "basis": (
                        "baseline binary, clean source state, artifact hashes, "
                        "and hardware identity are recorded"
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
                    "proof_grade_status": "not_established_by_report",
                    "basis": (
                        "candidate binary, clean source state, artifact hashes, "
                        "and hardware identity are recorded"
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
                    "proof_grade_status": "not_established_by_report",
                    "basis": (
                        "baseline/candidate binary, clean source state, "
                        "artifact hashes, and hardware identity are recorded"
                    ),
                },
            ],
            "ab_comparability": [
                {
                    "status": "comparison-grade",
                    "basis": (
                        "baseline/candidate preflight passed and metadata "
                        "comparability checks passed"
                    ),
                    "promotion_gate": (
                        "Still check correctness, provenance, repeatability, "
                        "and external gates before proof-grade claims."
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
                        "matched rows are present, but baseline-only or "
                        "candidate-only rows must be reviewed as sample-count "
                        "changes before interpreting deltas."
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
                    "proof_grade_status": "not_established_by_report",
                    "basis": "hardware A/B proof requires at least three matched runs",
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
                        "clean promotion evidence requires matched artifacts and hardware identity"
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
                {"capability": "device_result_digests", "present": False},
            ],
            "report_json_section_inventory": [
                {
                    "heading": "Trace Config Rows",
                    "json_path": "sections.trace_config_rows",
                    "json_section": "trace_config_rows",
                    "section_kind": "capture_configuration",
                    "claim_boundary": "requested_controls_not_recorded_evidence",
                },
                {
                    "heading": "Report Triage",
                    "json_path": "sections.report_triage",
                    "json_section": "report_triage",
                    "section_kind": "measurement_guidance",
                    "claim_boundary": "diagnostic_routing_not_evidence",
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
                    "json_path": "sections.introspection_capability_rows",
                    "json_section": "introspection_capability_rows",
                    "section_kind": "introspection",
                    "claim_boundary": "capability_presence_not_payload_evidence",
                },
                {
                    "heading": "Introspection Artifacts",
                    "json_path": "sections.introspection_artifacts",
                    "json_section": "introspection_artifacts",
                    "section_kind": "introspection",
                    "claim_boundary": "system_under_test_diagnostic",
                },
                {
                    "heading": "Introspection Section Inventory",
                    "json_path": "sections.introspection_section_inventory",
                    "json_section": "introspection_section_inventory",
                    "section_kind": "introspection_inventory",
                    "claim_boundary": "introspection_section_discovery",
                },
                {
                    "heading": "Provider Payload Boundary Inventory Rows",
                    "json_path": "sections.provider_payload_boundary_inventory_rows",
                    "json_section": "provider_payload_boundary_inventory_rows",
                    "section_kind": "introspection_inventory",
                    "claim_boundary": "payload_boundary_inventory_not_evidence",
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
                    "claim_boundary": "system_under_test_backend_provider_boundary",
                },
                {
                    "heading": "Backend Fail-Closed Root Causes",
                    "json_path": "sections.backend_fail_closed_root_causes",
                    "json_section": "backend_fail_closed_root_causes",
                    "section_kind": "backend_diagnostic",
                    "claim_boundary": (
                        "system_under_test_backend_fail_closed_diagnostic"
                    ),
                },
                {
                    "heading": "Debug Payload Artifact Summary Rows",
                    "json_path": "sections.debug_payload_artifact_summary_rows",
                    "json_section": "debug_payload_artifact_summary_rows",
                    "section_kind": "debug_payload_diagnostic",
                    "claim_boundary": "debug_payloads_can_perturb_timing",
                },
                {
                    "heading": "Token Quality Summary Rows",
                    "json_path": "sections.token_quality_summary_rows",
                    "json_section": "token_quality_summary_rows",
                    "section_kind": "sidecar",
                    "claim_boundary": "system_under_test_diagnostic_not_oracle",
                },
                {
                    "heading": "Oracle Reference Summary Rows",
                    "json_path": "sections.oracle_reference_summary_rows",
                    "json_section": "oracle_reference_summary_rows",
                    "section_kind": "sidecar",
                    "claim_boundary": (
                        "external_oracle_reference_anchor_not_sut_oracle_evidence"
                    ),
                },
                {
                    "heading": "Planning Decision Sidecar Rows",
                    "json_path": "sections.planning_decision_sidecar_rows",
                    "json_section": "planning_decision_sidecar_rows",
                    "section_kind": "sidecar",
                    "claim_boundary": (
                        "planning_decision_diagnostic_not_model_evidence"
                    ),
                },
                {
                    "heading": "Token Quality Sidecar Rows",
                    "json_path": "sections.token_quality_sidecar_rows",
                    "json_section": "token_quality_sidecar_rows",
                    "section_kind": "sidecar",
                    "claim_boundary": "system_under_test_diagnostic_not_oracle",
                },
                {
                    "heading": "Top-K Token Sidecar Rows",
                    "json_path": "sections.topk_token_sidecar_rows",
                    "json_section": "topk_token_sidecar_rows",
                    "section_kind": "sidecar",
                    "claim_boundary": ("system_under_test_topk_diagnostic_not_oracle"),
                },
                {
                    "heading": "Tensor Payload Sidecar Rows",
                    "json_path": "sections.tensor_payload_sidecar_rows",
                    "json_section": "tensor_payload_sidecar_rows",
                    "section_kind": "sidecar",
                    "claim_boundary": "system_under_test_payload_diagnostic",
                },
                {
                    "heading": "K/V Payload Digest Sidecar Rows",
                    "json_path": "sections.kv_payload_digest_sidecar_rows",
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
                    "claim_boundary": "system_under_test_final_logit_diagnostic",
                },
                {
                    "heading": "Activation Digest Sidecar Rows",
                    "json_path": "sections.activation_digest_sidecar_rows",
                    "json_section": "activation_digest_sidecar_rows",
                    "section_kind": "sidecar",
                    "claim_boundary": (
                        "system_under_test_activation_digest_diagnostic"
                    ),
                },
                {
                    "heading": "Scheduler Packet Lineage Sidecar Rows",
                    "json_path": "sections.scheduler_packet_lineage_sidecar_rows",
                    "json_section": "scheduler_packet_lineage_sidecar_rows",
                    "section_kind": "sidecar",
                    "claim_boundary": ("system_under_test_scheduler_packet_diagnostic"),
                },
                {
                    "heading": "Scheduler K/V Shard Lifecycle Sidecar Rows",
                    "json_path": ("sections.scheduler_kv_shard_lifecycle_sidecar_rows"),
                    "json_section": "scheduler_kv_shard_lifecycle_sidecar_rows",
                    "section_kind": "sidecar",
                    "claim_boundary": (
                        "system_under_test_scheduler_kv_lifecycle_diagnostic"
                    ),
                },
                {
                    "heading": "Scheduler Listener Sparse Logit Sidecar Rows",
                    "json_path": (
                        "sections.scheduler_listener_sparse_logit_sidecar_rows"
                    ),
                    "json_section": ("scheduler_listener_sparse_logit_sidecar_rows"),
                    "section_kind": "sidecar",
                    "claim_boundary": "system_under_test_scheduler_diagnostic",
                },
                {
                    "heading": "Device DMA Lifecycle Sidecar Rows",
                    "json_path": "sections.device_dma_lifecycle_sidecar_rows",
                    "json_section": "device_dma_lifecycle_sidecar_rows",
                    "section_kind": "sidecar",
                    "claim_boundary": "system_under_test_device_dma_diagnostic",
                },
                {
                    "heading": "Attention Page Trace Sidecar Rows",
                    "json_path": "sections.attention_page_trace_sidecar_rows",
                    "json_section": "attention_page_trace_sidecar_rows",
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
                    "next_action": "inspect_matching_introspection_report_sections",
                },
                {
                    "config_status": "requested_with_missing_sidecars",
                    "requested_sidecar_controls": (
                        "tensor_payloads,device_result_digests"
                    ),
                    "recorded_sidecar_capabilities": "tensor_payloads",
                    "missing_requested_sidecar_controls": "device_result_digests",
                    "introspection_level": "deep",
                    "compile_feature_trace_introspection": True,
                    "deep_introspection_effective": True,
                    "next_action": "enable_missing_sidecar_controls",
                },
            ],
            "introspection_capability_rows": [
                {
                    "capture_capability": "token_quality",
                    "capability_status": "recorded",
                    "matching_artifact_count": 1,
                    "claim_boundary": "system_under_test_diagnostic_not_oracle",
                    "next_action": "inspect_token_quality_rows",
                }
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
                    "boundary_status": "available_from_scheduler_protocol_boundary",
                    "producer_status": "provider_callback_present",
                    "producer_contract": (
                        "fpga_scheduler_batch_dispatch_emits_scheduler_kv_save_payload_digests"
                    ),
                    "payload_record_policy": "sha256_digest_plus_bounded_f32_sample",
                    "payload_sensitivity": "scheduler_kv_save_values",
                    "claim_boundary": "system_under_test_scheduler_kv_payload_diagnostic",
                    "next_action": "inspect_report_section",
                },
                {
                    "provider_id": "generic",
                    "payload_lane": "device_result_digests",
                    "capture_status": "route_available_no_provider_producer",
                    "capture_capability": "device_result_digests",
                    "artifact_kind": "device_result_digests",
                    "capture_control": "ARES_TRACE_RECORD_DEVICE_RESULTS=1",
                    "artifact_count": 0,
                    "matching_provider_artifact_count": 0,
                    "artifact_kind_recorded_count": 1,
                    "artifact_kind_recorded_backend_count": 1,
                    "artifact_kind_recorded_backend_ids": "fpga",
                    "report_section": "device_result_digest_sidecar_rows",
                    "boundary_status": "route_available_no_provider_producer_yet",
                    "producer_status": "runtime_route_only_no_provider_producer",
                    "producer_contract": "runtime_sidecar_route_only",
                    "payload_record_policy": "sha256_digest_plus_bounded_f32_sample",
                    "payload_sensitivity": "device_result_values",
                    "claim_boundary": (
                        "system_under_test_device_result_digest_diagnostic"
                    ),
                    "next_action": "wait_for_explicit_provider_payload_boundary",
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
                    "sha256": "b" * 64,
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
                    "metadata_keys": "provider_stage,target_plan_validation_status",
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
                    "metadata_keys": "provider_stage,root_cause_stage,root_cause",
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
                        "target_plan_validation_status=rejected_scheduler_runtime_table_missing"
                    ),
                    "model_id": "trace-model",
                    "request_id": "boundary-req-0",
                    "generation_id": "boundary-gen-0",
                    "targetplan_op_id": "tp.boundary.0",
                    "failure_reason": "targetplan_validation_failed",
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
                        "target_plan_validation_status=rejected_scheduler_runtime_table_missing"
                    ),
                    "event_kind": "forward_failed",
                    "failure_count": 1,
                    "example_model_id": "trace-model",
                    "example_request_id": "boundary-req-0",
                    "example_generation_id": "boundary-gen-0",
                    "example_targetplan_op_id": "tp.boundary.0",
                    "example_failure_reason": "targetplan_validation_failed",
                }
            ],
            "debug_payload_artifact_summary_rows": [
                {
                    "artifact_kind": "attention_page_trace",
                    "payload_summary_status": "recorded",
                    "row_count": "1",
                    "byte_count": "918",
                    "sampling_policy": "selected attention page summaries",
                    "token_window": "attention-page:7004",
                    "sensitivity": "local-only",
                    "compile_features": "trace-introspection",
                    "report_section": "attention_page_trace_sidecar_rows",
                    "debug_payload_boundary": "debug_payloads_can_perturb_timing",
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
                    "oracle_artifact_sha256": "a" * 64,
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
                    "oracle_reference_role": "external_hf_cpu_reference",
                    "hf_cpu_oracle_artifact_path": "correctness_hf_cpu_oracle.txt",
                    "hf_cpu_oracle_sha256": "a" * 64,
                    "expected_oracle_source": "hf_transformers_pytorch_cpu",
                    "oracle_reference_status": "external_reference_hash_recorded",
                    "sut_classification": "system_under_test",
                    "correctness_claim_status": "not_oracle_evidence",
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
                    "artifact_sha256": "b" * 64,
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
                    "target_plan_statement_index": "44",
                    "target_plan_statement_kind": "span",
                    "target_plan_statement_name": "provider_payload",
                    "layer": "31",
                    "tensor_payload_kind": "tensor_payload",
                    "tensor_name": "provider_payload",
                    "tensor_role": "provider_device_payload",
                    "element_type": "f32",
                    "shape": "[2]",
                    "element_count": "2",
                    "digest_sha256": "a" * 64,
                    "sample_value_count": "2",
                    "sample_finite_count": "2",
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
                    "targetplan_op_id": "",
                    "targetplan_action": "kv_cache",
                    "target_plan_statement_index": "21",
                    "target_plan_statement_kind": "kv_cache",
                    "target_plan_statement_name": "kv_save_layer_0_key",
                    "layer": "0",
                    "tensor_payload_kind": "kv_payload_digest",
                    "tensor_name": "scheduler_kv_save.layer_0.buf_1.k",
                    "tensor_role": "kv_key",
                    "element_type": "f32",
                    "shape": "[16]",
                    "element_count": "16",
                    "digest_sha256": "b" * 64,
                    "sample_value_count": "4",
                    "sample_finite_count": "4",
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
                    "target_plan_statement_index": "99",
                    "target_plan_statement_kind": "span",
                    "target_plan_statement_name": "ares_logits",
                    "layer": "31",
                    "intrinsic": "topk",
                    "tensor_payload_kind": "logit_slice",
                    "tensor_name": "ares_logits",
                    "tensor_role": "logits",
                    "element_type": "f32",
                    "shape": "[1, 32000]",
                    "element_count": "32000",
                    "digest_sha256": "a" * 64,
                    "sample_start": "7",
                    "sample_stride": "1",
                    "sample_value_count": "4",
                    "sample_finite_count": "2",
                    "sample_min": "-0.25",
                    "sample_max": "0.5",
                    "sample_nan_count": "1",
                    "sample_pos_inf_count": "0",
                    "sample_neg_inf_count": "1",
                    "sample_values": '[0.5, "-Infinity", "NaN", -0.25]',
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
                    "target_plan_statement_index": "42",
                    "target_plan_statement_kind": "rmsnorm",
                    "target_plan_statement_name": "layer_0_activation",
                    "layer": "0",
                    "intrinsic": "rmsnorm",
                    "tensor_payload_kind": "activation_digest",
                    "tensor_name": "layer_0.mlp.down_proj.activation",
                    "tensor_role": "activation",
                    "element_type": "f32",
                    "shape": "[1, 4096]",
                    "element_count": "4096",
                    "digest_sha256": "b" * 64,
                    "sample_start": "0",
                    "sample_stride": "8",
                    "sample_value_count": "4",
                    "sample_finite_count": "4",
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
                    "targetplan_op_id": "stmt.00004.matmul",
                    "targetplan_action": "matmul",
                    "target_plan_statement_index": "4",
                    "target_plan_statement_kind": "matmul",
                    "target_plan_statement_name": "wcls",
                    "layer": "",
                    "intrinsic": "fpga.matmul",
                    "tensor_payload_kind": "device_result_digest",
                    "tensor_name": "fpga_scheduler_forward_batch_result",
                    "tensor_role": "scheduler_device_result",
                    "element_type": "f32",
                    "shape": "[4]",
                    "element_count": "4",
                    "digest_sha256": SHA_C,
                    "sample_start": "0",
                    "sample_stride": "1",
                    "sample_value_count": "2",
                    "sample_finite_count": "2",
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
                    "executor_shape": "fullscheduler_forward_batch_v1",
                    "executor_status": "executed_fullscheduler_forward_batch_v1",
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
                    "prior_host_gof_staging_status": "page_info_published",
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
                    "executor_shape": "fullscheduler_forward_batch_v1",
                    "executor_status": "executed_fullscheduler_forward_batch_v1",
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
                    "prior_host_gof_staging_status": "page_info_published",
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
                    "executor_shape": "fullscheduler_forward_batch_v1",
                    "executor_status": "executed_fullscheduler_forward_batch_v1",
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
            "introspection_section_inventory": [
                {
                    "capture_capability": "token_quality",
                    "artifact_kind": "token_quality",
                    "heading": "Token Quality Summary Rows",
                    "json_section": "token_quality_summary_rows",
                    "capability_present": True,
                    "artifact_count": 1,
                    "section_status": "available",
                    "claim_boundary": "system_under_test_diagnostic_not_oracle",
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
                    "claim_boundary": "system_under_test_diagnostic_not_oracle",
                },
                {
                    "capture_capability": "topk_rows",
                    "artifact_kind": "token_quality",
                    "heading": "Top-K Token Sidecar Rows",
                    "json_section": "topk_token_sidecar_rows",
                    "capability_present": True,
                    "artifact_count": 1,
                    "section_status": "available",
                    "claim_boundary": ("system_under_test_topk_diagnostic_not_oracle"),
                },
                {
                    "capture_capability": "logit_slices",
                    "artifact_kind": "logit_slices",
                    "heading": "Logit Slice Sidecar Rows",
                    "json_section": "logit_slice_sidecar_rows",
                    "capability_present": True,
                    "artifact_count": 1,
                    "section_status": "available",
                    "claim_boundary": "system_under_test_final_logit_diagnostic",
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
                    "heading": "Scheduler Listener Sparse Logit Sidecar Rows",
                    "json_section": ("scheduler_listener_sparse_logit_sidecar_rows"),
                    "capability_present": True,
                    "artifact_count": 1,
                    "section_status": "available",
                    "claim_boundary": "system_under_test_scheduler_diagnostic",
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
                    "compile_features": "trace-introspection,deep-trace",
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
                    "sampling_policy": "bounded tensor payload summaries only",
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
                        "oracle_reference_summary_rows,token_quality_sidecar_rows,"
                        "token_quality_summary_rows,topk_token_sidecar_rows"
                    ),
                    "claim_boundaries": (
                        "external_oracle_reference_anchor_not_sut_oracle_evidence,"
                        "system_under_test_diagnostic_not_oracle,"
                        "system_under_test_topk_diagnostic_not_oracle"
                    ),
                }
            ],
        },
    }


class AresIngestCliTest(unittest.TestCase):
    def test_slugify_provider_model(self) -> None:
        self.assertEqual(
            slugify("meta-llama/Llama-3.1-8B-Instruct"),
            "meta-llama-llama-3-1-8b-instruct",
        )

    def test_refiner_defaults_to_codex(self) -> None:
        args = build_parser().parse_args(["Provider/Model"])
        cfg = config_from_args(args)

        self.assertEqual(cfg.refinement_command, DEFAULT_REFINER_COMMAND)
        self.assertEqual(cfg.driver, "codex")

    def test_driver_selection_supports_claude_and_custom_commands(self) -> None:
        claude_cfg = config_from_args(
            build_parser().parse_args(["--driver", "claude", "Provider/Model"])
        )
        self.assertEqual(claude_cfg.refinement_command, CLAUDE_REFINER_COMMAND)

        custom_cfg = config_from_args(
            build_parser().parse_args(
                [
                    "--driver",
                    "custom",
                    "--driver-command",
                    'printf "%s\\n" "$FIRST_FAILED_GATE"',
                    "Provider/Model",
                ]
            )
        )
        self.assertEqual(custom_cfg.driver, "custom")
        self.assertEqual(
            custom_cfg.refinement_command,
            'printf "%s\\n" "$FIRST_FAILED_GATE"',
        )

        alias_cfg = config_from_args(
            build_parser().parse_args(
                ["--refinement-command", "echo old-alias", "Provider/Model"]
            )
        )
        self.assertEqual(alias_cfg.refinement_command, "echo old-alias")

        with self.assertRaises(AresIngestError):
            config_from_args(
                build_parser().parse_args(["--driver", "custom", "Provider/Model"])
            )

    def test_no_refiner_disables_refinement_command(self) -> None:
        args = build_parser().parse_args(["--no-refiner", "Provider/Model"])
        cfg = config_from_args(args)

        self.assertIsNone(cfg.refinement_command)

    def test_initialize_run_writes_durable_state(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            args = build_parser().parse_args(
                [
                    "--ares-repo",
                    str(root),
                    "--run-dir",
                    str(run_dir),
                    "--setup-only",
                    "Provider/Model",
                ]
            )
            cfg = config_from_args(args)

            state = initialize_run(cfg)

            self.assertEqual(state["status"], "initialized_setup_only")
            self.assertTrue((run_dir / "model_spec.json").exists())
            self.assertTrue((run_dir / "state.json").exists())
            self.assertTrue((run_dir / "reward.json").exists())
            self.assertTrue((run_dir / "reward.txt").exists())
            self.assertTrue((run_dir / "handoff.md").exists())
            self.assertTrue((run_dir / "steering.json").exists())
            self.assertTrue((run_dir / "steering.md").exists())
            spec = json.loads((run_dir / "model_spec.json").read_text())
            self.assertEqual(spec["safe_model"], "provider-model")
            self.assertEqual(spec["gate_profile"], "cpu-only")
            self.assertNotIn("cpp_tvd", spec["required_gates"])
            self.assertEqual(spec["explicit_gates"]["model_spec"]["passed"], True)
            self.assertIn("shortcut_scan", spec["required_gates"])
            self.assertIn("artifact_consistency", spec["required_gates"])
            self.assertEqual(spec["expected_model_ids"], ["Provider/Model"])
            self.assertEqual(
                spec["explicit_gates"]["shortcut_scan"]["artifact_validator"],
                "shortcut_scan",
            )
            self.assertEqual(
                spec["policy"]["hf_cpu_goldens"],
                "capture_once_per_stable_oracle_tuple",
            )
            self.assertEqual(
                spec["policy"]["ordinary_backend_loop_reference"],
                "cached_hf_cpu_token_logit_goldens",
            )
            self.assertEqual(
                spec["policy"]["debug_loop_priority"],
                [
                    "cached_hf_cpu_logit_comparison",
                    "focused_backend_or_module_slice",
                    "short_depth_backend_generation",
                    "longer_depth_backend_generation",
                    "explicit_late_cpp_comparison_milestone",
                ],
            )
            self.assertEqual(spec["policy"]["fastest_verifier_first"], True)
            self.assertEqual(
                spec["policy"]["avoid_recapturing_unchanged_hf_logits"], True
            )
            self.assertEqual(
                spec["policy"]["defer_cpp_comparison_until_milestone_candidate"],
                True,
            )
            self.assertEqual(
                state["reward"]["gates"]["shortcut_scan"]["passed"],
                True,
            )
            self.assertEqual(state["reward"]["first_failed_gate"], "hf_cpu_oracle")
            self.assertEqual(state["refinement_loop"], "setup_only")
            self.assertEqual(
                [skill["name"] for skill in state["workflow_skills"]],
                ["command-wiggum", "ares-evidence", "ares-python", "command-fess"],
            )
            self.assertIn(
                "tools/oracles/hf-cpu/",
                state["workflow_skills"][2]["allowed_scope"],
            )
            handoff = (run_dir / "handoff.md").read_text()
            self.assertIn("## Workflow Skills", handoff)
            self.assertIn("`ares-python`", handoff)
            self.assertIn("validate-jsonl", handoff)

    def test_trace_report_json_is_recorded_in_state_handoff_and_prompt(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            report_path = run_dir / "trace-report.json"
            report_path.write_text(json.dumps(trace_report_payload()))
            (run_dir / "model_spec.json").write_text(
                json.dumps(
                    {
                        "model": "Provider/Model",
                        "trace_report_json": "trace-report.json",
                    }
                )
            )
            cfg = config_from_args(
                build_parser().parse_args(
                    [
                        "--ares-repo",
                        str(root),
                        "--run-dir",
                        str(run_dir),
                        "--setup-only",
                        "Provider/Model",
                    ]
                )
            )

            state = initialize_run(cfg)

            self.assertEqual(state["reward"]["first_failed_gate"], "hf_cpu_oracle")
            self.assertEqual(state["trace_report"]["path"], str(report_path.resolve()))
            self.assertEqual(len(state["trace_report"]["sha256"]), 64)
            self.assertEqual(state["trace_report"]["report_grade"], "diagnostic")
            self.assertEqual(
                state["trace_report"]["report_triage_status_counts"],
                {"needs_measurement": 1},
            )
            self.assertEqual(
                state["trace_report"]["report_triage_samples"][0][
                    "first_useful_section"
                ],
                "sections.next_measurements",
            )
            self.assertEqual(
                state["trace_report"]["report_triage_samples"][0]["claim_boundary"],
                "diagnostic_routing_not_evidence",
            )
            self.assertEqual(state["trace_report"]["supported_claim_count"], 1)
            self.assertEqual(
                state["trace_report"]["correctness_evidence_status_counts"],
                {"not_recorded": 1},
            )
            self.assertEqual(
                state["trace_report"]["correctness_evidence_proof_grade_status_counts"],
                {"not_established_by_report": 1},
            )
            self.assertEqual(
                state["trace_report"]["evidence_artifact_check_status_counts"],
                {"ok": 1},
            )
            self.assertEqual(
                state["trace_report"]["promotion_gate_summary_status_counts"],
                {"passed": 1},
            )
            self.assertEqual(
                state["trace_report"][
                    "promotion_gate_summary_proof_grade_status_counts"
                ],
                {"not_established_by_report": 1},
            )
            self.assertEqual(
                state["trace_report"]["trace_mode_guardrail_mode_counts"],
                {"timeline-lite": 1},
            )
            self.assertEqual(
                state["trace_report"]["trace_mode_guardrail_overhead_counts"],
                {"low_overhead": 1},
            )
            self.assertEqual(
                state["trace_report"]["ab_provenance_status_counts"],
                {"clean_provenance": 3},
            )
            self.assertEqual(
                state["trace_report"]["ab_provenance_source_state_counts"],
                {"clean": 2},
            )
            self.assertEqual(
                state["trace_report"]["ab_provenance_artifact_hash_status_counts"],
                {"matched": 3},
            )
            self.assertEqual(
                state["trace_report"]["ab_provenance_proof_grade_status_counts"],
                {"not_established_by_report": 3},
            )
            self.assertEqual(
                state["trace_report"]["ab_comparability_status_counts"],
                {"comparison-grade": 1},
            )
            self.assertEqual(
                state["trace_report"]["ab_coverage_status_counts"],
                {"partial_overlap": 1},
            )
            self.assertEqual(state["trace_report"]["ab_coverage_total_rows"], 5)
            self.assertEqual(state["trace_report"]["ab_coverage_matched_rows"], 4)
            self.assertEqual(
                state["trace_report"]["ab_coverage_baseline_only_rows"],
                1,
            )
            self.assertEqual(
                state["trace_report"]["ab_coverage_candidate_only_rows"],
                0,
            )
            self.assertEqual(
                state["trace_report"]["ab_repeatability_status_counts"],
                {"insufficient_for_proof": 1},
            )
            self.assertEqual(
                state["trace_report"]["ab_repeatability_proof_grade_status_counts"],
                {"not_established_by_report": 1},
            )
            self.assertEqual(state["trace_report"]["ab_repeatability_baseline_runs"], 1)
            self.assertEqual(
                state["trace_report"]["ab_repeatability_candidate_runs"], 1
            )
            self.assertEqual(
                state["trace_report"][
                    "ab_repeatability_required_matched_runs_for_hardware_proof"
                ],
                3,
            )
            self.assertEqual(state["trace_report"]["ab_repeatability_matched_rows"], 4)
            self.assertEqual(
                state["trace_report"]["capture_process_kind_counts"],
                {"runares": 1},
            )
            self.assertEqual(
                state["trace_report"]["capture_backend_counts"],
                {"fpga": 1},
            )
            self.assertEqual(
                state["trace_report"]["capture_trace_mode_counts"],
                {"timeline-lite": 1},
            )
            self.assertEqual(
                state["trace_report"]["run_provenance_source_state_counts"],
                {"clean": 1},
            )
            self.assertEqual(
                state["trace_report"]["artifact_identity_artifact_counts"],
                {"ares_plan": 1, "target_plan": 1},
            )
            self.assertEqual(
                state["trace_report"]["artifact_identity_load_status_counts"],
                {"loaded": 2},
            )
            self.assertEqual(
                state["trace_report"]["artifact_identity_check_status_counts"],
                {"ok": 1},
            )
            self.assertEqual(
                state["trace_report"]["capture_capability_present_counts"],
                {"False": 1, "True": 1},
            )
            self.assertEqual(
                state["trace_report"]["introspection_capability_status_counts"],
                {"recorded": 1},
            )
            self.assertEqual(state["trace_report"]["introspection_artifact_count"], 2)
            self.assertEqual(
                state["trace_report"]["introspection_artifact_status_counts"],
                {"missing": 1, "recorded": 1},
            )
            self.assertEqual(
                state["trace_report"]["introspection_artifact_kind_counts"],
                {"tensor_payload": 1, "token_quality": 1},
            )
            self.assertEqual(
                state["trace_report"]["introspection_artifact_format_counts"],
                {"tensor_payload_jsonl": 1, "token_quality_jsonl": 1},
            )
            self.assertEqual(
                state["trace_report"]["introspection_artifact_sensitivity_counts"],
                {"local-only": 1, "tensor_digest": 1},
            )
            self.assertEqual(
                state["trace_report"]["introspection_artifact_compile_feature_counts"],
                {"deep-trace": 1, "trace-introspection": 2},
            )
            self.assertEqual(
                state["trace_report"]["introspection_artifact_row_count_total"],
                5,
            )
            self.assertEqual(
                state["trace_report"]["introspection_artifact_byte_count_total"],
                200,
            )
            self.assertEqual(
                state["trace_report"]["trace_config_status_counts"],
                {
                    "requested_and_recorded": 1,
                    "requested_with_missing_sidecars": 1,
                },
            )
            self.assertEqual(
                state["trace_report"]["trace_config_missing_requested_sidecar_counts"],
                {"device_result_digests": 1},
            )
            self.assertEqual(
                state["trace_report"]["provider_payload_boundary_status_counts"],
                {
                    "recorded_artifact": 1,
                    "route_available_no_provider_producer": 1,
                },
            )
            self.assertEqual(
                state["trace_report"]["provider_payload_boundary_route_only_count"],
                1,
            )
            self.assertEqual(
                state["trace_report"]["provider_payload_boundary_route_only_lanes"],
                ["generic/device_result_digests"],
            )
            self.assertEqual(
                state["trace_report"]["provider_payload_boundary_recorded_count"],
                1,
            )
            self.assertEqual(
                state["trace_report"]["provider_payload_boundary_recorded_lanes"],
                ["fpga/kv_payload_digests"],
            )
            self.assertEqual(
                state["trace_report"]["trace_event_artifact_status_counts"],
                {"ok": 1},
            )
            self.assertEqual(
                state["trace_report"]["trace_event_artifact_event_kind_counts"],
                {"span_end": 1, "span_start": 1},
            )
            self.assertEqual(
                state["trace_report"]["backend_event_artifact_status_counts"],
                {"ok": 1},
            )
            self.assertEqual(
                state["trace_report"]["backend_event_artifact_event_kind_counts"],
                {"backend_selected": 1, "forward_failed": 1},
            )
            self.assertEqual(
                state["trace_report"]["backend_event_row_event_kind_counts"],
                {"backend_selected": 1, "forward_failed": 1},
            )
            self.assertEqual(
                state["trace_report"]["backend_event_row_backend_counts"],
                {"fpga": 2},
            )
            self.assertEqual(
                state["trace_report"]["backend_provider_boundary_status_counts"],
                {"fail_closed": 1, "ok": 1},
            )
            self.assertEqual(
                state["trace_report"]["backend_provider_boundary_stage_counts"],
                {"forward": 1, "session_open": 1},
            )
            self.assertEqual(
                state["trace_report"]["backend_provider_boundary_root_stage_counts"],
                {"targetplan_validation": 1},
            )
            self.assertEqual(
                state["trace_report"]["backend_fail_closed_root_cause_backend_counts"],
                {"fpga": 1},
            )
            self.assertEqual(
                state["trace_report"]["backend_fail_closed_root_cause_stage_counts"],
                {"forward": 1},
            )
            self.assertEqual(
                state["trace_report"][
                    "backend_fail_closed_root_cause_root_stage_counts"
                ],
                {"targetplan_validation": 1},
            )
            self.assertEqual(
                state["trace_report"]["timeline_query_summary_status_counts"],
                {"rendered": 1},
            )
            self.assertEqual(
                state["trace_report"]["debug_payload_artifact_summary_status_counts"],
                {"recorded": 1},
            )
            self.assertEqual(
                state["trace_report"]["token_quality_summary_status_counts"],
                {"present": 1},
            )
            self.assertEqual(
                state["trace_report"]["token_quality_summary_topk_status_counts"],
                {"selected_is_top1": 1},
            )
            self.assertEqual(
                state["trace_report"]["oracle_reference_summary_status_counts"],
                {"external_reference_hash_recorded": 1},
            )
            self.assertEqual(
                state["trace_report"]["oracle_reference_summary_correctness_counts"],
                {"not_oracle_evidence": 1},
            )
            self.assertEqual(
                state["trace_report"]["planning_decision_sidecar_row_kind_counts"],
                {"lean_planning_phase": 1},
            )
            self.assertEqual(
                state["trace_report"]["planning_decision_sidecar_phase_counts"],
                {"lean.target_plan_lower": 1},
            )
            self.assertEqual(
                state["trace_report"]["token_quality_sidecar_status_counts"],
                {"present": 1},
            )
            self.assertEqual(
                state["trace_report"]["token_quality_sidecar_finish_reason_counts"],
                {"stop": 1},
            )
            self.assertEqual(
                state["trace_report"]["topk_token_sidecar_selected_status_counts"],
                {"selected_token": 1},
            )
            self.assertEqual(
                state["trace_report"]["topk_token_sidecar_score_kind_counts"],
                {"logprob": 1},
            )
            self.assertEqual(
                state["trace_report"]["tensor_payload_sidecar_status_counts"],
                {"ok": 1},
            )
            self.assertEqual(
                state["trace_report"]["tensor_payload_sidecar_kind_counts"],
                {"tensor_payload": 1},
            )
            self.assertEqual(
                state["trace_report"]["tensor_payload_sidecar_role_counts"],
                {"provider_device_payload": 1},
            )
            self.assertEqual(
                state["trace_report"]["kv_payload_digest_sidecar_status_counts"],
                {"ok": 1},
            )
            self.assertEqual(
                state["trace_report"]["kv_payload_digest_sidecar_role_counts"],
                {"kv_key": 1},
            )
            self.assertEqual(
                state["trace_report"]["logit_slice_sidecar_role_counts"],
                {"logits": 1},
            )
            self.assertEqual(
                state["trace_report"]["logit_slice_sidecar_action_counts"],
                {"final_logits": 1},
            )
            self.assertEqual(
                state["trace_report"]["activation_digest_sidecar_role_counts"],
                {"activation": 1},
            )
            self.assertEqual(
                state["trace_report"]["activation_digest_sidecar_intrinsic_counts"],
                {"rmsnorm": 1},
            )
            self.assertEqual(
                state["trace_report"]["device_result_digest_sidecar_status_counts"],
                {"ok": 1},
            )
            self.assertEqual(
                state["trace_report"]["device_result_digest_sidecar_role_counts"],
                {"scheduler_device_result": 1},
            )
            self.assertEqual(
                state["trace_report"]["device_result_digest_sidecar_action_counts"],
                {"matmul": 1},
            )
            self.assertEqual(
                state["trace_report"]["device_result_digest_sidecar_intrinsic_counts"],
                {"fpga.matmul": 1},
            )
            self.assertEqual(
                state["trace_report"]["scheduler_packet_lineage_sidecar_status_counts"],
                {"ok": 1},
            )
            self.assertEqual(
                state["trace_report"][
                    "scheduler_packet_lineage_sidecar_executor_counts"
                ],
                {"executed_fullscheduler_forward_batch_v1": 1},
            )
            self.assertEqual(
                state["trace_report"][
                    "scheduler_kv_shard_lifecycle_sidecar_status_counts"
                ],
                {"ok": 1},
            )
            self.assertEqual(
                state["trace_report"][
                    "scheduler_kv_shard_lifecycle_sidecar_lifecycle_counts"
                ],
                {"observed": 1},
            )
            self.assertEqual(
                state["trace_report"][
                    "scheduler_listener_sparse_logit_sidecar_listener_status_counts"
                ],
                {"observed": 1},
            )
            self.assertEqual(
                state["trace_report"][
                    "scheduler_listener_sparse_logit_sidecar_executor_counts"
                ],
                {"executed_fullscheduler_forward_batch_v1": 1},
            )
            self.assertEqual(
                state["trace_report"]["device_dma_lifecycle_sidecar_status_counts"],
                {"ok": 1},
            )
            self.assertEqual(
                state["trace_report"]["device_dma_lifecycle_sidecar_stage_counts"],
                {"dma_completion": 1},
            )
            self.assertEqual(
                state["trace_report"]["device_dma_lifecycle_sidecar_queue_counts"],
                {"load_weight_q": 1},
            )
            self.assertEqual(
                state["trace_report"]["attention_page_trace_sidecar_status_counts"],
                {"ok": 1},
            )
            self.assertEqual(
                state["trace_report"]["attention_page_trace_sidecar_action_counts"],
                {"attention": 1},
            )
            self.assertEqual(
                state["trace_report"]["introspection_section_inventory_status_counts"],
                {"available": 7},
            )
            self.assertEqual(
                state["trace_report"][
                    "introspection_section_inventory_capability_counts"
                ],
                {
                    "activation_digests": 1,
                    "deep_introspection": 1,
                    "logit_slices": 1,
                    "scheduler_packet_lineage": 1,
                    "token_quality": 2,
                    "topk_rows": 1,
                },
            )
            self.assertEqual(state["trace_report"]["report_json_section_count"], 30)
            self.assertEqual(
                state["trace_report"]["report_json_section_kind_counts"],
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
            spec = json.loads((run_dir / "model_spec.json").read_text())
            trace_gate = spec["validated_gates"]["trace_report"]
            self.assertTrue(trace_gate["passed"])
            self.assertEqual(
                trace_gate["detail"]["sha256"], state["trace_report"]["sha256"]
            )
            handoff = (run_dir / "handoff.md").read_text()
            self.assertIn("## Trace Report", handoff)
            self.assertIn("Report JSON sections", handoff)
            self.assertIn("Report JSON section: sections.trace_config_rows", handoff)
            self.assertIn("Report JSON section: sections.ab_provenance", handoff)
            self.assertIn("Report JSON section: sections.ab_comparability", handoff)
            self.assertIn("Report JSON section: sections.ab_coverage", handoff)
            self.assertIn("Report JSON section: sections.ab_repeatability", handoff)
            self.assertIn(
                "Report JSON section: sections.introspection_artifacts",
                handoff,
            )
            self.assertIn("Capture trace modes", handoff)
            self.assertIn("A/B provenance", handoff)
            self.assertIn("A/B artifact hashes", handoff)
            self.assertIn("A/B comparability", handoff)
            self.assertIn("A/B coverage", handoff)
            self.assertIn("A/B repeatability", handoff)
            self.assertIn("A/B provenance: comparison", handoff)
            self.assertIn("status=clean_provenance", handoff)
            self.assertIn("artifact_hashes=matched", handoff)
            self.assertIn("A/B comparability: comparison-grade", handoff)
            self.assertIn("A/B coverage: partial_overlap", handoff)
            self.assertIn("align=targetplan-op", handoff)
            self.assertIn("matched=4", handoff)
            self.assertIn("A/B repeatability: insufficient_for_proof", handoff)
            self.assertIn("required_runs=3", handoff)
            self.assertIn("Introspection raw artifacts", handoff)
            self.assertIn("Introspection raw artifact: token_quality", handoff)
            self.assertIn("path=introspection_token_quality.jsonl", handoff)
            self.assertIn(f"sha256={SHA_A}", handoff)
            self.assertIn("features=trace-introspection,deep-trace", handoff)
            self.assertIn("Capture backends", handoff)
            self.assertIn("Run provenance source states", handoff)
            self.assertIn("Artifact identities", handoff)
            self.assertIn("Artifact identity checks", handoff)
            self.assertIn("Capture capabilities", handoff)
            self.assertIn("Capture: trace-run-001", handoff)
            self.assertIn("process=runares", handoff)
            self.assertIn("mode=timeline-lite", handoff)
            self.assertIn("Run provenance: target/debug/runares", handoff)
            self.assertIn("source_state=clean", handoff)
            self.assertIn("Artifact identity: ares_plan", handoff)
            self.assertIn("load_status=loaded", handoff)
            self.assertIn("Artifact identity check: <all>", handoff)
            self.assertIn("Capture capability: token_quality present=True", handoff)
            self.assertIn(
                "Capture capability: device_result_digests present=False",
                handoff,
            )
            self.assertIn("Trace config: requested_and_recorded", handoff)
            self.assertIn("recorded=tensor_payloads", handoff)
            self.assertIn("Missing requested sidecars", handoff)
            self.assertIn("Trace config: requested_with_missing_sidecars", handoff)
            self.assertIn("missing=device_result_digests", handoff)
            self.assertIn("Provider payload boundaries", handoff)
            self.assertIn("Provider payload boundary: fpga/kv_payload_digests", handoff)
            self.assertIn("Recorded provider payload lanes", handoff)
            self.assertIn(
                "Recorded provider payload boundary: fpga/kv_payload_digests", handoff
            )
            self.assertIn("Route-only provider payload lanes", handoff)
            self.assertIn("generic/device_result_digests", handoff)
            self.assertIn(
                "Route-only provider payload boundary: generic/device_result_digests",
                handoff,
            )
            self.assertIn("provider_artifacts=2", handoff)
            self.assertIn("provider_artifacts=0", handoff)
            self.assertIn("same_kind_artifacts=2", handoff)
            self.assertIn("same_kind_artifacts=1", handoff)
            self.assertIn("same_kind_backends=fpga", handoff)
            self.assertIn("producer=provider_callback_present", handoff)
            self.assertIn(
                "producer=runtime_route_only_no_provider_producer",
                handoff,
            )
            self.assertIn("control=ARES_TRACE_RECORD_DEVICE_RESULTS=1", handoff)
            self.assertIn("Trace event artifacts", handoff)
            self.assertIn("Trace event artifact kinds", handoff)
            self.assertIn("Trace event artifact: trace-events.jsonl", handoff)
            self.assertIn("event_kinds=span_start,span_end", handoff)
            self.assertIn("Backend event artifacts", handoff)
            self.assertIn("Backend event artifact kinds", handoff)
            self.assertIn("Backend event artifact: backend-events.jsonl", handoff)
            self.assertIn("matching=2", handoff)
            self.assertIn("Backend event rows", handoff)
            self.assertIn("Backend event backends", handoff)
            self.assertIn("Backend event row: fpga", handoff)
            self.assertIn(
                "metadata_keys=provider_stage,root_cause_stage,root_cause",
                handoff,
            )
            self.assertIn("Backend provider boundaries", handoff)
            self.assertIn("Backend provider boundary stages", handoff)
            self.assertIn("Backend provider root stages", handoff)
            self.assertIn("Backend fail-closed stages", handoff)
            self.assertIn("Backend fail-closed root stages", handoff)
            self.assertIn("Backend provider boundary: fpga", handoff)
            self.assertIn("event=forward_failed", handoff)
            self.assertIn("status=fail_closed", handoff)
            self.assertIn("root_stage=targetplan_validation", handoff)
            self.assertIn("op=tp.boundary.0", handoff)
            self.assertIn("Backend fail-closed root cause: fpga", handoff)
            self.assertIn(
                "root=target_plan_validation_status=rejected_scheduler_runtime_table_missing",
                handoff,
            )
            self.assertIn(
                "contract=fpga_scheduler_batch_dispatch_emits_scheduler_kv_save_payload_digests",
                handoff,
            )
            self.assertIn("contract=runtime_sidecar_route_only", handoff)
            self.assertIn("policy=sha256_digest_plus_bounded_f32_sample", handoff)
            self.assertIn("sensitivity=scheduler_kv_save_values", handoff)
            self.assertIn("sensitivity=device_result_values", handoff)
            self.assertIn("Capture backend event JSONL", handoff)
            self.assertIn("Debug payload artifacts", handoff)
            self.assertIn("Debug payload artifact: attention_page_trace", handoff)
            self.assertIn("sensitivity=local-only", handoff)
            self.assertIn(
                "payload_boundary=debug_payloads_can_perturb_timing",
                handoff,
            )
            self.assertIn("Tensor payload sidecars", handoff)
            self.assertIn("Tensor payload roles", handoff)
            self.assertIn("Tensor payload sidecar: request=7005", handoff)
            self.assertIn("kind=tensor_payload", handoff)
            self.assertIn("role=provider_device_payload", handoff)
            self.assertIn("stmt_name=provider_payload", handoff)
            self.assertIn("sample_finite=2", handoff)
            self.assertIn("sample_nan=0", handoff)
            self.assertIn("K/V payload digest roles", handoff)
            self.assertIn("K/V payload digest sidecar: request=7006", handoff)
            self.assertIn("role=kv_key", handoff)
            self.assertIn("stmt_index=21", handoff)
            self.assertIn("stmt_kind=kv_cache", handoff)
            self.assertIn("stmt_name=kv_save_layer_0_key", handoff)
            self.assertIn("sample_finite=4", handoff)
            self.assertIn("sample_min=0.125", handoff)
            self.assertIn("Scheduler packet executors", handoff)
            self.assertIn("Scheduler packet lineage: request=7002", handoff)
            self.assertIn("executor=executed_fullscheduler_forward_batch_v1", handoff)
            self.assertIn("tokens_reused=5", handoff)
            self.assertIn("sparse_tokens=3", handoff)
            self.assertIn("Scheduler K/V lifecycle status", handoff)
            self.assertIn("Scheduler K/V lifecycle: request=7002", handoff)
            self.assertIn("lifecycle=observed", handoff)
            self.assertIn("kv_context_rows=64", handoff)
            self.assertIn("staging=page_info_published", handoff)
            self.assertIn("Scheduler sparse listener status", handoff)
            self.assertIn("Scheduler sparse listener: request=7002", handoff)
            self.assertIn("listener=observed", handoff)
            self.assertIn("sparse_topk_tokens=3", handoff)
            self.assertIn("Device DMA lifecycle", handoff)
            self.assertIn("Device DMA stages", handoff)
            self.assertIn("Device DMA lifecycle: request=7002", handoff)
            self.assertIn("queue=load_weight_q", handoff)
            self.assertIn("cacheblock_bytes=8192", handoff)
            self.assertIn("Attention page traces", handoff)
            self.assertIn("Attention page actions", handoff)
            self.assertIn("Attention page trace: request=7004", handoff)
            self.assertIn("attention_row=7", handoff)
            self.assertIn("m_star=3.0", handoff)
            self.assertIn("Token quality summaries", handoff)
            self.assertIn("Token quality top-k status", handoff)
            self.assertIn("Token quality summary: request=7001", handoff)
            self.assertIn("token_index=0", handoff)
            self.assertIn("topk=selected_is_top1", handoff)
            self.assertIn("boundary=external_oracle_reference_present", handoff)
            self.assertIn("Oracle references", handoff)
            self.assertIn("Oracle-reference correctness boundary", handoff)
            self.assertIn("Oracle reference summary: request=7001", handoff)
            self.assertIn("correctness=not_oracle_evidence", handoff)
            self.assertIn("sut=system_under_test", handoff)
            self.assertIn("Planning decision sidecars", handoff)
            self.assertIn("Planning decision phases", handoff)
            self.assertIn("Planning decision sidecar: row=lean_planning_phase", handoff)
            self.assertIn("phase=lean.target_plan_lower", handoff)
            self.assertIn("targetplan_ops=4", handoff)
            self.assertIn("Token quality sidecars", handoff)
            self.assertIn("Token quality finish reasons", handoff)
            self.assertIn("Token quality sidecar: request=7001", handoff)
            self.assertIn("finish=stop", handoff)
            self.assertIn("topk_count=2", handoff)
            self.assertIn("Top-K token candidate status", handoff)
            self.assertIn("Top-K token sidecar: request=7001", handoff)
            self.assertIn("candidate_status=selected_token", handoff)
            self.assertIn("rank=0", handoff)
            self.assertIn("Logit slice sidecars", handoff)
            self.assertIn("Logit slice sidecar: request=7005", handoff)
            self.assertIn("action=final_logits", handoff)
            self.assertIn("stmt_index=99", handoff)
            self.assertIn("stmt_name=ares_logits", handoff)
            self.assertIn("sample_finite=2", handoff)
            self.assertIn("sample_nan=1", handoff)
            self.assertIn("Activation digest sidecars", handoff)
            self.assertIn("Activation digest sidecar: request=7007", handoff)
            self.assertIn("intrinsic=rmsnorm", handoff)
            self.assertIn("stmt_index=42", handoff)
            self.assertIn("stmt_kind=rmsnorm", handoff)
            self.assertIn("stmt_name=layer_0_activation", handoff)
            self.assertIn("Introspection capability: token_quality", handoff)
            self.assertIn("Introspection sections", handoff)
            self.assertIn(
                "Introspection section: token_quality_summary_rows",
                handoff,
            )
            self.assertIn("status=available", handoff)
            self.assertIn("capability=token_quality", handoff)
            self.assertIn("artifacts=1", handoff)
            self.assertIn("Timeline query summary", handoff)
            self.assertIn("Timeline query summary: join-key-coverage", handoff)
            self.assertIn("rows=3", handoff)
            self.assertIn("native_sql=True", handoff)
            self.assertIn(
                "Command: `bin/ares-trace-query --query join-key-coverage`",
                handoff,
            )
            self.assertIn("Supported claim: trace preflight is answerable", handoff)
            self.assertIn("Correctness evidence: hf_cpu_oracle_tokens_logits", handoff)
            self.assertIn(
                "Evidence artifact check: metadata.evidence_artifacts", handoff
            )
            self.assertIn("Promotion gate: capture_preflight", handoff)
            self.assertIn("Trace mode guardrail: timeline-lite", handoff)
            self.assertIn(
                "Introspection section: planning_decision_sidecar_rows",
                handoff,
            )
            self.assertIn("set ARES_BACKEND_EVENT_ARTIFACT_DIR", handoff)

            reward = json.loads((run_dir / "reward.json").read_text())
            prompt = write_refinement_prompt(
                cfg,
                iteration=1,
                spec=spec,
                reward=reward,
            ).read_text()
            self.assertIn("Trace report JSON:", prompt)
            self.assertIn("## Trace Report Summary", prompt)
            self.assertIn("sections.report_grade", prompt)
            self.assertIn("sections.report_triage", prompt)
            self.assertIn("sections.answerability", prompt)
            self.assertIn("Report triage detail: status=needs_measurement", prompt)
            self.assertIn("Supported claim: trace preflight is answerable", prompt)
            self.assertIn("Correctness evidence: hf_cpu_oracle_tokens_logits", prompt)
            self.assertIn(
                "Evidence artifact check: metadata.evidence_artifacts", prompt
            )
            self.assertIn("Promotion gate: capture_preflight", prompt)
            self.assertIn("Trace mode guardrail: timeline-lite", prompt)
            self.assertIn("sections.report_json_section_inventory", prompt)
            self.assertIn("sections.capture", prompt)
            self.assertIn("sections.run_provenance", prompt)
            self.assertIn("sections.artifact_identities", prompt)
            self.assertIn("sections.artifact_identity_checks", prompt)
            self.assertIn("sections.capture_capabilities", prompt)
            self.assertIn("sections.ab_provenance", prompt)
            self.assertIn("sections.ab_comparability", prompt)
            self.assertIn("sections.ab_coverage", prompt)
            self.assertIn("sections.ab_repeatability", prompt)
            self.assertIn("sections.trace_config_rows", prompt)
            self.assertIn("sections.provider_payload_boundary_inventory_rows", prompt)
            self.assertIn("sections.trace_event_artifacts", prompt)
            self.assertIn("sections.backend_event_artifacts", prompt)
            self.assertIn("sections.backend_event_rows", prompt)
            self.assertIn("sections.backend_provider_boundaries", prompt)
            self.assertIn("sections.backend_fail_closed_root_causes", prompt)
            self.assertIn("sections.debug_payload_artifact_summary_rows", prompt)
            self.assertIn("sections.planning_decision_sidecar_rows", prompt)
            self.assertIn("sections.token_quality_sidecar_rows", prompt)
            self.assertIn("sections.topk_token_sidecar_rows", prompt)
            self.assertIn("sections.tensor_payload_sidecar_rows", prompt)
            self.assertIn("sections.kv_payload_digest_sidecar_rows", prompt)
            self.assertIn("sections.logit_slice_sidecar_rows", prompt)
            self.assertIn("sections.activation_digest_sidecar_rows", prompt)
            self.assertIn("sections.scheduler_packet_lineage_sidecar_rows", prompt)
            self.assertIn(
                "sections.scheduler_kv_shard_lifecycle_sidecar_rows",
                prompt,
            )
            self.assertIn(
                "sections.scheduler_listener_sparse_logit_sidecar_rows",
                prompt,
            )
            self.assertIn("sections.device_dma_lifecycle_sidecar_rows", prompt)
            self.assertIn("sections.attention_page_trace_sidecar_rows", prompt)
            self.assertIn("sections.device_result_digest_sidecar_rows", prompt)
            self.assertIn("missing_requested_sidecar_controls", prompt)
            self.assertIn("sections.token_quality_summary_rows", prompt)
            self.assertIn("sections.oracle_reference_summary_rows", prompt)
            self.assertIn("sections.introspection_artifacts", prompt)
            self.assertIn("sections.introspection_section_inventory", prompt)
            self.assertIn("sections.timeline_query_summary", prompt)
            self.assertIn("Capture: trace-run-001", prompt)
            self.assertIn("process=runares", prompt)
            self.assertIn("Run provenance: target/debug/runares", prompt)
            self.assertIn("Artifact identity: ares_plan", prompt)
            self.assertIn("Artifact identity check: <all>", prompt)
            self.assertIn("Capture capability: token_quality present=True", prompt)
            self.assertIn("Trace event artifact: trace-events.jsonl", prompt)
            self.assertIn("Timeline query summary: join-key-coverage", prompt)
            self.assertIn("capture provenance", prompt)
            self.assertIn("artifact hashes, and capability booleans", prompt)
            self.assertIn("baseline/candidate delta", prompt)
            self.assertIn("comparison-grade evidence", prompt)
            self.assertIn("A/B provenance: comparison", prompt)
            self.assertIn("A/B comparability: comparison-grade", prompt)
            self.assertIn("A/B coverage: partial_overlap", prompt)
            self.assertIn("A/B repeatability: insufficient_for_proof", prompt)
            self.assertIn("Introspection raw artifact: token_quality", prompt)
            self.assertIn("path=introspection_token_quality.jsonl", prompt)
            self.assertIn(f"sha256={SHA_A}", prompt)
            self.assertIn("features=trace-introspection,deep-trace", prompt)
            self.assertIn("Provider payload boundary: fpga/kv_payload_digests", prompt)
            self.assertIn(
                "Recorded provider payload boundary: fpga/kv_payload_digests",
                prompt,
            )
            self.assertIn(
                "Route-only provider payload boundary: generic/device_result_digests",
                prompt,
            )
            self.assertIn("recorded provider-callback rows", prompt)
            self.assertIn("missing=device_result_digests", prompt)
            self.assertIn("provider_artifacts=2", prompt)
            self.assertIn("provider_artifacts=0", prompt)
            self.assertIn("same_kind_artifacts=2", prompt)
            self.assertIn("same_kind_artifacts=1", prompt)
            self.assertIn("same_kind_backends=fpga", prompt)
            self.assertIn("producer=provider_callback_present", prompt)
            self.assertIn(
                "producer=runtime_route_only_no_provider_producer",
                prompt,
            )
            self.assertIn("Backend event artifacts", prompt)
            self.assertIn("Backend event artifact: backend-events.jsonl", prompt)
            self.assertIn("Backend event row: fpga", prompt)
            self.assertIn(
                "raw backend JSONL artifacts",
                prompt,
            )
            self.assertIn("Backend provider boundaries", prompt)
            self.assertIn("Backend provider boundary: fpga", prompt)
            self.assertIn("event=forward_failed", prompt)
            self.assertIn("status=fail_closed", prompt)
            self.assertIn("root_stage=targetplan_validation", prompt)
            self.assertIn("Backend fail-closed root cause: fpga", prompt)
            self.assertIn("policy=sha256_digest_plus_bounded_f32_sample", prompt)
            self.assertIn("sensitivity=scheduler_kv_save_values", prompt)
            self.assertIn("Debug payload artifact: attention_page_trace", prompt)
            self.assertIn("features=trace-introspection", prompt)
            self.assertIn("Planning decision sidecar: row=lean_planning_phase", prompt)
            self.assertIn("phase=lean.target_plan_lower", prompt)
            self.assertIn("frontend and", prompt)
            self.assertIn("Token quality sidecar: request=7001", prompt)
            self.assertIn("finish=stop", prompt)
            self.assertIn("Top-K token sidecar: request=7001", prompt)
            self.assertIn("candidate_status=selected_token", prompt)
            self.assertIn("Tensor payload sidecar: request=7005", prompt)
            self.assertIn("kind=tensor_payload", prompt)
            self.assertIn("stmt_name=provider_payload", prompt)
            self.assertIn("digest_sha256=", prompt)
            self.assertIn("K/V payload digest sidecar: request=7006", prompt)
            self.assertIn("stmt_index=21", prompt)
            self.assertIn("stmt_kind=kv_cache", prompt)
            self.assertIn("stmt_name=kv_save_layer_0_key", prompt)
            self.assertIn("scheduler K/V digest rows", prompt)
            self.assertIn("Logit slice sidecar: request=7005", prompt)
            self.assertIn("action=final_logits", prompt)
            self.assertIn("stmt_index=99", prompt)
            self.assertIn("stmt_name=ares_logits", prompt)
            self.assertIn("Activation digest sidecar: request=7007", prompt)
            self.assertIn("intrinsic=rmsnorm", prompt)
            self.assertIn("stmt_index=42", prompt)
            self.assertIn("stmt_kind=rmsnorm", prompt)
            self.assertIn("stmt_name=layer_0_activation", prompt)
            self.assertIn("Device result digest sidecars", prompt)
            self.assertIn("Device result digest roles", prompt)
            self.assertIn("Device result digest actions", prompt)
            self.assertIn("Device result digest intrinsics", prompt)
            self.assertIn("Device result digest sidecar: request=7009", prompt)
            self.assertIn("action=matmul", prompt)
            self.assertIn("stmt_index=4", prompt)
            self.assertIn("stmt_kind=matmul", prompt)
            self.assertIn("stmt_name=wcls", prompt)
            self.assertIn("sample_finite=2", prompt)
            self.assertIn("intrinsic=fpga.matmul", prompt)
            self.assertIn("tensor=fpga_scheduler_forward_batch_result", prompt)
            self.assertIn("sample_min=1.25", prompt)
            self.assertIn("Scheduler packet lineage: request=7002", prompt)
            self.assertIn("scheduler packet shape", prompt)
            self.assertIn("Scheduler K/V lifecycle: request=7002", prompt)
            self.assertIn("Scheduler sparse listener: request=7002", prompt)
            self.assertIn("sparse-listener", prompt)
            self.assertIn("hardware-counter evidence", prompt)
            self.assertIn("Device DMA lifecycle: request=7002", prompt)
            self.assertIn("stage=dma_completion", prompt)
            self.assertIn("counter_delta=5", prompt)
            self.assertIn("Attention page trace: request=7004", prompt)
            self.assertIn("visible_tokens=65", prompt)
            self.assertIn("debug payload rows as performance proof", prompt)
            self.assertIn("without treating", prompt)
            self.assertIn("oracle evidence", prompt)
            self.assertIn("Token quality summary: request=7001", prompt)
            self.assertIn("token_index=0", prompt)
            self.assertIn("top1_margin=1.4", prompt)
            self.assertIn("oracle_reference=external_hf_cpu_reference", prompt)
            self.assertIn("Oracle reference summary: request=7001", prompt)
            self.assertIn("role=external_hf_cpu_reference", prompt)
            self.assertIn("correctness=not_oracle_evidence", prompt)
            self.assertIn("system-under-test rows as oracle evidence", prompt)
            self.assertIn("inspect_matching_introspection_report_sections", prompt)
            self.assertIn("sections.introspection_capability_rows", prompt)
            self.assertIn("sections.introspection_artifacts", prompt)
            self.assertIn("sections.introspection_artifact_summary_rows", prompt)
            self.assertIn(
                "Introspection section: token_quality_summary_rows",
                prompt,
            )
            self.assertIn("capability=token_quality", prompt)
            self.assertIn("Introspection artifact: token_quality", prompt)
            self.assertIn("set ARES_BACKEND_EVENT_ARTIFACT_DIR", prompt)

    def test_real_trace_report_renders_raw_introspection_section(self) -> None:
        path = FIXTURE_DIR / "ares_trace_report_introspection_real.json"
        gate = trace_report_gate(path)
        summary = trace_report_summary_from_spec(
            {"validated_gates": {"trace_report": gate}}
        )

        self.assertIsNotNone(summary)
        self.assertEqual((summary or {}).get("section_count"), 53)
        lines = "\n".join(render_trace_report_lines(summary or {}))

        expected_basis = (
            "Trace grade basis: preflight passed; no complete "
            "baseline/candidate comparison inputs"
        )
        self.assertIn(expected_basis, lines)
        self.assertIn(
            "Trace grade promotion gate: Capture comparable baseline/candidate artifacts",
            lines,
        )
        self.assertIn("Report sections available: count=53", lines)
        self.assertIn("Report triage detail: status=needs_measurement", lines)
        self.assertIn("next_priority=timeline_capture", lines)
        self.assertIn("analysis_commands", lines)
        self.assertIn("Answerability detail: metadata artifact identity", lines)
        self.assertIn("status=not_present", lines)
        self.assertIn("basis=metadata.artifacts: 0 row(s)", lines)
        self.assertIn("Preflight finding: warn: metadata.device_counters", lines)
        self.assertIn(
            "Evidence classification: diagnostic: preflight passed",
            lines,
        )
        self.assertIn("Report section inventory: Run Summary", lines)
        self.assertIn("query=run-summary", lines)
        self.assertIn("Report JSON section: sections.preflight", lines)
        self.assertIn("Report JSON section: sections.analysis_commands", lines)
        self.assertIn("Report JSON section: sections.report_grade", lines)
        self.assertIn("Report JSON section: sections.report_triage", lines)
        self.assertIn(
            "Report JSON section: sections.report_json_section_inventory",
            lines,
        )
        self.assertIn("Report JSON section: sections.preflight_findings", lines)
        self.assertIn("Report JSON section: sections.evidence_classification", lines)
        self.assertIn("Report JSON section: sections.report_section_inventory", lines)
        self.assertIn("Report JSON section: sections.introspection_artifacts", lines)
        self.assertIn("Introspection raw artifact: token_quality", lines)
        self.assertIn("path=introspection_token_quality_sidecar.jsonl", lines)
        self.assertIn(
            "sha256=4255461c82b837e9c6196aba804042ac967ee637602826962b70fe4a9576534e",
            lines,
        )

    def test_no_refiner_blocks_below_target(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"

            rc = main(
                [
                    "--ares-repo",
                    str(root),
                    "--run-dir",
                    str(run_dir),
                    "--no-refiner",
                    "Provider/Model",
                ]
            )

            self.assertEqual(rc, 3)
            state = json.loads((run_dir / "state.json").read_text())
            self.assertEqual(state["status"], "blocked_no_refiner")
            self.assertEqual(state["refinement_loop"], "one_failing_gate")
            self.assertEqual(state["history"][0]["first_failed_gate"], "hf_cpu_oracle")

    def test_evidence_gate_preserves_gate_specific_skill_context(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            cfg = config_from_args(
                build_parser().parse_args(
                    [
                        "--ares-repo",
                        str(root),
                        "--run-dir",
                        str(run_dir),
                        "Provider/Model",
                    ]
                )
            )
            reward = {
                "score": 0.48,
                "alpha_execution": 0.8,
                "tau_tokens": 0.0,
                "delta_inference": 0.0,
                "stage_cap": 0.5,
                "first_failed_gate": "artifact_consistency",
                "gates": {},
            }

            skills = selected_workflow_skills(
                cfg,
                first_failed_gate="artifact_consistency",
            )
            evidence_skills = [
                skill for skill in skills if skill["name"] == "ares-evidence"
            ]

            self.assertEqual(len(evidence_skills), 2)
            self.assertEqual(evidence_skills[1]["gate"], "artifact_consistency")
            self.assertIn("same model row", evidence_skills[1]["why"])
            write_handoff(cfg, reward, state={"workflow_skills": skills, "history": []})
            handoff = (run_dir / "handoff.md").read_text()
            self.assertIn("`ares-evidence` for `artifact_consistency`", handoff)
            prompt = write_refinement_prompt(
                cfg,
                iteration=1,
                spec={},
                reward=reward,
            ).read_text()
            self.assertIn("`ares-evidence` for `artifact_consistency`", prompt)
            self.assertIn("same model row", prompt)

    def test_model_spec_gate_selects_model_port_skill_context(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            cfg = config_from_args(
                build_parser().parse_args(
                    [
                        "--ares-repo",
                        str(root),
                        "--run-dir",
                        str(run_dir),
                        "Provider/Model",
                    ]
                )
            )
            reward = {
                "score": 0.0,
                "alpha_execution": 0.0,
                "tau_tokens": 0.0,
                "delta_inference": 0.0,
                "stage_cap": 0.03,
                "first_failed_gate": "model_spec",
                "gates": {},
            }

            skills = selected_workflow_skills(cfg, first_failed_gate="model_spec")
            model_skill = next(
                skill for skill in skills if skill.get("gate") == "model_spec"
            )

            self.assertEqual(model_skill["name"], "ares-model-port")
            self.assertIn("model row inventory", model_skill["why"])
            self.assertIn("HuggingFace Transformers", model_skill["why"])
            self.assertIn("vLLM", model_skill["why"])
            self.assertIn("llama.cpp", model_skill["why"])
            self.assertIn("MLX", model_skill["why"])
            self.assertIn("${ARES_PRIOR_ART_ROOT:-$HOME/db}", model_skill["why"])
            self.assertIn(
                "clone any missing official upstream repos", model_skill["why"]
            )
            self.assertIn(
                "https://github.com/vllm-project/vllm.git", model_skill["why"]
            )
            self.assertIn(
                "https://github.com/ggml-org/llama.cpp.git",
                model_skill["why"],
            )
            self.assertIn("https://github.com/ml-explore/mlx.git", model_skill["why"])
            self.assertEqual(
                model_skill["allowed_scope"],
                [
                    str(cfg.run_dir),
                    str(cfg.model_spec_path),
                    str(cfg.run_dir / "handoff.md"),
                    "prior-art checkout paths recorded in model_spec.json or handoff.md",
                    "${ARES_PRIOR_ART_ROOT:-$HOME/db}/vllm",
                    "${ARES_PRIOR_ART_ROOT:-$HOME/db}/llama.cpp",
                    "${ARES_PRIOR_ART_ROOT:-$HOME/db}/mlx",
                ],
            )
            run_dir.mkdir()
            write_handoff(cfg, reward, state={"workflow_skills": skills, "history": []})
            handoff = (run_dir / "handoff.md").read_text()
            self.assertIn("`ares-model-port` for `model_spec`", handoff)
            self.assertIn("vLLM, llama.cpp, and MLX", handoff)
            self.assertIn("${ARES_PRIOR_ART_ROOT:-$HOME/db}/vllm", handoff)
            self.assertIn("https://github.com/vllm-project/vllm.git", handoff)
            prompt = write_refinement_prompt(
                cfg,
                iteration=1,
                spec={},
                reward=reward,
            ).read_text()
            self.assertIn("vLLM, llama.cpp, and MLX", prompt)
            self.assertIn("${ARES_PRIOR_ART_ROOT:-$HOME/db}/llama.cpp", prompt)
            self.assertIn("https://github.com/ggml-org/llama.cpp.git", prompt)
            self.assertIn(
                "read or clone official vLLM, llama.cpp, and MLX checkouts",
                prompt,
            )
            self.assertNotIn(
                "- Work inside the Ares repo and this run directory only.",
                prompt,
            )

    def test_operator_steering_is_written_to_prompt_and_handoff(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            cfg = config_from_args(
                build_parser().parse_args(
                    [
                        "--ares-repo",
                        str(root),
                        "--run-dir",
                        str(run_dir),
                        "Provider/Model",
                    ]
                )
            )
            append_steering_note(
                cfg,
                iteration=2,
                text="Prefer HF-export evidence from the local snapshot.",
            )
            append_steering_resource(
                cfg,
                iteration=2,
                value="/tmp/local-hf-export",
                note="candidate bundle",
            )
            reward = {
                "score": 0.1,
                "alpha_execution": 0.2,
                "tau_tokens": 0.0,
                "delta_inference": 0.0,
                "stage_cap": 0.18,
                "first_failed_gate": "frontend_export",
                "gates": {},
            }

            write_handoff(cfg, reward, state={"history": []})
            handoff = (run_dir / "handoff.md").read_text()
            self.assertIn("## Operator Steering", handoff)
            self.assertIn("Prefer HF-export evidence", handoff)
            self.assertIn("/tmp/local-hf-export", handoff)

            prompt = write_refinement_prompt(
                cfg,
                iteration=2,
                spec={},
                reward=reward,
            ).read_text()
            self.assertIn("## Operator Steering", prompt)
            self.assertIn("Prefer HF-export evidence", prompt)
            self.assertIn("candidate bundle", prompt)

    def test_evaluate_run_validates_introspection_ladder_attachment(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            cfg = config_from_args(
                build_parser().parse_args(
                    [
                        "--ares-repo",
                        str(root),
                        "--run-dir",
                        str(run_dir),
                        "Provider/Model",
                    ]
                )
            )
            write_introspection_ladder(run_dir)
            cfg.model_spec_path.write_text(
                json.dumps(
                    {
                        "model": "Provider/Model",
                        "required_gates": ["model_spec", "hf_cpu_oracle"],
                        "explicit_gates": {
                            "model_spec": {"passed": True, "score": 1.0}
                        },
                        "introspection_ladder": "ladder.json",
                    }
                )
            )

            spec, reward = evaluate_run(cfg)

            gate = spec["validated_gates"]["introspection_ladder"]
            self.assertTrue(gate["passed"], gate.get("errors"))
            self.assertEqual(gate["artifact_validator"], "introspection_ladder")
            self.assertEqual(
                gate["detail"]["first_failed_comparison"]["from_stage"],
                "hf_cpu_oracle",
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
            self.assertEqual(reward["first_failed_gate"], "hf_cpu_oracle")
            self.assertTrue(reward["gates"]["introspection_ladder"]["passed"])
            prompt = write_refinement_prompt(
                cfg,
                iteration=1,
                spec=spec,
                reward=reward,
            ).read_text()
            self.assertIn("Validated introspection ladder", prompt)
            self.assertIn("first failing introspection stage", prompt)
            self.assertIn("source_hf_hypergraph", prompt)
            self.assertIn("first failed comparison", prompt)
            self.assertIn("`hf_cpu_oracle` -> `source_hf_hypergraph`", prompt)
            self.assertIn("first mismatch", prompt)
            self.assertIn("producer=`linear_0`", prompt)
            self.assertIn("max_abs_error=`0.25`", prompt)
            self.assertIn(gate["sha256"], prompt)
            self.assertIn("targetplan.stmt.00001.matmul", prompt)
            self.assertIn("backend-events.jsonl", prompt)
            self.assertIn("run.trace-meta.json", prompt)

    def test_introspection_ladder_is_recorded_in_state_handoff_and_history(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            cfg = config_from_args(
                build_parser().parse_args(
                    [
                        "--ares-repo",
                        str(root),
                        "--run-dir",
                        str(run_dir),
                        "--setup-only",
                        "Provider/Model",
                    ]
                )
            )
            write_introspection_ladder(run_dir)
            cfg.model_spec_path.write_text(
                json.dumps(
                    {
                        "model": "Provider/Model",
                        "required_gates": ["model_spec", "hf_cpu_oracle"],
                        "explicit_gates": {
                            "model_spec": {"passed": True, "score": 1.0}
                        },
                        "introspection_ladder": "ladder.json",
                    }
                )
            )

            state = initialize_run(cfg)

            ladder = state["introspection_ladder"]
            self.assertEqual(ladder["path"], str((run_dir / "ladder.json").resolve()))
            self.assertTrue(ladder["passed"])
            self.assertEqual(ladder["artifact_validator"], "introspection_ladder")
            self.assertEqual(ladder["evidence_class"], "semantic_localization")
            self.assertEqual(ladder["first_failing_stage"], "source_hf_hypergraph")
            self.assertEqual(ladder["next_owner"], "ares-python")
            self.assertEqual(
                ladder["first_failed_comparison"]["from_stage"],
                "hf_cpu_oracle",
            )
            self.assertEqual(
                ladder["first_failed_comparison"]["to_stage"],
                "source_hf_hypergraph",
            )
            self.assertEqual(
                ladder["first_failed_comparison"]["path"],
                "source.compare.json",
            )
            self.assertEqual(ladder["first_mismatch"]["id"], "logits")
            self.assertEqual(
                ladder["first_mismatch"]["producer_generator"],
                "linear_0",
            )
            self.assertEqual(
                ladder["trace_context"]["trace_labels"],
                ["targetplan.stmt.00001.matmul"],
            )
            handoff = (run_dir / "handoff.md").read_text()
            self.assertIn("## Introspection Ladder", handoff)
            self.assertIn("semantic_localization", handoff)
            self.assertIn("source.compare.json", handoff)
            self.assertIn("first failed comparison", handoff)
            self.assertIn("`hf_cpu_oracle` -> `source_hf_hypergraph`", handoff)
            self.assertIn("producer=`linear_0`", handoff)
            self.assertIn("max_abs_error=`0.25`", handoff)

            append_history(
                state,
                cfg=cfg,
                iteration=1,
                reward=json.loads((run_dir / "reward.json").read_text()),
                status="refiner_ran",
            )

            saved = json.loads(cfg.state_path.read_text())
            self.assertEqual(
                saved["history"][0]["introspection_ladder"]["first_mismatch"][
                    "max_abs_error"
                ],
                0.25,
            )

    def test_failure_state_preserves_validated_introspection_ladder(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            cfg = config_from_args(
                build_parser().parse_args(
                    [
                        "--ares-repo",
                        str(root),
                        "--run-dir",
                        str(run_dir),
                        "Provider/Model",
                    ]
                )
            )
            write_introspection_ladder(run_dir)
            cfg.model_spec_path.write_text(
                json.dumps(
                    {
                        "model": "Provider/Model",
                        "required_gates": ["model_spec", "hf_cpu_oracle"],
                        "explicit_gates": {
                            "model_spec": {"passed": True, "score": 1.0}
                        },
                        "introspection_ladder": "ladder.json",
                    }
                )
            )
            evaluate_run(cfg)

            write_failure_state(cfg, AresIngestError("forced failure"))

            saved = json.loads(cfg.state_path.read_text())
            self.assertEqual(saved["status"], "failed")
            self.assertEqual(
                saved["introspection_ladder"]["first_failed_comparison"]["to_stage"],
                "source_hf_hypergraph",
            )
            self.assertEqual(
                saved["introspection_ladder"]["first_mismatch"]["producer_generator"],
                "linear_0",
            )
            handoff = (run_dir / "handoff.md").read_text()
            self.assertIn("## Introspection Ladder", handoff)
            self.assertIn("first mismatch", handoff)

    def test_logit_drift_selects_introspection_skill_context(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = config_from_args(
                build_parser().parse_args(
                    [
                        "--ares-repo",
                        str(root),
                        "--run-dir",
                        str(root / "run"),
                        "Provider/Model",
                    ]
                )
            )
            reward = {
                "score": 0.58,
                "alpha_execution": 0.7,
                "tau_tokens": 0.0,
                "delta_inference": 0.0,
                "stage_cap": 0.70,
                "first_failed_gate": "one_token_logits",
                "gates": {},
            }

            skills = selected_workflow_skills(
                cfg,
                first_failed_gate="one_token_logits",
            )
            drift_skill = next(
                skill for skill in skills if skill.get("gate") == "one_token_logits"
            )

            self.assertEqual(drift_skill["name"], "ares-introspection")
            self.assertIn("semantic-localization ladder", drift_skill["why"])
            self.assertIn("bin/ares-introspect", drift_skill["allowed_scope"])
            cfg.run_dir.mkdir()
            write_handoff(cfg, reward, state={"workflow_skills": skills, "history": []})
            handoff = (cfg.run_dir / "handoff.md").read_text()
            self.assertIn("`ares-introspection` for `one_token_logits`", handoff)
            prompt = write_refinement_prompt(
                cfg,
                iteration=1,
                spec={"introspection_ladder": "ladder.json"},
                reward=reward,
            ).read_text()
            self.assertIn("`ares-introspection` for `one_token_logits`", prompt)
            self.assertIn("bin/ares-introspect ladder --graph", prompt)
            self.assertIn("introspection_ladder", prompt)

    def test_cpp_tvd_selects_comparison_evidence_context(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = config_from_args(
                build_parser().parse_args(
                    [
                        "--ares-repo",
                        str(root),
                        "--run-dir",
                        str(root / "run"),
                        "Provider/Model",
                    ]
                )
            )

            skills = selected_workflow_skills(cfg, first_failed_gate="cpp_tvd")
            cpp_skill = next(
                skill for skill in skills if skill.get("gate") == "cpp_tvd"
            )

            self.assertEqual(cpp_skill["name"], "ares-evidence")
            self.assertIn("comparison and rollback evidence", cpp_skill["why"])

    def test_mmlu_pro_gate_selects_mmlu_skill_context(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = config_from_args(
                build_parser().parse_args(
                    [
                        "--ares-repo",
                        str(root),
                        "--run-dir",
                        str(root / "run"),
                        "Provider/Model",
                    ]
                )
            )

            skills = selected_workflow_skills(cfg, first_failed_gate="mmlu_pro")
            mmlu_skill = next(
                skill for skill in skills if skill.get("gate") == "mmlu_pro"
            )

            self.assertEqual(mmlu_skill["name"], "ares-mmlu-pro")
            self.assertIn("MMLU Pro benchmark evidence", mmlu_skill["why"])
            self.assertIn("third_party/systems_test/", mmlu_skill["allowed_scope"])

    def test_targetplan_gate_selects_targetplan_skill_context(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = config_from_args(
                build_parser().parse_args(
                    [
                        "--ares-repo",
                        str(root),
                        "--run-dir",
                        str(root / "run"),
                        "Provider/Model",
                    ]
                )
            )

            skills = selected_workflow_skills(
                cfg,
                first_failed_gate="targetplan_valid",
            )
            targetplan_skill = next(
                skill for skill in skills if skill.get("gate") == "targetplan_valid"
            )

            self.assertEqual(targetplan_skill["name"], "ares-targetplan")
            self.assertIn("Rust validation", targetplan_skill["why"])
            self.assertEqual(
                targetplan_skill["allowed_scope"],
                [
                    "ingest/lean/",
                    "backend/",
                    "runtime/",
                    str(cfg.run_dir / "artifacts"),
                    str(cfg.model_spec_path),
                ],
            )
            reward = {
                "score": 0.38,
                "alpha_execution": 0.7,
                "tau_tokens": 0.0,
                "delta_inference": 0.0,
                "stage_cap": 0.48,
                "first_failed_gate": "targetplan_valid",
                "gates": {},
            }
            (cfg.run_dir).mkdir()
            write_handoff(cfg, reward, state={"workflow_skills": skills, "history": []})
            handoff = (cfg.run_dir / "handoff.md").read_text()
            self.assertIn("`ares-targetplan` for `targetplan_valid`", handoff)
            self.assertIn("runtime/", handoff)
            prompt = write_refinement_prompt(
                cfg,
                iteration=1,
                spec={},
                reward=reward,
            ).read_text()
            self.assertIn("`ares-targetplan` for `targetplan_valid`", prompt)
            self.assertIn("runtime handoff", prompt)
            self.assertIn(str(cfg.model_spec_path), prompt)

    def test_target_score_completion_does_not_invoke_refiner(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"

            rc = main(
                [
                    "--ares-repo",
                    str(root),
                    "--run-dir",
                    str(run_dir),
                    "--target-score",
                    "0.05",
                    "--refinement-command",
                    "exit 99",
                    "Provider/Model",
                ]
            )

            self.assertEqual(rc, 0)
            state = json.loads((run_dir / "state.json").read_text())
            self.assertEqual(state["status"], "complete")
            self.assertFalse((run_dir / "logs/01-refiner.log").exists())

    def test_refiner_loop_writes_prompt_log_and_history(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"

            rc = main(
                [
                    "--ares-repo",
                    str(root),
                    "--run-dir",
                    str(run_dir),
                    "--max-iterations",
                    "2",
                    "--stall-patience",
                    "5",
                    "--refinement-command",
                    'printf \'%s\\n\' "$FIRST_FAILED_GATE" > "$RUN_DIR/refiner-ran.txt"',
                    "Provider/Model",
                ]
            )

            self.assertEqual(rc, 2)
            prompt = run_dir / "prompts/refinement-01.md"
            self.assertTrue(prompt.exists())
            prompt_text = prompt.read_text()
            self.assertIn(
                "Work only the first failing gate: `hf_cpu_oracle`", prompt_text
            )
            self.assertIn("## Expected Workflow Skills", prompt_text)
            self.assertIn("`command-wiggum`", prompt_text)
            self.assertIn("`ares-python`", prompt_text)
            self.assertIn("Allowed scope:", prompt_text)
            self.assertIn("## Allowed Write Scope", prompt_text)
            self.assertIn("HF Transformers on PyTorch CPU", prompt_text)
            self.assertIn(
                f"ares-ingest-agent Provider/Model --ares-repo {root.resolve()} "
                f"--run-dir {run_dir.resolve()} --no-refiner",
                prompt_text,
            )
            self.assertEqual(
                (run_dir / "refiner-ran.txt").read_text().strip(), "hf_cpu_oracle"
            )
            self.assertTrue((run_dir / "logs/01-refiner.log").exists())
            state = json.loads((run_dir / "state.json").read_text())
            self.assertEqual(state["status"], "max_iterations")
            self.assertEqual(
                [skill["name"] for skill in state["workflow_skills"]],
                ["command-wiggum", "ares-evidence", "ares-python", "command-fess"],
            )
            self.assertEqual(
                [item["status"] for item in state["history"]],
                ["refiner_ran", "max_iterations"],
            )
            self.assertEqual(Path(state["latest_refinement_prompt"]), prompt.resolve())

    def test_cockpit_loop_records_events_and_streaming_state(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"

            rc = main(
                [
                    "--ares-repo",
                    str(root),
                    "--run-dir",
                    str(run_dir),
                    "--cockpit",
                    "--max-iterations",
                    "2",
                    "--stall-patience",
                    "5",
                    "--refinement-command",
                    "printf 'cockpit %s\\n' \"$FIRST_FAILED_GATE\"",
                    "Provider/Model",
                ]
            )

            self.assertEqual(rc, 2)
            state = json.loads((run_dir / "state.json").read_text())
            self.assertEqual(state["cockpit"]["enabled"], True)
            self.assertEqual(state["cockpit"]["stream_refiner_output"], True)
            self.assertEqual(state["driver"], "codex")
            self.assertTrue((run_dir / "cockpit.jsonl").exists())
            events = [
                json.loads(line)
                for line in (run_dir / "cockpit.jsonl").read_text().splitlines()
            ]
            self.assertTrue(any(event["status"] == "evaluated" for event in events))
            self.assertTrue(
                any(event["status"] == "continue_noninteractive" for event in events)
            )
            self.assertIn(
                "cockpit hf_cpu_oracle", (run_dir / "logs/01-refiner.log").read_text()
            )

    def test_cockpit_interactive_commands_record_steering_and_driver(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            cfg = config_from_args(
                build_parser().parse_args(
                    [
                        "--ares-repo",
                        str(root),
                        "--run-dir",
                        str(run_dir),
                        "--cockpit",
                        "--refinement-command",
                        "echo old-driver",
                        "Provider/Model",
                    ]
                )
            )
            reward = {
                "score": 0.075,
                "alpha_execution": 0.125,
                "tau_tokens": 0.0,
                "delta_inference": 0.0,
                "stage_cap": 0.1,
                "first_failed_gate": "hf_cpu_oracle",
                "gates": {},
            }
            commands = iter(
                [
                    "note use the local HF snapshot",
                    "resource /tmp/hf-export -- candidate bundle",
                    "driver echo new-driver",
                    "continue",
                ]
            )

            with (
                patch("sys.stdin.isatty", return_value=True),
                patch("builtins.input", side_effect=lambda _prompt: next(commands)),
            ):
                should_continue = cockpit_checkpoint(
                    cfg,
                    iteration=3,
                    reward=reward,
                    state={"history": []},
                    best_score=0.075,
                    stall_count=0,
                )

            self.assertTrue(should_continue)
            self.assertEqual(cfg.refinement_command, "echo new-driver")
            steering = json.loads((run_dir / "steering.json").read_text())
            self.assertEqual(steering["notes"][0]["text"], "use the local HF snapshot")
            self.assertEqual(steering["resources"][0]["value"], "/tmp/hf-export")
            self.assertEqual(steering["resources"][0]["note"], "candidate bundle")
            self.assertEqual(
                steering["driver_commands"][0]["command"],
                "echo new-driver",
            )
            events = [
                json.loads(line)
                for line in (run_dir / "cockpit.jsonl").read_text().splitlines()
            ]
            self.assertEqual(events[-1]["decision"], "continue")

    def test_missing_oracle_records_writes_failure_state_over_corrupt_state(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "state.json").write_text("{")
            (run_dir / "model_spec.json").write_text(
                json.dumps(
                    {
                        "model": "Provider/Model",
                        "required_gates": ["model_spec", "hf_cpu_oracle"],
                        "explicit_gates": {
                            "model_spec": {
                                "passed": True,
                                "score": 1.0,
                            },
                        },
                        "oracle_records": "missing-oracle.jsonl",
                    }
                )
            )

            with self.assertRaises(SystemExit) as raised:
                main(
                    [
                        "--ares-repo",
                        str(root),
                        "--run-dir",
                        str(run_dir),
                        "--no-refiner",
                        "Provider/Model",
                    ]
                )

            self.assertEqual(raised.exception.code, 1)
            state = json.loads((run_dir / "state.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertIn("missing JSON file", state["error"])
            self.assertIn("previous_state_error", state)

    def test_refiner_failure_writes_state_and_handoff(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"

            with self.assertRaises(SystemExit) as raised:
                main(
                    [
                        "--ares-repo",
                        str(root),
                        "--run-dir",
                        str(run_dir),
                        "--refinement-command",
                        "exit 7",
                        "Provider/Model",
                    ]
                )

            self.assertEqual(raised.exception.code, 1)
            state = json.loads((run_dir / "state.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertIn("refiner failed with exit 7", state["error"])
            self.assertIn(
                "ares-python",
                [skill["name"] for skill in state["workflow_skills"]],
            )
            self.assertNotIn(
                "failed",
                [skill.get("gate") for skill in state["workflow_skills"]],
            )
            handoff = (run_dir / "handoff.md").read_text()
            self.assertIn("Status: `failed`", handoff)
            self.assertIn("`ares-python` for `hf_cpu_oracle`", handoff)

    def test_main_setup_only_writes_state(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            rc = main(
                [
                    "--ares-repo",
                    str(root),
                    "--run-dir",
                    str(run_dir),
                    "--setup-only",
                    "Provider/Model",
                ]
            )

            self.assertEqual(rc, 0)
            state = json.loads((run_dir / "state.json").read_text())
            self.assertEqual(state["refinement_loop"], "setup_only")

    def test_backend_profile_writes_command_wrapper_plan(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"

            rc = main(
                [
                    "--ares-repo",
                    str(root),
                    "--run-dir",
                    str(run_dir),
                    "--gate-profile",
                    "backend",
                    "--setup-only",
                    "Provider/Model",
                ]
            )

            self.assertEqual(rc, 0)
            spec = json.loads((run_dir / "model_spec.json").read_text())
            self.assertEqual(spec["command_wrapper_plan"], "command_wrappers.json")
            plan = json.loads((run_dir / "command_wrappers.json").read_text())
            self.assertEqual(plan["schema"], "ares.autoagent.command_wrappers.v1")
            self.assertFalse(plan["execute_command_wrappers"])
            self.assertEqual(plan["command_gates"], [])
            self.assertEqual(
                {wrapper["name"] for wrapper in plan["wrappers"]},
                {"rinzler_chat_one_token", "rinzler_full_inference_smoke"},
            )
            self.assertTrue(
                all(wrapper["missing_inputs"] for wrapper in plan["wrappers"])
            )


if __name__ == "__main__":
    unittest.main()

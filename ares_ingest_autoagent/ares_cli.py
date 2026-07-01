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
        "frontend": "hf-export",
        "backend": "fpga",
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
            "hf_cpu_goldens": "capture_once_per_stable_oracle_tuple",
            "ordinary_backend_loop_reference": "cached_hf_cpu_token_logit_goldens",
            "debug_loop_priority": [
                "cached_hf_cpu_logit_comparison",
                "focused_backend_or_module_slice",
                "short_depth_backend_generation",
                "longer_depth_backend_generation",
                "explicit_late_cpp_comparison_milestone",
            ],
            "fastest_verifier_first": True,
            "avoid_recapturing_unchanged_hf_logits": True,
            "defer_cpp_comparison_until_milestone_candidate": True,
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


def introspection_ladder_summary_from_spec(
    spec: Mapping[str, Any],
) -> dict[str, Any] | None:
    gates = spec.get("validated_gates")
    if not isinstance(gates, Mapping):
        gates = spec.get("explicit_gates")
    if not isinstance(gates, Mapping):
        return None
    gate = gates.get("introspection_ladder")
    if not isinstance(gate, Mapping):
        return None
    detail = gate.get("detail")
    if not isinstance(detail, Mapping):
        detail = {}
    return {
        "path": gate.get("path"),
        "sha256": gate.get("sha256") or detail.get("sha256"),
        "passed": gate.get("passed"),
        "artifact_validator": gate.get("artifact_validator"),
        "schema": detail.get("schema"),
        "evidence_class": detail.get("evidence_class"),
        "status": detail.get("status"),
        "first_failing_stage": detail.get("first_failing_stage"),
        "next_owner": detail.get("next_owner"),
        "stage_order": detail.get("stage_order"),
        "comparison_count": detail.get("comparison_count"),
        "run_count": detail.get("run_count"),
        "graph_count": detail.get("graph_count"),
        "comparisons": detail.get("comparisons"),
        "first_failed_comparison": detail.get("first_failed_comparison"),
        "first_mismatch": detail.get("first_mismatch"),
        "trace_context": detail.get("trace_context"),
        "errors": gate.get("errors"),
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
        "section_count": detail.get("section_count"),
        "metadata": detail.get("metadata"),
        "trace": detail.get("trace"),
        "preflight_status": detail.get("preflight_status"),
        "preflight_ok_count": detail.get("preflight_ok_count"),
        "preflight_warn_count": detail.get("preflight_warn_count"),
        "preflight_fail_count": detail.get("preflight_fail_count"),
        "report_grade": detail.get("report_grade"),
        "proof_grade_status": detail.get("proof_grade_status"),
        "report_grade_basis": detail.get("report_grade_basis"),
        "report_grade_promotion_gate": detail.get("report_grade_promotion_gate"),
        "report_triage_count": detail.get("report_triage_count"),
        "report_triage_status_counts": detail.get("report_triage_status_counts"),
        "report_triage_samples": detail.get("report_triage_samples"),
        "answerability_count": detail.get("answerability_count"),
        "answerability_status_counts": detail.get("answerability_status_counts"),
        "supported_claim_count": detail.get("supported_claim_count"),
        "unsupported_claim_count": detail.get("unsupported_claim_count"),
        "next_measurement_count": detail.get("next_measurement_count"),
        "correctness_evidence_count": detail.get("correctness_evidence_count"),
        "correctness_evidence_status_counts": detail.get(
            "correctness_evidence_status_counts"
        ),
        "correctness_evidence_proof_grade_status_counts": detail.get(
            "correctness_evidence_proof_grade_status_counts"
        ),
        "evidence_artifact_check_count": detail.get("evidence_artifact_check_count"),
        "evidence_artifact_check_status_counts": detail.get(
            "evidence_artifact_check_status_counts"
        ),
        "promotion_gate_summary_count": detail.get("promotion_gate_summary_count"),
        "promotion_gate_summary_status_counts": detail.get(
            "promotion_gate_summary_status_counts"
        ),
        "promotion_gate_summary_proof_grade_status_counts": detail.get(
            "promotion_gate_summary_proof_grade_status_counts"
        ),
        "trace_mode_guardrail_count": detail.get("trace_mode_guardrail_count"),
        "trace_mode_guardrail_mode_counts": detail.get(
            "trace_mode_guardrail_mode_counts"
        ),
        "trace_mode_guardrail_overhead_counts": detail.get(
            "trace_mode_guardrail_overhead_counts"
        ),
        "ab_provenance_count": detail.get("ab_provenance_count"),
        "ab_provenance_status_counts": detail.get("ab_provenance_status_counts"),
        "ab_provenance_source_state_counts": detail.get(
            "ab_provenance_source_state_counts"
        ),
        "ab_provenance_artifact_hash_status_counts": detail.get(
            "ab_provenance_artifact_hash_status_counts"
        ),
        "ab_provenance_proof_grade_status_counts": detail.get(
            "ab_provenance_proof_grade_status_counts"
        ),
        "ab_comparability_count": detail.get("ab_comparability_count"),
        "ab_comparability_status_counts": detail.get("ab_comparability_status_counts"),
        "ab_coverage_count": detail.get("ab_coverage_count"),
        "ab_coverage_status_counts": detail.get("ab_coverage_status_counts"),
        "ab_coverage_total_rows": detail.get("ab_coverage_total_rows"),
        "ab_coverage_matched_rows": detail.get("ab_coverage_matched_rows"),
        "ab_coverage_baseline_only_rows": detail.get("ab_coverage_baseline_only_rows"),
        "ab_coverage_candidate_only_rows": detail.get(
            "ab_coverage_candidate_only_rows"
        ),
        "ab_repeatability_count": detail.get("ab_repeatability_count"),
        "ab_repeatability_status_counts": detail.get("ab_repeatability_status_counts"),
        "ab_repeatability_proof_grade_status_counts": detail.get(
            "ab_repeatability_proof_grade_status_counts"
        ),
        "ab_repeatability_baseline_runs": detail.get("ab_repeatability_baseline_runs"),
        "ab_repeatability_candidate_runs": detail.get(
            "ab_repeatability_candidate_runs"
        ),
        "ab_repeatability_required_matched_runs_for_hardware_proof": detail.get(
            "ab_repeatability_required_matched_runs_for_hardware_proof"
        ),
        "ab_repeatability_matched_rows": detail.get("ab_repeatability_matched_rows"),
        "report_json_section_count": detail.get("report_json_section_count"),
        "report_json_section_kind_counts": detail.get(
            "report_json_section_kind_counts"
        ),
        "report_section_inventory_count": detail.get("report_section_inventory_count"),
        "report_section_inventory_native_sql_counts": detail.get(
            "report_section_inventory_native_sql_counts"
        ),
        "preflight_finding_count": detail.get("preflight_finding_count"),
        "preflight_finding_kind_counts": detail.get("preflight_finding_kind_counts"),
        "evidence_classification_count": detail.get("evidence_classification_count"),
        "evidence_classification_kind_counts": detail.get(
            "evidence_classification_kind_counts"
        ),
        "capture_count": detail.get("capture_count"),
        "capture_process_kind_counts": detail.get("capture_process_kind_counts"),
        "capture_backend_counts": detail.get("capture_backend_counts"),
        "capture_trace_mode_counts": detail.get("capture_trace_mode_counts"),
        "run_provenance_count": detail.get("run_provenance_count"),
        "run_provenance_source_state_counts": detail.get(
            "run_provenance_source_state_counts"
        ),
        "artifact_identity_count": detail.get("artifact_identity_count"),
        "artifact_identity_artifact_counts": detail.get(
            "artifact_identity_artifact_counts"
        ),
        "artifact_identity_load_status_counts": detail.get(
            "artifact_identity_load_status_counts"
        ),
        "artifact_identity_check_count": detail.get("artifact_identity_check_count"),
        "artifact_identity_check_status_counts": detail.get(
            "artifact_identity_check_status_counts"
        ),
        "capture_capability_count": detail.get("capture_capability_count"),
        "capture_capability_present_counts": detail.get(
            "capture_capability_present_counts"
        ),
        "trace_config_count": detail.get("trace_config_count"),
        "trace_config_status_counts": detail.get("trace_config_status_counts"),
        "trace_config_missing_requested_sidecar_counts": detail.get(
            "trace_config_missing_requested_sidecar_counts"
        ),
        "provider_payload_boundary_count": detail.get(
            "provider_payload_boundary_count"
        ),
        "provider_payload_boundary_status_counts": detail.get(
            "provider_payload_boundary_status_counts"
        ),
        "provider_payload_boundary_route_only_count": detail.get(
            "provider_payload_boundary_route_only_count"
        ),
        "provider_payload_boundary_route_only_lanes": detail.get(
            "provider_payload_boundary_route_only_lanes"
        ),
        "provider_payload_boundary_recorded_count": detail.get(
            "provider_payload_boundary_recorded_count"
        ),
        "provider_payload_boundary_recorded_lanes": detail.get(
            "provider_payload_boundary_recorded_lanes"
        ),
        "trace_event_artifact_count": detail.get("trace_event_artifact_count"),
        "trace_event_artifact_status_counts": detail.get(
            "trace_event_artifact_status_counts"
        ),
        "trace_event_artifact_event_kind_counts": detail.get(
            "trace_event_artifact_event_kind_counts"
        ),
        "backend_event_artifact_count": detail.get("backend_event_artifact_count"),
        "backend_event_artifact_status_counts": detail.get(
            "backend_event_artifact_status_counts"
        ),
        "backend_event_artifact_event_kind_counts": detail.get(
            "backend_event_artifact_event_kind_counts"
        ),
        "backend_event_row_count": detail.get("backend_event_row_count"),
        "backend_event_row_event_kind_counts": detail.get(
            "backend_event_row_event_kind_counts"
        ),
        "backend_event_row_backend_counts": detail.get(
            "backend_event_row_backend_counts"
        ),
        "backend_provider_boundary_count": detail.get(
            "backend_provider_boundary_count"
        ),
        "backend_provider_boundary_status_counts": detail.get(
            "backend_provider_boundary_status_counts"
        ),
        "backend_provider_boundary_stage_counts": detail.get(
            "backend_provider_boundary_stage_counts"
        ),
        "backend_provider_boundary_root_stage_counts": detail.get(
            "backend_provider_boundary_root_stage_counts"
        ),
        "backend_fail_closed_root_cause_count": detail.get(
            "backend_fail_closed_root_cause_count"
        ),
        "backend_fail_closed_root_cause_backend_counts": detail.get(
            "backend_fail_closed_root_cause_backend_counts"
        ),
        "backend_fail_closed_root_cause_stage_counts": detail.get(
            "backend_fail_closed_root_cause_stage_counts"
        ),
        "backend_fail_closed_root_cause_root_stage_counts": detail.get(
            "backend_fail_closed_root_cause_root_stage_counts"
        ),
        "debug_payload_artifact_summary_count": detail.get(
            "debug_payload_artifact_summary_count"
        ),
        "debug_payload_artifact_summary_status_counts": detail.get(
            "debug_payload_artifact_summary_status_counts"
        ),
        "token_quality_summary_count": detail.get("token_quality_summary_count"),
        "token_quality_summary_status_counts": detail.get(
            "token_quality_summary_status_counts"
        ),
        "token_quality_summary_topk_status_counts": detail.get(
            "token_quality_summary_topk_status_counts"
        ),
        "oracle_reference_summary_count": detail.get("oracle_reference_summary_count"),
        "oracle_reference_summary_status_counts": detail.get(
            "oracle_reference_summary_status_counts"
        ),
        "oracle_reference_summary_correctness_counts": detail.get(
            "oracle_reference_summary_correctness_counts"
        ),
        "planning_decision_sidecar_count": detail.get(
            "planning_decision_sidecar_count"
        ),
        "planning_decision_sidecar_status_counts": detail.get(
            "planning_decision_sidecar_status_counts"
        ),
        "planning_decision_sidecar_row_kind_counts": detail.get(
            "planning_decision_sidecar_row_kind_counts"
        ),
        "planning_decision_sidecar_phase_counts": detail.get(
            "planning_decision_sidecar_phase_counts"
        ),
        "token_quality_sidecar_count": detail.get("token_quality_sidecar_count"),
        "token_quality_sidecar_status_counts": detail.get(
            "token_quality_sidecar_status_counts"
        ),
        "token_quality_sidecar_finish_reason_counts": detail.get(
            "token_quality_sidecar_finish_reason_counts"
        ),
        "topk_token_sidecar_count": detail.get("topk_token_sidecar_count"),
        "topk_token_sidecar_status_counts": detail.get(
            "topk_token_sidecar_status_counts"
        ),
        "topk_token_sidecar_selected_status_counts": detail.get(
            "topk_token_sidecar_selected_status_counts"
        ),
        "topk_token_sidecar_score_kind_counts": detail.get(
            "topk_token_sidecar_score_kind_counts"
        ),
        "tensor_payload_sidecar_count": detail.get("tensor_payload_sidecar_count"),
        "tensor_payload_sidecar_status_counts": detail.get(
            "tensor_payload_sidecar_status_counts"
        ),
        "tensor_payload_sidecar_kind_counts": detail.get(
            "tensor_payload_sidecar_kind_counts"
        ),
        "tensor_payload_sidecar_role_counts": detail.get(
            "tensor_payload_sidecar_role_counts"
        ),
        "kv_payload_digest_sidecar_count": detail.get(
            "kv_payload_digest_sidecar_count"
        ),
        "kv_payload_digest_sidecar_status_counts": detail.get(
            "kv_payload_digest_sidecar_status_counts"
        ),
        "kv_payload_digest_sidecar_role_counts": detail.get(
            "kv_payload_digest_sidecar_role_counts"
        ),
        "logit_slice_sidecar_count": detail.get("logit_slice_sidecar_count"),
        "logit_slice_sidecar_status_counts": detail.get(
            "logit_slice_sidecar_status_counts"
        ),
        "logit_slice_sidecar_role_counts": detail.get(
            "logit_slice_sidecar_role_counts"
        ),
        "logit_slice_sidecar_action_counts": detail.get(
            "logit_slice_sidecar_action_counts"
        ),
        "activation_digest_sidecar_count": detail.get(
            "activation_digest_sidecar_count"
        ),
        "activation_digest_sidecar_status_counts": detail.get(
            "activation_digest_sidecar_status_counts"
        ),
        "activation_digest_sidecar_role_counts": detail.get(
            "activation_digest_sidecar_role_counts"
        ),
        "activation_digest_sidecar_intrinsic_counts": detail.get(
            "activation_digest_sidecar_intrinsic_counts"
        ),
        "device_result_digest_sidecar_count": detail.get(
            "device_result_digest_sidecar_count"
        ),
        "device_result_digest_sidecar_status_counts": detail.get(
            "device_result_digest_sidecar_status_counts"
        ),
        "device_result_digest_sidecar_role_counts": detail.get(
            "device_result_digest_sidecar_role_counts"
        ),
        "device_result_digest_sidecar_action_counts": detail.get(
            "device_result_digest_sidecar_action_counts"
        ),
        "device_result_digest_sidecar_intrinsic_counts": detail.get(
            "device_result_digest_sidecar_intrinsic_counts"
        ),
        "scheduler_packet_lineage_sidecar_count": detail.get(
            "scheduler_packet_lineage_sidecar_count"
        ),
        "scheduler_packet_lineage_sidecar_status_counts": detail.get(
            "scheduler_packet_lineage_sidecar_status_counts"
        ),
        "scheduler_packet_lineage_sidecar_executor_counts": detail.get(
            "scheduler_packet_lineage_sidecar_executor_counts"
        ),
        "scheduler_kv_shard_lifecycle_sidecar_count": detail.get(
            "scheduler_kv_shard_lifecycle_sidecar_count"
        ),
        "scheduler_kv_shard_lifecycle_sidecar_status_counts": detail.get(
            "scheduler_kv_shard_lifecycle_sidecar_status_counts"
        ),
        "scheduler_kv_shard_lifecycle_sidecar_lifecycle_counts": detail.get(
            "scheduler_kv_shard_lifecycle_sidecar_lifecycle_counts"
        ),
        "scheduler_listener_sparse_logit_sidecar_count": detail.get(
            "scheduler_listener_sparse_logit_sidecar_count"
        ),
        "scheduler_listener_sparse_logit_sidecar_status_counts": detail.get(
            "scheduler_listener_sparse_logit_sidecar_status_counts"
        ),
        "scheduler_listener_sparse_logit_sidecar_listener_status_counts": detail.get(
            "scheduler_listener_sparse_logit_sidecar_listener_status_counts"
        ),
        "scheduler_listener_sparse_logit_sidecar_executor_counts": detail.get(
            "scheduler_listener_sparse_logit_sidecar_executor_counts"
        ),
        "device_dma_lifecycle_sidecar_count": detail.get(
            "device_dma_lifecycle_sidecar_count"
        ),
        "device_dma_lifecycle_sidecar_status_counts": detail.get(
            "device_dma_lifecycle_sidecar_status_counts"
        ),
        "device_dma_lifecycle_sidecar_stage_counts": detail.get(
            "device_dma_lifecycle_sidecar_stage_counts"
        ),
        "device_dma_lifecycle_sidecar_queue_counts": detail.get(
            "device_dma_lifecycle_sidecar_queue_counts"
        ),
        "attention_page_trace_sidecar_count": detail.get(
            "attention_page_trace_sidecar_count"
        ),
        "attention_page_trace_sidecar_status_counts": detail.get(
            "attention_page_trace_sidecar_status_counts"
        ),
        "attention_page_trace_sidecar_action_counts": detail.get(
            "attention_page_trace_sidecar_action_counts"
        ),
        "introspection_artifact_count": detail.get("introspection_artifact_count"),
        "introspection_artifact_status_counts": detail.get(
            "introspection_artifact_status_counts"
        ),
        "introspection_artifact_kind_counts": detail.get(
            "introspection_artifact_kind_counts"
        ),
        "introspection_artifact_format_counts": detail.get(
            "introspection_artifact_format_counts"
        ),
        "introspection_artifact_sensitivity_counts": detail.get(
            "introspection_artifact_sensitivity_counts"
        ),
        "introspection_artifact_compile_feature_counts": detail.get(
            "introspection_artifact_compile_feature_counts"
        ),
        "introspection_artifact_row_count_total": detail.get(
            "introspection_artifact_row_count_total"
        ),
        "introspection_artifact_byte_count_total": detail.get(
            "introspection_artifact_byte_count_total"
        ),
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
        "introspection_section_inventory_count": detail.get(
            "introspection_section_inventory_count"
        ),
        "introspection_section_inventory_status_counts": detail.get(
            "introspection_section_inventory_status_counts"
        ),
        "introspection_section_inventory_capability_counts": detail.get(
            "introspection_section_inventory_capability_counts"
        ),
        "timeline_query_summary_count": detail.get("timeline_query_summary_count"),
        "timeline_query_summary_status_counts": detail.get(
            "timeline_query_summary_status_counts"
        ),
        "next_measurement_samples": detail.get("next_measurement_samples"),
        "answerability_samples": detail.get("answerability_samples"),
        "unsupported_claim_samples": detail.get("unsupported_claim_samples"),
        "analysis_command_samples": detail.get("analysis_command_samples"),
        "report_json_section_samples": detail.get("report_json_section_samples"),
        "report_section_inventory_samples": detail.get(
            "report_section_inventory_samples"
        ),
        "preflight_finding_samples": detail.get("preflight_finding_samples"),
        "evidence_classification_samples": detail.get(
            "evidence_classification_samples"
        ),
        "capture_samples": detail.get("capture_samples"),
        "run_provenance_samples": detail.get("run_provenance_samples"),
        "artifact_identity_samples": detail.get("artifact_identity_samples"),
        "artifact_identity_check_samples": detail.get(
            "artifact_identity_check_samples"
        ),
        "capture_capability_samples": detail.get("capture_capability_samples"),
        "trace_config_samples": detail.get("trace_config_samples"),
        "trace_event_artifact_samples": detail.get("trace_event_artifact_samples"),
        "provider_payload_boundary_samples": detail.get(
            "provider_payload_boundary_samples"
        ),
        "provider_payload_boundary_route_only_samples": detail.get(
            "provider_payload_boundary_route_only_samples"
        ),
        "provider_payload_boundary_recorded_samples": detail.get(
            "provider_payload_boundary_recorded_samples"
        ),
        "backend_event_artifact_samples": detail.get("backend_event_artifact_samples"),
        "backend_event_samples": detail.get("backend_event_samples"),
        "backend_provider_boundary_samples": detail.get(
            "backend_provider_boundary_samples"
        ),
        "backend_fail_closed_root_cause_samples": detail.get(
            "backend_fail_closed_root_cause_samples"
        ),
        "debug_payload_artifact_summary_samples": detail.get(
            "debug_payload_artifact_summary_samples"
        ),
        "token_quality_summary_samples": detail.get("token_quality_summary_samples"),
        "oracle_reference_summary_samples": detail.get(
            "oracle_reference_summary_samples"
        ),
        "planning_decision_sidecar_samples": detail.get(
            "planning_decision_sidecar_samples"
        ),
        "token_quality_sidecar_samples": detail.get("token_quality_sidecar_samples"),
        "topk_token_sidecar_samples": detail.get("topk_token_sidecar_samples"),
        "tensor_payload_sidecar_samples": detail.get("tensor_payload_sidecar_samples"),
        "kv_payload_digest_sidecar_samples": detail.get(
            "kv_payload_digest_sidecar_samples"
        ),
        "logit_slice_sidecar_samples": detail.get("logit_slice_sidecar_samples"),
        "activation_digest_sidecar_samples": detail.get(
            "activation_digest_sidecar_samples"
        ),
        "device_result_digest_sidecar_samples": detail.get(
            "device_result_digest_sidecar_samples"
        ),
        "scheduler_packet_lineage_sidecar_samples": detail.get(
            "scheduler_packet_lineage_sidecar_samples"
        ),
        "scheduler_kv_shard_lifecycle_sidecar_samples": detail.get(
            "scheduler_kv_shard_lifecycle_sidecar_samples"
        ),
        "scheduler_listener_sparse_logit_sidecar_samples": detail.get(
            "scheduler_listener_sparse_logit_sidecar_samples"
        ),
        "device_dma_lifecycle_sidecar_samples": detail.get(
            "device_dma_lifecycle_sidecar_samples"
        ),
        "attention_page_trace_sidecar_samples": detail.get(
            "attention_page_trace_sidecar_samples"
        ),
        "introspection_capability_samples": detail.get(
            "introspection_capability_samples"
        ),
        "introspection_artifact_samples": detail.get("introspection_artifact_samples"),
        "introspection_artifact_summary_samples": detail.get(
            "introspection_artifact_summary_samples"
        ),
        "introspection_section_inventory_samples": detail.get(
            "introspection_section_inventory_samples"
        ),
        "timeline_query_summary_samples": detail.get("timeline_query_summary_samples"),
        "supported_claim_samples": detail.get("supported_claim_samples"),
        "correctness_evidence_samples": detail.get("correctness_evidence_samples"),
        "evidence_artifact_check_samples": detail.get(
            "evidence_artifact_check_samples"
        ),
        "promotion_gate_summary_samples": detail.get("promotion_gate_summary_samples"),
        "trace_mode_guardrail_samples": detail.get("trace_mode_guardrail_samples"),
        "ab_provenance_samples": detail.get("ab_provenance_samples"),
        "ab_comparability_samples": detail.get("ab_comparability_samples"),
        "ab_coverage_samples": detail.get("ab_coverage_samples"),
        "ab_repeatability_samples": detail.get("ab_repeatability_samples"),
        "section_names": detail.get("section_names"),
    }
    errors = gate.get("errors")
    if isinstance(errors, list) and errors:
        summary["errors"] = errors
    return {
        key: value for key, value in summary.items() if value not in (None, "", [], {})
    }


def _trace_report_sample_value_present(value: Any) -> bool:
    return value is not None and value != ""


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
    if summary.get("report_grade_basis"):
        lines.append(f"- Trace grade basis: {summary['report_grade_basis']}")
    if summary.get("report_grade_promotion_gate"):
        lines.append(
            f"- Trace grade promotion gate: {summary['report_grade_promotion_gate']}"
        )
    if summary.get("report_triage_status_counts"):
        lines.append(
            "- Report triage: "
            f"`{json.dumps(summary['report_triage_status_counts'], sort_keys=True)}`"
        )
    for sample in summary.get("report_triage_samples", [])[:4]:
        if not isinstance(sample, Mapping):
            continue
        status = sample.get("triage_status") or "unknown"
        parts = []
        for key, label in (
            ("first_blocked_gate", "gate"),
            ("first_blocked_gate_status", "gate_status"),
            ("first_next_measurement_priority", "next_priority"),
            ("first_next_measurement", "next"),
            ("first_useful_section", "section"),
            ("first_action", "action"),
            ("claim_boundary", "boundary"),
        ):
            value = sample.get(key)
            if value:
                parts.append(f"{label}={value}")
        line = f"- Report triage detail: status={status}"
        if parts:
            line += " " + " ".join(parts)
        lines.append(line)
    preflight_counts = {
        key: summary.get(f"preflight_{key}_count") for key in ("ok", "warn", "fail")
    }
    preflight_counts = {
        key: value
        for key, value in preflight_counts.items()
        if _trace_report_sample_value_present(value)
    }
    if preflight_counts:
        lines.append(
            f"- Preflight counts: `{json.dumps(preflight_counts, sort_keys=True)}`"
        )
    section_names = summary.get("section_names")
    if isinstance(section_names, list):
        display_names = [str(name) for name in section_names[:16]]
        suffix = ""
        if len(section_names) > len(display_names):
            suffix = f", ... (+{len(section_names) - len(display_names)} more)"
        count = summary.get("section_count", len(section_names))
        lines.append(
            "- Report sections available: "
            f"count={count} names={', '.join(display_names)}{suffix}"
        )
    if summary.get("answerability_status_counts"):
        lines.append(
            "- Answerability: "
            f"`{json.dumps(summary['answerability_status_counts'], sort_keys=True)}`"
        )
    for sample in summary.get("answerability_samples", [])[:6]:
        if not isinstance(sample, Mapping):
            continue
        question = sample.get("question")
        status = sample.get("status")
        basis = sample.get("basis")
        label = question or status or "unknown"
        parts = []
        if status:
            parts.append(f"status={status}")
        if basis:
            parts.append(f"basis={basis}")
        line = f"- Answerability detail: {label}"
        if parts:
            line += " " + " ".join(parts)
        lines.append(line)
    if summary.get("preflight_finding_kind_counts"):
        lines.append(
            "- Preflight findings: "
            "`"
            + json.dumps(
                summary["preflight_finding_kind_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("evidence_classification_kind_counts"):
        lines.append(
            "- Evidence classification: "
            "`"
            + json.dumps(
                summary["evidence_classification_kind_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("correctness_evidence_status_counts"):
        lines.append(
            "- Correctness evidence: "
            "`"
            + json.dumps(
                summary["correctness_evidence_status_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("evidence_artifact_check_status_counts"):
        lines.append(
            "- Evidence artifact checks: "
            "`"
            + json.dumps(
                summary["evidence_artifact_check_status_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("promotion_gate_summary_status_counts"):
        lines.append(
            "- Promotion gate summary: "
            "`"
            + json.dumps(
                summary["promotion_gate_summary_status_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("promotion_gate_summary_proof_grade_status_counts"):
        lines.append(
            "- Promotion proof-grade status: "
            "`"
            + json.dumps(
                summary["promotion_gate_summary_proof_grade_status_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("trace_mode_guardrail_mode_counts"):
        lines.append(
            "- Trace mode guardrails: "
            "`"
            + json.dumps(
                summary["trace_mode_guardrail_mode_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("trace_mode_guardrail_overhead_counts"):
        lines.append(
            "- Trace overhead boundaries: "
            "`"
            + json.dumps(
                summary["trace_mode_guardrail_overhead_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("ab_provenance_status_counts"):
        lines.append(
            "- A/B provenance: "
            "`"
            + json.dumps(summary["ab_provenance_status_counts"], sort_keys=True)
            + "`"
        )
    if summary.get("ab_provenance_artifact_hash_status_counts"):
        lines.append(
            "- A/B artifact hashes: "
            "`"
            + json.dumps(
                summary["ab_provenance_artifact_hash_status_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("ab_comparability_status_counts"):
        lines.append(
            "- A/B comparability: "
            "`"
            + json.dumps(
                summary["ab_comparability_status_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("ab_coverage_status_counts"):
        lines.append(
            "- A/B coverage: "
            "`" + json.dumps(summary["ab_coverage_status_counts"], sort_keys=True) + "`"
        )
    if summary.get("ab_repeatability_status_counts"):
        lines.append(
            "- A/B repeatability: "
            "`"
            + json.dumps(
                summary["ab_repeatability_status_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("report_json_section_kind_counts"):
        lines.append(
            "- Report JSON sections: "
            "`"
            + json.dumps(summary["report_json_section_kind_counts"], sort_keys=True)
            + "`"
        )
    if summary.get("report_section_inventory_native_sql_counts"):
        lines.append(
            "- Report section inventory: "
            "`"
            + json.dumps(
                summary["report_section_inventory_native_sql_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("capture_trace_mode_counts"):
        lines.append(
            "- Capture trace modes: "
            "`" + json.dumps(summary["capture_trace_mode_counts"], sort_keys=True) + "`"
        )
    if summary.get("capture_backend_counts"):
        lines.append(
            "- Capture backends: "
            "`" + json.dumps(summary["capture_backend_counts"], sort_keys=True) + "`"
        )
    if summary.get("run_provenance_source_state_counts"):
        lines.append(
            "- Run provenance source states: "
            "`"
            + json.dumps(
                summary["run_provenance_source_state_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("artifact_identity_artifact_counts"):
        lines.append(
            "- Artifact identities: "
            "`"
            + json.dumps(
                summary["artifact_identity_artifact_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("artifact_identity_check_status_counts"):
        lines.append(
            "- Artifact identity checks: "
            "`"
            + json.dumps(
                summary["artifact_identity_check_status_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("capture_capability_present_counts"):
        lines.append(
            "- Capture capabilities: "
            "`"
            + json.dumps(
                summary["capture_capability_present_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("trace_config_status_counts"):
        lines.append(
            "- Trace config: "
            f"`{json.dumps(summary['trace_config_status_counts'], sort_keys=True)}`"
        )
    if summary.get("trace_config_missing_requested_sidecar_counts"):
        lines.append(
            "- Missing requested sidecars: "
            "`"
            + json.dumps(
                summary["trace_config_missing_requested_sidecar_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("provider_payload_boundary_status_counts"):
        lines.append(
            "- Provider payload boundaries: "
            "`"
            + json.dumps(
                summary["provider_payload_boundary_status_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("provider_payload_boundary_route_only_lanes"):
        lines.append(
            "- Route-only provider payload lanes: "
            "`"
            + json.dumps(
                summary["provider_payload_boundary_route_only_lanes"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("provider_payload_boundary_recorded_lanes"):
        lines.append(
            "- Recorded provider payload lanes: "
            "`"
            + json.dumps(
                summary["provider_payload_boundary_recorded_lanes"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("trace_event_artifact_status_counts"):
        lines.append(
            "- Trace event artifacts: "
            "`"
            + json.dumps(
                summary["trace_event_artifact_status_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("trace_event_artifact_event_kind_counts"):
        lines.append(
            "- Trace event artifact kinds: "
            "`"
            + json.dumps(
                summary["trace_event_artifact_event_kind_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("backend_event_artifact_status_counts"):
        lines.append(
            "- Backend event artifacts: "
            "`"
            + json.dumps(
                summary["backend_event_artifact_status_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("backend_event_artifact_event_kind_counts"):
        lines.append(
            "- Backend event artifact kinds: "
            "`"
            + json.dumps(
                summary["backend_event_artifact_event_kind_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("backend_event_row_event_kind_counts"):
        lines.append(
            "- Backend event rows: "
            "`"
            + json.dumps(
                summary["backend_event_row_event_kind_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("backend_event_row_backend_counts"):
        lines.append(
            "- Backend event backends: "
            "`"
            + json.dumps(
                summary["backend_event_row_backend_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("backend_provider_boundary_status_counts"):
        lines.append(
            "- Backend provider boundaries: "
            "`"
            + json.dumps(
                summary["backend_provider_boundary_status_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("backend_provider_boundary_stage_counts"):
        lines.append(
            "- Backend provider boundary stages: "
            "`"
            + json.dumps(
                summary["backend_provider_boundary_stage_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("backend_provider_boundary_root_stage_counts"):
        lines.append(
            "- Backend provider root stages: "
            "`"
            + json.dumps(
                summary["backend_provider_boundary_root_stage_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("backend_fail_closed_root_cause_stage_counts"):
        lines.append(
            "- Backend fail-closed stages: "
            "`"
            + json.dumps(
                summary["backend_fail_closed_root_cause_stage_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("backend_fail_closed_root_cause_root_stage_counts"):
        lines.append(
            "- Backend fail-closed root stages: "
            "`"
            + json.dumps(
                summary["backend_fail_closed_root_cause_root_stage_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("debug_payload_artifact_summary_status_counts"):
        lines.append(
            "- Debug payload artifacts: "
            "`"
            + json.dumps(
                summary["debug_payload_artifact_summary_status_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("token_quality_summary_status_counts"):
        lines.append(
            "- Token quality summaries: "
            "`"
            + json.dumps(
                summary["token_quality_summary_status_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("token_quality_summary_topk_status_counts"):
        lines.append(
            "- Token quality top-k status: "
            "`"
            + json.dumps(
                summary["token_quality_summary_topk_status_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("oracle_reference_summary_status_counts"):
        lines.append(
            "- Oracle references: "
            "`"
            + json.dumps(
                summary["oracle_reference_summary_status_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("oracle_reference_summary_correctness_counts"):
        lines.append(
            "- Oracle-reference correctness boundary: "
            "`"
            + json.dumps(
                summary["oracle_reference_summary_correctness_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("planning_decision_sidecar_row_kind_counts"):
        lines.append(
            "- Planning decision sidecars: "
            "`"
            + json.dumps(
                summary["planning_decision_sidecar_row_kind_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("planning_decision_sidecar_phase_counts"):
        lines.append(
            "- Planning decision phases: "
            "`"
            + json.dumps(
                summary["planning_decision_sidecar_phase_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("token_quality_sidecar_status_counts"):
        lines.append(
            "- Token quality sidecars: "
            "`"
            + json.dumps(
                summary["token_quality_sidecar_status_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("token_quality_sidecar_finish_reason_counts"):
        lines.append(
            "- Token quality finish reasons: "
            "`"
            + json.dumps(
                summary["token_quality_sidecar_finish_reason_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("topk_token_sidecar_selected_status_counts"):
        lines.append(
            "- Top-K token candidate status: "
            "`"
            + json.dumps(
                summary["topk_token_sidecar_selected_status_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("topk_token_sidecar_score_kind_counts"):
        lines.append(
            "- Top-K token score kinds: "
            "`"
            + json.dumps(
                summary["topk_token_sidecar_score_kind_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("tensor_payload_sidecar_kind_counts"):
        lines.append(
            "- Tensor payload sidecars: "
            "`"
            + json.dumps(
                summary["tensor_payload_sidecar_kind_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("tensor_payload_sidecar_role_counts"):
        lines.append(
            "- Tensor payload roles: "
            "`"
            + json.dumps(
                summary["tensor_payload_sidecar_role_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("kv_payload_digest_sidecar_role_counts"):
        lines.append(
            "- K/V payload digest roles: "
            "`"
            + json.dumps(
                summary["kv_payload_digest_sidecar_role_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("logit_slice_sidecar_role_counts"):
        lines.append(
            "- Logit slice sidecars: "
            "`"
            + json.dumps(
                summary["logit_slice_sidecar_role_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("logit_slice_sidecar_action_counts"):
        lines.append(
            "- Logit slice actions: "
            "`"
            + json.dumps(
                summary["logit_slice_sidecar_action_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("activation_digest_sidecar_role_counts"):
        lines.append(
            "- Activation digest sidecars: "
            "`"
            + json.dumps(
                summary["activation_digest_sidecar_role_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("activation_digest_sidecar_intrinsic_counts"):
        lines.append(
            "- Activation digest intrinsics: "
            "`"
            + json.dumps(
                summary["activation_digest_sidecar_intrinsic_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("device_result_digest_sidecar_status_counts"):
        lines.append(
            "- Device result digest sidecars: "
            "`"
            + json.dumps(
                summary["device_result_digest_sidecar_status_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("device_result_digest_sidecar_role_counts"):
        lines.append(
            "- Device result digest roles: "
            "`"
            + json.dumps(
                summary["device_result_digest_sidecar_role_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("device_result_digest_sidecar_action_counts"):
        lines.append(
            "- Device result digest actions: "
            "`"
            + json.dumps(
                summary["device_result_digest_sidecar_action_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("device_result_digest_sidecar_intrinsic_counts"):
        lines.append(
            "- Device result digest intrinsics: "
            "`"
            + json.dumps(
                summary["device_result_digest_sidecar_intrinsic_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("scheduler_packet_lineage_sidecar_status_counts"):
        lines.append(
            "- Scheduler packet lineage: "
            "`"
            + json.dumps(
                summary["scheduler_packet_lineage_sidecar_status_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("scheduler_packet_lineage_sidecar_executor_counts"):
        lines.append(
            "- Scheduler packet executors: "
            "`"
            + json.dumps(
                summary["scheduler_packet_lineage_sidecar_executor_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("scheduler_kv_shard_lifecycle_sidecar_status_counts"):
        lines.append(
            "- Scheduler K/V shard lifecycle: "
            "`"
            + json.dumps(
                summary["scheduler_kv_shard_lifecycle_sidecar_status_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("scheduler_kv_shard_lifecycle_sidecar_lifecycle_counts"):
        lines.append(
            "- Scheduler K/V lifecycle status: "
            "`"
            + json.dumps(
                summary["scheduler_kv_shard_lifecycle_sidecar_lifecycle_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("scheduler_listener_sparse_logit_sidecar_listener_status_counts"):
        lines.append(
            "- Scheduler sparse listener status: "
            "`"
            + json.dumps(
                summary[
                    "scheduler_listener_sparse_logit_sidecar_listener_status_counts"
                ],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("scheduler_listener_sparse_logit_sidecar_executor_counts"):
        lines.append(
            "- Scheduler sparse listener executors: "
            "`"
            + json.dumps(
                summary["scheduler_listener_sparse_logit_sidecar_executor_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("device_dma_lifecycle_sidecar_status_counts"):
        lines.append(
            "- Device DMA lifecycle: "
            "`"
            + json.dumps(
                summary["device_dma_lifecycle_sidecar_status_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("device_dma_lifecycle_sidecar_stage_counts"):
        lines.append(
            "- Device DMA stages: "
            "`"
            + json.dumps(
                summary["device_dma_lifecycle_sidecar_stage_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("attention_page_trace_sidecar_status_counts"):
        lines.append(
            "- Attention page traces: "
            "`"
            + json.dumps(
                summary["attention_page_trace_sidecar_status_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("attention_page_trace_sidecar_action_counts"):
        lines.append(
            "- Attention page actions: "
            "`"
            + json.dumps(
                summary["attention_page_trace_sidecar_action_counts"],
                sort_keys=True,
            )
            + "`"
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
    if summary.get("introspection_artifact_status_counts"):
        lines.append(
            "- Introspection raw artifacts: "
            "`"
            + json.dumps(
                summary["introspection_artifact_status_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("introspection_artifact_compile_feature_counts"):
        lines.append(
            "- Introspection raw artifact compile features: "
            "`"
            + json.dumps(
                summary["introspection_artifact_compile_feature_counts"],
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
    if summary.get("introspection_section_inventory_status_counts"):
        lines.append(
            "- Introspection sections: "
            "`"
            + json.dumps(
                summary["introspection_section_inventory_status_counts"],
                sort_keys=True,
            )
            + "`"
        )
    if summary.get("timeline_query_summary_status_counts"):
        lines.append(
            "- Timeline query summary: "
            "`"
            + json.dumps(
                summary["timeline_query_summary_status_counts"],
                sort_keys=True,
            )
            + "`"
        )
    for sample in summary.get("report_json_section_samples", [])[:64]:
        if not isinstance(sample, Mapping):
            continue
        json_path = sample.get("json_path") or sample.get("json_section")
        heading = sample.get("heading")
        section_kind = sample.get("section_kind")
        requires_timeline = sample.get("requires_timeline_trace")
        boundary = sample.get("claim_boundary")
        if json_path:
            parts = []
            if heading:
                parts.append(f"heading={heading}")
            if section_kind:
                parts.append(f"kind={section_kind}")
            if _trace_report_sample_value_present(requires_timeline):
                parts.append(f"requires_timeline={requires_timeline}")
            if boundary:
                parts.append(f"boundary={boundary}")
            line = f"- Report JSON section: {json_path}"
            if parts:
                line += " " + " ".join(parts)
            lines.append(line)
    for sample in summary.get("report_section_inventory_samples", [])[:8]:
        if not isinstance(sample, Mapping):
            continue
        heading = sample.get("heading")
        query = sample.get("query")
        native_sql = sample.get("native_sql")
        portable_command = sample.get("portable_command")
        native_sql_command = sample.get("native_sql_command")
        label = heading or query or "unknown"
        parts = []
        if query:
            parts.append(f"query={query}")
        if _trace_report_sample_value_present(native_sql):
            parts.append(f"native_sql={native_sql}")
        line = f"- Report section inventory: {label}"
        if parts:
            line += " " + " ".join(parts)
        lines.append(line)
        if portable_command:
            lines.append(f"  Portable command: `{portable_command}`")
        if native_sql_command:
            lines.append(f"  Native SQL command: `{native_sql_command}`")
    for finding in summary.get("preflight_finding_samples", [])[:8]:
        if isinstance(finding, str) and finding:
            lines.append(f"- Preflight finding: {finding}")
    for classification in summary.get("evidence_classification_samples", [])[:4]:
        if isinstance(classification, str) and classification:
            lines.append(f"- Evidence classification: {classification}")
    for sample in summary.get("analysis_command_samples", [])[:3]:
        if not isinstance(sample, Mapping):
            continue
        purpose = sample.get("purpose")
        command = sample.get("command")
        if purpose:
            lines.append(f"- Analysis command: {purpose}")
        if command:
            lines.append(f"  Command: `{command}`")
    for sample in summary.get("capture_samples", [])[:3]:
        if not isinstance(sample, Mapping):
            continue
        trace_run_id = sample.get("trace_run_id")
        process_kind = sample.get("process_kind")
        trace_mode = sample.get("trace_mode")
        trace_sinks = sample.get("trace_sinks")
        backend_id = sample.get("backend_id")
        model_id = sample.get("model_id")
        target_plan_sha = sample.get("target_plan_sha256")
        worktree_dirty = sample.get("worktree_dirty")
        label = trace_run_id or process_kind or "unknown"
        parts = []
        if process_kind:
            parts.append(f"process={process_kind}")
        if trace_mode:
            parts.append(f"mode={trace_mode}")
        if trace_sinks:
            parts.append(f"sinks={trace_sinks}")
        if backend_id:
            parts.append(f"backend={backend_id}")
        if model_id:
            parts.append(f"model={model_id}")
        if target_plan_sha:
            parts.append(f"target_plan_sha256={target_plan_sha}")
        if _trace_report_sample_value_present(worktree_dirty):
            parts.append(f"dirty={worktree_dirty}")
        line = f"- Capture: {label}"
        if parts:
            line += " " + " ".join(parts)
        lines.append(line)
    for sample in summary.get("run_provenance_samples", [])[:3]:
        if not isinstance(sample, Mapping):
            continue
        binary = sample.get("binary")
        source_state = sample.get("source_state")
        git_sha = sample.get("git_sha")
        hardware_card_count = sample.get("hardware_card_count")
        hardware_cards = sample.get("hardware_cards")
        boundary = sample.get("provenance_boundary")
        label = binary or source_state or "unknown"
        parts = []
        if source_state:
            parts.append(f"source_state={source_state}")
        if git_sha:
            parts.append(f"git_sha={git_sha}")
        if _trace_report_sample_value_present(hardware_card_count):
            parts.append(f"hardware_cards={hardware_card_count}")
        if hardware_cards:
            parts.append(f"card_ids={hardware_cards}")
        if boundary:
            parts.append(f"boundary={boundary}")
        line = f"- Run provenance: {label}"
        if parts:
            line += " " + " ".join(parts)
        lines.append(line)
    for sample in summary.get("artifact_identity_samples", [])[:3]:
        if not isinstance(sample, Mapping):
            continue
        artifact = sample.get("artifact")
        path = sample.get("path")
        sha256 = sample.get("sha256")
        load_status = sample.get("load_status")
        status_source = sample.get("status_source")
        if artifact:
            parts = []
            if path:
                parts.append(f"path={path}")
            if sha256:
                parts.append(f"sha256={sha256}")
            if load_status:
                parts.append(f"load_status={load_status}")
            if status_source:
                parts.append(f"status_source={status_source}")
            line = f"- Artifact identity: {artifact}"
            if parts:
                line += " " + " ".join(parts)
            lines.append(line)
    for sample in summary.get("artifact_identity_check_samples", [])[:3]:
        if not isinstance(sample, Mapping):
            continue
        artifact = sample.get("artifact")
        check = sample.get("check")
        status = sample.get("status")
        detail = sample.get("detail")
        label = artifact or check or "unknown"
        parts = []
        if check:
            parts.append(f"check={check}")
        if status:
            parts.append(f"status={status}")
        if detail:
            parts.append(f"detail={detail}")
        line = f"- Artifact identity check: {label}"
        if parts:
            line += " " + " ".join(parts)
        lines.append(line)
    for sample in summary.get("capture_capability_samples", [])[:8]:
        if not isinstance(sample, Mapping):
            continue
        capability = sample.get("capability")
        present = sample.get("present")
        if capability:
            line = f"- Capture capability: {capability}"
            if _trace_report_sample_value_present(present):
                line += f" present={present}"
            lines.append(line)
    for sample in summary.get("trace_config_samples", [])[:3]:
        if not isinstance(sample, Mapping):
            continue
        status = sample.get("config_status")
        requested = sample.get("requested_sidecar_controls")
        recorded = sample.get("recorded_sidecar_capabilities")
        missing = sample.get("missing_requested_sidecar_controls")
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
            if missing:
                parts.append(f"missing={missing}")
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
    for sample in summary.get("ab_provenance_samples", [])[:4]:
        if not isinstance(sample, Mapping):
            continue
        role = sample.get("role")
        status = sample.get("provenance_status")
        source_state = sample.get("source_state")
        artifact_hash_status = sample.get("artifact_hash_status")
        hardware_cards = sample.get("hardware_cards")
        proof_grade_status = sample.get("proof_grade_status")
        basis = sample.get("basis")
        label = role or status or "unknown"
        parts = []
        if status:
            parts.append(f"status={status}")
        if source_state:
            parts.append(f"source_state={source_state}")
        if artifact_hash_status:
            parts.append(f"artifact_hashes={artifact_hash_status}")
        if hardware_cards:
            parts.append(f"hardware_cards={hardware_cards}")
        if proof_grade_status:
            parts.append(f"proof_grade={proof_grade_status}")
        if basis:
            parts.append(f"basis={basis}")
        line = f"- A/B provenance: {label}"
        if parts:
            line += " " + " ".join(parts)
        lines.append(line)
    for sample in summary.get("ab_comparability_samples", [])[:3]:
        if not isinstance(sample, Mapping):
            continue
        status = sample.get("status")
        basis = sample.get("basis")
        promotion_gate = sample.get("promotion_gate")
        if status:
            parts = []
            if basis:
                parts.append(f"basis={basis}")
            if promotion_gate:
                parts.append(f"gate={promotion_gate}")
            line = f"- A/B comparability: {status}"
            if parts:
                line += " " + " ".join(parts)
            lines.append(line)
    for sample in summary.get("ab_coverage_samples", [])[:3]:
        if not isinstance(sample, Mapping):
            continue
        status = sample.get("coverage_status")
        align = sample.get("align")
        matched_rows = sample.get("matched_rows")
        baseline_only = sample.get("baseline_only_rows")
        candidate_only = sample.get("candidate_only_rows")
        warnings = sample.get("warnings")
        basis = sample.get("basis")
        if status:
            parts = []
            if align:
                parts.append(f"align={align}")
            if _trace_report_sample_value_present(matched_rows):
                parts.append(f"matched={matched_rows}")
            if _trace_report_sample_value_present(baseline_only):
                parts.append(f"baseline_only={baseline_only}")
            if _trace_report_sample_value_present(candidate_only):
                parts.append(f"candidate_only={candidate_only}")
            if warnings:
                parts.append(f"warnings={warnings}")
            if basis:
                parts.append(f"basis={basis}")
            line = f"- A/B coverage: {status}"
            if parts:
                line += " " + " ".join(parts)
            lines.append(line)
    for sample in summary.get("ab_repeatability_samples", [])[:3]:
        if not isinstance(sample, Mapping):
            continue
        status = sample.get("status")
        align = sample.get("align")
        baseline_runs = sample.get("baseline_runs")
        candidate_runs = sample.get("candidate_runs")
        required_runs = sample.get("required_matched_runs_for_hardware_proof")
        matched_rows = sample.get("matched_rows")
        proof_grade_status = sample.get("proof_grade_status")
        basis = sample.get("basis")
        if status:
            parts = []
            if align:
                parts.append(f"align={align}")
            if _trace_report_sample_value_present(baseline_runs):
                parts.append(f"baseline_runs={baseline_runs}")
            if _trace_report_sample_value_present(candidate_runs):
                parts.append(f"candidate_runs={candidate_runs}")
            if _trace_report_sample_value_present(required_runs):
                parts.append(f"required_runs={required_runs}")
            if _trace_report_sample_value_present(matched_rows):
                parts.append(f"matched={matched_rows}")
            if proof_grade_status:
                parts.append(f"proof_grade={proof_grade_status}")
            if basis:
                parts.append(f"basis={basis}")
            line = f"- A/B repeatability: {status}"
            if parts:
                line += " " + " ".join(parts)
            lines.append(line)
    for sample in summary.get("trace_event_artifact_samples", [])[:3]:
        if not isinstance(sample, Mapping):
            continue
        path = sample.get("path")
        index = sample.get("index")
        status = sample.get("status")
        row_count = sample.get("row_count")
        matching_rows = sample.get("matching_trace_run_id_rows")
        event_kinds = sample.get("event_kinds")
        sha256 = sample.get("sha256")
        label = path or index or "unknown"
        parts = []
        if status:
            parts.append(f"status={status}")
        if _trace_report_sample_value_present(row_count):
            parts.append(f"rows={row_count}")
        if _trace_report_sample_value_present(matching_rows):
            parts.append(f"matching={matching_rows}")
        if event_kinds:
            parts.append(f"event_kinds={event_kinds}")
        if sha256:
            parts.append(f"sha256={sha256}")
        line = f"- Trace event artifact: {label}"
        if parts:
            line += " " + " ".join(parts)
        lines.append(line)
    for sample in summary.get("provider_payload_boundary_samples", [])[:3]:
        if not isinstance(sample, Mapping):
            continue
        provider_id = sample.get("provider_id")
        payload_lane = sample.get("payload_lane")
        status = sample.get("capture_status")
        provider_artifact_count = sample.get(
            "matching_provider_artifact_count"
        ) or sample.get("artifact_count")
        artifact_kind_count = sample.get("artifact_kind_recorded_count")
        artifact_backend_count = sample.get("artifact_kind_recorded_backend_count")
        artifact_backend_ids = sample.get("artifact_kind_recorded_backend_ids")
        report_section = sample.get("report_section")
        producer_status = sample.get("producer_status")
        producer_contract = sample.get("producer_contract")
        payload_policy = sample.get("payload_record_policy")
        payload_sensitivity = sample.get("payload_sensitivity")
        boundary = sample.get("claim_boundary")
        next_action = sample.get("next_action")
        if provider_id and payload_lane:
            parts = [f"status={status}" if status else ""]
            if provider_artifact_count is not None:
                parts.append(f"provider_artifacts={provider_artifact_count}")
            if artifact_kind_count is not None:
                parts.append(f"same_kind_artifacts={artifact_kind_count}")
            if artifact_backend_ids:
                parts.append(f"same_kind_backends={artifact_backend_ids}")
            elif artifact_backend_count is not None:
                parts.append(f"same_kind_backend_count={artifact_backend_count}")
            if report_section:
                parts.append(f"section={report_section}")
            if producer_status:
                parts.append(f"producer={producer_status}")
            if producer_contract:
                parts.append(f"contract={producer_contract}")
            if payload_policy:
                parts.append(f"policy={payload_policy}")
            if payload_sensitivity:
                parts.append(f"sensitivity={payload_sensitivity}")
            if boundary:
                parts.append(f"boundary={boundary}")
            line = f"- Provider payload boundary: {provider_id}/{payload_lane}"
            if any(parts):
                line += " " + " ".join(part for part in parts if part)
            lines.append(line)
        if next_action:
            lines.append(f"  Next action: `{next_action}`")
    for sample in summary.get("provider_payload_boundary_route_only_samples", [])[:3]:
        if not isinstance(sample, Mapping):
            continue
        provider_id = sample.get("provider_id")
        payload_lane = sample.get("payload_lane")
        status = sample.get("capture_status")
        capture_capability = sample.get("capture_capability")
        artifact_kind = sample.get("artifact_kind")
        capture_control = sample.get("capture_control")
        provider_artifact_count = sample.get(
            "matching_provider_artifact_count"
        ) or sample.get("artifact_count")
        artifact_kind_count = sample.get("artifact_kind_recorded_count")
        artifact_backend_ids = sample.get("artifact_kind_recorded_backend_ids")
        report_section = sample.get("report_section")
        producer_status = sample.get("producer_status")
        producer_contract = sample.get("producer_contract")
        payload_policy = sample.get("payload_record_policy")
        payload_sensitivity = sample.get("payload_sensitivity")
        boundary = sample.get("claim_boundary")
        next_action = sample.get("next_action")
        if provider_id and payload_lane:
            parts = [f"status={status}" if status else ""]
            if capture_capability:
                parts.append(f"capability={capture_capability}")
            if artifact_kind:
                parts.append(f"artifact={artifact_kind}")
            if capture_control:
                parts.append(f"control={capture_control}")
            if provider_artifact_count is not None:
                parts.append(f"provider_artifacts={provider_artifact_count}")
            if artifact_kind_count is not None:
                parts.append(f"same_kind_artifacts={artifact_kind_count}")
            if artifact_backend_ids:
                parts.append(f"same_kind_backends={artifact_backend_ids}")
            if report_section:
                parts.append(f"section={report_section}")
            if producer_status:
                parts.append(f"producer={producer_status}")
            if producer_contract:
                parts.append(f"contract={producer_contract}")
            if payload_policy:
                parts.append(f"policy={payload_policy}")
            if payload_sensitivity:
                parts.append(f"sensitivity={payload_sensitivity}")
            if boundary:
                parts.append(f"boundary={boundary}")
            line = (
                f"- Route-only provider payload boundary: {provider_id}/{payload_lane}"
            )
            if any(parts):
                line += " " + " ".join(part for part in parts if part)
            lines.append(line)
        if next_action:
            lines.append(f"  Next action: `{next_action}`")
    for sample in summary.get("provider_payload_boundary_recorded_samples", [])[:3]:
        if not isinstance(sample, Mapping):
            continue
        provider_id = sample.get("provider_id")
        payload_lane = sample.get("payload_lane")
        status = sample.get("capture_status")
        capture_capability = sample.get("capture_capability")
        artifact_kind = sample.get("artifact_kind")
        capture_control = sample.get("capture_control")
        provider_artifact_count = sample.get(
            "matching_provider_artifact_count"
        ) or sample.get("artifact_count")
        artifact_kind_count = sample.get("artifact_kind_recorded_count")
        artifact_backend_ids = sample.get("artifact_kind_recorded_backend_ids")
        report_section = sample.get("report_section")
        producer_status = sample.get("producer_status")
        producer_contract = sample.get("producer_contract")
        payload_policy = sample.get("payload_record_policy")
        payload_sensitivity = sample.get("payload_sensitivity")
        boundary = sample.get("claim_boundary")
        next_action = sample.get("next_action")
        if provider_id and payload_lane:
            parts = [f"status={status}" if status else ""]
            if capture_capability:
                parts.append(f"capability={capture_capability}")
            if artifact_kind:
                parts.append(f"artifact={artifact_kind}")
            if capture_control:
                parts.append(f"control={capture_control}")
            if provider_artifact_count is not None:
                parts.append(f"provider_artifacts={provider_artifact_count}")
            if artifact_kind_count is not None:
                parts.append(f"same_kind_artifacts={artifact_kind_count}")
            if artifact_backend_ids:
                parts.append(f"same_kind_backends={artifact_backend_ids}")
            if report_section:
                parts.append(f"section={report_section}")
            if producer_status:
                parts.append(f"producer={producer_status}")
            if producer_contract:
                parts.append(f"contract={producer_contract}")
            if payload_policy:
                parts.append(f"policy={payload_policy}")
            if payload_sensitivity:
                parts.append(f"sensitivity={payload_sensitivity}")
            if boundary:
                parts.append(f"boundary={boundary}")
            line = f"- Recorded provider payload boundary: {provider_id}/{payload_lane}"
            if any(parts):
                line += " " + " ".join(part for part in parts if part)
            lines.append(line)
        if next_action:
            lines.append(f"  Next action: `{next_action}`")
    for sample in summary.get("backend_event_artifact_samples", [])[:3]:
        if not isinstance(sample, Mapping):
            continue
        path = sample.get("path")
        index = sample.get("index")
        status = sample.get("status")
        row_count = sample.get("row_count")
        matching_rows = sample.get("matching_trace_run_id_rows")
        event_kinds = sample.get("event_kinds")
        sha256 = sample.get("sha256")
        label = path or index or "unknown"
        parts = []
        if status:
            parts.append(f"status={status}")
        if _trace_report_sample_value_present(row_count):
            parts.append(f"rows={row_count}")
        if _trace_report_sample_value_present(matching_rows):
            parts.append(f"matching={matching_rows}")
        if event_kinds:
            parts.append(f"event_kinds={event_kinds}")
        if sha256:
            parts.append(f"sha256={sha256}")
        line = f"- Backend event artifact: {label}"
        if parts:
            line += " " + " ".join(parts)
        lines.append(line)
    for sample in summary.get("backend_event_samples", [])[:3]:
        if not isinstance(sample, Mapping):
            continue
        backend_id = sample.get("backend_id")
        event_kind = sample.get("event_kind")
        request_id = sample.get("request_id")
        generation_id = sample.get("generation_id")
        targetplan_op_id = sample.get("targetplan_op_id")
        artifact = sample.get("artifact")
        metadata_keys = sample.get("metadata_keys")
        message = sample.get("message")
        label = backend_id or event_kind or "unknown"
        parts = []
        if event_kind:
            parts.append(f"event={event_kind}")
        if _trace_report_sample_value_present(request_id):
            parts.append(f"request={request_id}")
        if _trace_report_sample_value_present(generation_id):
            parts.append(f"generation={generation_id}")
        if targetplan_op_id:
            parts.append(f"op={targetplan_op_id}")
        if artifact:
            parts.append(f"artifact={artifact}")
        if metadata_keys:
            parts.append(f"metadata_keys={metadata_keys}")
        if message:
            parts.append(f"message={message}")
        line = f"- Backend event row: {label}"
        if parts:
            line += " " + " ".join(parts)
        lines.append(line)
    for sample in summary.get("backend_provider_boundary_samples", [])[:3]:
        if not isinstance(sample, Mapping):
            continue
        backend_id = sample.get("backend_id")
        event_kind = sample.get("event_kind")
        provider_stage = sample.get("provider_stage")
        boundary_status = sample.get("boundary_status")
        root_stage = sample.get("root_cause_stage")
        root_cause = sample.get("root_cause")
        request_id = sample.get("request_id")
        generation_id = sample.get("generation_id")
        targetplan_op_id = sample.get("targetplan_op_id")
        failure_reason = sample.get("failure_reason")
        message = sample.get("message")
        target_plan_status = sample.get("target_plan_validation_status")
        runtime_binding_status = sample.get("runtime_binding_status")
        hardware_gate_status = sample.get("hardware_gate_status")
        device_binding_status = sample.get("device_binding_status")
        weight_policy_status = sample.get("weight_policy_status")
        label = backend_id or event_kind or "unknown"
        parts = []
        if event_kind:
            parts.append(f"event={event_kind}")
        if provider_stage:
            parts.append(f"stage={provider_stage}")
        if boundary_status:
            parts.append(f"status={boundary_status}")
        if root_stage:
            parts.append(f"root_stage={root_stage}")
        if root_cause:
            parts.append(f"root={root_cause}")
        if _trace_report_sample_value_present(request_id):
            parts.append(f"request={request_id}")
        if _trace_report_sample_value_present(generation_id):
            parts.append(f"generation={generation_id}")
        if targetplan_op_id:
            parts.append(f"op={targetplan_op_id}")
        if target_plan_status:
            parts.append(f"target_plan={target_plan_status}")
        if runtime_binding_status:
            parts.append(f"runtime_binding={runtime_binding_status}")
        if hardware_gate_status:
            parts.append(f"hardware_gate={hardware_gate_status}")
        if device_binding_status:
            parts.append(f"device_binding={device_binding_status}")
        if weight_policy_status:
            parts.append(f"weight_policy={weight_policy_status}")
        if failure_reason:
            parts.append(f"failure={failure_reason}")
        if message:
            parts.append(f"message={message}")
        line = f"- Backend provider boundary: {label}"
        if parts:
            line += " " + " ".join(parts)
        lines.append(line)
    for sample in summary.get("backend_fail_closed_root_cause_samples", [])[:3]:
        if not isinstance(sample, Mapping):
            continue
        backend_id = sample.get("backend_id")
        provider_stage = sample.get("provider_stage")
        root_stage = sample.get("root_cause_stage")
        root_cause = sample.get("root_cause")
        event_kind = sample.get("event_kind")
        failure_count = sample.get("failure_count")
        example_request_id = sample.get("example_request_id")
        example_generation_id = sample.get("example_generation_id")
        example_op_id = sample.get("example_targetplan_op_id")
        example_failure = sample.get("example_failure_reason")
        label = backend_id or root_stage or "unknown"
        parts = []
        if provider_stage:
            parts.append(f"stage={provider_stage}")
        if root_stage:
            parts.append(f"root_stage={root_stage}")
        if root_cause:
            parts.append(f"root={root_cause}")
        if event_kind:
            parts.append(f"event={event_kind}")
        if _trace_report_sample_value_present(failure_count):
            parts.append(f"count={failure_count}")
        if _trace_report_sample_value_present(example_request_id):
            parts.append(f"request={example_request_id}")
        if _trace_report_sample_value_present(example_generation_id):
            parts.append(f"generation={example_generation_id}")
        if example_op_id:
            parts.append(f"op={example_op_id}")
        if example_failure:
            parts.append(f"failure={example_failure}")
        line = f"- Backend fail-closed root cause: {label}"
        if parts:
            line += " " + " ".join(parts)
        lines.append(line)
    for sample in summary.get("debug_payload_artifact_summary_samples", [])[:3]:
        if not isinstance(sample, Mapping):
            continue
        artifact_kind = sample.get("artifact_kind")
        status = sample.get("payload_summary_status")
        row_count = sample.get("row_count")
        byte_count = sample.get("byte_count")
        sensitivity = sample.get("sensitivity")
        token_window = sample.get("token_window")
        compile_features = sample.get("compile_features")
        report_section = sample.get("report_section")
        payload_boundary = sample.get("debug_payload_boundary")
        claim_boundary = sample.get("claim_boundary")
        if artifact_kind:
            parts = [f"status={status}" if status else ""]
            if row_count:
                parts.append(f"rows={row_count}")
            if byte_count:
                parts.append(f"bytes={byte_count}")
            if sensitivity:
                parts.append(f"sensitivity={sensitivity}")
            if token_window:
                parts.append(f"token_window={token_window}")
            if compile_features:
                parts.append(f"features={compile_features}")
            if report_section:
                parts.append(f"section={report_section}")
            if payload_boundary:
                parts.append(f"payload_boundary={payload_boundary}")
            if claim_boundary:
                parts.append(f"claim_boundary={claim_boundary}")
            line = f"- Debug payload artifact: {artifact_kind}"
            if any(parts):
                line += " " + " ".join(part for part in parts if part)
            lines.append(line)
    for sample in summary.get("planning_decision_sidecar_samples", [])[:3]:
        if not isinstance(sample, Mapping):
            continue
        row_kind = sample.get("row_kind")
        status = sample.get("status")
        frontend = sample.get("frontend")
        target_backend = sample.get("target_backend")
        selection_source = sample.get("selection_source")
        process_kind = sample.get("process_kind")
        logical_command = sample.get("logical_command")
        exit_code = sample.get("exit_code")
        duration_us = sample.get("duration_us")
        artifact_kind = sample.get("artifact_kind")
        artifact_sha = sample.get("artifact_sha256")
        planning_phase = sample.get("planning_phase")
        event_name = sample.get("event_name")
        targetplan_ops = sample.get("targetplan_op_count")
        claim_boundary = sample.get("claim_boundary")
        label = row_kind or planning_phase or event_name or "unknown"
        parts = []
        if status:
            parts.append(f"status={status}")
        if frontend:
            parts.append(f"frontend={frontend}")
        if target_backend:
            parts.append(f"target_backend={target_backend}")
        if selection_source:
            parts.append(f"selection={selection_source}")
        if process_kind:
            parts.append(f"process={process_kind}")
        if logical_command:
            parts.append(f"command={logical_command}")
        if _trace_report_sample_value_present(exit_code):
            parts.append(f"exit_code={exit_code}")
        if _trace_report_sample_value_present(duration_us):
            parts.append(f"duration_us={duration_us}")
        if artifact_kind:
            parts.append(f"artifact={artifact_kind}")
        if artifact_sha:
            parts.append(f"artifact_sha256={artifact_sha}")
        if planning_phase:
            parts.append(f"phase={planning_phase}")
        if _trace_report_sample_value_present(targetplan_ops):
            parts.append(f"targetplan_ops={targetplan_ops}")
        if claim_boundary:
            parts.append(f"boundary={claim_boundary}")
        line = f"- Planning decision sidecar: row={label}"
        if parts:
            line += " " + " ".join(parts)
        lines.append(line)
    for sample in summary.get("token_quality_sidecar_samples", [])[:3]:
        if not isinstance(sample, Mapping):
            continue
        request_id = sample.get("request_id")
        generation_id = sample.get("generation_id")
        token_index = sample.get("token_index")
        selected_token = sample.get("selected_token_id")
        topk_count = sample.get("topk_count")
        finish_reason = sample.get("finish_reason")
        eos_policy = sample.get("eos_policy")
        temperature = sample.get("temperature")
        top_p = sample.get("top_p")
        top_k = sample.get("top_k")
        oracle_reference = sample.get("oracle_reference")
        label = (
            request_id
            if _trace_report_sample_value_present(request_id)
            else generation_id
            if _trace_report_sample_value_present(generation_id)
            else "unknown"
        )
        parts = []
        if _trace_report_sample_value_present(generation_id):
            parts.append(f"generation={generation_id}")
        if _trace_report_sample_value_present(token_index):
            parts.append(f"token_index={token_index}")
        if _trace_report_sample_value_present(selected_token):
            parts.append(f"selected={selected_token}")
        if _trace_report_sample_value_present(topk_count):
            parts.append(f"topk_count={topk_count}")
        if finish_reason:
            parts.append(f"finish={finish_reason}")
        if eos_policy:
            parts.append(f"eos={eos_policy}")
        if _trace_report_sample_value_present(temperature):
            parts.append(f"temperature={temperature}")
        if _trace_report_sample_value_present(top_p):
            parts.append(f"top_p={top_p}")
        if _trace_report_sample_value_present(top_k):
            parts.append(f"top_k={top_k}")
        if oracle_reference:
            parts.append(f"oracle_reference={oracle_reference}")
        line = f"- Token quality sidecar: request={label}"
        if parts:
            line += " " + " ".join(parts)
        lines.append(line)
    for sample in summary.get("topk_token_sidecar_samples", [])[:3]:
        if not isinstance(sample, Mapping):
            continue
        request_id = sample.get("request_id")
        generation_id = sample.get("generation_id")
        token_index = sample.get("token_index")
        selected_token = sample.get("selected_token_id")
        candidate_token = sample.get("candidate_token_id")
        candidate_rank = sample.get("candidate_rank")
        candidate_score = sample.get("candidate_score")
        score_kind = sample.get("score_kind")
        candidate_status = sample.get("selected_candidate_status")
        oracle_reference = sample.get("oracle_reference")
        claim_boundary = sample.get("claim_boundary")
        label = (
            request_id
            if _trace_report_sample_value_present(request_id)
            else generation_id
            if _trace_report_sample_value_present(generation_id)
            else "unknown"
        )
        parts = []
        if _trace_report_sample_value_present(generation_id):
            parts.append(f"generation={generation_id}")
        if _trace_report_sample_value_present(token_index):
            parts.append(f"token_index={token_index}")
        if _trace_report_sample_value_present(selected_token):
            parts.append(f"selected={selected_token}")
        if _trace_report_sample_value_present(candidate_token):
            parts.append(f"candidate={candidate_token}")
        if _trace_report_sample_value_present(candidate_rank):
            parts.append(f"rank={candidate_rank}")
        if _trace_report_sample_value_present(candidate_score):
            parts.append(f"score={candidate_score}")
        if score_kind:
            parts.append(f"score_kind={score_kind}")
        if candidate_status:
            parts.append(f"candidate_status={candidate_status}")
        if oracle_reference:
            parts.append(f"oracle_reference={oracle_reference}")
        if claim_boundary:
            parts.append(f"boundary={claim_boundary}")
        line = f"- Top-K token sidecar: request={label}"
        if parts:
            line += " " + " ".join(parts)
        lines.append(line)
    for sample in summary.get("tensor_payload_sidecar_samples", [])[:3]:
        if not isinstance(sample, Mapping):
            continue
        request_id = sample.get("request_id")
        generation_id = sample.get("generation_id")
        token_index = sample.get("token_index")
        backend_id = sample.get("backend_id")
        payload_kind = sample.get("tensor_payload_kind")
        tensor_name = sample.get("tensor_name")
        tensor_role = sample.get("tensor_role")
        targetplan_op_id = sample.get("targetplan_op_id")
        targetplan_action = sample.get("targetplan_action")
        layer = sample.get("layer")
        shape = sample.get("shape")
        element_count = sample.get("element_count")
        digest = sample.get("digest_sha256")
        sample_count = sample.get("sample_value_count")
        sample_nan_count = sample.get("sample_nan_count")
        failure_reason = sample.get("failure_reason")
        label = (
            request_id
            if _trace_report_sample_value_present(request_id)
            else generation_id
            if _trace_report_sample_value_present(generation_id)
            else "unknown"
        )
        parts = []
        if _trace_report_sample_value_present(generation_id):
            parts.append(f"generation={generation_id}")
        if _trace_report_sample_value_present(token_index):
            parts.append(f"token_index={token_index}")
        if backend_id:
            parts.append(f"backend={backend_id}")
        if payload_kind:
            parts.append(f"kind={payload_kind}")
        if tensor_role:
            parts.append(f"role={tensor_role}")
        if targetplan_op_id:
            parts.append(f"op={targetplan_op_id}")
        if targetplan_action:
            parts.append(f"action={targetplan_action}")
        if _trace_report_sample_value_present(layer):
            parts.append(f"layer={layer}")
        if tensor_name:
            parts.append(f"tensor={tensor_name}")
        if shape:
            parts.append(f"shape={shape}")
        if _trace_report_sample_value_present(element_count):
            parts.append(f"elements={element_count}")
        if _trace_report_sample_value_present(sample_count):
            parts.append(f"samples={sample_count}")
        if _trace_report_sample_value_present(sample_nan_count):
            parts.append(f"sample_nan={sample_nan_count}")
        if digest:
            parts.append(f"digest_sha256={digest}")
        if failure_reason:
            parts.append(f"failure={failure_reason}")
        line = f"- Tensor payload sidecar: request={label}"
        if parts:
            line += " " + " ".join(parts)
        lines.append(line)
    for sample in summary.get("kv_payload_digest_sidecar_samples", [])[:3]:
        if not isinstance(sample, Mapping):
            continue
        request_id = sample.get("request_id")
        generation_id = sample.get("generation_id")
        token_index = sample.get("token_index")
        backend_id = sample.get("backend_id")
        tensor_name = sample.get("tensor_name")
        tensor_role = sample.get("tensor_role")
        targetplan_action = sample.get("targetplan_action")
        layer = sample.get("layer")
        element_count = sample.get("element_count")
        digest = sample.get("digest_sha256")
        sample_count = sample.get("sample_value_count")
        sample_min = sample.get("sample_min")
        sample_max = sample.get("sample_max")
        label = (
            request_id
            if _trace_report_sample_value_present(request_id)
            else generation_id
            if _trace_report_sample_value_present(generation_id)
            else "unknown"
        )
        parts = []
        if _trace_report_sample_value_present(generation_id):
            parts.append(f"generation={generation_id}")
        if _trace_report_sample_value_present(token_index):
            parts.append(f"token_index={token_index}")
        if backend_id:
            parts.append(f"backend={backend_id}")
        if tensor_role:
            parts.append(f"role={tensor_role}")
        if targetplan_action:
            parts.append(f"action={targetplan_action}")
        if _trace_report_sample_value_present(layer):
            parts.append(f"layer={layer}")
        if tensor_name:
            parts.append(f"tensor={tensor_name}")
        if _trace_report_sample_value_present(element_count):
            parts.append(f"elements={element_count}")
        if _trace_report_sample_value_present(sample_count):
            parts.append(f"samples={sample_count}")
        if _trace_report_sample_value_present(sample_min):
            parts.append(f"sample_min={sample_min}")
        if _trace_report_sample_value_present(sample_max):
            parts.append(f"sample_max={sample_max}")
        if digest:
            parts.append(f"digest_sha256={digest}")
        line = f"- K/V payload digest sidecar: request={label}"
        if parts:
            line += " " + " ".join(parts)
        lines.append(line)
    for sample in summary.get("logit_slice_sidecar_samples", [])[:3]:
        if not isinstance(sample, Mapping):
            continue
        request_id = sample.get("request_id")
        generation_id = sample.get("generation_id")
        token_index = sample.get("token_index")
        backend_id = sample.get("backend_id")
        targetplan_op_id = sample.get("targetplan_op_id")
        targetplan_action = sample.get("targetplan_action")
        layer = sample.get("layer")
        intrinsic = sample.get("intrinsic")
        tensor_name = sample.get("tensor_name")
        tensor_role = sample.get("tensor_role")
        element_count = sample.get("element_count")
        digest = sample.get("digest_sha256")
        sample_count = sample.get("sample_value_count")
        sample_nan_count = sample.get("sample_nan_count")
        failure_reason = sample.get("failure_reason")
        label = (
            request_id
            if _trace_report_sample_value_present(request_id)
            else generation_id
            if _trace_report_sample_value_present(generation_id)
            else "unknown"
        )
        parts = []
        if _trace_report_sample_value_present(generation_id):
            parts.append(f"generation={generation_id}")
        if _trace_report_sample_value_present(token_index):
            parts.append(f"token_index={token_index}")
        if backend_id:
            parts.append(f"backend={backend_id}")
        if targetplan_op_id:
            parts.append(f"op={targetplan_op_id}")
        if targetplan_action:
            parts.append(f"action={targetplan_action}")
        if _trace_report_sample_value_present(layer):
            parts.append(f"layer={layer}")
        if intrinsic:
            parts.append(f"intrinsic={intrinsic}")
        if tensor_role:
            parts.append(f"role={tensor_role}")
        if tensor_name:
            parts.append(f"tensor={tensor_name}")
        if _trace_report_sample_value_present(element_count):
            parts.append(f"elements={element_count}")
        if _trace_report_sample_value_present(sample_count):
            parts.append(f"samples={sample_count}")
        if _trace_report_sample_value_present(sample_nan_count):
            parts.append(f"sample_nan={sample_nan_count}")
        if digest:
            parts.append(f"digest_sha256={digest}")
        if failure_reason:
            parts.append(f"failure={failure_reason}")
        line = f"- Logit slice sidecar: request={label}"
        if parts:
            line += " " + " ".join(parts)
        lines.append(line)
    for sample in summary.get("activation_digest_sidecar_samples", [])[:3]:
        if not isinstance(sample, Mapping):
            continue
        request_id = sample.get("request_id")
        generation_id = sample.get("generation_id")
        token_index = sample.get("token_index")
        backend_id = sample.get("backend_id")
        targetplan_op_id = sample.get("targetplan_op_id")
        targetplan_action = sample.get("targetplan_action")
        layer = sample.get("layer")
        intrinsic = sample.get("intrinsic")
        tensor_name = sample.get("tensor_name")
        tensor_role = sample.get("tensor_role")
        element_count = sample.get("element_count")
        digest = sample.get("digest_sha256")
        sample_count = sample.get("sample_value_count")
        sample_min = sample.get("sample_min")
        sample_max = sample.get("sample_max")
        failure_reason = sample.get("failure_reason")
        label = (
            request_id
            if _trace_report_sample_value_present(request_id)
            else generation_id
            if _trace_report_sample_value_present(generation_id)
            else "unknown"
        )
        parts = []
        if _trace_report_sample_value_present(generation_id):
            parts.append(f"generation={generation_id}")
        if _trace_report_sample_value_present(token_index):
            parts.append(f"token_index={token_index}")
        if backend_id:
            parts.append(f"backend={backend_id}")
        if targetplan_op_id:
            parts.append(f"op={targetplan_op_id}")
        if targetplan_action:
            parts.append(f"action={targetplan_action}")
        if _trace_report_sample_value_present(layer):
            parts.append(f"layer={layer}")
        if intrinsic:
            parts.append(f"intrinsic={intrinsic}")
        if tensor_role:
            parts.append(f"role={tensor_role}")
        if tensor_name:
            parts.append(f"tensor={tensor_name}")
        if _trace_report_sample_value_present(element_count):
            parts.append(f"elements={element_count}")
        if _trace_report_sample_value_present(sample_count):
            parts.append(f"samples={sample_count}")
        if _trace_report_sample_value_present(sample_min):
            parts.append(f"sample_min={sample_min}")
        if _trace_report_sample_value_present(sample_max):
            parts.append(f"sample_max={sample_max}")
        if digest:
            parts.append(f"digest_sha256={digest}")
        if failure_reason:
            parts.append(f"failure={failure_reason}")
        line = f"- Activation digest sidecar: request={label}"
        if parts:
            line += " " + " ".join(parts)
        lines.append(line)
    for sample in summary.get("device_result_digest_sidecar_samples", [])[:3]:
        if not isinstance(sample, Mapping):
            continue
        request_id = sample.get("request_id")
        generation_id = sample.get("generation_id")
        token_index = sample.get("token_index")
        backend_id = sample.get("backend_id")
        targetplan_op_id = sample.get("targetplan_op_id")
        targetplan_action = sample.get("targetplan_action")
        layer = sample.get("layer")
        intrinsic = sample.get("intrinsic")
        tensor_name = sample.get("tensor_name")
        tensor_role = sample.get("tensor_role")
        shape = sample.get("shape")
        element_count = sample.get("element_count")
        digest = sample.get("digest_sha256")
        sample_count = sample.get("sample_value_count")
        sample_min = sample.get("sample_min")
        sample_max = sample.get("sample_max")
        sample_nan_count = sample.get("sample_nan_count")
        failure_reason = sample.get("failure_reason")
        label = (
            request_id
            if _trace_report_sample_value_present(request_id)
            else generation_id
            if _trace_report_sample_value_present(generation_id)
            else "unknown"
        )
        parts = []
        if _trace_report_sample_value_present(generation_id):
            parts.append(f"generation={generation_id}")
        if _trace_report_sample_value_present(token_index):
            parts.append(f"token_index={token_index}")
        if backend_id:
            parts.append(f"backend={backend_id}")
        if targetplan_op_id:
            parts.append(f"op={targetplan_op_id}")
        if targetplan_action:
            parts.append(f"action={targetplan_action}")
        if _trace_report_sample_value_present(layer):
            parts.append(f"layer={layer}")
        if intrinsic:
            parts.append(f"intrinsic={intrinsic}")
        if tensor_role:
            parts.append(f"role={tensor_role}")
        if tensor_name:
            parts.append(f"tensor={tensor_name}")
        if shape:
            parts.append(f"shape={shape}")
        if _trace_report_sample_value_present(element_count):
            parts.append(f"elements={element_count}")
        if _trace_report_sample_value_present(sample_count):
            parts.append(f"samples={sample_count}")
        if _trace_report_sample_value_present(sample_min):
            parts.append(f"sample_min={sample_min}")
        if _trace_report_sample_value_present(sample_max):
            parts.append(f"sample_max={sample_max}")
        if _trace_report_sample_value_present(sample_nan_count):
            parts.append(f"sample_nan={sample_nan_count}")
        if digest:
            parts.append(f"digest_sha256={digest}")
        if failure_reason:
            parts.append(f"failure={failure_reason}")
        line = f"- Device result digest sidecar: request={label}"
        if parts:
            line += " " + " ".join(parts)
        lines.append(line)
    for sample in summary.get("scheduler_packet_lineage_sidecar_samples", [])[:3]:
        if not isinstance(sample, Mapping):
            continue
        request_id = sample.get("request_id")
        generation_id = sample.get("generation_id")
        status = sample.get("status")
        executor_shape = sample.get("executor_shape")
        executor_status = sample.get("executor_status")
        attention_mode = sample.get("attention_mode")
        token_jobs = sample.get("token_job_count")
        runtime_tokens = sample.get("runtime_request_token_count")
        tokens_reused = sample.get("tokens_reused")
        visible_slots = sample.get("visible_token_slots")
        kv_context_rows = sample.get("kv_context_rows")
        kv_save_rows = sample.get("kv_save_rows")
        sparse_rows = sample.get("listener_sparse_rows")
        sparse_tokens = sample.get("listener_sparse_tokens")
        staging_status = sample.get("prior_host_gof_staging_status")
        dma_completions = sample.get("prior_host_gof_dma_completions")
        failure_reason = sample.get("failure_reason")
        label = (
            request_id
            if _trace_report_sample_value_present(request_id)
            else generation_id
            if _trace_report_sample_value_present(generation_id)
            else "unknown"
        )
        parts = []
        if _trace_report_sample_value_present(generation_id):
            parts.append(f"generation={generation_id}")
        if status:
            parts.append(f"status={status}")
        if executor_status:
            parts.append(f"executor={executor_status}")
        if executor_shape:
            parts.append(f"shape={executor_shape}")
        if attention_mode:
            parts.append(f"attention={attention_mode}")
        if _trace_report_sample_value_present(token_jobs):
            parts.append(f"token_jobs={token_jobs}")
        if _trace_report_sample_value_present(runtime_tokens):
            parts.append(f"runtime_tokens={runtime_tokens}")
        if _trace_report_sample_value_present(tokens_reused):
            parts.append(f"tokens_reused={tokens_reused}")
        if _trace_report_sample_value_present(visible_slots):
            parts.append(f"visible_slots={visible_slots}")
        if _trace_report_sample_value_present(kv_context_rows):
            parts.append(f"kv_context_rows={kv_context_rows}")
        if _trace_report_sample_value_present(kv_save_rows):
            parts.append(f"kv_save_rows={kv_save_rows}")
        if _trace_report_sample_value_present(sparse_rows):
            parts.append(f"sparse_rows={sparse_rows}")
        if _trace_report_sample_value_present(sparse_tokens):
            parts.append(f"sparse_tokens={sparse_tokens}")
        if staging_status:
            parts.append(f"staging={staging_status}")
        if _trace_report_sample_value_present(dma_completions):
            parts.append(f"dma_completions={dma_completions}")
        if failure_reason:
            parts.append(f"failure={failure_reason}")
        line = f"- Scheduler packet lineage: request={label}"
        if parts:
            line += " " + " ".join(parts)
        lines.append(line)
    for sample in summary.get("scheduler_kv_shard_lifecycle_sidecar_samples", [])[:3]:
        if not isinstance(sample, Mapping):
            continue
        request_id = sample.get("request_id")
        generation_id = sample.get("generation_id")
        status = sample.get("status")
        lifecycle_status = sample.get("kv_lifecycle_status")
        executor_status = sample.get("executor_status")
        attention_mode = sample.get("attention_mode")
        token_jobs = sample.get("token_job_count")
        kv_jobs = sample.get("kv_job_count")
        runtime_tokens = sample.get("runtime_request_token_count")
        visible_slots = sample.get("visible_token_slots")
        kv_context_rows = sample.get("kv_context_rows")
        kv_save_rows = sample.get("kv_save_rows")
        kv_pages = sample.get("kv_page_count")
        allocations = sample.get("hw_shard_allocation_requests")
        page_infos = sample.get("hw_gof_page_infos")
        staging_status = sample.get("prior_host_gof_staging_status")
        dma_completions = sample.get("prior_host_gof_dma_completions")
        failure_reason = sample.get("failure_reason")
        label = (
            request_id
            if _trace_report_sample_value_present(request_id)
            else generation_id
            if _trace_report_sample_value_present(generation_id)
            else "unknown"
        )
        parts = []
        if _trace_report_sample_value_present(generation_id):
            parts.append(f"generation={generation_id}")
        if status:
            parts.append(f"status={status}")
        if lifecycle_status:
            parts.append(f"lifecycle={lifecycle_status}")
        if executor_status:
            parts.append(f"executor={executor_status}")
        if attention_mode:
            parts.append(f"attention={attention_mode}")
        if _trace_report_sample_value_present(token_jobs):
            parts.append(f"token_jobs={token_jobs}")
        if _trace_report_sample_value_present(kv_jobs):
            parts.append(f"kv_jobs={kv_jobs}")
        if _trace_report_sample_value_present(runtime_tokens):
            parts.append(f"runtime_tokens={runtime_tokens}")
        if _trace_report_sample_value_present(visible_slots):
            parts.append(f"visible_slots={visible_slots}")
        if _trace_report_sample_value_present(kv_context_rows):
            parts.append(f"kv_context_rows={kv_context_rows}")
        if _trace_report_sample_value_present(kv_save_rows):
            parts.append(f"kv_save_rows={kv_save_rows}")
        if _trace_report_sample_value_present(kv_pages):
            parts.append(f"kv_pages={kv_pages}")
        if _trace_report_sample_value_present(allocations):
            parts.append(f"allocations={allocations}")
        if _trace_report_sample_value_present(page_infos):
            parts.append(f"page_infos={page_infos}")
        if staging_status:
            parts.append(f"staging={staging_status}")
        if _trace_report_sample_value_present(dma_completions):
            parts.append(f"dma_completions={dma_completions}")
        if failure_reason:
            parts.append(f"failure={failure_reason}")
        line = f"- Scheduler K/V lifecycle: request={label}"
        if parts:
            line += " " + " ".join(parts)
        lines.append(line)
    for sample in summary.get("scheduler_listener_sparse_logit_sidecar_samples", [])[
        :3
    ]:
        if not isinstance(sample, Mapping):
            continue
        request_id = sample.get("request_id")
        generation_id = sample.get("generation_id")
        status = sample.get("status")
        listener_status = sample.get("listener_sparse_status")
        executor_shape = sample.get("executor_shape")
        executor_status = sample.get("executor_status")
        attention_mode = sample.get("attention_mode")
        listener_rows = sample.get("listener_sparse_rows")
        listener_tokens = sample.get("listener_sparse_tokens")
        sparse_topk_rows = sample.get("sparse_topk_rows")
        sparse_topk_tokens = sample.get("sparse_topk_token_count")
        token_jobs = sample.get("token_job_count")
        minibatches = sample.get("minibatch_count")
        runtime_tokens = sample.get("runtime_request_token_count")
        tokens_reused = sample.get("tokens_reused")
        failure_reason = sample.get("failure_reason")
        label = (
            request_id
            if _trace_report_sample_value_present(request_id)
            else generation_id
            if _trace_report_sample_value_present(generation_id)
            else "unknown"
        )
        parts = []
        if _trace_report_sample_value_present(generation_id):
            parts.append(f"generation={generation_id}")
        if status:
            parts.append(f"status={status}")
        if listener_status:
            parts.append(f"listener={listener_status}")
        if executor_status:
            parts.append(f"executor={executor_status}")
        if executor_shape:
            parts.append(f"shape={executor_shape}")
        if attention_mode:
            parts.append(f"attention={attention_mode}")
        if _trace_report_sample_value_present(listener_rows):
            parts.append(f"listener_rows={listener_rows}")
        if _trace_report_sample_value_present(listener_tokens):
            parts.append(f"listener_tokens={listener_tokens}")
        if _trace_report_sample_value_present(sparse_topk_rows):
            parts.append(f"sparse_topk_rows={sparse_topk_rows}")
        if _trace_report_sample_value_present(sparse_topk_tokens):
            parts.append(f"sparse_topk_tokens={sparse_topk_tokens}")
        if _trace_report_sample_value_present(token_jobs):
            parts.append(f"token_jobs={token_jobs}")
        if _trace_report_sample_value_present(minibatches):
            parts.append(f"minibatches={minibatches}")
        if _trace_report_sample_value_present(runtime_tokens):
            parts.append(f"runtime_tokens={runtime_tokens}")
        if _trace_report_sample_value_present(tokens_reused):
            parts.append(f"tokens_reused={tokens_reused}")
        if failure_reason:
            parts.append(f"failure={failure_reason}")
        line = f"- Scheduler sparse listener: request={label}"
        if parts:
            line += " " + " ".join(parts)
        lines.append(line)
    for sample in summary.get("device_dma_lifecycle_sidecar_samples", [])[:3]:
        if not isinstance(sample, Mapping):
            continue
        request_id = sample.get("request_id")
        generation_id = sample.get("generation_id")
        status = sample.get("status")
        backend_id = sample.get("backend_id")
        targetplan_op_id = sample.get("targetplan_op_id")
        targetplan_action = sample.get("targetplan_action")
        device_stage = sample.get("device_stage")
        queue_id = sample.get("queue_id")
        device_index = sample.get("device_index")
        card_bus = sample.get("card_bus")
        dma_direction = sample.get("dma_direction")
        descriptor_count = sample.get("descriptor_count")
        byte_count = sample.get("byte_count")
        counter_name = sample.get("counter_name")
        counter_delta = sample.get("counter_value_delta")
        cacheblock_shard = sample.get("cacheblock_dma_shard_id")
        cacheblock_gof = sample.get("cacheblock_dma_gof_start_in_shard")
        cacheblock_bytes = sample.get("cacheblock_dma_transfer_byte_count")
        queue_depth_before = sample.get("queue_depth_before")
        queue_depth_after = sample.get("queue_depth_after")
        failure_reason = sample.get("failure_reason")
        label = (
            request_id
            if _trace_report_sample_value_present(request_id)
            else generation_id
            if _trace_report_sample_value_present(generation_id)
            else "unknown"
        )
        parts = []
        if _trace_report_sample_value_present(generation_id):
            parts.append(f"generation={generation_id}")
        if status:
            parts.append(f"status={status}")
        if backend_id:
            parts.append(f"backend={backend_id}")
        if targetplan_op_id:
            parts.append(f"op={targetplan_op_id}")
        if targetplan_action:
            parts.append(f"action={targetplan_action}")
        if device_stage:
            parts.append(f"stage={device_stage}")
        if queue_id:
            parts.append(f"queue={queue_id}")
        if _trace_report_sample_value_present(device_index):
            parts.append(f"device_index={device_index}")
        if card_bus:
            parts.append(f"card_bus={card_bus}")
        if dma_direction:
            parts.append(f"direction={dma_direction}")
        if _trace_report_sample_value_present(descriptor_count):
            parts.append(f"descriptors={descriptor_count}")
        if _trace_report_sample_value_present(byte_count):
            parts.append(f"bytes={byte_count}")
        if counter_name:
            parts.append(f"counter={counter_name}")
        if _trace_report_sample_value_present(counter_delta):
            parts.append(f"counter_delta={counter_delta}")
        if _trace_report_sample_value_present(cacheblock_shard):
            parts.append(f"cacheblock_shard={cacheblock_shard}")
        if _trace_report_sample_value_present(cacheblock_gof):
            parts.append(f"cacheblock_gof_start={cacheblock_gof}")
        if _trace_report_sample_value_present(cacheblock_bytes):
            parts.append(f"cacheblock_bytes={cacheblock_bytes}")
        if _trace_report_sample_value_present(queue_depth_before):
            parts.append(f"queue_before={queue_depth_before}")
        if _trace_report_sample_value_present(queue_depth_after):
            parts.append(f"queue_after={queue_depth_after}")
        if failure_reason:
            parts.append(f"failure={failure_reason}")
        line = f"- Device DMA lifecycle: request={label}"
        if parts:
            line += " " + " ".join(parts)
        lines.append(line)
    for sample in summary.get("attention_page_trace_sidecar_samples", [])[:3]:
        if not isinstance(sample, Mapping):
            continue
        request_id = sample.get("request_id")
        generation_id = sample.get("generation_id")
        status = sample.get("status")
        backend_id = sample.get("backend_id")
        targetplan_op_id = sample.get("targetplan_op_id")
        targetplan_action = sample.get("targetplan_action")
        layer = sample.get("layer")
        head = sample.get("head")
        kv_head = sample.get("kv_head")
        attention_row_index = sample.get("attention_row_index")
        batch = sample.get("batch")
        visible_tokens = sample.get("visible_tokens")
        page_start = sample.get("page_start")
        page_count = sample.get("page_count")
        page_v_count = sample.get("page_v_count")
        scaled_score_count = sample.get("scaled_score_count")
        exp_score_count = sample.get("exp_score_count")
        v_star_count = sample.get("v_star_count")
        m_star = sample.get("m_star")
        s_star = sample.get("s_star")
        was_valid = sample.get("was_valid")
        failure_reason = sample.get("failure_reason")
        label = (
            request_id
            if _trace_report_sample_value_present(request_id)
            else generation_id
            if _trace_report_sample_value_present(generation_id)
            else "unknown"
        )
        parts = []
        if _trace_report_sample_value_present(generation_id):
            parts.append(f"generation={generation_id}")
        if status:
            parts.append(f"status={status}")
        if backend_id:
            parts.append(f"backend={backend_id}")
        if targetplan_op_id:
            parts.append(f"op={targetplan_op_id}")
        if targetplan_action:
            parts.append(f"action={targetplan_action}")
        if _trace_report_sample_value_present(layer):
            parts.append(f"layer={layer}")
        if _trace_report_sample_value_present(head):
            parts.append(f"head={head}")
        if _trace_report_sample_value_present(kv_head):
            parts.append(f"kv_head={kv_head}")
        if _trace_report_sample_value_present(attention_row_index):
            parts.append(f"attention_row={attention_row_index}")
        if _trace_report_sample_value_present(batch):
            parts.append(f"batch={batch}")
        if _trace_report_sample_value_present(visible_tokens):
            parts.append(f"visible_tokens={visible_tokens}")
        if _trace_report_sample_value_present(page_start):
            parts.append(f"page_start={page_start}")
        if _trace_report_sample_value_present(page_count):
            parts.append(f"pages={page_count}")
        if _trace_report_sample_value_present(page_v_count):
            parts.append(f"page_v={page_v_count}")
        if _trace_report_sample_value_present(scaled_score_count):
            parts.append(f"scaled_scores={scaled_score_count}")
        if _trace_report_sample_value_present(exp_score_count):
            parts.append(f"exp_scores={exp_score_count}")
        if _trace_report_sample_value_present(v_star_count):
            parts.append(f"v_star={v_star_count}")
        if _trace_report_sample_value_present(m_star):
            parts.append(f"m_star={m_star}")
        if _trace_report_sample_value_present(s_star):
            parts.append(f"s_star={s_star}")
        if _trace_report_sample_value_present(was_valid):
            parts.append(f"was_valid={was_valid}")
        if failure_reason:
            parts.append(f"failure={failure_reason}")
        line = f"- Attention page trace: request={label}"
        if parts:
            line += " " + " ".join(parts)
        lines.append(line)
    for sample in summary.get("token_quality_summary_samples", [])[:3]:
        if not isinstance(sample, Mapping):
            continue
        request_id = sample.get("request_id")
        generation_id = sample.get("generation_id")
        token_index = sample.get("token_index")
        selected_token = sample.get("selected_token_id")
        topk_status = sample.get("selected_topk_status")
        score_kind = sample.get("score_kind")
        top1_token = sample.get("top1_token_id")
        top1_margin = sample.get("top1_margin")
        tokens_reused = sample.get("tokens_reused")
        runtime_count = sample.get("runtime_request_token_count")
        oracle_reference = sample.get("oracle_reference")
        claim_boundary = sample.get("claim_boundary")
        label = (
            request_id
            if _trace_report_sample_value_present(request_id)
            else generation_id
            if _trace_report_sample_value_present(generation_id)
            else "unknown"
        )
        parts = []
        if _trace_report_sample_value_present(generation_id):
            parts.append(f"generation={generation_id}")
        if _trace_report_sample_value_present(token_index):
            parts.append(f"token_index={token_index}")
        if _trace_report_sample_value_present(selected_token):
            parts.append(f"selected={selected_token}")
        if topk_status:
            parts.append(f"topk={topk_status}")
        if score_kind:
            parts.append(f"score_kind={score_kind}")
        if _trace_report_sample_value_present(top1_token):
            parts.append(f"top1={top1_token}")
        if _trace_report_sample_value_present(top1_margin):
            parts.append(f"top1_margin={top1_margin}")
        if _trace_report_sample_value_present(tokens_reused):
            parts.append(f"tokens_reused={tokens_reused}")
        if _trace_report_sample_value_present(runtime_count):
            parts.append(f"runtime_tokens={runtime_count}")
        if oracle_reference:
            parts.append(f"oracle_reference={oracle_reference}")
        if claim_boundary:
            parts.append(f"boundary={claim_boundary}")
        line = f"- Token quality summary: request={label}"
        if parts:
            line += " " + " ".join(parts)
        lines.append(line)
    for sample in summary.get("oracle_reference_summary_samples", [])[:3]:
        if not isinstance(sample, Mapping):
            continue
        request_id = sample.get("request_id")
        generation_id = sample.get("generation_id")
        token_index = sample.get("token_index")
        selected_token = sample.get("selected_token_id")
        role = sample.get("oracle_reference_role")
        source = sample.get("expected_oracle_source")
        oracle_status = sample.get("oracle_reference_status")
        correctness_status = sample.get("correctness_claim_status")
        sut_classification = sample.get("sut_classification")
        oracle_sha = sample.get("hf_cpu_oracle_sha256")
        claim_boundary = sample.get("claim_boundary")
        label = (
            request_id
            if _trace_report_sample_value_present(request_id)
            else generation_id
            if _trace_report_sample_value_present(generation_id)
            else "unknown"
        )
        parts = []
        if _trace_report_sample_value_present(generation_id):
            parts.append(f"generation={generation_id}")
        if _trace_report_sample_value_present(token_index):
            parts.append(f"token_index={token_index}")
        if _trace_report_sample_value_present(selected_token):
            parts.append(f"selected={selected_token}")
        if role:
            parts.append(f"role={role}")
        if source:
            parts.append(f"source={source}")
        if oracle_status:
            parts.append(f"status={oracle_status}")
        if correctness_status:
            parts.append(f"correctness={correctness_status}")
        if sut_classification:
            parts.append(f"sut={sut_classification}")
        if oracle_sha:
            parts.append(f"oracle_sha256={oracle_sha}")
        if claim_boundary:
            parts.append(f"boundary={claim_boundary}")
        line = f"- Oracle reference summary: request={label}"
        if parts:
            line += " " + " ".join(parts)
        lines.append(line)
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
    for sample in summary.get("introspection_artifact_samples", [])[:16]:
        if not isinstance(sample, Mapping):
            continue
        kind = sample.get("kind")
        artifact_kind = sample.get("artifact_kind")
        path = sample.get("path")
        sha256 = sample.get("sha256")
        status = sample.get("status")
        row_count = sample.get("row_count")
        byte_count = sample.get("byte_count")
        token_window = sample.get("token_window")
        sensitivity = sample.get("sensitivity")
        compile_features = sample.get("compile_features")
        label = kind or artifact_kind or path or "unknown"
        parts = []
        if artifact_kind:
            parts.append(f"artifact={artifact_kind}")
        if path:
            parts.append(f"path={path}")
        if sha256:
            parts.append(f"sha256={sha256}")
        if status:
            parts.append(f"status={status}")
        if _trace_report_sample_value_present(row_count):
            parts.append(f"rows={row_count}")
        if _trace_report_sample_value_present(byte_count):
            parts.append(f"bytes={byte_count}")
        if token_window:
            parts.append(f"window={token_window}")
        if sensitivity:
            parts.append(f"sensitivity={sensitivity}")
        if compile_features:
            parts.append(f"features={compile_features}")
        line = f"- Introspection raw artifact: {label}"
        if parts:
            line += " " + " ".join(parts)
        lines.append(line)
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
    for sample in summary.get("introspection_section_inventory_samples", [])[:16]:
        if not isinstance(sample, Mapping):
            continue
        json_section = sample.get("json_section")
        status = sample.get("section_status")
        capability = sample.get("capture_capability")
        artifact_kind = sample.get("artifact_kind")
        artifact_count = sample.get("artifact_count")
        boundary = sample.get("claim_boundary")
        if json_section:
            parts = [f"status={status}" if status else ""]
            if capability:
                parts.append(f"capability={capability}")
            if artifact_kind:
                parts.append(f"artifact={artifact_kind}")
            if _trace_report_sample_value_present(artifact_count):
                parts.append(f"artifacts={artifact_count}")
            if boundary:
                parts.append(f"boundary={boundary}")
            line = f"- Introspection section: {json_section}"
            if any(parts):
                line += " " + " ".join(part for part in parts if part)
            lines.append(line)
    for sample in summary.get("timeline_query_summary_samples", [])[:3]:
        if not isinstance(sample, Mapping):
            continue
        query = sample.get("query")
        section = sample.get("section")
        status = sample.get("status")
        row_count = sample.get("row_count")
        rendered_rows = sample.get("rendered_rows")
        native_sql = sample.get("native_sql")
        portable_command = sample.get("portable_command")
        label = query or section or "unknown"
        parts = []
        if status:
            parts.append(f"status={status}")
        if _trace_report_sample_value_present(row_count):
            parts.append(f"rows={row_count}")
        if _trace_report_sample_value_present(rendered_rows):
            parts.append(f"rendered={rendered_rows}")
        if _trace_report_sample_value_present(native_sql):
            parts.append(f"native_sql={native_sql}")
        line = f"- Timeline query summary: {label}"
        if parts:
            line += " " + " ".join(parts)
        lines.append(line)
        if portable_command:
            lines.append(f"  Command: `{portable_command}`")
    for sample in summary.get("supported_claim_samples", [])[:3]:
        if not isinstance(sample, Mapping):
            continue
        claim = sample.get("claim")
        basis = sample.get("basis")
        evidence_grade = sample.get("evidence_grade")
        if claim:
            parts = []
            if evidence_grade:
                parts.append(f"grade={evidence_grade}")
            if basis:
                parts.append(f"basis={basis}")
            line = f"- Supported claim: {claim}"
            if parts:
                line += " " + " ".join(parts)
            lines.append(line)
    for sample in summary.get("correctness_evidence_samples", [])[:3]:
        if not isinstance(sample, Mapping):
            continue
        evidence = sample.get("evidence")
        status = sample.get("status")
        proof = sample.get("proof_grade_status")
        next_gate = sample.get("next_gate")
        if evidence:
            parts = []
            if status:
                parts.append(f"status={status}")
            if proof:
                parts.append(f"proof={proof}")
            line = f"- Correctness evidence: {evidence}"
            if parts:
                line += " " + " ".join(parts)
            lines.append(line)
        if next_gate:
            lines.append(f"  Next gate: `{next_gate}`")
    for sample in summary.get("evidence_artifact_check_samples", [])[:3]:
        if not isinstance(sample, Mapping):
            continue
        check = sample.get("check")
        status = sample.get("status")
        detail = sample.get("detail")
        artifact = sample.get("evidence_artifact")
        if check:
            parts = []
            if artifact:
                parts.append(f"artifact={artifact}")
            if status:
                parts.append(f"status={status}")
            if detail:
                parts.append(f"detail={detail}")
            line = f"- Evidence artifact check: {check}"
            if parts:
                line += " " + " ".join(parts)
            lines.append(line)
    for sample in summary.get("promotion_gate_summary_samples", [])[:6]:
        if not isinstance(sample, Mapping):
            continue
        gate = sample.get("gate")
        status = sample.get("status")
        proof = sample.get("proof_grade_status")
        next_gate = sample.get("next_gate")
        if gate:
            parts = []
            if status:
                parts.append(f"status={status}")
            if proof:
                parts.append(f"proof={proof}")
            line = f"- Promotion gate: {gate}"
            if parts:
                line += " " + " ".join(parts)
            lines.append(line)
        if next_gate:
            lines.append(f"  Next gate: `{next_gate}`")
    for sample in summary.get("trace_mode_guardrail_samples", [])[:3]:
        if not isinstance(sample, Mapping):
            continue
        trace_mode = sample.get("trace_mode")
        guardrail = sample.get("claim_guardrail")
        overhead = sample.get("overhead_boundary")
        sinks = sample.get("trace_sinks")
        if trace_mode:
            parts = []
            if sinks:
                parts.append(f"sinks={sinks}")
            if overhead:
                parts.append(f"overhead={overhead}")
            if guardrail:
                parts.append(f"guardrail={guardrail}")
            line = f"- Trace mode guardrail: {trace_mode}"
            if parts:
                line += " " + " ".join(parts)
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
        "Prefer the report's `sections.report_grade`, `sections.report_triage`,",
        "`sections.answerability`, `sections.unsupported_claims`,",
        "`sections.next_measurements`,",
        "`sections.preflight_findings`, and `sections.evidence_classification`",
        "before ad hoc log parsing. Use",
        "`sections.report_json_section_inventory` and",
        "`sections.report_section_inventory` to discover available JSON",
        "sections and timeline queries. Then read `sections.capture`,",
        "`sections.run_provenance`,",
        "`sections.artifact_identities`, `sections.artifact_identity_checks`,",
        "and `sections.capture_capabilities` to verify capture provenance,",
        "artifact hashes, and capability booleans before heavier evidence. Read",
        "`sections.ab_provenance`, `sections.ab_comparability`,",
        "`sections.ab_coverage`, and `sections.ab_repeatability` before",
        "treating any baseline/candidate delta as comparison-grade evidence.",
        "Those A/B sections remain diagnostic until provenance, coverage,",
        "repeatability, correctness, and promotion gates are satisfied. Read",
        "`sections.trace_config_rows` to distinguish requested controls,",
        "recorded sidecars, and `missing_requested_sidecar_controls`, then use",
        "`sections.provider_payload_boundary_inventory_rows` to distinguish",
        "available, recorded, blocked, route-only runtime-sidecar, and",
        "other-backend provider/runtime payload lanes, then read",
        "`sections.trace_event_artifacts`,",
        "`sections.backend_event_artifacts` and",
        "`sections.backend_event_rows` to inspect raw backend JSONL artifacts",
        "and matching backend event rows before higher-level classification, then read",
        "`sections.backend_provider_boundaries` and",
        "`sections.backend_fail_closed_root_causes` to inspect provider",
        "validation stages and fail-closed root causes, then read",
        "`sections.debug_payload_artifact_summary_rows` to see payload",
        "sensitivity and timing-perturbation boundaries. Then read",
        "`sections.token_quality_summary_rows` and",
        "`sections.oracle_reference_summary_rows` to inspect selected-token",
        "top-k status and HF CPU reference anchors without treating",
        "system-under-test rows as oracle evidence. Then use",
        "`sections.introspection_capability_rows`,",
        "`sections.introspection_artifacts`, and",
        "`sections.introspection_artifact_summary_rows`, then",
        "`sections.introspection_section_inventory`, to decide which tracing",
        "sidecars exist before opening heavier sidecar-specific sections.",
        "When the inventory says they are available, read",
        "`sections.planning_decision_sidecar_rows` to inspect frontend and",
        "Lean planning diagnostics without treating planning sidecars as model",
        "evidence. Read `sections.token_quality_sidecar_rows` and",
        "`sections.topk_token_sidecar_rows` to inspect raw selected-token and",
        "top-k candidate rows without treating system-under-test rows as",
        "oracle evidence. Read `sections.tensor_payload_sidecar_rows`,",
        "`sections.kv_payload_digest_sidecar_rows`,",
        "`sections.logit_slice_sidecar_rows`, and",
        "`sections.activation_digest_sidecar_rows` to inspect provider/runtime",
        "payload summaries, scheduler K/V digest rows, final-logit slices,",
        "and activation digests without treating them as oracle evidence.",
        "Read",
        "`sections.scheduler_packet_lineage_sidecar_rows` and",
        "`sections.scheduler_kv_shard_lifecycle_sidecar_rows`, plus",
        "`sections.scheduler_listener_sparse_logit_sidecar_rows`, to inspect",
        "scheduler packet shape, K/V shard lifecycle, and sparse-listener",
        "delivery diagnostics without treating scheduler metadata as",
        "hardware-counter evidence. Also read",
        "`sections.device_dma_lifecycle_sidecar_rows`,",
        "`sections.attention_page_trace_sidecar_rows`, and",
        "`sections.device_result_digest_sidecar_rows` to inspect compact",
        "device DMA/queue lifecycle, attention page, and result digest diagnostics without",
        "treating debug payload rows as performance proof.",
        "Use `sections.timeline_query_summary` to choose named timeline",
        "queries only after the capture/report provenance says the trace can",
        "support them.",
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
            "why": "Keep C++ Tron/Rinzler classified as comparison and rollback evidence, and run the slow lane only after HF-backed Ares backend quality and performance look competitive.",
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


def render_introspection_ladder_lines(summary: Mapping[str, Any]) -> list[str]:
    lines = ["- Validated introspection ladder:"]
    if path := summary.get("path"):
        lines.append(f"  - report path: `{path}`")
    if digest := summary.get("sha256"):
        lines.append(f"  - report sha256: `{digest}`")
    if evidence_class := summary.get("evidence_class"):
        lines.append(f"  - evidence class: `{evidence_class}`")
    if status := summary.get("status"):
        lines.append(f"  - ladder status: `{status}`")
    stage_order = summary.get("stage_order")
    if isinstance(stage_order, list) and stage_order:
        stages = " -> ".join(str(stage) for stage in stage_order[:8])
        suffix = " -> ..." if len(stage_order) > 8 else ""
        lines.append(f"  - stage order: `{stages}{suffix}`")
    if first_stage := summary.get("first_failing_stage"):
        lines.append(f"  - first failing introspection stage: `{first_stage}`")
    if next_owner := summary.get("next_owner"):
        lines.append(f"  - next owner: `{next_owner}`")
    first_failed = summary.get("first_failed_comparison")
    if isinstance(first_failed, Mapping):
        comparison_line = _format_introspection_ladder_comparison(first_failed)
        if comparison_line:
            lines.append(f"  - first failed comparison: {comparison_line}")
    first_mismatch = summary.get("first_mismatch")
    if isinstance(first_mismatch, Mapping):
        mismatch_line = _format_introspection_ladder_mismatch(first_mismatch)
        if mismatch_line:
            lines.append(f"  - first mismatch: {mismatch_line}")
    trace_context = summary.get("trace_context")
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
            ("trace_metadata", "Ares trace metadata"),
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
    errors = summary.get("errors")
    if isinstance(errors, list) and errors:
        lines.append("- Validator errors: " + "; ".join(str(error) for error in errors))
    return lines


def _format_introspection_ladder_comparison(comparison: Mapping[str, Any]) -> str:
    from_stage = comparison.get("from_stage")
    to_stage = comparison.get("to_stage")
    if not from_stage or not to_stage:
        return ""
    parts = [f"`{from_stage}` -> `{to_stage}`"]
    for field in ("status", "result"):
        value = comparison.get(field)
        if value:
            parts.append(f"{field}=`{value}`")
    if path := comparison.get("path"):
        parts.append(f"path=`{path}`")
    if digest := comparison.get("sha256"):
        parts.append(f"sha256=`{digest}`")
    return " ".join(parts)


def _format_introspection_ladder_mismatch(mismatch: Mapping[str, Any]) -> str:
    labels = {
        "id": "id",
        "producer_generator": "producer",
        "value_id": "value",
        "tensor": "tensor",
        "metric": "metric",
        "max_abs_error": "max_abs_error",
        "max_rel_error": "max_rel_error",
        "tvd": "tvd",
        "top1_reference": "top1_reference",
        "top1_candidate": "top1_candidate",
        "token_index": "token_index",
        "statement_index": "statement_index",
        "statement_name": "statement_name",
        "operation_id": "operation_id",
        "trace_label": "trace_label",
    }
    parts = []
    for field, label in labels.items():
        value = mismatch.get(field)
        if value is not None:
            parts.append(f"{label}=`{value}`")
    return " ".join(parts)


def introspection_ladder_handoff_section(state: Mapping[str, Any] | None) -> str:
    if not isinstance(state, Mapping):
        return ""
    summary = state.get("introspection_ladder")
    if not isinstance(summary, Mapping):
        return ""
    lines = [
        "## Introspection Ladder",
        "",
        *render_introspection_ladder_lines(summary),
        "",
    ]
    return "\n".join(lines)


def introspection_ladder_prompt_lines(spec: Mapping[str, Any]) -> list[str]:
    summary = introspection_ladder_summary_from_spec(spec)
    if not summary:
        return []
    return render_introspection_ladder_lines(summary)


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
{introspection_ladder_handoff_section(state)}
## Operator Steering

{chr(10).join(steering_prompt_lines(cfg))}

## Rules

- HF Transformers on PyTorch CPU is the model-correctness oracle.
- Cache HF CPU token/logit artifacts once for the exact model/checkpoint,
  tokenizer, prompt-token context, decode depth, dtype/quantization policy,
  deterministic settings, and oracle/exporter code tuple; reuse those goldens
  for the fast Ares backend debug loop until that tuple changes.
- Prefer the cheapest verifier that proves the first failing gate. Do not
  recapture HF logits or run slower comparison lanes for ordinary Ares backend
  code changes.
- Order the debug loop by wall-clock cost: cached HF logit comparison first,
  focused backend/module slices next, short-depth generation after that, then
  longer-depth generation, with C++ comparison reserved for an explicit late
  milestone.
- Ares/Rust output is system-under-test evidence.
- C++ Tron/Rinzler is comparison and rollback evidence only. Do not involve
  that slow lane until the selected Ares backend has a competitive candidate
  against the cached HF goldens.
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
    spec.setdefault("frontend", "hf-export")
    spec.setdefault("backend", "fpga")
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
    introspection_ladder = introspection_ladder_summary_from_spec(spec)
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
    if introspection_ladder:
        state["introspection_ladder"] = introspection_ladder
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
                "- Inspect `sections.report_grade`, `sections.report_triage`, `sections.answerability`, `sections.unsupported_claims`, `sections.next_measurements`, `sections.preflight_findings`, and `sections.evidence_classification` before ad hoc parsing.",
                "- Read `sections.report_json_section_inventory` and `sections.report_section_inventory` to discover available report sections and timeline queries, then read `sections.capture`, `sections.run_provenance`, `sections.artifact_identities`, `sections.artifact_identity_checks`, `sections.capture_capabilities`, `sections.trace_config_rows` including `missing_requested_sidecar_controls`, `sections.provider_payload_boundary_inventory_rows` including recorded provider-callback rows and route-only runtime-sidecar rows, `sections.trace_event_artifacts`, `sections.backend_event_artifacts`, `sections.backend_event_rows`, `sections.backend_provider_boundaries`, `sections.backend_fail_closed_root_causes`, `sections.debug_payload_artifact_summary_rows`, `sections.token_quality_summary_rows`, `sections.oracle_reference_summary_rows`, `sections.introspection_capability_rows`, `sections.introspection_artifacts`, `sections.introspection_artifact_summary_rows`, and `sections.introspection_section_inventory` before choosing sidecar-specific report sections such as `sections.planning_decision_sidecar_rows`, `sections.token_quality_sidecar_rows`, `sections.topk_token_sidecar_rows`, `sections.tensor_payload_sidecar_rows`, `sections.kv_payload_digest_sidecar_rows`, `sections.logit_slice_sidecar_rows`, `sections.activation_digest_sidecar_rows`, `sections.scheduler_packet_lineage_sidecar_rows`, `sections.scheduler_listener_sparse_logit_sidecar_rows`, `sections.device_dma_lifecycle_sidecar_rows`, `sections.attention_page_trace_sidecar_rows`, and `sections.device_result_digest_sidecar_rows`; use `sections.timeline_query_summary` only after capture/report provenance supports timeline analysis.",
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
            "- Treat the captured token/logit rows as reusable goldens for this exact model/checkpoint, tokenizer, prompt-token context, decode depth, dtype/quantization policy, deterministic settings, and oracle/exporter code tuple.",
        ],
        "frontend_export": [
            "- Produce or select the frontend artifact declared by the model spec.",
            "- Use `frontend/hf-export` by default for current production-readiness work; keep `frontend/fx` as an explicit fallback.",
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
            "- Do not spend this slow comparison loop until cached-HF correctness and Ares backend performance suggest a competitive candidate.",
            "- Require matching replay context and dense-logit TVD artifacts.",
        ],
        "depth_performance": [
            "- Follow the 8 -> 64 -> 512 ladder against cached HF CPU token/logit goldens and keep correctness gates green before speed claims.",
            "- Record throughput, latency, TTFT, memory, and artifact hashes from the selected Ares backend without launching C++ comparison in the normal iteration loop.",
            "- Keep this loop HF-backed until short-depth correctness and performance justify a separate comparison checkpoint.",
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
            "- Capture HF CPU token/logit artifacts once per exact model/checkpoint, tokenizer, prompt-token context, decode depth, dtype/quantization policy, deterministic settings, and oracle/exporter code tuple; reuse those goldens for fast backend iteration until that tuple changes.",
            "- Prefer the fastest verifier that can prove the first failing gate; do not recapture HF logits or launch C++ comparison for ordinary Ares backend code changes.",
            "- Order the debug loop by wall-clock cost: cached HF logit comparison, focused backend/module slices, short-depth generation, longer-depth generation, then explicit late C++ comparison milestones.",
            "- C++ Tron/Rinzler is comparison, compliance, performance, and rollback evidence only.",
            "- Keep C++ Tron/Rinzler out of the normal debug loop; run it as an explicit milestone comparison after the selected Ares backend has HF-backed quality and competitive performance evidence.",
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
    introspection_ladder = None
    if cfg.model_spec_path.exists():
        spec = load_json(cfg.model_spec_path)
        trace_report = trace_report_summary_from_spec(spec)
        introspection_ladder = introspection_ladder_summary_from_spec(spec)
    if trace_report:
        state["trace_report"] = trace_report
    else:
        state.pop("trace_report", None)
    if introspection_ladder:
        state["introspection_ladder"] = introspection_ladder
    else:
        state.pop("introspection_ladder", None)
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
    if introspection_ladder:
        history_entry["introspection_ladder"] = introspection_ladder
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
            spec = load_json(cfg.model_spec_path)
            trace_report = trace_report_summary_from_spec(spec)
            introspection_ladder = introspection_ladder_summary_from_spec(spec)
        except AresIngestError:
            trace_report = None
            introspection_ladder = None
        if trace_report:
            state["trace_report"] = trace_report
        if introspection_ladder:
            state["introspection_ladder"] = introspection_ladder
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

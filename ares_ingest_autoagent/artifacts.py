"""Artifact validators for the Ares ingest AutoAgent scaffold."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import stat
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


HF_CPU_SCHEMA_ID = "ares.oracles.hf_cpu.record.v1"
HF_CPU_ORACLE_KIND = "huggingface_transformers_pytorch_cpu"
INTROSPECTION_LADDER_SCHEMA = "ares.introspection.ladder.v1"
LEAN_TARGET_PLAN_PRODUCER = {"language": "lean", "tool": "ingest-lean"}
ONE_TOKEN_LOGITS_SCHEMA = "ares.runtime.one_token_logits.v1"
ONE_TOKEN_PREFLIGHT_REPORT_SCHEMA = "ares.runtime.one_token_logits.preflight_report.v1"
EIGHT_TOKEN_GREEDY_SCHEMA = "ares.runtime.greedy_token_agreement.v1"
EIGHT_TOKEN_PREFLIGHT_REPORT_SCHEMA = (
  "ares.runtime.eight_token_greedy.preflight_report.v1"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.I)
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$", re.I)
REPLAY_CONTEXT_FIELDS = (
  "context_tokens",
  "context_tokens_role",
  "context_count",
  "new_count",
  "runtime_request_token_count",
  "context_prefix_token_count",
  "last_token",
)
CPP_COMPARISON_SOURCES = {
  "cpp_tron",
  "cpp_tron_rinzler",
  "cxx_tron",
  "cxx_tron_rinzler",
  "tron_rinzler_cpp",
}
ARES_RUNTIME_CANDIDATES = {"ares", "ares_rust", "ares_runtime", "runares", "rinzler"}
SCORING_WORKLOADS = {"independent_decode", "long_prefill"}
INTROSPECTION_STAGES = {
    "hf_cpu_oracle",
    "source_hf_hypergraph",
    "lean_imported_hypergraph",
    "lean_post_bridge_hypergraph",
    "lean_post_eqsat_hypergraph",
    "lean_extracted_hypergraph",
    "aresplan_semantic",
    "targetplan_colored",
    "backend",
}
INTROSPECTION_OWNERS = {
    "none",
    "ares-python",
    "ares-lean",
    "ares-targetplan",
    "ares-rust",
    "ares-perfetto",
    "model-inventory",
}
INTROSPECTION_TRACE_ARTIFACT_FIELDS = (
    "backend_events",
    "perfetto_traces",
    "stage_event_summaries",
    "perfetto_summaries",
    "trace_metadata",
)
INTROSPECTION_COMPARISON_DETAIL_LIMIT = 8
INTROSPECTION_FIRST_MISMATCH_SCALAR_FIELDS = (
    "id",
    "producer_generator",
    "value_id",
    "tensor",
    "metric",
    "max_abs_error",
    "max_rel_error",
    "tvd",
    "top1_reference",
    "top1_candidate",
    "token_index",
    "statement_index",
    "statement_name",
    "operation_id",
    "trace_label",
)
TRACE_REPORT_REQUIRED_SECTIONS = (
    "preflight",
    "analysis_commands",
    "report_grade",
    "report_triage",
    "answerability",
    "unsupported_claims",
    "next_measurements",
    "scheduler_packet_lineage_sidecar_rows",
)
TRACE_REPORT_TRIAGE_REQUIRED_FIELDS = (
    "triage_status",
    "report_grade",
    "proof_grade_status",
    "first_blocked_gate",
    "first_blocked_gate_status",
    "first_blocked_gate_basis",
    "first_next_measurement_priority",
    "first_next_measurement_reason",
    "first_next_measurement",
    "first_next_measurement_command_hint",
    "first_answerable_question",
    "first_unsupported_claim",
    "first_useful_section",
    "first_action",
    "claim_boundary",
)
TRACE_REPORT_TRIAGE_STATUSES = {
    "needs_measurement",
    "review_blocked_gate",
    "inspect_answerable_sections",
    "inspect_report_grade",
}
TRACE_REPORT_GRADES = {"inconclusive", "diagnostic", "comparison-grade"}
TRACE_REPORT_SCHEMA_VERSION = 2
MAX_TRACE_REPORT_BYTES = 64 * 1024 * 1024
MAX_TRACE_REPORT_JSON_ITEMS = 1_048_576
MAX_TRACE_METADATA_BYTES = 16 * 1024 * 1024
MAX_TRACE_METADATA_JSON_ITEMS = 262_144
MAX_SCHEDULER_LINEAGE_JSONL_BYTES = 64 * 1024 * 1024
MAX_SCHEDULER_LINEAGE_JSONL_LINE_BYTES = 4 * 1024 * 1024
MAX_SCHEDULER_LINEAGE_JSONL_ROWS = 65_536
MAX_SCHEDULER_LINEAGE_JSON_ITEMS_PER_ROW = 262_144
MAX_SCHEDULER_LINEAGE_JSON_ITEMS = 1_048_576
MAX_JSON_NESTING_DEPTH = 64
MAX_SCHEDULER_STRING_CHARS = 4096
PROVIDER_ADMISSION_MANIFEST_RELATIVE = Path(
  "runtime/core/crates/ares-plan-exec/fixtures/provider-admission-manifest.v1.json"
)
# V3 already defines the closed tagged source-authority union. Selecting exactly
# one legacy or successor core key extends that union without changing legacy
# bytes, so every live reader moves atomically while the envelope version stays.
SCHEDULER_GENERATED_STATE_TRANSPORT_SCHEMA = (
    "ares.scheduler.generated_state_transport.v3"
)
SCHEDULER_LEAN_FIXTURE_CORE_AUTHORITY_KEYS = (
    "legacy_core_sha256",
    "successor_core_sha256",
)
SCHEDULER_I64_MAX = (1 << 63) - 1
SCHEDULER_U64_MAX = (1 << 64) - 1
SCHEDULER_PUBLICATION_LANE_PROJECTION_FIELDS = (
    "publication_index",
    "publication_lane_id",
    "publication_owner_indices",
    "state_checkpoint_id",
    "state_branch_id",
    "state_parent_checkpoint_id",
    "state_parent_branch_id",
    "generated_state_transport_schema",
    "event",
    "model_id",
    "backend_id",
    "ares_plan_sha256",
    "target_plan_sha256",
    "source_authority",
    "publication_authority",
    "targetplan_op_ids",
    "qualified_row_count",
    "qualified_rows_sha256",
)
SCHEDULER_TRANSPORT_REPORT_FIELDS = frozenset(
    {
        "generated_state_transport_schema",
        "generated_state_event",
        "ares_plan_sha256",
        "target_plan_sha256",
        "source_authority_json",
        "publication_authority_json",
        "targetplan_op_ids",
    }
)
SCHEDULER_DIRECT_TRANSPORT_REPORT_FIELDS = frozenset(
    {
        "state_checkpoint_id",
        "state_branch_id",
        "state_parent_checkpoint_id",
        "state_parent_branch_id",
        "qualified_row_count",
        "qualified_rows_sha256",
    }
)
SCHEDULER_PUBLICATION_REPORT_FIELDS = frozenset(
    SCHEDULER_TRANSPORT_REPORT_FIELDS
    | {
        "publication_lanes",
        "publication_lanes_json",
        "publication_batch_id_json",
        "publication_batch_generation",
        "publication_lane_count",
        "publication_record_count",
        "publication_route_owner_index",
    }
)
SCHEDULER_REPORT_COMMON_FIELDS = frozenset(
    {
        "artifact_index",
        "row_index",
        "row_kind",
        "status",
        "evidence_role",
        "trace_run_id",
        "process_kind",
        "model_id",
        "backend_id",
        "request_id",
        "generation_id",
        "parent_location_id",
        "location_id",
        "executor_shape",
        "executor_status",
        "attention_mode",
        "runtime_request_token_count",
        "tokens_reused",
        "token_job_count",
        "kv_job_count",
        "listener_sparse_rows",
        "listener_sparse_tokens",
        "kv_save_rows",
        "kv_context_rows",
        "visible_token_slots",
        "kv_page_count",
        "hw_shard_allocation_requests",
        "hw_gof_page_infos",
        "minibatch_count",
        "sparse_topk_rows",
        "sparse_topk_token_count",
        "prior_hw_shard_allocation_requests",
        "prior_host_gof_ready_rows",
        "prior_host_gof_pending_dma_rows",
        "prior_host_gof_cacheblock_dma_plans",
        "prior_host_gof_materialized_cacheblocks",
        "prior_host_gof_dma_completions",
        "prior_host_gof_page_infos",
        "prior_host_gof_staging_status",
        "failure_reason",
    }
)
TRACE_REPORT_JSON_SECTION_SAMPLE_KEYS = (
    "preflight",
    "analysis_commands",
    "report_grade",
    "report_triage",
    "capture",
    "run_provenance",
    "artifact_identities",
    "artifact_identity_checks",
    "capture_capabilities",
    "trace_config_rows",
    "provider_payload_boundary_inventory_rows",
    "trace_event_artifacts",
    "backend_event_artifacts",
    "backend_event_rows",
    "backend_provider_boundaries",
    "backend_fail_closed_root_causes",
    "debug_payload_artifact_summary_rows",
    "token_quality_summary_rows",
    "oracle_reference_summary_rows",
    "planning_decision_sidecar_rows",
    "token_quality_sidecar_rows",
    "topk_token_sidecar_rows",
    "tensor_payload_sidecar_rows",
    "kv_payload_digest_sidecar_rows",
    "logit_slice_sidecar_rows",
    "activation_digest_sidecar_rows",
    "device_result_digest_sidecar_rows",
    "scheduler_packet_lineage_sidecar_rows",
    "scheduler_kv_shard_lifecycle_sidecar_rows",
    "scheduler_listener_sparse_logit_sidecar_rows",
    "device_dma_lifecycle_sidecar_rows",
    "attention_page_trace_sidecar_rows",
    "introspection_artifacts",
    "introspection_capability_rows",
    "introspection_artifact_summary_rows",
    "introspection_section_inventory",
    "supported_claims",
    "correctness_evidence",
    "evidence_artifact_checks",
    "promotion_gate_summary",
    "trace_mode_guardrails",
    "ab_provenance",
    "ab_comparability",
    "ab_coverage",
    "ab_repeatability",
    "report_json_section_inventory",
    "report_section_inventory",
    "preflight_findings",
    "evidence_classification",
    "answerability",
    "unsupported_claims",
    "next_measurements",
    "timeline_query_summary",
)

FLOATING_REVISION_NAMES = {
  "@",
  "dev",
  "develop",
  "development",
  "head",
  "latest",
  "main",
  "master",
  "stable",
  "trunk",
}
FLOATING_REVISION_PREFIXES = (
  "head~",
  "head^",
  "origin/",
  "refs/heads/",
  "refs/remotes/",
  "upstream/",
)


@dataclass(frozen=True)
class ArtifactValidation:
  passed: bool
  errors: tuple[str, ...]
  detail: dict[str, Any]

  def as_gate(
    self,
    *,
    label: str,
    validator_name: str,
    path: Path | None = None,
  ) -> dict[str, Any]:
    gate: dict[str, Any] = {
      "label": label,
      "artifact_validator": validator_name,
      "passed": self.passed,
      "score": 1.0 if self.passed else 0.0,
      "detail": self.detail,
    }
    if path is not None:
      gate["path"] = str(path)
      gate["exists"] = path.is_file()
    if self.errors:
      gate["errors"] = list(self.errors)
    return gate


def artifact_gate(
  path: Path,
  *,
  label: str,
  validator_name: str,
  validator: Any,
) -> dict[str, Any]:
  if not path.is_file():
    return {
      "label": label,
      "artifact_validator": validator_name,
      "path": str(path),
      "exists": path.exists(),
      "passed": False,
      "score": 0.0,
      "errors": ["artifact file is missing"],
    }
  try:
    payload = json.loads(path.read_text())
  except json.JSONDecodeError as exc:
    return {
      "label": label,
      "artifact_validator": validator_name,
      "path": str(path),
      "exists": True,
      "passed": False,
      "score": 0.0,
      "errors": [f"invalid JSON: {exc}"],
    }
  return validator(payload).as_gate(
    label=label,
    validator_name=validator_name,
    path=path,
  )


def evidence_gate(
  path: Path,
  *,
  label: str,
  validator_name: str,
  validator: Any,
) -> dict[str, Any]:
  if not path.is_file():
    return {
      "label": label,
      "artifact_validator": validator_name,
      "path": str(path),
      "exists": path.exists(),
      "passed": False,
      "score": 0.0,
      "errors": ["evidence file is missing"],
    }
  try:
    payload = _read_json_or_jsonl(path)
  except ValueError as exc:
    return {
      "label": label,
      "artifact_validator": validator_name,
      "path": str(path),
      "exists": True,
      "passed": False,
      "score": 0.0,
      "errors": [str(exc)],
    }
  return validator(payload).as_gate(
    label=label,
    validator_name=validator_name,
    path=path,
  )


def ares_plan_gate(path: Path, *, label: str = "generated AresPlan") -> dict[str, Any]:
  return artifact_gate(
    path,
    label=label,
    validator_name="ares_plan",
    validator=validate_ares_plan,
  )


def target_plan_gate(path: Path, *, label: str = "Lean TargetPlan") -> dict[str, Any]:
  return artifact_gate(
    path,
    label=label,
    validator_name="target_plan",
    validator=validate_target_plan,
  )


def artifact_consistency_gate(
  spec: Mapping[str, Any],
  *,
  oracle_payload: Any,
  validated_gates: Mapping[str, Any],
  label: str = "artifact model consistency",
) -> dict[str, Any]:
  return validate_artifact_consistency(
    spec,
    oracle_payload=oracle_payload,
    validated_gates=validated_gates,
  ).as_gate(
    label=label,
    validator_name="artifact_consistency",
  )


def backend_open_gate(
  path: Path, *, label: str = "backend provider open evidence"
) -> dict[str, Any]:
  return evidence_gate(
    path,
    label=label,
    validator_name="backend_open",
    validator=validate_backend_open_evidence,
  )


def one_token_logits_gate(
  path: Path, *, label: str = "one-token logits evidence"
) -> dict[str, Any]:
  if not path.is_file():
    return {
      "label": label,
      "artifact_validator": "one_token_logits",
      "path": str(path),
      "exists": path.exists(),
      "passed": False,
      "score": 0.0,
      "errors": ["evidence file is missing"],
    }
  try:
    payload = _read_json_or_jsonl(path)
  except ValueError as exc:
    return {
      "label": label,
      "artifact_validator": "one_token_logits",
      "path": str(path),
      "exists": True,
      "passed": False,
      "score": 0.0,
      "errors": [str(exc)],
    }
  return validate_one_token_logits_evidence(
    payload,
    base_dir=path.parent,
  ).as_gate(
    label=label,
    validator_name="one_token_logits",
    path=path,
  )


def cpp_tvd_gate(
  path: Path, *, label: str = "C++ comparison TVD evidence"
) -> dict[str, Any]:
  return evidence_gate(
    path,
    label=label,
    validator_name="cpp_tvd",
    validator=validate_cpp_tvd_evidence,
  )


def depth_performance_gate(
  path: Path, *, label: str = "depth ladder performance evidence"
) -> dict[str, Any]:
  return evidence_gate(
    path,
    label=label,
    validator_name="depth_performance",
    validator=validate_depth_performance_evidence,
  )


def mmlu_pro_gate(
  path: Path,
  *,
  label: str = "MMLU Pro benchmark evidence",
  expected_model: str | None = None,
  expected_backend: str | None = None,
  required_coverage_percent: float | None = None,
) -> dict[str, Any]:
  if not path.is_file():
    return {
      "label": label,
      "artifact_validator": "mmlu_pro",
      "path": str(path),
      "exists": path.exists(),
      "passed": False,
      "score": 0.0,
      "errors": ["evidence file is missing"],
    }
  try:
    payload = _read_json_or_jsonl(path)
  except ValueError as exc:
    return {
      "label": label,
      "artifact_validator": "mmlu_pro",
      "path": str(path),
      "exists": True,
      "passed": False,
      "score": 0.0,
      "errors": [str(exc)],
    }
  return validate_mmlu_pro_evidence(
    payload,
    base_dir=path.parent,
    expected_model=expected_model,
    expected_backend=expected_backend,
    required_coverage_percent=required_coverage_percent,
  ).as_gate(
    label=label,
    validator_name="mmlu_pro",
    path=path,
  )


def introspection_ladder_gate(
    path: Path,
    *,
    label: str = "introspection ladder evidence",
) -> dict[str, Any]:
    if not path.is_file():
        return {
            "label": label,
            "artifact_validator": "introspection_ladder",
            "path": str(path),
            "exists": path.exists(),
            "passed": False,
            "score": 0.0,
            "errors": ["introspection ladder file is missing"],
        }
    try:
        payload = _read_json_or_jsonl(path)
    except ValueError as exc:
        return {
            "label": label,
            "artifact_validator": "introspection_ladder",
            "path": str(path),
            "exists": True,
            "passed": False,
            "score": 0.0,
            "errors": [str(exc)],
        }
    gate = validate_introspection_ladder_report(
        payload,
        base_dir=path.parent,
    ).as_gate(
        label=label,
        validator_name="introspection_ladder",
        path=path,
    )
    gate["sha256"] = _sha256_file(path)
    return gate


def trace_report_gate(
    path: Path,
    *,
    label: str = "Ares trace report JSON",
    authority_root: Path | None = None,
) -> dict[str, Any]:
    try:
        data = _read_regular_file_bytes(
            path,
            maximum_bytes=MAX_TRACE_REPORT_BYTES,
            label="trace report JSON",
        )
    except ValueError as exc:
        return {
            "label": label,
            "artifact_validator": "trace_report",
            "path": str(path),
            "exists": path.exists(),
            "passed": False,
            "score": 0.0,
            "errors": [str(exc)],
        }
    digest = hashlib.sha256(data).hexdigest()
    try:
        payload = _strict_json_loads_bytes(
            data,
            label="trace report JSON",
            maximum_items=MAX_TRACE_REPORT_JSON_ITEMS,
        )
    except ValueError as exc:
        return {
            "label": label,
            "artifact_validator": "trace_report",
            "path": str(path),
            "exists": True,
            "passed": False,
            "score": 0.0,
            "detail": {"sha256": digest},
            "errors": [f"invalid JSON: {exc}"],
        }
    validation = validate_trace_report_json(
        payload,
        report_path=path,
        authority_root=authority_root,
    )
    gate = validation.as_gate(
        label=label,
        validator_name="trace_report",
        path=path,
    )
    detail = gate.setdefault("detail", {})
    if isinstance(detail, dict):
        detail["sha256"] = digest
    return gate


def token_agreement_gate(
  path: Path, *, label: str = "eight-token greedy evidence"
) -> dict[str, Any]:
  if not path.is_file():
    return {
      "label": label,
      "artifact_validator": "eight_token_greedy",
      "path": str(path),
      "exists": path.exists(),
      "passed": False,
      "score": 0.0,
      "errors": ["evidence file is missing"],
    }
  try:
    payload = _read_json_or_jsonl(path)
  except ValueError as exc:
    return {
      "label": label,
      "artifact_validator": "eight_token_greedy",
      "path": str(path),
      "exists": True,
      "passed": False,
      "score": 0.0,
      "errors": [str(exc)],
    }
  return validate_token_agreement_evidence(
    payload,
    base_dir=path.parent,
  ).as_gate(
    label=label,
    validator_name="eight_token_greedy",
    path=path,
  )


def build_greedy_token_evidence(
  token_result: Mapping[str, Any],
  *,
  reference: Path,
  candidate: Path,
  reference_payload: Any,
  candidate_payload: Any,
  expected_generated_tokens: int = 8,
  evidence_class: str = "system_under_test",
  oracle: str = HF_CPU_ORACLE_KIND,
  candidate_runtime: str = "ares",
) -> dict[str, Any]:
  evidence = dict(token_result)
  reference_generated = _generated_token_ids(reference_payload)
  candidate_generated = _generated_token_ids(candidate_payload)
  evidence.update(
    {
      "schema": "ares.runtime.greedy_token_agreement.v1",
      "evidence_class": evidence_class,
      "oracle": oracle,
      "candidate": candidate_runtime,
      "decode_strategy": "greedy",
      "expected_generated_tokens": expected_generated_tokens,
      "generated_tokens": (
        len(candidate_generated) if candidate_generated is not None else None
      ),
      "reference_generated_token_ids": reference_generated,
      "candidate_generated_token_ids": candidate_generated,
      "exact_match": _token_result_exact_match(token_result),
      "reference": {
        "path": str(reference),
        "sha256": _sha256_file(reference),
      },
      "candidate_output": {
        "path": str(candidate),
        "sha256": _sha256_file(candidate),
        "runtime": candidate_runtime,
      },
    }
  )
  return evidence


def is_floating_revision(revision: str) -> bool:
  value = revision.strip().lower()
  return value in FLOATING_REVISION_NAMES or value.startswith(
    FLOATING_REVISION_PREFIXES
  )


def validate_hf_cpu_oracle_record(record: Any) -> ArtifactValidation:
  errors: list[str] = []
  if not isinstance(record, dict):
    return _validation(False, ["record must be a JSON object"], {})

  _require_fields(
    errors,
    record,
    (
      "schema",
      "record_kind",
      "capture_id",
      "created_utc",
      "source",
      "model",
      "tokenizer",
      "run",
      "prompt",
      "generation",
      "logit_slices",
      "environment",
    ),
    "record",
  )
  if errors:
    return _validation(False, errors, {"record_kind": record.get("record_kind")})

  if record.get("schema") != HF_CPU_SCHEMA_ID:
    errors.append("record.schema must be ares.oracles.hf_cpu.record.v1")
  if record.get("record_kind") != "hf_cpu_oracle_capture":
    errors.append("record_kind must be hf_cpu_oracle_capture")

  source = _expect_object(errors, record.get("source"), "source")
  model = _expect_object(errors, record.get("model"), "model")
  tokenizer = _expect_object(errors, record.get("tokenizer"), "tokenizer")
  run = _expect_object(errors, record.get("run"), "run")
  prompt = _expect_object(errors, record.get("prompt"), "prompt")
  generation = _expect_object(errors, record.get("generation"), "generation")
  environment = _expect_object(errors, record.get("environment"), "environment")
  logit_slices = record.get("logit_slices")

  if source is not None:
    _require_fields(errors, source, ("oracle", "capture_script"), "source")
    if source.get("oracle") != HF_CPU_ORACLE_KIND:
      errors.append(f"source.oracle must be {HF_CPU_ORACLE_KIND}")
    _require_non_empty_string(
      errors, source.get("capture_script"), "source.capture_script"
    )

  if model is not None:
    _validate_revision_metadata(
      errors,
      model,
      "model",
      ("model_id", "requested_revision", "resolved_revision", "dtype"),
    )
  if tokenizer is not None:
    _validate_revision_metadata(
      errors,
      tokenizer,
      "tokenizer",
      ("tokenizer_id", "requested_revision", "resolved_revision"),
    )
  if run is not None:
    _validate_run(errors, run)
  if prompt is not None:
    _validate_prompt(errors, prompt)
  if generation is not None:
    _validate_generation(errors, generation)
  if environment is not None:
    _require_fields(
      errors,
      environment,
      ("python_version", "platform", "torch_version", "transformers_version"),
      "environment",
    )

  generated_ids = (
    generation.get("generated_token_ids") if isinstance(generation, dict) else None
  )
  selected_ids = _validate_logit_slices(errors, logit_slices)
  if isinstance(generated_ids, list) and selected_ids is not None:
    if selected_ids != generated_ids:
      errors.append(
        "generation.generated_token_ids must match logit_slices selected_token_id values"
      )

  detail = {
    "schema": record.get("schema"),
    "record_kind": record.get("record_kind"),
    "source_oracle": source.get("oracle") if isinstance(source, dict) else None,
    "generated_token_count": len(generated_ids)
    if isinstance(generated_ids, list)
    else None,
    "logit_slice_count": len(logit_slices) if isinstance(logit_slices, list) else None,
  }
  return _validation(not errors, errors, detail)


def validate_ares_plan(plan: Any) -> ArtifactValidation:
  errors: list[str] = []
  if not isinstance(plan, dict):
    return _validation(False, ["AresPlan must be a JSON object"], {})

  _require_fields(
    errors,
    plan,
    ("schema_version", "config", "weights", "buffers", "provenance"),
    "AresPlan",
  )
  version = plan.get("schema_version")
  if not isinstance(version, int) or version not in {1, 2}:
    errors.append("AresPlan.schema_version must be 1 or 2")
  _expect_object(errors, plan.get("config"), "AresPlan.config")
  _expect_string_list(errors, plan.get("weights"), "AresPlan.weights")
  if not isinstance(plan.get("buffers"), list):
    errors.append("AresPlan.buffers must be a list")

  provenance = _expect_object(errors, plan.get("provenance"), "AresPlan.provenance")
  if provenance is not None:
    _require_fields(
      errors,
      provenance,
      ("fx_hash", "rule_corpus_hash", "emitter_version"),
      "AresPlan.provenance",
    )
    _require_non_empty_string(
      errors,
      provenance.get("emitter_version"),
      "AresPlan.provenance.emitter_version",
    )
    emitter = provenance.get("emitter_version")
    if isinstance(emitter, str) and "ingest-lean" not in emitter:
      errors.append("AresPlan.provenance.emitter_version must name ingest-lean")

  body = None
  if version == 1:
    body = plan.get("spans")
    if not isinstance(body, list) or not body:
      errors.append("schema-v1 AresPlan.spans must be a non-empty list")
  elif version == 2:
    body = plan.get("stmts")
    if not isinstance(body, list) or not body:
      errors.append("schema-v2 AresPlan.stmts must be a non-empty list")

  detail = {
    "schema_version": version,
    "buffer_count": len(plan.get("buffers", []))
    if isinstance(plan.get("buffers"), list)
    else None,
    "weight_count": len(plan.get("weights", []))
    if isinstance(plan.get("weights"), list)
    else None,
    "statement_count": len(body) if isinstance(body, list) else None,
    "emitter_version": provenance.get("emitter_version")
    if isinstance(provenance, dict)
    else None,
  }
  return _validation(not errors, errors, detail)


def validate_backend_open_evidence(payload: Any) -> ArtifactValidation:
  errors: list[str] = []
  root = _payload_root(payload, errors, "backend open evidence")
  rows = _payload_rows(root)
  if root is None:
    return _validation(False, errors, {})

  backend_id = _first_string(root, rows, ("backend_id", "backend"))
  status = _first_string(root, rows, ("status", "state"))
  if backend_id is None:
    errors.append("backend evidence must name backend_id")
  if status not in {"open", "opened", "ok", "passed", "ready"}:
    errors.append("backend evidence status must be open/opened/ok/passed/ready")

  ares_sha = _first_string(root, rows, ("ares_plan_sha256", "ares_plan_hash"))
  target_sha = _first_string(root, rows, ("target_plan_sha256", "target_plan_hash"))
  ares_obj = root.get("ares_plan") if isinstance(root, dict) else None
  target_obj = root.get("target_plan") if isinstance(root, dict) else None
  if ares_sha is None and isinstance(ares_obj, dict):
    ares_sha = _first_string(ares_obj, [], ("sha256", "hash"))
  if target_sha is None and isinstance(target_obj, dict):
    target_sha = _first_string(target_obj, [], ("sha256", "hash"))
  _require_sha256(errors, ares_sha, "AresPlan")
  _require_sha256(errors, target_sha, "TargetPlan")

  target_backend = _first_string(root, rows, ("target_plan_backend", "target_backend"))
  if isinstance(target_obj, dict):
    nested_target_backend = _first_string(target_obj, [], ("backend_id", "backend"))
    target_backend = nested_target_backend or target_backend
  if target_backend is None:
    errors.append("backend evidence must name TargetPlan backend explicitly")
  if (
    backend_id is not None
    and target_backend is not None
    and target_backend != backend_id
  ):
    errors.append("TargetPlan backend must match opened backend")

  runtime_sidecars = _first_bool(
    root, rows, ("runtime_generated_sidecars", "runtime_generated_plan")
  )
  if _truthy_field(
    root, rows, ("runtime_generated_sidecars", "runtime_generated_plan")
  ):
    errors.append("backend evidence must not use runtime-generated plan sidecars")
  elif runtime_sidecars is not False:
    errors.append(
      "backend evidence must explicitly record runtime_generated_sidecars=false"
    )

  event_names = {_event_name(row) for row in rows}
  event_names.discard(None)
  if rows and not event_names.intersection(
    {"backend_open", "provider_open", "session_open", "runtime_open"}
  ):
    errors.append("backend event evidence must include a backend-open event row")

  detail = {
    "backend_id": backend_id,
    "status": status,
    "event_count": len(rows),
    "ares_plan_sha256": ares_sha,
    "target_plan_sha256": target_sha,
  }
  return _validation(not errors, errors, detail)


def validate_one_token_logits_evidence(
  payload: Any,
  *,
  base_dir: Path | None = None,
) -> ArtifactValidation:
  errors: list[str] = []
  root = _payload_root(payload, errors, "one-token logits evidence")
  if root is None:
    return _validation(False, errors, {})
  if not isinstance(root, dict):
    return _validation(
      False,
      [*errors, "one-token logits evidence must be a JSON object"],
      {},
    )

  schema = root.get("schema")
  if schema == ONE_TOKEN_PREFLIGHT_REPORT_SCHEMA:
    return _validation(
      False,
      ["one-token preflight reports are diagnostic and cannot close one_token_logits"],
      {
        "schema": schema,
        "evidence_class": _first_string(root, [], ("evidence_class", "classification")),
      },
    )
  if schema != ONE_TOKEN_LOGITS_SCHEMA:
    errors.append(f"one-token evidence schema must be {ONE_TOKEN_LOGITS_SCHEMA}")
  if _first_string(root, [], ("oracle", "oracle_source")) != HF_CPU_ORACLE_KIND:
    errors.append(f"one-token oracle must be {HF_CPU_ORACLE_KIND}")
  evidence_class = _first_string(root, [], ("evidence_class", "classification"))
  if evidence_class not in {"system_under_test", "promotion"}:
    errors.append("one-token evidence_class must be system_under_test or promotion")
  candidate = _first_string(root, [], ("candidate", "subject", "runtime"))
  if candidate not in ARES_RUNTIME_CANDIDATES:
    errors.append("one-token candidate must identify Ares system-under-test output")

  tvd, threshold = _validate_tvd(errors, root, "one-token")
  top1 = root.get("top1_agreement")
  same_argmax = root.get("same_argmax")
  if not isinstance(top1, int | float):
    errors.append("one-token top1_agreement must be numeric")
  elif float(top1) < 1.0:
    errors.append("one-token top1_agreement must be 1.0")
  if same_argmax is not True:
    errors.append("one-token same_argmax must be true")
  _validate_replay_context(errors, root.get("replay_context"), "one-token")
  model_id = _validate_model_provenance(
    errors,
    root.get("model_provenance"),
    "one-token model_provenance",
  )
  artifact_keys = _validate_one_token_artifacts(
    errors,
    root.get("artifacts"),
    base_dir,
  )

  passed = not errors and tvd is not None and threshold is not None and tvd <= threshold
  detail = {
    "tvd": tvd,
    "tvd_threshold": threshold,
    "top1_agreement": float(top1) if isinstance(top1, int | float) else None,
    "same_argmax": same_argmax,
    "candidate": candidate,
    "model_id": model_id,
    "artifact_keys": artifact_keys,
  }
  return _validation(passed, errors, detail)


def validate_cpp_tvd_evidence(payload: Any) -> ArtifactValidation:
  errors: list[str] = []
  root = _payload_root(payload, errors, "C++ comparison evidence")
  if root is None:
    return _validation(False, errors, {})
  if not isinstance(root, dict):
    return _validation(
      False,
      [*errors, "C++ comparison evidence must be a JSON object"],
      {},
    )

  evidence_class = _first_string(root, [], ("evidence_class", "classification"))
  if evidence_class != "comparison":
    errors.append("C++ TVD evidence_class must be comparison")
  source = _first_string(root, [], ("comparison_source", "source", "baseline"))
  if source not in CPP_COMPARISON_SOURCES:
    errors.append("comparison_source must identify C++ Tron/Rinzler")
  tvd, threshold = _validate_tvd(errors, root, "C++ TVD")
  _validate_replay_context(errors, root.get("replay_context"), "C++ TVD")
  oracle = _first_string(root, [], ("oracle", "oracle_source"))
  if oracle in CPP_COMPARISON_SOURCES:
    errors.append("C++ Tron/Rinzler must not be labeled as correctness oracle")

  passed = not errors and tvd is not None and threshold is not None and tvd <= threshold
  detail = {
    "comparison_source": source,
    "tvd": tvd,
    "tvd_threshold": threshold,
  }
  return _validation(passed, errors, detail)


def validate_depth_performance_evidence(payload: Any) -> ArtifactValidation:
  errors: list[str] = []
  root = _payload_root(payload, errors, "depth performance evidence")
  if root is None:
    return _validation(False, errors, {})
  if not isinstance(root, dict):
    return _validation(
      False,
      [*errors, "depth performance evidence must be a JSON object"],
      {},
    )

  evidence_class = _first_string(root, [], ("evidence_class", "classification"))
  if evidence_class not in {"system_under_test", "promotion", "comparison"}:
    errors.append("depth performance evidence_class must be explicit")
  workload = _normalize_workload(root.get("workload"))
  if workload not in SCORING_WORKLOADS:
    errors.append(
      "depth performance workload must be independent_decode or long_prefill"
    )
  if root.get("correctness_gates_green") is not True:
    errors.append("depth performance requires correctness_gates_green=true")

  depths = root.get("depths")
  seen_depths: set[int] = set()
  ordered_depths: list[int] = []
  if not isinstance(depths, list) or not depths:
    errors.append("depth performance depths must be a non-empty list")
  else:
    for index, entry in enumerate(depths):
      if not isinstance(entry, dict):
        errors.append("depth performance depth entries must be objects")
        continue
      depth = entry.get("generated_tokens", entry.get("depth"))
      if not isinstance(depth, int):
        errors.append(f"depths[{index}].generated_tokens must be an integer")
        continue
      ordered_depths.append(depth)
      seen_depths.add(depth)
      if entry.get("tokens_match") is not True:
        errors.append(f"depth {depth} must have tokens_match=true")
      tps = entry.get("throughput_tokens_per_second", entry.get("tokens_per_second"))
      if not isinstance(tps, int | float) or tps <= 0:
        errors.append(f"depth {depth} must record positive throughput")
  missing = [depth for depth in (8, 64, 512) if depth not in seen_depths]
  if missing:
    errors.append(
      "depth performance missing ladder depth(s): " + ", ".join(map(str, missing))
    )
  ladder_positions = [
    ordered_depths.index(depth) for depth in (8, 64, 512) if depth in seen_depths
  ]
  if len(ladder_positions) == 3 and ladder_positions != sorted(ladder_positions):
    errors.append("depth performance ladder must be ordered 8 -> 64 -> 512")

  detail = {
    "workload": workload,
    "depths": sorted(seen_depths),
    "depth_order": ordered_depths,
    "depth_count": len(seen_depths),
  }
  return _validation(not errors, errors, detail)


def validate_mmlu_pro_evidence(
  payload: Any,
  *,
  base_dir: Path | None = None,
  expected_model: str | None = None,
  expected_backend: str | None = None,
  required_coverage_percent: float | None = None,
) -> ArtifactValidation:
  errors: list[str] = []
  root = _payload_root(payload, errors, "MMLU Pro evidence")
  if root is None:
    return _validation(False, errors, {})
  if not isinstance(root, dict):
    return _validation(
      False,
      [*errors, "MMLU Pro evidence must be a JSON object"],
      {},
    )

  if root.get("schema") != "ares.benchmark.mmlu_pro.v1":
    errors.append("MMLU Pro schema must be ares.benchmark.mmlu_pro.v1")
  evidence_class = _first_string(root, [], ("evidence_class", "classification"))
  if evidence_class not in {"system_under_test", "promotion", "diagnostic"}:
    errors.append(
      "MMLU Pro evidence_class must be system_under_test, promotion, or diagnostic"
    )
  elif evidence_class == "diagnostic":
    errors.append("MMLU Pro diagnostic evidence cannot satisfy mmlu_pro gate")
  if root.get("status") != "passed":
    errors.append("MMLU Pro status must be passed")

  model = _first_string(root, [], ("model", "model_id", "mmlu_model"))
  backend = _first_string(root, [], ("backend", "backend_id"))
  openai_host = _first_string(root, [], ("openai_host", "endpoint"))
  if model is None:
    errors.append("MMLU Pro evidence must name model")
  elif expected_model is not None and model != expected_model:
    errors.append("MMLU Pro model must match model_spec mmlu_model/model")
  if backend is None:
    errors.append("MMLU Pro evidence must name backend")
  elif expected_backend is not None and backend != expected_backend:
    errors.append("MMLU Pro backend must match model_spec backend")
  if openai_host is None:
    errors.append("MMLU Pro evidence must name openai_host")

  coverage = root.get("coverage_percent", root.get("coverage"))
  if not isinstance(coverage, int | float) or float(coverage) <= 0:
    errors.append("MMLU Pro coverage_percent must be positive")
    coverage_value = None
  else:
    coverage_value = float(coverage)
  has_effective_coverage = "effective_coverage_percent" in root
  effective_coverage = root.get("effective_coverage_percent", coverage)
  if not isinstance(effective_coverage, int | float) or float(effective_coverage) < 0:
    errors.append("MMLU Pro effective_coverage_percent must be non-negative")
    effective_coverage_value = None
  else:
    effective_coverage_value = float(effective_coverage)
  attempted_question_count = root.get("attempted_question_count")
  attempted_question_count_value: int | None = None
  if attempted_question_count is not None:
    if (
      not isinstance(attempted_question_count, int | float)
      or attempted_question_count <= 0
    ):
      errors.append("MMLU Pro attempted_question_count must be positive")
    elif not float(attempted_question_count).is_integer():
      errors.append("MMLU Pro attempted_question_count must be an integer")
    else:
      attempted_question_count_value = int(attempted_question_count)
  question_limit = root.get("question_limit_per_subject")
  question_limit_value: int | None
  if question_limit is None:
    question_limit_value = 0
  elif not isinstance(question_limit, int) or question_limit < 0:
    errors.append("MMLU Pro question_limit_per_subject must be non-negative")
    question_limit_value = None
  else:
    question_limit_value = question_limit

  score = root.get("score_percent", root.get("score"))
  required = root.get(
    "required_score_percent",
    root.get("minimum_score_percent", root.get("threshold_percent")),
  )
  if not isinstance(score, int | float):
    errors.append("MMLU Pro score_percent must be numeric")
    score_value = None
  else:
    score_value = float(score)
  if not isinstance(required, int | float):
    errors.append("MMLU Pro required_score_percent must be numeric")
    required_value = None
  else:
    required_value = float(required)
  if (
    score_value is not None
    and required_value is not None
    and score_value < required_value
  ):
    errors.append("MMLU Pro score_percent must meet required_score_percent")

  endpoint_models = _expect_object(
    errors, root.get("endpoint_models"), "endpoint_models"
  )
  endpoint_model_count = None
  if endpoint_models is not None:
    _require_non_empty_string(
      errors,
      endpoint_models.get("path"),
      "endpoint_models.path",
    )
    _require_sha256(errors, endpoint_models.get("sha256"), "endpoint_models")
    _validate_referenced_sha256(
      errors,
      endpoint_models,
      base_dir,
      "endpoint_models",
    )
    endpoint_host = _first_string(
      endpoint_models,
      [],
      ("openai_host", "endpoint"),
    )
    if endpoint_host is None:
      errors.append("endpoint_models.openai_host must name the probed endpoint")
    elif openai_host is not None and endpoint_host != openai_host:
      errors.append("endpoint_models.openai_host must match top-level openai_host")
    served_models = _expect_string_list(
      errors,
      endpoint_models.get("models"),
      "endpoint_models.models",
    )
    if served_models is not None:
      endpoint_model_count = len(served_models)
      if model is not None and model not in served_models:
        errors.append("endpoint_models.models must include top-level model")

  systems_test = _expect_object(errors, root.get("systems_test"), "systems_test")
  systems_test_config_model = None
  if systems_test is not None:
    _require_non_empty_string(errors, systems_test.get("path"), "systems_test.path")
    _require_git_sha(errors, systems_test.get("commit"), "systems_test.commit")
    if systems_test.get("dirty") is not False:
      errors.append("systems_test.dirty must be false for promotion evidence")
    systems_test_config_model = _first_string(
      systems_test,
      [],
      ("config_model", "mmlu_model", "model"),
    )
    if systems_test_config_model is None:
      errors.append(
        "systems_test.config_model must record the scripts/mmlu_pro.py config model"
      )
    elif model is not None and systems_test_config_model != model:
      errors.append("systems_test.config_model must match top-level model")
    config = _expect_object(
      errors,
      systems_test.get("config"),
      "systems_test.config",
    )
    if config is not None:
      _require_non_empty_string(
        errors,
        config.get("path"),
        "systems_test.config.path",
      )
      _require_sha256(errors, config.get("sha256"), "systems_test.config")
      _validate_referenced_sha256(
        errors,
        config,
        base_dir,
        "systems_test.config",
      )
      _require_non_empty_string(
        errors,
        config.get("source_path"),
        "systems_test.config.source_path",
      )
      if "scripts/mmlu_pro.py" not in str(config.get("source_path", "")):
        errors.append(
          "systems_test.config.source_path must reference scripts/mmlu_pro.py"
        )
      config_model = _first_string(
        config,
        [],
        ("model", "config_model", "mmlu_model"),
      )
      if config_model is None:
        errors.append("systems_test.config.model must name the config row model")
      elif model is not None and config_model != model:
        errors.append("systems_test.config.model must match top-level model")
      nominal_users = config.get("nominal_users")
      if not isinstance(nominal_users, int | float) or nominal_users <= 0:
        errors.append("systems_test.config.nominal_users must be positive")
    command = systems_test.get("command")
    _require_non_empty_string(
      errors,
      command,
      "systems_test.command",
    )
    if not _command_runs_uv_mmlu_pro(command):
      errors.append("systems_test.command must run uv run mmlu_pro")
    if "SKIP_PROVISION=1" not in str(command or ""):
      errors.append("systems_test.command must set SKIP_PROVISION=1")
    command_question_limit = _command_env_non_negative_int(
      errors,
      command,
      "MMLU_MAX_QUESTIONS_PER_SUBJECT",
    )
    if command_question_limit is not None:
      if root.get("question_limit_per_subject") is None:
        question_limit_value = command_question_limit
      elif (
        question_limit_value is not None
        and question_limit_value != command_question_limit
      ):
        errors.append(
          "MMLU Pro question_limit_per_subject must match systems_test.command"
        )

  if question_limit_value and not has_effective_coverage:
    effective_coverage_value = 0.0
  if question_limit_value and (
    effective_coverage_value is not None and effective_coverage_value > 0.0
  ):
    errors.append(
      "MMLU Pro effective_coverage_percent must be zero when "
      "question_limit_per_subject is nonzero"
    )
  coverage_for_requirement = effective_coverage_value
  if question_limit_value:
    errors.append("MMLU Pro question_limit_per_subject must be zero for mmlu_pro gate")
    coverage_for_requirement = 0.0
  if (
    coverage_for_requirement is not None
    and required_coverage_percent is not None
    and coverage_for_requirement < required_coverage_percent
  ):
    errors.append("MMLU Pro effective_coverage_percent must meet required coverage")

  ares = _expect_object(errors, root.get("ares"), "ares")
  if ares is not None:
    _require_git_sha(errors, ares.get("commit"), "ares.commit")
    if ares.get("dirty") is not False:
      errors.append("ares.dirty must be false for promotion evidence")
    ares_backend = _first_string(ares, [], ("backend", "backend_id"))
    if ares_backend is None:
      errors.append("ares.backend must name the selected backend")
    elif backend is not None and ares_backend != backend:
      errors.append("ares.backend must match top-level backend")
    if ares.get("runtime_generated_sidecars") is not False:
      errors.append("ares.runtime_generated_sidecars must be false")
    _require_sha256(
      errors,
      ares.get("ares_plan_sha256", ares.get("ares_plan_hash")),
      "AresPlan",
    )
    _require_sha256(
      errors,
      ares.get("target_plan_sha256", ares.get("target_plan_hash")),
      "TargetPlan",
    )

  subjects = root.get("subjects")
  if not isinstance(subjects, list) or not subjects:
    errors.append("MMLU Pro subjects must be a non-empty list")
  else:
    subject_attempt_sum: int | None = 0
    for index, subject in enumerate(subjects):
      if not isinstance(subject, dict):
        errors.append(f"subjects[{index}] must be an object")
        subject_attempt_sum = None
        continue
      _require_non_empty_string(
        errors,
        subject.get("subject"),
        f"subjects[{index}].subject",
      )
      correct_value: float | None = None
      wrong_value: float | None = None
      for field in ("correct", "wrong", "score_percent"):
        value = subject.get(field)
        if not isinstance(value, int | float) or float(value) < 0:
          errors.append(f"subjects[{index}].{field} must be non-negative")
        elif field == "correct":
          correct_value = float(value)
        elif field == "wrong":
          wrong_value = float(value)
      subject_attempted = subject.get("attempted_question_count")
      subject_attempted_value: int | None = None
      if subject_attempted is not None:
        if (
          not isinstance(subject_attempted, int | float)
          or subject_attempted < 0
          or not float(subject_attempted).is_integer()
        ):
          errors.append(
            f"subjects[{index}].attempted_question_count must be a non-negative integer"
          )
          subject_attempt_sum = None
        else:
          subject_attempted_value = int(subject_attempted)
          if subject_attempt_sum is not None:
            subject_attempt_sum += subject_attempted_value
      elif correct_value is not None and wrong_value is not None:
        derived_attempted = correct_value + wrong_value
        if derived_attempted.is_integer():
          subject_attempted_value = int(derived_attempted)
          if subject_attempt_sum is not None:
            subject_attempt_sum += subject_attempted_value
        else:
          subject_attempt_sum = None
      if (
        subject_attempted_value is not None
        and correct_value is not None
        and wrong_value is not None
        and abs(subject_attempted_value - (correct_value + wrong_value)) > 1e-9
      ):
        errors.append(
          f"subjects[{index}].attempted_question_count must equal correct + wrong"
        )
      result_record_count = subject.get("result_record_count")
      if result_record_count is None:
        errors.append(f"subjects[{index}].result_record_count must be present")
      elif (
        not isinstance(result_record_count, int | float)
        or result_record_count < 0
        or not float(result_record_count).is_integer()
      ):
        errors.append(
          f"subjects[{index}].result_record_count must be a non-negative integer"
        )
      elif (
        subject_attempted_value is not None
        and int(result_record_count) != subject_attempted_value
      ):
        errors.append(
          f"subjects[{index}].result_record_count must equal attempted_question_count"
        )
    if (
      attempted_question_count_value is not None
      and subject_attempt_sum is not None
      and attempted_question_count_value != subject_attempt_sum
    ):
      errors.append(
        "MMLU Pro attempted_question_count must equal sum of subject attempts"
      )

  artifacts = root.get("artifacts")
  if not isinstance(artifacts, list) or not artifacts:
    errors.append("MMLU Pro artifacts must be a non-empty list")
  else:
    for index, artifact in enumerate(artifacts):
      if not isinstance(artifact, dict):
        errors.append(f"artifacts[{index}] must be an object")
        continue
      _require_non_empty_string(
        errors,
        artifact.get("path"),
        f"artifacts[{index}].path",
      )
      _require_sha256(errors, artifact.get("sha256"), f"artifacts[{index}]")
      _validate_referenced_sha256(
        errors,
        artifact,
        base_dir,
        f"artifacts[{index}]",
      )

  detail = {
    "model": model,
    "backend": backend,
    "coverage_percent": coverage_value,
    "effective_coverage_percent": effective_coverage_value,
    "question_limit_per_subject": question_limit_value,
    "score_percent": score_value,
    "required_score_percent": required_value,
    "required_coverage_percent": required_coverage_percent,
    "endpoint_model_count": endpoint_model_count,
    "systems_test_config_model": systems_test_config_model,
    "subject_count": len(subjects) if isinstance(subjects, list) else None,
    "artifact_count": len(artifacts) if isinstance(artifacts, list) else None,
  }
  return _validation(not errors, errors, detail)


def validate_introspection_ladder_report(
    payload: Any,
    *,
    base_dir: Path | None = None,
) -> ArtifactValidation:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return _validation(
            False,
            ["introspection ladder report must be a JSON object"],
            {},
        )

    if payload.get("schema") != INTROSPECTION_LADDER_SCHEMA:
        errors.append(
            f"introspection ladder schema must be {INTROSPECTION_LADDER_SCHEMA}"
        )
    if payload.get("evidence_class") != "semantic_localization":
        errors.append(
            "introspection ladder evidence_class must be semantic_localization"
        )

    status = payload.get("status")
    if status not in {"passed", "failed"}:
        errors.append("introspection ladder status must be passed or failed")

    stage_order = payload.get("stage_order")
    if not isinstance(stage_order, list) or not stage_order:
        errors.append("introspection ladder stage_order must be a non-empty list")
        stage_names: list[str] = []
    else:
        stage_names = []
        for index, stage in enumerate(stage_order):
            if not isinstance(stage, str) or stage not in INTROSPECTION_STAGES:
                errors.append(f"stage_order[{index}] is not a known ladder stage")
                continue
            stage_names.append(stage)

    comparisons = payload.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        errors.append("introspection ladder comparisons must be a non-empty list")
    else:
        for index, comparison in enumerate(comparisons):
            _validate_introspection_comparison_ref(
                errors,
                comparison,
                base_dir,
                f"comparisons[{index}]",
            )

    runs = payload.get("runs")
    if not isinstance(runs, list):
        errors.append("introspection ladder runs must be a list")
    else:
        for index, run in enumerate(runs):
            _validate_introspection_artifact_ref(
                errors,
                run,
                base_dir,
                f"runs[{index}]",
                expected_schema="ares.introspection.run.v1",
            )

    graphs = payload.get("graphs")
    if not isinstance(graphs, list):
        errors.append("introspection ladder graphs must be a list")
    else:
        for index, graph in enumerate(graphs):
            _validate_introspection_artifact_ref(
                errors,
                graph,
                base_dir,
                f"graphs[{index}]",
                expected_schema="ares.introspection.graph.v1",
                verify_path=False,
            )

    first_failing_stage = payload.get("first_failing_stage")
    next_owner = payload.get("next_owner")
    if status == "failed":
        if (
            not isinstance(first_failing_stage, str)
            or first_failing_stage not in INTROSPECTION_STAGES
        ):
            errors.append(
                "failed introspection ladder must name a known first_failing_stage"
            )
        elif stage_names and first_failing_stage not in stage_names:
            errors.append("first_failing_stage must appear in stage_order")
        if not isinstance(next_owner, str) or next_owner in {"", "none"}:
            errors.append("failed introspection ladder must name next_owner")
    elif status == "passed":
        if first_failing_stage is not None:
            errors.append("passed introspection ladder must not name a failing stage")
        if next_owner != "none":
            errors.append("passed introspection ladder next_owner must be none")

    if isinstance(next_owner, str) and next_owner not in INTROSPECTION_OWNERS:
        errors.append("introspection ladder next_owner is not a known owner")
    _require_non_empty_string(
        errors,
        payload.get("recommendation"),
        "introspection ladder recommendation",
    )
    trace_context = payload.get("trace_context")
    if "trace_context" in payload:
        if not isinstance(trace_context, Mapping):
            errors.append("introspection ladder trace_context must be a JSON object")
        else:
            _validate_introspection_trace_context(
                errors,
                trace_context,
                base_dir,
            )

    trace_context_detail = (
        _introspection_trace_context_detail(trace_context)
        if isinstance(trace_context, Mapping)
        else {}
    )
    comparison_details = _introspection_ladder_comparison_details(comparisons)
    first_failed_comparison = _introspection_first_failed_comparison_detail(comparisons)
    detail = {
        "schema": payload.get("schema"),
        "evidence_class": payload.get("evidence_class"),
        "status": status,
        "first_failing_stage": first_failing_stage,
        "next_owner": next_owner,
        "stage_order": stage_names,
        "comparison_count": len(comparisons) if isinstance(comparisons, list) else 0,
        "run_count": len(runs) if isinstance(runs, list) else 0,
        "graph_count": len(graphs) if isinstance(graphs, list) else 0,
        "trace_context": trace_context_detail,
        "comparisons": comparison_details,
    }
    if first_failed_comparison is not None:
        detail["first_failed_comparison"] = first_failed_comparison
        if isinstance(first_failed_comparison.get("first_mismatch"), Mapping):
            detail["first_mismatch"] = first_failed_comparison["first_mismatch"]
    return _validation(not errors, errors, detail)


def validate_token_agreement_evidence(
  payload: Any,
  *,
  base_dir: Path | None = None,
) -> ArtifactValidation:
  errors: list[str] = []
  root = _payload_root(payload, errors, "eight-token greedy evidence")
  if root is None:
    return _validation(False, errors, {})
  if not isinstance(root, dict):
    return _validation(
      False,
      [*errors, "eight-token greedy evidence must be a JSON object"],
      {},
    )

  schema = root.get("schema")
  if schema == EIGHT_TOKEN_PREFLIGHT_REPORT_SCHEMA:
    return _validation(
      False,
      [
        "eight-token preflight reports are diagnostic and cannot close "
        "eight_token_greedy"
      ],
      {
        "schema": schema,
        "evidence_class": _first_string(root, [], ("evidence_class", "classification")),
      },
    )
  if schema != EIGHT_TOKEN_GREEDY_SCHEMA:
    errors.append(f"eight-token evidence schema must be {EIGHT_TOKEN_GREEDY_SCHEMA}")
  evidence_class = _first_string(root, [], ("evidence_class", "classification"))
  if evidence_class not in {"system_under_test", "promotion"}:
    errors.append("eight-token evidence_class must be system_under_test or promotion")
  if _first_string(root, [], ("oracle", "oracle_source")) != HF_CPU_ORACLE_KIND:
    errors.append(f"eight-token oracle must be {HF_CPU_ORACLE_KIND}")
  candidate = _first_string(root, [], ("candidate", "subject", "runtime"))
  if candidate not in ARES_RUNTIME_CANDIDATES:
    errors.append("eight-token candidate must identify Ares system-under-test output")
  if root.get("decode_strategy") != "greedy":
    errors.append("eight-token decode_strategy must be greedy")
  model_id = _validate_model_provenance(
    errors,
    root.get("model_provenance"),
    "eight-token model_provenance",
  )

  expected = root.get("expected_generated_tokens", 8)
  if not isinstance(expected, int) or expected < 8:
    errors.append("eight-token expected_generated_tokens must be an integer >= 8")
    expected = 8
  reference_generated = _int_list(root.get("reference_generated_token_ids"))
  candidate_generated = _int_list(root.get("candidate_generated_token_ids"))
  if reference_generated is None:
    errors.append(
      "eight-token reference_generated_token_ids must be a list of integers"
    )
  if candidate_generated is None:
    errors.append(
      "eight-token candidate_generated_token_ids must be a list of integers"
    )

  generated = root.get("generated_tokens")
  if candidate_generated is not None:
    if not isinstance(generated, int):
      errors.append("eight-token generated_tokens must be an integer")
    elif generated != len(candidate_generated):
      errors.append(
        "eight-token generated_tokens must match candidate_generated_token_ids length"
      )
  elif not isinstance(generated, int):
    errors.append("eight-token generated_tokens must be an integer")
  if isinstance(generated, int) and generated < expected:
    errors.append(
      f"eight-token evidence generated {generated} token(s), expected at least {expected}"
    )
  if (
    reference_generated is not None
    and candidate_generated is not None
    and reference_generated != candidate_generated
  ):
    errors.append(
      "eight-token reference_generated_token_ids and candidate_generated_token_ids must match"
    )

  score = root.get("score")
  exact_fraction = root.get("exact_fraction")
  top1 = root.get("top1_agreement")
  if root.get("exact_match") is not True:
    errors.append("eight-token exact_match must be true")
  if not isinstance(score, int | float) or float(score) < 1.0:
    errors.append("eight-token score must be 1.0")
  if not isinstance(exact_fraction, int | float) or float(exact_fraction) < 1.0:
    errors.append("eight-token exact_fraction must be 1.0")
  if not isinstance(top1, int | float) or float(top1) < 1.0:
    errors.append("eight-token top1_agreement must be 1.0")

  reference = _expect_object(errors, root.get("reference"), "eight-token reference")
  if reference is not None:
    _require_non_empty_string(errors, reference.get("path"), "reference.path")
    _require_sha256(errors, reference.get("sha256"), "reference.sha256")
    _validate_referenced_sha256(errors, reference, base_dir, "reference")
  candidate_output = _expect_object(
    errors, root.get("candidate_output"), "eight-token candidate_output"
  )
  if candidate_output is not None:
    _require_non_empty_string(
      errors, candidate_output.get("path"), "candidate_output.path"
    )
    _require_sha256(errors, candidate_output.get("sha256"), "candidate_output.sha256")
    _validate_referenced_sha256(errors, candidate_output, base_dir, "candidate_output")
    runtime = candidate_output.get("runtime")
    if runtime is not None and runtime not in ARES_RUNTIME_CANDIDATES:
      errors.append("candidate_output.runtime must identify Ares")
  artifact_keys = _validate_eight_token_artifacts(
    errors,
    root.get("artifacts"),
    base_dir,
  )

  cases = root.get("cases")
  case_count = len(cases) if isinstance(cases, list) else 0
  if case_count == 0:
    errors.append("eight-token evidence must include at least one case")
  elif isinstance(cases, list):
    for index, case in enumerate(cases):
      if not isinstance(case, dict):
        errors.append(f"cases[{index}] must be an object")
        continue
      if case.get("exact_match") is not True:
        errors.append(f"cases[{index}].exact_match must be true")
      candidate_length = case.get("candidate_length")
      if not isinstance(candidate_length, int) or candidate_length < expected:
        errors.append(f"cases[{index}].candidate_length must be at least {expected}")

  detail = {
    "expected_generated_tokens": expected,
    "generated_tokens": generated,
    "case_count": case_count,
    "candidate": candidate,
    "model_id": model_id,
    "artifact_keys": artifact_keys,
  }
  return _validation(not errors, errors, detail)


def validate_target_plan(plan: Any) -> ArtifactValidation:
  errors: list[str] = []
  if not isinstance(plan, dict):
    return _validation(False, ["TargetPlan must be a JSON object"], {})

  _require_fields(
    errors,
    plan,
    (
      "schema_version",
      "producer",
      "backend_id",
      "model_id",
      "source",
      "declared_runtime_bindings",
      "hw_policy",
      "operations",
    ),
    "TargetPlan",
  )
  if plan.get("schema_version") != 1:
    errors.append("TargetPlan.schema_version must be 1")
  _require_non_empty_string(errors, plan.get("backend_id"), "TargetPlan.backend_id")
  _require_non_empty_string(errors, plan.get("model_id"), "TargetPlan.model_id")

  producer = _expect_object(errors, plan.get("producer"), "TargetPlan.producer")
  if producer is not None:
    for field, expected in LEAN_TARGET_PLAN_PRODUCER.items():
      if producer.get(field) != expected:
        errors.append(f"TargetPlan.producer.{field} must be {expected}")

  source = _expect_object(errors, plan.get("source"), "TargetPlan.source")
  source_statement_count = None
  if source is not None:
    _require_fields(
      errors,
      source,
      ("schema_version", "statement_count", "config", "provenance"),
      "TargetPlan.source",
    )
    if not isinstance(source.get("schema_version"), int):
      errors.append("TargetPlan.source.schema_version must be an integer")
    source_statement_count = source.get("statement_count")
    if not isinstance(source_statement_count, int) or source_statement_count <= 0:
      errors.append("TargetPlan.source.statement_count must be positive")
    _expect_object(errors, source.get("config"), "TargetPlan.source.config")
    source_provenance = _expect_object(
      errors,
      source.get("provenance"),
      "TargetPlan.source.provenance",
    )
    if source_provenance is not None:
      _require_fields(
        errors,
        source_provenance,
        ("fx_hash", "rule_corpus_hash", "emitter_version"),
        "TargetPlan.source.provenance",
      )
      _require_non_empty_string(
        errors,
        source_provenance.get("emitter_version"),
        "TargetPlan.source.provenance.emitter_version",
      )
      emitter = source_provenance.get("emitter_version")
      if isinstance(emitter, str) and "ingest-lean" not in emitter:
        errors.append(
          "TargetPlan.source.provenance.emitter_version must name ingest-lean"
        )

  declared_bindings = _expect_string_list(
    errors,
    plan.get("declared_runtime_bindings"),
    "TargetPlan.declared_runtime_bindings",
  )
  if declared_bindings is not None and not declared_bindings:
    errors.append("TargetPlan.declared_runtime_bindings must be non-empty")
  _expect_object(errors, plan.get("hw_policy"), "TargetPlan.hw_policy")
  operations = plan.get("operations")
  semantic_count = 0
  runtime_binding_names: set[str] = set()
  if not isinstance(operations, list) or not operations:
    errors.append("TargetPlan.operations must be a non-empty list")
  else:
    for index, operation in enumerate(operations):
      semantic_count += _validate_target_operation(
        errors,
        operation,
        index,
        runtime_binding_names,
      )
  if source_statement_count is not None and semantic_count != source_statement_count:
    errors.append(
      "TargetPlan semantic operation count must match source.statement_count"
    )
  if declared_bindings is not None:
    missing_bindings = sorted(set(declared_bindings).difference(runtime_binding_names))
    if missing_bindings:
      errors.append(
        "TargetPlan runtime binding operations missing: " + ", ".join(missing_bindings)
      )

  detail = {
    "schema_version": plan.get("schema_version"),
    "backend_id": plan.get("backend_id"),
    "model_id": plan.get("model_id"),
    "operation_count": len(operations) if isinstance(operations, list) else None,
    "semantic_operation_count": semantic_count,
    "runtime_binding_count": len(runtime_binding_names),
  }
  return _validation(not errors, errors, detail)


def validate_artifact_consistency(
  spec: Mapping[str, Any],
  *,
  oracle_payload: Any,
  validated_gates: Mapping[str, Any],
) -> ArtifactValidation:
  errors: list[str] = []
  expected_ids = _expected_model_ids(spec)
  oracle_ids = _oracle_model_ids(oracle_payload)
  target_model_id = _gate_detail_string(
    validated_gates.get("targetplan_valid"), "model_id"
  )

  if not expected_ids:
    errors.append("model_spec must name model or expected_model_ids")
  if not oracle_ids:
    errors.append("HF CPU oracle model_id is missing")
  if target_model_id is None:
    errors.append("TargetPlan model_id is missing")

  unexpected_oracle_ids = sorted(oracle_ids.difference(expected_ids))
  if unexpected_oracle_ids:
    errors.append(
      "HF CPU oracle model_id not allowed by model_spec: "
      + ", ".join(unexpected_oracle_ids)
    )
  if target_model_id is not None and target_model_id not in expected_ids:
    errors.append("TargetPlan model_id not allowed by model_spec: " + target_model_id)

  detail = {
    "expected_model_ids": sorted(expected_ids),
    "oracle_model_ids": sorted(oracle_ids),
    "target_plan_model_id": target_model_id,
  }
  return _validation(not errors, errors, detail)


def validate_trace_report_json(
    report: Any,
    *,
    report_path: Path | None = None,
    authority_root: Path | None = None,
) -> ArtifactValidation:
    errors: list[str] = []
    if not isinstance(report, dict):
        return _validation(False, ["trace report must be a JSON object"], {})

    if report.get("schema_version") != TRACE_REPORT_SCHEMA_VERSION:
        errors.append(
            f"trace report schema_version must be {TRACE_REPORT_SCHEMA_VERSION}"
        )
    _require_non_empty_string(errors, report.get("title"), "trace report title")
    inputs = _expect_object(errors, report.get("inputs"), "trace report inputs")
    sections = _expect_object(errors, report.get("sections"), "trace report sections")

    section_names: list[str] = []
    report_grade_rows: list[dict[str, Any]] = []
    report_triage_rows: list[dict[str, Any]] = []
    preflight_rows: list[dict[str, Any]] = []
    analysis_command_rows: list[dict[str, Any]] = []
    answerability_rows: list[dict[str, Any]] = []
    capture_rows: list[dict[str, Any]] = []
    run_provenance_rows: list[dict[str, Any]] = []
    artifact_identity_rows: list[dict[str, Any]] = []
    artifact_identity_check_rows: list[dict[str, Any]] = []
    capture_capability_rows: list[dict[str, Any]] = []
    supported_claim_rows: list[dict[str, Any]] = []
    unsupported_claim_rows: list[dict[str, Any]] = []
    next_measurement_rows: list[dict[str, Any]] = []
    correctness_evidence_rows: list[dict[str, Any]] = []
    evidence_artifact_check_rows: list[dict[str, Any]] = []
    promotion_gate_summary_rows: list[dict[str, Any]] = []
    trace_mode_guardrail_rows: list[dict[str, Any]] = []
    ab_provenance_rows: list[dict[str, Any]] = []
    ab_comparability_rows: list[dict[str, Any]] = []
    ab_coverage_rows: list[dict[str, Any]] = []
    ab_repeatability_rows: list[dict[str, Any]] = []
    report_json_section_rows: list[dict[str, Any]] = []
    report_section_inventory_rows: list[dict[str, Any]] = []
    preflight_finding_rows: list[str] = []
    evidence_classification_rows: list[str] = []
    trace_config_rows: list[dict[str, Any]] = []
    provider_payload_boundary_rows: list[dict[str, Any]] = []
    trace_event_artifact_rows: list[dict[str, Any]] = []
    backend_event_artifact_rows: list[dict[str, Any]] = []
    backend_event_rows: list[dict[str, Any]] = []
    backend_provider_boundary_rows: list[dict[str, Any]] = []
    backend_fail_closed_root_cause_rows: list[dict[str, Any]] = []
    debug_payload_artifact_summary_rows: list[dict[str, Any]] = []
    token_quality_summary_rows: list[dict[str, Any]] = []
    oracle_reference_summary_rows: list[dict[str, Any]] = []
    planning_decision_sidecar_rows: list[dict[str, Any]] = []
    token_quality_sidecar_rows: list[dict[str, Any]] = []
    topk_token_sidecar_rows: list[dict[str, Any]] = []
    tensor_payload_sidecar_rows: list[dict[str, Any]] = []
    kv_payload_digest_sidecar_rows: list[dict[str, Any]] = []
    logit_slice_sidecar_rows: list[dict[str, Any]] = []
    activation_digest_sidecar_rows: list[dict[str, Any]] = []
    device_result_digest_sidecar_rows: list[dict[str, Any]] = []
    scheduler_packet_lineage_sidecar_rows: list[dict[str, Any]] = []
    scheduler_kv_shard_lifecycle_sidecar_rows: list[dict[str, Any]] = []
    scheduler_listener_sparse_logit_sidecar_rows: list[dict[str, Any]] = []
    device_dma_lifecycle_sidecar_rows: list[dict[str, Any]] = []
    attention_page_trace_sidecar_rows: list[dict[str, Any]] = []
    introspection_artifact_rows: list[dict[str, Any]] = []
    introspection_capability_rows: list[dict[str, Any]] = []
    introspection_artifact_summary_rows: list[dict[str, Any]] = []
    introspection_section_inventory_rows: list[dict[str, Any]] = []
    timeline_query_summary_rows: list[dict[str, Any]] = []
    if sections is not None:
        section_names = sorted(str(name) for name in sections)
        for name in TRACE_REPORT_REQUIRED_SECTIONS:
            if name not in sections:
                errors.append(f"trace report sections missing required section: {name}")
        preflight_rows = _trace_report_section_rows(errors, sections, "preflight")
        analysis_command_rows = _trace_report_section_rows(
            errors, sections, "analysis_commands"
        )
        report_grade_rows = _trace_report_section_rows(errors, sections, "report_grade")
        report_triage_rows = _trace_report_section_rows(
            errors,
            sections,
            "report_triage",
        )
        _validate_trace_report_triage_rows(errors, report_triage_rows)
        answerability_rows = _trace_report_section_rows(
            errors, sections, "answerability"
        )
        capture_rows = _trace_report_section_rows(
            errors,
            sections,
            "capture",
            required=False,
        )
        run_provenance_rows = _trace_report_section_rows(
            errors,
            sections,
            "run_provenance",
            required=False,
        )
        artifact_identity_rows = _trace_report_section_rows(
            errors,
            sections,
            "artifact_identities",
            required=False,
        )
        artifact_identity_check_rows = _trace_report_section_rows(
            errors,
            sections,
            "artifact_identity_checks",
            required=False,
        )
        capture_capability_rows = _trace_report_section_rows(
            errors,
            sections,
            "capture_capabilities",
            required=False,
        )
        supported_claim_rows = _trace_report_section_rows(
            errors,
            sections,
            "supported_claims",
            required=False,
        )
        unsupported_claim_rows = _trace_report_section_rows(
            errors, sections, "unsupported_claims"
        )
        next_measurement_rows = _trace_report_section_rows(
            errors, sections, "next_measurements"
        )
        correctness_evidence_rows = _trace_report_section_rows(
            errors,
            sections,
            "correctness_evidence",
            required=False,
        )
        evidence_artifact_check_rows = _trace_report_section_rows(
            errors,
            sections,
            "evidence_artifact_checks",
            required=False,
        )
        promotion_gate_summary_rows = _trace_report_section_rows(
            errors,
            sections,
            "promotion_gate_summary",
            required=False,
        )
        trace_mode_guardrail_rows = _trace_report_section_rows(
            errors,
            sections,
            "trace_mode_guardrails",
            required=False,
        )
        ab_provenance_rows = _trace_report_section_rows(
            errors,
            sections,
            "ab_provenance",
            required=False,
        )
        ab_comparability_rows = _trace_report_section_rows(
            errors,
            sections,
            "ab_comparability",
            required=False,
        )
        ab_coverage_rows = _trace_report_section_rows(
            errors,
            sections,
            "ab_coverage",
            required=False,
        )
        ab_repeatability_rows = _trace_report_section_rows(
            errors,
            sections,
            "ab_repeatability",
            required=False,
        )
        report_json_section_rows = _trace_report_section_rows(
            errors,
            sections,
            "report_json_section_inventory",
            required=False,
        )
        report_section_inventory_rows = _trace_report_section_rows(
            errors,
            sections,
            "report_section_inventory",
            required=False,
        )
        preflight_finding_rows = _trace_report_string_rows(
            errors,
            sections,
            "preflight_findings",
            required=False,
        )
        evidence_classification_rows = _trace_report_string_rows(
            errors,
            sections,
            "evidence_classification",
            required=False,
        )
        trace_config_rows = _trace_report_section_rows(
            errors,
            sections,
            "trace_config_rows",
            required=False,
        )
        provider_payload_boundary_rows = _trace_report_section_rows(
            errors,
            sections,
            "provider_payload_boundary_inventory_rows",
            required=False,
        )
        trace_event_artifact_rows = _trace_report_section_rows(
            errors,
            sections,
            "trace_event_artifacts",
            required=False,
        )
        backend_event_artifact_rows = _trace_report_section_rows(
            errors,
            sections,
            "backend_event_artifacts",
            required=False,
        )
        backend_event_rows = _trace_report_section_rows(
            errors,
            sections,
            "backend_event_rows",
            required=False,
        )
        backend_provider_boundary_rows = _trace_report_section_rows(
            errors,
            sections,
            "backend_provider_boundaries",
            required=False,
        )
        backend_fail_closed_root_cause_rows = _trace_report_section_rows(
            errors,
            sections,
            "backend_fail_closed_root_causes",
            required=False,
        )
        debug_payload_artifact_summary_rows = _trace_report_section_rows(
            errors,
            sections,
            "debug_payload_artifact_summary_rows",
            required=False,
        )
        token_quality_summary_rows = _trace_report_section_rows(
            errors,
            sections,
            "token_quality_summary_rows",
            required=False,
        )
        oracle_reference_summary_rows = _trace_report_section_rows(
            errors,
            sections,
            "oracle_reference_summary_rows",
            required=False,
        )
        planning_decision_sidecar_rows = _trace_report_section_rows(
            errors,
            sections,
            "planning_decision_sidecar_rows",
            required=False,
        )
        token_quality_sidecar_rows = _trace_report_section_rows(
            errors,
            sections,
            "token_quality_sidecar_rows",
            required=False,
        )
        topk_token_sidecar_rows = _trace_report_section_rows(
            errors,
            sections,
            "topk_token_sidecar_rows",
            required=False,
        )
        tensor_payload_sidecar_rows = _trace_report_section_rows(
            errors,
            sections,
            "tensor_payload_sidecar_rows",
            required=False,
        )
        kv_payload_digest_sidecar_rows = _trace_report_section_rows(
            errors,
            sections,
            "kv_payload_digest_sidecar_rows",
            required=False,
        )
        logit_slice_sidecar_rows = _trace_report_section_rows(
            errors,
            sections,
            "logit_slice_sidecar_rows",
            required=False,
        )
        activation_digest_sidecar_rows = _trace_report_section_rows(
            errors,
            sections,
            "activation_digest_sidecar_rows",
            required=False,
        )
        device_result_digest_sidecar_rows = _trace_report_section_rows(
            errors,
            sections,
            "device_result_digest_sidecar_rows",
            required=False,
        )
        scheduler_packet_lineage_sidecar_rows = _trace_report_section_rows(
            errors,
            sections,
            "scheduler_packet_lineage_sidecar_rows",
        )
        _validate_scheduler_packet_lineage_report_rows(
            errors,
            scheduler_packet_lineage_sidecar_rows,
        )
        _validate_scheduler_packet_lineage_artifact_binding(
            errors,
            report,
            scheduler_packet_lineage_sidecar_rows,
            report_path,
            authority_root,
        )
        scheduler_kv_shard_lifecycle_sidecar_rows = _trace_report_section_rows(
            errors,
            sections,
            "scheduler_kv_shard_lifecycle_sidecar_rows",
            required=False,
        )
        scheduler_listener_sparse_logit_sidecar_rows = _trace_report_section_rows(
            errors,
            sections,
            "scheduler_listener_sparse_logit_sidecar_rows",
            required=False,
        )
        device_dma_lifecycle_sidecar_rows = _trace_report_section_rows(
            errors,
            sections,
            "device_dma_lifecycle_sidecar_rows",
            required=False,
        )
        attention_page_trace_sidecar_rows = _trace_report_section_rows(
            errors,
            sections,
            "attention_page_trace_sidecar_rows",
            required=False,
        )
        introspection_artifact_rows = _trace_report_section_rows(
            errors,
            sections,
            "introspection_artifacts",
            required=False,
        )
        introspection_capability_rows = _trace_report_section_rows(
            errors,
            sections,
            "introspection_capability_rows",
            required=False,
        )
        introspection_artifact_summary_rows = _trace_report_section_rows(
            errors,
            sections,
            "introspection_artifact_summary_rows",
            required=False,
        )
        introspection_section_inventory_rows = _trace_report_section_rows(
            errors,
            sections,
            "introspection_section_inventory",
            required=False,
        )
        timeline_query_summary_rows = _trace_report_section_rows(
            errors,
            sections,
            "timeline_query_summary",
            required=False,
        )

    if not preflight_rows:
        errors.append("trace report sections.preflight must include at least one row")
    if not analysis_command_rows:
        errors.append(
            "trace report sections.analysis_commands must include at least one row"
        )
    if not report_grade_rows:
        errors.append(
            "trace report sections.report_grade must include at least one row"
        )
    if not answerability_rows:
        errors.append(
            "trace report sections.answerability must include at least one row"
        )
    if not scheduler_packet_lineage_sidecar_rows:
        errors.append(
            "trace report sections.scheduler_packet_lineage_sidecar_rows "
            "must include at least one row"
        )

    first_grade = report_grade_rows[0] if report_grade_rows else {}
    first_preflight = preflight_rows[0] if preflight_rows else {}
    answerability_status_counts = _trace_report_value_counts(
        answerability_rows,
        "status",
    )
    provider_payload_route_only_rows = [
        row
        for row in provider_payload_boundary_rows
        if row.get("producer_contract") == "runtime_sidecar_route_only"
        or row.get("producer_status") == "runtime_route_only_no_provider_producer"
        or row.get("capture_status") == "route_available_no_provider_producer"
    ]
    provider_payload_recorded_rows = [
        row
        for row in provider_payload_boundary_rows
        if row.get("capture_status") == "recorded_artifact"
        or _trace_report_int(row.get("matching_provider_artifact_count")) > 0
        or _trace_report_int(row.get("artifact_count")) > 0
    ]
    report_json_section_sample_rows = [
        row
        for row in report_json_section_rows
        if row.get("json_section") in TRACE_REPORT_JSON_SECTION_SAMPLE_KEYS
    ]
    if not report_json_section_sample_rows:
        report_json_section_sample_rows = report_json_section_rows
    ab_coverage_row = ab_coverage_rows[0] if ab_coverage_rows else {}
    ab_repeatability_row = ab_repeatability_rows[0] if ab_repeatability_rows else {}
    introspection_artifact_row_count_total = 0
    introspection_artifact_byte_count_total = 0
    for index, row in enumerate(introspection_artifact_rows):
        introspection_artifact_row_count_total += _trace_report_optional_count(
            errors,
            row.get("row_count"),
            f"trace report sections.introspection_artifacts[{index}].row_count",
        )
        introspection_artifact_byte_count_total += _trace_report_optional_count(
            errors,
            row.get("byte_count"),
            f"trace report sections.introspection_artifacts[{index}].byte_count",
        )
    preflight_ok_count = _trace_report_optional_count(
        errors,
        first_preflight.get("ok"),
        "trace report sections.preflight[0].ok",
    )
    preflight_warn_count = _trace_report_optional_count(
        errors,
        first_preflight.get("warn"),
        "trace report sections.preflight[0].warn",
    )
    preflight_fail_count = _trace_report_optional_count(
        errors,
        first_preflight.get("fail"),
        "trace report sections.preflight[0].fail",
    )

    detail = {
        "schema_version": report.get("schema_version"),
        "title": report.get("title"),
        "metadata": inputs.get("metadata") if inputs is not None else None,
        "trace": inputs.get("trace") if inputs is not None else None,
        "section_count": len(section_names),
        "section_names": section_names,
        "preflight_status": first_preflight.get("status"),
        "preflight_ok_count": preflight_ok_count,
        "preflight_warn_count": preflight_warn_count,
        "preflight_fail_count": preflight_fail_count,
        "report_grade": first_grade.get("report_grade"),
        "proof_grade_status": first_grade.get("proof_grade_status"),
        "report_grade_basis": first_grade.get("basis"),
        "report_grade_promotion_gate": first_grade.get("promotion_gate"),
        "report_triage_count": len(report_triage_rows),
        "report_triage_status_counts": _trace_report_value_counts(
            report_triage_rows,
            "triage_status",
        ),
        "report_triage_samples": _trace_report_samples(
            report_triage_rows,
            (
                "triage_status",
                "report_grade",
                "first_blocked_gate",
                "first_blocked_gate_status",
                "first_next_measurement_priority",
                "first_next_measurement",
                "first_useful_section",
                "first_action",
                "claim_boundary",
            ),
            limit=4,
        ),
        "answerability_count": len(answerability_rows),
        "answerability_status_counts": dict(
            sorted(answerability_status_counts.items())
        ),
        "supported_claim_count": len(supported_claim_rows),
        "unsupported_claim_count": len(unsupported_claim_rows),
        "next_measurement_count": len(next_measurement_rows),
        "correctness_evidence_count": len(correctness_evidence_rows),
        "correctness_evidence_status_counts": _trace_report_value_counts(
            correctness_evidence_rows,
            "status",
        ),
        "correctness_evidence_proof_grade_status_counts": (
            _trace_report_value_counts(
                correctness_evidence_rows,
                "proof_grade_status",
            )
        ),
        "evidence_artifact_check_count": len(evidence_artifact_check_rows),
        "evidence_artifact_check_status_counts": _trace_report_value_counts(
            evidence_artifact_check_rows,
            "status",
        ),
        "promotion_gate_summary_count": len(promotion_gate_summary_rows),
        "promotion_gate_summary_status_counts": _trace_report_value_counts(
            promotion_gate_summary_rows,
            "status",
        ),
        "promotion_gate_summary_proof_grade_status_counts": (
            _trace_report_value_counts(
                promotion_gate_summary_rows,
                "proof_grade_status",
            )
        ),
        "trace_mode_guardrail_count": len(trace_mode_guardrail_rows),
        "trace_mode_guardrail_mode_counts": _trace_report_value_counts(
            trace_mode_guardrail_rows,
            "trace_mode",
        ),
        "trace_mode_guardrail_overhead_counts": _trace_report_value_counts(
            trace_mode_guardrail_rows,
            "overhead_boundary",
        ),
        "ab_provenance_count": len(ab_provenance_rows),
        "ab_provenance_status_counts": _trace_report_value_counts(
            ab_provenance_rows,
            "provenance_status",
        ),
        "ab_provenance_source_state_counts": _trace_report_value_counts(
            ab_provenance_rows,
            "source_state",
        ),
        "ab_provenance_artifact_hash_status_counts": _trace_report_value_counts(
            ab_provenance_rows,
            "artifact_hash_status",
        ),
        "ab_provenance_proof_grade_status_counts": _trace_report_value_counts(
            ab_provenance_rows,
            "proof_grade_status",
        ),
        "ab_comparability_count": len(ab_comparability_rows),
        "ab_comparability_status_counts": _trace_report_value_counts(
            ab_comparability_rows,
            "status",
        ),
        "ab_coverage_count": len(ab_coverage_rows),
        "ab_coverage_status_counts": _trace_report_value_counts(
            ab_coverage_rows,
            "coverage_status",
        ),
        "ab_coverage_total_rows": _trace_report_int(ab_coverage_row.get("total_rows")),
        "ab_coverage_matched_rows": _trace_report_int(
            ab_coverage_row.get("matched_rows")
        ),
        "ab_coverage_baseline_only_rows": _trace_report_int(
            ab_coverage_row.get("baseline_only_rows")
        ),
        "ab_coverage_candidate_only_rows": _trace_report_int(
            ab_coverage_row.get("candidate_only_rows")
        ),
        "ab_repeatability_count": len(ab_repeatability_rows),
        "ab_repeatability_status_counts": _trace_report_value_counts(
            ab_repeatability_rows,
            "status",
        ),
        "ab_repeatability_proof_grade_status_counts": _trace_report_value_counts(
            ab_repeatability_rows,
            "proof_grade_status",
        ),
        "ab_repeatability_baseline_runs": _trace_report_int(
            ab_repeatability_row.get("baseline_runs")
        ),
        "ab_repeatability_candidate_runs": _trace_report_int(
            ab_repeatability_row.get("candidate_runs")
        ),
        "ab_repeatability_required_matched_runs_for_hardware_proof": (
            _trace_report_int(
                ab_repeatability_row.get("required_matched_runs_for_hardware_proof")
            )
        ),
        "ab_repeatability_matched_rows": _trace_report_int(
            ab_repeatability_row.get("matched_rows")
        ),
        "report_json_section_count": len(report_json_section_rows),
        "report_json_section_kind_counts": _trace_report_value_counts(
            report_json_section_rows,
            "section_kind",
        ),
        "report_section_inventory_count": len(report_section_inventory_rows),
        "report_section_inventory_native_sql_counts": _trace_report_value_counts(
            report_section_inventory_rows,
            "native_sql",
        ),
        "preflight_finding_count": len(preflight_finding_rows),
        "preflight_finding_kind_counts": _trace_report_string_prefix_counts(
            preflight_finding_rows,
        ),
        "evidence_classification_count": len(evidence_classification_rows),
        "evidence_classification_kind_counts": _trace_report_string_prefix_counts(
            evidence_classification_rows,
        ),
        "capture_count": len(capture_rows),
        "capture_process_kind_counts": _trace_report_value_counts(
            capture_rows,
            "process_kind",
        ),
        "capture_backend_counts": _trace_report_value_counts(
            capture_rows,
            "backend_id",
        ),
        "capture_trace_mode_counts": _trace_report_value_counts(
            capture_rows,
            "trace_mode",
        ),
        "run_provenance_count": len(run_provenance_rows),
        "run_provenance_source_state_counts": _trace_report_value_counts(
            run_provenance_rows,
            "source_state",
        ),
        "artifact_identity_count": len(artifact_identity_rows),
        "artifact_identity_artifact_counts": _trace_report_value_counts(
            artifact_identity_rows,
            "artifact",
        ),
        "artifact_identity_load_status_counts": _trace_report_value_counts(
            artifact_identity_rows,
            "load_status",
        ),
        "artifact_identity_check_count": len(artifact_identity_check_rows),
        "artifact_identity_check_status_counts": _trace_report_value_counts(
            artifact_identity_check_rows,
            "status",
        ),
        "capture_capability_count": len(capture_capability_rows),
        "capture_capability_present_counts": _trace_report_value_counts(
            capture_capability_rows,
            "present",
        ),
        "trace_config_count": len(trace_config_rows),
        "trace_config_status_counts": _trace_report_value_counts(
            trace_config_rows,
            "config_status",
        ),
        "trace_config_missing_requested_sidecar_counts": (
            _trace_report_comma_value_counts(
                trace_config_rows,
                "missing_requested_sidecar_controls",
            )
        ),
        "provider_payload_boundary_count": len(provider_payload_boundary_rows),
        "provider_payload_boundary_status_counts": _trace_report_value_counts(
            provider_payload_boundary_rows,
            "capture_status",
        ),
        "provider_payload_boundary_route_only_count": len(
            provider_payload_route_only_rows
        ),
        "provider_payload_boundary_route_only_lanes": (
            _trace_report_provider_payload_lanes(provider_payload_route_only_rows)
        ),
        "provider_payload_boundary_recorded_count": len(provider_payload_recorded_rows),
        "provider_payload_boundary_recorded_lanes": (
            _trace_report_provider_payload_lanes(provider_payload_recorded_rows)
        ),
        "trace_event_artifact_count": len(trace_event_artifact_rows),
        "trace_event_artifact_status_counts": _trace_report_value_counts(
            trace_event_artifact_rows,
            "status",
        ),
        "trace_event_artifact_event_kind_counts": (
            _trace_report_comma_value_counts(
                trace_event_artifact_rows,
                "event_kinds",
            )
        ),
        "backend_event_artifact_count": len(backend_event_artifact_rows),
        "backend_event_artifact_status_counts": _trace_report_value_counts(
            backend_event_artifact_rows,
            "status",
        ),
        "backend_event_artifact_event_kind_counts": (
            _trace_report_comma_value_counts(
                backend_event_artifact_rows,
                "event_kinds",
            )
        ),
        "backend_event_row_count": len(backend_event_rows),
        "backend_event_row_event_kind_counts": _trace_report_value_counts(
            backend_event_rows,
            "event_kind",
        ),
        "backend_event_row_backend_counts": _trace_report_value_counts(
            backend_event_rows,
            "backend_id",
        ),
        "backend_provider_boundary_count": len(backend_provider_boundary_rows),
        "backend_provider_boundary_status_counts": _trace_report_value_counts(
            backend_provider_boundary_rows,
            "boundary_status",
        ),
        "backend_provider_boundary_stage_counts": _trace_report_value_counts(
            backend_provider_boundary_rows,
            "provider_stage",
        ),
        "backend_provider_boundary_root_stage_counts": _trace_report_value_counts(
            backend_provider_boundary_rows,
            "root_cause_stage",
        ),
        "backend_fail_closed_root_cause_count": len(
            backend_fail_closed_root_cause_rows
        ),
        "backend_fail_closed_root_cause_backend_counts": (
            _trace_report_value_counts(
                backend_fail_closed_root_cause_rows,
                "backend_id",
            )
        ),
        "backend_fail_closed_root_cause_stage_counts": (
            _trace_report_value_counts(
                backend_fail_closed_root_cause_rows,
                "provider_stage",
            )
        ),
        "backend_fail_closed_root_cause_root_stage_counts": (
            _trace_report_value_counts(
                backend_fail_closed_root_cause_rows,
                "root_cause_stage",
            )
        ),
        "debug_payload_artifact_summary_count": len(
            debug_payload_artifact_summary_rows
        ),
        "debug_payload_artifact_summary_status_counts": _trace_report_value_counts(
            debug_payload_artifact_summary_rows,
            "payload_summary_status",
        ),
        "token_quality_summary_count": len(token_quality_summary_rows),
        "token_quality_summary_status_counts": _trace_report_value_counts(
            token_quality_summary_rows,
            "status",
        ),
        "token_quality_summary_topk_status_counts": _trace_report_value_counts(
            token_quality_summary_rows,
            "selected_topk_status",
        ),
        "oracle_reference_summary_count": len(oracle_reference_summary_rows),
        "oracle_reference_summary_status_counts": _trace_report_value_counts(
            oracle_reference_summary_rows,
            "oracle_reference_status",
        ),
        "oracle_reference_summary_correctness_counts": _trace_report_value_counts(
            oracle_reference_summary_rows,
            "correctness_claim_status",
        ),
        "planning_decision_sidecar_count": len(planning_decision_sidecar_rows),
        "planning_decision_sidecar_status_counts": _trace_report_value_counts(
            planning_decision_sidecar_rows,
            "status",
        ),
        "planning_decision_sidecar_row_kind_counts": _trace_report_value_counts(
            planning_decision_sidecar_rows,
            "row_kind",
        ),
        "planning_decision_sidecar_phase_counts": _trace_report_value_counts(
            planning_decision_sidecar_rows,
            "planning_phase",
        ),
        "token_quality_sidecar_count": len(token_quality_sidecar_rows),
        "token_quality_sidecar_status_counts": _trace_report_value_counts(
            token_quality_sidecar_rows,
            "status",
        ),
        "token_quality_sidecar_finish_reason_counts": _trace_report_value_counts(
            token_quality_sidecar_rows,
            "finish_reason",
        ),
        "topk_token_sidecar_count": len(topk_token_sidecar_rows),
        "topk_token_sidecar_status_counts": _trace_report_value_counts(
            topk_token_sidecar_rows,
            "status",
        ),
        "topk_token_sidecar_selected_status_counts": _trace_report_value_counts(
            topk_token_sidecar_rows,
            "selected_candidate_status",
        ),
        "topk_token_sidecar_score_kind_counts": _trace_report_value_counts(
            topk_token_sidecar_rows,
            "score_kind",
        ),
        "tensor_payload_sidecar_count": len(tensor_payload_sidecar_rows),
        "tensor_payload_sidecar_status_counts": _trace_report_value_counts(
            tensor_payload_sidecar_rows,
            "status",
        ),
        "tensor_payload_sidecar_kind_counts": _trace_report_value_counts(
            tensor_payload_sidecar_rows,
            "tensor_payload_kind",
        ),
        "tensor_payload_sidecar_role_counts": _trace_report_value_counts(
            tensor_payload_sidecar_rows,
            "tensor_role",
        ),
        "kv_payload_digest_sidecar_count": len(kv_payload_digest_sidecar_rows),
        "kv_payload_digest_sidecar_status_counts": _trace_report_value_counts(
            kv_payload_digest_sidecar_rows,
            "status",
        ),
        "kv_payload_digest_sidecar_role_counts": _trace_report_value_counts(
            kv_payload_digest_sidecar_rows,
            "tensor_role",
        ),
        "logit_slice_sidecar_count": len(logit_slice_sidecar_rows),
        "logit_slice_sidecar_status_counts": _trace_report_value_counts(
            logit_slice_sidecar_rows,
            "status",
        ),
        "logit_slice_sidecar_role_counts": _trace_report_value_counts(
            logit_slice_sidecar_rows,
            "tensor_role",
        ),
        "logit_slice_sidecar_action_counts": _trace_report_value_counts(
            logit_slice_sidecar_rows,
            "targetplan_action",
        ),
        "activation_digest_sidecar_count": len(activation_digest_sidecar_rows),
        "activation_digest_sidecar_status_counts": _trace_report_value_counts(
            activation_digest_sidecar_rows,
            "status",
        ),
        "activation_digest_sidecar_role_counts": _trace_report_value_counts(
            activation_digest_sidecar_rows,
            "tensor_role",
        ),
        "activation_digest_sidecar_intrinsic_counts": _trace_report_value_counts(
            activation_digest_sidecar_rows,
            "intrinsic",
        ),
        "device_result_digest_sidecar_count": len(device_result_digest_sidecar_rows),
        "device_result_digest_sidecar_status_counts": _trace_report_value_counts(
            device_result_digest_sidecar_rows,
            "status",
        ),
        "device_result_digest_sidecar_role_counts": _trace_report_value_counts(
            device_result_digest_sidecar_rows,
            "tensor_role",
        ),
        "device_result_digest_sidecar_action_counts": _trace_report_value_counts(
            device_result_digest_sidecar_rows,
            "targetplan_action",
        ),
        "device_result_digest_sidecar_intrinsic_counts": _trace_report_value_counts(
            device_result_digest_sidecar_rows,
            "intrinsic",
        ),
        "scheduler_packet_lineage_sidecar_count": len(
            scheduler_packet_lineage_sidecar_rows
        ),
        "scheduler_packet_lineage_sidecar_status_counts": (
            _trace_report_value_counts(
                scheduler_packet_lineage_sidecar_rows,
                "status",
            )
        ),
        "scheduler_packet_lineage_sidecar_executor_counts": (
            _trace_report_value_counts(
                scheduler_packet_lineage_sidecar_rows,
                "executor_status",
            )
        ),
        "scheduler_kv_shard_lifecycle_sidecar_count": len(
            scheduler_kv_shard_lifecycle_sidecar_rows
        ),
        "scheduler_kv_shard_lifecycle_sidecar_status_counts": (
            _trace_report_value_counts(
                scheduler_kv_shard_lifecycle_sidecar_rows,
                "status",
            )
        ),
        "scheduler_kv_shard_lifecycle_sidecar_lifecycle_counts": (
            _trace_report_value_counts(
                scheduler_kv_shard_lifecycle_sidecar_rows,
                "kv_lifecycle_status",
            )
        ),
        "scheduler_listener_sparse_logit_sidecar_count": len(
            scheduler_listener_sparse_logit_sidecar_rows
        ),
        "scheduler_listener_sparse_logit_sidecar_status_counts": (
            _trace_report_value_counts(
                scheduler_listener_sparse_logit_sidecar_rows,
                "status",
            )
        ),
        "scheduler_listener_sparse_logit_sidecar_listener_status_counts": (
            _trace_report_value_counts(
                scheduler_listener_sparse_logit_sidecar_rows,
                "listener_sparse_status",
            )
        ),
        "scheduler_listener_sparse_logit_sidecar_executor_counts": (
            _trace_report_value_counts(
                scheduler_listener_sparse_logit_sidecar_rows,
                "executor_status",
            )
        ),
        "device_dma_lifecycle_sidecar_count": len(device_dma_lifecycle_sidecar_rows),
        "device_dma_lifecycle_sidecar_status_counts": _trace_report_value_counts(
            device_dma_lifecycle_sidecar_rows,
            "status",
        ),
        "device_dma_lifecycle_sidecar_stage_counts": _trace_report_value_counts(
            device_dma_lifecycle_sidecar_rows,
            "device_stage",
        ),
        "device_dma_lifecycle_sidecar_queue_counts": _trace_report_value_counts(
            device_dma_lifecycle_sidecar_rows,
            "queue_id",
        ),
        "attention_page_trace_sidecar_count": len(attention_page_trace_sidecar_rows),
        "attention_page_trace_sidecar_status_counts": _trace_report_value_counts(
            attention_page_trace_sidecar_rows,
            "status",
        ),
        "attention_page_trace_sidecar_action_counts": _trace_report_value_counts(
            attention_page_trace_sidecar_rows,
            "targetplan_action",
        ),
        "introspection_artifact_count": len(introspection_artifact_rows),
        "introspection_artifact_status_counts": _trace_report_value_counts(
            introspection_artifact_rows,
            "status",
        ),
        "introspection_artifact_kind_counts": _trace_report_value_counts(
            introspection_artifact_rows,
            "kind",
        ),
        "introspection_artifact_format_counts": _trace_report_value_counts(
            introspection_artifact_rows,
            "artifact_kind",
        ),
        "introspection_artifact_sensitivity_counts": _trace_report_value_counts(
            introspection_artifact_rows,
            "sensitivity",
        ),
        "introspection_artifact_compile_feature_counts": (
            _trace_report_comma_value_counts(
                introspection_artifact_rows,
                "compile_features",
            )
        ),
        "introspection_artifact_row_count_total": (
            introspection_artifact_row_count_total
        ),
        "introspection_artifact_byte_count_total": (
            introspection_artifact_byte_count_total
        ),
        "introspection_capability_count": len(introspection_capability_rows),
        "introspection_capability_status_counts": _trace_report_value_counts(
            introspection_capability_rows,
            "capability_status",
        ),
        "introspection_artifact_summary_count": len(
            introspection_artifact_summary_rows
        ),
        "introspection_artifact_summary_status_counts": _trace_report_value_counts(
            introspection_artifact_summary_rows,
            "summary_status",
        ),
        "introspection_section_inventory_count": len(
            introspection_section_inventory_rows
        ),
        "introspection_section_inventory_status_counts": _trace_report_value_counts(
            introspection_section_inventory_rows,
            "section_status",
        ),
        "introspection_section_inventory_capability_counts": (
            _trace_report_value_counts(
                introspection_section_inventory_rows,
                "capture_capability",
            )
        ),
        "timeline_query_summary_count": len(timeline_query_summary_rows),
        "timeline_query_summary_status_counts": _trace_report_value_counts(
            timeline_query_summary_rows,
            "status",
        ),
        "supported_claim_samples": _trace_report_samples(
            supported_claim_rows,
            ("claim", "basis", "evidence_grade"),
        ),
        "unsupported_claim_samples": _trace_report_samples(
            unsupported_claim_rows,
            ("claim", "reason", "basis"),
        ),
        "correctness_evidence_samples": _trace_report_samples(
            correctness_evidence_rows,
            (
                "evidence",
                "status",
                "evidence_role",
                "proof_grade_status",
                "basis",
                "next_gate",
            ),
        ),
        "evidence_artifact_check_samples": _trace_report_samples(
            evidence_artifact_check_rows,
            ("check", "evidence_artifact", "status", "detail"),
        ),
        "promotion_gate_summary_samples": _trace_report_samples(
            promotion_gate_summary_rows,
            ("gate", "status", "proof_grade_status", "basis", "next_gate"),
            limit=6,
        ),
        "trace_mode_guardrail_samples": _trace_report_samples(
            trace_mode_guardrail_rows,
            (
                "trace_mode",
                "trace_sinks",
                "role",
                "overhead_boundary",
                "claim_guardrail",
            ),
        ),
        "ab_provenance_samples": _trace_report_samples(
            ab_provenance_rows,
            (
                "role",
                "binary",
                "git_sha",
                "worktree_dirty",
                "source_state",
                "artifact_hash_count",
                "artifact_hash_status",
                "hardware_card_count",
                "hardware_cards",
                "provenance_status",
                "proof_grade_status",
                "basis",
            ),
            limit=4,
        ),
        "ab_comparability_samples": _trace_report_samples(
            ab_comparability_rows,
            ("status", "basis", "promotion_gate"),
        ),
        "ab_coverage_samples": _trace_report_samples(
            ab_coverage_rows,
            (
                "align",
                "coverage_status",
                "total_rows",
                "matched_rows",
                "baseline_only_rows",
                "candidate_only_rows",
                "warnings",
                "basis",
            ),
        ),
        "ab_repeatability_samples": _trace_report_samples(
            ab_repeatability_rows,
            (
                "status",
                "align",
                "baseline_runs",
                "candidate_runs",
                "required_matched_runs_for_hardware_proof",
                "matched_rows",
                "proof_grade_status",
                "basis",
            ),
        ),
        "next_measurement_samples": _trace_report_samples(
            next_measurement_rows,
            ("priority", "next_measurement", "reason", "command_hint"),
        ),
        "answerability_samples": _trace_report_samples(
            answerability_rows,
            ("question", "status", "basis"),
            limit=8,
        ),
        "analysis_command_samples": _trace_report_samples(
            analysis_command_rows,
            ("purpose", "command"),
        ),
        "report_json_section_samples": _trace_report_samples(
            report_json_section_sample_rows,
            (
                "heading",
                "json_path",
                "json_section",
                "section_kind",
                "requires_timeline_trace",
                "claim_boundary",
            ),
            limit=64,
        ),
        "report_section_inventory_samples": _trace_report_samples(
            report_section_inventory_rows,
            (
                "heading",
                "query",
                "native_sql",
                "portable_command",
                "native_sql_command",
            ),
            limit=8,
        ),
        "preflight_finding_samples": _trace_report_string_samples(
            preflight_finding_rows,
            limit=8,
        ),
        "evidence_classification_samples": _trace_report_string_samples(
            evidence_classification_rows,
            limit=4,
        ),
        "capture_samples": _trace_report_samples(
            capture_rows,
            (
                "metadata",
                "trace",
                "trace_run_id",
                "process_kind",
                "trace_mode",
                "trace_sinks",
                "model_id",
                "backend_id",
                "target_plan_sha256",
                "worktree_dirty",
            ),
        ),
        "run_provenance_samples": _trace_report_samples(
            run_provenance_rows,
            (
                "binary",
                "git_sha",
                "worktree_dirty",
                "source_state",
                "hardware_card_count",
                "hardware_cards",
                "provenance_boundary",
            ),
        ),
        "artifact_identity_samples": _trace_report_samples(
            artifact_identity_rows,
            ("artifact", "path", "sha256", "load_status", "status_source"),
        ),
        "artifact_identity_check_samples": _trace_report_samples(
            artifact_identity_check_rows,
            ("artifact", "check", "status", "detail"),
        ),
        "capture_capability_samples": _trace_report_samples(
            capture_capability_rows,
            ("capability", "present"),
            limit=8,
        ),
        "trace_config_samples": _trace_report_samples(
            trace_config_rows,
            (
                "config_status",
                "requested_sidecar_controls",
                "recorded_sidecar_capabilities",
                "missing_requested_sidecar_controls",
                "introspection_level",
                "compile_feature_trace_introspection",
                "deep_introspection_effective",
                "next_action",
            ),
        ),
        "trace_event_artifact_samples": _trace_report_samples(
            trace_event_artifact_rows,
            (
                "index",
                "path",
                "sha256",
                "row_count",
                "matching_trace_run_id_rows",
                "event_kinds",
                "status",
            ),
        ),
        "provider_payload_boundary_samples": _trace_report_samples(
            provider_payload_boundary_rows,
            (
                "provider_id",
                "payload_lane",
                "capture_status",
                "artifact_count",
                "matching_provider_artifact_count",
                "artifact_kind_recorded_count",
                "artifact_kind_recorded_backend_count",
                "artifact_kind_recorded_backend_ids",
                "report_section",
                "boundary_status",
                "producer_status",
                "producer_contract",
                "payload_record_policy",
                "payload_sensitivity",
                "claim_boundary",
                "next_action",
            ),
        ),
        "provider_payload_boundary_route_only_samples": _trace_report_samples(
            provider_payload_route_only_rows,
            (
                "provider_id",
                "payload_lane",
                "capture_status",
                "capture_capability",
                "artifact_kind",
                "capture_control",
                "artifact_count",
                "matching_provider_artifact_count",
                "artifact_kind_recorded_count",
                "artifact_kind_recorded_backend_count",
                "artifact_kind_recorded_backend_ids",
                "report_section",
                "boundary_status",
                "producer_status",
                "producer_contract",
                "payload_record_policy",
                "payload_sensitivity",
                "claim_boundary",
                "next_action",
            ),
        ),
        "provider_payload_boundary_recorded_samples": _trace_report_samples(
            provider_payload_recorded_rows,
            (
                "provider_id",
                "payload_lane",
                "capture_status",
                "capture_capability",
                "artifact_kind",
                "capture_control",
                "artifact_count",
                "matching_provider_artifact_count",
                "artifact_kind_recorded_count",
                "artifact_kind_recorded_backend_count",
                "artifact_kind_recorded_backend_ids",
                "report_section",
                "boundary_status",
                "producer_status",
                "producer_contract",
                "payload_record_policy",
                "payload_sensitivity",
                "claim_boundary",
                "next_action",
            ),
        ),
        "backend_event_artifact_samples": _trace_report_samples(
            backend_event_artifact_rows,
            (
                "index",
                "path",
                "sha256",
                "row_count",
                "matching_trace_run_id_rows",
                "event_kinds",
                "status",
            ),
        ),
        "backend_event_samples": _trace_report_samples(
            backend_event_rows,
            (
                "artifact_index",
                "row_index",
                "timestamp_ns",
                "backend_id",
                "event_kind",
                "model_id",
                "request_id",
                "generation_id",
                "targetplan_op_id",
                "artifact",
                "message",
                "metadata_keys",
            ),
        ),
        "backend_provider_boundary_samples": _trace_report_samples(
            backend_provider_boundary_rows,
            (
                "artifact_index",
                "row_index",
                "timestamp_ns",
                "backend_id",
                "event_kind",
                "provider_stage",
                "boundary_status",
                "root_cause_stage",
                "root_cause",
                "model_id",
                "request_id",
                "generation_id",
                "targetplan_op_id",
                "failure_reason",
                "message",
                "plan_artifact_status",
                "target_plan_artifact_status",
                "target_plan_validation_status",
                "runtime_binding_status",
                "backend_descriptor_status",
                "hardware_gate_status",
                "device_binding_status",
                "weight_policy_status",
                "scheduler_targetplan_execution_step_bridge_status",
            ),
        ),
        "backend_fail_closed_root_cause_samples": _trace_report_samples(
            backend_fail_closed_root_cause_rows,
            (
                "backend_id",
                "provider_stage",
                "root_cause_stage",
                "root_cause",
                "event_kind",
                "failure_count",
                "example_model_id",
                "example_request_id",
                "example_generation_id",
                "example_targetplan_op_id",
                "example_failure_reason",
            ),
        ),
        "debug_payload_artifact_summary_samples": _trace_report_samples(
            debug_payload_artifact_summary_rows,
            (
                "artifact_kind",
                "payload_summary_status",
                "row_count",
                "byte_count",
                "sampling_policy",
                "token_window",
                "sensitivity",
                "compile_features",
                "report_section",
                "debug_payload_boundary",
                "claim_boundary",
            ),
        ),
        "token_quality_summary_samples": _trace_report_samples(
            token_quality_summary_rows,
            (
                "status",
                "evidence_role",
                "request_id",
                "generation_id",
                "token_index",
                "selected_token_id",
                "selected_topk_status",
                "score_kind",
                "top1_token_id",
                "top1_score",
                "runner_up_token_id",
                "runner_up_score",
                "top1_margin",
                "temperature",
                "top_p",
                "top_k",
                "num_logprobs",
                "tokens_reused",
                "runtime_request_token_count",
                "oracle_reference",
                "oracle_artifact_sha256",
                "claim_boundary",
            ),
        ),
        "oracle_reference_summary_samples": _trace_report_samples(
            oracle_reference_summary_rows,
            (
                "status",
                "evidence_role",
                "request_id",
                "generation_id",
                "token_index",
                "selected_token_id",
                "oracle_reference_role",
                "hf_cpu_oracle_artifact_path",
                "hf_cpu_oracle_sha256",
                "expected_oracle_source",
                "oracle_reference_status",
                "sut_classification",
                "correctness_claim_status",
                "claim_boundary",
            ),
        ),
        "planning_decision_sidecar_samples": _trace_report_samples(
            planning_decision_sidecar_rows,
            (
                "row_kind",
                "status",
                "process_kind",
                "frontend",
                "target_backend",
                "selection_source",
                "source",
                "logical_command",
                "dispatch_command",
                "runner_count",
                "exit_code",
                "duration_us",
                "artifact_role",
                "artifact_kind",
                "artifact_path",
                "artifact_sha256",
                "artifact_byte_count",
                "planning_phase",
                "event_name",
                "category",
                "start_ms",
                "duration_ms",
                "planning_output_bytes",
                "targetplan_op_count",
                "claim_boundary",
            ),
        ),
        "token_quality_sidecar_samples": _trace_report_samples(
            token_quality_sidecar_rows,
            (
                "status",
                "evidence_role",
                "request_id",
                "generation_id",
                "token_index",
                "selected_token_id",
                "topk_count",
                "temperature",
                "top_p",
                "top_k",
                "eos_policy",
                "finish_reason",
                "oracle_reference",
            ),
        ),
        "topk_token_sidecar_samples": _trace_report_samples(
            topk_token_sidecar_rows,
            (
                "status",
                "evidence_role",
                "request_id",
                "generation_id",
                "token_index",
                "selected_token_id",
                "candidate_token_id",
                "candidate_rank",
                "candidate_score",
                "score_kind",
                "selected_candidate_status",
                "temperature",
                "top_p",
                "top_k",
                "oracle_reference",
                "claim_boundary",
            ),
        ),
        "tensor_payload_sidecar_samples": _trace_report_samples(
            tensor_payload_sidecar_rows,
            (
                "status",
                "evidence_role",
                "backend_id",
                "request_id",
                "generation_id",
                "token_index",
                "targetplan_op_id",
                "targetplan_action",
                "target_plan_statement_index",
                "target_plan_statement_kind",
                "target_plan_statement_name",
                "layer",
                "tensor_payload_kind",
                "tensor_name",
                "tensor_role",
                "element_type",
                "shape",
                "element_count",
                "digest_sha256",
                "sample_value_count",
                "sample_finite_count",
                "sample_min",
                "sample_max",
                "sample_nan_count",
                "sample_pos_inf_count",
                "sample_neg_inf_count",
                "sample_values",
                "failure_reason",
            ),
        ),
        "kv_payload_digest_sidecar_samples": _trace_report_samples(
            kv_payload_digest_sidecar_rows,
            (
                "status",
                "evidence_role",
                "backend_id",
                "request_id",
                "generation_id",
                "token_index",
                "targetplan_op_id",
                "targetplan_action",
                "target_plan_statement_index",
                "target_plan_statement_kind",
                "target_plan_statement_name",
                "layer",
                "tensor_payload_kind",
                "tensor_name",
                "tensor_role",
                "element_type",
                "shape",
                "element_count",
                "digest_sha256",
                "sample_value_count",
                "sample_finite_count",
                "sample_min",
                "sample_max",
                "sample_nan_count",
                "sample_pos_inf_count",
                "sample_neg_inf_count",
                "sample_values",
                "failure_reason",
            ),
        ),
        "logit_slice_sidecar_samples": _trace_report_samples(
            logit_slice_sidecar_rows,
            (
                "status",
                "evidence_role",
                "backend_id",
                "request_id",
                "generation_id",
                "token_index",
                "targetplan_op_id",
                "targetplan_action",
                "target_plan_statement_index",
                "target_plan_statement_kind",
                "target_plan_statement_name",
                "layer",
                "intrinsic",
                "tensor_payload_kind",
                "tensor_name",
                "tensor_role",
                "element_type",
                "shape",
                "element_count",
                "digest_sha256",
                "sample_start",
                "sample_stride",
                "sample_value_count",
                "sample_finite_count",
                "sample_min",
                "sample_max",
                "sample_nan_count",
                "sample_pos_inf_count",
                "sample_neg_inf_count",
                "sample_values",
                "failure_reason",
            ),
        ),
        "activation_digest_sidecar_samples": _trace_report_samples(
            activation_digest_sidecar_rows,
            (
                "status",
                "evidence_role",
                "backend_id",
                "request_id",
                "generation_id",
                "token_index",
                "targetplan_op_id",
                "targetplan_action",
                "target_plan_statement_index",
                "target_plan_statement_kind",
                "target_plan_statement_name",
                "layer",
                "intrinsic",
                "tensor_payload_kind",
                "tensor_name",
                "tensor_role",
                "element_type",
                "shape",
                "element_count",
                "digest_sha256",
                "sample_start",
                "sample_stride",
                "sample_value_count",
                "sample_finite_count",
                "sample_min",
                "sample_max",
                "sample_nan_count",
                "sample_pos_inf_count",
                "sample_neg_inf_count",
                "sample_values",
                "failure_reason",
            ),
        ),
        "device_result_digest_sidecar_samples": _trace_report_samples(
            device_result_digest_sidecar_rows,
            (
                "status",
                "evidence_role",
                "backend_id",
                "request_id",
                "generation_id",
                "token_index",
                "targetplan_op_id",
                "targetplan_action",
                "target_plan_statement_index",
                "target_plan_statement_kind",
                "target_plan_statement_name",
                "layer",
                "intrinsic",
                "tensor_payload_kind",
                "tensor_name",
                "tensor_role",
                "element_type",
                "shape",
                "element_count",
                "digest_sha256",
                "sample_start",
                "sample_stride",
                "sample_value_count",
                "sample_finite_count",
                "sample_min",
                "sample_max",
                "sample_nan_count",
                "sample_pos_inf_count",
                "sample_neg_inf_count",
                "sample_values",
                "failure_reason",
            ),
        ),
        "scheduler_packet_lineage_sidecar_samples": _trace_report_samples(
            scheduler_packet_lineage_sidecar_rows,
            (
                "status",
                "evidence_role",
                "request_id",
                "generation_id",
                "location_id",
                "parent_location_id",
                "executor_shape",
                "executor_status",
                "attention_mode",
                "token_job_count",
                "runtime_request_token_count",
                "tokens_reused",
                "visible_token_slots",
                "kv_context_rows",
                "kv_save_rows",
                "kv_page_count",
                "hw_shard_allocation_requests",
                "hw_gof_page_infos",
                "prior_host_gof_staging_status",
                "prior_host_gof_dma_completions",
                "listener_sparse_rows",
                "listener_sparse_tokens",
                "failure_reason",
            ),
        ),
        "scheduler_kv_shard_lifecycle_sidecar_samples": _trace_report_samples(
            scheduler_kv_shard_lifecycle_sidecar_rows,
            (
                "status",
                "evidence_role",
                "request_id",
                "generation_id",
                "location_id",
                "parent_location_id",
                "executor_shape",
                "executor_status",
                "attention_mode",
                "kv_lifecycle_status",
                "token_job_count",
                "kv_job_count",
                "runtime_request_token_count",
                "visible_token_slots",
                "kv_context_rows",
                "kv_save_rows",
                "kv_page_count",
                "hw_shard_allocation_requests",
                "hw_gof_page_infos",
                "prior_host_gof_staging_status",
                "prior_host_gof_dma_completions",
                "failure_reason",
            ),
        ),
        "scheduler_listener_sparse_logit_sidecar_samples": _trace_report_samples(
            scheduler_listener_sparse_logit_sidecar_rows,
            (
                "status",
                "listener_sparse_status",
                "evidence_role",
                "request_id",
                "generation_id",
                "location_id",
                "executor_shape",
                "executor_status",
                "attention_mode",
                "listener_sparse_rows",
                "listener_sparse_tokens",
                "sparse_topk_rows",
                "sparse_topk_token_count",
                "token_job_count",
                "minibatch_count",
                "runtime_request_token_count",
                "tokens_reused",
                "failure_reason",
            ),
        ),
        "device_dma_lifecycle_sidecar_samples": _trace_report_samples(
            device_dma_lifecycle_sidecar_rows,
            (
                "status",
                "evidence_role",
                "backend_id",
                "request_id",
                "generation_id",
                "targetplan_op_id",
                "targetplan_action",
                "location_id",
                "device_stage",
                "queue_id",
                "device_index",
                "card_bus",
                "dma_direction",
                "descriptor_count",
                "byte_count",
                "counter_name",
                "counter_value_delta",
                "cacheblock_dma_shard_id",
                "cacheblock_dma_gof_start_in_shard",
                "cacheblock_dma_k_transfer_count",
                "cacheblock_dma_v_transfer_count",
                "cacheblock_dma_transfer_byte_count",
                "cacheblock_gof_start_position",
                "cacheblock_k_word_checksum",
                "cacheblock_v_word_checksum",
                "queue_depth_before",
                "queue_depth_after",
                "failure_reason",
            ),
        ),
        "attention_page_trace_sidecar_samples": _trace_report_samples(
            attention_page_trace_sidecar_rows,
            (
                "status",
                "evidence_role",
                "backend_id",
                "request_id",
                "generation_id",
                "targetplan_op_id",
                "targetplan_action",
                "layer",
                "head",
                "kv_head",
                "attention_row_index",
                "batch",
                "visible_tokens",
                "page_start",
                "page_count",
                "page_v_count",
                "scaled_score_count",
                "exp_score_count",
                "v_star_count",
                "m_star",
                "s_star",
                "was_valid",
                "failure_reason",
            ),
        ),
        "introspection_capability_samples": _trace_report_samples(
            introspection_capability_rows,
            (
                "capture_capability",
                "capability_status",
                "matching_artifact_count",
                "claim_boundary",
                "next_action",
            ),
        ),
        "introspection_artifact_samples": _trace_report_samples(
            introspection_artifact_rows,
            (
                "index",
                "kind",
                "artifact_kind",
                "path",
                "sha256",
                "status",
                "row_count",
                "byte_count",
                "token_window",
                "sampling_policy",
                "sensitivity",
                "compile_features",
            ),
            limit=16,
        ),
        "introspection_artifact_summary_samples": _trace_report_samples(
            introspection_artifact_summary_rows,
            (
                "artifact_kind",
                "summary_status",
                "artifact_count",
                "local_present_count",
                "local_missing_count",
                "local_missing_path_count",
                "row_count_total",
                "report_sections",
                "claim_boundaries",
            ),
        ),
        "introspection_section_inventory_samples": _trace_report_samples(
            introspection_section_inventory_rows,
            (
                "capture_capability",
                "artifact_kind",
                "heading",
                "json_section",
                "capability_present",
                "artifact_count",
                "section_status",
                "claim_boundary",
            ),
            limit=16,
        ),
        "timeline_query_summary_samples": _trace_report_samples(
            timeline_query_summary_rows,
            (
                "section",
                "query",
                "row_count",
                "rendered_rows",
                "native_sql",
                "status",
                "portable_command",
                "native_sql_command",
            ),
        ),
    }
    return _validation(not errors, errors, detail)


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _generated_token_ids(payload: Any) -> list[int] | None:
  if not isinstance(payload, Mapping):
    return None
  for key in (
    "generated_token_ids",
    "generated_tokens",
    "candidate_generated_token_ids",
    "reference_generated_token_ids",
  ):
    tokens = _int_list(payload.get(key))
    if tokens is not None:
      return tokens
  generation = payload.get("generation")
  if isinstance(generation, Mapping):
    return _int_list(generation.get("generated_token_ids"))
  return None


def _int_list(value: Any) -> list[int] | None:
  if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
    return None
  return list(value)


def _validate_referenced_sha256(
  errors: list[str],
  value: Mapping[str, Any],
  base_dir: Path | None,
  label: str,
) -> None:
  if base_dir is None:
    errors.append(f"{label}.sha256 cannot be verified without evidence path context")
    return
  path_value = value.get("path")
  digest_value = value.get("sha256")
  if not isinstance(path_value, str) or not isinstance(digest_value, str):
    return
  path = Path(path_value)
  if not path.is_absolute():
    path = base_dir / path
  if not path.is_file():
    errors.append(f"{label}.path does not exist: {path}")
    return
  actual = _sha256_file(path)
  if actual != digest_value.lower():
    errors.append(f"{label}.sha256 does not match referenced file")


def _validate_one_token_artifacts(
  errors: list[str],
  value: Any,
  base_dir: Path | None,
) -> list[str]:
  artifacts = _expect_object(errors, value, "one-token artifacts")
  if artifacts is None:
    return []

  required = (
    ("hf_cpu_dense_logits_jsonl", True),
    ("ares_dense_logits_jsonl", True),
    ("runtime_binary", False),
    ("ares_plan", False),
    ("target_plan", False),
  )
  seen: list[str] = []
  for key, require_rows in required:
    label = f"one-token artifacts.{key}"
    record = _expect_object(errors, artifacts.get(key), label)
    if record is None:
      continue
    seen.append(key)
    _validate_file_artifact_record(
      errors,
      record,
      label,
      base_dir=base_dir,
      require_rows=require_rows,
    )
  return seen


def _validate_eight_token_artifacts(
  errors: list[str],
  value: Any,
  base_dir: Path | None,
) -> list[str]:
  artifacts = _expect_object(errors, value, "eight-token artifacts")
  if artifacts is None:
    return []

  required = (
    ("reference_tokens", False),
    ("candidate_tokens", False),
    ("runtime_binary", False),
    ("ares_plan", False),
    ("target_plan", False),
  )
  seen: list[str] = []
  for key, require_rows in required:
    label = f"eight-token artifacts.{key}"
    record = _expect_object(errors, artifacts.get(key), label)
    if record is None:
      continue
    seen.append(key)
    _validate_file_artifact_record(
      errors,
      record,
      label,
      base_dir=base_dir,
      require_rows=require_rows,
    )

  candidate_keys = {
    "ares_dense_logits_jsonl": True,
    "ares_token_lines": True,
  }
  present_candidate_keys = [
    key for key in candidate_keys if isinstance(artifacts.get(key), Mapping)
  ]
  if not present_candidate_keys:
    errors.append(
      "eight-token artifacts must include ares_dense_logits_jsonl or ares_token_lines"
    )
  for key in present_candidate_keys:
    seen.append(key)
    _validate_file_artifact_record(
      errors,
      artifacts[key],
      f"eight-token artifacts.{key}",
      base_dir=base_dir,
      require_rows=candidate_keys[key],
    )

  reference_keys = {
    "hf_cpu_dense_logits_jsonl": True,
    "hf_cpu_oracle_jsonl": True,
    "hf_cpu_reference_report": False,
  }
  present_reference_keys = [
    key for key in reference_keys if isinstance(artifacts.get(key), Mapping)
  ]
  if not present_reference_keys:
    errors.append(
      "eight-token artifacts must include hf_cpu_dense_logits_jsonl, "
      "hf_cpu_oracle_jsonl, or hf_cpu_reference_report"
    )
  for key in present_reference_keys:
    seen.append(key)
    _validate_file_artifact_record(
      errors,
      artifacts[key],
      f"eight-token artifacts.{key}",
      base_dir=base_dir,
      require_rows=reference_keys[key],
    )
  return seen


def _validate_file_artifact_record(
  errors: list[str],
  record: Mapping[str, Any],
  label: str,
  *,
  base_dir: Path | None,
  require_rows: bool = False,
) -> None:
  _require_non_empty_string(errors, record.get("path"), f"{label}.path")
  _require_sha256(errors, record.get("sha256"), label)
  size_bytes = record.get("size_bytes")
  if not isinstance(size_bytes, int) or size_bytes <= 0:
    errors.append(f"{label}.size_bytes must be a positive integer")
  if require_rows:
    rows = record.get("rows")
    if not isinstance(rows, int) or rows < 1:
      errors.append(f"{label}.rows must be a positive integer")
  _validate_referenced_sha256(errors, record, base_dir, label)


def _validate_model_provenance(
  errors: list[str],
  value: Any,
  label: str,
) -> str | None:
  provenance = _expect_object(errors, value, label)
  if provenance is None:
    return None

  model_id = provenance.get("model_id")
  _require_non_empty_string(errors, model_id, f"{label}.model_id")

  checkpoint = _expect_object(
    errors,
    provenance.get("checkpoint"),
    f"{label}.checkpoint",
  )
  if checkpoint is not None:
    _validate_revision_metadata(
      errors,
      checkpoint,
      f"{label}.checkpoint",
      ("model_id", "requested_revision", "resolved_revision", "checkpoint_class"),
    )
    _require_non_empty_string(
      errors,
      checkpoint.get("resolved_revision"),
      f"{label}.checkpoint.resolved_revision",
    )
    checkpoint_path = checkpoint.get("checkpoint_path")
    if checkpoint_path is not None:
      _require_non_empty_string(
        errors,
        checkpoint_path,
        f"{label}.checkpoint.checkpoint_path",
      )

  tokenizer = _expect_object(
    errors,
    provenance.get("tokenizer"),
    f"{label}.tokenizer",
  )
  if tokenizer is not None:
    _validate_revision_metadata(
      errors,
      tokenizer,
      f"{label}.tokenizer",
      ("tokenizer_id", "requested_revision", "resolved_revision"),
    )
    _require_non_empty_string(
      errors,
      tokenizer.get("resolved_revision"),
      f"{label}.tokenizer.resolved_revision",
    )

  input_policy = _expect_object(
    errors,
    provenance.get("input_policy"),
    f"{label}.input_policy",
  )
  if input_policy is not None:
    if input_policy.get("decode_strategy") != "greedy":
      errors.append(f"{label}.input_policy.decode_strategy must be greedy")
    if input_policy.get("context_tokens_role") != "hf_oracle_replay_context":
      errors.append(
        f"{label}.input_policy.context_tokens_role must be hf_oracle_replay_context"
      )

  return model_id if isinstance(model_id, str) and model_id.strip() else None


def _validate_introspection_trace_context(
    errors: list[str],
    trace_context: Mapping[str, Any],
    base_dir: Path | None,
) -> None:
    for field in ("autoagent_run_id", "request_id"):
        if field in trace_context:
            _require_non_empty_string(
                errors,
                trace_context.get(field),
                f"trace_context.{field}",
            )
    labels = trace_context.get("trace_labels")
    if labels is not None:
        if not isinstance(labels, list):
            errors.append("trace_context.trace_labels must be a list")
        else:
            for index, label in enumerate(labels):
                _require_non_empty_string(
                    errors,
                    label,
                    f"trace_context.trace_labels[{index}]",
                )
    for field in INTROSPECTION_TRACE_ARTIFACT_FIELDS:
        refs = trace_context.get(field)
        if refs is None:
            continue
        if not isinstance(refs, list):
            errors.append(f"trace_context.{field} must be a list")
            continue
        if not refs:
            errors.append(f"trace_context.{field} must not be empty when present")
            continue
        for index, ref in enumerate(refs):
            _validate_introspection_trace_artifact_ref(
                errors,
                ref,
                base_dir,
                f"trace_context.{field}[{index}]",
            )


def _validate_introspection_trace_artifact_ref(
    errors: list[str],
    value: Any,
    base_dir: Path | None,
    label: str,
) -> None:
    if not isinstance(value, Mapping):
        errors.append(f"{label} must be an object")
        return
    _require_non_empty_string(errors, value.get("path"), f"{label}.path")
    _require_introspection_sha256(errors, value.get("sha256"), f"{label}.sha256")
    for field in ("schema", "role", "profile"):
        if field in value:
            _require_non_empty_string(errors, value.get(field), f"{label}.{field}")
    if "path" in value and "sha256" in value:
        _validate_referenced_introspection_sha256(errors, value, base_dir, label)


def _introspection_trace_context_detail(value: Mapping[str, Any]) -> dict[str, Any]:
    detail: dict[str, Any] = {}
    for field in ("autoagent_run_id", "request_id"):
        item = value.get(field)
        if isinstance(item, str) and item:
            detail[field] = item
    labels = value.get("trace_labels")
    if isinstance(labels, list):
        detail["trace_labels"] = [
            label for label in labels if isinstance(label, str) and label
        ]
    for field in INTROSPECTION_TRACE_ARTIFACT_FIELDS:
        refs = value.get(field)
        if not isinstance(refs, list):
            continue
        normalized_refs = []
        for ref in refs:
            if not isinstance(ref, Mapping):
                continue
            normalized_ref: dict[str, Any] = {}
            path = ref.get("path")
            if isinstance(path, str) and path:
                normalized_ref["path"] = path
            digest = _normalize_introspection_sha256(ref.get("sha256"))
            if digest is not None:
                normalized_ref["sha256"] = f"sha256:{digest}"
            for metadata_field in ("schema", "role", "profile"):
                item = ref.get(metadata_field)
                if isinstance(item, str) and item:
                    normalized_ref[metadata_field] = item
            if normalized_ref:
                normalized_refs.append(normalized_ref)
        if normalized_refs:
            detail[field] = normalized_refs
    return detail


def _introspection_ladder_comparison_details(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [
        detail
        for detail in (
            _introspection_ladder_comparison_detail(comparison)
            for comparison in value[:INTROSPECTION_COMPARISON_DETAIL_LIMIT]
        )
        if detail
    ]


def _introspection_first_failed_comparison_detail(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, list):
        return None
    for comparison in value:
        if not isinstance(comparison, Mapping):
            continue
        if (
            comparison.get("status") == "failed"
            or comparison.get("result") == "diverged"
        ):
            return _introspection_ladder_comparison_detail(comparison)
    return None


def _introspection_ladder_comparison_detail(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    detail: dict[str, Any] = {}
    for field in ("from_stage", "to_stage", "status", "result", "path"):
        item = value.get(field)
        if isinstance(item, str) and item:
            detail[field] = item
    digest = _normalize_introspection_sha256(value.get("sha256"))
    if digest is not None:
        detail["sha256"] = f"sha256:{digest}"
    mismatch = value.get("first_mismatch")
    if isinstance(mismatch, Mapping):
        mismatch_detail = _introspection_first_mismatch_detail(mismatch)
        if mismatch_detail:
            detail["first_mismatch"] = mismatch_detail
    return detail


def _introspection_first_mismatch_detail(value: Mapping[str, Any]) -> dict[str, Any]:
    detail: dict[str, Any] = {}
    for field in INTROSPECTION_FIRST_MISMATCH_SCALAR_FIELDS:
        item = value.get(field)
        if _is_json_scalar(item):
            detail[field] = item
    return detail


def _is_json_scalar(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value)
    if isinstance(value, bool):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def _validate_introspection_comparison_ref(
    errors: list[str],
    value: Any,
    base_dir: Path | None,
    label: str,
) -> None:
    if not isinstance(value, Mapping):
        errors.append(f"{label} must be an object")
        return
    if value.get("schema") != "ares.introspection.compare.v1":
        errors.append(f"{label}.schema must be ares.introspection.compare.v1")
    for field in ("from_stage", "to_stage"):
        stage = value.get(field)
        if not isinstance(stage, str) or stage not in INTROSPECTION_STAGES:
            errors.append(f"{label}.{field} must be a known ladder stage")
    status = value.get("status")
    result = value.get("result")
    if status not in {"passed", "failed"}:
        errors.append(f"{label}.status must be passed or failed")
    if status == "passed" and result != "matches":
        errors.append(f"{label}.result must be matches when status is passed")
    if status == "failed":
        if result != "diverged":
            errors.append(f"{label}.result must be diverged when status is failed")
        mismatch = value.get("first_mismatch")
        if not isinstance(mismatch, Mapping):
            errors.append(f"{label}.first_mismatch must identify the divergence")
        else:
            _require_non_empty_string(
                errors,
                mismatch.get("id"),
                f"{label}.first_mismatch.id",
            )
    _validate_introspection_artifact_ref(
        errors,
        value,
        base_dir,
        label,
        expected_schema="ares.introspection.compare.v1",
    )


def _validate_introspection_artifact_ref(
    errors: list[str],
    value: Any,
    base_dir: Path | None,
    label: str,
    *,
    expected_schema: str,
    verify_path: bool = True,
) -> None:
    if not isinstance(value, Mapping):
        errors.append(f"{label} must be an object")
        return
    if value.get("schema") != expected_schema:
        errors.append(f"{label}.schema must be {expected_schema}")
    _require_introspection_sha256(errors, value.get("sha256"), f"{label}.sha256")
    if verify_path and "path" in value:
        _validate_referenced_introspection_sha256(errors, value, base_dir, label)


def _validate_referenced_introspection_sha256(
    errors: list[str],
    value: Mapping[str, Any],
    base_dir: Path | None,
    label: str,
) -> None:
    if base_dir is None:
        errors.append(
            f"{label}.sha256 cannot be verified without evidence path context"
        )
        return
    path_value = value.get("path")
    digest_value = _normalize_introspection_sha256(value.get("sha256"))
    if not isinstance(path_value, str) or digest_value is None:
        return
    path = Path(path_value)
    if not path.is_absolute():
        path = base_dir / path
    if not path.is_file():
        errors.append(f"{label}.path does not exist: {path}")
        return
    actual = _sha256_file(path)
    if actual != digest_value:
        errors.append(f"{label}.sha256 does not match referenced file")


def _token_result_generated_count(payload: Mapping[str, Any]) -> int | None:
  explicit = payload.get("generated_tokens", payload.get("generated_token_count"))
  if isinstance(explicit, int):
    return explicit
  cases = payload.get("cases")
  if not isinstance(cases, list):
    return None
  lengths = [
    case.get("candidate_length")
    for case in cases
    if isinstance(case, dict) and isinstance(case.get("candidate_length"), int)
  ]
  if not lengths:
    return None
  return min(lengths)


def _token_result_exact_match(payload: Mapping[str, Any]) -> bool:
  explicit = payload.get("exact_match")
  if isinstance(explicit, bool):
    return explicit
  cases = payload.get("cases")
  if not isinstance(cases, list) or not cases:
    return False
  return all(
    isinstance(case, dict) and case.get("exact_match") is True for case in cases
  )


def _validation(
  passed: bool,
  errors: Iterable[str],
  detail: dict[str, Any],
) -> ArtifactValidation:
  return ArtifactValidation(passed=passed, errors=tuple(errors), detail=detail)


def _read_regular_file_bytes(
  path: Path,
  *,
  maximum_bytes: int,
  label: str,
) -> bytes:
  """Read one immutable regular-file snapshot without following a final symlink."""
  flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
  flags |= getattr(os, "O_NOFOLLOW", 0)
  try:
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as handle:
      before = os.fstat(handle.fileno())
      if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} is not a regular file")
      if before.st_size > maximum_bytes:
        raise ValueError(f"{label} exceeds {maximum_bytes} bytes")
      data = handle.read(before.st_size + 1)
      after = os.fstat(handle.fileno())
  except OSError as exc:
    raise ValueError(f"could not read {label}: {exc}") from exc
  if len(data) > maximum_bytes:
    raise ValueError(f"{label} exceeds {maximum_bytes} bytes")
  if (
    not stat.S_ISREG(after.st_mode)
    or after.st_dev != before.st_dev
    or after.st_ino != before.st_ino
    or after.st_size != before.st_size
    or after.st_mtime_ns != before.st_mtime_ns
    or after.st_ctime_ns != before.st_ctime_ns
    or len(data) != before.st_size
  ):
    raise ValueError(f"{label} changed while being read")
  return data


def _json_structural_item_count(data: bytes, maximum_items: int) -> int:
  item_count = 1
  in_string = False
  escaped = False
  for byte in data:
    if in_string:
      if escaped:
        escaped = False
      elif byte == 0x5C:
        escaped = True
      elif byte == 0x22:
        in_string = False
    elif byte == 0x22:
      in_string = True
    elif byte in {0x2C, 0x3A}:
      item_count += 1
      if item_count > maximum_items:
        break
  return item_count


def _json_object_without_duplicate_keys(
  pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
  value: dict[str, Any] = {}
  for key, item in pairs:
    if key in value:
      raise ValueError(f"duplicate JSON object key {key!r}")
    value[key] = item
  return value


def _reject_nonstandard_json_constant(value: str) -> Any:
  raise ValueError(f"non-standard JSON constant {value!r}")


def _json_nesting_error(value: Any) -> str | None:
  stack: list[tuple[Any, int]] = [(value, 1)]
  while stack:
    item, depth = stack.pop()
    if depth > MAX_JSON_NESTING_DEPTH:
      return f"JSON nesting exceeds {MAX_JSON_NESTING_DEPTH} levels"
    if isinstance(item, dict):
      stack.extend((child, depth + 1) for child in item.values())
    elif isinstance(item, list):
      stack.extend((child, depth + 1) for child in item)
    elif isinstance(item, float) and not math.isfinite(item):
      return "JSON number is not finite"
  return None


def _strict_json_loads_bytes(
  data: bytes,
  *,
  label: str,
  maximum_items: int,
) -> Any:
  item_count = _json_structural_item_count(data, maximum_items)
  if item_count > maximum_items:
    raise ValueError(f"{label} exceeds {maximum_items} structural JSON items")
  try:
    encoded = data.decode("utf-8")
  except UnicodeDecodeError as exc:
    raise ValueError(f"{label} is not UTF-8 at byte {exc.start}") from exc
  try:
    value = json.loads(
      encoded,
      object_pairs_hook=_json_object_without_duplicate_keys,
      parse_constant=_reject_nonstandard_json_constant,
    )
  except json.JSONDecodeError as exc:
    raise ValueError(exc.msg) from exc
  nesting_error = _json_nesting_error(value)
  if nesting_error:
    raise ValueError(nesting_error)
  return value


def _scheduler_canonical_json(value: Any) -> str:
  return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _scheduler_nonnegative_u64(value: Any) -> bool:
  return (
    isinstance(value, int)
    and not isinstance(value, bool)
    and 0 <= value <= SCHEDULER_U64_MAX
  )


def _scheduler_positive_u64(value: Any) -> bool:
  return _scheduler_nonnegative_u64(value) and value > 0


def _scheduler_bounded_string(value: Any) -> bool:
  return (
    isinstance(value, str)
    and len(value) <= MAX_SCHEDULER_STRING_CHARS
    and not any(
      unicodedata.category(character) in {"Cc", "Cf", "Cs", "Zl", "Zp"}
      for character in value
    )
  )


def _scheduler_nonempty_string(value: Any) -> bool:
  return _scheduler_bounded_string(value) and bool(value)


def _scheduler_canonical_report_integer(
  value: Any,
  *,
  maximum: int,
  allow_empty: bool = False,
  positive: bool = False,
) -> bool:
  if allow_empty and value == "":
    return True
  if not isinstance(value, str) or re.fullmatch(r"0|[1-9][0-9]{0,19}", value) is None:
    return False
  parsed = int(value)
  return (parsed > 0 if positive else parsed >= 0) and parsed <= maximum


def _scheduler_qualified_id(value: Any, field: str) -> bool:
  return (
    isinstance(value, dict)
    and set(value) == {"arena_id", field}
    and _scheduler_positive_u64(value.get("arena_id"))
    and _scheduler_positive_u64(value.get(field))
  )


def _scheduler_unexpected_keys(
  value: Mapping[str, Any], expected: set[str]
) -> str | None:
  missing = sorted(expected - set(value))
  unexpected = sorted(set(value) - expected)
  if missing:
    return f"missing required fields {missing!r}"
  if unexpected:
    return f"contains unsupported fields {unexpected!r}"
  return None


def _scheduler_source_authority_error(value: Any) -> str | None:
  if not isinstance(value, dict):
    return "source authority is not an object"
  kind = value.get("kind")
  expected_keys = {
    "hf_export": {
      "kind",
      "manifest_sha256",
      "decode_state_contract_sha256",
    },
    "lean_fixture_synthetic_typed": {
      "kind",
      "typed_attention_view_sha256",
    },
  }.get(kind)
  if expected_keys is None:
    return f"unsupported source authority kind {kind!r}"
  if kind == "lean_fixture_synthetic_typed":
    selected_core_keys = [
      key for key in SCHEDULER_LEAN_FIXTURE_CORE_AUTHORITY_KEYS if key in value
    ]
    if len(selected_core_keys) != 1:
      return (
        "source authority 'lean_fixture_synthetic_typed' must carry exactly "
        "one of legacy_core_sha256 or successor_core_sha256"
      )
    expected_keys = expected_keys | {selected_core_keys[0]}
  if set(value) != expected_keys:
    return f"source authority keys do not match {kind!r}"
  if any(
    key != "kind"
    and (
      not isinstance(digest, str)
      or digest != digest.lower()
      or SHA256_RE.fullmatch(digest) is None
    )
    for key, digest in value.items()
  ):
    return f"source authority {kind!r} has a malformed digest"
  return None


def _scheduler_publication_authority_error(value: Any) -> str | None:
  if (
    not isinstance(value, dict)
    or set(value) != {"lane_id", "batch_id", "generation"}
    or not _scheduler_qualified_id(value.get("lane_id"), "lane_id")
    or not _scheduler_qualified_id(value.get("batch_id"), "batch_id")
    or not _scheduler_positive_u64(value.get("generation"))
  ):
    return "publication authority is invalid"
  if value["lane_id"]["arena_id"] != value["batch_id"]["arena_id"]:
    return "publication authority crosses arenas"
  if value["batch_id"]["batch_id"] != value["generation"]:
    return "publication batch id does not match publication generation"
  return None


def _scheduler_parse_canonical_json(
  fields: Mapping[str, Any], field: str, expected_type: type
) -> tuple[Any | None, str | None]:
  encoded = fields.get(field)
  if not isinstance(encoded, str):
    return None, f"{field} is not a string"
  try:
    value = json.loads(
      encoded,
      object_pairs_hook=_json_object_without_duplicate_keys,
      parse_constant=_reject_nonstandard_json_constant,
    )
  except (ValueError, RecursionError) as exc:
    return None, f"{field} is not valid bounded JSON: {exc}"
  nesting_error = _json_nesting_error(value)
  if nesting_error:
    return None, f"{field} is not valid bounded JSON: {nesting_error}"
  if not isinstance(value, expected_type):
    return None, f"{field} does not encode a {expected_type.__name__}"
  if encoded != _scheduler_canonical_json(value):
    return None, f"{field} is not canonical JSON"
  return value, None


def _scheduler_transport_report_error(fields: Mapping[str, Any]) -> str | None:
  missing = sorted(SCHEDULER_TRANSPORT_REPORT_FIELDS - set(fields))
  if missing:
    return f"generated-state transport report is missing fields {missing!r}"
  if (
    fields.get("generated_state_transport_schema")
    != SCHEDULER_GENERATED_STATE_TRANSPORT_SCHEMA
  ):
    return "generated_state_transport_schema is not v3"
  if fields.get("generated_state_event") != "publish":
    return "generated_state_event is not publish"
  for digest_key in ("ares_plan_sha256", "target_plan_sha256"):
    digest = fields.get(digest_key)
    if (
      not isinstance(digest, str)
      or digest != digest.lower()
      or SHA256_RE.fullmatch(digest) is None
    ):
      return f"{digest_key} is malformed"
  targetplan_op_ids = fields.get("targetplan_op_ids")
  if (
    not isinstance(targetplan_op_ids, list)
    or not targetplan_op_ids
    or not all(_scheduler_nonempty_string(op_id) for op_id in targetplan_op_ids)
    or len(set(targetplan_op_ids)) != len(targetplan_op_ids)
  ):
    return "targetplan_op_ids is not a non-empty unique identifier array"
  source, error = _scheduler_parse_canonical_json(
    fields, "source_authority_json", dict
  )
  if error:
    return error
  source_error = _scheduler_source_authority_error(source)
  if source_error:
    return f"source_authority_json: {source_error}"
  authority, error = _scheduler_parse_canonical_json(
    fields, "publication_authority_json", dict
  )
  if error:
    return error
  authority_error = _scheduler_publication_authority_error(authority)
  if authority_error:
    return f"publication_authority_json: {authority_error}"
  return None


def _scheduler_publication_lane_error(lane: Any) -> str | None:
  if not isinstance(lane, dict):
    return "publication lane projection is not an object"
  expected_fields = set(SCHEDULER_PUBLICATION_LANE_PROJECTION_FIELDS)
  shape_error = _scheduler_unexpected_keys(lane, expected_fields)
  if shape_error:
    return f"publication lane projection {shape_error}"
  if not _scheduler_nonnegative_u64(lane.get("publication_index")):
    return "publication lane projection index is not a u64"
  lane_id = lane.get("publication_lane_id")
  if not _scheduler_qualified_id(lane_id, "lane_id"):
    return "publication lane projection id is invalid"
  owners = lane.get("publication_owner_indices")
  if (
    not isinstance(owners, list)
    or not owners
    or not all(_scheduler_nonnegative_u64(owner) for owner in owners)
    or owners != sorted(set(owners))
  ):
    return "publication lane projection owners are not sorted unique u64 values"
  if (
    lane.get("generated_state_transport_schema")
    != SCHEDULER_GENERATED_STATE_TRANSPORT_SCHEMA
  ):
    return "publication lane projection transport schema is not v3"
  if lane.get("event") != "publish":
    return "publication lane projection event is not publish"
  for field in ("state_checkpoint_id", "state_branch_id"):
    if not _scheduler_positive_u64(lane.get(field)):
      return f"publication lane projection {field} is not a positive u64"
  parents = (
    lane.get("state_parent_checkpoint_id"),
    lane.get("state_parent_branch_id"),
  )
  if any(value is None for value in parents) != all(value is None for value in parents):
    return "publication lane projection parent checkpoint/branch presence differs"
  if any(value is not None and not _scheduler_positive_u64(value) for value in parents):
    return "publication lane projection parent identity is not null or positive u64"
  for field in ("model_id", "backend_id"):
    if not _scheduler_nonempty_string(lane.get(field)):
      return f"publication lane projection {field} is missing or empty"
  for digest_key in ("ares_plan_sha256", "target_plan_sha256"):
    digest = lane.get(digest_key)
    if (
      not isinstance(digest, str)
      or digest != digest.lower()
      or SHA256_RE.fullmatch(digest) is None
    ):
      return f"publication lane projection {digest_key} is malformed"
  targetplan_op_ids = lane.get("targetplan_op_ids")
  if (
    not isinstance(targetplan_op_ids, list)
    or not targetplan_op_ids
    or not all(_scheduler_nonempty_string(op_id) for op_id in targetplan_op_ids)
    or len(set(targetplan_op_ids)) != len(targetplan_op_ids)
  ):
    return "publication lane projection TargetPlan operation ids are invalid"
  if not _scheduler_positive_u64(lane.get("qualified_row_count")):
    return "publication lane projection qualified row count is not positive"
  qualified_digest = lane.get("qualified_rows_sha256")
  if (
    not isinstance(qualified_digest, str)
    or qualified_digest != qualified_digest.lower()
    or SHA256_RE.fullmatch(qualified_digest) is None
  ):
    return "publication lane projection qualified rows digest is malformed"
  source_error = _scheduler_source_authority_error(lane.get("source_authority"))
  if source_error:
    return f"publication lane projection {source_error}"
  authority = lane.get("publication_authority")
  authority_error = _scheduler_publication_authority_error(authority)
  if authority_error:
    return f"publication lane projection {authority_error}"
  if authority["lane_id"] != lane_id:
    return "publication lane projection id does not match publication authority"
  return None


def _scheduler_publication_report_error(fields: Mapping[str, Any]) -> str | None:
  missing = sorted(SCHEDULER_PUBLICATION_REPORT_FIELDS - set(fields))
  if missing:
    return f"publication report is missing fields {missing!r}"
  lanes = fields.get("publication_lanes")
  if not isinstance(lanes, list) or not lanes:
    return "publication_lanes is not a non-empty array"
  publication_identities: dict[tuple[int, int, int, int], int] = {}
  physical_lane_ids: set[tuple[int, int]] = set()
  physical_lane_records: dict[tuple[int, int], list[tuple[int, dict[str, Any]]]] = {}
  shared_plan_identity: str | None = None
  for index, lane in enumerate(lanes):
    lane_error = _scheduler_publication_lane_error(lane)
    if lane_error:
      return f"publication_lanes[{index}]: {lane_error}"
    if lane["publication_index"] != index:
      return f"publication_lanes[{index}] has a non-sequential publication index"
    lane_id = lane["publication_lane_id"]
    lane_key = (lane_id["arena_id"], lane_id["lane_id"])
    physical_lane_ids.add(lane_key)
    physical_lane_records.setdefault(lane_key, []).append((index, lane))
    publication_identity = (
      *lane_key,
      lane["state_checkpoint_id"],
      lane["state_branch_id"],
    )
    if publication_identity in publication_identities:
      return f"publication_lanes[{index}] duplicates a publication identity"
    publication_identities[publication_identity] = index
    plan_identity = _scheduler_canonical_json(
      {
        "model_id": lane["model_id"],
        "backend_id": lane["backend_id"],
        "ares_plan_sha256": lane["ares_plan_sha256"],
        "target_plan_sha256": lane["target_plan_sha256"],
        "source_authority": lane["source_authority"],
        "targetplan_op_ids": lane["targetplan_op_ids"],
      }
    )
    if shared_plan_identity is None:
      shared_plan_identity = plan_identity
    elif shared_plan_identity != plan_identity:
      return f"publication_lanes[{index}] has a different batch plan identity"
  for index, lane in enumerate(lanes):
    parent_checkpoint = lane["state_parent_checkpoint_id"]
    if parent_checkpoint is None:
      continue
    lane_id = lane["publication_lane_id"]
    parent_identity = (
      lane_id["arena_id"],
      lane_id["lane_id"],
      parent_checkpoint,
      lane["state_parent_branch_id"],
    )
    parent_index = publication_identities.get(parent_identity)
    if parent_index is not None and parent_index >= index:
      return f"publication_lanes[{index}] parent publication is not prior"
  for lane_records in physical_lane_records.values():
    root_indices = [
      index
      for index, lane in lane_records
      if lane["state_parent_checkpoint_id"] is None
    ]
    if len(root_indices) > 1:
      return "publication_lanes has multiple roots for one physical lane"
    if root_indices and root_indices[0] != lane_records[0][0]:
      return "publication_lanes physical lane root is not first"
    prior_identities: set[tuple[int, int]] = set()
    for record_index, (index, lane) in enumerate(lane_records):
      if record_index:
        parent_identity = (
          lane["state_parent_checkpoint_id"],
          lane["state_parent_branch_id"],
        )
        if parent_identity not in prior_identities:
          return (
            "publication_lanes physical lane has a non-prior parent "
            f"at index {index}"
          )
      prior_identities.add((lane["state_checkpoint_id"], lane["state_branch_id"]))
  lane_count = fields.get("publication_lane_count")
  if not isinstance(lane_count, str) or re.fullmatch(r"[1-9][0-9]*", lane_count) is None:
    return "publication_lane_count is not an integer string"
  if lane_count != str(len(physical_lane_ids)):
    return "publication_lane_count does not match distinct publication lanes"
  record_count = fields.get("publication_record_count")
  if not isinstance(record_count, str) or re.fullmatch(r"[1-9][0-9]*", record_count) is None:
    return "publication_record_count is not an integer string"
  if record_count != str(len(lanes)):
    return "publication_record_count does not match publication_lanes"
  parsed_lanes, error = _scheduler_parse_canonical_json(
    fields, "publication_lanes_json", list
  )
  if error:
    return error
  if parsed_lanes != lanes:
    return "publication_lanes_json does not match publication_lanes"
  expected_summaries = {
    "generated_state_transport_schema": lanes[0]["generated_state_transport_schema"],
    "generated_state_event": lanes[0]["event"],
    "ares_plan_sha256": lanes[0]["ares_plan_sha256"],
    "target_plan_sha256": lanes[0]["target_plan_sha256"],
    "targetplan_op_ids": lanes[0]["targetplan_op_ids"],
    "source_authority_json": _scheduler_canonical_json(
      [lane["source_authority"] for lane in lanes]
    ),
    "publication_authority_json": _scheduler_canonical_json(
      [lane["publication_authority"] for lane in lanes]
    ),
  }
  for field, expected in expected_summaries.items():
    if fields.get(field) != expected:
      return f"{field} does not match publication_lanes"
  if fields.get("model_id") != lanes[0]["model_id"]:
    return "publication model_id does not match publication_lanes"
  if fields.get("backend_id") != lanes[0]["backend_id"]:
    return "publication backend_id does not match publication_lanes"
  batch_ids = {
    _scheduler_canonical_json(lane["publication_authority"]["batch_id"])
    for lane in lanes
  }
  if len(batch_ids) != 1:
    return "publication_lanes do not share one publication batch id"
  batch_id, error = _scheduler_parse_canonical_json(
    fields, "publication_batch_id_json", dict
  )
  if error:
    return error
  if not _scheduler_qualified_id(batch_id, "batch_id"):
    return "publication_batch_id_json is not a qualified batch id"
  if batch_id != lanes[0]["publication_authority"]["batch_id"]:
    return "publication_batch_id_json does not match publication_lanes"
  generations = {lane["publication_authority"]["generation"] for lane in lanes}
  if len(generations) != 1:
    return "publication_lanes do not share one publication generation"
  generation = next(iter(generations))
  if fields.get("publication_batch_generation") != str(generation):
    return "publication_batch_generation does not match publication_lanes"
  if batch_id["batch_id"] != generation:
    return "publication_batch_id_json does not match publication generation"
  route_owner = fields.get("publication_route_owner_index")
  if not isinstance(route_owner, str) or re.fullmatch(
    r"(?:|0|[1-9][0-9]*)", route_owner
  ) is None:
    return "publication_route_owner_index is not empty or a nonnegative integer string"
  if route_owner and any(
    int(route_owner) not in lane["publication_owner_indices"] for lane in lanes
  ):
    return "publication_route_owner_index is absent from a publication lane"
  return None


def _scheduler_lineage_report_row_error(row: Any) -> str | None:
  if not isinstance(row, dict):
    return "scheduler lineage report row is not an object"
  row_kind = row.get("row_kind")
  status = row.get("status")
  evidence_role = row.get("evidence_role")
  failure_reason = row.get("failure_reason", "")
  if not isinstance(failure_reason, str):
    return "scheduler lineage report failure_reason is not a string"
  if failure_reason and not _scheduler_bounded_string(failure_reason):
    return "scheduler lineage report failure_reason contains unsupported text"
  generated_fields = SCHEDULER_TRANSPORT_REPORT_FIELDS & set(row)
  publication_fields = (
    SCHEDULER_PUBLICATION_REPORT_FIELDS - SCHEDULER_TRANSPORT_REPORT_FIELDS
  ) & set(row)
  direct_fields = SCHEDULER_DIRECT_TRANSPORT_REPORT_FIELDS & set(row)
  if row_kind in {"", "scheduler_forward_batch"}:
    expected_fields = SCHEDULER_REPORT_COMMON_FIELDS
  elif row_kind == "generated_state_transport":
    expected_fields = (
      SCHEDULER_REPORT_COMMON_FIELDS
      | SCHEDULER_TRANSPORT_REPORT_FIELDS
      | SCHEDULER_DIRECT_TRANSPORT_REPORT_FIELDS
    )
  elif row_kind == "generated_state_publication_batch":
    expected_fields = SCHEDULER_REPORT_COMMON_FIELDS | SCHEDULER_PUBLICATION_REPORT_FIELDS
  else:
    return f"unsupported scheduler lineage report row kind {row_kind!r}"
  if row_kind in {"", "scheduler_forward_batch"} and (
    generated_fields or publication_fields or direct_fields
  ):
    return f"{row_kind or 'scheduler lineage sentinel'} contains generated-state fields"
  shape_error = _scheduler_unexpected_keys(row, set(expected_fields))
  if shape_error:
    return f"scheduler lineage report {shape_error}"
  if row_kind == "":
    if status not in {"missing_sidecar", "invalid_metadata", "invalid_jsonl"}:
      return "scheduler lineage sentinel status is invalid"
    if evidence_role != "":
      return "scheduler lineage sentinel evidence role is not empty"
    empty_fields = SCHEDULER_REPORT_COMMON_FIELDS - {
      "artifact_index",
      "status",
      "failure_reason",
    }
    if any(row.get(field) != "" for field in empty_fields):
      return "scheduler lineage sentinel contains nonempty projected evidence"
    if status in {"invalid_metadata", "invalid_jsonl"} and not failure_reason:
      return "scheduler lineage invalid sentinel has no failure reason"
    return None
  for field in ("artifact_index", "row_index"):
    if not _scheduler_nonnegative_u64(row.get(field)):
      return f"scheduler lineage report {field} is not a u64"
  for field in ("trace_run_id", "process_kind", "model_id", "request_id"):
    if not _scheduler_nonempty_string(row.get(field)):
      return f"scheduler lineage report {field} is missing or invalid"
  if row.get("process_kind") != "rinzler":
    return "scheduler lineage report process_kind is not rinzler"
  request_id = row.get("request_id")
  if (
    not isinstance(request_id, str)
    or re.fullmatch(r"0|[1-9][0-9]{0,19}", request_id) is None
    or int(request_id) > SCHEDULER_U64_MAX
  ):
    return "scheduler lineage report request_id is not canonical"
  if row.get("generation_id") != f"rinzler-{request_id}":
    return "scheduler lineage report generation_id does not match request_id"
  backend_id = row.get("backend_id")
  if not isinstance(backend_id, str) or (backend_id and not _scheduler_nonempty_string(backend_id)):
    return "scheduler lineage report backend_id is invalid"
  if row_kind != "scheduler_forward_batch" and not backend_id:
    return "generated-state report backend_id is empty"
  if row_kind == "scheduler_forward_batch":
    if status not in {"ok", "error"}:
      return "scheduler-forward report status is not ok or error"
    if evidence_role != "system_under_test":
      return "scheduler-forward report evidence role is not system_under_test"
    if status == "ok" and failure_reason:
      return "scheduler-forward ok report has a failure reason"
    if status == "error" and not failure_reason:
      return "scheduler-forward error report has no failure reason"
    for field in ("parent_location_id", "location_id"):
      if not _scheduler_canonical_report_integer(
        row.get(field), maximum=SCHEDULER_U64_MAX, allow_empty=True
      ):
        return f"scheduler-forward report {field} is not empty or a canonical u64"
    if status == "ok" and not row.get("location_id"):
      return "scheduler-forward ok report has no location id"
    if status == "error" and row.get("location_id"):
      return "scheduler-forward error report has a location id"
    for field in ("executor_shape", "executor_status", "attention_mode"):
      value = row.get(field)
      if value and (
        not isinstance(value, str)
        or len(value) > 256
        or re.fullmatch(r"[a-z0-9][a-z0-9_.:-]*", value) is None
      ):
        return f"scheduler-forward report {field} is not a bounded identifier"
    u64_fields = {
      "runtime_request_token_count",
      "token_job_count",
      "kv_job_count",
      "listener_sparse_rows",
      "listener_sparse_tokens",
      "kv_save_rows",
      "kv_context_rows",
      "visible_token_slots",
      "kv_page_count",
      "hw_shard_allocation_requests",
      "hw_gof_page_infos",
      "minibatch_count",
      "sparse_topk_rows",
      "sparse_topk_token_count",
      "prior_hw_shard_allocation_requests",
      "prior_host_gof_ready_rows",
      "prior_host_gof_pending_dma_rows",
      "prior_host_gof_cacheblock_dma_plans",
      "prior_host_gof_materialized_cacheblocks",
      "prior_host_gof_dma_completions",
      "prior_host_gof_page_infos",
    }
    for field in u64_fields:
      if not _scheduler_canonical_report_integer(
        row.get(field),
        maximum=SCHEDULER_U64_MAX,
        allow_empty=field != "runtime_request_token_count",
      ):
        return f"scheduler-forward report {field} is not empty or a canonical u64"
    if not _scheduler_canonical_report_integer(
      row.get("tokens_reused"), maximum=SCHEDULER_I64_MAX
    ):
      return "scheduler-forward report tokens_reused is not a canonical nonnegative i64"
    staging_status = row.get("prior_host_gof_staging_status")
    if staging_status and (
      not isinstance(staging_status, str)
      or len(staging_status) > 256
      or re.fullmatch(r"[a-z0-9][a-z0-9_.:-]*", staging_status) is None
    ):
      return "scheduler-forward report prior_host_gof_staging_status is invalid"
    if status == "error":
      zero_fields = {
        "listener_sparse_tokens",
        "tokens_reused",
        "sparse_topk_rows",
        "sparse_topk_token_count",
      }
      empty_fields = (
        u64_fields - {"runtime_request_token_count"} - zero_fields
      ) | {
        "executor_shape",
        "executor_status",
        "attention_mode",
        "prior_host_gof_staging_status",
      }
      if any(row.get(field) != "0" for field in zero_fields) or any(
        row.get(field) != "" for field in empty_fields
      ):
        return "scheduler-forward error report contains successful result metadata"
    return None
  if row_kind == "generated_state_transport":
    if status != "present":
      return "generated-state transport report status is not present"
    if evidence_role != "system_under_test":
      return "generated-state transport report evidence role is not system_under_test"
    if publication_fields:
      return "generated-state transport report contains publication-batch fields"
    if failure_reason:
      return "generated-state transport report has a failure reason"
    shared_error = _scheduler_transport_report_error(row)
    if shared_error:
      return shared_error
    for field in ("state_checkpoint_id", "state_branch_id", "qualified_row_count"):
      if not _scheduler_canonical_report_integer(
        row.get(field), maximum=SCHEDULER_U64_MAX, positive=True
      ):
        return f"generated-state transport report {field} is not positive"
    parents = (
      row.get("state_parent_checkpoint_id"),
      row.get("state_parent_branch_id"),
    )
    if any(value == "" for value in parents) != all(value == "" for value in parents):
      return "generated-state transport report parent presence differs"
    if any(
      not _scheduler_canonical_report_integer(
        value,
        maximum=SCHEDULER_U64_MAX,
        allow_empty=True,
        positive=True,
      )
      for value in parents
    ):
      return "generated-state transport report parent identity is invalid"
    digest = row.get("qualified_rows_sha256")
    if (
      not isinstance(digest, str)
      or digest != digest.lower()
      or SHA256_RE.fullmatch(digest) is None
    ):
      return "generated-state transport report qualified rows digest is malformed"
    return None
  if status != "present":
    return "publication report status is not present"
  if evidence_role != "system_under_test":
    return "publication report evidence role is not system_under_test"
  if failure_reason:
    return "publication report has a failure reason"
  return _scheduler_publication_report_error(row)


def _validate_scheduler_packet_lineage_report_rows(
  errors: list[str], rows: list[dict[str, Any]]
) -> None:
  for index, row in enumerate(rows):
    error = _scheduler_lineage_report_row_error(row)
    if error:
      errors.append(
        "trace report sections.scheduler_packet_lineage_sidecar_rows"
        f"[{index}]: {error}"
      )


def _scheduler_as_string(value: Any) -> str:
  if value is None:
    return ""
  if isinstance(value, (str, int, float, bool)):
    return str(value)
  return json.dumps(value, sort_keys=True)


def _scheduler_common_projection(
  raw_row: Mapping[str, Any], artifact_index: int, row_index: int
) -> dict[str, Any]:
  failure_reason = json.dumps(
    _scheduler_as_string(raw_row.get("failure_reason")), ensure_ascii=True
  )[1:-1][:MAX_SCHEDULER_STRING_CHARS]
  return {
    "artifact_index": artifact_index,
    "row_index": row_index,
    "row_kind": _scheduler_as_string(raw_row.get("row_kind")),
    "status": _scheduler_as_string(raw_row.get("status")) or "present",
    "evidence_role": _scheduler_as_string(raw_row.get("evidence_role")),
    "trace_run_id": _scheduler_as_string(raw_row.get("trace_run_id")),
    "process_kind": _scheduler_as_string(raw_row.get("process_kind")),
    "model_id": _scheduler_as_string(raw_row.get("model_id")),
    "backend_id": _scheduler_as_string(raw_row.get("backend_id")),
    "request_id": _scheduler_as_string(raw_row.get("request_id")),
    "generation_id": _scheduler_as_string(raw_row.get("generation_id")),
    "parent_location_id": _scheduler_as_string(raw_row.get("parent_location_id")),
    "location_id": _scheduler_as_string(raw_row.get("location_id")),
    "executor_shape": _scheduler_as_string(
      raw_row.get("scheduler_executor_shape")
    ),
    "executor_status": _scheduler_as_string(
      raw_row.get("scheduler_batch_executor_status")
    ),
    "attention_mode": _scheduler_as_string(
      raw_row.get("scheduler_batch_attention_mode")
    ),
    "runtime_request_token_count": _scheduler_as_string(
      raw_row.get("runtime_request_token_count")
    ),
    "tokens_reused": _scheduler_as_string(raw_row.get("tokens_reused")),
    "token_job_count": _scheduler_as_string(
      raw_row.get("scheduler_batch_token_job_count")
    ),
    "kv_job_count": _scheduler_as_string(
      raw_row.get("scheduler_batch_kv_job_count")
    ),
    "listener_sparse_rows": _scheduler_as_string(
      raw_row.get("scheduler_batch_listener_sparse_logit_row_count")
    )
    or _scheduler_as_string(
      raw_row.get(
        "scheduler_batch_fullscheduler_forward_batch_v1_listener_sparse_row_count"
      )
    ),
    "listener_sparse_tokens": _scheduler_as_string(
      raw_row.get("scheduler_batch_listener_sparse_logit_token_count")
    )
    or _scheduler_as_string(raw_row.get("sparse_topk_token_count")),
    "kv_save_rows": _scheduler_as_string(
      raw_row.get("scheduler_batch_kv_save_count")
    )
    or _scheduler_as_string(
      raw_row.get("scheduler_batch_fullscheduler_forward_batch_v1_kv_save_row_count")
    ),
    "kv_context_rows": _scheduler_as_string(
      raw_row.get("scheduler_batch_kv_context_row_count")
    ),
    "visible_token_slots": _scheduler_as_string(
      raw_row.get("scheduler_batch_visible_token_slot_count")
    ),
    "kv_page_count": _scheduler_as_string(raw_row.get("scheduler_batch_kv_page_count")),
    "hw_shard_allocation_requests": _scheduler_as_string(
      raw_row.get("scheduler_batch_hw_shard_allocation_request_count")
    ),
    "hw_gof_page_infos": _scheduler_as_string(
      raw_row.get("scheduler_batch_hw_gof_page_info_count")
    ),
    "minibatch_count": _scheduler_as_string(
      raw_row.get("scheduler_batch_fullscheduler_forward_batch_v1_minibatch_count")
    ),
    "sparse_topk_rows": _scheduler_as_string(raw_row.get("sparse_topk_row_count")),
    "sparse_topk_token_count": _scheduler_as_string(
      raw_row.get("sparse_topk_token_count")
    ),
    "prior_hw_shard_allocation_requests": _scheduler_as_string(
      raw_row.get("scheduler_prior_hw_shard_allocation_request_count")
    ),
    "prior_host_gof_ready_rows": _scheduler_as_string(
      raw_row.get("scheduler_prior_host_gof_ready_row_count")
    ),
    "prior_host_gof_pending_dma_rows": _scheduler_as_string(
      raw_row.get("scheduler_prior_host_gof_pending_dma_row_count")
    ),
    "prior_host_gof_cacheblock_dma_plans": _scheduler_as_string(
      raw_row.get("scheduler_prior_host_gof_cacheblock_dma_plan_count")
    ),
    "prior_host_gof_materialized_cacheblocks": _scheduler_as_string(
      raw_row.get("scheduler_prior_host_gof_materialized_cacheblock_count")
    ),
    "prior_host_gof_dma_completions": _scheduler_as_string(
      raw_row.get("scheduler_prior_host_gof_dma_completion_count")
    ),
    "prior_host_gof_page_infos": _scheduler_as_string(
      raw_row.get("scheduler_prior_host_gof_page_info_count")
    ),
    "prior_host_gof_staging_status": _scheduler_as_string(
      raw_row.get("scheduler_prior_host_gof_staging_status")
    ),
    "failure_reason": failure_reason,
  }


def _scheduler_lane_projections(raw_row: Mapping[str, Any]) -> list[dict[str, Any]]:
  lanes = raw_row.get("lanes")
  if not isinstance(lanes, list) or not lanes:
    raise ValueError("raw publication batch has no lanes")
  projections: list[dict[str, Any]] = []
  for publication_index, lane in enumerate(lanes):
    if not isinstance(lane, dict):
      raise ValueError(f"raw publication lane {publication_index} is not an object")
    rows = lane.get("rows")
    if not isinstance(rows, list) or not rows:
      raise ValueError(f"raw publication lane {publication_index} has no rows")
    try:
      projection = {
        "publication_index": publication_index,
        **{
          field: lane[field]
          for field in SCHEDULER_PUBLICATION_LANE_PROJECTION_FIELDS
          if field
          not in {"publication_index", "qualified_row_count", "qualified_rows_sha256"}
        },
        "qualified_row_count": len(rows),
        "qualified_rows_sha256": hashlib.sha256(
          _scheduler_canonical_json(rows).encode("utf-8")
        ).hexdigest(),
      }
    except KeyError as exc:
      raise ValueError(
        f"raw publication lane {publication_index} is missing {exc.args[0]}"
      ) from exc
    projections.append(projection)
  return projections


def _scheduler_project_raw_row(
  raw_row: Any, artifact_index: int, row_index: int
) -> dict[str, Any]:
  if not isinstance(raw_row, dict):
    raise ValueError("raw scheduler row is not an object")
  if raw_row.get("schema_version") != 2:
    raise ValueError("raw scheduler row schema_version is not 2")
  projected = _scheduler_common_projection(raw_row, artifact_index, row_index)
  row_kind = projected["row_kind"]
  if row_kind == "scheduler_forward_batch":
    return projected
  if row_kind == "generated_state_transport":
    rows = raw_row.get("rows")
    if not isinstance(rows, list) or not rows:
      raise ValueError("raw generated-state transport has no qualified rows")
    source = raw_row.get("source_authority")
    authority = raw_row.get("publication_authority")
    projected.update(
      {
        "generated_state_transport_schema": _scheduler_as_string(
          raw_row.get("generated_state_transport_schema")
        ),
        "generated_state_event": _scheduler_as_string(raw_row.get("event")),
        "ares_plan_sha256": _scheduler_as_string(raw_row.get("ares_plan_sha256")),
        "target_plan_sha256": _scheduler_as_string(
          raw_row.get("target_plan_sha256")
        ),
        "targetplan_op_ids": raw_row.get("targetplan_op_ids", []),
        "state_checkpoint_id": _scheduler_as_string(
          raw_row.get("state_checkpoint_id")
        ),
        "state_branch_id": _scheduler_as_string(raw_row.get("state_branch_id")),
        "state_parent_checkpoint_id": _scheduler_as_string(
          raw_row.get("state_parent_checkpoint_id")
        ),
        "state_parent_branch_id": _scheduler_as_string(
          raw_row.get("state_parent_branch_id")
        ),
        "qualified_row_count": str(len(rows)),
        "qualified_rows_sha256": hashlib.sha256(
          _scheduler_canonical_json(rows).encode("utf-8")
        ).hexdigest(),
        "source_authority_json": (
          _scheduler_canonical_json(source) if isinstance(source, dict) else ""
        ),
        "publication_authority_json": (
          _scheduler_canonical_json(authority)
          if isinstance(authority, dict)
          else ""
        ),
      }
    )
    return projected
  if row_kind != "generated_state_publication_batch":
    raise ValueError(f"raw scheduler row kind {row_kind!r} is unsupported")
  lane_projections = _scheduler_lane_projections(raw_row)
  raw_lanes = raw_row["lanes"]
  first_lane = lane_projections[0]
  publication_batch_id = raw_row.get("publication_batch_id")
  distinct_lanes = {
    _scheduler_canonical_json(lane["publication_lane_id"])
    for lane in lane_projections
  }
  projected.update(
    {
      "model_id": first_lane["model_id"],
      "backend_id": first_lane["backend_id"],
      "generation_id": _scheduler_as_string(raw_lanes[0].get("generation_id")),
      "generated_state_transport_schema": first_lane[
        "generated_state_transport_schema"
      ],
      "generated_state_event": first_lane["event"],
      "ares_plan_sha256": first_lane["ares_plan_sha256"],
      "target_plan_sha256": first_lane["target_plan_sha256"],
      "targetplan_op_ids": first_lane["targetplan_op_ids"],
      "source_authority_json": _scheduler_canonical_json(
        [lane["source_authority"] for lane in lane_projections]
      ),
      "publication_authority_json": _scheduler_canonical_json(
        [lane["publication_authority"] for lane in lane_projections]
      ),
      "publication_lanes": lane_projections,
      "publication_lanes_json": _scheduler_canonical_json(lane_projections),
      "publication_batch_id_json": (
        _scheduler_canonical_json(publication_batch_id)
        if isinstance(publication_batch_id, dict)
        else ""
      ),
      "publication_batch_generation": _scheduler_as_string(
        raw_row.get("publication_batch_generation")
      ),
      "publication_lane_count": str(len(distinct_lanes)),
      "publication_record_count": str(len(raw_lanes)),
      "publication_route_owner_index": _scheduler_as_string(
        raw_row.get("publication_route_owner_index")
      ),
    }
  )
  return projected


def _scheduler_jsonl_rows(data: bytes) -> list[dict[str, Any]]:
  if len(data) > MAX_SCHEDULER_LINEAGE_JSONL_BYTES:
    raise ValueError(
      "scheduler packet-lineage JSONL exceeds "
      f"{MAX_SCHEDULER_LINEAGE_JSONL_BYTES} bytes"
    )
  rows: list[dict[str, Any]] = []
  aggregate_items = 0
  start = 0
  physical_rows = 0
  while start < len(data):
    physical_rows += 1
    if physical_rows > MAX_SCHEDULER_LINEAGE_JSONL_ROWS:
      raise ValueError(
        "scheduler packet-lineage JSONL exceeds "
        f"{MAX_SCHEDULER_LINEAGE_JSONL_ROWS} physical rows"
      )
    newline = data.find(b"\n", start)
    if newline < 0:
      encoded_line = data[start:]
      start = len(data)
    else:
      encoded_line = data[start:newline]
      start = newline + 1
    if not encoded_line.strip():
      continue
    if len(encoded_line) > MAX_SCHEDULER_LINEAGE_JSONL_LINE_BYTES:
      raise ValueError(
        f"scheduler packet-lineage JSONL line {physical_rows} exceeds "
        f"{MAX_SCHEDULER_LINEAGE_JSONL_LINE_BYTES} bytes"
      )
    item_count = _json_structural_item_count(
      encoded_line, MAX_SCHEDULER_LINEAGE_JSON_ITEMS_PER_ROW
    )
    if item_count > MAX_SCHEDULER_LINEAGE_JSON_ITEMS_PER_ROW:
      raise ValueError(
        f"scheduler packet-lineage JSONL line {physical_rows} exceeds "
        f"{MAX_SCHEDULER_LINEAGE_JSON_ITEMS_PER_ROW} structural JSON items"
      )
    aggregate_items += item_count
    if aggregate_items > MAX_SCHEDULER_LINEAGE_JSON_ITEMS:
      raise ValueError(
        "scheduler packet-lineage JSONL exceeds "
        f"{MAX_SCHEDULER_LINEAGE_JSON_ITEMS} structural JSON items"
      )
    value = _strict_json_loads_bytes(
      encoded_line,
      label=f"scheduler packet-lineage JSONL line {physical_rows}",
      maximum_items=MAX_SCHEDULER_LINEAGE_JSON_ITEMS_PER_ROW,
    )
    if not isinstance(value, dict):
      raise ValueError(
        f"scheduler packet-lineage JSONL line {physical_rows} is not an object"
      )
    rows.append(value)
  return rows


def _resolve_report_reference(path_value: str, owner_path: Path) -> Path:
  path = Path(path_value)
  return path if path.is_absolute() else owner_path.parent / path


def _scheduler_lean_authority_bindings(
  raw_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
  bindings: list[dict[str, Any]] = []
  for row in raw_rows:
    row_kind = row.get("row_kind")
    candidates: list[Mapping[str, Any]] = []
    if row_kind == "generated_state_transport":
      candidates = [row]
    elif row_kind == "generated_state_publication_batch":
      lanes = row.get("lanes")
      if isinstance(lanes, list):
        candidates = [lane for lane in lanes if isinstance(lane, Mapping)]
    for candidate in candidates:
      source_authority = candidate.get("source_authority")
      if (
        isinstance(source_authority, Mapping)
        and source_authority.get("kind") == "lean_fixture_synthetic_typed"
      ):
        bindings.append(
          {
            "model_id": candidate.get("model_id"),
            "backend_id": candidate.get("backend_id"),
            "ares_plan_sha256": candidate.get("ares_plan_sha256"),
            "target_plan_sha256": candidate.get("target_plan_sha256"),
            "source_authority": dict(source_authority),
          }
        )
  return bindings


def _validate_scheduler_lean_authority(
  errors: list[str],
  raw_rows: list[dict[str, Any]],
  authority_root: Path | None,
) -> None:
  bindings = _scheduler_lean_authority_bindings(raw_rows)
  if not bindings:
    return
  if authority_root is None:
    errors.append(
      "scheduler lineage Lean fixture authority requires an independent Ares root"
    )
    return
  manifest_path = authority_root / PROVIDER_ADMISSION_MANIFEST_RELATIVE
  try:
    manifest_data = _read_regular_file_bytes(
      manifest_path,
      maximum_bytes=MAX_TRACE_METADATA_BYTES,
      label="provider admission manifest",
    )
    manifest = _strict_json_loads_bytes(
      manifest_data,
      label="provider admission manifest",
      maximum_items=MAX_TRACE_METADATA_JSON_ITEMS,
    )
  except ValueError as exc:
    errors.append(str(exc))
    return
  if (
    not isinstance(manifest, dict)
    or manifest.get("schema") != "ares.provider_fixture_admission_manifest.v1"
    or manifest.get("schema_version") != 1
  ):
    errors.append("provider admission manifest is not schema version 1")
    return
  entries = manifest.get("entries")
  if not isinstance(entries, list):
    errors.append("provider admission manifest entries is not an array")
    return

  unique_bindings = {
    _scheduler_canonical_json(binding): binding for binding in bindings
  }
  for binding in unique_bindings.values():
    source = binding["source_authority"]
    expected_source_sha = "sha256:" + str(binding["ares_plan_sha256"])
    expected_target_sha = "sha256:" + str(binding["target_plan_sha256"])
    selected_core_keys = [
      key for key in SCHEDULER_LEAN_FIXTURE_CORE_AUTHORITY_KEYS if key in source
    ]
    if len(selected_core_keys) != 1:
      errors.append(
        "scheduler lineage Lean fixture authority does not select exactly one "
        "core authority"
      )
      continue
    core_key = selected_core_keys[0]
    expected_core_sha = "sha256:" + str(source.get(core_key))
    expected_typed_sha = "sha256:" + str(
      source.get("typed_attention_view_sha256")
    )
    matches = []
    for entry in entries:
      if not isinstance(entry, dict):
        continue
      ares_plan = entry.get("ares_plan")
      target_plan = entry.get("target_plan")
      if not isinstance(ares_plan, dict) or not isinstance(target_plan, dict):
        continue
      if (
        entry.get("source_sha256") == expected_source_sha
        and entry.get("target_sha256") == expected_target_sha
        and [
          key
          for key in SCHEDULER_LEAN_FIXTURE_CORE_AUTHORITY_KEYS
          if key in ares_plan
        ]
        == [core_key]
        and ares_plan.get(core_key) == expected_core_sha
        and ares_plan.get("derived_typed_attention_view_sha256")
        == expected_typed_sha
        and target_plan.get("model_id") == binding["model_id"]
        and target_plan.get("backend_id") == binding["backend_id"]
        and target_plan.get("source_ares_plan_sha256") == expected_source_sha
      ):
        matches.append(entry)
    if len(matches) != 1:
      errors.append(
        "scheduler lineage Lean fixture authority has "
        f"{len(matches)} exact provider-manifest matches for "
        f"{binding['model_id']!r}/{binding['backend_id']!r}"
      )
      continue
    entry = matches[0]
    for label, path_field, expected_sha in (
      ("AresPlan", "source_path", str(binding["ares_plan_sha256"])),
      ("TargetPlan", "target_path", str(binding["target_plan_sha256"])),
    ):
      relative_path = entry.get(path_field)
      if not isinstance(relative_path, str) or not relative_path:
        errors.append(
          f"provider admission manifest {label} path is missing for "
          f"{binding['model_id']!r}/{binding['backend_id']!r}"
        )
        continue
      artifact_path = authority_root / relative_path
      try:
        artifact_data = _read_regular_file_bytes(
          artifact_path,
          maximum_bytes=MAX_TRACE_REPORT_BYTES,
          label=f"provider admission {label}",
        )
      except ValueError as exc:
        errors.append(str(exc))
        continue
      if hashlib.sha256(artifact_data).hexdigest() != expected_sha:
        errors.append(
          f"provider admission {label} bytes do not match scheduler lineage"
        )


def _validate_scheduler_packet_lineage_artifact_binding(
  errors: list[str],
  report: Mapping[str, Any],
  report_rows: list[dict[str, Any]],
  report_path: Path | None,
  authority_root: Path | None,
) -> None:
  bound_rows = [row for row in report_rows if row.get("row_kind")]
  sections = report.get("sections")
  summary_rows = (
    sections.get("introspection_artifacts") if isinstance(sections, Mapping) else None
  )
  scheduler_summary_present = isinstance(summary_rows, list) and any(
    isinstance(summary, dict)
    and summary.get("kind") == "scheduler_packet_lineage"
    for summary in summary_rows
  )
  if not bound_rows and not scheduler_summary_present:
    return
  if report_path is None:
    if bound_rows:
      errors.append("scheduler lineage rows cannot be rebound without report path")
    return
  inputs = report.get("inputs")
  metadata_value = inputs.get("metadata") if isinstance(inputs, Mapping) else None
  if not isinstance(metadata_value, str) or not metadata_value:
    errors.append("trace report inputs.metadata is required for scheduler lineage")
    return
  metadata_path = _resolve_report_reference(metadata_value, report_path)
  try:
    metadata_data = _read_regular_file_bytes(
      metadata_path,
      maximum_bytes=MAX_TRACE_METADATA_BYTES,
      label="trace report metadata",
    )
    metadata = _strict_json_loads_bytes(
      metadata_data,
      label="trace report metadata",
      maximum_items=MAX_TRACE_METADATA_JSON_ITEMS,
    )
  except ValueError as exc:
    errors.append(str(exc))
    return
  if not isinstance(metadata, dict):
    errors.append("trace report metadata is not an object")
    return
  metadata_artifacts = metadata.get("introspection_artifacts")
  if not isinstance(metadata_artifacts, list):
    errors.append("trace report metadata introspection_artifacts is not an array")
    return
  scheduler_artifact_indices = [
    index
    for index, artifact in enumerate(metadata_artifacts)
    if isinstance(artifact, dict)
    and artifact.get("kind") == "scheduler_packet_lineage"
  ]
  if not scheduler_artifact_indices:
    if bound_rows:
      errors.append("trace report has scheduler lineage rows without an artifact")
    return
  if not isinstance(summary_rows, list):
    errors.append(
      "trace report sections.introspection_artifacts is required for scheduler lineage"
    )
    return
  rows_by_artifact: dict[int, list[dict[str, Any]]] = {}
  seen_coordinates: set[tuple[int, int]] = set()
  for row in bound_rows:
    artifact_index = row.get("artifact_index")
    row_index = row.get("row_index")
    if not _scheduler_nonnegative_u64(artifact_index) or not _scheduler_nonnegative_u64(
      row_index
    ):
      continue
    coordinate = (artifact_index, row_index)
    if coordinate in seen_coordinates:
      errors.append(
        "trace report scheduler lineage duplicates artifact/row coordinate "
        f"{coordinate}"
      )
      continue
    seen_coordinates.add(coordinate)
    rows_by_artifact.setdefault(artifact_index, []).append(row)
  for artifact_index in sorted(rows_by_artifact):
    if artifact_index not in scheduler_artifact_indices:
      errors.append(
        f"scheduler lineage artifact_index {artifact_index} is not a scheduler artifact"
      )
  for artifact_index in scheduler_artifact_indices:
    rows = rows_by_artifact.get(artifact_index, [])
    artifact = metadata_artifacts[artifact_index]
    assert isinstance(artifact, dict)
    matching_summaries = [
      summary
      for summary in summary_rows
      if isinstance(summary, dict)
      and summary.get("index") == artifact_index
      and summary.get("kind") == "scheduler_packet_lineage"
    ]
    if len(matching_summaries) != 1:
      errors.append(
        f"scheduler lineage artifact {artifact_index} has {len(matching_summaries)} "
        "matching report metadata rows"
      )
      continue
    summary = matching_summaries[0]
    raw_path_value = artifact.get("path")
    expected_sha = artifact.get("sha256")
    expected_rows = artifact.get("row_count")
    expected_bytes = artifact.get("byte_count")
    if not isinstance(raw_path_value, str) or not raw_path_value:
      errors.append(f"scheduler lineage artifact {artifact_index} path is invalid")
      continue
    if (
      not isinstance(expected_sha, str)
      or expected_sha != expected_sha.lower()
      or SHA256_RE.fullmatch(expected_sha) is None
    ):
      errors.append(f"scheduler lineage artifact {artifact_index} sha256 is invalid")
      continue
    if not _scheduler_nonnegative_u64(expected_rows) or not _scheduler_nonnegative_u64(
      expected_bytes
    ):
      errors.append(
        f"scheduler lineage artifact {artifact_index} row/byte counts are invalid"
      )
      continue
    raw_path = _resolve_report_reference(raw_path_value, metadata_path)
    try:
      raw_data = _read_regular_file_bytes(
        raw_path,
        maximum_bytes=MAX_SCHEDULER_LINEAGE_JSONL_BYTES,
        label="scheduler packet-lineage JSONL",
      )
      raw_rows = _scheduler_jsonl_rows(raw_data)
    except ValueError as exc:
      errors.append(str(exc))
      continue
    actual_sha = hashlib.sha256(raw_data).hexdigest()
    projected_indices = [row["row_index"] for row in rows]
    expected_indices = list(range(len(raw_rows)))
    if projected_indices != expected_indices:
      errors.append(
        f"scheduler lineage artifact {artifact_index} report projection is not "
        "complete and contiguous"
      )
    _validate_scheduler_lean_authority(errors, raw_rows, authority_root)
    binding_checks = {
      "metadata sha256": (expected_sha, actual_sha),
      "metadata row_count": (expected_rows, len(raw_rows)),
      "metadata byte_count": (expected_bytes, len(raw_data)),
      "report path": (summary.get("path"), raw_path_value),
      "report sha256": (summary.get("sha256"), actual_sha),
      "report row_count": (summary.get("row_count"), str(len(raw_rows))),
      "report byte_count": (summary.get("byte_count"), str(len(raw_data))),
    }
    for label, (reported, expected) in binding_checks.items():
      if reported != expected:
        errors.append(
          f"scheduler lineage artifact {artifact_index} {label} does not match raw sidecar"
        )
    for report_row in rows:
      row_index = report_row["row_index"]
      if row_index >= len(raw_rows):
        errors.append(
          f"scheduler lineage artifact {artifact_index} row_index {row_index} "
          "is outside the raw sidecar"
        )
        continue
      try:
        expected_row = _scheduler_project_raw_row(
          raw_rows[row_index], artifact_index, row_index
        )
      except ValueError as exc:
        errors.append(
          f"scheduler lineage raw row {artifact_index}:{row_index}: {exc}"
        )
        continue
      if report_row != expected_row:
        differing = sorted(
          field
          for field in set(report_row) | set(expected_row)
          if report_row.get(field) != expected_row.get(field)
        )
        errors.append(
          f"scheduler lineage report row {artifact_index}:{row_index} does not "
          f"match the raw projection at fields {differing!r}"
        )


def _trace_report_section_rows(
    errors: list[str],
    sections: Mapping[str, Any],
    name: str,
    *,
    required: bool = True,
) -> list[dict[str, Any]]:
    rows = sections.get(name)
    if rows is None and not required:
        return []
    if not isinstance(rows, list):
        errors.append(f"trace report sections.{name} must be a list")
        return []
    typed_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if isinstance(row, dict):
            typed_rows.append(row)
        else:
            errors.append(f"trace report sections.{name}[{index}] must be an object")
    return typed_rows


def _validate_trace_report_triage_rows(
    errors: list[str],
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        errors.append(
            "trace report sections.report_triage must contain at least one row"
        )
        return
    for index, row in enumerate(rows):
        context = f"trace report sections.report_triage[{index}]"
        _require_fields(errors, row, TRACE_REPORT_TRIAGE_REQUIRED_FIELDS, context)
        for field in TRACE_REPORT_TRIAGE_REQUIRED_FIELDS:
            if field in row and not isinstance(row.get(field), str):
                errors.append(f"{context}.{field} must be a string")

        triage_status = row.get("triage_status")
        if isinstance(triage_status, str) and (
            triage_status not in TRACE_REPORT_TRIAGE_STATUSES
        ):
            errors.append(
                f"{context}.triage_status must be one of: "
                + ", ".join(sorted(TRACE_REPORT_TRIAGE_STATUSES))
            )

        report_grade = row.get("report_grade")
        if isinstance(report_grade, str) and report_grade not in TRACE_REPORT_GRADES:
            errors.append(
                f"{context}.report_grade must be one of: "
                + ", ".join(sorted(TRACE_REPORT_GRADES))
            )

        proof_grade_status = row.get("proof_grade_status")
        if (
            isinstance(proof_grade_status, str)
            and proof_grade_status != "not_established_by_report"
        ):
            errors.append(
                f"{context}.proof_grade_status must be not_established_by_report"
            )

        first_useful_section = row.get("first_useful_section")
        if isinstance(first_useful_section, str):
            if re.fullmatch(r"sections\.[a-z0-9_]+", first_useful_section) is None:
                errors.append(
                    f"{context}.first_useful_section must match sections.<name>"
                )

        first_action = row.get("first_action")
        if isinstance(first_action, str) and not first_action.strip():
            errors.append(f"{context}.first_action must be a non-empty string")

        claim_boundary = row.get("claim_boundary")
        if (
            isinstance(claim_boundary, str)
            and claim_boundary != "diagnostic_routing_not_evidence"
        ):
            errors.append(
                f"{context}.claim_boundary must be diagnostic_routing_not_evidence"
            )


def _trace_report_string_rows(
    errors: list[str],
    sections: Mapping[str, Any],
    name: str,
    *,
    required: bool = True,
) -> list[str]:
    rows = sections.get(name)
    if rows is None and not required:
        return []
    if not isinstance(rows, list):
        errors.append(f"trace report sections.{name} must be a list")
        return []
    typed_rows: list[str] = []
    for index, row in enumerate(rows):
        if isinstance(row, str):
            typed_rows.append(row)
        else:
            errors.append(f"trace report sections.{name}[{index}] must be a string")
    return typed_rows


def _trace_report_value_counts(
    rows: list[dict[str, Any]],
    key: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(key)
        if isinstance(value, str) and value:
            counts[value] = counts.get(value, 0) + 1
        elif isinstance(value, (int, float, bool)):
            label = str(value)
            counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _trace_report_comma_value_counts(
    rows: list[dict[str, Any]],
    key: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str):
            continue
        for item in value.split(","):
            item = item.strip()
            if item:
                counts[item] = counts.get(item, 0) + 1
    return dict(sorted(counts.items()))


def _trace_report_int(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _trace_report_optional_count(errors: list[str], value: Any, label: str) -> int:
    if value is None or value == "":
        return 0
    if isinstance(value, bool):
        errors.append(f"{label} must be a non-negative integer")
        return 0
    if isinstance(value, int):
        if value < 0:
            errors.append(f"{label} must be a non-negative integer")
            return 0
        return value
    if isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError:
            errors.append(f"{label} must be a non-negative integer")
            return 0
        if parsed < 0:
            errors.append(f"{label} must be a non-negative integer")
            return 0
        return parsed
    errors.append(f"{label} must be a non-negative integer")
    return 0


def _trace_report_samples(
    rows: list[dict[str, Any]],
    keys: tuple[str, ...],
    *,
    limit: int = 3,
) -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []
    for row in rows[:limit]:
        sample: dict[str, str] = {}
        for key in keys:
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                sample[key] = value.strip()
            elif isinstance(value, (int, float, bool)):
                sample[key] = str(value)
        if sample:
            samples.append(sample)
    return samples


def _trace_report_string_samples(rows: list[str], *, limit: int = 3) -> list[str]:
    return [row.strip() for row in rows[:limit] if row.strip()]


def _trace_report_string_prefix_counts(rows: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        text = row.strip()
        if not text:
            continue
        prefix, separator, _ = text.partition(":")
        label = prefix.strip() if separator and prefix.strip() else "other"
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _trace_report_provider_payload_lanes(rows: list[dict[str, Any]]) -> list[str]:
    lanes: set[str] = set()
    for row in rows:
        provider_id = row.get("provider_id")
        payload_lane = row.get("payload_lane")
        if (
            isinstance(provider_id, str)
            and provider_id.strip()
            and isinstance(payload_lane, str)
            and payload_lane.strip()
        ):
            lanes.add(f"{provider_id.strip()}/{payload_lane.strip()}")
    return sorted(lanes)


def _read_json_or_jsonl(path: Path) -> Any:
  text = path.read_text()
  if path.suffix == ".jsonl":
    rows = []
    for lineno, line in enumerate(text.splitlines(), start=1):
      if not line.strip():
        continue
      try:
        rows.append(json.loads(line))
      except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSONL at line {lineno}: {exc}") from exc
    return rows
  try:
    return json.loads(text)
  except json.JSONDecodeError as exc:
    raise ValueError(f"invalid JSON: {exc}") from exc


def _payload_root(
  payload: Any,
  errors: list[str],
  label: str,
) -> dict[str, Any] | list[Any] | None:
  if isinstance(payload, dict | list):
    return payload
  errors.append(f"{label} must be a JSON object or JSONL event list")
  return None


def _payload_rows(payload: Any) -> list[dict[str, Any]]:
  if isinstance(payload, list):
    return [item for item in payload if isinstance(item, dict)]
  if isinstance(payload, dict) and isinstance(payload.get("events"), list):
    return [item for item in payload["events"] if isinstance(item, dict)]
  return []


def _first_string(
  payload: dict[str, Any] | list[Any],
  rows: list[dict[str, Any]],
  keys: tuple[str, ...],
) -> str | None:
  if isinstance(payload, dict):
    for key in keys:
      value = payload.get(key)
      if isinstance(value, str) and value.strip():
        return value.strip()
  for row in rows:
    for key in keys:
      value = row.get(key)
      if isinstance(value, str) and value.strip():
        return value.strip()
  return None


def _truthy_field(
  payload: dict[str, Any] | list[Any],
  rows: list[dict[str, Any]],
  keys: tuple[str, ...],
) -> bool:
  if isinstance(payload, dict):
    for key in keys:
      if payload.get(key) is True:
        return True
  return any(row.get(key) is True for row in rows for key in keys)


def _first_bool(
  payload: dict[str, Any] | list[Any],
  rows: list[dict[str, Any]],
  keys: tuple[str, ...],
) -> bool | None:
  if isinstance(payload, dict):
    for key in keys:
      value = payload.get(key)
      if isinstance(value, bool):
        return value
  for row in rows:
    for key in keys:
      value = row.get(key)
      if isinstance(value, bool):
        return value
  return None


def _event_name(row: dict[str, Any]) -> str | None:
  for key in ("event", "event_name", "name", "phase"):
    value = row.get(key)
    if isinstance(value, str) and value.strip():
      return value.strip()
  return None


def _require_sha256(errors: list[str], value: Any, label: str) -> None:
  if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
    errors.append(f"{label} SHA-256 must be a 64-character hex string")


def _require_introspection_sha256(
    errors: list[str],
    value: Any,
    label: str,
) -> None:
    if _normalize_introspection_sha256(value) is None:
        errors.append(f"{label} must be a SHA-256 hex string or sha256:<hex> digest")


def _normalize_introspection_sha256(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    digest = value[7:] if value.startswith("sha256:") else value
    if SHA256_RE.fullmatch(digest) is None:
        return None
    return digest.lower()


def _require_git_sha(errors: list[str], value: Any, label: str) -> None:
  if not isinstance(value, str) or GIT_COMMIT_RE.fullmatch(value) is None:
    errors.append(f"{label} must be a 40-character hex git commit")


def _validate_tvd(
  errors: list[str],
  payload: dict[str, Any],
  label: str,
) -> tuple[float | None, float | None]:
  tvd = payload.get("tvd", payload.get("max_tvd"))
  threshold = payload.get("tvd_threshold", 0.01)
  if not isinstance(tvd, int | float) or tvd < 0:
    errors.append(f"{label} tvd/max_tvd must be a non-negative number")
    tvd_value = None
  else:
    tvd_value = float(tvd)
  if not isinstance(threshold, int | float) or threshold <= 0:
    errors.append(f"{label} tvd_threshold must be a positive number")
    threshold_value = None
  else:
    threshold_value = float(threshold)
  return tvd_value, threshold_value


def _validate_replay_context(
  errors: list[str],
  value: Any,
  label: str,
) -> None:
  if not isinstance(value, dict):
    errors.append(f"{label} replay_context must be an object")
    return
  missing = [field for field in REPLAY_CONTEXT_FIELDS if field not in value]
  if missing:
    errors.append(
      f"{label} replay_context missing required field(s): {', '.join(missing)}"
    )
  if not isinstance(value.get("context_tokens"), list):
    errors.append(f"{label} replay_context.context_tokens must be a list")
  for field in (
    "context_count",
    "new_count",
    "runtime_request_token_count",
    "context_prefix_token_count",
  ):
    if not isinstance(value.get(field), int):
      errors.append(f"{label} replay_context.{field} must be an integer")


def _normalize_workload(value: Any) -> str | None:
  if not isinstance(value, str) or not value.strip():
    return None
  return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _require_fields(
  errors: list[str],
  payload: dict[str, Any],
  fields: Iterable[str],
  context: str,
) -> None:
  missing = [field for field in fields if field not in payload]
  if missing:
    errors.append(f"{context} missing required field(s): {', '.join(missing)}")


def _expect_object(
  errors: list[str],
  value: Any,
  context: str,
) -> dict[str, Any] | None:
  if isinstance(value, dict):
    return value
  errors.append(f"{context} must be an object")
  return None


def _expect_string_list(
  errors: list[str],
  value: Any,
  context: str,
) -> list[str] | None:
  if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
    errors.append(f"{context} must be a list of strings")
    return None
  return value


def _require_non_empty_string(errors: list[str], value: Any, context: str) -> None:
  if not isinstance(value, str) or not value.strip():
    errors.append(f"{context} must be a non-empty string")


def _command_runs_uv_mmlu_pro(command: Any) -> bool:
  if not isinstance(command, str):
    return False
  try:
    tokens = shlex.split(command)
  except ValueError:
    tokens = command.split()
  options_with_values = {
    "--cache-dir",
    "--config-file",
    "--config-setting",
    "--directory",
    "--env-file",
    "--exclude-newer",
    "--extra",
    "--find-links",
    "--from",
    "--index",
    "--index-url",
    "--keyring-provider",
    "--link-mode",
    "--no-binary",
    "--no-build",
    "--no-build-isolation-package",
    "--only-binary",
    "--project",
    "--python",
    "--python-platform",
    "--refresh-package",
    "--reinstall-package",
    "--resolution",
    "--upgrade-package",
    "--with",
    "--with-editable",
    "--with-requirements",
  }
  for index, token in enumerate(tokens[:-1]):
    if token == "uv" and tokens[index + 1] == "run":
      arg_index = index + 2
      while arg_index < len(tokens):
        arg = tokens[arg_index]
        if arg == "--":
          arg_index += 1
          break
        if not arg.startswith("-"):
          break
        if "=" in arg:
          arg_index += 1
          continue
        arg_index += 1
        if arg in options_with_values and arg_index < len(tokens):
          arg_index += 1
      return arg_index < len(tokens) and tokens[arg_index] == "mmlu_pro"
  return False


def _command_env_non_negative_int(
  errors: list[str],
  command: Any,
  name: str,
) -> int | None:
  if not isinstance(command, str):
    return None
  try:
    tokens = shlex.split(command)
  except ValueError:
    tokens = command.split()
  prefix = f"{name}="
  values: list[str] = []
  for token in tokens:
    if not token.startswith(prefix):
      continue
    values.append(token[len(prefix) :])
  if not values:
    return None
  if len(values) > 1:
    errors.append(f"systems_test.command {name} must not be repeated")
    return None
  raw = values[0]
  if raw == "":
    return None
  try:
    value = int(raw)
  except ValueError:
    errors.append(f"systems_test.command {name} must be an integer")
    return None
  if value < 0:
    errors.append(f"systems_test.command {name} must be non-negative")
    return None
  return value


def _validate_revision_metadata(
  errors: list[str],
  metadata: dict[str, Any],
  label: str,
  fields: tuple[str, ...],
) -> None:
  _require_fields(errors, metadata, fields, label)
  revision = metadata.get("requested_revision")
  _require_non_empty_string(errors, revision, f"{label}.requested_revision")
  if isinstance(revision, str) and is_floating_revision(revision):
    errors.append(f"{label}.requested_revision must not be floating")


def _expected_model_ids(spec: Mapping[str, Any]) -> set[str]:
  expected: set[str] = set()
  for field in (
    "expected_model_ids",
    "artifact_model_ids",
    "model_aliases",
    "model_id",
    "model",
  ):
    expected.update(_string_values(spec.get(field)))
  return expected


def _oracle_model_ids(payload: Any) -> set[str]:
  records = payload if isinstance(payload, list) else [payload]
  model_ids: set[str] = set()
  for record in records:
    if not isinstance(record, dict):
      continue
    model = record.get("model")
    if not isinstance(model, dict):
      continue
    model_ids.update(_string_values(model.get("model_id")))
  return model_ids


def _gate_detail_string(gate: Any, field: str) -> str | None:
  if not isinstance(gate, dict):
    return None
  detail = gate.get("detail")
  if not isinstance(detail, dict):
    return None
  value = detail.get(field)
  return value if isinstance(value, str) and value else None


def _string_values(value: Any) -> set[str]:
  if isinstance(value, str) and value:
    return {value}
  if isinstance(value, list):
    return {item for item in value if isinstance(item, str) and item}
  return set()


def _validate_run(errors: list[str], run: dict[str, Any]) -> None:
  _require_fields(
    errors,
    run,
    (
      "seed",
      "decode_strategy",
      "max_new_tokens",
      "top_k",
      "torch_deterministic_algorithms",
      "local_files_only",
      "trust_remote_code",
    ),
    "run",
  )
  if run.get("decode_strategy") != "greedy":
    errors.append("run.decode_strategy must be greedy")
  if not isinstance(run.get("max_new_tokens"), int) or run.get("max_new_tokens") < 0:
    errors.append("run.max_new_tokens must be a non-negative integer")
  if not isinstance(run.get("top_k"), int) or run.get("top_k") < 1:
    errors.append("run.top_k must be a positive integer")
  for field in (
    "torch_deterministic_algorithms",
    "local_files_only",
    "trust_remote_code",
  ):
    if not isinstance(run.get(field), bool):
      errors.append(f"run.{field} must be a boolean")


def _validate_prompt(errors: list[str], prompt: dict[str, Any]) -> None:
  _require_fields(
    errors,
    prompt,
    ("kind", "text", "token_ids", "token_count", "add_special_tokens"),
    "prompt",
  )
  if prompt.get("kind") not in {"raw", "chat"}:
    errors.append("prompt.kind must be raw or chat")
  token_ids = prompt.get("token_ids")
  if not isinstance(token_ids, list) or not all(
    isinstance(token_id, int) for token_id in token_ids
  ):
    errors.append("prompt.token_ids must be a list of integers")
    return
  if prompt.get("token_count") != len(token_ids):
    errors.append("prompt.token_count must match prompt.token_ids length")
  if not isinstance(prompt.get("add_special_tokens"), bool):
    errors.append("prompt.add_special_tokens must be a boolean")


def _validate_generation(errors: list[str], generation: dict[str, Any]) -> None:
  _require_fields(
    errors,
    generation,
    (
      "generated_token_ids",
      "generated_token_count",
      "generated_text",
      "finish_reason",
      "eos_token_id",
      "eos_token_ids",
      "stop_token_id",
    ),
    "generation",
  )
  generated_ids = generation.get("generated_token_ids")
  if not isinstance(generated_ids, list) or not all(
    isinstance(token_id, int) for token_id in generated_ids
  ):
    errors.append("generation.generated_token_ids must be a list of integers")
    return
  if generation.get("generated_token_count") != len(generated_ids):
    errors.append(
      "generation.generated_token_count must match generated_token_ids length"
    )
  if not isinstance(generation.get("generated_text"), str):
    errors.append("generation.generated_text must be a string")
  if generation.get("finish_reason") not in {"max_new_tokens", "eos_token"}:
    errors.append("generation.finish_reason must be max_new_tokens or eos_token")
  eos_ids = generation.get("eos_token_ids")
  if not isinstance(eos_ids, list) or not all(
    isinstance(token_id, int) for token_id in eos_ids
  ):
    errors.append("generation.eos_token_ids must be a list of integers")
  stop_token_id = generation.get("stop_token_id")
  if generation.get("finish_reason") == "eos_token":
    if not isinstance(stop_token_id, int):
      errors.append("generation.stop_token_id must be an integer on eos stop")
    elif isinstance(eos_ids, list) and stop_token_id not in eos_ids:
      errors.append("generation.stop_token_id must appear in eos_token_ids")
    elif generated_ids and generated_ids[-1] != stop_token_id:
      errors.append("generation.stop_token_id must match final generated token")
  elif stop_token_id is not None:
    errors.append("generation.stop_token_id must be null unless finish is eos_token")


def _validate_logit_slices(errors: list[str], value: Any) -> list[int] | None:
  if not isinstance(value, list):
    errors.append("logit_slices must be a list")
    return None
  selected: list[int] = []
  for expected_step, entry in enumerate(value):
    if not isinstance(entry, dict):
      errors.append("logit_slices entries must be objects")
      return None
    if entry.get("step") != expected_step:
      errors.append("logit_slices step values must be contiguous from zero")
    position = entry.get("position")
    context_token_count = entry.get("context_token_count")
    if not isinstance(position, int) or position < 0:
      errors.append("logit_slices[].position must be a non-negative integer")
    if not isinstance(context_token_count, int) or context_token_count < 0:
      errors.append("logit_slices[].context_token_count must be a non-negative integer")
    elif isinstance(position, int) and context_token_count != position + 1:
      errors.append("logit_slices[].context_token_count must equal position + 1")
    token_id = entry.get("selected_token_id")
    if not isinstance(token_id, int):
      errors.append("logit_slices[].selected_token_id must be an integer")
    else:
      selected.append(token_id)
    if not isinstance(entry.get("selected_token_text"), str):
      errors.append("logit_slices[].selected_token_text must be a string")
    selected_logit = entry.get("selected_token_logit")
    if not isinstance(selected_logit, int | float):
      errors.append("logit_slices[].selected_token_logit must be numeric")
    _validate_top_k(errors, entry.get("top_k"), entry)
  return selected


def _validate_top_k(
  errors: list[str],
  value: Any,
  logit_slice: dict[str, Any],
) -> None:
  if not isinstance(value, list) or not value:
    errors.append("logit_slices[].top_k must be a non-empty list")
    return
  previous_rank = 0
  for item in value:
    if not isinstance(item, dict):
      errors.append("top_k entries must be objects")
      return
    rank = item.get("rank")
    if not isinstance(rank, int) or rank != previous_rank + 1:
      errors.append("top_k ranks must be contiguous from one")
    if not isinstance(item.get("token_id"), int):
      errors.append("top_k token_id must be an integer")
    if not isinstance(item.get("token_text"), str):
      errors.append("top_k token_text must be a string")
    if not isinstance(item.get("logit"), int | float):
      errors.append("top_k logit must be numeric")
    previous_rank = rank if isinstance(rank, int) else previous_rank
  first = value[0]
  if isinstance(first, dict):
    if logit_slice.get("selected_token_logit") != first.get("logit"):
      errors.append("selected_token_logit must match top top_k logit")
    if logit_slice.get("selected_token_id") != first.get("token_id"):
      errors.append("selected_token_id must match top top_k token_id")


def _validate_target_operation(
  errors: list[str],
  operation: Any,
  index: int,
  runtime_binding_names: set[str],
) -> int:
  context = f"TargetPlan.operations[{index}]"
  if not isinstance(operation, dict):
    errors.append(f"{context} must be an object")
    return 0
  _require_fields(
    errors,
    operation,
    ("id", "role", "action", "source", "requirements"),
    context,
  )
  _require_non_empty_string(errors, operation.get("id"), f"{context}.id")
  role = operation.get("role")
  action = operation.get("action")
  if role not in {"semantic", "runtime_binding", "auxiliary"}:
    errors.append(f"{context}.role is invalid")
  if action not in {
    "kernel",
    "matmul",
    "attention",
    "kv_cache",
    "top_k",
    "moe_routing",
    "transfer",
    "barrier",
    "queue",
    "intrinsic",
    "runtime_binding",
  }:
    errors.append(f"{context}.action is invalid")
  source = _expect_object(errors, operation.get("source"), f"{context}.source")
  _expect_object(errors, operation.get("requirements"), f"{context}.requirements")
  if role == "runtime_binding" and source is not None:
    name = source.get("name")
    if isinstance(name, str) and name:
      runtime_binding_names.add(name)
    else:
      errors.append(f"{context}.source.name must name the runtime binding")
  if role == "semantic" and source is not None:
    if source.get("type") != "ares_plan_statement":
      errors.append(f"{context}.source.type must be ares_plan_statement")
    if not isinstance(source.get("statement_index"), int):
      errors.append(f"{context}.source.statement_index must be an integer")
    return 1
  return 0

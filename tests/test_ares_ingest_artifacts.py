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
    mmlu_pro_gate,
    one_token_logits_gate,
    trace_report_gate,
    token_agreement_gate,
    validate_cpp_tvd_evidence,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write_json_artifact(path: Path, payload: dict) -> dict[str, str]:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return {"path": path.name, "sha256": sha256_file(path)}


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
            f"OPENAI_HOST={openai_host} SKIP_PROVISION=1 "
            f"MMLU_MODEL={model} uv run mmlu_pro"
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
                                        "sections."
                                        "provider_payload_boundary_inventory_rows"
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
                                        "external_oracle_reference_anchor_not_"
                                        "sut_oracle_evidence"
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
                                        "system_under_test_scheduler_kv_payload_"
                                        "diagnostic"
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
                                    "introspection_level": "payload",
                                    "compile_feature_trace_introspection": True,
                                    "deep_introspection_effective": True,
                                    "next_action": (
                                        "inspect_matching_introspection_report_sections"
                                    ),
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
                                    "boundary_status": (
                                        "available_from_scheduler_protocol_boundary"
                                    ),
                                    "claim_boundary": (
                                        "system_under_test_scheduler_kv_payload_"
                                        "diagnostic"
                                    ),
                                    "next_action": "inspect_report_section",
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
                                        "system_under_test_numeric_"
                                        "localization_diagnostic"
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
                                        "external_oracle_reference_present; "
                                        "row_remains_system_under_test"
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
                                    "artifact_count": 0,
                                    "section_status": "capability_without_artifact",
                                    "claim_boundary": (
                                        "planning_decision_diagnostic_not_model_evidence"
                                    ),
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
                                        "token_quality_summary_rows,"
                                        "oracle_reference_summary_rows"
                                    ),
                                    "claim_boundaries": (
                                        "system_under_test_diagnostic_not_oracle"
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
            self.assertEqual(gate["detail"]["preflight_status"], "pass")
            self.assertEqual(gate["detail"]["unsupported_claim_count"], 1)
            self.assertEqual(gate["detail"]["next_measurement_count"], 1)
            self.assertEqual(
                gate["detail"]["introspection_capability_status_counts"],
                {"recorded": 1},
            )
            self.assertEqual(
                gate["detail"]["introspection_artifact_summary_status_counts"],
                {"recorded_and_locally_present": 1},
            )
            self.assertEqual(
                gate["detail"]["introspection_section_inventory_status_counts"],
                {"available": 1, "capability_without_artifact": 1},
            )
            self.assertEqual(
                gate["detail"]["introspection_section_inventory_capability_counts"],
                {"deep_introspection": 1, "token_quality": 1},
            )
            self.assertEqual(
                gate["detail"]["trace_config_status_counts"],
                {"requested_and_recorded": 1},
            )
            self.assertEqual(
                gate["detail"]["provider_payload_boundary_status_counts"],
                {"recorded_artifact": 1},
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
            self.assertEqual(gate["detail"]["report_json_section_count"], 10)
            self.assertEqual(
                gate["detail"]["report_json_section_kind_counts"],
                {
                    "capture_configuration": 1,
                    "debug_payload_diagnostic": 1,
                    "introspection": 1,
                    "introspection_inventory": 2,
                    "measurement_guidance": 1,
                    "sidecar": 4,
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
                "capability_without_artifact",
            )
            self.assertEqual(len(gate["detail"]["sha256"]), 64)

    def test_trace_report_gate_accepts_real_ares_trace_report_fixture(self) -> None:
        path = FIXTURE_DIR / "ares_trace_report_introspection_real.json"

        gate = trace_report_gate(path)

        self.assertTrue(gate["passed"], gate.get("errors"))
        self.assertEqual(gate["artifact_validator"], "trace_report")
        self.assertEqual(gate["detail"]["report_grade"], "diagnostic")
        self.assertEqual(gate["detail"]["preflight_status"], "pass")
        self.assertEqual(gate["detail"]["report_json_section_count"], 51)
        self.assertEqual(
            gate["detail"]["trace_config_status_counts"],
            {"requested_and_recorded": 1},
        )
        self.assertEqual(
            gate["detail"]["provider_payload_boundary_status_counts"],
            {
                "blocked_no_supported_boundary": 1,
                "capability_without_matching_provider_artifact": 5,
                "recorded_artifact": 5,
            },
        )
        self.assertEqual(
            gate["detail"]["introspection_capability_status_counts"],
            {
                "capability_without_artifact": 1,
                "compiled": 1,
                "recorded": 10,
            },
        )
        self.assertEqual(
            gate["detail"]["introspection_artifact_summary_status_counts"],
            {"recorded_and_locally_present": 8},
        )
        self.assertEqual(
            gate["detail"]["introspection_section_inventory_status_counts"],
            {"available": 14, "capability_without_artifact": 1},
        )
        self.assertEqual(
            gate["detail"]["introspection_section_inventory_capability_counts"],
            {
                "activation_digests": 1,
                "attention_page_trace": 1,
                "debug_payloads": 1,
                "deep_introspection": 1,
                "device_dma_lifecycle": 1,
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
        section_paths = {
            sample["json_path"]
            for sample in gate["detail"]["report_json_section_samples"]
        }
        self.assertIn("sections.trace_config_rows", section_paths)
        self.assertIn("sections.debug_payload_artifact_summary_rows", section_paths)
        self.assertIn("sections.token_quality_summary_rows", section_paths)
        self.assertIn("sections.oracle_reference_summary_rows", section_paths)
        self.assertIn("sections.tensor_payload_sidecar_rows", section_paths)
        self.assertIn("sections.kv_payload_digest_sidecar_rows", section_paths)
        self.assertIn("sections.introspection_capability_rows", section_paths)
        self.assertIn("sections.introspection_artifact_summary_rows", section_paths)
        self.assertIn("sections.introspection_section_inventory", section_paths)
        self.assertIn(
            "provider_payload_boundary_inventory_rows",
            gate["detail"]["section_names"],
        )
        self.assertEqual(
            gate["detail"]["trace_config_samples"][0]["requested_sidecar_controls"],
            "topk,logit_slices,tensor_payloads,activation_digests,kv_payload_digests",
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
        self.assertIn("device_dma_lifecycle_sidecar_rows", introspection_sections)
        self.assertIn("attention_page_trace_sidecar_rows", introspection_sections)
        self.assertIn("kv_payload_digest_sidecar_rows", introspection_sections)

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

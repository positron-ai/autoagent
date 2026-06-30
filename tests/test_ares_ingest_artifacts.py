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
                gate["detail"]["trace_config_status_counts"],
                {"requested_and_recorded": 1},
            )
            self.assertEqual(gate["detail"]["report_json_section_count"], 3)
            self.assertEqual(
                gate["detail"]["report_json_section_kind_counts"],
                {
                    "capture_configuration": 1,
                    "introspection": 1,
                    "measurement_guidance": 1,
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
                gate["detail"]["introspection_capability_samples"][0][
                    "matching_artifact_count"
                ],
                "1",
            )
            self.assertEqual(len(gate["detail"]["sha256"]), 64)

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

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ares_ingest_autoagent.artifacts import (
    artifact_consistency_gate,
    backend_open_gate,
    build_greedy_token_evidence,
    cpp_tvd_gate,
    depth_performance_gate,
    one_token_logits_gate,
    token_agreement_gate,
    validate_cpp_tvd_evidence,
)


SHA_A = "a" * 64
SHA_B = "b" * 64


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
            reference.write_text(json.dumps(list(range(8))))
            candidate.write_text(json.dumps(list(range(8))))
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
            reference.write_text(json.dumps([1, 2, 3]))
            candidate.write_text(json.dumps([1, 2, 3]))
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
                    )
                )
            )

            gate = token_agreement_gate(path)

            self.assertFalse(gate["passed"])
            self.assertIn("expected at least 8", " ".join(gate["errors"]))

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


if __name__ == "__main__":
    unittest.main()

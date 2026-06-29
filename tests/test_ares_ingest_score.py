from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ares_ingest_autoagent.artifacts import build_greedy_token_evidence
from ares_ingest_autoagent.score import (
    BACKEND_GATES,
    COMPARISON_GATES,
    CPU_ONLY_GATES,
    FULL_GATES,
    compute_reward,
    main,
)


def artifact_gate(validator: str) -> dict:
    return {"passed": True, "score": 1.0, "artifact_validator": validator}


def token_evidence(length: int = 8, *, exact: bool = True) -> dict:
    return {
        "schema": "ares.runtime.greedy_token_agreement.v1",
        "evidence_class": "system_under_test",
        "oracle": "huggingface_transformers_pytorch_cpu",
        "candidate": "ares",
        "decode_strategy": "greedy",
        "expected_generated_tokens": 8,
        "generated_tokens": length,
        "reference_generated_token_ids": list(range(length)),
        "candidate_generated_token_ids": list(range(length))
        if exact
        else [99] * length,
        "score": 1.0 if exact else 0.0,
        "exact_match": exact,
        "exact_fraction": 1.0 if exact else 0.875,
        "top1_agreement": 1.0 if exact else 0.875,
        "reference": {"path": "reference.json", "sha256": "a" * 64},
        "candidate_output": {
            "path": "candidate.json",
            "sha256": "b" * 64,
            "runtime": "ares",
        },
        "cases": [
            {
                "name": "default",
                "exact_match": exact,
                "candidate_length": length,
            }
        ],
    }


def oracle_record(
    kind: str = "hf_cpu_oracle_capture",
    oracle: str = "huggingface_transformers_pytorch_cpu",
) -> dict:
    return {
        "schema": "ares.oracles.hf_cpu.record.v1",
        "record_kind": kind,
        "capture_id": "test-capture",
        "created_utc": "2026-06-26T00:00:00Z",
        "source": {
            "oracle": oracle,
            "capture_script": "tools/oracles/hf-cpu/capture_hf_cpu_oracle.py",
        },
        "model": {
            "model_id": "synthetic/model",
            "requested_revision": "0123456789abcdef0123456789abcdef01234567",
            "resolved_revision": "0123456789abcdef0123456789abcdef01234567",
            "dtype": "float32",
        },
        "tokenizer": {
            "tokenizer_id": "synthetic/model",
            "requested_revision": "0123456789abcdef0123456789abcdef01234567",
            "resolved_revision": "0123456789abcdef0123456789abcdef01234567",
        },
        "run": {
            "seed": 0,
            "decode_strategy": "greedy",
            "max_new_tokens": 2,
            "top_k": 2,
            "torch_deterministic_algorithms": True,
            "local_files_only": True,
            "trust_remote_code": False,
        },
        "prompt": {
            "kind": "raw",
            "text": "Hello",
            "token_ids": [1, 7],
            "token_count": 2,
            "add_special_tokens": True,
        },
        "generation": {
            "generated_token_ids": [3, 2],
            "generated_token_count": 2,
            "generated_text": " world</s>",
            "finish_reason": "eos_token",
            "eos_token_id": 2,
            "eos_token_ids": [2],
            "stop_token_id": 2,
        },
        "logit_slices": [
            {
                "step": 0,
                "position": 1,
                "context_token_count": 2,
                "selected_token_id": 3,
                "selected_token_text": " world",
                "selected_token_logit": 12.5,
                "top_k": [
                    {"rank": 1, "token_id": 3, "token_text": " world", "logit": 12.5},
                    {"rank": 2, "token_id": 4, "token_text": " there", "logit": 8.0},
                ],
            },
            {
                "step": 1,
                "position": 2,
                "context_token_count": 3,
                "selected_token_id": 2,
                "selected_token_text": "</s>",
                "selected_token_logit": 9.25,
                "top_k": [
                    {"rank": 1, "token_id": 2, "token_text": "</s>", "logit": 9.25},
                    {"rank": 2, "token_id": 5, "token_text": "!", "logit": 4.0},
                ],
            },
        ],
        "environment": {
            "python_version": "3.12.13",
            "platform": "test-platform",
            "torch_version": "test-torch",
            "transformers_version": "test-transformers",
        },
    }


class AresIngestScoreTest(unittest.TestCase):
    def test_perfect_required_gates_reaches_one(self) -> None:
        reward = compute_reward(
            gates_payload={
                "gates": {
                    "model_spec": True,
                    "frontend_export": True,
                    "lean_ingest": True,
                }
            },
            validated_gates_payload={
                "gates": {
                    "aresplan_valid": artifact_gate("ares_plan"),
                    "targetplan_valid": artifact_gate("target_plan"),
                    "artifact_consistency": artifact_gate("artifact_consistency"),
                    "shortcut_scan": artifact_gate("shortcut_scan"),
                    "backend_open": artifact_gate("backend_open"),
                    "one_token_logits": artifact_gate("one_token_logits"),
                    "cpp_tvd": artifact_gate("cpp_tvd"),
                    "depth_performance": artifact_gate("depth_performance"),
                }
            },
            oracle_payload=oracle_record(),
            token_payload=token_evidence(),
            performance_payload={
                "workload": "independent_decode",
                "measured_tokens_per_second": 100.0,
                "speed_of_light_tokens_per_second": 100.0,
            },
        )

        self.assertEqual(reward["first_failed_gate"], "complete")
        self.assertEqual(reward["stage_cap"], 1.0)
        self.assertEqual(reward["score"], 1.0)
        self.assertEqual(reward["alpha_execution"], 1.0)
        self.assertEqual(reward["tau_tokens"], 1.0)
        self.assertEqual(reward["delta_inference"], 1.0)

    def test_default_required_gates_do_not_require_cpp_or_hardware(self) -> None:
        reward = compute_reward(
            gates_payload={
                "gates": {
                    "model_spec": True,
                    "frontend_export": True,
                    "lean_ingest": True,
                }
            },
            validated_gates_payload={
                "gates": {
                    "aresplan_valid": artifact_gate("ares_plan"),
                    "targetplan_valid": artifact_gate("target_plan"),
                    "artifact_consistency": artifact_gate("artifact_consistency"),
                    "shortcut_scan": artifact_gate("shortcut_scan"),
                }
            },
            oracle_payload=oracle_record(),
            token_payload={"score": 1.0},
            performance_payload={
                "workload": "independent_decode",
                "measured_tokens_per_second": 100.0,
                "speed_of_light_tokens_per_second": 100.0,
            },
        )

        for gate in CPU_ONLY_GATES:
            self.assertIn(gate, reward["gates"])
        self.assertEqual(reward["first_failed_gate"], "complete")
        self.assertNotIn("cpp_tvd", reward["gates"])

    def test_full_profile_requires_mmlu_pro_after_depth_performance(self) -> None:
        reward = compute_reward(
            gates_payload={
                "gates": {
                    "model_spec": True,
                    "frontend_export": True,
                    "lean_ingest": True,
                }
            },
            validated_gates_payload={
                "gates": {
                    "aresplan_valid": artifact_gate("ares_plan"),
                    "targetplan_valid": artifact_gate("target_plan"),
                    "artifact_consistency": artifact_gate("artifact_consistency"),
                    "shortcut_scan": artifact_gate("shortcut_scan"),
                    "backend_open": artifact_gate("backend_open"),
                    "one_token_logits": artifact_gate("one_token_logits"),
                    "eight_token_greedy": artifact_gate("eight_token_greedy"),
                    "depth_performance": artifact_gate("depth_performance"),
                }
            },
            oracle_payload=oracle_record(),
            token_payload=token_evidence(),
            performance_payload={
                "workload": "independent_decode",
                "measured_tokens_per_second": 100.0,
                "speed_of_light_tokens_per_second": 100.0,
            },
            required_gates=FULL_GATES,
        )

        self.assertEqual(reward["first_failed_gate"], "mmlu_pro")
        self.assertEqual(reward["stage_cap"], 0.98)
        self.assertLess(reward["score"], 1.0)

    def test_full_profile_does_not_require_cpp_comparison(self) -> None:
        self.assertIn("depth_performance", FULL_GATES)
        self.assertNotIn("cpp_tvd", FULL_GATES)
        self.assertIn("cpp_tvd", COMPARISON_GATES)
        self.assertLess(
            COMPARISON_GATES.index("depth_performance"),
            COMPARISON_GATES.index("cpp_tvd"),
        )

    def test_missing_targetplan_caps_fast_token_match(self) -> None:
        reward = compute_reward(
            gates_payload={
                "gates": {
                    "model_spec": True,
                    "frontend_export": True,
                    "lean_ingest": True,
                }
            },
            validated_gates_payload={
                "gates": {"aresplan_valid": artifact_gate("ares_plan")}
            },
            oracle_payload=oracle_record(),
            token_payload={"score": 1.0},
            performance_payload={"delta": 1.0},
        )

        self.assertEqual(reward["first_failed_gate"], "targetplan_valid")
        self.assertEqual(reward["stage_cap"], 0.48)
        self.assertLessEqual(reward["score"], 0.48)

    def test_mock_oracle_cannot_satisfy_oracle_gate(self) -> None:
        reward = compute_reward(
            gates_payload={"gates": {"model_spec": True}},
            oracle_payload=oracle_record(
                kind="mock_fixture",
                oracle="mock_fixture_not_oracle",
            ),
            required_gates=("model_spec", "hf_cpu_oracle"),
        )

        self.assertEqual(reward["first_failed_gate"], "hf_cpu_oracle")
        self.assertFalse(reward["gates"]["hf_cpu_oracle"]["passed"])

    def test_incomplete_oracle_cannot_satisfy_oracle_gate(self) -> None:
        reward = compute_reward(
            gates_payload={"gates": {"model_spec": True}},
            oracle_payload={
                "record_kind": "hf_cpu_oracle_capture",
                "source": {"oracle": "huggingface_transformers_pytorch_cpu"},
            },
            required_gates=("model_spec", "hf_cpu_oracle"),
        )

        self.assertEqual(reward["first_failed_gate"], "hf_cpu_oracle")
        self.assertFalse(reward["gates"]["hf_cpu_oracle"]["passed"])

    def test_explicit_artifact_gate_cannot_replace_oracle_payload(self) -> None:
        reward = compute_reward(
            gates_payload={"gates": {"model_spec": True, "hf_cpu_oracle": True}},
            required_gates=("model_spec", "hf_cpu_oracle"),
        )

        self.assertEqual(reward["first_failed_gate"], "hf_cpu_oracle")
        self.assertFalse(reward["gates"]["hf_cpu_oracle"]["passed"])

    def test_explicit_artifact_gate_cannot_replace_plan_validators(self) -> None:
        reward = compute_reward(
            gates_payload={
                "gates": {
                    "model_spec": True,
                    "frontend_export": True,
                    "lean_ingest": True,
                    "aresplan_valid": True,
                    "targetplan_valid": True,
                }
            },
            oracle_payload=oracle_record(),
            required_gates=(
                "model_spec",
                "hf_cpu_oracle",
                "frontend_export",
                "lean_ingest",
                "aresplan_valid",
                "targetplan_valid",
            ),
        )

        self.assertEqual(reward["first_failed_gate"], "aresplan_valid")
        self.assertFalse(reward["gates"]["aresplan_valid"]["passed"])

    def test_explicit_artifact_gate_cannot_replace_shortcut_scan(self) -> None:
        reward = compute_reward(
            gates_payload={
                "gates": {
                    "model_spec": True,
                    "frontend_export": True,
                    "lean_ingest": True,
                    "shortcut_scan": True,
                }
            },
            validated_gates_payload={
                "gates": {
                    "aresplan_valid": artifact_gate("ares_plan"),
                    "targetplan_valid": artifact_gate("target_plan"),
                    "artifact_consistency": artifact_gate("artifact_consistency"),
                }
            },
            oracle_payload=oracle_record(),
            required_gates=(
                "model_spec",
                "hf_cpu_oracle",
                "frontend_export",
                "lean_ingest",
                "aresplan_valid",
                "targetplan_valid",
                "artifact_consistency",
                "shortcut_scan",
            ),
        )

        self.assertEqual(reward["first_failed_gate"], "shortcut_scan")
        self.assertFalse(reward["gates"]["shortcut_scan"]["passed"])

    def test_explicit_artifact_gate_cannot_replace_consistency_validator(
        self,
    ) -> None:
        reward = compute_reward(
            gates_payload={
                "gates": {
                    "model_spec": True,
                    "frontend_export": True,
                    "lean_ingest": True,
                    "artifact_consistency": True,
                }
            },
            validated_gates_payload={
                "gates": {
                    "aresplan_valid": artifact_gate("ares_plan"),
                    "targetplan_valid": artifact_gate("target_plan"),
                }
            },
            oracle_payload=oracle_record(),
            required_gates=(
                "model_spec",
                "hf_cpu_oracle",
                "frontend_export",
                "lean_ingest",
                "aresplan_valid",
                "targetplan_valid",
                "artifact_consistency",
            ),
        )

        self.assertEqual(reward["first_failed_gate"], "artifact_consistency")
        self.assertFalse(reward["gates"]["artifact_consistency"]["passed"])

    def test_explicit_runtime_gates_cannot_replace_artifact_validators(self) -> None:
        reward = compute_reward(
            gates_payload={
                "gates": {
                    "model_spec": True,
                    "frontend_export": True,
                    "lean_ingest": True,
                    "backend_open": artifact_gate("backend_open"),
                    "one_token_logits": artifact_gate("one_token_logits"),
                    "eight_token_greedy": True,
                    "cpp_tvd": artifact_gate("cpp_tvd"),
                    "depth_performance": artifact_gate("depth_performance"),
                }
            },
            validated_gates_payload={
                "gates": {
                    "aresplan_valid": artifact_gate("ares_plan"),
                    "targetplan_valid": artifact_gate("target_plan"),
                    "artifact_consistency": artifact_gate("artifact_consistency"),
                    "shortcut_scan": artifact_gate("shortcut_scan"),
                }
            },
            oracle_payload=oracle_record(),
            token_payload=token_evidence(),
            performance_payload={
                "workload": "independent_decode",
                "measured_tokens_per_second": 100.0,
                "speed_of_light_tokens_per_second": 100.0,
            },
            required_gates=(
                "model_spec",
                "hf_cpu_oracle",
                "frontend_export",
                "lean_ingest",
                "aresplan_valid",
                "targetplan_valid",
                "artifact_consistency",
                "shortcut_scan",
                "backend_open",
                "one_token_logits",
                "eight_token_greedy",
                "cpp_tvd",
                "depth_performance",
            ),
        )

        self.assertEqual(reward["first_failed_gate"], "backend_open")
        self.assertFalse(reward["gates"]["backend_open"]["passed"])

    def test_explicit_mmlu_pro_gate_cannot_replace_validator(self) -> None:
        reward = compute_reward(
            gates_payload={
                "gates": {
                    "model_spec": True,
                    "frontend_export": True,
                    "lean_ingest": True,
                    "mmlu_pro": True,
                }
            },
            validated_gates_payload={
                "gates": {
                    "aresplan_valid": artifact_gate("ares_plan"),
                    "targetplan_valid": artifact_gate("target_plan"),
                    "artifact_consistency": artifact_gate("artifact_consistency"),
                    "shortcut_scan": artifact_gate("shortcut_scan"),
                    "backend_open": artifact_gate("backend_open"),
                    "one_token_logits": artifact_gate("one_token_logits"),
                    "eight_token_greedy": artifact_gate("eight_token_greedy"),
                    "cpp_tvd": artifact_gate("cpp_tvd"),
                    "depth_performance": artifact_gate("depth_performance"),
                }
            },
            oracle_payload=oracle_record(),
            token_payload=token_evidence(),
            performance_payload={
                "workload": "independent_decode",
                "measured_tokens_per_second": 100.0,
                "speed_of_light_tokens_per_second": 100.0,
            },
            required_gates=FULL_GATES,
        )

        self.assertEqual(reward["first_failed_gate"], "mmlu_pro")
        self.assertFalse(reward["gates"]["mmlu_pro"]["passed"])

    def test_explicit_eight_token_gate_cannot_replace_token_evidence(self) -> None:
        reward = compute_reward(
            gates_payload={
                "gates": {
                    "model_spec": True,
                    "frontend_export": True,
                    "lean_ingest": True,
                    "eight_token_greedy": True,
                }
            },
            validated_gates_payload={
                "gates": {
                    "aresplan_valid": artifact_gate("ares_plan"),
                    "targetplan_valid": artifact_gate("target_plan"),
                    "artifact_consistency": artifact_gate("artifact_consistency"),
                    "shortcut_scan": artifact_gate("shortcut_scan"),
                    "backend_open": artifact_gate("backend_open"),
                    "one_token_logits": artifact_gate("one_token_logits"),
                }
            },
            oracle_payload=oracle_record(),
            required_gates=(
                "model_spec",
                "hf_cpu_oracle",
                "frontend_export",
                "lean_ingest",
                "aresplan_valid",
                "targetplan_valid",
                "artifact_consistency",
                "shortcut_scan",
                "backend_open",
                "one_token_logits",
                "eight_token_greedy",
            ),
        )

        self.assertEqual(reward["first_failed_gate"], "eight_token_greedy")
        self.assertFalse(reward["gates"]["eight_token_greedy"]["passed"])

    def test_cli_writes_reward_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gates = root / "gates.json"
            oracle = root / "oracle.jsonl"
            out_json = root / "reward.json"
            out_txt = root / "reward.txt"
            gates.write_text(json.dumps({"gates": {"model_spec": True}}))
            oracle.write_text(json.dumps(oracle_record()) + "\n")

            rc = main(
                [
                    "--gates",
                    str(gates),
                    "--oracle",
                    str(oracle),
                    "--required-gates",
                    "model_spec",
                    "hf_cpu_oracle",
                    "--output-json",
                    str(out_json),
                    "--output-txt",
                    str(out_txt),
                ]
            )

            self.assertEqual(rc, 0)
            reward = json.loads(out_json.read_text())
            self.assertEqual(reward["first_failed_gate"], "complete")
            self.assertEqual(out_txt.read_text().strip(), f"{reward['score']:.12g}")

    def test_cli_rejects_stale_token_source_digest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gates = root / "gates.json"
            validated_gates = root / "validated-gates.json"
            oracle = root / "oracle.jsonl"
            tokens = root / "tokens.json"
            reference = root / "reference.json"
            candidate = root / "candidate.json"
            out_json = root / "reward.json"
            out_txt = root / "reward.txt"
            reference_payload = {"generated_token_ids": list(range(8))}
            candidate_payload = {"generated_token_ids": list(range(8))}

            gates.write_text(
                json.dumps(
                    {
                        "gates": {
                            "model_spec": True,
                            "frontend_export": True,
                            "lean_ingest": True,
                        }
                    }
                )
            )
            validated_gates.write_text(
                json.dumps(
                    {
                        "gates": {
                            "aresplan_valid": artifact_gate("ares_plan"),
                            "targetplan_valid": artifact_gate("target_plan"),
                            "artifact_consistency": artifact_gate(
                                "artifact_consistency"
                            ),
                            "shortcut_scan": artifact_gate("shortcut_scan"),
                            "backend_open": artifact_gate("backend_open"),
                            "one_token_logits": artifact_gate("one_token_logits"),
                        }
                    }
                )
            )
            oracle.write_text(json.dumps(oracle_record()) + "\n")
            reference.write_text(json.dumps(reference_payload))
            candidate.write_text(json.dumps(candidate_payload))
            tokens.write_text(
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

            rc = main(
                [
                    "--gates",
                    str(gates),
                    "--validated-gates",
                    str(validated_gates),
                    "--oracle",
                    str(oracle),
                    "--tokens",
                    str(tokens),
                    "--required-gates",
                    *BACKEND_GATES,
                    "--output-json",
                    str(out_json),
                    "--output-txt",
                    str(out_txt),
                ]
            )

            self.assertEqual(rc, 0)
            reward = json.loads(out_json.read_text())
            self.assertEqual(reward["first_failed_gate"], "eight_token_greedy")
            self.assertFalse(reward["gates"]["eight_token_greedy"]["passed"])
            self.assertEqual(reward["tau_tokens"], 0.0)


if __name__ == "__main__":
    unittest.main()

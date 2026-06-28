from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
EVALUATOR = (
    REPO_ROOT / "templates/ares-ingest-harbor-task/files/evaluate_ares_ingest.py"
)


def oracle_record() -> dict[str, Any]:
    return {
        "schema": "ares.oracles.hf_cpu.record.v1",
        "record_kind": "hf_cpu_oracle_capture",
        "capture_id": "test-capture",
        "created_utc": "2026-06-26T00:00:00Z",
        "source": {
            "oracle": "huggingface_transformers_pytorch_cpu",
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
                    {"rank": 1, "token_id": 3, "token_text": " world", "logit": 12.5}
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
                    {"rank": 1, "token_id": 2, "token_text": "</s>", "logit": 9.25}
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


def valid_ares_plan() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "config": {
            "dim": 4,
            "n_layers": 1,
            "n_heads": 1,
            "n_kv_head": 1,
            "head_size": 4,
            "vocab_size": 8,
            "wcls_is_compensated": False,
            "attn_softcapping": False,
            "attn_sinks": False,
            "is_eagle": False,
        },
        "weights": ["w"],
        "buffers": [
            {
                "name": "input_ids",
                "role": "cut_in",
                "type": {
                    "kind": "array",
                    "scalar_type": "int",
                    "outer": "singleton",
                    "inner": "1",
                },
            },
            {
                "name": "ares_logits",
                "role": "output",
                "type": {
                    "kind": "array",
                    "scalar_type": "real",
                    "outer": "singleton",
                    "inner": "8",
                },
            },
        ],
        "stmts": [
            {
                "stmt": "matmul",
                "weight": "w",
                "input": "input_ids",
                "result": "ares_logits",
            }
        ],
        "provenance": {
            "fx_hash": "test-fx",
            "rule_corpus_hash": "test-rules",
            "emitter_version": "ingest-lean test",
            "target_executor": "tron",
            "hardware_policy": "tron",
            "lowering_path": "Ingest.Plan.ToJson",
        },
    }


def valid_target_plan() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "producer": {
            "language": "lean",
            "tool": "ingest-lean",
            "module": "Ingest.TargetPlan",
            "version": "0.1.0",
        },
        "backend_id": "tron",
        "model_id": "synthetic/model",
        "source": {
            "schema_version": 2,
            "statement_count": 1,
            "config": valid_ares_plan()["config"],
            "provenance": valid_ares_plan()["provenance"],
        },
        "declared_runtime_bindings": ["input_ids", "ares_logits"],
        "hw_policy": {
            "fallback_policy": "forbidden",
            "placement": "single_device",
            "notes": {},
        },
        "operations": [
            {
                "id": "runtime.input_ids",
                "role": "runtime_binding",
                "action": "runtime_binding",
                "source": {"type": "runtime_binding", "name": "input_ids"},
                "requirements": {},
            },
            {
                "id": "runtime.ares_logits",
                "role": "runtime_binding",
                "action": "runtime_binding",
                "source": {"type": "runtime_binding", "name": "ares_logits"},
                "requirements": {},
            },
            {
                "id": "stmt.00000.matmul",
                "role": "semantic",
                "action": "matmul",
                "source": {
                    "type": "ares_plan_statement",
                    "statement_index": 0,
                    "statement_kind": "matmul",
                    "statement_name": "w",
                },
                "requirements": {},
            },
        ],
    }


def replay_context() -> dict[str, Any]:
    return {
        "context_tokens": [1, 2],
        "context_tokens_role": "prompt",
        "context_count": 2,
        "new_count": 1,
        "runtime_request_token_count": 2,
        "context_prefix_token_count": 0,
        "last_token": 2,
    }


def backend_open_evidence() -> dict[str, Any]:
    return {
        "schema": "ares.runtime.backend_open.v1",
        "evidence_class": "system_under_test",
        "status": "opened",
        "backend_id": "tron",
        "ares_plan": {"path": "ares-plan.json", "sha256": "a" * 64},
        "target_plan": {
            "path": "tron.target-plan.json",
            "sha256": "b" * 64,
            "backend_id": "tron",
        },
        "events": [{"event": "backend_open", "backend_id": "tron"}],
        "runtime_generated_sidecars": False,
    }


def one_token_logits_evidence() -> dict[str, Any]:
    return {
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


def cpp_tvd_evidence() -> dict[str, Any]:
    return {
        "schema": "ares.comparison.cpp_tvd.v1",
        "evidence_class": "comparison",
        "comparison_source": "cpp_tron_rinzler",
        "candidate": "ares",
        "tvd": 0.001,
        "tvd_threshold": 0.01,
        "replay_context": replay_context(),
    }


def depth_performance_evidence() -> dict[str, Any]:
    return {
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


class AresIngestEvaluatorTest(unittest.TestCase):
    def run_evaluator(
        self, spec: dict[str, Any], task_file_texts: dict[str, str]
    ) -> dict[str, Any]:
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root)

        ares_repo = root / "ares"
        task_files = root / "task-files"
        work_dir = root / "work"
        logs_dir = root / "logs/verifier"
        reward_json = logs_dir / "reward.json"
        reward_txt = root / "logs/reward.txt"
        ares_repo.mkdir()
        task_files.mkdir()

        for name, text in task_file_texts.items():
            (task_files / name).write_text(text)

        full_spec = {
            "model": "synthetic/model",
            "work_dir": str(work_dir),
            **spec,
        }
        (task_files / "model_spec.json").write_text(json.dumps(full_spec))

        env = os.environ.copy()
        pythonpath = [str(REPO_ROOT)]
        if existing := env.get("PYTHONPATH"):
            pythonpath.append(existing)
        env.update(
            {
                "ARES_REPO": str(ares_repo),
                "TASK_FILES_DIR": str(task_files),
                "VERIFIER_LOG_DIR": str(logs_dir),
                "REWARD_JSON": str(reward_json),
                "REWARD_TXT": str(reward_txt),
                "PYTHONPATH": ":".join(pythonpath),
            }
        )
        subprocess.run([sys.executable, str(EVALUATOR)], env=env, check=True)
        return {
            "work_dir": work_dir,
            "logs_dir": logs_dir,
            "reward_json": reward_json,
            "reward_txt": reward_txt,
            "reward": json.loads(reward_json.read_text()),
        }

    def test_evaluator_scores_full_artifact_backed_profile(self) -> None:
        result = self.run_evaluator(
            {
                "required_gates": [
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
                ],
                "explicit_gates": {
                    "model_spec": True,
                    "frontend_export": True,
                    "lean_ingest": True,
                    "eight_token_greedy": True,
                },
                "oracle_records": "oracle.jsonl",
                "ares_plan": "ares-plan.json",
                "target_plan": "tron.target-plan.json",
                "backend_open_evidence": "backend-open.json",
                "one_token_logits_evidence": "one-token.json",
                "cpp_tvd_evidence": "cpp-tvd.json",
                "depth_performance_evidence": "depth.json",
                "token_comparison": {
                    "reference": "reference_tokens.json",
                    "candidate": "candidate_tokens.json",
                },
                "performance_comparison": {
                    "measured": "measured.log",
                    "speed_of_light": "speed.json",
                    "workload": "independent_decode",
                },
            },
            {
                "oracle.jsonl": json.dumps(oracle_record()) + "\n",
                "ares-plan.json": json.dumps(valid_ares_plan()) + "\n",
                "tron.target-plan.json": json.dumps(valid_target_plan()) + "\n",
                "backend-open.json": json.dumps(backend_open_evidence()) + "\n",
                "one-token.json": json.dumps(one_token_logits_evidence()) + "\n",
                "cpp-tvd.json": json.dumps(cpp_tvd_evidence()) + "\n",
                "depth.json": json.dumps(depth_performance_evidence()) + "\n",
                "reference_tokens.json": json.dumps([1, 2, 3]),
                "candidate_tokens.json": json.dumps([1, 2, 3]),
                "measured.log": "Throughput 80.0 tok/s\n",
                "speed.json": json.dumps(
                    {
                        "targets": {
                            "independent_decode": {
                                "speed_of_light_tokens_per_second": 100.0
                            }
                        }
                    }
                ),
            },
        )

        reward = result["reward"]
        self.assertEqual(reward["first_failed_gate"], "complete")
        self.assertEqual(reward["stage_cap"], 1.0)
        self.assertEqual(reward["tau_tokens"], 1.0)
        self.assertEqual(reward["delta_inference"], 0.8)
        self.assertTrue((result["work_dir"] / "tokens.json").exists())
        self.assertTrue((result["work_dir"] / "performance.json").exists())

    def test_evaluator_rejects_target_plan_model_mismatch(self) -> None:
        target_plan = valid_target_plan()
        target_plan["model_id"] = "fixture/model"
        result = self.run_evaluator(
            {
                "required_gates": [
                    "model_spec",
                    "hf_cpu_oracle",
                    "frontend_export",
                    "lean_ingest",
                    "aresplan_valid",
                    "targetplan_valid",
                    "artifact_consistency",
                    "shortcut_scan",
                ],
                "explicit_gates": {
                    "model_spec": True,
                    "frontend_export": True,
                    "lean_ingest": True,
                },
                "oracle_records": "oracle.jsonl",
                "ares_plan": "ares-plan.json",
                "target_plan": "tron.target-plan.json",
            },
            {
                "oracle.jsonl": json.dumps(oracle_record()) + "\n",
                "ares-plan.json": json.dumps(valid_ares_plan()) + "\n",
                "tron.target-plan.json": json.dumps(target_plan) + "\n",
            },
        )

        reward = result["reward"]
        self.assertEqual(reward["first_failed_gate"], "artifact_consistency")
        self.assertFalse(reward["gates"]["artifact_consistency"]["passed"])

    def test_evaluator_rejects_placeholder_plan_json(self) -> None:
        result = self.run_evaluator(
            {
                "required_gates": [
                    "model_spec",
                    "hf_cpu_oracle",
                    "frontend_export",
                    "lean_ingest",
                    "aresplan_valid",
                    "targetplan_valid",
                ],
                "explicit_gates": {
                    "model_spec": True,
                    "frontend_export": True,
                    "lean_ingest": True,
                },
                "oracle_records": "oracle.jsonl",
                "ares_plan": "ares-plan.json",
                "target_plan": "tron.target-plan.json",
            },
            {
                "oracle.jsonl": json.dumps(oracle_record()) + "\n",
                "ares-plan.json": "{}\n",
                "tron.target-plan.json": "{}\n",
            },
        )

        reward = result["reward"]
        self.assertEqual(reward["first_failed_gate"], "aresplan_valid")
        self.assertFalse(reward["gates"]["aresplan_valid"]["passed"])

    def test_explicit_gates_cannot_replace_missing_oracle_artifact(self) -> None:
        result = self.run_evaluator(
            {
                "required_gates": [
                    "model_spec",
                    "hf_cpu_oracle",
                    "frontend_export",
                    "lean_ingest",
                    "aresplan_valid",
                    "targetplan_valid",
                ],
                "explicit_gates": {
                    "model_spec": True,
                    "hf_cpu_oracle": True,
                    "frontend_export": True,
                    "lean_ingest": True,
                    "aresplan_valid": True,
                    "targetplan_valid": True,
                },
            },
            {},
        )

        reward = result["reward"]
        self.assertEqual(reward["first_failed_gate"], "hf_cpu_oracle")
        self.assertFalse(reward["gates"]["hf_cpu_oracle"]["passed"])

    def test_explicit_gates_cannot_replace_missing_plan_artifacts(self) -> None:
        result = self.run_evaluator(
            {
                "required_gates": [
                    "model_spec",
                    "hf_cpu_oracle",
                    "frontend_export",
                    "lean_ingest",
                    "aresplan_valid",
                    "targetplan_valid",
                ],
                "explicit_gates": {
                    "model_spec": True,
                    "frontend_export": True,
                    "lean_ingest": True,
                    "aresplan_valid": True,
                    "targetplan_valid": True,
                },
                "oracle_records": "oracle.jsonl",
            },
            {"oracle.jsonl": json.dumps(oracle_record()) + "\n"},
        )

        reward = result["reward"]
        self.assertEqual(reward["first_failed_gate"], "aresplan_valid")
        self.assertFalse(reward["gates"]["aresplan_valid"]["passed"])

    def test_explicit_gates_cannot_replace_missing_target_plan_artifact(self) -> None:
        result = self.run_evaluator(
            {
                "required_gates": [
                    "model_spec",
                    "hf_cpu_oracle",
                    "frontend_export",
                    "lean_ingest",
                    "aresplan_valid",
                    "targetplan_valid",
                ],
                "explicit_gates": {
                    "model_spec": True,
                    "frontend_export": True,
                    "lean_ingest": True,
                    "targetplan_valid": True,
                },
                "oracle_records": "oracle.jsonl",
                "ares_plan": "ares-plan.json",
            },
            {
                "oracle.jsonl": json.dumps(oracle_record()) + "\n",
                "ares-plan.json": json.dumps(valid_ares_plan()) + "\n",
            },
        )

        reward = result["reward"]
        self.assertEqual(reward["first_failed_gate"], "targetplan_valid")
        self.assertFalse(reward["gates"]["targetplan_valid"]["passed"])

    def test_evaluator_writes_nonexecuted_command_wrapper_plan(self) -> None:
        result = self.run_evaluator(
            {
                "required_gates": [
                    "model_spec",
                    "backend_open",
                    "one_token_logits",
                ],
                "explicit_gates": {"model_spec": True},
                "weights": "/weights/synthetic",
                "ares_plan": "ares-plan.json",
                "target_plan": "tron.target-plan.json",
            },
            {},
        )

        plan = json.loads((result["work_dir"] / "command_wrappers.json").read_text())
        self.assertEqual(plan["schema"], "ares.autoagent.command_wrappers.v1")
        self.assertFalse(plan["execute_command_wrappers"])
        self.assertEqual(plan["command_gates"], [])
        self.assertEqual(
            {wrapper["name"] for wrapper in plan["wrappers"]},
            {"rinzler_chat_one_token", "rinzler_full_inference_smoke"},
        )
        self.assertEqual(
            sorted(path.name for path in result["logs_dir"].glob("rinzler*.log")),
            [],
            "wrapper commands should not run unless execute_command_wrappers is true",
        )


if __name__ == "__main__":
    unittest.main()

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

    def test_evaluator_scores_cpu_only_artifacts(self) -> None:
        result = self.run_evaluator(
            {
                "required_gates": [
                    "model_spec",
                    "hf_cpu_oracle",
                    "frontend_export",
                    "lean_ingest",
                    "aresplan_valid",
                    "targetplan_valid",
                    "backend_open",
                    "eight_token_greedy",
                    "depth_performance",
                ],
                "explicit_gates": {
                    "model_spec": True,
                    "frontend_export": True,
                    "lean_ingest": True,
                    "backend_open": True,
                    "eight_token_greedy": True,
                    "depth_performance": True,
                },
                "oracle_records": "oracle.jsonl",
                "ares_plan": "ares-plan.json",
                "target_plan": "tron.target-plan.json",
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


if __name__ == "__main__":
    unittest.main()

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
        oracle = {
            "record_kind": "hf_cpu_oracle_capture",
            "source": {"oracle": "huggingface_transformers_pytorch_cpu"},
            "prompt_token_ids": [1, 2, 3],
            "generated_token_ids": [4],
        }
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
                "oracle.jsonl": json.dumps(oracle) + "\n",
                "ares-plan.json": "{}\n",
                "tron.target-plan.json": "{}\n",
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


if __name__ == "__main__":
    unittest.main()

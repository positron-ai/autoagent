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
    REPO_ROOT / "templates/tron-ingest-harbor-task/files/evaluate_tron_ingest.py"
)


class TronIngestEvaluatorTest(unittest.TestCase):
    def run_evaluator(
        self,
        spec: dict[str, Any],
        *,
        task_file_texts: dict[str, str] | None = None,
        env_overrides: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root)

        tron_repo = root / "tron"
        ingest_dir = tron_repo / "ingest"
        task_files = root / "task-files"
        work_dir = root / "work"
        logs_dir = root / "logs/verifier"
        reward_json = logs_dir / "reward.json"
        reward_txt = root / "logs/reward.txt"

        ingest_dir.mkdir(parents=True)
        task_files.mkdir()

        for name, text in (task_file_texts or {}).items():
            (task_files / name).write_text(text)

        full_spec = {
            "hf_model": "synthetic/model",
            "work_dir": str(work_dir),
            "run_typedfx": False,
            "run_bulk": False,
            **spec,
        }
        (task_files / "model_spec.json").write_text(json.dumps(full_spec))

        env = os.environ.copy()
        pythonpath = [str(REPO_ROOT)]
        if existing := env.get("PYTHONPATH"):
            pythonpath.append(existing)
        env.update(
            {
                "TRON_REPO": str(tron_repo),
                "TASK_FILES_DIR": str(task_files),
                "VERIFIER_LOG_DIR": str(logs_dir),
                "REWARD_JSON": str(reward_json),
                "REWARD_TXT": str(reward_txt),
                "PYTHONPATH": ":".join(pythonpath),
            }
        )
        if env_overrides:
            env.update(env_overrides)
        subprocess.run([sys.executable, str(EVALUATOR)], env=env, check=True)

        return {
            "root": root,
            "tron_repo": tron_repo,
            "task_files": task_files,
            "work_dir": work_dir,
            "logs_dir": logs_dir,
            "reward_json": reward_json,
            "reward_txt": reward_txt,
            "reward": json.loads(reward_json.read_text()),
            "gates": json.loads((work_dir / "gates.json").read_text())["gates"],
        }

    def test_evaluator_writes_reward_with_command_tokens_and_performance(self) -> None:
        result = self.run_evaluator(
            {
                "required_gates": ["cpp_compile"],
                "explicit_gates": {
                    "architecture": True,
                    "eqsat_structure": True,
                    "typedfx_logits": True,
                    "bulk_logits": True,
                    "cpu_logits": True,
                    "fpga_logits": True,
                },
                "command_gates": [
                    {
                        "name": "cpp_compile",
                        "cwd": "tron_repo",
                        "command": "printf compiled",
                        "timeout_sec": 10,
                    }
                ],
                "token_comparison": {
                    "reference": "reference_tokens.json",
                    "candidate": "candidate_tokens.json",
                },
                "performance_comparison": {
                    "measured": "measured.log",
                    "speed_of_light": "speed_of_light.json",
                    "workload": "independent_decode",
                },
            },
            task_file_texts={
                "reference_tokens.json": json.dumps([1, 2, 3, 4]),
                "candidate_tokens.json": json.dumps([1, 2, 3, 9]),
                "measured.log": "Throughput 80.0 tok/s.\n",
                "speed_of_light.json": json.dumps(
                    {
                        "targets": {
                            "independent_decode": {
                                "speed_of_light_tokens_per_second": 100.0,
                            }
                        }
                    }
                ),
            },
        )

        work_dir = result["work_dir"]
        logs_dir = result["logs_dir"]
        self.assertTrue(result["reward_json"].exists())
        self.assertTrue(result["reward_txt"].exists())
        self.assertTrue((work_dir / "tokens.json").exists())
        self.assertTrue((work_dir / "performance.json").exists())
        self.assertTrue((logs_dir / "cpp_compile.log").exists())

        reward = result["reward"]
        tokens = json.loads((work_dir / "tokens.json").read_text())
        performance = json.loads((work_dir / "performance.json").read_text())

        self.assertEqual(reward["first_failed_gate"], "complete")
        self.assertEqual(reward["stage_cap"], 1.0)
        self.assertTrue(reward["gates"]["cpp_compile"]["passed"])
        self.assertGreater(reward["tau"], 0.0)
        self.assertLess(reward["tau"], 1.0)
        self.assertEqual(reward["delta"], 0.8)
        self.assertEqual(
            tokens["reference"],
            str(result["task_files"] / "reference_tokens.json"),
        )
        self.assertEqual(performance["measured_tokens_per_second"], 80.0)
        self.assertEqual(performance["workload"], "independent_decode")

    def test_evaluator_rejects_reuse_workload_for_performance_delta(self) -> None:
        result = self.run_evaluator(
            {
                "required_gates": ["cpp_compile"],
                "explicit_gates": {
                    "architecture": True,
                    "eqsat_structure": True,
                    "typedfx_logits": True,
                    "bulk_logits": True,
                    "cpp_compile": True,
                    "cpu_logits": True,
                    "fpga_logits": True,
                },
                "performance_comparison": {
                    "measured": "measured.json",
                    "speed_of_light": "speed_of_light.json",
                    "workload": "independent_decode",
                },
            },
            task_file_texts={
                "measured.json": json.dumps(
                    {
                        "workload": "prefix_cache_reuse",
                        "measured_tokens_per_second": 900.0,
                    }
                ),
                "speed_of_light.json": json.dumps(
                    {
                        "targets": {
                            "independent_decode": {
                                "speed_of_light_tokens_per_second": 300.0,
                            }
                        }
                    }
                ),
            },
        )

        performance = json.loads((result["work_dir"] / "performance.json").read_text())

        self.assertEqual(result["reward"]["delta"], 0.0)
        self.assertEqual(performance["measured_workload"], "prefix_cache_reuse")
        self.assertEqual(performance["required_workload"], "independent_decode")
        self.assertIn("does not match", performance["error"])

    def test_command_gate_fail_regex_fails_zero_exit_command(self) -> None:
        result = self.run_evaluator(
            {
                "required_gates": ["cpu_logits"],
                "explicit_gates": {
                    "architecture": True,
                    "eqsat_structure": True,
                    "typedfx_logits": True,
                    "bulk_logits": True,
                    "cpp_compile": True,
                    "fpga_logits": True,
                },
                "command_gates": [
                    {
                        "name": "cpu_logits",
                        "cwd": "tron_repo",
                        "command": "printf 'Busted!\\n'",
                        "timeout_sec": 10,
                        "fail_regexes": ["Busted!"],
                    }
                ],
            }
        )

        reward = result["reward"]
        gate = result["gates"]["cpu_logits"]
        self.assertEqual(reward["first_failed_gate"], "cpu_logits")
        self.assertEqual(reward["stage_cap"], 0.82)
        self.assertFalse(reward["gates"]["cpu_logits"]["passed"])
        self.assertEqual(gate["returncode"], 0)
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["checks"]["fail_regexes"][0]["match"], "Busted!")

    def test_command_gate_gets_tron_libstdcxx_path(self) -> None:
        result = self.run_evaluator(
            {
                "required_gates": ["cpp_compile"],
                "explicit_gates": {
                    "architecture": True,
                    "eqsat_structure": True,
                    "typedfx_logits": True,
                    "bulk_logits": True,
                    "cpu_logits": True,
                    "fpga_logits": True,
                },
                "command_gates": [
                    {
                        "name": "cpp_compile",
                        "cwd": "tron_repo",
                        "command": 'case "$LD_LIBRARY_PATH" in /tmp/libstdcxx:/tmp/existing*) exit 0;; *) printf \'%s\\n\' "$LD_LIBRARY_PATH"; exit 1;; esac',
                        "timeout_sec": 10,
                    }
                ],
            },
            env_overrides={
                "TRON_INGEST_LIBSTDCXX_PATH": "/tmp/libstdcxx",
                "LD_LIBRARY_PATH": "/tmp/existing",
            },
        )

        gate = result["gates"]["cpp_compile"]
        self.assertTrue(gate["passed"])

    def test_command_gate_requires_pass_regexes(self) -> None:
        result = self.run_evaluator(
            {
                "required_gates": ["cpp_compile"],
                "explicit_gates": {
                    "architecture": True,
                    "eqsat_structure": True,
                    "typedfx_logits": True,
                    "bulk_logits": True,
                    "cpu_logits": True,
                    "fpga_logits": True,
                },
                "command_gates": [
                    {
                        "name": "cpp_compile",
                        "cwd": "tron_repo",
                        "command": "printf 'compile finished\\n'",
                        "timeout_sec": 10,
                        "pass_regexes": ["compile finished", "link finished"],
                    }
                ],
            }
        )

        reward = result["reward"]
        gate = result["gates"]["cpp_compile"]
        self.assertEqual(reward["first_failed_gate"], "cpp_compile")
        self.assertFalse(reward["gates"]["cpp_compile"]["passed"])
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["checks"]["missing_pass_regexes"], ["link finished"])

    def test_command_gate_numeric_threshold_fails_zero_exit_command(self) -> None:
        result = self.run_evaluator(
            {
                "required_gates": ["cpu_logits"],
                "explicit_gates": {
                    "architecture": True,
                    "eqsat_structure": True,
                    "typedfx_logits": True,
                    "bulk_logits": True,
                    "cpp_compile": True,
                    "fpga_logits": True,
                },
                "command_gates": [
                    {
                        "name": "cpu_logits",
                        "cwd": "tron_repo",
                        "command": "printf 'Relative error in L2: 0.42\\n'",
                        "timeout_sec": 10,
                        "numeric_checks": [
                            {
                                "name": "relative_l2",
                                "regex": "Relative error in L2:\\s*([0-9.eE+-]+)",
                                "max": 0.1,
                            }
                        ],
                    }
                ],
            }
        )

        reward = result["reward"]
        gate = result["gates"]["cpu_logits"]
        numeric = gate["checks"]["numeric_checks"][0]
        self.assertEqual(reward["first_failed_gate"], "cpu_logits")
        self.assertFalse(reward["gates"]["cpu_logits"]["passed"])
        self.assertFalse(gate["passed"])
        self.assertEqual(numeric["name"], "relative_l2")
        self.assertEqual(numeric["value"], 0.42)
        self.assertFalse(numeric["passed"])
        self.assertEqual(numeric["max"], 0.1)


if __name__ == "__main__":
    unittest.main()

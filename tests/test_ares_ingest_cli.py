from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ares_ingest_autoagent.ares_cli import (
    DEFAULT_REFINER_COMMAND,
    build_parser,
    config_from_args,
    initialize_run,
    main,
    slugify,
)


class AresIngestCliTest(unittest.TestCase):
    def test_slugify_provider_model(self) -> None:
        self.assertEqual(
            slugify("meta-llama/Llama-3.1-8B-Instruct"),
            "meta-llama-llama-3-1-8b-instruct",
        )

    def test_refiner_defaults_to_codex(self) -> None:
        args = build_parser().parse_args(["Provider/Model"])
        cfg = config_from_args(args)

        self.assertEqual(cfg.refinement_command, DEFAULT_REFINER_COMMAND)

    def test_no_refiner_disables_refinement_command(self) -> None:
        args = build_parser().parse_args(["--no-refiner", "Provider/Model"])
        cfg = config_from_args(args)

        self.assertIsNone(cfg.refinement_command)

    def test_initialize_run_writes_durable_state(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            args = build_parser().parse_args(
                [
                    "--ares-repo",
                    str(root),
                    "--run-dir",
                    str(run_dir),
                    "--setup-only",
                    "Provider/Model",
                ]
            )
            cfg = config_from_args(args)

            state = initialize_run(cfg)

            self.assertEqual(state["status"], "initialized_setup_only")
            self.assertTrue((run_dir / "model_spec.json").exists())
            self.assertTrue((run_dir / "state.json").exists())
            self.assertTrue((run_dir / "reward.json").exists())
            self.assertTrue((run_dir / "reward.txt").exists())
            self.assertTrue((run_dir / "handoff.md").exists())
            spec = json.loads((run_dir / "model_spec.json").read_text())
            self.assertEqual(spec["safe_model"], "provider-model")
            self.assertEqual(spec["gate_profile"], "cpu-only")
            self.assertNotIn("cpp_tvd", spec["required_gates"])
            self.assertEqual(spec["explicit_gates"]["model_spec"]["passed"], True)
            self.assertIn("shortcut_scan", spec["required_gates"])
            self.assertIn("artifact_consistency", spec["required_gates"])
            self.assertEqual(spec["expected_model_ids"], ["Provider/Model"])
            self.assertEqual(
                spec["explicit_gates"]["shortcut_scan"]["artifact_validator"],
                "shortcut_scan",
            )
            self.assertEqual(
                state["reward"]["gates"]["shortcut_scan"]["passed"],
                True,
            )
            self.assertEqual(state["reward"]["first_failed_gate"], "hf_cpu_oracle")
            self.assertEqual(state["refinement_loop"], "setup_only")
            self.assertEqual(
                [skill["name"] for skill in state["workflow_skills"]],
                ["command-wiggum", "ares-evidence", "ares-python", "command-fess"],
            )
            self.assertIn(
                "tools/oracles/hf-cpu/",
                state["workflow_skills"][2]["allowed_scope"],
            )
            handoff = (run_dir / "handoff.md").read_text()
            self.assertIn("## Workflow Skills", handoff)
            self.assertIn("`ares-python`", handoff)
            self.assertIn("validate-jsonl", handoff)

    def test_no_refiner_blocks_below_target(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"

            rc = main(
                [
                    "--ares-repo",
                    str(root),
                    "--run-dir",
                    str(run_dir),
                    "--no-refiner",
                    "Provider/Model",
                ]
            )

            self.assertEqual(rc, 3)
            state = json.loads((run_dir / "state.json").read_text())
            self.assertEqual(state["status"], "blocked_no_refiner")
            self.assertEqual(state["refinement_loop"], "one_failing_gate")
            self.assertEqual(state["history"][0]["first_failed_gate"], "hf_cpu_oracle")

    def test_target_score_completion_does_not_invoke_refiner(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"

            rc = main(
                [
                    "--ares-repo",
                    str(root),
                    "--run-dir",
                    str(run_dir),
                    "--target-score",
                    "0.05",
                    "--refinement-command",
                    "exit 99",
                    "Provider/Model",
                ]
            )

            self.assertEqual(rc, 0)
            state = json.loads((run_dir / "state.json").read_text())
            self.assertEqual(state["status"], "complete")
            self.assertFalse((run_dir / "logs/01-refiner.log").exists())

    def test_refiner_loop_writes_prompt_log_and_history(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"

            rc = main(
                [
                    "--ares-repo",
                    str(root),
                    "--run-dir",
                    str(run_dir),
                    "--max-iterations",
                    "2",
                    "--stall-patience",
                    "5",
                    "--refinement-command",
                    'printf \'%s\\n\' "$FIRST_FAILED_GATE" > "$RUN_DIR/refiner-ran.txt"',
                    "Provider/Model",
                ]
            )

            self.assertEqual(rc, 2)
            prompt = run_dir / "prompts/refinement-01.md"
            self.assertTrue(prompt.exists())
            prompt_text = prompt.read_text()
            self.assertIn(
                "Work only the first failing gate: `hf_cpu_oracle`", prompt_text
            )
            self.assertIn("## Expected Workflow Skills", prompt_text)
            self.assertIn("`command-wiggum`", prompt_text)
            self.assertIn("`ares-python`", prompt_text)
            self.assertIn("Allowed scope:", prompt_text)
            self.assertIn("## Allowed Write Scope", prompt_text)
            self.assertIn("HF Transformers on PyTorch CPU", prompt_text)
            self.assertIn(
                f"ares-ingest-agent Provider/Model --ares-repo {root.resolve()} "
                f"--run-dir {run_dir.resolve()} --no-refiner",
                prompt_text,
            )
            self.assertEqual(
                (run_dir / "refiner-ran.txt").read_text().strip(), "hf_cpu_oracle"
            )
            self.assertTrue((run_dir / "logs/01-refiner.log").exists())
            state = json.loads((run_dir / "state.json").read_text())
            self.assertEqual(state["status"], "max_iterations")
            self.assertEqual(
                [skill["name"] for skill in state["workflow_skills"]],
                ["command-wiggum", "ares-evidence", "ares-python", "command-fess"],
            )
            self.assertEqual(
                [item["status"] for item in state["history"]],
                ["refiner_ran", "max_iterations"],
            )
            self.assertEqual(Path(state["latest_refinement_prompt"]), prompt.resolve())

    def test_missing_oracle_records_writes_failure_state_over_corrupt_state(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "state.json").write_text("{")
            (run_dir / "model_spec.json").write_text(
                json.dumps(
                    {
                        "model": "Provider/Model",
                        "required_gates": ["model_spec", "hf_cpu_oracle"],
                        "explicit_gates": {
                            "model_spec": {
                                "passed": True,
                                "score": 1.0,
                            },
                        },
                        "oracle_records": "missing-oracle.jsonl",
                    }
                )
            )

            with self.assertRaises(SystemExit) as raised:
                main(
                    [
                        "--ares-repo",
                        str(root),
                        "--run-dir",
                        str(run_dir),
                        "--no-refiner",
                        "Provider/Model",
                    ]
                )

            self.assertEqual(raised.exception.code, 1)
            state = json.loads((run_dir / "state.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertIn("missing JSON file", state["error"])
            self.assertIn("previous_state_error", state)

    def test_refiner_failure_writes_state_and_handoff(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"

            with self.assertRaises(SystemExit) as raised:
                main(
                    [
                        "--ares-repo",
                        str(root),
                        "--run-dir",
                        str(run_dir),
                        "--refinement-command",
                        "exit 7",
                        "Provider/Model",
                    ]
                )

            self.assertEqual(raised.exception.code, 1)
            state = json.loads((run_dir / "state.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertIn("refiner failed with exit 7", state["error"])
            self.assertIn("Status: `failed`", (run_dir / "handoff.md").read_text())

    def test_main_setup_only_writes_state(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            rc = main(
                [
                    "--ares-repo",
                    str(root),
                    "--run-dir",
                    str(run_dir),
                    "--setup-only",
                    "Provider/Model",
                ]
            )

            self.assertEqual(rc, 0)
            state = json.loads((run_dir / "state.json").read_text())
            self.assertEqual(state["refinement_loop"], "setup_only")

    def test_backend_profile_writes_command_wrapper_plan(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"

            rc = main(
                [
                    "--ares-repo",
                    str(root),
                    "--run-dir",
                    str(run_dir),
                    "--gate-profile",
                    "backend",
                    "--setup-only",
                    "Provider/Model",
                ]
            )

            self.assertEqual(rc, 0)
            spec = json.loads((run_dir / "model_spec.json").read_text())
            self.assertEqual(spec["command_wrapper_plan"], "command_wrappers.json")
            plan = json.loads((run_dir / "command_wrappers.json").read_text())
            self.assertEqual(plan["schema"], "ares.autoagent.command_wrappers.v1")
            self.assertFalse(plan["execute_command_wrappers"])
            self.assertEqual(plan["command_gates"], [])
            self.assertEqual(
                {wrapper["name"] for wrapper in plan["wrappers"]},
                {"rinzler_chat_one_token", "rinzler_full_inference_smoke"},
            )
            self.assertTrue(
                all(wrapper["missing_inputs"] for wrapper in plan["wrappers"])
            )


if __name__ == "__main__":
    unittest.main()

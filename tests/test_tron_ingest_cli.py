from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tron_ingest_autoagent.ingest_cli import (
    DEFAULT_REFINER_COMMAND,
    IngestError,
    build_parser,
    bootstrap_dir,
    config_from_args,
    deep_merge,
    reward_fingerprint,
    slugify,
    write_failure_state,
)


class TronIngestCliTest(unittest.TestCase):
    def test_slugify_provider_model(self) -> None:
        self.assertEqual(slugify("JackFram/llama-68m"), "jackfram-llama-68m")
        self.assertEqual(
            slugify("Qwen/Qwen2.5-32B-Instruct"),
            "qwen-qwen2-5-32b-instruct",
        )

    def test_deep_merge_preserves_unrelated_spec_keys(self) -> None:
        merged = deep_merge(
            {
                "explicit_gates": {"cpp_compile": {"passed": True}},
                "required_gates": ["cpp_compile"],
            },
            {
                "explicit_gates": {"cpu_logits": {"passed": True}},
                "token_results_json": "tokens.json",
            },
        )

        self.assertEqual(
            merged["explicit_gates"],
            {
                "cpp_compile": {"passed": True},
                "cpu_logits": {"passed": True},
            },
        )
        self.assertEqual(merged["required_gates"], ["cpp_compile"])
        self.assertEqual(merged["token_results_json"], "tokens.json")

    def test_reward_fingerprint_omits_verbose_gate_detail(self) -> None:
        fingerprint = reward_fingerprint(
            {
                "score": 0.82,
                "alpha": 1.0,
                "tau": 0.5,
                "delta": 0.1,
                "stage_cap": 0.82,
                "first_failed_gate": "cpu_logits",
                "gates": {
                    "cpu_logits": {
                        "passed": False,
                        "score": 0.0,
                        "log": "/tmp/large.log",
                    }
                },
            }
        )

        self.assertEqual(
            fingerprint["gates"]["cpu_logits"],
            {"passed": False, "score": 0.0},
        )

    def test_refiner_defaults_to_autonomous_codex_command(self) -> None:
        args = build_parser().parse_args(["Provider/Model"])
        cfg = config_from_args(args)

        self.assertEqual(cfg.refinement_command, DEFAULT_REFINER_COMMAND)

    def test_no_refiner_disables_autonomous_loop(self) -> None:
        args = build_parser().parse_args(["--no-refiner", "Provider/Model"])
        cfg = config_from_args(args)

        self.assertIsNone(cfg.refinement_command)

    def test_default_has_no_iteration_cap(self) -> None:
        args = build_parser().parse_args(["Provider/Model"])
        cfg = config_from_args(args)

        self.assertIsNone(cfg.max_iterations)

    def test_explicit_iteration_cap_is_preserved(self) -> None:
        args = build_parser().parse_args(["--max-iterations", "3", "Provider/Model"])
        cfg = config_from_args(args)

        self.assertEqual(cfg.max_iterations, 3)

    def test_create_worktree_failure_state_does_not_create_target(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "ingest-provider-model"
            args = build_parser().parse_args(
                [
                    "--create-worktree",
                    "--worktree-root",
                    str(root),
                    "--worktree",
                    str(target),
                    "Provider/Model",
                ]
            )
            cfg = config_from_args(args)

            write_failure_state(cfg, IngestError("boom"))

            self.assertFalse(target.exists())
            self.assertTrue((bootstrap_dir(cfg) / "state.json").exists())


if __name__ == "__main__":
    unittest.main()

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

            self.assertEqual(state["status"], "initialized")
            self.assertTrue((run_dir / "model_spec.json").exists())
            self.assertTrue((run_dir / "state.json").exists())
            self.assertTrue((run_dir / "reward.json").exists())
            self.assertTrue((run_dir / "reward.txt").exists())
            self.assertTrue((run_dir / "handoff.md").exists())
            spec = json.loads((run_dir / "model_spec.json").read_text())
            self.assertEqual(spec["safe_model"], "provider-model")
            self.assertEqual(spec["explicit_gates"]["model_spec"]["passed"], True)


if __name__ == "__main__":
    unittest.main()

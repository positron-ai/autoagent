from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ares_ingest_autoagent.commands import (
    build_command_wrapper_plan,
    command_gates_from_plan,
    full_inference_smoke_wrapper,
    rinzler_chat_wrapper,
    side_by_side_comparison_wrapper,
)


class AresIngestCommandWrapperTest(unittest.TestCase):
    def test_rinzler_chat_wrapper_builds_dry_run_command(self) -> None:
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            spec = {
                "model": "synthetic/model",
                "backend": "cpu",
                "weights": "/weights/synthetic",
                "ares_plan": "artifacts/ares-plan.json",
                "target_plan": "artifacts/cpu.target-plan.json",
                "prompt": "Hello",
            }

            wrapper = rinzler_chat_wrapper(spec, run_dir=run_dir)

            self.assertTrue(wrapper.enabled)
            self.assertIn("bin/ares-rinzler-chat", wrapper.command)
            self.assertIn("--cpu", wrapper.command)
            self.assertIn("--dry-run", wrapper.command)
            self.assertIn("--dense-logits-jsonl", wrapper.command)
            self.assertEqual(wrapper.evidence_class, "system_under_test")
            self.assertFalse(wrapper.promotion_eligible)

    def test_backend_wrapper_requires_generated_artifact_inputs(self) -> None:
        with TemporaryDirectory() as tmp:
            wrapper = rinzler_chat_wrapper(
                {"model": "synthetic/model", "backend": "tron"},
                run_dir=Path(tmp),
            )

            self.assertFalse(wrapper.enabled)
            self.assertEqual(
                wrapper.missing_inputs,
                ("weights", "ares_plan", "target_plan"),
            )

    def test_full_inference_wrapper_uses_smoke_script_env(self) -> None:
        with TemporaryDirectory() as tmp:
            wrapper = full_inference_smoke_wrapper(
                {
                    "model": "synthetic/model",
                    "backend": "tron",
                    "weights": "/weights/synthetic",
                    "ares_plan": "artifacts/ares-plan.json",
                    "target_plan": "artifacts/tron.target-plan.json",
                },
                run_dir=Path(tmp),
            )

            self.assertTrue(wrapper.enabled)
            self.assertIn(
                "bin/ci/ci-ares-rinzler-full-inference-smoke.sh",
                wrapper.command,
            )
            self.assertIn("ARES_RINZLER_FULL_INFERENCE_DRY_RUN=1", wrapper.command)
            self.assertIn("ARES_RINZLER_FULL_INFERENCE_ARES_PLAN", wrapper.command)

    def test_side_by_side_wrapper_classifies_cpp_as_comparison(self) -> None:
        with TemporaryDirectory() as tmp:
            wrapper = side_by_side_comparison_wrapper(
                {
                    "model": "synthetic/model",
                    "backend": "tron",
                    "weights": "/weights/synthetic",
                    "comparison": {
                        "cpp_rinzler_bin": "/tron/gen/rinzler",
                        "rust_model_path": "/weights/synthetic-with-plans",
                    },
                },
                run_dir=Path(tmp) / "run",
                ares_repo=Path(tmp) / "ares",
            )

            self.assertTrue(wrapper.enabled)
            self.assertEqual(wrapper.evidence_class, "comparison")
            self.assertIn(
                "bin/ci/ci-rinzler-fpga-vs-tron-comparison.sh",
                wrapper.command,
            )
            self.assertIn("ARES_RINZLER_COMPARE_REQUIRE_TVD=1", wrapper.command)
            self.assertIn("CPP_RINZLER_BIN=/tron/gen/rinzler", wrapper.command)

    def test_plan_materializes_command_gates_only_when_explicitly_enabled(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            spec = {
                "model": "synthetic/model",
                "backend": "tron",
                "weights": "/weights/synthetic",
                "ares_plan": "artifacts/ares-plan.json",
                "target_plan": "artifacts/tron.target-plan.json",
                "required_gates": ["model_spec", "backend_open"],
            }

            plan = build_command_wrapper_plan(
                spec,
                run_dir=Path(tmp) / "run",
                ares_repo=Path(tmp) / "ares",
            )
            self.assertEqual(plan["command_gates"], [])

            string_false_plan = build_command_wrapper_plan(
                {
                    **spec,
                    "command_wrapper_config": {"dry_run": "false"},
                    "execute_command_wrappers": "false",
                },
                run_dir=Path(tmp) / "run",
                ares_repo=Path(tmp) / "ares",
            )
            self.assertFalse(string_false_plan["dry_run"])
            self.assertEqual(string_false_plan["command_gates"], [])

            enabled_plan = build_command_wrapper_plan(
                {**spec, "execute_command_wrappers": True},
                run_dir=Path(tmp) / "run",
                ares_repo=Path(tmp) / "ares",
            )

            self.assertGreaterEqual(len(enabled_plan["wrappers"]), 2)
            self.assertGreaterEqual(len(enabled_plan["command_gates"]), 2)
            self.assertEqual(
                command_gates_from_plan(enabled_plan),
                enabled_plan["command_gates"],
            )


if __name__ == "__main__":
    unittest.main()

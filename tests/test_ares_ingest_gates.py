from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from ares_ingest_autoagent.gates import shortcut_scan_gate


class AresIngestGateTest(unittest.TestCase):
    def test_shortcut_scan_accepts_plain_generated_artifact_fixtures(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = root / "backend/tron/tests/data/tiny.tron.target_plan.json"
            fixture.parent.mkdir(parents=True)
            fixture.write_text("{}\n")

            gate = shortcut_scan_gate(root)

            self.assertTrue(gate["passed"])
            self.assertEqual(gate["artifact_validator"], "shortcut_scan")

    def test_shortcut_scan_rejects_runtime_generated_targetplan(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sidecar = (
                root
                / "runtime/runares/artifacts/model.runtime_generated.target_plan.json"
            )
            sidecar.parent.mkdir(parents=True)
            sidecar.write_text("{}\n")

            gate = shortcut_scan_gate(root)

            self.assertFalse(gate["passed"])
            hits = gate["detail"]["forbidden_hits"]
            self.assertEqual(hits[0]["kind"], "runtime_generated_plan_sidecar")
            self.assertEqual(
                hits[0]["path"],
                "runtime/runares/artifacts/model.runtime_generated.target_plan.json",
            )

    def test_shortcut_scan_rejects_hand_authored_rust_model_plugin(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            plugin = root / "backend/tron/model_plugins/llama.rs"
            plugin.parent.mkdir(parents=True)
            plugin.write_text("pub fn forward() {}\n")

            gate = shortcut_scan_gate(root)

            self.assertFalse(gate["passed"])
            hits = gate["detail"]["forbidden_hits"]
            self.assertEqual(hits[0]["kind"], "hand_authored_rust_model_plugin")
            self.assertEqual(hits[0]["path"], "backend/tron/model_plugins/llama.rs")


if __name__ == "__main__":
    unittest.main()

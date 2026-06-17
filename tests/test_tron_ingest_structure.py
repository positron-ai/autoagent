from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tron_ingest_autoagent.structure import analyze_patterns, main


class TronIngestStructureTest(unittest.TestCase):
    def test_detects_expected_patterns_from_generated_python(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            generated = root / "model_bulk.py"
            generated.write_text(
                "\n".join(
                    [
                        "from tron_ingest_runtime import apply_tron_rope",
                        "from tron_ingest_runtime import apply_tron_sdpa",
                        "from tron_ingest_runtime import rms_norm, swish_mul",
                        "x = apply_tron_rope(x, inv_freq, 'chunked')",
                        "y = apply_tron_sdpa(scale, q, k, v)",
                        "z = rms_norm(y, weight, 1e-6)",
                        "w = swish_mul(a, b)",
                    ]
                )
            )

            result = analyze_patterns(
                roots=[root],
                expected_patterns=["tron_sdpa", "tron_rope", "rms_norm", "swishmul"],
            )

            self.assertTrue(result["passed"])
            self.assertEqual(result["score"], 1.0)
            self.assertEqual(result["missing_patterns"], [])
            self.assertGreater(result["counts"]["tron_sdpa"], 0)
            self.assertGreater(result["counts"]["tron_rope"], 0)

    def test_reports_missing_patterns(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "model_bulk.py").write_text("x = rms_norm(x, weight, 1e-6)")

            result = analyze_patterns(
                roots=[root],
                expected_patterns=["tron_sdpa", "rms_norm"],
            )

            self.assertFalse(result["passed"])
            self.assertEqual(result["score"], 0.5)
            self.assertEqual(result["missing_patterns"], ["tron_sdpa"])

    def test_infers_expected_patterns_from_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "metadata.json").write_text(
                json.dumps(
                    {
                        "parameters": [
                            {"target": "model.layers.0.self_attn.q_proj.weight"},
                            {"target": "model.layers.0.self_attn.k_proj.weight"},
                            {"target": "model.layers.0.self_attn.v_proj.weight"},
                            {"target": "model.layers.0.input_layernorm.weight"},
                            {"target": "model.layers.0.mlp.gate_proj.weight"},
                            {"target": "model.layers.0.mlp.up_proj.weight"},
                        ]
                    }
                )
            )
            (root / "generated.py").write_text(
                "apply_tron_sdpa(q,k,v); apply_tron_rope(x); rms_norm(x,w); swish_mul(a,b)"
            )

            result = analyze_patterns(roots=[root])

            self.assertTrue(result["passed"])
            self.assertEqual(
                result["expected_patterns"],
                ["rms_norm", "swishmul", "tron_rope", "tron_sdpa"],
            )

    def test_cli_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "generated.py").write_text("apply_tron_sdpa(q,k,v)")
            out = root / "structure.json"

            rc = main(
                [
                    "--artifact-root",
                    str(root),
                    "--expected-pattern",
                    "tron_sdpa",
                    "--output-json",
                    str(out),
                ]
            )

            self.assertEqual(rc, 0)
            written = json.loads(out.read_text())
            self.assertTrue(written["passed"])
            self.assertEqual(written["score"], 1.0)


if __name__ == "__main__":
    unittest.main()

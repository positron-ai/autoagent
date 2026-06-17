from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tron_ingest_autoagent.architecture import analyze_architecture, main


def write_artifacts(root: Path) -> None:
    (root / "fx_export").mkdir()
    (root / "fx_export/metadata.json").write_text(
        json.dumps(
            {
                "config": {
                    "num_key_value_heads": 4,
                    "num_hidden_layers": 2,
                    "vocab_size": 32000,
                    "hidden_size": 16,
                    "intermediate_size": 64,
                    "rope_theta": 10000.0,
                    "num_attention_heads": 4,
                    "head_dim": 4,
                    "max_position_embeddings": 2048,
                },
                "parameters": [
                    {"target": "model.layers.0.self_attn.q_proj.weight"},
                    {"target": "model.layers.0.self_attn.k_proj.weight"},
                    {"target": "model.layers.0.self_attn.v_proj.weight"},
                    {"target": "model.layers.0.input_layernorm.weight"},
                    {"target": "model.layers.0.mlp.gate_proj.weight"},
                    {"target": "model.layers.0.mlp.up_proj.weight"},
                    {"target": "model.layers.0.mlp.down_proj.weight"},
                ],
            }
        )
    )
    (root / "generated.py").write_text(
        "self.rope_layout = 'chunked'\n"
        "x = rms_norm(x, weight, 1e-6)\n"
        "y = swish_mul(a, b)\n"
    )


class TronIngestArchitectureTest(unittest.TestCase):
    def test_scores_consistent_architecture(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_artifacts(root)

            result = analyze_architecture(root)

            self.assertTrue(result["passed"])
            self.assertEqual(result["score"], 1.0)
            architecture = result["architecture"]
            self.assertEqual(architecture["hidden_size"], 16)
            self.assertEqual(architecture["attention_type"], "MHA")
            self.assertEqual(architecture["rope_layout"], "chunked")
            self.assertEqual(architecture["normalization"], "RMSNorm")
            self.assertEqual(architecture["ffn_activation"], "SwiGLU/SiLU")

    def test_expected_field_mismatch_reduces_score(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_artifacts(root)

            result = analyze_architecture(root, expected={"hidden_size": 32})

            self.assertFalse(result["passed"])
            self.assertLess(result["score"], 1.0)
            self.assertFalse(result["checks"]["expected_hidden_size"]["passed"])

    def test_cli_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_artifacts(root)
            out = root / "architecture.json"

            rc = main(["--artifact-root", str(root), "--output-json", str(out)])

            self.assertEqual(rc, 0)
            written = json.loads(out.read_text())
            self.assertTrue(written["passed"])
            self.assertEqual(written["architecture"]["num_hidden_layers"], 2)


if __name__ == "__main__":
    unittest.main()

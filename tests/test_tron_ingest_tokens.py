from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tron_ingest_autoagent.tokens import compare_payloads, main


class TronIngestTokensTest(unittest.TestCase):
    def test_exact_token_match_scores_one(self) -> None:
        result = compare_payloads([1, 2, 3], [1, 2, 3])

        self.assertEqual(result["tau"], 1.0)
        self.assertEqual(result["cases"][0]["exact_fraction"], 1.0)
        self.assertTrue(result["cases"][0]["exact_match"])

    def test_partial_token_match_scores_components(self) -> None:
        result = compare_payloads([1, 2, 3, 4], [1, 2, 9, 4])
        case = result["cases"][0]

        self.assertFalse(case["exact_match"])
        self.assertEqual(case["exact_fraction"], 0.75)
        self.assertEqual(case["prefix_match"], 0.5)
        self.assertGreater(case["edit_similarity"], 0.0)
        self.assertLess(result["tau"], 1.0)

    def test_case_payloads_match_by_name(self) -> None:
        reference = {
            "cases": [
                {"name": "a", "tokens": [1, 2]},
                {"name": "b", "tokens": [3, 4]},
            ]
        }
        candidate = {
            "cases": [
                {"name": "b", "tokens": [3, 9]},
                {"name": "a", "tokens": [1, 2]},
            ]
        }

        result = compare_payloads(reference, candidate)

        self.assertEqual(len(result["cases"]), 2)
        self.assertLess(result["tau"], 1.0)
        self.assertGreater(result["tau"], 0.0)

    def test_cli_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ref = root / "ref.json"
            cand = root / "cand.json"
            out = root / "tokens.json"
            ref.write_text(json.dumps([1, 2, 3]))
            cand.write_text(json.dumps([1, 2, 4]))

            rc = main(["--reference", str(ref), "--candidate", str(cand), "--output-json", str(out)])

            self.assertEqual(rc, 0)
            written = json.loads(out.read_text())
            self.assertIn("tau", written)
            self.assertLess(written["tau"], 1.0)


if __name__ == "__main__":
    unittest.main()

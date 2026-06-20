from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tron_ingest_autoagent.score import compute_reward, main


def logit_payload(*, functionally_equivalent: bool, max_tvd: float = 0.0) -> dict:
    return {
        "model": "test-model",
        "tvd_threshold": 0.01,
        "all_functionally_equivalent": functionally_equivalent,
        "all_strictly_passed": functionally_equivalent,
        "results": [
            {
                "name": "short_hello",
                "passed": functionally_equivalent,
                "functionally_equivalent": functionally_equivalent,
                "max_abs_error": 0.0 if functionally_equivalent else 1.0,
                "mean_abs_error": 0.0 if functionally_equivalent else 0.5,
                "max_tvd": max_tvd,
                "mean_tvd": max_tvd,
                "top1_agreement": 1.0 if functionally_equivalent else 0.25,
                "top5_agreement": 1.0 if functionally_equivalent else 0.50,
                "cosine_similarity": 1.0 if functionally_equivalent else 0.2,
            }
        ],
    }


class TronIngestScoreTest(unittest.TestCase):
    def test_perfect_required_gates_reaches_one(self) -> None:
        reward = compute_reward(
            gates_payload={
                "gates": {
                    "architecture": True,
                    "eqsat_structure": True,
                    "cpp_compile": True,
                    "cpu_logits": True,
                    "fpga_logits": True,
                }
            },
            typedfx_payload=logit_payload(functionally_equivalent=True),
            bulk_payload=logit_payload(functionally_equivalent=True),
            token_payload={"score": 1.0},
            performance_payload={
                "measured_tokens_per_second": 100.0,
                "speed_of_light_tokens_per_second": 100.0,
            },
        )

        self.assertEqual(reward["first_failed_gate"], "complete")
        self.assertEqual(reward["stage_cap"], 1.0)
        self.assertEqual(reward["alpha"], 1.0)
        self.assertEqual(reward["tau"], 1.0)
        self.assertEqual(reward["delta"], 1.0)
        self.assertAlmostEqual(reward["score"], 1.0)

    def test_near_one_floating_scores_snap_to_one(self) -> None:
        payload = logit_payload(functionally_equivalent=True)
        payload["results"][0]["cosine_similarity"] = 0.9999999999996

        reward = compute_reward(
            gates_payload={
                "gates": {
                    "architecture": True,
                    "eqsat_structure": True,
                    "cpp_compile": True,
                    "cpu_logits": True,
                    "fpga_logits": True,
                }
            },
            typedfx_payload=payload,
            bulk_payload=payload,
            token_payload={"score": 1.0},
            performance_payload={"delta": 1.0},
        )

        self.assertEqual(reward["first_failed_gate"], "complete")
        self.assertEqual(reward["alpha"], 1.0)
        self.assertEqual(reward["score"], 1.0)

    def test_prefix_cache_reuse_cannot_score_delta(self) -> None:
        reward = compute_reward(
            gates_payload={
                "gates": {
                    "architecture": True,
                    "eqsat_structure": True,
                    "cpp_compile": True,
                    "cpu_logits": True,
                    "fpga_logits": True,
                }
            },
            typedfx_payload=logit_payload(functionally_equivalent=True),
            bulk_payload=logit_payload(functionally_equivalent=True),
            token_payload={"score": 1.0},
            performance_payload={
                "workload": "prefix_cache_reuse",
                "delta": 1.0,
            },
        )

        self.assertEqual(reward["delta"], 0.0)
        self.assertAlmostEqual(reward["raw_score"], 0.9)

    def test_semantic_gate_caps_fast_but_wrong_run(self) -> None:
        reward = compute_reward(
            gates_payload={
                "gates": {
                    "architecture": True,
                    "eqsat_structure": True,
                    "cpp_compile": True,
                    "cpu_logits": True,
                    "fpga_logits": True,
                }
            },
            typedfx_payload=logit_payload(functionally_equivalent=False, max_tvd=0.8),
            bulk_payload=logit_payload(functionally_equivalent=True),
            token_payload={"score": 1.0},
            performance_payload={"delta": 1.0},
        )

        self.assertEqual(reward["first_failed_gate"], "typedfx_logits")
        self.assertEqual(reward["stage_cap"], 0.35)
        self.assertLessEqual(reward["score"], 0.35)

    def test_bulk_logit_file_infers_prior_gates(self) -> None:
        reward = compute_reward(
            bulk_payload=logit_payload(functionally_equivalent=True),
            required_gates=("fx_export", "typedfx_parse", "bulk_logits"),
        )

        self.assertEqual(reward["first_failed_gate"], "complete")
        self.assertTrue(reward["gates"]["fx_export"]["passed"])
        self.assertTrue(reward["gates"]["typedfx_parse"]["passed"])
        self.assertTrue(reward["gates"]["bulk_logits"]["passed"])

    def test_structure_payload_sets_eqsat_gate(self) -> None:
        reward = compute_reward(
            gates_payload={"gates": {"architecture": True}},
            structure_payload={
                "passed": True,
                "score": 1.0,
                "expected_patterns": ["tron_sdpa"],
                "missing_patterns": [],
            },
            typedfx_payload=logit_payload(functionally_equivalent=True),
            bulk_payload=logit_payload(functionally_equivalent=True),
            required_gates=(
                "fx_export",
                "typedfx_parse",
                "typedfx_logits",
                "eqsat_structure",
                "bulk_logits",
            ),
        )

        self.assertEqual(reward["first_failed_gate"], "complete")
        self.assertTrue(reward["gates"]["eqsat_structure"]["passed"])

    def test_architecture_payload_sets_architecture_gate(self) -> None:
        reward = compute_reward(
            architecture_payload={
                "passed": True,
                "score": 1.0,
                "architecture": {"hidden_size": 16},
            },
            required_gates=(),
        )

        self.assertTrue(reward["gates"]["architecture"]["passed"])
        self.assertEqual(reward["alpha_components"]["architecture"], 1.0)

    def test_no_artifacts_scores_zero(self) -> None:
        reward = compute_reward()

        self.assertEqual(reward["first_failed_gate"], "not_started")
        self.assertEqual(reward["stage_cap"], 0.0)
        self.assertEqual(reward["score"], 0.0)

    def test_cli_writes_harbor_reward_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            typedfx = root / "logit_results.json"
            bulk = root / "bulk_logit_results.json"
            gates = root / "gates.json"
            tokens = root / "tokens.json"
            perf = root / "performance.json"
            out_json = root / "reward.json"
            out_txt = root / "reward.txt"

            typedfx.write_text(json.dumps(logit_payload(functionally_equivalent=True)))
            bulk.write_text(json.dumps(logit_payload(functionally_equivalent=True)))
            gates.write_text(
                json.dumps(
                    {
                        "gates": {
                            "architecture": True,
                            "eqsat_structure": True,
                            "cpp_compile": True,
                            "cpu_logits": True,
                            "fpga_logits": True,
                        }
                    }
                )
            )
            tokens.write_text(json.dumps({"score": 1.0}))
            perf.write_text(
                json.dumps(
                    {
                        "measured_tokens_per_second": 90.0,
                        "speed_of_light_tokens_per_second": 100.0,
                    }
                )
            )

            rc = main(
                [
                    "--gates",
                    str(gates),
                    "--typedfx-logits",
                    str(typedfx),
                    "--bulk-logits",
                    str(bulk),
                    "--tokens",
                    str(tokens),
                    "--performance",
                    str(perf),
                    "--output-json",
                    str(out_json),
                    "--output-txt",
                    str(out_txt),
                ]
            )

            self.assertEqual(rc, 0)
            self.assertTrue(out_json.exists())
            self.assertTrue(out_txt.exists())
            written = json.loads(out_json.read_text())
            self.assertAlmostEqual(written["delta"], 0.9)
            self.assertEqual(out_txt.read_text().strip(), f"{written['score']:.12g}")


if __name__ == "__main__":
    unittest.main()

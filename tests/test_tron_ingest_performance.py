from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tron_ingest_autoagent.performance import (
    extract_limit,
    extract_measured,
    main,
    score_performance,
)


class TronIngestPerformanceTest(unittest.TestCase):
    def test_scores_delta_from_numbers(self) -> None:
        result = score_performance(75.0, 100.0)

        self.assertEqual(result["delta"], 0.75)
        self.assertEqual(result["score"], 0.75)
        self.assertFalse(result["passed"])

    def test_caps_delta_at_one(self) -> None:
        result = score_performance(125.0, 100.0)

        self.assertEqual(result["delta"], 1.0)
        self.assertTrue(result["passed"])

    def test_rejects_prefix_cache_reuse_for_delta(self) -> None:
        result = score_performance(
            900.0,
            300.0,
            measured_workload="prefix_cache_reuse",
        )

        self.assertEqual(result["delta"], 0.0)
        self.assertFalse(result["passed"])
        self.assertIn("diagnostic-only", result["error"])

    def test_extracts_named_measured_workload(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "measured.json"
            path.write_text(
                json.dumps(
                    {
                        "workloads": [
                            {
                                "workload": "prefix_cache_reuse",
                                "measured_tokens_per_second": 900.0,
                            },
                            {
                                "workload": "independent_decode",
                                "measured_tokens_per_second": 75.0,
                            },
                        ]
                    }
                )
            )

            measured, detail = extract_measured(
                path,
                workload="independent_decode",
            )

            self.assertEqual(measured, 75.0)
            self.assertEqual(detail["workload"], "independent_decode")

    def test_missing_required_measured_workload_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "measured.json"
            path.write_text(
                json.dumps(
                    {
                        "workloads": [
                            {
                                "workload": "prefix_cache_reuse",
                                "measured_tokens_per_second": 900.0,
                            }
                        ]
                    }
                )
            )

            measured, detail = extract_measured(
                path,
                workload="independent_decode",
            )
            result = score_performance(
                measured,
                300.0,
                measured_workload=detail.get("workload"),
                required_workload="independent_decode",
            )

            self.assertIsNone(measured)
            self.assertEqual(detail["available_workloads"], ["prefix_cache_reuse"])
            self.assertIn("was not found", detail["error"])
            self.assertEqual(result["delta"], 0.0)
            self.assertFalse(result["passed"])

    def test_extracts_named_speed_target(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "speed.json"
            path.write_text(
                json.dumps(
                    {
                        "targets": {
                            "independent_decode": {
                                "speed_of_light_tokens_per_second": 300.0,
                            },
                            "long_prefill": {
                                "speed_of_light_tokens_per_second": 1000.0,
                            },
                        }
                    }
                )
            )

            limit, detail = extract_limit(path, workload="long_prefill")

            self.assertEqual(limit, 1000.0)
            self.assertEqual(detail["workload"], "long_prefill")

    def test_missing_required_speed_target_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "speed.json"
            path.write_text(
                json.dumps(
                    {
                        "targets": {
                            "prefix_cache_reuse": {
                                "speed_of_light_tokens_per_second": 3000.0,
                            }
                        }
                    }
                )
            )

            limit, detail = extract_limit(path, workload="independent_decode")
            result = score_performance(
                900.0,
                limit,
                required_workload="independent_decode",
            )

            self.assertIsNone(limit)
            self.assertEqual(detail["available_workloads"], ["prefix_cache_reuse"])
            self.assertIn("was not found", detail["error"])
            self.assertEqual(result["delta"], 0.0)
            self.assertFalse(result["passed"])

    def test_extracts_throughput_from_tron_log(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "runtron.log"
            path.write_text(
                "Generating 128 response tokens with 1920 context took "
                "0.836916 s at 152.942 average tok/s\n"
                "Throughput 149.250 new and 200.000 all tok/s.\n"
            )

            measured, detail = extract_measured(path)

            self.assertEqual(measured, 149.250)
            self.assertEqual(detail["method"], "log_regex_preferred")

    def test_prefers_generated_throughput_over_prompt_parse_rate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "runtron.log"
            path.write_text(
                "Parsing the prompt took 0.004 s at 1899.210 tokens/s\n"
                "Generating 1 response tokens with 7 context took "
                "0.001 s at 796.712 average tok/s\n"
                "Throughput 836.575 new and 836.575 all tok/s.\n"
            )

            measured, detail = extract_measured(path)

            self.assertEqual(measured, 836.575)
            self.assertEqual(detail["values"], [836.575])

    def test_cli_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            measured = root / "measured.log"
            limit = root / "speed.json"
            out = root / "performance.json"
            measured.write_text("Throughput 80.0 tok/s.\n")
            limit.write_text(json.dumps({"speed_of_light_tokens_per_second": 100.0}))

            rc = main(
                [
                    "--measured",
                    str(measured),
                    "--speed-of-light",
                    str(limit),
                    "--workload",
                    "independent_decode",
                    "--output-json",
                    str(out),
                ]
            )

            self.assertEqual(rc, 0)
            written = json.loads(out.read_text())
            self.assertEqual(written["delta"], 0.8)
            self.assertEqual(written["measured_tokens_per_second"], 80.0)


if __name__ == "__main__":
    unittest.main()

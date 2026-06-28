from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ares_ingest_autoagent.score import compute_reward, main


def oracle_record(
    kind: str = "hf_cpu_oracle_capture",
    oracle: str = "huggingface_transformers_pytorch_cpu",
) -> dict:
    return {
        "record_kind": kind,
        "source": {"oracle": oracle},
        "prompt_token_ids": [1, 2, 3],
        "generated_token_ids": [4],
    }


class AresIngestScoreTest(unittest.TestCase):
    def test_perfect_required_gates_reaches_one(self) -> None:
        reward = compute_reward(
            gates_payload={
                "gates": {
                    "model_spec": True,
                    "frontend_export": True,
                    "lean_ingest": True,
                    "aresplan_valid": True,
                    "targetplan_valid": True,
                    "backend_open": True,
                    "one_token_logits": True,
                    "eight_token_greedy": True,
                    "cpp_tvd": True,
                    "depth_performance": True,
                }
            },
            oracle_payload=oracle_record(),
            token_payload={"score": 1.0},
            performance_payload={
                "workload": "independent_decode",
                "measured_tokens_per_second": 100.0,
                "speed_of_light_tokens_per_second": 100.0,
            },
        )

        self.assertEqual(reward["first_failed_gate"], "complete")
        self.assertEqual(reward["stage_cap"], 1.0)
        self.assertEqual(reward["score"], 1.0)
        self.assertEqual(reward["alpha_execution"], 1.0)
        self.assertEqual(reward["tau_tokens"], 1.0)
        self.assertEqual(reward["delta_inference"], 1.0)

    def test_missing_targetplan_caps_fast_token_match(self) -> None:
        reward = compute_reward(
            gates_payload={
                "gates": {
                    "model_spec": True,
                    "frontend_export": True,
                    "lean_ingest": True,
                    "aresplan_valid": True,
                }
            },
            oracle_payload=oracle_record(),
            token_payload={"score": 1.0},
            performance_payload={"delta": 1.0},
        )

        self.assertEqual(reward["first_failed_gate"], "targetplan_valid")
        self.assertEqual(reward["stage_cap"], 0.48)
        self.assertLessEqual(reward["score"], 0.48)

    def test_mock_oracle_cannot_satisfy_oracle_gate(self) -> None:
        reward = compute_reward(
            gates_payload={"gates": {"model_spec": True}},
            oracle_payload=oracle_record(
                kind="mock_fixture",
                oracle="mock_fixture_not_oracle",
            ),
            required_gates=("model_spec", "hf_cpu_oracle"),
        )

        self.assertEqual(reward["first_failed_gate"], "hf_cpu_oracle")
        self.assertFalse(reward["gates"]["hf_cpu_oracle"]["passed"])

    def test_cli_writes_reward_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gates = root / "gates.json"
            oracle = root / "oracle.jsonl"
            out_json = root / "reward.json"
            out_txt = root / "reward.txt"
            gates.write_text(json.dumps({"gates": {"model_spec": True}}))
            oracle.write_text(json.dumps(oracle_record()) + "\n")

            rc = main(
                [
                    "--gates",
                    str(gates),
                    "--oracle",
                    str(oracle),
                    "--required-gates",
                    "model_spec",
                    "hf_cpu_oracle",
                    "--output-json",
                    str(out_json),
                    "--output-txt",
                    str(out_txt),
                ]
            )

            self.assertEqual(rc, 0)
            reward = json.loads(out_json.read_text())
            self.assertEqual(reward["first_failed_gate"], "complete")
            self.assertEqual(out_txt.read_text().strip(), f"{reward['score']:.12g}")


if __name__ == "__main__":
    unittest.main()

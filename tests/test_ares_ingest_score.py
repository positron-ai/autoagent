from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import ares_ingest_autoagent.artifacts as artifacts_module
from ares_ingest_autoagent.artifacts import (
    HfCpuOracleAuthority,
    build_greedy_token_evidence,
    load_hf_cpu_oracle_authority_file,
    validate_hf_cpu_oracle_evidence_files,
    validate_hf_cpu_oracle_record,
)
from ares_ingest_autoagent.score import (
    BACKEND_GATES,
    COMPARISON_GATES,
    CPU_ONLY_GATES,
    FULL_GATES,
    GATE_PROFILES,
    STAGE_CAPS,
    compute_reward,
    main,
)


ARES_ROOT = Path(__file__).resolve().parents[3]
ORACLE_SCRIPT = ARES_ROOT / "tools/oracles/hf-cpu/capture_hf_cpu_oracle.py"
ORACLE_SPEC = importlib.util.spec_from_file_location(
    "test_ares_ingest_score_hf_cpu_oracle",
    ORACLE_SCRIPT,
)
assert ORACLE_SPEC is not None and ORACLE_SPEC.loader is not None
ORACLE_MODULE = importlib.util.module_from_spec(ORACLE_SPEC)
ORACLE_SPEC.loader.exec_module(ORACLE_MODULE)


def artifact_gate(validator: str) -> dict:
    return {"passed": True, "score": 1.0, "artifact_validator": validator}


def valid_ares_plan() -> dict:
    return {
        "schema_version": 2,
        "config": {},
        "weights": [],
        "buffers": [],
        "stmts": [{"stmt": "matmul"}],
        "provenance": {
            "fx_hash": "test-fx",
            "rule_corpus_hash": "test-rules",
            "emitter_version": "ingest-lean test",
            "hf_export_model_type": "synthetic",
        },
    }


def valid_target_plan() -> dict:
    return {
        "schema_version": 1,
        "producer": {
            "language": "lean",
            "tool": "ingest-lean",
            "module": "Ingest.TargetPlan",
            "version": "0.1.0",
        },
        "backend_id": "tron",
        "model_id": "synthetic/model",
        "source": {
            "schema_version": 2,
            "statement_count": 1,
            "config": {},
            "provenance": valid_ares_plan()["provenance"],
        },
        "declared_runtime_bindings": ["input_ids"],
        "hw_policy": {},
        "operations": [
            {
                "id": "runtime.input_ids",
                "role": "runtime_binding",
                "action": "runtime_binding",
                "source": {"type": "runtime_binding", "name": "input_ids"},
                "requirements": {},
            },
            {
                "id": "stmt.00000.matmul",
                "role": "semantic",
                "action": "matmul",
                "source": {
                    "type": "ares_plan_statement",
                    "statement_index": 0,
                    "statement_kind": "matmul",
                    "statement_name": "w",
                },
                "requirements": {},
            },
        ],
    }


def token_evidence(length: int = 8, *, exact: bool = True) -> dict:
    return {
        "schema": "ares.runtime.greedy_token_agreement.v1",
        "evidence_class": "system_under_test",
        "oracle": "huggingface_transformers_pytorch_cpu",
        "candidate": "ares",
        "decode_strategy": "greedy",
        "expected_generated_tokens": 8,
        "generated_tokens": length,
        "reference_generated_token_ids": list(range(length)),
        "candidate_generated_token_ids": list(range(length))
        if exact
        else [99] * length,
        "score": 1.0 if exact else 0.0,
        "exact_match": exact,
        "exact_fraction": 1.0 if exact else 0.875,
        "top1_agreement": 1.0 if exact else 0.875,
        "reference": {"path": "reference.json", "sha256": "a" * 64},
        "candidate_output": {
            "path": "candidate.json",
            "sha256": "b" * 64,
            "runtime": "ares",
        },
        "cases": [
            {
                "name": "default",
                "exact_match": exact,
                "candidate_length": length,
            }
        ],
    }


def oracle_record(
    kind: str = "hf_cpu_oracle_capture",
    oracle: str = "huggingface_transformers_pytorch_cpu",
) -> dict:
    record = {
        "schema": ORACLE_MODULE.SCHEMA_ID,
        "record_kind": "hf_cpu_oracle_capture",
        "invocation_id": ORACLE_MODULE.new_capture_invocation_id(),
        "invocation_record_count": 1,
        "capture_id": "",
        "created_utc": "2026-06-26T00:00:00Z",
        "source": {
            "oracle": "huggingface_transformers_pytorch_cpu",
            "capture_script": "tools/oracles/hf-cpu/capture_hf_cpu_oracle.py",
            "capture_script_sha256": ORACLE_MODULE.capture_source_sha256(),
        },
        "model": {
            "model_id": "synthetic/model",
            "requested_revision": "0123456789abcdef0123456789abcdef01234567",
            "resolved_revision": "0123456789abcdef0123456789abcdef01234567",
            "name_or_path": "synthetic/model",
            "dtype": "float32",
            "config": {
                "architectures": ["SyntheticForCausalLM"],
                "model_type": "synthetic",
                "torch_dtype": "float32",
                "vocab_size": 6,
            },
        },
        "tokenizer": {
            "tokenizer_id": "synthetic/model",
            "requested_revision": "0123456789abcdef0123456789abcdef01234567",
            "resolved_revision": "0123456789abcdef0123456789abcdef01234567",
            "name_or_path": "synthetic/model",
            "vocab_size": 6,
            "model_max_length": 1024,
            "padding_side": "right",
            "truncation_side": "right",
            "eos_token_id": 2,
            "bos_token_id": 1,
            "chat_template_sha256": None,
        },
        "run": {
            "seed": 0,
            "decode_strategy": "greedy",
            "max_new_tokens": 2,
            "top_k": 2,
            "torch_deterministic_algorithms": True,
            "local_files_only": True,
            "trust_remote_code": False,
            "stop_on_eos": True,
            "ignore_eos": False,
            "effective_eos_token_ids": [2],
        },
        "prompt": {
            "kind": "raw",
            "text": "Hello",
            "token_ids": [1, 0],
            "token_count": 2,
            "add_special_tokens": True,
            "capture_index": 0,
        },
        "artifacts": {
            "dense_logits_mode": "paired_promotion",
            "expected_dense_row_count": 2,
        },
        "generation": {
            "generated_token_ids": [3, 2],
            "generated_token_count": 2,
            "generated_text": " world</s>",
            "finish_reason": "eos_token",
            "eos_token_id": 2,
            "eos_token_ids": [2],
            "stop_token_id": 2,
        },
        "logit_slices": [
            {
                "step": 0,
                "position": 1,
                "context_token_count": 2,
                "selected_token_id": 3,
                "selected_token_text": " world",
                "selected_token_logit": 12.5,
                "dense_logits_sha256": "",
                "top_k": [
                    {"rank": 1, "token_id": 3, "token_text": " world", "logit": 12.5},
                    {"rank": 2, "token_id": 4, "token_text": " there", "logit": 8.0},
                ],
            },
            {
                "step": 1,
                "position": 2,
                "context_token_count": 3,
                "selected_token_id": 2,
                "selected_token_text": "</s>",
                "selected_token_logit": 9.25,
                "dense_logits_sha256": "",
                "top_k": [
                    {"rank": 1, "token_id": 2, "token_text": "</s>", "logit": 9.25},
                    {"rank": 2, "token_id": 5, "token_text": "!", "logit": 4.0},
                ],
            },
        ],
        "environment": {
            "python_version": "3.12.13",
            "platform": "test-platform",
            "torch_version": "test-torch",
            "transformers_version": "test-transformers",
            "torch_device": "cpu",
            "torch_num_threads": 1,
            "torch_num_interop_threads": 1,
            "git_commit": "a" * 40,
            "git_dirty": False,
        },
    }
    record["capture_id"] = ORACLE_MODULE.semantic_capture_id_for_record(record)
    dense_rows = oracle_dense_rows(record)
    for logit_slice, dense_row in zip(record["logit_slices"], dense_rows):
        logit_slice["dense_logits_sha256"] = ORACLE_MODULE.sha256_json(dense_row)
    record["record_kind"] = kind
    record["source"]["oracle"] = oracle
    return record


def oracle_dense_rows(record: dict | None = None) -> list[dict]:
    if record is None:
        record = oracle_record()
    capture_id = record["capture_id"]
    return [
        {
            "capture_id": capture_id,
            "invocation_id": record["invocation_id"],
            "step_index": 0,
            "context_tokens": [1, 0],
            "context_tokens_role": "hf_oracle_replay_context",
            "context_count": 2,
            "new_count": 2,
            "runtime_request_token_count": 2,
            "context_prefix_token_count": 0,
            "last_token": 0,
            "logits": [0.0, -1.0, -2.0, 12.5, 8.0, -3.0],
        },
        {
            "capture_id": capture_id,
            "invocation_id": record["invocation_id"],
            "step_index": 1,
            "context_tokens": [1, 0, 3],
            "context_tokens_role": "hf_oracle_replay_context",
            "context_count": 3,
            "new_count": 1,
            "runtime_request_token_count": 1,
            "context_prefix_token_count": 2,
            "last_token": 3,
            "logits": [0.0, -1.0, 9.25, 1.0, 0.0, 4.0],
        },
    ]


def publish_oracle_transaction(
    root: Path,
    record: dict | None = None,
    dense_rows: list[dict] | None = None,
) -> tuple[Path, Path]:
    record = oracle_record() if record is None else record
    dense_rows = oracle_dense_rows(record) if dense_rows is None else dense_rows
    oracle_payload = ORACLE_MODULE.canonical_json_line(record)
    dense_payload = b"".join(
        ORACLE_MODULE.canonical_json_line(row) for row in dense_rows
    )
    oracle_stage_path = root / "oracle-stage.jsonl"
    dense_stage_path = root / "dense-stage.jsonl"
    oracle_stage_path.write_bytes(oracle_payload)
    dense_stage_path.write_bytes(dense_payload)
    oracle_stage = ORACLE_MODULE.verify_staged_artifact(
        oracle_stage_path,
        expected_size=len(oracle_payload),
        expected_sha256=hashlib.sha256(oracle_payload).hexdigest(),
    )
    dense_stage = ORACLE_MODULE.verify_staged_artifact(
        dense_stage_path,
        expected_size=len(dense_payload),
        expected_sha256=hashlib.sha256(dense_payload).hexdigest(),
    )
    oracle_path = root / "oracle.jsonl"
    dense_path = root / "dense.jsonl"
    transaction = ORACLE_MODULE.publish_capture_transaction(
        oracle_stage=oracle_stage,
        oracle_output=oracle_path,
        dense_stage=dense_stage,
        dense_output=dense_path,
    )
    assert transaction is not None
    return oracle_path, dense_path


def operator_authority_document(
    oracle_path: Path,
    dense_path: Path,
) -> dict[str, object]:
    transaction = ORACLE_MODULE.validate_capture_transaction(oracle_path, dense_path)
    return {
        "schema": artifacts_module.HF_CPU_ORACLE_AUTHORITY_SCHEMA,
        "transaction_schema": transaction.schema,
        "transaction_digest": transaction.digest,
        "canonical_validator_sha256": ORACLE_MODULE.capture_source_sha256(),
        "canonical_schema_sha256": artifacts_module.HF_CPU_ORACLE_SCHEMA_SHA256,
        "oracle_sha256": transaction.oracle_sha256,
        "oracle_size_bytes": transaction.oracle_size_bytes,
        "oracle_record_count": len(transaction.oracle_records),
        "dense_sha256": transaction.dense_sha256,
        "dense_size_bytes": transaction.dense_size_bytes,
        "dense_row_count": len(transaction.dense_rows),
    }


def load_operator_authority(
    oracle_path: Path,
    dense_path: Path,
) -> HfCpuOracleAuthority:
    authority_path = oracle_path.parent / "operator-oracle-authority.json"
    authority_path.write_text(
        json.dumps(operator_authority_document(oracle_path, dense_path)) + "\n"
    )
    return load_hf_cpu_oracle_authority_file(authority_path)


class AresIngestScoreTest(unittest.TestCase):
    def retained_oracle_files(
        self,
        record: dict | None = None,
        dense_rows: list[dict] | None = None,
    ) -> tuple[Path, Path]:
        record = oracle_record() if record is None else record
        dense_rows = oracle_dense_rows(record) if dense_rows is None else dense_rows
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        return publish_oracle_transaction(root, record, dense_rows)

    def compute_reward_with_oracle(self, **kwargs: object) -> dict:
        oracle_path, oracle_dense_path = self.retained_oracle_files()
        validation = validate_hf_cpu_oracle_evidence_files(
            oracle_path,
            oracle_dense_path,
        )
        self.assertTrue(validation.passed, validation.errors)
        if kwargs.get("promotion_authority") is True:
            kwargs.setdefault(
                "expected_oracle_authority",
                load_operator_authority(oracle_path, oracle_dense_path),
            )
        validated_payload = kwargs.get("validated_gates_payload")
        if isinstance(validated_payload, dict):
            gates = validated_payload.get("gates")
            consistency = gates.get("artifact_consistency") if isinstance(gates, dict) else None
            if isinstance(consistency, dict):
                consistency.setdefault("detail", {})[
                    "oracle_transaction_digest"
                ] = validation.detail["transaction_digest"]
        if kwargs.get("promotion_authority") is True:
            root = oracle_path.parent
            ares_plan_path = root / "ares-plan.json"
            target_plan_path = root / "target-plan.json"
            ares_plan_path.write_text(json.dumps(valid_ares_plan()))
            target_plan_path.write_text(json.dumps(valid_target_plan()))
            kwargs.setdefault("artifact_spec", {"model": "synthetic/model"})
            kwargs.setdefault("ares_plan_path", ares_plan_path)
            kwargs.setdefault("target_plan_path", target_plan_path)
            kwargs.setdefault("authority_root", root)
        return compute_reward(
            oracle_path=oracle_path,
            oracle_dense_path=oracle_dense_path,
            **kwargs,
        )

    def test_oracle_record_uses_canonical_raw_validator(self) -> None:
        validation = validate_hf_cpu_oracle_record(oracle_record())

        self.assertTrue(validation.passed, validation.errors)
        self.assertEqual(validation.detail["model_type"], "synthetic")
        self.assertEqual(
            validation.detail["validation_scope"],
            "raw_record_shape_only",
        )
        self.assertEqual(
            validation.detail["claim_ceiling"],
            "diagnostic_only_nonpromotion",
        )
        self.assertFalse(validation.detail["promotion_eligible"])

    def test_canonical_validator_source_is_pinned_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            marker = root / "executed"
            forged_validator = root / "capture_hf_cpu_oracle.py"
            forged_validator.write_text(
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('executed')\n"
            )
            artifacts_module._canonical_hf_cpu_oracle_module.cache_clear()
            try:
                with mock.patch.object(
                    artifacts_module,
                    "HF_CPU_ORACLE_VALIDATOR",
                    forged_validator,
                ):
                    validation = validate_hf_cpu_oracle_record(oracle_record())
                executed = marker.exists()
            finally:
                artifacts_module._canonical_hf_cpu_oracle_module.cache_clear()

        self.assertFalse(validation.passed)
        self.assertIn("frozen source SHA-256", " ".join(validation.errors))
        self.assertFalse(executed)

    def test_canonical_record_schema_is_pinned_before_validator_execution(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            forged_schema = Path(td) / "hf_cpu_oracle_record.schema.json"
            forged_schema.write_text("{}\n")
            artifacts_module._canonical_hf_cpu_oracle_module.cache_clear()
            try:
                with mock.patch.object(
                    artifacts_module,
                    "HF_CPU_ORACLE_SCHEMA",
                    forged_schema,
                ):
                    validation = validate_hf_cpu_oracle_record(oracle_record())
            finally:
                artifacts_module._canonical_hf_cpu_oracle_module.cache_clear()

        self.assertFalse(validation.passed)
        self.assertIn("frozen schema SHA-256", " ".join(validation.errors))

    def test_oracle_record_rejects_effective_raw_mutations(self) -> None:
        cases = {
            "capture_id": lambda record: record.__setitem__(
                "capture_id", "hf-cpu-" + "0" * 64
            ),
            "exporter_hash": lambda record: record["source"].__setitem__(
                "capture_script_sha256", "0" * 64
            ),
            "missing_model_type": lambda record: record["model"]["config"].pop(
                "model_type"
            ),
            "unknown_model_type": lambda record: record["model"]["config"].__setitem__(
                "model_type", "unknown"
            ),
            "non_cpu": lambda record: record["environment"].__setitem__(
                "torch_device", "cuda"
            ),
            "floating_revision": lambda record: record["model"].__setitem__(
                "requested_revision", "main"
            ),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name):
                record = oracle_record()
                mutate(record)

                validation = validate_hf_cpu_oracle_record(record)

                self.assertFalse(validation.passed)

    def test_oracle_record_rejects_foreign_capture_script_path(self) -> None:
        record = oracle_record()
        record["source"]["capture_script"] = (
            "third_party/autoagent/ares_ingest_autoagent/score.py"
        )
        record["capture_id"] = ORACLE_MODULE.semantic_capture_id_for_record(record)

        validation = validate_hf_cpu_oracle_record(record)

        self.assertFalse(validation.passed)
        self.assertIn("resolve to the canonical producer", " ".join(validation.errors))

    def test_committed_oracle_transaction_satisfies_live_validator(self) -> None:
        oracle_path, dense_path = self.retained_oracle_files()

        diagnostic_validation = validate_hf_cpu_oracle_evidence_files(
            oracle_path,
            dense_path,
        )
        self.assertTrue(diagnostic_validation.passed, diagnostic_validation.errors)
        authority = load_operator_authority(oracle_path, dense_path)
        validation = validate_hf_cpu_oracle_evidence_files(
            oracle_path,
            dense_path,
            expected_authority=authority,
            require_authority=True,
        )

        self.assertTrue(validation.passed, validation.errors)
        self.assertTrue(validation.detail["promotion_eligible"])
        self.assertTrue(validation.detail["oracle_authority_matched"])
        self.assertEqual(
            validation.detail["transaction_commit_path"],
            str(oracle_path.resolve()),
        )
        self.assertEqual(
            validation.detail["canonical_validator_sha256"],
            ORACLE_MODULE.capture_source_sha256(),
        )
        self.assertEqual(
            validation.detail["canonical_schema_sha256"],
            artifacts_module.HF_CPU_ORACLE_SCHEMA_SHA256,
        )

    def test_live_validator_rejects_dense_orphan_with_stale_v1_marker(self) -> None:
        oracle_path, dense_path = self.retained_oracle_files()
        oracle_path.unlink()
        (oracle_path.parent / ".ares-hf-cpu-transaction-v1-forged").mkdir()

        validation = validate_hf_cpu_oracle_evidence_files(
            oracle_path,
            dense_path,
        )

        self.assertFalse(validation.passed)
        self.assertFalse(validation.detail["promotion_eligible"])

    def test_live_validator_rejects_coherent_post_commit_tamper(self) -> None:
        oracle_path, dense_path = self.retained_oracle_files()
        original_validation = validate_hf_cpu_oracle_evidence_files(
            oracle_path,
            dense_path,
        )
        self.assertTrue(original_validation.passed, original_validation.errors)
        authority = load_operator_authority(oracle_path, dense_path)
        record = oracle_record()
        dense_rows = oracle_dense_rows(record)
        dense_rows[0]["logits"][0] = -0.25
        record["logit_slices"][0]["dense_logits_sha256"] = ORACLE_MODULE.sha256_json(
            dense_rows[0]
        )
        oracle_path.chmod(0o600)
        dense_path.chmod(0o600)
        oracle_path.write_bytes(ORACLE_MODULE.canonical_json_line(record))
        dense_path.write_bytes(
            b"".join(ORACLE_MODULE.canonical_json_line(row) for row in dense_rows)
        )
        oracle_path.chmod(ORACLE_MODULE.CAPTURE_OUTPUT_MODE)
        dense_path.chmod(ORACLE_MODULE.CAPTURE_OUTPUT_MODE)

        validation = validate_hf_cpu_oracle_evidence_files(
            oracle_path,
            dense_path,
            expected_authority=authority,
            require_authority=True,
        )

        self.assertFalse(validation.passed)
        self.assertFalse(validation.detail["promotion_eligible"])
        self.assertIn("does not match", " ".join(validation.errors))

    def test_oracle_authority_file_is_strict_and_exact(self) -> None:
        oracle_path, dense_path = self.retained_oracle_files()
        validation = validate_hf_cpu_oracle_evidence_files(oracle_path, dense_path)
        self.assertTrue(validation.passed, validation.errors)
        authority = operator_authority_document(oracle_path, dense_path)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            authority_path = root / "authority.json"
            authority_path.write_text(json.dumps(authority) + "\n")
            self.assertEqual(
                load_hf_cpu_oracle_authority_file(authority_path).value,
                authority,
            )

            invalid_payloads = {
                "duplicate_key": (
                    '{"schema":"duplicate",'
                    + json.dumps(authority, separators=(",", ":"))[1:]
                ),
                "nonfinite": json.dumps(
                    {**authority, "oracle_size_bytes": float("nan")}
                ),
                "unexpected_field": json.dumps({**authority, "extra": True}),
            }
            for name, payload in invalid_payloads.items():
                with self.subTest(name=name):
                    authority_path.write_text(payload + "\n")
                    with self.assertRaises(ValueError):
                        load_hf_cpu_oracle_authority_file(authority_path)

    def test_in_memory_authority_mapping_cannot_self_authorize(self) -> None:
        oracle_path, dense_path = self.retained_oracle_files()
        authority = operator_authority_document(oracle_path, dense_path)

        validation = validate_hf_cpu_oracle_evidence_files(
            oracle_path,
            dense_path,
            expected_authority=authority,  # type: ignore[arg-type]
            require_authority=True,
        )

        self.assertFalse(validation.passed)
        self.assertIn("trusted file loader", " ".join(validation.errors))

    def test_promotion_profile_rejects_missing_operator_authority(self) -> None:
        oracle_path, dense_path = self.retained_oracle_files()
        missing_authority = validate_hf_cpu_oracle_evidence_files(
            oracle_path,
            dense_path,
            require_authority=True,
        )
        self.assertFalse(missing_authority.passed)
        self.assertIn("authority is required", " ".join(missing_authority.errors))
        root = oracle_path.parent
        ares_plan_path = root / "ares-plan.json"
        target_plan_path = root / "target-plan.json"
        ares_plan_path.write_text(json.dumps(valid_ares_plan()))
        target_plan_path.write_text(json.dumps(valid_target_plan()))

        reward = compute_reward(
            gates_payload={
                "gates": {
                    "model_spec": True,
                    "frontend_export": True,
                    "lean_ingest": True,
                }
            },
            oracle_path=oracle_path,
            oracle_dense_path=dense_path,
            artifact_spec={"model": "synthetic/model"},
            ares_plan_path=ares_plan_path,
            target_plan_path=target_plan_path,
            authority_root=root,
            required_gates=CPU_ONLY_GATES,
            promotion_authority=True,
        )

        self.assertEqual(reward["first_failed_gate"], "hf_cpu_oracle")
        self.assertFalse(reward["gates"]["hf_cpu_oracle"]["passed"])
        self.assertFalse(reward["promotion_eligible"])

    def test_live_validator_rejects_raw_dense_capture_mismatch(self) -> None:
        oracle_path, dense_path = self.retained_oracle_files()
        dense_rows = oracle_dense_rows()
        dense_rows[0]["capture_id"] = "hf-cpu-" + "0" * 64
        dense_path.chmod(0o600)
        dense_path.write_bytes(
            b"".join(ORACLE_MODULE.canonical_json_line(row) for row in dense_rows)
        )
        dense_path.chmod(ORACLE_MODULE.CAPTURE_OUTPUT_MODE)

        validation = validate_hf_cpu_oracle_evidence_files(
            oracle_path,
            dense_path,
        )

        self.assertFalse(validation.passed)
        self.assertFalse(validation.detail["promotion_eligible"])

    def test_reward_rejects_consistency_bound_to_another_valid_transaction(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            transaction_a_root = root / "a"
            transaction_b_root = root / "b"
            transaction_a_root.mkdir()
            transaction_b_root.mkdir()
            oracle_a, dense_a = publish_oracle_transaction(transaction_a_root)
            record_b = oracle_record()
            record_b["created_utc"] = "2026-06-26T00:00:01Z"
            oracle_b, dense_b = publish_oracle_transaction(
                transaction_b_root,
                record_b,
            )
            validation_a = validate_hf_cpu_oracle_evidence_files(oracle_a, dense_a)
            self.assertTrue(validation_a.passed, validation_a.errors)

            reward = compute_reward(
                gates_payload={"gates": {"model_spec": True}},
                validated_gates_payload={
                    "gates": {
                        "artifact_consistency": {
                            "passed": True,
                            "score": 1.0,
                            "artifact_validator": "artifact_consistency",
                            "detail": {
                                "oracle_transaction_digest": validation_a.detail[
                                    "transaction_digest"
                                ]
                            },
                        }
                    }
                },
                oracle_path=oracle_b,
                oracle_dense_path=dense_b,
                required_gates=(
                    "model_spec",
                    "hf_cpu_oracle",
                    "artifact_consistency",
                ),
            )

        self.assertEqual(reward["first_failed_gate"], "artifact_consistency")
        self.assertFalse(reward["gates"]["artifact_consistency"]["passed"])

    def test_reward_rejects_consistency_without_transaction_digest(self) -> None:
        oracle_path, dense_path = self.retained_oracle_files()

        reward = compute_reward(
            gates_payload={"gates": {"model_spec": True}},
            validated_gates_payload={
                "gates": {
                    "artifact_consistency": artifact_gate(
                        "artifact_consistency"
                    )
                }
            },
            oracle_path=oracle_path,
            oracle_dense_path=dense_path,
            required_gates=(
                "model_spec",
                "hf_cpu_oracle",
                "artifact_consistency",
            ),
        )

        self.assertEqual(reward["first_failed_gate"], "artifact_consistency")
        self.assertFalse(reward["gates"]["artifact_consistency"]["passed"])

    def test_live_validator_rejects_foreign_capture_script_transaction(self) -> None:
        record = oracle_record()
        record["source"]["capture_script"] = (
            "third_party/autoagent/ares_ingest_autoagent/score.py"
        )
        record["capture_id"] = ORACLE_MODULE.semantic_capture_id_for_record(record)
        dense_rows = oracle_dense_rows(record)
        for logit_slice, dense_row in zip(record["logit_slices"], dense_rows):
            logit_slice["dense_logits_sha256"] = ORACLE_MODULE.sha256_json(dense_row)
        oracle_path, dense_path = self.retained_oracle_files(record, dense_rows)

        validation = validate_hf_cpu_oracle_evidence_files(
            oracle_path,
            dense_path,
        )

        self.assertFalse(validation.passed)
        self.assertIn("canonical producer", " ".join(validation.errors))

    def test_in_memory_oracle_rows_are_never_live_promotion_evidence(self) -> None:
        record = oracle_record()
        reward = compute_reward(
            gates_payload={"gates": {"model_spec": True}},
            oracle_payload=record,
            oracle_dense_payload=oracle_dense_rows(record),
            required_gates=("model_spec", "hf_cpu_oracle"),
        )

        self.assertEqual(reward["first_failed_gate"], "hf_cpu_oracle")
        self.assertFalse(reward["gates"]["hf_cpu_oracle"]["passed"])

    def test_custom_gate_subset_without_oracle_is_nonpromotion_diagnostic(self) -> None:
        reward = compute_reward(
            gates_payload={"gates": {"model_spec": True}},
            token_payload={"score": 1.0},
            performance_payload={
                "workload": "independent_decode",
                "measured_tokens_per_second": 100.0,
                "speed_of_light_tokens_per_second": 100.0,
            },
            required_gates=("model_spec",),
        )

        self.assertEqual(reward["first_failed_gate"], "diagnostic_complete")
        self.assertLess(reward["score"], 1.0)
        self.assertEqual(reward["required_gates_authority"], "custom_diagnostic")
        self.assertEqual(reward["claim_ceiling"], "diagnostic_only_nonpromotion")
        self.assertFalse(reward["promotion_eligible"])

    def test_missing_dense_file_cannot_satisfy_oracle_gate(self) -> None:
        oracle_path, _dense_path = self.retained_oracle_files()
        reward = compute_reward(
            gates_payload={"gates": {"model_spec": True}},
            oracle_path=oracle_path,
            required_gates=("model_spec", "hf_cpu_oracle"),
        )

        self.assertEqual(reward["first_failed_gate"], "hf_cpu_oracle")
        self.assertFalse(reward["gates"]["hf_cpu_oracle"]["passed"])

    def test_perfect_required_gates_reaches_one(self) -> None:
        reward = self.compute_reward_with_oracle(
            gates_payload={
                "gates": {
                    "model_spec": True,
                    "frontend_export": True,
                    "lean_ingest": True,
                }
            },
            validated_gates_payload={
                "gates": {
                    "aresplan_valid": artifact_gate("ares_plan"),
                    "targetplan_valid": artifact_gate("target_plan"),
                    "artifact_consistency": artifact_gate("artifact_consistency"),
                    "shortcut_scan": artifact_gate("shortcut_scan"),
                    "backend_open": artifact_gate("backend_open"),
                    "one_token_logits": artifact_gate("one_token_logits"),
                    "cpp_tvd": artifact_gate("cpp_tvd"),
                    "depth_performance": artifact_gate("depth_performance"),
                }
            },
            token_payload=token_evidence(),
            performance_payload={
                "workload": "independent_decode",
                "measured_tokens_per_second": 100.0,
                "speed_of_light_tokens_per_second": 100.0,
            },
            promotion_authority=True,
        )

        self.assertEqual(reward["first_failed_gate"], "complete")
        self.assertTrue(reward["promotion_eligible"])
        self.assertEqual(reward["claim_ceiling"], "promotion_candidate")
        self.assertEqual(reward["stage_cap"], 1.0)
        self.assertEqual(reward["score"], 1.0)
        self.assertEqual(reward["alpha_execution"], 1.0)
        self.assertEqual(reward["tau_tokens"], 1.0)
        self.assertEqual(reward["delta_inference"], 1.0)

    def test_default_required_gates_do_not_require_cpp_or_hardware(self) -> None:
        reward = self.compute_reward_with_oracle(
            gates_payload={
                "gates": {
                    "model_spec": True,
                    "frontend_export": True,
                    "lean_ingest": True,
                }
            },
            validated_gates_payload={
                "gates": {
                    "aresplan_valid": artifact_gate("ares_plan"),
                    "targetplan_valid": artifact_gate("target_plan"),
                    "artifact_consistency": artifact_gate("artifact_consistency"),
                    "shortcut_scan": artifact_gate("shortcut_scan"),
                }
            },
            token_payload={"score": 1.0},
            performance_payload={
                "workload": "independent_decode",
                "measured_tokens_per_second": 100.0,
                "speed_of_light_tokens_per_second": 100.0,
            },
            promotion_authority=True,
        )

        for gate in CPU_ONLY_GATES:
            self.assertIn(gate, reward["gates"])
        self.assertEqual(reward["first_failed_gate"], "complete")
        self.assertNotIn("cpp_tvd", reward["gates"])

    def test_full_profile_requires_mmlu_pro_after_depth_performance(self) -> None:
        reward = self.compute_reward_with_oracle(
            gates_payload={
                "gates": {
                    "model_spec": True,
                    "frontend_export": True,
                    "lean_ingest": True,
                }
            },
            validated_gates_payload={
                "gates": {
                    "aresplan_valid": artifact_gate("ares_plan"),
                    "targetplan_valid": artifact_gate("target_plan"),
                    "artifact_consistency": artifact_gate("artifact_consistency"),
                    "shortcut_scan": artifact_gate("shortcut_scan"),
                    "backend_open": artifact_gate("backend_open"),
                    "one_token_logits": artifact_gate("one_token_logits"),
                    "eight_token_greedy": artifact_gate("eight_token_greedy"),
                    "depth_performance": artifact_gate("depth_performance"),
                }
            },
            token_payload=token_evidence(),
            performance_payload={
                "workload": "independent_decode",
                "measured_tokens_per_second": 100.0,
                "speed_of_light_tokens_per_second": 100.0,
            },
            required_gates=FULL_GATES,
        )

        self.assertEqual(reward["first_failed_gate"], "mmlu_pro")
        self.assertEqual(reward["stage_cap"], 0.98)
        self.assertLess(reward["score"], 1.0)

    def test_full_profile_does_not_require_cpp_comparison(self) -> None:
        self.assertIn("depth_performance", FULL_GATES)
        self.assertNotIn("cpp_tvd", FULL_GATES)
        self.assertIn("cpp_tvd", COMPARISON_GATES)
        self.assertLess(
            COMPARISON_GATES.index("depth_performance"),
            COMPARISON_GATES.index("cpp_tvd"),
        )

    def test_profile_stage_caps_are_monotonic(self) -> None:
        for profile, gates in GATE_PROFILES.items():
            with self.subTest(profile=profile):
                caps = [STAGE_CAPS[gate] for gate in gates]
                self.assertEqual(caps, sorted(caps))

    def test_missing_targetplan_caps_fast_token_match(self) -> None:
        reward = self.compute_reward_with_oracle(
            gates_payload={
                "gates": {
                    "model_spec": True,
                    "frontend_export": True,
                    "lean_ingest": True,
                }
            },
            validated_gates_payload={
                "gates": {"aresplan_valid": artifact_gate("ares_plan")}
            },
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

    def test_incomplete_oracle_cannot_satisfy_oracle_gate(self) -> None:
        reward = compute_reward(
            gates_payload={"gates": {"model_spec": True}},
            oracle_payload={
                "record_kind": "hf_cpu_oracle_capture",
                "source": {"oracle": "huggingface_transformers_pytorch_cpu"},
            },
            required_gates=("model_spec", "hf_cpu_oracle"),
        )

        self.assertEqual(reward["first_failed_gate"], "hf_cpu_oracle")
        self.assertFalse(reward["gates"]["hf_cpu_oracle"]["passed"])

    def test_explicit_artifact_gate_cannot_replace_oracle_payload(self) -> None:
        reward = compute_reward(
            gates_payload={"gates": {"model_spec": True, "hf_cpu_oracle": True}},
            required_gates=("model_spec", "hf_cpu_oracle"),
        )

        self.assertEqual(reward["first_failed_gate"], "hf_cpu_oracle")
        self.assertFalse(reward["gates"]["hf_cpu_oracle"]["passed"])

    def test_named_validated_gate_cannot_replace_oracle_transaction(self) -> None:
        reward = compute_reward(
            gates_payload={"gates": {"model_spec": True}},
            validated_gates_payload={
                "gates": {"hf_cpu_oracle": artifact_gate("hf_cpu_oracle")}
            },
            required_gates=("model_spec", "hf_cpu_oracle"),
        )

        self.assertEqual(reward["first_failed_gate"], "hf_cpu_oracle")
        self.assertFalse(reward["gates"]["hf_cpu_oracle"]["passed"])

    def test_explicit_artifact_gate_cannot_replace_plan_validators(self) -> None:
        reward = self.compute_reward_with_oracle(
            gates_payload={
                "gates": {
                    "model_spec": True,
                    "frontend_export": True,
                    "lean_ingest": True,
                    "aresplan_valid": True,
                    "targetplan_valid": True,
                }
            },
            required_gates=(
                "model_spec",
                "hf_cpu_oracle",
                "frontend_export",
                "lean_ingest",
                "aresplan_valid",
                "targetplan_valid",
            ),
        )

        self.assertEqual(reward["first_failed_gate"], "aresplan_valid")
        self.assertFalse(reward["gates"]["aresplan_valid"]["passed"])

    def test_explicit_artifact_gate_cannot_replace_shortcut_scan(self) -> None:
        reward = self.compute_reward_with_oracle(
            gates_payload={
                "gates": {
                    "model_spec": True,
                    "frontend_export": True,
                    "lean_ingest": True,
                    "shortcut_scan": True,
                }
            },
            validated_gates_payload={
                "gates": {
                    "aresplan_valid": artifact_gate("ares_plan"),
                    "targetplan_valid": artifact_gate("target_plan"),
                    "artifact_consistency": artifact_gate("artifact_consistency"),
                }
            },
            required_gates=(
                "model_spec",
                "hf_cpu_oracle",
                "frontend_export",
                "lean_ingest",
                "aresplan_valid",
                "targetplan_valid",
                "artifact_consistency",
                "shortcut_scan",
            ),
        )

        self.assertEqual(reward["first_failed_gate"], "shortcut_scan")
        self.assertFalse(reward["gates"]["shortcut_scan"]["passed"])

    def test_explicit_artifact_gate_cannot_replace_consistency_validator(
        self,
    ) -> None:
        reward = self.compute_reward_with_oracle(
            gates_payload={
                "gates": {
                    "model_spec": True,
                    "frontend_export": True,
                    "lean_ingest": True,
                    "artifact_consistency": True,
                }
            },
            validated_gates_payload={
                "gates": {
                    "aresplan_valid": artifact_gate("ares_plan"),
                    "targetplan_valid": artifact_gate("target_plan"),
                }
            },
            required_gates=(
                "model_spec",
                "hf_cpu_oracle",
                "frontend_export",
                "lean_ingest",
                "aresplan_valid",
                "targetplan_valid",
                "artifact_consistency",
            ),
        )

        self.assertEqual(reward["first_failed_gate"], "artifact_consistency")
        self.assertFalse(reward["gates"]["artifact_consistency"]["passed"])

    def test_explicit_runtime_gates_cannot_replace_artifact_validators(self) -> None:
        reward = self.compute_reward_with_oracle(
            gates_payload={
                "gates": {
                    "model_spec": True,
                    "frontend_export": True,
                    "lean_ingest": True,
                    "backend_open": artifact_gate("backend_open"),
                    "one_token_logits": artifact_gate("one_token_logits"),
                    "eight_token_greedy": True,
                    "cpp_tvd": artifact_gate("cpp_tvd"),
                    "depth_performance": artifact_gate("depth_performance"),
                }
            },
            validated_gates_payload={
                "gates": {
                    "aresplan_valid": artifact_gate("ares_plan"),
                    "targetplan_valid": artifact_gate("target_plan"),
                    "artifact_consistency": artifact_gate("artifact_consistency"),
                    "shortcut_scan": artifact_gate("shortcut_scan"),
                }
            },
            token_payload=token_evidence(),
            performance_payload={
                "workload": "independent_decode",
                "measured_tokens_per_second": 100.0,
                "speed_of_light_tokens_per_second": 100.0,
            },
            required_gates=(
                "model_spec",
                "hf_cpu_oracle",
                "frontend_export",
                "lean_ingest",
                "aresplan_valid",
                "targetplan_valid",
                "artifact_consistency",
                "shortcut_scan",
                "backend_open",
                "one_token_logits",
                "eight_token_greedy",
                "cpp_tvd",
                "depth_performance",
            ),
        )

        self.assertEqual(reward["first_failed_gate"], "backend_open")
        self.assertFalse(reward["gates"]["backend_open"]["passed"])

    def test_explicit_mmlu_pro_gate_cannot_replace_validator(self) -> None:
        reward = self.compute_reward_with_oracle(
            gates_payload={
                "gates": {
                    "model_spec": True,
                    "frontend_export": True,
                    "lean_ingest": True,
                    "mmlu_pro": True,
                }
            },
            validated_gates_payload={
                "gates": {
                    "aresplan_valid": artifact_gate("ares_plan"),
                    "targetplan_valid": artifact_gate("target_plan"),
                    "artifact_consistency": artifact_gate("artifact_consistency"),
                    "shortcut_scan": artifact_gate("shortcut_scan"),
                    "backend_open": artifact_gate("backend_open"),
                    "one_token_logits": artifact_gate("one_token_logits"),
                    "eight_token_greedy": artifact_gate("eight_token_greedy"),
                    "cpp_tvd": artifact_gate("cpp_tvd"),
                    "depth_performance": artifact_gate("depth_performance"),
                }
            },
            token_payload=token_evidence(),
            performance_payload={
                "workload": "independent_decode",
                "measured_tokens_per_second": 100.0,
                "speed_of_light_tokens_per_second": 100.0,
            },
            required_gates=FULL_GATES,
        )

        self.assertEqual(reward["first_failed_gate"], "mmlu_pro")
        self.assertFalse(reward["gates"]["mmlu_pro"]["passed"])

    def test_explicit_eight_token_gate_cannot_replace_token_evidence(self) -> None:
        reward = self.compute_reward_with_oracle(
            gates_payload={
                "gates": {
                    "model_spec": True,
                    "frontend_export": True,
                    "lean_ingest": True,
                    "eight_token_greedy": True,
                }
            },
            validated_gates_payload={
                "gates": {
                    "aresplan_valid": artifact_gate("ares_plan"),
                    "targetplan_valid": artifact_gate("target_plan"),
                    "artifact_consistency": artifact_gate("artifact_consistency"),
                    "shortcut_scan": artifact_gate("shortcut_scan"),
                    "backend_open": artifact_gate("backend_open"),
                    "one_token_logits": artifact_gate("one_token_logits"),
                }
            },
            required_gates=(
                "model_spec",
                "hf_cpu_oracle",
                "frontend_export",
                "lean_ingest",
                "aresplan_valid",
                "targetplan_valid",
                "artifact_consistency",
                "shortcut_scan",
                "backend_open",
                "one_token_logits",
                "eight_token_greedy",
            ),
        )

        self.assertEqual(reward["first_failed_gate"], "eight_token_greedy")
        self.assertFalse(reward["gates"]["eight_token_greedy"]["passed"])

    def test_cli_writes_reward_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gates = root / "gates.json"
            out_json = root / "reward.json"
            out_txt = root / "reward.txt"
            gates.write_text(json.dumps({"gates": {"model_spec": True}}))
            oracle, dense = publish_oracle_transaction(root)

            rc = main(
                [
                    "--gates",
                    str(gates),
                    "--oracle",
                    str(oracle),
                    "--oracle-dense",
                    str(dense),
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
            self.assertEqual(reward["first_failed_gate"], "diagnostic_complete")
            self.assertEqual(
                reward["claim_ceiling"],
                "diagnostic_only_nonpromotion",
            )
            self.assertFalse(reward["promotion_eligible"])
            self.assertEqual(out_txt.read_text().strip(), f"{reward['score']:.12g}")

    def test_cli_rejects_stale_token_source_digest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gates = root / "gates.json"
            validated_gates = root / "validated-gates.json"
            tokens = root / "tokens.json"
            reference = root / "reference.json"
            candidate = root / "candidate.json"
            out_json = root / "reward.json"
            out_txt = root / "reward.txt"
            reference_payload = {"generated_token_ids": list(range(8))}
            candidate_payload = {"generated_token_ids": list(range(8))}

            gates.write_text(
                json.dumps(
                    {
                        "gates": {
                            "model_spec": True,
                            "frontend_export": True,
                            "lean_ingest": True,
                        }
                    }
                )
            )
            validated_gates.write_text(
                json.dumps(
                    {
                        "gates": {
                            "aresplan_valid": artifact_gate("ares_plan"),
                            "targetplan_valid": artifact_gate("target_plan"),
                            "artifact_consistency": artifact_gate(
                                "artifact_consistency"
                            ),
                            "shortcut_scan": artifact_gate("shortcut_scan"),
                            "backend_open": artifact_gate("backend_open"),
                            "one_token_logits": artifact_gate("one_token_logits"),
                        }
                    }
                )
            )
            oracle, dense = publish_oracle_transaction(root)
            oracle_validation = validate_hf_cpu_oracle_evidence_files(oracle, dense)
            self.assertTrue(oracle_validation.passed, oracle_validation.errors)
            validated_payload = json.loads(validated_gates.read_text())
            validated_payload["gates"]["artifact_consistency"]["detail"] = {
                "oracle_transaction_digest": oracle_validation.detail[
                    "transaction_digest"
                ]
            }
            validated_gates.write_text(json.dumps(validated_payload))
            reference.write_text(json.dumps(reference_payload))
            candidate.write_text(json.dumps(candidate_payload))
            tokens.write_text(
                json.dumps(
                    build_greedy_token_evidence(
                        {
                            "score": 1.0,
                            "exact_match": True,
                            "exact_fraction": 1.0,
                            "top1_agreement": 1.0,
                            "cases": [
                                {
                                    "name": "default",
                                    "exact_match": True,
                                    "candidate_length": 8,
                                }
                            ],
                        },
                        reference=reference,
                        candidate=candidate,
                        reference_payload=reference_payload,
                        candidate_payload=candidate_payload,
                    )
                )
            )
            candidate.write_text(json.dumps({"generated_token_ids": [7] * 8}))

            rc = main(
                [
                    "--gates",
                    str(gates),
                    "--validated-gates",
                    str(validated_gates),
                    "--oracle",
                    str(oracle),
                    "--oracle-dense",
                    str(dense),
                    "--tokens",
                    str(tokens),
                    "--required-gates",
                    *BACKEND_GATES,
                    "--output-json",
                    str(out_json),
                    "--output-txt",
                    str(out_txt),
                ]
            )

            self.assertEqual(rc, 0)
            reward = json.loads(out_json.read_text())
            self.assertEqual(reward["first_failed_gate"], "eight_token_greedy")
            self.assertFalse(reward["gates"]["eight_token_greedy"]["passed"])
            self.assertEqual(reward["tau_tokens"], 0.0)

    def test_cli_validated_gate_receipt_is_diagnostic_only(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gates_path = root / "gates.json"
            validated_gates_path = root / "validated-gates.json"
            output_json = root / "reward.json"
            output_txt = root / "reward.txt"
            oracle_path, dense_path = publish_oracle_transaction(root)
            validation = validate_hf_cpu_oracle_evidence_files(
                oracle_path,
                dense_path,
            )
            self.assertTrue(validation.passed, validation.errors)
            gates_path.write_text(
                json.dumps(
                    {
                        "gates": {
                            "model_spec": True,
                            "frontend_export": True,
                            "lean_ingest": True,
                        }
                    }
                )
            )
            consistency = artifact_gate("artifact_consistency")
            consistency["detail"] = {
                "oracle_transaction_digest": validation.detail[
                    "transaction_digest"
                ]
            }
            validated_gates_path.write_text(
                json.dumps(
                    {
                        "gates": {
                            "aresplan_valid": artifact_gate("ares_plan"),
                            "targetplan_valid": artifact_gate("target_plan"),
                            "artifact_consistency": consistency,
                            "shortcut_scan": artifact_gate("shortcut_scan"),
                        }
                    }
                )
            )

            rc = main(
                [
                    "--gates",
                    str(gates_path),
                    "--validated-gates",
                    str(validated_gates_path),
                    "--oracle",
                    str(oracle_path),
                    "--oracle-dense",
                    str(dense_path),
                    "--required-gates",
                    *CPU_ONLY_GATES,
                    "--output-json",
                    str(output_json),
                    "--output-txt",
                    str(output_txt),
                ]
            )

            self.assertEqual(rc, 0)
            reward = json.loads(output_json.read_text())
            self.assertEqual(reward["first_failed_gate"], "diagnostic_complete")
            self.assertEqual(reward["stage_cap"], 0.99)
            self.assertEqual(reward["validation_authority"], "diagnostic_receipt")
            self.assertEqual(
                reward["claim_ceiling"],
                "diagnostic_only_nonpromotion",
            )
            self.assertFalse(reward["promotion_eligible"])


if __name__ == "__main__":
    unittest.main()

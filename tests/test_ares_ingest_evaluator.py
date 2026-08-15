from __future__ import annotations

import json
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable
from unittest import mock

from ares_ingest_autoagent.artifacts import (
  HfCpuOracleAuthority,
  load_hf_cpu_oracle_authority_file,
  validate_hf_cpu_oracle_evidence_files,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
ARES_ROOT = REPO_ROOT.parents[1]
EVALUATOR = (
  REPO_ROOT / "templates/ares-ingest-harbor-task/tests/evaluate_ares_ingest.py"
)
HARBOR_TEST_SCRIPT = REPO_ROOT / "templates/ares-ingest-harbor-task/tests/test.sh"
ORACLE_SCRIPT = ARES_ROOT / "tools/oracles/hf-cpu/capture_hf_cpu_oracle.py"
ORACLE_SPEC = importlib.util.spec_from_file_location(
  "test_ares_ingest_evaluator_hf_cpu_oracle",
  ORACLE_SCRIPT,
)
assert ORACLE_SPEC is not None and ORACLE_SPEC.loader is not None
ORACLE_MODULE = importlib.util.module_from_spec(ORACLE_SPEC)
ORACLE_SPEC.loader.exec_module(ORACLE_MODULE)
EVALUATOR_SPEC = importlib.util.spec_from_file_location(
  "test_ares_ingest_evaluator_module",
  EVALUATOR,
)
assert EVALUATOR_SPEC is not None and EVALUATOR_SPEC.loader is not None
EVALUATOR_MODULE = importlib.util.module_from_spec(EVALUATOR_SPEC)
EVALUATOR_SPEC.loader.exec_module(EVALUATOR_MODULE)


def oracle_record() -> dict[str, Any]:
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
        "vocab_size": 8,
      },
    },
    "tokenizer": {
      "tokenizer_id": "synthetic/model",
      "requested_revision": "0123456789abcdef0123456789abcdef01234567",
      "resolved_revision": "0123456789abcdef0123456789abcdef01234567",
      "name_or_path": "synthetic/model",
      "vocab_size": 8,
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
      "token_ids": [1, 7],
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
  for logit_slice, dense_row in zip(record["logit_slices"], oracle_dense_rows(record)):
    logit_slice["dense_logits_sha256"] = ORACLE_MODULE.sha256_json(dense_row)
  return record


def oracle_dense_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
  capture_id = record["capture_id"]
  return [
    {
      "capture_id": capture_id,
      "invocation_id": record["invocation_id"],
      "step_index": 0,
      "context_tokens": [1, 7],
      "context_tokens_role": "hf_oracle_replay_context",
      "context_count": 2,
      "new_count": 2,
      "runtime_request_token_count": 2,
      "context_prefix_token_count": 0,
      "last_token": 7,
      "logits": [0.0, -1.0, -2.0, 12.5, 8.0, -3.0, -4.0, -5.0],
    },
    {
      "capture_id": capture_id,
      "invocation_id": record["invocation_id"],
      "step_index": 1,
      "context_tokens": [1, 7, 3],
      "context_tokens_role": "hf_oracle_replay_context",
      "context_count": 3,
      "new_count": 1,
      "runtime_request_token_count": 1,
      "context_prefix_token_count": 2,
      "last_token": 3,
      "logits": [0.0, -1.0, 9.25, 1.0, 0.0, 4.0, -4.0, -5.0],
    },
  ]


def oracle_transaction_files() -> dict[str, str]:
  record = oracle_record()
  return {
    "oracle.jsonl": ORACLE_MODULE.canonical_json_line(record).decode(),
    "oracle-dense.jsonl": b"".join(
      ORACLE_MODULE.canonical_json_line(row) for row in oracle_dense_rows(record)
    ).decode(),
  }


def publish_oracle_transaction(
  root: Path,
  *,
  oracle_name: str,
  dense_name: str,
  oracle_text: str,
  dense_text: str,
) -> Any:
  oracle_payload = oracle_text.encode()
  dense_payload = dense_text.encode()
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
  transaction = ORACLE_MODULE.publish_capture_transaction(
    oracle_stage=oracle_stage,
    oracle_output=root / oracle_name,
    dense_stage=dense_stage,
    dense_output=root / dense_name,
  )
  assert transaction is not None
  return transaction


def operator_authority_document(transaction: Any) -> dict[str, Any]:
  return {
    "schema": "ares.autoagent.hf_cpu_oracle_authority.v1",
    "transaction_schema": transaction.schema,
    "transaction_digest": transaction.digest,
    "canonical_validator_sha256": ORACLE_MODULE.capture_source_sha256(),
    "canonical_schema_sha256": (
      "0ff9238de88d9017742ed956d0036cf4919241140efa5fff84792bdedc04cae7"
    ),
    "oracle_sha256": transaction.oracle_sha256,
    "oracle_size_bytes": transaction.oracle_size_bytes,
    "oracle_record_count": len(transaction.oracle_records),
    "dense_sha256": transaction.dense_sha256,
    "dense_size_bytes": transaction.dense_size_bytes,
    "dense_row_count": len(transaction.dense_rows),
  }


def remove_fixture_tree(root: Path) -> None:
  shutil.rmtree(root)


def valid_ares_plan() -> dict[str, Any]:
  return {
    "schema_version": 2,
    "config": {
      "dim": 4,
      "n_layers": 1,
      "n_heads": 1,
      "n_kv_head": 1,
      "head_size": 4,
      "vocab_size": 8,
      "wcls_is_compensated": False,
      "attn_softcapping": False,
      "attn_sinks": False,
      "is_eagle": False,
    },
    "weights": ["w"],
    "buffers": [
      {
        "name": "input_ids",
        "role": "cut_in",
        "type": {
          "kind": "array",
          "scalar_type": "int",
          "outer": "singleton",
          "inner": "1",
        },
      },
      {
        "name": "ares_logits",
        "role": "output",
        "type": {
          "kind": "array",
          "scalar_type": "real",
          "outer": "singleton",
          "inner": "8",
        },
      },
    ],
    "stmts": [
      {
        "stmt": "matmul",
        "weight": "w",
        "input": "input_ids",
        "result": "ares_logits",
      }
    ],
    "provenance": {
      "fx_hash": "test-fx",
      "rule_corpus_hash": "test-rules",
      "emitter_version": "ingest-lean test",
      "hf_export_model_type": "synthetic",
      "target_executor": "tron",
      "hardware_policy": "tron",
      "lowering_path": "Ingest.Plan.ToJson",
    },
  }


def valid_target_plan() -> dict[str, Any]:
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
      "config": valid_ares_plan()["config"],
      "provenance": valid_ares_plan()["provenance"],
    },
    "declared_runtime_bindings": ["input_ids", "ares_logits"],
    "hw_policy": {
      "fallback_policy": "forbidden",
      "placement": "single_device",
      "notes": {},
    },
    "operations": [
      {
        "id": "runtime.input_ids",
        "role": "runtime_binding",
        "action": "runtime_binding",
        "source": {"type": "runtime_binding", "name": "input_ids"},
        "requirements": {},
      },
      {
        "id": "runtime.ares_logits",
        "role": "runtime_binding",
        "action": "runtime_binding",
        "source": {"type": "runtime_binding", "name": "ares_logits"},
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


def replay_context() -> dict[str, Any]:
  return {
    "context_tokens": [1, 2],
    "context_tokens_role": "prompt",
    "context_count": 2,
    "new_count": 1,
    "runtime_request_token_count": 2,
    "context_prefix_token_count": 0,
    "last_token": 2,
  }


def model_provenance() -> dict[str, Any]:
  revision = "0123456789abcdef0123456789abcdef01234567"
  return {
    "model_id": "synthetic/model",
    "checkpoint": {
      "model_id": "synthetic/model",
      "requested_revision": revision,
      "resolved_revision": revision,
      "checkpoint_class": "SyntheticForCausalLM",
      "dtype": "bfloat16",
    },
    "tokenizer": {
      "tokenizer_id": "synthetic/model",
      "requested_revision": revision,
      "resolved_revision": revision,
      "tokenizer_class": "SyntheticTokenizer",
    },
    "input_policy": {
      "decode_strategy": "greedy",
      "context_tokens_role": "hf_oracle_replay_context",
      "prompt_suite_id": "synthetic_prompt_suite",
    },
  }


def text_file_record(path: str, text: str, rows: int | None = None) -> dict[str, Any]:
  record: dict[str, Any] = {
    "path": path,
    "sha256": hashlib.sha256(text.encode()).hexdigest(),
    "size_bytes": len(text.encode()),
  }
  if rows is not None:
    record["rows"] = rows
  return record


def backend_open_evidence() -> dict[str, Any]:
  return {
    "schema": "ares.runtime.backend_open.v1",
    "evidence_class": "system_under_test",
    "status": "opened",
    "backend_id": "tron",
    "ares_plan": {"path": "ares-plan.json", "sha256": "a" * 64},
    "target_plan": {
      "path": "tron.target-plan.json",
      "sha256": "b" * 64,
      "backend_id": "tron",
    },
    "events": [{"event": "backend_open", "backend_id": "tron"}],
    "runtime_generated_sidecars": False,
  }


def one_token_logits_evidence(artifacts: dict[str, Any]) -> dict[str, Any]:
  return {
    "schema": "ares.runtime.one_token_logits.v1",
    "evidence_class": "system_under_test",
    "oracle": "huggingface_transformers_pytorch_cpu",
    "candidate": "ares",
    "tvd": 0.001,
    "tvd_threshold": 0.01,
    "top1_agreement": 1.0,
    "same_argmax": True,
    "replay_context": replay_context(),
    "model_provenance": model_provenance(),
    "artifacts": artifacts,
  }


def eight_token_greedy_evidence(
  *,
  reference_text: str,
  candidate_text: str,
  hf_dense_text: str,
  ares_dense_text: str,
  ares_plan_text: str,
  target_plan_text: str,
) -> dict[str, Any]:
  generated = [1, 2, 3, 4, 5, 6, 7, 8]
  return {
    "schema": "ares.runtime.greedy_token_agreement.v1",
    "evidence_class": "system_under_test",
    "oracle": "huggingface_transformers_pytorch_cpu",
    "candidate": "ares",
    "decode_strategy": "greedy",
    "expected_generated_tokens": 8,
    "generated_tokens": 8,
    "reference_generated_token_ids": generated,
    "candidate_generated_token_ids": generated,
    "exact_match": True,
    "score": 1.0,
    "exact_fraction": 1.0,
    "top1_agreement": 1.0,
    "model_provenance": model_provenance(),
    "reference": text_file_record("reference_tokens.json", reference_text),
    "candidate_output": {
      **text_file_record("candidate_tokens.json", candidate_text),
      "runtime": "ares",
    },
    "cases": [
      {
        "name": "default",
        "exact_match": True,
        "candidate_length": 8,
        "reference_length": 8,
        "matching_prefix_tokens": 8,
      }
    ],
    "artifacts": {
      "reference_tokens": text_file_record(
        "reference_tokens.json",
        reference_text,
      ),
      "candidate_tokens": text_file_record(
        "candidate_tokens.json",
        candidate_text,
      ),
      "hf_cpu_dense_logits_jsonl": text_file_record(
        "hf-dense.jsonl",
        hf_dense_text,
        rows=1,
      ),
      "ares_dense_logits_jsonl": text_file_record(
        "ares-dense.jsonl",
        ares_dense_text,
        rows=1,
      ),
      "runtime_binary": text_file_record(
        "runares",
        "synthetic-runares-binary",
      ),
      "ares_plan": text_file_record("ares-plan.json", ares_plan_text),
      "target_plan": text_file_record(
        "tron.target-plan.json",
        target_plan_text,
      ),
    },
  }


def cpp_tvd_evidence() -> dict[str, Any]:
  return {
    "schema": "ares.comparison.cpp_tvd.v1",
    "evidence_class": "comparison",
    "comparison_source": "cpp_tron_rinzler",
    "candidate": "ares",
    "tvd": 0.001,
    "tvd_threshold": 0.01,
    "replay_context": replay_context(),
  }


def depth_performance_evidence() -> dict[str, Any]:
  return {
    "schema": "ares.performance.depth_ladder.v1",
    "evidence_class": "system_under_test",
    "workload": "independent_decode",
    "correctness_gates_green": True,
    "depths": [
      {
        "generated_tokens": 8,
        "tokens_match": True,
        "throughput_tokens_per_second": 80.0,
      },
      {
        "generated_tokens": 64,
        "tokens_match": True,
        "throughput_tokens_per_second": 70.0,
      },
      {
        "generated_tokens": 512,
        "tokens_match": True,
        "throughput_tokens_per_second": 60.0,
      },
    ],
  }


def mmlu_pro_evidence(
  report_text: str,
  endpoint_models_text: str,
  systems_test_config_text: str,
  *,
  model: str = "synthetic/model",
  backend: str = "tron",
) -> dict[str, Any]:
  report_sha = hashlib.sha256(report_text.encode()).hexdigest()
  endpoint_models_sha = hashlib.sha256(endpoint_models_text.encode()).hexdigest()
  systems_test_config_sha = hashlib.sha256(
    systems_test_config_text.encode()
  ).hexdigest()
  return {
    "schema": "ares.benchmark.mmlu_pro.v1",
    "evidence_class": "system_under_test",
    "status": "passed",
    "model": model,
    "backend": backend,
    "openai_host": "http://127.0.0.1:8000/v1",
    "coverage_percent": 10,
    "effective_coverage_percent": 10,
    "attempted_question_count": 100,
    "question_limit_per_subject": 0,
    "score_percent": 72.0,
    "required_score_percent": 70.0,
    "endpoint_models": {
      "path": "endpoint-models.json",
      "sha256": endpoint_models_sha,
      "openai_host": "http://127.0.0.1:8000/v1",
      "models": [model],
    },
    "subjects": [
      {
        "subject": "total",
        "correct": 72,
        "wrong": 28,
        "attempted_question_count": 100,
        "result_record_count": 100,
        "score_percent": 72.0,
      }
    ],
    "systems_test": {
      "path": "third_party/systems_test",
      "commit": "1" * 40,
      "dirty": False,
      "config_model": model,
      "config": {
        "path": "systems-test-config-row.json",
        "sha256": systems_test_config_sha,
        "source_path": "scripts/mmlu_pro.py",
        "model": model,
        "nominal_users": 1,
      },
      "command": "OPENAI_HOST=http://127.0.0.1:8000/v1 SKIP_PROVISION=1 uv run mmlu_pro",
    },
    "ares": {
      "commit": "2" * 40,
      "dirty": False,
      "backend": backend,
      "runtime_generated_sidecars": False,
      "ares_plan_sha256": "a" * 64,
      "target_plan_sha256": "b" * 64,
    },
    "artifacts": [{"path": "mmlu-report.txt", "sha256": report_sha}],
  }


class AresIngestEvaluatorTest(unittest.TestCase):
  def test_harbor_invokes_verifier_owned_evaluator(self) -> None:
    script = HARBOR_TEST_SCRIPT.read_text()

    self.assertIn("python3 /tests/evaluate_ares_ingest.py", script)
    self.assertNotIn("/task/files/evaluate_ares_ingest.py", script)

  def cpu_only_spec(self) -> dict[str, Any]:
    return {
      "required_gates": [
        "model_spec",
        "hf_cpu_oracle",
        "frontend_export",
        "lean_ingest",
        "aresplan_valid",
        "targetplan_valid",
        "artifact_consistency",
        "shortcut_scan",
      ],
      "explicit_gates": {
        "model_spec": True,
        "frontend_export": True,
        "lean_ingest": True,
      },
      "oracle_records": "oracle.jsonl",
      "oracle_dense_logits": "oracle-dense.jsonl",
      "ares_plan": "ares-plan.json",
      "target_plan": "tron.target-plan.json",
    }

  def cpu_only_task_files(self) -> dict[str, str]:
    return {
      **oracle_transaction_files(),
      "ares-plan.json": json.dumps(valid_ares_plan()) + "\n",
      "tron.target-plan.json": json.dumps(valid_target_plan()) + "\n",
    }

  def run_evaluator(
    self,
    spec: dict[str, Any],
    task_file_texts: dict[str, str],
    *,
    commit_oracle: bool = True,
    supply_oracle_authority: bool = True,
    authority_mutator: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    receipt_replacer: (
      Callable[[Path, dict[str, Any]], Callable[[], None] | None] | None
    ) = None,
  ) -> dict[str, Any]:
    root = Path(tempfile.mkdtemp())
    self.addCleanup(remove_fixture_tree, root)

    ares_repo = root / "ares"
    task_files = root / "task-files"
    work_dir = root / "work"
    logs_dir = root / "logs/verifier"
    reward_json = logs_dir / "reward.json"
    reward_txt = root / "logs/reward.txt"
    ares_repo.mkdir()
    task_files.mkdir()

    oracle_name = spec.get("oracle_records")
    dense_name = spec.get("oracle_dense_logits")
    publish_oracle = (
      commit_oracle
      and isinstance(oracle_name, str)
      and isinstance(dense_name, str)
      and oracle_name in task_file_texts
      and dense_name in task_file_texts
    )
    for name, text in task_file_texts.items():
      if publish_oracle and name in {oracle_name, dense_name}:
        continue
      (task_files / name).write_text(text)
    authority_path = None
    oracle_authority = None
    if publish_oracle:
      assert isinstance(oracle_name, str) and isinstance(dense_name, str)
      transaction = publish_oracle_transaction(
        task_files,
        oracle_name=oracle_name,
        dense_name=dense_name,
        oracle_text=task_file_texts[oracle_name],
        dense_text=task_file_texts[dense_name],
      )
      validation = validate_hf_cpu_oracle_evidence_files(
        task_files / oracle_name,
        task_files / dense_name,
      )
      self.assertTrue(validation.passed, validation.errors)
      oracle_authority = operator_authority_document(transaction)
      if authority_mutator is not None:
        oracle_authority = authority_mutator(dict(oracle_authority))
      if supply_oracle_authority:
        verifier_tests = root / "verifier-tests"
        verifier_tests.mkdir()
        authority_path = verifier_tests / "oracle-authority.json"
        authority_path.write_text(json.dumps(oracle_authority) + "\n")

    full_spec = {
      "model": "synthetic/model",
      "work_dir": str(work_dir),
      **spec,
    }
    (task_files / "model_spec.json").write_text(json.dumps(full_spec))

    env = os.environ.copy()
    pythonpath = [str(REPO_ROOT)]
    if existing := env.get("PYTHONPATH"):
      pythonpath.append(existing)
    env.update(
      {
        "ARES_REPO": str(ares_repo),
        "TASK_FILES_DIR": str(task_files),
        "MODEL_SPEC": str(task_files / "model_spec.json"),
        "VERIFIER_LOG_DIR": str(logs_dir),
        "REWARD_JSON": str(reward_json),
        "REWARD_TXT": str(reward_txt),
        "PYTHONPATH": ":".join(pythonpath),
      }
    )
    if authority_path is not None:
      env["ORACLE_AUTHORITY_FILE"] = str(authority_path)
    else:
      env.pop("ORACLE_AUTHORITY_FILE", None)
    if receipt_replacer is None:
      subprocess.run([sys.executable, str(EVALUATOR)], env=env, check=True)
    else:
      compute_reward = EVALUATOR_MODULE.compute_reward

      def replace_receipt_then_score(**kwargs: Any) -> dict[str, Any]:
        restore = receipt_replacer(work_dir / "validated_gates.json", kwargs)
        try:
          return compute_reward(**kwargs)
        finally:
          if restore is not None:
            restore()

      with (
        mock.patch.dict(os.environ, env, clear=False),
        mock.patch.object(
          EVALUATOR_MODULE,
          "compute_reward",
          side_effect=replace_receipt_then_score,
        ),
      ):
        self.assertEqual(EVALUATOR_MODULE.main(), 0)
    return {
      "work_dir": work_dir,
      "task_files": task_files,
      "logs_dir": logs_dir,
      "authority_path": authority_path,
      "oracle_authority": oracle_authority,
      "reward_json": reward_json,
      "reward_txt": reward_txt,
      "reward": json.loads(reward_json.read_text()),
    }

  def test_promotion_rejects_missing_operator_oracle_authority(self) -> None:
    result = self.run_evaluator(
      self.cpu_only_spec(),
      self.cpu_only_task_files(),
      supply_oracle_authority=False,
    )

    reward = result["reward"]
    validation = validate_hf_cpu_oracle_evidence_files(
      result["task_files"] / "oracle.jsonl",
      result["task_files"] / "oracle-dense.jsonl",
      require_authority=True,
    )
    self.assertFalse(validation.passed)
    self.assertIn("authority is required", " ".join(validation.errors))
    self.assertEqual(reward["first_failed_gate"], "hf_cpu_oracle")
    self.assertFalse(reward["gates"]["hf_cpu_oracle"]["passed"])
    self.assertFalse(reward["promotion_eligible"])

  def test_promotion_rejects_mismatched_operator_oracle_authority(self) -> None:
    result = self.run_evaluator(
      self.cpu_only_spec(),
      self.cpu_only_task_files(),
      authority_mutator=lambda authority: {
        **authority,
        "transaction_digest": "0" * 64,
      },
    )

    reward = result["reward"]
    mismatched_authority = load_hf_cpu_oracle_authority_file(
      result["authority_path"]
    )
    validation = validate_hf_cpu_oracle_evidence_files(
      result["task_files"] / "oracle.jsonl",
      result["task_files"] / "oracle-dense.jsonl",
      expected_authority=mismatched_authority,
      require_authority=True,
    )
    self.assertFalse(validation.passed)
    self.assertIn("does not match", " ".join(validation.errors))
    self.assertEqual(reward["first_failed_gate"], "hf_cpu_oracle")
    self.assertFalse(reward["gates"]["hf_cpu_oracle"]["passed"])
    self.assertFalse(reward["promotion_eligible"])

  def test_candidate_task_file_evaluator_cannot_influence_promotion(self) -> None:
    task_files = self.cpu_only_task_files()
    task_files["evaluate_ares_ingest.py"] = (
      'raise RuntimeError("candidate-writable evaluator executed")\n'
    )

    result = self.run_evaluator(self.cpu_only_spec(), task_files)

    self.assertTrue(
      (result["task_files"] / "evaluate_ares_ingest.py").exists()
    )
    self.assertEqual(result["reward"]["first_failed_gate"], "complete")
    self.assertTrue(result["reward"]["promotion_eligible"])

  def test_coherent_oracle_replacement_and_authority_swap_fails_closed(
    self,
  ) -> None:
    observed: dict[str, Any] = {}

    def replace_transaction_and_authority(
      _path: Path,
      kwargs: dict[str, Any],
    ) -> Callable[[], None]:
      authority_path = Path(os.environ["ORACLE_AUTHORITY_FILE"])
      original_authority_bytes = authority_path.read_bytes()
      original_authority = json.loads(original_authority_bytes)
      expected_authority = kwargs["expected_oracle_authority"]
      self.assertIsInstance(expected_authority, HfCpuOracleAuthority)
      self.assertEqual(
        expected_authority.value,
        original_authority,
      )

      replacement_root = authority_path.parent / "replacement"
      replacement_root.mkdir()
      replacement_files = oracle_transaction_files()
      replacement_oracle = replacement_root / "oracle.jsonl"
      replacement_dense = replacement_root / "oracle-dense.jsonl"
      replacement_transaction = publish_oracle_transaction(
        replacement_root,
        oracle_name=replacement_oracle.name,
        dense_name=replacement_dense.name,
        oracle_text=replacement_files[replacement_oracle.name],
        dense_text=replacement_files[replacement_dense.name],
      )
      replacement_validation = validate_hf_cpu_oracle_evidence_files(
        replacement_oracle,
        replacement_dense,
      )
      self.assertTrue(
        replacement_validation.passed,
        replacement_validation.errors,
      )
      replacement_authority = operator_authority_document(
        replacement_transaction
      )

      current_oracle = kwargs["oracle_path"]
      current_dense = kwargs["oracle_dense_path"]
      current_oracle.chmod(0o600)
      current_dense.chmod(0o600)
      current_oracle.write_bytes(replacement_oracle.read_bytes())
      current_dense.write_bytes(replacement_dense.read_bytes())
      current_oracle.chmod(ORACLE_MODULE.CAPTURE_OUTPUT_MODE)
      current_dense.chmod(ORACLE_MODULE.CAPTURE_OUTPUT_MODE)
      authority_path.write_text(json.dumps(replacement_authority) + "\n")
      observed["replacement_authority"] = replacement_authority
      observed["original_authority_bytes"] = original_authority_bytes

      def restore_authority() -> None:
        authority_path.write_bytes(original_authority_bytes)

      return restore_authority

    result = self.run_evaluator(
      self.cpu_only_spec(),
      self.cpu_only_task_files(),
      receipt_replacer=replace_transaction_and_authority,
    )

    reward = result["reward"]
    self.assertEqual(reward["first_failed_gate"], "hf_cpu_oracle")
    self.assertFalse(reward["gates"]["hf_cpu_oracle"]["passed"])
    self.assertFalse(reward["promotion_eligible"])
    authority_path = result["authority_path"]
    self.assertIsNotNone(authority_path)
    self.assertEqual(
      authority_path.read_bytes(),
      observed["original_authority_bytes"],
    )
    self.assertNotEqual(
      result["oracle_authority"],
      observed["replacement_authority"],
    )
    replacement_authority_path = (
      result["authority_path"].parent / "replacement-authority-check.json"
    )
    replacement_authority_path.write_text(
      json.dumps(observed["replacement_authority"]) + "\n"
    )
    replacement_authority = load_hf_cpu_oracle_authority_file(
      replacement_authority_path
    )
    replacement_validation = validate_hf_cpu_oracle_evidence_files(
      result["task_files"] / "oracle.jsonl",
      result["task_files"] / "oracle-dense.jsonl",
      expected_authority=replacement_authority,
      require_authority=True,
    )
    self.assertTrue(replacement_validation.passed, replacement_validation.errors)
    original_authority = load_hf_cpu_oracle_authority_file(
      result["authority_path"]
    )
    original_validation = validate_hf_cpu_oracle_evidence_files(
      result["task_files"] / "oracle.jsonl",
      result["task_files"] / "oracle-dense.jsonl",
      expected_authority=original_authority,
      require_authority=True,
    )
    self.assertFalse(original_validation.passed)
    self.assertIn("does not match", " ".join(original_validation.errors))

  def test_replacing_post_validation_receipt_cannot_grant_promotion(self) -> None:
    def replace_receipt(path: Path, kwargs: dict[str, Any]) -> None:
      forged = json.loads(path.read_text())
      transaction = ORACLE_MODULE.validate_capture_transaction(
        kwargs["oracle_path"],
        kwargs["oracle_dense_path"],
      )
      forged["gates"]["targetplan_valid"] = {
        "passed": True,
        "score": 1.0,
        "artifact_validator": "target_plan",
      }
      forged["gates"]["artifact_consistency"] = {
        "passed": True,
        "score": 1.0,
        "artifact_validator": "artifact_consistency",
        "detail": {"oracle_transaction_digest": transaction.digest},
      }
      path.write_text(json.dumps(forged))

    result = self.run_evaluator(
      {
        "required_gates": [
          "model_spec",
          "hf_cpu_oracle",
          "frontend_export",
          "lean_ingest",
          "aresplan_valid",
          "targetplan_valid",
          "artifact_consistency",
          "shortcut_scan",
        ],
        "explicit_gates": {
          "model_spec": True,
          "frontend_export": True,
          "lean_ingest": True,
        },
        "oracle_records": "oracle.jsonl",
        "oracle_dense_logits": "oracle-dense.jsonl",
        "ares_plan": "ares-plan.json",
      },
      {
        **oracle_transaction_files(),
        "ares-plan.json": json.dumps(valid_ares_plan()) + "\n",
      },
      receipt_replacer=replace_receipt,
    )

    forged = json.loads(
      (result["work_dir"] / "validated_gates.json").read_text()
    )
    self.assertTrue(forged["gates"]["targetplan_valid"]["passed"])
    self.assertEqual(result["reward"]["first_failed_gate"], "targetplan_valid")
    self.assertFalse(result["reward"]["promotion_eligible"])

  def test_replacing_target_plan_after_validation_fails_final_join(self) -> None:
    def replace_target_plan(_path: Path, kwargs: dict[str, Any]) -> None:
      target_path = kwargs["target_plan_path"]
      target = json.loads(target_path.read_text())
      target["source"]["provenance"]["hf_export_model_type"] = "phi3"
      target_path.write_text(json.dumps(target))

    result = self.run_evaluator(
      {
        "required_gates": [
          "model_spec",
          "hf_cpu_oracle",
          "frontend_export",
          "lean_ingest",
          "aresplan_valid",
          "targetplan_valid",
          "artifact_consistency",
          "shortcut_scan",
        ],
        "explicit_gates": {
          "model_spec": True,
          "frontend_export": True,
          "lean_ingest": True,
        },
        "oracle_records": "oracle.jsonl",
        "oracle_dense_logits": "oracle-dense.jsonl",
        "ares_plan": "ares-plan.json",
        "target_plan": "tron.target-plan.json",
      },
      {
        **oracle_transaction_files(),
        "ares-plan.json": json.dumps(valid_ares_plan()) + "\n",
        "tron.target-plan.json": json.dumps(valid_target_plan()) + "\n",
      },
      receipt_replacer=replace_target_plan,
    )

    receipt = json.loads(
      (result["work_dir"] / "validated_gates.json").read_text()
    )["gates"]
    self.assertTrue(receipt["targetplan_valid"]["passed"])
    self.assertTrue(receipt["artifact_consistency"]["passed"])
    self.assertEqual(result["reward"]["first_failed_gate"], "artifact_consistency")
    self.assertFalse(result["reward"]["gates"]["artifact_consistency"]["passed"])
    self.assertFalse(result["reward"]["promotion_eligible"])

  def test_post_validation_shortcut_is_caught_by_final_scan(self) -> None:
    def add_shortcut(_path: Path, kwargs: dict[str, Any]) -> None:
      shortcut = kwargs["authority_root"] / "runtime/model_plugins/fixture.rs"
      shortcut.parent.mkdir(parents=True)
      shortcut.write_text("// forbidden hand-authored model plugin\n")

    result = self.run_evaluator(
      {
        "required_gates": [
          "model_spec",
          "hf_cpu_oracle",
          "frontend_export",
          "lean_ingest",
          "aresplan_valid",
          "targetplan_valid",
          "artifact_consistency",
          "shortcut_scan",
        ],
        "explicit_gates": {
          "model_spec": True,
          "frontend_export": True,
          "lean_ingest": True,
        },
        "oracle_records": "oracle.jsonl",
        "oracle_dense_logits": "oracle-dense.jsonl",
        "ares_plan": "ares-plan.json",
        "target_plan": "tron.target-plan.json",
      },
      {
        **oracle_transaction_files(),
        "ares-plan.json": json.dumps(valid_ares_plan()) + "\n",
        "tron.target-plan.json": json.dumps(valid_target_plan()) + "\n",
      },
      receipt_replacer=add_shortcut,
    )

    receipt = json.loads(
      (result["work_dir"] / "validated_gates.json").read_text()
    )["gates"]
    self.assertTrue(receipt["shortcut_scan"]["passed"])
    self.assertEqual(result["reward"]["first_failed_gate"], "shortcut_scan")
    self.assertFalse(result["reward"]["gates"]["shortcut_scan"]["passed"])
    self.assertFalse(result["reward"]["promotion_eligible"])

  def test_evaluator_scores_full_artifact_backed_profile(self) -> None:
    mmlu_report = "Total, 72/100, 72.00%\n"
    ares_plan = json.dumps(valid_ares_plan()) + "\n"
    target_plan = json.dumps(valid_target_plan()) + "\n"
    hf_dense = '{"logits":[0.0,1.0]}\n'
    ares_dense = '{"logits":[0.0,1.0]}\n'
    reference_tokens = json.dumps(
      {
        "token_ids": [99, 98],
        "generated_token_ids": [1, 2, 3, 4, 5, 6, 7, 8],
      }
    )
    candidate_tokens = json.dumps(
      {
        "token_ids": [99, 98],
        "generated_token_ids": [1, 2, 3, 4, 5, 6, 7, 8],
      }
    )
    one_token_artifacts = {
      "hf_cpu_dense_logits_jsonl": text_file_record(
        "hf-dense.jsonl",
        hf_dense,
        rows=1,
      ),
      "ares_dense_logits_jsonl": text_file_record(
        "ares-dense.jsonl",
        ares_dense,
        rows=1,
      ),
      "runtime_binary": text_file_record(
        "runares",
        "synthetic-runares-binary",
      ),
      "ares_plan": text_file_record("ares-plan.json", ares_plan),
      "target_plan": text_file_record("tron.target-plan.json", target_plan),
    }
    endpoint_models = (
      json.dumps({"data": [{"id": "synthetic/model", "object": "model"}]}) + "\n"
    )
    systems_test_config = (
      json.dumps(
        {
          "name": "synthetic_model",
          "sample_name": "synthetic_model",
          "model": "synthetic/model",
          "nominal_users": 1,
          "user_sets": [],
        }
      )
      + "\n"
    )
    result = self.run_evaluator(
      {
        "required_gates": [
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
          "mmlu_pro",
        ],
        "explicit_gates": {
          "model_spec": True,
          "frontend_export": True,
          "lean_ingest": True,
        },
        "oracle_records": "oracle.jsonl",
        "oracle_dense_logits": "oracle-dense.jsonl",
        "ares_plan": "ares-plan.json",
        "target_plan": "tron.target-plan.json",
        "backend_open_evidence": "backend-open.json",
        "one_token_logits_evidence": "one-token.json",
        "eight_token_greedy_evidence": "eight-token.json",
        "cpp_tvd_evidence": "cpp-tvd.json",
        "depth_performance_evidence": "depth.json",
        "mmlu_pro_evidence": "mmlu-pro.json",
        "performance_comparison": {
          "measured": "measured.log",
          "speed_of_light": "speed.json",
          "workload": "independent_decode",
        },
      },
      {
        **oracle_transaction_files(),
        "ares-plan.json": ares_plan,
        "tron.target-plan.json": target_plan,
        "backend-open.json": json.dumps(backend_open_evidence()) + "\n",
        "one-token.json": json.dumps(one_token_logits_evidence(one_token_artifacts))
        + "\n",
        "eight-token.json": json.dumps(
          eight_token_greedy_evidence(
            reference_text=reference_tokens,
            candidate_text=candidate_tokens,
            hf_dense_text=hf_dense,
            ares_dense_text=ares_dense,
            ares_plan_text=ares_plan,
            target_plan_text=target_plan,
          )
        )
        + "\n",
        "hf-dense.jsonl": hf_dense,
        "ares-dense.jsonl": ares_dense,
        "runares": "synthetic-runares-binary",
        "cpp-tvd.json": json.dumps(cpp_tvd_evidence()) + "\n",
        "depth.json": json.dumps(depth_performance_evidence()) + "\n",
        "mmlu-pro.json": json.dumps(
          mmlu_pro_evidence(
            mmlu_report,
            endpoint_models,
            systems_test_config,
          )
        )
        + "\n",
        "mmlu-report.txt": mmlu_report,
        "endpoint-models.json": endpoint_models,
        "systems-test-config-row.json": systems_test_config,
        "reference_tokens.json": reference_tokens,
        "candidate_tokens.json": candidate_tokens,
        "measured.log": "Throughput 80.0 tok/s\n",
        "speed.json": json.dumps(
          {
            "targets": {
              "independent_decode": {"speed_of_light_tokens_per_second": 100.0}
            }
          }
        ),
      },
    )

    reward = result["reward"]
    self.assertEqual(reward["first_failed_gate"], "complete")
    self.assertTrue(reward["gates"]["hf_cpu_oracle"]["passed"])
    self.assertTrue(reward["promotion_eligible"])
    self.assertEqual(reward["claim_ceiling"], "promotion_candidate")
    self.assertEqual(reward["validation_authority"], "same_process")
    self.assertEqual(reward["stage_cap"], 1.0)
    self.assertEqual(reward["tau_tokens"], 1.0)
    self.assertEqual(reward["delta_inference"], 0.8)
    self.assertTrue(reward["gates"]["eight_token_greedy"]["passed"])
    validated_gates = json.loads(
      (result["work_dir"] / "validated_gates.json").read_text()
    )["gates"]
    for plan_gate in ("aresplan_valid", "targetplan_valid"):
      self.assertEqual(
        validated_gates[plan_gate]["detail"]["snapshot_validation"],
        "retained_descriptor_rejoin",
      )
      self.assertEqual(len(validated_gates[plan_gate]["detail"]["sha256"]), 64)
    eight_token_detail = validated_gates["eight_token_greedy"]
    self.assertEqual(
      eight_token_detail["artifact_validator"],
      "eight_token_greedy",
    )
    self.assertIn(
      "hf_cpu_dense_logits_jsonl",
      eight_token_detail["detail"]["artifact_keys"],
    )
    self.assertTrue((result["work_dir"] / "performance.json").exists())

  def test_evaluator_rejects_mmlu_pro_for_wrong_model(self) -> None:
    mmlu_report = "Total, 72/100, 72.00%\n"
    endpoint_models = (
      json.dumps({"data": [{"id": "wrong/model", "object": "model"}]}) + "\n"
    )
    systems_test_config = (
      json.dumps(
        {
          "name": "wrong_model",
          "sample_name": "wrong_model",
          "model": "wrong/model",
          "nominal_users": 1,
          "user_sets": [],
        }
      )
      + "\n"
    )
    result = self.run_evaluator(
      {
        "required_gates": ["model_spec", "mmlu_pro"],
        "explicit_gates": {"model_spec": True},
        "backend": "tron",
        "mmlu_pro_evidence": "mmlu-pro.json",
      },
      {
        "mmlu-pro.json": json.dumps(
          mmlu_pro_evidence(
            mmlu_report,
            endpoint_models,
            systems_test_config,
            model="wrong/model",
          )
        )
        + "\n",
        "mmlu-report.txt": mmlu_report,
        "endpoint-models.json": endpoint_models,
        "systems-test-config-row.json": systems_test_config,
      },
    )

    reward = result["reward"]
    self.assertEqual(reward["first_failed_gate"], "mmlu_pro")
    self.assertFalse(reward["gates"]["mmlu_pro"]["passed"])

  def test_evaluator_rejects_target_plan_model_mismatch(self) -> None:
    target_plan = valid_target_plan()
    target_plan["model_id"] = "fixture/model"
    result = self.run_evaluator(
      {
        "required_gates": [
          "model_spec",
          "hf_cpu_oracle",
          "frontend_export",
          "lean_ingest",
          "aresplan_valid",
          "targetplan_valid",
          "artifact_consistency",
          "shortcut_scan",
        ],
        "explicit_gates": {
          "model_spec": True,
          "frontend_export": True,
          "lean_ingest": True,
        },
        "oracle_records": "oracle.jsonl",
        "oracle_dense_logits": "oracle-dense.jsonl",
        "ares_plan": "ares-plan.json",
        "target_plan": "tron.target-plan.json",
      },
      {
        **oracle_transaction_files(),
        "ares-plan.json": json.dumps(valid_ares_plan()) + "\n",
        "tron.target-plan.json": json.dumps(target_plan) + "\n",
      },
    )

    reward = result["reward"]
    self.assertEqual(reward["first_failed_gate"], "artifact_consistency")
    self.assertFalse(reward["gates"]["artifact_consistency"]["passed"])

  def test_evaluator_rejects_plan_model_type_mismatch(self) -> None:
    for plan_kind in ("ares", "target"):
      with self.subTest(plan_kind=plan_kind):
        ares_plan = valid_ares_plan()
        target_plan = valid_target_plan()
        provenance = (
          ares_plan["provenance"]
          if plan_kind == "ares"
          else target_plan["source"]["provenance"]
        )
        provenance["hf_export_model_type"] = "phi3"
        result = self.run_evaluator(
          {
            "required_gates": [
              "model_spec",
              "hf_cpu_oracle",
              "frontend_export",
              "lean_ingest",
              "aresplan_valid",
              "targetplan_valid",
              "artifact_consistency",
              "shortcut_scan",
            ],
            "explicit_gates": {
              "model_spec": True,
              "frontend_export": True,
              "lean_ingest": True,
            },
            "oracle_records": "oracle.jsonl",
            "oracle_dense_logits": "oracle-dense.jsonl",
            "ares_plan": "ares-plan.json",
            "target_plan": "tron.target-plan.json",
          },
          {
            **oracle_transaction_files(),
            "ares-plan.json": json.dumps(ares_plan) + "\n",
            "tron.target-plan.json": json.dumps(target_plan) + "\n",
          },
        )

        reward = result["reward"]
        self.assertEqual(reward["first_failed_gate"], "artifact_consistency")
        self.assertFalse(reward["gates"]["artifact_consistency"]["passed"])

  def test_evaluator_rejects_non_concrete_plan_model_type(self) -> None:
    cases = (("missing", None), ("empty", ""), ("unknown", "unknown"))
    for plan_kind in ("ares", "target"):
      for field, value in cases:
        with self.subTest(plan_kind=plan_kind, field=field):
          ares_plan = valid_ares_plan()
          target_plan = valid_target_plan()
          provenance = (
            ares_plan["provenance"]
            if plan_kind == "ares"
            else target_plan["source"]["provenance"]
          )
          if value is None:
            del provenance["hf_export_model_type"]
          else:
            provenance["hf_export_model_type"] = value
          result = self.run_evaluator(
            {
              "required_gates": [
                "model_spec",
                "hf_cpu_oracle",
                "frontend_export",
                "lean_ingest",
                "aresplan_valid",
                "targetplan_valid",
              ],
              "explicit_gates": {
                "model_spec": True,
                "frontend_export": True,
                "lean_ingest": True,
              },
              "oracle_records": "oracle.jsonl",
              "oracle_dense_logits": "oracle-dense.jsonl",
              "ares_plan": "ares-plan.json",
              "target_plan": "tron.target-plan.json",
            },
            {
              **oracle_transaction_files(),
              "ares-plan.json": json.dumps(ares_plan) + "\n",
              "tron.target-plan.json": json.dumps(target_plan) + "\n",
            },
          )

          expected_gate = "aresplan_valid" if plan_kind == "ares" else "targetplan_valid"
          reward = result["reward"]
          self.assertEqual(reward["first_failed_gate"], expected_gate)
          self.assertFalse(reward["gates"][expected_gate]["passed"])

  def test_evaluator_rejects_placeholder_plan_json(self) -> None:
    result = self.run_evaluator(
      {
        "required_gates": [
          "model_spec",
          "hf_cpu_oracle",
          "frontend_export",
          "lean_ingest",
          "aresplan_valid",
          "targetplan_valid",
        ],
        "explicit_gates": {
          "model_spec": True,
          "frontend_export": True,
          "lean_ingest": True,
        },
        "oracle_records": "oracle.jsonl",
        "oracle_dense_logits": "oracle-dense.jsonl",
        "ares_plan": "ares-plan.json",
        "target_plan": "tron.target-plan.json",
      },
      {
        **oracle_transaction_files(),
        "ares-plan.json": "{}\n",
        "tron.target-plan.json": "{}\n",
      },
    )

    reward = result["reward"]
    self.assertEqual(reward["first_failed_gate"], "aresplan_valid")
    self.assertFalse(reward["gates"]["aresplan_valid"]["passed"])

  def test_explicit_gates_cannot_replace_missing_oracle_artifact(self) -> None:
    result = self.run_evaluator(
      {
        "required_gates": [
          "model_spec",
          "hf_cpu_oracle",
          "frontend_export",
          "lean_ingest",
          "aresplan_valid",
          "targetplan_valid",
        ],
        "explicit_gates": {
          "model_spec": True,
          "hf_cpu_oracle": True,
          "frontend_export": True,
          "lean_ingest": True,
          "aresplan_valid": True,
          "targetplan_valid": True,
        },
      },
      {},
    )

    reward = result["reward"]
    self.assertEqual(reward["first_failed_gate"], "hf_cpu_oracle")
    self.assertFalse(reward["gates"]["hf_cpu_oracle"]["passed"])

  def test_evaluator_rejects_uncommitted_oracle_pair(self) -> None:
    result = self.run_evaluator(
      {
        "required_gates": ["model_spec", "hf_cpu_oracle"],
        "explicit_gates": {"model_spec": True},
        "oracle_records": "oracle.jsonl",
        "oracle_dense_logits": "oracle-dense.jsonl",
      },
      oracle_transaction_files(),
      commit_oracle=False,
    )

    reward = result["reward"]
    self.assertEqual(reward["first_failed_gate"], "hf_cpu_oracle")
    self.assertFalse(reward["gates"]["hf_cpu_oracle"]["passed"])

  def test_evaluator_rejects_raw_without_dense_spec_binding(self) -> None:
    result = self.run_evaluator(
      {
        "required_gates": ["model_spec", "hf_cpu_oracle"],
        "explicit_gates": {"model_spec": True},
        "oracle_records": "oracle.jsonl",
      },
      oracle_transaction_files(),
      commit_oracle=False,
    )

    reward = result["reward"]
    self.assertEqual(reward["first_failed_gate"], "hf_cpu_oracle")
    self.assertFalse(reward["gates"]["hf_cpu_oracle"]["passed"])

  def test_explicit_gates_cannot_replace_missing_plan_artifacts(self) -> None:
    result = self.run_evaluator(
      {
        "required_gates": [
          "model_spec",
          "hf_cpu_oracle",
          "frontend_export",
          "lean_ingest",
          "aresplan_valid",
          "targetplan_valid",
        ],
        "explicit_gates": {
          "model_spec": True,
          "frontend_export": True,
          "lean_ingest": True,
          "aresplan_valid": True,
          "targetplan_valid": True,
        },
        "oracle_records": "oracle.jsonl",
        "oracle_dense_logits": "oracle-dense.jsonl",
      },
      oracle_transaction_files(),
    )

    reward = result["reward"]
    self.assertEqual(reward["first_failed_gate"], "aresplan_valid")
    self.assertFalse(reward["gates"]["aresplan_valid"]["passed"])

  def test_explicit_gates_cannot_replace_missing_target_plan_artifact(self) -> None:
    result = self.run_evaluator(
      {
        "required_gates": [
          "model_spec",
          "hf_cpu_oracle",
          "frontend_export",
          "lean_ingest",
          "aresplan_valid",
          "targetplan_valid",
        ],
        "explicit_gates": {
          "model_spec": True,
          "frontend_export": True,
          "lean_ingest": True,
          "targetplan_valid": True,
        },
        "oracle_records": "oracle.jsonl",
        "oracle_dense_logits": "oracle-dense.jsonl",
        "ares_plan": "ares-plan.json",
      },
      {
        **oracle_transaction_files(),
        "ares-plan.json": json.dumps(valid_ares_plan()) + "\n",
      },
    )

    reward = result["reward"]
    self.assertEqual(reward["first_failed_gate"], "targetplan_valid")
    self.assertFalse(reward["gates"]["targetplan_valid"]["passed"])

  def test_evaluator_writes_nonexecuted_command_wrapper_plan(self) -> None:
    result = self.run_evaluator(
      {
        "required_gates": [
          "model_spec",
          "backend_open",
          "one_token_logits",
        ],
        "explicit_gates": {"model_spec": True},
        "weights": "/weights/synthetic",
        "ares_plan": "ares-plan.json",
        "target_plan": "tron.target-plan.json",
      },
      {},
    )

    plan = json.loads((result["work_dir"] / "command_wrappers.json").read_text())
    self.assertEqual(plan["schema"], "ares.autoagent.command_wrappers.v1")
    self.assertFalse(plan["execute_command_wrappers"])
    self.assertEqual(plan["command_gates"], [])
    self.assertEqual(
      {wrapper["name"] for wrapper in plan["wrappers"]},
      {"rinzler_chat_one_token", "rinzler_full_inference_smoke"},
    )
    self.assertEqual(
      sorted(path.name for path in result["logs_dir"].glob("rinzler*.log")),
      [],
      "wrapper commands should not run unless execute_command_wrappers is true",
    )


if __name__ == "__main__":
  unittest.main()

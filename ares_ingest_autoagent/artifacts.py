"""Artifact validators for the Ares ingest AutoAgent scaffold."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


HF_CPU_SCHEMA_ID = "ares.oracles.hf_cpu.record.v1"
HF_CPU_ORACLE_KIND = "huggingface_transformers_pytorch_cpu"
LEAN_TARGET_PLAN_PRODUCER = {"language": "lean", "tool": "ingest-lean"}

FLOATING_REVISION_NAMES = {
    "@",
    "dev",
    "develop",
    "development",
    "head",
    "latest",
    "main",
    "master",
    "stable",
    "trunk",
}
FLOATING_REVISION_PREFIXES = (
    "head~",
    "head^",
    "origin/",
    "refs/heads/",
    "refs/remotes/",
    "upstream/",
)


@dataclass(frozen=True)
class ArtifactValidation:
    passed: bool
    errors: tuple[str, ...]
    detail: dict[str, Any]

    def as_gate(
        self,
        *,
        label: str,
        validator_name: str,
        path: Path | None = None,
    ) -> dict[str, Any]:
        gate: dict[str, Any] = {
            "label": label,
            "artifact_validator": validator_name,
            "passed": self.passed,
            "score": 1.0 if self.passed else 0.0,
            "detail": self.detail,
        }
        if path is not None:
            gate["path"] = str(path)
            gate["exists"] = path.is_file()
        if self.errors:
            gate["errors"] = list(self.errors)
        return gate


def artifact_gate(
    path: Path,
    *,
    label: str,
    validator_name: str,
    validator: Any,
) -> dict[str, Any]:
    if not path.is_file():
        return {
            "label": label,
            "artifact_validator": validator_name,
            "path": str(path),
            "exists": path.exists(),
            "passed": False,
            "score": 0.0,
            "errors": ["artifact file is missing"],
        }
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return {
            "label": label,
            "artifact_validator": validator_name,
            "path": str(path),
            "exists": True,
            "passed": False,
            "score": 0.0,
            "errors": [f"invalid JSON: {exc}"],
        }
    return validator(payload).as_gate(
        label=label,
        validator_name=validator_name,
        path=path,
    )


def ares_plan_gate(path: Path, *, label: str = "generated AresPlan") -> dict[str, Any]:
    return artifact_gate(
        path,
        label=label,
        validator_name="ares_plan",
        validator=validate_ares_plan,
    )


def target_plan_gate(path: Path, *, label: str = "Lean TargetPlan") -> dict[str, Any]:
    return artifact_gate(
        path,
        label=label,
        validator_name="target_plan",
        validator=validate_target_plan,
    )


def is_floating_revision(revision: str) -> bool:
    value = revision.strip().lower()
    return value in FLOATING_REVISION_NAMES or value.startswith(
        FLOATING_REVISION_PREFIXES
    )


def validate_hf_cpu_oracle_record(record: Any) -> ArtifactValidation:
    errors: list[str] = []
    if not isinstance(record, dict):
        return _validation(False, ["record must be a JSON object"], {})

    _require_fields(
        errors,
        record,
        (
            "schema",
            "record_kind",
            "capture_id",
            "created_utc",
            "source",
            "model",
            "tokenizer",
            "run",
            "prompt",
            "generation",
            "logit_slices",
            "environment",
        ),
        "record",
    )
    if errors:
        return _validation(False, errors, {"record_kind": record.get("record_kind")})

    if record.get("schema") != HF_CPU_SCHEMA_ID:
        errors.append("record.schema must be ares.oracles.hf_cpu.record.v1")
    if record.get("record_kind") != "hf_cpu_oracle_capture":
        errors.append("record_kind must be hf_cpu_oracle_capture")

    source = _expect_object(errors, record.get("source"), "source")
    model = _expect_object(errors, record.get("model"), "model")
    tokenizer = _expect_object(errors, record.get("tokenizer"), "tokenizer")
    run = _expect_object(errors, record.get("run"), "run")
    prompt = _expect_object(errors, record.get("prompt"), "prompt")
    generation = _expect_object(errors, record.get("generation"), "generation")
    environment = _expect_object(errors, record.get("environment"), "environment")
    logit_slices = record.get("logit_slices")

    if source is not None:
        _require_fields(errors, source, ("oracle", "capture_script"), "source")
        if source.get("oracle") != HF_CPU_ORACLE_KIND:
            errors.append(f"source.oracle must be {HF_CPU_ORACLE_KIND}")
        _require_non_empty_string(
            errors, source.get("capture_script"), "source.capture_script"
        )

    if model is not None:
        _validate_revision_metadata(
            errors,
            model,
            "model",
            ("model_id", "requested_revision", "resolved_revision", "dtype"),
        )
    if tokenizer is not None:
        _validate_revision_metadata(
            errors,
            tokenizer,
            "tokenizer",
            ("tokenizer_id", "requested_revision", "resolved_revision"),
        )
    if run is not None:
        _validate_run(errors, run)
    if prompt is not None:
        _validate_prompt(errors, prompt)
    if generation is not None:
        _validate_generation(errors, generation)
    if environment is not None:
        _require_fields(
            errors,
            environment,
            ("python_version", "platform", "torch_version", "transformers_version"),
            "environment",
        )

    generated_ids = (
        generation.get("generated_token_ids") if isinstance(generation, dict) else None
    )
    selected_ids = _validate_logit_slices(errors, logit_slices)
    if isinstance(generated_ids, list) and selected_ids is not None:
        if selected_ids != generated_ids:
            errors.append(
                "generation.generated_token_ids must match logit_slices selected_token_id values"
            )

    detail = {
        "schema": record.get("schema"),
        "record_kind": record.get("record_kind"),
        "source_oracle": source.get("oracle") if isinstance(source, dict) else None,
        "generated_token_count": len(generated_ids)
        if isinstance(generated_ids, list)
        else None,
        "logit_slice_count": len(logit_slices)
        if isinstance(logit_slices, list)
        else None,
    }
    return _validation(not errors, errors, detail)


def validate_ares_plan(plan: Any) -> ArtifactValidation:
    errors: list[str] = []
    if not isinstance(plan, dict):
        return _validation(False, ["AresPlan must be a JSON object"], {})

    _require_fields(
        errors,
        plan,
        ("schema_version", "config", "weights", "buffers", "provenance"),
        "AresPlan",
    )
    version = plan.get("schema_version")
    if not isinstance(version, int) or version not in {1, 2}:
        errors.append("AresPlan.schema_version must be 1 or 2")
    _expect_object(errors, plan.get("config"), "AresPlan.config")
    _expect_string_list(errors, plan.get("weights"), "AresPlan.weights")
    if not isinstance(plan.get("buffers"), list):
        errors.append("AresPlan.buffers must be a list")

    provenance = _expect_object(errors, plan.get("provenance"), "AresPlan.provenance")
    if provenance is not None:
        _require_fields(
            errors,
            provenance,
            ("fx_hash", "rule_corpus_hash", "emitter_version"),
            "AresPlan.provenance",
        )
        _require_non_empty_string(
            errors,
            provenance.get("emitter_version"),
            "AresPlan.provenance.emitter_version",
        )
        emitter = provenance.get("emitter_version")
        if isinstance(emitter, str) and "ingest-lean" not in emitter:
            errors.append("AresPlan.provenance.emitter_version must name ingest-lean")

    body = None
    if version == 1:
        body = plan.get("spans")
        if not isinstance(body, list) or not body:
            errors.append("schema-v1 AresPlan.spans must be a non-empty list")
    elif version == 2:
        body = plan.get("stmts")
        if not isinstance(body, list) or not body:
            errors.append("schema-v2 AresPlan.stmts must be a non-empty list")

    detail = {
        "schema_version": version,
        "buffer_count": len(plan.get("buffers", []))
        if isinstance(plan.get("buffers"), list)
        else None,
        "weight_count": len(plan.get("weights", []))
        if isinstance(plan.get("weights"), list)
        else None,
        "statement_count": len(body) if isinstance(body, list) else None,
        "emitter_version": provenance.get("emitter_version")
        if isinstance(provenance, dict)
        else None,
    }
    return _validation(not errors, errors, detail)


def validate_target_plan(plan: Any) -> ArtifactValidation:
    errors: list[str] = []
    if not isinstance(plan, dict):
        return _validation(False, ["TargetPlan must be a JSON object"], {})

    _require_fields(
        errors,
        plan,
        (
            "schema_version",
            "producer",
            "backend_id",
            "model_id",
            "source",
            "declared_runtime_bindings",
            "hw_policy",
            "operations",
        ),
        "TargetPlan",
    )
    if plan.get("schema_version") != 1:
        errors.append("TargetPlan.schema_version must be 1")
    _require_non_empty_string(errors, plan.get("backend_id"), "TargetPlan.backend_id")
    _require_non_empty_string(errors, plan.get("model_id"), "TargetPlan.model_id")

    producer = _expect_object(errors, plan.get("producer"), "TargetPlan.producer")
    if producer is not None:
        for field, expected in LEAN_TARGET_PLAN_PRODUCER.items():
            if producer.get(field) != expected:
                errors.append(f"TargetPlan.producer.{field} must be {expected}")

    source = _expect_object(errors, plan.get("source"), "TargetPlan.source")
    source_statement_count = None
    if source is not None:
        _require_fields(
            errors,
            source,
            ("schema_version", "statement_count", "config", "provenance"),
            "TargetPlan.source",
        )
        if not isinstance(source.get("schema_version"), int):
            errors.append("TargetPlan.source.schema_version must be an integer")
        source_statement_count = source.get("statement_count")
        if not isinstance(source_statement_count, int) or source_statement_count <= 0:
            errors.append("TargetPlan.source.statement_count must be positive")
        _expect_object(errors, source.get("config"), "TargetPlan.source.config")
        source_provenance = _expect_object(
            errors,
            source.get("provenance"),
            "TargetPlan.source.provenance",
        )
        if source_provenance is not None:
            _require_fields(
                errors,
                source_provenance,
                ("fx_hash", "rule_corpus_hash", "emitter_version"),
                "TargetPlan.source.provenance",
            )
            _require_non_empty_string(
                errors,
                source_provenance.get("emitter_version"),
                "TargetPlan.source.provenance.emitter_version",
            )
            emitter = source_provenance.get("emitter_version")
            if isinstance(emitter, str) and "ingest-lean" not in emitter:
                errors.append(
                    "TargetPlan.source.provenance.emitter_version must name ingest-lean"
                )

    declared_bindings = _expect_string_list(
        errors,
        plan.get("declared_runtime_bindings"),
        "TargetPlan.declared_runtime_bindings",
    )
    if declared_bindings is not None and not declared_bindings:
        errors.append("TargetPlan.declared_runtime_bindings must be non-empty")
    _expect_object(errors, plan.get("hw_policy"), "TargetPlan.hw_policy")
    operations = plan.get("operations")
    semantic_count = 0
    runtime_binding_names: set[str] = set()
    if not isinstance(operations, list) or not operations:
        errors.append("TargetPlan.operations must be a non-empty list")
    else:
        for index, operation in enumerate(operations):
            semantic_count += _validate_target_operation(
                errors,
                operation,
                index,
                runtime_binding_names,
            )
    if source_statement_count is not None and semantic_count != source_statement_count:
        errors.append(
            "TargetPlan semantic operation count must match source.statement_count"
        )
    if declared_bindings is not None:
        missing_bindings = sorted(
            set(declared_bindings).difference(runtime_binding_names)
        )
        if missing_bindings:
            errors.append(
                "TargetPlan runtime binding operations missing: "
                + ", ".join(missing_bindings)
            )

    detail = {
        "schema_version": plan.get("schema_version"),
        "backend_id": plan.get("backend_id"),
        "model_id": plan.get("model_id"),
        "operation_count": len(operations) if isinstance(operations, list) else None,
        "semantic_operation_count": semantic_count,
        "runtime_binding_count": len(runtime_binding_names),
    }
    return _validation(not errors, errors, detail)


def _validation(
    passed: bool,
    errors: Iterable[str],
    detail: dict[str, Any],
) -> ArtifactValidation:
    return ArtifactValidation(passed=passed, errors=tuple(errors), detail=detail)


def _require_fields(
    errors: list[str],
    payload: dict[str, Any],
    fields: Iterable[str],
    context: str,
) -> None:
    missing = [field for field in fields if field not in payload]
    if missing:
        errors.append(f"{context} missing required field(s): {', '.join(missing)}")


def _expect_object(
    errors: list[str],
    value: Any,
    context: str,
) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    errors.append(f"{context} must be an object")
    return None


def _expect_string_list(
    errors: list[str],
    value: Any,
    context: str,
) -> list[str] | None:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        errors.append(f"{context} must be a list of strings")
        return None
    return value


def _require_non_empty_string(errors: list[str], value: Any, context: str) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{context} must be a non-empty string")


def _validate_revision_metadata(
    errors: list[str],
    metadata: dict[str, Any],
    label: str,
    fields: tuple[str, ...],
) -> None:
    _require_fields(errors, metadata, fields, label)
    revision = metadata.get("requested_revision")
    _require_non_empty_string(errors, revision, f"{label}.requested_revision")
    if isinstance(revision, str) and is_floating_revision(revision):
        errors.append(f"{label}.requested_revision must not be floating")


def _validate_run(errors: list[str], run: dict[str, Any]) -> None:
    _require_fields(
        errors,
        run,
        (
            "seed",
            "decode_strategy",
            "max_new_tokens",
            "top_k",
            "torch_deterministic_algorithms",
            "local_files_only",
            "trust_remote_code",
        ),
        "run",
    )
    if run.get("decode_strategy") != "greedy":
        errors.append("run.decode_strategy must be greedy")
    if not isinstance(run.get("max_new_tokens"), int) or run.get("max_new_tokens") < 0:
        errors.append("run.max_new_tokens must be a non-negative integer")
    if not isinstance(run.get("top_k"), int) or run.get("top_k") < 1:
        errors.append("run.top_k must be a positive integer")
    for field in (
        "torch_deterministic_algorithms",
        "local_files_only",
        "trust_remote_code",
    ):
        if not isinstance(run.get(field), bool):
            errors.append(f"run.{field} must be a boolean")


def _validate_prompt(errors: list[str], prompt: dict[str, Any]) -> None:
    _require_fields(
        errors,
        prompt,
        ("kind", "text", "token_ids", "token_count", "add_special_tokens"),
        "prompt",
    )
    if prompt.get("kind") not in {"raw", "chat"}:
        errors.append("prompt.kind must be raw or chat")
    token_ids = prompt.get("token_ids")
    if not isinstance(token_ids, list) or not all(
        isinstance(token_id, int) for token_id in token_ids
    ):
        errors.append("prompt.token_ids must be a list of integers")
        return
    if prompt.get("token_count") != len(token_ids):
        errors.append("prompt.token_count must match prompt.token_ids length")
    if not isinstance(prompt.get("add_special_tokens"), bool):
        errors.append("prompt.add_special_tokens must be a boolean")


def _validate_generation(errors: list[str], generation: dict[str, Any]) -> None:
    _require_fields(
        errors,
        generation,
        (
            "generated_token_ids",
            "generated_token_count",
            "generated_text",
            "finish_reason",
            "eos_token_id",
            "eos_token_ids",
            "stop_token_id",
        ),
        "generation",
    )
    generated_ids = generation.get("generated_token_ids")
    if not isinstance(generated_ids, list) or not all(
        isinstance(token_id, int) for token_id in generated_ids
    ):
        errors.append("generation.generated_token_ids must be a list of integers")
        return
    if generation.get("generated_token_count") != len(generated_ids):
        errors.append(
            "generation.generated_token_count must match generated_token_ids length"
        )
    if not isinstance(generation.get("generated_text"), str):
        errors.append("generation.generated_text must be a string")
    if generation.get("finish_reason") not in {"max_new_tokens", "eos_token"}:
        errors.append("generation.finish_reason must be max_new_tokens or eos_token")
    eos_ids = generation.get("eos_token_ids")
    if not isinstance(eos_ids, list) or not all(
        isinstance(token_id, int) for token_id in eos_ids
    ):
        errors.append("generation.eos_token_ids must be a list of integers")
    stop_token_id = generation.get("stop_token_id")
    if generation.get("finish_reason") == "eos_token":
        if not isinstance(stop_token_id, int):
            errors.append("generation.stop_token_id must be an integer on eos stop")
        elif isinstance(eos_ids, list) and stop_token_id not in eos_ids:
            errors.append("generation.stop_token_id must appear in eos_token_ids")
        elif generated_ids and generated_ids[-1] != stop_token_id:
            errors.append("generation.stop_token_id must match final generated token")
    elif stop_token_id is not None:
        errors.append(
            "generation.stop_token_id must be null unless finish is eos_token"
        )


def _validate_logit_slices(errors: list[str], value: Any) -> list[int] | None:
    if not isinstance(value, list):
        errors.append("logit_slices must be a list")
        return None
    selected: list[int] = []
    for expected_step, entry in enumerate(value):
        if not isinstance(entry, dict):
            errors.append("logit_slices entries must be objects")
            return None
        if entry.get("step") != expected_step:
            errors.append("logit_slices step values must be contiguous from zero")
        position = entry.get("position")
        context_token_count = entry.get("context_token_count")
        if not isinstance(position, int) or position < 0:
            errors.append("logit_slices[].position must be a non-negative integer")
        if not isinstance(context_token_count, int) or context_token_count < 0:
            errors.append(
                "logit_slices[].context_token_count must be a non-negative integer"
            )
        elif isinstance(position, int) and context_token_count != position + 1:
            errors.append("logit_slices[].context_token_count must equal position + 1")
        token_id = entry.get("selected_token_id")
        if not isinstance(token_id, int):
            errors.append("logit_slices[].selected_token_id must be an integer")
        else:
            selected.append(token_id)
        if not isinstance(entry.get("selected_token_text"), str):
            errors.append("logit_slices[].selected_token_text must be a string")
        selected_logit = entry.get("selected_token_logit")
        if not isinstance(selected_logit, int | float):
            errors.append("logit_slices[].selected_token_logit must be numeric")
        _validate_top_k(errors, entry.get("top_k"), entry)
    return selected


def _validate_top_k(
    errors: list[str],
    value: Any,
    logit_slice: dict[str, Any],
) -> None:
    if not isinstance(value, list) or not value:
        errors.append("logit_slices[].top_k must be a non-empty list")
        return
    previous_rank = 0
    for item in value:
        if not isinstance(item, dict):
            errors.append("top_k entries must be objects")
            return
        rank = item.get("rank")
        if not isinstance(rank, int) or rank != previous_rank + 1:
            errors.append("top_k ranks must be contiguous from one")
        if not isinstance(item.get("token_id"), int):
            errors.append("top_k token_id must be an integer")
        if not isinstance(item.get("token_text"), str):
            errors.append("top_k token_text must be a string")
        if not isinstance(item.get("logit"), int | float):
            errors.append("top_k logit must be numeric")
        previous_rank = rank if isinstance(rank, int) else previous_rank
    first = value[0]
    if isinstance(first, dict):
        if logit_slice.get("selected_token_logit") != first.get("logit"):
            errors.append("selected_token_logit must match top top_k logit")
        if logit_slice.get("selected_token_id") != first.get("token_id"):
            errors.append("selected_token_id must match top top_k token_id")


def _validate_target_operation(
    errors: list[str],
    operation: Any,
    index: int,
    runtime_binding_names: set[str],
) -> int:
    context = f"TargetPlan.operations[{index}]"
    if not isinstance(operation, dict):
        errors.append(f"{context} must be an object")
        return 0
    _require_fields(
        errors,
        operation,
        ("id", "role", "action", "source", "requirements"),
        context,
    )
    _require_non_empty_string(errors, operation.get("id"), f"{context}.id")
    role = operation.get("role")
    action = operation.get("action")
    if role not in {"semantic", "runtime_binding", "auxiliary"}:
        errors.append(f"{context}.role is invalid")
    if action not in {
        "kernel",
        "matmul",
        "attention",
        "kv_cache",
        "top_k",
        "moe_routing",
        "transfer",
        "barrier",
        "queue",
        "intrinsic",
        "runtime_binding",
    }:
        errors.append(f"{context}.action is invalid")
    source = _expect_object(errors, operation.get("source"), f"{context}.source")
    _expect_object(errors, operation.get("requirements"), f"{context}.requirements")
    if role == "runtime_binding" and source is not None:
        name = source.get("name")
        if isinstance(name, str) and name:
            runtime_binding_names.add(name)
        else:
            errors.append(f"{context}.source.name must name the runtime binding")
    if role == "semantic" and source is not None:
        if source.get("type") != "ares_plan_statement":
            errors.append(f"{context}.source.type must be ares_plan_statement")
        if not isinstance(source.get("statement_index"), int):
            errors.append(f"{context}.source.statement_index must be an integer")
        return 1
    return 0

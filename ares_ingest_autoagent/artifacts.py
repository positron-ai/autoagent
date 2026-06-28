"""Artifact validators for the Ares ingest AutoAgent scaffold."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


HF_CPU_SCHEMA_ID = "ares.oracles.hf_cpu.record.v1"
HF_CPU_ORACLE_KIND = "huggingface_transformers_pytorch_cpu"
LEAN_TARGET_PLAN_PRODUCER = {"language": "lean", "tool": "ingest-lean"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.I)
REPLAY_CONTEXT_FIELDS = (
    "context_tokens",
    "context_tokens_role",
    "context_count",
    "new_count",
    "runtime_request_token_count",
    "context_prefix_token_count",
    "last_token",
)
CPP_COMPARISON_SOURCES = {
    "cpp_tron",
    "cpp_tron_rinzler",
    "cxx_tron",
    "cxx_tron_rinzler",
    "tron_rinzler_cpp",
}
SCORING_WORKLOADS = {"independent_decode", "long_prefill"}

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


def evidence_gate(
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
            "errors": ["evidence file is missing"],
        }
    try:
        payload = _read_json_or_jsonl(path)
    except ValueError as exc:
        return {
            "label": label,
            "artifact_validator": validator_name,
            "path": str(path),
            "exists": True,
            "passed": False,
            "score": 0.0,
            "errors": [str(exc)],
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


def backend_open_gate(
    path: Path, *, label: str = "backend provider open evidence"
) -> dict[str, Any]:
    return evidence_gate(
        path,
        label=label,
        validator_name="backend_open",
        validator=validate_backend_open_evidence,
    )


def one_token_logits_gate(
    path: Path, *, label: str = "one-token logits evidence"
) -> dict[str, Any]:
    return evidence_gate(
        path,
        label=label,
        validator_name="one_token_logits",
        validator=validate_one_token_logits_evidence,
    )


def cpp_tvd_gate(
    path: Path, *, label: str = "C++ comparison TVD evidence"
) -> dict[str, Any]:
    return evidence_gate(
        path,
        label=label,
        validator_name="cpp_tvd",
        validator=validate_cpp_tvd_evidence,
    )


def depth_performance_gate(
    path: Path, *, label: str = "depth ladder performance evidence"
) -> dict[str, Any]:
    return evidence_gate(
        path,
        label=label,
        validator_name="depth_performance",
        validator=validate_depth_performance_evidence,
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


def validate_backend_open_evidence(payload: Any) -> ArtifactValidation:
    errors: list[str] = []
    root = _payload_root(payload, errors, "backend open evidence")
    rows = _payload_rows(root)
    if root is None:
        return _validation(False, errors, {})

    backend_id = _first_string(root, rows, ("backend_id", "backend"))
    status = _first_string(root, rows, ("status", "state"))
    if backend_id is None:
        errors.append("backend evidence must name backend_id")
    if status not in {"open", "opened", "ok", "passed", "ready"}:
        errors.append("backend evidence status must be open/opened/ok/passed/ready")

    ares_sha = _first_string(root, rows, ("ares_plan_sha256", "ares_plan_hash"))
    target_sha = _first_string(root, rows, ("target_plan_sha256", "target_plan_hash"))
    ares_obj = root.get("ares_plan") if isinstance(root, dict) else None
    target_obj = root.get("target_plan") if isinstance(root, dict) else None
    if ares_sha is None and isinstance(ares_obj, dict):
        ares_sha = _first_string(ares_obj, [], ("sha256", "hash"))
    if target_sha is None and isinstance(target_obj, dict):
        target_sha = _first_string(target_obj, [], ("sha256", "hash"))
    _require_sha256(errors, ares_sha, "AresPlan")
    _require_sha256(errors, target_sha, "TargetPlan")

    target_backend = _first_string(
        root, rows, ("target_plan_backend", "target_backend", "backend_id")
    )
    if (
        backend_id is not None
        and target_backend is not None
        and target_backend != backend_id
    ):
        errors.append("TargetPlan backend must match opened backend")

    if _truthy_field(
        root, rows, ("runtime_generated_sidecars", "runtime_generated_plan")
    ):
        errors.append("backend evidence must not use runtime-generated plan sidecars")

    event_names = {_event_name(row) for row in rows}
    event_names.discard(None)
    if rows and not event_names.intersection(
        {"backend_open", "provider_open", "session_open", "runtime_open"}
    ):
        errors.append("backend event evidence must include a backend-open event row")

    detail = {
        "backend_id": backend_id,
        "status": status,
        "event_count": len(rows),
        "ares_plan_sha256": ares_sha,
        "target_plan_sha256": target_sha,
    }
    return _validation(not errors, errors, detail)


def validate_one_token_logits_evidence(payload: Any) -> ArtifactValidation:
    errors: list[str] = []
    root = _payload_root(payload, errors, "one-token logits evidence")
    if root is None:
        return _validation(False, errors, {})
    if not isinstance(root, dict):
        return _validation(
            False,
            [*errors, "one-token logits evidence must be a JSON object"],
            {},
        )

    if _first_string(root, [], ("oracle", "oracle_source")) != HF_CPU_ORACLE_KIND:
        errors.append(f"one-token oracle must be {HF_CPU_ORACLE_KIND}")
    evidence_class = _first_string(root, [], ("evidence_class", "classification"))
    if evidence_class not in {"system_under_test", "diagnostic", "promotion"}:
        errors.append(
            "one-token evidence_class must classify Ares as system under test"
        )

    tvd, threshold = _validate_tvd(errors, root, "one-token")
    top1 = root.get("top1_agreement")
    same_argmax = root.get("same_argmax")
    if not isinstance(top1, int | float):
        errors.append("one-token top1_agreement must be numeric")
    elif float(top1) < 1.0:
        errors.append("one-token top1_agreement must be 1.0")
    if same_argmax is not True:
        errors.append("one-token same_argmax must be true")
    _validate_replay_context(errors, root.get("replay_context"), "one-token")

    passed = (
        not errors and tvd is not None and threshold is not None and tvd <= threshold
    )
    detail = {
        "tvd": tvd,
        "tvd_threshold": threshold,
        "top1_agreement": float(top1) if isinstance(top1, int | float) else None,
        "same_argmax": same_argmax,
    }
    return _validation(passed, errors, detail)


def validate_cpp_tvd_evidence(payload: Any) -> ArtifactValidation:
    errors: list[str] = []
    root = _payload_root(payload, errors, "C++ comparison evidence")
    if root is None:
        return _validation(False, errors, {})
    if not isinstance(root, dict):
        return _validation(
            False,
            [*errors, "C++ comparison evidence must be a JSON object"],
            {},
        )

    evidence_class = _first_string(root, [], ("evidence_class", "classification"))
    if evidence_class != "comparison":
        errors.append("C++ TVD evidence_class must be comparison")
    source = _first_string(root, [], ("comparison_source", "source", "baseline"))
    if source not in CPP_COMPARISON_SOURCES:
        errors.append("comparison_source must identify C++ Tron/Rinzler")
    tvd, threshold = _validate_tvd(errors, root, "C++ TVD")
    _validate_replay_context(errors, root.get("replay_context"), "C++ TVD")
    if root.get("oracle") == "cpp_tron":
        errors.append("C++ Tron/Rinzler must not be labeled as correctness oracle")

    passed = (
        not errors and tvd is not None and threshold is not None and tvd <= threshold
    )
    detail = {
        "comparison_source": source,
        "tvd": tvd,
        "tvd_threshold": threshold,
    }
    return _validation(passed, errors, detail)


def validate_depth_performance_evidence(payload: Any) -> ArtifactValidation:
    errors: list[str] = []
    root = _payload_root(payload, errors, "depth performance evidence")
    if root is None:
        return _validation(False, errors, {})
    if not isinstance(root, dict):
        return _validation(
            False,
            [*errors, "depth performance evidence must be a JSON object"],
            {},
        )

    evidence_class = _first_string(root, [], ("evidence_class", "classification"))
    if evidence_class not in {"system_under_test", "promotion", "comparison"}:
        errors.append("depth performance evidence_class must be explicit")
    workload = _normalize_workload(root.get("workload"))
    if workload not in SCORING_WORKLOADS:
        errors.append(
            "depth performance workload must be independent_decode or long_prefill"
        )
    if root.get("correctness_gates_green") is not True:
        errors.append("depth performance requires correctness_gates_green=true")

    depths = root.get("depths")
    seen_depths: set[int] = set()
    if not isinstance(depths, list) or not depths:
        errors.append("depth performance depths must be a non-empty list")
    else:
        for index, entry in enumerate(depths):
            if not isinstance(entry, dict):
                errors.append("depth performance depth entries must be objects")
                continue
            depth = entry.get("generated_tokens", entry.get("depth"))
            if not isinstance(depth, int):
                errors.append(f"depths[{index}].generated_tokens must be an integer")
                continue
            seen_depths.add(depth)
            if entry.get("tokens_match") is not True:
                errors.append(f"depth {depth} must have tokens_match=true")
            tps = entry.get(
                "throughput_tokens_per_second", entry.get("tokens_per_second")
            )
            if not isinstance(tps, int | float) or tps <= 0:
                errors.append(f"depth {depth} must record positive throughput")
    missing = [depth for depth in (8, 64, 512) if depth not in seen_depths]
    if missing:
        errors.append(
            "depth performance missing ladder depth(s): " + ", ".join(map(str, missing))
        )

    detail = {
        "workload": workload,
        "depths": sorted(seen_depths),
        "depth_count": len(seen_depths),
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


def _read_json_or_jsonl(path: Path) -> Any:
    text = path.read_text()
    if path.suffix == ".jsonl":
        rows = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {lineno}: {exc}") from exc
        return rows
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc


def _payload_root(
    payload: Any,
    errors: list[str],
    label: str,
) -> dict[str, Any] | list[Any] | None:
    if isinstance(payload, dict | list):
        return payload
    errors.append(f"{label} must be a JSON object or JSONL event list")
    return None


def _payload_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("events"), list):
        return [item for item in payload["events"] if isinstance(item, dict)]
    return []


def _first_string(
    payload: dict[str, Any] | list[Any],
    rows: list[dict[str, Any]],
    keys: tuple[str, ...],
) -> str | None:
    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for row in rows:
        for key in keys:
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _truthy_field(
    payload: dict[str, Any] | list[Any],
    rows: list[dict[str, Any]],
    keys: tuple[str, ...],
) -> bool:
    if isinstance(payload, dict):
        for key in keys:
            if payload.get(key) is True:
                return True
    return any(row.get(key) is True for row in rows for key in keys)


def _event_name(row: dict[str, Any]) -> str | None:
    for key in ("event", "event_name", "name", "phase"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _require_sha256(errors: list[str], value: Any, label: str) -> None:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        errors.append(f"{label} SHA-256 must be a 64-character hex string")


def _validate_tvd(
    errors: list[str],
    payload: dict[str, Any],
    label: str,
) -> tuple[float | None, float | None]:
    tvd = payload.get("tvd", payload.get("max_tvd"))
    threshold = payload.get("tvd_threshold", 0.01)
    if not isinstance(tvd, int | float) or tvd < 0:
        errors.append(f"{label} tvd/max_tvd must be a non-negative number")
        tvd_value = None
    else:
        tvd_value = float(tvd)
    if not isinstance(threshold, int | float) or threshold <= 0:
        errors.append(f"{label} tvd_threshold must be a positive number")
        threshold_value = None
    else:
        threshold_value = float(threshold)
    return tvd_value, threshold_value


def _validate_replay_context(
    errors: list[str],
    value: Any,
    label: str,
) -> None:
    if not isinstance(value, dict):
        errors.append(f"{label} replay_context must be an object")
        return
    missing = [field for field in REPLAY_CONTEXT_FIELDS if field not in value]
    if missing:
        errors.append(
            f"{label} replay_context missing required field(s): {', '.join(missing)}"
        )
    if not isinstance(value.get("context_tokens"), list):
        errors.append(f"{label} replay_context.context_tokens must be a list")
    for field in (
        "context_count",
        "new_count",
        "runtime_request_token_count",
        "context_prefix_token_count",
    ):
        if not isinstance(value.get(field), int):
            errors.append(f"{label} replay_context.{field} must be an integer")


def _normalize_workload(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


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

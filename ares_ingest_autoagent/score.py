"""Score Ares model-ingest artifacts for AutoAgent and Harbor.

The score is a hill-climbing signal for agents. It is not a production
readiness substitute: hard gates still decide whether a model row can be
promoted.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tron_ingest_autoagent.performance import is_scoring_workload
from tron_ingest_autoagent.score import (
    clamp,
    score_performance_results,
    score_token_results,
)

from ares_ingest_autoagent.artifacts import validate_hf_cpu_oracle_record


STAGE_CAPS: dict[str, float] = {
    "not_started": 0.0,
    "fork_submodule": 0.03,
    "model_spec": 0.05,
    "hf_cpu_oracle": 0.10,
    "frontend_export": 0.18,
    "lean_ingest": 0.28,
    "aresplan_valid": 0.38,
    "targetplan_valid": 0.48,
    "artifact_consistency": 0.50,
    "shortcut_scan": 0.52,
    "backend_open": 0.58,
    "one_token_logits": 0.70,
    "eight_token_greedy": 0.80,
    "cpp_tvd": 0.88,
    "depth_performance": 0.95,
    "complete": 1.00,
}

CPU_ONLY_GATES: tuple[str, ...] = (
    "model_spec",
    "hf_cpu_oracle",
    "frontend_export",
    "lean_ingest",
    "aresplan_valid",
    "targetplan_valid",
    "artifact_consistency",
    "shortcut_scan",
)

BACKEND_GATES: tuple[str, ...] = (
    *CPU_ONLY_GATES,
    "backend_open",
    "one_token_logits",
    "eight_token_greedy",
)

COMPARISON_GATES: tuple[str, ...] = (*BACKEND_GATES, "cpp_tvd")
FULL_GATES: tuple[str, ...] = (*COMPARISON_GATES, "depth_performance")
STANDARD_GATES: tuple[str, ...] = CPU_ONLY_GATES

GATE_PROFILES: dict[str, tuple[str, ...]] = {
    "cpu-only": CPU_ONLY_GATES,
    "backend": BACKEND_GATES,
    "comparison": COMPARISON_GATES,
    "full": FULL_GATES,
}

ALPHA_EXECUTION_WEIGHTS: dict[str, float] = {
    "model_spec": 0.05,
    "hf_cpu_oracle": 0.15,
    "frontend_export": 0.10,
    "lean_ingest": 0.10,
    "aresplan_valid": 0.15,
    "targetplan_valid": 0.15,
    "artifact_consistency": 0.05,
    "shortcut_scan": 0.05,
    "backend_open": 0.10,
    "one_token_logits": 0.20,
}

ARTIFACT_GATE_VALIDATORS: dict[str, str] = {
    "hf_cpu_oracle": "hf_cpu_oracle",
    "aresplan_valid": "ares_plan",
    "targetplan_valid": "target_plan",
    "artifact_consistency": "artifact_consistency",
    "shortcut_scan": "shortcut_scan",
    "backend_open": "backend_open",
    "one_token_logits": "one_token_logits",
    "cpp_tvd": "cpp_tvd",
    "depth_performance": "depth_performance",
}


@dataclass(frozen=True)
class Gate:
    passed: bool
    score: float
    detail: Any = None


def read_json(path: Path | None) -> Any:
    if path is None:
        return None
    return json.loads(path.read_text())


def as_gate(value: Any) -> Gate:
    if isinstance(value, Gate):
        return value
    if isinstance(value, bool):
        return Gate(passed=value, score=1.0 if value else 0.0)
    if isinstance(value, (int, float)):
        score = clamp(float(value))
        return Gate(passed=score >= 1.0, score=score)
    if isinstance(value, dict):
        raw_score = value.get("score", value.get("value"))
        if raw_score is None and "passed" in value:
            raw_score = 1.0 if value["passed"] else 0.0
        score = clamp(float(raw_score or 0.0))
        return Gate(
            passed=bool(value.get("passed", score >= 1.0)),
            score=score,
            detail=value,
        )
    return Gate(False, 0.0, value)


def extract_gates(payload: Any) -> dict[str, Gate]:
    if payload is None:
        return {}
    if isinstance(payload, dict) and "gates" in payload:
        payload = payload["gates"]
    if not isinstance(payload, dict):
        raise TypeError("gate payload must be a JSON object")
    return {name: as_gate(value) for name, value in payload.items()}


def merge_gate(gates: dict[str, Gate], name: str, gate: Gate) -> None:
    existing = gates.get(name)
    if existing is not None and existing.passed is False:
        return
    gates[name] = gate


def gate_has_artifact_validator(gate: Gate, validator_name: str) -> bool:
    return (
        isinstance(gate.detail, dict)
        and gate.detail.get("artifact_validator") == validator_name
    )


def fail_closed_artifact_gate(name: str, validator_name: str) -> Gate:
    return Gate(
        passed=False,
        score=0.0,
        detail={
            "artifact_validator": validator_name,
            "error": f"{name} requires validator evidence, not an explicit gate",
        },
    )


def enforce_artifact_gate_evidence(
    gates: dict[str, Gate],
    required_gates: tuple[str, ...],
) -> None:
    for name, validator_name in ARTIFACT_GATE_VALIDATORS.items():
        if name not in required_gates:
            continue
        gate = gates.get(name)
        if gate is None or not gate.passed:
            continue
        if not gate_has_artifact_validator(gate, validator_name):
            gates[name] = fail_closed_artifact_gate(name, validator_name)


def reject_untrusted_artifact_gates(
    gates: dict[str, Gate],
    required_gates: tuple[str, ...],
) -> None:
    for name, validator_name in ARTIFACT_GATE_VALIDATORS.items():
        if name in gates and name in required_gates:
            gates[name] = fail_closed_artifact_gate(name, validator_name)


def merge_validated_gates(
    gates: dict[str, Gate],
    validated_gates_payload: Any,
) -> None:
    for name, gate in extract_gates(validated_gates_payload).items():
        gates[name] = gate


def first_failed_gate(gates: dict[str, Gate], required: tuple[str, ...]) -> str:
    if not gates:
        return "not_started"
    for name in required:
        gate = gates.get(name)
        if gate is None or not gate.passed:
            return name
    return "complete"


def required_gates_for_profile(profile: str) -> tuple[str, ...]:
    try:
        return GATE_PROFILES[profile]
    except KeyError as exc:
        known = ", ".join(sorted(GATE_PROFILES))
        raise ValueError(
            f"unknown gate profile {profile!r}; expected one of {known}"
        ) from exc


def compute_alpha_execution(
    gates: dict[str, Gate],
    required_gates: tuple[str, ...] = STANDARD_GATES,
) -> tuple[float, dict[str, float]]:
    components: dict[str, float] = {}
    active_weights = {
        name: weight
        for name, weight in ALPHA_EXECUTION_WEIGHTS.items()
        if name in required_gates
    }
    for name in active_weights:
        gate = gates.get(name)
        components[name] = gate.score if gate is not None else 0.0
    weight_total = sum(active_weights.values())
    alpha = (
        sum(active_weights[name] * components[name] for name in active_weights)
        / weight_total
        if weight_total
        else 0.0
    )
    return clamp(alpha), components


def _score_logit_payload(payload: Any) -> Gate:
    if not isinstance(payload, dict):
        return Gate(False, 0.0, {"error": "missing logit payload"})
    if "score" in payload:
        return as_gate(payload)

    tvd = payload.get("tvd")
    if tvd is None:
        tvd = payload.get("max_tvd")
    top1 = payload.get("top1_agreement")
    same_argmax = payload.get("same_argmax")
    passed = bool(payload.get("passed", False))

    tvd_score = 0.0
    if isinstance(tvd, (int, float)):
        threshold = float(payload.get("tvd_threshold", 0.01) or 0.01)
        tvd_score = 1.0 if tvd <= threshold else clamp(1.0 - float(tvd))
    top1_score = clamp(float(top1)) if isinstance(top1, (int, float)) else 0.0
    argmax_score = 1.0 if same_argmax is True else 0.0
    score = clamp(0.60 * tvd_score + 0.25 * top1_score + 0.15 * argmax_score)
    return Gate(passed=passed or score >= 1.0, score=score, detail=payload)


def _score_oracle_records(payload: Any) -> Gate:
    if payload is None:
        return Gate(False, 0.0, {"error": "missing oracle payload"})
    records = payload if isinstance(payload, list) else [payload]
    if not records or not all(isinstance(record, dict) for record in records):
        return Gate(False, 0.0, {"error": "oracle payload must be object/list"})

    validations = []
    for record in records:
        validations.append(validate_hf_cpu_oracle_record(record))
    passed = all(validation.passed for validation in validations)
    return Gate(
        passed=passed,
        score=1.0 if passed else 0.0,
        detail={
            "artifact_validator": "hf_cpu_oracle",
            "records": len(records),
            "valid_records": sum(1 for validation in validations if validation.passed),
            "errors": [
                {"index": index, "errors": list(validation.errors)}
                for index, validation in enumerate(validations)
                if validation.errors
            ],
        },
    )


def compute_reward(
    *,
    gates_payload: Any = None,
    validated_gates_payload: Any = None,
    oracle_payload: Any = None,
    token_payload: Any = None,
    performance_payload: Any = None,
    one_token_payload: Any = None,
    required_gates: tuple[str, ...] = STANDARD_GATES,
) -> dict[str, Any]:
    gates = extract_gates(gates_payload)
    reject_untrusted_artifact_gates(gates, required_gates)
    if validated_gates_payload is not None:
        merge_validated_gates(gates, validated_gates_payload)

    if oracle_payload is not None:
        gates["hf_cpu_oracle"] = _score_oracle_records(oracle_payload)

    if one_token_payload is not None:
        gates["one_token_logits"] = _score_logit_payload(one_token_payload)

    enforce_artifact_gate_evidence(gates, required_gates)

    first_failed = first_failed_gate(gates, required_gates)
    stage_cap = STAGE_CAPS.get(first_failed, 0.0)

    alpha_execution, alpha_components = compute_alpha_execution(gates, required_gates)
    tau_tokens = score_token_results(token_payload)
    delta_inference = score_performance_results(performance_payload)
    if not is_scoring_workload(
        performance_payload.get("workload")
        if isinstance(performance_payload, dict)
        else None
    ):
        delta_inference = 0.0

    raw = clamp(0.60 * alpha_execution + 0.25 * tau_tokens + 0.15 * delta_inference)
    score = min(raw, stage_cap)
    if math.isclose(score, 1.0, rel_tol=0.0, abs_tol=1e-12):
        score = 1.0

    return {
        "score": score,
        "raw_score": raw,
        "alpha_execution": alpha_execution,
        "tau_tokens": tau_tokens,
        "delta_inference": delta_inference,
        "stage_cap": stage_cap,
        "first_failed_gate": first_failed,
        "gates": {
            name: {"passed": gate.passed, "score": gate.score}
            for name, gate in sorted(gates.items())
        },
        "alpha_execution_components": alpha_components,
        "weights": {
            "alpha_execution": 0.60,
            "tau_tokens": 0.25,
            "delta_inference": 0.15,
        },
    }


def parse_gate_override(text: str) -> tuple[str, Gate]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("gate override must be NAME=VALUE")
    name, value = text.split("=", 1)
    normalized = value.strip().lower()
    if normalized in {"true", "pass", "passed", "1", "yes"}:
        return name, Gate(True, 1.0, {"source": "cli"})
    if normalized in {"false", "fail", "failed", "0", "no"}:
        return name, Gate(False, 0.0, {"source": "cli"})
    try:
        score = clamp(float(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid gate value: {value!r}") from exc
    return name, Gate(score >= 1.0, score, {"source": "cli"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gates", type=Path, help="JSON file containing gates")
    parser.add_argument(
        "--validated-gates",
        type=Path,
        help="JSON file containing gates produced by artifact validators",
    )
    parser.add_argument("--oracle", type=Path, help="HF CPU oracle JSON/JSONL summary")
    parser.add_argument("--tokens", type=Path, help="Token agreement JSON")
    parser.add_argument("--performance", type=Path, help="Performance JSON")
    parser.add_argument("--one-token", type=Path, help="One-token logits/TVD JSON")
    parser.add_argument(
        "--gate",
        action="append",
        default=[],
        type=parse_gate_override,
        metavar="NAME=VALUE",
        help="Add or override a gate",
    )
    parser.add_argument(
        "--required-gates",
        nargs="+",
        default=list(STANDARD_GATES),
        help="Gate order used for first-failure and stage cap computation",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("/logs/verifier/reward.json"),
        help="Reward JSON path",
    )
    parser.add_argument(
        "--output-txt",
        type=Path,
        default=Path("/logs/reward.txt"),
        help="Reward text path",
    )
    parser.add_argument("--print-json", action="store_true")
    return parser


def _read_json_or_jsonl(path: Path | None) -> Any:
    if path is None:
        return None
    text = path.read_text()
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    gates_payload = read_json(args.gates)
    if args.gate:
        gates = extract_gates(gates_payload)
        gates.update(dict(args.gate))
        gates_payload = {"gates": gates}

    reward = compute_reward(
        gates_payload=gates_payload,
        validated_gates_payload=read_json(args.validated_gates),
        oracle_payload=_read_json_or_jsonl(args.oracle),
        token_payload=read_json(args.tokens),
        performance_payload=read_json(args.performance),
        one_token_payload=read_json(args.one_token),
        required_gates=tuple(args.required_gates),
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(reward, indent=2) + "\n")
    args.output_txt.parent.mkdir(parents=True, exist_ok=True)
    args.output_txt.write_text(f"{reward['score']:.12g}\n")

    if args.print_json:
        print(json.dumps(reward, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

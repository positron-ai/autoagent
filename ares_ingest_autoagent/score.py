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
from typing import Any, Mapping

from tron_ingest_autoagent.performance import is_scoring_workload
from tron_ingest_autoagent.score import (
    clamp,
    score_performance_results,
    score_token_results,
)

from ares_ingest_autoagent.artifacts import (
    HfCpuOracleAuthority,
    ares_plan_gate,
    artifact_consistency_gate,
    load_hf_cpu_oracle_evidence_files,
    target_plan_gate,
    validate_token_agreement_evidence,
)
from ares_ingest_autoagent.gates import shortcut_scan_gate


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
    "depth_performance": 0.95,
    "mmlu_pro": 0.98,
    "cpp_tvd": 0.99,
    "diagnostic_complete": 0.99,
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

FULL_GATES: tuple[str, ...] = (*BACKEND_GATES, "depth_performance", "mmlu_pro")
COMPARISON_GATES: tuple[str, ...] = (*FULL_GATES, "cpp_tvd")
STANDARD_GATES: tuple[str, ...] = CPU_ONLY_GATES

GATE_PROFILES: dict[str, tuple[str, ...]] = {
    "cpu-only": CPU_ONLY_GATES,
    "backend": BACKEND_GATES,
    # The fast production-candidate loop stays on HF CPU oracle artifacts and
    # the selected Ares backend. Recompute HF goldens only when the oracle tuple
    # changes; C++ comparison is an explicit late checkpoint.
    "full": FULL_GATES,
    "comparison": COMPARISON_GATES,
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
    "mmlu_pro": 0.05,
}

ARTIFACT_GATE_VALIDATORS: dict[str, str] = {
    "hf_cpu_oracle": "hf_cpu_oracle",
    "aresplan_valid": "ares_plan",
    "targetplan_valid": "target_plan",
    "artifact_consistency": "artifact_consistency",
    "shortcut_scan": "shortcut_scan",
    "backend_open": "backend_open",
    "one_token_logits": "one_token_logits",
    "eight_token_greedy": "eight_token_greedy",
    "cpp_tvd": "cpp_tvd",
    "depth_performance": "depth_performance",
    "mmlu_pro": "mmlu_pro",
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


def checked_in_gate_profile(required_gates: tuple[str, ...]) -> str | None:
    required_set = frozenset(required_gates)
    if len(required_set) != len(required_gates):
        return None
    return next(
        (
            name
            for name, profile_gates in GATE_PROFILES.items()
            if required_set == frozenset(profile_gates)
        ),
        None,
    )


def enforce_oracle_consistency_digest(
    gates: dict[str, Gate],
    required_gates: tuple[str, ...],
) -> None:
    if not {"hf_cpu_oracle", "artifact_consistency"}.issubset(required_gates):
        return
    oracle_gate = gates.get("hf_cpu_oracle")
    consistency_gate = gates.get("artifact_consistency")
    if (
        oracle_gate is None
        or not oracle_gate.passed
        or consistency_gate is None
        or not consistency_gate.passed
    ):
        return
    actual = (
        oracle_gate.detail.get("transaction_digest")
        if isinstance(oracle_gate.detail, dict)
        else None
    )
    consistency_validation = (
        consistency_gate.detail.get("detail")
        if isinstance(consistency_gate.detail, dict)
        else None
    )
    expected = (
        consistency_validation.get("oracle_transaction_digest")
        if isinstance(consistency_validation, dict)
        else None
    )
    if not isinstance(expected, str) or expected != actual:
        gates["artifact_consistency"] = Gate(
            False,
            0.0,
            {
                "artifact_validator": "artifact_consistency",
                "promotion_eligible": False,
                "claim_ceiling": "diagnostic_only_nonpromotion",
                "expected_oracle_transaction_digest": expected,
                "actual_oracle_transaction_digest": actual,
                "errors": [
                    "artifact consistency must bind the exact canonical HF CPU "
                    "oracle transaction used by reward scoring"
                ],
            },
        )


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


def _oracle_validation_gate(validation: Any) -> Gate:
    return Gate(
        passed=validation.passed,
        score=1.0 if validation.passed else 0.0,
        detail={
            **validation.detail,
            "errors": list(validation.errors),
        },
    )


def _score_oracle_records(payload: Any, dense_payload: Any) -> Gate:
    return Gate(
        False,
        0.0,
        {
            "artifact_validator": "hf_cpu_oracle",
            "transaction_validator": (
                "tools/oracles/hf-cpu/capture_hf_cpu_oracle.py:"
                "validate_capture_transaction"
            ),
            "provided_raw_payload": payload is not None,
            "provided_dense_payload": dense_payload is not None,
            "promotion_eligible": False,
            "claim_ceiling": "diagnostic_only_nonpromotion",
            "errors": [
                "live reward requires a canonically committed raw/dense "
                "transaction at retained file paths"
            ],
        },
    )


def _score_oracle_files(
    oracle_path: Path | None,
    dense_path: Path | None,
    *,
    expected_authority: HfCpuOracleAuthority | None,
    require_authority: bool,
) -> tuple[Gate, Any | None]:
    if oracle_path is None or dense_path is None:
        missing = "raw" if oracle_path is None else "dense"
        return (
            Gate(
                False,
                0.0,
                {
                    "artifact_validator": "hf_cpu_oracle",
                    "transaction_validator": (
                        "tools/oracles/hf-cpu/capture_hf_cpu_oracle.py:"
                        "validate_capture_transaction"
                    ),
                    "promotion_eligible": False,
                    "claim_ceiling": "diagnostic_only_nonpromotion",
                    "errors": [f"missing retained HF CPU oracle {missing} file"],
                },
            ),
            None,
        )
    evidence = load_hf_cpu_oracle_evidence_files(
        oracle_path,
        dense_path,
        expected_authority=expected_authority,
        require_authority=require_authority,
    )
    return (
        _oracle_validation_gate(evidence.validation),
        evidence,
    )


def _missing_promotion_evidence_gate(
    name: str,
    validator_name: str,
) -> dict[str, Any]:
    return {
        "artifact_validator": validator_name,
        "passed": False,
        "score": 0.0,
        "promotion_eligible": False,
        "claim_ceiling": "diagnostic_only_nonpromotion",
        "errors": [f"same-process promotion requires retained {name} evidence"],
    }


def _final_plan_gates(
    *,
    artifact_spec: Mapping[str, Any] | None,
    ares_plan_path: Path | None,
    target_plan_path: Path | None,
    oracle_evidence: Any | None,
    required_gates: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    validated: dict[str, dict[str, Any]] = {}
    if "aresplan_valid" in required_gates:
        validated["aresplan_valid"] = (
            ares_plan_gate(ares_plan_path)
            if ares_plan_path is not None
            else _missing_promotion_evidence_gate("AresPlan", "ares_plan")
        )
    if "targetplan_valid" in required_gates:
        validated["targetplan_valid"] = (
            target_plan_gate(target_plan_path)
            if target_plan_path is not None
            else _missing_promotion_evidence_gate("TargetPlan", "target_plan")
        )
    if "artifact_consistency" in required_gates:
        if artifact_spec is None:
            validated["artifact_consistency"] = _missing_promotion_evidence_gate(
                "model specification",
                "artifact_consistency",
            )
        else:
            oracle_validation = (
                oracle_evidence.validation if oracle_evidence is not None else None
            )
            validated["artifact_consistency"] = artifact_consistency_gate(
                artifact_spec,
                oracle_payload=(
                    oracle_evidence.oracle_records
                    if oracle_validation is not None and oracle_validation.passed
                    else None
                ),
                oracle_transaction_digest=(
                    oracle_validation.detail.get("transaction_digest")
                    if oracle_validation is not None and oracle_validation.passed
                    else None
                ),
                validated_gates=validated,
            )
    return validated


def _score_token_agreement_payload(
    payload: Any,
    *,
    base_dir: Path | None = None,
) -> Gate:
    validation = validate_token_agreement_evidence(payload, base_dir=base_dir)
    return Gate(
        passed=validation.passed,
        score=1.0 if validation.passed else 0.0,
        detail=validation.as_gate(
            label="eight-token greedy evidence",
            validator_name="eight_token_greedy",
        ),
    )


def compute_reward(
    *,
    gates_payload: Any = None,
    validated_gates_payload: Any = None,
    oracle_payload: Any = None,
    oracle_dense_payload: Any = None,
    oracle_path: Path | None = None,
    oracle_dense_path: Path | None = None,
    artifact_spec: Mapping[str, Any] | None = None,
    ares_plan_path: Path | None = None,
    target_plan_path: Path | None = None,
    authority_root: Path | None = None,
    expected_oracle_authority: HfCpuOracleAuthority | None = None,
    token_payload: Any = None,
    token_payload_base_dir: Path | None = None,
    performance_payload: Any = None,
    one_token_payload: Any = None,
    required_gates: tuple[str, ...] = STANDARD_GATES,
    promotion_authority: bool = False,
) -> dict[str, Any]:
    gates = extract_gates(gates_payload)
    reject_untrusted_artifact_gates(gates, required_gates)
    if validated_gates_payload is not None:
        merge_validated_gates(gates, validated_gates_payload)

    oracle_evidence = None
    require_oracle_authority = (
        promotion_authority
        and checked_in_gate_profile(required_gates) is not None
        and "hf_cpu_oracle" in required_gates
    )
    if oracle_path is not None or oracle_dense_path is not None:
        gates["hf_cpu_oracle"], oracle_evidence = _score_oracle_files(
            oracle_path,
            oracle_dense_path,
            expected_authority=expected_oracle_authority,
            require_authority=require_oracle_authority,
        )
    elif oracle_payload is not None or oracle_dense_payload is not None:
        gates["hf_cpu_oracle"] = _score_oracle_records(
            oracle_payload,
            oracle_dense_payload,
        )
    elif "hf_cpu_oracle" in required_gates:
        gates["hf_cpu_oracle"], oracle_evidence = _score_oracle_files(
            None,
            None,
            expected_authority=expected_oracle_authority,
            require_authority=require_oracle_authority,
        )

    if promotion_authority:
        for name, gate in _final_plan_gates(
            artifact_spec=artifact_spec,
            ares_plan_path=ares_plan_path,
            target_plan_path=target_plan_path,
            oracle_evidence=oracle_evidence,
            required_gates=required_gates,
        ).items():
            gates[name] = as_gate(gate)
        if "shortcut_scan" in required_gates:
            gates["shortcut_scan"] = as_gate(
                shortcut_scan_gate(authority_root)
                if authority_root is not None
                else _missing_promotion_evidence_gate(
                    "shortcut scan",
                    "shortcut_scan",
                )
            )

    enforce_oracle_consistency_digest(gates, required_gates)

    if one_token_payload is not None:
        gates["one_token_logits"] = _score_logit_payload(one_token_payload)

    if (
        token_payload is not None
        and "eight_token_greedy" in required_gates
        and "eight_token_greedy" not in gates
    ):
        gates["eight_token_greedy"] = _score_token_agreement_payload(
            token_payload,
            base_dir=token_payload_base_dir,
        )

    enforce_artifact_gate_evidence(gates, required_gates)

    gate_profile = checked_in_gate_profile(required_gates)
    first_failed = first_failed_gate(gates, required_gates)
    if first_failed == "complete" and (
        gate_profile is None or not promotion_authority
    ):
        first_failed = "diagnostic_complete"
    stage_cap = STAGE_CAPS.get(first_failed, 0.0)

    alpha_execution, alpha_components = compute_alpha_execution(gates, required_gates)
    tau_tokens = score_token_results(token_payload)
    token_gate = gates.get("eight_token_greedy")
    if "eight_token_greedy" in required_gates and (
        token_gate is None or not token_gate.passed
    ):
        tau_tokens = 0.0
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

    promotion_eligible = (
        promotion_authority
        and gate_profile is not None
        and first_failed == "complete"
    )

    return {
        "score": score,
        "raw_score": raw,
        "alpha_execution": alpha_execution,
        "tau_tokens": tau_tokens,
        "delta_inference": delta_inference,
        "stage_cap": stage_cap,
        "first_failed_gate": first_failed,
        "gate_profile": gate_profile,
        "required_gates_authority": (
            "checked_in_profile" if gate_profile is not None else "custom_diagnostic"
        ),
        "validation_authority": (
            "same_process" if promotion_authority else "diagnostic_receipt"
        ),
        "promotion_eligible": promotion_eligible,
        "claim_ceiling": (
            "promotion_candidate"
            if promotion_eligible
            else "diagnostic_only_nonpromotion"
        ),
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
    parser.add_argument("--oracle", type=Path, help="Canonical HF CPU raw JSONL")
    parser.add_argument(
        "--oracle-dense",
        type=Path,
        help="Canonical dense-logits JSONL committed by --oracle",
    )
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
        oracle_path=args.oracle,
        oracle_dense_path=args.oracle_dense,
        token_payload=read_json(args.tokens),
        token_payload_base_dir=args.tokens.parent if args.tokens is not None else None,
        performance_payload=read_json(args.performance),
        one_token_payload=read_json(args.one_token),
        required_gates=tuple(args.required_gates),
        promotion_authority=False,
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

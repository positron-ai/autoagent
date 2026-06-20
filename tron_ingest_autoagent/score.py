"""Score Tron ingest artifacts for Harbor.

The scorer is intentionally file-format tolerant. It consumes the JSON outputs
that already exist in the Tron ingest flow, plus optional gate, token, and
performance summaries, and emits the two files Harbor verifiers expect:

  /logs/reward.txt
  /logs/verifier/reward.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tron_ingest_autoagent.performance import is_scoring_workload, normalize_workload


STAGE_CAPS: dict[str, float] = {
    "not_started": 0.0,
    "fx_export": 0.05,
    "typedfx_parse": 0.15,
    "typedfx_logits": 0.35,
    "eqsat_structure": 0.45,
    "bulk_logits": 0.60,
    "cpp_compile": 0.70,
    "cpu_logits": 0.82,
    "fpga_logits": 0.92,
    "complete": 1.00,
}

STANDARD_GATES: tuple[str, ...] = (
    "fx_export",
    "typedfx_parse",
    "typedfx_logits",
    "eqsat_structure",
    "bulk_logits",
    "cpp_compile",
    "cpu_logits",
    "fpga_logits",
)

ALPHA_WEIGHTS: dict[str, float] = {
    "architecture": 0.10,
    "typedfx_logits": 0.15,
    "eqsat_structure": 0.15,
    "bulk_logits": 0.20,
    "cpp_compile": 0.10,
    "cpu_logits": 0.20,
    "fpga_logits": 0.10,
}


@dataclass(frozen=True)
class Gate:
    passed: bool
    score: float
    detail: Any = None


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if math.isnan(value):
        return low
    if math.isclose(value, low, rel_tol=0.0, abs_tol=1e-12):
        return low
    if math.isclose(value, high, rel_tol=0.0, abs_tol=1e-12):
        return high
    return min(max(value, low), high)


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
        raw_score = value.get("score")
        if raw_score is None:
            raw_score = value.get("value")
        if raw_score is None and "passed" in value:
            raw_score = 1.0 if value["passed"] else 0.0
        score = clamp(float(raw_score or 0.0))
        passed = bool(value.get("passed", score >= 1.0))
        return Gate(passed=passed, score=score, detail=value)
    return Gate(passed=False, score=0.0, detail=value)


def extract_gates(payload: Any) -> dict[str, Gate]:
    if payload is None:
        return {}
    if isinstance(payload, dict) and "gates" in payload:
        payload = payload["gates"]
    if not isinstance(payload, dict):
        raise TypeError(
            "gate payload must be a JSON object or an object with a 'gates' key"
        )
    return {name: as_gate(value) for name, value in payload.items()}


def merge_inferred_gate(
    gates: dict[str, Gate],
    name: str,
    gate: Gate,
) -> None:
    existing = gates.get(name)
    if existing is None:
        gates[name] = gate
        return
    if existing.passed is False:
        return
    gates[name] = gate


def _score_logit_case(case: dict[str, Any], tvd_threshold: float) -> float:
    if case.get("functionally_equivalent") is True:
        tvd_score = 1.0
    else:
        max_tvd = float(case.get("max_tvd", 1.0) or 1.0)
        tvd_score = clamp(1.0 - ((max_tvd - tvd_threshold) / (1.0 - tvd_threshold)))

    top1 = clamp(float(case.get("top1_agreement", 0.0) or 0.0))
    top5 = clamp(float(case.get("top5_agreement", 0.0) or 0.0))
    cosine = float(case.get("cosine_similarity", -1.0) or -1.0)
    cosine_score = clamp((cosine + 1.0) / 2.0)

    return clamp(0.50 * tvd_score + 0.25 * top1 + 0.15 * top5 + 0.10 * cosine_score)


def score_logit_results(payload: Any) -> Gate:
    if not isinstance(payload, dict):
        return Gate(False, 0.0, {"error": "missing logit payload"})

    results = payload.get("results")
    if not isinstance(results, list) or not results:
        passed = bool(payload.get("all_functionally_equivalent", False))
        return Gate(passed=passed, score=1.0 if passed else 0.0, detail=payload)

    threshold = float(payload.get("tvd_threshold", 0.01) or 0.01)
    scores = [
        _score_logit_case(case, threshold)
        for case in results
        if isinstance(case, dict) and case.get("functionally_equivalent") is not None
    ]
    if not scores:
        return Gate(False, 0.0, payload)

    score = sum(scores) / len(scores)
    passed = bool(payload.get("all_functionally_equivalent", False))
    return Gate(passed=passed, score=score, detail=payload)


def _find_numeric(payload: Any, names: tuple[str, ...]) -> float | None:
    if isinstance(payload, dict):
        for name in names:
            value = payload.get(name)
            if isinstance(value, (int, float)):
                return float(value)
        for value in payload.values():
            found = _find_numeric(value, names)
            if found is not None:
                return found
    if isinstance(payload, list):
        values = [_find_numeric(item, names) for item in payload]
        numbers = [value for value in values if value is not None]
        if numbers:
            return sum(numbers) / len(numbers)
    return None


def _find_workload(payload: Any) -> str | None:
    if isinstance(payload, dict):
        for name in (
            "workload",
            "workload_type",
            "benchmark_category",
            "category",
        ):
            workload = normalize_workload(payload.get(name))
            if workload is not None:
                return workload
    return None


def score_token_results(payload: Any) -> float:
    if payload is None:
        return 0.0
    if isinstance(payload, (int, float)):
        return clamp(float(payload))
    if not isinstance(payload, dict):
        return 0.0
    direct = _find_numeric(payload, ("tau", "score", "token_score", "token_agreement"))
    if direct is not None:
        return clamp(direct)

    exact = _find_numeric(
        payload, ("exact_match", "exact_token_match", "exact_fraction")
    )
    prefix = _find_numeric(
        payload, ("prefix_match", "prefix_fraction", "exact_prefix_fraction")
    )
    edit = _find_numeric(payload, ("edit_similarity", "sequence_similarity"))
    top1 = _find_numeric(payload, ("top1_agreement",))

    components: list[tuple[float, float]] = []
    if exact is not None:
        components.append((0.40, clamp(exact)))
    if prefix is not None:
        components.append((0.25, clamp(prefix)))
    if edit is not None:
        components.append((0.20, clamp(edit)))
    if top1 is not None:
        components.append((0.15, clamp(top1)))
    if not components:
        return 0.0

    total_weight = sum(weight for weight, _ in components)
    return clamp(sum(weight * value for weight, value in components) / total_weight)


def score_performance_results(payload: Any) -> float:
    if payload is None:
        return 0.0
    if isinstance(payload, (int, float)):
        return clamp(float(payload))
    if not isinstance(payload, dict):
        return 0.0
    if not is_scoring_workload(_find_workload(payload)):
        return 0.0

    direct = _find_numeric(payload, ("delta", "score", "performance_score"))
    if direct is not None:
        return clamp(direct)

    measured = _find_numeric(
        payload,
        (
            "measured_tokens_per_second",
            "measured_tps",
            "tokens_per_second",
            "throughput_tps",
            "throughput",
        ),
    )
    limit = _find_numeric(
        payload,
        (
            "speed_of_light_tokens_per_second",
            "speed_of_light_tps",
            "theoretical_tokens_per_second",
            "theoretical_tps",
            "speed_of_light",
        ),
    )
    if measured is None or limit is None or limit <= 0:
        return 0.0
    return clamp(measured / limit)


def infer_prerequisites(gates: dict[str, Gate], through: str) -> None:
    for name in STANDARD_GATES:
        if name == through:
            return
        gates.setdefault(name, Gate(True, 1.0, {"inferred": True}))


def first_failed_gate(gates: dict[str, Gate], required: tuple[str, ...]) -> str:
    if not gates:
        return "not_started"
    for name in required:
        gate = gates.get(name)
        if gate is None or not gate.passed:
            return name
    return "complete"


def compute_alpha(gates: dict[str, Gate]) -> tuple[float, dict[str, float]]:
    components: dict[str, float] = {}
    for name in ALPHA_WEIGHTS:
        gate = gates.get(name)
        components[name] = gate.score if gate is not None else 0.0
    alpha = sum(ALPHA_WEIGHTS[name] * components[name] for name in ALPHA_WEIGHTS)
    return clamp(alpha), components


def compute_reward(
    *,
    gates_payload: Any = None,
    architecture_payload: Any = None,
    structure_payload: Any = None,
    typedfx_payload: Any = None,
    bulk_payload: Any = None,
    token_payload: Any = None,
    performance_payload: Any = None,
    required_gates: tuple[str, ...] = STANDARD_GATES,
) -> dict[str, Any]:
    gates = extract_gates(gates_payload)

    if architecture_payload is not None:
        merge_inferred_gate(gates, "architecture", as_gate(architecture_payload))

    if structure_payload is not None:
        merge_inferred_gate(gates, "eqsat_structure", as_gate(structure_payload))

    if typedfx_payload is not None:
        typedfx_gate = score_logit_results(typedfx_payload)
        infer_prerequisites(gates, "typedfx_logits")
        merge_inferred_gate(gates, "typedfx_logits", typedfx_gate)

    if bulk_payload is not None:
        bulk_gate = score_logit_results(bulk_payload)
        infer_prerequisites(gates, "bulk_logits")
        merge_inferred_gate(gates, "bulk_logits", bulk_gate)

    first_failed = first_failed_gate(gates, required_gates)
    stage_cap = STAGE_CAPS[first_failed]

    alpha, alpha_components = compute_alpha(gates)
    tau = score_token_results(token_payload)
    delta = score_performance_results(performance_payload)

    raw = clamp(0.70 * alpha + 0.20 * tau + 0.10 * delta)
    score = min(raw, stage_cap)

    return {
        "score": score,
        "raw_score": raw,
        "alpha": alpha,
        "tau": tau,
        "delta": delta,
        "stage_cap": stage_cap,
        "first_failed_gate": first_failed,
        "gates": {
            name: {
                "passed": gate.passed,
                "score": gate.score,
            }
            for name, gate in sorted(gates.items())
        },
        "alpha_components": alpha_components,
        "weights": {
            "alpha": 0.70,
            "tau": 0.20,
            "delta": 0.10,
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
    parser.add_argument(
        "--gates", type=Path, help="JSON file containing explicit gates"
    )
    parser.add_argument("--architecture", type=Path, help="Architecture analysis JSON")
    parser.add_argument(
        "--eqsat-structure", type=Path, help="EqSat structural analysis JSON"
    )
    parser.add_argument(
        "--typedfx-logits", type=Path, help="TypedFx logit_results.json"
    )
    parser.add_argument("--bulk-logits", type=Path, help="Bulk bulk_logit_results.json")
    parser.add_argument("--tokens", type=Path, help="Token agreement JSON")
    parser.add_argument(
        "--performance", type=Path, help="Performance/speed-of-light JSON"
    )
    parser.add_argument(
        "--gate",
        action="append",
        default=[],
        type=parse_gate_override,
        metavar="NAME=VALUE",
        help="Add or override a gate, e.g. cpp_compile=true or cpu_logits=0.8",
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
    parser.add_argument(
        "--print-json", action="store_true", help="Print reward JSON to stdout"
    )
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
        architecture_payload=read_json(args.architecture),
        structure_payload=read_json(args.eqsat_structure),
        typedfx_payload=read_json(args.typedfx_logits),
        bulk_payload=read_json(args.bulk_logits),
        token_payload=read_json(args.tokens),
        performance_payload=read_json(args.performance),
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

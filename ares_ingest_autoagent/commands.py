"""Command wrapper plans for Ares runtime and comparison evidence."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


DEFAULT_PROMPT = "In one sentence, explain why attention caches matter."
DEFAULT_TIMEOUT_SEC = 3600
COMPARISON_TIMEOUT_SEC = 24 * 3600
MMLU_PRO_TIMEOUT_SEC = 24 * 3600
BACKEND_GATES = {"backend_open", "one_token_logits", "eight_token_greedy"}
COMPARISON_GATES = {"cpp_tvd"}
MMLU_PRO_GATES = {"mmlu_pro"}


@dataclass(frozen=True)
class CommandWrapper:
    name: str
    gate: str
    command: str
    cwd: str
    timeout_sec: int
    evidence_class: str
    evidence_outputs: Mapping[str, str]
    enabled: bool
    missing_inputs: tuple[str, ...] = ()
    promotion_eligible: bool = False
    notes: tuple[str, ...] = ()
    pass_regexes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "gate": self.gate,
            "command": self.command,
            "cwd": self.cwd,
            "timeout_sec": self.timeout_sec,
            "evidence_class": self.evidence_class,
            "evidence_outputs": dict(self.evidence_outputs),
            "enabled": self.enabled,
            "promotion_eligible": self.promotion_eligible,
            "missing_inputs": list(self.missing_inputs),
            "notes": list(self.notes),
        }
        if self.pass_regexes:
            payload["pass_regexes"] = list(self.pass_regexes)
        return payload

    def as_command_gate(self) -> dict[str, Any]:
        gate: dict[str, Any] = {
            "name": self.name,
            "command": self.command,
            "cwd": self.cwd,
            "timeout_sec": self.timeout_sec,
        }
        if self.pass_regexes:
            gate["pass_regexes"] = list(self.pass_regexes)
        return gate


def build_command_wrapper_plan(
    spec: Mapping[str, Any],
    *,
    run_dir: Path,
    ares_repo: Path,
) -> dict[str, Any]:
    """Build opt-in command wrappers for runtime/comparison evidence capture."""

    required = {str(gate) for gate in spec.get("required_gates", [])}
    wrapper_cfg = _object(spec.get("command_wrapper_config"))
    config_errors: list[str] = []
    dry_run = _bool(
        wrapper_cfg.get("dry_run", True),
        default=True,
        name="command_wrapper_config.dry_run",
        errors=config_errors,
    )
    execute = _bool(
        spec.get("execute_command_wrappers", False),
        default=False,
        name="execute_command_wrappers",
        errors=config_errors,
    )
    if config_errors:
        execute = False

    wrappers: list[CommandWrapper] = []
    if required.intersection(BACKEND_GATES) or wrapper_cfg.get("backend"):
        wrappers.extend(
            [
                rinzler_chat_wrapper(spec, run_dir=run_dir, dry_run=dry_run),
                full_inference_smoke_wrapper(spec, run_dir=run_dir, dry_run=dry_run),
            ]
        )
    if required.intersection(COMPARISON_GATES) or wrapper_cfg.get("comparison"):
        wrappers.append(
            side_by_side_comparison_wrapper(
                spec,
                run_dir=run_dir,
                ares_repo=ares_repo,
                dry_run=dry_run,
            )
        )
    if required.intersection(MMLU_PRO_GATES) or wrapper_cfg.get("mmlu_pro"):
        wrappers.append(
            mmlu_pro_wrapper(
                spec,
                run_dir=run_dir,
                ares_repo=ares_repo,
                dry_run=dry_run,
            )
        )

    return {
        "schema": "ares.autoagent.command_wrappers.v1",
        "model": spec.get("model"),
        "backend": spec.get("backend", "tron"),
        "dry_run": dry_run,
        "execute_command_wrappers": execute,
        "config_errors": config_errors,
        "wrappers": [wrapper.as_dict() for wrapper in wrappers],
        "command_gates": [
            wrapper.as_command_gate()
            for wrapper in wrappers
            if wrapper.enabled and execute
        ],
    }


def rinzler_chat_wrapper(
    spec: Mapping[str, Any],
    *,
    run_dir: Path,
    dry_run: bool = True,
) -> CommandWrapper:
    model = _string(spec.get("model"))
    backend = _string(spec.get("backend")) or "tron"
    weights = _string(_first(spec, "weights", "checkpoint", "model_weights"))
    ares_plan = _string(spec.get("ares_plan"))
    prompt = _prompt(spec)
    max_tokens = int(_first(spec, "max_tokens", "one_token_max_tokens") or 1)
    summary = run_dir / "artifacts" / "rinzler-chat-summary.json"
    dense_logits = run_dir / "artifacts" / "rinzler-chat-dense-logits.jsonl"

    missing = _missing(
        {
            "model": model,
            "weights": weights,
            "ares_plan": ares_plan,
        }
    )
    args = [
        "bin/ares-rinzler-chat",
        "--model",
        model or "<model>",
        "--weights",
        weights or "<weights>",
        "--max-tokens",
        str(max_tokens),
        "--min-tokens",
        "1",
        "--summary",
        str(summary),
        "--dense-logits-jsonl",
        str(dense_logits),
        "--keep-artifacts",
    ]
    if backend == "cpu":
        args.append("--cpu")
    else:
        args.extend(["--backend", backend])
    if ares_plan:
        args.extend(["--ares-plan", ares_plan])
    if dry_run:
        args.append("--dry-run")
    args.extend(["--", prompt])

    notes = [
        "Produces Ares system-under-test runtime artifacts only.",
        "Post-process dense logits against HF CPU oracle before scoring one_token_logits.",
        "Summary output alone is not validator-backed backend_open_evidence.",
        "This launcher does not consume target_plan directly; attach TargetPlan validator evidence separately before scoring backend gates.",
    ]
    if ares_plan:
        notes.append(
            "When this script stages a TargetPlan from --ares-plan, the result is diagnostic until evidence proves runtime_generated_sidecars=false."
        )
    return CommandWrapper(
        name="rinzler_chat_one_token",
        gate="one_token_logits",
        command=shlex.join(args),
        cwd="ares_repo",
        timeout_sec=DEFAULT_TIMEOUT_SEC,
        evidence_class="system_under_test",
        evidence_outputs={
            "summary": str(summary),
            "dense_logits_jsonl": str(dense_logits),
        },
        enabled=not missing,
        missing_inputs=missing,
        notes=tuple(notes),
        pass_regexes=("rinzler full inference dry run",) if dry_run else (),
    )


def full_inference_smoke_wrapper(
    spec: Mapping[str, Any],
    *,
    run_dir: Path,
    dry_run: bool = True,
) -> CommandWrapper:
    model = _string(spec.get("model"))
    backend = _string(spec.get("backend")) or "tron"
    weights = _string(_first(spec, "weights", "checkpoint", "model_weights"))
    ares_plan = _string(spec.get("ares_plan"))
    prompt = _prompt(spec)
    max_tokens = int(_first(spec, "max_tokens", "smoke_max_tokens") or 8)
    summary = run_dir / "artifacts" / "full-inference-summary.json"
    dense_logits = run_dir / "artifacts" / "full-inference-dense-logits.jsonl"

    missing = _missing(
        {
            "model": model,
            "weights": weights,
            "ares_plan": ares_plan,
        }
    )
    env = {
        "ARES_RINZLER_FULL_INFERENCE_MODEL": model or "<model>",
        "ARES_RINZLER_FULL_INFERENCE_WEIGHTS": weights or "<weights>",
        "ARES_RINZLER_FULL_INFERENCE_BACKEND": backend,
        "ARES_RINZLER_FULL_INFERENCE_PROMPT": prompt,
        "ARES_RINZLER_FULL_INFERENCE_MAX_TOKENS": str(max_tokens),
        "ARES_RINZLER_FULL_INFERENCE_MIN_TOKENS": "1",
        "ARES_RINZLER_FULL_INFERENCE_SUMMARY_JSON": str(summary),
        "ARES_RINZLER_FULL_INFERENCE_DENSE_LOGITS_JSONL": str(dense_logits),
        "ARES_RINZLER_FULL_INFERENCE_KEEP_ARTIFACTS": "1",
    }
    if ares_plan:
        env["ARES_RINZLER_FULL_INFERENCE_ARES_PLAN"] = ares_plan
    if dry_run:
        env["ARES_RINZLER_FULL_INFERENCE_DRY_RUN"] = "1"

    command = shlex.join(
        ["env", *_env_pairs(env), "bin/ci/ci-ares-rinzler-full-inference-smoke.sh"]
    )
    return CommandWrapper(
        name="rinzler_full_inference_smoke",
        gate="backend_open",
        command=command,
        cwd="ares_repo",
        timeout_sec=DEFAULT_TIMEOUT_SEC,
        evidence_class="system_under_test",
        evidence_outputs={
            "summary": str(summary),
            "dense_logits_jsonl": str(dense_logits),
        },
        enabled=not missing,
        missing_inputs=missing,
        notes=(
            "Launches the same smoke wrapper used by bin/ares-rinzler-chat.",
            "Its summary must be transformed into validator-backed backend_open or token evidence before scoring.",
            "This launcher does not consume target_plan directly; attach TargetPlan validator evidence separately before scoring backend gates.",
        ),
        pass_regexes=("rinzler full inference dry run",) if dry_run else (),
    )


def side_by_side_comparison_wrapper(
    spec: Mapping[str, Any],
    *,
    run_dir: Path,
    ares_repo: Path,
    dry_run: bool = True,
) -> CommandWrapper:
    model = _string(spec.get("model"))
    backend = _string(spec.get("backend")) or "tron"
    weights = _string(_first(spec, "weights", "checkpoint", "model_weights"))
    comparison = _object(spec.get("comparison"))
    cpp_bin = _string(_first(comparison, "cpp_rinzler_bin", "cpp_bin"))
    rust_model_path = _string(_first(comparison, "rust_model_path", "ares_plan_bundle"))
    output_dir = run_dir / "artifacts" / "side-by-side"
    cpp_logits = output_dir / "cpp-dense-logits.jsonl"
    rust_logits = output_dir / "ares-dense-logits.jsonl"
    events_dir = output_dir / "backend-events"

    missing = _missing(
        {
            "model": model,
            "weights": weights,
            "comparison.cpp_rinzler_bin": cpp_bin,
            "comparison.rust_model_path": rust_model_path,
        }
    )
    env = {
        "ARES_RINZLER_COMPARE_MODEL": model or "<model>",
        "ARES_RINZLER_COMPARE_WEIGHTS": weights or "<weights>",
        "ARES_RINZLER_COMPARE_RUST_BACKEND": backend,
        "ARES_RINZLER_COMPARE_OUTPUT_DIR": str(output_dir),
        "ARES_RINZLER_COMPARE_RUST_LOGITS_JSONL": str(rust_logits),
        "ARES_RINZLER_COMPARE_CPP_LOGITS_JSONL": str(cpp_logits),
        "ARES_RINZLER_COMPARE_RUST_BACKEND_EVENTS_DIR": str(events_dir),
        "ARES_RINZLER_COMPARE_REQUIRE_TVD": "1",
        "ARES_RINZLER_COMPARE_TVD_THRESHOLD": str(
            comparison.get("tvd_threshold", spec.get("tvd_threshold", 0.01))
        ),
    }
    if cpp_bin:
        env["CPP_RINZLER_BIN"] = cpp_bin
    if rust_model_path:
        env["ARES_RINZLER_COMPARE_RUST_MODEL_PATH"] = rust_model_path
    if dry_run:
        env["ARES_RINZLER_COMPARE_DRY_RUN"] = "1"

    command = shlex.join(
        ["env", *_env_pairs(env), "bin/ci/ci-rinzler-fpga-vs-tron-comparison.sh"]
    )
    return CommandWrapper(
        name="cpp_rinzler_side_by_side",
        gate="cpp_tvd",
        command=command,
        cwd=str(ares_repo),
        timeout_sec=COMPARISON_TIMEOUT_SEC,
        evidence_class="comparison",
        evidence_outputs={
            "output_dir": str(output_dir),
            "cpp_dense_logits_jsonl": str(cpp_logits),
            "ares_dense_logits_jsonl": str(rust_logits),
            "backend_events_dir": str(events_dir),
        },
        enabled=not missing,
        missing_inputs=missing,
        notes=(
            "C++ Tron/Rinzler output is comparison and rollback evidence only.",
            "This wrapper never supplies HF CPU oracle correctness evidence.",
            "Do not run this slow lane in the normal debug loop; attach it only after HF-backed Ares backend quality and performance are competitive enough to justify milestone comparison.",
        ),
        pass_regexes=("dry-run",) if dry_run else (),
    )


def mmlu_pro_wrapper(
    spec: Mapping[str, Any],
    *,
    run_dir: Path,
    ares_repo: Path,
    dry_run: bool = True,
) -> CommandWrapper:
    model = _string(_first(spec, "mmlu_model", "model"))
    backend = _string(spec.get("backend")) or "tron"
    mmlu_cfg = _object(spec.get("mmlu_pro"))
    openai_host = _string(
        _first(mmlu_cfg, "openai_host", "endpoint")
        or _first(spec, "openai_host", "endpoint")
    )
    coverage = _first(
        mmlu_cfg,
        "coverage_percent",
        "coverage",
        "mmlu_coverage",
    ) or _first(spec, "mmlu_coverage", "coverage_percent")
    max_retries = _first(mmlu_cfg, "max_retries", "mmlu_max_retries") or 10
    output_dir = run_dir / "artifacts" / "mmlu-pro"

    missing = _missing(
        {
            "model": model,
            "openai_host": openai_host,
            "coverage": str(coverage) if coverage is not None else None,
        }
    )
    env = {
        "OPENAI_HOST": openai_host or "<openai_host>",
        "SKIP_PROVISION": "1",
        "MMLU_MODEL": model or "<model>",
        "MMLU_COVERAGE": str(coverage) if coverage is not None else "<coverage>",
        "MMLU_MAX_RETRIES": str(max_retries),
    }

    run_mmlu = shlex.join(["env", *_env_pairs(env), "uv", "run", "mmlu_pro"])
    if dry_run:
        command = shlex.join(["printf", "%s\n", f"MMLU Pro dry run: {run_mmlu}"])
    else:
        command = " && ".join(
            [
                shlex.join(["mkdir", "-p", str(output_dir)]),
                shlex.join(["rm", "-rf", "eval_results", "mmlu_output"]),
                run_mmlu,
                shlex.join(
                    [
                        "rm",
                        "-rf",
                        str(output_dir / "eval_results"),
                        str(output_dir / "mmlu_output"),
                    ]
                ),
                shlex.join(
                    ["cp", "-R", "eval_results", str(output_dir / "eval_results")]
                ),
                shlex.join(
                    ["cp", "-R", "mmlu_output", str(output_dir / "mmlu_output")]
                ),
            ]
        )
    return CommandWrapper(
        name="systems_test_mmlu_pro",
        gate="mmlu_pro",
        command=command,
        cwd=str(ares_repo / "third_party" / "systems_test"),
        timeout_sec=MMLU_PRO_TIMEOUT_SEC,
        evidence_class="system_under_test",
        evidence_outputs={
            "eval_results": str(output_dir / "eval_results"),
            "mmlu_output": str(output_dir / "mmlu_output"),
        },
        enabled=not missing,
        missing_inputs=missing,
        notes=(
            f"Runs MMLU Pro against the selected Ares {backend} endpoint.",
            "Wrapper command is dry-run unless command_wrapper_config.dry_run is false.",
            "Verify /v1/models first and use the exact served model id.",
            "MMLU_MODEL must match a hardcoded scripts/mmlu_pro.py config entry; otherwise systems_test skips the run.",
            "Add and commit a systems_test config entry for new model ids before using promotion evidence.",
            "Use SKIP_PROVISION=1 for Ares-owned endpoints.",
            "OPENAI_TOKEN is inherited from the caller environment when the endpoint requires it.",
            "For provisioned DUT mode, keep the -tpN suffix in the model id.",
            "Retain eval_results and mmlu_output under the run artifacts directory.",
            "Post-process outputs into validator-backed mmlu_pro_evidence before scoring.",
        ),
        pass_regexes=("MMLU Pro dry run",) if dry_run else ("MMLU",),
    )


def command_gates_from_plan(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": wrapper["name"],
            "command": wrapper["command"],
            "cwd": wrapper["cwd"],
            "timeout_sec": wrapper["timeout_sec"],
            **(
                {"pass_regexes": wrapper["pass_regexes"]}
                if wrapper.get("pass_regexes")
                else {}
            ),
        }
        for wrapper in plan.get("wrappers", [])
        if wrapper.get("enabled") and not wrapper.get("missing_inputs")
    ]


def _object(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _first(mapping: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        value = mapping.get(name)
        if value:
            return value
    return None


def _prompt(spec: Mapping[str, Any]) -> str:
    value = spec.get("prompt")
    if isinstance(value, str) and value:
        return value
    prompts = spec.get("prompts")
    if isinstance(prompts, list):
        for entry in prompts:
            if isinstance(entry, str) and entry:
                return entry
            if isinstance(entry, Mapping) and isinstance(entry.get("text"), str):
                return entry["text"]
    return DEFAULT_PROMPT


def _missing(fields: Mapping[str, str | None]) -> tuple[str, ...]:
    return tuple(name for name, value in fields.items() if not value)


def _bool(
    value: Any,
    *,
    default: bool,
    name: str,
    errors: list[str],
) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    errors.append(f"{name} must be boolean-like true/false, got {value!r}")
    return default


def _env_pairs(env: Mapping[str, str]) -> list[str]:
    return [f"{name}={value}" for name, value in sorted(env.items())]

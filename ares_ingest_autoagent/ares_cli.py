"""Create and manage Ares AutoAgent model-ingest run directories."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ares_ingest_autoagent.artifacts import (
    ares_plan_gate,
    backend_open_gate,
    cpp_tvd_gate,
    depth_performance_gate,
    one_token_logits_gate,
    target_plan_gate,
)
from ares_ingest_autoagent.commands import build_command_wrapper_plan
from ares_ingest_autoagent.gates import shortcut_scan_gate, write_json
from ares_ingest_autoagent.score import (
    GATE_PROFILES,
    compute_reward,
    required_gates_for_profile,
)


DEFAULT_REFINER_COMMAND = (
    "codex exec --dangerously-bypass-approvals-and-sandbox "
    '-C "$ARES_REPO" --add-dir "$AUTOAGENT_REPO" - < "$REFINEMENT_PROMPT"'
)


@dataclass
class AresIngestConfig:
    model_slug: str
    safe_model: str
    ares_repo: Path
    autoagent_root: Path
    run_dir: Path
    logs_dir: Path
    state_path: Path
    model_spec_path: Path
    target_score: float
    max_iterations: int | None
    stall_patience: int
    min_improvement: float
    refinement_command: str | None
    gate_profile: str
    setup_only: bool


class AresIngestError(RuntimeError):
    """Expected Ares AutoAgent command failure."""


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower())
    slug = re.sub(r"-+", "-", slug).strip("-")
    if not slug:
        raise ValueError(f"cannot derive safe model name from {text!r}")
    return slug


def default_run_dir(ares_repo: Path, safe_model: str) -> Path:
    return ares_repo / ".autoagent" / "ares-ingest" / safe_model / utc_stamp()


def build_model_spec(cfg: AresIngestConfig) -> dict[str, Any]:
    return {
        "model": cfg.model_slug,
        "safe_model": cfg.safe_model,
        "ares_repo": str(cfg.ares_repo),
        "frontend": "fx",
        "backend": "tron",
        "gate_profile": cfg.gate_profile,
        "required_gates": list(required_gates_for_profile(cfg.gate_profile)),
        "explicit_gates": {
            "model_spec": {
                "passed": True,
                "score": 1.0,
                "source": "ares-ingest-agent setup",
            }
        },
        "oracle": {
            "required_source": "huggingface_transformers_pytorch_cpu",
            "required_record_kind": "hf_cpu_oracle_capture",
        },
        "policy": {
            "no_hand_authored_rust_model_plugins": True,
            "no_runtime_generated_execution_sidecars": True,
            "runtime_path": "frontend -> Lean ingest -> AresPlan -> TargetPlan -> backend provider",
        },
    }


def reward_fingerprint(reward: dict[str, Any]) -> dict[str, Any]:
    gates = reward.get("gates", {})
    return {
        "score": reward.get("score"),
        "alpha_execution": reward.get("alpha_execution"),
        "tau_tokens": reward.get("tau_tokens"),
        "delta_inference": reward.get("delta_inference"),
        "stage_cap": reward.get("stage_cap"),
        "first_failed_gate": reward.get("first_failed_gate"),
        "gates": {
            name: {
                "passed": detail.get("passed"),
                "score": detail.get("score"),
            }
            for name, detail in gates.items()
            if isinstance(detail, dict)
        },
    }


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise AresIngestError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AresIngestError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AresIngestError(f"{path} must contain a JSON object")
    return payload


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"history": []}
    return load_json(path)


def load_state_or_recover(path: Path) -> dict[str, Any]:
    try:
        return load_state(path)
    except AresIngestError as state_error:
        return {
            "history": [],
            "previous_state_error": str(state_error),
        }


def resolve_run_path(path: str, cfg: AresIngestConfig) -> Path:
    raw = Path(path)
    if raw.is_absolute():
        return raw
    for base in (cfg.run_dir, cfg.ares_repo):
        candidate = base / raw
        if candidate.exists():
            return candidate
    return cfg.run_dir / raw


def read_json_or_jsonl(path: Path) -> Any:
    try:
        text = path.read_text()
    except FileNotFoundError as exc:
        raise AresIngestError(f"missing JSON file: {path}") from exc
    if path.suffix == ".jsonl":
        rows = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise AresIngestError(
                    f"invalid JSONL in {path}:{lineno}: {exc}"
                ) from exc
        return rows
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise AresIngestError(f"invalid JSON in {path}: {exc}") from exc


def write_handoff(
    cfg: AresIngestConfig,
    reward: dict[str, Any],
    *,
    state: Mapping[str, Any] | None = None,
) -> Path:
    path = cfg.run_dir / "handoff.md"
    history_count = len(state.get("history", [])) if isinstance(state, Mapping) else 0
    status = (
        state.get("status") if isinstance(state, Mapping) else "initialized_setup_only"
    )
    latest_prompt = (
        state.get("latest_refinement_prompt") if isinstance(state, Mapping) else None
    )
    text = f"""# Ares AutoAgent Run Handoff

## Objective

Bring `{cfg.model_slug}` into Ares through the generated pipeline.

## Current Reward

```json
{json.dumps(reward_fingerprint(reward), indent=2)}
```

Status: `{status}`

History entries: `{history_count}`

Latest refinement prompt: `{latest_prompt or "none"}`

## Rules

- HF Transformers on PyTorch CPU is the model-correctness oracle.
- Ares/Rust output is system-under-test evidence.
- C++ Tron/Rinzler is comparison and rollback evidence only.
- Runtime execution must flow through frontend artifacts, Lean ingest,
  generated AresPlan, Lean TargetPlan, and a backend provider.
- Do not add hand-authored Rust model plugins or runtime-generated plan
  sidecars.

## Next Action

Work the first failing gate: `{reward["first_failed_gate"]}`.
"""
    path.write_text(text)
    return path


def run_command(
    command: str,
    *,
    cfg: AresIngestConfig,
    env: Mapping[str, str],
    log: Path,
    timeout: int | None = None,
) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    merged_env.update({key: str(value) for key, value in env.items()})
    with log.open("w", encoding="utf-8") as out:
        out.write("+ " + command + "\n")
        out.flush()
        try:
            proc = subprocess.run(
                ["bash", "-lc", command],
                cwd=cfg.ares_repo,
                env=merged_env,
                stdout=out,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            out.write(f"\nTIMEOUT after {timeout} seconds\n")
            raise AresIngestError(f"refiner timed out; see {log}") from exc
    return proc.returncode


def config_from_args(args: argparse.Namespace) -> AresIngestConfig:
    ares_repo = Path(args.ares_repo).resolve()
    safe_model = slugify(args.model)
    run_dir = (
        Path(args.run_dir).resolve()
        if args.run_dir
        else default_run_dir(ares_repo, safe_model)
    )
    return AresIngestConfig(
        model_slug=args.model,
        safe_model=safe_model,
        ares_repo=ares_repo,
        autoagent_root=Path(__file__).resolve().parents[1],
        run_dir=run_dir,
        logs_dir=run_dir / "logs",
        state_path=run_dir / "state.json",
        model_spec_path=run_dir / "model_spec.json",
        target_score=args.target_score,
        max_iterations=None
        if args.max_iterations is None or args.max_iterations <= 0
        else args.max_iterations,
        stall_patience=max(1, args.stall_patience),
        min_improvement=max(0.0, args.min_improvement),
        refinement_command=None if args.no_refiner else args.refinement_command,
        gate_profile=args.gate_profile,
        setup_only=args.setup_only,
    )


def evaluate_run(cfg: AresIngestConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    spec = (
        load_json(cfg.model_spec_path)
        if cfg.model_spec_path.exists()
        else build_model_spec(cfg)
    )
    spec.setdefault("model", cfg.model_slug)
    spec.setdefault("safe_model", cfg.safe_model)
    spec.setdefault("ares_repo", str(cfg.ares_repo))
    spec.setdefault("frontend", "fx")
    spec.setdefault("backend", "tron")
    spec.setdefault("gate_profile", cfg.gate_profile)
    spec.setdefault(
        "required_gates", list(required_gates_for_profile(cfg.gate_profile))
    )
    if not isinstance(spec["required_gates"], list) or not all(
        isinstance(gate, str) for gate in spec["required_gates"]
    ):
        raise AresIngestError("model_spec required_gates must be a list of strings")
    explicit_gates = spec.setdefault("explicit_gates", {})
    if not isinstance(explicit_gates, dict):
        raise AresIngestError("model_spec explicit_gates must be a JSON object")
    explicit_gates.setdefault(
        "model_spec",
        {
            "passed": True,
            "score": 1.0,
            "source": "ares-ingest-agent setup",
        },
    )
    validated_gates: dict[str, Any] = {}
    validated_gates["shortcut_scan"] = shortcut_scan_gate(cfg.ares_repo)

    oracle_payload = None
    if oracle_spec := spec.get("oracle_records"):
        oracle_payload = read_json_or_jsonl(resolve_run_path(str(oracle_spec), cfg))
    if ares_plan := spec.get("ares_plan"):
        validated_gates["aresplan_valid"] = ares_plan_gate(
            resolve_run_path(str(ares_plan), cfg)
        )
    if target_plan := spec.get("target_plan"):
        validated_gates["targetplan_valid"] = target_plan_gate(
            resolve_run_path(str(target_plan), cfg)
        )
    if backend_open := spec.get("backend_open_evidence"):
        validated_gates["backend_open"] = backend_open_gate(
            resolve_run_path(str(backend_open), cfg)
        )
    if one_token := spec.get("one_token_logits_evidence") or spec.get(
        "one_token_results_json"
    ):
        validated_gates["one_token_logits"] = one_token_logits_gate(
            resolve_run_path(str(one_token), cfg)
        )
    if cpp_tvd := spec.get("cpp_tvd_evidence"):
        validated_gates["cpp_tvd"] = cpp_tvd_gate(resolve_run_path(str(cpp_tvd), cfg))
    if depth_performance := spec.get("depth_performance_evidence"):
        validated_gates["depth_performance"] = depth_performance_gate(
            resolve_run_path(str(depth_performance), cfg)
        )
    command_wrapper_plan = build_command_wrapper_plan(
        spec,
        run_dir=cfg.run_dir,
        ares_repo=cfg.ares_repo,
    )
    if command_wrapper_plan["wrappers"]:
        command_wrapper_path = cfg.run_dir / "command_wrappers.json"
        write_json(command_wrapper_path, command_wrapper_plan)
        spec["command_wrapper_plan"] = command_wrapper_path.name
    else:
        spec.pop("command_wrapper_plan", None)
    spec["validated_gates"] = validated_gates
    spec["explicit_gates"].update(validated_gates)

    write_json(cfg.model_spec_path, spec)
    reward = compute_reward(
        gates_payload={"gates": spec["explicit_gates"]},
        validated_gates_payload={"gates": validated_gates},
        oracle_payload=oracle_payload,
        required_gates=tuple(spec["required_gates"]),
    )
    write_json(cfg.run_dir / "reward.json", reward)
    (cfg.run_dir / "reward.txt").write_text(f"{reward['score']:.12g}\n")
    return spec, reward


def initialize_run(cfg: AresIngestConfig) -> dict[str, Any]:
    _, reward = evaluate_run(cfg)
    state = {
        "status": "initialized_setup_only",
        "model": cfg.model_slug,
        "safe_model": cfg.safe_model,
        "target_score": cfg.target_score,
        "max_iterations": cfg.max_iterations,
        "refinement_command": cfg.refinement_command,
        "gate_profile": cfg.gate_profile,
        "refinement_loop": "setup_only",
        "reward": reward_fingerprint(reward),
        "history": [],
    }
    write_json(cfg.state_path, state)
    write_handoff(cfg, reward, state=state)
    return state


def gate_guidance(
    first_failed_gate: str,
    *,
    cfg: AresIngestConfig,
    spec: Mapping[str, Any],
    iteration: int,
) -> list[str]:
    common = [
        f"- Model spec: `{cfg.model_spec_path}`",
        f"- Reward JSON: `{cfg.run_dir / 'reward.json'}`",
        f"- Run handoff: `{cfg.run_dir / 'handoff.md'}`",
        f"- Verifier/refiner logs: `{cfg.logs_dir}`",
        f"- This iteration's refiner log: `{cfg.logs_dir / f'{iteration:02d}-refiner.log'}`",
    ]
    for field in ("oracle_records", "ares_plan", "target_plan"):
        if value := spec.get(field):
            common.append(
                f"- `{field}` artifact: `{resolve_run_path(str(value), cfg)}`"
            )
    if value := spec.get("command_wrapper_plan"):
        common.append(f"- Command wrapper plan: `{resolve_run_path(str(value), cfg)}`")
    specific: dict[str, list[str]] = {
        "hf_cpu_oracle": [
            "- Capture or attach a real HF Transformers + PyTorch CPU oracle record.",
            "- Do not use Ares/Rust, C++ Tron, mocks, or generated fixtures as the oracle.",
            "- Record model/tokenizer revisions, prompt tokens, generated ids, and logit slices.",
        ],
        "frontend_export": [
            "- Produce or select the frontend artifact declared by the model spec.",
            "- Keep `frontend/fx` as the default route unless this task explicitly asks for HF export.",
        ],
        "lean_ingest": [
            "- Run the Lean ingest path on the frontend artifact and keep logs in this run directory.",
            "- Do not move ahead-of-time planning into Rust.",
        ],
        "aresplan_valid": [
            "- Attach a generated AresPlan JSON artifact and set `ares_plan` in `model_spec.json`.",
            "- The AresPlan must carry Lean ingest provenance and a non-empty generated body.",
        ],
        "targetplan_valid": [
            "- Attach a Lean-emitted TargetPlan JSON artifact and set `target_plan` in `model_spec.json`.",
            "- Missing TargetPlan artifacts must fail before backend execution.",
        ],
        "shortcut_scan": [
            "- Remove any hand-authored Rust model-family plugin or runtime-generated plan sidecar.",
            "- Do not mark this gate manually; it is refreshed from the Ares source tree.",
        ],
        "backend_open": [
            "- Open the selected backend only with generated AresPlan and TargetPlan artifacts.",
            "- Record backend id, hardware gate variables, and event/log paths.",
        ],
        "one_token_logits": [
            "- Compare one-token dense logits against the HF CPU oracle with replay metadata.",
            "- Ares/Rust output is system-under-test evidence, never the oracle.",
        ],
        "eight_token_greedy": [
            "- Prove 8-token greedy identity before considering 64-token or 512-token gates.",
        ],
        "cpp_tvd": [
            "- Treat C++ Tron/Rinzler as comparison/rollback evidence, not correctness oracle.",
            "- Require matching replay context and dense-logit TVD artifacts.",
        ],
        "depth_performance": [
            "- Follow the 8 -> 64 -> 512 ladder and keep correctness gates green before speed claims.",
            "- Record throughput, latency, TTFT, memory, and artifact hashes.",
        ],
    }
    return common + specific.get(first_failed_gate, [])


def write_refinement_prompt(
    cfg: AresIngestConfig,
    *,
    iteration: int,
    spec: Mapping[str, Any],
    reward: dict[str, Any],
) -> Path:
    prompt_dir = cfg.run_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    path = prompt_dir / f"refinement-{iteration:02d}.md"
    first_failed_gate = str(reward.get("first_failed_gate", "unknown"))
    verify_command = (
        "ares-ingest-agent "
        f"{cfg.model_slug} "
        f"--ares-repo {cfg.ares_repo} "
        f"--run-dir {cfg.run_dir} "
        "--no-refiner"
    )
    prompt = "\n".join(
        [
            f"# Ares Ingest Refinement {iteration}",
            "",
            "## Objective",
            "",
            f"Bring `{cfg.model_slug}` into Ares through the generated pipeline.",
            "",
            "## Current Gate",
            "",
            f"Work only the first failing gate: `{first_failed_gate}`.",
            "",
            "Current reward:",
            "",
            "```json",
            json.dumps(reward_fingerprint(reward), indent=2, sort_keys=True),
            "```",
            "",
            "## Paths",
            "",
            f"- Ares repo: `{cfg.ares_repo}`",
            f"- AutoAgent repo: `{cfg.autoagent_root}`",
            f"- Run directory: `{cfg.run_dir}`",
            f"- Model spec: `{cfg.model_spec_path}`",
            f"- Reward JSON: `{cfg.run_dir / 'reward.json'}`",
            f"- Handoff: `{cfg.run_dir / 'handoff.md'}`",
            f"- Logs: `{cfg.logs_dir}`",
            "",
            "## Relevant Artifacts To Inspect",
            "",
            *gate_guidance(
                first_failed_gate,
                cfg=cfg,
                spec=spec,
                iteration=iteration,
            ),
            "",
            "## Ares Rules",
            "",
            "- HF Transformers on PyTorch CPU is the only model-correctness oracle.",
            "- C++ Tron/Rinzler is comparison, compliance, performance, and rollback evidence only.",
            "- Ares/Rust output is system-under-test evidence.",
            "- Runtime execution must flow through frontend artifacts, Lean ingest, generated AresPlan, Lean TargetPlan, and a backend provider.",
            "- Do not add hand-authored Rust model plugins or runtime-generated AresPlan/TargetPlan sidecars.",
            "- Do not use generated-plan runtime shortcuts as production-readiness evidence.",
            "",
            "## Allowed Write Scope",
            "",
            "- Work inside the Ares repo and this run directory only.",
            "- Keep AutoAgent source changes out of model-ingest refinement unless the failing gate is an AutoAgent tool bug.",
            "- Keep changes focused on the first failing gate; leave later gates for later iterations.",
            "",
            "## Required Verification Before Returning",
            "",
            "- Rerun the cheapest command that proves the first failing gate changed.",
            f"- Rerun `{verify_command}` or update `model_spec.json` with the new artifact paths so the next verifier pass can score it.",
            "- Update `handoff.md` or another run-local note if you discover durable state the next agent needs.",
            "",
        ]
    )
    path.write_text(prompt)
    return path


def run_refiner(
    cfg: AresIngestConfig,
    *,
    iteration: int,
    spec: Mapping[str, Any],
    reward: dict[str, Any],
) -> Path:
    if not cfg.refinement_command:
        raise AresIngestError("no refinement command configured")
    prompt_path = write_refinement_prompt(
        cfg,
        iteration=iteration,
        spec=spec,
        reward=reward,
    )
    env = {
        "ARES_REPO": str(cfg.ares_repo),
        "AUTOAGENT_REPO": str(cfg.autoagent_root),
        "RUN_DIR": str(cfg.run_dir),
        "MODEL_SPEC": str(cfg.model_spec_path),
        "REWARD_JSON": str(cfg.run_dir / "reward.json"),
        "REFINEMENT_PROMPT": str(prompt_path),
        "FIRST_FAILED_GATE": str(reward.get("first_failed_gate", "unknown")),
        "ITERATION": str(iteration),
        "MODEL_SLUG": cfg.model_slug,
        "MODEL_SAFE": cfg.safe_model,
    }
    log = cfg.logs_dir / f"{iteration:02d}-refiner.log"
    rc = run_command(
        cfg.refinement_command,
        cfg=cfg,
        env=env,
        log=log,
        timeout=24 * 3600,
    )
    if rc != 0:
        raise AresIngestError(f"refiner failed with exit {rc}; see {log}")
    return prompt_path


def append_history(
    state: dict[str, Any],
    *,
    cfg: AresIngestConfig,
    iteration: int,
    reward: dict[str, Any],
    status: str,
    prompt_path: Path | None = None,
) -> None:
    state.setdefault("history", [])
    state.setdefault("model", cfg.model_slug)
    state.setdefault("safe_model", cfg.safe_model)
    state.setdefault("run_dir", str(cfg.run_dir))
    state.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    state["status"] = status
    state["target_score"] = cfg.target_score
    state["max_iterations"] = cfg.max_iterations
    state["refinement_command"] = cfg.refinement_command
    state["gate_profile"] = cfg.gate_profile
    state["refinement_loop"] = "one_failing_gate"
    state["reward"] = reward_fingerprint(reward)
    if prompt_path is not None:
        state["latest_refinement_prompt"] = str(prompt_path)
    state["history"].append(
        {
            "iteration": iteration,
            "status": status,
            "model_spec": str(cfg.model_spec_path),
            "reward_json": str(cfg.run_dir / "reward.json"),
            "logs_dir": str(cfg.logs_dir),
            **reward_fingerprint(reward),
        }
    )
    write_json(cfg.state_path, state)
    write_handoff(cfg, reward, state=state)


def write_failure_state(cfg: AresIngestConfig, error: BaseException) -> None:
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    state = load_state_or_recover(cfg.state_path)
    state["status"] = "failed"
    state["error"] = str(error)
    state.setdefault("model", cfg.model_slug)
    state.setdefault("safe_model", cfg.safe_model)
    state.setdefault("run_dir", str(cfg.run_dir))
    state.setdefault("gate_profile", cfg.gate_profile)
    state.setdefault("refinement_loop", "one_failing_gate")
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(cfg.state_path, state)
    reward_path = cfg.run_dir / "reward.json"
    if reward_path.exists():
        write_handoff(cfg, load_json(reward_path), state=state)


def run_loop(cfg: AresIngestConfig) -> int:
    if cfg.setup_only:
        initialize_run(cfg)
        return 0

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    state = load_state_or_recover(cfg.state_path)
    best_score = max(
        [float(item.get("score", 0.0) or 0.0) for item in state.get("history", [])]
        or [0.0]
    )
    stall_count = 0
    previous_fingerprint: dict[str, Any] | None = None
    iteration = 1

    while cfg.max_iterations is None or iteration <= cfg.max_iterations:
        spec, reward = evaluate_run(cfg)
        score = float(reward.get("score", 0.0) or 0.0)
        fingerprint = reward_fingerprint(reward)

        if score >= cfg.target_score:
            append_history(
                state,
                cfg=cfg,
                iteration=iteration,
                reward=reward,
                status="complete",
            )
            print_summary(cfg, reward, status="complete")
            return 0

        if score > best_score + cfg.min_improvement:
            best_score = score
            stall_count = 0
        else:
            stall_count += 1
        if previous_fingerprint == fingerprint:
            stall_count += 1
        previous_fingerprint = fingerprint

        if not cfg.refinement_command:
            append_history(
                state,
                cfg=cfg,
                iteration=iteration,
                reward=reward,
                status="blocked_no_refiner",
            )
            print_summary(cfg, reward, status="blocked_no_refiner")
            return 3

        if stall_count >= cfg.stall_patience:
            append_history(
                state,
                cfg=cfg,
                iteration=iteration,
                reward=reward,
                status="stalled",
            )
            print_summary(cfg, reward, status="stalled")
            return 2

        if cfg.max_iterations is not None and iteration >= cfg.max_iterations:
            append_history(
                state,
                cfg=cfg,
                iteration=iteration,
                reward=reward,
                status="max_iterations",
            )
            print_summary(cfg, reward, status="max_iterations")
            return 2

        prompt_path = run_refiner(cfg, iteration=iteration, spec=spec, reward=reward)
        append_history(
            state,
            cfg=cfg,
            iteration=iteration,
            reward=reward,
            status="refiner_ran",
            prompt_path=prompt_path,
        )
        iteration += 1

    _, reward = evaluate_run(cfg)
    append_history(
        state,
        cfg=cfg,
        iteration=iteration - 1,
        reward=reward,
        status="max_iterations",
    )
    print_summary(cfg, reward, status="max_iterations")
    return 2


def print_summary(
    cfg: AresIngestConfig, reward: Mapping[str, Any], *, status: str
) -> None:
    print(
        "ares-ingest-agent "
        f"status={status} "
        f"score={reward.get('score')} "
        f"alpha_execution={reward.get('alpha_execution')} "
        f"tau_tokens={reward.get('tau_tokens')} "
        f"delta_inference={reward.get('delta_inference')} "
        f"first_failed_gate={reward.get('first_failed_gate')}"
    )
    print(f"ares_repo={cfg.ares_repo}")
    print(f"run_dir={cfg.run_dir}")
    print(f"state={cfg.state_path}")
    print(f"reward={cfg.run_dir / 'reward.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="HuggingFace model id or local model label")
    parser.add_argument("--ares-repo", default=".", help="Ares repository root")
    parser.add_argument("--run-dir", help="Explicit run directory")
    parser.add_argument("--target-score", type=float, default=1.0)
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=0,
        help="Safety cap on verifier/refiner iterations; 0 means no cap",
    )
    parser.add_argument("--stall-patience", type=int, default=2)
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.001,
        help="Minimum score improvement that resets stall detection",
    )
    parser.add_argument(
        "--no-refiner",
        action="store_true",
        help="Run evaluation only; stop below target with blocked_no_refiner",
    )
    parser.add_argument(
        "--refinement-command",
        default=DEFAULT_REFINER_COMMAND,
        help="Shell command to run between non-terminal verifier iterations",
    )
    parser.add_argument(
        "--gate-profile",
        choices=sorted(GATE_PROFILES),
        default="cpu-only",
        help="Gate profile for the generated model_spec.json",
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Create run state and exit without invoking the refiner loop",
    )
    parser.add_argument("--print-json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg: AresIngestConfig | None = None
    try:
        cfg = config_from_args(args)
        rc = run_loop(cfg)
        if args.print_json:
            print(json.dumps(load_state(cfg.state_path), indent=2))
        return rc
    except AresIngestError as exc:
        failure_note = ""
        if cfg is not None:
            try:
                write_failure_state(cfg, exc)
            except Exception as state_exc:
                failure_note = f" (also failed to write failure state: {state_exc})"
        parser.exit(1, f"ares-ingest-agent: error: {exc}{failure_note}\n")
    except KeyboardInterrupt:
        parser.exit(130, "ares-ingest-agent: interrupted\n")


if __name__ == "__main__":
    raise SystemExit(main())

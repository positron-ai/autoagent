"""Create and manage Ares AutoAgent model-ingest run directories."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ares_ingest_autoagent.gates import write_json
from ares_ingest_autoagent.score import compute_reward


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
    target_score: float
    max_iterations: int | None
    refinement_command: str | None


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
        "required_gates": [
            "model_spec",
            "hf_cpu_oracle",
            "frontend_export",
            "lean_ingest",
            "aresplan_valid",
            "targetplan_valid",
            "backend_open",
            "one_token_logits",
            "eight_token_greedy",
            "cpp_tvd",
            "depth_performance",
        ],
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


def write_handoff(cfg: AresIngestConfig, reward: dict[str, Any]) -> Path:
    path = cfg.run_dir / "handoff.md"
    text = f"""# Ares AutoAgent Run Handoff

## Objective

Bring `{cfg.model_slug}` into Ares through the generated pipeline.

## Current Reward

```json
{json.dumps(reward_fingerprint(reward), indent=2)}
```

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
        target_score=args.target_score,
        max_iterations=args.max_iterations,
        refinement_command=None if args.no_refiner else args.refinement_command,
    )


def initialize_run(cfg: AresIngestConfig) -> dict[str, Any]:
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    spec = build_model_spec(cfg)
    write_json(cfg.run_dir / "model_spec.json", spec)
    gates_payload = {"gates": spec["explicit_gates"]}
    reward = compute_reward(
        gates_payload=gates_payload,
        required_gates=tuple(spec["required_gates"]),
    )
    write_json(cfg.run_dir / "reward.json", reward)
    (cfg.run_dir / "reward.txt").write_text(f"{reward['score']:.12g}\n")
    state = {
        "status": "initialized",
        "model": cfg.model_slug,
        "safe_model": cfg.safe_model,
        "target_score": cfg.target_score,
        "max_iterations": cfg.max_iterations,
        "refinement_command": cfg.refinement_command,
        "reward": reward_fingerprint(reward),
    }
    write_json(cfg.run_dir / "state.json", state)
    write_handoff(cfg, reward)
    return state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="HuggingFace model id or local model label")
    parser.add_argument("--ares-repo", default=".", help="Ares repository root")
    parser.add_argument("--run-dir", help="Explicit run directory")
    parser.add_argument("--target-score", type=float, default=1.0)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--no-refiner", action="store_true")
    parser.add_argument(
        "--refinement-command",
        default=DEFAULT_REFINER_COMMAND,
        help="Command used between non-terminal verifier runs",
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Create run state and exit without invoking a refiner",
    )
    parser.add_argument("--print-json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    state = initialize_run(cfg)
    if args.print_json:
        print(json.dumps(state, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

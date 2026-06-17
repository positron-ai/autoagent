"""Command-line driver for Tron ingest AutoAgent refinement.

The intended human-facing command is:

    ingest PROVIDER/MODEL

Run it from a dedicated Tron worktree.  The command owns setup, generation,
build, verifier execution, durable state, and stop criteria.  If a refinement
command is configured, it is called between verifier runs until the target
score is reached or progress stalls.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUTOAGENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRON_MAIN = Path("/home/jwiegley/tron/main")
DEFAULT_WORKTREE_ROOT = Path("/home/jwiegley/tron")
DEFAULT_TRON_CACHE = Path("/tmp/tron-nix-cache")
DEFAULT_AUTOAGENT_CACHE = Path("/tmp/autoagent-nix-cache")
MAIN_WORKTREE_GUARD = DEFAULT_TRON_MAIN.resolve()
DEFAULT_REFINER_COMMAND = (
    "codex exec --dangerously-bypass-approvals-and-sandbox "
    '-C "$TRON_REPO" --add-dir "$AUTOAGENT_REPO" - < "$REFINEMENT_PROMPT"'
)


class IngestError(RuntimeError):
    """Expected command/setup failure with a user-facing message."""


@dataclass
class CommandResult:
    returncode: int
    log: Path


@dataclass
class IngestConfig:
    model_slug: str
    safe_model: str
    model_cpp: str
    tron_worktree: Path
    autoagent_root: Path
    run_dir: Path
    logs_dir: Path
    state_path: Path
    model_spec_path: Path
    hf_dir: Path
    weights: Path | None
    tron_main: Path
    worktree_root: Path
    branch: str
    executors: list[str]
    max_seq_length: int
    target_score: float
    max_iterations: int | None
    stall_patience: int
    min_improvement: float
    refinement_command: str | None
    spec_overlays: list[Path] = field(default_factory=list)
    create_worktree: bool = False
    skip_download: bool = False
    skip_convert: bool = False
    skip_generate: bool = False
    skip_build: bool = False
    setup_only: bool = False
    evaluate_only: bool = False
    no_refiner: bool = False
    tron_cache: Path = DEFAULT_TRON_CACHE
    autoagent_cache: Path = DEFAULT_AUTOAGENT_CACHE


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def bootstrap_dir(cfg: IngestConfig) -> Path:
    return cfg.worktree_root / ".autoagent/ingest-bootstrap" / cfg.safe_model / cfg.run_dir.name


def slugify(text: str) -> str:
    lowered = text.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered)
    slug = re.sub(r"-+", "-", slug).strip("-")
    if not slug:
        raise IngestError(f"cannot derive a safe model name from {text!r}")
    return slug


def deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, Mapping)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def reward_fingerprint(reward: Mapping[str, Any]) -> dict[str, Any]:
    gates = reward.get("gates", {})
    gate_bits = {}
    if isinstance(gates, Mapping):
        for name, detail in gates.items():
            if isinstance(detail, Mapping):
                gate_bits[name] = {
                    "passed": detail.get("passed"),
                    "score": detail.get("score"),
                }
            else:
                gate_bits[name] = detail
    return {
        "score": reward.get("score"),
        "alpha": reward.get("alpha"),
        "tau": reward.get("tau"),
        "delta": reward.get("delta"),
        "stage_cap": reward.get("stage_cap"),
        "first_failed_gate": reward.get("first_failed_gate"),
        "gates": gate_bits,
    }


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise IngestError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise IngestError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise IngestError(f"{path} must contain a JSON object")
    return payload


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"history": []}
    return load_json(path)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def command_display(argv: list[str] | str) -> str:
    if isinstance(argv, str):
        return argv
    return " ".join(shlex.quote(str(arg)) for arg in argv)


def run_command(
    argv: list[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None,
    log: Path,
    timeout: int | None = None,
    check: bool = True,
) -> CommandResult:
    log.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update({key: str(value) for key, value in env.items()})

    with log.open("w", encoding="utf-8") as out:
        out.write("+ " + command_display(argv) + "\n")
        out.flush()
        try:
            proc = subprocess.run(
                argv,
                cwd=cwd,
                env=merged_env,
                stdout=out,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            out.write(f"\nTIMEOUT after {timeout} seconds\n")
            raise IngestError(f"command timed out; see {log}") from exc

    if check and proc.returncode != 0:
        raise IngestError(f"command failed with exit {proc.returncode}; see {log}")
    return CommandResult(returncode=proc.returncode, log=log)


def run_shell(
    command: str,
    *,
    cwd: Path,
    env: Mapping[str, str] | None,
    log: Path,
    timeout: int | None = None,
    check: bool = True,
) -> CommandResult:
    return run_command(
        ["bash", "-lc", command],
        cwd=cwd,
        env=env,
        log=log,
        timeout=timeout,
        check=check,
    )


def tron_nix(
    cfg: IngestConfig,
    command: str,
    *,
    log_name: str,
    cwd: Path | None = None,
    timeout: int | None = None,
    check: bool = True,
) -> CommandResult:
    env = {"XDG_CACHE_HOME": str(cfg.tron_cache)}
    return run_command(
        [
            "nix",
            "develop",
            "--no-write-lock-file",
            f"path:{cfg.tron_worktree}",
            "-c",
            "bash",
            "-lc",
            command,
        ],
        cwd=cwd or cfg.tron_worktree,
        env=env,
        log=cfg.logs_dir / log_name,
        timeout=timeout,
        check=check,
    )


def ensure_tron_worktree(cfg: IngestConfig) -> None:
    if cfg.create_worktree:
        bootstrap_logs = (
            cfg.logs_dir
            if (cfg.tron_worktree / ".git").exists()
            else bootstrap_dir(cfg) / "logs"
        )
        if not (cfg.tron_main / ".git").exists():
            raise IngestError(f"missing Tron main checkout: {cfg.tron_main}")
        if cfg.tron_worktree.exists() and (cfg.tron_worktree / ".git").exists():
            return
        if subprocess.run(
            ["git", "-C", str(cfg.tron_main), "status", "--porcelain"],
            stdout=subprocess.PIPE,
            text=True,
            check=True,
        ).stdout.strip():
            raise IngestError(f"{cfg.tron_main} is not clean")
        run_command(
            ["git", "-C", str(cfg.tron_main), "fetch", "origin"],
            cwd=cfg.tron_main,
            env=None,
            log=bootstrap_logs / "git-fetch.log",
        )
        run_command(
            ["git", "-C", str(cfg.tron_main), "switch", "main"],
            cwd=cfg.tron_main,
            env=None,
            log=bootstrap_logs / "git-switch-main.log",
        )
        run_command(
            ["git", "-C", str(cfg.tron_main), "pull", "--ff-only"],
            cwd=cfg.tron_main,
            env=None,
            log=bootstrap_logs / "git-pull-main.log",
        )
        exists = subprocess.run(
            ["git", "-C", str(cfg.tron_main), "show-ref", "--verify", "--quiet", f"refs/heads/{cfg.branch}"],
            check=False,
        ).returncode == 0
        if exists:
            cmd = ["git", "-C", str(cfg.tron_main), "worktree", "add", str(cfg.tron_worktree), cfg.branch]
        else:
            cmd = [
                "git",
                "-C",
                str(cfg.tron_main),
                "worktree",
                "add",
                "-b",
                cfg.branch,
                str(cfg.tron_worktree),
                "origin/main",
            ]
        run_command(cmd, cwd=cfg.tron_main, env=None, log=bootstrap_logs / "git-worktree-add.log")

    resolved = cfg.tron_worktree.resolve()
    if resolved == MAIN_WORKTREE_GUARD:
        raise IngestError(
            "refusing to run in /home/jwiegley/tron/main; create a branch worktree first"
        )
    required = [cfg.tron_worktree / "ingest/build-model.py", cfg.tron_worktree / "CMakeLists.txt"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise IngestError(
            f"{cfg.tron_worktree} does not look like a Tron worktree; missing {', '.join(missing)}"
        )


def download_model(cfg: IngestConfig) -> None:
    if cfg.skip_download:
        return
    if cfg.hf_dir.exists() and any(cfg.hf_dir.iterdir()):
        return
    cmd = (
        "python3 - "
        f"{shlex.quote(cfg.model_slug)} {shlex.quote(str(cfg.hf_dir))} <<'PY'\n"
        "from huggingface_hub import snapshot_download\n"
        "import sys\n"
        "snapshot_download(sys.argv[1], local_dir=sys.argv[2])\n"
        "PY"
    )
    tron_nix(cfg, cmd, log_name="download.log", timeout=24 * 3600)


def choose_weights(cfg: IngestConfig) -> Path:
    if cfg.weights is not None:
        return cfg.weights
    if (cfg.hf_dir / "model.safetensors").exists() or (
        cfg.hf_dir / "model.safetensors.index.json"
    ).exists():
        return cfg.hf_dir
    return Path(f"/tmp/tron-{cfg.safe_model}-safetensors")


def convert_weights(cfg: IngestConfig) -> None:
    if cfg.skip_convert:
        return
    if (cfg.weights / "model.safetensors").exists() or (
        cfg.weights / "model.safetensors.index.json"
    ).exists():
        return
    cmd = (
        "printf '%s\\n%s\\nN\\n' "
        f"{shlex.quote(str(cfg.hf_dir))} {shlex.quote(str(cfg.weights))} "
        "| python3 bin/convert_to_safetensor.py"
    )
    tron_nix(cfg, cmd, log_name="convert-safetensors.log", timeout=24 * 3600)


def generate_plugin(cfg: IngestConfig, iteration: int) -> None:
    if cfg.skip_generate:
        return
    executor_args = " ".join(
        f"-e {shlex.quote(executor)}" for executor in cfg.executors
    )
    cmd = f"""
set -euo pipefail
cd ingest
python3 build-model.py \\
  --model {shlex.quote(str(cfg.weights))} \\
  --slug {shlex.quote(cfg.safe_model)} \\
  --name {shlex.quote(cfg.model_cpp)} \\
  --trace-dir {shlex.quote(f"traces/ingested-{cfg.safe_model}")} \\
  --plugin-dir ../gen/src/tron/h/tron/plugins \\
  --default-weights {shlex.quote(str(cfg.weights))} \\
  --config ../config/models.local.yaml \\
  --max-seq-length {cfg.max_seq_length} \\
  --dump-all \\
  {executor_args}
"""
    tron_nix(cfg, cmd, log_name=f"{iteration:02d}-generate.log", timeout=6 * 3600)


def build_runtron(cfg: IngestConfig, iteration: int) -> bool:
    if cfg.skip_build:
        return False
    cmd = f"""
set -euo pipefail
cmake --preset native -DCMAKE_BUILD_TYPE=RelWithDebInfo \\
  -DBUILD_PRODUCTION_MODELS=OFF \\
  -DBUILD_TEST_MODELS=OFF \\
  -DDEV_MODEL_CONFIG={shlex.quote(str(cfg.tron_worktree / "config/models.local.yaml"))} \\
  -DENABLE_FUSE_STATS=ON
cmake --build gen --target runtron -j "${{NIX_BUILD_CORES:-16}}"
"""
    tron_nix(cfg, cmd, log_name=f"{iteration:02d}-build-runtron.log", timeout=8 * 3600)
    return True


def build_model_spec(cfg: IngestConfig, *, cpp_compile_passed: bool) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "hf_model": str(cfg.weights),
        "work_dir": str(cfg.run_dir / "work"),
        "device": "cpu",
        "dtype": "float32",
        "run_typedfx": True,
        "run_bulk": True,
        "typedfx_timeout_sec": 3600,
        "bulk_timeout_sec": 3600,
        "required_gates": [
            "fx_export",
            "typedfx_parse",
            "typedfx_logits",
            "eqsat_structure",
            "bulk_logits",
            "cpp_compile",
            "cpu_logits",
            "fpga_logits",
        ],
        "explicit_gates": {},
        "command_gates": [],
        "token_comparison": None,
        "performance_comparison": None,
        "token_results_json": "",
        "performance_results_json": "",
    }
    if cpp_compile_passed:
        spec["explicit_gates"]["cpp_compile"] = {
            "passed": True,
            "score": 1.0,
            "command": "cmake --preset native ... && cmake --build gen --target runtron",
        }
    for overlay in cfg.spec_overlays:
        spec = deep_merge(spec, load_json(overlay))
    return spec


def run_verifier(cfg: IngestConfig, iteration: int) -> dict[str, Any]:
    cmd = (
        f"TRON_REPO={shlex.quote(str(cfg.tron_worktree))} "
        f"TASK_FILES_DIR={shlex.quote(str(cfg.autoagent_root / 'templates/tron-ingest-harbor-task/files'))} "
        f"MODEL_SPEC={shlex.quote(str(cfg.model_spec_path))} "
        f"VERIFIER_LOG_DIR={shlex.quote(str(cfg.run_dir / 'logs/verifier'))} "
        f"REWARD_JSON={shlex.quote(str(cfg.run_dir / 'reward.json'))} "
        f"REWARD_TXT={shlex.quote(str(cfg.run_dir / 'reward.txt'))} "
        f"PYTHONPATH={shlex.quote(str(cfg.autoagent_root))}:\"${{PYTHONPATH:-}}\" "
        f"python3 {shlex.quote(str(cfg.autoagent_root / 'templates/tron-ingest-harbor-task/files/evaluate_tron_ingest.py'))}"
    )
    tron_nix(cfg, cmd, log_name=f"{iteration:02d}-verifier.log", timeout=8 * 3600)
    return load_json(cfg.run_dir / "reward.json")


def write_refinement_prompt(
    cfg: IngestConfig,
    *,
    iteration: int,
    reward: Mapping[str, Any],
) -> Path:
    path = cfg.run_dir / f"refinement-{iteration:02d}.md"
    path.write_text(
        "\n".join(
            [
                f"# Tron Ingest Refinement {iteration}",
                "",
                f"Model: `{cfg.model_slug}`",
                f"Tron worktree: `{cfg.tron_worktree}`",
                f"Run directory: `{cfg.run_dir}`",
                f"Model spec: `{cfg.model_spec_path}`",
                f"Reward JSON: `{cfg.run_dir / 'reward.json'}`",
                "",
                "Current reward:",
                "",
                "```json",
                json.dumps(reward_fingerprint(reward), indent=2, sort_keys=True),
                "```",
                "",
                "Work stage by stage. Establish the first failing gate, inspect only",
                "the artifacts needed to explain it, make one focused intervention,",
                "and rerun the cheapest relevant validation before another change.",
                "Do all Tron implementation work in the listed Tron worktree, never",
                "in /home/jwiegley/tron/main. Keep AutoAgent handoff notes current if",
                "the change alters the durable process or discovered state.",
                "",
            ]
        )
    )
    return path


def run_refiner(
    cfg: IngestConfig,
    *,
    iteration: int,
    reward: Mapping[str, Any],
) -> None:
    if not cfg.refinement_command:
        raise IngestError("no refinement command configured")
    prompt_path = write_refinement_prompt(cfg, iteration=iteration, reward=reward)
    env = {
        "AUTOAGENT_REPO": str(cfg.autoagent_root),
        "TRON_REPO": str(cfg.tron_worktree),
        "MODEL_SLUG": cfg.model_slug,
        "MODEL_SAFE": cfg.safe_model,
        "MODEL_CPP": cfg.model_cpp,
        "RUN_DIR": str(cfg.run_dir),
        "MODEL_SPEC": str(cfg.model_spec_path),
        "REWARD_JSON": str(cfg.run_dir / "reward.json"),
        "REFINEMENT_PROMPT": str(prompt_path),
        "ITERATION": str(iteration),
        "SCORE": str(reward.get("score", 0.0)),
        "FIRST_FAILED_GATE": str(reward.get("first_failed_gate", "unknown")),
    }
    run_shell(
        cfg.refinement_command,
        cwd=cfg.tron_worktree,
        env=env,
        log=cfg.logs_dir / f"{iteration:02d}-refiner.log",
        timeout=24 * 3600,
    )


def append_history(
    state: dict[str, Any],
    *,
    cfg: IngestConfig,
    iteration: int,
    reward: Mapping[str, Any],
    status: str,
) -> None:
    state.setdefault("model_slug", cfg.model_slug)
    state.setdefault("safe_model", cfg.safe_model)
    state.setdefault("tron_worktree", str(cfg.tron_worktree))
    state.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    state["status"] = status
    state.setdefault("history", []).append(
        {
            "iteration": iteration,
            "status": status,
            "reward_json": str(cfg.run_dir / "reward.json"),
            "model_spec": str(cfg.model_spec_path),
            "logs_dir": str(cfg.logs_dir),
            **reward_fingerprint(reward),
        }
    )
    write_json(cfg.state_path, state)


def write_failure_state(cfg: IngestConfig, error: BaseException) -> None:
    state_path = cfg.state_path
    logs_dir = cfg.logs_dir
    if cfg.create_worktree and not (cfg.tron_worktree / ".git").exists():
        state_path = bootstrap_dir(cfg) / "state.json"
        logs_dir = bootstrap_dir(cfg) / "logs"
    try:
        state = load_state(state_path)
    except IngestError:
        state = {"history": []}
    state.setdefault("model_slug", cfg.model_slug)
    state.setdefault("safe_model", cfg.safe_model)
    state.setdefault("tron_worktree", str(cfg.tron_worktree))
    state.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    state["status"] = "failed"
    state["error"] = str(error)
    state["run_dir"] = str(cfg.run_dir)
    state["state_path"] = str(state_path)
    state["logs_dir"] = str(logs_dir)
    write_json(state_path, state)


def run_loop(cfg: IngestConfig) -> int:
    ensure_tron_worktree(cfg)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)

    download_model(cfg)
    cfg.weights = choose_weights(cfg)
    convert_weights(cfg)

    state = load_state(cfg.state_path)
    best_score = max(
        [float(item.get("score", 0.0) or 0.0) for item in state.get("history", [])]
        or [0.0]
    )
    stall_count = 0
    previous_fingerprint: dict[str, Any] | None = None

    if cfg.setup_only:
        write_json(
            cfg.state_path,
            {
                **state,
                "status": "setup_complete",
                "model_slug": cfg.model_slug,
                "tron_worktree": str(cfg.tron_worktree),
                "run_dir": str(cfg.run_dir),
                "weights": str(cfg.weights),
            },
        )
        print(f"setup_complete run_dir={cfg.run_dir}")
        return 0

    iteration = 1
    while cfg.max_iterations is None or iteration <= cfg.max_iterations:
        cpp_compile_passed = False
        if not cfg.evaluate_only:
            generate_plugin(cfg, iteration)
            cpp_compile_passed = build_runtron(cfg, iteration)

        spec = build_model_spec(cfg, cpp_compile_passed=cpp_compile_passed)
        write_json(cfg.model_spec_path, spec)
        reward = run_verifier(cfg, iteration)
        score = float(reward.get("score", 0.0) or 0.0)
        fingerprint = reward_fingerprint(reward)

        status = "evaluated"
        if score >= cfg.target_score:
            append_history(state, cfg=cfg, iteration=iteration, reward=reward, status="complete")
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
            append_history(state, cfg=cfg, iteration=iteration, reward=reward, status="stalled")
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

        append_history(state, cfg=cfg, iteration=iteration, reward=reward, status=status)
        run_refiner(cfg, iteration=iteration, reward=reward)
        iteration += 1

    reward = load_json(cfg.run_dir / "reward.json")
    append_history(state, cfg=cfg, iteration=iteration - 1, reward=reward, status="max_iterations")
    print_summary(cfg, reward, status="max_iterations")
    return 2


def print_summary(cfg: IngestConfig, reward: Mapping[str, Any], *, status: str) -> None:
    print(
        "ingest "
        f"status={status} "
        f"score={reward.get('score')} "
        f"alpha={reward.get('alpha')} "
        f"tau={reward.get('tau')} "
        f"delta={reward.get('delta')} "
        f"first_failed_gate={reward.get('first_failed_gate')}"
    )
    print(f"worktree={cfg.tron_worktree}")
    print(f"run_dir={cfg.run_dir}")
    print(f"state={cfg.state_path}")
    print(f"reward={cfg.run_dir / 'reward.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ingest",
        description="Run the Tron ingest AutoAgent loop for a HuggingFace PROVIDER/MODEL slug.",
    )
    parser.add_argument("model_slug", metavar="PROVIDER/MODEL")
    parser.add_argument(
        "--tron-worktree",
        "--worktree",
        type=Path,
        default=Path.cwd(),
        help="Target Tron worktree; defaults to the current directory",
    )
    parser.add_argument("--autoagent", type=Path, default=AUTOAGENT_ROOT)
    parser.add_argument("--tron-main", type=Path, default=DEFAULT_TRON_MAIN)
    parser.add_argument("--worktree-root", type=Path, default=DEFAULT_WORKTREE_ROOT)
    parser.add_argument("--create-worktree", action="store_true")
    parser.add_argument("--branch", default="")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--hf-dir", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--max-seq-length", type=int, default=64)
    parser.add_argument("--executor", action="append", default=[])
    parser.add_argument("--target-score", type=float, default=1.0)
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=0,
        help="Safety cap on verifier/refiner iterations; 0 means no cap",
    )
    parser.add_argument("--stall-patience", type=int, default=2)
    parser.add_argument("--min-improvement", type=float, default=0.001)
    parser.add_argument(
        "--refinement-command",
        default=os.environ.get("TRON_INGEST_REFINER", DEFAULT_REFINER_COMMAND),
        help=(
            "Shell command to run between non-terminal verifier iterations; "
            "defaults to codex exec"
        ),
    )
    parser.add_argument(
        "--no-refiner",
        action="store_true",
        help="Run setup and evaluation only; stop below target with blocked_no_refiner",
    )
    parser.add_argument(
        "--spec-overlay",
        type=Path,
        action="append",
        default=[],
        help="JSON object merged into the generated model_spec.json; repeatable",
    )
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--tron-cache", type=Path, default=Path(os.environ.get("TRON_CACHE", DEFAULT_TRON_CACHE)))
    parser.add_argument(
        "--autoagent-cache",
        type=Path,
        default=Path(os.environ.get("AUTOAGENT_CACHE", DEFAULT_AUTOAGENT_CACHE)),
    )
    return parser


def config_from_args(args: argparse.Namespace) -> IngestConfig:
    safe_model = slugify(args.model_slug)
    model_cpp = f"ingested_{safe_model.replace('-', '_')}"
    branch = args.branch or f"jw/ingest-{safe_model}"
    tron_worktree = args.tron_worktree
    if args.create_worktree and tron_worktree == Path.cwd():
        tron_worktree = args.worktree_root / f"ingest-{safe_model}"
    run_root = args.run_root or (tron_worktree / ".autoagent/ingest" / safe_model)
    run_dir = args.run_dir or (run_root / utc_stamp())
    hf_dir = args.hf_dir or Path(f"/tmp/tron-{safe_model}-hf")
    weights = args.weights
    logs_dir = run_dir / "logs"
    return IngestConfig(
        model_slug=args.model_slug,
        safe_model=safe_model,
        model_cpp=model_cpp,
        tron_worktree=tron_worktree.resolve(),
        autoagent_root=args.autoagent.resolve(),
        run_dir=run_dir.resolve(),
        logs_dir=logs_dir.resolve(),
        state_path=(run_dir / "state.json").resolve(),
        model_spec_path=(run_dir / "model_spec.json").resolve(),
        hf_dir=hf_dir.resolve(),
        weights=weights.resolve() if weights is not None else None,
        tron_main=args.tron_main.resolve(),
        worktree_root=args.worktree_root.resolve(),
        branch=branch,
        executors=args.executor or ["host", "tp1"],
        max_seq_length=args.max_seq_length,
        target_score=args.target_score,
        max_iterations=None if args.max_iterations <= 0 else args.max_iterations,
        stall_patience=max(1, args.stall_patience),
        min_improvement=max(0.0, args.min_improvement),
        refinement_command=None if args.no_refiner else args.refinement_command,
        spec_overlays=[path.resolve() for path in args.spec_overlay],
        create_worktree=args.create_worktree,
        skip_download=args.skip_download,
        skip_convert=args.skip_convert,
        skip_generate=args.skip_generate,
        skip_build=args.skip_build,
        setup_only=args.setup_only,
        evaluate_only=args.evaluate_only,
        no_refiner=args.no_refiner,
        tron_cache=args.tron_cache,
        autoagent_cache=args.autoagent_cache,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg: IngestConfig | None = None
    try:
        cfg = config_from_args(args)
        return run_loop(cfg)
    except IngestError as exc:
        if cfg is not None and cfg.tron_worktree.resolve() != MAIN_WORKTREE_GUARD:
            try:
                write_failure_state(cfg, exc)
            except Exception:
                pass
        print(f"ingest: error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("ingest: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

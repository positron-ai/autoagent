# Ares Ingest AutoAgent Runbook

This is the Ares-specific scaffold for model-ingest AutoAgent runs. The current
CLI initializes durable run state for the Ares generated pipeline:

```text
frontend artifacts -> Lean ingest -> AresPlan -> TargetPlan -> backend provider
```

Do not use this tool to add hand-authored Rust model plugins or runtime-created
execution sidecars.

## Quick Start

From an Ares checkout, create setup state without invoking the refiner:

```bash
ares-ingest-agent PROVIDER/MODEL --ares-repo "$PWD" --setup-only
```

This creates:

```text
.autoagent/ares-ingest/<safe-model>/<timestamp>/
  model_spec.json
  reward.json
  reward.txt
  state.json
  handoff.md
```

The first reward normally stops at `hf_cpu_oracle`, because setup only creates
the model spec, the shortcut scan evidence, and the run handoff. The next agent
should capture a real HF CPU oracle record, then work one failing gate at a
time.

To run the one-failing-gate loop, omit `--setup-only`:

```bash
ares-ingest-agent PROVIDER/MODEL --ares-repo "$PWD" --max-iterations 2
```

Each verifier pass refreshes deterministic gates, writes `reward.json`,
`reward.txt`, `state.json`, and `handoff.md`, then stops at target score,
stall, max iterations, or `--no-refiner`. When a refiner is enabled, the CLI
writes `prompts/refinement-NN.md` with the current first failing gate, Ares
evidence rules, allowed write scope, and verification requirements, then runs
the configured shell command with `REFINEMENT_PROMPT`, `ARES_REPO`,
`AUTOAGENT_REPO`, `RUN_DIR`, `MODEL_SPEC`, `REWARD_JSON`, and
`FIRST_FAILED_GATE` in its environment.

Use `--no-refiner` for evaluation-only runs; below target, it exits with
`blocked_no_refiner` recorded in `state.json`.

## Evidence Rules

- HuggingFace Transformers on PyTorch CPU is the correctness oracle.
- C++ Tron/Rinzler is comparison, compliance, performance, and rollback
  evidence only.
- Ares/Rust output is system-under-test evidence.
- Performance does not compensate for missing correctness gates.

## Default Gate Order

1. `model_spec`
2. `hf_cpu_oracle`
3. `frontend_export`
4. `lean_ingest`
5. `aresplan_valid`
6. `targetplan_valid`
7. `shortcut_scan`

The default `cpu-only` profile stops after generated artifact validation and a
source-tree shortcut scan. The scan rejects hand-authored Rust model-family
plugin paths and runtime-generated AresPlan/TargetPlan sidecars as promotion
evidence.

Optional profiles extend the required gate list:

- `backend`: adds `backend_open`, `one_token_logits`, and
  `eight_token_greedy`.
- `comparison`: adds `cpp_tvd` on top of `backend`.
- `full`: adds `depth_performance` on top of `comparison`.

Use hardware or C++ comparison gates only when the machine has the required
cards, checkpoints, generated artifacts, and comparison binaries.

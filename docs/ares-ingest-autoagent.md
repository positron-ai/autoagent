# Ares Ingest AutoAgent Runbook

This is the Ares-specific scaffold for model-ingest AutoAgent runs. The current
CLI initializes durable run state for the Ares generated pipeline:

```text
frontend artifacts -> Lean ingest -> AresPlan -> TargetPlan -> backend provider
```

Do not use this tool to add hand-authored Rust model plugins or runtime-created
execution sidecars.

## Quick Start

From an Ares checkout:

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

The refinement loop is not implemented in this scaffold yet. Invoking
`ares-ingest-agent` without `--setup-only` exits with an argparse error instead
of claiming to run a refiner.

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

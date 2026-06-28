# Ares Ingest AutoAgent Runbook

This is the Ares-specific scaffold for model-ingest AutoAgent runs. The CLI
evaluates durable run state for the Ares generated pipeline:

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
evidence rules, selected workflow skills, allowed write scope, and verification
requirements, then runs the configured shell command with `REFINEMENT_PROMPT`,
`ARES_REPO`, `AUTOAGENT_REPO`, `RUN_DIR`, `MODEL_SPEC`, `REWARD_JSON`, and
`FIRST_FAILED_GATE` in its environment.

Use `--no-refiner` for evaluation-only runs; below target, it exits with
`blocked_no_refiner` recorded in `state.json`.

## Workflow Skills

Every verifier pass records `workflow_skills` in `state.json` and mirrors the
same list in `handoff.md`. The list names the skill or workflow expected for
the next gate, why it was selected, which files or artifacts it may touch, and
the verification command that should prove the result. `command-wiggum` and
`ares-evidence` are always present; the current failing gate adds the relevant
Ares language, profiling, or gate-specific evidence context, and unfinished
gates include `command-fess` for conditional post-commit claim audits.
The `targetplan_valid` gate uses `ares-targetplan` because it crosses Lean
lowering, Rust validation, and runtime provider handoff.

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
7. `artifact_consistency`
8. `shortcut_scan`

The default `cpu-only` profile stops after generated artifact validation and a
source-tree shortcut scan. `artifact_consistency` requires the HF CPU oracle and
TargetPlan model ids to match the expected row, or an explicit
`expected_model_ids` alias list in `model_spec.json`. The scan rejects
hand-authored Rust model-family plugin paths and runtime-generated
AresPlan/TargetPlan sidecars as promotion evidence.

Optional profiles extend the required gate list:

- `backend`: adds `backend_open`, `one_token_logits`, and
  `eight_token_greedy`.
- `comparison`: adds `cpp_tvd` on top of `backend`.
- `full`: adds `depth_performance` on top of `comparison`.

Use hardware or C++ comparison gates only when the machine has the required
cards, checkpoints, generated artifacts, and comparison binaries.

## Runtime Evidence Fields

Attach optional evidence files in `model_spec.json` when using profiles beyond
`cpu-only`:

- `backend_open_evidence`: JSON or JSONL backend-open event evidence with
  AresPlan and TargetPlan SHA-256 values and no runtime-generated sidecars.
- `one_token_logits_evidence`: Ares system-under-test logits/TVD evidence
  compared against HF CPU oracle rows, with replay context.
- `eight_token_greedy_evidence`: Ares system-under-test greedy token evidence
  with at least eight generated tokens, HF CPU oracle provenance, source
  digests, and exact token identity.
- `cpp_tvd_evidence`: C++ Tron/Rinzler dense-logit TVD comparison evidence.
  This never replaces HF CPU oracle correctness evidence.
- `depth_performance_evidence`: 8/64/512 depth-ladder evidence with token
  correctness still green before performance is scored.

## Command Wrapper Plans

For `backend`, `comparison`, and `full` profiles, the CLI writes
`command_wrappers.json` in the run directory. The plan contains exact wrapper
commands for:

- `bin/ares-rinzler-chat` one-token/full-inference runtime artifacts;
- `bin/ci/ci-ares-rinzler-full-inference-smoke.sh` runtime smoke artifacts;
- `bin/ci/ci-rinzler-fpga-vs-tron-comparison.sh` C++ side-by-side comparison.

Wrappers default to dry-run mode and do not execute unless `model_spec.json`
sets `execute_command_wrappers` to `true`. Their outputs are launch artifacts,
not scoring evidence by themselves. Attach post-processed
`backend_open_evidence`, `one_token_logits_evidence`, `cpp_tvd_evidence`, or
`depth_performance_evidence` files to close the corresponding validator-backed
gates. A `token_comparison` block also writes `tokens.json` as
`ares.runtime.greedy_token_agreement.v1` evidence and validates it as
`eight_token_greedy` when that gate is required; its reference and candidate
files must expose generated-only token ids as `generated_token_ids` or
`generation.generated_token_ids`. The runtime launchers currently consume
`ares_plan`; keep `target_plan` attached separately as validator evidence
instead of treating the wrapper command itself as TargetPlan proof.

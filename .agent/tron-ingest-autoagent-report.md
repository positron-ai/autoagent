# Tron Ingest AutoAgent Report

Last updated: 2026-06-16

## Executive Summary

This session built the first concrete AutoAgent framework layer for Tron model
ingest iteration.

The work now includes:

- A durable task/handoff ledger.
- A Tron-specific AutoAgent operating program.
- A reusable `alpha`/`tau`/`delta` scorer.
- Artifact-based architecture and EqSat structural analyzers.
- Token-agreement and performance artifact utilities for `tau` and `delta`.
- Unit tests for the scorer and analyzers.
- A Harbor task template for Tron ingest evaluation.
- A live ignored task copy under `tasks/tron-ingest-template`.
- AutoAgent harness tools for long-running shell commands and structured
  artifact inspection.
- A host-side smoke baseline against
  `hf-internal-testing/tiny-random-LlamaForCausalLM`.

Parity is not achieved yet. This machine cannot run the full process end to
end. Docker is not running, and the full Tron validation stack needs to move to
a machine with the complete runtime, model weights, build environment,
performance measurement setup, and FPGA access where needed.

Implementation work on this machine has now stopped by request. This report is
the handoff point for continuing on a full Tron machine.

## Repositories And Paths

- AutoAgent repo: `/Users/johnw/src/autoagent`
- Tron repo: `/Users/johnw/work/positron/tron/main`
- Tron ingest skill:
  `/Users/johnw/work/positron/tron/main/.claude/skills/tron-ingest/SKILL.md`
- Handoff ledger:
  `/Users/johnw/src/autoagent/.agent/tron-ingest-parity-handoff.md`
- AutoAgent operating program:
  `/Users/johnw/src/autoagent/.agent/tron-ingest-agent-program.md`
- This report:
  `/Users/johnw/src/autoagent/.agent/tron-ingest-autoagent-report.md`

## Objective

Build an AutoAgent-based framework that enhances the current Tron ingest skill
and runs iterative improvement loops until an ingested model reaches parity.

Parity is represented by:

```text
score = min(0.70 * alpha + 0.20 * tau + 0.10 * delta, stage_cap)
```

Where:

- `alpha`: semantic preservation.
- `tau`: token-output agreement with the reference model.
- `delta`: performance relative to speed-of-light inferencing.

All values are normalized to `[0, 1]`, where `1` is perfect.

## Research Completed

### Local AutoAgent Repo

The local AutoAgent repo is a small Harbor-compatible harness:

- `agent.py`: OpenAI Agents SDK harness plus fixed Harbor adapter.
- `program.md`: meta-agent instructions for improving the harness.
- `Dockerfile.base`: base image for Harbor task containers.
- `pyproject.toml`: Python deps: `openai-agents`, `harbor`, `numpy`,
  `pandas`, `openpyxl`.

Important constraint from `program.md`: do not change `MODEL` from `gpt-5`
unless the human explicitly says to.

### AutoAgent Paper

Paper studied:

- arXiv: `https://arxiv.org/abs/2502.05957`
- Title: “AutoAgent: A Fully-Automated and Zero-Code Framework for LLM Agents”
- Authors: Jiabin Tang, Tianyu Fan, Chao Huang.

Relevant ideas extracted:

- AutoAgent treats agent construction as an iterative self-play/customization
  process.
- The paper framework uses specialized system agents, tool creation, workflow
  creation, self-managed file memory, and an orchestrator/worker structure.
- The useful idea for Tron is not to copy HKUDS AutoAgent wholesale, but to
  import the evaluator-driven loop: create artifacts, evaluate them, use
  structured feedback, and iterate.

### HKUDS AutoAgent Upstream

Repository studied:

- `https://github.com/HKUDS/AutoAgent`

Relevant architecture:

- `MetaChain` runner around LiteLLM.
- System agents: web, coding, file, orchestrator.
- Meta agents: tool editor, agent editor, workflow former/creator.
- Workflows as event-driven forms.
- XML-ish structured forms for creating agents/tools/workflows.

Important finding:

- The upstream implementation is much larger than this local repo.
- It contains useful patterns, but the local AutoAgent repo is a Harbor harness
  optimizer, not the HKUDS framework.

### Harbor

Harbor is the right evaluation substrate for this work because:

- Tasks produce deterministic verifier output.
- Verifiers write `/logs/reward.txt`.
- Richer metadata can be written to `/logs/verifier/reward.json`.
- AutoAgent can hill-climb against that scalar reward.

### OpenAI Agents SDK

Relevant SDK patterns:

- Function tools should expose structured actions rather than only raw shell.
- `agent.as_tool()` can wrap specialist agents for manager-style orchestration.
- Handoffs and agent-as-tool can support architecture, equivalence, and
  performance roles later.

### Tron Ingest Skill And References

The existing `tron-ingest` skill is already a strong staged-debugging guide.
Important points:

- Do not brute-force the final C++ plugin.
- Validate logit equivalence at intermediate stages.
- Maintain an intervention journal.
- Use sub-agents for heavy debugging to avoid context pollution.
- Use stage bisection:
  TypedFx -> EqSat -> Bulk -> C++ host -> FPGA.

Reference files read:

- `references/ir-stages.md`
- `references/eqsat-rules.md`
- `references/tracing-provenance.md`

Key Tron stages:

- FX export.
- TypedFx parse.
- TypedFx Python round-trip.
- RewriteFx and equality saturation.
- Bulk IR.
- Loopy IR.
- Tron IR.
- C++ plugin generation.
- CPU logit validation.
- FPGA validation.

## Design Decisions

### Use A Stage Cap

A weighted sum alone is too forgiving. A fast model with broken semantics could
score well if `tau` or `delta` is high. The framework therefore caps the final
score by the deepest validated semantic stage.

Current cap policy:

| Stage | Cap |
| --- | ---: |
| Not started | 0.00 |
| FX export fails | 0.05 |
| TypedFx parse fails | 0.15 |
| TypedFx logits fail | 0.35 |
| EqSat structurally wrong | 0.45 |
| Bulk logits fail | 0.60 |
| C++ compile fails | 0.70 |
| CPU logits fail | 0.82 |
| FPGA logits fail | 0.92 |
| All gates pass | 1.00 |

### Keep The Scorer Artifact-Based

The scorer reads files produced by the pipeline rather than depending on live
in-process state. This makes it usable by:

- Harbor verifiers.
- AutoAgent tools.
- Local manual runs.
- Future machines with different runtime layouts.

### Keep Tron Repo Untouched For Now

The Tron worktree already had unrelated uncommitted changes:

```text
 M h/tron/hardware/tensor.hpp
 M h/tron/models/model.hpp
 M src/pos/worker.cpp
?? agent-loops-claude-code-report.md
```

This session did not modify those files.

## Files Added Or Changed

### Added

- `.agent/tron-ingest-parity-handoff.md`
- `.agent/tron-ingest-agent-program.md`
- `.agent/tron-ingest-autoagent-report.md`
- `flake.nix`
- `tron_ingest_autoagent/__init__.py`
- `tron_ingest_autoagent/architecture.py`
- `tron_ingest_autoagent/performance.py`
- `tron_ingest_autoagent/score.py`
- `tron_ingest_autoagent/structure.py`
- `tron_ingest_autoagent/tokens.py`
- `tests/test_tron_ingest_architecture.py`
- `tests/test_tron_ingest_evaluator.py`
- `tests/test_tron_ingest_performance.py`
- `tests/test_tron_ingest_score.py`
- `tests/test_tron_ingest_structure.py`
- `tests/test_tron_ingest_tokens.py`
- `templates/tron-ingest-harbor-task/README.md`
- `templates/tron-ingest-harbor-task/task.toml`
- `templates/tron-ingest-harbor-task/instruction.md`
- `templates/tron-ingest-harbor-task/environment/Dockerfile`
- `templates/tron-ingest-harbor-task/files/model_spec.json`
- `templates/tron-ingest-harbor-task/files/evaluate_tron_ingest.py`
- `templates/tron-ingest-harbor-task/tests/test.sh`

### Changed

- `agent.py`
- `Dockerfile.base`

### Ignored Runtime Copy

- `tasks/tron-ingest-template`

This is copied from `templates/tron-ingest-harbor-task` and is ignored by Git
because `tasks/` is already gitignored.

## Implementation Details

### Nix Flake

Added `flake.nix` for the AutoAgent repo.

It provides:

- `autoagent-sync`
- `autoagent-check`
- `autoagent-build-base`
- `autoagent-run`

It pins through `nixpkgs-unstable`, uses Python 3.12, and configures uv to use
the Nix Python.

Important caveat:

Until `flake.nix` is tracked by Git, use:

```bash
nix develop --no-write-lock-file "path:$PWD"
```

Plain `nix develop` may not see untracked flake files inside a Git worktree.

### Scorer

Added:

```text
tron_ingest_autoagent/score.py
```

It reads:

- Explicit gate JSON.
- Architecture analysis JSON.
- EqSat structural analysis JSON.
- TypedFx `logit_results.json`.
- Bulk `bulk_logit_results.json`.
- Optional token agreement JSON.
- Optional performance JSON.

It writes:

```text
/logs/reward.txt
/logs/verifier/reward.json
```

The JSON includes:

- `score`
- `raw_score`
- `alpha`
- `tau`
- `delta`
- `stage_cap`
- `first_failed_gate`
- `gates`
- `alpha_components`
- `weights`

### AutoAgent Harness

Updated the editable section of `agent.py`.

Changes:

- Expanded `SYSTEM_PROMPT` for staged verification discipline.
- Increased `MAX_TURNS` from `30` to `80`.
- Added bounded shell timeout support up to 3600 seconds.
- Added tools:
  - `list_files`
  - `read_text_file`
  - `summarize_json`
  - `score_tron_ingest`

The fixed Harbor adapter boundary was not modified.

### Architecture Analyzer

Added:

```text
tron_ingest_autoagent/architecture.py
```

It extracts architecture information from ingest `metadata.json` and generated
artifacts:

- hidden size
- layer count
- query heads
- key/value heads
- head dimension
- vocabulary size
- FFN intermediate size
- RoPE theta/layout
- attention type
- normalization
- FFN activation
- MoE presence

It validates basic consistency checks such as:

- `num_attention_heads * head_dim == hidden_size`
- `num_key_value_heads <= num_attention_heads`
- positive layer/vocab/intermediate dimensions

Optional `expected_architecture` fields in `model_spec.json` are checked as
exact matches.

### EqSat Structural Analyzer

Added:

```text
tron_ingest_autoagent/structure.py
```

It scans available structural artifacts and scores expected hardware-oriented
patterns:

- `tron_sdpa`
- `tron_rope`
- `rms_norm`
- `swishmul`

The analyzer can use rich IR dumps when present, but also works from generated
Bulk Python and generated C++ headers, which are already emitted by the current
TypedFx/Bulk logit tests.

### Token Agreement Utility

Added:

```text
tron_ingest_autoagent/tokens.py
```

It compares reference and candidate token outputs and emits `tau` JSON. Inputs
can be:

- JSON arrays.
- Text token-id files.
- Strings.
- JSON objects with `tokens`, `token_ids`, `output_tokens`, `text`, or `cases`.

Metrics emitted:

- exact match
- positional exact fraction
- prefix match fraction
- edit distance
- edit similarity
- top-1 agreement
- aggregate `tau`

### Performance Utility

Added:

```text
tron_ingest_autoagent/performance.py
```

It compares measured throughput against speed-of-light throughput and emits
`delta` JSON. Inputs can be structured JSON or logs containing common Tron
throughput forms such as:

```text
Throughput 149.250 new and 200.000 all tok/s.
... at 152.942 average tok/s ...
```

The utility uses generated-token throughput when `new/all` throughput is
present.

### Dockerfile

Updated `Dockerfile.base` to copy `tron_ingest_autoagent` into the base image:

```dockerfile
COPY tron_ingest_autoagent ./tron_ingest_autoagent
```

This allows task containers to call:

```bash
python3 -m tron_ingest_autoagent.score
```

### Harbor Task Template

Added:

```text
templates/tron-ingest-harbor-task/
```

It contains:

- Task instruction.
- Task TOML.
- Environment Dockerfile.
- Model spec JSON.
- Evaluator script.
- Verifier `test.sh`.

The evaluator supports local overrides:

- `TRON_REPO`
- `TASK_FILES_DIR`
- `MODEL_SPEC`
- `VERIFIER_LOG_DIR`
- `REWARD_JSON`
- `REWARD_TXT`
- `TRON_INGEST_PYTHON`
- `TRON_ALLOW_HF_TRANSFER`

The evaluator now writes:

```text
WORK_DIR/architecture.json
WORK_DIR/eqsat_structure.json
```

It also supports `command_gates` in `model_spec.json`, so the full Tron
machine can add gate commands without changing the evaluator:

```json
{
  "command_gates": [
    {
      "name": "cpp_compile",
      "cwd": "tron_repo",
      "command": "make build",
      "timeout_sec": 7200
    },
    {
      "name": "cpu_logits",
      "cwd": "tron_repo",
      "command": "bin/compare_intermediates MODEL_TRON_NAME",
      "timeout_sec": 3600
    }
  ]
}
```

Current command-gate caveat:

- `command_gates` are currently pass/fail by process exit status.
- Before treating `compare_intermediates` or similar tools as authoritative,
  either make those tools return nonzero on semantic failure or extend
  `evaluate_tron_ingest.py` with log-aware checks such as `fail_regexes` and
  numeric threshold parsing.

It also supports:

```json
{
  "token_comparison": {
    "reference": "reference_tokens.json",
    "candidate": "candidate_tokens.json",
    "output": "tokens.json"
  },
  "performance_comparison": {
    "measured": "runtron.log",
    "speed_of_light": "speed_of_light.json",
    "output": "performance.json"
  }
}
```

The evaluator path is covered by `tests/test_tron_ingest_evaluator.py`, which
runs the actual script entrypoint in a subprocess with a synthetic Tron repo,
passing command gate, token comparison, and performance comparison fixtures.

The evaluator uses:

```bash
uv run --frozen --project ingest/runtime --extra test python ...
```

This is necessary because the Tron Nix shell on this machine exposed Cabal/GHC
but its raw Python did not import `torch`.

The evaluator disables HuggingFace fast transfer by default because this
machine had `HF_HUB_ENABLE_HF_TRANSFER=1` without the `hf_transfer` package in
the uv runtime.

## Tests And Validation

### AutoAgent Environment

Passed:

```bash
python3 -m py_compile agent.py tron_ingest_autoagent/score.py \
  tron_ingest_autoagent/architecture.py tron_ingest_autoagent/structure.py \
  tron_ingest_autoagent/tokens.py tron_ingest_autoagent/performance.py \
  tests/test_tron_ingest_score.py \
  tests/test_tron_ingest_architecture.py tests/test_tron_ingest_structure.py \
  tests/test_tron_ingest_tokens.py tests/test_tron_ingest_performance.py \
  tests/test_tron_ingest_evaluator.py \
  templates/tron-ingest-harbor-task/files/evaluate_tron_ingest.py
python3 -m unittest tests/test_tron_ingest_score.py \
  tests/test_tron_ingest_structure.py tests/test_tron_ingest_architecture.py \
  tests/test_tron_ingest_tokens.py tests/test_tron_ingest_performance.py \
  tests/test_tron_ingest_evaluator.py
nix develop --no-write-lock-file "path:$PWD" -c autoagent-check
nix develop --no-write-lock-file "path:$PWD" -c \
  uv run python -m unittest tests/test_tron_ingest_score.py \
    tests/test_tron_ingest_structure.py tests/test_tron_ingest_architecture.py \
    tests/test_tron_ingest_tokens.py tests/test_tron_ingest_performance.py \
    tests/test_tron_ingest_evaluator.py
nix flake check --no-write-lock-file "path:$PWD"
```

The local suite currently contains 23 tests.

### Docker

Attempted:

```bash
docker build -f Dockerfile.base -t autoagent-base .
```

Failed because Docker is not running:

```text
failed to connect to the docker API at
unix:///Users/johnw/.docker/run/docker.sock
```

### Tron Host Toolchain

Plain host shell:

- `uv` exists.
- `cabal` not found.
- `ghc` not found.

Tron Nix shell:

```bash
cd /Users/johnw/work/positron/tron/main
nix develop --no-write-lock-file -c cabal --version
```

Succeeded and provided:

- GHC 9.12.2.
- Cabal 3.16.0.0.

However, raw Python in the Tron Nix shell did not import `torch`, despite the
banner saying torch was available. Using the uv-managed ingest runtime solved
this for the smoke baseline.

### Initial Failed Attempts And Fixes

First host-side evaluator run failed because the Python process lost the Tron
runtime package path.

Fix:

- Preserve/prepend `ingest/runtime` in the child `PYTHONPATH`.

Second run failed because `HF_HUB_ENABLE_HF_TRANSFER=1` was set while
`hf_transfer` was absent.

Fix:

- Evaluator now sets `HF_HUB_ENABLE_HF_TRANSFER=0` by default unless
  `TRON_ALLOW_HF_TRANSFER=1`.

Third run completed.

## Host-Side Smoke Baseline

Baseline model:

```text
hf-internal-testing/tiny-random-LlamaForCausalLM
```

Command:

```bash
cd /Users/johnw/work/positron/tron/main
rm -rf /tmp/tron-autoagent-smoke /tmp/tron-ingest-template
mkdir -p /tmp/tron-autoagent-smoke/logs/verifier
nix develop --no-write-lock-file -c sh -c \
  'TRON_REPO=/Users/johnw/work/positron/tron/main \
   TASK_FILES_DIR=/Users/johnw/src/autoagent/templates/tron-ingest-harbor-task/files \
   VERIFIER_LOG_DIR=/tmp/tron-autoagent-smoke/logs/verifier \
   REWARD_JSON=/tmp/tron-autoagent-smoke/logs/verifier/reward.json \
   REWARD_TXT=/tmp/tron-autoagent-smoke/logs/reward.txt \
   PYTHONPATH=/Users/johnw/src/autoagent:$PYTHONPATH \
   python /Users/johnw/src/autoagent/templates/tron-ingest-harbor-task/files/evaluate_tron_ingest.py'
```

Result:

```text
score=0.42
raw_score=0.42
alpha=0.60
tau=0.0
delta=0.0
stage_cap=1.0
first_failed_gate=complete
```

TypedFx:

```text
5/5 strict tolerance passed
5/5 functional equivalence passed
max_tvd=0.0
top1=100%
```

Bulk:

```text
5/5 strict tolerance passed
5/5 functional equivalence passed
max_tvd=0.0
top1=100%
```

Architecture:

```text
score=1.0
hidden_size=16
num_hidden_layers=2
num_attention_heads=4
num_key_value_heads=4
head_dim=4
attention_type=MHA
normalization=RMSNorm
ffn_activation=SwiGLU/SiLU
```

EqSat structure:

```text
score=1.0
matched: rms_norm, swishmul, tron_rope, tron_sdpa
missing: none
```

Artifacts:

```text
/tmp/tron-autoagent-smoke/logs/verifier/reward.json
/tmp/tron-autoagent-smoke/logs/verifier/typedfx.log
/tmp/tron-autoagent-smoke/logs/verifier/bulk.log
/tmp/tron-ingest-template/architecture.json
/tmp/tron-ingest-template/eqsat_structure.json
/tmp/tron-ingest-template/typedfx/logit_results.json
/tmp/tron-ingest-template/bulk/bulk_logit_results.json
/tmp/tron-ingest-template/typedfx/generated/tiny_random_llamaforcausallm.hpp
/tmp/tron-ingest-template/bulk/generated/tiny_random_llamaforcausallm.hpp
```

Interpretation:

- The framework can run real architecture, EqSat-structure, TypedFx, and Bulk
  semantic gates on this machine.
- All currently configured CPU-side gates pass for the tiny smoke model.
- `tau` and `delta` remain zero because token-generation and speed-of-light
  performance evaluators are not wired yet.

## Current Limitations

The framework is not complete enough to claim parity.

Known gaps:

- Docker/Harbor execution was not tested because Docker is not running.
- No real target model has been selected.
- The template performs architecture, EqSat-structure, TypedFx, and Bulk
  CPU-side gates.
- Token agreement and performance utilities are implemented and wired into the
  task template, but they need real reference/candidate token and throughput
  artifacts from the full Tron machine.
- C++ compile, CPU plugin execution, and FPGA gates can be wired through
  `command_gates`, but have not been validated on this machine.
- Command gates need hardening if a verifier command can print failure details
  while exiting successfully.
- The full process needs to move to a machine that can run the real Tron stack.

## Migration Plan For A Full Tron Machine

1. Move or clone the AutoAgent repo.

2. Ensure the current changes are present:

   ```bash
   cd /Users/johnw/src/autoagent
   git status --short
   ```

3. Start Docker.

4. Build the AutoAgent base image:

   ```bash
   cd /Users/johnw/src/autoagent
   docker build -f Dockerfile.base -t autoagent-base .
   ```

5. Copy the template to a real task:

   ```bash
   mkdir -p tasks
   cp -R templates/tron-ingest-harbor-task tasks/tron-ingest-<model>
   ```

6. Edit:

   ```text
   tasks/tron-ingest-<model>/files/model_spec.json
   ```

   Set:

   - `hf_model`
   - `work_dir`
   - `device`
   - `dtype`
   - `expected_architecture`, if exact architecture assertions are known
   - `expected_eqsat_patterns`
   - `command_gates` for `cpp_compile`, `cpu_logits`, and eventually
     `fpga_logits`
   - required gates
   - token evaluator output path
   - performance evaluator output path

7. Run a baseline Harbor task:

   ```bash
   cd /Users/johnw/src/autoagent
   AUTOAGENT_CONCURRENCY=1 AUTOAGENT_JOB_NAME=tron-ingest-baseline \
     nix develop --no-write-lock-file "path:$PWD" -c \
     autoagent-run --task-name tron-ingest-<model> -l 1
   ```

8. Inspect:

   ```text
   jobs/tron-ingest-baseline/*/verifier/reward.json
   jobs/tron-ingest-baseline/*/agent/trajectory.json
   ```

9. Iterate:

   - If first failed gate is `typedfx_logits`, debug FX/TypedFx/EqSat.
   - If first failed gate is `bulk_logits`, debug `TypedFxBulk.hs`.
   - If first failed gate is `cpp_compile`, debug generated plugin and config.
   - If first failed gate is `cpu_logits`, use `compare_intermediates`.
   - If first failed gate is `fpga_logits`, confirm FPGA access and debug
     hardware/plugin divergence.
   - If alpha passes but delta is low, inspect missed EqSat patterns and
     performance bottlenecks.

## Next Engineering Tasks

Recommended next tasks, in order:

1. Harden command gates.

   Input:

   - Verifier logs from commands such as `compare_intermediates`.
   - Known failure markers and numeric tolerances.

   Output:

   - Either verifier commands return nonzero on semantic failure, or
     `evaluate_tron_ingest.py` supports log checks such as `fail_regexes` and
     numeric threshold checks.

   Example target behavior:

   ```json
   {
     "name": "cpu_logits",
     "cwd": "tron_repo",
     "command": "bin/compare_intermediates MODEL_TRON_NAME",
     "timeout_sec": 3600,
     "fail_regexes": ["Busted!"],
     "numeric_checks": [
       {
         "name": "relative_l2",
         "regex": "Relative error in L2:\\s*([0-9.eE+-]+)",
         "max": 0.1,
         "aggregate": "max"
       }
     ]
   }
   ```

2. Add C++ compile gate for the target model.

   Input:

   - Generated plugin.
   - Tron build command.

   Output:

   - Gate `cpp_compile`.
   - Compiler error summary.

3. Add CPU plugin logit gate.

   Input:

   - `gen/runtron`.
   - Reference logits.
   - `bin/compare_intermediates`.

   Output:

   - Gate `cpu_logits`.
   - First diverging layer/tensor.

4. Produce token artifacts for the target model and enable
   `token_comparison`.

   Input:

   - Fixed prompt suite.
   - Reference HF generation.
   - Tron generation.

   Output:

   - `tau` JSON.

5. Produce speed-of-light and measured-throughput artifacts for the target
   model and enable `performance_comparison`.

   Input:

   - Model architecture.
   - Hardware limits.
   - Measured throughput.

   Output:

   - `delta` JSON.

6. Add FPGA gate only on a machine with FPGA access.

## Restart Checklist

Use this order in a fresh AI session:

1. Read:

   ```text
   .agent/tron-ingest-parity-handoff.md
   .agent/tron-ingest-agent-program.md
   .agent/tron-ingest-autoagent-report.md
   ```

2. Check worktrees:

   ```bash
   cd /Users/johnw/src/autoagent && git status --short
   cd /Users/johnw/work/positron/tron/main && git status --short
   ```

3. Re-run cheap checks:

   ```bash
   cd /Users/johnw/src/autoagent
   python3 -m unittest tests/test_tron_ingest_score.py \
     tests/test_tron_ingest_structure.py tests/test_tron_ingest_architecture.py \
     tests/test_tron_ingest_tokens.py tests/test_tron_ingest_performance.py \
     tests/test_tron_ingest_evaluator.py
   nix develop --no-write-lock-file "path:$PWD" -c autoagent-check
   ```

4. If on a full Tron machine, start Docker and run Harbor.

5. If not on a full Tron machine, continue implementing the missing evaluator
   gates without claiming parity.

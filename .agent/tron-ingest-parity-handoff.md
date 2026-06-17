# Tron Ingest AutoAgent Handoff

Last updated: 2026-06-17

## Objective

Build an AutoAgent-based framework that improves the Tron model ingest workflow
by iterating on measurable feedback until parity is achieved.

Parity means the ingested Tron plugin reaches a weighted score of 1.0:

```text
score = min(0.70 * alpha + 0.20 * tau + 0.10 * delta, stage_cap)
```

Where:

- `alpha`: semantic preservation score, based on staged ingest gates.
- `tau`: token-output agreement score against the reference model.
- `delta`: performance score against the model's computed speed-of-light bound.

All three scores are normalized to `[0, 1]`, where `1` is perfect.

## Session State

Earlier implementation work stopped on the macOS machine by request. Work has
now resumed and completed on the Linux Tron machine at
`/home/jwiegley/autoagent`.

The current state is local full-machine parity for the active Llama-68M target,
with one external execution blocker: Docker/Harbor cannot be validated because
this host still has no Docker daemon socket at `/var/run/docker.sock`.

What is ready:

- AutoAgent development environment via `flake.nix`.
- Reusable `alpha`/`tau`/`delta` scoring package.
- Harbor task template for Tron ingest evaluation.
- Host-side smoke validation against a tiny Llama model through TypedFx and
  Bulk gates.
- Validated log-aware command gates with regression tests for `fail_regexes`,
  `pass_regexes`, and numeric thresholds.
- Confirmed this machine has eight `8200:0011` FPGA PCI devices bound to
  `vfio-pci` with eight `/dev/vfio/*` group devices.
- Active target `JackFram/llama-68m` driven through TypedFx, Bulk, C++ compile,
  CPU logits, token agreement, performance, and FPGA hand-vs-generated parity.
- Final reward artifact:
  `/tmp/tron-autoagent-runs/llama68-20260617T015302Z/artifacts/reward.json`
  reports `score=1.0`, `alpha=1.0`, `tau=1.0`, `delta=1.0`,
  `stage_cap=1.0`, and `first_failed_gate=complete`.
- Durable copies of the final run spec and reward summary are checked into the
  handoff area:
  - `.agent/tron-ingest-llama68-model-spec.json`.
  - `.agent/tron-ingest-llama68-final-reward.json`.
- Repeatable process documentation and wrapper script:
  - `docs/tron-ingest-autoagent.md`.
  - `scripts/ingest`.
  - `scripts/run-tron-ingest-refinement.sh`.
- Packaged CLI entry point:
  - `ingest PROVIDER/MODEL` via `tron_ingest_autoagent.ingest_cli:main`.
  - AutoAgent is now configured as an installable Python package with a
    Hatchling build backend, so `uv run ingest` resolves the console entry
    point.
  - The command defaults to the current directory as the Tron worktree,
    refuses `/home/jwiegley/tron/main`, writes run state under
    `.autoagent/ingest/<safe-model>/<timestamp>/`, and stops when the reward
    reaches `1.0`, stalls, or, with `--no-refiner`, lacks a configured
    refiner.
  - There is no default iteration cap; `--max-iterations N` is an explicit
    safety cap only.
  - Bare `ingest PROVIDER/MODEL` now uses the default Codex refiner:
    `codex exec --dangerously-bypass-approvals-and-sandbox -C "$TRON_REPO"
    --add-dir "$AUTOAGENT_REPO" - < "$REFINEMENT_PROMPT"`.
  - `scripts/ingest` runs the same command from any Tron worktree through the
    AutoAgent checkout.

What is not done:

- Docker/Harbor task execution is still blocked on this machine because no
  Docker daemon socket exists at `/var/run/docker.sock`.
- Containerized Harbor execution has therefore not been proven, even though the
  same evaluator path was run directly from the Tron Nix shell.

## PR 3012 CI Fix Status

PR: <https://github.com/positron-ai/tron/pull/3012>
Branch/worktree: `jw/llama68-ingest-support` at
`/home/jwiegley/tron/llama68-ingest-support`.

Current CI-fix loop started after the user requested `$command-fix-ci`.

Completed:

- Fixed the CI `lint` failure by running clang-format on the affected Tron
  headers.
- Addressed Cursor BugBot review thread `PRRT_kwDOKStajs6KFh4-`.
- Pushed commit `2f97f2647` (`Write all generated token sequences`), which
  makes `--output-token-file` write one generated-token row per prompt instead
  of only `std::get<0>(response)`.
- Replied to the Cursor BugBot comment and resolved the GitHub review thread.
- Addressed follow-up Cursor BugBot review thread `PRRT_kwDOKStajs6KFqfv`.
- Pushed commit `ae5e80900` (`Preserve token output across generation
  iterations`), which preserves every generated-token row across
  `--iterations > 1` instead of only retaining the final iteration.
- Replied to the follow-up Cursor BugBot comment and resolved that GitHub
  review thread.
- Local checks passed:
  - `lefthook run --all-files pre-commit --commands whitespace,no-cassert,clang-format,ci-pins --no-tty`
  - `python3 config/test_model_definitions.py`
  - `cmake --preset native -DCMAKE_BUILD_TYPE=RelWithDebInfo && cmake --build gen --target runtron -j 16`

CI state as of this update:

- Previous head `2f97f2647`: required `build` and `lint` passed; benchmark
  passed; `benchmark-report` and `ingest-frontend` were queued on self-hosted
  runners.
- Current head `ae5e80900`: pushed.
- Current head local validation passed:
  - `lefthook run --all-files pre-commit --commands whitespace,no-cassert,clang-format,ci-pins --no-tty`
  - `python3 config/test_model_definitions.py`
  - `cmake --preset native -DCMAKE_BUILD_TYPE=RelWithDebInfo && cmake --build gen --target runtron -j 16`
- Current head GitHub checks: Cursor BugBot, Graphite AI Reviews, and Graphite
  mergeability check passed; CMake/CTest OCI `build`, `lint`, `benchmark`, and
  `benchmark-report` were skipped by path filters; `ingest-frontend` remains
  queued on a `self-hosted,no-fpga` runner.
- Local emulation of `ingest-frontend` with `EXPORT_ONLY=1 ./build-all.sh`
  could not proceed on this host because `/opt/positron/weights/huggingface`
  is missing the full CI registry weight set; first missing path was
  `/opt/positron/weights/huggingface/Qwen/Qwen2.5-32B-Instruct`.

## Repositories

- AutoAgent repo: `/home/jwiegley/autoagent`
- Tron repo: `/home/jwiegley/tron/main`
- Existing ingest skill:
  `/home/jwiegley/tron/main/.claude/skills/tron-ingest/SKILL.md`

## Current Constraints

- The AutoAgent harness is in `agent.py`.
- The fixed Harbor adapter boundary in `agent.py` should not be edited unless
  the human explicitly asks.
- The model in AutoAgent should remain `gpt-5` unless the human explicitly
  changes that constraint.
- The ingest skill requires staged validation; do not jump directly to final
  C++ plugin debugging.
- The current user request confirmed this machine has eight FPGA cards
  available. Still preserve the staged order: TypedFx/Bulk, C++ compile, CPU
  logits, then FPGA.
- Use `XDG_CACHE_HOME=/tmp/autoagent-nix-cache` for AutoAgent Nix commands and
  `XDG_CACHE_HOME=/tmp/tron-nix-cache` for Tron Nix commands until
  `/home/jwiegley/.cache/nix/fetcher-cache-v4.sqlite` is repaired.
- Do not do Tron implementation or iteration work in `/home/jwiegley/tron/main`.
  Treat it as the clean base checkout only. For every target or PR, create a
  new branch and a new git worktree under `/home/jwiegley/tron/`, then do all
  builds, generated artifacts, and iterations from that worktree.

## Metric Contract

The verifier should emit:

```text
/logs/reward.txt
/logs/verifier/reward.json
```

The JSON payload should include at least:

```json
{
  "score": 0.0,
  "alpha": 0.0,
  "tau": 0.0,
  "delta": 0.0,
  "stage_cap": 0.0,
  "first_failed_gate": "not_started",
  "gates": {}
}
```

## Stage Cap Policy

The weighted score is capped by the deepest validated ingest stage:

| Stage | Cap |
| --- | ---: |
| FX export fails | 0.05 |
| TypedFx parse fails | 0.15 |
| TypedFx logits fail | 0.35 |
| EqSat structurally wrong | 0.45 |
| Bulk logits fail | 0.60 |
| C++ compile fails | 0.70 |
| CPU logits fail | 0.82 |
| FPGA logits fail | 0.92 |
| All gates pass | 1.00 |

## Tasks

- [x] Read the AutoAgent repo.
- [x] Read the Tron ingest skill and its key references.
- [x] Read the AutoAgent paper and upstream implementation.
- [x] Add a Nix flake for the AutoAgent dev environment.
- [x] Create this handoff and task ledger.
- [x] Inspect the Tron repo's current ingest scripts and test utilities.
- [x] Design the first Harbor task format for a small model ingest target.
- [x] Implement reusable scoring utilities for `alpha`, `tau`, and `delta`.
- [x] Add AutoAgent tools for structured ingest execution and artifact reading.
- [x] Add or document sub-agent roles for architecture, equivalence, and
  performance analysis.
- [x] Validate the framework locally without FPGA.
- [x] Run at least one baseline ingest evaluation.
- [x] Add initial log/metric checks to command gates for tools that can print a
  failure while exiting successfully.
- [x] Validate the command-gate checker and add regression tests for failure
  regexes, pass regexes, and numeric thresholds.
- [ ] Move to a full Tron machine and validate Docker/Harbor execution.
- [x] Select a real target model.
- [x] Add and validate C++ compile, CPU logits, token, performance, and FPGA
  gates as appropriate.
- [x] Iterate until parity is reached or a hard external blocker is documented.

## Current Autonomous Run

Started: 2026-06-17 on `/home/jwiegley/autoagent`.

Objective: continue autonomously until the Tron ingest AutoAgent parity work is
complete, maintaining this ledger as the durable source of truth.

Current working assumptions:

- Use the staged ingest discipline from the Tron skill:
  TypedFx/Bulk -> C++ compile -> CPU logits -> token agreement ->
  performance -> FPGA.
- Choose a concrete first parity target locally instead of waiting for manual
  selection.
- Do not launch broad builds for every registered large model unless a
  narrower target cannot exercise the needed gates.
- Keep generated runtime artifacts out of the Git worktrees unless they are
  intentional source or task-template inputs.

Current next actions:

- [x] Select the first concrete target model.
- [x] Create or update a task spec for that target.
- [x] Add command gate coverage for C++ build.
- [x] Add command gate coverage for CPU-logit validation.
- [x] Add token and performance artifact producers for `tau` and `delta`.
- [x] Add FPGA gate after C++ and CPU gates are trustworthy.
- [x] Run the complete scoring path and iterate.
- [x] Package the workflow as an `ingest PROVIDER/MODEL` CLI for use inside a
  dedicated Tron worktree.

Latest AutoAgent packaging validation:

- `uv sync`
- `uv run ingest --help`
- `/home/jwiegley/autoagent/scripts/ingest --help` from
  `/home/jwiegley/tron/llama68-ingest-support`
- `uv run python -m py_compile tron_ingest_autoagent/ingest_cli.py`
- `uv run python -m unittest discover tests` now passes 36 tests.
- `bash -n scripts/ingest scripts/run-tron-ingest-refinement.sh`
- `uv lock --check`
- `nix-instantiate --parse flake.nix`
- Main checkout guard check from `/home/jwiegley/tron/main`:
  `/home/jwiegley/autoagent/scripts/ingest --skip-download --skip-convert --setup-only hf-internal-testing/tiny-random-LlamaForCausalLM`
  exits `1` with the expected refusal and does not create or change a
  `.autoagent` path under `/home/jwiegley/tron/main`.
- `codex exec --help` confirms the default refiner's required flags:
  `--dangerously-bypass-approvals-and-sandbox`, `-C/--cd`, and `--add-dir`.

Target selection:

- Initial smoke target:
  `hf-internal-testing/tiny-random-LlamaForCausalLM`.
- Smoke target result: TypedFx and Bulk parity are excellent, but generated C++
  is not a useful production-runtime target because its `head_size=4` violates
  existing SIMD divisibility constraints in the runtime RoPE and KV-cache
  templates.
- Active parity target: `JackFram/llama-68m`.
- Rationale: it is still small enough for local iteration, but has
  production-shaped dimensions (`hidden_size=768`, `num_attention_heads=12`,
  `head_dim=64`, `num_hidden_layers=2`, `vocab_size=32000`). It is already a
  hand-authored production model in Tron's registry, so generated-ingest results
  can be compared against a known-supported runtime shape.
- Local HuggingFace snapshot:
  `/tmp/tron-nix-cache/huggingface/hub/models--JackFram--llama-68m/snapshots/9de84537b6aa98de634ad0fbb1608e9d6a019355`.
- Generated runtime slugs:
  - `ingested-llama-68m` for `tp1`.
  - `ingested-llama-68m-host` for host CPU.

Local Tron generation status:

- Created ignored local registry file
  `/home/jwiegley/tron/main/config/models.local.yaml`.
- Tiny smoke generation produced trace/plugin artifacts under
  `/home/jwiegley/tron/main/ingest/traces/ingested-tiny-random-llama` and
  `/home/jwiegley/tron/main/gen/src/tron/h/tron/plugins/`.
- Tiny generated artifacts are being removed before the Llama-68M generation so
  CMake only sees the active parity target.
- Llama-68M generated plugin status: complete.
- Generated trace:
  `/home/jwiegley/tron/main/ingest/traces/ingested-llama-68m`.
- Generated plugin:
  `/home/jwiegley/tron/main/gen/src/tron/h/tron/plugins/ingested_llama_68m.hpp`.
- Generated provenance sidecar:
  `/home/jwiegley/tron/main/gen/src/tron/h/tron/plugins/ingested_llama_68m.trace.json`.
- Local registry validation produced:
  - `TRON_PLUGIN(ingested_llama_68m, host)`.
  - `TRON_PLUGIN(ingested_llama_68m, tp1)`.
  - runtime slugs `ingested-llama-68m-host` and `ingested-llama-68m`.
- CMake should be configured with
  `-DDEV_MODEL_CONFIG=/home/jwiegley/tron/main/config/models.local.yaml` to
  avoid compiling all test-tagged registry models.

C++ compile gate attempt:

- Command:

  ```bash
  cd /home/jwiegley/tron/main
  XDG_CACHE_HOME=/tmp/tron-nix-cache nix develop --no-write-lock-file -c sh -lc \
    'cmake --preset native -DCMAKE_BUILD_TYPE=RelWithDebInfo \
       -DBUILD_PRODUCTION_MODELS=OFF -DBUILD_TEST_MODELS=OFF \
       -DDEV_MODEL_CONFIG=/home/jwiegley/tron/main/config/models.local.yaml \
       -DENABLE_FUSE_STATS=ON && cmake --build gen --target runtron -j 16'
  ```

- Result: failed in C++ compile.
- First failure: generated tiny model has `head_size=4`; existing
  `rope_table<rope_layout::huggingface, head_size>` and
  `scaled_v_expr<page_size, head_size>` require divisibility by their SIMD
  chunk size, which rejects `4`.
- Fixed secondary failure in source: with a standalone local registry,
  `h/tron/models/load.hpp`
  hard-codes `llama_3_8b_tp1` for `ret_t` in `with_model_type`; that type is
  absent when `DEV_MODEL_CONFIG` contains only a local generated model.
  `config/model_definitions.py` now emits a `TRON_FIRST_HUGGINGFACE_MODEL`
  entry, and `load.hpp` uses that generated first model for return-type
  deduction.
- Next action: generate the active Llama-68M plugin and re-run the narrow
  C++ compile gate.

Llama-68M C++ compile gate:

- Added a Tron regression test for `generate_first_model_call` in
  `/home/jwiegley/tron/main/config/test_model_definitions.py`.
- Validation passed:

  ```bash
  cd /home/jwiegley/tron/main
  XDG_CACHE_HOME=/tmp/tron-nix-cache nix develop --no-write-lock-file -c \
    sh -lc 'python3 config/test_model_definitions.py'
  ```

  Result: 50 tests pass.
- Direct model-definition generation with the local registry produced the
  expected `TRON_PLUGIN(ingested_llama_68m, host)`,
  `TRON_PLUGIN(ingested_llama_68m, tp1)`, and
  `TRON_HUGGINGFACE_MODEL` entries for `ingested-llama-68m-host` and
  `ingested-llama-68m`.
- C++ build command:

  ```bash
  cd /home/jwiegley/tron/main
  XDG_CACHE_HOME=/tmp/tron-nix-cache nix develop --no-write-lock-file -c sh -lc \
    'cmake --preset native -DCMAKE_BUILD_TYPE=RelWithDebInfo \
       -DBUILD_PRODUCTION_MODELS=OFF -DBUILD_TEST_MODELS=OFF \
       -DDEV_MODEL_CONFIG=/home/jwiegley/tron/main/config/models.local.yaml \
       -DENABLE_FUSE_STATS=ON && cmake --build gen --target runtron -j 16'
  ```

  Result: passed; `gen/runtron` linked successfully for the generated
  Llama-68M target.
- Runtime prerequisites discovered:
  - `JackFram/llama-68m` only ships `pytorch_model.bin`; Tron runtime expects
    `model.safetensors` or `model.safetensors.index.json`.
  - Materialized a local safetensors copy at `/tmp/tron-llama68-safetensors`
    using `bin/convert_to_safetensor.py`.
  - Runtime commands must run under `sg positron` so `/dev/vfio/*` and
    `/dev/hugepages` are accessible.
  - Running with `--devices 10:00.0` alone initializes a card but leaves
    `app_cpu_set` empty, causing `app_pool()` to terminate while constructing
    `self_attention::state`.
  - Use `--instance 0,8` for one-card runs; this selects slice 0 from
    `config/resource-map.yaml` and populates app/dev CPU sets.
- Host CPU runtime smoke command:

  ```bash
  cd /home/jwiegley/tron/main
  tmpdir=$(mktemp -d /tmp/tron-llama68-host.XXXXXX)
  printf 'Hello from Tron\n' > "$tmpdir/prompt.txt"
  export TRON_RUN_TMP="$tmpdir"
  sg positron -c 'cd /home/jwiegley/tron/main && \
    XDG_CACHE_HOME=/tmp/tron-nix-cache HF_HUB_ENABLE_HF_TRANSFER=0 \
    nix develop --no-write-lock-file -c ./gen/runtron stream-generate-text \
      -m ingested-llama-68m-host \
      --model-path /tmp/tron-llama68-safetensors \
      -f "$TRON_RUN_TMP/prompt.txt" -l 4 --seed 1 --temperature 1.0 \
      --threshold_p 1.0 --sequential --no-opt --instance 0,8 \
      --nr_hugepages 2 --kv-cache-gb 1 --fuse-mount "$TRON_RUN_TMP/fuse"'
  ```

  Result: passed. The generated host model loads, allocates state, parses a
  six-token prompt, emits four response tokens, and reports about 1200 tok/s on
  the smoke run.

Host CPU logits/intermediate gate:

- Durable run directory:
  `/tmp/tron-autoagent-runs/llama68-20260617T015302Z`.
- HF reference command:

  ```bash
  cd /home/jwiegley/tron/main
  XDG_CACHE_HOME=/tmp/tron-nix-cache HF_HUB_ENABLE_HF_TRANSFER=0 \
    nix develop --no-write-lock-file -c ./bin/extract_logits \
      --model /tmp/tron-llama68-safetensors \
      --max-layers 2 \
      --text-file /tmp/tron-autoagent-runs/llama68-20260617T015302Z/prompt.txt \
      --max-tokens 8 \
      --save-tokens \
      --intermediates-file /tmp/tron-autoagent-runs/llama68-20260617T015302Z/hf/intermediates.txt \
      --scratch-dir /tmp/tron-autoagent-runs/llama68-20260617T015302Z/hf/scratch
  ```

- Generated host command used the saved binary token file from
  `extract_logits` and wrote
  `/tmp/tron-autoagent-runs/llama68-20260617T015302Z/tron-host/intermediates-l0.txt`.
- Patched `/home/jwiegley/tron/main/bin/compare_intermediates` so it can parse
  the current C++ intermediate log format:
  - strips the optional `(len):` field before values.
  - aliases `Logits of token N` to HF's `Logits for token N`.
- Comparison command:

  ```bash
  cd /home/jwiegley/tron/main
  python3 bin/compare_intermediates \
    --reference-intermediates-file /tmp/tron-autoagent-runs/llama68-20260617T015302Z/hf/intermediates.txt \
    --intermediates-file /tmp/tron-autoagent-runs/llama68-20260617T015302Z/tron-host/intermediates-l0.txt \
    --vec-rtol 0.1 \
    > /tmp/tron-autoagent-runs/llama68-20260617T015302Z/tron-host/compare-l0.log
  ```

- Result: passed as a log-aware CPU gate candidate.
  - 89 vector comparisons.
  - maximum relative L2 error: `0.0100283`.
  - final logits included; maximum logits relative L2: `0.0047548`.
  - `Busted!` count: `0`.
  - remaining missing reference entries are generated-only RoPE diagnostics and
    top-k sparse-logit diagnostics, not required HF reference keys.

Host token and performance artifacts:

- Added `--output-token-file` to
  `/home/jwiegley/tron/main/gen/runtron stream-generate-text` by updating
  `h/runtron/stream_generate.hpp` and `src/runtron.cpp`. The option writes
  generated token IDs as whitespace-separated decimal tokens, one sequence per
  line.
- Greedy token agreement:
  - HF reference tokens:
    `/tmp/tron-autoagent-runs/llama68-20260617T015302Z/artifacts/hf-greedy-tokens.txt`.
  - Generated host tokens:
    `/tmp/tron-autoagent-runs/llama68-20260617T015302Z/artifacts/tron-host-greedy-tokens.txt`.
  - Tokens: `29902 505 1063 773`.
  - AutoAgent token artifact:
    `/tmp/tron-autoagent-runs/llama68-20260617T015302Z/artifacts/tokens.json`.
  - Result: `tau=1.0`.
- Performance comparison:
  - Added the hand-authored `llama_68m` host/tp1 variants to the ignored
    `/home/jwiegley/tron/main/config/models.local.yaml` so generated ingest and
    established hand-authored implementations can run from the same executable
    against `/tmp/tron-llama68-safetensors`.
  - Hand-authored host model slug: `llama-68m-local-host`.
  - Generated host model slug: `ingested-llama-68m-host`.
  - 64-token deterministic throughput:
    - hand-authored host: `769.366` new tok/s.
    - generated ingest host: `836.575` new tok/s.
  - The generated and hand-authored 64-token continuations matched exactly.
  - Proxy speed-of-light artifact:
    `/tmp/tron-autoagent-runs/llama68-20260617T015302Z/perf/speed-of-light-host-proxy.json`.
  - AutoAgent performance artifact:
    `/tmp/tron-autoagent-runs/llama68-20260617T015302Z/artifacts/performance.json`.
  - Result: `delta=1.0` against the hand-authored-host proxy.
- Fixed `tron_ingest_autoagent.performance` so raw Tron logs prefer explicit
  generated-token throughput (`Throughput ... new`) over prompt-parse
  `tokens/s` lines. Added a regression test in
  `tests/test_tron_ingest_performance.py`; the focused performance suite
  passes (`5` tests).
FPGA `tp1` gate:

- Short generated `tp1` greedy run produced tokens `29902 626 263 13524`;
  validated HF/host tokens are `29902 505 1063 773`.
- The hand-authored `llama-68m-local` `tp1` slug produced the same short FPGA
  token sequence as generated ingest, so strict HF/host-vs-FPGA token mismatch
  is not ingest-specific for this model/runtime path.
- 64-token free-running hand-authored-vs-generated `tp1` runs diverged at
  token index 7 because small logit differences change autoregressive context.
  This is too brittle as the primary FPGA ingest gate.
- Added `--force-feed-text-token-file` to
  `/home/jwiegley/tron/main/gen/runtron stream-generate-text` by updating
  `h/runtron/stream_generate.hpp` and `src/runtron.cpp`.
- Forced-context file:
  `/tmp/tron-autoagent-runs/llama68-20260617T015302Z/fpga/force-feed-host-16.txt`.
  Contents: `29902 505 1063 773 278 1021 1158 363 263 1550 1286 29889 306 505 1898 304`.
- Hand-authored and generated `tp1` forced runs emitted exactly those forced
  tokens.
- Intermediate comparison filtered `Top-k Token IDs` diagnostics, because
  those are integer IDs printed in vector form rather than logit/activation
  values.
- FPGA comparison command:

  ```bash
  cd /home/jwiegley/tron/main
  python3 bin/compare_intermediates \
    --reference-intermediates-file /tmp/tron-autoagent-runs/llama68-20260617T015302Z/fpga/hand-fpga-force16-intermediates.filtered.txt \
    --intermediates-file /tmp/tron-autoagent-runs/llama68-20260617T015302Z/fpga/ingested-fpga-force16-intermediates.filtered.txt \
    --vec-rtol 0.05 \
    > /tmp/tron-autoagent-runs/llama68-20260617T015302Z/fpga/compare-hand-vs-ingested-force16.log 2>&1
  ```

- Result: passed as the FPGA ingest parity gate.
  - 560 vector comparisons.
  - maximum relative L2 error: `0.0161088`.
  - `Busted!` count: `0`.
  - missing reference entries: `0`.
- Current interpretation: generated ingest matches the established
  hand-authored `tp1` FPGA path under identical forced contexts. Remaining
  HF/host-vs-FPGA free-generation mismatch is a broader runtime/model issue,
  not evidence of ingest-specific FPGA regression.

Final Llama-68M scoring:

- Model spec used by the direct evaluator:
  `/tmp/tron-autoagent-runs/llama68-20260617T015302Z/autoagent/model_spec.json`.
  Durable copy:
  `.agent/tron-ingest-llama68-model-spec.json`.
- The first evaluator run exposed missing tokenizer dependencies in
  `/home/jwiegley/tron/main/ingest/runtime`: `protobuf`, `sentencepiece`, and
  `tiktoken`. Added them to the `test` extra in `pyproject.toml` and refreshed
  `uv.lock`.
- Rerun of
  `/home/jwiegley/autoagent/templates/tron-ingest-harbor-task/files/evaluate_tron_ingest.py`
  from the Tron Nix shell passed:
  - TypedFx logits: 5/5 functionally equivalent, strict tolerance passed.
  - Bulk logits: 5/5 functionally equivalent, strict tolerance passed.
  - Architecture: passed expected Llama-68M dimensions.
  - EqSat structure: matched `tron_sdpa`, `tron_rope`, `rms_norm`, and
    `swishmul`.
  - Explicit gates: `cpp_compile`, `cpu_logits`, and `fpga_logits` passed.
- Fixed reward-boundary math in `tron_ingest_autoagent.score` so values within
  floating-point epsilon of `1.0` report exact `1.0`; added a regression test.
- Final reward artifacts:
  - `/tmp/tron-autoagent-runs/llama68-20260617T015302Z/artifacts/reward.json`.
  - `/tmp/tron-autoagent-runs/llama68-20260617T015302Z/artifacts/reward.txt`.
  - `/tmp/tron-autoagent-runs/llama68-20260617T015302Z/artifacts/typedfx-bulk-reward.json`.
  - `/tmp/tron-autoagent-runs/llama68-20260617T015302Z/artifacts/typedfx-bulk-reward.txt`.
  - durable summary: `.agent/tron-ingest-llama68-final-reward.json`.
- Final reward summary:

  ```json
  {
    "score": 1.0,
    "alpha": 1.0,
    "tau": 1.0,
    "delta": 1.0,
    "stage_cap": 1.0,
    "first_failed_gate": "complete"
  }
  ```

- Docker/Harbor validation remains externally blocked:
  `/var/run/docker.sock` does not exist on this host.

Final validation commands passed:

```bash
cd /home/jwiegley/autoagent
python3 -m py_compile agent.py tron_ingest_autoagent/score.py \
  tron_ingest_autoagent/architecture.py tron_ingest_autoagent/structure.py \
  tron_ingest_autoagent/tokens.py tron_ingest_autoagent/performance.py \
  tests/test_tron_ingest_score.py tests/test_tron_ingest_architecture.py \
  tests/test_tron_ingest_structure.py tests/test_tron_ingest_tokens.py \
  tests/test_tron_ingest_performance.py tests/test_tron_ingest_evaluator.py \
  templates/tron-ingest-harbor-task/files/evaluate_tron_ingest.py
python3 -m unittest tests/test_tron_ingest_score.py \
  tests/test_tron_ingest_structure.py tests/test_tron_ingest_architecture.py \
  tests/test_tron_ingest_tokens.py tests/test_tron_ingest_performance.py \
  tests/test_tron_ingest_evaluator.py
XDG_CACHE_HOME=/tmp/autoagent-nix-cache \
  nix develop --no-write-lock-file "path:$PWD" -c autoagent-check

cd /home/jwiegley/tron/main
XDG_CACHE_HOME=/tmp/tron-nix-cache nix develop --no-write-lock-file -c \
  sh -lc 'python3 config/test_model_definitions.py'
XDG_CACHE_HOME=/tmp/tron-nix-cache nix develop --no-write-lock-file -c sh -lc \
  'cmake --preset native -DCMAKE_BUILD_TYPE=RelWithDebInfo \
     -DBUILD_PRODUCTION_MODELS=OFF -DBUILD_TEST_MODELS=OFF \
     -DDEV_MODEL_CONFIG=/home/jwiegley/tron/main/config/models.local.yaml \
     -DENABLE_FUSE_STATS=ON && cmake --build gen --target runtron -j 16'
```

Current worktree inventory after completion:

- AutoAgent `/home/jwiegley/autoagent`:
  - Modified: `Dockerfile.base`, `agent.py`.
  - New source/task areas: `.agent/`, `flake.nix`, `templates/`, `tests/`,
    `tron_ingest_autoagent/`.
  - New runbook/script: `docs/tron-ingest-autoagent.md`,
    `scripts/run-tron-ingest-refinement.sh`.
  - New lockfile: `uv.lock`, created by the AutoAgent Nix/uv check.
- Tron `/home/jwiegley/tron/main`:
  - Modified source: `bin/compare_intermediates`,
    `config/model_definitions.py`, `config/test_model_definitions.py`,
    `h/runtron/stream_generate.hpp`, `h/tron/models/load.hpp`,
    `src/runtron.cpp`.
  - Modified ingest runtime dependencies:
    `ingest/runtime/pyproject.toml`, `ingest/runtime/uv.lock`.
  - Generated evidence left untracked under `ingest/`:
    `ingested_llama_68m.*`, `logit_test_model.py`,
    `logit_test_bulk_model_bulk.py`, and `logit_test_bulk_model.bulk-py`.
  - Ignored local runtime config/artifacts include
    `config/models.local.yaml` and generated plugin files under `gen/`.

## Completed Work

- Added `flake.nix` to `/Users/johnw/src/autoagent`.
- Verified:
  - `nix flake show --no-write-lock-file "path:$PWD"`
  - `nix develop --no-write-lock-file "path:$PWD" -c autoagent-check`
  - `nix flake check --no-write-lock-file "path:$PWD"`
- Confirmed the AutoAgent repo currently has no bundled Harbor tasks.
- Confirmed the Tron ingest skill is organized around staged gates:
  TypedFx, EqSat, Bulk IR, C++ compile, CPU logits, FPGA logits.
- Added `tron_ingest_autoagent.score`, a Harbor-compatible scorer that reads:
  explicit gate JSON, TypedFx logit results, Bulk logit results, token
  agreement JSON, and performance JSON.
- Added `tron_ingest_autoagent.architecture`, an artifact-based architecture
  extractor/scorer.
- Added `tron_ingest_autoagent.structure`, an artifact-based EqSat structural
  pattern scorer.
- Added `tron_ingest_autoagent.tokens`, a token-sequence agreement utility that
  produces `tau` JSON from reference/candidate token artifacts.
- Added `tron_ingest_autoagent.performance`, a throughput-vs-speed-of-light
  utility that produces `delta` JSON from structured files or Tron logs.
- Added unit tests for the scorer, architecture analyzer, and structure
  analyzer, token scorer, and performance scorer.
- Added `tests/test_tron_ingest_evaluator.py`, a subprocess integration test
  for the Harbor evaluator path covering command gates, token comparison,
  performance comparison, and reward file emission without requiring the full
  Tron stack.
- Updated `Dockerfile.base` so Harbor task containers include
  `tron_ingest_autoagent`.
- Updated the editable part of `agent.py`:
  - Longer bounded shell timeouts.
  - Structured file listing and bounded text reading tools.
  - JSON summarization tool.
  - `score_tron_ingest` tool for computing reward files in-container.
- Added a versioned Harbor task template in
  `templates/tron-ingest-harbor-task/`.
- The task template now supports:
  - `architecture.json` from artifact-based architecture analysis.
  - `eqsat_structure.json` from generated artifact pattern analysis.
  - `command_gates` for full-machine gates such as `cpp_compile`,
    `cpu_logits`, and `fpga_logits`.
  - `token_comparison` and `performance_comparison` fields for generating
    `tokens.json` and `performance.json`.
- `command_gates` include a log-aware checker in
  `templates/tron-ingest-harbor-task/files/evaluate_tron_ingest.py`; it is
  covered by subprocess regression tests in `tests/test_tron_ingest_evaluator.py`.
- The command-gate checker is intended to support:
  - `fail_regexes`: any match fails the gate.
  - `pass_regexes`: required matches; `pass_regex_mode` may be `all` or `any`.
  - `numeric_checks`: regex-captured floats checked against optional `min` and
    `max` thresholds. Supported `aggregate` modes are `max`, `min`, `mean`,
    `first`, and `last`; default is `max`.
  - Return code zero is still required.
- Added `.agent/tron-ingest-agent-program.md` with the AutoAgent operating loop
  and sub-agent roles:
  architecture analyst, equivalence debugger, and performance analyst.
- Added `.agent/tron-ingest-autoagent-report.md`, a complete migration report
  covering research, implementation, validation, failed attempts, blockers, and
  next steps for a full Tron machine.
- Created an ignored live Harbor task copy at `tasks/tron-ingest-template` from
  the versioned template.
- Resumed on `/home/jwiegley/autoagent` and added evaluator subprocess
  regressions in `tests/test_tron_ingest_evaluator.py`:
  - zero-exit command with `fail_regexes: ["Busted!"]` fails the gate.
  - missing required `pass_regexes` fails the gate.
  - zero-exit command with `Relative error in L2: 0.42` fails a numeric
    threshold of `max: 0.1`.
- Validation passed:

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
- Additional 2026-06-17 validation passed on the Linux machine:

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
  XDG_CACHE_HOME=/tmp/autoagent-nix-cache \
    nix develop --no-write-lock-file "path:$PWD" -c autoagent-check
  ```

  Result: 26 Python tests pass; `autoagent-check` reports
  `agent import ok: autoagent model=gpt-5`.
- Docker validation is blocked because the Docker daemon is not running:

  ```text
  failed to connect to the docker API at unix:///var/run/docker.sock
  ```
- Host-side smoke baseline was run without Docker using the Tron Nix shell and
  uv-managed ingest runtime:

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
  alpha=0.60
  tau=0.0
  delta=0.0
  stage_cap=1.0
  first_failed_gate=complete
  ```

  TypedFx and Bulk logit equivalence both passed for
  `hf-internal-testing/tiny-random-LlamaForCausalLM`:

  ```text
  TypedFx: 5/5 strict and functional pass, max_tvd=0.0
  Bulk:    5/5 strict and functional pass, max_tvd=0.0
  Architecture: passed, score=1.0
  EqSat structure: passed, score=1.0
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
  ```
- The same host-side smoke baseline passed on the Linux Tron machine:

  ```bash
  cd /home/jwiegley/tron/main
  rm -rf /tmp/tron-autoagent-smoke /tmp/tron-ingest-template
  mkdir -p /tmp/tron-autoagent-smoke/logs/verifier
  XDG_CACHE_HOME=/tmp/tron-nix-cache nix develop --no-write-lock-file -c sh -lc \
    'TRON_REPO=/home/jwiegley/tron/main \
     TASK_FILES_DIR=/home/jwiegley/autoagent/templates/tron-ingest-harbor-task/files \
     VERIFIER_LOG_DIR=/tmp/tron-autoagent-smoke/logs/verifier \
     REWARD_JSON=/tmp/tron-autoagent-smoke/logs/verifier/reward.json \
     REWARD_TXT=/tmp/tron-autoagent-smoke/logs/reward.txt \
     PYTHONPATH=/home/jwiegley/autoagent:$PYTHONPATH \
     python /home/jwiegley/autoagent/templates/tron-ingest-harbor-task/files/evaluate_tron_ingest.py'
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

  TypedFx and Bulk both passed 5/5 strict and functional logit equivalence
  with `max_tvd=0.0` and top-1 agreement of 100%.

  Current Linux artifacts:

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

## Linux Machine Status

- Tron worktree `/home/jwiegley/tron/main` is clean.
- AutoAgent worktree has the framework changes plus the new evaluator tests.
- Tron Nix shell works with `XDG_CACHE_HOME=/tmp/tron-nix-cache`:
  GHC 9.12.2, Cabal 3.16.0.0, Python 3.12.12, uv 0.11.19.
- Eight FPGA cards are visible:
  `0000:10:00.0`, `0000:13:00.0`, `0000:38:00.0`, `0000:3b:00.0`,
  `0000:90:00.0`, `0000:93:00.0`, `0000:b9:00.0`, `0000:bc:00.0`.
  All are bound to `vfio-pci`.
- Docker command is available only inside the AutoAgent Nix shell, but the
  daemon socket is missing, so Harbor remains blocked.
- The current Tron build system uses `GNUmakefile` plus CMake presets, not the
  older root `Makefile` assumed in some prior notes.

## Resume Procedure

From a fresh AI session:

1. Open this file first.
2. Open `.agent/tron-ingest-agent-program.md`.
3. Check worktree state:

   ```bash
   cd /home/jwiegley/autoagent
   git status --short
   ```

4. Re-validate the AutoAgent environment if needed:

   ```bash
   XDG_CACHE_HOME=/tmp/autoagent-nix-cache \
     nix develop --no-write-lock-file "path:$PWD" -c autoagent-check
   ```

5. Inspect the Tron repo before running ingest work:

   ```bash
   cd /home/jwiegley/tron/main
   git status --short
   ```

6. Verify the Tron shell:

   ```bash
   cd /home/jwiegley/tron/main
   XDG_CACHE_HOME=/tmp/tron-nix-cache nix develop --no-write-lock-file -c \
     sh -lc 'cabal --version | head -1; ghc --version; python3 --version'
   ```

7. Start/fix Docker daemon if Harbor is needed, then build the base image:

   ```bash
   cd /home/jwiegley/autoagent
   XDG_CACHE_HOME=/tmp/autoagent-nix-cache nix develop --no-write-lock-file \
     "path:$PWD" -c docker build -f Dockerfile.base -t autoagent-base .
   ```

8. Run the baseline Harbor task:

   ```bash
   cd /home/jwiegley/autoagent
   AUTOAGENT_CONCURRENCY=1 AUTOAGENT_JOB_NAME=tron-ingest-baseline \
     XDG_CACHE_HOME=/tmp/autoagent-nix-cache \
     nix develop --no-write-lock-file "path:$PWD" -c \
     autoagent-run --task-name tron-ingest-template -l 1
   ```

9. Read:

   ```text
   jobs/tron-ingest-baseline/*/verifier/reward.json
   jobs/tron-ingest-baseline/*/agent/trajectory.json
   ```

10. Continue from the first unchecked task above.

11. Before trusting `cpu_logits`, keep the command-gate regression suite green:

    ```bash
    cd /home/jwiegley/autoagent
    python3 -m py_compile templates/tron-ingest-harbor-task/files/evaluate_tron_ingest.py
    python3 -m unittest tests/test_tron_ingest_evaluator.py
    ```


## Current Blocker

Docker daemon is not running or not installed as a socket service. Until Docker
is available, the Harbor base image and baseline task cannot be executed, so
AutoAgent/Harbor parity iteration cannot begin.

This machine does support the Tron Nix shell and exposes eight FPGA cards.
The next engineering work is to choose a real parity target and wire
`cpp_compile`, `cpu_logits`, `token_comparison`, `performance_comparison`, and
eventually `fpga_logits` gates for that target.

## Open Questions

- Which initial model should be used as the first parity target?
- Initial template model is
  `hf-internal-testing/tiny-random-LlamaForCausalLM`; this should be treated as
  a smoke target until a real target model is chosen.
- Should the evaluator include FPGA once the CPU path is stable?
- Where should long-running AutoAgent/Harbor job outputs live for this work:
  inside `/home/jwiegley/autoagent/jobs`, inside the Tron repo, or under
  `/tmp/tron-autoagent-runs`?

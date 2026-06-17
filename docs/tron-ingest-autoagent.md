# Tron Ingest AutoAgent Runbook

This is the repeatable process for using AutoAgent-style refinement to ingest a
new HuggingFace model into Tron, given a `PROVIDER/MODEL` slug.

The important rule: do not do implementation or iteration work in
`/home/jwiegley/tron/main`. Treat it as the clean base checkout only. Every
target gets its own branch and worktree under `/home/jwiegley/tron/`.

## Packaged Command

From a dedicated Tron branch worktree:

```bash
ingest PROVIDER/MODEL
```

Example:

```bash
cd /home/jwiegley/tron/ingest-llama-68m
ingest JackFram/llama-68m
```

If the `ingest` command is not already on `PATH`, run it through the AutoAgent
checkout wrapper:

```bash
/home/jwiegley/autoagent/scripts/ingest PROVIDER/MODEL
```

Or enter the AutoAgent Nix shell, which provides `ingest`:

```bash
nix develop /home/jwiegley/autoagent
cd /home/jwiegley/tron/ingest-my-model
ingest PROVIDER/MODEL
```

The command:

- refuses to run in `/home/jwiegley/tron/main`;
- uses the current directory as the Tron worktree by default;
- downloads the HuggingFace snapshot to `/tmp/tron-<safe-model-name>-hf`;
- converts `pytorch_model.bin` to safetensors when needed;
- runs `ingest/build-model.py` in the Tron worktree;
- writes/updates that worktree's ignored `config/models.local.yaml`;
- builds `gen/runtron` against the local model config;
- writes a verifier `model_spec.json`;
- runs the direct AutoAgent/Harbor evaluator;
- writes durable run state under
  `.autoagent/ingest/<safe-model-name>/<timestamp>/`;
- invokes the default Codex refiner between non-terminal verifier runs;
- repeats until the score reaches `--target-score` or the score legitimately
  stalls.

The older worktree-creating wrapper remains available:

```bash
/home/jwiegley/autoagent/scripts/run-tron-ingest-refinement.sh PROVIDER/MODEL
```

It is now just a compatibility wrapper around:

```bash
ingest --create-worktree PROVIDER/MODEL
```

Common options:

```bash
ingest PROVIDER/MODEL \
  --run-dir .autoagent/ingest/my-model/manual-run \
  --max-seq-length 128 \
  --executor host \
  --executor tp1 \
  --max-iterations 8 \
  --stall-patience 2
```

By default there is no iteration cap: the loop stops only at target score,
stall, error, or `--no-refiner`. Use `--max-iterations N` only when you want an
explicit safety cap.

Run setup and evaluation only, without invoking an autonomous refiner:

```bash
ingest PROVIDER/MODEL --no-refiner
```

Resume an existing worktree and skip setup stages:

```bash
ingest PROVIDER/MODEL \
  --weights /tmp/tron-my-model-safetensors \
  --skip-download \
  --skip-convert \
  --skip-generate
```

Create the worktree from `/home/jwiegley/tron/main`:

```bash
ingest PROVIDER/MODEL \
  --create-worktree \
  --branch jw/ingest-my-model \
  --worktree /home/jwiegley/tron/ingest-my-model
```

## Stop Criteria

`ingest` stops with `status=complete` when the reward reaches the target score
of `1.0`:

```text
score = min(0.70 * alpha + 0.20 * tau + 0.10 * delta, stage_cap)
```

When the score does not improve by at least `--min-improvement` for
`--stall-patience` verifier iterations, or when the same reward fingerprint
repeats, the loop stops with `status=stalled`.

There is no default iteration cap. If `--max-iterations N` is supplied, the loop
also stops with `status=max_iterations` after the Nth scored verifier run.

If `--no-refiner` is configured and the first verifier run is below the target
score, the loop stops with `status=blocked_no_refiner`. This is intentional:
the deterministic setup/evaluator path has done all it can, and an agent or
human intervention is needed for the failing gate.

Every run writes:

```text
.autoagent/ingest/<safe-model-name>/<timestamp>/
  model_spec.json
  reward.json
  reward.txt
  state.json
  logs/
```

## Refinement Hook

By default, `ingest` calls Codex between non-terminal verifier runs:

```bash
codex exec --dangerously-bypass-approvals-and-sandbox \
  -C "$TRON_REPO" \
  --add-dir "$AUTOAGENT_REPO" \
  - < "$REFINEMENT_PROMPT"
```

Override the default when you want another agent command:

```bash
TRON_INGEST_REFINER='codex exec --dangerously-bypass-approvals-and-sandbox -C "$TRON_REPO" --add-dir "$AUTOAGENT_REPO" - < "$REFINEMENT_PROMPT"' \
  ingest PROVIDER/MODEL
```

Or pass it explicitly:

```bash
ingest PROVIDER/MODEL \
  --refinement-command 'codex exec --dangerously-bypass-approvals-and-sandbox -C "$TRON_REPO" --add-dir "$AUTOAGENT_REPO" - < "$REFINEMENT_PROMPT"'
```

The command is executed from the Tron worktree and receives these environment
variables:

```text
AUTOAGENT_REPO
TRON_REPO
MODEL_SLUG
MODEL_SAFE
MODEL_CPP
RUN_DIR
MODEL_SPEC
REWARD_JSON
REFINEMENT_PROMPT
ITERATION
SCORE
FIRST_FAILED_GATE
```

The generated `REFINEMENT_PROMPT` points at the current reward, logs, and model
spec, and instructs the agent to work one failing gate at a time.

## Spec Overlays

The generic command can usually drive a model through export, TypedFx, Bulk,
and C++ compile. Full `1.0` parity also requires CPU, FPGA, token, and
performance evidence. Provide those model/runtime-specific gates with a JSON
overlay:

```bash
ingest PROVIDER/MODEL --spec-overlay .autoagent/spec-overlays/full-parity.json
```

Example overlay:

```json
{
  "explicit_gates": {
    "cpu_logits": { "passed": true, "score": 1.0 },
    "fpga_logits": { "passed": true, "score": 1.0 }
  },
  "token_results_json": ".autoagent/artifacts/tokens.json",
  "performance_results_json": ".autoagent/artifacts/performance.json"
}
```

## Manual Workflow

Use this when the script needs customization or when debugging a failed stage.

Set variables:

```bash
export MODEL_SLUG='PROVIDER/MODEL'
export MODEL_SAFE=$(printf '%s' "$MODEL_SLUG" |
  tr '[:upper:]/.' '[:lower:]--' |
  tr -cd 'a-z0-9-')
export MODEL_CPP=ingested_${MODEL_SAFE//-/_}
export BRANCH=jw/ingest-$MODEL_SAFE
export WT=/home/jwiegley/tron/ingest-$MODEL_SAFE
export RUN=/tmp/tron-autoagent-runs/$MODEL_SAFE-$(date -u +%Y%m%dT%H%M%SZ)
export HF_DIR=/tmp/tron-$MODEL_SAFE-hf
export WEIGHTS=/tmp/tron-$MODEL_SAFE-safetensors
```

Create the worktree from the clean base checkout:

```bash
cd /home/jwiegley/tron/main
git fetch origin
git switch main
git pull --ff-only
git worktree add -b "$BRANCH" "$WT" origin/main
cd "$WT"
mkdir -p "$RUN"
```

Download the HuggingFace model:

```bash
XDG_CACHE_HOME=/tmp/tron-nix-cache nix develop --no-write-lock-file -c \
  python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download("$MODEL_SLUG", local_dir="$HF_DIR")
PY
```

If the snapshot already contains `model.safetensors` or
`model.safetensors.index.json`, use `HF_DIR` as `WEIGHTS`:

```bash
export WEIGHTS="$HF_DIR"
```

If it only contains `pytorch_model.bin`, convert it:

```bash
printf '%s\n%s\nN\n' "$HF_DIR" "$WEIGHTS" | \
  XDG_CACHE_HOME=/tmp/tron-nix-cache nix develop --no-write-lock-file -c \
  python3 bin/convert_to_safetensor.py
```

Generate the plugin and local model registry entry:

```bash
XDG_CACHE_HOME=/tmp/tron-nix-cache nix develop --no-write-lock-file -c sh -lc "
  cd ingest &&
  python3 build-model.py \
    --model '$WEIGHTS' \
    --slug '$MODEL_SAFE' \
    --name '$MODEL_CPP' \
    --trace-dir 'traces/ingested-$MODEL_SAFE' \
    --plugin-dir '../gen/src/tron/h/tron/plugins' \
    --default-weights '$WEIGHTS' \
    --config '../config/models.local.yaml' \
    --max-seq-length 64 \
    --dump-all \
    -e host -e tp1
"
```

Build only the local model set:

```bash
XDG_CACHE_HOME=/tmp/tron-nix-cache nix develop --no-write-lock-file -c sh -lc "
  cmake --preset native -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DBUILD_PRODUCTION_MODELS=OFF \
    -DBUILD_TEST_MODELS=OFF \
    -DDEV_MODEL_CONFIG=$WT/config/models.local.yaml \
    -DENABLE_FUSE_STATS=ON &&
  cmake --build gen --target runtron -j 16
"
```

Write the initial evaluator spec:

```bash
cat > "$RUN/model_spec.json" <<JSON
{
  "hf_model": "$WEIGHTS",
  "work_dir": "$RUN/work",
  "device": "cpu",
  "dtype": "float32",
  "run_typedfx": true,
  "run_bulk": true,
  "typedfx_timeout_sec": 3600,
  "bulk_timeout_sec": 3600,
  "required_gates": [
    "fx_export",
    "typedfx_parse",
    "typedfx_logits",
    "eqsat_structure",
    "bulk_logits",
    "cpp_compile"
  ],
  "explicit_gates": {
    "cpp_compile": {
      "passed": true,
      "score": 1.0
    }
  },
  "command_gates": [],
  "token_comparison": null,
  "performance_comparison": null,
  "token_results_json": "",
  "performance_results_json": ""
}
JSON
```

Run the direct verifier:

```bash
cd /home/jwiegley/autoagent
XDG_CACHE_HOME=/tmp/tron-nix-cache nix develop --no-write-lock-file "path:$WT" -c sh -lc "
  TRON_REPO='$WT' \
  TASK_FILES_DIR=/home/jwiegley/autoagent/templates/tron-ingest-harbor-task/files \
  MODEL_SPEC='$RUN/model_spec.json' \
  VERIFIER_LOG_DIR='$RUN/logs' \
  REWARD_JSON='$RUN/reward.json' \
  REWARD_TXT='$RUN/reward.txt' \
  PYTHONPATH=/home/jwiegley/autoagent:\$PYTHONPATH \
  python /home/jwiegley/autoagent/templates/tron-ingest-harbor-task/files/evaluate_tron_ingest.py
"
```

Inspect the result:

```bash
cat "$RUN/reward.json"
cat "$RUN/logs/typedfx.log"
cat "$RUN/logs/bulk.log"
```

## Full Parity Gates

The initial script validates the frontend/ingest path through C++ compile. Full
parity means extending the same `model_spec.json` with CPU, token, performance,
and FPGA artifacts, then rerunning the scorer until:

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

Use this staged order:

1. TypedFx logits.
2. Bulk logits.
3. EqSat structure.
4. C++ compile.
5. CPU intermediate/logit comparison.
6. Token agreement.
7. Performance comparison.
8. FPGA parity.

For runtime commands that need FPGA or hugepages, run under `sg positron -c`.
Prefer `--instance 0,8` for one-card runs; using only `--devices` can leave the
application CPU set empty.

The useful `runtron` validation options are:

- `--output-token-file FILE`: writes generated token IDs.
- `--force-feed-text-token-file FILE`: replays a whitespace-separated token
  continuation for deterministic host-vs-FPGA comparison.
- `--intermediates-file FILE`: writes intermediate activation/logit logs.
- `--log-intermediates`: implied by `--intermediates-file`.

Compare intermediate logs with:

```bash
cd "$WT"
python3 bin/compare_intermediates \
  --reference-intermediates-file "$REFERENCE" \
  --intermediates-file "$CANDIDATE" \
  --vec-rtol 0.05
```

When free-running FPGA token sequences diverge because of small logit
differences, use forced contexts from a validated host sequence and compare
hand-authored-vs-generated `tp1` intermediate logs. Treat strict HF/host-vs-FPGA
free-generation divergence as an ingest bug only if the established
hand-authored runtime path does not show the same behavior.

## Docker/Harbor Mode

If Docker is available, build the base image and run Harbor from AutoAgent:

```bash
cd /home/jwiegley/autoagent
XDG_CACHE_HOME=/tmp/autoagent-nix-cache \
  nix develop --no-write-lock-file "path:$PWD" -c autoagent-build-base

XDG_CACHE_HOME=/tmp/autoagent-nix-cache \
  AUTOAGENT_JOB_NAME=tron-ingest-$MODEL_SAFE \
  nix develop --no-write-lock-file "path:$PWD" -c autoagent-run
```

On the current machine, Docker is blocked when `/var/run/docker.sock` does not
exist. In that case, use the direct verifier path above.

## PR Workflow

Commit and push from the target worktree, never from `/home/jwiegley/tron/main`:

```bash
cd "$WT"
git status --short
git diff --check
git add <source-files>
git commit -m "Support ingest for <model family>"
git push -u origin "$BRANCH"
gh pr create --repo positron-ai/tron --base main --head "$BRANCH"
```

Do not commit generated `ingest/*.bulk`, `ingest/*.tron`, or local
`config/models.local.yaml` unless the PR is specifically meant to version those
artifacts.

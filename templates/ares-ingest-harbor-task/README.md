# Ares Ingest Harbor Task Template

This template evaluates an Ares model-ingest target through the generated Ares
pipeline contract:

```text
frontend artifacts -> Lean ingest -> AresPlan -> TargetPlan -> backend provider
```

The verifier writes:

```text
/logs/reward.txt
/logs/verifier/reward.json
```

The reward is:

```text
score = min(
  0.60 * alpha_execution +
  0.25 * tau_tokens +
  0.15 * delta_inference,
  stage_cap
)
```

HF Transformers on PyTorch CPU is the only model-correctness oracle. Capture
HF token/logit artifacts once for the exact model/checkpoint, tokenizer,
prompt-token context, decode depth, dtype/quantization policy, deterministic
generation settings, and oracle/exporter code tuple, then reuse those artifacts
as goldens while iterating on the selected Ares backend. C++ Tron/Rinzler
artifacts are comparison, compliance, performance, and rollback evidence only;
do not put that slow lane in the normal debug loop until HF-backed backend
quality and performance show a competitive candidate. The intended debug-loop
order is cached HF logit comparison, focused backend/module slices,
short-depth generation, longer-depth generation, and only then explicit late
C++ comparison milestones.

## Expected Runtime Layout

The verifier expects the Ares repository at `ARES_REPO` inside the task
container. The default is `/ares`.

For local development, mount or copy:

```text
/Users/johnw/hera/ares-ingest-skill -> /ares
```

## Useful `model_spec.json` Fields

- `explicit_gates`: gates already proven by supplied artifacts.
- `expected_model_ids`: allowed model ids for cross-artifact consistency when a
  registry row, local checkpoint path, and HF model id are legitimate aliases.
- `prior_art_checkouts`: optional explicit paths for vLLM, llama.cpp, and MLX
  checkouts. If absent, the `ares-model-port` workflow uses
  `${ARES_PRIOR_ART_ROOT:-$HOME/db}` as the cache root and clones any missing
  official upstream repositories there for prior-art inspection:
  `https://github.com/vllm-project/vllm.git`,
  `https://github.com/ggml-org/llama.cpp.git`, and
  `https://github.com/ml-explore/mlx.git`.
- `oracle_records`: HF CPU oracle JSONL file, relative to `work_dir`,
  `files/`, or `ARES_REPO`.
- `ares_plan`: generated AresPlan JSON artifact.
- `target_plan`: Lean-emitted backend TargetPlan JSON artifact.
- `backend_open_evidence`: backend open JSON or JSONL event evidence that
  names the backend, AresPlan/TargetPlan SHA-256 values, and proves no
  runtime-generated plan sidecars participated.
- `one_token_logits_evidence`: one-token Ares-vs-HF CPU logit/TVD evidence
  with replay-context metadata.
- `eight_token_greedy_evidence`: validator-backed greedy token evidence with
  at least eight generated tokens, source digests, HF CPU oracle provenance, and
  exact Ares-vs-oracle token identity.
- `introspection_ladder`: optional `ares.introspection.ladder.v1` report for
  token/logit/TVD drift localization. It is validated as non-scoring
  semantic-localization evidence and surfaced in refiner prompts; it does not
  replace HF CPU oracle, TargetPlan, backend, token, C++ comparison, or
  production-readiness evidence. When the report's `trace_context` references
  backend event JSONL, stage-event summaries, Perfetto traces, or Perfetto
  summaries, those paths and SHA-256 values are validated before prompt
  exposure, and conventional `role`, `profile`, and retained `schema` metadata
  is preserved in the validated gate detail for tracing consumers.
- `cpp_tvd_evidence`: C++ Tron/Rinzler comparison TVD evidence. This is
  comparison/rollback evidence, not an oracle.
- `depth_performance_evidence`: 8/64/512 depth-ladder performance evidence
  with correctness gates still green.
- `mmlu_pro_evidence`: MMLU Pro evidence with schema
  `ares.benchmark.mmlu_pro.v1`, produced from `third_party/systems_test` against
  the selected Ares model/backend endpoint. Verify `/v1/models` first and use
  the exact API-facing model id as `MMLU_MODEL`; that id must also have a
  matching `scripts/mmlu_pro.py` config entry in `third_party/systems_test`.
  The evidence must include hashed `/v1/models` and selected config-row
  artifacts and must meet the model spec's required coverage.
- `mmlu_model` and `mmlu_pro`: optional MMLU Pro run configuration. Use
  `mmlu_model` when the API-facing model id differs from `model`, and set
  `mmlu_pro.openai_host` plus `mmlu_pro.required_coverage_percent` for full
  profile runs.
- `command_wrapper_config`: optional settings for AutoAgent-generated runtime
  and C++ comparison wrapper commands. Wrappers default to dry-run mode.
- `execute_command_wrappers`: set to `true` only when the task should launch
  generated wrapper commands as command gates.
- `shortcut_scan`: optional boolean to run the shortcut/static-sidecar scan
  even when it is not listed in `required_gates`.
- `command_gates`: optional command-backed gates.
- `token_comparison`: optional reference/candidate token files for
  `tau_tokens`; when `eight_token_greedy` is required, the evaluator converts
  this comparison into `ares.runtime.greedy_token_agreement.v1` evidence and
  validates it before closing the gate. The token files must expose
  generated-only ids as `generated_token_ids` or
  `generation.generated_token_ids`.
- `performance_comparison`: optional measured/speed target files for
  `delta_inference`.

The default setup profile should require only CPU-side gates through
`targetplan_valid`, `artifact_consistency`, and `shortcut_scan`. Enable backend
and performance gates when the task environment supplies the generated
artifacts, runtime backend, model checkpoint, and cached HF goldens needed for
a fast backend loop. Enable C++ comparison only as an explicit late milestone
when the selected backend is already HF-correct and plausibly competitive.
Enable MMLU Pro only when an OpenAI-compatible endpoint and systems_test outputs
are available. Wrapper command output still must be transformed into
validator-backed evidence files before the backend, token, C++ TVD,
depth/performance, or MMLU Pro gates can pass. The runtime wrappers consume
`ares_plan`; keep `target_plan` attached separately for TargetPlan validation
rather than treating wrapper launch output as TargetPlan proof.

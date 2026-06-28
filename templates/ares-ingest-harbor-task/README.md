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

HF Transformers on PyTorch CPU is the only model-correctness oracle. C++
Tron/Rinzler artifacts are comparison, compliance, performance, and rollback
evidence only.

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
- `cpp_tvd_evidence`: C++ Tron/Rinzler comparison TVD evidence. This is
  comparison/rollback evidence, not an oracle.
- `depth_performance_evidence`: 8/64/512 depth-ladder performance evidence
  with correctness gates still green.
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
`targetplan_valid`, `artifact_consistency`, and `shortcut_scan`. Enable backend,
C++ comparison, or performance gates only when the task environment supplies the
corresponding generated artifacts, runtime backend, model checkpoint, and
comparison binaries. Wrapper command output still must be transformed into
validator-backed evidence files before the backend, token, C++ TVD, or
depth/performance gates can pass. The runtime wrappers consume `ares_plan`;
keep `target_plan` attached separately for TargetPlan validation rather than
treating wrapper launch output as TargetPlan proof.

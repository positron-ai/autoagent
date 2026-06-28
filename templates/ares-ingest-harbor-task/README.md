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
- `oracle_records`: HF CPU oracle JSONL file, relative to `work_dir`,
  `files/`, or `ARES_REPO`.
- `ares_plan`: generated AresPlan JSON artifact.
- `target_plan`: Lean-emitted backend TargetPlan JSON artifact.
- `shortcut_scan`: optional boolean to run the shortcut/static-sidecar scan
  even when it is not listed in `required_gates`.
- `command_gates`: optional command-backed gates.
- `token_comparison`: optional reference/candidate token files for
  `tau_tokens`.
- `performance_comparison`: optional measured/speed target files for
  `delta_inference`.

The default setup profile should require only CPU-side gates through
`targetplan_valid` plus `shortcut_scan`. Enable backend, C++ comparison, or
performance gates only when the task environment supplies the corresponding
generated artifacts, runtime backend, model checkpoint, and comparison binaries.

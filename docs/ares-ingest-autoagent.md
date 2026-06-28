# Ares Ingest AutoAgent Runbook

This is the Ares-specific successor to the Tron ingest AutoAgent workflow. It
uses the same agentic refinement loop, but the target is the Ares generated
pipeline:

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
the model spec and the run handoff. The next agent should capture a real HF
CPU oracle record, then work one failing gate at a time.

## Evidence Rules

- HuggingFace Transformers on PyTorch CPU is the correctness oracle.
- C++ Tron/Rinzler is comparison, compliance, performance, and rollback
  evidence only.
- Ares/Rust output is system-under-test evidence.
- Performance does not compensate for missing correctness gates.

## Gate Order

1. `model_spec`
2. `hf_cpu_oracle`
3. `frontend_export`
4. `lean_ingest`
5. `aresplan_valid`
6. `targetplan_valid`
7. `backend_open`
8. `one_token_logits`
9. `eight_token_greedy`
10. `cpp_tvd`
11. `depth_performance`

Use hardware or C++ comparison gates only when the machine has the required
cards, checkpoints, generated artifacts, and comparison binaries.

# Ares Ingest AutoAgent Handoff

Last updated: 2026-06-29

## Objective

Evolve the AutoAgent fork into an Ares model-ingest refinement tool and pin it
as `third_party/autoagent` from the Ares repository.

## Completed

- Added the `ares_ingest_autoagent` package and `ares-ingest-agent` CLI.
- Added Ares reward scoring for `alpha_execution`, `tau_tokens`, and
  `delta_inference`, with stage caps that stop speed from hiding missing
  correctness gates.
- Added the one-failing-gate verifier/refiner loop, durable reward/state/
  handoff artifacts, and refinement prompts.
- Added an Ares Harbor task template with CPU-only artifact gates and optional
  backend/comparison/full profiles.
- Added validator-backed HF CPU oracle, AresPlan, TargetPlan,
  artifact-consistency, shortcut-scan, backend-open, one-token logits, C++ TVD,
  depth-performance, and eight-token greedy evidence gates.
- Added dry-run-by-default runtime and C++ comparison wrapper plans.
- Validated synthetic and public CPU-only diagnostic rows through the staged
  loop.
- Added an Ares runbook.

## Remaining

- Add HF CPU oracle capture wrappers.
- Add Perfetto trace parsing once the Ares perfetto skill is ported.
- Recapture or copy the public CPU-only row into durable evidence storage before
  using it as promotion evidence.
- Escalate to real backend/comparison/performance rows only when generated
  artifacts, checkpoints, and hardware are available.
- Keep the default refinement loop on cached HF CPU goldens and the selected
  Ares backend. Do not launch slow C++ Tron/Rinzler comparison until HF-backed
  backend quality and performance show a competitive candidate worth comparing.

## Ares Rules

- HF Transformers on PyTorch CPU is the correctness oracle.
- HF CPU token/logit captures should be produced once for the exact captured
  tuple and reused as goldens until that tuple changes.
- C++ Tron/Rinzler is comparison, compliance, performance, and rollback
  evidence only.
- Ares runtime execution must flow through frontend artifacts, Lean ingest,
  generated AresPlan, Lean TargetPlan, and a backend provider.
- Do not restore hand-authored Rust model plugins.

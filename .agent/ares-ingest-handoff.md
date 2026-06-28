# Ares Ingest AutoAgent Handoff

Last updated: 2026-06-28

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

## Ares Rules

- HF Transformers on PyTorch CPU is the correctness oracle.
- C++ Tron/Rinzler is comparison and rollback evidence only.
- Ares runtime execution must flow through frontend artifacts, Lean ingest,
  generated AresPlan, Lean TargetPlan, and a backend provider.
- Do not restore hand-authored Rust model plugins.

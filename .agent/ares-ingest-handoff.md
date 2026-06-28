# Ares Ingest AutoAgent Handoff

Last updated: 2026-06-28

## Objective

Evolve the AutoAgent fork into an Ares model-ingest refinement tool and pin it
as `third_party/autoagent` from the Ares repository.

## Completed

- Added the first `ares_ingest_autoagent` package skeleton.
- Added Ares reward scoring for `alpha_execution`, `tau_tokens`, and
  `delta_inference`.
- Added a lightweight `ares-ingest-agent` CLI that creates durable run
  directories and handoff files without hardware.
- Added an Ares Harbor task template with a CPU-only verifier.
- Added an Ares runbook.

## Remaining

- Expand artifact gates to call real Ares wrappers directly.
- Add HF CPU oracle capture wrappers.
- Add runtime comparison parsers for Rinzler side-by-side summaries and backend
  event JSONL.
- Add Perfetto trace parsing once the Ares perfetto skill is ported.
- Validate a real model row through the staged loop.

## Ares Rules

- HF Transformers on PyTorch CPU is the correctness oracle.
- C++ Tron/Rinzler is comparison and rollback evidence only.
- Ares runtime execution must flow through frontend artifacts, Lean ingest,
  generated AresPlan, Lean TargetPlan, and a backend provider.
- Do not restore hand-authored Rust model plugins.

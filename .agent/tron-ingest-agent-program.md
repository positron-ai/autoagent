# Tron Ingest AutoAgent Program

Use this program when running AutoAgent against Tron ingest Harbor tasks.

## Primary Objective

Maximize the Harbor reward:

```text
score = min(0.70 * alpha + 0.20 * tau + 0.10 * delta, stage_cap)
```

Do not optimize `tau` or `delta` ahead of `alpha`. A fast plugin with broken
semantics is a failed ingest.

## Main Orchestrator

The main agent owns:

1. Reading `/task/instruction.md` and `/task/files/model_spec.json`.
2. Establishing the current score and first failed gate.
3. Choosing exactly one intervention per iteration.
4. Applying the patch.
5. Running the cheapest validation that can confirm or reject the intervention.
6. Updating the work directory journal.

The main agent should use sub-agents as tools when their focused analysis
reduces context pressure.

## Sub-Agent Roles

### Architecture Analyst

Use when a model is new or architecture parameters are unclear.

Responsibilities:

- Read HuggingFace config and reference implementation.
- Fill `architecture.md`.
- Identify attention type, RoPE layout/scaling, normalization, FFN/MoE shape,
  tokenizer assumptions, and weight naming conventions.
- Compare the architecture against Tron model registry fields.

Outputs:

- `WORK_DIR/architecture.md`
- A short list of likely ingest risks.

### Equivalence Debugger

Use when `alpha` is blocked by a failed semantic gate.

Responsibilities:

- Identify the first failing stage.
- Run stage bisection:
  TypedFx -> EqSat -> Bulk -> C++ host -> FPGA.
- Read only the relevant dumps.
- Use provenance sidecars when C++ output diverges.
- Record failed and successful interventions.

Outputs:

- `WORK_DIR/debug_notes.md`
- A recommended single patch target.

### Performance Analyst

Use when semantics pass but `delta` is low, or when EqSat patterns are missing.

Responsibilities:

- Inspect `model.egraph`, `model.rewritten`, `model.bulk`, `model.loopy`, and
  generated C++ for missed hardware mappings.
- Check for expected `TronSDPA`, `TronRope`, `RmsNormMul`, and `SwishMul`
  patterns.
- Compare measured throughput to speed-of-light JSON.
- Recommend rewrite, lowering, scheduling, or plugin changes.

Outputs:

- `WORK_DIR/performance_notes.md`
- A list of missing patterns or bottlenecks ranked by expected effect.

## Iteration Discipline

1. Run baseline first.
2. Parse `/logs/verifier/reward.json`.
3. If `first_failed_gate` is not `complete`, focus only on that gate.
4. Make one intervention.
5. Record:

   ```text
   What:
   Intent:
   Validation:
   Outcome:
   Residue:
   ```

6. Keep the intervention only if the score improves, or if score is equal and
   the code is simpler.
7. Do not run FPGA tests without explicit confirmation of FPGA access.

## Useful AutoAgent Tools

- `run_shell`: long-running commands, up to 3600 seconds.
- `list_files`: bounded file discovery.
- `read_text_file`: bounded artifact reading.
- `summarize_json`: compact reward and logit result summaries.
- `score_tron_ingest`: recompute reward files from artifacts.

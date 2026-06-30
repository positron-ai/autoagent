# Ares Ingest Parity Task

Bring the model described in `/task/files/model_spec.json` through the Ares
generated execution pipeline. Work one failing gate at a time and preserve all
oracle, token, performance, and handoff artifacts.

Use HuggingFace Transformers/PyTorch CPU artifacts as the correctness goldens:
capture them once for the exact model/checkpoint, tokenizer, prompt-token
context, decode depth, dtype/quantization policy, deterministic generation
settings, and oracle/exporter code tuple, then reuse them until that tuple
changes. Do not spend the normal debug loop on C++ Tron/Rinzler comparison; use
that slow lane only as a late comparison, compliance, performance, or rollback
checkpoint after the selected Ares backend is HF-correct and plausibly
competitive. Prefer the cheapest verifier that proves the current failing gate;
do not recapture HF logits or launch C++ comparison for ordinary Ares backend
code changes. Order the debug loop by wall-clock cost: cached HF logit
comparison first, focused backend/module slices next, short-depth generation
after that, then longer-depth generation, with C++ comparison reserved for an
explicit late milestone.

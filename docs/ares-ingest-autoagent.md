# Ares Ingest AutoAgent Notes

The canonical Ares operator and developer guide lives in the parent Ares
repository at
[`doc/model-ingest-autoagent.md`](../../../doc/model-ingest-autoagent.md).

Keep Ares-facing usage documentation there, linked from the Ares README. This
fork-local file is only a package breadcrumb for developers browsing the
AutoAgent checkout directly.

From an Ares checkout, use the ambient repository environment and command:

```bash
command -v ares-ingest-agent
ares-ingest-agent PROVIDER/MODEL --cockpit --max-iterations 2
```

The Ares guide documents the cockpit UI, driver selection, steering files, run
directory layout, gate discipline, prior-art checkout policy, and evidence
rules.

The fast iteration rule is HF-first: capture HuggingFace Transformers/PyTorch
CPU token/logit artifacts once for the exact model/checkpoint, tokenizer,
prompt-token context, decode depth, dtype/quantization policy, deterministic
generation settings, and oracle/exporter code tuple, then reuse those artifacts
as goldens for backend development. Keep the slow C++ Tron/Rinzler lane out of
the normal AutoAgent debug loop; run it only as an explicit late
comparison/rollback checkpoint after the selected Ares backend has HF-backed
quality and competitive performance.

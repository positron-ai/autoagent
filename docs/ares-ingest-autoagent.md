# Ares Ingest AutoAgent Notes

The canonical Ares operator and developer guide lives in the parent Ares
repository at
[`doc/model-ingest-autoagent.md`](../../../doc/model-ingest-autoagent.md).

Keep Ares-facing usage documentation there, linked from the Ares README. This
fork-local file is only a package breadcrumb for developers browsing the
AutoAgent checkout directly.

From an Ares checkout, use the dedicated ingest shell and command:

```bash
nix develop .#ingest
ares-ingest-agent PROVIDER/MODEL --cockpit --max-iterations 2
```

The Ares guide documents the cockpit UI, driver selection, steering files, run
directory layout, gate discipline, prior-art checkout policy, and evidence
rules.

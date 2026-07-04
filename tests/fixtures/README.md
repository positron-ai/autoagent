# AutoAgent Test Fixtures

`ares_trace_report_introspection_real.json` is checked-in output from:

```bash
bin/ares-trace-report \
  --metadata tools/ares-trace/fixtures/introspection_artifacts.trace-meta.json \
  --format json \
  --limit 1
```

Keep it as real Ares report JSON so AutoAgent trace-report intake tests cover
the produced `sections.*` shape, not only synthetic dictionaries.

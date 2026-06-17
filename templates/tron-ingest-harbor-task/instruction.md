# Tron Ingest Parity Task

Ingest the model described in `/task/files/model_spec.json` into Tron.

Optimize the weighted parity score:

```text
score = min(0.70 * alpha + 0.20 * tau + 0.10 * delta, stage_cap)
```

Work stage by stage:

1. Establish the current failing gate.
2. Read only the artifacts needed to explain that gate.
3. Make one focused intervention.
4. Run the cheapest relevant validation.
5. Record the intervention in the configured work directory.

Do not skip ahead to final C++ debugging if an earlier gate fails.

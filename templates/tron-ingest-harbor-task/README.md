# Tron Ingest Harbor Task Template

This template turns a Tron model ingest target into a Harbor task that
AutoAgent can optimize against.

Copy this directory into `tasks/<task-name>/`, then edit
`files/model_spec.json`.

The task verifier writes:

```text
/logs/reward.txt
/logs/verifier/reward.json
```

The reward uses:

```text
score = min(0.70 * alpha + 0.20 * tau + 0.10 * delta, stage_cap)
```

## Expected Runtime Layout

The verifier expects the Tron repository to be available at `TRON_REPO`
inside the task container. The default is `/tron`.

For local development, mount or copy:

```text
/Users/johnw/work/positron/tron/main -> /tron
```

The initial template runs CPU-side gates only. Add token-generation and
performance commands to `model_spec.json` once the basic semantic loop is
stable.

## Useful `model_spec.json` Fields

- `expected_architecture`: optional JSON object of exact architecture fields to
  compare against the extracted metadata, such as `hidden_size`,
  `num_hidden_layers`, `num_attention_heads`, or `attention_type`.
- `expected_eqsat_patterns`: structural patterns that should appear in
  generated artifacts. Supported values are `tron_sdpa`, `tron_rope`,
  `rms_norm`, and `swishmul`.
- `command_gates`: optional command-backed gates for the full Tron machine.
- `token_comparison`: optional reference/candidate token files to produce
  `tau`.
- `performance_comparison`: optional measured/speed-of-light files to produce
  `delta`.

Example command gates:

```json
{
  "command_gates": [
    {
      "name": "cpp_compile",
      "cwd": "tron_repo",
      "command": "make build",
      "timeout_sec": 7200
    },
    {
      "name": "cpu_logits",
      "cwd": "tron_repo",
      "command": "bin/compare_intermediates MODEL_TRON_NAME",
      "timeout_sec": 3600
    }
  ]
}
```

For FPGA validation, only add an `fpga_logits` command gate on a machine with
confirmed FPGA access.

Example token/performance comparisons:

```json
{
  "token_comparison": {
    "reference": "reference_tokens.json",
    "candidate": "candidate_tokens.json",
    "output": "tokens.json"
  },
  "performance_comparison": {
    "measured": "runtron.log",
    "speed_of_light": "speed_of_light.json",
    "output": "performance.json"
  }
}
```

Token files can be JSON arrays, text token-id files, strings, or JSON objects
with `tokens`, `token_ids`, `output_tokens`, `text`, or `cases`. Performance
files can be structured JSON or logs containing `tok/s` / `tokens/s` lines.

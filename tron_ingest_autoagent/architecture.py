"""Extract and score model architecture from Tron ingest artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _first_metadata(root: Path) -> Path | None:
    if root.is_file() and root.name == "metadata.json":
        return root
    if root.is_dir():
        matches = sorted(root.rglob("metadata.json"))
        if matches:
            return matches[0]
    return None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _collect_text(root: Path, max_bytes: int = 2_000_000) -> str:
    suffixes = {".py", ".hpp", ".json", ".typedfx", ".bulk"}
    texts: list[str] = []
    if root.is_file():
        files = [root]
    else:
        files = [
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix in suffixes
        ]
    for path in sorted(files):
        try:
            data = path.read_bytes()[:max_bytes]
            texts.append(data.decode("utf-8", errors="replace"))
        except OSError:
            pass
    return "\n".join(texts)


def extract_architecture(root: Path) -> dict[str, Any]:
    metadata_path = _first_metadata(root)
    metadata = _read_json(metadata_path) if metadata_path else {}
    config = metadata.get("config", {}) if isinstance(metadata, dict) else {}
    parameters = metadata.get("parameters", []) if isinstance(metadata, dict) else []
    text = _collect_text(root)

    n_heads = config.get("num_attention_heads")
    n_kv_heads = config.get("num_key_value_heads", n_heads)
    head_dim = config.get("head_dim")
    hidden_size = config.get("hidden_size")

    if n_heads and n_kv_heads:
        if n_heads == n_kv_heads:
            attention_type = "MHA"
        elif n_kv_heads == 1:
            attention_type = "MQA"
        else:
            attention_type = "GQA"
    else:
        attention_type = None

    parameter_targets = [
        str(parameter.get("target", ""))
        for parameter in parameters
        if isinstance(parameter, dict)
    ]

    has_gate = any(".mlp.gate_proj" in target for target in parameter_targets)
    has_up = any(".mlp.up_proj" in target for target in parameter_targets)
    has_down = any(
        ".mlp.down_proj" in target or ".mlp.experts.down_proj" in target
        for target in parameter_targets
    )
    has_gate_up = any(
        ".mlp.experts.gate_up_proj" in target for target in parameter_targets
    )
    has_linear_attn = any(".linear_attn." in target for target in parameter_targets)
    has_experts = any(
        "expert" in target.lower() or ".experts." in target
        for target in parameter_targets
    )
    has_rms = (
        "rms_norm" in text or "rmsnorm" in text.lower() or "input_layernorm" in text
    )

    rope_layout = None
    if match := re.search(r"rope_layout\s*=\s*['\"]([^'\"]+)['\"]", text):
        rope_layout = match.group(1)
    elif match := re.search(r"rope_layout::([A-Za-z0-9_]+)", text):
        rope_layout = match.group(1)

    return {
        "metadata_path": str(metadata_path) if metadata_path else None,
        "hidden_size": hidden_size,
        "num_hidden_layers": config.get("num_hidden_layers"),
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv_heads,
        "head_dim": head_dim,
        "vocab_size": config.get("vocab_size"),
        "intermediate_size": config.get("intermediate_size"),
        "max_position_embeddings": config.get("max_position_embeddings"),
        "rope_theta": config.get("rope_theta"),
        "rope_layout": rope_layout,
        "attention_type": attention_type,
        "normalization": "RMSNorm" if has_rms else None,
        "ffn_activation": "SwiGLU/SiLU"
        if (has_gate and has_up and has_down) or (has_gate_up and has_down)
        else None,
        "linear_attention": has_linear_attn,
        "moe": has_experts,
    }


def score_architecture(
    architecture: dict[str, Any],
    expected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}

    def add_check(name: str, passed: bool, detail: Any = None) -> None:
        checks[name] = {"passed": passed, "detail": detail}

    hidden_size = architecture.get("hidden_size")
    n_heads = architecture.get("num_attention_heads")
    n_kv_heads = architecture.get("num_key_value_heads")
    head_dim = architecture.get("head_dim")
    n_layers = architecture.get("num_hidden_layers")
    vocab_size = architecture.get("vocab_size")
    intermediate = architecture.get("intermediate_size")
    has_linear_attn = bool(architecture.get("linear_attention"))
    projected_attention = (
        n_heads * head_dim
        if isinstance(n_heads, int) and isinstance(head_dim, int)
        else None
    )
    exact_head_dim = (
        isinstance(hidden_size, int)
        and isinstance(projected_attention, int)
        and projected_attention == hidden_size
    )
    expanded_head_dim = (
        has_linear_attn
        and isinstance(hidden_size, int)
        and isinstance(projected_attention, int)
        and projected_attention >= hidden_size
    )

    add_check(
        "has_hidden_size", isinstance(hidden_size, int) and hidden_size > 0, hidden_size
    )
    add_check("has_layers", isinstance(n_layers, int) and n_layers > 0, n_layers)
    add_check("has_attention_heads", isinstance(n_heads, int) and n_heads > 0, n_heads)
    add_check("has_vocab", isinstance(vocab_size, int) and vocab_size > 0, vocab_size)
    add_check(
        "has_intermediate_size",
        isinstance(intermediate, int) and intermediate > 0,
        intermediate,
    )
    add_check(
        "head_dim_consistent",
        exact_head_dim or expanded_head_dim,
        {
            "hidden_size": hidden_size,
            "num_attention_heads": n_heads,
            "head_dim": head_dim,
            "linear_attention": has_linear_attn,
        },
    )
    add_check(
        "kv_heads_valid",
        isinstance(n_heads, int)
        and isinstance(n_kv_heads, int)
        and 0 < n_kv_heads <= n_heads
        and n_heads % n_kv_heads == 0,
        {"num_attention_heads": n_heads, "num_key_value_heads": n_kv_heads},
    )

    if expected:
        for key, expected_value in expected.items():
            actual = architecture.get(key)
            add_check(
                f"expected_{key}",
                actual == expected_value,
                {"expected": expected_value, "actual": actual},
            )

    passed_count = sum(1 for check in checks.values() if check["passed"])
    score = passed_count / len(checks) if checks else 0.0

    return {
        "passed": score == 1.0,
        "score": score,
        "architecture": architecture,
        "checks": checks,
    }


def analyze_architecture(
    root: Path, expected: dict[str, Any] | None = None
) -> dict[str, Any]:
    return score_architecture(extract_architecture(root), expected)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument(
        "--expected", type=Path, help="JSON object with expected architecture fields"
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--print-json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    expected = _read_json(args.expected) if args.expected else None
    result = analyze_architecture(args.artifact_root, expected)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    if args.print_json:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

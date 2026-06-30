from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from ares_ingest_autoagent.ares_cli import (
    CLAUDE_REFINER_COMMAND,
    DEFAULT_REFINER_COMMAND,
    AresIngestError,
    append_steering_note,
    append_steering_resource,
    build_parser,
    config_from_args,
    cockpit_checkpoint,
    initialize_run,
    main,
    selected_workflow_skills,
    slugify,
    write_handoff,
    write_refinement_prompt,
)


def trace_report_payload() -> dict:
    return {
        "schema_version": 1,
        "title": "Synthetic Ares Trace Report",
        "summary": "diagnostic trace report",
        "inputs": {
            "metadata": "run.trace-meta.json",
            "trace": "run.chrome.json",
        },
        "sections": {
            "preflight": [{"status": "pass", "ok": 2, "warn": 0}],
            "analysis_commands": [
                {
                    "purpose": "report",
                    "command": "bin/ares-trace-report --format json",
                }
            ],
            "report_grade": [
                {
                    "report_grade": "diagnostic",
                    "proof_grade_status": "not_established_by_report",
                }
            ],
            "answerability": [
                {
                    "question": "backend JSONL evidence",
                    "status": "not_present",
                    "basis": "missing backend_event_artifacts",
                }
            ],
            "unsupported_claims": [
                {
                    "claim": "backend JSONL evidence is unsupported",
                    "reason": "not_present",
                }
            ],
            "next_measurements": [
                {
                    "priority": "backend_jsonl",
                    "next_measurement": "Capture backend event JSONL",
                    "reason": "backend JSONL evidence not present",
                    "command_hint": "set ARES_BACKEND_EVENT_ARTIFACT_DIR",
                }
            ],
            "report_json_section_inventory": [
                {
                    "heading": "Trace Config Rows",
                    "json_path": "sections.trace_config_rows",
                    "json_section": "trace_config_rows",
                    "section_kind": "capture_configuration",
                    "claim_boundary": "requested_controls_not_recorded_evidence",
                },
                {
                    "heading": "Introspection Capability Rows",
                    "json_path": "sections.introspection_capability_rows",
                    "json_section": "introspection_capability_rows",
                    "section_kind": "introspection",
                    "claim_boundary": "capability_presence_not_payload_evidence",
                },
                {
                    "heading": "Provider Payload Boundary Inventory Rows",
                    "json_path": "sections.provider_payload_boundary_inventory_rows",
                    "json_section": "provider_payload_boundary_inventory_rows",
                    "section_kind": "introspection_inventory",
                    "claim_boundary": "payload_boundary_inventory_not_evidence",
                },
                {
                    "heading": "Debug Payload Artifact Summary Rows",
                    "json_path": "sections.debug_payload_artifact_summary_rows",
                    "json_section": "debug_payload_artifact_summary_rows",
                    "section_kind": "debug_payload_diagnostic",
                    "claim_boundary": "debug_payloads_can_perturb_timing",
                },
                {
                    "heading": "Token Quality Summary Rows",
                    "json_path": "sections.token_quality_summary_rows",
                    "json_section": "token_quality_summary_rows",
                    "section_kind": "sidecar",
                    "claim_boundary": "system_under_test_diagnostic_not_oracle",
                },
                {
                    "heading": "Oracle Reference Summary Rows",
                    "json_path": "sections.oracle_reference_summary_rows",
                    "json_section": "oracle_reference_summary_rows",
                    "section_kind": "sidecar",
                    "claim_boundary": (
                        "external_oracle_reference_anchor_not_sut_oracle_evidence"
                    ),
                },
                {
                    "heading": "Next Measurements",
                    "json_path": "sections.next_measurements",
                    "json_section": "next_measurements",
                    "section_kind": "measurement_guidance",
                    "claim_boundary": "next_action_not_evidence",
                },
            ],
            "trace_config_rows": [
                {
                    "config_status": "requested_and_recorded",
                    "requested_sidecar_controls": "tensor_payloads",
                    "recorded_sidecar_capabilities": "tensor_payloads",
                    "introspection_level": "payload",
                    "compile_feature_trace_introspection": True,
                    "deep_introspection_effective": True,
                    "next_action": "inspect_matching_introspection_report_sections",
                }
            ],
            "introspection_capability_rows": [
                {
                    "capture_capability": "token_quality",
                    "capability_status": "recorded",
                    "matching_artifact_count": 1,
                    "claim_boundary": "system_under_test_diagnostic_not_oracle",
                    "next_action": "inspect_token_quality_rows",
                }
            ],
            "provider_payload_boundary_inventory_rows": [
                {
                    "provider_id": "fpga",
                    "payload_lane": "kv_payload_digests",
                    "capture_status": "recorded_artifact",
                    "artifact_count": 2,
                    "matching_provider_artifact_count": 2,
                    "artifact_kind_recorded_count": 2,
                    "artifact_kind_recorded_backend_count": 1,
                    "artifact_kind_recorded_backend_ids": "fpga",
                    "report_section": "kv_payload_digest_sidecar_rows",
                    "boundary_status": "available_from_scheduler_protocol_boundary",
                    "claim_boundary": "system_under_test_scheduler_kv_payload_diagnostic",
                    "next_action": "inspect_report_section",
                }
            ],
            "debug_payload_artifact_summary_rows": [
                {
                    "artifact_kind": "attention_page_trace",
                    "payload_summary_status": "recorded",
                    "row_count": "1",
                    "byte_count": "918",
                    "sampling_policy": "selected attention page summaries",
                    "token_window": "attention-page:7004",
                    "sensitivity": "local-only",
                    "compile_features": "trace-introspection",
                    "report_section": "attention_page_trace_sidecar_rows",
                    "debug_payload_boundary": "debug_payloads_can_perturb_timing",
                    "claim_boundary": (
                        "system_under_test_numeric_localization_diagnostic"
                    ),
                }
            ],
            "token_quality_summary_rows": [
                {
                    "status": "present",
                    "evidence_role": "system_under_test",
                    "request_id": "7001",
                    "generation_id": "rinzler-7001",
                    "token_index": 0,
                    "selected_token_id": "42",
                    "selected_topk_status": "selected_is_top1",
                    "score_kind": "logprob",
                    "top1_token_id": "42",
                    "top1_score": "-0.1",
                    "runner_up_token_id": "7",
                    "runner_up_score": "-1.5",
                    "top1_margin": "1.4",
                    "temperature": "0.7",
                    "top_p": "0.9",
                    "top_k": "8",
                    "num_logprobs": "2",
                    "tokens_reused": "2",
                    "runtime_request_token_count": "4",
                    "oracle_reference": "external_hf_cpu_reference",
                    "oracle_artifact_sha256": "a" * 64,
                    "claim_boundary": (
                        "external_oracle_reference_present; "
                        "row_remains_system_under_test"
                    ),
                }
            ],
            "oracle_reference_summary_rows": [
                {
                    "status": "present",
                    "evidence_role": "system_under_test",
                    "request_id": "7001",
                    "generation_id": "rinzler-7001",
                    "token_index": 0,
                    "selected_token_id": "42",
                    "oracle_reference_role": "external_hf_cpu_reference",
                    "hf_cpu_oracle_artifact_path": "correctness_hf_cpu_oracle.txt",
                    "hf_cpu_oracle_sha256": "a" * 64,
                    "expected_oracle_source": "hf_transformers_pytorch_cpu",
                    "oracle_reference_status": "external_reference_hash_recorded",
                    "sut_classification": "system_under_test",
                    "correctness_claim_status": "not_oracle_evidence",
                    "claim_boundary": (
                        "external_hf_cpu_reference_anchor_only; "
                        "token_quality_row_remains_system_under_test"
                    ),
                }
            ],
            "introspection_artifact_summary_rows": [
                {
                    "artifact_kind": "token_quality",
                    "summary_status": "recorded_and_locally_present",
                    "artifact_count": 1,
                    "local_present_count": 1,
                    "local_missing_count": 0,
                    "row_count_total": 1,
                    "report_sections": (
                        "token_quality_summary_rows,oracle_reference_summary_rows"
                    ),
                    "claim_boundaries": "system_under_test_diagnostic_not_oracle",
                }
            ],
        },
    }


class AresIngestCliTest(unittest.TestCase):
    def test_slugify_provider_model(self) -> None:
        self.assertEqual(
            slugify("meta-llama/Llama-3.1-8B-Instruct"),
            "meta-llama-llama-3-1-8b-instruct",
        )

    def test_refiner_defaults_to_codex(self) -> None:
        args = build_parser().parse_args(["Provider/Model"])
        cfg = config_from_args(args)

        self.assertEqual(cfg.refinement_command, DEFAULT_REFINER_COMMAND)
        self.assertEqual(cfg.driver, "codex")

    def test_driver_selection_supports_claude_and_custom_commands(self) -> None:
        claude_cfg = config_from_args(
            build_parser().parse_args(["--driver", "claude", "Provider/Model"])
        )
        self.assertEqual(claude_cfg.refinement_command, CLAUDE_REFINER_COMMAND)

        custom_cfg = config_from_args(
            build_parser().parse_args(
                [
                    "--driver",
                    "custom",
                    "--driver-command",
                    'printf "%s\\n" "$FIRST_FAILED_GATE"',
                    "Provider/Model",
                ]
            )
        )
        self.assertEqual(custom_cfg.driver, "custom")
        self.assertEqual(
            custom_cfg.refinement_command,
            'printf "%s\\n" "$FIRST_FAILED_GATE"',
        )

        alias_cfg = config_from_args(
            build_parser().parse_args(
                ["--refinement-command", "echo old-alias", "Provider/Model"]
            )
        )
        self.assertEqual(alias_cfg.refinement_command, "echo old-alias")

        with self.assertRaises(AresIngestError):
            config_from_args(
                build_parser().parse_args(["--driver", "custom", "Provider/Model"])
            )

    def test_no_refiner_disables_refinement_command(self) -> None:
        args = build_parser().parse_args(["--no-refiner", "Provider/Model"])
        cfg = config_from_args(args)

        self.assertIsNone(cfg.refinement_command)

    def test_initialize_run_writes_durable_state(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            args = build_parser().parse_args(
                [
                    "--ares-repo",
                    str(root),
                    "--run-dir",
                    str(run_dir),
                    "--setup-only",
                    "Provider/Model",
                ]
            )
            cfg = config_from_args(args)

            state = initialize_run(cfg)

            self.assertEqual(state["status"], "initialized_setup_only")
            self.assertTrue((run_dir / "model_spec.json").exists())
            self.assertTrue((run_dir / "state.json").exists())
            self.assertTrue((run_dir / "reward.json").exists())
            self.assertTrue((run_dir / "reward.txt").exists())
            self.assertTrue((run_dir / "handoff.md").exists())
            self.assertTrue((run_dir / "steering.json").exists())
            self.assertTrue((run_dir / "steering.md").exists())
            spec = json.loads((run_dir / "model_spec.json").read_text())
            self.assertEqual(spec["safe_model"], "provider-model")
            self.assertEqual(spec["gate_profile"], "cpu-only")
            self.assertNotIn("cpp_tvd", spec["required_gates"])
            self.assertEqual(spec["explicit_gates"]["model_spec"]["passed"], True)
            self.assertIn("shortcut_scan", spec["required_gates"])
            self.assertIn("artifact_consistency", spec["required_gates"])
            self.assertEqual(spec["expected_model_ids"], ["Provider/Model"])
            self.assertEqual(
                spec["explicit_gates"]["shortcut_scan"]["artifact_validator"],
                "shortcut_scan",
            )
            self.assertEqual(
                spec["policy"]["hf_cpu_goldens"],
                "capture_once_per_stable_oracle_tuple",
            )
            self.assertEqual(
                spec["policy"]["ordinary_backend_loop_reference"],
                "cached_hf_cpu_token_logit_goldens",
            )
            self.assertEqual(
                spec["policy"]["debug_loop_priority"],
                [
                    "cached_hf_cpu_logit_comparison",
                    "focused_backend_or_module_slice",
                    "short_depth_backend_generation",
                    "longer_depth_backend_generation",
                    "explicit_late_cpp_comparison_milestone",
                ],
            )
            self.assertEqual(spec["policy"]["fastest_verifier_first"], True)
            self.assertEqual(
                spec["policy"]["avoid_recapturing_unchanged_hf_logits"], True
            )
            self.assertEqual(
                spec["policy"]["defer_cpp_comparison_until_milestone_candidate"],
                True,
            )
            self.assertEqual(
                state["reward"]["gates"]["shortcut_scan"]["passed"],
                True,
            )
            self.assertEqual(state["reward"]["first_failed_gate"], "hf_cpu_oracle")
            self.assertEqual(state["refinement_loop"], "setup_only")
            self.assertEqual(
                [skill["name"] for skill in state["workflow_skills"]],
                ["command-wiggum", "ares-evidence", "ares-python", "command-fess"],
            )
            self.assertIn(
                "tools/oracles/hf-cpu/",
                state["workflow_skills"][2]["allowed_scope"],
            )
            handoff = (run_dir / "handoff.md").read_text()
            self.assertIn("## Workflow Skills", handoff)
            self.assertIn("`ares-python`", handoff)
            self.assertIn("validate-jsonl", handoff)

    def test_trace_report_json_is_recorded_in_state_handoff_and_prompt(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            report_path = run_dir / "trace-report.json"
            report_path.write_text(json.dumps(trace_report_payload()))
            (run_dir / "model_spec.json").write_text(
                json.dumps(
                    {
                        "model": "Provider/Model",
                        "trace_report_json": "trace-report.json",
                    }
                )
            )
            cfg = config_from_args(
                build_parser().parse_args(
                    [
                        "--ares-repo",
                        str(root),
                        "--run-dir",
                        str(run_dir),
                        "--setup-only",
                        "Provider/Model",
                    ]
                )
            )

            state = initialize_run(cfg)

            self.assertEqual(state["reward"]["first_failed_gate"], "hf_cpu_oracle")
            self.assertEqual(state["trace_report"]["path"], str(report_path.resolve()))
            self.assertEqual(len(state["trace_report"]["sha256"]), 64)
            self.assertEqual(state["trace_report"]["report_grade"], "diagnostic")
            self.assertEqual(
                state["trace_report"]["introspection_capability_status_counts"],
                {"recorded": 1},
            )
            self.assertEqual(
                state["trace_report"]["trace_config_status_counts"],
                {"requested_and_recorded": 1},
            )
            self.assertEqual(
                state["trace_report"]["provider_payload_boundary_status_counts"],
                {"recorded_artifact": 1},
            )
            self.assertEqual(
                state["trace_report"]["debug_payload_artifact_summary_status_counts"],
                {"recorded": 1},
            )
            self.assertEqual(
                state["trace_report"]["token_quality_summary_status_counts"],
                {"present": 1},
            )
            self.assertEqual(
                state["trace_report"]["token_quality_summary_topk_status_counts"],
                {"selected_is_top1": 1},
            )
            self.assertEqual(
                state["trace_report"]["oracle_reference_summary_status_counts"],
                {"external_reference_hash_recorded": 1},
            )
            self.assertEqual(
                state["trace_report"]["oracle_reference_summary_correctness_counts"],
                {"not_oracle_evidence": 1},
            )
            self.assertEqual(state["trace_report"]["report_json_section_count"], 7)
            self.assertEqual(
                state["trace_report"]["report_json_section_kind_counts"],
                {
                    "capture_configuration": 1,
                    "debug_payload_diagnostic": 1,
                    "introspection": 1,
                    "introspection_inventory": 1,
                    "measurement_guidance": 1,
                    "sidecar": 2,
                },
            )
            spec = json.loads((run_dir / "model_spec.json").read_text())
            trace_gate = spec["validated_gates"]["trace_report"]
            self.assertTrue(trace_gate["passed"])
            self.assertEqual(
                trace_gate["detail"]["sha256"], state["trace_report"]["sha256"]
            )
            handoff = (run_dir / "handoff.md").read_text()
            self.assertIn("## Trace Report", handoff)
            self.assertIn("Report JSON sections", handoff)
            self.assertIn("Report JSON section: sections.trace_config_rows", handoff)
            self.assertIn("Trace config: requested_and_recorded", handoff)
            self.assertIn("recorded=tensor_payloads", handoff)
            self.assertIn("Provider payload boundaries", handoff)
            self.assertIn("Provider payload boundary: fpga/kv_payload_digests", handoff)
            self.assertIn("provider_artifacts=2", handoff)
            self.assertIn("same_kind_artifacts=2", handoff)
            self.assertIn("same_kind_backends=fpga", handoff)
            self.assertIn("Capture backend event JSONL", handoff)
            self.assertIn("Debug payload artifacts", handoff)
            self.assertIn("Debug payload artifact: attention_page_trace", handoff)
            self.assertIn("sensitivity=local-only", handoff)
            self.assertIn(
                "payload_boundary=debug_payloads_can_perturb_timing",
                handoff,
            )
            self.assertIn("Token quality summaries", handoff)
            self.assertIn("Token quality top-k status", handoff)
            self.assertIn("Token quality summary: request=7001", handoff)
            self.assertIn("token_index=0", handoff)
            self.assertIn("topk=selected_is_top1", handoff)
            self.assertIn("boundary=external_oracle_reference_present", handoff)
            self.assertIn("Oracle references", handoff)
            self.assertIn("Oracle-reference correctness boundary", handoff)
            self.assertIn("Oracle reference summary: request=7001", handoff)
            self.assertIn("correctness=not_oracle_evidence", handoff)
            self.assertIn("sut=system_under_test", handoff)
            self.assertIn("Introspection capability: token_quality", handoff)
            self.assertIn("set ARES_BACKEND_EVENT_ARTIFACT_DIR", handoff)

            reward = json.loads((run_dir / "reward.json").read_text())
            prompt = write_refinement_prompt(
                cfg,
                iteration=1,
                spec=spec,
                reward=reward,
            ).read_text()
            self.assertIn("Trace report JSON:", prompt)
            self.assertIn("## Trace Report Summary", prompt)
            self.assertIn("sections.answerability", prompt)
            self.assertIn("sections.report_json_section_inventory", prompt)
            self.assertIn("sections.trace_config_rows", prompt)
            self.assertIn("sections.provider_payload_boundary_inventory_rows", prompt)
            self.assertIn("sections.debug_payload_artifact_summary_rows", prompt)
            self.assertIn("sections.token_quality_summary_rows", prompt)
            self.assertIn("sections.oracle_reference_summary_rows", prompt)
            self.assertIn("Provider payload boundary: fpga/kv_payload_digests", prompt)
            self.assertIn("provider_artifacts=2", prompt)
            self.assertIn("same_kind_artifacts=2", prompt)
            self.assertIn("same_kind_backends=fpga", prompt)
            self.assertIn("Debug payload artifact: attention_page_trace", prompt)
            self.assertIn("features=trace-introspection", prompt)
            self.assertIn("Token quality summary: request=7001", prompt)
            self.assertIn("token_index=0", prompt)
            self.assertIn("top1_margin=1.4", prompt)
            self.assertIn("oracle_reference=external_hf_cpu_reference", prompt)
            self.assertIn("Oracle reference summary: request=7001", prompt)
            self.assertIn("role=external_hf_cpu_reference", prompt)
            self.assertIn("correctness=not_oracle_evidence", prompt)
            self.assertIn("system-under-test rows as oracle evidence", prompt)
            self.assertIn("inspect_matching_introspection_report_sections", prompt)
            self.assertIn("sections.introspection_capability_rows", prompt)
            self.assertIn("sections.introspection_artifact_summary_rows", prompt)
            self.assertIn("Introspection artifact: token_quality", prompt)
            self.assertIn("set ARES_BACKEND_EVENT_ARTIFACT_DIR", prompt)

    def test_no_refiner_blocks_below_target(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"

            rc = main(
                [
                    "--ares-repo",
                    str(root),
                    "--run-dir",
                    str(run_dir),
                    "--no-refiner",
                    "Provider/Model",
                ]
            )

            self.assertEqual(rc, 3)
            state = json.loads((run_dir / "state.json").read_text())
            self.assertEqual(state["status"], "blocked_no_refiner")
            self.assertEqual(state["refinement_loop"], "one_failing_gate")
            self.assertEqual(state["history"][0]["first_failed_gate"], "hf_cpu_oracle")

    def test_evidence_gate_preserves_gate_specific_skill_context(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            cfg = config_from_args(
                build_parser().parse_args(
                    [
                        "--ares-repo",
                        str(root),
                        "--run-dir",
                        str(run_dir),
                        "Provider/Model",
                    ]
                )
            )
            reward = {
                "score": 0.48,
                "alpha_execution": 0.8,
                "tau_tokens": 0.0,
                "delta_inference": 0.0,
                "stage_cap": 0.5,
                "first_failed_gate": "artifact_consistency",
                "gates": {},
            }

            skills = selected_workflow_skills(
                cfg,
                first_failed_gate="artifact_consistency",
            )
            evidence_skills = [
                skill for skill in skills if skill["name"] == "ares-evidence"
            ]

            self.assertEqual(len(evidence_skills), 2)
            self.assertEqual(evidence_skills[1]["gate"], "artifact_consistency")
            self.assertIn("same model row", evidence_skills[1]["why"])
            write_handoff(cfg, reward, state={"workflow_skills": skills, "history": []})
            handoff = (run_dir / "handoff.md").read_text()
            self.assertIn("`ares-evidence` for `artifact_consistency`", handoff)
            prompt = write_refinement_prompt(
                cfg,
                iteration=1,
                spec={},
                reward=reward,
            ).read_text()
            self.assertIn("`ares-evidence` for `artifact_consistency`", prompt)
            self.assertIn("same model row", prompt)

    def test_model_spec_gate_selects_model_port_skill_context(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            cfg = config_from_args(
                build_parser().parse_args(
                    [
                        "--ares-repo",
                        str(root),
                        "--run-dir",
                        str(run_dir),
                        "Provider/Model",
                    ]
                )
            )
            reward = {
                "score": 0.0,
                "alpha_execution": 0.0,
                "tau_tokens": 0.0,
                "delta_inference": 0.0,
                "stage_cap": 0.03,
                "first_failed_gate": "model_spec",
                "gates": {},
            }

            skills = selected_workflow_skills(cfg, first_failed_gate="model_spec")
            model_skill = next(
                skill for skill in skills if skill.get("gate") == "model_spec"
            )

            self.assertEqual(model_skill["name"], "ares-model-port")
            self.assertIn("model row inventory", model_skill["why"])
            self.assertIn("HuggingFace Transformers", model_skill["why"])
            self.assertIn("vLLM", model_skill["why"])
            self.assertIn("llama.cpp", model_skill["why"])
            self.assertIn("MLX", model_skill["why"])
            self.assertIn("${ARES_PRIOR_ART_ROOT:-$HOME/db}", model_skill["why"])
            self.assertIn(
                "clone any missing official upstream repos", model_skill["why"]
            )
            self.assertIn(
                "https://github.com/vllm-project/vllm.git", model_skill["why"]
            )
            self.assertIn(
                "https://github.com/ggml-org/llama.cpp.git",
                model_skill["why"],
            )
            self.assertIn("https://github.com/ml-explore/mlx.git", model_skill["why"])
            self.assertEqual(
                model_skill["allowed_scope"],
                [
                    str(cfg.run_dir),
                    str(cfg.model_spec_path),
                    str(cfg.run_dir / "handoff.md"),
                    "prior-art checkout paths recorded in model_spec.json or handoff.md",
                    "${ARES_PRIOR_ART_ROOT:-$HOME/db}/vllm",
                    "${ARES_PRIOR_ART_ROOT:-$HOME/db}/llama.cpp",
                    "${ARES_PRIOR_ART_ROOT:-$HOME/db}/mlx",
                ],
            )
            run_dir.mkdir()
            write_handoff(cfg, reward, state={"workflow_skills": skills, "history": []})
            handoff = (run_dir / "handoff.md").read_text()
            self.assertIn("`ares-model-port` for `model_spec`", handoff)
            self.assertIn("vLLM, llama.cpp, and MLX", handoff)
            self.assertIn("${ARES_PRIOR_ART_ROOT:-$HOME/db}/vllm", handoff)
            self.assertIn("https://github.com/vllm-project/vllm.git", handoff)
            prompt = write_refinement_prompt(
                cfg,
                iteration=1,
                spec={},
                reward=reward,
            ).read_text()
            self.assertIn("vLLM, llama.cpp, and MLX", prompt)
            self.assertIn("${ARES_PRIOR_ART_ROOT:-$HOME/db}/llama.cpp", prompt)
            self.assertIn("https://github.com/ggml-org/llama.cpp.git", prompt)
            self.assertIn(
                "read or clone official vLLM, llama.cpp, and MLX checkouts",
                prompt,
            )
            self.assertNotIn(
                "- Work inside the Ares repo and this run directory only.",
                prompt,
            )

    def test_operator_steering_is_written_to_prompt_and_handoff(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            cfg = config_from_args(
                build_parser().parse_args(
                    [
                        "--ares-repo",
                        str(root),
                        "--run-dir",
                        str(run_dir),
                        "Provider/Model",
                    ]
                )
            )
            append_steering_note(
                cfg,
                iteration=2,
                text="Prefer HF-export evidence from the local snapshot.",
            )
            append_steering_resource(
                cfg,
                iteration=2,
                value="/tmp/local-hf-export",
                note="candidate bundle",
            )
            reward = {
                "score": 0.1,
                "alpha_execution": 0.2,
                "tau_tokens": 0.0,
                "delta_inference": 0.0,
                "stage_cap": 0.18,
                "first_failed_gate": "frontend_export",
                "gates": {},
            }

            write_handoff(cfg, reward, state={"history": []})
            handoff = (run_dir / "handoff.md").read_text()
            self.assertIn("## Operator Steering", handoff)
            self.assertIn("Prefer HF-export evidence", handoff)
            self.assertIn("/tmp/local-hf-export", handoff)

            prompt = write_refinement_prompt(
                cfg,
                iteration=2,
                spec={},
                reward=reward,
            ).read_text()
            self.assertIn("## Operator Steering", prompt)
            self.assertIn("Prefer HF-export evidence", prompt)
            self.assertIn("candidate bundle", prompt)

    def test_cpp_tvd_selects_comparison_evidence_context(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = config_from_args(
                build_parser().parse_args(
                    [
                        "--ares-repo",
                        str(root),
                        "--run-dir",
                        str(root / "run"),
                        "Provider/Model",
                    ]
                )
            )

            skills = selected_workflow_skills(cfg, first_failed_gate="cpp_tvd")
            cpp_skill = next(
                skill for skill in skills if skill.get("gate") == "cpp_tvd"
            )

            self.assertEqual(cpp_skill["name"], "ares-evidence")
            self.assertIn("comparison and rollback evidence", cpp_skill["why"])

    def test_mmlu_pro_gate_selects_mmlu_skill_context(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = config_from_args(
                build_parser().parse_args(
                    [
                        "--ares-repo",
                        str(root),
                        "--run-dir",
                        str(root / "run"),
                        "Provider/Model",
                    ]
                )
            )

            skills = selected_workflow_skills(cfg, first_failed_gate="mmlu_pro")
            mmlu_skill = next(
                skill for skill in skills if skill.get("gate") == "mmlu_pro"
            )

            self.assertEqual(mmlu_skill["name"], "ares-mmlu-pro")
            self.assertIn("MMLU Pro benchmark evidence", mmlu_skill["why"])
            self.assertIn("third_party/systems_test/", mmlu_skill["allowed_scope"])

    def test_targetplan_gate_selects_targetplan_skill_context(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = config_from_args(
                build_parser().parse_args(
                    [
                        "--ares-repo",
                        str(root),
                        "--run-dir",
                        str(root / "run"),
                        "Provider/Model",
                    ]
                )
            )

            skills = selected_workflow_skills(
                cfg,
                first_failed_gate="targetplan_valid",
            )
            targetplan_skill = next(
                skill for skill in skills if skill.get("gate") == "targetplan_valid"
            )

            self.assertEqual(targetplan_skill["name"], "ares-targetplan")
            self.assertIn("Rust validation", targetplan_skill["why"])
            self.assertEqual(
                targetplan_skill["allowed_scope"],
                [
                    "ingest/lean/",
                    "backend/",
                    "runtime/",
                    str(cfg.run_dir / "artifacts"),
                    str(cfg.model_spec_path),
                ],
            )
            reward = {
                "score": 0.38,
                "alpha_execution": 0.7,
                "tau_tokens": 0.0,
                "delta_inference": 0.0,
                "stage_cap": 0.48,
                "first_failed_gate": "targetplan_valid",
                "gates": {},
            }
            (cfg.run_dir).mkdir()
            write_handoff(cfg, reward, state={"workflow_skills": skills, "history": []})
            handoff = (cfg.run_dir / "handoff.md").read_text()
            self.assertIn("`ares-targetplan` for `targetplan_valid`", handoff)
            self.assertIn("runtime/", handoff)
            prompt = write_refinement_prompt(
                cfg,
                iteration=1,
                spec={},
                reward=reward,
            ).read_text()
            self.assertIn("`ares-targetplan` for `targetplan_valid`", prompt)
            self.assertIn("runtime handoff", prompt)
            self.assertIn(str(cfg.model_spec_path), prompt)

    def test_target_score_completion_does_not_invoke_refiner(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"

            rc = main(
                [
                    "--ares-repo",
                    str(root),
                    "--run-dir",
                    str(run_dir),
                    "--target-score",
                    "0.05",
                    "--refinement-command",
                    "exit 99",
                    "Provider/Model",
                ]
            )

            self.assertEqual(rc, 0)
            state = json.loads((run_dir / "state.json").read_text())
            self.assertEqual(state["status"], "complete")
            self.assertFalse((run_dir / "logs/01-refiner.log").exists())

    def test_refiner_loop_writes_prompt_log_and_history(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"

            rc = main(
                [
                    "--ares-repo",
                    str(root),
                    "--run-dir",
                    str(run_dir),
                    "--max-iterations",
                    "2",
                    "--stall-patience",
                    "5",
                    "--refinement-command",
                    'printf \'%s\\n\' "$FIRST_FAILED_GATE" > "$RUN_DIR/refiner-ran.txt"',
                    "Provider/Model",
                ]
            )

            self.assertEqual(rc, 2)
            prompt = run_dir / "prompts/refinement-01.md"
            self.assertTrue(prompt.exists())
            prompt_text = prompt.read_text()
            self.assertIn(
                "Work only the first failing gate: `hf_cpu_oracle`", prompt_text
            )
            self.assertIn("## Expected Workflow Skills", prompt_text)
            self.assertIn("`command-wiggum`", prompt_text)
            self.assertIn("`ares-python`", prompt_text)
            self.assertIn("Allowed scope:", prompt_text)
            self.assertIn("## Allowed Write Scope", prompt_text)
            self.assertIn("HF Transformers on PyTorch CPU", prompt_text)
            self.assertIn(
                f"ares-ingest-agent Provider/Model --ares-repo {root.resolve()} "
                f"--run-dir {run_dir.resolve()} --no-refiner",
                prompt_text,
            )
            self.assertEqual(
                (run_dir / "refiner-ran.txt").read_text().strip(), "hf_cpu_oracle"
            )
            self.assertTrue((run_dir / "logs/01-refiner.log").exists())
            state = json.loads((run_dir / "state.json").read_text())
            self.assertEqual(state["status"], "max_iterations")
            self.assertEqual(
                [skill["name"] for skill in state["workflow_skills"]],
                ["command-wiggum", "ares-evidence", "ares-python", "command-fess"],
            )
            self.assertEqual(
                [item["status"] for item in state["history"]],
                ["refiner_ran", "max_iterations"],
            )
            self.assertEqual(Path(state["latest_refinement_prompt"]), prompt.resolve())

    def test_cockpit_loop_records_events_and_streaming_state(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"

            rc = main(
                [
                    "--ares-repo",
                    str(root),
                    "--run-dir",
                    str(run_dir),
                    "--cockpit",
                    "--max-iterations",
                    "2",
                    "--stall-patience",
                    "5",
                    "--refinement-command",
                    "printf 'cockpit %s\\n' \"$FIRST_FAILED_GATE\"",
                    "Provider/Model",
                ]
            )

            self.assertEqual(rc, 2)
            state = json.loads((run_dir / "state.json").read_text())
            self.assertEqual(state["cockpit"]["enabled"], True)
            self.assertEqual(state["cockpit"]["stream_refiner_output"], True)
            self.assertEqual(state["driver"], "codex")
            self.assertTrue((run_dir / "cockpit.jsonl").exists())
            events = [
                json.loads(line)
                for line in (run_dir / "cockpit.jsonl").read_text().splitlines()
            ]
            self.assertTrue(any(event["status"] == "evaluated" for event in events))
            self.assertTrue(
                any(event["status"] == "continue_noninteractive" for event in events)
            )
            self.assertIn(
                "cockpit hf_cpu_oracle", (run_dir / "logs/01-refiner.log").read_text()
            )

    def test_cockpit_interactive_commands_record_steering_and_driver(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            cfg = config_from_args(
                build_parser().parse_args(
                    [
                        "--ares-repo",
                        str(root),
                        "--run-dir",
                        str(run_dir),
                        "--cockpit",
                        "--refinement-command",
                        "echo old-driver",
                        "Provider/Model",
                    ]
                )
            )
            reward = {
                "score": 0.075,
                "alpha_execution": 0.125,
                "tau_tokens": 0.0,
                "delta_inference": 0.0,
                "stage_cap": 0.1,
                "first_failed_gate": "hf_cpu_oracle",
                "gates": {},
            }
            commands = iter(
                [
                    "note use the local HF snapshot",
                    "resource /tmp/hf-export -- candidate bundle",
                    "driver echo new-driver",
                    "continue",
                ]
            )

            with (
                patch("sys.stdin.isatty", return_value=True),
                patch("builtins.input", side_effect=lambda _prompt: next(commands)),
            ):
                should_continue = cockpit_checkpoint(
                    cfg,
                    iteration=3,
                    reward=reward,
                    state={"history": []},
                    best_score=0.075,
                    stall_count=0,
                )

            self.assertTrue(should_continue)
            self.assertEqual(cfg.refinement_command, "echo new-driver")
            steering = json.loads((run_dir / "steering.json").read_text())
            self.assertEqual(steering["notes"][0]["text"], "use the local HF snapshot")
            self.assertEqual(steering["resources"][0]["value"], "/tmp/hf-export")
            self.assertEqual(steering["resources"][0]["note"], "candidate bundle")
            self.assertEqual(
                steering["driver_commands"][0]["command"],
                "echo new-driver",
            )
            events = [
                json.loads(line)
                for line in (run_dir / "cockpit.jsonl").read_text().splitlines()
            ]
            self.assertEqual(events[-1]["decision"], "continue")

    def test_missing_oracle_records_writes_failure_state_over_corrupt_state(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "state.json").write_text("{")
            (run_dir / "model_spec.json").write_text(
                json.dumps(
                    {
                        "model": "Provider/Model",
                        "required_gates": ["model_spec", "hf_cpu_oracle"],
                        "explicit_gates": {
                            "model_spec": {
                                "passed": True,
                                "score": 1.0,
                            },
                        },
                        "oracle_records": "missing-oracle.jsonl",
                    }
                )
            )

            with self.assertRaises(SystemExit) as raised:
                main(
                    [
                        "--ares-repo",
                        str(root),
                        "--run-dir",
                        str(run_dir),
                        "--no-refiner",
                        "Provider/Model",
                    ]
                )

            self.assertEqual(raised.exception.code, 1)
            state = json.loads((run_dir / "state.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertIn("missing JSON file", state["error"])
            self.assertIn("previous_state_error", state)

    def test_refiner_failure_writes_state_and_handoff(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"

            with self.assertRaises(SystemExit) as raised:
                main(
                    [
                        "--ares-repo",
                        str(root),
                        "--run-dir",
                        str(run_dir),
                        "--refinement-command",
                        "exit 7",
                        "Provider/Model",
                    ]
                )

            self.assertEqual(raised.exception.code, 1)
            state = json.loads((run_dir / "state.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertIn("refiner failed with exit 7", state["error"])
            self.assertIn(
                "ares-python",
                [skill["name"] for skill in state["workflow_skills"]],
            )
            self.assertNotIn(
                "failed",
                [skill.get("gate") for skill in state["workflow_skills"]],
            )
            handoff = (run_dir / "handoff.md").read_text()
            self.assertIn("Status: `failed`", handoff)
            self.assertIn("`ares-python` for `hf_cpu_oracle`", handoff)

    def test_main_setup_only_writes_state(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            rc = main(
                [
                    "--ares-repo",
                    str(root),
                    "--run-dir",
                    str(run_dir),
                    "--setup-only",
                    "Provider/Model",
                ]
            )

            self.assertEqual(rc, 0)
            state = json.loads((run_dir / "state.json").read_text())
            self.assertEqual(state["refinement_loop"], "setup_only")

    def test_backend_profile_writes_command_wrapper_plan(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"

            rc = main(
                [
                    "--ares-repo",
                    str(root),
                    "--run-dir",
                    str(run_dir),
                    "--gate-profile",
                    "backend",
                    "--setup-only",
                    "Provider/Model",
                ]
            )

            self.assertEqual(rc, 0)
            spec = json.loads((run_dir / "model_spec.json").read_text())
            self.assertEqual(spec["command_wrapper_plan"], "command_wrappers.json")
            plan = json.loads((run_dir / "command_wrappers.json").read_text())
            self.assertEqual(plan["schema"], "ares.autoagent.command_wrappers.v1")
            self.assertFalse(plan["execute_command_wrappers"])
            self.assertEqual(plan["command_gates"], [])
            self.assertEqual(
                {wrapper["name"] for wrapper in plan["wrappers"]},
                {"rinzler_chat_one_token", "rinzler_full_inference_smoke"},
            )
            self.assertTrue(
                all(wrapper["missing_inputs"] for wrapper in plan["wrappers"])
            )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tron_ingest_autoagent.ingest_cli import (
    DEFAULT_REFINER_COMMAND,
    IngestError,
    build_parser,
    build_runtron,
    bootstrap_dir,
    config_from_args,
    choose_weights,
    convert_weights,
    deep_merge,
    download_model,
    ensure_local_model_config,
    generate_plugin,
    has_pytorch_weights,
    has_safetensors_weights,
    reward_fingerprint,
    slugify,
    write_failure_state,
)


class TronIngestCliTest(unittest.TestCase):
    def test_slugify_provider_model(self) -> None:
        self.assertEqual(slugify("JackFram/llama-68m"), "jackfram-llama-68m")
        self.assertEqual(
            slugify("Qwen/Qwen2.5-32B-Instruct"),
            "qwen-qwen2-5-32b-instruct",
        )

    def test_deep_merge_preserves_unrelated_spec_keys(self) -> None:
        merged = deep_merge(
            {
                "explicit_gates": {"cpp_compile": {"passed": True}},
                "required_gates": ["cpp_compile"],
            },
            {
                "explicit_gates": {"cpu_logits": {"passed": True}},
                "token_results_json": "tokens.json",
            },
        )

        self.assertEqual(
            merged["explicit_gates"],
            {
                "cpp_compile": {"passed": True},
                "cpu_logits": {"passed": True},
            },
        )
        self.assertEqual(merged["required_gates"], ["cpp_compile"])
        self.assertEqual(merged["token_results_json"], "tokens.json")

    def test_reward_fingerprint_omits_verbose_gate_detail(self) -> None:
        fingerprint = reward_fingerprint(
            {
                "score": 0.82,
                "alpha": 1.0,
                "tau": 0.5,
                "delta": 0.1,
                "stage_cap": 0.82,
                "first_failed_gate": "cpu_logits",
                "gates": {
                    "cpu_logits": {
                        "passed": False,
                        "score": 0.0,
                        "log": "/tmp/large.log",
                    }
                },
            }
        )

        self.assertEqual(
            fingerprint["gates"]["cpu_logits"],
            {"passed": False, "score": 0.0},
        )

    def test_refiner_defaults_to_autonomous_codex_command(self) -> None:
        args = build_parser().parse_args(["Provider/Model"])
        cfg = config_from_args(args)

        self.assertEqual(cfg.refinement_command, DEFAULT_REFINER_COMMAND)

    def test_no_refiner_disables_autonomous_loop(self) -> None:
        args = build_parser().parse_args(["--no-refiner", "Provider/Model"])
        cfg = config_from_args(args)

        self.assertIsNone(cfg.refinement_command)

    def test_generate_uses_meta_device_by_default(self) -> None:
        args = build_parser().parse_args(["Provider/Model"])
        cfg = config_from_args(args)
        cfg.weights = Path("/tmp/weights")

        with patch("tron_ingest_autoagent.ingest_cli.tron_nix") as tron_nix:
            generate_plugin(cfg, iteration=1)

        command = tron_nix.call_args.args[1]
        self.assertIn("--meta-device", command)

    def test_no_meta_device_disables_generate_flag(self) -> None:
        args = build_parser().parse_args(["--no-meta-device", "Provider/Model"])
        cfg = config_from_args(args)
        cfg.weights = Path("/tmp/weights")

        with patch("tron_ingest_autoagent.ingest_cli.tron_nix") as tron_nix:
            generate_plugin(cfg, iteration=1)

        command = tron_nix.call_args.args[1]
        self.assertNotIn("--meta-device", command)

    def test_generate_does_not_dump_all_by_default(self) -> None:
        args = build_parser().parse_args(["Provider/Model"])
        cfg = config_from_args(args)
        cfg.weights = Path("/tmp/weights")

        with patch("tron_ingest_autoagent.ingest_cli.tron_nix") as tron_nix:
            generate_plugin(cfg, iteration=1)

        command = tron_nix.call_args.args[1]
        self.assertNotIn("--dump-all", command)

    def test_dump_all_opt_in_for_generate(self) -> None:
        args = build_parser().parse_args(["--dump-all", "Provider/Model"])
        cfg = config_from_args(args)
        cfg.weights = Path("/tmp/weights")

        with patch("tron_ingest_autoagent.ingest_cli.tron_nix") as tron_nix:
            generate_plugin(cfg, iteration=1)

        command = tron_nix.call_args.args[1]
        self.assertIn("--dump-all", command)

    def test_build_clears_dev_model_config_cache_entry(self) -> None:
        args = build_parser().parse_args(["Provider/Model"])
        cfg = config_from_args(args)

        with patch("tron_ingest_autoagent.ingest_cli.tron_nix") as tron_nix:
            build_runtron(cfg, iteration=1)

        command = tron_nix.call_args.args[1]
        self.assertIn("-DDEV_MODEL_CONFIG=", command)
        self.assertNotIn("config/models.local.yaml", command)

    def test_default_has_no_iteration_cap(self) -> None:
        args = build_parser().parse_args(["Provider/Model"])
        cfg = config_from_args(args)

        self.assertIsNone(cfg.max_iterations)

    def test_explicit_iteration_cap_is_preserved(self) -> None:
        args = build_parser().parse_args(["--max-iterations", "3", "Provider/Model"])
        cfg = config_from_args(args)

        self.assertEqual(cfg.max_iterations, 3)

    def test_create_worktree_failure_state_does_not_create_target(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "ingest-provider-model"
            args = build_parser().parse_args(
                [
                    "--create-worktree",
                    "--worktree-root",
                    str(root),
                    "--worktree",
                    str(target),
                    "Provider/Model",
                ]
            )
            cfg = config_from_args(args)

            write_failure_state(cfg, IngestError("boom"))

            self.assertFalse(target.exists())
            self.assertTrue((bootstrap_dir(cfg) / "state.json").exists())

    def test_ensure_local_model_config_copies_template(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / "config"
            config_dir.mkdir()
            template = config_dir / "models.local.template.yaml"
            template.write_text("models: {}\n")
            (root / "ingest").mkdir()
            (root / "ingest" / "build-model.py").write_text("")
            (root / "CMakeLists.txt").write_text("")
            args = build_parser().parse_args(
                ["--worktree", str(root), "Provider/Model"]
            )
            cfg = config_from_args(args)

            ensure_local_model_config(cfg)

            self.assertEqual(
                (config_dir / "models.local.yaml").read_text(),
                "models: {}\n",
            )

    def test_download_model_resumes_partial_snapshot(self) -> None:
        with TemporaryDirectory() as tmp:
            hf_dir = Path(tmp) / "hf"
            hf_dir.mkdir()
            (hf_dir / "config.json").write_text("{}\n")
            args = build_parser().parse_args(
                ["--hf-dir", str(hf_dir), "Provider/Model"]
            )
            cfg = config_from_args(args)

            with patch("tron_ingest_autoagent.ingest_cli.tron_nix") as tron_nix:
                download_model(cfg)

            tron_nix.assert_called_once()
            self.assertEqual(tron_nix.call_args.kwargs["log_name"], "download.log")

    def test_download_model_skips_complete_safetensors_snapshot(self) -> None:
        with TemporaryDirectory() as tmp:
            hf_dir = Path(tmp) / "hf"
            hf_dir.mkdir()
            (hf_dir / "config.json").write_text("{}\n")
            (hf_dir / "model.safetensors").write_bytes(b"")
            args = build_parser().parse_args(
                ["--hf-dir", str(hf_dir), "Provider/Model"]
            )
            cfg = config_from_args(args)

            with patch("tron_ingest_autoagent.ingest_cli.tron_nix") as tron_nix:
                download_model(cfg)

            tron_nix.assert_not_called()

    def test_choose_weights_uses_sharded_safetensors_snapshot(self) -> None:
        with TemporaryDirectory() as tmp:
            hf_dir = Path(tmp) / "hf"
            hf_dir.mkdir()
            (hf_dir / "model-00001-of-00001.safetensors").write_bytes(b"")
            (hf_dir / "model.safetensors.index.json").write_text(
                '{"weight_map": {"model.embed_tokens.weight": '
                '"model-00001-of-00001.safetensors"}}\n'
            )
            args = build_parser().parse_args(
                ["--hf-dir", str(hf_dir), "Provider/Model"]
            )
            cfg = config_from_args(args)

            self.assertTrue(has_safetensors_weights(cfg.hf_dir))
            self.assertEqual(choose_weights(cfg), cfg.hf_dir)

    def test_choose_weights_converts_pytorch_shards(self) -> None:
        with TemporaryDirectory() as tmp:
            hf_dir = Path(tmp) / "hf"
            hf_dir.mkdir()
            (hf_dir / "pytorch_model-00001-of-00001.bin").write_bytes(b"")
            (hf_dir / "pytorch_model.bin.index.json").write_text(
                '{"weight_map": {"model.embed_tokens.weight": '
                '"pytorch_model-00001-of-00001.bin"}}\n'
            )
            args = build_parser().parse_args(
                ["--hf-dir", str(hf_dir), "Provider/Model"]
            )
            cfg = config_from_args(args)

            self.assertTrue(has_pytorch_weights(cfg.hf_dir))
            self.assertEqual(
                choose_weights(cfg),
                Path("/tmp/tron-provider-model-safetensors"),
            )

    def test_convert_weights_reports_missing_supported_weights(self) -> None:
        with TemporaryDirectory() as tmp:
            hf_dir = Path(tmp) / "hf"
            hf_dir.mkdir()
            (hf_dir / "config.json").write_text("{}\n")
            weights = Path(tmp) / "weights"
            args = build_parser().parse_args(
                [
                    "--hf-dir",
                    str(hf_dir),
                    "--weights",
                    str(weights),
                    "Provider/Model",
                ]
            )
            cfg = config_from_args(args)

            with patch("tron_ingest_autoagent.ingest_cli.tron_nix") as tron_nix:
                with self.assertRaisesRegex(IngestError, "supported model weights"):
                    convert_weights(cfg)

            tron_nix.assert_not_called()

    def test_convert_weights_skips_existing_safetensors_shards(self) -> None:
        with TemporaryDirectory() as tmp:
            weights = Path(tmp) / "weights"
            weights.mkdir()
            (weights / "model-00001-of-00001.safetensors").write_bytes(b"")
            (weights / "model.safetensors.index.json").write_text(
                '{"weight_map": {"model.embed_tokens.weight": '
                '"model-00001-of-00001.safetensors"}}\n'
            )
            args = build_parser().parse_args(
                ["--weights", str(weights), "Provider/Model"]
            )
            cfg = config_from_args(args)

            with patch("tron_ingest_autoagent.ingest_cli.tron_nix") as tron_nix:
                convert_weights(cfg)

            tron_nix.assert_not_called()


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env bash
set -euo pipefail

mkdir -p /logs/verifier
python3 /task/files/evaluate_tron_ingest.py

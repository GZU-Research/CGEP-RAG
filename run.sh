#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${CGEPRAG_PYTHON:-python}"
exec "${PYTHON_BIN}" "${PROJECT_DIR}/codes/run_methods/run_cgep_rag.py" "$@"

#!/usr/bin/env bash
# Backward-compatible strict 8-GPU probe.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
export RJOB_GPU_TARGET=8
exec bash "${SCRIPT_DIR}/probe_seta_resources.sh" "$@"

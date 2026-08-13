#!/usr/bin/env bash
# Convenience wrapper for the paired SETA+DIVE-PO 8-GPU RJob.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
export TRAIN_VARIANT=divepo
exec bash "${SCRIPT_DIR}/submit_seta_dapo_8g_dind.sh" "$@"

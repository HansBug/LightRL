#!/usr/bin/env bash
# Submit one privileged SETA DAPO/DIVE-PO RJob. Supported layouts are 8 GPUs
# (4 actor + 4 rollout) and the capacity fallback 5 GPUs (4 actor + 1 rollout).
# The Pod has an independent
# network namespace, so its private dockerd may use the kernel network stack
# directly without touching host Docker or paying slirp's user-space overhead.
set -euo pipefail

REPO_ROOT="/mnt/shared-storage-user/puyuan/code/LightRL"
timestamp="$(date +%Y%m%d-%H%M%S)"
TRAIN_VARIANT="${TRAIN_VARIANT:-dapo}"
case "${TRAIN_VARIANT}" in
  dapo|divepo) ;;
  *) printf 'TRAIN_VARIANT must be dapo or divepo\n' >&2; exit 2 ;;
esac
RUN_ID="${RUN_ID:-seta-fixed12-score-${TRAIN_VARIANT}-8g-dind-${timestamp}}"
RJOB_NAME="${RJOB_NAME:-lightrl-seta-${TRAIN_VARIANT}-8g-dind-${timestamp}}"
IMAGE="${LIGHTRL_RJOB_IMAGE:-registry.h.pjlab.org.cn/ailab-rlinfra-rlinfra_gpu/rft:20260408}"
PERSIST_ROOT="${LIGHTRL_PERSIST_ROOT:-/mnt/shared-storage-gpfs2/trustcyberdata/private/docker-infra/tmp/puyuan/lightrl}"
POOL_SERVER_PYTHON="${POOL_SERVER_PYTHON:-${PERSIST_ROOT}/runtime/seta-worker-py312-v1/bin/python}"
RESOURCE_PROBE_SCRIPT="${REPO_ROOT}/deploy/rjob/probe_seta_resources.sh"
RJOB_GPU_TARGET="${RJOB_GPU_TARGET:-8}"

case "${RJOB_GPU_TARGET}" in
  8)
    ACTOR_GPUS="${ACTOR_GPUS:-4}"
    ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"
    TP_SIZE="${TP_SIZE:-4}"
    ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
    ;;
  5)
    ACTOR_GPUS="${ACTOR_GPUS:-4}"
    ROLLOUT_GPUS="${ROLLOUT_GPUS:-1}"
    TP_SIZE="${TP_SIZE:-4}"
    ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
    ;;
  *)
    printf 'Supported RJOB_GPU_TARGET values are 5 and 8, got %s\n' "${RJOB_GPU_TARGET}" >&2
    exit 2
    ;;
esac
(( ACTOR_GPUS + ROLLOUT_GPUS == RJOB_GPU_TARGET )) || {
  printf 'GPU topology mismatch: actor=%s rollout=%s target=%s\n' \
    "${ACTOR_GPUS}" "${ROLLOUT_GPUS}" "${RJOB_GPU_TARGET}" >&2
  exit 2
}

[[ -x "${POOL_SERVER_PYTHON}" ]] || {
  printf 'Missing SETA worker Python: %s\n' "${POOL_SERVER_PYTHON}" >&2
  exit 1
}
# Submission only verifies that the persistent runtime contains both packages.
# A deep import initializes LiteLLM and may block on its optional online model
# cost-map fetch when the control host has transient DNS issues. The RJob's
# worker `serve` doctor still performs the real deep-import/runtime gate before
# any training process is started.
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${POOL_SERVER_PYTHON}" -c '
import importlib.util
for name in ("agentic_rl.platform.worker_cli", "camel"):
    if importlib.util.find_spec(name) is None:
        raise SystemExit(f"missing SETA worker dependency: {name}")
' >/dev/null

# Probe an 8-GPU candidate first and size CPU/RAM from that same node. This is
# intentionally non-bypassable: an operator override may shrink the safe
# request, but can never submit without a fresh prediction or exceed it.
selection="$(RJOB_GPU_TARGET="${RJOB_GPU_TARGET}" bash "${RESOURCE_PROBE_SCRIPT}")"
IFS='|' read -r RESOURCE_NODE PROBED_GPU PROBED_CPU PROBED_MEMORY_MB <<<"${selection}"
RJOB_GPU="${RJOB_GPU:-${PROBED_GPU}}"
RJOB_CPU="${RJOB_CPU:-${PROBED_CPU}}"
RJOB_MEMORY_MB="${RJOB_MEMORY_MB:-${PROBED_MEMORY_MB}}"
(( RJOB_GPU == RJOB_GPU_TARGET )) || {
  printf 'SETA formal run requires exactly the probed %s GPUs, got %s\n' \
    "${RJOB_GPU_TARGET}" "${RJOB_GPU}" >&2
  exit 3
}
(( RJOB_CPU <= PROBED_CPU && RJOB_MEMORY_MB <= PROBED_MEMORY_MB )) || {
  printf 'Requested resources exceed predicted candidate %s: requested=%sGPU/%sCPU/%sMiB available-safe=%sGPU/%sCPU/%sMiB\n' \
    "${RESOURCE_NODE}" "${RJOB_GPU}" "${RJOB_CPU}" "${RJOB_MEMORY_MB}" \
    "${PROBED_GPU}" "${PROBED_CPU}" "${PROBED_MEMORY_MB}" >&2
  exit 3
}

printf 'Submitting RJob=%s RUN_ID=%s target=%s resources=%sGPU/%sCPU/%sMiB\n' \
  "${RJOB_NAME}" "${RUN_ID}" "${RESOURCE_NODE}" "${RJOB_GPU}" "${RJOB_CPU}" "${RJOB_MEMORY_MB}"
rjob submit \
  --name="${RJOB_NAME}" \
  --gpu="${RJOB_GPU}" --cpu="${RJOB_CPU}" --memory="${RJOB_MEMORY_MB}" \
  --charged-group=narmodel_gpu --private-machine=group -P 1 \
  --priority="${RJOB_PRIORITY:-5}" --preemptible=no \
  --image="${IMAGE}" \
  --mount=gpfs://gpfs1/puyuan:/mnt/shared-storage-user/puyuan \
  --mount=gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public \
  --mount=gpfs://gpfs2/trustcyberdata:/mnt/shared-storage-gpfs2/trustcyberdata \
  --custom-resources brainpp.cn/fuse=1 \
  --privileged=true --host-network=false --store-host-nvme=true \
  --share-host-shm=true --termination-grace-period-seconds=30 \
  -e INSIDE_RJOB=1 RUN_ID="${RUN_ID}" TRAIN_VARIANT="${TRAIN_VARIANT}" \
     NUM_GPUS="${RJOB_GPU}" ACTOR_GPUS="${ACTOR_GPUS}" \
     ROLLOUT_GPUS="${ROLLOUT_GPUS}" TP_SIZE="${TP_SIZE}" \
     ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE}" \
     DIND_SMOKE_ONLY="${DIND_SMOKE_ONLY:-0}" \
     LIGHTRL_DIND_ISOLATED_NETWORK=1 HOST_NETWORK=false \
     RJOB_DIND_NETWORK_MODE=pod-isolated \
     WANDB_MODE=offline WANDB_ENABLE=1 \
  -- bash "${REPO_ROOT}/deploy/rjob/run_seta_dapo_8g_dind.sh"

printf 'RJOB_NAME=%s\nRUN_ID=%s\n' "${RJOB_NAME}" "${RUN_ID}"

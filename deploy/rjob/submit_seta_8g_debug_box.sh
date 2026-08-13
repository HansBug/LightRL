#!/usr/bin/env bash
# Reserve one reusable 5- or 8-GPU SETA debug box after probing a schedulable
# node. CPU and RAM come from the same candidate with race margins; when no
# candidate exists, the script exits without creating a Pending RJob.
set -euo pipefail

REPO_ROOT="/mnt/shared-storage-user/puyuan/code/LightRL"
timestamp="$(date +%Y%m%d-%H%M%S)"
RJOB_GPU_TARGET="${RJOB_GPU_TARGET:-8}"
case "${RJOB_GPU_TARGET}" in 5|8) ;; *) printf 'RJOB_GPU_TARGET must be 5 or 8\n' >&2; exit 2 ;; esac
RJOB_NAME="${RJOB_NAME:-lightrl-seta-${RJOB_GPU_TARGET}g-debug-${timestamp}}"
IMAGE="${LIGHTRL_RJOB_IMAGE:-registry.h.pjlab.org.cn/ailab-rlinfra-rlinfra_gpu/rft:20260408}"
RESOURCE_PROBE_SCRIPT="${REPO_ROOT}/deploy/rjob/probe_seta_resources.sh"

selection="$(RJOB_GPU_TARGET="${RJOB_GPU_TARGET}" bash "${RESOURCE_PROBE_SCRIPT}")"
IFS='|' read -r RESOURCE_NODE PROBED_GPU PROBED_CPU PROBED_MEMORY_MB <<<"${selection}"
RJOB_GPU="${RJOB_GPU:-${PROBED_GPU}}"
RJOB_CPU="${RJOB_CPU:-${PROBED_CPU}}"
RJOB_MEMORY_MB="${RJOB_MEMORY_MB:-${PROBED_MEMORY_MB}}"

(( RJOB_GPU == RJOB_GPU_TARGET )) || {
  printf 'SETA debug box requires exactly the probed %s GPUs, got %s\n' \
    "${RJOB_GPU_TARGET}" "${RJOB_GPU}" >&2
  exit 3
}
(( RJOB_CPU <= PROBED_CPU && RJOB_MEMORY_MB <= PROBED_MEMORY_MB )) || {
  printf 'Requested resources exceed predicted candidate %s: requested=%sGPU/%sCPU/%sMiB available-safe=%sGPU/%sCPU/%sMiB\n' \
    "${RESOURCE_NODE}" "${RJOB_GPU}" "${RJOB_CPU}" "${RJOB_MEMORY_MB}" \
    "${PROBED_GPU}" "${PROBED_CPU}" "${PROBED_MEMORY_MB}" >&2
  exit 3
}

printf 'Submitting reusable debug RJob=%s target=%s resources=%sGPU/%sCPU/%sMiB\n' \
  "${RJOB_NAME}" "${RESOURCE_NODE}" "${RJOB_GPU}" "${RJOB_CPU}" "${RJOB_MEMORY_MB}"
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
  -e INSIDE_RJOB=1 LIGHTRL_DIND_ISOLATED_NETWORK=1 HOST_NETWORK=false \
     RJOB_DIND_NETWORK_MODE=pod-isolated WANDB_MODE=offline WANDB_ENABLE=1 \
  -- sleep infinity

printf 'RJOB_NAME=%s\n' "${RJOB_NAME}"

#!/usr/bin/env bash
# Print one machine-readable row: NODE|GPU|CPU|MEMORY_MIB.
# The RJob predict table itself is sent to stderr for operator visibility.
set -euo pipefail

IMAGE="${LIGHTRL_RJOB_IMAGE:-registry.h.pjlab.org.cn/ailab-rlinfra-rlinfra_gpu/rft:20260408}"
TARGET_GPU="${RJOB_GPU_TARGET:-8}"
MIN_CPU="${RJOB_MIN_CPU:-64}"
MIN_MEMORY_MIB="${RJOB_MIN_MEMORY_MB:-800000}"
CPU_MARGIN="${RJOB_CPU_MARGIN:-2}"
MEMORY_MARGIN_MIB="${RJOB_MEMORY_MARGIN_MB:-32768}"
PROBE_FILE="$(mktemp /tmp/lightrl-rjob-predict.XXXXXX)"
trap 'rm -f "${PROBE_FILE}"' EXIT

[[ "${TARGET_GPU}" =~ ^[1-8]$ ]] || {
  printf 'RJOB_GPU_TARGET must be an integer in [1, 8], got %s\n' "${TARGET_GPU}" >&2
  exit 2
}

PYTHONUNBUFFERED=1 rjob submit \
  --name="seta-${TARGET_GPU}g-probe-$(date +%m%d-%H%M)" --predict-only=true \
  --gpu="${TARGET_GPU}" --cpu=1 --memory=1000 \
  --charged-group=narmodel_gpu --private-machine=group -P 1 \
  --priority="${RJOB_PRIORITY:-5}" --preemptible=no \
  --image="${IMAGE}" \
  --mount=gpfs://gpfs1/puyuan:/mnt/shared-storage-user/puyuan \
  --mount=gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public \
  --mount=gpfs://gpfs2/trustcyberdata:/mnt/shared-storage-gpfs2/trustcyberdata \
  --custom-resources brainpp.cn/fuse=1 \
  --privileged=true --host-network=false --store-host-nvme=true \
  --share-host-shm=true --termination-grace-period-seconds=30 \
  -- /bin/true | tee "${PROBE_FILE}" >&2

selection="$({
  awk -F'|' \
    -v target_gpu="${TARGET_GPU}" \
    -v min_cpu="${MIN_CPU}" \
    -v min_mem_mib="${MIN_MEMORY_MIB}" \
    -v cpu_margin="${CPU_MARGIN}" \
    -v mem_margin_mib="${MEMORY_MARGIN_MIB}" '
      /gpu-.*host\.h\.pjlab\.org\.cn/ {
        node=$2; cpu=$3+0; mem_gib=$4+0; gpu=$5+0
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", node)
        mem_mib=int(mem_gib*1024)
        if (gpu < target_gpu || cpu < min_cpu || mem_mib < min_mem_mib) next
        # Keep a small race margin between prediction and actual submission.
        request_cpu=int(cpu)-cpu_margin
        request_mem=int((mem_mib-mem_margin_mib)/1000)*1000
        if (request_cpu < min_cpu || request_mem < min_mem_mib) next
        if (!found || request_cpu > best_cpu ||
            (request_cpu == best_cpu && request_mem > best_mem)) {
          found=1; best_node=node; best_cpu=request_cpu; best_mem=request_mem
        }
      }
      END {
        if (found) printf "%s|%d|%d|%d\n", best_node, target_gpu, best_cpu, best_mem
      }
    ' "${PROBE_FILE}"
} || true)"

if [[ -z "${selection}" ]]; then
  printf '[seta-rjob-probe] no node currently has %s GPU plus at least %s CPU/%s MiB; not submitting\n' \
    "${TARGET_GPU}" "${MIN_CPU}" "${MIN_MEMORY_MIB}" >&2
  exit 3
fi

printf '%s\n' "${selection}"

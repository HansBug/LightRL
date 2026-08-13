#!/usr/bin/env bash
# RJob foreground entrypoint: private dockerd + SETA worker + formal SETA run.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

TRAIN_VARIANT="${TRAIN_VARIANT:-dapo}"
case "${TRAIN_VARIANT}" in
  dapo|divepo) ;;
  *) printf '[seta-rjob] ERROR: TRAIN_VARIANT must be dapo or divepo\n' >&2; exit 2 ;;
esac

log() { printf '[seta-%s-rjob] %s\n' "${TRAIN_VARIANT}" "$*"; }
die() { log "ERROR: $*"; exit 1; }

RUN_ID="${RUN_ID:?RUN_ID is required}"
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs/refactor}"
RUN_DIR="${RUNS_ROOT}/${RUN_ID}"
PERSIST_ROOT="${LIGHTRL_PERSIST_ROOT:-/mnt/shared-storage-gpfs2/trustcyberdata/private/docker-infra/tmp/puyuan/lightrl}"
CKPT_ROOT="${CKPT_ROOT:-${PERSIST_ROOT}/checkpoints}"
WANDB_DIR="${WANDB_DIR:-${PERSIST_ROOT}/wandb/${RUN_ID}}"
WORKER_URLS="http://127.0.0.1:${ENV_SERVER_PORT:-18081}"
DIND_STATE_ROOT="/run/lightrl-dind/${RUN_ID:0:32}"

# The formal job is offline.  Never propagate credentials to Ray or saved
# command/config metadata even if the outer RJob environment happens to have one.
unset WANDB_API_KEY WANDB_KEY

# Privileged RJobs can make host /dev/nvidia* nodes visible even when the
# scheduler allocated only a subset.  `nvidia-smi -L` therefore is not a safe
# device boundary.  Pin CUDA/Ray to the UUID list injected by the platform so
# a 5-GPU fallback cannot collide with workloads on the other three GPUs.
if [[ -n "${NUM_GPUS:-}" ]]; then
  if [[ "${NVIDIA_VISIBLE_DEVICES:-}" == "void" && "${NUM_GPUS}" == "8" ]]; then
    # Full-node jobs own every physical GPU, so privileged visibility is safe.
    unset CUDA_VISIBLE_DEVICES
    log "platform GPU list is void, but this is a full 8-GPU allocation; using all host GPUs"
  else
    [[ "${NVIDIA_VISIBLE_DEVICES:-}" != "" && \
       "${NVIDIA_VISIBLE_DEVICES}" != "all" && \
       "${NVIDIA_VISIBLE_DEVICES}" != "void" ]] \
      || die "partial-GPU run requires an exec session with a finite NVIDIA_VISIBLE_DEVICES UUID list; raw=${NVIDIA_VISIBLE_DEVICES:-unset}"
    IFS=',' read -r -a allocated_gpu_uuids <<<"${NVIDIA_VISIBLE_DEVICES}"
    (( ${#allocated_gpu_uuids[@]} == NUM_GPUS )) \
      || die "NVIDIA_VISIBLE_DEVICES raw=${NVIDIA_VISIBLE_DEVICES@Q} count=${#allocated_gpu_uuids[@]} disagrees with NUM_GPUS=${NUM_GPUS}"
    declare -A seen_gpu_uuids=()
    for gpu_uuid in "${allocated_gpu_uuids[@]}"; do
      [[ "${gpu_uuid}" == GPU-* ]] || die "invalid allocated GPU UUID: ${gpu_uuid}"
      [[ -z "${seen_gpu_uuids[${gpu_uuid}]:-}" ]] || die "duplicate allocated GPU UUID: ${gpu_uuid}"
      seen_gpu_uuids["${gpu_uuid}"]=1
    done
    export CUDA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES}"
    log "pinned CUDA_VISIBLE_DEVICES to ${#allocated_gpu_uuids[@]} scheduler-allocated GPU UUIDs"
  fi
fi

RJOB_EGRESS_PROXY_SERVICE_HOST="${RJOB_EGRESS_PROXY_SERVICE_HOST:-httpproxy-headless.kubebrain.svc.pjlab.local}"
RJOB_EGRESS_PROXY_SERVICE_PORT="${RJOB_EGRESS_PROXY_SERVICE_PORT:-3128}"
# pu-dev:18129 is an unauthenticated tunnel to node53's localhost-only relay,
# which in turn reaches the credential-hiding runtime proxy.  No upstream
# credential crosses this boundary.  Service endpoints remain failover probes.
RJOB_EGRESS_PROXY="${RJOB_EGRESS_PROXY:-http://100.98.250.77:18129}"

select_egress_proxy() {
  local probe_url="${RJOB_EGRESS_PROXY_PROBE_URL:-https://registry-1.docker.io/v2/}"
  local -a candidates=()
  local ip candidate attempt

  if [[ -n "${RJOB_EGRESS_PROXY}" ]]; then
    candidates+=("${RJOB_EGRESS_PROXY}")
  fi
  mapfile -t proxy_ips < <(
    getent ahostsv4 "${RJOB_EGRESS_PROXY_SERVICE_HOST}" \
      | awk '{print $1}' | sort -u
  )
  for ip in "${proxy_ips[@]}"; do
    candidates+=("http://${ip}:${RJOB_EGRESS_PROXY_SERVICE_PORT}")
  done
  (( ${#candidates[@]} > 0 )) \
    || die "could not resolve any egress proxy endpoint"

  for candidate in "${candidates[@]}"; do
    for attempt in 1 2; do
      if curl --proxy "${candidate}" --noproxy '' --silent --show-error \
        --connect-timeout 4 --max-time 10 --output /dev/null "${probe_url}"; then
        RJOB_EGRESS_PROXY="${candidate}"
        log "selected healthy credential-free egress proxy endpoint=${candidate#http://}"
        export RJOB_EGRESS_PROXY
        return 0
      fi
      log "egress proxy probe failed endpoint=${candidate#http://} attempt=${attempt}"
    done
  done
  die "no healthy egress proxy endpoint can reach ${probe_url}"
}

select_egress_proxy
# Replace the RJob image's hostname-based defaults.  Leaving lowercase values
# untouched would make Docker build args bypass the endpoint selected above.
export HTTP_PROXY="${RJOB_EGRESS_PROXY}"
export HTTPS_PROXY="${RJOB_EGRESS_PROXY}"
export http_proxy="${RJOB_EGRESS_PROXY}"
export https_proxy="${RJOB_EGRESS_PROXY}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,::1,.pjlab.org.cn,.pjlab.local,.svc,10.0.0.0/8,100.64.0.0/10,172.16.0.0/12}"
export no_proxy="${no_proxy:-${NO_PROXY}}"

# Starting private Docker adds docker0 before the trainer discovers its local
# address.  `hostname -I | awk '{print $1}'` can then select the bridge IP,
# which is not a stable NCCL bootstrap interface and may fail lazily on the
# first actor all-reduce.  Pin Ray, torch.distributed, NCCL, and Gloo to the
# Pod's default-route interface instead.
RJOB_TRAIN_IFACE="$(ip -4 route show default 2>/dev/null | awk 'NR == 1 {for (i=1; i<=NF; i++) if ($i == "dev") {print $(i+1); exit}}')"
[[ -n "${RJOB_TRAIN_IFACE}" ]] || die "could not resolve the Pod default-route interface"
[[ "${RJOB_TRAIN_IFACE}" != "lo" && \
   "${RJOB_TRAIN_IFACE}" != docker* && \
   "${RJOB_TRAIN_IFACE}" != br-* && \
   "${RJOB_TRAIN_IFACE}" != veth* ]] \
  || die "unsafe trainer interface selected: ${RJOB_TRAIN_IFACE}"
RJOB_TRAIN_IP="$(ip -4 -o addr show dev "${RJOB_TRAIN_IFACE}" scope global 2>/dev/null | awk 'NR == 1 {split($4, a, "/"); print a[1]}')"
[[ -n "${RJOB_TRAIN_IP}" ]] || die "could not resolve an IPv4 address for ${RJOB_TRAIN_IFACE}"
export MASTER_ADDR="${RJOB_TRAIN_IP}"
export NCCL_SOCKET_IFNAME="=${RJOB_TRAIN_IFACE}"
export GLOO_SOCKET_IFNAME="${RJOB_TRAIN_IFACE}"
log "pinned distributed bootstrap to ${RJOB_TRAIN_IFACE}/${MASTER_ADDR} (excluding private Docker bridges)"

[[ "$(id -u)" -eq 0 ]] || die "RJob image must run this privileged entrypoint as root"
[[ ! -e "${RUN_DIR}" ]] || die "run directory already exists: ${RUN_DIR}"
mkdir -p "${RUN_DIR}/.launch-once" "${RUN_DIR}/reproducibility" "${CKPT_ROOT}" "${WANDB_DIR}"

export RUN_ID RUNS_ROOT RUN_DIR CKPT_ROOT WANDB_DIR WORKER_URLS
export WANDB_MODE=offline WANDB_ENABLE=1
export HOST_NETWORK="${HOST_NETWORK:-false}"
export LIGHTRL_DIND_ISOLATED_NETWORK="${LIGHTRL_DIND_ISOLATED_NETWORK:-1}"
export RJOB_DIND_NETWORK_MODE="${RJOB_DIND_NETWORK_MODE:-pod-isolated}"
export RJOB_DIND_INSTANCE="${RUN_ID}"
export RJOB_DIND_STATE_ROOT="${DIND_STATE_ROOT}"
export DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT:-/nvme/lightrl-dind/${RUN_ID}/docker}"
export DOCKER_OFFLINE_BUNDLE_DIR="${DOCKER_OFFLINE_BUNDLE_DIR:-${PERSIST_ROOT}/runtime/docker-rft-20260408-ubuntu22.04}"
export DOCKER_CONFIG="${DOCKER_CONFIG:-${DIND_STATE_ROOT}/docker-config}"
export POOL_SERVER_VENV="${POOL_SERVER_VENV:-${PERSIST_ROOT}/runtime/seta-worker-py312-v1}"
export POOL_SERVER_PYTHON="${POOL_SERVER_PYTHON:-${POOL_SERVER_VENV}/bin/python}"
export WORKER_MAX_TOTAL_RUNS="${WORKER_MAX_TOTAL_RUNS:-40}"
export WORKER_MAX_TASKS="${WORKER_MAX_TASKS:-32}"
export WORKER_MAX_RUNS_PER_TASK="${WORKER_MAX_RUNS_PER_TASK:-8}"
export WORKER_MAX_CONCURRENT_RESETS="${WORKER_MAX_CONCURRENT_RESETS:-32}"
# One training rollout has four distinct prompts.  A 64-CPU debug box can build
# all four in one wave; a full-node 176-CPU box uses eight so fixed12 evaluation
# needs at most two waves. Duplicate builds remain coalesced by the worker.
VISIBLE_CPUS="$(nproc 2>/dev/null || echo 1)"
if (( VISIBLE_CPUS >= 128 )); then
  DEFAULT_CONCURRENT_BUILDS=8
else
  DEFAULT_CONCURRENT_BUILDS=4
fi
export WORKER_MAX_CONCURRENT_BUILDS="${WORKER_MAX_CONCURRENT_BUILDS:-${DEFAULT_CONCURRENT_BUILDS}}"
export WORKER_MAX_CONCURRENT_CLOSES="${WORKER_MAX_CONCURRENT_CLOSES:-32}"
# Cold SETA image builds routinely exceed 90 seconds. Queueing behind the
# bounded build semaphore is healthy and must not be misclassified as failure.
export WORKER_DOCKER_BUILD_QUEUE_TIMEOUT="${WORKER_DOCKER_BUILD_QUEUE_TIMEOUT:-1200}"
export TERMINAL_RL_POOL_NAMESPACE="${TERMINAL_RL_POOL_NAMESPACE:-default}"

TRAINER_PID=""
POOL_PID=""
RAY_CLI="${RAY_CLI:-/mnt/shared-storage-user/puyuan/conda_envs/lightrft_py312/bin/ray}"
RAY_OWNED=0
DOCKER_RESTARTS=0
DOCKER_MAX_RESTARTS="${DOCKER_MAX_RESTARTS:-3}"
DOCKER_RESTART_TRIGGER="${DOCKER_RESTART_TRIGGER:-2}"

stop_owned_ray() {
  [[ "${RAY_OWNED}" == "1" ]] || return 0
  if [[ -x "${RAY_CLI}" ]]; then
    log "stopping run-owned local Ray cluster"
    timeout 45 "${RAY_CLI}" stop --force >/dev/null 2>&1 || \
      log "WARN: Ray stop did not finish cleanly; continuing infrastructure cleanup"
  fi
  RAY_OWNED=0
}

terminate_trainer() {
  [[ -n "${TRAINER_PID}" ]] || return 0
  if kill -0 "${TRAINER_PID}" 2>/dev/null; then
    kill -TERM "${TRAINER_PID}" 2>/dev/null || true
    # The launcher waits on `ray job submit`; stopping its owned cluster also
    # releases that child instead of allowing cleanup to block indefinitely.
    stop_owned_ray
    for _ in $(seq 1 30); do
      kill -0 "${TRAINER_PID}" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "${TRAINER_PID}" 2>/dev/null; then
      log "WARN: trainer ignored TERM for 30s; sending KILL"
      kill -KILL "${TRAINER_PID}" 2>/dev/null || true
    fi
  fi
  wait "${TRAINER_PID}" 2>/dev/null || true
  TRAINER_PID=""
}

restart_private_dockerd() {
  if (( DOCKER_RESTARTS >= DOCKER_MAX_RESTARTS )); then
    log "WARN: private dockerd restart budget exhausted (${DOCKER_RESTARTS}/${DOCKER_MAX_RESTARTS})"
    return 1
  fi
  DOCKER_RESTARTS=$((DOCKER_RESTARTS + 1))
  log "attempting private dockerd recovery (${DOCKER_RESTARTS}/${DOCKER_MAX_RESTARTS})"

  local daemon_pid=""
  if [[ -s "${DIND_STATE_ROOT}/docker.pid" ]]; then
    daemon_pid="$(<"${DIND_STATE_ROOT}/docker.pid")"
  elif [[ -s "${DIND_STATE_ROOT}/dockerd-launcher.pid" ]]; then
    daemon_pid="$(<"${DIND_STATE_ROOT}/dockerd-launcher.pid")"
  fi
  if [[ "${daemon_pid}" =~ ^[0-9]+$ ]] && kill -0 "${daemon_pid}" 2>/dev/null; then
    kill -TERM "${daemon_pid}" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "${daemon_pid}" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "${daemon_pid}" 2>/dev/null; then
      kill -KILL "${daemon_pid}" 2>/dev/null || true
    fi
  fi
  rm -f "${DIND_STATE_ROOT}/docker.pid" "${DIND_STATE_ROOT}/docker.sock"
  if timeout 240 bash "${REPO_ROOT}/deploy/workers/start_rjob_dind_worker.sh" doctor \
      && timeout 15 docker info >/dev/null 2>&1; then
    log "private dockerd recovery succeeded; live-restore cache/root retained"
    return 0
  fi
  log "WARN: private dockerd recovery attempt failed"
  tail -100 "${DIND_STATE_ROOT}/dockerd.log" 2>/dev/null || true
  return 1
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  terminate_trainer
  stop_owned_ray
  if [[ -f "${DIND_STATE_ROOT}/pool_server.pid" ]]; then
    POOL_PID="$(<"${DIND_STATE_ROOT}/pool_server.pid")"
    kill -TERM "${POOL_PID}" 2>/dev/null || true
  fi
  if [[ -f "${DIND_STATE_ROOT}/docker.pid" ]]; then
    kill -TERM "$(<"${DIND_STATE_ROOT}/docker.pid")" 2>/dev/null || true
  fi
  if [[ -f "${DIND_STATE_ROOT}/slirp4netns.pid" ]]; then
    kill -TERM "$(<"${DIND_STATE_ROOT}/slirp4netns.pid")" 2>/dev/null || true
  fi
  log "exit status=${status}; checkpoint=${CKPT_ROOT}/${RUN_ID}; wandb=${WANDB_DIR}"
  exit "${status}"
}
trap cleanup EXIT INT TERM

mkdir -p "${DOCKER_CONFIG}"
python3 "${REPO_ROOT}/deploy/workers/configure_seta_runtime_proxy.py" \
  --docker-config "${DOCKER_CONFIG}/config.json" \
  --proxy-url "${RJOB_EGRESS_PROXY}"

log "starting private Docker and localhost SETA worker"
bash "${REPO_ROOT}/deploy/workers/start_rjob_dind_worker.sh" serve
# shellcheck disable=SC1090
source "${DIND_STATE_ROOT}/worker.env"
POOL_PID="$(<"${DIND_STATE_ROOT}/pool_server.pid")"

log "prewarming all SETA Docker Hub base images through the explicit mirror"
bash "${REPO_ROOT}/deploy/workers/prewarm_seta_base_images.sh" \
  | tee "${RUN_DIR}/reproducibility/seta-base-prewarm.log"

SNAPSHOT_DIR="${RUN_DIR}/reproducibility/worker-snapshot"
python3 "${REPO_ROOT}/tools/reproducibility/capture_local_seta_worker_state.py" \
  --repo-root "${REPO_ROOT}" \
  --output-dir "${SNAPSHOT_DIR}" \
  --worker-url "${WORKER_URLS}"
python3 - "${SNAPSHOT_DIR}/manifest.json" "${RUN_DIR}/reproducibility/worker-state.json" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
output = Path(sys.argv[2])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
payload = {
    "worker_snapshot": "worker-snapshot",
    "worker_url": manifest["worker_url"],
    "source_archive_sha256": manifest["source_archive_sha256"],
    "source_sha256_manifest_sha256": manifest["source_sha256_manifest_sha256"],
}
output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

docker info --format '{{json .}}' >"${RUN_DIR}/reproducibility/docker-info.json"
curl --noproxy '*' --fail --silent --show-error "${WORKER_URLS}/status" \
  >"${RUN_DIR}/reproducibility/worker-status-before-train.json"

log "running representative SETA reset/evaluate smoke before formal training"
python3 "${REPO_ROOT}/tools/dev/smoke_seta_worker.py" \
  --worker-url "${WORKER_URLS}" \
  --dataset "${REPO_ROOT}/benchmarks/seta_env_convert/train_minus_eval12.filtered.jsonl" \
  --task-name "${SETA_RJOB_SMOKE_TASK:-338}" \
  --verify-runtime-proxy-url "${RJOB_EGRESS_PROXY}" \
  | tee "${RUN_DIR}/reproducibility/seta-worker-smoke.jsonl"

if [[ "${DIND_SMOKE_ONLY:-0}" == "1" ]]; then
  log "DIND_SMOKE_ONLY=1; infrastructure smoke completed without launching training"
  exit 0
fi

# A sleep-box is deliberately reusable, but two local Ray control planes must
# never share it. Fail closed if an earlier or manually launched run is active.
if pgrep -x raylet >/dev/null 2>&1 || pgrep -x gcs_server >/dev/null 2>&1; then
  die "an existing local Ray cluster is active; stop it before launching another formal run"
fi

log "launching formal 1000-rollout ${TRAIN_VARIANT} training topology=${NUM_GPUS:-8}GPU/${ACTOR_GPUS:-4}actor/${ROLLOUT_GPUS:-4}rollout"
RAY_OWNED=1
FORMAL_PRECREATED_RUN_DIR=1 \
BACKGROUND=0 \
WORKER_STATE_SNAPSHOT="${SNAPSHOT_DIR}" \
SETA_EXECUTION_PROFILE="${SETA_EXECUTION_PROFILE:-sequential-throughput-v1}" \
CHECKPOINT_MIN_FREE_GB="${CHECKPOINT_MIN_FREE_GB:-128}" \
bash "${REPO_ROOT}/runs/refactor/launch_seta_fixed12_score_8g.sh" "${TRAIN_VARIANT}" &
TRAINER_PID=$!

failures=0
docker_failures=0
while kill -0 "${TRAINER_PID}" 2>/dev/null; do
  healthy=1
  if ! timeout 15 docker info >/dev/null 2>&1; then
    log "infrastructure probe failed: private dockerd unavailable"
    docker_failures=$((docker_failures + 1))
    if (( docker_failures >= DOCKER_RESTART_TRIGGER )) \
        && restart_private_dockerd; then
      docker_failures=0
    else
      healthy=0
    fi
  else
    docker_failures=0
  fi
  if ! kill -0 "${POOL_PID}" 2>/dev/null; then
    log "infrastructure probe failed: SETA pool process exited"
    healthy=0
  elif ! curl --noproxy '*' --fail --silent --max-time 10 "${WORKER_URLS}/healthz" >/dev/null; then
    log "infrastructure probe failed: SETA pool health endpoint unavailable"
    healthy=0
  fi
  if [[ "${healthy}" == "1" ]]; then
    failures=0
  else
    failures=$((failures + 1))
  fi
  if (( failures >= 3 )); then
    log "ERROR: infrastructure failed three consecutive probes; terminating trainer"
    tail -100 "${DIND_STATE_ROOT}/dockerd.log" 2>/dev/null || true
    tail -100 "${DIND_STATE_ROOT}/pool_server.log" 2>/dev/null || true
    terminate_trainer
    exit 70
  fi
  sleep 20
done

set +e
wait "${TRAINER_PID}"
status=$?
set -e
TRAINER_PID=""
if [[ "${status}" -ne 0 ]]; then
  log "formal trainer exited with status=${status}"
  exit "${status}"
fi
log "formal trainer completed successfully"

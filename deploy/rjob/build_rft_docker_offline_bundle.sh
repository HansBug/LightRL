#!/usr/bin/env bash
# Build a complete Docker/Compose .deb bundle from the exact RFT base image,
# then run the same DinD doctor used by the formal job.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
BUNDLE_DIR="${DOCKER_OFFLINE_BUNDLE_DIR:-/mnt/shared-storage-gpfs2/trustcyberdata/private/docker-infra/tmp/puyuan/lightrl/runtime/docker-rft-20260408-ubuntu22.04}"
STAGING_DIR="${BUNDLE_DIR}.staging"

log() { printf '[docker-bundle] %s\n' "$*"; }
[[ "$(id -u)" -eq 0 ]] || { log "ERROR: root is required"; exit 1; }
command -v apt-get >/dev/null 2>&1 || { log "ERROR: apt-get unavailable"; exit 1; }

# RJob Pods do not have direct Internet egress.  The cluster-local headless
# proxy is credential-free and resolvable inside the Pod network namespace.
RJOB_EGRESS_PROXY="${RJOB_EGRESS_PROXY:-http://httpproxy-headless.kubebrain.svc.pjlab.local:3128}"
export HTTP_PROXY="${HTTP_PROXY:-${RJOB_EGRESS_PROXY}}"
export HTTPS_PROXY="${HTTPS_PROXY:-${RJOB_EGRESS_PROXY}}"
export http_proxy="${http_proxy:-${HTTP_PROXY}}"
export https_proxy="${https_proxy:-${HTTPS_PROXY}}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,::1,.pjlab.org.cn,.pjlab.local,.svc,10.0.0.0/8,172.16.0.0/12}"
export no_proxy="${no_proxy:-${NO_PROXY}}"

rm -rf "${STAGING_DIR}"
mkdir -p "${STAGING_DIR}"
rm -f /var/cache/apt/archives/*.deb

log "refreshing package metadata in the disposable bundle-builder RJob"
apt-get update

compose_package=""
for candidate in docker-compose-v2 docker-compose-plugin; do
  if apt-cache show "${candidate}" >/dev/null 2>&1; then
    compose_package="${candidate}"
    break
  fi
done
[[ -n "${compose_package}" ]] || {
  log "ERROR: neither docker-compose-v2 nor docker-compose-plugin is available"
  exit 1
}

packages=(docker.io "${compose_package}" fuse-overlayfs iptables uidmap ca-certificates)
log "downloading Docker bundle packages: ${packages[*]}"
DEBIAN_FRONTEND=noninteractive apt-get install --download-only -y --no-install-recommends "${packages[@]}"
cp -a /var/cache/apt/archives/*.deb "${STAGING_DIR}/"

(
  cd "${STAGING_DIR}"
  sha256sum ./*.deb | sort > SHA256SUMS
)
cat >"${STAGING_DIR}/BUILD_INFO.txt" <<EOF
base_image=registry.h.pjlab.org.cn/ailab-rlinfra-rlinfra_gpu/rft:20260408
os_release=$(tr '\n' ' ' </etc/os-release)
compose_package=${compose_package}
created_at=$(date --iso-8601=seconds)
EOF

rm -rf "${BUNDLE_DIR}"
mv "${STAGING_DIR}" "${BUNDLE_DIR}"
log "bundle committed: ${BUNDLE_DIR} ($(find "${BUNDLE_DIR}" -maxdepth 1 -name '*.deb' | wc -l) packages)"

if [[ "${BUNDLE_BUILD_ONLY:-0}" == "1" ]]; then
  log "BUNDLE_BUILD_ONLY=1: download phase complete; isolated DinD doctor must run in a separate RJob"
  exit 0
fi

export DOCKER_OFFLINE_BUNDLE_DIR="${BUNDLE_DIR}"
export LIGHTRL_DIND_ISOLATED_NETWORK=1 HOST_NETWORK=false
export RUN_ID="${RUN_ID:-rft-docker-bundle-doctor}"
bash "${REPO_ROOT}/deploy/workers/start_rjob_dind_worker.sh" doctor
log "bundle and DinD doctor passed"

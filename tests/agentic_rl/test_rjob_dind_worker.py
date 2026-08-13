from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "deploy" / "workers" / "start_rjob_dind_worker.sh"
POOL_SCRIPT = ROOT / "deploy" / "workers" / "run_pool_server_pu_v2.sh"
RJOB_ENTRYPOINT = ROOT / "deploy" / "rjob" / "run_seta_dapo_8g_dind.sh"
RJOB_SUBMIT = ROOT / "deploy" / "rjob" / "submit_seta_dapo_8g_dind.sh"
RJOB_DIVE_SUBMIT = ROOT / "deploy" / "rjob" / "submit_seta_divepo_8g_dind.sh"
RJOB_RESOURCE_PROBE = ROOT / "deploy" / "rjob" / "probe_seta_8g_resources.sh"
RJOB_FLEX_RESOURCE_PROBE = ROOT / "deploy" / "rjob" / "probe_seta_resources.sh"
RJOB_DEBUG_SUBMIT = ROOT / "deploy" / "rjob" / "submit_seta_8g_debug_box.sh"
BASE_PREWARM = ROOT / "deploy" / "workers" / "prewarm_seta_base_images.sh"


def test_rjob_dind_requires_network_isolation_and_nvme():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "LIGHTRL_DIND_ISOLATED_NETWORK" in text
    assert "refusing hostNetwork RJob unless RJOB_DIND_NETWORK_MODE=slirp-netns" in text
    assert "RJOB_DIND_NETWORK_MODE" in text
    assert "slirp4netns" in text
    assert "unshare --net -- bash -c" in text
    assert "--ready-fd=9" in text
    assert "--store-host-nvme=true" in text
    assert 'DOCKER_DATA_ROOT must be under /nvme' in text
    assert 'DEFAULT_DOCKER_SOCKET="/var/run/docker.sock"' in text
    assert "CAMEL Docker socket compatibility path is unusable" in text
    assert "points to another live private RJob daemon" in text
    assert "/run/lightrl-dind/*/docker.sock" in text
    assert '"live-restore": True' in text
    assert '"registry-mirrors": registry_mirrors' in text
    assert "DOCKER_REGISTRY_MIRRORS accepts only HTTPS endpoints" in text


def test_rjob_dind_fails_closed_without_docker_bundle():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "DOCKER_OFFLINE_BUNDLE_DIR" in text
    assert "docker compose version" in text
    assert '[[ "${DOCKER_STORAGE_DRIVER}" != "vfs" ]]' in text
    assert "DockerRootDir mismatch" in text


def test_inner_gpu_runtime_is_opt_in():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'ENABLE_INNER_GPU="${RJOB_DIND_ENABLE_INNER_GPU:-0}"' in text
    assert "require_command nvidia-container-runtime" in text
    assert 'if enable_inner_gpu == "1"' in text


def test_seta_pool_python_preflight_uses_real_entrypoint_import():
    text = POOL_SCRIPT.read_text(encoding="utf-8")

    assert "import agentic_rl.platform.worker_cli" in text
    assert "import camel" in text
    assert "importlib.util.find_spec" not in text
    assert 'export WORKER_MAX_TOTAL_RUNS=' in text
    assert '--max-total-runs "${WORKER_MAX_TOTAL_RUNS}"' in text
    assert "effective_max_total=${WORKER_MAX_TOTAL_RUNS}" in text


def test_formal_rjob_pins_persistent_worker_runtime_and_schedulable_resources():
    entrypoint = RJOB_ENTRYPOINT.read_text(encoding="utf-8")
    submit = RJOB_SUBMIT.read_text(encoding="utf-8")

    assert 'runtime/seta-worker-py312-v1' in entrypoint
    assert 'TRAIN_VARIANT="${TRAIN_VARIANT:-dapo}"' in entrypoint
    assert 'CUDA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES}"' in entrypoint
    assert 'count=${#allocated_gpu_uuids[@]}' in entrypoint
    assert 'partial-GPU run requires an exec session' in entrypoint
    assert 'this is a full 8-GPU allocation' in entrypoint
    assert 'ip -4 route show default' in entrypoint
    assert 'export MASTER_ADDR="${RJOB_TRAIN_IP}"' in entrypoint
    assert 'export NCCL_SOCKET_IFNAME="=${RJOB_TRAIN_IFACE}"' in entrypoint
    assert 'export GLOO_SOCKET_IFNAME="${RJOB_TRAIN_IFACE}"' in entrypoint
    assert 'excluding private Docker bridges' in entrypoint
    assert '"${TRAIN_VARIANT}" &' in entrypoint
    assert 'export POOL_SERVER_PYTHON=' in entrypoint
    assert 'DIND_SMOKE_ONLY' in entrypoint
    assert "select_egress_proxy" in entrypoint
    assert "getent ahostsv4" in entrypoint
    assert "registry-1.docker.io/v2/" in entrypoint
    assert "no healthy egress proxy endpoint" in entrypoint
    assert "http://100.98.250.77:18129" in entrypoint
    assert "100.64.0.0/10" in entrypoint
    assert "importlib.util.find_spec" in submit
    assert 'for name in ("agentic_rl.platform.worker_cli", "camel")' in submit
    assert '--gpu="${RJOB_GPU}"' in submit
    assert 'probe_seta_resources.sh' in submit
    assert '--cpu="${RJOB_CPU}"' in submit
    assert '--memory="${RJOB_MEMORY_MB}"' in submit
    assert "RJOB_SKIP_RESOURCE_PROBE" not in submit
    assert 'RJOB_GPU == RJOB_GPU_TARGET' in submit
    assert 'ACTOR_GPUS + ROLLOUT_GPUS == RJOB_GPU_TARGET' in submit
    assert 'NUM_GPUS="${RJOB_GPU}"' in submit
    assert 'ACTOR_GPUS="${ACTOR_GPUS}"' in submit
    probe_wrapper = RJOB_RESOURCE_PROBE.read_text(encoding="utf-8")
    assert 'export RJOB_GPU_TARGET=8' in probe_wrapper
    probe = RJOB_FLEX_RESOURCE_PROBE.read_text(encoding="utf-8")
    assert "--predict-only=true" in probe
    assert '--gpu="${TARGET_GPU}" --cpu=1 --memory=1000' in probe
    assert 'RJOB_GPU_TARGET:-8' in probe
    assert 'RJOB_MIN_CPU:-64' in probe
    assert 'RJOB_MIN_MEMORY_MB:-800000' in probe
    assert 'WORKER_MAX_TOTAL_RUNS="${WORKER_MAX_TOTAL_RUNS:-40}"' in entrypoint
    assert 'WORKER_MAX_CONCURRENT_RESETS="${WORKER_MAX_CONCURRENT_RESETS:-32}"' in entrypoint
    assert 'RAY_OWNED=0' in entrypoint
    assert 'stop_owned_ray' in entrypoint
    assert 'pgrep -x raylet' in entrypoint
    assert 'timeout 45 "${RAY_CLI}" stop --force' in entrypoint
    assert 'restart_private_dockerd' in entrypoint
    assert 'DOCKER_MAX_RESTARTS="${DOCKER_MAX_RESTARTS:-3}"' in entrypoint
    assert 'start_rjob_dind_worker.sh" doctor' in entrypoint
    assert 'prewarm_seta_base_images.sh' in entrypoint
    assert "--host-network=false" in submit
    assert "RJOB_DIND_NETWORK_MODE=pod-isolated" in submit
    assert 'TRAIN_VARIANT="${TRAIN_VARIANT:-dapo}"' in submit
    assert 'TRAIN_VARIANT="${TRAIN_VARIANT}"' in submit
    assert "TRAIN_VARIANT=divepo" in RJOB_DIVE_SUBMIT.read_text(encoding="utf-8")


def test_debug_box_also_requires_same_node_resource_probe():
    text = RJOB_DEBUG_SUBMIT.read_text(encoding="utf-8")

    assert 'RJOB_GPU_TARGET:-8' in text
    assert 'RJOB_GPU_TARGET="${RJOB_GPU_TARGET}" bash "${RESOURCE_PROBE_SCRIPT}"' in text
    assert 'RJOB_GPU == RJOB_GPU_TARGET' in text
    assert 'RJOB_CPU <= PROBED_CPU' in text
    assert 'RJOB_MEMORY_MB <= PROBED_MEMORY_MB' in text
    assert '--gpu="${RJOB_GPU}" --cpu="${RJOB_CPU}" --memory="${RJOB_MEMORY_MB}"' in text
    assert '-- sleep infinity' in text


def test_seta_base_prewarm_uses_explicit_mirror_and_canonical_tags():
    text = BASE_PREWARM.read_text(encoding="utf-8")

    assert "docker.m.daocloud.io/library" in text
    assert 'find "${DATASET_DIR}"' in text
    assert '$1 == "FROM"' in text
    assert 'toupper($1)' not in text
    assert 'docker pull "${mirrored}"' in text
    assert 'docker tag "${mirrored}" "${canonical}"' in text
    assert 'if [[ "${canonical}" == */* ]]' in text

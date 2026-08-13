# LightRL Node53/RJob SETA Worker 运维手册

本文说明四类容易混淆、但彼此独立的操作：

1. 从开发机连接 node53；
2. 在 node53 配置长期运行的 Docker server 与 SETA worker；
3. 通过 `rjob exec` 进入已有训练盒进行调试；
4. 在 RJob 内部启动该任务专用的 Docker server 与 SETA worker。

文中的命令以仓库
`/mnt/shared-storage-user/puyuan/code/LightRL`、RJob namespace
`ailab-narmodel` 和 SETA worker 端口 `18081` 为例。涉及代理凭据的值不得写入仓库、
训练配置、RJob 环境或任务容器。

## 1. 先选择正确的运行方式

| 目标 | 推荐方式 | Worker URL | Docker 数据位置 | 适用范围 |
|---|---|---|---|---|
| 维护共享 SETA worker | node53 常驻服务 | node53 本机为 `127.0.0.1:18081` | node53 本地 Docker root | 日常维护、共享 smoke |
| RJob 使用 node53 worker | node53 + pu-dev SSH 隧道 | `http://<pu-dev-IP>:18081` | node53 | 旧实验、外部共享 worker |
| 正式单任务隔离训练 | RJob 内置 Docker/worker | `http://127.0.0.1:18081` | RJob `/nvme` | 当前 8-GPU DAPO 正式方案 |
| 反复调试训练脚本 | `sleep infinity` RJob + exec | 按所选 worker 方案 | 不固定 | 短 smoke、定位启动错误 |

选择原则：

- 新的 8-GPU 长训优先使用 RJob 内置 worker，避免共享 node53 的 12-slot 容量、
  SSH 隧道和其他实验互相影响。
- node53 方案仍适合集中维护、代表性 SETA smoke，以及平台不允许 privileged
  RJob 时的回退。
- `rjob exec` 是进入 RJob 的控制手段，不等于 worker 部署方案；exec 盒既可以访问
  外部 node53，也可以在满足权限时启动自己的 Docker server。

整体链路如下：

```text
开发机（共享盘）
├─ SSH/rsync ──HTTP CONNECT──> node53
│                              ├─ dockerd
│                              └─ lightrl-pool53 :18081
│                                      ▲
│                    pu-dev SSH -L 隧道 │ 供旧 RJob 使用
│                                      │
└─ rjob/brainctl ─────────────> GPU RJob
                               ├─ actor / SGLang（GPU）
                               └─ 可选：私有 dockerd + SETA pool（CPU）
```

## 2. node53 的环境约束

node53 地址为 `10.140.84.53`，也曾暴露 `10.140.86.53`。使用前先理解以下限制：

- node53 没有 `/mnt/shared-storage-user`，代码和离线依赖必须显式同步到
  `/home/puyuan/lightrl_worker`。
- node53 shell 无正常公网出口；GitHub、apt、pip 不能按普通开发机方式直连。
- GPU RJob 的 `100.97.x/100.98.x` 网段不能直接访问 `10.140.x`，因此训练盒不能把
  `10.140.84.53:18081` 写入 `WORKER_URLS`。
- `jump.pjlab.org.cn` 的交互 shell 级联只适合首次安装 SSH 公钥。它会污染 rsync/tar
  协议、吞掉远端 argv，并禁止端口转发，不能作为日常数据通道。
- pu-dev IP 由 DHCP 分配。将它作为 node53 隧道入口时，IP 变化会使旧
  `WORKER_URLS` 失效。

## 3. 连接 node53

### 3.1 一次性安装公钥

在能够访问集群 HTTP CONNECT 代理的开发机上生成密钥，并配置直连别名：

```bash
ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519 -q

Host pj-baremetal
    HostName 10.140.84.53
    User puyuan
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    ProxyCommand nc -X connect -x httpproxy-headless.kubebrain.svc.pjlab.local:3128 %h %p
    StrictHostKeyChecking no
```

上面的 `Host` 内容应写入开发机 `~/.ssh/config`。首次仍需经 jump 的交互认证把
`~/.ssh/id_ed25519.pub` 追加到 node53 的 `~/.ssh/authorized_keys`。密码向管理员
获取，不要保存在仓库或脚本中。

### 3.2 日常 SSH 与文件同步

公钥安装后全部使用 `pj-baremetal`，不要再走 jump shell：

```bash
ssh pj-baremetal 'hostname && id'
rsync -az --partial ./dir/ pj-baremetal:/home/puyuan/dir/
```

这里的 `ProxyCommand` 只做 TCP CONNECT，不进入 bastion shell，因此远端 argv、
rsync 和 `ssh -L` 都能正常工作。

### 3.3 连接失败的最小检查

```bash
ssh -vv pj-baremetal 'hostname'
ip -4 addr show eth0                  # 在 pu-dev 核对当前隧道入口 IP
ssh pj-baremetal 'ip -4 addr show'    # 核对 node53 地址
```

如果 rsync 报 `protocol version mismatch`，或 SSH 只显示 MOTD 而远端命令没有输出，
说明仍误用了 jump shell。

## 4. node53 Docker server 配置

本章只讨论 node53 的长期共享 dockerd。RJob 内的私有 dockerd 见第 7 章。

### 4.1 同步代码和离线 Python 依赖

在开发机同步 worker 所需代码：

```bash
cd /mnt/shared-storage-user/puyuan/code/LightRL
rsync -az --partial --exclude='__pycache__' \
  agentic_rl benchmarks/seta_env benchmarks/seta_env_convert \
  examples/validation deploy/workers tools/dev pyproject.toml \
  pj-baremetal:/home/puyuan/lightrl_worker/
```

node53 无法在线安装 Python 包。应先在兼容 Linux/Python 环境下载 wheels，再同步并
离线安装：

```bash
pip download -d wheels53 \
  'fastapi>=0.115' 'uvicorn>=0.30' 'aiohttp>=3.9' 'psutil>=5.9' \
  'terminal-bench>=0.2.18' 'camel-ai>=0.2.90' setuptools wheel 'mcp==1.25.0'
rsync -az wheels53/ pj-baremetal:/home/puyuan/lightrl_worker/wheels53/

ssh pj-baremetal 'cd /home/puyuan/lightrl_worker && \
  python3 -m venv .venv && \
  .venv/bin/pip install --no-index --find-links wheels53 \
  fastapi uvicorn aiohttp psutil terminal-bench camel-ai "mcp==1.25.0"'
```

`camel-ai` 是 terminal runtime 的模块级依赖；`mcp` 必须固定为 `1.25.0`，2.0.0
没有当前代码使用的 `mcp.server.FastMCP`。

### 4.2 dockerd 拉取镜像的代理

node53 的镜像 pull 依赖 dockerd 级代理，配置位置为 `/etc/docker/daemon.json`：

```json
{
  "proxies": {
    "http-proxy": "http://<USER>:<PASSWORD>@proxy.pjlab.org.cn:23128",
    "https-proxy": "http://<USER>:<PASSWORD>@proxy.pjlab.org.cn:23128",
    "no-proxy": "localhost,127.0.0.1,0.0.0.0,::1,10.0.0.0/8,.pjlab.org.cn"
  }
}
```

该文件包含凭据，必须保持 root-only，不能复制到 repo 或训练 artifact。修改后只能在
没有正式 rollout 的维护窗口重启 Docker：

```bash
ssh pj-baremetal 'sudo systemctl restart docker && sudo docker info'
```

不要在 node53 普通 shell 中盲目 `export HTTP_PROXY`；shell 代理和 dockerd 代理是
不同链路，错误环境变量反而可能导致 DNS 失败。

### 4.3 任务容器的无凭据 runtime proxy

SETA 有三条联网链路，必须分别处理：

1. dockerd 拉取基础镜像；
2. Docker build 阶段执行 apt/uv/curl；
3. 已启动任务容器中的 grader 或工具联网。

生产方案使用本地 `tinyproxy`：认证上游只对 root 可见，任务容器只看到无凭据地址
`http://172.17.0.1:3129`。相关文件为：

```text
deploy/workers/lightrl-seta-runtime-proxy.service
deploy/workers/lightrl-pool53-runtime-proxy.conf
deploy/workers/lightrl-pool53-proxy.conf
deploy/workers/configure_seta_runtime_proxy.py
```

安全边界：

- `/etc/seta_build_proxy.env` 仅允许 tinyproxy unit 读取，权限为 `0600 root:root`。
- worker 使用专用
  `DOCKER_CONFIG=/home/puyuan/lightrl_worker/.docker-seta`，不得修改
  `/root/.docker/config.json`。
- worker drop-in 只能注入无凭据的 `172.17.0.1:3129`，严禁加载
  `/etc/seta_build_proxy.env`。
- 诊断只输出 `present/node_local/has_credentials` 等布尔值，不输出代理原文。

部署后核对实际 systemd 配置：

```bash
ssh pj-baremetal \
  "sudo systemctl cat lightrl-pool53 | \
   grep -E 'DOCKER_CONFIG|SKIP_PROXY_ENV|HTTP_PROXY=http://172.17.0.1:3129'"

ssh pj-baremetal \
  'if sudo cmp -s \
     /etc/systemd/system/lightrl-pool53.service.d/proxy.conf \
     /etc/systemd/system/lightrl-pool53.service.d/runtime-proxy.conf; then \
     echo ERROR-duplicated-dropins; exit 1; fi'
```

第一条应同时命中 `DOCKER_CONFIG`、`SKIP_PROXY_ENV` 和无凭据代理；两个 drop-in
必须不同。

## 5. node53 SETA pool 服务

### 5.1 systemd 服务职责

node53 上的 `/etc/systemd/system/lightrl-pool53.service` 负责运行 SETA pool，核心
配置如下：

```ini
[Service]
User=root
WorkingDirectory=/home/puyuan/lightrl_worker
Environment=REPO_ROOT=/home/puyuan/lightrl_worker
Environment=TERMINAL_RL_POOL_NAMESPACE=default
Environment=ENV_SERVER_PORT=18081
Environment=WORKER_MAX_TOTAL_RUNS=12
Environment=WORKER_MAX_TASKS=16
Environment=WORKER_MAX_RUNS_PER_TASK=16
Environment=WORKER_MAX_CONCURRENT_BUILDS=4
Environment=WORKER_MAX_CONCURRENT_RESETS=8
Environment=WORKER_RESET_ADMISSION_TIMEOUT=600
Environment=WORKER_DOCKER_BUILD_QUEUE_TIMEOUT=1200
Environment=WORKER_ORPHAN_DOCKER_SWEEP=0
Environment=SKIP_PROXY_ENV=1
Environment=CONTAINER_PIDS_LIMIT=512
Environment=CONTAINER_MEMORY_LIMIT=16g
Environment=DOCKER_NETWORK_LIFECYCLE_LOCK=/tmp/lightrl-default_docker_network_lifecycle.lock
ExecStart=/usr/bin/bash deploy/workers/run_pool_server_pu_v2.sh
Restart=always
RestartSec=10
```

关键约束：

- legacy SETA Compose 必须使用 `TERMINAL_RL_POOL_NAMESPACE=default`。
- node53 同时运行其他 Docker 服务，所以必须设置
  `WORKER_ORPHAN_DOCKER_SWEEP=0`，避免误删非 LightRL 对象。
- Docker network 生命周期锁必须使用独立、可写路径。
- `WORKER_MAX_TOTAL_RUNS=12` 是宿主总容量，统计范围是
  `active ∪ retiring`，不是只看 active lease。
- 同镜像首次构建应串行去重，build queue timeout 与最长 reset 预算应保持一致。
- `/allocate` 返回 429 通常表示 retiring 网络尚未清完，是正常回压；客户端调用顺序
  必须为 `admission → allocate → reset`。

### 5.2 启停与健康检查

所有重启都应在没有正式 rollout 的维护窗口执行：

```bash
ssh pj-baremetal 'sudo systemctl daemon-reload && \
  sudo systemctl restart lightrl-seta-runtime-proxy lightrl-pool53'

ssh pj-baremetal 'sudo systemctl status \
  lightrl-seta-runtime-proxy lightrl-pool53 --no-pager'

ssh pj-baremetal 'curl --noproxy "*" -fsS \
  http://127.0.0.1:18081/healthz && \
  curl --noproxy "*" -fsS http://127.0.0.1:18081/status'
```

正式 rollout 活跃时不要重启 Docker、pool 或隧道。即使服务立即恢复，已有
keepalive/reset/evaluate 连接也会中断并污染当前训练或 fixed-eval 点位。

### 5.3 代表性 SETA smoke

任何 worker、代理或 Docker 配置变更后都必须做真实 reset/evaluate：

```bash
ssh pj-baremetal 'cd /home/puyuan/lightrl_worker && \
  .venv/bin/python tools/dev/smoke_seta_worker.py \
  --worker-url http://127.0.0.1:18081 \
  --dataset benchmarks/seta_env_convert/eval_fixed48_v2.jsonl \
  --task-name 705 --eval-timeout 1200 \
  --verify-runtime-proxy-url http://172.17.0.1:3129'
```

验收标准：

- 命令退出码为 0；
- 输出含 `runtime_proxy_verified=true`；
- 不出现 `eval_parse_failed`、`eval_timeout` 或 `eval_no_results`；
- close 完成后 `/status` 中 active、retiring、reserved、pending 和 failure 回到 0。

## 6. 让 RJob 使用 node53 外部 worker

GPU RJob 不能直连 `10.140.84.53`。必须由网络稳定的 pu-dev 建立本地端口转发：

```text
GPU RJob -> pu-dev:18081 -> SSH tunnel -> node53:127.0.0.1:18081
```

### 6.1 pu-dev 隧道服务

在 pu-dev 安装 `/etc/systemd/system/lightrl-tunnel53.service`：

```ini
[Service]
ExecStart=/usr/bin/ssh -N -L 0.0.0.0:18081:127.0.0.1:18081 \
  -o ServerAliveInterval=30 \
  -o ExitOnForwardFailure=yes \
  -o BatchMode=yes pj-baremetal
Restart=always
RestartSec=5
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now lightrl-tunnel53
curl --noproxy '*' -fsS http://127.0.0.1:18081/healthz
ip -4 addr show eth0
```

训练端使用 pu-dev 当前地址，而不是 node53 地址：

```bash
export WORKER_URLS=http://<PU_DEV_IP>:18081
```

pu-dev IP 变化时，要同步更新下一次实验的 `WORKER_URLS` 和 worker 快照；已经运行的
正式实验不要在中途改 URL。

### 6.2 从 RJob 内验证完整链路

先从开发机验证隧道，再进入 RJob 验证：

```bash
curl --noproxy '*' -fsS http://<PU_DEV_IP>:18081/healthz

env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  brainctl -n ailab-narmodel --request-timeout=120s \
  exec replica/<REPLICA> -- \
  curl --noproxy '*' -fsS http://<PU_DEV_IP>:18081/healthz
```

如果普通 curl 返回 `502 ... EOF`，先加 `--noproxy '*'`。这通常是本机 curl 误走
HTTP proxy，并不意味着 node53 pool 已失效。

### 6.3 共享 worker 的并发边界

- 单个 DAPO 基线最多使用 node53 的 12 个 slot。
- 双实验共享 node53 时，应满足
  `每侧 client cap × 2 <= WORKER_MAX_TOTAL_RUNS`，当前推荐 `6 × 2 <= 12`。
- 限流必须发生在 `/allocate` 之前，不能先占远端 lease 再等待本地 semaphore。
- fixed-eval 的 semaphore 是每个 dataset 独立创建的。两个 dataset 各 cap=4 时，
  峰值可能是 8 个 lease，而不是 4 个。

## 7. 在 RJob 内部启动 Docker server 与 SETA worker

这一方案让每个训练任务拥有独立 dockerd、Docker image/cache 和 SETA pool，避免
依赖 node53 及 pu-dev 隧道。SETA 容器使用 CPU，RJob 分配的 GPU 仍直接提供给
actor/SGLang；内层 Docker 默认不需要 NVIDIA container runtime。

### 7.1 安全拓扑

当前正式方案已验证为 Pod 隔离网络，不再依赖外层 host network 或 slirp。RJob 通过
一条无凭据 relay 访问 node53 上已有的 credential-hiding proxy：

```text
8-GPU RJob（privileged，hostNetwork=false）
├─ actor / SGLang / Ray
├─ 私有 dockerd
│  ├─ data-root=/nvme/lightrl-dind/<run-id>/docker
│  └─ unix socket=/run/lightrl-dind/<run-id>/docker.sock
├─ SETA pool http://127.0.0.1:18081
└─ HTTP proxy → pu-dev:18129 SSH tunnel
                → node53:3130 localhost relay
                → node53 credential-hiding proxy:3129
```

dockerd 的 bridge/iptables 只存在于 Pod network namespace，不修改 GPU 宿主机网络。
上游认证信息始终只存在于 node53 的 root-only 配置，不能复制进 RJob。pu-dev 上的
`lightrl-proxy-tunnel53-v2.service` 当前是 transient unit，pu-dev 重启后必须先恢复并
验证它；node53 侧 relay 容器名为 `lightrl-rjob-proxy-relay`，只发布 localhost。

### 7.2 RJob 必需资源

提交参数至少包含：

```text
--privileged=true
--host-network=false
--store-host-nvme=true
--custom-resources brainpp.cn/fuse=1
--share-host-shm=true
```

同时设置：

```text
LIGHTRL_DIND_ISOLATED_NETWORK=1
RJOB_DIND_NETWORK_MODE=pod-isolated
RJOB_EGRESS_PROXY=http://100.98.250.77:18129
```

`hostNetwork=true + slirp-netns` 保留为诊断回退，不再是正式默认值。

无论哪种方式，都不要把 `/nvme` 作为唯一 checkpoint 存储；Pod 删除或换机后本地盘
可能丢失。

### 7.3 离线 Docker/Compose bundle

`rft:20260408` 和 `ml-base:22.04-pjlab` 都不自带完整 Docker Engine/Compose。
当前 bundle 位于：

```text
/mnt/shared-storage-gpfs2/trustcyberdata/private/docker-infra/tmp/puyuan/lightrl/
  runtime/docker-rft-20260408-ubuntu22.04/
```

它是带 `Packages`、`ROOT_PACKAGES` 和逐包校验信息的本地 APT 仓，包含 Docker、
Compose、iptables、fuse-overlayfs、iproute2、slirp4netns 和 uidmap。更新 bundle 时在
可联网环境运行：

```bash
python3 tools/reproducibility/download_ubuntu_docker_bundle.py --help
```

RJob 启动时只从该本地仓安装实际缺失依赖，不联网 apt，也不使用
`dpkg -i *.deb` 无条件覆盖系统库。

### 7.4 单独验证 dockerd

在符合第 7.2 节条件的 RJob 内执行：

```bash
export RUN_ID=<RUN_ID>
export HOST_NETWORK=false
export LIGHTRL_DIND_ISOLATED_NETWORK=1
export RJOB_DIND_NETWORK_MODE=pod-isolated
export RJOB_DIND_INSTANCE="$RUN_ID"
export DOCKER_DATA_ROOT="/nvme/lightrl-dind/$RUN_ID/docker"
export DOCKER_OFFLINE_BUNDLE_DIR=/mnt/shared-storage-gpfs2/trustcyberdata/private/docker-infra/tmp/puyuan/lightrl/runtime/docker-rft-20260408-ubuntu22.04

bash deploy/workers/start_rjob_dind_worker.sh doctor
source "/run/lightrl-dind/${RUN_ID:0:32}/worker.env"
docker info
```

`doctor` 会核对 root/privileged、NVMe、Docker/Compose、storage driver、data-root、
独立 socket 和 Docker network create/inspect/remove。它还把 Pod 私有 socket 以
符号链接暴露到 `/var/run/docker.sock`，兼容 CAMEL 0.2.90 的硬编码客户端；若该路径
已指向其他 daemon 会 fail-close。正式长训禁止使用 `vfs`；优先
`overlay2`，只有后端文件系统不支持时才允许 `fuse-overlayfs`。

已通过的参考 smoke：
`seta-dind-socket-smoke-20260812-194203`。task 338 冷构建 77.2 秒、容器启动
0.5 秒、grader 56.1 秒、全链路 137.4 秒；reset/evaluate/close 均为 HTTP 200，且
`runtime_proxy_verified=true`。

### 7.5 启动私有 SETA pool

`serve` 在 doctor 基础上继续启动 localhost pool：

```bash
bash deploy/workers/start_rjob_dind_worker.sh serve
source "/run/lightrl-dind/${RUN_ID:0:32}/worker.env"
curl --noproxy '*' -fsS "$WORKER_URLS/healthz"
curl --noproxy '*' -fsS "$WORKER_URLS/status"
```

正式训练前仍必须执行真实 SETA reset/evaluate，不能只用 `docker info` 代替：

```bash
python3 tools/dev/smoke_seta_worker.py \
  --worker-url "$WORKER_URLS" \
  --dataset benchmarks/seta_env_convert/train_minus_eval12.filtered.jsonl \
  --task-name 338 \
  --verify-runtime-proxy-url \
    http://100.98.250.77:18129
```

### 7.6 提交正式 DAPO 或 DIVE-PO 任务

仓库已有一体化入口：

```bash
RUN_ID=<NEW_RUN_ID> \
RJOB_NAME=<NEW_RJOB_NAME> \
bash deploy/rjob/submit_seta_dapo_8g_dind.sh

# DAPO 首个 rollout 验证通过且 DIVE-PO review 完成后才可提交
RUN_ID=<NEW_DIVE_RUN_ID> \
RJOB_NAME=<NEW_DIVE_RJOB_NAME> \
bash deploy/rjob/submit_seta_divepo_8g_dind.sh

# 只有实时探针确认 5 卡可调度、8 卡不可调度时，才使用容量回退布局：
# 4 卡 actor + 1 卡 TP1 rollout；CPU/内存仍由同节点余量自动计算。
RJOB_GPU_TARGET=5 \
RUN_ID=<NEW_DIVE_5G_RUN_ID> \
RJOB_NAME=<NEW_DIVE_5G_RJOB_NAME> \
bash deploy/rjob/submit_seta_divepo_8g_dind.sh
```

两个提交入口共用同一个执行脚本，只通过 `TRAIN_VARIANT=dapo|divepo` 切换算法，
因此 fixed12、worker、并发和存储协议一致。提交前会先调用
`deploy/rjob/probe_seta_resources.sh`：按 `RJOB_GPU_TARGET` 用极小 CPU/内存做
predict-only，只从同时至少剩余 64 CPU/800000 MiB 的节点中选择目标；实际申请量取
该同一节点实时余量并预留 2 CPU/32 GiB 竞态空间。没有合格节点时直接退出，不提交
一个注定长期排队的任务。`probe_seta_8g_resources.sh` 是强制 8 卡的兼容入口。

正式训练当前只接受两种经过拓扑校验的布局：8 卡=`4 actor + 4 rollout`，5 卡容量
回退=`4 actor + 1 rollout`，二者 actor 都保持 TP4，rollout engine 都是 TP1。5 卡
任务仍保持相同训练 batch、1000 rollout-step、fixed12、seed 和算法超参，因此可以按
rollout-step 比较分数；但 rollout 推理并行度只有四分之一，墙钟吞吐和 GPU 利用率不应
与 8 卡结果作同构比较。人工覆盖只能缩小探针给出的 CPU/内存，不能超过实时安全余量。

特权 RJob 可能让 `nvidia-smi -L` 看见宿主全部 8 卡，即使调度器只分配了 5 卡；不能
据此把物理编号 0–4 当作本任务资源。平台给特权 Pod 主进程设置的是
`NVIDIA_VISIBLE_DEVICES=void`，但 `brainctl exec` 会注入真实分配的 UUID 列表。因此
部分卡任务必须先以 `sleep infinity` 占住资源，再从 exec 会话启动训练；正式入口会
校验该列表必须恰有 `NUM_GPUS` 个唯一 UUID，并显式导出为 `CUDA_VISIBLE_DEVICES` 后
才启动 Ray。直接把训练脚本作为 5 卡 RJob 主命令会 fail-close；完整 8 卡任务拥有
整机全部 GPU，可安全接受主进程的 `void`。验收部分卡任务时还应按 UUID 对照
`nvidia-smi --query-compute-apps`，确认新进程没有落到未分配卡。
SGLang 适配层必须把 UUID 列表解释为本地 CUDA ordinal 空间 `0..N-1`；不能对
`CUDA_VISIBLE_DEVICES` 的每一项强制 `int()`。否则 actor 可创建，但 rollout engine
会在模型加载前退出。仓库已为数值编号和 GPU/MIG UUID 两种形式保留回归测试。
执行脚本会按顺序：

1. 安装/启动私有 dockerd；
2. 启动 localhost SETA pool；
3. 扫描全部 SETA Dockerfile，预取所有基础镜像；Docker Hub 官方镜像经显式 mirror，
   GHCR 等显式 registry 保持原地址；
4. 保存 Docker 与 worker 可复现快照；
5. 对 task 338 执行 reset/evaluate 和代理门禁；
6. 门禁通过后启动所选算法的 1000 rollout-step fixed12 训练；
7. 每 20 秒检查 dockerd、pool PID 和 `/healthz`；worker 由退避 supervisor 拉起，
   dockerd 连续失败时在原 NVMe data-root 上限次重启，仍不可用才终止 trainer。

专用 Pod 的吞吐默认值是 client active=32、worker reserved=40、每 task 最多 8、
reset/close=32。多出的 8 个 reserved slot 仅吸收异步 retiring close，避免交接瞬间
429，不会让 client 超过 32 个活跃环境。build 并发按 Pod CPU 自适应：64 CPU 为 4，
≥128 CPU 为 8；build 去重仍保证同一 task 只构建一次。node53 共享池仍使用其独立的
12-slot 容量，不能照搬此值。

镜像失败分两类：Dockerfile 解析/缺文件等确定性错误进入长期 blacklist；429、
502/503/504、TLS/DNS/连接异常只进入默认 60 秒短冷却并返回
`TASK_IMAGE_BUILD_TRANSIENT`，不得把外部注册表故障缓存成一小时的坏任务。

存储分工：

```text
/nvme/lightrl-dind/<run-id>/docker
  只存可重建的 image/container/build cache

/mnt/shared-storage-gpfs2/trustcyberdata/private/docker-infra/tmp/puyuan/lightrl/
  checkpoints/<run-id>   持久 checkpoint
  wandb/<run-id>         W&B offline run

runs/refactor/<run-id>/
  logs/metrics.jsonl、train.log、配置和可复现快照
```

## 8. 使用 RJob exec 调试

### 8.1 申请常驻调试盒

调试多次启动时，可申请一个 `sleep infinity` RJob。统一使用下面的脚本；它会先预测
同一候选节点的 8 卡、CPU 和内存余量，保留竞态余量后再提交。没有合格候选节点时
直接退出，不创建长期 Pending 的任务：

```bash
bash deploy/rjob/submit_seta_8g_debug_box.sh
```

如需缩小申请，可设置 `RJOB_CPU` / `RJOB_MEMORY_MB`；脚本会拒绝超过预测安全值的覆盖。

通过下面命令取得 replica 名；后续 exec 使用 replica，不是 RJob metadata name：

```bash
rjob get <RJOB_NAME>
```

### 8.2 exec 前移除代理环境

`brainctl` 继承 headless proxy 时可能稳定出现 TLS handshake timeout 或 websocket
bad handshake。正确入口为：

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  brainctl -n ailab-narmodel --request-timeout=120s \
  exec replica/<REPLICA> -- bash -lc '<幂等命令>'
```

只有移除代理后仍发生偶发控制面抖动，才进行重试；被重试的命令必须幂等。

### 8.3 防止重复启动实验

控制面超时不代表远端命令没有执行。使用原子目录作为 launch-once 标记：

```bash
RUN_ID=<RUN_ID>
RUN_DIR=/mnt/shared-storage-user/puyuan/code/LightRL/runs/refactor/$RUN_ID
mkdir -p "$RUN_DIR"
if mkdir "$RUN_DIR/.launch-once" 2>/dev/null; then
  nohup env RUN_ID="$RUN_ID" bash <LAUNCH_SCRIPT> \
    >"$RUN_DIR/launcher.log" 2>&1 </dev/null &
  echo $! >"$RUN_DIR/launcher.pid"
  echo LAUNCHED
else
  echo ALREADY_LAUNCHED
fi
```

不要依赖持续的 exec 连接。进程应使用 `nohup`，日志和 PID 写共享盘。重启前先检查
PID、Ray 状态和 `.launch-once`，不要根据一次 exec 超时猜测远端状态。

### 8.4 从共享盘监控

```bash
tail -f runs/refactor/<RUN_ID>/logs/metrics.jsonl
tail -100 runs/refactor/<RUN_ID>/logs/train.log
grep -oE 'Generate failed[^\n]*' \
  runs/refactor/<RUN_ID>/logs/train.log | sort | uniq -c

rjob get <RJOB_NAME>
rjob events <RJOB_NAME>
```

RJob exec 输出可能截断，长输出先写入共享文件再读取。运行中的 Python/Ray worker 已
加载的模块不会随共享盘编辑自动更新，但新派生进程会读到新文件；正式运行中不要
修改主链路代码。

## 9. 长训容量与吞吐经验

- 旧 8 卡 fixed12 配置每 100 rollout-step 约 49 小时；actor 每步训练约 78–96 秒，
  rollout 等待占 94%–95%。首要瓶颈通常是环境 reset/build、tool loop 和最慢
  trajectory，不是 checkpoint 或反向计算。
- RJob 内置专用 worker 的 4×8 rollout 使用 client 32 active + worker 40 reserved；
  node53 共享双跑仍是每侧 6、总计 12，二者不可混用。
- Qwen3-8B 可依次 smoke `TP4/cap6 → TP4/cap12 → TP1×4/cap12`。每次只改一个
  变量，并用 step wall time、P95、失败率和 `perf/request/*` 选择 profile。
- `sequential-throughput-v1` 使用 4×TP1 rollout engine、cap32，并启用 rollout
  engine health monitor。1000-step 正式任务不要逐步保存 debug `.pt`。
- pool `/status` 的 `lifecycle_latency_sec` 提供 reset admission、reset、exec_tool、
  evaluate、close 的 count、成功率、mean/P50/P95/max。先根据数据确定瓶颈，再改并发。
- 不要预留常驻 warm network；它会直接占用正式 lease 容量。
- 扩 Docker address pool、重启 Docker 或切换 Firecracker/OverlayBD 都属于新实验，
  只能在无活跃 rollout 的窗口执行，并完成 SETA 语义回归。

## 10. 故障速查

### 10.1 node53、隧道与共享 worker

| 现象 | 原因 | 处理 |
|---|---|---|
| rsync `protocol version mismatch` | 误用 jump shell | 改用 `ssh pj-baremetal` |
| SSH 只有 MOTD、远端 argv 无输出 | 误用 jump shell | 改用 `pj-baremetal` TCP 直连 |
| node53 pip/git/DNS 失败 | node53 无普通公网出口 | 开发机下载后 rsync 离线安装 |
| pool crash-loop：缺 `camel` | venv 缺 `camel-ai` | 按 4.1 节离线安装 |
| `cannot import FastMCP` | 装了不兼容的 mcp 2.x | 固定 `mcp==1.25.0` |
| `/reset` 500 且 lock permission denied | 生命周期锁路径被占用 | 设置独立 `DOCKER_NETWORK_LIFECYCLE_LOCK` |
| non-default pool 拒绝 legacy Compose | namespace 不兼容 | SETA 使用 `namespace=default`，关闭 orphan sweep |
| build 内 apt 超时 | 只配置了 dockerd pull proxy | 检查 4.3 节 build/runtime proxy |
| 构建排队导致 reset 500/503 | 首次镜像构建超过短 admission | admission=600s、build queue=1200s |
| `all predefined address pools...` | active+retiring 超过 12 | 限制总 lease，等待 retiring 网络清理 |
| allocate timeout 后持续 429 | 先 allocate 后 admission，或退避不足 | 固定 `admission → allocate → reset` |
| `has_credentials=true` | worker 泄露了认证上游 | 立即停止实验，移除错误 EnvironmentFile |
| 容器 `Resource temporarily unavailable` | PID 上限过低 | `CONTAINER_PIDS_LIMIT=512` |
| GPU RJob 直连 node53 超时 | 两网段无 L3 路由 | 使用第 6 章 pu-dev 隧道 |
| pu-dev URL 返回 502 EOF | curl 误走 HTTP proxy | 加 `--noproxy '*'` 后复核 |

### 10.2 RJob 内置 Docker/worker

| 现象 | 原因 | 处理 |
|---|---|---|
| 镜像内没有 `dockerd`/Compose | RFT 镜像默认不包含 | 设置 `DOCKER_OFFLINE_BUNDLE_DIR` |
| `/nvme` 不存在或不可写 | 未申请 host NVMe | 加 `--store-host-nvme=true` |
| backing fs 不支持 overlay2 | 节点本地盘类型不同 | 申请 fuse；只回退 `fuse-overlayfs`，禁用 vfs |
| hostNetwork 模式被启动器拒绝 | 没有 slirp 子 netns 声明 | 设置 `RJOB_DIND_NETWORK_MODE=slirp-netns` |
| isolated RJob 无法 pull/build | 集群 proxy Service 不可达，或 relay/tunnel 未运行 | 检查 pu-dev `:18129`、node53 relay 与无凭据 Docker Hub 401 探针 |
| CAMEL 报 `/var/run/docker.sock` 不存在 | CAMEL 0.2.90 忽略 `DOCKER_HOST` | 使用启动器创建的私有 socket 兼容链接；禁止连接宿主 daemon |
| `docker info` 成功但 SETA reset 失败 | 代理/build/grader 链路仍未通过 | 执行 task 338 的完整 smoke |
| trainer 中途出现 worker 失败 | dockerd/pool 异常或资源耗尽 | 检查 RJob 日志、dockerd.log、pool_server.log；不要当作普通低 reward |
| 首次 actor all-reduce 报 NCCL `Call to bind failed` | `hostname -I` 误选私有 `docker0` 为 `MASTER_ADDR` | 从默认路由取 Pod 接口/IP，并让 Ray runtime 显式继承 `MASTER_ADDR`、`NCCL_SOCKET_IFNAME`、`GLOO_SOCKET_IFNAME`；不得选择 docker/br/veth |

### 10.3 RJob 与训练控制面

| 现象 | 原因 | 处理 |
|---|---|---|
| `brainctl exec` TLS/websocket 错误 | 继承了 HTTP(S) proxy | exec 前移除大小写代理环境 |
| exec 超时后出现两个实验 | 重试了非幂等启动命令 | 使用 `.launch-once` 原子标记 |
| RJob 启动 20 秒内失败且无训练日志 | 平台/configmap 抖动 | 查 `rjob events`；连续复现时换短 metadata name 并联系平台，不能归因训练代码 |
| Running Pod 被外部删除 | 平台生命周期事件 | 查 event、OOM 与训练尾部；从 tracker 指向的完整 checkpoint 恢复 |
| 8 卡长期 Inqueue | gang 资源暂不可满足 | 保持任务排队或改为经验证的较小 profile，不能假装已启动 |

## 11. 正式实验验收清单

### 使用 node53 外部 worker

- [ ] `ssh pj-baremetal` 和 rsync 正常；
- [ ] node53 本机 `/healthz`、`/status` 正常；
- [ ] runtime proxy smoke 通过且无凭据泄露；
- [ ] pu-dev 隧道使用当前 IP，`--noproxy '*'` 可访问；
- [ ] 从目标 RJob 内能访问 `<PU_DEV_IP>:18081`；
- [ ] 客户端 cap 与 node53 12-slot 总容量一致；
- [ ] worker 源码 hash、URL 和部署快照写入 run artifact。

### 使用 RJob 内置 worker

- [ ] RJob 为 privileged，并挂载 `/nvme` 和 GPFS2 持久目录；
- [ ] `hostNetwork=false`，dockerd 位于 Pod 隔离 network namespace；
- [ ] pu-dev/node53 无凭据 relay 链路健康，RJob 中不含上游认证信息；
- [ ] Docker/Compose、storage driver、data-root 和独立 socket 验证通过；
- [ ] task 338 reset/evaluate 与代理门禁通过；
- [ ] `WORKER_URLS=http://127.0.0.1:18081`；
- [ ] dockerd/pool/trainer 监督生效；
- [ ] checkpoint、W&B offline 和结构化训练日志写持久存储；
- [ ] 使用全新 `RUN_ID`，dry-run 创建过的 ID 不再用于正式训练。

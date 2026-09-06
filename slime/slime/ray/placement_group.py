import logging
import os
import socket

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .actor_group import RayTrainGroup
from .rollout import RolloutManager

logger = logging.getLogger(__name__)


@ray.remote(num_gpus=1)
class InfoActor:
    def get_ip_and_gpu_id(self):
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]


def sort_key(x):
    index, node_identifier, gpu_id = x
    # Sort by node IP number and then by GPU ID
    try:
        # try to parse it as an IP address.
        ip_address = node_identifier
        node_ip_parts = list(map(int, ip_address.split(".")))
    except ValueError:
        # Try to resolve the hostname to an IP address.
        try:
            ip_address = socket.gethostbyname(node_identifier)
            node_ip_parts = list(map(int, ip_address.split(".")))
        except (socket.gaierror, TypeError):
            # Instead, we convert each character of the original identifier string
            # to its ASCII value. This provides a stable and consistent numerical
            # representation that allows for sorting.
            node_ip_parts = [ord(c) for c in node_identifier]

    return (node_ip_parts, gpu_id)


def _create_placement_group(num_gpus):
    """Create a placement group with the specified number of GPUs."""
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    # PACK is the historical default, but a colocated multi-node smoke can
    # intentionally use fewer bundles than the physical cluster (for
    # example, 16 total GPUs across four 4-GPU replicas).  In that shape PACK
    # puts all bundles on the first two 8-GPU nodes, while the rollout
    # launcher still computes four node ranks.  Allow the launcher to request
    # SPREAD so Ray balances bundles across the available replicas and the
    # computed node ranks match the actual placement.  Existing jobs retain
    # PACK unless they opt in via the environment variable.
    strategy = os.environ.get("SLIME_RAY_PLACEMENT_GROUP_STRATEGY", "PACK").upper()
    if strategy not in {"PACK", "SPREAD", "STRICT_PACK", "STRICT_SPREAD"}:
        raise ValueError(
            "SLIME_RAY_PLACEMENT_GROUP_STRATEGY must be one of "
            "PACK, SPREAD, STRICT_PACK, STRICT_SPREAD"
        )
    pg = placement_group(bundles, strategy=strategy)
    num_bundles = len(bundles)

    ray.get(pg.ready())
    if os.environ.get("SLIME_RAY_PLACEMENT_GPU_PROBE", "1") == "0":
        logger.info("Skipping InfoActor GPU probing; using placement-group bundle order as GPU order.")
        pg_reordered_bundle_indices = list(range(num_bundles))
        pg_reordered_gpu_ids = list(range(num_bundles))
        for i in range(num_bundles):
            logger.info(f"  bundle {i:4}, actual_bundle_index: {i:4}, gpu: {i} (assumed)")
        return pg, pg_reordered_bundle_indices, pg_reordered_gpu_ids

    # use info actor to get the GPU id
    info_actors = []
    for i in range(num_bundles):
        info_actors.append(
            InfoActor.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i,
                )
            ).remote()
        )
    gpu_ids = ray.get([actor.get_ip_and_gpu_id.remote() for actor in info_actors])
    for actor in info_actors:
        ray.kill(actor)

    bundle_infos = [(i, gpu_ids[i][0], gpu_ids[i][1]) for i in range(num_bundles)]
    sorted_bundle_infos = sorted(bundle_infos, key=sort_key)
    pg_reordered_bundle_indices = [info[0] for info in sorted_bundle_infos]
    # Map from logical index -> physical GPU ID
    pg_reordered_gpu_ids = [gpu_ids[info[0]][1] for info in sorted_bundle_infos]

    for i in range(num_bundles):
        actual_bundle_index = pg_reordered_bundle_indices[i]
        logger.info(
            f"  bundle {i:4}, actual_bundle_index: {actual_bundle_index:4}, "
            f"node: {gpu_ids[actual_bundle_index][0]}, gpu: {gpu_ids[actual_bundle_index][1]}"
        )

    return pg, pg_reordered_bundle_indices, pg_reordered_gpu_ids


def create_placement_groups(args):
    """Create placement groups for actor and rollout engines."""

    num_gpus = 0
    prm_offset = None
    if args.debug_train_only:
        num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
        rollout_offset = 0
        if args.use_critic:
            num_gpus += args.critic_num_nodes * args.critic_num_gpus_per_node
            critic_offset = args.actor_num_nodes * args.actor_num_gpus_per_node
    elif args.debug_rollout_only:
        num_gpus = args.rollout_num_gpus
        rollout_offset = 0
    elif args.colocate:
        actor_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
        # In the usual colocated topology training and rollout share the same
        # GPU set, so the placement group only needs the training bundles.
        # A constrained smoke test may explicitly request more rollout GPUs
        # than training GPUs (e.g. 3-node TP8/PP3 Megatron + a fourth
        # rollout-only node).  Keep the actor bundles at the front, but create
        # enough total bundles for every rollout engine; rollout actors reserve
        # 0 GPUs in colocate mode and therefore reuse/offload the actor devices
        # while the extra bundles provide the additional physical GPUs.
        rollout_on_extra_bundles = False
        if (
            os.environ.get("SLIME_ALLOW_COLOCATE_ROLLOUT_GPU_MISMATCH", "0") == "1"
            and args.rollout_num_gpus
            and args.rollout_num_gpus != actor_gpus
        ):
            if args.rollout_num_gpus < actor_gpus:
                # A colocated smoke may want one wide TP engine on a subset
                # of the actor devices (e.g. actor TP16/PP2 on 32 GPUs but a
                # single TP16 rollout on 16 GPUs).  Reusing the actor
                # bundles avoids requesting 48 physical GPUs from a
                # 32-GPU RJob; the rollout actors reserve zero GPU and are
                # explicitly offloaded/onloaded around generation.  Keep
                # the old extra-bundle behavior as the default for callers
                # that really provision a rollout-only slice.
                if os.environ.get("SLIME_COLOCATE_REUSE_ACTOR_BUNDLES", "0") == "1":
                    num_gpus = actor_gpus
                    rollout_offset = 0
                else:
                    num_gpus = actor_gpus + args.rollout_num_gpus
                    rollout_offset = actor_gpus
                    rollout_on_extra_bundles = True
            else:
                # When rollout needs more GPUs than training, use the whole
                # bundle set for rollout.  Colocated rollout actors reserve
                # zero GPUs and reuse the actor devices after offload.
                num_gpus = max(actor_gpus, args.rollout_num_gpus)
            logger.info(
                "Colocate GPU mismatch placement: actor_gpus=%s rollout_gpus=%s total_bundles=%s extra_rollout=%s",
                actor_gpus,
                args.rollout_num_gpus,
                num_gpus,
                rollout_on_extra_bundles,
            )
        else:
            num_gpus = actor_gpus
        if not rollout_on_extra_bundles:
            rollout_offset = 0
        if args.use_critic:
            num_gpus += args.critic_num_nodes * args.critic_num_gpus_per_node
            critic_offset = args.actor_num_nodes * args.actor_num_gpus_per_node
    else:
        num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node + args.rollout_num_gpus
        rollout_offset = args.actor_num_nodes * args.actor_num_gpus_per_node
        if args.use_critic:
            num_gpus += args.critic_num_nodes * args.critic_num_gpus_per_node
            critic_offset = args.actor_num_nodes * args.actor_num_gpus_per_node
            rollout_offset += args.critic_num_nodes * args.critic_num_gpus_per_node
        if args.prm_enable and args.prm_num_gpus > 0:
            prm_offset = rollout_offset + args.rollout_num_gpus
            num_gpus += args.prm_num_gpus

    logger.info(f"Creating placement group with {num_gpus} GPUs...")
    pg, actor_pg_reordered_bundle_indices, actor_pg_reordered_gpu_ids = _create_placement_group(num_gpus)

    rollout_pg_reordered_bundle_indices = actor_pg_reordered_bundle_indices[rollout_offset:]
    rollout_pg_reordered_gpu_ids = actor_pg_reordered_gpu_ids[rollout_offset:]
    # When a smaller rollout engine reuses a larger actor allocation, the
    # actor list contains (say) 8 bundles per node while the rollout engine
    # needs only 4.  Passing the unsliced list would assign rollout node
    # ranks 0/1 to the same physical node (the first 8-bundle chunk), which
    # makes SGLang's multi-node NCCL rendezvous fail.  Preserve one
    # rollout-sized prefix from each actor node so rank i*local_gpus maps to
    # a distinct node.  This is enabled only for the explicit reuse mode.
    if (
        args.colocate
        and os.environ.get("SLIME_COLOCATE_REUSE_ACTOR_BUNDLES", "0") == "1"
        and args.rollout_num_gpus
        and args.rollout_num_gpus < args.actor_num_nodes * args.actor_num_gpus_per_node
    ):
        actor_local = args.actor_num_gpus_per_node
        rollout_local = min(args.num_gpus_per_node, actor_local)
        rollout_nodes = args.rollout_num_gpus // rollout_local
        selected_bundle_indices = []
        selected_gpu_ids = []
        for node_index in range(rollout_nodes):
            start = node_index * actor_local
            end = start + rollout_local
            selected_bundle_indices.extend(actor_pg_reordered_bundle_indices[start:end])
            selected_gpu_ids.extend(actor_pg_reordered_gpu_ids[start:end])
        rollout_pg_reordered_bundle_indices = selected_bundle_indices
        rollout_pg_reordered_gpu_ids = selected_gpu_ids
        logger.info(
            "Colocate rollout bundle reuse: actor_local=%s rollout_local=%s rollout_nodes=%s bundles=%s",
            actor_local,
            rollout_local,
            rollout_nodes,
            len(selected_bundle_indices),
        )
    if prm_offset is not None:
        prm_pg_reordered_bundle_indices = actor_pg_reordered_bundle_indices[prm_offset : prm_offset + args.prm_num_gpus]
        prm_pg_reordered_gpu_ids = actor_pg_reordered_gpu_ids[prm_offset : prm_offset + args.prm_num_gpus]
        rollout_pg_reordered_bundle_indices = actor_pg_reordered_bundle_indices[rollout_offset:prm_offset]
        rollout_pg_reordered_gpu_ids = actor_pg_reordered_gpu_ids[rollout_offset:prm_offset]
    if args.use_critic:
        critic_pg_reordered_bundle_indices = actor_pg_reordered_bundle_indices[critic_offset:]
        critic_pg_reordered_gpu_ids = actor_pg_reordered_gpu_ids[critic_offset:]

    return {
        "actor": (pg, actor_pg_reordered_bundle_indices, actor_pg_reordered_gpu_ids),
        "critic": (pg, critic_pg_reordered_bundle_indices, critic_pg_reordered_gpu_ids) if args.use_critic else None,
        "rollout": (pg, rollout_pg_reordered_bundle_indices, rollout_pg_reordered_gpu_ids),
        "prm": (pg, prm_pg_reordered_bundle_indices, prm_pg_reordered_gpu_ids) if prm_offset is not None else None,
    }


def allocate_train_group(args, num_nodes, num_gpus_per_node, pg):
    # Megatron constructs a full model shard in every actor.  With GLM-5.1
    # each shard is roughly 34--37 GiB, so the historical 0.4-GPU fractional
    # reservation lets several actors land on one H200 and causes an OOM
    # during model construction.  Reserve one whole placement-group bundle
    # per Megatron rank; retain fractional placement for FSDP backends whose
    # actors are explicitly designed to share a device.
    num_gpus_per_actor = 1.0 if args.train_backend == "megatron" else 0.4
    return RayTrainGroup(
        args=args,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        pg=pg,
        num_gpus_per_actor=num_gpus_per_actor,
    )


def create_training_models(args, pgs, rollout_manager):
    actor_model = allocate_train_group(
        args=args,
        num_nodes=args.actor_num_nodes,
        num_gpus_per_node=args.actor_num_gpus_per_node,
        pg=pgs["actor"],
    )
    if args.use_critic:
        critic_model = allocate_train_group(
            args=args,
            num_nodes=args.critic_num_nodes,
            num_gpus_per_node=args.critic_num_gpus_per_node,
            pg=pgs["critic"],
        )
        critic_init_handle = critic_model.async_init(args, role="critic", with_ref=False)
    else:
        critic_model = None

    start_rollout_ids = ray.get(
        actor_model.async_init(args, role="actor", with_ref=args.kl_coef != 0 or args.use_kl_loss)
    )

    assert len(set(start_rollout_ids)) == 1
    if args.start_rollout_id is None:
        args.start_rollout_id = start_rollout_ids[0]

    if args.use_critic:
        ray.get(critic_init_handle)
        actor_model.connect(critic_model)

    # The synchronous smoke path may intentionally initialize Megatron before
    # constructing SGLang/rollout (to keep the actor's GPU footprint isolated).
    # In that phase there is no manager to bind yet; train.py attaches it after
    # create_rollout_manager returns.
    if rollout_manager is not None:
        actor_model.set_rollout_manager(rollout_manager)
    if args.rollout_global_dataset and rollout_manager is not None:
        ray.get(rollout_manager.load.remote(args.start_rollout_id - 1))

    return actor_model, critic_model


def create_rollout_manager(args, pg, prm_pg=None):
    # RolloutManager is a plain Ray actor (rather than a train actor), so in
    # direct ``slime/train.py`` mode it does not inherit the caller shell's
    # compatibility variables.  Preserve the SGLang/GLM smoke switches here;
    # SGLangEngine then copies them into its own runtime environment before
    # spawning the server process.
    rollout_env_vars = {
        key: os.environ.get(key, default)
                for key, default in {
                    "SGLANG_GLM51_SAFE_MLA_CONCAT": "0",
                    "SGLANG_DISABLE_NSA_DP_ATTENTION": "0",
                    "SGLANG_NSA_FORCE_MLA": "0",
                    "SGLANG_DISABLE_FLASHINFER_RMSNORM": "0",
                    "SGLANG_RMSNORM_CPU_FALLBACK": "0",
                    "SGLANG_RMSNORM_BYPASS": "0",
                    "SGLANG_DISABLE_PYNCCL": "0",
            "SGLANG_WEIGHT_LOADER_DISABLE_MMAP": "0",
            "SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT": "",
            "SGLANG_USE_MESSAGE_QUEUE_BROADCASTER": "1",
            "SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS": "0",
            "LIGHTRL_RETRY_CUDA_ALLOC": "0",
        }.items()
    }
    rollout_manager = RolloutManager.options(
        num_cpus=1,
        num_gpus=0,
        runtime_env={"env_vars": rollout_env_vars},
    ).remote(args, pg, prm_pg)

    # calculate num_rollout from num_epoch
    num_rollout_per_epoch = None
    if args.num_rollout is None:
        num_rollout_per_epoch = ray.get(rollout_manager.get_num_rollout_per_epoch.remote())
        args.num_rollout = num_rollout_per_epoch * args.num_epoch
        assert args.num_rollout > 0

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="snapshot"))
        ray.get(rollout_manager.check_weights.remote(action="reset_tensors"))

    if args.offload_rollout:
        ray.get(rollout_manager.offload.remote())

    return rollout_manager, num_rollout_per_epoch

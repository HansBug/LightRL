import logging
import os
import re
from pathlib import Path

# TODO: may need to copy those 2 functions and do refactoring.
from megatron.training.checkpointing import load_checkpoint as _load_checkpoint_megatron
from megatron.training.checkpointing import save_checkpoint
from megatron.training.global_vars import get_args

from slime.utils import megatron_bridge_utils

from .bridge_compat import bridge_from_pretrained
from .lora import without_megatron_lora_parameters

try:
    # Here we patch out the `validate_non_overlapping_shards_metadata` in both functions
    # because it is really slow for large models with many shards.
    # TODO: find a less hacky way to do this.
    import torch.distributed as dist
    import torch.distributed._shard.sharding_spec as shard_spec
    from torch.distributed._shard.sharded_tensor import ShardedTensor
    from torch.distributed._shard.sharded_tensor.metadata import ShardedTensorMetadata
    from torch.distributed._shard.sharded_tensor.shard import Shard
    from torch.distributed._shard.sharded_tensor.utils import _parse_and_validate_remote_device
    from torch.distributed._shard.sharding_spec.api import EnumerableShardingSpec

    def __post_init__(self):
        pass

    EnumerableShardingSpec.__post_init__ = __post_init__

    @classmethod
    def _init_from_local_shards_and_global_metadata(  # type: ignore[override]
        cls,
        local_shards: list[Shard],
        sharded_tensor_metadata: ShardedTensorMetadata,
        process_group=None,
        init_rrefs=False,
        sharding_spec=None,
    ) -> ShardedTensor:
        """
        Initialize a ShardedTensor with local shards and a global
        ShardedTensorMetadata built on each rank.

        Warning: This API is experimental and subject to change. It does
                 not do cross rank validations, and fully rely on the user
                 for the correctness of sharded_tensor_metadata on each rank
        """
        process_group = cls._normalize_pg(process_group)
        current_rank = dist.get_rank()  # intentional to get global rank

        shards_metadata = sharded_tensor_metadata.shards_metadata

        local_shard_metadatas = []

        # collect local shard metadatas from the global sharded_tensor_metadata
        for shard_metadata in shards_metadata:  # type: ignore[attr-defined]
            rank, local_device = _parse_and_validate_remote_device(process_group, shard_metadata.placement)

            if current_rank == rank:
                local_shard_metadatas.append(shard_metadata)

        shards_metadata = sharded_tensor_metadata.shards_metadata
        tensor_properties = sharded_tensor_metadata.tensor_properties

        if sharding_spec is None:
            spec = shard_spec._infer_sharding_spec_from_shards_metadata(shards_metadata)
        else:
            spec = sharding_spec

        sharded_tensor = ShardedTensor.__new__(
            ShardedTensor,
            spec,
            sharded_tensor_metadata.size,
            dtype=tensor_properties.dtype,
            layout=tensor_properties.layout,
            pin_memory=tensor_properties.pin_memory,
            requires_grad=tensor_properties.requires_grad,
        )

        # done validation, add local_shards
        sharded_tensor._local_shards = local_shards
        sharded_tensor._prepare_init(process_group=process_group, init_rrefs=init_rrefs)

        # run post initialization, i.e. map registration, rpc initialization
        sharded_tensor._post_init()
        return sharded_tensor

    ShardedTensor._init_from_local_shards_and_global_metadata = _init_from_local_shards_and_global_metadata

except ImportError:
    pass

logger = logging.getLogger(__name__)

__all__ = ["save_checkpoint"]


def _bridge_cpu_safetensor_context(enabled: bool):
    """Return a context manager that forces mbridge safetensors to CPU.

    mbridge's FP8 SafeTensorIO opens files with ``device="cuda"``.  During
    GLM-5.1 initialization this creates a second device-side copy of a shard
    tensor while the Megatron model is already resident, leaving only a few
    MiB of headroom.  CPU staging keeps the temporary dequantized tensor off
    CUDA; the normal mbridge scatter subsequently copies only the local split
    to the parameter device.
    """
    from contextlib import nullcontext

    if not enabled:
        return nullcontext()

    class _CpuSafeOpen:
        def __enter__(self):
            self._patched = []
            module_names = (
                "mbridge.models.ext.deepseek_v3.dequant_fp8_safetensor_io",
                "mbridge.core.safetensor_io",
            )
            for module_name in module_names:
                try:
                    module = __import__(module_name, fromlist=["safe_open"])
                    original = getattr(module, "safe_open", None)
                except Exception:
                    continue
                if original is None:
                    continue

                def cpu_safe_open(*args, _original=original, **kwargs):
                    kwargs["device"] = "cpu"
                    return _original(*args, **kwargs)

                setattr(module, "safe_open", cpu_safe_open)
                self._patched.append((module, original))
            return self

        def __exit__(self, exc_type, exc, tb):
            for module, original in self._patched:
                module.safe_open = original
            return False

    return _CpuSafeOpen()


def _bridge_cached_state_dict_context(models, enabled: bool):
    """Cache each model's state dict while mbridge imports HF weights.

    mbridge's loader looks up ``model.state_dict()[name]`` inside its loop
    over every local parameter.  Transformer Engine modules synchronize CUDA
    from ``get_extra_state`` while building a state dict, so repeating that
    full traversal thousands of times can make a checkpoint load appear
    hung.  The returned mapping contains the same parameter references (not
    clones), therefore subsequent ``copy_`` calls still update the model.
    """

    from contextlib import nullcontext

    if not enabled:
        return nullcontext()

    class _CachedStateDicts:
        def __enter__(self):
            self._patched = []
            for model in models:
                original = model.state_dict
                cached = original()

                def cached_state_dict(*args, _cached=cached, **kwargs):
                    # mbridge calls this without arguments.  Returning the
                    # same references is intentional; no tensor is copied.
                    return _cached

                marker = object()
                previous = model.__dict__.get("state_dict", marker)
                model.__dict__["state_dict"] = cached_state_dict
                self._patched.append((model, previous, marker))
            logger.info("Cached state_dict for %d mbridge model chunk(s)", len(self._patched))
            return self

        def __exit__(self, exc_type, exc, tb):
            for model, previous, marker in self._patched:
                if previous is marker:
                    model.__dict__.pop("state_dict", None)
                else:
                    model.__dict__["state_dict"] = previous
            return False

    return _CachedStateDicts()


def load_checkpoint(ddp_model, optimizer, opt_param_scheduler, checkpointing_context, skip_load_to_model_and_opt):
    # ref: how megatron `load_checkpoint` gets directory
    args = get_args()
    load_path = args.load

    assert Path(load_path).exists() and _is_dir_nonempty(
        load_path
    ), f"{args.load=} does not exist or is an empty directory. Did you specify the wrong folder?"

    if _is_megatron_checkpoint(load_path):
        return _load_checkpoint_megatron(
            ddp_model=ddp_model,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=skip_load_to_model_and_opt,
        )
    else:
        return _load_checkpoint_hf(
            ddp_model=ddp_model,
            optimizer=optimizer,
            args=args,
            load_path=load_path,
        )


def _is_megatron_checkpoint(path: str | Path) -> bool:
    return (Path(path) / "latest_checkpointed_iteration.txt").is_file() or bool(
        re.fullmatch(r"iter_\d{7}", Path(path).name)
    )


def _load_checkpoint_hf(ddp_model, optimizer, args, load_path: str):
    assert args.megatron_to_hf_mode == "bridge", "Only bridge mode is supported for loading HF checkpoint"
    import slime_plugins.megatron_bridge  # noqa: F401

    logger.info(f"Load checkpoint from HuggingFace model into Megatron (path={load_path})")

    with megatron_bridge_utils.patch_megatron_model(ddp_model), without_megatron_lora_parameters(ddp_model):
        bridge = bridge_from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        if hasattr(bridge, "load_hf_weights"):
            bridge.load_hf_weights(ddp_model)
        else:
            # mbridge's default loader keeps every tensor from a shard file in
            # a GPU-side dictionary before scattering it.  For very large FP8
            # checkpoints (notably GLM-5.1), that creates a transient allocation
            # on top of an already full model and can fail by a few MiB.  The
            # memory-efficient path reads one HF tensor at a time and is
            # opt-in so existing jobs keep their historical throughput unless
            # a constrained run requests the safer behavior.
            memory_efficient = os.environ.get(
                "LIGHTRL_BRIDGE_MEMORY_EFFICIENT", "0"
            ).strip().lower() in {"1", "true", "yes", "on"}
            cpu_staging = os.environ.get(
                "LIGHTRL_BRIDGE_CPU_STAGING", "0"
            ).strip().lower() in {"1", "true", "yes", "on"}
            debug_weights = os.environ.get(
                "LIGHTRL_DEBUG_BRIDGE_WEIGHTS", "0"
            ).strip().lower() in {"1", "true", "yes", "on"}
            if debug_weights and hasattr(bridge, "_weight_split_across_tp"):
                original_split = bridge._weight_split_across_tp

                def debug_split(mcore_name, mcore_weight, param, tp_split_size):
                    split_weights = original_split(
                        mcore_name, mcore_weight, param, tp_split_size
                    )
                    param_shape = tuple(param.shape)
                    split_shapes = [tuple(weight.shape) for weight in split_weights]
                    if any(shape != param_shape for shape in split_shapes):
                        print(
                            "[GLM bridge split mismatch] "
                            f"name={mcore_name} param={param_shape} "
                            f"mcore={tuple(mcore_weight.shape)} splits={split_shapes}",
                            flush=True,
                        )
                    return split_weights

                bridge._weight_split_across_tp = debug_split
            with _bridge_cpu_safetensor_context(cpu_staging):
                cache_state_dict = os.environ.get(
                    "LIGHTRL_BRIDGE_CACHE_STATE_DICT", "1"
                ).strip().lower() in {"1", "true", "yes", "on"}
                with _bridge_cached_state_dict_context(ddp_model, cache_state_dict):
                    bridge.load_weights(
                        ddp_model, args.hf_checkpoint, memory_efficient=memory_efficient
                    )

    # Copied from Megatron-core :: load_checkpoint (with simplifications)
    if (args.fp16 or args.bf16) and optimizer is not None:
        assert not args.load_main_params_from_ckpt
        optimizer.reload_model_params()

    # We can see `successfully loaded checkpoint from ... [ t 1/2, p 1/1 ] at iteration 0`
    # when loading Megatron, thus it is 0
    iteration = 0
    num_floating_point_operations_so_far = 0
    return iteration, num_floating_point_operations_so_far


def _is_dir_nonempty(path):
    with os.scandir(path) as it:
        return any(it)

"""Small process-wide compatibility hooks for the optional SGLang runtime.

Python imports ``sitecustomize`` in every interpreter whose ``PYTHONPATH``
contains this repository.  Keep the hook completely inert by default; the
GLM smoke sets the opt-in environment variable because SGLang's scheduler
children are spawned outside slime's wrapper and otherwise reset the
module-level model-loading barrier timeout to 300 seconds.
"""

import os


# A full GLM-5.1 TP16 shard can leave a small amount of cached-but-unusable
# CUDA memory while Transformer Engine constructs the final expert layer.  An
# opt-in retry after ``empty_cache`` recovers that block without changing model
# precision or architecture.  Keep this narrowly scoped: retrying every CUDA
# allocation globally would hide genuine OOMs in normal training.
if os.environ.get("LIGHTRL_RETRY_CUDA_ALLOC") == "1":
    try:
        import torch as _torch

        _torch_empty = _torch.empty

        def _empty_with_cache_retry(*args, **kwargs):
            try:
                return _torch_empty(*args, **kwargs)
            except _torch.OutOfMemoryError:
                _torch.cuda.empty_cache()
                return _torch_empty(*args, **kwargs)

        _torch.empty = _empty_with_cache_retry
    except (ImportError, AttributeError, TypeError):
        # PyTorch is optional for lightweight CLI processes.
        pass


_timeout = os.environ.get("SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT")
if _timeout:
    try:
        from sglang.srt.model_executor import model_runner

        model_runner.UNBALANCED_MODEL_LOADING_TIMEOUT_S = int(_timeout)
    except (ImportError, TypeError, ValueError):
        # SGLang is optional for most LightRL commands.
        pass


# GLM-5.1 has a 192-wide non-rotary head.  The SGLang Triton helper used by
# the DeepSeek-compatible path currently requires every ``tl.arange`` range
# to be a power of two, so its first real request fails during compilation.
# Keep the workaround opt-in: the torch copies below preserve the exact
# layout/dtype conversion while avoiding that kernel only for the smoke test.
if os.environ.get("SGLANG_GLM51_SAFE_MLA_CONCAT") == "1":
    try:
        import torch
        from sglang.srt.layers.attention import utils as _attention_utils

        def _safe_concat_and_cast_mha_k(k, k_nope, k_rope):
            nope_dim = k_nope.shape[-1]
            k[..., :nope_dim].copy_(k_nope.to(dtype=k.dtype))
            # The Triton kernel broadcasts the single rope head to all local
            # heads; express the same operation without allocating a concat.
            k[..., nope_dim:].copy_(
                k_rope.to(dtype=k.dtype).expand(-1, k.shape[1], -1)
            )

        _attention_utils.concat_and_cast_mha_k_triton = _safe_concat_and_cast_mha_k

        # deepseek_v2 imports the helper by value, so update that binding too.
        from sglang.srt.models import deepseek_v2 as _deepseek_v2

        _deepseek_v2.concat_and_cast_mha_k_triton = _safe_concat_and_cast_mha_k
    except (ImportError, RuntimeError, TypeError, ValueError):
        # Leave non-SGLang commands and incompatible SGLang builds untouched.
        pass


# SGLang's NSA handler deliberately chooses ``MHA_ONE_SHOT`` for short
# prefixes (the normal fast path when the prompt is below ``index_topk``).
# On the private multi-node GLM smoke that path can stall after the prefill
# log without producing a detokenizer heartbeat.  Keep an opt-in escape hatch
# that forces the handler to use the real sparse MLA implementation; this is
# intentionally disabled by default so production inference keeps SGLang's
# normal heuristic.
if os.environ.get("SGLANG_NSA_FORCE_MLA") == "1":
    try:
        from sglang.srt.models import deepseek_v2 as _deepseek_v2

        def _force_nsa_mla(_attn, _forward_batch):
            return _deepseek_v2.AttnForwardMethod.MLA

        _deepseek_v2.AttentionBackendRegistry._handlers["nsa"] = _force_nsa_mla
    except (ImportError, RuntimeError, TypeError, AttributeError):
        # Keep lightweight/non-SGLang processes and incompatible builds inert.
        pass


# The private H200/DinD image can block in the FlashInfer RMSNorm custom op's
# ``cudaFuncSetAttribute`` on the first request (all TP ranks remain at 100%
# GPU while the detokenizer heartbeat never advances).  An opt-in native
# implementation is considerably slower but uses ordinary PyTorch kernels and
# is useful for proving the end-to-end training path.  Keep the production
# fast path untouched.
if os.environ.get("SGLANG_DISABLE_FLASHINFER_RMSNORM") == "1":
    try:
        from sglang.srt.layers import layernorm as _layernorm

        _layernorm.RMSNorm.forward_cuda = _layernorm.RMSNorm.forward_native
    except (ImportError, RuntimeError, TypeError, AttributeError):
        # Leave lightweight/non-SGLang processes and incompatible builds inert.
        pass


# A few H200/DinD combinations also stall in PyTorch's first CUDA reduction
# when the native implementation is selected.  For a tiny end-to-end smoke,
# make the normalization reduction explicitly on CPU and copy the result back
# to the original device.  This is intentionally opt-in and should never be
# used for production serving; it trades latency for a deterministic path that
# lets us validate the surrounding rollout/training plumbing.
if os.environ.get("SGLANG_RMSNORM_CPU_FALLBACK") == "1":
    try:
        from sglang.srt.layers import layernorm as _layernorm

        _rmsnorm_forward_native = _layernorm.RMSNorm.forward_native

        def _rmsnorm_forward_cpu_fallback(self, x, residual=None):
            device = x.device
            x_cpu = x.to(device="cpu")
            residual_cpu = None if residual is None else residual.to(device="cpu")
            result = _rmsnorm_forward_native(self, x_cpu, residual_cpu)
            if isinstance(result, tuple):
                return tuple(value.to(device=device) for value in result)
            return result.to(device=device)

        _layernorm.RMSNorm.forward_cuda = _rmsnorm_forward_cpu_fallback
    except (ImportError, RuntimeError, TypeError, AttributeError):
        # Keep lightweight/non-SGLang processes and incompatible builds inert.
        pass


# Diagnostic escape hatch for an H200 kernel/runtime issue: skip the RMSNorm
# arithmetic entirely while checking whether later attention/MoE kernels can
# make progress.  This intentionally produces non-normalized activations and
# is suitable only for a reachability smoke, never for model quality tests.
if os.environ.get("SGLANG_RMSNORM_BYPASS") == "1":
    try:
        from sglang.srt.layers import layernorm as _layernorm

        def _rmsnorm_bypass(self, x, residual=None):
            if residual is None:
                return x
            return x, residual

        _layernorm.RMSNorm.forward_cuda = _rmsnorm_bypass
    except (ImportError, RuntimeError, TypeError, AttributeError):
        # Keep lightweight/non-SGLang processes and incompatible builds inert.
        pass


# Some private RJob network topologies can initialize Megatron's torch NCCL
# process group successfully while SGLang's auxiliary PyNccl communicator
# fails during multi-node TP setup.  GroupCoordinator already supports the
# standard torch.distributed path; make the workaround opt-in so production
# SGLang keeps its faster PyNccl path by default.
if os.environ.get("SGLANG_DISABLE_PYNCCL") == "1":
    try:
        from sglang.srt.distributed import parallel_state as _parallel_state

        _group_coordinator_init = _parallel_state.GroupCoordinator.__init__

        def _group_coordinator_init_without_pynccl(self, *args, **kwargs):
            kwargs["use_pynccl"] = False
            return _group_coordinator_init(self, *args, **kwargs)

        _parallel_state.GroupCoordinator.__init__ = _group_coordinator_init_without_pynccl
    except (ImportError, RuntimeError, TypeError, AttributeError):
        # Keep the hook inert for lightweight processes or older SGLang builds.
        pass

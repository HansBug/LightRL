import os
import re

import torch
from mbridge.core import register_model
from mbridge.models import DeepseekV3Bridge


@register_model("glm_moe_dsa")
class GLMMoEDSABridge(DeepseekV3Bridge):
    """Megatron-Bridge adapter for GLM-5.1 HF checkpoints."""

    _DSA_INDEXER_MAPPING = {
        "core_attention.indexer.linear_wq_b.weight": ["model.layers.{layer_number}.self_attn.indexer.wq_b.weight"],
        "core_attention.indexer.linear_wk.weight": ["model.layers.{layer_number}.self_attn.indexer.wk.weight"],
        "core_attention.indexer.linear_weights_proj.weight": [
            "model.layers.{layer_number}.self_attn.indexer.weights_proj.weight"
        ],
        "core_attention.indexer.k_norm.weight": ["model.layers.{layer_number}.self_attn.indexer.k_norm.weight"],
        "core_attention.indexer.k_norm.bias": ["model.layers.{layer_number}.self_attn.indexer.k_norm.bias"],
    }
    # The installed Megatron-Core exposes MLA's norms as standalone
    # ``q_layernorm``/``kv_layernorm`` parameters, while mbridge's DeepSeek
    # mapping expects the older fused ``linear_*_up_proj.layer_norm_weight``
    # spelling.  GLM-5.1 stores the same tensors under the HF q_a/kv_a norm
    # names, so handle both mcore variants explicitly.
    _GLM_MLA_NORM_MAPPING = {
        "q_layernorm.weight": ["model.layers.{layer_number}.self_attn.q_a_layernorm.weight"],
        "q_layernorm.bias": ["model.layers.{layer_number}.self_attn.q_a_layernorm.bias"],
        "kv_layernorm.weight": ["model.layers.{layer_number}.self_attn.kv_a_layernorm.weight"],
        "kv_layernorm.bias": ["model.layers.{layer_number}.self_attn.kv_a_layernorm.bias"],
    }

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        assert "_extra_state" not in mcore_weights_name, "extra_state should not be loaded"
        dsa_match = re.match(r"decoder\.layers\.(\d+)\.self_attention\.(.+)", mcore_weights_name)
        if dsa_match:
            layer_number, rest = dsa_match.groups()
            if rest in self._DSA_INDEXER_MAPPING:
                return [x.format(layer_number=layer_number) for x in self._DSA_INDEXER_MAPPING[rest]]
            if rest in self._GLM_MLA_NORM_MAPPING:
                return [x.format(layer_number=layer_number) for x in self._GLM_MLA_NORM_MAPPING[rest]]
        return super()._weight_name_mapping_mcore_to_hf(mcore_weights_name)

    def _weight_to_mcore_format(self, mcore_weights_name: str, hf_weights):
        """Adapt GLM's rotary-Q layout to Megatron MLA's fused Q-up weight.

        GLM-5.1 stores ``q_b_proj`` using its non-rotary query dimension.  The
        normal path now configures Megatron with the matching
        ``qk_nope_head_dim``; the padding fallback remains for older bridge
        configurations that still request an extra rotary-Q slice.
        """
        weight = super()._weight_to_mcore_format(mcore_weights_name, hf_weights)
        if not mcore_weights_name.endswith("self_attention.linear_q_up_proj.weight"):
            return weight

        # GLM exposes ``qk_head_dim`` as the total query-key dimension
        # (non-rotary + rotary), whereas Megatron MLA's ``qk_head_dim``
        # argument denotes only the non-rotary portion.  Prefer the explicit
        # HF field when present; otherwise retain the DeepSeek-compatible
        # fallback for older checkpoints.
        qk_head_dim = int(
            getattr(
                self.hf_config,
                "qk_nope_head_dim",
                getattr(self.hf_config, "qk_head_dim", 0),
            )
            or 0
        )
        qk_rope_head_dim = int(getattr(self.hf_config, "qk_rope_head_dim", 0) or 0)
        num_heads = int(getattr(self.hf_config, "num_attention_heads", 0) or 0)
        expected_rows = num_heads * (qk_head_dim + qk_rope_head_dim)
        if expected_rows <= weight.shape[0] or qk_rope_head_dim <= 0:
            return weight
        missing_rows = expected_rows - weight.shape[0]
        padding = weight.new_zeros((missing_rows, weight.shape[1]))
        return torch.cat((weight, padding), dim=0)

    def _build_config(self):
        # The bridge normally constructs GLM with GPU-side parameter
        # initialization and TEGroupedMLP.  Both choices are ideal for a
        # production run, but leave only a few MiB of headroom on a 140-GiB
        # H200 at TP16.  Keep the defaults unchanged and expose opt-in smoke
        # switches so a constrained debug RJob can construct the model from
        # CPU and/or use Megatron's legacy grouped-MoE implementation.
        use_cpu_initialization = os.environ.get("GLM_CPU_INITIALIZATION", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        use_legacy_grouped_gemm = os.environ.get("GLM_LEGACY_GROUPED_GEMM", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        first_k_dense_replace = int(getattr(self.hf_config, "first_k_dense_replace", 0) or 0)
        num_hidden_layers = int(getattr(self.hf_config, "num_hidden_layers", 0) or 0)
        moe_layer_freq = [0] * min(first_k_dense_replace, num_hidden_layers) + [1] * max(
            0, num_hidden_layers - first_k_dense_replace
        )
        qk_head_dim = int(
            getattr(
                self.hf_config,
                "qk_nope_head_dim",
                getattr(self.hf_config, "qk_head_dim", 0),
            )
            or 0
        )
        return self._build_base_config(
            use_cpu_initialization=use_cpu_initialization,
            # MLA + DSA
            multi_latent_attention=True,
            experimental_attention_variant="dsa",
            q_lora_rank=self.hf_config.q_lora_rank,
            kv_lora_rank=self.hf_config.kv_lora_rank,
            qk_head_dim=qk_head_dim,
            qk_pos_emb_head_dim=self.hf_config.qk_rope_head_dim,
            v_head_dim=self.hf_config.v_head_dim,
            dsa_indexer_n_heads=self.hf_config.index_n_heads,
            dsa_indexer_head_dim=self.hf_config.index_head_dim,
            dsa_indexer_topk=self.hf_config.index_topk,
            # MoE
            moe_ffn_hidden_size=self.hf_config.moe_intermediate_size,
            moe_shared_expert_intermediate_size=self.hf_config.n_shared_experts
            * self.hf_config.moe_intermediate_size,
            moe_router_bias_update_rate=0.0,
            moe_router_topk=self.hf_config.num_experts_per_tok,
            num_moe_experts=self.hf_config.n_routed_experts,
            moe_grouped_gemm=True,
            moe_use_legacy_grouped_gemm=use_legacy_grouped_gemm,
            moe_router_score_function=self.hf_config.scoring_func,
            moe_router_enable_expert_bias=True,
            moe_router_pre_softmax=True,
            moe_router_topk_scaling_factor=self.hf_config.routed_scaling_factor,
            moe_router_load_balancing_type="none",
            # GLM-5.1 keeps the first three decoder blocks dense.  The
            # generic DeepSeek bridge defaults to MoE at every layer, which
            # both creates the wrong parameter layout and asks the loader for
            # a nonexistent router weight in dense blocks.
            moe_layer_freq=moe_layer_freq,
            # GLM/DeepSeek-style details
            qk_layernorm=True,
            add_qkv_bias=False,
            add_bias_linear=False,
            persist_layer_norm=True,
            bias_activation_fusion=True,
            bias_dropout_fusion=True,
            # Megatron DSA currently routes through multi-latent attention, and
            # TransformerConfig rejects rotary_interleaved with MLA.
            rotary_interleaved=False,
        )

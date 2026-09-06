"""Compatibility helpers for the bridge packages used by slime."""

from __future__ import annotations


def bridge_from_pretrained(path: str, *, trust_remote_code: bool = True):
    """Return a bridge instance for the bridge API available in this image."""

    try:
        from megatron.bridge import AutoBridge
    except ModuleNotFoundError as exc:
        if exc.name != "megatron.bridge":
            raise
        from mbridge import AutoBridge

        # Register slime's model adapters (including GLM-5.1) for mbridge.
        import slime_plugins.megatron_bridge  # noqa: F401

    if hasattr(AutoBridge, "from_hf_pretrained"):
        return AutoBridge.from_hf_pretrained(path, trust_remote_code=trust_remote_code)
    try:
        return AutoBridge.from_pretrained(path, trust_remote_code=trust_remote_code)
    except ValueError as exc:
        # Older Transformers does not know GLM-5.1's ``glm_moe_dsa`` type,
        # although mbridge does.  Load the JSON into a generic config (which
        # preserves all fields) and let mbridge select its registered bridge.
        if "does not recognize this architecture" not in str(exc):
            raise
        from transformers import PretrainedConfig

        config = PretrainedConfig.from_pretrained(path)
        config.model_type = "glm_moe_dsa"
        # Transformers >=4.XX stores the RoPE base in the nested
        # ``rope_parameters`` object, whereas this mbridge release still
        # reads the legacy top-level attribute.
        rope_parameters = getattr(config, "rope_parameters", None)
        if isinstance(rope_parameters, dict) and "rope_theta" in rope_parameters:
            config.rope_theta = rope_parameters["rope_theta"]
        return AutoBridge.from_config(config)

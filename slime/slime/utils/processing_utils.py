import base64
import io
import json
import logging
from pathlib import Path

from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast, ProcessorMixin

logger = logging.getLogger(__name__)

# Default image patch size for vision-language models
# Note: Qwen3-VL uses 16, Qwen2.5-VL uses 14
# Reference: https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/README.md
DEFAULT_PATCH_SIZE = 14


def load_tokenizer(name_or_path: str, **kwargs):
    try:
        return AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    except ValueError as exc:
        # GLM-5.1 publishes a valid tokenizer.json but its tokenizer_config
        # names the Transformers-5-only ``TokenizersBackend`` class.  The
        # prepared training image currently has Transformers 4.57, where that
        # class is unavailable.  Construct the equivalent fast tokenizer
        # directly from tokenizer.json rather than mutating the shared model
        # cache or requiring a runtime package upgrade.
        model_dir = Path(name_or_path)
        config_path = model_dir / "tokenizer_config.json"
        tokenizer_path = model_dir / "tokenizer.json"
        if (
            not config_path.is_file()
            or not tokenizer_path.is_file()
            or "TokenizersBackend" not in str(exc)
        ):
            raise

        config = json.loads(config_path.read_text())
        fast_kwargs = {
            key: config[key]
            for key in (
                "bos_token",
                "eos_token",
                "unk_token",
                "sep_token",
                "pad_token",
                "cls_token",
                "mask_token",
                "model_max_length",
                "padding_side",
                "truncation_side",
                "clean_up_tokenization_spaces",
            )
            if key in config
        }
        extra_special_tokens = config.get("extra_special_tokens")
        if extra_special_tokens:
            fast_kwargs["additional_special_tokens"] = extra_special_tokens
        logger.warning(
            "Falling back to PreTrainedTokenizerFast for %s because the "
            "TokenizersBackend class is unavailable: %s",
            name_or_path,
            exc,
        )
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path), **fast_kwargs)
        # Keep the model metadata available to callers such as save_pretrained
        # and chat-template consumers.
        tokenizer.init_kwargs.update(
            {key: value for key, value in config.items() if key != "tokenizer_class"}
        )
        return tokenizer


def load_processor(name_or_path: str, **kwargs):
    try:
        proc = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except (OSError, ValueError) as e:
        logger.warning(f"Failed to load processor from {name_or_path}: {e}")
        proc = None

    # If HF returned a tokenizer, discard it.
    if isinstance(proc, PreTrainedTokenizerBase) or not isinstance(proc, ProcessorMixin):
        proc = None

    return proc


def process_vision_info(prompt, processor):
    # temporary solution, will write image utils for slime later
    from qwen_vl_utils import process_vision_info

    if hasattr(processor.image_processor, "patch_size"):
        image_patch_size = processor.image_processor.patch_size
    else:
        logger.info(f"Using default patch size: {DEFAULT_PATCH_SIZE}")
        image_patch_size = DEFAULT_PATCH_SIZE
    images, videos = process_vision_info(prompt, image_patch_size=image_patch_size)
    multimodal_inputs = {"images": images, "videos": videos}
    return multimodal_inputs


def encode_image_for_rollout_engine(image) -> str:
    """Load an image from path, ensure RGB, encode as PNG base64 string."""
    buffer = io.BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

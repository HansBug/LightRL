from __future__ import annotations

import hashlib
import json
import logging
import os
from types import SimpleNamespace
from typing import Any

from slime.utils.misc import load_function

logger = logging.getLogger(__name__)

_PREFLIGHT_STATE: dict[str, Any] = {}


def _preflight_strict() -> bool:
    return os.environ.get("LWM_TARGET_PROVIDER_STRICT", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def preflight_world_model_target_provider(args: Any, provider_path: str) -> None:
    """Exercise the target provider once so misconfiguration fails fast.

    A ``latent_wm`` arm whose provider cannot produce targets silently degrades
    into a plain baseline while burning the same cluster budget.  Running the
    provider on a dummy sample at the first attach call surfaces missing
    checkpoints (``LWM_TARGET_CHECKPOINT``), unloadable encoders, and import
    errors at rollout 0 instead of mid-run.  Strictness is controlled with
    ``LWM_TARGET_PROVIDER_STRICT`` (default ``1`` = raise; ``0`` = log and
    continue without targets).
    """
    if _PREFLIGHT_STATE.get(provider_path):
        return
    provider = load_function(provider_path)
    record = {
        "schema": "openclaw_terminal_latent_world_model_v2",
        "next_observation_text": "world-model provider preflight probe",
        "feedback_text": "world-model provider preflight probe",
    }
    dummy = SimpleNamespace(metadata={"world_model": record}, train_metadata={})
    try:
        provided = provider(
            args=args,
            samples=[dummy],
            turn_records=[],
            task_meta={},
            run_ctx=None,
        )
        if provided is None:
            target = (dummy.metadata.get("world_model") or {}).get("target_latents")
        else:
            target = provided[0] if provided else None
        if target is None:
            raise RuntimeError("provider produced no target for the preflight probe sample")
    except Exception as exc:
        message = (
            f"world-model target provider {provider_path} failed its preflight probe: "
            f"{type(exc).__name__}: {exc}. The latent_wm arm would receive no targets. "
            "Fix the provider configuration (e.g. LWM_TARGET_CHECKPOINT / "
            "LWM_TARGET_ENCODER_MODEL) or set LWM_TARGET_PROVIDER_STRICT=0 to "
            "degrade to no-target operation with a loud log instead."
        )
        if _preflight_strict():
            # Do not mark the probe as done: every attach retries it, so a
            # broken provider keeps failing fast instead of silently degrading
            # to a no-target baseline on driver retries.
            raise RuntimeError(message) from exc
        # Non-strict mode degrades once; skip future probes to avoid log spam.
        _PREFLIGHT_STATE[provider_path] = True
        logger.error("%s", message)
    else:
        _PREFLIGHT_STATE[provider_path] = True
        logger.info("world-model target provider %s passed preflight probe", provider_path)


def is_world_model_enabled(args: Any) -> bool:
    raw = getattr(args, "world_model_enable", False)
    if isinstance(raw, str):
        raw = raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(raw)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def stable_hash(value: Any, *, digest_size: int = 16) -> str:
    payload = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=digest_size).hexdigest()


def _truncate(text: Any, max_chars: int, *, strategy: str = "head") -> str:
    text = "" if text is None else str(text)
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if strategy == "tail":
        return text[-max_chars:]
    if strategy == "head_tail":
        marker = "\n[openclaw_truncated_middle]\n"
        if max_chars <= len(marker) + 2:
            return text[:max_chars]
        keep = max_chars - len(marker)
        head_chars = max(1, keep // 4)
        tail_chars = keep - head_chars
        return text[:head_chars] + marker + text[-tail_chars:]
    return text[:max_chars]


def _tool_action_text(call: dict[str, Any], max_chars: int) -> str:
    name = str(call.get("tool_name") or call.get("name") or "tool")
    args = call.get("args")
    if args is None:
        args = call.get("arguments")
    args_text = json.dumps(_jsonable(args), ensure_ascii=False, sort_keys=True, default=str)
    return _truncate(f"{name}({args_text})", max_chars, strategy="head_tail")


def _render_tool_result(value: Any) -> str:
    """Render one raw tool result without terminal/prompt decoration."""
    if isinstance(value, dict):
        fields = [
            f"{key}={value[key]}"
            for key in ("stdout", "stderr", "exit_code")
            if value.get(key) is not None
        ]
        if fields:
            return "\n".join(fields)
        return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, default=str)
    return str(value)


def _extract_action_text(turn: dict[str, Any], max_chars: int) -> str:
    parts: list[str] = []
    assistant = str(turn.get("assistant_output") or "").strip()
    if assistant:
        parts.append(assistant)
    for call in turn.get("tool_calls") or []:
        if isinstance(call, dict):
            parts.append(_tool_action_text(call, max_chars))
    return _truncate("\n".join(parts), max_chars, strategy="head_tail")


def _extract_context_text(turn: dict[str, Any], *, task_meta: dict[str, Any], max_chars: int) -> str:
    context_messages = turn.get("context_messages") or []
    payload = {
        "task_name": task_meta.get("task_name"),
        "task_path": task_meta.get("task_path"),
        "data_source": task_meta.get("data_source"),
        "turn_idx": turn.get("turn_idx"),
        "context_messages": context_messages,
    }
    return _truncate(
        json.dumps(_jsonable(payload), ensure_ascii=False, sort_keys=True, default=str),
        max_chars,
        strategy="head_tail",
    )


def _extract_observation_text(
    turn: dict[str, Any],
    *,
    status: Any,
    reward: dict[str, Any],
    eval_details: dict[str, Any] | None,
    eval_error: str | None,
    max_chars: int,
) -> str:
    observations: list[str] = []
    for call in turn.get("tool_calls") or []:
        if isinstance(call, dict) and call.get("result") is not None:
            observations.append(_render_tool_result(call.get("result")))
    if not observations:
        reason = None
        if isinstance(eval_details, dict):
            reason = eval_details.get("reason") or eval_details.get("message")
        observations.append(
            json.dumps(
                {
                    "status": getattr(status, "value", str(status)),
                    "score": reward.get("score"),
                    "raw_score": reward.get("raw_score"),
                    "eval_reason": reason,
                    "eval_error": eval_error,
                },
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
        )
    return _truncate("\n".join(observations), max_chars, strategy="head_tail")


def build_terminal_world_model_record(
    *,
    sample: Any,
    turn: dict[str, Any],
    task_meta: dict[str, Any],
    run_ctx: Any,
    status: Any,
    eval_details: dict[str, Any] | None,
    eval_error: str | None,
    max_chars: int,
    next_turn: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reward = sample.reward if isinstance(getattr(sample, "reward", None), dict) else {}
    context_text = _extract_context_text(turn, task_meta=task_meta, max_chars=max_chars)
    action_text = _extract_action_text(turn, max_chars)
    next_observation_text = _extract_observation_text(
        turn,
        status=status,
        reward=reward,
        eval_details=eval_details,
        eval_error=eval_error,
        max_chars=max_chars,
    )
    context_messages = turn.get("context_messages") or []
    response_length = int(getattr(sample, "response_length", 0) or 0)
    token_count = len(getattr(sample, "tokens", []) or [])
    sample_metadata = getattr(sample, "metadata", {}) if isinstance(getattr(sample, "metadata", {}), dict) else {}
    turn_idx = int(turn.get("turn_idx", sample_metadata.get("turn_idx", 0) or 0))
    num_turns = sample_metadata.get("num_turns")
    status_value = getattr(status, "value", str(status))
    abnormal_done = status_value in {"truncated", "aborted", "failed"}
    final_turn = num_turns is not None and turn_idx >= int(num_turns) - 1
    next_context_text = None
    if next_turn is not None:
        next_context_text = _extract_context_text(next_turn, task_meta=task_meta, max_chars=max_chars)
    return {
        "schema": "openclaw_terminal_latent_world_model_v2",
        "hidden_source": "policy_prompt_action_spans",
        "task_name": task_meta.get("task_name"),
        "task_path": task_meta.get("task_path"),
        "data_source": task_meta.get("data_source"),
        "uid": getattr(run_ctx, "uid", None),
        "group_index": getattr(run_ctx, "group_index", None),
        "sample_index": getattr(run_ctx, "sample_index", None),
        "rollout_id": getattr(run_ctx, "rollout_id", None),
        "train_step": getattr(run_ctx, "train_step", None),
        "turn_idx": turn_idx,
        "num_turns": num_turns,
        "status": status_value,
        "done": bool(final_turn or (num_turns is None and abnormal_done)),
        "context_hash": stable_hash(context_messages),
        "action_hash": stable_hash(action_text),
        "next_observation_hash": stable_hash(next_observation_text),
        "context_token_len": max(0, token_count - response_length),
        "action_token_len": response_length,
        "context_text": context_text,
        "context_text_source": "context_messages.head_tail",
        "context_text_truncation": "head_tail",
        "next_context_text": next_context_text,
        "next_context_hash": stable_hash(next_turn.get("context_messages") or []) if next_turn else None,
        "action_text": action_text,
        "next_observation_text": next_observation_text,
        "reward_score": reward.get("score"),
        "reward_base_score": reward.get("base_score"),
        "reward_raw_score": reward.get("raw_score"),
        "has_tool_result": any(
            isinstance(call, dict) and bool(call.get("result"))
            for call in (turn.get("tool_calls") or [])
        ),
    }


def _build_turn_index(turn_records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Index turn records by every turn_idx they cover.

    Besides the record's own ``turn_idx``, a record also covers the sub-turn
    indices listed in ``sdk_model_turns`` — per-interaction samples carry those
    indices, and without them the lookup silently drops every sample that does
    not come from a turn's final model interaction.
    """
    index: dict[int, dict[str, Any]] = {}
    for i, turn in enumerate(turn_records or []):
        primary = int(turn.get("turn_idx", i))
        index.setdefault(primary, turn)
        for sub in turn.get("sdk_model_turns") or []:
            if isinstance(sub, dict) and sub.get("turn_idx") is not None:
                index.setdefault(int(sub["turn_idx"]), turn)
    return index


def attach_terminal_world_model_metadata(
    *,
    args: Any,
    samples: list[Any],
    turn_records: list[dict[str, Any]],
    task_meta: dict[str, Any],
    run_ctx: Any,
    status: Any,
    eval_details: dict[str, Any] | None = None,
    eval_error: str | None = None,
) -> None:
    if not is_world_model_enabled(args) or not samples:
        return
    max_chars = int(getattr(args, "world_model_metadata_max_chars", 4096) or 4096)
    turn_by_idx = _build_turn_index(turn_records)
    matched = 0
    fallback = 0
    missed = 0
    for sample in samples:
        metadata = sample.metadata if isinstance(getattr(sample, "metadata", None), dict) else {}
        sample.metadata = metadata
        turn_idx = int(metadata.get("turn_idx", 0) or 0)
        turn = turn_by_idx.get(turn_idx)
        if turn is None and turn_records:
            # Positional fallback: clamp to the nearest recorded turn so the
            # sample still carries an observation instead of being silently
            # skipped (a missing record zeroes that sample's aux supervision).
            fallback_idx = min(max(turn_idx, 0), len(turn_records) - 1)
            turn = turn_records[fallback_idx]
            fallback += 1
        if turn is None:
            missed += 1
            continue
        matched += 1
        record = build_terminal_world_model_record(
            sample=sample,
            turn=turn,
            task_meta=task_meta,
            run_ctx=run_ctx,
            status=status,
            eval_details=eval_details,
            eval_error=eval_error,
            max_chars=max_chars,
            next_turn=turn_by_idx.get(turn_idx + 1),
        )
        sample.metadata["world_model"] = record
        train_metadata = dict(sample.train_metadata or {})
        train_metadata["world_model"] = record
        sample.train_metadata = train_metadata
    if fallback or missed:
        logger.warning(
            "world-model metadata attach coverage: matched=%d/%d samples (fallback=%d, missed=%d)",
            matched,
            len(samples),
            fallback,
            missed,
        )

    provider_path = getattr(args, "world_model_target_provider_path", None)
    if not provider_path:
        return
    preflight_world_model_target_provider(args, provider_path)
    provider = load_function(provider_path)
    try:
        provided = provider(
            args=args,
            samples=samples,
            turn_records=turn_records,
            task_meta=task_meta,
            run_ctx=run_ctx,
        )
    except Exception as exc:
        # A broken provider (missing checkpoint/env, encoder OOM, ...) must not
        # turn otherwise healthy rollouts into generate failures.  Degrade to
        # no-target for this batch and make the failure loud instead; use the
        # provider preflight for a startup-time hard check.
        logger.error(
            "world-model target provider %s failed for this batch; continuing without targets: %s: %s",
            provider_path,
            type(exc).__name__,
            exc,
        )
        return
    if provided is None:
        # A provider may mutate sample.metadata directly.  This is useful when
        # it keeps a resident frozen encoder and avoids copying large tensors.
        return
    if not isinstance(provided, (list, tuple)) or len(provided) != len(samples):
        raise ValueError(
            "world-model target provider must return one latent payload per sample "
            "or mutate sample.metadata and return None"
        )
    attached = 0
    for sample, target in zip(samples, provided, strict=True):
        world_model = sample.metadata.setdefault("world_model", {})
        if target is not None:
            world_model["target_latents"] = target
            attached += 1
        train_metadata = dict(sample.train_metadata or {})
        train_metadata["world_model"] = world_model
        sample.train_metadata = train_metadata
    logger.info("Attached frozen latent WM targets with provider=%s", provider_path)
    if attached == 0:
        logger.warning(
            "world-model target provider %s returned targets for 0/%d samples",
            provider_path,
            len(samples),
        )

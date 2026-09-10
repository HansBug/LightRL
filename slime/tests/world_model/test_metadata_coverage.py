import logging
import sys
import types
from types import SimpleNamespace

import pytest

from slime.world_model import metadata as wm_metadata
from slime.world_model.metadata import attach_terminal_world_model_metadata


def _sample(turn_idx):
    return SimpleNamespace(
        tokens=[1, 2, 3, 4, 5],
        response_length=2,
        reward={"score": 0.5, "base_score": 0.5, "raw_score": 1.0},
        metadata={"turn_idx": turn_idx, "num_turns": 3},
        train_metadata=None,
    )


def _turn_records_with_sdk_sub_turns():
    return [
        {
            "turn_idx": 2,
            "context_messages": [{"role": "user", "content": "hi"}],
            "assistant_output": "first",
            "tool_calls": [{"tool_name": "bash", "result": "out-0"}],
            "sdk_model_turns": [{"turn_idx": 0}, {"turn_idx": 1}, {"turn_idx": 2}],
        },
        {
            "turn_idx": 5,
            "context_messages": [{"role": "user", "content": "hi"}],
            "assistant_output": "second",
            "tool_calls": [{"tool_name": "bash", "result": "out-1"}],
            "sdk_model_turns": [{"turn_idx": 3}, {"turn_idx": 4}, {"turn_idx": 5}],
        },
    ]


def _attach(samples, turn_records, **args_overrides):
    args = SimpleNamespace(world_model_enable=True, world_model_metadata_max_chars=256, **args_overrides)
    attach_terminal_world_model_metadata(
        args=args,
        samples=samples,
        turn_records=turn_records,
        task_meta={},
        run_ctx=SimpleNamespace(),
        status=SimpleNamespace(value="completed"),
    )


def test_sdk_model_turns_indices_match_parent_record():
    samples = [_sample(turn_idx=i) for i in range(6)]
    _attach(samples, _turn_records_with_sdk_sub_turns())
    for i, sample in enumerate(samples):
        assert "world_model" in sample.metadata, f"sample turn_idx={i} missed"
    assert samples[0].metadata["world_model"]["next_observation_text"] == "out-0"
    assert samples[4].metadata["world_model"]["next_observation_text"] == "out-1"


def test_positional_fallback_attaches_nearest_record(caplog):
    sample = _sample(turn_idx=9)
    records = _turn_records_with_sdk_sub_turns()
    with caplog.at_level(logging.WARNING):
        _attach([sample], records)
    assert "world_model" in sample.metadata
    assert sample.metadata["world_model"]["next_observation_text"] == "out-1"
    assert any("fallback=1" in record.getMessage() for record in caplog.records)


def test_provider_exception_degrades_to_no_targets(monkeypatch, caplog):
    module = types.ModuleType("test_raising_provider")

    def provider(**kwargs):
        raise RuntimeError("LWM_TARGET_CHECKPOINT is required")

    module.provider = provider
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("LWM_TARGET_PROVIDER_STRICT", "0")
    wm_metadata._PREFLIGHT_STATE.clear()
    sample = _sample(turn_idx=2)
    with caplog.at_level(logging.ERROR):
        _attach(
            [sample],
            _turn_records_with_sdk_sub_turns(),
            world_model_target_provider_path="test_raising_provider.provider",
        )
    # Rollout must survive a broken provider: record attached, no target.
    assert sample.metadata["world_model"]["next_observation_text"] == "out-0"
    assert sample.metadata["world_model"].get("target_latents") is None
    assert any("failed for this batch" in record.getMessage() for record in caplog.records)


def test_preflight_strict_raises_on_broken_provider(monkeypatch):
    module = types.ModuleType("test_strict_raising_provider")

    def provider(**kwargs):
        raise RuntimeError("missing checkpoint")

    module.provider = provider
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("LWM_TARGET_PROVIDER_STRICT", "1")
    wm_metadata._PREFLIGHT_STATE.clear()
    with pytest.raises(RuntimeError, match="preflight probe"):
        _attach(
            [_sample(turn_idx=2)],
            _turn_records_with_sdk_sub_turns(),
            world_model_target_provider_path="test_strict_raising_provider.provider",
        )


def test_preflight_runs_once_per_provider_path(monkeypatch):
    module = types.ModuleType("test_counting_provider")
    calls = {"n": 0}

    def provider(**kwargs):
        calls["n"] += 1
        return [[0.1, 0.2]] * len(kwargs["samples"])

    module.provider = provider
    monkeypatch.setitem(sys.modules, module.__name__, module)
    wm_metadata._PREFLIGHT_STATE.clear()
    samples = [_sample(turn_idx=2), _sample(turn_idx=5)]
    _attach(samples, _turn_records_with_sdk_sub_turns(), world_model_target_provider_path="test_counting_provider.provider")
    _attach(samples, _turn_records_with_sdk_sub_turns(), world_model_target_provider_path="test_counting_provider.provider")
    # 1 preflight probe + 2 real calls.
    assert calls["n"] == 3

import asyncio
import time
from types import SimpleNamespace

import pytest

from slime.rollout import sglang_rollout
from slime.rollout.sglang_rollout import (
    _rollout_generation_max_seconds,
    _rollout_generation_watchdog_seconds,
    _should_fail_all_failed_round,
)
from slime.utils.types import Sample


def test_rollout_generation_max_seconds_defaults_to_bounded():
    assert _rollout_generation_max_seconds(SimpleNamespace()) == 2400.0


def test_rollout_generation_max_seconds_respects_override_and_disable():
    assert _rollout_generation_max_seconds(SimpleNamespace(rollout_generation_max_seconds=600.0)) == 600.0
    assert _rollout_generation_max_seconds(SimpleNamespace(rollout_generation_max_seconds=0.0)) is None
    assert _rollout_generation_max_seconds(SimpleNamespace(rollout_generation_max_seconds="bad")) == 2400.0


def test_rollout_generation_watchdog_seconds_defaults_and_fallback():
    assert _rollout_generation_watchdog_seconds(SimpleNamespace()) == 300.0
    assert _rollout_generation_watchdog_seconds(SimpleNamespace(rollout_generation_watchdog_seconds=60.0)) == 60.0
    assert _rollout_generation_watchdog_seconds(SimpleNamespace(rollout_generation_watchdog_seconds=-1.0)) == 300.0


def test_all_failed_round_detection():
    assert _should_fail_all_failed_round(kept_groups=4, failed_groups=4)
    assert not _should_fail_all_failed_round(kept_groups=4, failed_groups=3)
    assert not _should_fail_all_failed_round(kept_groups=0, failed_groups=0)


class _FakeState:
    def __init__(self, group_results):
        self.remaining_batch_size = 0
        self.pendings = set()
        self.aborted = False
        self._group_results = list(group_results)

    def submit_generate_tasks(self, samples):
        for _group in samples:
            result = self._group_results.pop(0) if self._group_results else "hang"
            if result == "hang":
                task = asyncio.create_task(asyncio.sleep(3600))
            else:
                task = asyncio.create_task(_immediate(result))
            self.pendings.add(task)
        self.remaining_batch_size += len(samples)

    def reset(self):
        self.remaining_batch_size = 0
        self.pendings = set()
        self.aborted = False


async def _immediate(value):
    return value


def _args(**overrides):
    base = dict(
        rollout_global_dataset=True,
        rollout_batch_size=1,
        n_samples_per_prompt=1,
        over_sampling_batch_size=1,
        dynamic_sampling_max_seconds=None,
        dynamic_sampling_filter_path=None,
        dynamic_sampling_failed_group_abort_min_groups=None,
        dynamic_sampling_failed_group_abort_ratio=1.0,
        rollout_generation_max_seconds=1.0,
        rollout_generation_watchdog_seconds=0.2,
        partial_rollout=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _failed_sample():
    return Sample(
        status=Sample.Status.FAILED,
        remove_sample=True,
        prompt="p",
        response="",
        label="l",
        reward={"score": 0.0},
    )


def _install_fake_state(monkeypatch, state):
    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda args: state)
    monkeypatch.setattr(sglang_rollout, "_annotate_rollout_groups", lambda *a, **k: None)


def test_never_resolving_group_task_fails_loudly_instead_of_hanging(monkeypatch):
    state = _FakeState(group_results=["hang"])
    _install_fake_state(monkeypatch, state)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="dynamic sampling timed out"):
        asyncio.run(sglang_rollout.generate_rollout_async(_args(), rollout_id=0, data_source=lambda n: [[object()]]))
    elapsed = time.monotonic() - started
    # The 3600s fake task must be cut by the ~1s round deadline, not awaited.
    assert elapsed < 30.0


def test_all_failed_round_is_rejected_with_env_storm_marker(monkeypatch):
    state = _FakeState(group_results=[[_failed_sample()]])
    _install_fake_state(monkeypatch, state)
    with pytest.raises(RuntimeError, match="all-failed rollout groups"):
        asyncio.run(sglang_rollout.generate_rollout_async(_args(), rollout_id=0, data_source=lambda n: [[object()]]))

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local scheduler lifecycle checks without model or clock nondeterminism."""

from types import SimpleNamespace

import pytest

from h3_arrival_scheduler import Arrival, replay


class FakeEngine:
    def __init__(self):
        self.now = 0.0

    def reset(self):
        pass

    def prepare(self, request_id, prompt, seed):
        return SimpleNamespace(request_id=request_id, index=0, done=False)

    def step(self, active):
        self.now += len(active)
        for state in active:
            state.index += 1
            state.done = state.index == 4

    def finish(self, done):
        return [s.request_id for s in done], [s.request_id for s in done]

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


@pytest.mark.parametrize("policy", ["b1", "static_b2", "continuous_b2"])
def test_arrival_replay_preserves_steps_and_never_admits_early(policy):
    engine = FakeEngine()
    requests = [Arrival(i, at, "prompt", 42 + i) for i, at in enumerate((0, 1, 3, 10))]
    report, outputs = replay(engine, requests, policy, engine.clock, engine.sleep)
    assert set(outputs) == {0, 1, 2, 3}
    for row in report["requests"]:
        assert row["admitted"] >= row["arrival"]
        assert row["total_seconds"] == row["queue_seconds"] + row["active_seconds"]
    steps = [event for event in report["events"] if event["event"] == "step"]
    for request in requests:
        assert [
            event["indices"][event["ids"].index(request.request_id)]
            for event in steps
            if request.request_id in event["ids"]
        ] == [0, 1, 2, 3]
    assert max(len(event["ids"]) for event in steps) <= (1 if policy == "b1" else 2)
    mixed = any(len(set(event["indices"])) > 1 for event in steps)
    assert mixed == (policy == "continuous_b2")


def test_continuous_refills_slot_without_resetting_survivor():
    engine = FakeEngine()
    requests = [Arrival(i, at, "prompt", 42 + i) for i, at in enumerate((0, 1, 2))]
    report, _ = replay(engine, requests, "continuous_b2", engine.clock, engine.sleep)
    steps = [e for e in report["events"] if e["event"] == "step"]
    assert any(e["ids"] == [1, 2] and e["indices"] == [3, 0] for e in steps)


def test_invalid_arrivals_rejected():
    with pytest.raises(ValueError, match="nonnegative and sorted"):
        replay(FakeEngine(), [Arrival(0, -1, "prompt", 42)], "b1")

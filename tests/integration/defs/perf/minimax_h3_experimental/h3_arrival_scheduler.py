# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-worker open-loop arrival replay, with real denoising-step boundaries."""

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Arrival:
    request_id: int
    at: float
    prompt: str
    seed: int


def replay(
    engine,
    requests: list[Arrival],
    policy: str,
    clock=time.perf_counter,
    sleep=time.sleep,
):
    if policy not in ("b1", "static_b2", "continuous_b2"):
        raise ValueError(f"Unsupported policy: {policy}")
    if not requests or len({r.request_id for r in requests}) != len(requests):
        raise ValueError("Requests must be nonempty and have unique IDs")
    if any(r.at < 0 for r in requests) or requests != sorted(
        requests, key=lambda r: r.at
    ):
        raise ValueError("Arrival times must be nonnegative and sorted")
    engine.reset()
    capacity = 1 if policy == "b1" else 2
    active, completed, events, outputs = [], [], [], {}
    rows = {
        r.request_id: {
            "id": r.request_id,
            "arrival": r.at,
            "seed": r.seed,
            "prompt": r.prompt,
        }
        for r in requests
    }
    next_request = 0
    started = clock()
    while active or next_request < len(requests):
        if not active or policy == "continuous_b2":
            while len(active) < capacity and next_request < len(requests):
                request = requests[next_request]
                now = clock() - started
                if request.at > now:
                    break
                rows[request.request_id]["admitted"] = now
                state = engine.prepare(request.request_id, request.prompt, request.seed)
                rows[request.request_id]["prepared"] = clock() - started
                active.append(state)
                next_request += 1
                events.append(
                    {
                        "event": "admit",
                        "at": now,
                        "id": request.request_id,
                        "active": [s.request_id for s in active],
                    }
                )
        if not active:
            sleep(max(0, started + requests[next_request].at - clock()))
            continue
        begin = clock() - started
        before = [s.index for s in active]
        engine.step(active)
        events.append(
            {
                "event": "step",
                "at": begin,
                "end": clock() - started,
                "ids": [s.request_id for s in active],
                "indices": before,
            }
        )
        done = [s for s in active if s.done]
        if done:
            video, audio = engine.finish(done)
            finished = clock() - started
            for i, state in enumerate(done):
                row = rows[state.request_id]
                row.update(
                    completed=finished,
                    queue_seconds=row["admitted"] - row["arrival"],
                    active_seconds=finished - row["admitted"],
                    total_seconds=finished - row["arrival"],
                )
                completed.append(row)
                outputs[state.request_id] = (video[i], audio[i])
                events.append(
                    {"event": "finish", "at": finished, "id": state.request_id}
                )
            active = [s for s in active if not s.done]
    duration = clock() - started
    return {
        "policy": policy,
        "duration_seconds": duration,
        "videos_per_minute": 60 * len(requests) / duration,
        "requests": completed,
        "events": events,
    }, outputs

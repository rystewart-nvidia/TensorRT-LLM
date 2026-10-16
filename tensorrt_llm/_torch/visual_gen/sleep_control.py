# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coordinate sleep/wake with generation requests."""

from _thread import LockType
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import count
from queue import Queue
from threading import TIMEOUT_MAX, Lock
from time import monotonic
from typing import Literal, TypeVar

_T = TypeVar("_T")
_SleepState = Literal["awake", "sleeping", "asleep", "waking", "failed", "closed"]


@contextmanager
def _lock_before_deadline(lock: LockType, deadline: float | None) -> Iterator[None]:
    acquired = (
        lock.acquire()
        if deadline is None
        else lock.acquire(timeout=min(max(0.0, deadline - monotonic()), TIMEOUT_MAX))
    )
    if not acquired:
        raise TimeoutError("Timed out waiting to start VisualGen sleep/wake")
    try:
        yield
    finally:
        lock.release()


@dataclass(frozen=True)
class _SleepControlRequest:
    request_id: int
    operation: Literal["sleep", "wake_up"]


class _SleepRequestQueue(Queue):
    """Queue requests and sleep/wake commands; count only generation for metrics."""

    def _init(self, maxsize):
        super()._init(maxsize)
        self._generation_count = 0

    def _put(self, item):
        super()._put(item)
        if item is not None and not isinstance(item, _SleepControlRequest):
            self._generation_count += 1

    def _get(self):
        item = super()._get()
        if item is not None and not isinstance(item, _SleepControlRequest):
            self._generation_count -= 1
        return item

    def generation_qsize(self) -> int:
        with self.mutex:
            return self._generation_count


class _SleepController:
    """Serialize sleep/wake and reject new generation until the worker confirms it is awake."""

    def __init__(self) -> None:
        self._admission_lock = Lock()
        self._transition_lock = Lock()
        self._state: _SleepState = "awake"
        # Non-negative IDs belong to generation; -1 is the worker READY signal.
        self._request_ids = count(-2, -1)

    @property
    def is_sleeping(self) -> bool:
        with self._admission_lock:
            return self._state == "asleep"

    @property
    def state(self) -> _SleepState:
        with self._admission_lock:
            if self._state == "closed":
                raise RuntimeError("VisualGen has been shut down")
            return self._state

    def enqueue(self, submit: Callable[[], _T]) -> _T:
        with self._admission_lock:
            if self._state != "awake":
                raise RuntimeError(
                    f"VisualGen is {self._state}; generation requires an awake instance"
                )
            return submit()

    def close(self) -> None:
        with self._admission_lock:
            self._state = "closed"

    def transition(
        self,
        operation: Literal["sleep", "wake_up"],
        send_and_wait: Callable[[_SleepControlRequest], None],
        *,
        deadline: float | None = None,
    ) -> None:
        source: _SleepState
        pending: _SleepState
        target: _SleepState
        source, pending, target = (
            ("awake", "sleeping", "asleep")
            if operation == "sleep"
            else ("asleep", "waking", "awake")
        )
        with _lock_before_deadline(self._transition_lock, deadline):
            with _lock_before_deadline(self._admission_lock, deadline):
                if self._state == target:
                    return
                if self._state != source:
                    raise RuntimeError(f"Cannot {operation} VisualGen in state {self._state}")
                if deadline is not None and monotonic() >= deadline:
                    raise TimeoutError(f"VisualGen {operation} timed out before submission")
                self._state = pending
                request = _SleepControlRequest(next(self._request_ids), operation)
            try:
                send_and_wait(request)
                with self._admission_lock:
                    if self._state == "closed":
                        raise RuntimeError("VisualGen was shut down during sleep/wake")
                    self._state = target
            finally:
                with self._admission_lock:
                    if self._state == pending:
                        self._state = "failed"

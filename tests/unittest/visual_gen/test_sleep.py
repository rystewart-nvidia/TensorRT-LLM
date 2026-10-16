# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test public sleep/wake admission and worker responses."""

import asyncio
import json
import pickle
import queue
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from tensorrt_llm import VisualGen, VisualGenArgs
from tensorrt_llm._torch.visual_gen import sleep_control
from tensorrt_llm._torch.visual_gen.executor import (
    DiffusionExecutor,
    DiffusionRemoteClient,
    DiffusionRequest,
    DiffusionResponse,
    _IterationStatsTracker,
)
from tensorrt_llm._torch.visual_gen.sleep_control import (
    _SleepController,
    _SleepControlRequest,
    _SleepRequestQueue,
)
from tensorrt_llm.visual_gen import VisualGenMemoryStatus

pytestmark = pytest.mark.cpu_only


def test_sleep_config_default_and_validation(tmp_path):
    assert VisualGenArgs().sleep_restore_mode is None
    for mode in (None, "CPU", "PINNED"):
        args = VisualGenArgs(sleep_restore_mode=mode)
        assert VisualGenArgs.model_validate_json(args.model_dump_json()).sleep_restore_mode == mode
    path = tmp_path / "args.yml"
    path.write_text("sleep_restore_mode: PINNED\n")
    assert VisualGenArgs.from_yaml(path).sleep_restore_mode == "PINNED"
    for mode in ("NONE", "MEMSET", "pinned", True):
        with pytest.raises(ValidationError, match="sleep_restore_mode"):
            VisualGenArgs(sleep_restore_mode=mode)


def test_sleep_schema_accepts_exactly_the_public_backing_modes():
    path = (
        Path(__file__).resolve().parents[3]
        / "tensorrt_llm/schemas/trtllm-serve-visual-gen-config.schema.json"
    )
    schema = json.loads(path.read_text())
    validator = Draft202012Validator(schema)
    for mode in (None, "CPU", "PINNED", "NONE", "MEMSET", "pinned", True):
        assert validator.is_valid({"sleep_restore_mode": mode}) == (mode in (None, "CPU", "PINNED"))


def test_sleep_closes_admission_before_waiting_and_serializes_wake():
    controller = _SleepController()
    entered, finish = Event(), Event()
    commands = []

    def send(request):
        commands.append(pickle.loads(pickle.dumps(request)))
        if request.operation == "sleep":
            entered.set()
            assert finish.wait(10)

    with ThreadPoolExecutor(max_workers=2) as threads:
        sleeping = threads.submit(controller.transition, "sleep", send)
        try:
            assert entered.wait(10)
            submit = Mock()
            with pytest.raises(RuntimeError, match="sleeping"):
                controller.enqueue(submit)
            submit.assert_not_called()
            waking = threads.submit(controller.transition, "wake_up", send)
            assert not sleeping.done()
            assert not waking.done()
            assert not controller.is_sleeping
        finally:
            finish.set()
        sleeping.result(timeout=10)
        waking.result(timeout=10)
    assert [command.operation for command in commands] == ["sleep", "wake_up"]
    assert [command.request_id for command in commands] == [-2, -3]
    assert controller.enqueue(lambda: "accepted") == "accepted"


def test_repeated_transitions_are_idempotent_and_pools_independent():
    first, second = _SleepController(), _SleepController()
    send = Mock()
    for _ in range(2):
        first.transition("sleep", send)
    assert first.is_sleeping
    assert not second.is_sleeping
    assert second.enqueue(lambda: 1) == 1
    for _ in range(2):
        first.transition("wake_up", send)
    assert send.call_count == 2
    assert first.enqueue(lambda: 2) == 2


@pytest.mark.parametrize("operation", ["sleep", "wake_up"])
def test_transition_failure_stays_closed(operation):
    controller = _SleepController()
    if operation == "wake_up":
        controller.transition("sleep", lambda _request: None)
    with pytest.raises(RuntimeError, match="offload failed"):
        controller.transition(operation, Mock(side_effect=RuntimeError("offload failed")))
    with pytest.raises(RuntimeError, match="failed"):
        controller.enqueue(lambda: pytest.fail("Generation was admitted after failure"))
    for retry in ("sleep", "wake_up"):
        with pytest.raises(RuntimeError, match="failed"):
            controller.transition(retry, Mock())


def test_shutdown_during_transition_cannot_reopen_admission():
    controller = _SleepController()
    with pytest.raises(RuntimeError, match="shut down"):
        controller.transition("sleep", lambda _request: controller.close())
    with pytest.raises(RuntimeError, match="closed"):
        controller.enqueue(lambda: pytest.fail("Generation was admitted after shutdown"))


def test_expired_deadline_between_locks_does_not_submit_or_leak_locks(monkeypatch):
    controller = _SleepController()
    send = Mock()
    monkeypatch.setattr(sleep_control, "monotonic", Mock(side_effect=[0.0, 1.1, 1.1]))
    with pytest.raises(TimeoutError, match="before submission"):
        controller.transition("sleep", send, deadline=1.0)
    send.assert_not_called()
    assert controller.enqueue(lambda: "accepted") == "accepted"
    assert controller._transition_lock.acquire(blocking=False)
    controller._transition_lock.release()


def test_control_queue_counts_only_generation_but_retains_fifo_order():
    requests = _SleepRequestQueue()
    entries = [DiffusionRequest(0, ["space"]), _SleepControlRequest(-2, "sleep"), None]
    for entry in entries:
        requests.put(entry)
    assert requests.qsize() == 3
    assert requests.generation_qsize() == 1
    assert requests.get_nowait() is entries[0]
    assert requests.generation_qsize() == 0
    assert not requests.empty()
    assert requests.get_nowait() is entries[1]
    assert requests.get_nowait() is None
    assert requests.empty()
    assert requests.generation_qsize() == 0


class _Pipeline:
    def __init__(self):
        self.is_sleeping = False
        self.operations = []

    def sleep(self):
        self.operations.append("sleep")
        self.is_sleeping = True

    def wake_up(self):
        self.operations.append("wake_up")
        self.is_sleeping = False


def _worker():
    worker = DiffusionExecutor.__new__(DiffusionExecutor)
    worker.rank = worker.device_id = 0
    worker.world_size = 1
    worker.visual_gen_args = VisualGenArgs(sleep_restore_mode="PINNED")
    worker.pipeline = _Pipeline()
    worker.response_queue = queue.Queue()
    return worker


@pytest.mark.parametrize("operation", ["sleep", "wake_up"])
def test_worker_executes_control_and_reports_failures(operation):
    worker = _worker()
    worker._process_sleep_control(_SleepControlRequest(-2, operation))
    response = worker.response_queue.get_nowait()
    assert response.error_msg is None
    assert response.output == {"sleeping": operation == "sleep"}
    assert worker.pipeline.operations == [operation]
    setattr(worker.pipeline, operation, Mock(side_effect=RuntimeError("transfer failed")))
    worker._process_sleep_control(_SleepControlRequest(-3, operation))
    failure = worker.response_queue.get_nowait()
    assert failure.request_id == -3
    assert failure.error_msg == "transfer failed"


def test_worker_rejects_control_without_opt_in_or_on_multiple_ranks():
    worker = _worker()
    worker.visual_gen_args = VisualGenArgs()
    worker._process_sleep_control(_SleepControlRequest(-2, "sleep"))
    assert "not enabled" in worker.response_queue.get_nowait().error_msg
    worker.visual_gen_args.sleep_restore_mode = "PINNED"
    worker.world_size = 2
    worker._process_sleep_control(_SleepControlRequest(-3, "sleep"))
    assert "single GPU" in worker.response_queue.get_nowait().error_msg
    assert worker.pipeline.operations == []


@pytest.mark.parametrize("mode", [None, "CPU", "PINNED"])
def test_worker_loader_forwards_only_opted_in_sleep_configuration(monkeypatch, mode):
    from tensorrt_llm._torch.visual_gen import executor as executor_module

    worker = _worker()
    worker.visual_gen_args.sleep_restore_mode = mode
    pipeline = SimpleNamespace(
        default_generation_params={},
        extra_param_specs={},
        supports_image_edit=False,
        ref_slot_specs={},
    )
    loader = Mock()
    loader.load.return_value = pipeline
    monkeypatch.setattr(executor_module, "PipelineLoader", Mock(return_value=loader))
    monkeypatch.setattr(executor_module.dist, "barrier", lambda: None)
    worker._load_pipeline()
    kwargs = {"skip_warmup": worker.visual_gen_args.compilation_config.skip_warmup}
    if mode is not None:
        kwargs["sleep_restore_mode"] = mode
    loader.load.assert_called_once_with(**kwargs)
    assert worker.pipeline is pipeline
    assert worker.response_queue.get_nowait().output["status"] == "READY"


def test_public_multi_gpu_sleep_is_rejected_before_worker_startup():
    args = VisualGenArgs(sleep_restore_mode="PINNED", parallel_config={"ulysses_size": 2})
    with pytest.raises(ValueError, match="single GPU"):
        DiffusionRemoteClient(args)


def test_worker_loop_orders_generation_before_sleep_and_bypasses_media_handling():
    worker = _worker()
    worker.sender_thread = None
    incoming = queue.Queue()
    worker.requests_ipc = incoming
    for request in (
        DiffusionRequest(0, ["space"]),
        DiffusionRequest(1, ["moon"]),
        _SleepControlRequest(-2, "sleep"),
        _SleepControlRequest(-3, "wake_up"),
        DiffusionRequest(2, ["planet"]),
        None,
    ):
        incoming.put(request)
    worker.process_request = lambda req: worker.pipeline.operations.append(req.request_id)
    worker.serve_forever()
    assert worker.pipeline.operations == [0, 1, "sleep", "wake_up", 2]
    assert worker.response_queue.get_nowait().output == {"sleeping": True}
    assert worker.response_queue.get_nowait().output == {"sleeping": False}


@contextmanager
def _client():
    """Exercise client request handling without a GPU or IPC."""
    client = DiffusionRemoteClient.__new__(DiffusionRemoteClient)
    client._sleep_controller = _SleepController()
    client._shutdown_started = False
    client._worker_failure = None
    client.pending_requests = _SleepRequestQueue()
    client.completed_responses = {}
    client._abandoned_request_ids = set()
    client._iter_stats = _IterationStatsTracker()
    client._request_to_send = None
    ready = Event()

    def run():
        client._event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(client._event_loop)
        client.lock = asyncio.Lock()
        client.response_event = asyncio.Event()
        ready.set()
        client._event_loop.run_forever()
        pending = asyncio.all_tasks(client._event_loop)
        for task in pending:
            task.cancel()
        client._event_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        client._event_loop.close()

    client.background_thread = Thread(target=run)
    client.background_thread.start()
    assert ready.wait(10)
    try:
        yield client
    finally:
        client._shutdown_started = True
        if client._sleep_controller is not None:
            client._sleep_controller.close()
        if not client._event_loop.is_closed():
            client._event_loop.call_soon_threadsafe(client._event_loop.stop)
        client.background_thread.join(timeout=10)
        assert not client.background_thread.is_alive()


def _ack(client, response):
    asyncio.run_coroutine_threadsafe(client._store_response(response), client._event_loop).result(
        10
    )


@contextmanager
def _transition_threads(client, max_workers=1):
    with ThreadPoolExecutor(max_workers=max_workers) as threads:
        try:
            yield threads
        finally:
            client._shutdown_started = True
            client._sleep_controller.close()


def test_client_drains_queued_requests_and_waits_for_acknowledgement():
    with _client() as client, _transition_threads(client) as threads:
        requests = [DiffusionRequest(i, ["space"]) for i in range(2)]
        assert client.enqueue_requests(requests) == [0, 1]
        sleeping = threads.submit(client.sleep, policy="drain")
        first = client.pending_requests.get(timeout=10)
        second = client.pending_requests.get(timeout=10)
        command = client.pending_requests.get(timeout=10)
        try:
            assert [first.request_id, second.request_id] == [0, 1]
            assert command.operation == "sleep"
            assert not sleeping.done()
            with pytest.raises(RuntimeError, match="sleeping"):
                client.enqueue_requests([DiffusionRequest(2, ["rejected"])])
        finally:
            _ack(client, DiffusionResponse(command.request_id, output={"sleeping": True}))
        sleeping.result(timeout=10)
        assert client.is_sleeping
        with pytest.raises(RuntimeError, match="asleep"):
            client.enqueue_requests([DiffusionRequest(3, ["rejected"])])
        waking = threads.submit(client.wake_up)
        command = client.pending_requests.get(timeout=10)
        _ack(client, DiffusionResponse(command.request_id, output={"sleeping": False}))
        waking.result(timeout=10)
        assert client.enqueue_requests([DiffusionRequest(4, ["accepted"])]) == [4]
        assert not client.is_sleeping
        snapshots = client.get_iteration_stats()
        assert snapshots
        assert all(item["numActiveRequests"] == 0 for item in snapshots)


@pytest.mark.parametrize("failure", ["worker_error", "bad_ack", "worker_exit", "shutdown"])
def test_client_transition_errors_do_not_hang_or_reopen_admission(failure):
    with _client() as client, _transition_threads(client) as threads:
        sleeping = threads.submit(client.sleep)
        command = client.pending_requests.get(timeout=10)
        if failure == "worker_exit":
            client._worker_failure = "worker exited"
            expected = "worker exited"
        elif failure == "shutdown":
            client._shutdown_started = True
            client._sleep_controller.close()
            expected = "shut down"
        else:
            _ack(
                client,
                DiffusionResponse(
                    command.request_id,
                    error_msg="transfer failed" if failure == "worker_error" else None,
                    output={"sleeping": False},
                ),
            )
            expected = "transfer failed" if failure == "worker_error" else "acknowledge"
        with pytest.raises(RuntimeError, match=expected):
            sleeping.result(timeout=10)
        with pytest.raises(RuntimeError, match="failed|closed"):
            client.enqueue_requests([DiffusionRequest(1, ["must reject"])])
        if failure in ("worker_exit", "shutdown"):
            with pytest.raises(RuntimeError, match=expected):
                client.get_memory_status()
        else:
            assert client.get_memory_status().state == "failed"


@pytest.mark.parametrize("policy", ["after_current", "priority", "DRAIN", None, True, 1])
def test_public_release_rejects_unsupported_policy_without_changing_admission(policy):
    visual_gen = VisualGen.__new__(VisualGen)
    with _client() as client:
        visual_gen.executor = client
        try:
            with pytest.raises(ValueError, match="Only the 'drain' sleep policy"):
                visual_gen.release(policy=policy, timeout=0)
            assert client.pending_requests.empty()
            assert visual_gen.get_memory_status().state == "running"
            assert client.enqueue_requests([DiffusionRequest(0, ["accepted"])]) == [0]
        finally:
            visual_gen.executor = None


@pytest.mark.parametrize("operation", ["sleep", "wake_up"])
@pytest.mark.parametrize("timeout", [-1, float("inf"), float("-inf"), float("nan"), True])
def test_invalid_timeout_does_not_change_state_or_submit_work(operation, timeout):
    with _client() as client:
        client._send_sleep_control = Mock(
            side_effect=AssertionError("Invalid timeout reached worker dispatch")
        )
        with pytest.raises(ValueError, match="finite and non-negative"):
            getattr(client, operation)(timeout=timeout)
        client._send_sleep_control.assert_not_called()
        assert client.pending_requests.empty()
        assert client.enqueue_requests([DiffusionRequest(0, ["accepted"])]) == [0]


@pytest.mark.parametrize("operation", ["sleep", "wake_up"])
def test_zero_timeout_before_submission_preserves_state(operation):
    with _client() as client:
        if operation == "wake_up":
            client._sleep_controller.transition("sleep", lambda _request: None)
        with pytest.raises(TimeoutError, match="before submission"):
            getattr(client, operation)(timeout=0)
        assert client.pending_requests.empty()
        assert client.is_sleeping == (operation == "wake_up")
        if operation == "sleep":
            assert client.enqueue_requests([DiffusionRequest(0, ["accepted"])]) == [0]
        else:
            with pytest.raises(RuntimeError, match="asleep"):
                client.enqueue_requests([DiffusionRequest(0, ["rejected"])])


def test_waiting_wake_timeout_does_not_interrupt_sleep():
    with _client() as client, _transition_threads(client, max_workers=2) as threads:
        sleeping = threads.submit(client.sleep)
        command = client.pending_requests.get(timeout=10)
        try:
            waking = threads.submit(client.wake_up, timeout=0.05)
            with pytest.raises(TimeoutError, match="waiting to start"):
                waking.result(timeout=10)
            assert waking.done()
            assert not sleeping.done()
            assert client.pending_requests.empty()
        finally:
            _ack(client, DiffusionResponse(command.request_id, output={"sleeping": True}))
        sleeping.result(timeout=10)
        assert client.is_sleeping
        waking = threads.submit(client.wake_up, timeout=10)
        command = client.pending_requests.get(timeout=10)
        _ack(client, DiffusionResponse(command.request_id, output={"sleeping": False}))
        waking.result(timeout=10)
        assert client.enqueue_requests([DiffusionRequest(0, ["accepted"])]) == [0]


def test_timeout_waiting_for_admission_keeps_instance_usable():
    with _client() as client, _transition_threads(client) as threads:
        client._sleep_controller._admission_lock.acquire()
        try:
            sleeping = threads.submit(client.sleep, timeout=0.05)
            with pytest.raises(TimeoutError, match="waiting to start"):
                sleeping.result(timeout=10)
            assert sleeping.done()
            assert client.pending_requests.empty()
        finally:
            client._sleep_controller._admission_lock.release()
        sleeping = threads.submit(client.sleep, timeout=10)
        command = client.pending_requests.get(timeout=10)
        _ack(client, DiffusionResponse(command.request_id, output={"sleeping": True}))
        sleeping.result(timeout=10)
        assert client.is_sleeping


@pytest.mark.parametrize("operation", ["sleep", "wake_up"])
@pytest.mark.parametrize("block_response_loop", [False, True])
def test_timeout_after_submission_rejects_generation_and_discards_late_reply(
    operation, block_response_loop
):
    with _client() as client, _transition_threads(client) as threads:
        if operation == "wake_up":
            client._sleep_controller.transition("sleep", lambda _request: None)
        abandoned, blocked, resume = Event(), Event(), Event()
        original_abandon = client.abandon_request_id

        async def observe_abandon(request_id):
            await original_abandon(request_id)
            abandoned.set()

        def block_loop():
            blocked.set()
            assert resume.wait(10), "Test did not release the response loop"

        client.abandon_request_id = observe_abandon
        if block_response_loop:
            client._event_loop.call_soon_threadsafe(block_loop)
        try:
            if block_response_loop:
                assert blocked.wait(10)
            transition = threads.submit(getattr(client, operation), timeout=0.05)
            command = client.pending_requests.get(timeout=10)
            with pytest.raises(TimeoutError, match="recreate the instance"):
                transition.result(timeout=10)
            assert transition.done()
            assert client.background_thread.is_alive()
            assert client.get_memory_status().state == "failed"
            with pytest.raises(RuntimeError, match="failed"):
                client.enqueue_requests([DiffusionRequest(0, ["rejected"])])
        finally:
            resume.set()
        assert abandoned.wait(10), "Timed-out control response was not abandoned"
        _ack(
            client,
            DiffusionResponse(command.request_id, output={"sleeping": operation == "sleep"}),
        )
        assert command.request_id not in client.completed_responses
        assert command.request_id not in client._abandoned_request_ids
        assert not client.is_sleeping
        assert client.get_memory_status().state == "failed"
        for retry in (client.sleep, client.wake_up):
            with pytest.raises(RuntimeError, match="failed"):
                retry(timeout=1)


def test_response_waiter_timeout_is_not_mistaken_for_polling():
    with _client() as client, _transition_threads(client) as threads:

        async def fail_wait(_request_id):
            raise TimeoutError("Response reader failed")

        client.await_responses = fail_wait
        sleeping = threads.submit(client.sleep)
        with pytest.raises(TimeoutError, match="Response reader failed"):
            sleeping.result(timeout=10)
        assert sleeping.done()
        with pytest.raises(RuntimeError, match="failed"):
            client.enqueue_requests([DiffusionRequest(0, ["rejected"])])


def test_control_dispatch_excludes_generation_stats_and_reports_send_error():
    with _client() as client:
        client.requests_ipc = Mock()
        client.pending_requests.put(_SleepControlRequest(-2, "sleep"))
        assert client.get_current_iteration_snapshot()["numQueuedRequests"] == 0
        client._process_requests()
        client.requests_ipc.put_nowait.assert_called_once_with(_SleepControlRequest(-2, "sleep"))
        assert client.get_iteration_stats() == []
        client.requests_ipc.put_nowait.side_effect = OSError("socket failed")
        client.pending_requests.put(_SleepControlRequest(-3, "sleep"))
        client._process_requests()
        response = client.await_responses_sync(-3, timeout=10)
        assert response.error_msg == "socket failed"


def test_default_client_path_does_not_require_lifecycle_controller():
    with _client() as client:
        client._sleep_controller = None
        assert client.enqueue_requests([DiffusionRequest(0, ["space"])]) == [0]
        assert client.pending_requests.get_nowait().request_id == 0
        assert not client.is_sleeping
        for operation in (client.sleep, client.wake_up, client.get_memory_status):
            with pytest.raises(RuntimeError, match="not enabled"):
                operation()


def test_rejected_generation_consumes_unsent_shared_memory_handles():
    with _client() as client:
        reference = SimpleNamespace(content=b"reference payload")
        request = DiffusionRequest(
            0,
            ["space"],
            params=SimpleNamespace(image_reference=[reference]),
        )
        request.refs_to_shm()
        assert request.ref_handles
        assert reference.content == b""
        client._sleep_controller.transition("sleep", lambda _request: None)
        with pytest.raises(RuntimeError, match="asleep"):
            client.enqueue_requests([request])
        assert request.ref_handles is None
        assert reference.content == b"reference payload"
        assert client.pending_requests.empty()


def test_sleep_cannot_deadlock_its_own_response_thread():
    with _client() as client:

        async def transition_on_response_thread():
            with pytest.raises(RuntimeError, match="response loop"):
                client.sleep()

        asyncio.run_coroutine_threadsafe(
            transition_on_response_thread(), client._event_loop
        ).result(10)
        assert client.enqueue_requests([DiffusionRequest(0, ["accepted"])]) == [0]


def test_public_methods_use_executor_and_handle_shutdown():
    visual_gen = VisualGen.__new__(VisualGen)
    with _client() as client, _transition_threads(client, max_workers=2) as threads:
        visual_gen.executor = client
        initial = visual_gen.get_memory_status()
        assert isinstance(initial, VisualGenMemoryStatus)
        assert initial.state == "running"
        for operation, expected_state in ((visual_gen.release, True), (visual_gen.resume, False)):
            transition = threads.submit(operation, timeout=10)
            command = client.pending_requests.get(timeout=10)
            try:
                status = threads.submit(visual_gen.get_memory_status).result(timeout=10)
                assert status.state == ("parking" if expected_state else "waking")
                assert initial.state == "running"
                with pytest.raises(RuntimeError, match="sleeping|waking"):
                    client.enqueue_requests([DiffusionRequest(0, ["rejected"])])
            finally:
                _ack(
                    client,
                    DiffusionResponse(command.request_id, output={"sleeping": expected_state}),
                )
            transition.result(timeout=10)
            assert visual_gen.get_memory_status().state == (
                "parked" if expected_state else "running"
            )
            assert status.state == ("parking" if expected_state else "waking")
        assert client.enqueue_requests([DiffusionRequest(1, ["accepted"])]) == [1]
        visual_gen.executor = None
    for operation in (visual_gen.release, visual_gen.resume, visual_gen.get_memory_status):
        with pytest.raises(RuntimeError, match="shut down"):
            operation()


@pytest.mark.parametrize("state", ["awake", "asleep", "closed", "unknown", "RUNNING", None])
def test_memory_status_rejects_invalid_state_names(state):
    with pytest.raises(ValidationError, match="state"):
        VisualGenMemoryStatus(state=state)
    assert VisualGenMemoryStatus(state="running").model_dump() == {"state": "running"}


@pytest.mark.parametrize("failure", ["closed", "response_loop"])
def test_memory_status_does_not_report_running_after_shutdown_or_response_loop_exit(failure):
    with _client() as client:
        assert client.get_memory_status().state == "running"
        if failure == "closed":
            client._sleep_controller.close()
            expected = "shut down"
        else:
            client._event_loop.call_soon_threadsafe(client._event_loop.stop)
            client.background_thread.join(timeout=10)
            assert not client.background_thread.is_alive()
            expected = "response loop is not running"
        with pytest.raises(RuntimeError, match=expected):
            client.get_memory_status()


class _PausedExecutorRead(VisualGen):
    """Hold the first executor read while another thread completes shutdown."""

    def __init__(self, executor, read, shutdown_finished):
        self._pause_executor_read = False
        self.executor = executor
        self._executor_read = read
        self._shutdown_finished = shutdown_finished
        self._pause_executor_read = True

    def __getattribute__(self, name):
        value = super().__getattribute__(name)
        if name == "executor" and self._pause_executor_read:
            self._pause_executor_read = False
            self._executor_read.set()
            assert self._shutdown_finished.wait(10), "Concurrent shutdown did not finish"
        return value


@pytest.mark.parametrize("operation", ["release", "resume", "get_memory_status"])
def test_public_lifecycle_keeps_executor_reference_during_shutdown(operation):
    read, shutdown_finished = Event(), Event()
    executor = SimpleNamespace(
        sleep=Mock(side_effect=RuntimeError("VisualGen has been shut down")),
        wake_up=Mock(side_effect=RuntimeError("VisualGen has been shut down")),
        get_memory_status=Mock(side_effect=RuntimeError("VisualGen has been shut down")),
    )
    executor.shutdown = Mock()
    visual_gen = _PausedExecutorRead(executor, read, shutdown_finished)

    def call_public_api():
        return getattr(visual_gen, operation)()

    with ThreadPoolExecutor(max_workers=1) as threads:
        pending = threads.submit(call_public_api)
        try:
            assert read.wait(10), "Public API did not read its executor"
            visual_gen.shutdown()
        finally:
            shutdown_finished.set()
        with pytest.raises(RuntimeError, match="has been shut down"):
            pending.result(timeout=10)
        kwargs = {} if operation == "get_memory_status" else {"timeout": None}
        if operation == "release":
            kwargs["policy"] = "drain"
        worker_operation = {
            "release": "sleep",
            "resume": "wake_up",
            "get_memory_status": "get_memory_status",
        }[operation]
        getattr(executor, worker_operation).assert_called_once_with(**kwargs)
    executor.shutdown.assert_called_once_with()
    assert visual_gen.executor is None

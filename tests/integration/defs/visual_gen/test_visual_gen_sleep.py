# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test public sleep/wake with an H3 checkpoint and queued requests."""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest
import torch
from test_common.llm_data import get_checkpoint

from tensorrt_llm import VisualGen, VisualGenArgs, VisualGenParams


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_h3_public_sleep_drains_requests_and_preserves_outputs():
    if torch.cuda.get_device_properties(0).total_memory < 140 * 1024**3:
        pytest.skip("H3 FP8 sleep test requires a GPU with at least 140 GiB")
    checkpoint = os.environ.get("MINIMAX_H3_CHECKPOINT") or get_checkpoint("MiniMax-H3")
    assert Path(checkpoint).is_dir(), f"MiniMax-H3 checkpoint not found: {checkpoint}"
    args = VisualGenArgs(
        sleep_restore_mode="PINNED",
        quant_config={"quant_algo": "FP8_BLOCK_SCALES", "dynamic": True},
        cuda_graph_config={"enable": False},
        torch_compile_config={"enable": False},
        compilation_config={"skip_warmup": True},
        attention_config={"backend": "VANILLA"},
    )
    params = VisualGenParams(
        seed=42, height=128, width=128, num_frames=124, frame_rate=24.0, num_inference_steps=4
    )
    prompt = "A spacecraft passes an icy moon, with a deep engine rumble."
    with VisualGen(checkpoint, args=args) as visual_gen:
        expected = visual_gen.generate(prompt, params)
        reference_video = expected.video.cpu().clone()
        reference_audio = expected.audio.cpu().clone()
        assert reference_video.shape == (1, 124, 128, 128, 3)
        assert reference_video.float().std() > 0
        assert reference_audio.shape[:2] == (1, 2)
        assert torch.isfinite(reference_audio).all()
        assert reference_audio.abs().max() > 0
        del expected
        free_before = torch.cuda.mem_get_info()[0]

        # Observe admission closure without injecting delays into model execution.
        entered = Event()
        send_control = visual_gen.executor._send_sleep_control

        def send_and_signal(request, *, deadline=None):
            entered.set()
            return send_control(request, deadline=deadline)

        visual_gen.executor._send_sleep_control = send_and_signal
        pending = [visual_gen.generate_async(prompt, params) for _ in range(2)]
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=1) as threads:
            sleeping = threads.submit(visual_gen.release, policy="drain", timeout=240)
            try:
                assert entered.wait(10), "Sleep did not close admission"
                with pytest.raises(RuntimeError, match="sleeping|asleep"):
                    visual_gen.generate_async(prompt, params)
                sleeping.result(timeout=240)
            finally:
                if not sleeping.done():
                    visual_gen.shutdown()
                elif visual_gen.executor is not None:
                    visual_gen.executor._send_sleep_control = send_control
        sleep_seconds = time.perf_counter() - start
        assert visual_gen.get_memory_status().state == "parked"
        released_gib = (torch.cuda.mem_get_info()[0] - free_before) / 1024**3
        assert released_gib > 50, f"Only {released_gib:.2f} GiB was released"
        for result in pending:
            output = result.result(timeout=10)
            torch.testing.assert_close(output.video.cpu(), reference_video, rtol=0, atol=0)
            torch.testing.assert_close(output.audio.cpu(), reference_audio, rtol=0, atol=0)
        with pytest.raises(RuntimeError, match="asleep"):
            visual_gen.generate_async(prompt, params)

        start = time.perf_counter()
        visual_gen.resume(timeout=60)
        wake_seconds = time.perf_counter() - start
        assert visual_gen.get_memory_status().state == "running"
        output = visual_gen.generate(prompt, params)
        torch.testing.assert_close(output.video.cpu(), reference_video, rtol=0, atol=0)
        torch.testing.assert_close(output.audio.cpu(), reference_audio, rtol=0, atol=0)
        print(
            {
                "sleep_including_drain_s": sleep_seconds,
                "wake_s": wake_seconds,
                "released_gib": released_gib,
            }
        )

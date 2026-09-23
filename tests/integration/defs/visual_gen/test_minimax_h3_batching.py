# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Local/QA checkpoint checks; intentionally not registered in CI."""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from tensorrt_llm import VisualGen, VisualGenArgs, VisualGenParams

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module")
def engine() -> Iterator[VisualGen]:
    checkpoint = os.environ.get("MINIMAX_H3_CHECKPOINT")
    assert checkpoint and Path(checkpoint).is_dir(), "Set MINIMAX_H3_CHECKPOINT to local weights"
    args = VisualGenArgs(
        model=checkpoint,
        quant_config={"quant_algo": "FP8_BLOCK_SCALES", "dynamic": True},
        attention_config={"backend": "VANILLA"},
        cuda_graph_config={"enable": False},
        compilation_config={"skip_warmup": True},
        torch_compile_config={"enable": False},
    )
    visual_gen = VisualGen(model=checkpoint, args=args)
    try:
        yield visual_gen
    finally:
        visual_gen.shutdown()


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_minimax_h3_public_api_static_batch(engine: VisualGen, batch_size: int) -> None:
    prompts = [
        "A spacecraft passes an orbital station, with a deep engine rumble.",
        "A bioluminescent alien walks across a rocky moon beneath two enormous planets, "
        "with crunching footsteps and eerie chirping calls.",
    ] * batch_size
    prompts = prompts[:batch_size]
    outputs = engine.generate(
        inputs=prompts,
        params=VisualGenParams(
            height=128,
            width=128,
            num_frames=124,
            frame_rate=24,
            num_inference_steps=2,
            seed=42,
        ),
    )
    assert isinstance(outputs, list) and len(outputs) == batch_size
    assert len({output.request_id for output in outputs}) == 1
    for output in outputs:
        assert output.error is None, output.error
        assert output.video.shape == (124, 128, 128, 3)
        assert output.video.dtype == torch.uint8
        assert output.video.float().std() > 0
        assert output.audio.ndim == 2 and output.audio.shape[0] == 2
        assert torch.isfinite(output.audio).all() and output.audio.abs().max() > 0
        assert output.frame_rate == 24 and output.audio_sample_rate == 32000

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

"""Local/QA GPU regressions for experimental MiniMax-H3 static batching."""

import pytest
import torch
from test_minimax_h3_transformer import _initialize_weights, _make_model_config, _model_inputs

from tensorrt_llm._torch.visual_gen.models.minimax_h3 import transformer_minimax_h3 as h3

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_masked_batch_matches_independent_transformer_samples() -> None:
    torch.manual_seed(11)
    model = (
        h3.MiniMaxH3Transformer3DModel(_make_model_config(num_layers=2, num_refiner_layers=2))
        .to("cuda")
        .eval()
    )
    _initialize_weights(model, scale=0.1)
    short = _model_inputs("cuda")
    long = _model_inputs("cuda")
    long.update(
        encoder_hidden_states=torch.randn(1, 3, 5, device="cuda"),
        token_tags=torch.tensor([1, 1, 1, 0, 2, 0], device="cuda"),
        timestep_indices=torch.tensor([0, 0, 0, 1, 1, 0], device="cuda"),
        position_ids=torch.tensor(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [3, 1, 0], [4, 0, 1]], device="cuda"
        ),
        text_indices=torch.tensor([0, 1, 2], device="cuda"),
        video_indices=torch.tensor([3, 5], device="cuda"),
        audio_indices=torch.tensor([4], device="cuda"),
    )
    mask = torch.tensor([[False, False, True], [True, True, True]], device="cuda")
    padded_short = torch.cat(
        (torch.full((1, 2, 5), 100.0, device="cuda"), short["encoder_hidden_states"]), dim=1
    )
    embeddings = torch.cat((padded_short, long["encoder_hidden_states"]))
    with torch.inference_mode():
        expected = [model(**inputs) for inputs in (short, long)]
        context = model.prepare_static_context(embeddings, long["position_ids"], mask)
        batched = long | {
            "hidden_states": torch.cat((short["hidden_states"], long["hidden_states"])),
            "audio_hidden_states": torch.cat(
                (short["audio_hidden_states"], long["audio_hidden_states"])
            ),
            "encoder_hidden_states": None,
            "position_ids": None,
            "static_context": context,
        }
        actual = model(**batched)
        for index, reference in enumerate(expected):
            torch.testing.assert_close(
                actual.sample[index : index + 1], reference.sample, rtol=2e-2, atol=2e-3
            )
            torch.testing.assert_close(
                actual.audio_sample[index : index + 1], reference.audio_sample, rtol=2e-2, atol=2e-3
            )

        # Perturb only the second request and the first request's padded tokens.
        embeddings[0, :2] *= -10
        embeddings[1] *= -10
        batched["static_context"] = model.prepare_static_context(
            embeddings, long["position_ids"], mask
        )
        changed = model(**batched)
        torch.testing.assert_close(changed.sample[0], actual.sample[0], rtol=0, atol=0)
        torch.testing.assert_close(changed.audio_sample[0], actual.audio_sample[0], rtol=0, atol=0)
        assert not torch.equal(changed.sample[1], actual.sample[1])


def test_padded_text_rejects_backend_without_mask_support() -> None:
    model = h3.MiniMaxH3Transformer3DModel(_make_model_config()).to("cuda")
    model._supports_key_padding_mask = False
    with pytest.raises(NotImplementedError, match="use VANILLA"):
        model.prepare_static_context(
            torch.ones(2, 2, 5, device="cuda"),
            torch.zeros(4, 3, device="cuda"),
            torch.tensor([[False, True], [True, True]], device="cuda"),
        )

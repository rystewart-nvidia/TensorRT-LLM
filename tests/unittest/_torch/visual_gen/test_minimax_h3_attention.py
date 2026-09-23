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

"""Local/QA tests of H3 offset attention and its SDPA fallback contract."""

import pytest
import torch

from tensorrt_llm._torch.visual_gen.attention_backend.vanilla import VanillaAttention
from tensorrt_llm._torch.visual_gen.models.minimax_h3 import attention as h3


@pytest.mark.parametrize(
    "mask,expected",
    [
        (None, False),
        ([[False, True], [True, True]], True),
        ([[True, False]], False),
        ([[True, False, True]], False),
        ([[False, False]], False),
        ([[False, True], [False, False]], False),
        ([[]], False),
        ([[1, 1]], False),
    ],
)
def test_suffix_mask_contract(mask, expected: bool) -> None:
    tensor = None if mask is None else torch.tensor(mask)
    assert h3.is_nonempty_suffix_mask(tensor) == expected


@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("kind", ["none", "suffix", "holes", "empty"])
def test_cpu_fallback_preserves_sdpa(batch: int, kind: str) -> None:
    q, k, v = [torch.randn(batch, 2, 7, 128) for _ in range(3)]
    mask = None
    if kind != "none":
        mask = torch.ones(batch, 7, dtype=torch.bool)
        mask[:, :2] = False
        if kind == "holes":
            mask[:, 4] = False
        elif kind == "empty":
            mask[0] = False
    args = dict(num_heads=2, head_dim=128)
    expected = VanillaAttention(**args).forward(q, k, v, key_padding_mask=mask)
    actual = h3.MiniMaxH3BatchedAttention(**args).forward(
        q, k, v, key_padding_mask=mask, suffix_mask=h3.is_nonempty_suffix_mask(mask)
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("batch", [2, 4])
def test_offset_attention_matches_sdpa_and_isolates_samples(batch: int, monkeypatch) -> None:
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("H3 offset attention requires SM100")
    assert h3.flash_attn4._flash_attn_fwd is not None
    torch.manual_seed(7)
    q, k, v = [
        torch.randn(batch, 67, 4, 128, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
        for _ in range(3)
    ]
    lengths = torch.arange(batch, device="cuda") * 11 + 1
    mask = torch.arange(67, device="cuda")[None] >= 67 - lengths[:, None]
    args = dict(num_heads=4, head_dim=128)
    backend = h3.MiniMaxH3BatchedAttention(**args)
    calls = []
    original = h3.flash_attn4._flash_attn_fwd

    def record(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(h3.flash_attn4, "_flash_attn_fwd", record)
    expected = VanillaAttention(**args).forward(q, k, v, key_padding_mask=mask)
    actual = backend.forward(q, k, v, key_padding_mask=mask, suffix_mask=True)
    assert len(calls) == 1
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    k[1:] *= -5
    v[1:] *= -5
    k[0, :, :-1] = 100
    v[0, :, :-1] = 100
    changed = backend.forward(q, k, v, key_padding_mask=mask, suffix_mask=True)
    torch.testing.assert_close(changed[0], actual[0], atol=0, rtol=0)
    assert not torch.equal(changed[1], actual[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("case", ["singleton", "unmasked", "holes", "missing_fa4", "arch", "dtype"])
def test_gpu_fallback_does_not_call_fa4(case: str, monkeypatch) -> None:
    batch = 1 if case == "singleton" else 2
    dtype = torch.float32 if case == "dtype" else torch.bfloat16
    q, k, v = [torch.randn(batch, 4, 7, 128, device="cuda", dtype=dtype) for _ in range(3)]
    mask = torch.ones(batch, 7, device="cuda", dtype=torch.bool)
    mask[:, :2] = False
    if case == "unmasked":
        mask = None
    elif case == "holes":
        mask[:, 4] = False
    elif case == "arch":
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (9, 0))

    def unexpected(*args, **kwargs):
        raise AssertionError("Fallback must not invoke FA4")

    monkeypatch.setattr(
        h3.flash_attn4, "_flash_attn_fwd", None if case == "missing_fa4" else unexpected
    )
    args = dict(num_heads=4, head_dim=128)
    expected = VanillaAttention(**args).forward(q, k, v, key_padding_mask=mask)
    actual = h3.MiniMaxH3BatchedAttention(**args).forward(
        q, k, v, key_padding_mask=mask, suffix_mask=h3.is_nonempty_suffix_mask(mask)
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

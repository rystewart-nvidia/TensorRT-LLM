# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local/QA safety, layout, and compile tests for the fused H3 prototype."""

import pytest
import torch

from h3_fused_qk_rope import fused_qk_norm_rope


def inputs():
    torch.manual_seed(19)
    qkv = torch.randn(2, 13, 8 * 128, device="cuda", dtype=torch.bfloat16)
    q, k, _ = qkv.split((4 * 128, 2 * 128, 2 * 128), dim=-1)
    angles = torch.randn(13, 48, device="cuda").repeat(1, 2)
    return (
        q.view(2, 13, 4, 128),
        k.view(2, 13, 2, 128),
        torch.randn(128, device="cuda", dtype=torch.bfloat16),
        torch.randn(128, device="cuda", dtype=torch.bfloat16),
        angles.cos(),
        angles.sin(),
        1e-6,
        1e-6,
    )


def test_compile_gqa_and_tail():
    args = inputs()
    with torch.inference_mode():
        eager = fused_qk_norm_rope(*args)
        compiled = torch.compile(fused_qk_norm_rope, fullgraph=True)(*args)
    for index, (a, b) in enumerate(zip(eager, compiled)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        x, w = args[index], args[2 + index]
        expected = x.float() * torch.rsqrt(
            x.float().square().mean(-1, keepdim=True) + 1e-6
        )
        expected = (expected * w.float()).to(torch.bfloat16)
        torch.testing.assert_close(
            a[..., 96:], expected[..., 96:], rtol=0.01, atol=0.01
        )


@pytest.mark.parametrize("case", ["weight", "rope_length", "rotary_odd", "dtype"])
def test_invalid_inputs(case):
    args = list(inputs())
    if case == "weight":
        args[2] = args[2][:-1]
    elif case == "rope_length":
        args[4] = args[4][:-1]
    elif case == "rotary_odd":
        args[4], args[5] = args[4][:, :-1].contiguous(), args[5][:, :-1].contiguous()
    else:
        args[0] = args[0].float()
    with pytest.raises(ValueError):
        fused_qk_norm_rope(*args)


def test_inputs_unchanged_and_requests_isolated():
    args = inputs()
    before = [x.clone() for x in args[:6]]
    with torch.inference_mode():
        reference = fused_qk_norm_rope(*args)
        changed = args[0].clone()
        changed[1, 1, 1, 1] += 10
        actual = fused_qk_norm_rope(changed, *args[1:])
    for original, saved in zip(args[:6], before):
        torch.testing.assert_close(original, saved, rtol=0, atol=0)
    torch.testing.assert_close(actual[0][0], reference[0][0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], reference[1], rtol=0, atol=0)
    assert not torch.equal(actual[0][1], reference[0][1])

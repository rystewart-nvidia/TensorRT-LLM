# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernel-level correctness and warmed timing against compiled H3 operations."""

import json
from pathlib import Path

import torch
from tensorrt_llm._torch.visual_gen.models.minimax_h3.transformer_minimax_h3 import (
    apply_minimax_h3_rotary_emb,
)

from h3_fused_qk_rope import fused_qk_norm_rope
from profile_h3_throughput import measure


def reference(q, k, qw, kw, cos, sin, qeps, keps):
    values = []
    for x, w, eps in ((q, qw, qeps), (k, kw, keps)):
        f = x.float()
        normalized = (f * torch.rsqrt(f.square().mean(-1, keepdim=True) + eps)).to(
            x.dtype
        )
        values.append(apply_minimax_h3_rotary_emb(w * normalized, cos, sin))
    return tuple(values)


def main():
    torch.manual_seed(17)
    root = Path("/work/outputs/b200-efficiency")
    root.mkdir(parents=True, exist_ok=True)
    compiled = torch.compile(reference, dynamic=False)
    rows = []
    for b, s, h, d, rotary in (
        (1, 13, 3, 128, 96),
        (2, 43, 8, 128, 96),
        (1, 43, 4, 64, 32),
        (2, 43, 4, 128, 128),
        (1, 19000, 56, 128, 96),
        (2, 19000, 56, 128, 96),
    ):
        qkv = torch.randn(b, s, 3 * h * d, device="cuda", dtype=torch.bfloat16)
        q, k, _ = [x.view(b, s, h, d) for x in qkv.chunk(3, dim=-1)]
        qw, kw = [torch.randn(d, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        angles = torch.randn(s, rotary // 2, device="cuda").repeat(1, 2)
        cos, sin = angles.cos(), angles.sin()
        args = (q, k, qw, kw, cos, sin, 1e-6, 1e-6)
        actual = fused_qk_norm_rope(*args)
        eager = reference(*args)
        expected = compiled(*args)
        errors = []
        for val, ref, uncompiled in zip(actual, expected, eager):
            torch.testing.assert_close(val, ref, rtol=0.02, atol=0.02)
            errors.append(
                {
                    "compiled_max_abs": float((val.float() - ref.float()).abs().max()),
                    "compiled_relative_l2": float(
                        torch.linalg.vector_norm(val.float() - ref.float())
                        / torch.linalg.vector_norm(ref.float())
                    ),
                    "eager_max_abs": float(
                        (val.float() - uncompiled.float()).abs().max()
                    ),
                }
            )
        if b == 2 and s == 43:
            changed = q.clone()
            changed[1] += 10
            isolated = fused_qk_norm_rope(changed, *args[1:])
            torch.testing.assert_close(isolated[0][0], actual[0][0], rtol=0, atol=0)
            torch.testing.assert_close(isolated[1], actual[1], rtol=0, atol=0)
        times = {}
        for name, fn in [("compiled", compiled), ("fused", fused_qk_norm_rope)]:
            samples = measure(lambda fn=fn, args=args: fn(*args), repeats=10)
            times[name] = {
                "mean_ms": sum(samples) / len(samples),
                "samples_ms": samples,
            }
        rows.append(
            {
                "shape": [b, s, h, d],
                "rotary_dim": rotary,
                "errors": errors,
                "timings": times,
            }
        )
        (root / "qk-rope-probe.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(rows[-1]), flush=True)
        del qkv, q, k, actual, eager, expected, args


if __name__ == "__main__":
    with torch.inference_mode():
        main()

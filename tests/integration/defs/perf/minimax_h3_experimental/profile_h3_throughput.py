# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local/QA H3 throughput probes; run only in an allocated GPU container."""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from flash_attn.cute.interface import _flash_attn_fwd
from tensorrt_llm._torch.visual_gen.attention_backend.flash_attn4 import (
    FlashAttn4Attention,
)

ROOT = Path("/work/outputs/b200-throughput")


def compact_attention(backend, q, k, v, mask=None):
    # Gather after RoPE. Only K/V order changes, never sample boundaries or Q order.
    if mask is not None:
        indices = torch.argsort(
            mask.to(torch.int32), dim=1, descending=True, stable=True
        )
        indices = indices[:, :, None, None].expand(-1, -1, k.shape[2], k.shape[3])
        k, v = k.gather(1, indices), v.gather(1, indices)
        lengths = mask.sum(dim=1, dtype=torch.int32)
        output, _ = backend._fwd(q, k, v, False, seqused_k=lengths)
        return output
    return backend.forward(q, k, v)


@torch.compiler.disable
def offset_attention(backend, q, k, v, mask):
    """Left-padding-only probe: each sample has one contiguous valid suffix."""
    batch, sequence = mask.shape
    lengths = mask.sum(dim=1, dtype=torch.int32)
    offsets = torch.arange(batch + 1, device=q.device, dtype=torch.int32) * sequence
    # Intervals can include the next sample's padding; seqused_k excludes it.
    key_offsets = torch.cat((offsets[:-1] + sequence - lengths, offsets[-1:]))
    output, *_ = _flash_attn_fwd(
        q,
        k.flatten(0, 1),
        v.flatten(0, 1),
        cu_seqlens_k=key_offsets,
        seqused_k=lengths,
        max_seqlen_q=sequence,
        max_seqlen_k=sequence,
        softmax_scale=backend.scale,
        causal=False,
        return_lse=False,
        num_splits=0,
    )
    return output.view_as(q)


def offset_microbench():
    ROOT.mkdir(parents=True, exist_ok=True)
    backend = FlashAttn4Attention(num_heads=56, head_dim=128, dtype=torch.bfloat16)
    rows = []
    for batch, sequence in ((1, 43), (2, 43), (4, 43), (2, 19000), (4, 19000)):
        q, k, v = [
            torch.randn(batch, sequence, 56, 128, device="cuda", dtype=torch.bfloat16)
            for _ in range(3)
        ]
        mask = (
            torch.arange(sequence, device="cuda")[None]
            >= (torch.arange(batch, device="cuda") * 3 + 5)[:, None]
        )
        expected = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=mask[:, None, None, :],
        ).transpose(1, 2)
        for name, fn in [("compact", compact_attention), ("offset", offset_attention)]:
            actual = fn(backend, q, k, v, mask)
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
            changed_k, changed_v = k.clone(), v.clone()
            changed_k[~mask], changed_v[~mask] = 100, -100
            if batch > 1:
                changed_k[-1], changed_v[-1] = -100, 100
            changed = fn(backend, q, changed_k, changed_v, mask)
            stop = batch - 1 if batch > 1 else 1
            torch.testing.assert_close(changed[:stop], actual[:stop], rtol=0, atol=0)
            samples = measure(partial(fn, backend, q, k, v, mask))
            row = {
                "batch": batch,
                "sequence": sequence,
                "kernel": name,
                "mean_ms": sum(samples) / len(samples),
                "milliseconds": samples,
                "max_abs": float((actual.float() - expected.float()).abs().max()),
                "isolation_passed": True,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            (ROOT / "offset-microbench.json").write_text(
                json.dumps(rows, indent=2) + "\n"
            )
            del actual, changed_k, changed_v, changed
        del q, k, v, mask, expected
    for batch in (1, 2, 4):
        q, k, v = [
            torch.randn(batch, 19000, 56, 128, device="cuda", dtype=torch.bfloat16)
            for _ in range(3)
        ]
        lengths = torch.full((batch,), 19000, device="cuda", dtype=torch.int32)

        def sdpa(q=q, k=k, v=v):
            return F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            ).transpose(1, 2)

        def fa4_lengths(q=q, k=k, v=v, lengths=lengths):
            return backend._fwd(q, k, v, False, seqused_k=lengths)[0]

        expected = sdpa()
        for name, fn in [("sdpa_unmasked", sdpa), ("fa4_full_lengths", fa4_lengths)]:
            actual = fn()
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
            samples = measure(fn)
            row = {
                "batch": batch,
                "sequence": 19000,
                "kernel": name,
                "mean_ms": sum(samples) / len(samples),
                "milliseconds": samples,
                "max_abs": float((actual.float() - expected.float()).abs().max()),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            (ROOT / "offset-microbench.json").write_text(
                json.dumps(rows, indent=2) + "\n"
            )
            del actual
        del q, k, v, lengths, expected


def measure(fn, repeats=5):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        output = fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
        del output
    return samples


def microbench():
    ROOT.mkdir(parents=True, exist_ok=True)
    backend = FlashAttn4Attention(num_heads=56, head_dim=128, dtype=torch.bfloat16)
    report = []
    torch.manual_seed(42)
    for batch in (1, 2, 4):
        q, k, v = [
            torch.randn(batch, 19000, 56, 128, device="cuda", dtype=torch.bfloat16)
            for _ in range(3)
        ]
        mask = torch.ones(batch, 19000, device="cuda", dtype=torch.bool)
        mask[-1, :20] = False
        for padded in (False, True):
            current_mask = mask if padded else None

            def sdpa(q=q, k=k, v=v, current_mask=current_mask):
                return F.scaled_dot_product_attention(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    attn_mask=None
                    if current_mask is None
                    else current_mask[:, None, None, :],
                ).transpose(1, 2)

            def fa4(q=q, k=k, v=v, current_mask=current_mask):
                return compact_attention(backend, q, k, v, current_mask)

            reference = sdpa()
            for name, fn in [("sdpa", sdpa), ("fa4_compact", fa4)]:
                samples = measure(fn)
                actual = fn()
                delta = (actual.float() - reference.float()).abs()
                row = {
                    "batch": batch,
                    "padded": padded,
                    "kernel": name,
                    "milliseconds": samples,
                    "mean_ms": sum(samples) / len(samples),
                    "max_abs": float(delta.max()),
                    "mean_abs": float(delta.mean()),
                }
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as prof:
                    fn()
                    torch.cuda.synchronize()
                row["operators"] = prof.key_averages().table(
                    sort_by="self_cuda_time_total", row_limit=12
                )
                prof.export_chrome_trace(
                    str(ROOT / f"attention-b{batch}-pad{padded}-{name}.json")
                )
                report.append(row)
                print(json.dumps(row), flush=True)
                (ROOT / "attention-microbench.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
                del actual, delta
            del reference
        del q, k, v, mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["microbench", "offset"], default="microbench"
    )
    args = parser.parse_args()
    with torch.inference_mode():
        if args.mode == "offset":
            offset_microbench()
        else:
            microbench()

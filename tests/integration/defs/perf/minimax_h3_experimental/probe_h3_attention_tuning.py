# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local B200 QA sweep; imports TRT-LLM before FA4 compatibility-sensitive modules."""

import argparse
import json
import random
import statistics
import time
import traceback
from pathlib import Path

import torch
from tensorrt_llm._torch.visual_gen.attention_backend.flash_attn4 import (
    FlashAttn4Attention,
    _flash_attn_fwd,
)


def main(sequence: int, target: Path, extra_tiles: bool = False) -> None:
    from flash_attn.cute import utils

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    torch.manual_seed(123)
    backend = FlashAttn4Attention(num_heads=56, head_dim=128, dtype=torch.bfloat16)
    configs = [(128, n, two) for two in (True, False) for n in (64, 128, 192, 256)]
    if extra_tiles:
        configs = [(128, 128, True)] + [
            (64, n, two) for two in (True, False) for n in (64, 128)
        ]
    report = {"sequence": sequence, "heads": 56, "head_dim": 128, "runs": []}
    original = utils._fa_disable_2cta_enabled
    for batch in (1, 2, 4):
        q, k, v = [
            torch.randn(batch, sequence, 56, 128, device="cuda", dtype=torch.bfloat16)
            for _ in range(3)
        ]
        padding = torch.tensor(([0, 20] * 2)[:batch], device="cuda", dtype=torch.int32)
        lengths = sequence - padding
        offsets = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * sequence
        offsets = torch.cat((offsets[:-1] + padding, offsets[-1:]))
        mask = torch.arange(sequence, device="cuda")[None] >= padding[:, None]
        reference = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=mask[:, None, None, :],
        ).transpose(1, 2)

        def call(
            config: tuple[int, int, bool],
            q=q,
            k=k,
            v=v,
            offsets=offsets,
            lengths=lengths,
        ) -> torch.Tensor:
            m, n, two = config
            utils._fa_disable_2cta_enabled = not two
            output, *_ = _flash_attn_fwd(
                q,
                k.flatten(0, 1),
                v.flatten(0, 1),
                cu_seqlens_k=offsets,
                seqused_k=lengths,
                max_seqlen_q=sequence,
                max_seqlen_k=sequence,
                softmax_scale=backend.scale,
                causal=False,
                return_lse=False,
                num_splits=0,
                tile_mn=(m, n),
            )
            return output

        passing = []
        for config in configs:
            row = {"batch": batch, "tile": list(config[:2]), "two_cta": config[2]}
            begin = time.perf_counter()
            try:
                output = call(config)
                torch.cuda.synchronize()
                torch.testing.assert_close(output, reference, rtol=0.02, atol=0.02)
                delta = output.float() - reference.float()
                row.update(
                    passed=True,
                    max_abs=float(delta.abs().max()),
                    relative_l2=float(delta.norm() / reference.float().norm()),
                    compile_and_check_seconds=time.perf_counter() - begin,
                )
                for _ in range(3):
                    call(config)
                passing.append((config, row))
                del output, delta
            except (AssertionError, RuntimeError, ValueError) as error:
                row.update(
                    passed=False,
                    error=f"{type(error).__name__}: {error}",
                    traceback=traceback.format_exc(),
                )
                print(json.dumps(row), flush=True)
            report["runs"].append(row)
            target.write_text(json.dumps(report, indent=2) + "\n")

        rng = random.Random(42)
        for _ in range(20):
            order = list(passing)
            rng.shuffle(order)
            for config, row in order:
                torch.cuda.synchronize()
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                begin = time.perf_counter()
                start.record()
                output = call(config)
                end.record()
                end.synchronize()
                row.setdefault("wall_ms", []).append(
                    1000 * (time.perf_counter() - begin)
                )
                row.setdefault("cuda_ms", []).append(start.elapsed_time(end))
                del output
        for _, row in passing:
            row["median_cuda_ms"] = statistics.median(row["cuda_ms"])
            row["median_wall_ms"] = statistics.median(row["wall_ms"])
            print(json.dumps(row), flush=True)
        target.write_text(json.dumps(report, indent=2) + "\n")
        del q, k, v, reference, mask
    utils._fa_disable_2cta_enabled = original


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=19000)
    parser.add_argument("--extra-tiles", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/work/outputs/b200-attention-tuning/microbench.json"),
    )
    args = parser.parse_args()
    with torch.inference_mode():
        main(args.sequence, args.output, args.extra_tiles)

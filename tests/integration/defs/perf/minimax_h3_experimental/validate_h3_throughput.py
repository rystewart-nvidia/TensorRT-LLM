# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare sampled full-resolution outputs from the paired throughput runs."""

import argparse
import json
from pathlib import Path

import lpips
import torch
from PIL import Image
from tensorrt_llm._torch.visual_gen.attention_backend.flash_attn4 import (
    FlashAttn4Attention,
)

from diagnose_batching_audio import distance
from profile_h3_throughput import compact_attention

ROOT = Path("/work/outputs/b200-throughput-space")


def attention_checks():
    backend = FlashAttn4Attention(num_heads=4, head_dim=128, dtype=torch.bfloat16)
    torch.manual_seed(123)
    count = 0
    for batch in (1, 2, 4):
        q, k, v = [
            torch.randn(batch, 43, 4, 128, device="cuda", dtype=torch.bfloat16)
            for _ in range(3)
        ]
        for kind in ("left", "right", "holes", "all_valid"):
            mask = torch.ones(batch, 43, device="cuda", dtype=torch.bool)
            if kind == "left":
                mask[:, :7] = False
            elif kind == "right":
                mask[:, -7:] = False
            elif kind == "holes":
                mask[:, 2::3] = False
            if batch > 1:
                mask[-1] = True
            expected = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                attn_mask=mask[:, None, None, :],
            ).transpose(1, 2)
            actual = compact_attention(backend, q, k, v, mask)
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
            changed_k, changed_v = k.clone(), v.clone()
            changed_k[~mask], changed_v[~mask] = 100, -100
            if batch > 1:
                changed_k[-1], changed_v[-1] = -100, 100
            changed = compact_attention(backend, q, changed_k, changed_v, mask)
            stable = batch - 1 if batch > 1 else 1
            torch.testing.assert_close(
                changed[:stable], actual[:stable], rtol=0, atol=0
            )
            count += 1
    return count


def video_distance(metric, a, b):
    scores = []
    for start in range(0, a.shape[0], 4):
        reference = a[start : start + 4].permute(0, 3, 1, 2).cuda().float() / 127.5 - 1
        actual = b[start : start + 4].permute(0, 3, 1, 2).cuda().float() / 127.5 - 1
        scores.extend(metric(actual, reference).flatten().tolist())
    return sum(scores) / len(scores)


def main(backend):
    report = {
        "attention_checks_passed": attention_checks(),
        "samples": [],
        "thresholds": {"video_lpips": 0.03, "audio_log_stft": 0.02},
        "video_sampling": "31 frames per sample, every fourth frame of 124",
    }
    metric = lpips.LPIPS(net="alex").eval().cuda()
    for case in ("b1", "b2_mixed"):
        baseline = torch.load(ROOT / f"baseline-{case}-quality.pt", weights_only=True)
        candidate = torch.load(ROOT / f"{backend}-{case}-quality.pt", weights_only=True)
        for index in range(baseline["video"].shape[0]):
            a, b = baseline["video"][index], candidate["video"][index]
            row = {
                "case": case,
                "sample": index,
                "lpips": video_distance(metric, a, b),
                "audio_log_stft": distance(
                    candidate["audio"][index], baseline["audio"][index]
                ),
                "video_mae": float((a.float() - b.float()).abs().mean()),
                "audio_finite": bool(torch.isfinite(candidate["audio"][index]).all()),
            }
            report["samples"].append(row)
            print(json.dumps(row), flush=True)
            canvas = Image.new("RGB", (480 * 4, 272 * 2))
            for col, frame in enumerate((0, 10, 20, 30)):
                for y, tensor in enumerate((a, b)):
                    canvas.paste(
                        Image.fromarray(tensor[frame].numpy()).resize((480, 272)),
                        (480 * col, 272 * y),
                    )
            canvas.save(ROOT / f"{case}-{index}-comparison.jpg")
    report["batch_vs_single_reference"] = []
    for index, case in enumerate(("b1", "b1_short")):
        single = torch.load(ROOT / f"baseline-{case}-quality.pt", weights_only=True)
        for name in ("baseline", backend):
            batch = torch.load(ROOT / f"{name}-b2_mixed-quality.pt", weights_only=True)
            row = {
                "backend": name,
                "sample": index,
                "lpips": video_distance(
                    metric, single["video"][0], batch["video"][index]
                ),
                "audio_log_stft": distance(batch["audio"][index], single["audio"][0]),
            }
            report["batch_vs_single_reference"].append(row)
            print(json.dumps(row), flush=True)
    report["quality_passed"] = all(
        row["lpips"] <= 0.03 and row["audio_log_stft"] <= 0.02 and row["audio_finite"]
        for row in report["samples"]
    )
    (ROOT / "quality.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["hybrid", "offset"], default="hybrid")
    args = parser.parse_args()
    with torch.inference_mode():
        main(args.backend)

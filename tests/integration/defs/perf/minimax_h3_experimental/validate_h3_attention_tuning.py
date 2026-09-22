# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paired media differences for the attention-only local/QA experiment."""

import json
from pathlib import Path

import lpips
import torch

from diagnose_batching_audio import distance
from validate_h3_throughput import video_distance


def main() -> None:
    root = Path("/work/outputs/b200-attention-tuning")
    metric = lpips.LPIPS(net="alex").eval().cuda()
    rows = []
    for batch in (1, 2, 4):
        source = root if batch == 1 else root.with_name("b200-attention-tuning-retry")
        baseline = torch.load(
            source / f"baseline-b{batch}-quality.pt", weights_only=True
        )
        candidate = torch.load(
            source / f"compiled-b{batch}-quality.pt", weights_only=True
        )
        for index in range(batch):
            a, b = baseline["video"][index], candidate["video"][index]
            row = {
                "batch": batch,
                "sample": index,
                "video_lpips": video_distance(metric, a, b),
                "video_mae": float((a.float() - b.float()).abs().mean()),
                "video_sampled_exact": torch.equal(a, b),
                "audio_log_stft": distance(
                    candidate["audio"][index], baseline["audio"][index]
                ),
                "audio_exact": torch.equal(
                    candidate["audio"][index], baseline["audio"][index]
                ),
                "audio_finite": bool(torch.isfinite(candidate["audio"][index]).all()),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
        del baseline, candidate
    (root / "quality.json").write_text(
        json.dumps(
            {
                "video_sampling": "31 full-resolution frames, every fourth frame of 124",
                "audio": "complete generated stereo waveform",
                "samples": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    with torch.inference_mode():
        main()

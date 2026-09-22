# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Report full-resolution sampled-frame and complete-audio A/B differences."""

import json
from pathlib import Path

import lpips
import torch

from diagnose_batching_audio import distance
from validate_h3_throughput import video_distance


def main():
    root = Path("/work/outputs/b200-efficiency-verified")
    metric = lpips.LPIPS(net="alex").eval().cuda()
    rows = []
    for batch in (1, 2):
        a = torch.load(root / f"baseline-b{batch}-quality.pt", weights_only=True)
        b = torch.load(root / f"fused-b{batch}-quality.pt", weights_only=True)
        for i in range(batch):
            row = {
                "batch": batch,
                "sample": i,
                "video_lpips": video_distance(metric, a["video"][i], b["video"][i]),
                "video_mae": float(
                    (a["video"][i].float() - b["video"][i].float()).abs().mean()
                ),
                "audio_log_stft": distance(b["audio"][i], a["audio"][i]),
                "audio_finite": bool(torch.isfinite(b["audio"][i]).all()),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
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

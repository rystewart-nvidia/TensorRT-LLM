# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize clean measurements without treating interrupted warmups as results."""

import json
import statistics
from pathlib import Path


def main() -> None:
    root = Path("/work/outputs/b200-attention-tuning")
    retry_root = root.with_name("b200-attention-tuning-retry")
    original = json.loads((root / "pipeline.json").read_text())
    retry = json.loads((retry_root / "pipeline.json").read_text())
    groups = []
    selections = [("initial", original, "baseline", batch) for batch in (1, 2, 4)] + [
        ("initial", original, "metadata", 2),
        ("initial", original, "compiled", 1),
        ("retry", retry, "compiled", 2),
        ("retry", retry, "compiled", 4),
        ("retry", retry, "baseline", 4),
        ("retry", retry, "baseline_recheck", 2),
    ]
    for phase, source, mode, batch in selections:
        runs = [
            r
            for r in source["runs"]
            if r["mode"] == mode and r["batch"] == batch and not r["warmup"]
        ]
        if not runs:
            raise ValueError(f"Missing completed case: {phase}/{mode}/B{batch}")
        times = [r["seconds"] for r in runs]
        mean = statistics.mean(times)
        groups.append(
            {
                "phase": phase,
                "mode": mode,
                "batch": batch,
                "measurements": len(times),
                "seconds": mean,
                "videos_per_minute": 60 * batch / mean,
                "min_seconds": min(times),
                "max_seconds": max(times),
                "stdev_seconds": statistics.stdev(times) if len(times) > 1 else None,
                "mean_pre": statistics.mean(r["pre"] for r in runs),
                "mean_denoise": statistics.mean(r["denoise"] for r in runs),
                "mean_post": statistics.mean(r["post"] for r in runs),
            }
        )
    deltas = []
    for batch in (1, 2, 4):
        a = next(g for g in groups if g["mode"] == "baseline" and g["batch"] == batch)
        b = next(g for g in groups if g["mode"] == "compiled" and g["batch"] == batch)
        deltas.append(
            {
                "batch": batch,
                "throughput_change_percent": 100 * (a["seconds"] / b["seconds"] - 1),
            }
        )
    summary = {
        "groups": groups,
        "compiled_vs_baseline": deltas,
        "excluded": [
            {
                "phase": "retry",
                "mode": "baseline",
                "batch": 2,
                "run": 1,
                "reason": "Focused tests overlapped the first approximately 2.1 seconds",
            }
        ],
        "first_step_initial": original["first_step"],
        "first_step_retry": retry["first_step"],
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

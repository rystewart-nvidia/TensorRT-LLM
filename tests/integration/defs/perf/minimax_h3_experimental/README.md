<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# MiniMax H3 Local/QA Experiment Checkpoint

This directory checkpoints the experimental scripts and reports used for static
batching, offset FA4 attention, and step-level continuous batching on one B200.
It is not a supported serving API or a CI-registered benchmark. Reports retain
historical decisions and failures; attention-run-record.md is the latest result.

The scripts retain their original container layout to preserve reproducibility:

- `/work`: writable scratch workspace containing copies of these scripts.
- `/work/TensorRT-LLM-fork`: this source checkout, including model changes.
- `/work/TensorRT-LLM`: original source checkout, used for setup diagnostics only.
- `/work/checkpoint-path.txt`: local checkpoint path as visible inside Docker.
- `/hf_cache`: read-only Hugging Face cache containing the model checkpoint.
- `/work/cache`: writable compiler caches; `/work/outputs`: generated artifacts.

Run workloads only inside Docker on an allocated compute node. Exact image,
resource limits, commands, validation, and timing definitions are in the reports.
The setup script overlays the three H3 source files into the installed package.
`run-batching-tests.sh unit` runs H3 regressions; the standalone scheduler test is
`test_h3_arrival_scheduler.py`. `benchmark_h3_continuous.py --stages baseline serve`
runs matched calibration and arrival replays. Existing media is not overwritten;
use a fresh workspace/output directory for a new experiment.

Model weights, caches, raw traces, and generated media are intentionally not in
Git. Original artifacts remain under the shared scratch project
`/home/scratch.rystewart_wwfo/trtllm-video-gen-dynamo-testing/outputs/`.
Relative artifact links in the archived reports refer to that workspace.

The offset-attention adapter is still experimental, installed by the benchmark
at runtime; it is not the library default. Continuous batching supports only
the homogeneous T2VA shape tested here. Saved-media equivalence metrics are
diagnostics, not a substitute for perceptual review.

The Q/K normalization/partial-RoPE fusion experiment is in h3_fused_qk_rope.py,
with focused tests, a compiled-reference probe, a dispatch-verified full-pipeline
benchmark, and media/velocity diagnostics. It measured about 4% additional
throughput but has unresolved full-transformer numerical differences. It is
not enabled by default; read efficiency-run-record.md before using it.

The attention-only follow-up sweeps the pinned FA4 tile/CTA configurations and
tests generation-scoped metadata reuse plus a compile-compatible custom-op
boundary. It leaves the Q/K fusion off and preserves BF16 attention. The new
adapter and its tests remain local/QA only; they are not production defaults.
Read attention-run-record.md for the interrupted attempt, corrected warmup
dispatch check, excluded measurement, full-pipeline results, and media paths.

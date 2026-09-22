<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# H3 Space Video Review

## Purpose

Generate complete playable MP4s with audio for original and optimized static
batching, B2 and B4. Use the spacecraft and alien prompts in benchmark_batching.py.
Matching samples use seed 42 + sample index in both backends. B4 repeats the
prompt pair. This is media review, not a statistically robust throughput study.
One shape warmup and one measured generation per case; export is outside timing.
No pixel/perceptual similarity threshold is used as a quality gate.

## Allocation

- Date: 2026-09-22 UTC.
- Login endpoint: computelab-sc-01, load-balanced frontend-12 on initial check.
- Job: 4463616, h3-space-review, one B200, 8 CPUs, 180 GiB, 45-minute limit.
- Partition: b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb.
- Account/QOS: wwfo-nala_fallback / batch-short.
- Compute node: umbriel-b200-035; direct SSH, exactly one visible B200 verified.
- GPU UUID: GPU-520cf485-ff7c-d45e-99dc-47467c40e22b.
- Container: h3-space-4463616, --rm, 8 CPUs, 180 GiB, 16 GiB shared memory.
- Image: nvcr.io/nvidia/tensorrt-llm/release@sha256:f7753134fa2049d4fccbcc2e73258d5c71c6dd34d06ec5be38ae750cf6ca121d.
- Shared project mounted at /work; HF cache read-only at /hf_cache.
- Unrelated user job 4462606 is untouched.

Provisioning command (on login, no workload):

```sh
salloc --no-shell --immediate=60 --job-name=h3-space-review \
  --partition=b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb \
  --account=wwfo-nala_fallback --qos=batch-short --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --mem=180G --gres=gpu:b200:1 --time=00:45:00
```

## Execution

The existing three-file batching overlay is installed by
prepare-batching-container.sh. prepare-space-review.sh also installs ffmpeg.
run-space-review.sh executes the offset and baseline backends in that order,
each with --cases b2_mixed b4_mixed --repeats 1, then checks each MP4 with
verify_media.py. All workload execution is inside Docker on the allocated node.
Model, precision, resolution (960x544), 124 frames at 24 FPS, and 28 schedule
points are unchanged. Attention optimization remains a runtime-only prototype.

New outputs: outputs/b200-throughput-space. Each MP4 gets a JSON sidecar with
the exact prompt, seed, backend, batch size, and ffprobe results. Full video/audio
decode is checked; verify_media.py additionally checks nonblank/moving frames
and finite, nonsilent stereo audio, and creates preview sheets.

## Setup Notes

- Initial direct SSH required accepting the new node's first-use host key into
  /tmp/h3-space-known-hosts. Preflight then passed.
- Local ruff was unavailable; formatting/lint will run inside the container.
- Container setup completed successfully. Ruff formatting and lint passed.

Workload command (direct SSH to allocated compute node):

```sh
docker exec --user 151948:30 -e HOME=/tmp \
  -e TORCHINDUCTOR_CACHE_DIR=/work/cache/inductor \
  -e TRITON_CACHE_DIR=/work/cache/triton -e XDG_CACHE_HOME=/work/cache \
  -e CUDA_CACHE_PATH=/work/cache/cuda \
  h3-space-4463616 bash /work/run-space-review.sh
```

Container environment also sets HF_HOME=/hf_cache, HF_HUB_OFFLINE=1,
TRANSFORMERS_OFFLINE=1, TORCH_HOME=/work/cache/torch, and OMP_NUM_THREADS=8.
Logs: space-container-setup.log, space-review.log, space-offset.log,
space-baseline.log. The fork worktree was confirmed clean before execution.

## Results and Cleanup

- run-space-review.sh exited successfully. All 12 MP4s were exported with every
  frame and stereo audio, plus prompt/seed sidecars, validation JSON, and previews.
- All 12 passed ffmpeg full decode, ffprobe frame count/dimensions/frame rate,
  and nonblank/moving video plus finite, nonsilent stereo audio checks.
- Each file contains 124 frames, 960x544 at 24 FPS, with 32 kHz stereo AAC.
- All generation calls passed transformer batch traces of [B] * 27.
- Visual preview inspection covered all four optimized B4 samples. Spacecraft
  and alien scenes rendered. No claim of full perceptual quality or A/V sync
  validation is made; full clips are available for user review.
- Optimized backend replaced 52 attention modules; same model and precision.
- Fork source remains unchanged; all export work is in workspace harnesses.

Single warmed saved-generation timings, excluding export:

| Backend | B2 seconds | B4 seconds |
| --- | ---: | ---: |
| Original (baseline) | 88.1413 | 182.1740 |
| Optimized (offset) | 50.9613 | 103.2744 |

These are one observation per case, not a new robust throughput estimate or a
comparison against B1. Cold model load for offset was 281.42 seconds; subsequent
baseline load was 32.10 seconds. Loading and warmup are excluded above.

Confirmed only sleep infinity remained in the container after the workload.
Container stop exited successfully and docker ps -a confirmed removal.
Allocation 4463616 was released; Slurm confirmed CANCELLED with runtime 00:28:37.
All execution sessions finished. The unrelated allocation was not modified.

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# H3 Continuous Batching Experiment

## Scope and Definitions

Prototype step-level admission with a maximum of two active requests, homogeneous
960x544 T2VA, 124 frames at 24 FPS, 28 schedule points (27 transformer evaluations).
Same space prompts, FP8-block weights/BF16 attention, and runtime offset FA4
optimization as the prior review. No approximate generation settings changed.
No public API, Dynamo serving integration, or CI registration is added.

The fork transformer now accepts request-local [B,S] timestep indices in addition
to its existing shared [S] indices. h3_step_engine.py keeps individual video/audio
latents and scheduler instances. h3_arrival_scheduler.py implements FIFO B1,
static B2, and continuous B2. New requests are prepared only after arrival;
the replay does not pre-encode future prompts. No artificial batch-fill delay.
Preparation and decode run on the same worker, so their blocking cost is included.

Arrival traces (four requests, alternating prompts, seeds 42-45):

- Burst: [0, 0, 0, 0] seconds.
- Staggered: [0, 10, 20, 30] seconds.
- Light: [0, 35, 70, 105] seconds.

Queue time is arrival to admission. Active time is admission through preparation,
denoising, decode, and CPU output transfer, including any time the active request
waits while another request is prepared or decoded. Total is arrival to CPU media
ready. MP4 encoding/verification happens after each replay and is excluded.
Replay throughput includes its idle periods and drain; it is not saturated kernel
capacity. Four-request p50/p95 values are descriptive, not stable tail estimates.
Matched pipeline B1/B2 calibration separately excludes CPU media transfer/export,
with one warmup and two measured runs per shape/prompt case.

## Allocation

- Date: 2026-09-22 UTC.
- Login: computelab-sc-01, load-balanced frontend-05 on initial check.
- Job: 4464454, h3-continuous-qa.
- Node: umbriel-b200-035, direct SSH preflight passed with exactly one B200.
- GPU: GPU-27e9f59b-f968-8f87-dac2-95ef5d829a26.
- 8 CPUs, 180 GiB RAM, one GPU, 60-minute allocation limit.
- Partition: b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb.
- Account/QOS: wwfo-nala_fallback / batch-short.
- Container: h3-continuous-4464454, --rm, --cpus=8 --memory=180g --shm-size=16g.
- Image: nvcr.io/nvidia/tensorrt-llm/release@sha256:f7753134fa2049d4fccbcc2e73258d5c71c6dd34d06ec5be38ae750cf6ca121d.
- Project bind-mounted at /work; HF cache at /hf_cache read-only.
- Unrelated job 4462606 is untouched. Login used only for provisioning/status.

```sh
salloc --no-shell --immediate=60 --job-name=h3-continuous-qa \
  --partition=b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb \
  --account=wwfo-nala_fallback --qos=batch-short --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --mem=180G --gres=gpu:b200:1 --time=01:00:00
```

## Execution

Root setup in container: bash /work/prepare-space-review.sh, which overlays the
three model files from the fork and installs test dependencies plus ffmpeg.
Workload runs as UID:GID 151948:30, HOME=/tmp, HF offline, with all model/compiler
caches in shared scratch. The exact workload entrypoint is run-continuous-qa.sh.
Outputs go to outputs/b200-continuous-space; prior media is not overwritten.

## Results

- Formatting and lint passed; initial import-order lint issue was corrected.
- Scheduler lifecycle suite: 5 passed, covering capacity, no early admission,
  exact per-request step progression, mixed-step execution, slot refill without
  resetting the surviving request, and invalid arrival rejection.
- Existing H3 suites plus new mixed-timestep regressions: 136 passed in 27.91s.
  No tests were registered in CI. Log: continuous-qa.log.
- Initial real-checkpoint check stopped at 10.95% relative L2 difference between
  mixed-batch video velocity and unpadded/unmasked singleton velocity. Preserved
  in step-validation-failed.json and continuous-qa.log; no timings ran in that
  attempt. First model load took 311.01 seconds.
- Controlled diagnostic holds padding, FA4 backend, and conditioning-table shape
  fixed while slicing independent singleton forwards from the same packed input.
  It checks [0,0], [0,7], and reversed [7,0] step indices for both streams.
  All six video comparisons were exact; audio relative L2 was <= 9.30e-6.
- Uncontrolled singleton comparisons already differed by 10.55-16.28% (video)
  and 5.21-7.24% (audio) in the shared-step control. Thus the original diagnostic
  did not isolate mixed-timestep correctness. This does not establish perceptual
  quality equivalence. The 2% tolerance was not relaxed; the controlled reference
  removes unrelated padding/backend/table-shape changes. Both sets are preserved
  in step-validation.json and continuous-controlled-validation.log.
- Independent scheduler objects and one-step progress updates were verified.
- Performance resumed with --stages baseline serve --label performance;
  log continuous-performance.log.
- Matched warmed pipeline calibration (two measurements per case): spacecraft
  B1 26.73375s, alien B1 26.66766s, optimized B2 50.24457s. Matched B1 pair
  total 53.40140s, about 2.247 videos/min; B2 about 2.388 videos/min (+6.28%).
  All four reference MP4s passed media checks. Raw: matched-baseline.json.
- Full-resolution mixed-timestep warmup completed before arrival replays.

## Arrival Replay Results

All policies use our optimized attention path. B2 is a capacity limit, not a
requirement to wait for two arrivals. Latencies below are arrival-to-media-ready.

| Arrivals | Policy | Videos/min | Mean latency (s) | p50 (s) | p95 (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| Burst | FIFO B1 | 2.240 | 66.97 | 66.96 | 103.14 |
| Burst | Static B2 | 2.356 | 76.67 | 76.67 | 101.87 |
| Burst | Continuous B2 | 2.383 | 75.53 | 75.53 | 100.73 |
| Staggered | FIFO B1 | 2.239 | 52.00 | 52.01 | 74.67 |
| Staggered | Static B2 | 2.310 | 56.24 | 62.12 | 72.87 |
| Staggered | Continuous B2 | 2.321 | 58.51 | 60.91 | 72.89 |
| Light | FIFO B1 | 1.822 | 26.77 | 26.78 | 26.79 |
| Light | Static B2 | 1.822 | 26.76 | 26.76 | 26.79 |
| Light | Continuous B2 | 1.821 | 26.78 | 26.78 | 26.79 |

At light load all policies execute singletons with negligible queue time.
Throughput is arrival-limited and includes idle time and final drain.

Continuous B2 executed 42 mixed-timestep forwards in the staggered trace.
It reduced mean queue time from 17.70s to 12.83s, but increased mean active time
from 38.54s to 45.68s. The first request completed at 38.80s versus about 26.8s
with B1. Earlier admission is not automatically lower end-to-end latency.
Relative to static B2, median improved 1.21s, p95 was effectively unchanged,
and mean latency worsened 2.27s. Replay throughput was 3.65% above B1, not the
6.28% saturated calibration gain.

Burst continuous B2 used no mixed timesteps and behaves like static batching.
The static burst replay's first two forwards were unusually slow (~2.2s versus
~1.68s normally), so its ~1% throughput gap versus continuous is not evidence of
a scheduling advantage. These are exploratory, four-request, single-replay
traces, not statistically conclusive serving benchmarks.

The prototype establishes feasibility, but does not demonstrate the requested
latency reduction while retaining the full throughput gain. With only modest
B2 compute efficiency, sharing the GPU slows each active request substantially.
An age-aware admission policy could defer a newcomer while an older request is
near completion, but that trades batching opportunities for latency; it needs
measurement rather than an assumption of improvement. Broader kernel/pipeline
efficiency work is likely more promising for a substantial gain.

## Final Validation and Artifacts

- Performance process exited successfully after all nine arrival replays.
- All 40 full MP4s passed automated media validation, including exactly 124
  decoded frames, moving/nonblank video, and valid generated audio.
- Staggered continuous spacecraft/alien contact sheets were visually inspected;
  they were coherent. Full perceptual review remains separate from these checks.
- Final Ruff and git diff --check passed. A diagnostic closure lint warning was
  fixed by explicitly binding its capture dictionary; execution is unchanged.
- Scheduler regressions reran successfully: 5 passed in 0.05s. Together with
  the 136 H3 regressions, 141 local/QA tests passed. No CI registration added.
- [Output index and exact prompts](outputs/b200-continuous-space/README.md).
- Raw replay reports, events, calibration, diagnostics, media metadata and
  contact sheets live in outputs/b200-continuous-space/.
- Internal transformer/test changes are uncommitted on
  feat/minimax-h3-static-batching; the scheduling harness remains a local/QA
  prototype, not a production VisualGen or Dynamo serving API.

## Cleanup

The performance process exited 0. Docker top showed only the idle container
entrypoint; h3-continuous-4464454 was stopped and its removal verified. Job
4464454 was released and confirmed CANCELLED, with allocation runtime 40m30s.
No background test worker remains and unrelated allocations were untouched.

Workload command on the allocated compute node:

```sh
docker exec --user 151948:30 -e HOME=/tmp \
  -e TORCHINDUCTOR_CACHE_DIR=/work/cache/inductor \
  -e TRITON_CACHE_DIR=/work/cache/triton -e XDG_CACHE_HOME=/work/cache \
  -e CUDA_CACHE_PATH=/work/cache/cuda \
  h3-continuous-4464454 bash /work/run-continuous-qa.sh
```

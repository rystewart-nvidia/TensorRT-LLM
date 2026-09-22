<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# H3 Batched Attention Tuning

2026-09-22. Experimental local/QA work on feat/minimax-h3-static-batching,
starting from 2f2bbbeb. No Q/K normalization/RoPE fusion in this experiment.
No public API or CI registration changes.

## Resources

Slurm job 4466093, h3-attention-tuning, one B200 on umbriel-b200-068.
GPU-f44e2821-7443-8f59-8977-6d84a9156965, verified afresh by direct SSH.
8 CPUs, 180 GiB RAM, 60-minute limit. Account wwfo-nala_fallback, QOS batch-short,
partition b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb.
Login frontend-07 used only for allocation/status. No other active user jobs
at initial inspection. Container h3-attention-4466093, ID 0d464292f213,
--rm --cpus=8 --memory=180g --shm-size=16g, selected allocated UUID explicitly.
Image nvcr.io/nvidia/tensorrt-llm/release@sha256:f7753134fa2049d4fccbcc2e73258d5c71c6dd34d06ec5be38ae750cf6ca121d.
Scratch project mounted /work, HF cache /hf_cache read-only; all other caches
under /work/cache. Workloads UID:GID 151948:30, HOME=/tmp, HF offline.
Setup command: bash /work/prepare-space-review.sh; log attention-container-setup.log.

## Plan

Inspect installed FA4, screen tile and CTA configurations with independent
correctness checks, then compare full-pipeline B1/B2/B4 for useful candidates.
Separately test precomputed metadata and a compile-compatible custom-op boundary.
Preserve BF16 attention, FP8-block weights, steps, resolution, and space prompts.
Use warmed, synchronized end-to-end timing, not profiler totals, for claims.

Installed FA4 defaults to tile (128,128), 2 Q stages for long sequences,
persistent scheduling and eligible 2-CTA instructions. Its CLC scheduler is
already disabled for variable-length MHA. Hopper-only knobs (mma_pv_is_rs,
intra_wg_overlap, num_threads) do not tune the SM100 implementation, so are
not swept. Split-KV is not an obvious fit for already-large MHA workloads.

## Results

Initial microbenchmark completed: B1/B2/B4, S19000/H56/D128, BF16, mixed valid
suffix lengths, precomputed offsets, 20 shuffled warmed measurements per valid
configuration. The default tile (128,128) with 2 CTA was fastest in each batch.
Median CUDA-event ms for B1/B2/B4: 7.1104 / 14.3258 / 28.0995. Disabling 2 CTA:
7.5835 / 15.1668 / 30.1287. Tile (128,64) was slower. N=192 and N=256 were
rejected with AssertionError by the installed kernel. These failures are
preserved; no timings are attributed to rejected configurations. Initial sweep
uses the prior approximate 19k shape, not claimed to be exact prompt packing.
No kernel tuning candidate promoted from this sweep.

Prepared-attention runtime adapter leaves token refinement and unmasked B1
attention unchanged. It validates nonempty valid-suffix masks and precomputes
lengths/offsets once per static generation context. The benchmark explicitly
resets its metadata before every pipeline call; it is not a general mutable
context or serving cache. Two variants isolate metadata reuse alone (same
disabled-compiler boundary) and metadata reuse plus an opaque custom operation
with fake-tensor support. No normalization, RoPE, precision, or FA4 math changes.

Seven focused GPU tests passed in 18.43s: B1/B2/B4 strided QKV, masked SDPA
reference checks, exact custom-op vs eager output, padding and sample isolation,
fullgraph compilation, CUDA-graph capture/replay, invalid masks, and metadata
reset/context replacement. The full pipeline does not enable CUDA graphs;
capture is a focused compatibility test, not a performance claim.

Pipeline benchmark runs baseline B1/B2/B4, metadata-only B2, compiled B1/B2/B4,
then baseline B2 recheck. One warmup and three measurements per main case;
recheck two measurements. Warmup's second transformer step is profiled to prove
50 custom-op calls for compiled B2/B4 and zero otherwise. Warmups and profiling
are excluded from timings. First-step velocity differences are recorded against
same-prompt/seed baseline before each mode's timed runs. Full video/audio is
exported and media-validated after the final measured run, outside timers.
Final results and the corrected recovery procedure are recorded below.

## Interrupted Candidate Attempt

Initial process exited 1 during compiled B2 warmup. The step-2 profiler counted
59 CPU custom-op events but exactly 50 actual FA4 GPU launches. Additional
compiler/fake execution contaminated a check placed too early in warmup.
No compiled B2 timing was recorded; the original baseline, metadata-only and B1
control timings/media remain valid. The original failed dispatch trace and log
are preserved, not overwritten. Moved profiling to step 10, retaining the exact
50-call assertion rather than relaxing it.

Compiled B2 first-step differences before the stop: video relative L2 0.0405881,
audio 0.0620260. Metadata-only B2 and compiled B1 were exact. The custom-op
boundary changes whole-block compilation, so this is not an attention-math-only
numerical experiment despite leaving FA4 math and precision unchanged. Do not
promote it without understanding that effect.

Recovery: benchmark_h3_attention_tuning.py --recovery, log
attention-pipeline-retry.log, fresh output directory b200-attention-tuning-retry.
Loads once; fresh B2/B4 baseline checks (one warmup + one measurement), compiled
B2/B4 (one warmup + two measurements), and B2 baseline recheck (one warmup + one
measurement). Saves full paired B2/B4 media. B1 control already completed in the
original process. This reduced recovery matrix avoids repeating the completed
three-run baseline and metadata-only studies. Requested ten minutes additional
allocation limit for the unexpected cold reload and validation; release promptly
after work completes.

Slurm denied the time-limit extension (Access/permission denied). The original
60-minute deadline is unchanged. The recovery model load unexpectedly took only
33.80 seconds (initial load 309.57 seconds). Final focused tests were started
during the expected reload interval and passed (7 tests, 17.20s), but overlapped
warmup and the first ~2.1 seconds of the recovery B2 baseline measurement.
Test log completed 23:24:37.544 UTC; recovery B2 warmup report was written at
23:24:35.430 UTC. Exclude that one recovery baseline B2 timing from performance
claims; its saved media remains useful for quality comparison. The initial
three baseline measurements and the later recheck are uncontaminated. A stop
request found the tests already finished (pkill exit 1). No candidate inference
timings overlap tests. Defer the exact-shape probe until the benchmark exits.

## Final Performance Result

Recovery completed with exit 0. Step-10 profiles verified exactly 50 custom-op
calls for compiled B2/B4 and zero for baseline/recheck. Metadata was built once
per complete candidate generation. No Q/K fusion or CUDA graphs enabled in the
pipeline. All timers include pre/denoise/post and synchronized wall time, but
exclude export, profiling, and warmup. Latencies below are per complete batch.

| Batch | Baseline seconds | Compiled seconds | Baseline videos/min | Compiled videos/min | Throughput change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 27.16696 | 27.17380 | 2.20857 | 2.20801 | -0.03% |
| 2 | 51.79374 | 54.35950 | 2.31688 | 2.20753 | -4.72% |
| 4 | 105.31859 | 109.49180 | 2.27880 | 2.19195 | -3.81% |

Baseline and B1 control: three warmed measurements. Compiled B2/B4: two warmed
measurements in recovery. B2 metadata-only: three measurements, mean 51.87671s,
2.31318 videos/min, effectively unchanged (-0.16%). These are exploratory small
samples, not confidence intervals. The fresh uncontaminated B4 baseline check
was 105.45618s; final B2 recheck was 51.96097s. Their small drift does not explain
the regression. The contaminated recovery B2 baseline is explicitly excluded
by summarize_h3_attention_tuning.py; raw results are retained.

The slowdown is in denoising: B2 46.95046 -> 49.51721s, B4 96.17280 ->
100.35307s. Pre/post remain essentially unchanged. Profiles show different,
more expensive surrounding pointwise kernels after removing the graph break.
This suggests less favorable whole-block compiler fusion, rather than a faster
attention kernel being hidden by export/decode. It is a profile-based inference,
not a fully localized compiler diagnosis.

Decision: no throughput win. Keep both adapters experimental and disabled by
default. Preserve the previous best offset-FA4 path and the separately
checkpointed Q/K fusion experiment unchanged. Do not attribute the previous
Q/K fusion's ~4% gain to this unsuccessful attention-boundary experiment.

## Remaining Tile Probe

After all pipeline timing completed, tried the exact S19339 shape with default
(128,128) and smaller-query M64 tiles. Default B1 passed the SDPA correctness
comparison. The first M64/N64/2CTA configuration terminated the process in the
pinned CuTe compiler: IR verification rejected a tmem_load alignment (required
2 columns / 8 bytes, inferred alignment 1 byte). Exit 1. No M64 timings are
valid, and subsequent shapes/configurations did not run. The partial JSON and
attention-exact-microbench.log are preserved. This is not evidence that every
possible FA4 configuration was exhaustively tested; the initial successful
sweep supports keeping the default among the tested working configurations.

## Media and Numerical Checks

20 full MP4s passed media checks: 124 frames, 960x544, 24 FPS, 32kHz stereo,
nonblank/moving video and finite/nonsilent audio. Candidate B2 samples 0/1 and
B4 samples 2/3 contact sheets were inspected and coherent. This is not a full
perceptual review. Seven focused tests passed initially and again after the
import-order fix (17.20s on final run); lint passed for all six new Python files.

First-step video/audio relative L2: metadata-only B2 and compiled B1 exactly
zero; compiled B2 0.04058815 / 0.06202597 (reproduced in both processes), compiled
B4 0.04001284 / 0.05222929. Final B2 baseline recheck was exact. No FA4 precision
or math was changed, but the custom-op boundary changes whole-block compiler
fusion and numerical behavior. These differences do not establish worse visual
quality. Isolated custom-op/eager attention remained exact in focused tests.

Paired media differences (31 full-resolution frames per video, complete audio):

| Batch/sample | LPIPS | Video MAE (0-255) | Audio log-STFT |
| --- | ---: | ---: | ---: |
| B1/0 | 0 | 0 | 0 |
| B2/0 | 0.219221 | 13.83947 | 0.030584 |
| B2/1 | 0.157657 | 7.87517 | 0.010837 |
| B4/0 | 0.219221 | 13.83947 | 0.030582 |
| B4/1 | 0.157657 | 7.87517 | 0.010837 |
| B4/2 | 0.330730 | 23.42382 | 0.018741 |
| B4/3 | 0.127043 | 10.15151 | 0.011974 |

B1 sampled video and complete audio were exact. All candidate audio was finite.
These are descriptive differences, not a perceptual-quality verdict or a passed
equivalence threshold. No acceptance threshold was relaxed. The candidate is
rejected as a throughput optimization regardless of the media differences.

## Cleanup

Recovery benchmark, final tests, summary, and media comparison exited 0.
The initial benchmark and exact-shape M64 tile probe exited 1 as documented.
Docker top showed only sleep infinity before cleanup. Stopped
h3-attention-4466093, verified its --rm removal, then released job 4466093.
Slurm confirmed CANCELLED after 56m53s, ending 2026-09-22 23:44:30 UTC
(16:44:30 in Slurm's displayed timezone). No task process or container remains.
GPU driver 610.57.04; configured power limit 1000 W, unchanged throughout.
The requested time extension was denied and never applied.

## Commands and Artifacts

Inside h3-attention-4466093, UID:GID 151948:30, HOME=/tmp:

```sh
python -u /work/probe_h3_attention_tuning.py
python -m pytest -q --noconftest -c /dev/null -p no:cacheprovider /work/test_h3_prepared_attention.py
python -u /work/benchmark_h3_attention_tuning.py
python -u /work/benchmark_h3_attention_tuning.py --recovery
python -u /work/probe_h3_attention_tuning.py --sequence 19339 --extra-tiles --output /work/outputs/b200-attention-tuning/exact-shape-microbench.json
python /work/summarize_h3_attention_tuning.py
python -u /work/validate_h3_attention_tuning.py
```

All generated data remains in scratch, not Git:
- outputs/b200-attention-tuning: initial results, B1 paired media, microbench,
  summary.json with explicit exclusion, and paired media metrics.
- outputs/b200-attention-tuning-retry: complete dispatch-verified B2/B4 paired
  media and recovery timings.
- attention-*.log: setup, both pipeline attempts, both test runs, both tile
  probes, and media comparison.

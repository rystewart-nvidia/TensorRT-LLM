<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# H3 Kernel Efficiency Experiment

## Checkpoint

Prior work committed with DCO sign-off as ce2e58f3 on
feat/minimax-h3-static-batching. Includes request-local timestep support,
regressions, and local/QA harness/report snapshots under
tests/integration/defs/perf/minimax_h3_experimental. No CI registration.
The checkpoint was pushed to the user's origin fork using its configured SSH key.

## Allocation and Scope

2026-09-22, Slurm job 4465256, one B200 on umbriel-b200-068, 8 CPUs,
180 GiB RAM, 60-minute limit. Account wwfo-nala_fallback, QOS batch-short.
Partition b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb.
Current verified GPU: GPU-3da4eea0-c8f3-34f8-fcde-fbb8432c5b46.
Container h3-efficiency-4465256, image
nvcr.io/nvidia/tensorrt-llm/release@sha256:f7753134fa2049d4fccbcc2e73258d5c71c6dd34d06ec5be38ae750cf6ca121d.
Scratch project mounted at /work and HF cache read-only at /hf_cache.
All workloads run inside Docker on the allocated node, never on login.
Unrelated job 4462606 is untouched.
Driver 610.57.04; observed configured GPU power limit 1000 W, unchanged.

The initial Docker request mistakenly reused the prior allocation's GPU UUID
on the same node; the cluster proxy rejected it before container creation.
Retried with the current UUID returned by preflight. Initial SSH preflight
required recording the assigned host key in /tmp/h3-kernel-known-hosts.

## Hypothesis

Existing profiles already show compiler-fused AdaLN gathers/pointwise operations.
Do not duplicate that optimization. Q/K per-head normalization and partial
split-half RoPE instead use separate reductions/pointwise passes, roughly 9% of
the prior B2 GPU profile. Prototype a dedicated combined kernel, preserving H3's
96-of-128 partial rotary layout and the compiled path's rounding points. Compare against the
compiled current implementation, not only eager PyTorch. Keep best offset FA4
attention, checkpoint, precision, resolution, and schedule unchanged.

Run kernel correctness/isolation tests first, then a short shape-matched
microbenchmark. Only proceed to full-model A/B if promising. No public API
change or approximation is intended.

## Kernel Development

- Initial probe import order triggered a cutlass/quack initialization error.
  Importing TRT-LLM before the standalone FA4 probe helper resolved it.
- The first kernel explicitly reproduced eager BF16 intermediate rounding.
  Its first small case matched eager exactly, but compiled output differed by
  about 0.35% relative L2. A later small case failed the compiled elementwise
  tolerance on one value. Logs retain this failure.
- Inspected Inductor-generated code with TORCH_LOGS=output_code: normalization,
  weight multiplication, and cosine/sine loads stay FP32; the rotated sine term
  is materialized in BF16 before addition. Revised the fused kernel to preserve
  this compiled path rather than eager intermediates. Compiled elementwise
  tolerance remains rtol=0.02, atol=0.02. Eager differences remain reported,
  but eager is not the deployed reference.
- Revised probe passed all six shapes and exact cross-sample isolation checks.
  H3 long-sequence B1 Q/K pair: compiled 1.1353ms, fused 0.4136ms. B2:
  compiled 2.2046ms, fused 0.7895ms (10 warmed samples each). Small-shape
  measurements are dominated by dispatch overhead and not speedup claims.
  Long-sequence compiled relative L2 <= 1.11e-5. These are kernel timings,
  not end-to-end gains. Full-rotation control differs more (~0.2% relative L2)
  because Inductor chooses different fusion, but passed the same tolerance.
- Full-pipeline A/B started with benchmark_h3_efficiency.py: best offset FA4,
  spacecraft B1 and spacecraft/alien B2, one warmup and three measured runs per
  mode/shape, then baseline recheck with one warmup and two measurements.
  Wall time includes synchronized pipeline pre/denoise/post, excludes export.
  Full MP4s plus sampled video/full audio tensors are saved for review/metrics.
- First full-model attempt was interrupted after candidate B1 timings were
  indistinguishable from baseline and no recompilation appeared. Its candidate
  timings are invalid until dispatch is verified; all original artifacts/logs
  remain under outputs/b200-efficiency and efficiency-pipeline.log. Corrected
  the runtime adapter to gate on projected Q/K dtype rather than incoming hidden
  dtype, reset Dynamo when switching modes, and added a profiled dispatch gate:
  exactly 50 fused custom-op calls per denoising step for the candidate, zero for
  baseline. The full A/B retry writes outputs/b200-efficiency-verified, without
  overwriting the first attempt. Profiles are separate from timed generation.

## Verified Pipeline Results

The dispatch-verified retry exited 0. One warmup and three measurements per main
mode/shape; baseline recheck has one warmup and two measurements. Same loaded
model, node, GPU, prompts, seeds, precision, attention, and 27 transformer
evaluations. Latency includes synchronized pipeline pre/denoise/post, excluding
MP4 export and profiling. B2 latency is for the whole two-video batch.

| Batch | Baseline seconds | Fused seconds | Baseline videos/min | Fused videos/min | Throughput gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 26.6386 | 25.6537 | 2.2524 | 2.3388 | 3.84% |
| 2 | 49.9499 | 48.0555 | 2.4024 | 2.4971 | 3.94% |

Baseline recheck: B1 26.6121s, B2 50.0058s. The small before/after drift does not
explain the candidate gains. Main-run ranges: baseline B1 26.6227-26.6559s;
candidate B1 25.6317-25.6685s; baseline B2 49.8991-49.9894s; candidate B2
48.0381-48.0804s. These are small exploratory samples, not a confidence interval.

Separate profiles prove 50 candidate custom-op calls per step for B1 and B2,
and zero for both baseline and baseline recheck. In the B1 profile, the four
normalization/RoPE passes total 74.580ms; fused Q/K kernels total 26.154ms.
Attention and FP8 GEMMs remain the dominant work. Profiler totals are diagnostic;
the table uses independent unprofiled, full-pipeline timings.

All six saved full MP4s passed media validation, including 124 decoded frames
and generated stereo audio. Candidate spacecraft and alien contact sheets were
visually inspected and coherent; this is not a full perceptual equivalence test.

The candidate remains a local/QA runtime adapter, not a library default or a
production serving API. Its rounding behavior targets the pinned compiled H3
implementation. Broader precision/platform coverage is needed before upstream
integration. No CI list or public API was changed.

## Numerical Validation and Decision

Six focused GPU regression tests passed in 6.27s: compiled dispatch, GQA layout,
unrotated tail, input immutability, request isolation, and invalid inputs. Final
lint and Python syntax checks passed. Kernel probe checks also passed as above.

Paired media metrics (31 full-resolution video frames and complete audio):

| Batch/sample | Video LPIPS | Video MAE (0-255) | Audio log-STFT |
| --- | ---: | ---: | ---: |
| B1 spacecraft | 0.24238 | 16.12188 | 0.05379 |
| B2 spacecraft | 0.33298 | 26.23060 | 0.03627 |
| B2 alien | 0.17448 | 8.84452 | 0.00933 |

These are substantial differences, not a finding that perceptual quality is
worse. Valid/coherent output and mathematical intent do not establish numerical
equivalence. No quality threshold was relaxed and no quality-equivalence claim
is made for the candidate.

A follow-up real-checkpoint test compared first-step velocities on the exact
same prepared inputs/static context, with unchanged offset attention. Relative
L2 differences: B1 video 3.7674%, audio 3.4318%; B2 video 3.5091%, audio 2.0488%.
This is larger than the isolated Q/K kernel probe and remains unresolved; the
isolated probe does not establish full-transformer numerical equivalence.
Raw data: first-step-velocity.json, log efficiency-velocity.log.

Decision: preserve the measured ~4% speedup as an experimental candidate only.
Leave the prior library/default behavior unchanged. Next work should localize
the full-transformer discrepancy (including surrounding compiler fusion and
rounding), then retest media before promoting this path. Broader kernel/pipeline
work can also target the still-dominant attention/GEMM stages.

Artifacts: outputs/b200-efficiency-verified/pipeline.json, per-mode dispatch
summaries and Chrome traces, MP4s and metadata, sampled video/full-audio tensors,
and quality.json. The earlier outputs/b200-efficiency directory is explicitly
marked as an unverified pipeline attempt; its kernel microbenchmark remains valid.

Workload commands inside the allocated container (UID:GID 151948:30, HOME=/tmp;
compiler caches under /work/cache):

```sh
python -u /work/probe_h3_fused_qk_rope.py
python -u /work/benchmark_h3_efficiency.py
python -m pytest -q --noconftest -c /dev/null -p no:cacheprovider /work/test_h3_fused_qk_rope.py
python -u /work/validate_h3_efficiency.py
python -u /work/diagnose_h3_fused_velocity.py
```

## Cleanup

The verified benchmark, quality comparison, six-test suite, and first-step
diagnostic all exited successfully. Only the idle container entrypoint remained
before cleanup. h3-efficiency-4465256 was stopped and removal verified. Slurm job
4465256 was released and confirmed CANCELLED after 47m27s. No background worker
or task container remains; unrelated allocations were untouched.

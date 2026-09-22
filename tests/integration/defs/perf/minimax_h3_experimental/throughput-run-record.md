<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# H3 Throughput Investigation

## Scope

Prioritize single-B200 throughput, not continuous batching. Compare real static
batches, not a sequence of individual generations labeled as concurrency.
No changes to public APIs or CI registration.

## Environment

The measurements and artifacts in this record used these original prompts:

1. A woman with long brown hair and light skin smiles at the camera while standing
   in a sunlit park, her hair gently blowing in the breeze as she tilts her head slightly to the side.
2. Ocean waves roll onto a sunny beach with the sound of surf.

After this run, the harness defaults were changed to spacecraft/alien prompts.
Future runs write to `outputs/b200-throughput-space`; existing artifacts in
`outputs/b200-throughput` are unchanged. No space-prompt inference has run yet.

- Date: 2026-09-22.
- Source: TensorRT-LLM-fork, feat/minimax-h3-static-batching, f40695be.
- Slurm allocation: 4462868, one B200, 8 CPUs, 180 GiB, 60 minutes.
- Node: umbriel-b200-068 (same node as earlier B1 baseline).
- Account/QOS: wwfo-nala_fallback / batch-short.
- Partition: b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb.
- GPU: GPU-62bcf4b1-462b-f6cf-320c-7e4d853b857c, 183359 MiB.
- Container: h3-throughput-4462868.
- Image: nvcr.io/nvidia/tensorrt-llm/release@sha256:f7753134fa2049d4fccbcc2e73258d5c71c6dd34d06ec5be38ae750cf6ca121d.
- Scratch-mounted /work and read-only /hf_cache; caches under /work/cache.
- Login endpoint resolves to computelab-sc-01-frontend-11, the documented
  load-balanced frontend. Provisioning/status only on login, direct node SSH
  and Docker for all workloads. Exactly one GPU visible before container start.
- Unrelated pending user job 4462606 left untouched.

## Experiments

- `profile_h3_throughput.py --mode microbench`: warmed BF16 attention kernels,
  B1/B2/B4, H56/S19000/D128, with and without padding. FA4 candidate compacts
  valid K/V tokens after RoPE within each sample, retaining query order.
  Kernel timings and traces are diagnostic, not end-to-end speedup claims.

### Setup Failures and Microbenchmark

- Initial root-run import failed creating /work/cache/torchinductor; retry with
  a new cache directory also failed. Shared scratch is root-squashed. Both
  failures are preserved in throughput-microbench{,-retry}.log.
- Corrected execution to `docker exec --user 151948:30 -e HOME=/tmp`, with
  TORCHINDUCTOR_CACHE_DIR=/work/cache/inductor, TRITON_CACHE_DIR=/work/cache/triton,
  XDG_CACHE_HOME=/work/cache, CUDA_CACHE_PATH=/work/cache/cuda.
- Microbenchmark passed; throughput-microbench-user.log and
  outputs/b200-throughput/attention-microbench.json contain full results.
- Unmasked SDPA vs FA4 mean milliseconds: B1 6.92 vs 9.36;
  B2 15.92 vs 21.38; B4 31.68 vs 44.91.
- Masked SDPA vs FA4+compaction: B1 16.87 vs 7.28;
  B2 33.16 vs 17.28; B4 67.12 vs 35.27.
- BF16 random-tensor attention max absolute difference 0.0004883, mean 0.0000224.
  This is not a generated-media quality result.
- Full pipeline baseline command: `python /work/benchmark_h3_throughput.py
  --backend baseline --profile`. One persistent model, shape-specific warmup,
  two measured runs, full 28-point schedule and 544x960x124 output. B1,
  B2 repeated prompt, B2 mixed lengths, B4 mixed lengths. Every generation
  asserts 27 transformer forwards at the requested full batch dimension.

### Controlled Baseline

- Model loading: 342.60 seconds, excluded from throughput.
- B1 warmed wall times: 26.7534 and 26.7802 seconds.
- B2 repeated-prompt warmed wall times: 53.3298 and 53.3768 seconds.
- B1 first-denoising-step profile: about 416.5 ms attention, 252.4 ms FP8 GEMMs,
  24.5 ms activation quantization. Attention is about 51% of reported GPU time,
  GEMMs about 31%. Trace: baseline-b1-transformer.json.
- The batch dimension doubles work without a material B2 throughput gain in
  this matched-node unmasked control. No claim of hardware saturation from
  utilization alone.
- Additional left-padding-only FA4 offset probe prepared. It uses per-sample
  K/V start offsets and explicit valid lengths, preserving sample boundaries
  without gathering K/V. It must pass parity and isolation checks before use.
- B2 mixed-length warmed wall times: 87.0879 and 87.2025 seconds.
- B2 mixed first-step kernel totals: attention 2245.14 ms (75.05%),
  GEMMs 505.06 ms, activation quantization 39.90 ms, other 201.33 ms.
  Trace: baseline-b2_mixed-transformer.json. Profiling runs are separate from
  throughput measurements.
- B4 mixed warmed wall times: 183.1452 and 182.8850 seconds.
- Baseline completed with exit code 0. Mean wall seconds / videos per minute:
  B1 26.7668 / 2.2416; B2 equal 53.3533 / 2.2492;
  B2 mixed 87.1452 / 1.3770; B4 mixed 183.0151 / 1.3114.

### Offset Candidate

- `profile_h3_throughput.py --mode offset` passed. Five shapes covering
  B1/B2/B4, short and ~19k-token sequences; compaction and offset variants
  each passed SDPA comparison and exact padded-token/cross-sample isolation.
- Long-sequence B2 compaction 16.61 ms vs offsets 14.18 ms;
  B4 compaction 33.96 ms vs offsets 31.89 ms. These are a second microbenchmark
  run; raw samples are preserved instead of claiming precise small speedups.
- FA4 with explicit full lengths did not consistently beat unmasked SDPA
  across B1/B2/B4. Keep unmasked SDPA unchanged.
- Selected experimental runtime-only OffsetAttention: VANILLA for no mask,
  FA4 with K/V offsets and explicit valid lengths for left-padded H3 batches.
  No K/V compaction copy, no RoPE changes, no precision/schedule changes.
  FA4 source confirms seqused_k takes precedence over interval length; tests
  include a nonzero first offset and perturb the adjacent sample/padded values.
- Command: `python /work/benchmark_h3_throughput.py --backend offset
  --cases b1 b2_mixed b4_mixed --profile`.
- Fork source remains unchanged pending end-to-end performance and quality
  validation. Candidate code is in the workspace profiling/benchmark harness.
- Candidate model load: 35.41 seconds with filesystem/cache state warm.
  Replaced 52 H3 attention instances (50 denoising blocks, two text refiners).
- Candidate B1 warmed times: 26.7430 and 26.7865 seconds, effectively unchanged.
- Candidate B2 mixed warmed times: 50.3250 and 50.4867 seconds, mean 50.4058.
  2.3807 videos/minute, 1.729x old mixed-length B2 throughput. The approximately
  6% advantage over B1 is small; it is not a 1.729x advantage over sequential B1.
- Candidate B2 first-step kernel totals: attention 836.11 ms (down from 2245.14),
  GEMMs 516.59 ms, activation quantization 49.43 ms, other 243.66 ms.
  This confirms the end-to-end gain comes primarily from attention.
- Candidate B4 warmed times: 102.7566 and 102.8031 seconds, mean 102.7798.
  2.3351 videos/minute, below candidate B2. Candidate pipeline exited 0.
- A short-prompt B1 reference (seed 43) will match the second batch member and
  complete the sequential prompt-mix and full-resolution quality controls.

## Final Results

The matched short-prompt B1 run completed with exit code 0. Warmed times were
26.6905 and 26.7314 seconds (mean 26.7110). Long-prompt mean was 26.7668 seconds.
The sum of the two independent B1 means is 53.4778 seconds per matched prompt
pair, or 2.2439 videos/minute. This is a sequential reference, not a concurrent
request benchmark. Every B2/B4 generation was a true static batch.

| Configuration | Mean seconds per generation/batch | Videos/minute |
| --- | ---: | ---: |
| B1 matched prompt mix | 26.7389 per video | 2.2439 |
| Original B2, equal-length control | 53.3533 | 2.2492 |
| Original B2, mixed lengths | 87.1452 | 1.3770 |
| Offset FA4 B2, mixed lengths | 50.4058 | 2.3807 |
| Original B4, mixed lengths | 183.0151 | 1.3114 |
| Offset FA4 B4, mixed lengths | 102.7798 | 2.3351 |

Offset B2 is 6.094% above the matched B1 throughput; B4 is 4.063% above it.
The large 1.729x B2 improvement is relative to the slow original padded batch,
not relative to B1. Larger batches did not improve throughput beyond B2.
All measurements use two warmed repetitions, with shape-specific warmup excluded.
Wall time includes synchronized pipeline pre/denoise/post, not media export.
No continuous batching, fewer steps, reduced resolution, or new quantization.

## Validation and Decision

`python /work/validate_h3_throughput.py --backend offset` completed with exit 0
as a metrics collection run, but `quality_passed` is **false**. Thresholds were
preselected and not relaxed: video LPIPS <= 0.03, audio log-STFT <= 0.02.
Video comparison uses 31 full-resolution frames per sample (every fourth frame
of 124); audio comparison uses the full generated stereo waveform.

- 12 additional GPU compaction checks passed, covering B1/B2/B4 with left,
  right, interior-hole, and all-valid masks, plus exact sample isolation.
- The offset microbenchmark separately passed five shape cases, each checking
  numerical agreement and exact padding/cross-sample isolation.
- Candidate B1 is bit-identical to baseline in saved video frames and audio.
- Candidate vs original B2 batch: video LPIPS 0.14946 and 0.03570;
  audio log-STFT 0.04892 and 0.00417. Both video comparisons and the first
  audio comparison exceed the selected equivalence limits.
- Original B2 vs matching individual references: LPIPS 0.10212 / 0.01561,
  audio 0.03786 / 0.00408. The original batch already has full-resolution drift.
- Offset B2 vs matching individual references: LPIPS 0.09521 / 0.03491,
  audio 0.04183 / 0.00443. The candidate does not resolve parity.
- Contact sheets show valid, prompt-appropriate content but changed pose/framing.
  Metric failure establishes lack of output equivalence, not proven perceptual
  degradation. B4 received full-step shape/finite-output checks, not a separate
  full-resolution perceptual comparison.

Keep the offset optimization as a workspace-only experimental prototype. Do not
enable it by default or claim quality-validated throughput. The fork remains at
f40695be with no source/CI changes in this investigation. B1 remains the practical
default; future work should prioritize per-video kernel efficiency and numerical
stability over increasing batch size or building a continuous scheduler.

Artifacts: outputs/b200-throughput/{baseline,baseline-short,offset}-pipeline.json,
quality.json, raw paired quality tensors, contact sheets, attention microbench
JSON, and Chrome traces. Harnesses: profile_h3_throughput.py,
benchmark_h3_throughput.py, validate_h3_throughput.py. Exact environment and
commands are recorded above; source helper imports reuse benchmark_batching.py
and diagnose_batching_audio.py from the preceding experiment.

## Cleanup

- Final harness cleanup bound synchronous loop callbacks explicitly and applied
  import/dictionary/formatting fixes. Attention algorithms were unchanged.
- Ruff check and Python syntax compilation passed for all three new harnesses.
- Confirmed only `sleep infinity` remained in the task container after workloads.
- Stopped h3-throughput-4462868; `docker ps -a` confirmed it was removed.
- Released allocation 4462868 with scancel at approximately 50 minutes elapsed.
  Slurm initially reported COMPLETING while its epilog ran.
- Final Slurm state was CANCELLED with runtime 00:50:03. Container stop exited
  successfully; all benchmark and validation sessions finished.
- No changes to unrelated allocation 4462606, other containers, or fork source.

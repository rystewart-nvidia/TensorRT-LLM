<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# MiniMax-H3 batching implementation and validation

## Source

This run used the original park and beach prompts, not the later space prompts:

1. A woman with long brown hair and light skin smiles at the camera while standing
   in a sunlit park, her hair gently blowing in the breeze as she tilts her head slightly to the side.
2. Ocean waves roll onto a sunny beach with the sound of surf.

The harness now defaults to spacecraft/alien prompts and writes new runs to
`outputs/b200-batching-space`. Existing `outputs/b200-batching` media is unchanged.

- Fork: git@github.com:rystewart-nvidia/TensorRT-LLM.git
- Checkout: TensorRT-LLM-fork (the previous upstream experiment tree is untouched).
- Branch: feat/minimax-h3-static-batching
- Commit: f40695be730b27a429a8e1a1922bfde227d6614a, signed off and pushed to origin.
- Branch URL: https://github.com/rystewart-nvidia/TensorRT-LLM/tree/feat/minimax-h3-static-batching
- Base: 89e2cf0f
- SSH identity: id_ed25519_rystewart-nvidia.
- GPU regression tests are local/QA only, per user direction.

## Allocation and Environment

- Date: 2026-09-22.
- Allocation 4461936: one B200, 16 CPUs, 220 GiB RAM, 90 minutes. Immediate
  allocation timed out; confirmed absent from squeue, no compute ran.
- Allocation 4461972: one B200, 8 CPUs, 180 GiB RAM, 60 minutes.
- Node: umbriel-b200-040.
- Partition: b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb.
- Account/QOS: wwfo-nala_fallback / batch-short.
- Exactly one GPU visible: GPU-b246ad1c-c16e-747d-d86f-f799160d94c8,
  NVIDIA B200, 183359 MiB.
- Container: minimax-h3-batching-4461972.
- Image: nvcr.io/nvidia/tensorrt-llm/release@sha256:f7753134fa2049d4fccbcc2e73258d5c71c6dd34d06ec5be38ae750cf6ca121d.
- Runtime: TRT-LLM 1.3.0rc27, Torch 2.14.0a0+4fdf77b940.nv26.8.63802676.
- Only packing.py, pipeline_minimax_h3.py, and transformer_minimax_h3.py were
  overlaid into the installed runtime. Their original container versions were
  byte-identical to the previous upstream checkout (setup diff produced no output).
  This is Python-overlay validation, not a full build of the fork's main branch.
- Models read-only from /hf_cache; all persistent caches/artifacts in /work,
  both mounted from scratch.rystewart_wwfo. No compute on the login node.

## Commands and Evidence

- Container preparation: `bash /work/prepare-batching-container.sh`.
- Lint: `ruff format` and `ruff check` on the seven changed/new Python files.
  All lint checks passed.
- Unit/GPU tests: `bash /work/run-batching-tests.sh unit`.
  Result: 133 passed in 43.13 seconds; no failures or skipped tests.
  The focused run bypasses repository-wide conftest/plugin infrastructure, using
  `--noconftest -c /dev/null` and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.
- Logs: batching-container-setup.log, batching-unit.log, batching-e2e.log,
  batching-apt.log.
- Checkpoint smoke tests: `bash /work/run-batching-tests.sh e2e`.
  Pytest result: 2 passed in 413.42 seconds, batch sizes 2 and 4, each with
  different-length prompts, 128x128, 124 frames, 2 configured schedule points.
  The shell wrapper subsequently exited 2 because it was edited while bash was
  executing it (to disable pytest's /dev cache warning). The current wrapper
  passes `bash -n`; both actual pytest cases completed successfully and all
  workers exited. No model error was hidden by this harness failure.
- Git identity recovered from account GECOS and the specified public SSH-key
  comment: Ryan Stewart <rystewart@nvidia.com>. Configured in this clone only.
- Full-step quality/timing script: `/work/benchmark_batching.py`.
  Individual generations in the quality section are references only, not a
  concurrency benchmark. Timed batch runs verify 27 transformer calls each with
  the full requested batch dimension.

## Initial Full-Step Results

FP8_BLOCK_SCALES, VANILLA, torch.compile enabled, one B200, 960x544,
124 frames at 24 FPS, 28 schedule points (27 transformer evaluations).
Two different-length prompts; batch 4 repeats that pair. One shape-specific
warmup excluded, followed by two measured batched generations. Pipeline wall
time includes CUDA synchronization but excludes CPU media transfer and MP4 encoding.

| Batch | Warm wall times (seconds) | Mean (seconds) | Videos/second | Peak allocated / reserved GiB |
| --- | --- | --- | --- | --- |
| 2 | 87.6032, 87.6014 | 87.6023 | 0.02283 | 108.66 / 113.13 |
| 4 | 192.3086, 181.0756 | 186.6921 | 0.02143 | 113.37 / 125.32 |

This is genuine batching, confirmed by a forward pre-hook: every generation
made exactly 27 transformer calls at the full batch dimension. The mixed-length
prototype is not a throughput improvement over the prior single-prompt B200 run
(26.7807 seconds through the public API). That earlier baseline was on a different
B200 node and includes API overhead, so this is not a tightly controlled speedup study.

Full-step numerical comparison at the repository's 128x128 test resolution:

| Sample | Video LPIPS | Audio multi-resolution log-STFT distance |
| --- | --- | --- |
| 0 | 0.00788594 | 0.08803010 |
| 1 | 0.01776262 | 0.04238291 |

The preselected exploratory limits were LPIPS <= 0.03 and audio <= 0.02.
Video passed; audio did not. `quality_passed` remains false, and thresholds were
not relaxed after measuring. These low-resolution outputs test numerical parity,
not representative visual quality. Audio requires further validation before
production use.

Artifacts: `outputs/b200-batching/metrics.json`, paired raw quality tensors and
contact sheets, and six full-resolution MP4s. `batching-benchmark.log` completed
with exit code 0. No OOM occurred.

## Diagnostic Controls

`diagnose_batching_audio.py` completed with exit code 0. Logs are in
`batching-diagnostic.log`; JSON results are under `outputs/b200-batching/`.

- Equal-length repeated prompts, same full video settings, one warmup and one
  measured run per batch: batch 2 took 53.8014 seconds (0.03717 videos/s);
  batch 4 took 108.9338 seconds (0.03672 videos/s). Every call again asserted
  exactly 27 transformer evaluations at the full batch dimension.
- Compared with the earlier 26.7807-second single-video run, equal-length
  batching is roughly flat in throughput; mixed-length padding is substantially
  slower. No speedup claim is supported by these results.
- A direct SDPA dispatch probe at [B=2, H=56, S=19000, D=128], BF16, using
  HND views of NHD storage, selected `aten::_scaled_dot_product_cudnn_attention`
  both with and without a boolean padding mask. A backend switch was not observed.
- Forcing an all-valid attention mask in individual low-resolution references
  did not eliminate audio drift: masked-single vs batch distances were 0.11374
  and 0.02749. Masked vs unmasked single distances were 0.08708 and 0.02604.
  The source of numerical drift remains unresolved; do not equate non-silent,
  finite audio with established perceptual equivalence.
- Final QA now reuses one loaded VisualGen engine for batch-2 and batch-4
  smoke tests. This avoids reloading the checkpoint for each parameterization.

## Cleanup

Final source was recopied into the container, then
`bash /work/run-batching-final-validation.sh` completed with exit code 0:

- Six MP4s passed full decode, resolution, frame-count, FPS, stereo sample-rate,
  motion/non-uniformity, and finite/non-silent audio checks.
- 133 focused unit/GPU tests passed in 24.45 seconds.
- Both real-checkpoint public API tests passed in 56.71 seconds using one
  persistent engine, first batch 2 and then batch 4.
- Ruff and `git diff --check` passed.
- Preview contact sheets were inspected: expected park/portrait and beach
  content, with distinct outputs for different per-sample seeds.
- Log: `batching-final-validation.log`; media validation JSON and previews live
  alongside each MP4.

Confirmed no remaining worker processes before cleanup (only `sleep infinity`).
Stopped the task container; `docker ps -a` confirmed it was removed. Released
allocation 4461972 with `scancel` after validation. No other allocations or
containers were modified.
Slurm subsequently confirmed `JobState=CANCELLED`, runtime 42:54, after its
COMPLETING/epilog phase. The pushed branch's remote hash was verified as
`f40695be730b27a429a8e1a1922bfde227d6614a`; the fork worktree is clean.

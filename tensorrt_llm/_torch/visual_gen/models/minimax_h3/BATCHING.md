<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Experimental Static Batching

Pass a list of text prompts to the existing `VisualGen.generate(inputs=..., params=...)`
API. One request runs a joint video/audio denoising loop with a leading batch
dimension. The API returns one `VisualGenOutput` per prompt in input order.
Each result contains an unbatched video `[frames, height, width, 3]` and stereo
audio `[2, samples]`.

## Scope

- Text-to-video with audio, one GPU, common resolution, frame count, and schedule.
- One video per prompt. Repeat a prompt in the list for multiple samples.
- Sample `i` uses `params.seed + i`, including repeated prompts. A single prompt
  keeps the existing seed behavior.
- Unequal token lengths require `attention_config.backend: VANILLA`. Other
  backends reject padded text instead of silently ignoring its mask.
- Keyframe-conditioned requests remain single-sample only.
- Text encoding is per prompt; text refinement, all denoising steps, and VAE
  decoding are batched. The model weights are shared, but activation memory grows
  with batch size. There is no automatic batch-size reduction on OOM.
- This does not add dynamic batching or combine separately submitted requests.
  Serving/Dynamo request coalescing needs a separate scheduler change.

## Padding Contract

Text is left-padded before refinement, with a per-sample boolean mask threaded
through the text refiner and the joint transformer. H3 positions audio/video
after text. Left padding translates all valid temporal coordinates by the same
offset, preserving relative RoPE positions despite a shared packed layout.
Padded embeddings cannot act as attention keys. Video and audio unpacking retain
sample order, including stereo channel order.

## Local Validation

The experimental GPU tests are deliberately not registered in CI. Run in a
TRT-LLM environment with a supported GPU:

```bash
pytest tests/unittest/_torch/visual_gen/test_minimax_h3_pipeline.py \
       tests/unittest/_torch/visual_gen/test_minimax_h3_scheduler_packing.py \
       tests/unittest/_torch/visual_gen/test_minimax_h3_transformer.py \
       tests/unittest/_torch/visual_gen/test_minimax_h3_batching.py

LLM_MODELS_ROOT=/path/to/models MINIMAX_H3_CHECKPOINT=/path/to/MiniMax-H3 \
    pytest tests/integration/defs/visual_gen/test_minimax_h3_batching.py
```

The checkpoint tests exercise prompt batches of two and four through the public
executor API. The synthetic tests verify one transformer call per denoising step,
per-sample RNG, unequal prompt lengths, padding isolation, sample isolation, and
media ordering. Numerical comparisons permit floating-point accumulation drift;
real-checkpoint perceptual comparisons are also needed before production use.

## Initial B200 Evaluation

The implementation is functional, but is not yet a demonstrated throughput
optimization. Tests used one B200, the official H3 checkpoint revision
`42ed227ee7df40d41602854ae760620d6eb651fe`, dynamic FP8 block scales, VANILLA
attention, and torch.compile. The three modified model files were overlaid onto
the official TRT-LLM 1.3.0rc27 container; this was not a full source build.

At 960x544, 124 frames, 24 FPS, and 28 schedule points, each request made exactly
27 transformer calls with the entire batch present. After one excluded warmup,
two mixed-length prompt runs measured:

| Batch | Pipeline wall time | Videos/second | Peak allocated GPU memory |
| --- | --- | --- | --- |
| 2 | 87.60 s | 0.02283 | 108.66 GiB |
| 4 | 181.08-192.31 s | 0.02143 (aggregate) | 113.37 GiB |

Timing includes CUDA synchronization, but excludes model loading, CPU media
transfer, and MP4 encoding. Equal-length controls measured 53.80 s for batch 2
and 108.93 s for batch 4, showing a substantial padding-associated cost. Each
control had one excluded warmup and one measured run. The earlier single-video B200
baseline was 26.78 s through the public API on another node; it is context,
not a controlled same-node speedup comparison.

At the repository's 128x128 numerical-test resolution, full-step batch-vs-single
video LPIPS was 0.00789 / 0.01776. Audio multi-resolution log-STFT distance was
0.08803 / 0.04238. Video met the exploratory 0.03 limit; audio **did not meet**
the preselected 0.02 limit. These small-resolution comparisons are numerical
checks, not representative visual-quality evaluations. Matching the single
sample's attention mask did not eliminate audio drift. Both mask variants
dispatched through cuDNN SDPA, so a backend switch was not established.

Treat audio equivalence and padded-attention performance as open validation
items. Do not infer production readiness or a throughput gain from the passing
shape/isolation tests.

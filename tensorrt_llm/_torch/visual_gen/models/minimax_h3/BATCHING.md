<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# H3 Static Batching

Pass a list of text prompts through the existing
`VisualGen.generate(inputs=..., params=...)` API. One request runs a joint
video/audio denoising loop with a leading batch dimension. The API returns one
`VisualGenOutput` per prompt in input order, containing video
`[frames, height, width, 3]` and stereo audio `[2, samples]`.

## Request Contract

- Text-to-video with audio, one GPU, common resolution, frame count, and schedule.
- One video per prompt; repeat prompts for multiple samples. Sample `i` uses
  `params.seed + i`. Single-prompt RNG behavior is unchanged.
- Unequal token lengths require `attention_config.backend: VANILLA`. Other
  backends reject padded text instead of silently ignoring the mask.
- Keyframe-conditioned requests remain single-sample only.
- Text encoding is per prompt; refinement, denoising, and VAE decoding are batched.
  Activation memory grows with batch size; there is no automatic reduction on OOM.
- Separate requests are not coalesced. This is not continuous batching or a
  serving/Dynamo scheduler integration.

## Attention Contract

Text is left-padded before refinement. A per-sample boolean mask excludes padded
keys in the refiner and joint transformer. Left padding translates all valid
temporal coordinates equally, preserving relative RoPE positions in the shared
packed layout. Video/audio unpacking preserves sample and stereo channel order.

The H3-local VANILLA implementation uses offset FlashAttention 4 for masked
batches on SM100 with BF16 attention and 128-dimensional heads, when FA4 is
available. It validates nonempty contiguous suffix masks outside the block loop.
Offsets and valid lengths select each sample's keys without compacting K/V.

Singletons, unmasked batches, other architectures/dtypes, missing FA4, and other
mask layouts retain SDPA. Non-VANILLA backend selections are not replaced.
Shared attention implementations and other models are unchanged. No Q/K
normalization, RoPE, precision, or generation settings are changed. Floating-point
accumulation can differ from independent singleton generation; batching does not
promise bitwise-identical media or a universal throughput gain.

## Local/QA Validation

The new GPU tests are intentionally not registered in CI. Run in a built TRT-LLM
environment with a supported GPU:

```bash
pytest tests/unittest/_torch/visual_gen/test_minimax_h3_pipeline.py \
       tests/unittest/_torch/visual_gen/test_minimax_h3_scheduler_packing.py \
       tests/unittest/_torch/visual_gen/test_minimax_h3_transformer.py \
       tests/unittest/_torch/visual_gen/test_minimax_h3_batching.py \
       tests/unittest/_torch/visual_gen/test_minimax_h3_attention.py

LLM_MODELS_ROOT=/path/to/models MINIMAX_H3_CHECKPOINT=/path/to/MiniMax-H3 \
    pytest tests/integration/defs/visual_gen/test_minimax_h3_batching.py
```

The checkpoint tests exercise singleton and batched outputs through the public
executor API. Synthetic tests cover true batched transformer calls, per-sample
RNG, unequal lengths, padding/sample isolation, media ordering, and fallback
behavior. Performance and real-checkpoint perceptual comparisons are separate
validation steps; passing shape/isolation tests is not a quality guarantee.

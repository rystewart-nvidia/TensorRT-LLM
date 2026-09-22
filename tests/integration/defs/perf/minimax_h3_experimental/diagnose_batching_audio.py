# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isolate attention-backend numerical drift from batch padding."""

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark_batching import PROMPTS
from tensorrt_llm import VisualGenArgs
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader


def distance(actual, expected):
    values = []
    for n_fft in (512, 1024, 2048):
        window = torch.hann_window(n_fft)
        a = torch.stft(actual.float(), n_fft, hop_length=n_fft // 4,
                       window=window, return_complex=True).abs()
        b = torch.stft(expected.float(), n_fft, hop_length=n_fft // 4,
                       window=window, return_complex=True).abs()
        values.append(float(F.l1_loss(torch.log1p(a), torch.log1p(b))))
    return sum(values) / len(values)


def main():
    root = Path('/work/outputs/b200-batching')
    dispatch = {}
    query = torch.randn(2, 19000, 56, 128, device='cuda', dtype=torch.bfloat16).transpose(1, 2)
    mask = torch.ones(2, 1, 1, 19000, device='cuda', dtype=torch.bool)
    mask[1, :, :, :5] = False
    for name, attention_mask in [('unmasked', None), ('masked', mask)]:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            result = F.scaled_dot_product_attention(query, query, query, attn_mask=attention_mask)
            torch.cuda.synchronize()
        dispatch[name] = [entry.key for entry in profile.key_averages()
                          if 'scaled_dot_product' in entry.key]
        del result
    del query, mask
    (root / 'attention-dispatch.json').write_text(json.dumps(dispatch, indent=2) + '\n')
    print(json.dumps(dispatch), flush=True)
    args = VisualGenArgs.from_yaml(
        '/work/TensorRT-LLM-fork/examples/visual_gen/configs/minimax-h3-fp8-blockwise-1gpu.yaml'
    )
    args.model = Path('/work/checkpoint-path.txt').read_text().strip()
    pipeline = PipelineLoader(args).load(skip_warmup=True)
    original = pipeline._encode_prompts

    def encode(prompts, keyframes):
        embeds, tags, mask = original(prompts, keyframes)
        if mask is None:
            mask = torch.ones(embeds.shape[:2], device=embeds.device, dtype=torch.bool)
        return embeds, tags, mask

    pipeline._encode_prompts = encode
    report = []
    for index, prompt in enumerate(PROMPTS):
        output = pipeline.forward(prompt=prompt, seed=42 + index, height=128, width=128,
                                  num_frames=124, frame_rate=24, num_inference_steps=28)
        data = torch.load(root / f'quality-{index}.pt', weights_only=True)
        actual = output.audio[0].cpu()
        row = dict(sample=index,
                   masked_single_vs_batch=distance(actual, data['batch_audio']),
                   masked_single_vs_unmasked_single=distance(actual, data['reference_audio'][0]))
        report.append(row)
        print(json.dumps(row), flush=True)
        del output
    (root / 'audio-diagnostic.json').write_text(json.dumps(report, indent=2) + '\n')

    # Equal-length batches exercise the same implementation without a padding mask.
    pipeline._encode_prompts = original
    timings = []
    batch_trace = []
    hook = pipeline.transformer.register_forward_pre_hook(
        lambda module, args, kwargs: batch_trace.append(int(kwargs['hidden_states'].shape[0])),
        with_kwargs=True,
    )
    for batch in (2, 4):
        for run in range(2):
            batch_trace.clear()
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = pipeline.forward(prompt=[PROMPTS[0]] * batch, seed=42,
                                      height=544, width=960, num_frames=124,
                                      frame_rate=24, num_inference_steps=28)
            torch.cuda.synchronize()
            wall = time.perf_counter() - started
            assert batch_trace == [batch] * 27, batch_trace
            row = dict(batch_size=batch, warmup=(run == 0), wall_seconds=wall,
                       denoise=output.denoise, videos_per_second=batch / wall)
            timings.append(row)
            print(json.dumps(row), flush=True)
            (root / 'equal-length-timings.json').write_text(json.dumps(timings, indent=2) + '\n')
            del output
    hook.remove()


if __name__ == '__main__':
    main()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate true MiniMax-H3 static batches and collect B200 timings."""

import json
import time
from pathlib import Path

import lpips
import torch
import torch.nn.functional as F
from PIL import Image

from tensorrt_llm import VisualGenArgs
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm.visual_gen.output import VisualGenOutput

ROOT = Path('/work/outputs/b200-batching-space')
PROMPTS = [
    'A cinematic tracking shot follows a weathered spacecraft skimming over the icy '
    'rings of a giant planet. Blue thrusters flare as it banks past tumbling ice '
    'fragments, revealing a vast orbital station ahead. A deep engine rumble and '
    'crackling radio chatter accompany the flyby.',
    'A bioluminescent alien walks across a rocky moon beneath two enormous planets, '
    'with crunching footsteps and eerie chirping calls.',
]


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    config = VisualGenArgs.from_yaml(
        '/work/TensorRT-LLM-fork/examples/visual_gen/configs/minimax-h3-fp8-blockwise-1gpu.yaml'
    )
    config.model = Path('/work/checkpoint-path.txt').read_text().strip()
    config.compilation_config.skip_warmup = True
    report = {'config': config.model_dump(mode='json'), 'prompts': PROMPTS,
              'quality': [], 'runs': [],
              'quality_thresholds': {'lpips': 0.03, 'audio_log_stft': 0.02}}

    def save():
        (ROOT / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')

    started = time.perf_counter()
    pipeline = PipelineLoader(config).load(skip_warmup=True)
    report['load_seconds'] = time.perf_counter() - started
    batch_trace = []

    def trace(module, args, kwargs):
        batch_trace.append(int(kwargs['hidden_states'].shape[0]))

    hook = pipeline.transformer.register_forward_pre_hook(trace, with_kwargs=True)

    def generate(prompts, seed, height, width, label):
        batch_trace.clear()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = pipeline.forward(
            prompt=prompts, seed=seed, height=height, width=width,
            num_frames=124, frame_rate=24.0, num_inference_steps=28,
        )
        torch.cuda.synchronize()
        wall = time.perf_counter() - started
        batch = len(prompts) if isinstance(prompts, list) else 1
        assert batch_trace == [batch] * 27, batch_trace
        assert output.video.shape == (batch, 124, height, width, 3)
        assert output.audio.shape[:2] == (batch, 2)
        assert torch.isfinite(output.audio).all()
        row = dict(label=label, batch_size=batch, wall_seconds=wall,
                   pre_denoise=output.pre_denoise, denoise=output.denoise,
                   post_denoise=output.post_denoise,
                   videos_per_second=batch / wall,
                   peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                   peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                   transformer_batch_trace=list(batch_trace))
        report['runs'].append(row)
        print(json.dumps(row), flush=True)
        save()
        video, audio = output.video.cpu(), output.audio.cpu()
        del output
        return video, audio

    # Individual calls here are correctness references, never reported as concurrency.
    references = [generate(p, 42 + i, 128, 128, f'quality-single-{i}') for i, p in enumerate(PROMPTS)]
    batch_video, batch_audio = generate(PROMPTS, 42, 128, 128, 'quality-batch-2')
    metric = lpips.LPIPS(net='alex').eval()
    for index, (video, audio) in enumerate(references):
        candidate = batch_video[index]
        scores = []
        with torch.inference_mode():
            for start in range(0, 124, 16):
                a = candidate[start:start + 16].permute(0, 3, 1, 2).float() / 127.5 - 1
                b = video[0, start:start + 16].permute(0, 3, 1, 2).float() / 127.5 - 1
                scores.extend(metric(a, b).flatten().tolist())
        distances = []
        for n_fft in (512, 1024, 2048):
            window = torch.hann_window(n_fft)
            a = torch.stft(batch_audio[index].float(), n_fft, hop_length=n_fft // 4,
                           window=window, return_complex=True).abs()
            b = torch.stft(audio[0].float(), n_fft, hop_length=n_fft // 4,
                           window=window, return_complex=True).abs()
            distances.append(float(F.l1_loss(torch.log1p(a), torch.log1p(b))))
        row = dict(sample=index, lpips=sum(scores) / len(scores),
                   audio_log_stft=sum(distances) / len(distances),
                   video_mae=float((candidate.float() - video[0].float()).abs().mean()))
        report['quality'].append(row)
        torch.save(dict(reference_video=video, reference_audio=audio,
                        batch_video=candidate, batch_audio=batch_audio[index]),
                   ROOT / f'quality-{index}.pt')
        canvas = Image.new('RGB', (128 * 4, 128 * 2))
        for col, frame in enumerate((0, 41, 82, 123)):
            canvas.paste(Image.fromarray(video[0, frame].numpy()), (col * 128, 0))
            canvas.paste(Image.fromarray(candidate[frame].numpy()), (col * 128, 128))
        canvas.save(ROOT / f'quality-{index}.png')
        print(json.dumps(row), flush=True)
        save()
    del metric, references, batch_video, batch_audio
    report['quality_passed'] = all(
        row['lpips'] <= 0.03 and row['audio_log_stft'] <= 0.02 for row in report['quality']
    )
    save()

    for batch in (2, 4):
        prompts = (PROMPTS * 2)[:batch]
        for run in range(3):
            video, audio = generate(prompts, 42, 544, 960,
                                    f'batch-{batch}-' + ('warmup' if run == 0 else f'measured-{run}'))
            if run == 2:
                for index in range(batch):
                    VisualGenOutput(
                        request_id=index, video=video[index], audio=audio[index],
                        frame_rate=24.0, audio_sample_rate=32000,
                    ).save(ROOT / f'batch-{batch}-sample-{index}.mp4')
            del video, audio
    hook.remove()
    save()


if __name__ == '__main__':
    main()

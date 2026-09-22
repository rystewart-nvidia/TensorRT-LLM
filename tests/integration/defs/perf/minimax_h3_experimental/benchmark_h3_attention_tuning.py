# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paired local/QA pipeline benchmark, without experimental Q/K RoPE fusion."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch
from tensorrt_llm import VisualGenArgs
from tensorrt_llm._torch.visual_gen.models.minimax_h3.transformer_minimax_h3 import (
    MiniMaxH3Attention,
)
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm.visual_gen.output import VisualGenOutput

from benchmark_batching import PROMPTS
from benchmark_h3_throughput import OffsetAttention
from h3_prepared_attention import PreparedAttention, PreparedMetadata

ROOT = Path("/work/outputs/b200-attention-tuning")


def main(recovery: bool = False) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    target = ROOT / "pipeline.json"
    if target.exists():
        raise FileExistsError(target)
    config = VisualGenArgs.from_yaml(
        "/work/TensorRT-LLM-fork/examples/visual_gen/configs/minimax-h3-fp8-blockwise-1gpu.yaml"
    )
    config.model = Path("/work/checkpoint-path.txt").read_text().strip()
    config.compilation_config.skip_warmup = True
    report = {
        "config": config.model_dump(mode="json"),
        "runs": [],
        "prompts": PROMPTS,
        "dispatch": [],
        "first_step": [],
    }
    begin = time.perf_counter()
    pipeline = PipelineLoader(config).load(skip_warmup=True)
    report["load_seconds"] = time.perf_counter() - begin
    print(f"Loaded in {report['load_seconds']:.2f}s", flush=True)
    bank = PreparedMetadata()
    trace = []
    profile = None
    reference_velocity = {}
    mode, batch, run = "baseline", 1, 0

    def save() -> None:
        target.write_text(json.dumps(report, indent=2) + "\n")

    def before(module, args: tuple, kwargs: dict) -> None:
        nonlocal profile
        if mode not in ("baseline", "baseline_recheck"):
            bank.before_transformer(module, args, kwargs)
        trace.append(kwargs["hidden_states"].shape[0])
        if run == 0 and len(trace) == 10:
            profile = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            )
            profile.__enter__()

    def after(module, args: tuple, kwargs: dict, output) -> None:
        nonlocal profile
        if run == 0 and len(trace) == 1:
            values = tuple(x.detach().float().cpu() for x in output)
            if mode == "baseline":
                reference_velocity[batch] = values
            else:
                for stream, (actual, expected) in enumerate(
                    zip(values, reference_velocity[batch])
                ):
                    delta = actual - expected
                    report["first_step"].append(
                        {
                            "mode": mode,
                            "batch": batch,
                            "stream": stream,
                            "relative_l2": float(delta.norm() / expected.norm()),
                            "max_abs": float(delta.abs().max()),
                        }
                    )
            save()
        if profile is not None:
            torch.cuda.synchronize()
            profile.__exit__(None, None, None)
            entries = profile.key_averages()
            count = sum(
                e.count for e in entries if e.key == "h3_qa::prepared_attention"
            )
            expected = 50 if mode == "compiled" and batch > 1 else 0
            summary = {
                "mode": mode,
                "batch": batch,
                "custom_calls": count,
                "expected": expected,
                "sequence": bank.sequence,
                "operators": entries.table(
                    sort_by="self_cuda_time_total", row_limit=30
                ),
            }
            profile.export_chrome_trace(str(ROOT / f"{mode}-b{batch}-dispatch.json"))
            (ROOT / f"{mode}-b{batch}-dispatch-summary.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
            report["dispatch"].append(summary)
            save()
            assert count == expected, summary
            print(
                f"Verified {mode} B{batch}: {count} prepared attention calls",
                flush=True,
            )
            profile = None

    pre_hook = pipeline.transformer.register_forward_pre_hook(before, with_kwargs=True)
    post_hook = pipeline.transformer.register_forward_hook(after, with_kwargs=True)
    cases = (
        ("baseline", (1, 2, 4)),
        ("metadata", (2,)),
        ("compiled", (1, 2, 4)),
        ("baseline_recheck", (2,)),
    )
    if recovery:
        cases = (("baseline", (2, 4)), ("compiled", (2, 4)), ("baseline_recheck", (2,)))
    report["recovery"] = recovery
    report["profile_step"] = 10
    for mode, batches in cases:
        for name, module in pipeline.transformer.named_modules():
            if isinstance(module, MiniMaxH3Attention):
                kwargs = {
                    "num_heads": module.local_num_attention_heads,
                    "head_dim": module.head_dim,
                    "num_kv_heads": module.local_num_key_value_heads,
                    "dtype": torch.bfloat16,
                }
                if mode in ("metadata", "compiled") and name.startswith(
                    "transformer_blocks."
                ):
                    module.attn = PreparedAttention(
                        bank, opaque=mode == "compiled", **kwargs
                    )
                else:
                    module.attn = OffsetAttention(**kwargs)
        torch._dynamo.reset()
        for batch in batches:
            prompts = (PROMPTS * 2)[:batch]
            repeats = 2 if mode == "baseline_recheck" else 3
            if recovery:
                repeats = 2 if mode == "compiled" else 1
            for run in range(repeats + 1):
                bank.clear()
                trace.clear()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                begin = time.perf_counter()
                output = pipeline.forward(
                    prompt=prompts,
                    seed=42,
                    height=544,
                    width=960,
                    num_frames=124,
                    frame_rate=24,
                    num_inference_steps=28,
                )
                torch.cuda.synchronize()
                seconds = time.perf_counter() - begin
                assert trace == [batch] * 27, trace
                assert output.video.shape == (batch, 124, 544, 960, 3)
                assert torch.isfinite(output.audio).all()
                if mode in ("metadata", "compiled") and batch > 1:
                    assert bank.builds == 1, bank.builds
                row = {
                    "mode": mode,
                    "batch": batch,
                    "run": run,
                    "warmup": run == 0,
                    "seconds": seconds,
                    "videos_per_minute": 60 * batch / seconds,
                    "pre": output.pre_denoise,
                    "denoise": output.denoise,
                    "post": output.post_denoise,
                    "metadata_builds": bank.builds,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                }
                report["runs"].append(row)
                save()
                print(json.dumps(row), flush=True)
                if run == repeats and mode in ("baseline", "compiled"):
                    video, audio = output.video.cpu(), output.audio.cpu()
                    torch.save(
                        {"video": video[:, ::4].contiguous(), "audio": audio},
                        ROOT / f"{mode}-b{batch}-quality.pt",
                    )
                    for index, prompt in enumerate(prompts):
                        path = ROOT / f"{mode}-b{batch}-sample-{index}.mp4"
                        VisualGenOutput(
                            video=video[index],
                            audio=audio[index],
                            frame_rate=24,
                            audio_sample_rate=32000,
                        ).save(path)
                        path.with_suffix(".json").write_text(
                            json.dumps(
                                {
                                    "prompt": prompt,
                                    "seed": 42 + index,
                                    "mode": mode,
                                    "batch": batch,
                                    "num_frames": 124,
                                    "num_inference_steps": 28,
                                },
                                indent=2,
                            )
                            + "\n"
                        )
                        subprocess.run(
                            ["python", "/work/verify_media.py", str(path)],
                            check=True,
                            stdout=subprocess.DEVNULL,
                        )
                        print(f"Saved and verified {path}", flush=True)
                    del video, audio
                del output
    pre_hook.remove()
    post_hook.remove()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovery", action="store_true")
    args = parser.parse_args()
    if args.recovery:
        ROOT = Path("/work/outputs/b200-attention-tuning-retry")
    with torch.inference_mode():
        main(args.recovery)

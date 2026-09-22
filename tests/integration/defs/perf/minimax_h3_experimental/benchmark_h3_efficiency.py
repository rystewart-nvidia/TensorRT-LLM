# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local/QA paired full-pipeline test of H3 QK normalization/RoPE fusion."""

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
from h3_fused_qk_rope import install_fused_qk_rope
from h3_step_engine import H3StepEngine

ROOT = Path("/work/outputs/b200-efficiency-verified")


def verify_dispatch(pipeline, mode, batch):
    engine = H3StepEngine(pipeline)
    states = [engine.prepare(i, p, 42 + i) for i, p in enumerate(PROMPTS[:batch])]
    engine.predict(states)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        engine.predict(states)
        torch.cuda.synchronize()
    entries = profile.key_averages()
    count = sum(e.count for e in entries if e.key == "h3_qa::qk_norm_rope")
    expected = 50 if mode == "fused" else 0
    profile.export_chrome_trace(str(ROOT / f"{mode}-b{batch}-dispatch.json"))
    summary = {
        "mode": mode,
        "batch": batch,
        "fused_operator_calls": count,
        "operators": entries.table(sort_by="self_cuda_time_total", row_limit=25),
    }
    (ROOT / f"{mode}-b{batch}-dispatch-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    assert count == expected, summary
    print(f"Verified {mode} B{batch}: {count} fused QK/RoPE calls per step", flush=True)
    engine.reset()


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "pipeline.json").exists():
        raise FileExistsError(ROOT / "pipeline.json")
    config = VisualGenArgs.from_yaml(
        "/work/TensorRT-LLM-fork/examples/visual_gen/configs/minimax-h3-fp8-blockwise-1gpu.yaml"
    )
    config.model = Path("/work/checkpoint-path.txt").read_text().strip()
    config.compilation_config.skip_warmup = True
    report = {"config": config.model_dump(mode="json"), "runs": [], "prompts": PROMPTS}
    begin = time.perf_counter()
    pipeline = PipelineLoader(config).load(skip_warmup=True)
    report["load_seconds"] = time.perf_counter() - begin
    print(f"Model load: {report['load_seconds']:.3f}s", flush=True)
    for module in pipeline.transformer.modules():
        if isinstance(module, MiniMaxH3Attention):
            module.attn = OffsetAttention(
                num_heads=module.local_num_attention_heads,
                head_dim=module.head_dim,
                num_kv_heads=module.local_num_key_value_heads,
                dtype=torch.bfloat16,
            )
    original = MiniMaxH3Attention.forward
    trace = []
    hook = pipeline.transformer.register_forward_pre_hook(
        lambda module, pos, kw: trace.append(kw["hidden_states"].shape[0]),
        with_kwargs=True,
    )
    for mode in ("baseline", "fused", "baseline_recheck"):
        if mode == "fused":
            install_fused_qk_rope()
        else:
            MiniMaxH3Attention.forward = original
        torch._dynamo.reset()
        for batch in (1, 2):
            verify_dispatch(pipeline, mode, batch)
            repeats = 2 if mode == "baseline_recheck" else 3
            for run in range(repeats + 1):
                trace.clear()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                begin = time.perf_counter()
                output = pipeline.forward(
                    prompt=PROMPTS[:batch],
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
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                }
                report["runs"].append(row)
                (ROOT / "pipeline.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(row), flush=True)
                if run == repeats and mode != "baseline_recheck":
                    video, audio = output.video.cpu(), output.audio.cpu()
                    torch.save(
                        {"video": video[:, ::4].contiguous(), "audio": audio},
                        ROOT / f"{mode}-b{batch}-quality.pt",
                    )
                    for i, prompt in enumerate(PROMPTS[:batch]):
                        path = ROOT / f"{mode}-b{batch}-sample-{i}.mp4"
                        VisualGenOutput(
                            video=video[i],
                            audio=audio[i],
                            frame_rate=24,
                            audio_sample_rate=32000,
                        ).save(path)
                        path.with_suffix(".json").write_text(
                            json.dumps(
                                {
                                    "prompt": prompt,
                                    "seed": 42 + i,
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
    hook.remove()
    MiniMaxH3Attention.forward = original


if __name__ == "__main__":
    with torch.inference_mode():
        main()

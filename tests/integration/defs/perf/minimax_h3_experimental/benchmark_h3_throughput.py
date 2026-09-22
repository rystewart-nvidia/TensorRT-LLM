# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paired warmed pipeline timings for H3; experimental attention stays outside the fork."""

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import torch
from tensorrt_llm import VisualGenArgs
from tensorrt_llm._torch.visual_gen.attention_backend.flash_attn4 import (
    FlashAttn4Attention,
)
from tensorrt_llm._torch.visual_gen.attention_backend.vanilla import VanillaAttention
from tensorrt_llm._torch.visual_gen.models.minimax_h3.transformer_minimax_h3 import (
    MiniMaxH3Attention,
)
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm.visual_gen.output import VisualGenOutput

from benchmark_batching import PROMPTS
from profile_h3_throughput import compact_attention, offset_attention

ROOT = Path("/work/outputs/b200-throughput-space")


class CompactFlashAttention(FlashAttn4Attention):
    def forward(self, q, k, v, *, key_padding_mask=None, **kwargs):
        return compact_attention(self, q, k, v, key_padding_mask)


class HybridAttention(VanillaAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.masked_backend = FlashAttn4Attention(**kwargs)

    def forward(self, q, k, v, *, key_padding_mask=None, **kwargs):
        if key_padding_mask is None:
            return super().forward(q, k, v, **kwargs)
        return compact_attention(
            self.masked_backend,
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            key_padding_mask,
        ).transpose(1, 2)


class OffsetAttention(HybridAttention):
    def forward(self, q, k, v, *, key_padding_mask=None, **kwargs):
        if key_padding_mask is None:
            return VanillaAttention.forward(self, q, k, v, **kwargs)
        return offset_attention(
            self.masked_backend,
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            key_padding_mask,
        ).transpose(1, 2)


def main(args):
    for command in ("ffmpeg", "ffprobe"):
        if shutil.which(command) is None:
            raise RuntimeError(
                f"{command} is required to save and verify review videos"
            )
    ROOT.mkdir(parents=True, exist_ok=True)
    for name in args.cases:
        if list(ROOT.glob(f"{args.backend}-{name}-sample-*.mp4")):
            raise FileExistsError(
                f"Review media already exists for {args.backend}/{name}"
            )
    config = VisualGenArgs.from_yaml(
        "/work/TensorRT-LLM-fork/examples/visual_gen/configs/minimax-h3-fp8-blockwise-1gpu.yaml"
    )
    config.model = Path("/work/checkpoint-path.txt").read_text().strip()
    config.compilation_config.skip_warmup = True
    report = {
        "config": config.model_dump(mode="json"),
        "prompts": PROMPTS,
        "runs": [],
        "profiles": [],
        "experiment": "runtime-only FA4 replacement, unchanged checkpoint/precision/steps",
    }
    target = ROOT / f"{args.backend}{args.report_suffix}-pipeline.json"

    def save():
        target.write_text(json.dumps(report, indent=2) + "\n")

    started = time.perf_counter()
    pipeline = PipelineLoader(config).load(skip_warmup=True)
    report["load_seconds"] = time.perf_counter() - started
    if args.backend in ("fa4", "hybrid", "offset"):
        backend_class = {
            "fa4": CompactFlashAttention,
            "hybrid": HybridAttention,
            "offset": OffsetAttention,
        }[args.backend]
        count = 0
        for module in pipeline.transformer.modules():
            if isinstance(module, MiniMaxH3Attention):
                module.attn = backend_class(
                    num_heads=module.local_num_attention_heads,
                    head_dim=module.head_dim,
                    num_kv_heads=module.local_num_key_value_heads,
                    dtype=torch.bfloat16,
                )
                count += 1
        report["replaced_attention_modules"] = count
        assert count > 0
    batch_trace = []
    hook = pipeline.transformer.register_forward_pre_hook(
        lambda module, pos, kw: batch_trace.append(int(kw["hidden_states"].shape[0])),
        with_kwargs=True,
    )
    save()
    cases = {
        "b1": PROMPTS[:1],
        "b1_short": PROMPTS[1:],
        "b2_equal": [PROMPTS[0]] * 2,
        "b2_mixed": PROMPTS,
        "b4_mixed": PROMPTS * 2,
    }
    for name in args.cases:
        prompts = cases[name]
        seed = 43 if name == "b1_short" else 42
        for run in range(args.repeats + 1):
            batch_trace.clear()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = pipeline.forward(
                prompt=prompts,
                seed=seed,
                height=544,
                width=960,
                num_frames=124,
                frame_rate=24,
                num_inference_steps=28,
            )
            torch.cuda.synchronize()
            wall = time.perf_counter() - started
            batch = len(prompts)
            assert batch_trace == [batch] * 27, batch_trace
            assert output.video.shape == (batch, 124, 544, 960, 3)
            assert torch.isfinite(output.audio).all()
            row = {
                "case": name,
                "prompts": prompts,
                "seed": seed,
                "backend": args.backend,
                "warmup": run == 0,
                "iteration": run,
                "batch": batch,
                "wall_seconds": wall,
                "pre": output.pre_denoise,
                "denoise": output.denoise,
                "post": output.post_denoise,
                "videos_per_second": batch / wall,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "transformer_batch_trace": list(batch_trace),
            }
            report["runs"].append(row)
            print(json.dumps(row), flush=True)
            save()
            # Export every frame and the full audio outside the inference timer.
            if run == args.repeats:
                video, audio = output.video.cpu(), output.audio.cpu()
                row["media"] = []
                for index, prompt in enumerate(prompts):
                    path = ROOT / f"{args.backend}-{name}-sample-{index}.mp4"
                    VisualGenOutput(
                        request_id=index,
                        video=video[index],
                        audio=audio[index],
                        frame_rate=24.0,
                        audio_sample_rate=32000,
                    ).save(path)
                    probe = json.loads(
                        subprocess.check_output(
                            [
                                "ffprobe",
                                "-v",
                                "error",
                                "-count_frames",
                                "-show_streams",
                                "-show_format",
                                "-of",
                                "json",
                                str(path),
                            ]
                        )
                    )
                    streams = {s["codec_type"]: s for s in probe["streams"]}
                    assert int(streams["video"]["nb_read_frames"]) == 124, probe
                    assert streams["video"]["width"] == 960, probe
                    assert streams["video"]["height"] == 544, probe
                    assert streams["video"]["r_frame_rate"] == "24/1", probe
                    assert int(streams["audio"]["sample_rate"]) == 32000, probe
                    assert streams["audio"]["channels"] == 2, probe
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-v",
                            "error",
                            "-xerror",
                            "-i",
                            str(path),
                            "-map",
                            "0:v:0",
                            "-map",
                            "0:a:0",
                            "-f",
                            "null",
                            "-",
                        ],
                        check=True,
                    )
                    metadata = {
                        "file": path.name,
                        "prompt": prompt,
                        "seed": seed + index,
                        "backend": args.backend,
                        "batch_size": batch,
                        "sample_index": index,
                        "num_inference_steps": 28,
                        "frame_rate": 24,
                        "audio_sample_rate": 32000,
                        "probe": probe,
                        "decode_verified": True,
                    }
                    path.with_suffix(".json").write_text(
                        json.dumps(metadata, indent=2) + "\n"
                    )
                    row["media"].append(metadata)
                    print(f"Saved and verified {path}", flush=True)
                    save()
                del video, audio
            # Save representative full-resolution media for paired quality checks.
            if run == args.repeats and name in ("b1", "b1_short", "b2_mixed"):
                torch.save(
                    {"video": output.video[:, ::4].cpu(), "audio": output.audio.cpu()},
                    ROOT / f"{args.backend}-{name}-quality.pt",
                )
            del output
        if args.profile and name in ("b1", "b2_mixed"):
            profile = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            )
            state = {"count": 0}

            def before(module, pos, kw, state=state, profile=profile):
                state["count"] += 1
                if state["count"] == 1:
                    profile.start()

            def after(module, pos, output, state=state, profile=profile):
                if state["count"] == 1:
                    torch.cuda.synchronize()
                    profile.stop()

            pre = pipeline.transformer.register_forward_pre_hook(
                before, with_kwargs=True
            )
            post = pipeline.transformer.register_forward_hook(after)
            try:
                output = pipeline.forward(
                    prompt=prompts,
                    seed=seed,
                    height=544,
                    width=960,
                    num_frames=124,
                    frame_rate=24,
                    num_inference_steps=2,
                )
                del output
            finally:
                pre.remove()
                post.remove()
            profile.export_chrome_trace(
                str(ROOT / f"{args.backend}-{name}-transformer.json")
            )
            table = profile.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=35
            )
            report["profiles"].append({"case": name, "operators": table})
            print(table, flush=True)
            save()
    hook.remove()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=["baseline", "fa4", "hybrid", "offset"], required=True
    )
    parser.add_argument(
        "--cases", nargs="+", default=["b1", "b2_equal", "b2_mixed", "b4_mixed"]
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--report-suffix", default="")
    with torch.inference_mode():
        main(parser.parse_args())

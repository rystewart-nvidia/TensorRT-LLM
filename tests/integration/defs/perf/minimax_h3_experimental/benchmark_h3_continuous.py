# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Matched B1/B2 timings and exploratory open-loop H3 scheduling experiments."""

import argparse
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from tensorrt_llm import VisualGenArgs
from tensorrt_llm._torch.visual_gen.models.minimax_h3.transformer_minimax_h3 import (
    MiniMaxH3Attention,
)
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm.visual_gen.output import VisualGenOutput

from benchmark_batching import PROMPTS
from benchmark_h3_throughput import OffsetAttention
from h3_arrival_scheduler import Arrival, replay
from h3_step_engine import H3StepEngine

ROOT = Path("/work/outputs/b200-continuous-space")
SCENARIOS = {
    "burst": [0, 0, 0, 0],
    "staggered": [0, 10, 20, 30],
    "light": [0, 35, 70, 105],
}


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def save_media(stem, video, audio, prompt, seed):
    path = ROOT / f"{stem}.mp4"
    if path.exists():
        raise FileExistsError(path)
    VisualGenOutput(
        video=video, audio=audio, frame_rate=24, audio_sample_rate=32000
    ).save(path)
    save_json(
        path.with_suffix(".json"),
        {"prompt": prompt, "seed": seed, "num_frames": 124, "num_inference_steps": 28},
    )
    subprocess.run(
        ["python", "/work/verify_media.py", str(path)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    print(f"Saved and verified {path}", flush=True)


def validate_steps(pipeline):
    engine = H3StepEngine(pipeline, height=128, width=128)
    a = engine.prepare(0, PROMPTS[0], 42)
    b = engine.prepare(1, PROMPTS[1], 43)
    for _ in range(7):
        engine.step([b])
    result = []
    for states, progress in (([a, b], [0, 0]), ([a, b], [0, 7]), ([b, a], [7, 0])):
        for state, index in zip(states, progress):
            state.index = index
        expected = [engine.predict([state]) for state in states]
        captured = {}
        hook = pipeline.transformer.register_forward_pre_hook(
            lambda module, pos, kw, captured=captured: captured.update(kw),
            with_kwargs=True,
        )
        try:
            actual = engine.predict(states)
        finally:
            hook.remove()
        packed_expected = []
        for row in range(2):
            context = captured["static_context"]
            inputs = captured | {
                "hidden_states": captured["hidden_states"][row : row + 1],
                "audio_hidden_states": captured["audio_hidden_states"][row : row + 1],
                "timestep": captured["timestep"][row : row + 1],
                "timestep_indices": (
                    captured["timestep_indices"][row]
                    if captured["timestep_indices"].ndim == 2
                    else captured["timestep_indices"]
                ),
                "static_context": replace(
                    context,
                    text_embeds=context.text_embeds[row : row + 1],
                    text_attention_mask=context.text_attention_mask[row : row + 1],
                ),
            }
            packed_expected.append(pipeline.transformer(**inputs))
        for row in range(2):
            for stream, (batch_tensor, individual) in enumerate(
                zip(actual, expected[row])
            ):
                ref, val = individual[0].float(), batch_tensor[row].float()
                relative_l2 = float(
                    torch.linalg.vector_norm(val - ref)
                    / torch.linalg.vector_norm(ref).clamp_min(1e-8)
                )
                packed_ref = packed_expected[row][stream][0].float()
                packed_relative_l2 = float(
                    torch.linalg.vector_norm(val - packed_ref)
                    / torch.linalg.vector_norm(packed_ref).clamp_min(1e-8)
                )
                record = {
                    "indices": [s.index for s in states],
                    "row": row,
                    "stream": stream,
                    "relative_l2": relative_l2,
                    "same_padding_and_table_relative_l2": packed_relative_l2,
                }
                result.append(record)
                print(json.dumps(record), flush=True)
                if packed_relative_l2 > 0.02:
                    save_json(ROOT / "step-validation-controlled-failed.json", result)
                    raise AssertionError(f"Mixed-timestep velocity mismatch: {record}")
    old_a, old_b = a.index, b.index
    engine.step([a, b])
    assert (a.index, b.index) == (old_a + 1, old_b + 1)
    assert a.video_scheduler is not b.video_scheduler
    assert a.audio_scheduler is not b.audio_scheduler
    save_json(ROOT / "step-validation.json", {"checks": result, "passed": True})
    engine.reset()


def baseline(pipeline):
    records = []
    for name, prompts, seed in (
        ("b1_spacecraft", PROMPTS[:1], 42),
        ("b1_alien", PROMPTS[1:], 43),
        ("b2", PROMPTS, 42),
    ):
        for run in range(3):
            torch.cuda.synchronize()
            begin = time.perf_counter()
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
            seconds = time.perf_counter() - begin
            row = {
                "case": name,
                "batch": len(prompts),
                "warmup": run == 0,
                "seconds": seconds,
                "videos_per_minute": len(prompts) * 60 / seconds,
            }
            records.append(row)
            save_json(ROOT / "matched-baseline.json", records)
            print(json.dumps(row), flush=True)
            if run == 2:
                for index, prompt in enumerate(prompts):
                    save_media(
                        f"reference-{name}-{index}",
                        output.video[index].cpu(),
                        output.audio[index].cpu(),
                        prompt,
                        seed + index,
                    )
            del output


def warm_engine(engine):
    states = [engine.prepare(i, p, 42 + i) for i, p in enumerate(PROMPTS)]
    for _ in range(3):
        engine.step(states[:1])
    while not all(s.done for s in states):
        active = [s for s in states if not s.done]
        engine.step(active)
        done = [s for s in active if s.done]
        if done:
            engine.finish(done)
    engine.reset()
    print("Full-resolution mixed-timestep engine warmup complete", flush=True)


def summarize(report):
    for field in ("queue_seconds", "active_seconds", "total_seconds"):
        values = [row[field] for row in report["requests"]]
        report[field] = {
            "mean": float(np.mean(values)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
        }
    steps = [e for e in report["events"] if e["event"] == "step"]
    report["mixed_timestep_steps"] = sum(len(set(e["indices"])) > 1 for e in steps)
    for row in report["requests"]:
        executed = [
            e["indices"][e["ids"].index(row["id"])]
            for e in steps
            if row["id"] in e["ids"]
        ]
        assert executed == list(range(27)), executed


def main(args):
    ROOT.mkdir(parents=True, exist_ok=True)
    config = VisualGenArgs.from_yaml(
        "/work/TensorRT-LLM-fork/examples/visual_gen/configs/minimax-h3-fp8-blockwise-1gpu.yaml"
    )
    config.model = Path("/work/checkpoint-path.txt").read_text().strip()
    config.compilation_config.skip_warmup = True
    save_json(ROOT / f"config-{args.label}.json", config.model_dump(mode="json"))
    started = time.perf_counter()
    pipeline = PipelineLoader(config).load(skip_warmup=True)
    print(f"Model load seconds: {time.perf_counter() - started:.3f}", flush=True)
    count = 0
    for module in pipeline.transformer.modules():
        if isinstance(module, MiniMaxH3Attention):
            module.attn = OffsetAttention(
                num_heads=module.local_num_attention_heads,
                head_dim=module.head_dim,
                num_kv_heads=module.local_num_key_value_heads,
                dtype=torch.bfloat16,
            )
            count += 1
    assert count == 52, count
    if "validate" in args.stages:
        validate_steps(pipeline)
    if "baseline" in args.stages:
        baseline(pipeline)
    if "serve" in args.stages:
        engine = H3StepEngine(pipeline)
        warm_engine(engine)
        for scenario in args.scenarios:
            requests = [
                Arrival(i, at, PROMPTS[i % 2], 42 + i)
                for i, at in enumerate(SCENARIOS[scenario])
            ]
            for policy in args.policies:
                target = ROOT / f"{scenario}-{policy}.json"
                if target.exists():
                    raise FileExistsError(target)
                print(f"Starting scenario={scenario} policy={policy}", flush=True)
                report, outputs = replay(engine, requests, policy)
                report["scenario"] = scenario
                summarize(report)
                save_json(target, report)
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in report.items()
                            if k not in ("events", "requests")
                        }
                    ),
                    flush=True,
                )
                for request in requests:
                    video, audio = outputs[request.request_id]
                    save_media(
                        f"{scenario}-{policy}-{request.request_id}",
                        video,
                        audio,
                        request.prompt,
                        request.seed,
                    )
                del outputs
                engine.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["validate", "baseline", "serve"],
        default=["validate", "baseline", "serve"],
    )
    parser.add_argument(
        "--scenarios", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS)
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=["b1", "static_b2", "continuous_b2"],
        default=["b1", "static_b2", "continuous_b2"],
    )
    parser.add_argument("--label", default="initial")
    with torch.inference_mode():
        main(parser.parse_args())

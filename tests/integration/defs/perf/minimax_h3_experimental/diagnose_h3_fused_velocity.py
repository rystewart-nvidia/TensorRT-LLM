# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare baseline/candidate first-step velocities on identical packed inputs."""

import json
from pathlib import Path

import torch
from tensorrt_llm import VisualGenArgs
from tensorrt_llm._torch.visual_gen.models.minimax_h3.transformer_minimax_h3 import (
    MiniMaxH3Attention,
)
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader

from benchmark_batching import PROMPTS
from benchmark_h3_throughput import OffsetAttention
from h3_fused_qk_rope import install_fused_qk_rope
from h3_step_engine import H3StepEngine


def main():
    root = Path("/work/outputs/b200-efficiency-verified")
    config = VisualGenArgs.from_yaml(
        "/work/TensorRT-LLM-fork/examples/visual_gen/configs/minimax-h3-fp8-blockwise-1gpu.yaml"
    )
    config.model = Path("/work/checkpoint-path.txt").read_text().strip()
    config.compilation_config.skip_warmup = True
    pipeline = PipelineLoader(config).load(skip_warmup=True)
    for module in pipeline.transformer.modules():
        if isinstance(module, MiniMaxH3Attention):
            module.attn = OffsetAttention(
                num_heads=module.local_num_attention_heads,
                head_dim=module.head_dim,
                num_kv_heads=module.local_num_key_value_heads,
                dtype=torch.bfloat16,
            )
    engine = H3StepEngine(pipeline)
    states = [engine.prepare(i, p, 42 + i) for i, p in enumerate(PROMPTS)]
    expected = {
        batch: tuple(x.cpu() for x in engine.predict(states[:batch]))
        for batch in (1, 2)
    }
    install_fused_qk_rope()
    torch._dynamo.reset()
    rows = []
    for batch in (1, 2):
        actual = engine.predict(states[:batch])
        for stream, (a, b) in enumerate(zip(expected[batch], actual)):
            a, b = a.float(), b.cpu().float()
            row = {
                "batch": batch,
                "stream": stream,
                "relative_l2": float(
                    torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(a)
                ),
                "max_abs": float((a - b).abs().max()),
                "mean_abs": float((a - b).abs().mean()),
                "reference_rms": float(a.square().mean().sqrt()),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    (root / "first-step-velocity.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    with torch.inference_mode():
        main()

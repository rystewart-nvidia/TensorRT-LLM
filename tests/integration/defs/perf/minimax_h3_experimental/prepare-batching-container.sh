#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
package=$(python -c 'import importlib.util; print(next(iter(importlib.util.find_spec("tensorrt_llm").submodule_search_locations)))')
target="$package/_torch/visual_gen/models/minimax_h3"
mkdir -p /tmp/minimax-h3-original
for file in packing.py pipeline_minimax_h3.py transformer_minimax_h3.py; do
    cp "$target/$file" "/tmp/minimax-h3-original/$file"
    diff -u "/tmp/minimax-h3-original/$file" "/work/TensorRT-LLM/tensorrt_llm/_torch/visual_gen/models/minimax_h3/$file" || true
    cp "/work/TensorRT-LLM-fork/tensorrt_llm/_torch/visual_gen/models/minimax_h3/$file" "$target/$file"
done
python -m pip install pytest ruff lpips

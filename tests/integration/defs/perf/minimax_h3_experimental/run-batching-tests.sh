#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd /work
export LLM_MODELS_ROOT=/hf_cache
export MINIMAX_H3_CHECKPOINT
MINIMAX_H3_CHECKPOINT=$(< /work/checkpoint-path.txt)
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTEST_ADDOPTS='-p no:cacheprovider'
export PYTHONPATH=/work/TensorRT-LLM-fork/tests/unittest/_torch/visual_gen
repo=/work/TensorRT-LLM-fork
case "${1:-unit}" in
    unit)
        python -m pytest -q --noconftest -c /dev/null \
          "$repo/tests/unittest/_torch/visual_gen/test_minimax_h3_pipeline.py" \
          "$repo/tests/unittest/_torch/visual_gen/test_minimax_h3_scheduler_packing.py" \
          "$repo/tests/unittest/_torch/visual_gen/test_minimax_h3_transformer.py" \
          "$repo/tests/unittest/_torch/visual_gen/test_minimax_h3_batching.py"
        ;;
    e2e)
        python -m pytest -q -s --noconftest -c /dev/null \
          "$repo/tests/integration/defs/visual_gen/test_minimax_h3_batching.py"
        ;;
    *) exit 2 ;;
esac

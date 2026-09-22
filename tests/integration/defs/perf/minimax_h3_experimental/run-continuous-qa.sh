#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd /work
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTEST_ADDOPTS='-p no:cacheprovider'
python -m ruff check h3_step_engine.py h3_arrival_scheduler.py \
    benchmark_h3_continuous.py test_h3_arrival_scheduler.py
python -m pytest -q --noconftest -c /dev/null test_h3_arrival_scheduler.py
bash run-batching-tests.sh unit
python -u benchmark_h3_continuous.py 2>&1 | tee continuous-benchmark.log

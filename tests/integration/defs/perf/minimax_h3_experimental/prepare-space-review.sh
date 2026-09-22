#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
bash /work/prepare-batching-container.sh
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ffmpeg

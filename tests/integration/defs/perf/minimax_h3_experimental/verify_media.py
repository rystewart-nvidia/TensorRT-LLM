# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode and inspect the example artifact; write a small visual preview."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

path = Path(sys.argv[1])
probe = json.loads(subprocess.check_output([
    "ffprobe", "-v", "error", "-count_frames", "-show_streams",
    "-show_format", "-of", "json", str(path),
]))
video = next(s for s in probe["streams"] if s["codec_type"] == "video")
audio = next(s for s in probe["streams"] if s["codec_type"] == "audio")
assert (video["width"], video["height"]) == (960, 544), video
assert int(video["nb_read_frames"]) == 124, video
assert video["r_frame_rate"] == "24/1", video
assert audio["channels"] == 2, audio
assert int(audio["sample_rate"]) == 32000, audio
pixels = subprocess.check_output([
    "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0",
    "-vf", "scale=240:136", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
])
frames = np.frombuffer(pixels, dtype=np.uint8).reshape(-1, 136, 240, 3)
wave = np.frombuffer(subprocess.check_output([
    "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0",
    "-f", "f32le", "-acodec", "pcm_f32le", "-",
]), dtype=np.float32).reshape(-1, 2)
assert len(frames) == 124
assert np.isfinite(wave).all()
frame_std = frames.std(axis=(1, 2, 3))
motion = np.abs(np.diff(frames.astype(np.float32), axis=0)).mean()
rms = np.sqrt(np.mean(wave.astype(np.float64) ** 2, axis=0))
assert frame_std.min() > 1, "Uniform or blank frame detected"
assert motion > 0.1, "Video appears static"
assert (rms > 1e-6).all(), "Silent audio channel detected"
summary = {
    "file": str(path), "bytes": path.stat().st_size,
    "video": video, "audio": audio,
    "minimum_frame_std": float(frame_std.min()),
    "mean_adjacent_frame_difference": float(motion),
    "audio_rms_by_channel": rms.tolist(),
    "audio_peak": float(np.abs(wave).max()),
    "checks_passed": True,
}
path.with_suffix(".validation.json").write_text(json.dumps(summary, indent=2) + "\n")
sheet = Image.new("RGB", (720, 272))
for index, frame_index in enumerate(np.linspace(0, len(frames) - 1, 6, dtype=int)):
    sheet.paste(Image.fromarray(frames[frame_index]), ((index % 3) * 240, (index // 3) * 136))
sheet.save(path.with_suffix(".preview.jpg"))
print(json.dumps(summary, indent=2))

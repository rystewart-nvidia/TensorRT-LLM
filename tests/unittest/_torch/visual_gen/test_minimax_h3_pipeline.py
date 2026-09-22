# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Synthetic pipeline-level tests for MiniMax-H3."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from diffusers import MiniMaxH3Scheduler
from PIL import Image

from tensorrt_llm._torch.visual_gen.config import (
    DiffusionPipelineConfig,
    discover_pipeline_components,
)
from tensorrt_llm._torch.visual_gen.models.minimax_h3 import pipeline_minimax_h3 as h3_pipeline
from tensorrt_llm._torch.visual_gen.models.minimax_h3.packing import (
    MINIMAX_H3_TEXT_TAG,
    resolve_canvas_size,
)
from tensorrt_llm._torch.visual_gen.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm._torch.visual_gen.pipeline_registry import AutoPipeline, PipelineComponent
from tensorrt_llm.visual_gen.args import TorchCompileConfig, VisualGenArgs

pytestmark = pytest.mark.cpu_only


class _FakeMiniMaxH3Transformer:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(patch_size=(1, 2, 2))
        self.training = True
        self.static_context_calls = 0
        self.forward_calls = 0
        self.inference_modes: list[bool] = []
        self.attention_timesteps: list[torch.Tensor] = []
        self.batch_sizes: list[int] = []
        self.text_attention_mask: torch.Tensor | None = None

    def eval(self) -> "_FakeMiniMaxH3Transformer":
        self.training = False
        return self

    def prepare_static_context(
        self,
        prompt_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        text_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.static_context_calls += 1
        self.text_attention_mask = text_attention_mask
        return prompt_embeds, position_ids

    def __call__(
        self,
        *,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        static_context: tuple[torch.Tensor, torch.Tensor],
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.attention_timesteps.append(kwargs["timestep"].clone())
        assert static_context is not None
        self.forward_calls += 1
        self.batch_sizes.append(hidden_states.shape[0])
        self.inference_modes.append(torch.is_inference_mode_enabled())
        # Non-zero: an all-zero velocity is what _check_denoise_step treats
        # as a corrupt step, so a fake must not emit one.
        return torch.ones_like(hidden_states), torch.ones_like(audio_hidden_states)


class _SyntheticMiniMaxH3Pipeline(MiniMaxH3Pipeline):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)
        # BasePipeline.__init__ is bypassed here, so set the backing attribute
        # that BasePipeline.device reads.
        self._device = torch.device("cpu")
        self.pipeline_config = SimpleNamespace(visual_gen_mapping=None)
        self.transformer = _FakeMiniMaxH3Transformer()
        self.vae = SimpleNamespace(
            config=SimpleNamespace(latent_channels=2),
            spatial_compression_ratio=2,
        )
        self.audio_vae = SimpleNamespace(
            config=SimpleNamespace(latent_channels=3, sampling_rate=32000)
        )
        self.scheduler = MiniMaxH3Scheduler(shift=12.0)
        self.audio_scheduler = MiniMaxH3Scheduler(shift=3.0)
        self._is_warmup = True

    def _encode_prompt(
        self,
        prompt: str,
        keyframes: list[Image.Image],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del prompt, keyframes
        return torch.ones(1, 2, 4), torch.full((2,), MINIMAX_H3_TEXT_TAG)

    def _encode_keyframes(
        self,
        keyframes: list[Image.Image],
        latent_height: int,
        latent_width: int,
        generator: torch.Generator,
    ) -> None:
        del keyframes, latent_height, latent_width, generator
        return None

    def _decode_video(
        self,
        rows: torch.Tensor,
        num_condition_rows: int,
        num_latent_frames: int,
        latent_height: int,
        latent_width: int,
    ) -> torch.Tensor:
        del num_condition_rows, num_latent_frames, latent_height, latent_width
        return torch.zeros(rows.shape[0], 124, 32, 32, 3, dtype=torch.uint8)

    def _decode_audio(self, rows: torch.Tensor, num_audio_latents: int) -> torch.Tensor:
        return torch.zeros(rows.shape[0], 2, num_audio_latents * 800)


def test_pipeline_reuses_static_context_across_joint_denoise_steps() -> None:
    pipeline = _SyntheticMiniMaxH3Pipeline()

    output = pipeline.forward(
        prompt="a synthetic prompt",
        seed=42,
        height=32,
        width=32,
        num_frames=124,
        frame_rate=24.0,
        num_inference_steps=4,
    )

    assert pipeline.transformer.static_context_calls == 1
    assert pipeline.transformer.forward_calls == 3
    assert output.video.shape == (1, 124, 32, 32, 3)
    assert output.video.dtype == torch.uint8
    assert output.audio.shape[0:2] == (1, 2)
    assert output.audio_sample_rate == 32000
    assert output.frame_rate == 24.0
    actual_times = torch.cat(pipeline.transformer.attention_timesteps)
    expected_times = 1.0 - torch.minimum(
        pipeline.scheduler.timesteps, pipeline.audio_scheduler.timesteps
    )
    torch.testing.assert_close(actual_times, expected_times)
    assert bool(((actual_times >= 0) & (actual_times <= 1)).all())
    assert bool((actual_times[1:] <= actual_times[:-1]).all())


def test_shared_denoise_refreshes_cache_state_per_request() -> None:
    pipeline = _SyntheticMiniMaxH3Pipeline()
    refreshed = []
    pipeline.cache_accelerator = SimpleNamespace(
        is_enabled=lambda: True, refresh=refreshed.append, get_stats=lambda: None
    )
    for _ in range(2):
        pipeline.forward(
            prompt="test",
            seed=42,
            height=32,
            width=32,
            num_frames=124,
            frame_rate=24.0,
            num_inference_steps=4,
        )
    assert refreshed == [3, 3]


def test_pipeline_runs_generation_in_inference_mode() -> None:
    pipeline = _SyntheticMiniMaxH3Pipeline()

    pipeline.forward(
        prompt="a synthetic prompt",
        seed=42,
        height=32,
        width=32,
        num_frames=124,
        frame_rate=24.0,
        num_inference_steps=2,
    )

    assert pipeline.transformer.inference_modes == [True]


@pytest.mark.parametrize("batch_size", [2, 4])
def test_static_batch_shares_denoising_steps(batch_size: int) -> None:
    pipeline = _SyntheticMiniMaxH3Pipeline()
    output = pipeline.forward(
        prompt=["test"] * batch_size,
        seed=42,
        height=32,
        width=32,
        num_frames=124,
        frame_rate=24.0,
        num_inference_steps=4,
    )
    assert pipeline.transformer.batch_sizes == [batch_size] * 3
    assert pipeline.transformer.static_context_calls == 1
    assert output.video.shape == (batch_size, 124, 32, 32, 3)
    assert output.audio.shape[:2] == (batch_size, 2)


def test_prompt_batch_left_pads_and_masks_variable_lengths(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _SyntheticMiniMaxH3Pipeline()

    def encode(prompt: str, keyframes: list[Image.Image]) -> tuple[torch.Tensor, torch.Tensor]:
        del keyframes
        return torch.full((1, len(prompt), 4), len(prompt)), torch.ones(
            len(prompt), dtype=torch.long
        )

    monkeypatch.setattr(pipeline, "_encode_prompt", encode)
    embeds, tags, mask = pipeline._encode_prompts(["a", "abcd"], [])
    assert embeds.shape == (2, 4, 4)
    assert tags.tolist() == [MINIMAX_H3_TEXT_TAG] * 4
    assert mask.tolist() == [[False, False, False, True], [True, True, True, True]]
    assert not embeds[0, :3].any()
    assert (embeds[0, 3] == 1).all()
    assert (embeds[1] == 4).all()


def test_batched_latents_preserve_each_sample_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _SyntheticMiniMaxH3Pipeline()
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []

    def capture_denoise(**kwargs: object) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        video = kwargs["latents"]
        audio = kwargs["extra_streams"]["audio"][0]
        captured.append((video.clone(), audio.clone()))
        return video, {"audio": audio}

    monkeypatch.setattr(pipeline, "denoise", capture_denoise)
    settings = dict(height=32, width=32, num_frames=124, frame_rate=24.0, num_inference_steps=2)
    pipeline.forward(prompt=["first", "second"], seed=42, **settings)
    pipeline.forward(prompt="second", seed=43, **settings)
    for batched, single in zip(captured[0], captured[1]):
        torch.testing.assert_close(batched[1:2], single, rtol=0, atol=0)
        assert not torch.equal(batched[0], batched[1])


@pytest.mark.parametrize("prompt", [[], "", ["valid", ""], ["valid", 3]])
def test_batch_rejects_invalid_prompts(prompt: str | list[str]) -> None:
    with pytest.raises(ValueError, match="non-empty prompt"):
        _SyntheticMiniMaxH3Pipeline().forward(
            prompt=prompt,
            seed=0,
            height=32,
            width=32,
            num_frames=124,
            frame_rate=24.0,
            num_inference_steps=2,
        )


def test_batch_rejects_keyframe_conditioning() -> None:
    with pytest.raises(ValueError, match="text-to-video only"):
        _SyntheticMiniMaxH3Pipeline().forward(
            prompt=["first", "second"],
            seed=0,
            height=32,
            width=32,
            num_frames=124,
            frame_rate=24.0,
            num_inference_steps=2,
            keyframes=[Image.new("RGB", (32, 32))],
        )


def test_one_zero_velocity_sample_cannot_hide_in_a_batch() -> None:
    with pytest.raises(RuntimeError, match="all zeros"):
        h3_pipeline._check_denoise_step(
            torch.stack((torch.ones(4, 2), torch.zeros(4, 2))), "video", 0
        )


@pytest.mark.parametrize("override", [False, True])
def test_shared_denoise_stream_schedules(override: bool) -> None:
    """L2/T1: paired scheduler latents vs a manual loop, with per-stream time overrides."""
    pipeline = _SyntheticMiniMaxH3Pipeline()

    class Scheduler:
        def __init__(self, timesteps: torch.Tensor) -> None:
            self.timesteps = timesteps
            self.seen: list[float] = []

        def step(self, noise: torch.Tensor, t: torch.Tensor, latents: torch.Tensor, **kwargs):
            self.seen.append(float(t))
            return (latents - t * noise,)

    times = torch.tensor([0.9, 0.6, 0.2])
    audio_times = torch.tensor([0.8, 0.4, 0.1]) if override else times
    video_scheduler, audio_scheduler = Scheduler(times), Scheduler(audio_times)
    initial = torch.arange(1, 9, dtype=torch.float32).reshape(1, 4, 2)
    indices = []

    def predict(video, extra, index, timestep, embeds, extras):
        indices.append(index)
        return video * 0.1, {"audio": extra["audio"] * 0.2}

    video, extra = pipeline.denoise(
        initial.clone(),
        video_scheduler,
        torch.ones(1, 2, 4),
        1.0,
        predict,
        extra_streams={"audio": (initial.clone(), audio_scheduler)},
        extra_stream_timesteps={"audio": audio_times} if override else None,
    )
    expected_video, expected_audio = initial.clone(), initial.clone()
    for t, audio_t in zip(times, audio_times):
        expected_video = expected_video - t * (expected_video * 0.1)
        expected_audio = expected_audio - audio_t * (expected_audio * 0.2)
    assert indices == [0, 1, 2]
    assert video_scheduler.seen == times.tolist()
    assert audio_scheduler.seen == audio_times.tolist()
    for actual, expected in ((video, expected_video), (extra["audio"], expected_audio)):
        relative_l2 = (actual - expected).norm() / expected.norm()
        cosine = torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
        max_error = (actual - expected).abs().max()
        print(f"scheduler T1: relative_l2={relative_l2}, cosine={cosine}, max_abs={max_error}")
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
        assert relative_l2 <= 1e-2 and cosine >= 0.9999


@pytest.mark.parametrize(
    "schedule", [{"missing": torch.ones(3)}, {"audio": torch.ones(2)}, {"audio": torch.ones(3, 1)}]
)
def test_shared_denoise_rejects_invalid_stream_schedule(schedule) -> None:
    pipeline = _SyntheticMiniMaxH3Pipeline()
    pipeline.scheduler.set_timesteps(4)
    with pytest.raises(ValueError, match="schedule"):
        pipeline.denoise(
            torch.ones(1, 4, 2),
            pipeline.scheduler,
            torch.ones(1, 2, 4),
            1.0,
            lambda *args: pytest.fail("Invalid schedule reached the transformer"),
            extra_streams={"audio": (torch.ones(1, 4, 3), pipeline.audio_scheduler)},
            extra_stream_timesteps=schedule,
        )


def test_h3_shared_loop_matches_native_scheduler_rollout(monkeypatch: pytest.MonkeyPatch) -> None:
    """L2/T1: both H3 latent streams vs the former unbatched native scheduler rollout."""
    pipeline = _SyntheticMiniMaxH3Pipeline()
    original_denoise = pipeline.denoise
    checked = []

    def compare_rollout(**kwargs):
        video = kwargs["latents"][0].clone()
        audio = kwargs["extra_streams"]["audio"][0][0].clone()
        video_scheduler = MiniMaxH3Scheduler(shift=12.0)
        audio_scheduler = MiniMaxH3Scheduler(shift=3.0)
        video_scheduler.set_timesteps(4)
        audio_scheduler.set_timesteps(4)
        # The fake joint transformer emits ones; use the exact same initial latents.
        for video_t, audio_t in zip(video_scheduler.timesteps, audio_scheduler.timesteps):
            video = video_scheduler.step(torch.ones_like(video), video_t, video, return_dict=False)[
                0
            ]
            audio = audio_scheduler.step(torch.ones_like(audio), audio_t, audio, return_dict=False)[
                0
            ]
        actual_video, actual_extra = original_denoise(**kwargs)
        for actual, expected in ((actual_video[0], video), (actual_extra["audio"][0], audio)):
            relative_l2 = (actual - expected).norm() / expected.norm()
            cosine = torch.nn.functional.cosine_similarity(
                actual.flatten(), expected.flatten(), dim=0
            )
            max_error = (actual - expected).abs().max()
            print(f"H3 rollout T1: relative_l2={relative_l2}, cosine={cosine}, max_abs={max_error}")
            torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
            assert relative_l2 <= 1e-2 and cosine >= 0.9999
        checked.append(True)
        return actual_video, actual_extra

    monkeypatch.setattr(pipeline, "denoise", compare_rollout)
    pipeline.forward(
        prompt="test",
        seed=42,
        height=32,
        width=32,
        num_frames=124,
        frame_rate=24.0,
        num_inference_steps=4,
    )
    assert checked == [True]


def test_request_generator_preserves_released_cpu_rng_contract() -> None:
    generator = MiniMaxH3Pipeline._request_generator(123)

    assert generator.device.type == "cpu"
    assert torch.equal(
        torch.randn(8, generator=generator),
        torch.randn(8, generator=torch.Generator().manual_seed(123)),
    )


def test_pipeline_unwraps_component_scoped_transformer_weights() -> None:
    pipeline = _SyntheticMiniMaxH3Pipeline()
    captured: dict[str, object] = {}
    expected_weights = {"proj_in.weight": torch.ones(1)}
    pipeline.transformer.load_weights = lambda weights: captured.update(weights=weights)

    pipeline.load_weights({PipelineComponent.TRANSFORMER: expected_weights})

    assert captured["weights"] is expected_weights
    assert not pipeline.transformer.training


def test_pipeline_keeps_torch_compile_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        h3_pipeline.BasePipeline,
        "__init__",
        lambda self, config: torch.nn.Module.__init__(self),
    )

    def _config(torch_compile: TorchCompileConfig) -> SimpleNamespace:
        return SimpleNamespace(
            mapping=SimpleNamespace(world_size=1),
            attention=SimpleNamespace(backend="VANILLA"),
            cache=None,
            cpu_offload_config=SimpleNamespace(enable=False),
            cuda_graph=SimpleNamespace(enable=False),
            torch_compile=torch_compile,
        )

    # The contract is the asymmetry: MiniMax-H3 rejects CUDA graphs and caching
    # but accepts torch.compile, and never quietly turns it off. Asserting the
    # config default alone would only exercise TorchCompileConfig.
    for torch_compile in (TorchCompileConfig(), TorchCompileConfig(enable=True)):
        config = _config(torch_compile)
        MiniMaxH3Pipeline(config)
        assert config.torch_compile.enable, "pipeline disabled torch.compile"

    disabled = _config(TorchCompileConfig(enable=False))
    MiniMaxH3Pipeline(disabled)
    assert not disabled.torch_compile.enable, "pipeline force-enabled torch.compile"

    rejected = _config(TorchCompileConfig(enable=True))
    rejected.cuda_graph = SimpleNamespace(enable=True)
    with pytest.raises(NotImplementedError, match="CUDA graphs"):
        MiniMaxH3Pipeline(rejected)


def test_modular_manifest_discovers_h3_pipeline_and_transformer_config(tmp_path: Path) -> None:
    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer" / "config.json").write_text(
        json.dumps({"hidden_size": 32}),
        encoding="utf-8",
    )
    (tmp_path / "modular_model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "MiniMaxH3ModularPipeline",
                "transformer": ["diffusers", "MiniMaxH3Transformer3DModel"],
            }
        ),
        encoding="utf-8",
    )

    components = discover_pipeline_components(tmp_path)

    assert components == {"transformer": tmp_path / "transformer" / "config.json"}
    assert AutoPipeline._detect_from_checkpoint(str(tmp_path)) == ("MiniMaxH3ModularPipeline")


def test_public_visual_gen_lists_minimax_h3() -> None:
    from tensorrt_llm import VisualGen

    assert "MiniMaxAI/MiniMax-H3" in VisualGen.supported_models()


def test_modular_manifest_preserves_pipeline_metadata(tmp_path: Path) -> None:
    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer" / "config.json").write_text('{"hidden_size": 32}')
    (tmp_path / "modular_model_index.json").write_text(
        json.dumps(
            {
                "transformer": ["diffusers", "WanTransformer3DModel"],
                "transformer_2": ["diffusers", "WanTransformer3DModel"],
                "boundary_ratio": 0.9,
                "expand_timesteps": True,
            }
        )
    )
    config = DiffusionPipelineConfig.from_pretrained(str(tmp_path))
    assert config.primary_pretrained_config.boundary_ratio == 0.9
    assert config.primary_pretrained_config.expand_timesteps is True


@pytest.mark.parametrize("capability", [(8, 9), (9, 0), (10, 1), (12, 0)])
def test_trtllm_attention_rejects_unvalidated_gpu(
    monkeypatch: pytest.MonkeyPatch, capability: tuple[int, int]
) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    config = SimpleNamespace(
        mapping=SimpleNamespace(world_size=1),
        attention=SimpleNamespace(backend="TRTLLM"),
    )
    with pytest.raises(NotImplementedError, match="SM100"):
        MiniMaxH3Pipeline(config)


@pytest.mark.parametrize("capability", [(10, 0), (10, 3)])
def test_trtllm_attention_accepts_supported_gpu(
    monkeypatch: pytest.MonkeyPatch, capability: tuple[int, int]
) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    monkeypatch.setattr(
        h3_pipeline.BasePipeline, "__init__", lambda self, config: torch.nn.Module.__init__(self)
    )
    config = SimpleNamespace(
        mapping=SimpleNamespace(world_size=1),
        attention=SimpleNamespace(backend="TRTLLM"),
        cache=None,
        cpu_offload_config=SimpleNamespace(enable=False),
        cuda_graph=SimpleNamespace(enable=False),
    )
    MiniMaxH3Pipeline(config)


def test_default_generation_steps_match_reference_app() -> None:
    pipeline = _SyntheticMiniMaxH3Pipeline()
    assert pipeline.default_generation_params["num_inference_steps"] == 28


def test_hf_download_is_scoped_to_the_supported_fl2va_components(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _capture_partial_download(
        model: str,
        allow_patterns: list[str],
        revision: str | None = None,
    ) -> Path:
        captured.update(
            model=model,
            allow_patterns=allow_patterns,
            revision=revision,
        )
        return tmp_path

    def _reject_full_download(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("MiniMax-H3 must not download the full 354 GB repository")

    monkeypatch.setattr(
        "tensorrt_llm._torch.visual_gen.pipeline_loader.download_hf_partial",
        _capture_partial_download,
        raising=False,
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.visual_gen.pipeline_loader.download_hf_model",
        _reject_full_download,
    )
    loader = PipelineLoader(
        VisualGenArgs(
            model="MiniMaxAI/MiniMax-H3",
            revision="5d9b308a59ab12e67147f191e184baf704185bd1",
        ),
        device="cpu",
    )

    resolved = loader._resolve_checkpoint_dir("MiniMaxAI/MiniMax-H3")

    assert resolved == str(tmp_path)
    assert captured == {
        "model": "MiniMaxAI/MiniMax-H3",
        "allow_patterns": [
            "modular_model_index.json",
            "LICENSE",
            "README.md",
            "transformer/*",
            "text_encoder/*",
            "tokenizer/*",
            "processor/*",
            "vae/*",
            "audio_vae/*",
            "scheduler/*",
            "audio_scheduler/*",
        ],
        "revision": "5d9b308a59ab12e67147f191e184baf704185bd1",
    }


def test_infer_supports_a_last_frame_without_a_first_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _SyntheticMiniMaxH3Pipeline()
    captured: dict[str, object] = {}
    marker = object()

    def _return_marker(*args: object, **kwargs: object) -> object:
        del args, kwargs
        return marker

    monkeypatch.setattr(h3_pipeline, "load_image", _return_marker)

    def _capture_forward(**kwargs: object) -> str:
        captured.update(kwargs)
        return "output"

    pipeline.forward = _capture_forward
    request = SimpleNamespace(
        prompt="finish at this frame",
        params=SimpleNamespace(
            negative_prompt=None,
            num_images_per_prompt=1,
            image_reference=[
                SimpleNamespace(content="/tmp/last.png", format="path", role="last_frame")
            ],
            extra_params=None,
            seed=42,
            height=768,
            width=1344,
            num_frames=124,
            frame_rate=24.0,
            num_inference_steps=4,
        ),
    )

    assert pipeline.infer(request) == "output"
    assert captured["keyframes"] == [marker]
    assert captured["keyframe_anchors"] == ("last",)


@pytest.mark.parametrize("prompts", [["one public API prompt"], ["first", "second"]])
def test_infer_preserves_the_executor_prompt_batch(prompts: list[str]) -> None:
    pipeline = _SyntheticMiniMaxH3Pipeline()
    captured: dict[str, object] = {}

    def _capture_forward(**kwargs: object) -> str:
        captured.update(kwargs)
        return "output"

    pipeline.forward = _capture_forward
    request = SimpleNamespace(
        prompt=prompts,
        prepared_inputs={"keyframes": [], "keyframe_anchors": ()},
        params=SimpleNamespace(
            negative_prompt=None,
            num_images_per_prompt=1,
            image_reference=None,
            extra_params=None,
            seed=0,
            height=128,
            width=128,
            num_frames=124,
            frame_rate=24.0,
            num_inference_steps=2,
        ),
    )

    assert pipeline.infer(request) == "output"
    assert captured["prompt"] == (prompts[0] if len(prompts) == 1 else prompts)


def test_batched_audio_decode_preserves_samples_and_channels() -> None:
    pipeline = _SyntheticMiniMaxH3Pipeline()
    pipeline.audio_vae = SimpleNamespace(
        config=SimpleNamespace(latents_mean=[0.0], latents_std=[1.0]),
        decode=lambda latents, return_dict: (latents,),
    )
    rows = torch.arange(2 * 2 * 5, dtype=torch.float32).reshape(2, 10, 1)
    audio = MiniMaxH3Pipeline._decode_audio(pipeline, rows, 5)
    assert audio.shape == (2, 2, 5)
    torch.testing.assert_close(audio, rows.reshape(2, 2, 5))


def test_batched_video_decode_drops_each_samples_condition_rows() -> None:
    pipeline = _SyntheticMiniMaxH3Pipeline()
    pipeline.vae = SimpleNamespace(
        config=SimpleNamespace(latent_channels=3, latents_mean=[0.0] * 3, latents_std=[1.0] * 3),
        decode=lambda latents, return_dict: (latents,),
    )
    rows = torch.zeros(2, 2, 12)
    rows[:, 0] = float("nan")
    rows[1, 1] = 1.0
    video = MiniMaxH3Pipeline._decode_video(pipeline, rows, 1, 1, 2, 2)
    assert video.shape == (2, 1, 2, 2, 3)
    assert video.dtype == torch.uint8
    assert (video[1] > video[0]).all()


def test_prepare_request_derives_default_canvas_from_last_only_keyframe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _SyntheticMiniMaxH3Pipeline()
    last_frame = Image.new("RGB", (512, 1024))

    def _return_last_frame(*args: object, **kwargs: object) -> Image.Image:
        del args, kwargs
        return last_frame

    monkeypatch.setattr(h3_pipeline, "load_image", _return_last_frame)
    request = SimpleNamespace(
        params=SimpleNamespace(
            image_reference=[
                SimpleNamespace(content="/tmp/portrait.png", format="path", role="last_frame")
            ],
            height=None,
            width=None,
            extra_params=None,
        ),
        prepared_inputs={},
    )

    pipeline.prepare_request(request)

    assert (request.params.height, request.params.width) == resolve_canvas_size(512, 1024)
    assert request.prepared_inputs["keyframes"] == [last_frame]
    assert request.prepared_inputs["keyframe_anchors"] == ("last",)


def test_prepare_request_uses_default_t2va_canvas_and_preserves_explicit_size() -> None:
    pipeline = _SyntheticMiniMaxH3Pipeline()
    default_request = SimpleNamespace(
        params=SimpleNamespace(
            image_reference=None,
            height=None,
            width=None,
            extra_params=None,
        ),
        prepared_inputs={},
    )
    explicit_request = SimpleNamespace(
        params=SimpleNamespace(
            image_reference=None,
            height=512,
            width=768,
            extra_params=None,
        ),
        prepared_inputs={},
    )

    pipeline.prepare_request(default_request)
    pipeline.prepare_request(explicit_request)

    assert (default_request.params.height, default_request.params.width) == resolve_canvas_size(
        16, 9
    )
    assert (explicit_request.params.height, explicit_request.params.width) == (512, 768)
    assert pipeline.default_generation_params["height"] is None
    assert pipeline.default_generation_params["width"] is None


def test_prepare_request_rejects_partial_canvas_override() -> None:
    pipeline = _SyntheticMiniMaxH3Pipeline()
    request = SimpleNamespace(
        params=SimpleNamespace(
            image_reference=None,
            height=768,
            width=None,
            extra_params=None,
        ),
        prepared_inputs={},
    )

    with pytest.raises(ValueError, match="height and width must be set together"):
        pipeline.prepare_request(request)


@pytest.mark.parametrize(
    ("velocity", "match"),
    [
        (torch.zeros(1, 4, 2), "all zeros"),
        (torch.full((1, 4, 2), float("nan")), "not finite"),
        (torch.full((1, 4, 2), float("inf")), "not finite"),
    ],
)
def test_denoise_step_rejects_unusable_velocity(velocity: torch.Tensor, match: str) -> None:
    """A blank or non-finite velocity must fail loudly, not decode to black."""
    with pytest.raises(RuntimeError, match=match):
        h3_pipeline._check_denoise_step(velocity, "video", 3)


def test_denoise_step_accepts_a_normal_velocity() -> None:
    h3_pipeline._check_denoise_step(torch.randn(1, 4, 2), "video", 3)


@pytest.mark.parametrize("generated_value,condition_value", [(0.0, 1.0), (1.0, float("nan"))])
def test_denoise_validates_only_generated_rows(
    monkeypatch: pytest.MonkeyPatch, generated_value: float, condition_value: float
) -> None:
    pipeline = _SyntheticMiniMaxH3Pipeline()
    # A 32x32 image becomes 16x16 latents, or 64 rows of 2x2 patches.
    condition_rows = 64
    monkeypatch.setattr(pipeline, "_encode_keyframes", lambda *args: torch.zeros(condition_rows, 8))

    class ConditionedTransformer(_FakeMiniMaxH3Transformer):
        def __call__(
            self,
            *,
            hidden_states: torch.Tensor,
            audio_hidden_states: torch.Tensor,
            **kwargs: object,
        ):
            video = torch.full_like(hidden_states, generated_value)
            video[:, :condition_rows] = condition_value
            return video, torch.ones_like(audio_hidden_states)

    pipeline.transformer = ConditionedTransformer()
    request = dict(
        prompt="test",
        seed=42,
        height=32,
        width=32,
        num_frames=124,
        frame_rate=24.0,
        num_inference_steps=2,
        keyframes=[Image.new("RGB", (32, 32))],
        keyframe_anchors=("first",),
    )
    if generated_value == 0:
        with pytest.raises(RuntimeError, match="all zeros"):
            pipeline.forward(**request)
    else:
        assert pipeline.forward(**request).video.shape[1] == 124


def _png_bytes(color: tuple[int, int, int] = (10, 20, 30)) -> bytes:
    """A real PNG payload, matching what prepare_reference_slots hands a worker."""
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (64, 64), color).save(buf, format="PNG")
    return buf.getvalue()


def test_keyframes_decode_resolved_reference_bytes() -> None:
    """References reach the worker as bytes, not paths.

    ``prepare_reference_slots`` rewrites every reference to ``format="bytes"``
    before the request is broadcast, so decoding must not route through the
    URL/path loader.  This deliberately does not patch ``load_image``.
    """
    payload = _png_bytes()
    req = SimpleNamespace(
        prompt="a test prompt",
        params=SimpleNamespace(
            image_reference=[
                SimpleNamespace(content=payload, format="bytes", role="first_frame"),
                SimpleNamespace(content=payload, format="bytes", role="last_frame"),
            ],
            extra_params=None,
        ),
    )
    pipeline = _SyntheticMiniMaxH3Pipeline()
    keyframes, anchors = pipeline._load_request_keyframes(req)

    assert anchors == ("first", "last")
    assert [image.size for image in keyframes] == [(64, 64), (64, 64)]
    assert all(image.mode == "RGB" for image in keyframes)


@pytest.mark.parametrize(
    ("roles", "expected_anchors"),
    [
        ((), ()),
        (("first_frame",), ("first",)),
        (("last_frame",), ("last",)),
        (("first_frame", "last_frame"), ("first", "last")),
        # Declaration order must not change the emitted anchor order: the first
        # keyframe is stretched and the rest cropped (`stretch=index == 0`).
        (("last_frame", "first_frame"), ("first", "last")),
    ],
)
def test_every_supported_keyframe_combination(
    roles: tuple[str, ...], expected_anchors: tuple[str, ...]
) -> None:
    """T2VA, first-only, last-only and first+last are all reachable."""
    payload = _png_bytes()
    req = SimpleNamespace(
        prompt="a test prompt",
        params=SimpleNamespace(
            image_reference=[
                SimpleNamespace(content=payload, format="bytes", role=role) for role in roles
            ]
            or None,
            extra_params=None,
        ),
    )
    keyframes, anchors = _SyntheticMiniMaxH3Pipeline()._load_request_keyframes(req)

    assert anchors == expected_anchors
    assert len(keyframes) == len(expected_anchors)


def test_keyframe_reference_requires_an_explicit_role() -> None:
    """Both roles are optional, so upstream validation cannot infer one.

    There is no positional fallback: an unroled reference is rejected rather
    than silently treated as the first frame.
    """
    req = SimpleNamespace(
        prompt="a test prompt",
        params=SimpleNamespace(
            image_reference=[SimpleNamespace(content=_png_bytes(), format="bytes", role=None)],
            extra_params=None,
        ),
    )
    with pytest.raises(ValueError, match="requires role"):
        _SyntheticMiniMaxH3Pipeline()._load_request_keyframes(req)


def test_keyframe_reference_rejects_a_duplicate_role() -> None:
    payload = _png_bytes()
    req = SimpleNamespace(
        prompt="a test prompt",
        params=SimpleNamespace(
            image_reference=[
                SimpleNamespace(content=payload, format="bytes", role="last_frame"),
                SimpleNamespace(content=payload, format="bytes", role="last_frame"),
            ],
            extra_params=None,
        ),
    )
    with pytest.raises(ValueError, match="single last_frame"):
        _SyntheticMiniMaxH3Pipeline()._load_request_keyframes(req)


def test_keyframe_reference_role_must_be_a_keyframe_slot() -> None:
    req = SimpleNamespace(
        prompt="a test prompt",
        params=SimpleNamespace(
            image_reference=[
                SimpleNamespace(content=_png_bytes(), format="bytes", role="reference")
            ],
            extra_params=None,
        ),
    )
    with pytest.raises(ValueError, match="first_frame"):
        _SyntheticMiniMaxH3Pipeline()._load_request_keyframes(req)

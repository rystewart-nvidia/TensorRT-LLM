# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local/QA H3 step execution prototype: homogeneous T2VA, at most two requests."""

from dataclasses import dataclass

import torch
from diffusers import MiniMaxH3Scheduler
from tensorrt_llm._torch.visual_gen.models.minimax_h3.packing import (
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    audio_latent_num_frames,
    build_packed_sequence,
    build_row_timesteps,
    video_latent_num_frames,
)
from tensorrt_llm._torch.visual_gen.models.minimax_h3.pipeline_minimax_h3 import (
    _check_denoise_step,
)
from tensorrt_llm._torch.visual_gen.models.minimax_h3.transformer_minimax_h3 import (
    MiniMaxH3StaticContext,
)


@dataclass
class RequestState:
    request_id: int
    prompt: str
    seed: int
    text: torch.Tensor
    text_tags: torch.Tensor
    video: torch.Tensor
    audio: torch.Tensor
    video_scheduler: MiniMaxH3Scheduler
    audio_scheduler: MiniMaxH3Scheduler
    index: int = 0

    @property
    def done(self) -> bool:
        return self.index == len(self.video_scheduler.timesteps)


class H3StepEngine:
    """Own per-request latents and schedulers; share only the transformer forward."""

    def __init__(self, pipeline, height: int = 544, width: int = 960) -> None:
        self.pipeline = pipeline
        self.height = height
        self.width = width
        self.num_frames = 124
        self.num_steps = 28
        self.latent_frames = video_latent_num_frames(self.num_frames)
        self.latent_height = height // pipeline.vae.spatial_compression_ratio
        self.latent_width = width // pipeline.vae.spatial_compression_ratio
        self.audio_length = audio_latent_num_frames(self.num_frames)
        self._members = ()
        self._states = ()
        self._bundle = None
        self._layouts = {}

    def reset(self) -> None:
        self._members = ()
        self._states = ()
        self._bundle = None

    def prepare(self, request_id: int, prompt: str, seed: int) -> RequestState:
        p = self.pipeline
        embeds, tags = p._encode_prompt(prompt, [])
        text = p.transformer.context_embedder(
            embeds.to(p.transformer.context_embedder.dtype)
        )
        text = p.transformer.token_refiner(text)
        video, audio = p._prepare_latents(
            num_latent_frames=self.latent_frames,
            latent_height=self.latent_height,
            latent_width=self.latent_width,
            num_audio_latents=self.audio_length,
            generator=p._request_generator(seed),
            condition_latents=None,
        )
        video_scheduler = MiniMaxH3Scheduler.from_config(p.scheduler.config)
        audio_scheduler = MiniMaxH3Scheduler.from_config(p.audio_scheduler.config)
        for scheduler in (video_scheduler, audio_scheduler):
            scheduler.set_timesteps(self.num_steps, device=p.device)
        if len(video_scheduler.timesteps) != len(audio_scheduler.timesteps):
            raise ValueError("Video and audio schedule lengths must match")
        return RequestState(
            request_id,
            prompt,
            seed,
            text,
            tags,
            video,
            audio,
            video_scheduler,
            audio_scheduler,
        )

    def _pack(self, states: list[RequestState]):
        members = tuple(id(state) for state in states)
        if members == self._members:
            return self._bundle
        p = self.pipeline
        lengths = [state.text.shape[1] for state in states]
        longest = states[lengths.index(max(lengths))]
        key = tuple(longest.text_tags.tolist())
        if key not in self._layouts:
            layout = build_packed_sequence(
                longest.text_tags,
                self.latent_frames,
                self.latent_height,
                self.latent_width,
                self.audio_length,
                p.transformer.config.patch_size,
                [],
            )
            plans = [
                tuple(
                    value.to(p.device)
                    for value in build_row_timesteps(
                        layout,
                        float(v),
                        float(a),
                        max(float(v), MINIMAX_H3_KEYFRAME_NOISE_AUG),
                        1.0,
                    )
                )
                for v, a in zip(
                    longest.video_scheduler.timesteps.cpu(),
                    longest.audio_scheduler.timesteps.cpu(),
                )
            ]
            self._layouts[key] = {
                "rotary": p.transformer.rope(layout.position_ids.to(p.device)),
                "tags": layout.token_tags.to(p.device),
                "video_indices": layout.video_indices.to(p.device),
                "audio_indices": layout.audio_indices.to(p.device),
                "text_indices": layout.text_indices.to(p.device),
                "plans": plans,
            }
        layout = self._layouts[key]
        text = torch.cat(
            [
                torch.cat(
                    (s.text.new_zeros(1, max(lengths) - n, s.text.shape[-1]), s.text),
                    dim=1,
                )
                for s, n in zip(states, lengths)
            ]
        )
        mask = None
        if min(lengths) != max(lengths):
            mask = torch.arange(max(lengths), device=p.device)[None] >= (
                max(lengths) - torch.tensor(lengths, device=p.device)[:, None]
            )
        context = MiniMaxH3StaticContext(text, layout["rotary"], mask)
        self._members = members
        self._states = tuple(states)
        self._bundle = layout, context
        return self._bundle

    def predict(self, states: list[RequestState]) -> tuple[torch.Tensor, torch.Tensor]:
        if not 1 <= len(states) <= 2 or any(state.done for state in states):
            raise ValueError("Expected one or two unfinished requests")
        layout, context = self._pack(states)
        tables, indices, offset = [], [], 0
        for state in states:
            table, row_indices = layout["plans"][state.index]
            tables.append(table)
            indices.append(row_indices + offset)
            offset += len(table)
        if len({state.index for state in states}) == 1:
            table, row_indices = layout["plans"][states[0].index]
        else:
            table, row_indices = torch.cat(tables), torch.stack(indices)
        attention_times = torch.stack(
            [
                1.0
                - torch.minimum(
                    s.video_scheduler.timesteps[s.index],
                    s.audio_scheduler.timesteps[s.index],
                )
                for s in states
            ]
        )
        return self.pipeline.transformer(
            hidden_states=torch.stack([s.video for s in states]),
            audio_hidden_states=torch.stack([s.audio for s in states]),
            encoder_hidden_states=None,
            timestep=attention_times,
            conditioning_timesteps=table,
            timestep_indices=row_indices,
            token_tags=layout["tags"],
            position_ids=None,
            video_indices=layout["video_indices"],
            audio_indices=layout["audio_indices"],
            text_indices=layout["text_indices"],
            return_dict=False,
            static_context=context,
        )

    def step(self, states: list[RequestState]) -> None:
        video_velocity, audio_velocity = self.predict(states)
        _check_denoise_step(video_velocity, "video", min(s.index for s in states))
        _check_denoise_step(audio_velocity, "audio", min(s.index for s in states))
        for row, state in enumerate(states):
            state.video = state.video_scheduler.step(
                video_velocity[row].float(),
                state.video_scheduler.timesteps[state.index],
                state.video,
                return_dict=False,
            )[0]
            state.audio = state.audio_scheduler.step(
                audio_velocity[row].float(),
                state.audio_scheduler.timesteps[state.index],
                state.audio,
                return_dict=False,
            )[0]
            state.index += 1
        torch.cuda.synchronize()

    def finish(self, states: list[RequestState]) -> tuple[torch.Tensor, torch.Tensor]:
        if not states or not all(state.done for state in states):
            raise ValueError("Only finished requests may be decoded")
        video = self.pipeline._decode_video(
            torch.stack([s.video for s in states]),
            0,
            self.latent_frames,
            self.latent_height,
            self.latent_width,
        )
        audio = self.pipeline._decode_audio(
            torch.stack([s.audio for s in states]),
            self.audio_length,
        )
        return video.cpu(), audio.cpu()

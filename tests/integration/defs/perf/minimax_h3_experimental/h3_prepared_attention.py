# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime-only H3 attention metadata and compiler-boundary experiments."""

from dataclasses import dataclass

import torch
from tensorrt_llm._torch.visual_gen.attention_backend.flash_attn4 import _flash_attn_fwd
from tensorrt_llm._torch.visual_gen.attention_backend.vanilla import VanillaAttention

from benchmark_h3_throughput import OffsetAttention


def prepare_offsets(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate a [B,S] Boolean valid suffix and return int32 FA4 metadata."""
    if mask.dtype != torch.bool or mask.ndim != 2 or not mask.is_cuda:
        raise ValueError("Expected a CUDA Boolean [B,S] mask")
    batch, sequence = mask.shape
    lengths = mask.sum(dim=1, dtype=torch.int32)
    expected = (
        torch.arange(sequence, device=mask.device)[None] >= sequence - lengths[:, None]
    )
    if not bool((lengths > 0).all()) or not torch.equal(mask, expected):
        raise ValueError("Only nonempty contiguous valid suffixes are supported")
    offsets = torch.arange(batch + 1, device=mask.device, dtype=torch.int32) * sequence
    return torch.cat((offsets[:-1] + sequence - lengths, offsets[-1:])), lengths


def _forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    offsets: torch.Tensor,
    lengths: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    output, *_ = _flash_attn_fwd(
        q,
        k.flatten(0, 1),
        v.flatten(0, 1),
        cu_seqlens_k=offsets,
        seqused_k=lengths,
        max_seqlen_q=q.shape[1],
        max_seqlen_k=k.shape[1],
        softmax_scale=scale,
        causal=False,
        return_lse=False,
        num_splits=0,
    )
    return output


@torch.compiler.disable
def eager_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    offsets: torch.Tensor,
    lengths: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    return _forward(q, k, v, offsets, lengths, scale)


@torch.library.custom_op("h3_qa::prepared_attention", mutates_args=())
def compiled_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    offsets: torch.Tensor,
    lengths: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("This H3 QA adapter expects equal [B,S,H,D] Q/K/V shapes")
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("This H3 QA adapter expects BF16 Q/K/V")
    if offsets.shape != (q.shape[0] + 1,) or lengths.shape != (q.shape[0],):
        raise ValueError("Metadata batch dimensions do not match Q/K/V")
    return _forward(q, k, v, offsets, lengths, scale)


@compiled_attention.register_fake
def _fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    offsets: torch.Tensor,
    lengths: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    return torch.empty(q.shape, dtype=q.dtype, device=q.device)


@dataclass
class PreparedMetadata:
    """Generation-scoped metadata; reset before every pipeline.forward call.

    H3 static contexts and layout tensors must remain immutable within a call.
    Identity checks retain their owners and cannot collide via recycled IDs.
    """

    context: object = None
    tags: torch.Tensor | None = None
    text_indices: torch.Tensor | None = None
    offsets: torch.Tensor | None = None
    lengths: torch.Tensor | None = None
    sequence: int = 0
    builds: int = 0

    def clear(self) -> None:
        self.context = self.tags = self.text_indices = None
        self.offsets = self.lengths = None
        self.sequence = self.builds = 0

    def before_transformer(self, module, args: tuple, kwargs: dict) -> None:
        context = kwargs["static_context"]
        tags, text_indices = kwargs["token_tags"], kwargs["text_indices"]
        if (
            context is self.context
            and tags is self.tags
            and text_indices is self.text_indices
        ):
            return
        self.context, self.tags, self.text_indices = context, tags, text_indices
        self.sequence = tags.numel()
        if context.text_attention_mask is None and not bool((tags < 0).any()):
            self.offsets = self.lengths = None
            return
        mask = (tags >= 0)[None].expand(context.text_embeds.shape[0], -1).clone()
        if context.text_attention_mask is not None:
            mask[:, text_indices] &= context.text_attention_mask
        self.offsets, self.lengths = prepare_offsets(mask)
        self.builds += 1


class PreparedAttention(OffsetAttention):
    def __init__(self, metadata: PreparedMetadata, opaque: bool, **kwargs) -> None:
        super().__init__(**kwargs)
        self.metadata = metadata
        self.opaque = opaque

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if key_padding_mask is None:
            return VanillaAttention.forward(self, q, k, v, **kwargs)
        if self.metadata.offsets is None or q.shape[2] != self.metadata.sequence:
            raise ValueError("Prepared metadata missing or sequence length mismatch")
        fn = compiled_attention if self.opaque else eager_attention
        return fn(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            self.metadata.offsets,
            self.metadata.lengths,
            self.masked_backend.scale,
        ).transpose(1, 2)

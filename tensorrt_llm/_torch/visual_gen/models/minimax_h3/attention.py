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

"""H3-specific attention for left-padded prompt batches."""

import torch

from ...attention_backend import flash_attn4
from ...attention_backend.vanilla import PredefinedAttentionMask, VanillaAttention


def is_nonempty_suffix_mask(mask: torch.Tensor | None) -> bool:
    """Check once outside the block loop; arbitrary masks must stay on SDPA."""
    if mask is None or mask.dtype != torch.bool or mask.ndim != 2 or mask.shape[1] == 0:
        return False
    return bool((mask[:, -1].all() & ~(mask[:, :-1] & ~mask[:, 1:]).any()).item())


class MiniMaxH3BatchedAttention(VanillaAttention):
    """Keep SDPA except for supported, explicitly validated H3 suffix batches."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
        suffix_mask: bool = False,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        **kwargs,
    ) -> torch.Tensor:
        if (
            suffix_mask
            and attention_mask == PredefinedAttentionMask.FULL
            and key_padding_mask is not None
            and q.shape[0] > 1
            and q.is_cuda
            and q.dtype == k.dtype == v.dtype == torch.bfloat16
            and self.head_dim == 128
            and self.num_heads == self.num_kv_heads
            and q.shape == k.shape == v.shape
            and torch.cuda.get_device_capability(q.device) == (10, 0)
            and flash_attn4._flash_attn_fwd is not None
        ):
            return self._offset_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), key_padding_mask
            ).transpose(1, 2)
        return super().forward(
            q, k, v, attention_mask=attention_mask, key_padding_mask=key_padding_mask, **kwargs
        )

    @torch.compiler.disable
    def _offset_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        batch, sequence = mask.shape
        lengths = mask.sum(dim=1, dtype=torch.int32)
        offsets = torch.arange(batch + 1, dtype=torch.int32, device=q.device) * sequence
        # Each valid K/V span starts after its padding. seqused_k prevents a span
        # from reading the padding at the start of the following batch element.
        key_offsets = torch.cat((offsets[:-1] + sequence - lengths, offsets[-1:]))
        output, *_ = flash_attn4._flash_attn_fwd(
            q,
            k.flatten(0, 1),
            v.flatten(0, 1),
            cu_seqlens_k=key_offsets,
            seqused_k=lengths,
            max_seqlen_q=sequence,
            max_seqlen_k=sequence,
            softmax_scale=self.scale,
            causal=False,
            return_lse=False,
            num_splits=0,
        )
        return output.view_as(q)
